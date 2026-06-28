#!/usr/bin/env python3
"""
CAT-SAM Style Training — Adapter + Prompt Token + Conditioned Decoder
======================================================================
将 CAT-SAM 的三个核心思路 (Adapter / Prompt Token / 条件化解码) 迁移到 FastSAM。

Architecture:
    Image → FastSAM Encoder (frozen) → Adapters → P3/P4/P8
                                                       │
    Support Set → FastSAM → P4 feat → PrototypePrompt ─┤
    GenericPrompt ──────────────────────────────────────┤
                                                       ↓
                                                  Prompt Fusion
                                                       │
                                                       ↓
    Query Image → FastSAM → Adapters → P3/P4 ──→ CATStyleDecoder → Mask
                                                   (CrossAttn + Gate + FiLM)

训练策略 | Training Strategy:
    Stage 1 (本脚本): Freeze FastSAM, Train Adapter + Prompt + Decoder
    Stage 2 (后续):    Joint fine-tune with SPM routing

用法 | Usage::
    python tools/train/train_catsam_style.py \
        --dataset isaid5i \
        --src-root data/iSAID_processed \
        --tile-root data/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 1 --epochs 60 \
        --device cuda
"""

import sys, argparse, json, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "train"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "instance"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS

from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.adapter.conv_adapter import MultiScaleAdapter
from adatile.prompt import GenericPrompt, PrototypePrompt
from adatile.decoder.conditioned_decoder import CATStyleDecoder

logger = get_logger("catsam_train")


def parse_args():
    p = argparse.ArgumentParser(
        description="CAT-SAM Style Training: Adapter + Prompt + Conditioned Decoder")

    # 数据 | Data
    p.add_argument("--dataset", type=str, default="isaid5i")
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--tile-root", type=str, default="data/iSAID5i_tiles/tile_896")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1, choices=[1, 3, 5])

    # 模型 | Model
    p.add_argument("--fusion-dim", type=int, default=256)
    p.add_argument("--prompt-dim", type=int, default=256)
    p.add_argument("--num-prompt-tokens", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--adapter-reduction", type=int, default=16,
                   help="Adapter 通道压缩比 (16=~460K params)")

    # 训练 | Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-episodes", type=int, default=150)
    p.add_argument("--workers", type=int, default=4)

    # 硬件 | Hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")

    # 输出 | Output
    p.add_argument("--output-dir", type=str, default="runs/catsam_training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=10)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 训练引擎 | Training Engine
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(
    backbone: FastSAMBackbone,
    decoder: CATStyleDecoder,
    prompt_generic: GenericPrompt,
    prompt_proto: PrototypePrompt,
    train_ds,
    val_ds,
    novel_ids: list,
    shot: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.RandomState,
    scaler=None,
) -> dict:
    """训练一个 epoch | Train one epoch (episodic)."""
    backbone.eval()
    decoder.train()
    prompt_generic.train()
    prompt_proto.train()

    losses = []
    all_ious = []

    n_episodes = 200  # 每 epoch 采样 episode 数 | Episodes per epoch

    for ep_idx in tqdm(range(n_episodes), desc="CAT-SAM [train]", unit="ep"):
        # ── 随机选择类别 | Random class selection ──
        cls_id = int(rng.choice(novel_ids))
        candidates = train_ds.class_to_images(cls_id)
        if len(candidates) < shot + 1:
            continue

        # ── 采样 Support + Query | Sample Support + Query ──
        idxs = rng.choice(candidates, shot + 1, replace=False).tolist()
        s_idxs = idxs[:shot]
        q_idx = idxs[shot]

        # ── Support → prototype ──
        s_imgs = torch.stack([train_ds.load_image(si) for si in s_idxs]).to(device)
        s_masks = torch.stack([
            train_ds.render_class_mask(si, cls_id) for si in s_idxs
        ]).to(device).float()
        s_feats = backbone(s_imgs)
        s_p4 = s_feats["p4"]  # [S, 1280, H/16, W/16]

        # Prototype from support P4
        proto = prompt_proto(s_p4, s_masks)  # [S, 1, D] → mean → [1, D]

        # ── Query image → features ──
        q_img = train_ds.load_image(q_idx).unsqueeze(0).to(device)
        q_mask = train_ds.render_class_mask(q_idx, cls_id).unsqueeze(0).to(device).float()
        q_feats = backbone(q_img)

        # ── Build Prompt | 构建条件 ──
        generic = prompt_generic(batch_size=1)   # [1, N, D]
        combined = torch.cat([generic, proto.unsqueeze(0)], dim=1)  # [1, N+1, D]
        # Simple fusion: mean along token dim for decoder
        prompt = combined.mean(dim=1, keepdim=False)  # [1, D]
        prompt = prompt.unsqueeze(0)  # [1, 1, D] — for decoder compatibility

        # ── Decoder forward | 解码 ──
        # CATStyleDecoder expects prompt [B, N, D]
        decoder_prompt = torch.cat([generic, proto.unsqueeze(0)], dim=1)  # [1, N+1, D]
        logit = decoder(
            q_feats["p3"], q_feats["p4"],
            decoder_prompt,
            target_size=tuple(q_mask.shape[-2:]),
        )

        # ── Loss: Dice + BCE | 损失: Dice + BCE ──
        prob = torch.sigmoid(logit)
        bce = F.binary_cross_entropy_with_logits(logit, q_mask)
        smooth = 1.0
        intersection = (prob * q_mask).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (prob.sum() + q_mask.sum() + smooth)
        loss = bce + dice

        # Backward
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        # IoU
        pred = (prob > 0.5).float()
        inter = (pred * q_mask).sum()
        union = (pred + q_mask).clamp(0, 1).sum()
        iou = (inter / union.clamp(min=1)).item()
        all_ious.append(iou)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "miou": float(np.mean(all_ious)) if all_ious else 0.0,
    }


@torch.no_grad()
def validate_epoch(
    backbone: FastSAMBackbone,
    decoder: CATStyleDecoder,
    prompt_generic: GenericPrompt,
    prompt_proto: PrototypePrompt,
    train_ds,
    val_ds,
    novel_ids: list,
    shot: int,
    device: torch.device,
    rng: np.random.RandomState,
    n_episodes: int = 150,
) -> dict:
    """验证 | Validation (episodic)."""
    backbone.eval()
    decoder.eval()
    prompt_generic.eval()
    prompt_proto.eval()

    per_class_ious = {c: [] for c in novel_ids}
    all_ious = []

    for ep_idx in tqdm(range(n_episodes), desc="CAT-SAM [val]", unit="ep"):
        cls_id = int(rng.choice(novel_ids))
        train_cand = train_ds.class_to_images(cls_id)
        val_cand = val_ds.class_to_images(cls_id)
        if len(train_cand) < shot or not val_cand:
            continue

        s_idxs = rng.choice(train_cand, shot, replace=False).tolist()
        q_idx = int(rng.choice(val_cand))

        # Support
        s_imgs = torch.stack([train_ds.load_image(si) for si in s_idxs]).to(device)
        s_masks = torch.stack([
            train_ds.render_class_mask(si, cls_id) for si in s_idxs
        ]).to(device).float()
        s_feats = backbone(s_imgs)
        proto = prompt_proto(s_feats["p4"], s_masks)

        # Query
        q_img = val_ds.load_image(q_idx).unsqueeze(0).to(device)
        q_mask = val_ds.render_class_mask(q_idx, cls_id).unsqueeze(0).to(device).float()
        q_feats = backbone(q_img)

        # Prompt
        generic = prompt_generic(batch_size=1)
        decoder_prompt = torch.cat([generic, proto.unsqueeze(0)], dim=1)

        # Decode
        logit = decoder(
            q_feats["p3"], q_feats["p4"],
            decoder_prompt,
            target_size=tuple(q_mask.shape[-2:]),
        )

        # IoU
        pred = (torch.sigmoid(logit) > 0.5).float()
        inter = (pred * q_mask).sum()
        union = (pred + q_mask).clamp(0, 1).sum()
        iou = (inter / union.clamp(min=1)).item()
        all_ious.append(iou)
        per_class_ious[cls_id].append(iou)

    per_cls = {}
    for c in novel_ids:
        ious = per_class_ious[c]
        per_cls[c] = float(np.mean(ious)) if ious else 0.0

    return {
        "miou": float(np.mean(all_ious)) if all_ious else 0.0,
        "per_class": per_cls,
        "n_episodes": len(all_ious),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main | 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logging ──
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "catsam_training.jsonl")))

    logger.log_info("config", f"Fold={args.fold}, Shot={args.shot}, Device={device}")
    logger.log_info("config", f"Fusion dim={args.fusion_dim}, Prompt dim={args.prompt_dim}")
    logger.log_info("config", f"Prompt tokens={args.num_prompt_tokens}, Heads={args.num_heads}")

    # ── 数据 | Data ──
    from train_fewshot import PreCutTileAdapter
    train_ds = PreCutTileAdapter(args.tile_root, "train")
    val_ds = PreCutTileAdapter(args.tile_root, "val")
    fold_info = ISAID5I_FOLDS[args.fold]
    novel_ids = fold_info["novel"]
    logger.log_info("data", f"Novel classes: {[ISAID5I_CATEGORIES[c] for c in novel_ids]}")

    # ── Backbone + Adapters | 骨架 + 适配器 ──
    logger.log_info("model", "Loading FastSAM backbone + Adapters...")
    backbone = FastSAMBackbone(device=str(device))
    backbone.freeze()

    adapters = MultiScaleAdapter(
        p3_channels=640, p4_channels=1280, p8_channels=1280,
        reduction=args.adapter_reduction,
    ).to(device)
    backbone.set_adapters(adapters)

    n_bb = sum(p.numel() for p in backbone.model.model.parameters())
    n_ad = sum(p.numel() for p in adapters.parameters())
    logger.log_info("model",
                    f"Backbone: {n_bb:,} params (frozen), "
                    f"Adapters: {n_ad:,} params (trainable)")

    # ── Prompt Tokens | 提示令牌 ──
    prompt_generic = GenericPrompt(
        num_tokens=args.num_prompt_tokens, dim=args.prompt_dim,
    ).to(device)
    prompt_proto = PrototypePrompt(
        feat_dim=1280, proto_dim=args.prompt_dim,
    ).to(device)

    n_gp = sum(p.numel() for p in prompt_generic.parameters())
    n_pp = sum(p.numel() for p in prompt_proto.parameters())
    logger.log_info("model",
                    f"GenericPrompt: {n_gp:,} params, "
                    f"PrototypePrompt: {n_pp:,} params")

    # ── CAT-Style Decoder | CAT 风格解码器 ──
    decoder = CATStyleDecoder(
        feat_dim_p3=640, feat_dim_p4=1280,
        fusion_dim=args.fusion_dim,
        prompt_dim=args.prompt_dim,
        num_heads=args.num_heads,
    ).to(device)

    n_dec = sum(p.numel() for p in decoder.parameters())
    n_train = n_ad + n_gp + n_pp + n_dec
    logger.log_info("model",
                    f"CATStyleDecoder: {n_dec:,} params")
    logger.log_info("model",
                    f"Total trainable: {n_train:,} params "
                    f"(Adapters={n_ad:,} + Prompt={n_gp+n_pp:,} + Decoder={n_dec:,})")

    # ── Optimizer | 优化器 ──
    trainable_params = (
        list(adapters.parameters()) +
        list(prompt_generic.parameters()) +
        list(prompt_proto.parameters()) +
        list(decoder.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    # ── 训练循环 | Training Loop ──
    rng = np.random.RandomState(args.seed)
    val_rng = np.random.RandomState(args.seed + 1000)
    best_miou = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(
            backbone, decoder, prompt_generic, prompt_proto,
            train_ds, val_ds, novel_ids, args.shot,
            optimizer, device, rng, scaler,
        )

        val_m = validate_epoch(
            backbone, decoder, prompt_generic, prompt_proto,
            train_ds, val_ds, novel_ids, args.shot,
            device, val_rng, n_episodes=args.val_episodes,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.log_info("epoch",
                        f"E{epoch:4d} | "
                        f"train: loss={train_m['loss']:.4f} mIoU={train_m['miou']:.4f} | "
                        f"val: mIoU={val_m['miou']:.4f} | lr={lr_now:.2e}")

        # Per-class log
        per_cls_str = " ".join(
            f"{ISAID5I_CATEGORIES[c][:6]}={val_m['per_class'].get(c,0):.3f}"
            for c in novel_ids
        )
        logger.log_info("per_class", f"  val per-class: {per_cls_str}")

        # Save best
        if val_m["miou"] > best_miou:
            best_miou = val_m["miou"]
            torch.save({
                "epoch": epoch,
                "val_miou": best_miou,
                "per_class": val_m["per_class"],
                "adapter_state": adapters.state_dict(),
                "prompt_generic_state": prompt_generic.state_dict(),
                "prompt_proto_state": prompt_proto.state_dict(),
                "decoder_state": decoder.state_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
            }, out_dir / "catsam_best.pt")
            logger.log_info("save", f"  Best model saved (mIoU={best_miou:.4f})")

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch, "val_miou": val_m["miou"],
                "decoder_state": decoder.state_dict(),
                "adapter_state": adapters.state_dict(),
            }, out_dir / f"catsam_e{epoch:04d}.pt")

    elapsed = time.time() - t_start
    logger.log_info("done", f"Training complete in {elapsed/60:.1f} min")
    logger.log_info("done", f"Best val mIoU: {best_miou:.4f}")

    # ── 对标 | Benchmark ──
    logger.log_info("done", "")
    logger.log_info("done", "Benchmark vs existing decoders:")
    logger.log_info("done", f"  FiLM P3P4 (baseline):     0.158 (Fold 0, 1-shot)")
    logger.log_info("done", f"  CAT-SAM Style (this run):  {best_miou:.3f}")
    if best_miou > 0.158:
        gain = (best_miou - 0.158) / 0.158 * 100
        logger.log_info("done", f"  Improvement: +{gain:.0f}% over FiLM baseline")
    logger.log_info("done", f"  Checkpoint: {out_dir / 'catsam_best.pt'}")


if __name__ == "__main__":
    main()
