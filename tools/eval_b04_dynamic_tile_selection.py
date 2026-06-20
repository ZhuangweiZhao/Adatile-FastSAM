#!/usr/bin/env python3
"""
B-04: Dynamic Tile Selection — FDR × Decoder 端到端验证
==========================================================

Paper B 核心实验。将 FDR (Foreground Density Router) 接入真实分割管线，
测量 Accuracy vs Compute 曲线。

实验设计 | Design:
    Step 1: 训练 FastSAM + LightDecoder (15 类 iSAID, tile 级别)
    Step 2: FDR 预测全图 tile 重要性 → Top-K 选择
    Step 3: 只解码选中 tile → 对比全量解码的 mIoU + 计算量

核心问题 | Core question:
    Top40% tiles → mIoU 下降多少? GFLOPs 节省多少?

预期 | Expected:
    Top40% → mIoU ↓ <2%, GFLOPs ↓ ~60%, FPS ↑ ~2×

用法 | Usage:
    python tools/eval_b04_dynamic_tile_selection.py
    python tools/eval_b04_dynamic_tile_selection.py --train-images 300 --decoder-epochs 10
"""

import sys, argparse, json, datetime, os, time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone

logger = get_logger("b04_dynamic")
logger.add_backend(ConsoleBackend())

ACTUAL_TO_CODE_ID = {
    1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 10, 7: 6, 8: 9,
    9: 7, 10: 8, 11: 11, 12: 13, 13: 12, 14: 15, 15: 14,
}
TILE_SIZE = 1024
NUM_CLASSES = 15  # 15 foreground classes (1-15)
NUM_OUT_CH = 16    # 0=background + 15 foreground = 16 output channels
BACKBONE_STRIDE = 32
FDR_FEAT_PER_TILE = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--tile-root", type=str, default="data/iSAID_tiles")
    p.add_argument("--train-images", type=int, default=200)
    p.add_argument("--decoder-epochs", type=int, default=20)
    p.add_argument("--fdr-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=2048)
    p.add_argument("--top-k-list", type=str, default="10,20,30,40,50,70,100")
    p.add_argument("--output-dir", type=str, default="runs/b04_dynamic_tile")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-decoder-train", action="store_true",
                   help="跳过 decoder 训练, 使用已有权重 | Skip decoder train")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Light Decoder — 轻量 15 类分割头 | Lightweight 15-class decoder
# ═══════════════════════════════════════════════════════════════════

class LightDecoder(nn.Module):
    """
    从 FastSAM P4 特征预测 15 类分割 | 15-class segmentation from FastSAM P4.
    P4: [B, 1280, H/16, W/16] → [B, 15, H, W].

    先 1×1 降维（1280→256），再 3×3 做空间推理，减少参数量。
    1×1 channel reduction first (1280→256), then 3×3 spatial reasoning.
    """

    def __init__(self, in_channels: int = 1280, num_classes: int = 15):
        super().__init__()
        self.decoder = nn.Sequential(
            # 1×1 降维 | Channel reduction: 1280 → 256
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 3×3 空间推理 | Spatial reasoning
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 上采样 + 细化 | Upsample + refine
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1, bias=True),
        )

    def forward(self, p4: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            p4: [B, 1280, H/16, W/16].
            target_size: (H, W) 上采样目标尺寸 | upsample target.
        Returns:
            [B, 15, H, W] logits.
        """
        out = self.decoder(p4)
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode="bilinear",
                               align_corners=False)
        return out


# ═══════════════════════════════════════════════════════════════════
# FDR 包装器 (复用 B-03 设计) | FDR wrapper (reuse B-03 design)
# ═══════════════════════════════════════════════════════════════════

class FDRPredictor(nn.Module):
    """
    FDR: MV3 backbone → DensityHead → tile importance scores.
    处理全图, 输出 per-tile 重要性分数.
    """

    def __init__(self):
        super().__init__()
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        from adatile.sparse.spatial_router import DensityHead
        self.density_head = DensityHead(576, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] → importance [B, 1, H/32, W/32]."""
        return self.density_head(self.backbone(x))

    @torch.no_grad()
    def predict_tile_scores(self, img_np: np.ndarray, device: str) -> np.ndarray:
        """
        全图 → tile 分数 | Full image → tile scores.
        Args:
            img_np: [H, W, 3] uint8.
        Returns:
            scores: [n_ty, n_tx] float32 per-tile importance.
        """
        H, W = img_np.shape[:2]

        # Resize to 2048
        from PIL import Image
        scale = 2048 / max(H, W)
        nH, nW = int(H*scale), int(W*scale)
        img_resized = np.array(Image.fromarray(img_np).resize((nW, nH), Image.BILINEAR))
        ph, pw = (32-nH%32)%32, (32-nW%32)%32
        if ph>0 or pw>0:
            img_resized = np.pad(img_resized, ((0,ph),(0,pw),(0,0)), mode="constant")

        img_t = torch.from_numpy(img_resized.astype(np.float32)/255.0)
        img_t = img_t.permute(2,0,1).unsqueeze(0).to(device)  # [1,3,2048,2048]
        imp = self.forward(img_t)  # [1, 1, 64, 64]
        hp, wp = imp.shape[2], imp.shape[3]

        n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W + TILE_SIZE - 1) // TILE_SIZE
        scores = np.zeros((n_ty, n_tx), dtype=np.float32)

        for ty in range(n_ty):
            for tx in range(n_tx):
                y0 = int(ty * TILE_SIZE * scale / 2048 * hp)
                y1 = int(min(ty*TILE_SIZE+TILE_SIZE, H) * scale / 2048 * hp)
                x0 = int(tx * TILE_SIZE * scale / 2048 * wp)
                x1 = int(min(tx*TILE_SIZE+TILE_SIZE, W) * scale / 2048 * wp)
                y0, y1 = max(0, min(y0, hp-1)), max(y0+1, min(y1, hp))
                x0, x1 = max(0, min(x0, wp-1)), max(x0+1, min(x1, wp))
                scores[ty, tx] = float(imp[0, 0, y0:y1, x0:x1].mean())

        return scores


# ═══════════════════════════════════════════════════════════════════
# Step 1: 训练 Decoder | Train Decoder
# ═══════════════════════════════════════════════════════════════════

def train_decoder(args, device):
    """训练 LightDecoder on iSAID tiles | Train LightDecoder on iSAID tiles."""
    from adatile.datasets.isaid_tiles import FastISAIDTileDataset

    logger.log_info("b04/decoder", "=" * 40)
    logger.log_info("b04/decoder",
                    f"Step 1: Training LightDecoder ({args.decoder_epochs} epochs)")

    train_ds = FastISAIDTileDataset(args.tile_root, split="train", semantic=True)
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)

    # 限制训练数据 | Limit training data
    max_tiles = min(len(train_ds), args.train_images * 8)
    train_ds._tiles = train_ds._tiles[:max_tiles]
    logger.log_info("b04/decoder", f"Train tiles: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 模型 | Model
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, NUM_OUT_CH).to(device)
    n_p = sum(p.numel() for p in decoder.parameters())
    logger.log_info("b04/decoder", f"LightDecoder: {n_p:,} params (16ch: bg+15fg)")

    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.decoder_epochs, eta_min=1e-6)

    best_miou, best_state = 0.0, None

    for epoch in range(1, args.decoder_epochs + 1):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"  Dec E{epoch}", leave=False):
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            with torch.no_grad():
                feats = backbone(img)
            logit = decoder(feats["p4"], target_size=tgt.shape[1:])

            # Focal Loss (γ=2.0, α=0.25) — 小目标/难样本聚焦
            # Focal Loss — focus on hard, small-object samples
            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** 2.0  # γ=2.0
            focal_loss = (focal_weight * ce).mean()

            # Dice Loss (多类, 排除背景) — 直接优化 IoU, 不用 one-hot 避免 OOM
            # Multi-class Dice Loss (exclude bg) — no one-hot to avoid OOM
            probs = F.softmax(logit, dim=1)
            dice_sum = 0.0
            valid_dice = 0
            for c in range(1, NUM_OUT_CH):  # exclude background class 0
                p_c = probs[:, c]                    # [B, H, W]
                t_c = (tgt == c).float()             # [B, H, W] (bool→float, no one-hot)
                inter = (p_c * t_c).sum()
                union = p_c.sum() + t_c.sum() + 1e-8
                if t_c.sum() > 0:
                    dice_sum += (2 * inter / union)
                    valid_dice += 1
            dice_loss = 1.0 - (dice_sum / max(valid_dice, 1))

            loss = 0.5 * focal_loss + 0.5 * dice_loss

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)

        # ── 验证 + 诊断 | Validation + Diagnostics ──
        decoder.eval()
        fg_mious, fg_dices = [], []
        all_pred_vals, all_tgt_vals = set(), set()
        total_bg_pred, total_px = 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                tgt = batch["mask"].to(device)
                feats = backbone(img)
                logit = decoder(feats["p4"], target_size=tgt.shape[1:])
                pred = logit.argmax(dim=1)

                # 诊断: 唯一值 | Diagnostic: unique values
                all_pred_vals.update(pred.unique().cpu().tolist())
                all_tgt_vals.update(tgt.unique().cpu().tolist())
                total_bg_pred += (pred == 0).sum().item()
                total_px += pred.numel()

                # FG-mIoU (foreground classes 1-15)
                miou_v = 0.0; valid = 0
                for c in range(1, NUM_CLASSES + 1):
                    pc = (pred == c); tc = (tgt == c)
                    inter = (pc & tc).sum().float()
                    union = (pc | tc).sum().float()
                    if union > 0: miou_v += inter / union; valid += 1
                if valid > 0: fg_mious.append((miou_v / valid).item())

                # Dice per class (foreground only, no one-hot)
                probs_val = F.softmax(logit, dim=1)
                dice_v = 0.0; dv = 0
                for c in range(1, NUM_OUT_CH):
                    pc_v = probs_val[:, c]
                    tc_v = (tgt == c).float()
                    inter_v = (pc_v * tc_v).sum()
                    union_v = pc_v.sum() + tc_v.sum() + 1e-8
                    if tc_v.sum() > 0:
                        dice_v += (2 * inter_v / union_v).item(); dv += 1
                if dv > 0: fg_dices.append(dice_v / dv)

        miou = float(np.mean(fg_mious)) if fg_mious else 0.0
        dice_val = float(np.mean(fg_dices)) if fg_dices else 0.0
        bg_pred_ratio = total_bg_pred / max(total_px, 1)

        # 诊断日志 | Diagnostic log (every 5 epochs or epoch 1)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.decoder_epochs:
            logger.log_info("b04/debug",
                            f"E{epoch:2d} pred.unique={sorted(all_pred_vals)} "
                            f"gt.unique={sorted(all_tgt_vals)} "
                            f"bg_pred_ratio={bg_pred_ratio:.3f}")
        logger.log_info("b04/decoder",
                        f"E{epoch}/{args.decoder_epochs} loss={avg_loss:.4f} "
                        f"FG-mIoU={miou:.4f} Dice={dice_val:.4f} "
                        f"bg_pred%={bg_pred_ratio*100:.1f}%")

        if miou > best_miou:
            best_miou = miou
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}

    if best_state:
        decoder.load_state_dict(best_state)
    logger.log_info("b04/decoder", f"Best decoder mIoU={best_miou:.4f}")
    return decoder, backbone


# ═══════════════════════════════════════════════════════════════════
# Step 2 & 3: FDR + Dynamic Selection + Evaluation
# ═══════════════════════════════════════════════════════════════════

def render_semantic_mask(annotations, h, w):
    import cv2
    sem = np.full((h, w), 255, dtype=np.uint8)  # 255 = ignore
    for ann in annotations:
        cat_id = ACTUAL_TO_CODE_ID.get(ann.get("category_id", 0), 0)
        if cat_id <= 0: continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0,0,0,0])
            sem[max(0,int(bbox[1])):min(h,int(bbox[1]+bbox[3])),
                max(0,int(bbox[0])):min(w,int(bbox[0]+bbox[2]))] = cat_id
            continue
        if isinstance(seg, dict): continue
        for poly in (seg if isinstance(seg[0], list) else [seg]):
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:,:,0] = np.clip(pts[:,:,0], 0, w-1)
            pts[:,:,1] = np.clip(pts[:,:,1], 0, h-1)
            cv2.fillPoly(sem, [pts], cat_id)
    return sem


@torch.no_grad()
def run_dynamic_selection(args, fdr, decoder, backbone, device):
    """端到端动态选择评估 | End-to-end dynamic selection evaluation."""
    logger.log_info("b04/dynamic", "=" * 40)
    logger.log_info("b04/dynamic", "Step 2: Dynamic Tile Selection + Evaluation")

    # 加载测试图片 | Load test images
    src_root = Path(args.src_root)
    with open(src_root/"train"/"annotations"/"instances_train.json") as f:
        coco = json.load(f)

    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # 选验证图片 | Select validation images
    img_dir = src_root / "train" / "images"
    val_images = []
    for img_info in coco["images"][:30]:  # 前 30 张用于快速评估 | First 30 for quick eval
        anns = img_id_to_anns.get(img_info["id"], [])
        if anns:
            img_path = str(img_dir / img_info["file_name"])
            if Path(img_path).exists():
                val_images.append((img_info, img_path, anns))

    logger.log_info("b04/dynamic", f"Test images: {len(val_images)}")

    K_LIST = [int(k) for k in args.top_k_list.split(",")]
    decoder.eval()
    backbone.eval()

    # Per-K 统计 | Per-K statistics
    all_results = {k: {"mious": [], "pred_times": [], "decode_times": [], "n_tiles": []}
                   for k in K_LIST}

    for img_info, img_path, anns in tqdm(val_images, desc="  Dynamic eval"):
        from PIL import Image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H_orig, W_orig = img_np.shape[:2]
        gt_mask = render_semantic_mask(anns, H_orig, W_orig)

        # Resize to max 4096 (OOM protection) | Resize for OOM protection
        max_dim = 3072
        if max(H_orig, W_orig) > max_dim:
            scale = max_dim / max(H_orig, W_orig)
            nH, nW = int(H_orig*scale), int(W_orig*scale)
            img_np = np.array(Image.fromarray(img_np).resize((nW, nH), Image.BILINEAR))
            gt_mask = np.array(Image.fromarray(gt_mask).resize((nW, nH), Image.NEAREST))

        H, W = img_np.shape[:2]
        H_pad = (32 - H % 32) % 32
        W_pad = (32 - W % 32) % 32

        # FDR 预测 tile 重要性 | FDR predicts tile importance
        tile_scores = fdr.predict_tile_scores(img_np, device)
        n_ty, n_tx = tile_scores.shape

        for K in K_LIST:
            if K >= 100:
                selected_mask = np.ones((n_ty, n_tx), dtype=bool)
            else:
                n_keep = max(1, int(n_ty * n_tx * K / 100))
                flat_idx = np.argsort(tile_scores.flatten())[::-1][:n_keep]
                selected_mask = np.zeros((n_ty, n_tx), dtype=bool)
                for fi in flat_idx:
                    selected_mask[fi // n_tx, fi % n_tx] = True

            # 只解码选中 tile | Decode only selected tiles
            pred_full = np.full((H + H_pad, W + W_pad), 0, dtype=np.int64)
            n_selected = 0
            t_decode = 0.0

            for ty in range(n_ty):
                for tx in range(n_tx):
                    if not selected_mask[ty, tx]:
                        continue
                    n_selected += 1

                    y0, y1 = ty*TILE_SIZE, min(ty*TILE_SIZE+TILE_SIZE, H)
                    x0, x1 = tx*TILE_SIZE, min(tx*TILE_SIZE+TILE_SIZE, W)
                    tile_img = img_np[y0:y1, x0:x1]
                    th, tw = tile_img.shape[:2]

                    # Pad tile to 1024
                    if th < TILE_SIZE or tw < TILE_SIZE:
                        padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                        padded[:th, :tw] = tile_img
                        tile_img = padded

                    tile_t = torch.from_numpy(tile_img.astype(np.float32)/255.0)
                    tile_t = tile_t.permute(2, 0, 1).unsqueeze(0).to(device)

                    t0 = time.perf_counter()
                    feats = backbone(tile_t)
                    logit = decoder(feats["p4"], target_size=(TILE_SIZE, TILE_SIZE))
                    pred = logit.argmax(dim=1)[0].cpu().numpy()  # [1024, 1024]
                    t_decode += time.perf_counter() - t0

                    pred_full[y0:y0+min(th, TILE_SIZE), x0:x0+min(tw, TILE_SIZE)] = \
                        pred[:th, :tw]

            # 裁剪 | Crop
            pred_full = pred_full[:H, :W]

            # 计算 mIoU (仅前景类, 排除背景) | Compute mIoU (foreground only)
            miou_v, valid = 0.0, 0
            for c in range(1, NUM_CLASSES + 1):
                pc = (pred_full == c)
                tc = (gt_mask == c)
                inter = (pc & tc).sum()
                union = (pc | tc).sum()
                if union > 0: miou_v += inter/union; valid += 1

            all_results[K]["mious"].append(miou_v/valid if valid > 0 else 0.0)
            all_results[K]["decode_times"].append(t_decode)
            all_results[K]["n_tiles"].append(n_selected)

    # ── 汇总 | Summary ──
    logger.log_info("b04/summary", f"{'='*60}")
    logger.log_info("b04/summary", "B-04 Dynamic Tile Selection — Results")
    logger.log_info("b04/summary",
                    f"  {'K%':<6} {'mIoU':>8} {'ΔmIoU':>8} {'Tiles':>8} "
                    f"{'Decode(ms)':>10} {'Speedup':>8}")

    base_miou = np.mean(all_results[100]["mious"])
    base_time = np.mean(all_results[100]["decode_times"])
    logger.log_info("b04/summary",
                    f"  {'100%':<6} {base_miou*100:>7.2f}% {'-':>8} "
                    f"{int(np.mean(all_results[100]['n_tiles'])):>8} "
                    f"{base_time*1000:>9.1f}ms {'1.00×':>8}")

    for K in sorted(K_LIST):
        if K == 100: continue
        miou = np.mean(all_results[K]["mious"])
        dt = np.mean(all_results[K]["decode_times"])
        nt = int(np.mean(all_results[K]["n_tiles"]))
        dmiou = (miou - base_miou) * 100
        speedup = base_time / max(dt, 1e-8)
        logger.log_info("b04/summary",
                        f"  {K:>4}%  {miou*100:>7.2f}% {dmiou:>+7.2f}% {nt:>8} "
                        f"{dt*1000:>9.1f}ms {speedup:>7.2f}×")

    logger.log_info("b04/summary", f"{'='*60}")

    return all_results, base_miou, base_time


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.log_info("b04/start",
                    f"B-04 Dynamic Tile Selection | "
                    f"decoder_epochs={args.decoder_epochs} "
                    f"top_k={args.top_k_list}")

    # ── Step 1: Train Decoder ──
    if args.skip_decoder_train:
        # 加载已有权重 | Load existing weights
        logger.log_info("b04/decoder", "Skipping decoder training (--skip-decoder-train)")
        decoder = LightDecoder(1280, NUM_OUT_CH).to(device)
        backbone = FastSAMBackbone(freeze_backbone=True).eval()
        ckpt_path = output_dir / "decoder_best.pt"
        if ckpt_path.exists():
            decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.log_info("b04/decoder", f"Loaded decoder from {ckpt_path}")
    else:
        decoder, backbone = train_decoder(args, device)
        torch.save(decoder.state_dict(), output_dir / "decoder_best.pt")

    # ── Step 2: Train FDR ──
    logger.log_info("b04/fdr", f"Step 2: Training FDR ({args.fdr_epochs} epochs)")

    # 复用 B-02/B-03 的 FDR 训练逻辑 (简化版: 全图→tile scores)
    fdr = FDRPredictor().to(device)
    n_fdr = sum(p.numel() for p in fdr.parameters() if p.requires_grad)
    logger.log_info("b04/fdr", f"FDR trainable: {n_fdr:,}")

    # 数据 | Data
    src_root = Path(args.src_root)
    with open(src_root/"train"/"annotations"/"instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    img_dir = src_root / "train" / "images"
    fdr_images = []
    for img_info in coco["images"][:args.train_images]:
        anns = img_id_to_anns.get(img_info["id"], [])
        if anns and (img_dir/img_info["file_name"]).exists():
            fdr_images.append((img_info["file_name"], str(img_dir/img_info["file_name"]),
                              anns))

    logger.log_info("b04/fdr", f"FDR training images: {len(fdr_images)}")

    # 训练 FDR | Train FDR
    opt = torch.optim.Adam(fdr.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.fdr_epochs, eta_min=1e-6)

    for epoch in range(1, args.fdr_epochs + 1):
        fdr.train()
        total_loss, n = 0.0, 0
        for img_id, img_path, anns in tqdm(fdr_images[:50], desc=f"  FDR E{epoch}", leave=False):
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
            H, W = img.shape[:2]
            mask = render_semantic_mask(anns, H, W)

            # Resize + pad
            scale = args.image_size / max(H, W)
            nH, nW = int(H*scale), int(W*scale)
            img_r = np.array(Image.fromarray(img).resize((nW, nH), Image.BILINEAR))
            mask_r = np.array(Image.fromarray(mask).resize((nW, nH), Image.NEAREST))
            ph, pw = (32-nH%32)%32, (32-nW%32)%32
            if ph>0 or pw>0:
                img_r = np.pad(img_r, ((0,ph),(0,pw),(0,0)), mode="constant")
                mask_r = np.pad(mask_r, ((0,ph),(0,pw)), mode="constant", constant_values=255)

            img_t = torch.from_numpy(img_r.astype(np.float32)/255.0)
            img_t = img_t.permute(2,0,1).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_r).to(device)

            # GT tile scores
            H2, W2 = mask_r.shape
            n_ty, n_tx = (H2+TILE_SIZE-1)//TILE_SIZE, (W2+TILE_SIZE-1)//TILE_SIZE
            gts_list = []
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty*TILE_SIZE, min(ty*TILE_SIZE+TILE_SIZE, H2)
                    x0, x1 = tx*TILE_SIZE, min(tx*TILE_SIZE+TILE_SIZE, W2)
                    tm = mask_r[y0:y1, x0:x1]
                    # mask_r 是 numpy, 用 np 计算 GT | mask_r is numpy, compute GT with np
                    fg = float((tm > 0).sum())
                    gts_list.append(fg / max((y1-y0)*(x1-x0), 1))
            gt_scores = torch.tensor(gts_list, dtype=torch.float32, device=device).reshape(n_ty, n_tx)

            imp = fdr(img_t)
            _, _, hp, wp = imp.shape
            preds, gts = [], []
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty*FDR_FEAT_PER_TILE, min(ty*FDR_FEAT_PER_TILE+FDR_FEAT_PER_TILE, hp)
                    x0, x1 = tx*FDR_FEAT_PER_TILE, min(tx*FDR_FEAT_PER_TILE+FDR_FEAT_PER_TILE, wp)
                    if y1>y0 and x1>x0:
                        preds.append(imp[0, 0, y0:y1, x0:x1].mean())
                        gts.append(gt_scores[ty, tx])
            if preds:
                loss = F.mse_loss(torch.stack(preds), torch.stack(gts))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n += 1
        sch.step()
        if epoch%5==0 or epoch==1:
            logger.log_info("b04/fdr", f"FDR E{epoch}/{args.fdr_epochs} loss={total_loss/max(n,1):.4f}")

    # ── Step 3: Dynamic Selection Evaluation ──
    results, base_miou, base_time = run_dynamic_selection(args, fdr, decoder, backbone, device)

    # ── Plot ──
    K_LIST = sorted([int(k) for k in args.top_k_list.split(",")])
    mious = [np.mean(results[k]["mious"])*100 for k in K_LIST]
    times = [np.mean(results[k]["decode_times"])*1000 for k in K_LIST]
    tiles = [int(np.mean(results[k]["n_tiles"])) for k in K_LIST]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(K_LIST, mious, "o-", color="#27AE60", lw=2.5, ms=8, label="FDR Top-K")
    ax.axhline(y=base_miou*100, color="gray", ls="--", alpha=0.5, label=f"All tiles ({base_miou*100:.1f}%)")
    ax.set_xlabel("Tiles Kept (%)"); ax.set_ylabel("mIoU (%)")
    ax.set_title("Accuracy vs Tile Budget"); ax.legend(); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(K_LIST, times, "s-", color="#3498DB", lw=2.5, ms=8)
    ax.set_xlabel("Tiles Kept (%)"); ax.set_ylabel("Decode Time (ms)")
    ax.set_title("Latency vs Tile Budget"); ax.grid(alpha=0.25)

    ax = axes[2]
    savings = [(1 - t/times[-1])*100 for t in times]
    ax.plot(K_LIST, savings, "D-", color="#E74C3C", lw=2.5, ms=8)
    ax.set_xlabel("Tiles Kept (%)"); ax.set_ylabel("Time Saved (%)")
    ax.set_title("Compute Savings"); ax.grid(alpha=0.25)
    # 标注关键点 | Annotate key points
    for k_pct in [40, 50, 70]:
        if k_pct in K_LIST:
            idx = K_LIST.index(k_pct)
            ax.annotate(f"Top{k_pct}%\n{savings[idx]:.0f}% saved",
                       (k_pct, savings[idx]), textcoords="offset points",
                       xytext=(5, 10), fontsize=8)

    fig.suptitle("B-04: Dynamic Tile Selection — FDR + FastSAM Decoder",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir/"dynamic_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Save ──
    summary = {
        "experiment": "B-04 Dynamic Tile Selection",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "base_miou": float(base_miou),
        "per_k_results": {str(k): {"miou": float(np.mean(results[k]["mious"])),
                                     "decode_time_ms": float(np.mean(results[k]["decode_times"])*1000),
                                     "avg_tiles": int(np.mean(results[k]["n_tiles"]))}
                          for k in K_LIST},
    }
    with open(output_dir/"dynamic_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b04/output", f"Saved: {output_dir}/")


if __name__ == "__main__":
    main()
