#!/usr/bin/env python3
"""
E005-B1: Partial Backbone Fine-tune — 解冻 Backbone 最后一个 Block.
=====================================================================

单变量对照实验 | Single-variable controlled experiment:
    E005-D (Frozen Backbone + LightDecoder + CosineLR)
        vs
    E005-B1 (Partial Fine-tune Backbone + LightDecoder + CosineLR)

唯一变量 | Only variable:
    解冻 SPPF (layer [9]) — backbone 最后一个可学习 block (1.03M params)
    Unfreeze SPPF (layer [9]) — backbone's last learnable block.

核心问题 | Core question:
    FastSAM 预训练特征是否已经足够？
    当前 0.628 的限制来自 Backbone 还是 Decoder？

两种情景 | Two scenarios:
    情景 1: 0.628 → 0.72+  → FastSAM 有 Domain Gap → Backbone Adaptation 值得做
    情景 2: 0.628 → 0.64   → FastSAM 已经很好 → 重点转向 Decoder/Proto/SPM

用法 | Usage:
    python tools/eval_e005b1_partial_finetune.py
    python tools/eval_e005b1_partial_finetune.py --epochs 50 --lr 5e-4
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import compute_dice, compute_miou, format_param_count
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder


def parse_args():
    p = argparse.ArgumentParser(description="E005-B1: Partial Backbone Fine-tune")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Initial learning rate (CosineAnnealingLR start)")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum learning rate (CosineAnnealingLR eta_min)")
    p.add_argument("--unfreeze-layers", type=str, default="9",
                   help="Comma-separated layer indices to unfreeze (default: '9' = SPPF only). "
                        "Use '7,9' for Conv+SPPF (4.72M). Use '15' for FPN C2f (1.95M).")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def train_one_epoch(decoder, backbone, ds, optimizer, device):
    """
    训练一个 epoch | Train one epoch.

    与 E005-D 的关键区别 | Key difference from E005-D:
        backbone 需要梯度流（部分层解冻）→ 不用 torch.no_grad()
        Backbone needs gradient flow (partial unfreeze) → no torch.no_grad()
    """
    decoder.train()
    total_loss = 0.0
    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)

        # Backbone forward WITH gradient tracking（部分层解冻需要梯度）
        # Backbone forward WITH gradient tracking (partial unfreeze needs gradients)
        features = backbone(image)

        logit = decoder.forward(features, target_size=tuple(gt_mask.shape[1:]))
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(1), gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ds)


@torch.no_grad()
def evaluate(decoder, backbone, ds, device):
    """评测 | Evaluate. (与 E005-D 完全相同 | Identical to E005-D)"""
    decoder.eval()
    dices, mious = [], []
    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        features = backbone(image)
        pred = decoder.predict(features, target_size=tuple(gt_mask.shape[1:]))
        dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())
        pred_lbl = pred.squeeze(0).squeeze(0).long()
        gt_lbl = gt_mask.squeeze(0).long()
        mious.append(compute_miou(pred_lbl, gt_lbl, num_classes=2)["miou"])
    return {"dice_mean": float(np.mean(dices)), "dice_std": float(np.std(dices)),
            "miou_mean": float(np.mean(mious)), "miou_std": float(np.std(mious))}


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    unfreeze_indices = [int(x.strip()) for x in args.unfreeze_layers.split(",")]

    print("=" * 70)
    print("  E005-B1: Partial Backbone Fine-tune — Frozen vs Partial FT")
    print("  核心问题 | Core: 0.628 的限制来自 Backbone 还是 Decoder?")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e005b1_partial_ft")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # ── Step 1: 加载 Backbone 并全冻结 ──
    print("\n[1/6] Load FastSAM + Full Freeze")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # 触发一次前向以探测 P4/P8 索引 | Trigger forward to probe P4/P8 indices
    print("  Probing P4/P8 hook locations...")
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    with torch.no_grad():
        backbone.forward(dummy)
    print(f"  P4 → layer [{backbone._hook_p4_idx}], P8 → layer [{backbone._hook_p8_idx}]")

    # ── Step 2: 选择性解冻指定层 ──
    print(f"\n[2/6] Partial Unfreeze: layers {unfreeze_indices}")
    sequential = backbone.model.model.model  # YOLOv8 Sequential

    # 先确保全部冻结 | Ensure all frozen first
    for param in backbone.model.model.parameters():
        param.requires_grad = False

    # 解冻指定层 | Unfreeze specified layers
    bb_unfrozen = 0
    for idx in unfreeze_indices:
        layer = sequential[idx]
        for param in layer.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in layer.parameters())
        bb_unfrozen += n
        name = type(layer).__name__
        print(f"  ✓ Layer [{idx}] {name}: {format_param_count(n)} unfrozen")

    # 允许 backbone forward 时启用梯度 | Enable gradient in backbone forward
    backbone._freeze_backbone = False

    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    bb_trainable = sum(p.numel() for p in backbone.model.model.parameters() if p.requires_grad)
    print(f"  Backbone unfrozen/total: {format_param_count(bb_trainable)} / "
          f"{format_param_count(bb_total)} ({100*bb_trainable/bb_total:.1f}%)")
    recorder.logger.log_info("e005b1/unfreeze",
        f"layers={unfreeze_indices}, bb_unfrozen={bb_trainable}, bb_total={bb_total}, "
        f"ratio={bb_trainable/bb_total:.6f}")

    # ── Step 3: LightDecoder ──
    print("\n[3/6] LightDecoder")
    decoder = LightDecoder(in_channels=1280).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {format_param_count(n_dec)} ({n_dec:,})")
    n_trainable = bb_trainable + n_dec
    print(f"  Total trainable: {format_param_count(n_trainable)} "
          f"(Decoder: {format_param_count(n_dec)} + Backbone: {format_param_count(bb_trainable)})")
    recorder.logger.log_info("e005b1/params",
        f"decoder={n_dec}, bb_unfrozen={bb_trainable}, total_trainable={n_trainable}")

    # ── Step 4: Data ──
    print("\n[4/6] Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")

    # ── Step 5: Optimizer + Scheduler (包含 backbone 解冻参数) ──
    print(f"\n[5/6] Optimizer + CosineAnnealingLR (lr={args.lr}→{args.lr_min})")
    # 合并 decoder 和 backbone 解冻参数 | Merge decoder + unfrozen backbone params
    all_params = list(decoder.parameters()) + [
        p for p in backbone.model.model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )
    print(f"  Optimizer param groups: {len(optimizer.param_groups[0]['params'])} tensors")
    print(f"  Scheduler: CosineAnnealingLR(T_max={args.epochs}, eta_min={args.lr_min})")
    recorder.logger.log_info("e005b1/scheduler",
        f"CosineAnnealingLR(T_max={args.epochs}, eta_min={args.lr_min}), "
        f"optim_params={len(optimizer.param_groups[0]['params'])}")

    # ── Step 6: Train ──
    print(f"\n[6/6] Train ({args.epochs} epochs)")
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        current_lr = scheduler.get_last_lr()[0]

        loss = train_one_epoch(decoder, backbone, train_ds, optimizer, device)
        m = evaluate(decoder, backbone, val_ds, device)

        recorder.record_metric("loss/train", loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", m["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", m["miou_mean"], step=epoch, phase="val")
        recorder.record_metric("lr", current_lr, step=epoch, phase="train")

        if m["dice_mean"] > best_dice:
            best_dice = m["dice_mean"]
            # 保存完整 state（decoder + backbone 解冻层）| Save full state (decoder + backbone unfrozen)
            best_state = {
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
                "backbone_unfrozen": {
                    k: v.clone()
                    for k, v in backbone.model.model.state_dict().items()
                    if any(k.startswith(f"model.{idx}.") or k.startswith(f"{idx}.")
                           for idx in unfreeze_indices)
                },
            }

        scheduler.step()

        print(f"  Epoch {epoch:3d}/{args.epochs}  lr={current_lr:.2e}  loss={loss:.4f}  "
              f"Dice={m['dice_mean']:.4f}±{m['dice_std']:.4f}  "
              f"mIoU={m['miou_mean']:.4f}"
              f"{' *' if m['dice_mean'] == best_dice else ''}")

    # ── Final evaluation with best state ──
    decoder.load_state_dict(best_state["decoder"])
    # 恢复 backbone 解冻层的最佳权重 | Restore best weights for unfrozen backbone layers
    backbone.model.model.load_state_dict(best_state["backbone_unfrozen"], strict=False)
    final = evaluate(decoder, backbone, val_ds, device)

    # ── Comparison ──
    e005_best, e005d_best = 0.628, 0.628
    delta_vs_e005 = final["dice_mean"] - e005_best
    delta_vs_e005d = final["dice_mean"] - e005d_best

    print(f"\n{'=' * 70}")
    print(f"  E005-B1 结果 | Results (Partial Fine-tune: layers {unfreeze_indices})")
    print(f"  {'─' * 50}")
    print(f"  Best Dice (val):     {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  Best mIoU (val):     {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  BB unfrozen:         {format_param_count(bb_trainable)} ({100*bb_trainable/bb_total:.1f}%)")
    print(f"  Total trainable:     {format_param_count(n_trainable)}")
    print(f"  ─────────────────────")
    print(f"  E005   Frozen BB + fixed LR:     {e005_best:.3f}")
    print(f"  E005-D Frozen BB + CosineLR:     {e005d_best:.3f}")
    print(f"  E005-B1 Partial FT + CosineLR:   {final['dice_mean']:.3f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E005 (frozen):     {delta_vs_e005:+.4f}")
    print(f"  Δ vs E005-D (frozen+LR): {delta_vs_e005d:+.4f}")
    print(f"  ─────────────────────")

    # 情景解读 | Scenario interpretation
    if delta_vs_e005 > 0.05:
        scenario = "情景 1"
        interpretation = (
            f"  ✓ 提升显著 ({delta_vs_e005:+.3f}) → FastSAM 存在 Domain Gap\n"
            f"    后续路线 → Backbone Adaptation 值得做，可能成为论文创新点\n"
            f"    Scenario 1: Significant gain → Backbone Adaptation is valuable"
        )
    elif delta_vs_e005 > 0.02:
        scenario = "情景 1-2 之间"
        interpretation = (
            f"  △ 有提升 ({delta_vs_e005:+.3f}) → 部分 Domain Gap，Backbone 有改善空间\n"
            f"    后续路线 → 可尝试更激进的解冻策略 (E005-B2)\n"
            f"    Moderate gain → some domain gap, try more aggressive unfreeze"
        )
    else:
        scenario = "情景 2"
        interpretation = (
            f"  → 提升极小 ({delta_vs_e005:+.3f}) → FastSAM Backbone 已经很好\n"
            f"    后续路线 → 重点转向 Decoder / Proto / SPM / AdaTile\n"
            f"    Scenario 2: Minimal gain → backbone is good, focus on decoder side"
        )

    print(f"  【{scenario}】")
    print(interpretation)
    print(f"{'=' * 70}")

    # 记录总结 | Record summary
    recorder.record_metric("e005b1/dice_mean", final["dice_mean"], phase="val",
                           tags=["e005b1", "summary"])
    recorder.record_metric("e005b1/delta_vs_e005", delta_vs_e005,
                           phase="val", tags=["e005b1", "summary"])
    recorder.record_metric("e005b1/delta_vs_e005d", delta_vs_e005d,
                           phase="val", tags=["e005b1", "summary"])
    recorder.record_metric("e005b1/bb_unfrozen_params", bb_trainable,
                           phase="val", tags=["e005b1", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
