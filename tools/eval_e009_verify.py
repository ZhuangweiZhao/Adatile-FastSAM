#!/usr/bin/env python3
"""
E009 快速诊断 | Quick Diagnosis: train/val/test 独立对比

用已训练的 SPMHead 在三组数据上分别比较 Learned vs Fixed vs Full。
如果 train Dice 接近 val Dice → 真实提升。
如果 train Dice 正常 (0.46) 而 val Dice 异常高 (0.52) → 过拟合 val / 泄漏。

用法 | Usage:
    python tools/eval_e009_verify.py --checkpoint runs/exp_e009_spm_k4_*/spm_head_s2.pt
"""

import sys, argparse, glob as _glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice
from eval_e009_spm_router import ProtoHead, SPMHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--n-protos", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--router-k", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def eval_on_dataset(spm_head, backbone, dataset, device, args):
    """Evaluate Learned/Fixed/Full Dice on a dataset."""
    spm_head.eval()
    dice_learned, dice_fixed, dice_full = [], [], []
    k = args.router_k

    for idx in tqdm(range(len(dataset)), desc=f"  Eval ({len(dataset)} imgs)", leave=False):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)
        features = backbone(image)
        p4 = features["p4"]

        # Full
        logit_f, _, _ = spm_head.forward_full(p4)
        # Learned
        logit_l, _, _, _ = spm_head.forward_routed(p4, mode="learned", k=k)
        # Fixed
        logit_x, _, _, _ = spm_head.forward_routed(p4, mode="fixed", k=k)

        for logit, dlist in [(logit_f, dice_full), (logit_l, dice_learned), (logit_x, dice_fixed)]:
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
            if pred.dim() == 2: pred = pred.unsqueeze(0)
            gm = gt_mask
            if gm.dim() == 2: gm = gm.unsqueeze(0)
            dlist.append(compute_dice(pred, gm).item())

    return {
        "learned": (float(np.mean(dice_learned)), float(np.std(dice_learned))),
        "fixed":   (float(np.mean(dice_fixed)), float(np.std(dice_fixed))),
        "full":    (float(np.mean(dice_full)), float(np.std(dice_full))),
    }


def main():
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt_path = args.checkpoint
    if "*" in ckpt_path:
        matches = _glob.glob(ckpt_path)
        ckpt_path = matches[0] if matches else ckpt_path
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Reconstruct model
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)
    proto_head.load_state_dict(ckpt["proto_head"])

    # 从 checkpoint 推断 router_arch | Infer router_arch from checkpoint
    router_arch = ckpt.get("router_arch", "conv3x3")
    spm_head = SPMHead(proto_head, n_protos=args.n_protos,
                        router_k=args.router_k,
                        router_arch=router_arch).to(device)
    spm_head.router.load_state_dict(ckpt["router"])

    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # Evaluate on all three splits
    print(f"\n{'=' * 70}")
    print(f"  E009 Verification: Train / Val / Test")
    print(f"  Router K={args.router_k}")
    print(f"  {'=' * 70}")

    for split in ["train", "val", "test"]:
        ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split=split)
        results = eval_on_dataset(spm_head, backbone, ds, device, args)

        l_mean, l_std = results["learned"]
        f_mean, f_std = results["fixed"]
        full_mean, full_std = results["full"]
        delta = l_mean - f_mean
        delta_full = l_mean - full_mean

        print(f"\n  [{split.upper():5s}] {len(ds):3d} images")
        print(f"    Full:    {full_mean:.4f} ± {full_std:.4f}")
        print(f"    Fixed:   {f_mean:.4f} ± {f_std:.4f}")
        print(f"    Learned: {l_mean:.4f} ± {l_std:.4f}")
        print(f"    Δ(L-F):  {delta:+.4f}  |  Δ(L-Full): {delta_full:+.4f}")

    print(f"\n  {'=' * 70}")
    print(f"  解释 | Interpretation:")
    print(f"    - Train/Val/Test 三者一致 → 真实提升")
    print(f"    - Val 异常高但 Train/Test 正常 → 过拟合 val")
    print(f"    - Train 异常高 → 过拟合 train (正常, router 在 train 上训练)")
    print(f"  {'=' * 70}")


if __name__ == "__main__":
    main()
