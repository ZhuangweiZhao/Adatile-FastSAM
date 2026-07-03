#!/usr/bin/env python3
"""
Novel 类 Few-Shot 评估 | Novel Class Few-Shot Evaluation.
==========================================================

在 Base 类全量训练完成后，测试 Novel 类的 K-shot 分割能力。
After full Base-class training, evaluate K-shot segmentation on Novel classes.

方法 | Method: Prototype Matching
    K 张 support tile → 提取 P4/P3 特征 → FG prototype (均值) → 与 query 做余弦相似度匹配
    K support tiles → extract P4/P3 features → FG prototype (mean) → cosine similarity on query

用法 | Usage::
    # 使用 P3+P4 [C] checkpoint, Fold 0, 1-shot
    python tools/eval/eval_novel_fewshot.py \
        --ckpt runs/supervised_C_p3p4_nocb_256/best_model.pt \
        --fold 0 --k-shot 1

    # P4-only, 5-shot, 多 seed 平均
    python tools/eval/eval_novel_fewshot.py \
        --ckpt runs/supervised_H_full_p4_nocb_256/best_model.pt \
        --fold 0 --k-shot 5 --seeds 3

    # P3+P4 多尺度 prototype
    python tools/eval/eval_novel_fewshot.py \
        --ckpt runs/supervised_C_p3p4_nocb_256/best_model.pt \
        --fold 0 --k-shot 1 --use-p3
"""

from __future__ import annotations

import sys, argparse, json, os
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder, LightDecoderP3, LightDecoderP3P4
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "iSAID-5i" / "iSAID")
NUM_CLASSES = 15
NUM_OUT_CH = 16
IGNORE_INDEX = 255


# ═══════════════════════════════════════════════════════════════════
# Novel 类查询数据集 | Novel Class Query Dataset
# ═══════════════════════════════════════════════════════════════════

class NovelQueryDataset(Dataset):
    """
    加载指定 Novel 类的所有 val tile | Load all val tiles for a specific Novel class.
    """

    def __init__(self, data_root: str, fold: int, novel_class_id: int):
        self.root = Path(data_root)
        self.novel_class_id = novel_class_id

        # ── 加载 val split | Load val split ──
        list_file = self.root / "val" / "val_list" / f"split{fold}_val.txt"
        with open(list_file) as f:
            raw_names = [line.strip() for line in f if line.strip()]

        self._tile_names = []
        for raw in raw_names:
            clean = self._clean_tile_name(raw)
            if clean:
                self._tile_names.append(clean)

        # ── 筛选: 只保留含该 Novel 类的 tile | Filter: only tiles containing the Novel class ──
        mask_dir = self.root / "val" / "semantic_png"
        self._samples = []  # [(img_path, mask_path)]
        for name in self._tile_names:
            mask_path = self._find_mask(mask_dir, name)
            if not mask_path:
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if novel_class_id not in np.unique(mask):
                continue
            img_path = self._find_img(name)
            if img_path:
                self._samples.append((str(img_path), str(mask_path)))

    @staticmethod
    def _clean_tile_name(raw: str) -> str | None:
        raw = raw.strip()
        for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
            idx = raw.find(suffix)
            if idx > 0:
                return raw[:idx]
        return raw.rsplit(".", 1)[0] if "." in raw else raw

    def _find_img(self, tile_name: str) -> Path | None:
        for ext in [".png", ".jpg"]:
            p = self.root / "val" / "images" / f"{tile_name}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _find_mask(mask_dir: Path, tile_name: str) -> Path | None:
        for suffix in ["_instance_color_RGB.png", ".png"]:
            p = mask_dir / f"{tile_name}{suffix}"
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        img_path, mask_path = self._samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # ── 预处理 | Preprocessing ──
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5  # normalize to [-1, 1]

        # Binary mask: this Novel class = FG, everything else = BG
        binary_mask = (mask == self.novel_class_id).astype(np.float32)
        mask_t = torch.from_numpy(binary_mask)

        return {
            "image": img_t,
            "mask": mask_t,
            "image_id": Path(img_path).stem,
        }


# ═══════════════════════════════════════════════════════════════════
# Support 集采样 | Support Set Sampling
# ═══════════════════════════════════════════════════════════════════

def sample_support_tiles(
    data_root: str,
    fold: int,
    novel_class_id: int,
    k: int,
    seed: int,
) -> list[dict]:
    """
    从 train 集随机采样 K 张含指定 Novel 类的 tile 作为 support。
    Randomly sample K support tiles containing a specific Novel class from train set.

    :return: [{"image": tensor[3,H,W], "mask": tensor[H,W]}, ...]
    """
    import random
    rng = random.Random(seed)

    root = Path(data_root)
    list_file = root / "train" / "train_list" / f"split{fold}_train.txt"
    with open(list_file) as f:
        raw_names = [line.strip() for line in f if line.strip()]

    tile_names = []
    for raw in raw_names:
        raw = raw.strip()
        for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
            idx = raw.find(suffix)
            if idx > 0:
                tile_names.append(raw[:idx])
                break
        else:
            tile_names.append(raw.rsplit(".", 1)[0] if "." in raw else raw)

    # ── 找含该 Novel 类的 tile | Find tiles containing the Novel class ──
    mask_dir = root / "train" / "semantic_png"
    img_dir = root / "train" / "images"
    candidates = []
    for name in tile_names:
        for suffix in ["_instance_color_RGB.png", ".png"]:
            mask_path = mask_dir / f"{name}{suffix}"
            if mask_path.exists():
                break
        else:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if novel_class_id not in np.unique(mask):
            continue

        # 找图片 | Find image
        img_path = None
        for ext in [".png", ".jpg"]:
            p = img_dir / f"{name}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        binary_mask = (mask == novel_class_id).astype(np.float32)
        candidates.append({"img_path": str(img_path), "mask": binary_mask, "name": name})

    if not candidates:
        return []

    n_pick = min(k, len(candidates))
    picked = rng.sample(candidates, n_pick)

    # ── 加载图像 | Load images ──
    supports = []
    for item in picked:
        img = cv2.imread(item["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        mask_t = torch.from_numpy(item["mask"])
        supports.append({"image": img_t, "mask": mask_t, "name": item["name"]})

    return supports


# ═══════════════════════════════════════════════════════════════════
# Prototype 计算 | Prototype Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_prototype(
    backbone: FastSAMBackbone,
    supports: list[dict],
    device: torch.device,
    use_p3: bool = False,
) -> dict[str, torch.Tensor]:
    """
    从 K 张 support tile 计算 FG prototype。
    Compute FG prototype from K support tiles.

    :return: {"p4": [1280] or None, "p3": [960] or None}
    """
    all_p4 = []
    all_p3 = []

    for s in supports:
        img = s["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        mask = s["mask"].to(device)                # [H, W]

        feats = backbone(img)

        # P4 prototype
        f_p4 = feats["p4"]  # [1, 1280, H/16, W/16]
        mask_p4 = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                size=f_p4.shape[2:], mode="nearest").squeeze(0).squeeze(0)
        fg_mask_p4 = (mask_p4 > 0.5)
        if fg_mask_p4.sum() > 0:
            proto_p4 = f_p4[:, :, fg_mask_p4].mean(dim=-1).squeeze(0)  # [1280]
            all_p4.append(proto_p4)

        # P3 prototype
        if use_p3 and "p3" in feats:
            f_p3 = feats["p3"]  # [1, 960, H/8, W/8]
            mask_p3 = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                    size=f_p3.shape[2:], mode="nearest").squeeze(0).squeeze(0)
            fg_mask_p3 = (mask_p3 > 0.5)
            if fg_mask_p3.sum() > 0:
                proto_p3 = f_p3[:, :, fg_mask_p3].mean(dim=-1).squeeze(0)  # [960]
                all_p3.append(proto_p3)

    result = {}
    if all_p4:
        result["p4"] = torch.stack(all_p4).mean(dim=0)  # [1280]
    else:
        result["p4"] = None

    if all_p3:
        result["p3"] = torch.stack(all_p3).mean(dim=0)  # [960]
    else:
        result["p3"] = None

    return result


# ═══════════════════════════════════════════════════════════════════
# Prototype 匹配推理 | Prototype Matching Inference
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def prototype_match(
    backbone: FastSAMBackbone,
    query_img: torch.Tensor,
    prototype: dict[str, torch.Tensor | None],
    device: torch.device,
) -> torch.Tensor:
    """
    用 cosine similarity 匹配 prototype，输出二值 mask。
    Match prototype via cosine similarity, output binary mask.

    :return: binary mask [256, 256]
    """
    img = query_img.unsqueeze(0).to(device)
    feats = backbone(img)

    # ── P4 matching ──
    f_p4 = feats["p4"].squeeze(0)  # [1280, H/16, W/16]
    proto_p4 = prototype.get("p4")

    if proto_p4 is not None:
        # Cosine similarity
        f_p4_norm = F.normalize(f_p4, dim=0)  # normalize along channel dim
        proto_norm = F.normalize(proto_p4, dim=0)
        sim_p4 = torch.einsum("c,chw->hw", proto_norm, f_p4_norm)  # [H/16, W/16]
        sim_p4 = sim_p4.unsqueeze(0).unsqueeze(0)  # [1, 1, H/16, W/16]
        pred_p4 = F.interpolate(sim_p4, size=(256, 256), mode="bilinear",
                                align_corners=False).squeeze()  # [256, 256]
    else:
        pred_p4 = torch.zeros(256, 256, device=device)

    # ── P3 matching ──
    proto_p3 = prototype.get("p3")
    if proto_p3 is not None and "p3" in feats:
        f_p3 = feats["p3"].squeeze(0)  # [960, H/8, W/8]
        f_p3_norm = F.normalize(f_p3, dim=0)
        proto_norm3 = F.normalize(proto_p3, dim=0)
        sim_p3 = torch.einsum("c,chw->hw", proto_norm3, f_p3_norm)
        sim_p3 = sim_p3.unsqueeze(0).unsqueeze(0)
        pred_p3 = F.interpolate(sim_p3, size=(256, 256), mode="bilinear",
                                align_corners=False).squeeze()
    else:
        pred_p3 = torch.zeros(256, 256, device=device)

    # ── 融合: P4 + P3 (可选) | Fusion: P4 + P3 (if available) ──
    if proto_p3 is not None:
        pred = (pred_p4 + pred_p3) / 2.0
    else:
        pred = pred_p4

    # ── 二值化: 阈值 0 | Binarize: threshold at 0 ──
    binary = (pred > 0).float()
    return binary


# ═══════════════════════════════════════════════════════════════════
# IoU 计算 | IoU Computation
# ═══════════════════════════════════════════════════════════════════

def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """计算二值分割的 IoU | Compute binary segmentation IoU."""
    pred = pred.bool()
    gt = gt.bool()
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    if union == 0:
        return 0.0
    return (intersection / union).item()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Novel 类 Few-Shot 评估 | Novel Class Few-Shot Evaluation")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Base 类训练好的 checkpoint 路径 | Path to Base-trained checkpoint")
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="iSAID-5i 数据根目录 | Data root")
    p.add_argument("--fold", type=int, default=0,
                   help="Fold ID (0/1/2)")
    p.add_argument("--k-shot", type=int, default=1,
                   help="每类 support 数量 | Number of support examples per class")
    p.add_argument("--seeds", type=int, default=1,
                   help="重复次数 (不同 random seed 平均) | Number of trials with different seeds")
    p.add_argument("--use-p3", action="store_true",
                   help="使用 P3+P4 多尺度 prototype | Use P3+P4 multi-scale prototype")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="设备 | Device")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Query 推理 batch size | Query inference batch size")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(42)

    novel_ids = ISAID5I_FOLDS[args.fold]["novel"]
    novel_names = {c: ISAID5I_CATEGORIES.get(c, f"class_{c}") for c in novel_ids}

    print("=" * 80)
    print(f"Novel Class Few-Shot Evaluation | Novel 类少样本评估")
    print(f"  Fold {args.fold}, K={args.k_shot}-shot, seeds={args.seeds}")
    print(f"  Novel classes: {novel_names}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Multi-scale: {'P3+P4' if args.use_p3 else 'P4-only'}")
    print("=" * 80)

    # ── 加载模型 | Load model ──
    print("\nLoading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # 判断 decoder 类型 | Determine decoder type
    # checkpoint 的 config 存在 "args" 键 (vars(parse_args())) | config is in "args" key
    decoder_config = ckpt.get("args", {})
    use_p3_decoder = decoder_config.get("use_p3", False)
    use_p3_only = decoder_config.get("use_p3_only", False)

    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    if use_p3_only:
        decoder = LightDecoderP3(in_channels=960, num_classes=NUM_OUT_CH).to(device)
    elif use_p3_decoder:
        decoder = LightDecoderP3P4(num_classes=NUM_OUT_CH).to(device)
    else:
        decoder = LightDecoder(in_channels=1280, num_classes=NUM_OUT_CH).to(device)

    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    print(f"  Decoder type: {type(decoder).__name__}")

    # ── 逐 Novel 类评估 | Per-Novel-Class Evaluation ──
    all_results = {}  # {class_id: {k_shot: [iou_per_seed]}}

    for novel_id in novel_ids:
        cls_name = novel_names[novel_id]
        print(f"\n{'─' * 60}")
        print(f"Class {novel_id}: {cls_name}")
        print(f"{'─' * 60}")

        # ── 加载 query 集 | Load query set ──
        query_ds = NovelQueryDataset(args.data_root, args.fold, novel_id)
        print(f"  Query tiles: {len(query_ds)}")

        if len(query_ds) == 0:
            print(f"  WARNING: No query tiles found for class {novel_id}, skipping")
            all_results[novel_id] = []
            continue

        query_loader = DataLoader(query_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # ── 多 seed 平均 | Multi-seed evaluation ──
        seed_ious = []

        for seed_idx in range(args.seeds):
            support_seed = 100 + seed_idx * 7  # 避免与训练 seed 冲突

            # 采样 support | Sample support
            supports = sample_support_tiles(
                args.data_root, args.fold, novel_id, args.k_shot, support_seed)
            n_support = len(supports)
            if n_support == 0:
                print(f"  Seed {seed_idx}: No support tiles found, skipping")
                continue
            print(f"  Seed {seed_idx}: {n_support} support tiles "
                  f"(sampled from {n_support} available)")

            # 计算 prototype | Compute prototype
            proto = compute_prototype(backbone, supports, device, use_p3=args.use_p3)
            has_p4 = proto["p4"] is not None
            has_p3 = proto["p3"] is not None
            print(f"    Prototype: P4={'OK' if has_p4 else 'N/A'}, "
                  f"P3={'OK' if has_p3 else 'N/A'}")

            if not has_p4 and not has_p3:
                print(f"    WARNING: No valid prototype, mIoU=0")
                seed_ious.append(0.0)
                continue

            # ── 在 query 集上评估 | Evaluate on query set ──
            query_ious = []
            for batch in tqdm(query_loader, desc=f"    Query eval", leave=False):
                for i in range(len(batch["image"])):
                    pred = prototype_match(backbone, batch["image"][i], proto, device)
                    iou = compute_iou(pred.cpu(), batch["mask"][i])
                    query_ious.append(iou)

            mean_iou = np.mean(query_ious) if query_ious else 0.0
            seed_ious.append(mean_iou)
            print(f"    mIoU = {mean_iou:.4f}")

        avg_iou = np.mean(seed_ious) if seed_ious else 0.0
        std_iou = np.std(seed_ious) if len(seed_ious) > 1 else 0.0
        print(f"  → Class {cls_name}: mIoU = {avg_iou:.4f} ± {std_iou:.4f} "
              f"(over {len(seed_ious)} seeds)")
        all_results[novel_id] = {"miou": avg_iou, "std": std_iou, "seeds": seed_ious}

    # ── 汇总 | Summary ──
    print(f"\n{'=' * 80}")
    print("Summary | 汇总")
    print(f"{'=' * 80}")
    print(f"{'ID':>3s}  {'Class':<22s}  {'mIoU':>8s}  {'±Std':>8s}")
    print(f"{'─' * 50}")

    valid_mious = []
    for novel_id in novel_ids:
        cls_name = novel_names[novel_id]
        r = all_results.get(novel_id, {})
        miou = r.get("miou", 0.0)
        std = r.get("std", 0.0)
        print(f"{novel_id:>3d}  {cls_name:<22s}  {miou:>8.4f}  {std:>8.4f}")
        if miou > 0 or r:
            valid_mious.append(miou)

    mean_miou = np.mean(valid_mious) if valid_mious else 0.0
    print(f"{'─' * 50}")
    print(f"  {'Mean Novel mIoU':<22s}  {mean_miou:>8.4f}")
    print(f"{'=' * 80}")

    # ── 保存结果 | Save results ──
    ckpt_dir = Path(args.ckpt).parent.name
    out_file = Path("runs") / f"novel_fewshot_{ckpt_dir}_k{args.k_shot}_fold{args.fold}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "fold": args.fold,
        "k_shot": args.k_shot,
        "num_seeds": args.seeds,
        "use_p3": args.use_p3,
        "checkpoint": str(args.ckpt),
        "novel_classes": {str(k): novel_names[k] for k in novel_ids},
        "per_class": {str(k): v for k, v in all_results.items()},
        "mean_novel_miou": float(mean_miou),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    main()
