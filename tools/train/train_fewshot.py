#!/usr/bin/env python3
"""
Few-Shot Training Entry Point (glue code) | 少样本训练入口（胶水层）.
=====================================================

复用 eval_c04 的 train_episode/train_and_evaluate/evaluate_full 核心循环。
只需替换 Dataset 为 FewShotEpisodeDataset, 其余不变。

用法 | Usage::
    python tools/train/train_fewshot.py \
        --src-root data/iSAID_processed \
        --fold 0 --shot 1 --epochs 40 \
        --decoder film --feature-level p3p4 \
        --use-dynamic-proto --num-prototypes 4
"""

import sys, argparse, json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.utils.env import get_env_info
from adatile.backbone import FastSAMBackbone
from adatile.utils.label_mapping import ISAID_CATEGORIES
from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper
from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset
from adatile.datasets.fewshot_split import get_novel_classes

# ── 复用 C-04 核心训练循环 | Reuse C-04 core training loop ──
from tools.instance.eval_c04_full_fewshot import (
    build_decoder, train_and_evaluate, evaluate_full,
)
from tools.instance.eval_c02a_fastsam_fewshot import ISAIDInstanceDataset


def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Training Entry Point")
    # ── Data ──
    p.add_argument("--src-root", type=str, required=True,
                   help="iSAID COCO JSON root (e.g. data/iSAID_processed)")
    p.add_argument("--tile-root", type=str, default=None,
                   help="Pre-cut tiles root (e.g. data/iSAID_tiles). "
                   "If set, skips slow dynamic tiling and uses pre-cut tiles directly.")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1, help="K-shot (1/3/5)")
    p.add_argument("--tile-size", type=int, default=896)
    p.add_argument("--tile-stride", type=int, default=512)
    p.add_argument("--crop-support", type=int, default=1, choices=[0, 1],
                   help="ROI crop support images around GT mask (1=on, 0=off)")
    # ── Model ──
    p.add_argument("--decoder", type=str, default="film",
                   choices=["baseline", "film", "crossattn", "contrastive"])
    p.add_argument("--feature-level", type=str, default="p3p4",
                   choices=["p3", "p4", "p8", "p3p4"])
    p.add_argument("--num-prototypes", type=int, default=4)
    p.add_argument("--freeze-encoder", action="store_true", default=True,
                   help="Freeze FastSAM backbone (default: True)")
    p.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank (0=disabled). Placeholder for future.")
    # ── Training ──
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--episodes-per-epoch", type=int, default=150)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val-episodes-per-class", type=int, default=10)
    p.add_argument("--val-batch-size", type=int, default=8,
                   help="验证批大小 (6GB=4, 12GB=12, 24GB=32)")
    p.add_argument("--tile-cache-size", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--swa-start-epoch", type=int, default=0)
    p.add_argument("--use-dynamic-proto", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rur-ceiling", type=str, default="runs/tile_recall_ceiling.json")
    # ── Output ──
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── 预切 Tile 快速适配器 | Pre-Cut Tile Fast Adapter ──
# 当 --tile-root 指定时，跳过 ISAIDTileWrapper 的动态 tile grid 构建，
# 直接从预切好的 tiles 加载，秒级初始化 (vs 动态构建的几分钟)。
# 自动检测两种预切格式:
#   Format A (iSAID_tiles_precut): dict metadata + labels/ + pre-built class_to_tiles
#   Format B (iSAID_tiles): list metadata + masks/ + per-tile class_distribution

class PreCutTileAdapter:
    """
    Lightweight adapter: pre-cut tiles → FewShotEpisodeDataset compatible API.
    轻量适配器: 预切 tiles → FewShotEpisodeDataset 兼容接口.

    提供 class_to_images / load_image / render_class_mask 三个必需方法。
    Auto-detects metadata format (dict vs list).
    """

    def __init__(self, tile_root: str, split: str):
        import cv2
        self._cv2 = cv2
        self.split = split
        self._root = Path(tile_root)

        # 检测格式: dict (precut) vs list (legacy) | Detect format
        meta_path = self._root / "metadata" / f"{split}.json"
        if not meta_path.exists():
            # precut format: metadata.json at split level | 新格式: split 级别
            meta_path = self._root / split / "metadata.json"
            self._fmt = "precut"
        else:
            self._fmt = "legacy"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self._root}/metadata/{split}.json "
                f"or {self._root}/{split}/metadata.json"
            )

        with open(meta_path) as f:
            self._meta = json.load(f)

        # ── 根据格式设置路径和构建 class_to_tiles ──
        from collections import defaultdict

        if self._fmt == "precut":
            # Format A: iSAID_tiles_precut
            # {tile_size, stride, n_tiles, tiles: [{tile_name, classes, ...}], class_to_tiles: {str→[int]}}
            self._img_dir = self._root / split / "images"
            self._mask_dir = self._root / split / "labels"  # "labels" not "masks"!
            self._tile_size = self._meta.get("tile_size", 896)
            self._tiles = self._meta["tiles"]
            # Pre-built class_to_tiles (string keys → convert to int)
            self._cls_to_tiles = {
                int(k): v for k, v in self._meta.get("class_to_tiles", {}).items()
            }
        else:
            # Format B: iSAID_tiles (legacy list format)
            # [{tile_name, img_id, class_distribution: {str→pixels}, ...}, ...]
            self._img_dir = self._root / "images" / split
            self._mask_dir = self._root / "masks" / split
            self._tile_size = 1024  # legacy tiles are 1024×1024
            self._tiles = self._meta  # list of tile dicts
            # Build class_to_tiles from per-tile class_distribution
            self._cls_to_tiles = defaultdict(list)
            for i, tile_info in enumerate(self._meta):
                for cls_str in tile_info.get("class_distribution", {}):
                    self._cls_to_tiles[int(cls_str)].append(i)

        total_tiles = len(self._tiles)
        n_classes = len(self._cls_to_tiles)
        print(f"[PreCutTileAdapter] {split}: {total_tiles} tiles ({self._tile_size}px), "
              f"{n_classes} classes, fmt={self._fmt}")

    def class_to_images(self, class_id: int):
        """class_id → tile index list (兼容 ISAIDTileWrapper API)."""
        return self._cls_to_tiles.get(class_id, [])

    def load_image(self, tile_idx: int) -> torch.Tensor:
        """加载 tile 图像 → [3, H, W] float32 (兼容 ISAIDTileWrapper API)."""
        tile_info = self._tiles[tile_idx]
        if self._fmt == "precut":
            fname = f"{tile_info['tile_name']}.png"
        else:
            fname = tile_info["tile_name"]
        img = self._cv2.imread(str(self._img_dir / fname), self._cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Corrupted image: {fname}")
        img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def render_class_mask(self, tile_idx: int, class_id: int) -> torch.Tensor:
        """渲染指定类别的二值掩码 → [H, W] float32 (兼容 ISAIDTileWrapper API)."""
        tile_info = self._tiles[tile_idx]
        if self._fmt == "precut":
            fname = f"{tile_info['tile_name']}_label.png"
        else:
            fname = tile_info["tile_name"]
        mask = self._cv2.imread(str(self._mask_dir / fname), self._cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Corrupted mask: {fname}")
        # 预切 tile mask 是 dense label (0-15), 提取指定类别即可
        return torch.from_numpy((mask == class_id).astype(np.float32))

    def __len__(self):
        return len(self._tiles)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── Output dir ──
    if args.output_dir is None:
        args.output_dir = (f"runs/fewshot_f{args.fold}_k{args.shot}_"
                          f"{datetime.now().strftime('%m%d_%H%M')}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("fewshot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── Novel classes for this fold ──
    novel_ids = get_novel_classes(args.fold)
    novel_classes = {cid: ISAID_CATEGORIES[cid] for cid in novel_ids
                     if cid in ISAID_CATEGORIES}
    logger.log_info("fewshot/config",
                   f"Fold {args.fold}, Shot={args.shot}, "
                   f"Novel={list(novel_classes.values())}")

    # ── Load datasets ──
    if args.tile_root:
        # 快速路径: 预切 tiles → 秒级初始化 | Fast path: pre-cut tiles → seconds
        logger.log_info("fewshot/data",
                       f"Using pre-cut tiles: {args.tile_root}")
        train_tiles = PreCutTileAdapter(args.tile_root, "train")
        val_tiles = PreCutTileAdapter(args.tile_root, "val")
        # 使用适配器检测到的 tile 尺寸 | Use adapter-detected tile size
        args.tile_size = getattr(train_tiles, '_tile_size', 896)
    else:
        # 原始路径: 全图 → 动态 tile grid (慢) | Original: full-image → dynamic tile grid (slow)
        logger.log_info("fewshot/data",
                       f"Using dynamic tile wrapper (tile={args.tile_size}, stride={args.tile_stride})")
        train_ds = ISAIDInstanceDataset(args.src_root, split="train")
        val_ds = ISAIDInstanceDataset(args.src_root, split="val")
        train_tiles = ISAIDTileWrapper(train_ds, tile_size=args.tile_size,
                                        stride=args.tile_stride)
        val_tiles = ISAIDTileWrapper(val_ds, tile_size=args.tile_size,
                                      stride=args.tile_stride)

    # ── Episode Datasets ──
    train_ep = FewShotEpisodeDataset(
        train_tiles, fold=args.fold, shot=args.shot, split="train",
        episodes_per_epoch=args.episodes_per_epoch, seed=args.seed,
        crop_support=bool(args.crop_support),
    )
    # For validation: use the SAME fold, but val split tiles
    # validate_all_classes_batched needs class_to_images on val tiles
    # We wrap val_tiles with the same episode interface for compat
    val_ep = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=args.shot, split="val",
        episodes_per_epoch=args.eval_episodes, seed=args.seed + 1,
        crop_support=False,
    )

    logger.log_info("fewshot/data",
                   f"Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}")

    # ── Backbone ──
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_encoder).to(device).eval()
    logger.log_info("fewshot/model",
                   f"FastSAM backbone (frozen={args.freeze_encoder}) on {device}")

    # ── Feature dims ──
    with torch.no_grad():
        probe = backbone(torch.randn(1, 3, 896, 896).to(device))
        if args.feature_level == "p3p4":
            feat_dim_p3 = probe["p3"].shape[1]
            feat_dim_p4 = probe["p4"].shape[1]
            feat_dim = feat_dim_p3
        else:
            feat_dim = probe[args.feature_level].shape[1]

    # ── Decoder ──
    decoder_type = args.decoder
    if args.feature_level == "p3p4" and args.decoder == "film":
        decoder_type = "p3p4film"
    decoder_kwargs = {"feat_dim": feat_dim, "num_prototypes": args.num_prototypes}
    if args.feature_level == "p3p4":
        decoder_kwargs["feat_dim_p3"] = feat_dim_p3
        decoder_kwargs["feat_dim_p4"] = feat_dim_p4
    decoder = build_decoder(decoder_type, **decoder_kwargs).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.log_info("fewshot/model", f"Decoder: {n_params:,} trainable params")

    # ── Train (reuse C-04 core loop) ──
    # train_and_evaluate uses: decoder, backbone, train_ds, val_ds, device,
    #   shot, target_classes, args, logger, output_dir, decoder_type, feature_level
    decoder, result, best_val = train_and_evaluate(
        decoder, backbone, train_ep, val_ep, device,
        args.shot, novel_classes, args, logger, out_dir,
        decoder_type=decoder_type, feature_level=args.feature_level,
    )

    # ── Final eval on Novel classes only ──
    logger.log_info("fewshot/eval", "Final evaluation on Novel classes...")
    final_result = evaluate_full(
        decoder, backbone, train_ep, val_ep, device,
        args.shot, args.eval_episodes, novel_classes,
        logger, "fewshot/final", feature_level=args.feature_level,
    )

    # ── Save results ──
    summary = {
        "experiment": "Few-Shot Instance Segmentation on iSAID",
        "fold": args.fold, "shot": args.shot,
        "novel_classes": {str(k): v for k, v in novel_classes.items()},
        "timestamp": datetime.now().isoformat(),
        "environment": get_env_info(),
        "results": {
            "novel": {
                "miou_mean": final_result["miou_mean"],
                "miou_std": final_result["miou_std"],
                "per_class_iou": final_result["per_class_iou"],
            },
            "best_val_miou": best_val,
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.log_info("fewshot/done",
                   f"Novel mIoU={final_result['miou_mean']*100:.2f}%  "
                   f"Saved → {out_dir}/results.json")


if __name__ == "__main__":
    main()
