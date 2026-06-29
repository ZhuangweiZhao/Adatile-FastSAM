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
from adatile.utils.label_mapping import ISAID_CATEGORIES, ISAID5I_CATEGORIES, ISAID5I_FOLDS
from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper
from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset
from adatile.datasets.fewshot_split import get_novel_classes
from adatile.datasets.isaid5i import ISAID5iDataset

# ── 复用 C-04 核心训练循环 | Reuse C-04 core training loop ──
from tools.instance.eval_c04_full_fewshot import (
    build_decoder, train_and_evaluate, evaluate_full,
)
from tools.instance.eval_c02a_fastsam_fewshot import ISAIDInstanceDataset


def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Training Entry Point")
    # ── Data ──
    p.add_argument("--dataset", type=str, default="fastsam",
                   choices=["fastsam", "isaid5i"],
                   help="Dataset protocol: 'fastsam' (896px tiles, custom splits) "
                   "or 'isaid5i' (official iSAID-5i benchmark, 256px tiles, standard folds)")
    p.add_argument("--src-root", type=str, required=True,
                   help="fastsam: iSAID COCO JSON root | isaid5i: iSAID-5i dataset root "
                   "(or any path when using --tile-root with custom tiles)")
    p.add_argument("--tile-root", type=str, default=None,
                   help="Custom pre-cut tiles root. Works with BOTH protocols: "
                   "fastsam: loads via PreCutTileAdapter | isaid5i: uses official folds + custom tiles")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1, help="K-shot (1/3/5)")
    p.add_argument("--tile-size", type=int, default=896)
    p.add_argument("--tile-stride", type=int, default=512)
    p.add_argument("--crop-support", type=int, default=1, choices=[0, 1],
                   help="ROI crop support images around GT mask (1=on, 0=off)")
    # ── Model ──
    p.add_argument("--decoder", type=str, default="film",
                   choices=["baseline", "film", "crossattn", "contrastive",
                            "p3p4film", "p3p4crossattn"])
    p.add_argument("--feature-level", type=str, default="p3p4",
                   choices=["p3", "p4", "p8", "p3p4"])
    p.add_argument("--num-prototypes", type=int, default=4)
    p.add_argument("--freeze-encoder", action="store_true", default=True,
                   help="Freeze FastSAM backbone (default: True)")
    p.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="LoRA rank for FastSAM backbone (0=disabled, 4=light, 8=medium). "
                   "Adds ~50K params per rank to neck layers.")
    # ── Training ──
    p.add_argument("--epochs", type=int, default=60,
                   help="训练轮数 (少量推荐60，充分推荐100)")
    p.add_argument("--episodes-per-epoch", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="权重衰减 (few-shot 推荐 5e-4，减少过拟合)")
    p.add_argument("--warmup-epochs", type=int, default=10,
                   help="学习率预热轮数 (长预热稳定早期训练)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val-episodes-per-class", type=int, default=30,
                   help="每类验证episode数 (30→更可靠的checkpoint选择)")
    p.add_argument("--val-batch-size", type=int, default=2,
                   help="验证批大小 (6GB=1-2, 12GB=6, 24GB=12, 896px建议≤2)")
    p.add_argument("--tile-cache-size", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA衰减率 (0.997→更快适应，适合noisy few-shot)")
    p.add_argument("--swa-start-epoch", type=int, default=0,
                   help="SWA起始轮 (0=auto=60%epochs)")
    p.add_argument("--early-stop-patience", type=int, default=15,
                   help="早停耐心 (0=禁用)")
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

    Tile Cache: 内置 LRU tile 缓存，避免验证时重复磁盘 I/O。
    Built-in LRU tile cache to avoid repeated disk I/O during validation.
    """

    # 类级别缓存容量 | Class-level cache capacity
    _CACHE_MAX = 600  # tiles (~600 × 10MB = 6GB CPU RAM)

    def __init__(self, tile_root: str, split: str):
        import cv2
        self._cv2 = cv2
        self.split = split
        self._root = Path(tile_root)

        # LRU image cache: tile_idx → (image_tensor, mask_tensor)
        self._img_cache: dict[int, torch.Tensor] = {}
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._cache_order: list[int] = []  # 最近使用顺序 | access order (last=most recent)

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
        """加载 tile 图像 → [3, H, W] float32 (兼容 ISAIDTileWrapper API).
        带 LRU 缓存 | With LRU cache."""
        if tile_idx in self._img_cache:
            self._touch_cache(tile_idx)
            return self._img_cache[tile_idx]

        tile_info = self._tiles[tile_idx]
        if self._fmt == "precut":
            fname = f"{tile_info['tile_name']}.png"
        else:
            fname = tile_info["tile_name"]
        img = self._cv2.imread(str(self._img_dir / fname), self._cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Corrupted image: {fname}")
        img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        self._img_cache[tile_idx] = tensor
        self._cache_order.append(tile_idx)
        self._evict_if_needed()
        return tensor

    def render_class_mask(self, tile_idx: int, class_id: int) -> torch.Tensor:
        """渲染指定类别的二值掩码 → [H, W] float32 (兼容 ISAIDTileWrapper API).
        带 LRU 缓存 | With LRU cache (caches raw label, extracts class_id each time)."""
        # 缓存原始 label map | Cache raw label map
        if tile_idx not in self._mask_cache:
            tile_info = self._tiles[tile_idx]
            if self._fmt == "precut":
                fname = f"{tile_info['tile_name']}_label.png"
            else:
                fname = tile_info["tile_name"]
            mask = self._cv2.imread(str(self._mask_dir / fname), self._cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise ValueError(f"Corrupted mask: {fname}")
            self._mask_cache[tile_idx] = torch.from_numpy(mask.astype(np.int64))
            self._cache_order.append(tile_idx)
            self._evict_if_needed()
        else:
            self._touch_cache(tile_idx)

        return (self._mask_cache[tile_idx] == class_id).float()

    def _touch_cache(self, tile_idx: int):
        """将 tile_idx 移到访问顺序末尾 | Move tile_idx to end of access order."""
        if tile_idx in self._cache_order:
            self._cache_order.remove(tile_idx)
        self._cache_order.append(tile_idx)

    def _evict_if_needed(self):
        """No-op: cache cleaned manually each validation."""
        pass

    def clear_cache(self):
        """清空 tile 缓存（每次验证前调用）| Clear tile cache (called before each validation)."""
        self._img_cache.clear()
        self._mask_cache.clear()
        self._cache_order.clear()

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

    # ── Dataset protocol selection | 数据集协议选择 ──
    if args.dataset == "isaid5i":
        # ═══ 标准 iSAID-5i 协议 | Standard iSAID-5i Protocol ═══
        # 使用官方 Fold 划分 + 标准类别 ID
        # --tile-root 可选: 指定自定义尺寸的预切 tiles (如 tile_896)
        # 不指定时: 使用官方 256×256 ISAID5iDataset
        from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS
        CATEGORIES = ISAID5I_CATEGORIES
        novel_ids = ISAID5I_FOLDS[args.fold]["novel"]
        base_ids = ISAID5I_FOLDS[args.fold]["base"]

        if args.tile_root:
            # 自定义 tile 尺寸 + 官方 Fold | Custom tile size + official folds
            logger.log_info("fewshot/config",
                           f"Protocol: iSAID-5i (official folds) + custom tiles | "
                           f"Fold {args.fold}, Shot={args.shot}")
            logger.log_info("fewshot/config",
                           f"Tile root: {args.tile_root}")
            train_tiles = PreCutTileAdapter(args.tile_root, "train")
            val_tiles = PreCutTileAdapter(args.tile_root, "val")
            args.tile_size = getattr(train_tiles, '_tile_size', 896)
        else:
            # 官方 256×256 tiles | Official 256×256 tiles
            args.tile_size = 256
            logger.log_info("fewshot/config",
                           f"Protocol: iSAID-5i (official) | Fold {args.fold}, Shot={args.shot}, "
                           f"Tile={args.tile_size}px")
            train_tiles = ISAID5iDataset(args.src_root, split="train", fold=args.fold)
            val_tiles = ISAID5iDataset(args.src_root, split="val", fold=args.fold)

        logger.log_info("fewshot/config",
                       f"Base classes: {[CATEGORIES[c] for c in base_ids if c in CATEGORIES]}")
        logger.log_info("fewshot/config",
                       f"Novel classes: {[CATEGORIES[c] for c in novel_ids if c in CATEGORIES]}")

    else:
        # ═══ AdaTile-FastSAM 协议 (896px tiles) | AdaTile-FastSAM Protocol ═══
        # .. warning::
        #    此路径使用 ISAID_CATEGORIES ID 体系 + ISAID_FEWSHOT_FOLDS,
        #    Train on Novel + Test on Novel — 非标准 FSS!
        #    仅用于内部快速验证 (debug), 论文正式实验请用 --dataset isaid5i.
        CATEGORIES = ISAID_CATEGORIES
        novel_ids = get_novel_classes(args.fold)
        logger.log_warn("fewshot/config",
                       "⚠ 非标准 FSS 协议: Train-on-Novel + Test-on-Novel. "
                       "论文正式实验请使用 --dataset isaid5i.")
        logger.log_info("fewshot/config",
                       f"Protocol: AdaTile-FastSAM | Fold {args.fold}, Shot={args.shot}")

        if args.tile_root:
            # 快速路径: 预切 tiles → 秒级初始化 | Fast path: pre-cut tiles
            logger.log_info("fewshot/data",
                           f"Using pre-cut tiles: {args.tile_root}")
            train_tiles = PreCutTileAdapter(args.tile_root, "train")
            val_tiles = PreCutTileAdapter(args.tile_root, "val")
            args.tile_size = getattr(train_tiles, '_tile_size', 896)
        else:
            # 原始路径: 全图 → 动态 tile grid (慢) | Original: full-image → dynamic
            logger.log_info("fewshot/data",
                           f"Using dynamic tile wrapper (tile={args.tile_size}, "
                           f"stride={args.tile_stride})")
            train_ds = ISAIDInstanceDataset(args.src_root, split="train")
            val_ds = ISAIDInstanceDataset(args.src_root, split="val")
            train_tiles = ISAIDTileWrapper(train_ds, tile_size=args.tile_size,
                                            stride=args.tile_stride)
            val_tiles = ISAIDTileWrapper(val_ds, tile_size=args.tile_size,
                                          stride=args.tile_stride)

    novel_classes = {cid: CATEGORIES[cid] for cid in novel_ids if cid in CATEGORIES}
    logger.log_info("fewshot/config",
                   f"Novel classes: {list(novel_classes.values())}")

    # ── Episode Datasets ──
    if args.dataset == "isaid5i":
        # iSAID-5i 协议: train_ds 仅含 Base 类 tiles
        # Meta-training: Base 类 episodes (学会 "如何从 Support 学习")
        # Meta-testing:  Novel 类 episodes (评估泛化)
        base_classes = {cid: CATEGORIES[cid] for cid in base_ids if cid in CATEGORIES}
        logger.log_info("fewshot/config",
                       f"Meta-training on Base classes: {list(base_classes.values())}")

        train_ep = FewShotEpisodeDataset(
            train_tiles, fold=args.fold, shot=args.shot, split="train",
            episodes_per_epoch=args.episodes_per_epoch, seed=args.seed,
            crop_support=bool(args.crop_support),
            novel_classes=base_ids,  # ← 训练时采样 Base 类
            category_names=CATEGORIES,
        )
        val_ep = FewShotEpisodeDataset(
            val_tiles, fold=args.fold, shot=args.shot, split="val",
            episodes_per_epoch=args.eval_episodes, seed=args.seed + 1,
            crop_support=False,
            novel_classes=novel_ids,  # ← 验证时采样 Novel 类
            category_names=CATEGORIES,
        )
        # 训练目标: Base 类 | Training target: Base classes
        train_target = base_classes
        eval_target = novel_classes
    else:
        # AdaTile-FastSAM 协议: episodic training on Novel classes
        train_ep = FewShotEpisodeDataset(
            train_tiles, fold=args.fold, shot=args.shot, split="train",
            episodes_per_epoch=args.episodes_per_epoch, seed=args.seed,
            crop_support=bool(args.crop_support),
            novel_classes=novel_ids,
            category_names=CATEGORIES,
        )
        val_ep = FewShotEpisodeDataset(
            val_tiles, fold=args.fold, shot=args.shot, split="val",
            episodes_per_epoch=args.eval_episodes, seed=args.seed + 1,
            crop_support=False,
            novel_classes=novel_ids,
            category_names=CATEGORIES,
        )
        train_target = novel_classes
        eval_target = novel_classes

    logger.log_info("fewshot/data",
                   f"Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}")

    # ── Backbone ──
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_encoder).to(device).eval()

    # ── LoRA | 低秩适配 ──
    if args.lora_rank > 0:
        n_lora = backbone.apply_lora(rank=args.lora_rank)
        logger.log_info("fewshot/model",
                       f"FastSAM backbone (frozen={args.freeze_encoder}, "
                       f"LoRA rank={args.lora_rank}, +{n_lora:,} params) on {device}")
    else:
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
    elif args.feature_level == "p3p4" and args.decoder == "crossattn":
        decoder_type = "p3p4crossattn"
    decoder_kwargs = {"feat_dim": feat_dim, "num_prototypes": args.num_prototypes}
    if args.feature_level == "p3p4":
        decoder_kwargs["feat_dim_p3"] = feat_dim_p3
        decoder_kwargs["feat_dim_p4"] = feat_dim_p4
    decoder = build_decoder(decoder_type, **decoder_kwargs).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.log_info("fewshot/model", f"Decoder: {n_params:,} trainable params")

    # ── Train (reuse C-04 core loop) ──
    decoder, result, best_val = train_and_evaluate(
        decoder, backbone, train_ep, val_ep, device,
        args.shot, train_target, args, logger, out_dir,
        decoder_type=decoder_type, feature_level=args.feature_level,
    )

    # ── Final eval on Novel classes ──
    if args.dataset == "isaid5i":
        logger.log_info("fewshot/eval", "Final evaluation on Novel classes (iSAID-5i protocol)...")
        # iSAID-5i: train_ds 不含 Novel 类，评估时 support 和 query 都从 val_ds
        # evaluate_full 内部会从 val_ds 采样 support+query (train_ds 用于获取 class_to_images)
        final_result = evaluate_full(
            decoder, backbone, val_ep, val_ep, device,
            args.shot, args.eval_episodes, eval_target,
            logger, "fewshot/final", feature_level=args.feature_level,
        )
    else:
        logger.log_info("fewshot/eval", "Final evaluation on Novel classes...")
        final_result = evaluate_full(
            decoder, backbone, train_ep, val_ep, device,
            args.shot, args.eval_episodes, eval_target,
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
