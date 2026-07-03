"""
iSAID 实例分割数据集 | iSAID Instance Segmentation Dataset.
=============================================================

支持两种模式 | Two Modes:

1. **Tile Mode (256² tiles)**: 使用 iSAID-5i 预切 tiles，返回 per-instance masks。
   适用于标准 few-shot fine-tuning。每张 tile 可能包含多个类的多个实例。
   Uses iSAID-5i pre-cut tiles, returns per-instance masks.
   Suitable for standard few-shot fine-tuning. Each tile may contain multiple instances
   of multiple classes.

2. **Full-Image Mode (COCO JSON)**: 使用 iSAID COCO 格式全图，支持 tile wrapper。
   适用于高分辨率实例分割。通过 tile wrapper 将大图切分为重叠 tiles。
   Uses iSAID COCO-format full images, supports tile wrapper.
   Suitable for high-resolution instance segmentation via tiling.

返回格式 | Return Format:
    Tile mode: {"image": [3, H, W], "instances": [{"mask": [H,W], "category_id": int, "bbox": [x,y,w,h]}], "image_id": str}
    Full-img:  {"image": [3, H, W], "instances": [...], "image_id": int}

用法 | Usage::

    from adatile.datasets.isaid_instance import ISAIDInstanceDataset

    # Tile mode — few-shot fine-tuning
    ds = ISAIDInstanceDataset(
        root="data/iSAID-5i/iSAID", split="train", fold=0, mode="tile",
    )
    sample = ds[0]
    # sample["instances"] = [{"mask": [256,256] bool, "category_id": 9, "bbox": [x,y,w,h]}, ...]

    # Full-image mode — high-res inference
    ds = ISAIDInstanceDataset(
        root="data/iSAID_processed", split="val", mode="full",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IGNORE_INDEX = 255


# ═══════════════════════════════════════════════════════════════════
# 数据集 | Dataset
# ═══════════════════════════════════════════════════════════════════

class ISAIDInstanceDataset(Dataset):
    """
    iSAID 实例分割数据集 | iSAID Instance Segmentation Dataset.

    加载 iSAID 数据并返回 per-instance masks（每个实例独立掩码），
    而非 per-class union mask（同类实例合并）。
    Loads iSAID data and returns per-instance masks (one per object),
    not per-class union masks (all instances of same class merged).

    Parameters
    ----------
    root : str
        iSAID-5i 数据根目录 | iSAID-5i data root.
    split : str
        "train" 或 "val".
    fold : int
        Fold ID (0/1/2). 必须 ≥0.
    mode : str
        "tile": iSAID-5i 预切 256² tiles (默认) | Pre-cut 256² tiles (default).
        "full": 全图 COCO 格式 | Full-image COCO format.
    novel_ids : list[int] | None
        Novel 类 ID 列表。None → 从 fold 自动获取。
        List of Novel class IDs. None → auto-from fold.
    base_ids : list[int] | None
        Base 类 ID 列表。None → 从 fold 自动获取。
    """

    def __init__(
        self,
        root: str = "data/iSAID-5i/iSAID",
        split: str = "train",
        fold: int = 0,
        mode: str = "tile",
        novel_ids: list[int] | None = None,
        base_ids: list[int] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.mode = mode

        # ── 类别划分 | Class Split ──
        if fold >= 0:
            fold_info = ISAID5I_FOLDS[fold]
            self.novel_ids = set(novel_ids) if novel_ids else set(fold_info["novel"])
            self.base_ids = set(base_ids) if base_ids else set(fold_info["base"])
        else:
            self.novel_ids = set(novel_ids) if novel_ids else set()
            self.base_ids = set(base_ids) if base_ids else set(range(1, 16))

        self.all_class_ids = sorted(self.novel_ids | self.base_ids)

        # ── 路径设置 | Path Setup ──
        if mode == "tile":
            self._img_dir = self.root / split / "images"
            self._mask_dir = self.root / split / "semantic_png"

            if not self._img_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {self._img_dir}")
            if not self._mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {self._mask_dir}")

            # 加载 split 文件 | Load split file
            list_dir = self.root / split / f"{split}_list"
            split_file = list_dir / f"split{fold}_{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file) as f:
                raw_names = [line.strip() for line in f if line.strip()]

            self._tile_names = []
            for raw in raw_names:
                clean = self._clean_tile_name(raw)
                if clean:
                    self._tile_names.append(clean)

            # ── 构建 class→tiles 索引 | Build class→tiles index ──
            self._class_to_tiles: dict[int, list[int]] = {}
            self._tile_class_info: list[dict[int, int]] = []  # [{class_id: instance_count}]
            self._build_tile_index()

        elif mode == "full":
            # 全图模式 | Full-image mode
            self._img_dir = self.root / split / "images"
            anno_path = self.root / split / "annotations" / f"instances_{split}.json"

            if not anno_path.exists():
                raise FileNotFoundError(f"COCO annotation not found: {anno_path}")

            with open(anno_path) as f:
                coco_data = json.load(f)

            self._image_infos = coco_data["images"]
            self._annotations = coco_data["annotations"]
            self._categories = coco_data.get("categories", [])

            # 按 image_id 索引标注 | Index annotations by image_id
            self._ann_by_image: dict[int, list[dict]] = {}
            for ann in self._annotations:
                img_id = ann["image_id"]
                self._ann_by_image.setdefault(img_id, []).append(ann)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'tile' or 'full'.")

        # ── 日志 | Log ──
        n_samples = len(self._tile_names) if mode == "tile" else len(self._image_infos)
        print(f"[ISAIDInstanceDataset] {split}, fold={fold}, mode={mode}: "
              f"{n_samples} samples, "
              f"Base={sorted(self.base_ids)}, Novel={sorted(self.novel_ids)}")

    # ── 索引构建 (Tile 模式) | Index Building (Tile Mode) ──

    def _build_tile_index(self) -> None:
        """
        扫描所有 tile 的语义掩码，建立 class→tiles 映射并统计每 tile 的实例数。
        Scan all tile semantic masks, build class→tiles mapping and per-tile instance counts.

        iSAID-5i 的语义掩码是 per-pixel class ID（0-15），不是 per-instance 掩码。
        我们通过连通分量分析来识别每个类中的独立实例。
        iSAID-5i semantic masks are per-pixel class IDs (0-15), not per-instance masks.
        We use connected component analysis to identify individual instances per class.
        """
        for tile_idx, tile_name in enumerate(self._tile_names):
            mask_path = self._get_mask_path(tile_name)
            if not mask_path.exists():
                self._tile_class_info.append({})
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                self._tile_class_info.append({})
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            tile_classes: dict[int, int] = {}  # {class_id: instance_count}
            for cls_id in self.all_class_ids:
                cls_mask = (mask == cls_id).astype(np.uint8)
                if cls_mask.sum() == 0:
                    continue

                # 连通分量分析 → 实例计数
                # Connected component analysis → instance count
                num_labels, labels = cv2.connectedComponents(cls_mask, connectivity=8)
                n_instances = num_labels - 1  # 减去背景 | Subtract background

                if n_instances > 0:
                    tile_classes[cls_id] = n_instances
                    self._class_to_tiles.setdefault(cls_id, []).append(tile_idx)

            self._tile_class_info.append(tile_classes)

        # ── 日志 | Log ──
        for cls_id in sorted(self._class_to_tiles.keys()):
            cls_name = ISAID5I_CATEGORIES.get(cls_id, f"cls{cls_id}")
            n_tiles = len(self._class_to_tiles[cls_id])
            print(f"  {cls_name} (cls{cls_id}): {n_tiles} tiles")

    # ── 文件名处理 | Filename Handling ──

    @staticmethod
    def _clean_tile_name(raw: str) -> str | None:
        """从 split 文件行提取干净的 tile 名 | Extract clean tile name from split file line."""
        raw = raw.strip()
        if not raw:
            return None
        for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
            idx = raw.find(suffix)
            if idx > 0:
                return raw[:idx]
        return raw.rsplit(".", 1)[0] if "." in raw else raw

    def _get_img_path(self, tile_name: str) -> Path:
        """获取图像路径 | Get image path."""
        p = self._img_dir / f"{tile_name}.png"
        if p.exists():
            return p
        return self._img_dir / f"{tile_name}.jpg"

    def _get_mask_path(self, tile_name: str) -> Path:
        """获取语义掩码路径 | Get semantic mask path."""
        p = self._mask_dir / f"{tile_name}_instance_color_RGB.png"
        if p.exists():
            return p
        p = self._mask_dir / f"{tile_name}.png"
        if p.exists():
            return p
        return self._mask_dir / f"{tile_name}_instance_color_RGB.png"

    # ── 数据加载 | Data Loading ──

    def __len__(self) -> int:
        return len(self._tile_names) if self.mode == "tile" else len(self._image_infos)

    def __getitem__(self, idx: int) -> dict:
        """返回 per-instance 数据 | Returns per-instance data."""
        if self.mode == "tile":
            return self._get_tile_item(idx)
        else:
            return self._get_full_item(idx)

    def _get_tile_item(self, idx: int) -> dict:
        """
        加载 tile 模式的 per-instance 数据 | Load tile-mode per-instance data.

        :return: {"image": [3, 256, 256], "instances": [{"mask": [H,W] bool, "category_id": int, "bbox": [x,y,w,h]}]}
        """
        tile_name = self._tile_names[idx]

        # ── 加载图像 | Load image ──
        img_path = self._get_img_path(tile_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]

        # ── 加载掩码 | Load mask ──
        mask_path = self._get_mask_path(tile_name)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return {"image": img_tensor, "instances": [], "tile_name": tile_name}
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # ── Per-instance mask extraction | 提取每实例掩码 ──
        instances = []
        for cls_id in self.all_class_ids:
            cls_mask = (mask == cls_id).astype(np.uint8)
            if cls_mask.sum() == 0:
                continue

            # 连通分量分解 | Connected component decomposition
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cls_mask, connectivity=8
            )

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < 16:  # 过滤极小噪声 | Filter tiny noise
                    continue

                inst_mask = (labels == label_id)  # [H, W] bool
                x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP], \
                             stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT]

                instances.append({
                    "mask": torch.from_numpy(inst_mask).bool(),
                    "category_id": cls_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(area),
                })

        return {
            "image": img_tensor,
            "instances": instances,
            "tile_name": tile_name,
        }

    def _get_full_item(self, idx: int) -> dict:
        """
        加载全图模式的 per-instance 数据 | Load full-image-mode per-instance data.

        :return: {"image": [3, H, W], "instances": [...], "image_id": int}
        """
        img_info = self._image_infos[idx]
        image_id = img_info["id"]
        h, w = img_info["height"], img_info["width"]

        # ── 加载图像 | Load image ──
        img_name = img_info.get("file_name", f"{image_id}.png")
        img_path = self._img_dir / img_name
        if not img_path.exists():
            img_path = self._img_dir / f"{image_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: tried {self._img_dir / img_name} and {self._img_dir / f'{image_id}.png'}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]

        # ── 提取 per-instance masks | Extract per-instance masks ──
        instances = []
        for ann in self._ann_by_image.get(image_id, []):
            cat_id = ann["category_id"]

            # 过滤非目标类别 | Filter non-target classes
            if cat_id not in self.all_class_ids:
                continue

            # 渲染实例掩码 | Render instance mask
            inst_mask = self._render_mask(ann, h, w)
            if inst_mask.sum() < 16:
                continue

            bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO format: [x, y, w, h]

            instances.append({
                "mask": torch.from_numpy(inst_mask > 0.5).bool(),
                "category_id": cat_id,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "area": float(ann.get("area", inst_mask.sum())),
            })

        return {
            "image": img_tensor,
            "instances": instances,
            "image_id": image_id,
        }

    @staticmethod
    def _render_mask(ann: dict, h: int, w: int) -> np.ndarray:
        """
        渲染单个实例的二值掩码 | Render binary mask for a single instance.

        支持 polygon 和 RLE 分割格式。
        Supports polygon and RLE segmentation formats.

        :param ann: COCO 标注字典 | COCO annotation dict.
        :param h: 图像高度 | Image height.
        :param w: 图像宽度 | Image width.
        :return: [H, W] float32 binary mask.
        """
        mask = np.zeros((h, w), dtype=np.float32)

        seg = ann.get("segmentation", [])
        if not seg:
            # 回退到 bbox | Fallback to bbox
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if bbox[2] > 0 and bbox[3] > 0:
                x, y, bw, bh = [int(v) for v in bbox]
                x, y = max(0, x), max(0, y)
                bw, bh = min(bw, w - x), min(bh, h - y)
                mask[y:y+bh, x:x+bw] = 1.0
            return mask

        try:
            if isinstance(seg, dict):
                # RLE 格式 | RLE format
                return ISAIDInstanceDataset._decode_rle(seg, h, w)
            elif isinstance(seg, list):
                if len(seg) == 0:
                    return mask
                if isinstance(seg[0], list):
                    # 多个 polygon | Multi-polygon
                    polys = seg
                elif isinstance(seg[0], (int, float)):
                    # 单个 polygon | Single polygon: [x1, y1, x2, y2, ...]
                    polys = [seg]
                else:
                    return mask

                for poly in polys:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    # 裁剪到图像边界 | Clip to image boundary
                    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
                    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
                    cv2.fillPoly(mask, [pts], 1.0)
        except Exception:
            pass  # 渲染失败时返回空掩码 | Return empty mask on render failure

        return mask

    @staticmethod
    def _decode_rle(rle: dict, h: int, w: int) -> np.ndarray:
        """
        解码 RLE 分割 | Decode RLE segmentation.

        :param rle: {"size": [H, W], "counts": "..."} RLE dict.
        :param h: 图像高度 | Image height.
        :param w: 图像宽度 | Image width.
        :return: [H, W] float32 binary mask.
        """
        try:
            from pycocotools.mask import decode
            return decode(rle).astype(np.float32).squeeze()
        except ImportError:
            pass

        # 手动解码 | Manual decode (fallback)
        mask = np.zeros((h, w), dtype=np.float32)
        try:
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            # 简单 RLE 解码 (仅支持未压缩) | Simple RLE decode (uncompressed only)
            if isinstance(counts, str) and not counts.startswith("["):
                # 二进制 RLE (pycocotools 格式)
                import struct
                from io import BytesIO
                s = BytesIO()
                # 这是简化实现。完整解码需要 pycocotools。
                # This is simplified. Full decode needs pycocotools.
                pass
        except Exception:
            pass
        return mask

    # ── 查询接口 | Query Interface ──

    def class_to_tiles(self, class_id: int) -> list[int]:
        """
        获取包含指定类的所有 tile 索引 | Get all tile indices containing a given class.

        :param class_id: 类别 ID (1-15) | Class ID (1-15).
        :return: 包含该类的 tile 索引列表 | List of tile indices.
        """
        if self.mode != "tile":
            raise NotImplementedError("class_to_tiles only available in tile mode")
        return self._class_to_tiles.get(class_id, [])

    def tile_instance_count(self, idx: int, class_id: int) -> int:
        """
        获取指定 tile 中某类的实例数 | Get instance count for a class in a tile.

        :param idx: tile 索引 | Tile index.
        :param class_id: 类别 ID | Class ID.
        :return: 实例数 | Instance count.
        """
        if self.mode != "tile":
            raise NotImplementedError("tile_instance_count only available in tile mode")
        return self._tile_class_info[idx].get(class_id, 0)

    def get_novel_tiles(self) -> list[int]:
        """获取包含 Novel 类的所有 tile 索引 | Get all tile indices containing Novel classes."""
        novel_tiles = set()
        for cls_id in self.novel_ids:
            novel_tiles.update(self.class_to_tiles(cls_id))
        return sorted(novel_tiles)

    def get_base_tiles(self) -> list[int]:
        """获取包含 Base 类的所有 tile 索引 | Get all tile indices containing Base classes."""
        base_tiles = set()
        for cls_id in self.base_ids:
            base_tiles.update(self.class_to_tiles(cls_id))
        return sorted(base_tiles)


# ═══════════════════════════════════════════════════════════════════
# Few-Shot 采样辅助 | Few-Shot Sampling Helpers
# ═══════════════════════════════════════════════════════════════════

def sample_k_shot_tiles(
    dataset: ISAIDInstanceDataset,
    class_ids: list[int],
    k: int,
    seed: int = 42,
) -> list[int]:
    """
    为每个类随机采样 K 个 tile（用于 few-shot fine-tuning）。
    Randomly sample K tiles per class (for few-shot fine-tuning).

    确保每个类至少有 K 个 tile（如果可用），取所有选中 tile 的并集。
    Ensures at least K tiles per class (if available), takes union of all selected tiles.

    :param dataset: iSAID 实例数据集 | iSAID instance dataset (tile mode, train only).
    :param class_ids: 要采样的类别 ID 列表 | List of class IDs to sample.
    :param k: 每类最少 tile 数 | Minimum tiles per class.
    :param seed: 随机种子 | Random seed.
    :return: 被选中的 tile 索引列表 | Selected tile index list.
    """
    import random
    rng = random.Random(seed)

    selected: set[int] = set()
    for cls_id in class_ids:
        candidates = dataset.class_to_tiles(cls_id)
        if not candidates:
            print(f"  [WARNING] Class {cls_id}: 0 tiles available!")
            continue
        n_pick = min(k, len(candidates))
        picked = rng.sample(candidates, n_pick)
        selected.update(picked)

    result = sorted(selected)
    print(f"[sample_k_shot_tiles] k={k}, seed={seed}: "
          f"selected {len(result)} tiles across {len(class_ids)} classes")
    return result


def instances_to_dense_mask(
    instances: list[dict],
    h: int,
    w: int,
    ignore_ids: set[int] | None = None,
) -> torch.Tensor:
    """
    将 per-instance 掩码列表转换为 dense category mask。
    Convert per-instance mask list to dense category mask.

    用于训练的损失计算（需要 [H, W] label map）。
    For training loss computation (needs [H, W] label map).

    :param instances: per-instance 字典列表 | List of per-instance dicts.
    :param h: 掩码高度 | Mask height.
    :param w: 掩码宽度 | Mask width.
    :param ignore_ids: 要忽略（设为 255）的类别 ID | Class IDs to ignore (set to 255).
    :return: [H, W] int64 category label map.
    """
    dense = torch.zeros(h, w, dtype=torch.int64)

    for inst in instances:
        cat_id = inst["category_id"]
        if ignore_ids and cat_id in ignore_ids:
            dense[inst["mask"]] = IGNORE_INDEX
        else:
            dense[inst["mask"]] = cat_id

    return dense
