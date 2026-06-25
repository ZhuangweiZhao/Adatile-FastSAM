"""
ISAIDDataset — iSAID 航拍实例分割数据集加载器。
=====================================================
iSAID aerial instance segmentation dataset loader.

15 个类别，图像分辨率 800×800 到 4000×4000 像素。
15 classes, image resolution 800×800 to 4000×4000 pixels.

基于 COCO JSON 格式标注。
Based on COCO JSON format annotations.

数据集结构 | Dataset structure:
    iSAID/
    ├── train/
    │   ├── images/              # *.png 航拍图像 | Aerial images
    │   └── annotations/
    │       └── instances_train.json  # COCO format
    ├── val/
    │   ├── images/
    │   └── annotations/
    │       └── instances_val.json
    └── test/
        └── images/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from adatile.datasets.base import BaseSegDataset
from adatile.logging import get_logger


# ── iSAID 15 个类别 | 15 iSAID Categories ────────────────────

ISAID_CATEGORIES: list[dict] = [
    {"id": 1, "name": "small_vehicle"},
    {"id": 2, "name": "large_vehicle"},
    {"id": 3, "name": "plane"},
    {"id": 4, "name": "storage_tank"},
    {"id": 5, "name": "ship"},
    {"id": 6, "name": "harbor"},
    {"id": 7, "name": "ground_track_field"},
    {"id": 8, "name": "soccer_ball_field"},
    {"id": 9, "name": "tennis_court"},
    {"id": 10, "name": "swimming_pool"},
    {"id": 11, "name": "road"},
    {"id": 12, "name": "basketball_court"},
    {"id": 13, "name": "bridge"},
    {"id": 14, "name": "helicopter"},
    {"id": 15, "name": "roundabout"},
]

# 类别 ID → 名称映射 | Category ID → name mapping
_CAT_ID_TO_NAME: dict[int, str] = {c["id"]: c["name"] for c in ISAID_CATEGORIES}
_CAT_NAME_TO_ID: dict[str, int] = {c["name"]: c["id"] for c in ISAID_CATEGORIES}


class ISAIDDataset(BaseSegDataset):
    """
    iSAID 航拍实例分割数据集 | iSAID Aerial Instance Segmentation Dataset.

    特性 | Features:
        - COCO JSON 格式标注解析 | COCO JSON annotation parsing
        - 大图瓦片处理 | Large-image tile-based processing
        - 掩码归一化（{0,255} → {0,1}）| Mask normalization
        - 密集标签模式：返回 [H,W] 类别标签 | Dense label mode: returns [H,W] class labels
        - 日志系统集成 | Logging system integration

    两种输出模式 | Two output modes:
        dense_labels=False (默认 | default):
            sample["masks"] → [N_inst, H, W] 实例二值掩码 | instance binary masks
        dense_labels=True:
            sample["mask"]  → [H, W] 密集类别标签 | dense category labels (0=bg, 1-15)
            sample["masks"] → 不可用 | not available

    ----------
    root_dir : str
        iSAID 数据集根目录 | iSAID dataset root directory.
        期望结构见 prep_isaid.py | Expected structure: see prep_isaid.py.
    split : str
        数据集划分 ("train", "val") | Dataset split.
    tile_size : int | None
        瓦片尺寸。None = 全图加载。大图建议 1024。
        Tile size. None = full-image loading. Recommended 1024 for large images.
    tile_overlap : float
        瓦片重叠比例 (0.0 ~ 1.0) | Tile overlap ratio.
    dense_labels : bool
        True = 密集标签模式 (返回[H,W]类别标签) | dense label mode (returns [H,W] category labels).
    transforms : callable | None
        数据增强变换 | Optional data augmentation transforms.
    """

    def __init__(
        self,
        root_dir: str = "datasets/iSAID",
        split: str = "train",
        tile_size: int | None = None,
        tile_overlap: float = 0.0,
        dense_labels: bool = False,
        transforms=None,
    ) -> None:
        """
        ----------
        dense_labels : bool
            True = 返回密集类别标签 [H, W] (category IDs)
                   Return dense category label map [H, W] (category IDs).
            False = 返回实例掩码 [N, H, W] (二值 per-instance)
                    Return instance masks [N, H, W] (binary per-instance).
        """
        super().__init__(root_dir=root_dir, split=split, transforms=transforms)
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        # 密集标签模式：返回 [H,W] 类别标签 而非 [N,H,W] 实例掩码
        # Dense label mode: return [H,W] category labels instead of [N,H,W] instance masks
        self._dense_labels = dense_labels

        # 加载 COCO 标注 | Load COCO annotations
        self._coco_data = self._load_coco()
        self._image_infos: list[dict] = self._coco_data["images"]
        self._annotations: list[dict] = self._coco_data["annotations"]
        self._categories: list[dict] = self._coco_data.get("categories", ISAID_CATEGORIES)

        # 构建 image_id → annotations 索引 | Build image_id → annotations index
        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in self._annotations:
            img_id = ann["image_id"]
            self._ann_by_image.setdefault(img_id, []).append(ann)

        # 图像目录 | Image directory
        self._img_dir = self.root_dir / self.split / "images"

        # 预计算 tile 索引（仅在 tile 模式）| Precompute tile index (tile mode only)
        self._tile_index: list[tuple[int, int, int, int]] = []  # (img_idx, x, y, tile_idx)
        if self._tile_size is not None:
            self._build_tile_index()

        # 日志数据集统计 | Log dataset statistics
        self.logger.log_info(
            "dataset/isaid_init",
            f"iSAID {split}: {len(self)} images, "
            f"{len(self._annotations)} instances, "
            f"{len(self._categories)} categories, "
            f"tile_size={tile_size}, "
            f"dense_labels={dense_labels}",
        )

    def _build_tile_index(self) -> None:
        """
        构建瓦片索引：将每张图切分为瓦片，记录每个瓦片的位置。
        Build tile index: split each image into tiles, record each tile position.
        """
        stride = int(self._tile_size * (1 - self._tile_overlap))
        for img_idx, info in enumerate(self._image_infos):
            h, w = info["height"], info["width"]
            tile_idx = 0
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    self._tile_index.append((img_idx, x, y, tile_idx))
                    tile_idx += 1

    def _extract_tile(
        self, image: torch.Tensor, masks: torch.Tensor, x: int, y: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从全图中提取瓦片 | Extract tile from full image.
        边界瓦片自动补零到 tile_size × tile_size (YOLOv8 要求尺寸一致)。
        Edge tiles auto-padded to tile_size × tile_size for YOLOv8 compatibility.

        :param image: [C, H, W] 全图 | Full image.
        :type image: torch.Tensor

        :param masks: [N, H, W] 全尺寸掩码 | Full-size masks. x, y:  瓦片左上角坐标 | Tile top-left coordinates.
        :type masks: torch.Tensor

        :param x: 
        :type x: int

        :param y: 
        :type y: int

        :return: (tile_image [C, ts, ts], tile_masks [N, ts, ts])
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        ts = self._tile_size
        _, h, w = image.shape
        th = min(ts, h - y)
        tw = min(ts, w - x)

        # 提取实际像素 | Extract actual pixels
        tile_img = image[:, y:y + th, x:x + tw]
        tile_masks = masks[:, y:y + th, x:x + tw]

        # 边界瓦片补零 | Pad edge tiles to ts × ts
        if th < ts or tw < ts:
            pad_h = ts - th
            pad_w = ts - tw
            tile_img = F.pad(tile_img, (0, pad_w, 0, pad_h), value=0)
            if tile_masks.dim() == 3:
                tile_masks = F.pad(tile_masks, (0, pad_w, 0, pad_h), value=0)

        return tile_img, tile_masks

    # ── 抽象方法实现 | Abstract Method Implementations ────────

    def _load_image(self, index: int) -> torch.Tensor:
        """
        加载并归一化图像 [C, H, W] | Load and normalize image.

        读取 PNG → RGB → float32 [0, 1]。
        """
        info = self._image_infos[index]
        img_path = self._img_dir / info["file_name"]

        # PIL 读取 → RGB → numpy → tensor
        # PIL read → RGB → numpy → tensor
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, C], [0, 1]

        # [H, W, C] → [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        return img_tensor

    def _load_masks(self, index: int) -> torch.Tensor:
        """
        加载实例掩码 [N, H, W] | Load instance masks.

        从 COCO 多边形标注渲染二值掩码。
        Renders binary masks from COCO polygon annotations.

        使用 pycocotools 的 annToMask（如果可用），
        否则用简化的多边形光栅化。
        Uses pycocotools annToMask if available,
        otherwise simplified polygon rasterization.
        """
        info = self._image_infos[index]
        img_id = info["id"]
        h, w = info["height"], info["width"]

        anns = self._ann_by_image.get(img_id, [])
        if not anns:
            # 无实例：返回空掩码 [0, H, W]
            # No instances: return empty mask [0, H, W]
            return torch.zeros(0, h, w, dtype=torch.float32)

        masks_list = []
        for ann in anns:
            mask = self._render_mask(ann, h, w)
            masks_list.append(mask)

        # 堆叠为 [N, H, W] | Stack to [N, H, W]
        masks = torch.stack(masks_list, dim=0)
        return masks

    def _load_dense_mask(self, index: int) -> torch.Tensor:
        """
        加载密集类别标签掩码 [H, W] | Load dense category label mask (instance→dense).

        将同一图像的所有实例按 category_id 渲染到单通道标签图。
        Renders all instances into a single-channel label map by category_id.

        渲染规则 | Rendering rule:
            - 背景像素 → 0 (ignore_index 用于 loss)
            - Background pixels → 0 (used as ignore_index in loss)
            - 实例像素 → category_id (1-15)
            - Instance pixels → category_id (1-15)
            - 重叠区域 → 后渲染的实例覆盖先渲染的
            - Overlapping regions → later instances overwrite earlier ones

        :param index: 图像在 self._image_infos 中的索引
        :type index: int

        :return: torch.Tensor [H, W] int64, 密集类别标签 | dense category labels
        :rtype: torch.Tensor
        """
        info = self._image_infos[index]
        img_id = info["id"]
        h, w = info["height"], info["width"]

        anns = self._ann_by_image.get(img_id, [])
        if not anns:
            # 无标注图像 → 全零标签 (全部算背景)
            # No annotation → all-zero label (all background)
            return torch.zeros(h, w, dtype=torch.long)

        # 按 category_id 分层渲染 | Layer-by-category rendering
        # 后渲染的覆盖先渲染的 (处理重叠实例)
        # Later renders overwrite earlier (handles overlapping instances)
        dense = np.zeros((h, w), dtype=np.int32)

        for ann in anns:
            cat_id = ann.get("category_id", 0)  # 1-15, 0 为无效
            if cat_id <= 0:
                continue
            # 渲染单实例二值掩码 | Render single-instance binary mask
            mask = self._render_mask(ann, h, w)  # [H, W] float32, {0, 1}
            # 将该实例的像素赋值为类别 ID | Assign category ID to instance pixels
            dense = np.where(mask.numpy() > 0.5, cat_id, dense)

        return torch.from_numpy(dense.astype(np.int64))

    def _load_image_id(self, index: int) -> int:
        """返回 COCO image_id | Return COCO image_id."""
        return self._image_infos[index]["id"]

    def __len__(self) -> int:
        """
        数据集样本数量 | Number of samples in dataset.
        Tile 模式：返回瓦片总数；否则返回图像数。
        Tile mode: returns total tiles; otherwise returns image count.
        """
        if self._tile_size is not None:
            return len(self._tile_index)
        return len(self._image_infos)

    def __getitem__(self, index: int) -> dict:
        """
        获取样本（支持 tile 模式）| Get sample (supports tile mode).

        Tile 模式：通过 tile_index 映射到 (img_idx, x, y)。
        非 tile 模式：直接使用父类实现。
        Tile mode: maps through tile_index to (img_idx, x, y).
        Non-tile mode: uses parent implementation directly.
        """
        if self._tile_size is not None:
            img_idx, x, y, _ = self._tile_index[index]

            if self._dense_labels:
                image = self._load_image(img_idx)
                dense_mask = self._load_dense_mask(img_idx)
                image_id = self._load_image_id(img_idx)
                ts = self._tile_size
                _, h, w = image.shape
                th, tw = min(ts, h - y), min(ts, w - x)
                tile_img = image[:, y:y+th, x:x+tw]
                tile_mask = dense_mask[y:y+th, x:x+tw]
                # 边界补零 | Pad edge tiles to ts×ts
                if th < ts or tw < ts:
                    tile_img = F.pad(tile_img, (0, ts-tw, 0, ts-th), value=0)
                    tile_mask = F.pad(tile_mask, (0, ts-tw, 0, ts-th), value=0)
                sample = {
                    "image": tile_img,
                    "mask": tile_mask,
                    "image_id": image_id,
                    "image_size": (ts, ts),
                }
                return self._apply_transforms(sample)

            # 加载全图 + 掩码 | Load full image + masks
            image = self._load_image(img_idx)
            masks = self._load_masks(img_idx)
            image_id = self._load_image_id(img_idx)

            # 提取瓦片 | Extract tile
            tile_img, tile_masks = self._extract_tile(image, masks, x, y)

            sample = {
                "image": tile_img,
                "masks": tile_masks,
                "image_id": image_id,
                "image_size": tuple(tile_img.shape[1:]),  # tile (H, W)
            }
            return self._apply_transforms(sample)

        # 非 tile 模式 | Non-tile mode
        if self._dense_labels:
            image = self._load_image(index)
            dense_mask = self._load_dense_mask(index)
            image_id = self._load_image_id(index)
            sample = {
                "image": image,
                "mask": dense_mask,       # [H, W] long, category IDs
                "image_id": image_id,
                "image_size": tuple(image.shape[1:]),
            }
            return self._apply_transforms(sample)
        return super().__getitem__(index)

    # ── 掩码渲染 | Mask Rendering ─────────────────────────────

    def _render_mask(self, ann: dict, h: int, w: int) -> torch.Tensor:
        """
        渲染单个实例掩码 | Render single instance mask.

        支持 COCO 多边形 segmentation 格式。
        Supports COCO polygon segmentation format.

        :param ann: COCO 标注字典 | COCO annotation dict.
        :type ann: dict

        :param h: 图像高度 | Image height.
        :type h: int

        :param w: 图像宽度 | Image width.
        :type w: int

        :return: torch.Tensor [H, W] float32, 二值 (0/1) | Binary (0/1).
        :rtype: torch.Tensor
        """
        seg = ann.get("segmentation", [])

        if not seg:
            # 无分割 → 使用 bbox | No segmentation → use bbox
            return self._render_bbox(ann["bbox"], h, w)

        # 多边形光栅化 | Polygon rasterization
        mask = np.zeros((h, w), dtype=np.uint8)

        if isinstance(seg, list) and isinstance(seg[0], list):
            # 多个多边形（带孔洞的情况）| Multiple polygons (for objects with holes)
            for poly in seg:
                self._draw_polygon(mask, poly, h, w)
        elif isinstance(seg, list) and isinstance(seg[0], (int, float)):
            # 单个多边形 | Single polygon
            self._draw_polygon(mask, seg, h, w)
        elif isinstance(seg, dict):
            # RLE 格式（暂不实现）| RLE format (not implemented yet)
            pass

        return torch.from_numpy(mask.astype(np.float32))

    def _draw_polygon(self, mask: np.ndarray, poly: list[float], h: int, w: int) -> None:
        """
        在 numpy 数组上绘制多边形 | Draw polygon on numpy array.

        使用 OpenCV fillPoly（如果可用），否则使用 PIL。
        Uses OpenCV fillPoly if available, otherwise PIL.
        """
        try:
            import cv2
        except ImportError:
            self._draw_polygon_pil(mask, poly, h, w)
            return

        # OpenCV 绘制 | OpenCV drawing
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        # 裁剪到图像边界 | Clip to image boundary
        pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
        pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)

    def _draw_polygon_pil(self, mask: np.ndarray, poly: list[float], h: int, w: int) -> None:
        """
        PIL 回退方案：多边形光栅化 | PIL fallback: polygon rasterization.
        将 float 坐标转为整数点对。| Convert float coords to int point pairs.
        """
        from PIL import ImageDraw

        # 转为 (x, y) 点对 | Convert to (x, y) point pairs
        pts = [(int(poly[i]), int(poly[i + 1])) for i in range(0, len(poly), 2)]
        # 裁剪 | Clip
        pts = [(max(0, min(x, w - 1)), max(0, min(y, h - 1))) for x, y in pts]

        pil_mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(pil_mask).polygon(pts, fill=1)
        mask[:] = np.array(pil_mask)

    def _render_bbox(self, bbox: list[float], h: int, w: int) -> torch.Tensor:
        """
        bbox 回退方案：矩形掩码 | Bbox fallback: rectangular mask.
        bbox: [x, y, width, height].
        """
        x, y, bw, bh = bbox
        mask = np.zeros((h, w), dtype=np.float32)

        # 裁剪到边界 | Clip to bounds
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + bw))
        y2 = min(h, int(y + bh))

        mask[y1:y2, x1:x2] = 1.0
        return torch.from_numpy(mask)

    # ── COCO 加载 | COCO Loading ──────────────────────────────

    def _load_coco(self) -> dict:
        """
        加载 COCO JSON 标注文件 | Load COCO JSON annotation file.

        文件命名约定：instances_{split}.json
        File naming convention: instances_{split}.json

        test split 无标注, 返回空结构 (优雅降级)
        test split has no annotations, returns empty dict (graceful degradation)
        """
        ann_path = self.root_dir / self.split / "annotations" / f"instances_{self.split}.json"
        if not ann_path.exists():
            if self.split == "test":
                self.logger.log_info(
                    "dataset/isaid_no_annotations",
                    f"iSAID {self.split}: no annotation file (expected for test split), "
                    f"返回空标注 | returning empty annotations",
                )
                return {"images": self._scan_images(), "annotations": [], "categories": []}
            raise FileNotFoundError(
                f"COCO 标注文件未找到 | COCO annotation file not found: {ann_path}"
            )
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _scan_images(self) -> list[dict]:
        """
        扫描图像目录生成 image_info | Scan image directory to generate image_info.
        用于无标注的 split (test)。
        """
        from PIL import Image
        img_dir = self.root_dir / self.split / "images"
        infos = []
        for png in sorted(img_dir.glob("*.png")):
            with Image.open(png) as img:
                w, h = img.size
            infos.append({
                "id": len(infos),
                "file_name": png.name,
                "height": h,
                "width": w,
            })
        return infos

    # ── 公共属性 | Public Properties ──────────────────────────

    @property
    def num_classes(self) -> int:
        """类别总数 | Total number of classes."""
        return len(self._categories)

    def category_name(self, cat_id: int) -> str:
        """
        根据类别 ID 获取名称 | Get category name by ID.

        :param cat_id: 类别 ID (1-15) | Category ID.
        :type cat_id: int

        :return: str: 类别名称 | Category name.
        :rtype: str
        """
        return _CAT_ID_TO_NAME.get(cat_id, f"unknown_{cat_id}")

    def category_id(self, name: str) -> int:
        """
        根据名称获取类别 ID | Get category ID by name.

        :param name: 类别名称 | Category name.
        :type name: str

        :return: int: 类别 ID, 未知返回 -1 | Category ID, -1 if unknown.
        :rtype: int
        """
        return _CAT_NAME_TO_ID.get(name, -1)
