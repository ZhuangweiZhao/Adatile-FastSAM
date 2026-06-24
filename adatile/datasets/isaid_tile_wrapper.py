"""
iSAID Tile Dataset Wrapper — 瓦片切分策略 | Tile Splitting Strategy.
======================================================================

将全图 COCO 实例分割数据集包装为 tile 级别访问，支持滑窗切分、
bbox 重叠检测、LRU 图像缓存。从 eval_c03_catsam_fewshot.py 解耦提取。

Wraps full-image COCO instance segmentation dataset for tile-level access,
with sliding-window splitting, bbox-overlap class→tile mapping, and LRU
image caching. Extracted from eval_c03_catsam_fewshot.py for reuse.

设计原则 | Design principles:
    - Duck-typing: 不直接导入 ISAIDInstanceDataset，任何实现了所需接口的对象均可使用
      Duck-typing: does NOT import ISAIDInstanceDataset — any object with the
      required interface works.
    - 零外部依赖(除 adatile + torch + cv2) | Zero external deps beyond adatile/torch/cv2.
    - 双语注释 + 日志 | Bilingual comments + logging.

Wrapped dataset 必须实现以下接口 | Required interface:
    - _img_infos: list[dict]  (each dict has "id", "file_name")
    - _img_anns: dict[image_id → list[COCO annotations]]
    - src_root: Path | str
    - split: str
    - load_image(idx) → torch.Tensor [3, H, W]
    - render_class_mask(idx, class_id) → torch.Tensor [H, W]
    - class_to_images(class_id) → list[int]

用法 | Usage::
    >>> from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper
    >>> tile_ds = ISAIDTileWrapper(dataset, tile_size=896, stride=512)
    >>> img = tile_ds.load_image(0)          # [3, 896, 896]
    >>> mask = tile_ds.render_class_mask(0, 5)  # [896, 896] for ship
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from adatile.logging import get_logger

logger = get_logger("isaid_tile_wrapper")


class ISAIDTileWrapper:
    """
    将全图数据集包装为 tile 级别访问 | Full-image → tile-level wrapper.

    全图 (up to 4000×4000) → sliding window tiles, stride < tile_size.
    每张 tile 保留原始分辨率细节（不降采样），避免小目标丢失。
    Each tile preserves original resolution detail — no downscaling.
    No small-object loss from aggressive resizing.

    Tile P4 feature map: [1280, tile_size/16, tile_size/16] at stride 16.

    Parameters
    ----------
    dataset : object
        Duck-typed dataset with _img_infos, _img_anns, src_root, split,
        load_image(), render_class_mask(), class_to_images().
    tile_size : int
        Tile edge length in pixels. Default 896.
    stride : int
        Sliding window stride in pixels. Must be ≤ tile_size.
        Overlap = tile_size - stride.

    Examples
    --------
    >>> tile_ds = ISAIDTileWrapper(dataset, tile_size=896, stride=512)
    >>> len(tile_ds)  # number of tiles
    >>> img = tile_ds.load_image(0)           # [3, 896, 896] in [0, 1]
    >>> mask = tile_ds.render_class_mask(0, 5)  # [896, 896] binary
    """

    def __init__(self, dataset: Any, tile_size: int = 896, stride: int = 512):
        # ── 参数校验 | Parameter validation ──
        if stride > tile_size:
            raise ValueError(
                f"stride ({stride}) must be ≤ tile_size ({tile_size})"
            )
        if stride <= 0 or tile_size <= 0:
            raise ValueError(
                f"tile_size ({tile_size}) and stride ({stride}) must be > 0"
            )

        self.ds = dataset
        self.tile_size = tile_size
        self.stride = stride
        self.overlap = tile_size - stride  # e.g. 384 when tile=896, stride=512

        # ── LRU 图像缓存: 避免每张 tile 都从磁盘加载 4000×4000 原图 ──
        # LRU image cache: avoid re-reading full images from disk per tile
        # 可外部调节 _cache_max: RTX 5090=64, RTX 3060=32, default=8
        self._img_cache: dict[int, np.ndarray] = {}  # img_idx → [H, W, 3]
        self._cache_max: int = 8  # ~8 × 48MB = 384MB RAM (for 4000×4000 images)

        # ── 预计算全部 tile 坐标 & class→tile 映射 ──
        # Pre-compute all tile coordinates + class→tile mapping
        # 优化: 使用 COCO metadata 获取尺寸 (避免每张图 ~48MB 的 cv2.imread)
        # Opt: use COCO metadata for dims (avoids ~48MB cv2.imread per image)
        # 优化: O(1) bbox→tile 映射 (整数除法代替 O(n_tiles × n_bboxes) 遍历)
        # Opt: O(1) bbox→tile via integer division instead of O(tiles×bboxes) scan
        self._tiles: list[tuple[int, int, int]] = []  # [(img_idx, x0, y0), ...]
        self._class_to_tiles: dict[int, set[int]] = defaultdict(set)
        skipped_empty = 0
        total_bbox_checks = 0  # 统计 | stats

        for img_idx in tqdm(range(len(dataset)), desc="Tile grid", leave=False):
            img_info = dataset._img_infos[img_idx]
            # COCO JSON 已有 width/height — 无需 cv2.imread
            # COCO JSON already has width/height — no cv2.imread needed
            w: int = img_info.get("width", 0)
            h: int = img_info.get("height", 0)

            if w <= 0 or h <= 0:
                # Fallback: 如果 metadata 缺失则读文件 | Fallback if metadata missing
                img_path = str(
                    dataset.src_root / dataset.split / "images" / img_info["file_name"]
                )
                img = cv2.imread(img_path)
                if img is None:
                    logger.log_info("tile_grid", f"Skip unreadable: {img_path}")
                    skipped_empty += 1
                    continue
                h, w = img.shape[:2]
                logger.log_info("tile_grid",
                               f"Fallback cv2.imread for {img_info['file_name']}: {w}x{h}")

            if h < tile_size or w < tile_size:
                skipped_empty += 1
                continue

            # ── 滑窗切分 | Sliding window ──
            tile_start = len(self._tiles)
            n_cols = (w - tile_size) // stride + 1
            n_rows = (h - tile_size) // stride + 1

            for row in range(n_rows):
                y0 = row * stride
                for col in range(n_cols):
                    x0 = col * stride
                    self._tiles.append((img_idx, x0, y0))

            # ── Class→tile O(1) 映射: 对每个 bbox 直接计算覆盖的 tile 范围 ──
            # O(1) bbox→tile: for each bbox, compute tile range via integer division
            anns = dataset._img_anns.get(img_info["id"], [])
            if not anns:
                continue

            # 按类别分组 bbox | Group bboxes by category
            cat_to_bboxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
            for ann in anns:
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if bbox[2] > 0 and bbox[3] > 0:
                    cat_to_bboxes[ann["category_id"]].append(
                        (bbox[0], bbox[1], bbox[2], bbox[3])
                    )

            for cat_id, bboxes in cat_to_bboxes.items():
                for bx, by, bw, bh in bboxes:
                    # 整数除法计算 bbox 覆盖的 tile 范围 | Integer division for tile range
                    tx_start = int(bx // stride)
                    tx_end = min(int((bx + bw - 1) // stride) + 1, n_cols)
                    ty_start = int(by // stride)
                    ty_end = min(int((by + bh - 1) // stride) + 1, n_rows)

                    # 直接索引 tile: tile_i = tile_start + row * n_cols + col
                    # Direct tile index: tile_i = tile_start + row * n_cols + col
                    for ty in range(ty_start, ty_end):
                        row_offset = ty * n_cols
                        for tx in range(tx_start, tx_end):
                            self._class_to_tiles[cat_id].add(tile_start + row_offset + tx)
                    total_bbox_checks += 1

        # 转为排序列表 | Convert to sorted lists
        self._class_to_tiles = {
            int(k): sorted(v)
            for k, v in self._class_to_tiles.items()
        }

        total_tiles = len(self._tiles)
        n_classes = len(self._class_to_tiles)
        logger.log_info(
            "tile_wrapper",
            f"{len(dataset)} imgs → {total_tiles} tiles "
            f"(skipped {skipped_empty} too small), "
            f"{n_classes} classes have tiles",
        )
        logger.log_metric(
            "total_tiles", total_tiles,
            tags=["tile_wrapper", f"size={tile_size}", f"stride={stride}"],
        )
        logger.log_metric("skipped_empty", skipped_empty, tags=["tile_wrapper"])
        logger.log_metric("n_classes_with_tiles", n_classes, tags=["tile_wrapper"])

    # ── 数据集协议 | Dataset protocol ──────────────────────────────────

    def __len__(self) -> int:
        """总 tile 数量 | Total number of tiles."""
        return len(self._tiles)

    def class_to_images(self, class_id: int) -> list[int]:
        """
        返回包含指定类别的 tile 索引列表。
        Returns tile indices that contain the given class.

        :param class_id: 类别 ID | Category ID
        :return: list of tile indices
        """
        return self._class_to_tiles.get(int(class_id), [])

    @property
    def src_root(self):
        """代理到内部 dataset 的 src_root | Proxy to inner dataset src_root."""
        return self.ds.src_root

    @property
    def split(self):
        """代理到内部 dataset 的 split | Proxy to inner dataset split."""
        return self.ds.split

    @property
    def _img_infos(self):
        """代理到内部 dataset 的 _img_infos | Proxy to inner dataset _img_infos."""
        return self.ds._img_infos

    @property
    def _img_anns(self):
        """代理到内部 dataset 的 _img_anns | Proxy to inner dataset _img_anns."""
        return self.ds._img_anns

    # ── 图像加载 | Image loading ────────────────────────────────────

    def _load_original_image(self, img_idx: int) -> np.ndarray:
        """
        加载全分辨率原始图像（带 LRU 缓存）。
        Load full-resolution original image with LRU cache.

        缓存策略 | Cache strategy:
            - 最多缓存 _cache_max 张原图 | Max _cache_max full images cached
            - 存储 copy，返回引用 — 调用方不应修改返回值
            - Store a copy, return reference — caller must not mutate
        """
        if img_idx in self._img_cache:
            return self._img_cache[img_idx]

        img_info = self.ds._img_infos[img_idx]
        img_path = str(
            self.ds.src_root / self.ds.split / "images" / img_info["file_name"]
        )
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = img[..., ::-1]  # BGR → RGB

        # LRU 淘汰 | LRU eviction
        if len(self._img_cache) >= self._cache_max:
            oldest = next(iter(self._img_cache))
            del self._img_cache[oldest]

        # 缓存副本: cv2.imread 返回的 buffer 可能被后续 imread 复用
        # Store copy: cv2.imread buffer may be reused by subsequent imread calls
        self._img_cache[img_idx] = img.copy()
        return self._img_cache[img_idx]

    def load_image(self, tile_idx: int) -> torch.Tensor:
        """
        加载单张 tile → [3, tile_size, tile_size], 值域 [0, 1]。
        Load single tile → [3, tile_size, tile_size] tensor in [0, 1].

        :param tile_idx: tile 索引 | Tile index (0-based)
        :return: float tensor [3, tile_size, tile_size]
        """
        img_idx, x0, y0 = self._tiles[int(tile_idx)]
        img = self._load_original_image(img_idx)
        tile = img[y0 : y0 + self.tile_size, x0 : x0 + self.tile_size]
        tile_t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        return tile_t

    # ── Mask 渲染 | Mask rendering ──────────────────────────────────

    def render_class_mask(self, tile_idx: int, class_id: int) -> torch.Tensor:
        """
        渲染单张 tile 的目标类别 union mask → [tile_size, tile_size]。
        Render union mask for target class on a single tile.

        策略 | Strategy:
            Polygon 坐标相对于 tile 左上角平移后直接渲染，无需渲染全图再裁剪。
            Polygon coords are shifted relative to tile origin — no full-image
            rendering needed, saving memory and computation.

        支持格式 | Supported formats:
            - Polygon: [[x1,y1,x2,y2,...], ...]
            - RLE (pycocotools): dict with 'counts' key (rare in iSAID, fallback)
            - Bbox: [x, y, w, h] (last resort fallback)

        :param tile_idx: tile 索引 | Tile index
        :param class_id: 目标类别 ID | Target category ID
        :return: binary float tensor [tile_size, tile_size], values 0.0 or 1.0
        """
        tile_idx = int(tile_idx)
        img_idx, x0, y0 = self._tiles[tile_idx]
        img_info = self.ds._img_infos[img_idx]
        anns = self.ds._img_anns.get(img_info["id"], [])
        ts = self.tile_size

        mask = np.zeros((ts, ts), dtype=np.uint8)

        for ann in anns:
            if ann["category_id"] != int(class_id):
                continue

            seg = ann.get("segmentation", [])
            if isinstance(seg, dict):
                # RLE: 渲染全分辨率后裁剪 | Render full-res then crop
                # (rare in iSAID, kept as fallback)
                img = self._load_original_image(img_idx)
                h, w = img.shape[:2]
                from pycocotools import mask as coco_mask

                rle = coco_mask.frPyObjects(seg, h, w)
                if isinstance(rle, list):
                    rle = coco_mask.merge(rle)
                full_mask = coco_mask.decode(rle).astype(np.uint8)
                mask = np.maximum(mask, full_mask[y0 : y0 + ts, x0 : x0 + ts])

            elif seg and isinstance(seg[0], (int, float)):
                # 单个多边形 | Single polygon
                self._fill_polygon_on_tile(mask, [seg], x0, y0, ts)

            elif seg and isinstance(seg[0], list):
                # 多个多边形 | Multiple polygons
                self._fill_polygon_on_tile(mask, seg, x0, y0, ts)

            else:
                # Bbox 回退 | Bbox fallback
                bbox = ann.get("bbox", [0, 0, 0, 0])
                bx, by, bw, bh = bbox
                ix0 = int(max(0, bx - x0))
                iy0 = int(max(0, by - y0))
                ix1 = int(min(ts, bx + bw - x0))
                iy1 = int(min(ts, by + bh - y0))
                if ix1 > ix0 and iy1 > iy0:
                    mask[iy0:iy1, ix0:ix1] = 1

        return torch.from_numpy(mask).float()

    @staticmethod
    def _fill_polygon_on_tile(
        mask: np.ndarray, polys: list, x0: int, y0: int, ts: int
    ) -> None:
        """
        在 tile 上绘制多边形（坐标相对于 tile 原点平移）。
        Draw polygons on tile (coordinates shifted relative to tile origin).

        :param mask: [ts, ts] uint8 mask to fill into (mutated in-place)
        :param polys: list of polygons, each [x1,y1,x2,y2,...]
        :param x0: tile left edge in full-image coordinates
        :param y0: tile top edge in full-image coordinates
        :param ts: tile size in pixels
        """
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            # 平移到 tile 坐标系 | Shift to tile coordinate system
            pts = pts - [x0, y0]
            # 裁剪到 tile 边界 | Clip to tile bounds
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, ts - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, ts - 1)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

    # ── Tile 坐标查询 | Tile coordinate query ──────────────────────

    def get_tile_info(self, tile_idx: int) -> dict:
        """
        返回 tile 的元信息 | Return tile metadata.

        :return: dict with keys: img_idx, x0, y0, tile_size, overlap
        """
        img_idx, x0, y0 = self._tiles[int(tile_idx)]
        return {
            "img_idx": img_idx,
            "x0": x0,
            "y0": y0,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
        }

    def get_total_tiles(self) -> int:
        """返回 tile 总数 | Return total tile count."""
        return len(self._tiles)

    # ── 全图级 P4 预计算支持 | Full-image P4 precomputation support ──

    def get_source_image_count(self) -> int:
        """
        返回源图像数量 (用于全图级 P4 预计算)。
        Returns number of unique source images (for full-image P4 precomputation).

        全图级预计算: 对每张源图像仅运行一次 backbone forward,
        然后从 feature map 裁剪各 tile 的 P4 → ~167× 加速。
        Full-image precomputation: backbone forward once per source image,
        then crop tile P4s from feature map → ~167× speedup.
        """
        return len(self.ds)

    def get_tiles_for_image(self, img_idx: int) -> list[tuple[int, int, int]]:
        """
        返回属于指定源图像的所有 tile 信息。
        Returns all tile info for a given source image.

        :param img_idx: 源图像索引 | Source image index
        :return: list of (tile_idx, x0, y0) tuples sorted by tile_idx
        """
        result = []
        for tile_idx, (ti_img_idx, x0, y0) in enumerate(self._tiles):
            if ti_img_idx == img_idx:
                result.append((tile_idx, x0, y0))
        return result

    def load_full_image(self, img_idx: int) -> torch.Tensor:
        """
        加载完整源图像 → [3, H, W], 值域 [0, 1]。
        Load full source image → [3, H, W] in [0, 1].

        用于全图级 P4 预计算 | Used for full-image P4 precomputation.
        """
        img = self._load_original_image(img_idx)  # [H, W, 3] RGB
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
