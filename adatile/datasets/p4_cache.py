"""
P4 Feature Cache — Frozen Backbone 特征预计算 | Frozen Backbone Feature Pre-computation.
============================================================================================

由于 FastSAM backbone 在训练期间完全冻结，每张 tile 的 P4 特征是确定性的。
预计算一次并缓存，训练时直接查表，消除所有 backbone forward 开销。

Since the FastSAM backbone is fully frozen during training, P4 features are deterministic
per tile. Pre-compute once, cache, and look up during training — eliminating all backbone
forward overhead.

策略 | Strategy:
    RTX 5090 (32 GB VRAM):
        - 全图模式 (~200 images): P4 fp16 on GPU ≈ 11 GB, 可行
        - Tile 模式 (~20k tiles): P4 fp16 ≈ 160+ GB >> 32 GB → 强制 CPU pinned
        - Full-image (~200 imgs): P4 fp16 on GPU ≈ 11 GB, viable
        - Tile mode (~20k tiles): P4 fp16 ≈ 160+ GB >> 32 GB → forced CPU pinned

    RTX 3060 (12 GB VRAM):
        - 一律 CPU pinned memory + 异步传输
        - Always CPU pinned memory + async transfer

用法 | Usage::
    >>> from adatile.datasets.p4_cache import P4Cache
    >>> cache = P4Cache(dataset, backbone, device='cuda', fp16=True)
    >>> cache.build()  # 预计算所有 tile 的 P4 | Pre-compute all tile P4s
    >>> p4 = cache[42]  # → [1280, 56, 56] fp16 tensor on target device
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from adatile.logging import get_logger

logger = get_logger("p4_cache")


class P4Cache:
    """
    Frozen backbone P4 特征缓存 | Frozen backbone P4 feature cache.

    预计算所有输入样本的 P4 特征，支持 GPU/CPU 存储和 fp16/fp32 精度。
    Pre-computes P4 features for all input samples. Supports GPU/CPU storage
    and fp16/fp32 precision.

    Parameters
    ----------
    dataset : object
        Duck-typed dataset with __len__ and load_image(idx)->Tensor[3,H,W].
    backbone : nn.Module
        Frozen FastSAM backbone in eval mode. Forward returns dict with 'p4' key.
    device : str
        Storage device: 'cuda' for GPU cache, 'cpu' for CPU pinned memory.
    fp16 : bool
        Store features in float16 (half memory, slight precision loss acceptable
        since features are already normalized by the backbone).
    batch_size : int
        Batch size for pre-computation backbone forward passes.
    pin_memory : bool
        Use pinned (page-locked) CPU memory for faster GPU transfers.
        Only applies when device='cpu'.

    Attributes
    ----------
    _cache : dict[int, torch.Tensor]
        Index → P4 feature tensor on storage device.
    feat_dim : int
        P4 feature channel dimension (typically 1280).
    spatial_size : tuple[int, int]
        P4 feature spatial size (H/16, W/16).
    total_size_gb : float
        Total cache size in GB.
    """

    def __init__(
        self,
        dataset,
        backbone: torch.nn.Module,
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 8,
        pin_memory: bool = True,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.backbone = backbone
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self.pin_memory = pin_memory and (device == "cpu")
        self.num_workers = num_workers

        self._cache: dict[int, torch.Tensor] = {}
        self._built = False
        self.feat_dim: int = 0
        self.spatial_size: tuple[int, int] = (0, 0)

    # ── 构建缓存 | Build Cache ──────────────────────────────────────────

    @torch.no_grad()
    def build(self, desc: str = "P4 cache") -> "P4Cache":
        """
        预计算所有样本的 P4 特征并存入缓存。
        Pre-compute P4 features for all samples and store in cache.

        内存估算 | Memory estimation:
            每 tile: 1280 × H/16 × W/16 × 2 (fp16) or × 4 (fp32) bytes
            Per tile: ~8 MB fp16, ~16 MB fp32 at 896×896 input.
            ~1,500–25,000 tiles typical → 12–200 GB fp16 → CPU pinned recommended.

        :return: self (for method chaining)
        """
        n_samples = len(self.dataset)
        if n_samples == 0:
            logger.log_info("p4_cache", "Empty dataset, cache not built")
            return self

        # ── Warmup: 计算单个样本确定 feature shape ──
        # Warmup: compute single sample to determine feature shape
        # 计算设备: 检测 backbone 所在设备, fallback CUDA
        # Compute device: detect from backbone, fallback CUDA
        storage_device = torch.device(self.device)
        try:
            backbone_device = next(self.backbone.parameters()).device
        except StopIteration:
            # FastSAM 的 YOLO wrapper 可能导致 parameters() 为空
            # FastSAM YOLO wrapper may produce empty parameters()
            backbone_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        compute_device = backbone_device  # backbone always on CUDA for inference

        sample_img = self.dataset.load_image(0)
        if sample_img.dim() == 3:
            sample_img = sample_img.unsqueeze(0)  # [1, 3, H, W]
        sample_img = sample_img.to(compute_device)

        # 使用 autocast 加速 warmup (backbone 在 fp16 下更快)
        # Use autocast for warmup speed (backbone faster in fp16)
        with torch.amp.autocast('cuda', enabled=self.fp16):
            sample_feat = self.backbone(sample_img)["p4"]  # [1, C, h, w]

        self.feat_dim = sample_feat.shape[1]
        self.spatial_size = tuple(sample_feat.shape[2:])  # (h, w)

        bytes_per_elem = 2 if self.fp16 else 4
        per_sample_gb = (
            self.feat_dim
            * self.spatial_size[0]
            * self.spatial_size[1]
            * bytes_per_elem
            / 1e9
        )
        total_est_gb = per_sample_gb * n_samples
        logger.log_info(
            "p4_cache",
            f"Pre-computing P4 for {n_samples} samples: "
            f"[{self.feat_dim}, {self.spatial_size[0]}, {self.spatial_size[1]}], "
            f"~{per_sample_gb:.3f} GB/sample, ~{total_est_gb:.1f} GB total "
            f"({'fp16' if self.fp16 else 'fp32'}, storage={self.device})",
        )

        # ── 逐 batch 预计算 | Batch-wise pre-computation ──
        t0 = time.perf_counter()
        # backbone 保持在计算设备 (CUDA), 不需要移动
        # backbone stays on compute device (CUDA), no need to move
        self.backbone.eval()

        for start in tqdm(
            range(0, n_samples, self.batch_size), desc=desc, unit="batch"
        ):
            end = min(start + self.batch_size, n_samples)
            batch_indices = list(range(start, end))

            # 加载 batch 图像 (并行 I/O) | Load batch images (parallel I/O)
            batch_imgs = [None] * len(batch_indices)

            def _load_one(idx: int) -> tuple[int, torch.Tensor]:
                """加载单个 tile, 返回 (batch_pos, tensor) | Load one tile."""
                img = self.dataset.load_image(idx)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                return batch_indices.index(idx), img

            if self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                    futures = {pool.submit(self.dataset.load_image, idx): i
                              for i, idx in enumerate(batch_indices)}
                    for future in as_completed(futures):
                        i = futures[future]
                        img = future.result()
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        batch_imgs[i] = img
            else:
                for i, idx in enumerate(batch_indices):
                    img = self.dataset.load_image(idx)
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    batch_imgs[i] = img

            batch_tensor = torch.cat(batch_imgs, dim=0).to(compute_device)

            # Backbone forward (使用 autocast 加速大 batch)
            # Backbone forward (autocast for speed on large batch)
            with torch.amp.autocast('cuda', enabled=self.fp16):
                feats = self.backbone(batch_tensor)["p4"]  # [B, C, h, w]

            # 存入选定设备 + 可选精度 | Store on target device + optional precision
            for i, idx in enumerate(batch_indices):
                feat = feats[i].detach()  # [C, h, w]
                if self.fp16 and feat.dtype != torch.float16:
                    feat = feat.half()

                if storage_device.type == "cpu" and self.pin_memory:
                    # Pinned memory for fast GPU transfer
                    feat_cpu = torch.empty(
                        feat.shape,
                        dtype=feat.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    feat_cpu.copy_(feat.cpu(), non_blocking=False)
                    self._cache[idx] = feat_cpu
                elif storage_device.type == "cpu":
                    self._cache[idx] = feat.cpu()
                else:
                    # Keep on GPU (same device as computation)
                    self._cache[idx] = feat.to(storage_device)

            # 释放 batch tensor (帮助 GPU 内存管理)
            # Free batch tensor (helps GPU memory management)
            del batch_tensor, feats

        dt = time.perf_counter() - t0
        self._built = True
        self.total_size_gb = sum(
            t.numel() * t.element_size() for t in self._cache.values()
        ) / 1e9

        logger.log_info(
            "p4_cache",
            f"P4 cache built: {len(self._cache)} samples, "
            f"{self.total_size_gb:.1f} GB, {dt:.0f}s "
            f"({n_samples / dt:.1f} samples/s, "
            f"storage={'GPU' if self.device == 'cuda' else 'CPU pinned' if self.pin_memory else 'CPU'})",
        )
        logger.log_metric(
            "p4_cache_build_time_s", dt, tags=["p4_cache", f"n={n_samples}"]
        )
        logger.log_metric(
            "p4_cache_size_gb",
            self.total_size_gb,
            tags=["p4_cache", f"fp16={self.fp16}"],
        )

        return self

    @torch.no_grad()
    def build_fast(self, tile_wrapper, desc: str = "P4 cache (fast)") -> "P4Cache":
        """
        全图级 P4 预计算: 每张源图像一次 backbone forward, 然后裁剪 tile P4。
        Full-image P4 precomputation: one backbone forward per source image,
        then crop tile P4s from the feature map.

        对比逐 tile 预计算 (~23k backbone forwards):
        vs per-tile precomputation (~23k backbone forwards):
            141 images × 1 forward = 141 forwards → **~167× faster**.

        要求 tile_wrapper 提供:
        Requires tile_wrapper to provide:
            - get_source_image_count() → int
            - get_tiles_for_image(img_idx) → list[(tile_idx, x0, y0)]
            - load_full_image(img_idx) → Tensor[3, H, W]
            - tile_size → int

        :param tile_wrapper: ISAIDTileWrapper 实例
        :param desc: 进度条标签 | Progress bar label
        :return: self
        """
        n_images = tile_wrapper.get_source_image_count()
        n_tiles = len(tile_wrapper)
        if n_images == 0 or n_tiles == 0:
            logger.log_info("p4_cache", "Empty dataset, cache not built")
            return self

        # ── 确定设备 | Determine devices ──
        storage_device = torch.device(self.device)
        try:
            backbone_device = next(self.backbone.parameters()).device
        except StopIteration:
            backbone_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        compute_device = backbone_device
        self.backbone.eval()

        # ── Warmup: 用小尺寸确定 P4 feature shape (避免大图 OOM) ──
        # Warmup: use small crop to determine P4 shape (avoids full-image OOM)
        sample_img = tile_wrapper.load_full_image(0)
        if sample_img.dim() == 4:
            sample_img = sample_img[0]
        # 只取 1024×1024 区域做 warmup | Only use 1024×1024 crop for warmup
        _, h_full, w_full = sample_img.shape
        crop_h = min(1024, h_full)
        crop_w = min(1024, w_full)
        sample_crop = sample_img[:, :crop_h, :crop_w]
        # Pad 到 32 倍数 | Pad to 32×
        _, ch, cw = sample_crop.shape
        ph = (32 - ch % 32) % 32
        pw = (32 - cw % 32) % 32
        sample_crop = torch.nn.functional.pad(sample_crop, (0, pw, 0, ph), value=0)
        sample_crop = sample_crop.unsqueeze(0).to(compute_device)
        with torch.amp.autocast('cuda', enabled=self.fp16):
            sample_p4 = self.backbone(sample_crop)["p4"]
        del sample_crop, sample_img
        self.feat_dim = sample_p4.shape[1]
        self.spatial_size = (tile_wrapper.tile_size // 16, tile_wrapper.tile_size // 16)

        per_tile_gb = (self.feat_dim * self.spatial_size[0] * self.spatial_size[1]
                       * (2 if self.fp16 else 4)) / 1e9
        total_est_gb = per_tile_gb * n_tiles
        logger.log_info(
            "p4_cache",
            f"Fast pre-computation: {n_images} source images → {n_tiles} tiles, "
            f"~{per_tile_gb:.3f} GB/tile, ~{total_est_gb:.1f} GB total "
            f"({'fp16' if self.fp16 else 'fp32'}, storage={self.device})",
        )

        # ── 逐源图像分块处理 | Per-source-image chunked processing ──
        # 全图 > 2048px 时 OOM → 分块处理 (chunk=2048, overlap=tile_size)
        # Full image > 2048px causes OOM → chunked processing
        t0 = time.perf_counter()
        ts = tile_wrapper.tile_size
        pts = ts // 16  # P4 spatial tile size per tile
        MAX_CHUNK = 2048  # 安全 chunk 尺寸: backbone 在此尺寸下不 OOM
        CHUNK_OVERLAP = ts  # chunk 重叠 = tile_size, 确保边界 tile 完整

        for img_idx in tqdm(range(n_images), desc=desc, unit="img"):
            tiles = tile_wrapper.get_tiles_for_image(img_idx)
            if not tiles:
                continue

            # ── 加载全图 (CPU) | Load full image (CPU) ──
            full_img = tile_wrapper.load_full_image(img_idx)  # [3, H, W] on CPU
            if full_img.dim() == 4:
                full_img = full_img[0]
            _, h, w = full_img.shape

            # ── 分块策略 | Chunking strategy ──
            need_chunk = (h > MAX_CHUNK or w > MAX_CHUNK)

            if not need_chunk:
                # 小图: pad 到 32 倍数 → backbone forward
                # Small image: pad to 32× → backbone forward
                _, h_img, w_img = full_img.shape
                pad_h_img = (32 - h_img % 32) % 32
                pad_w_img = (32 - w_img % 32) % 32
                if pad_h_img > 0 or pad_w_img > 0:
                    batch = torch.nn.functional.pad(
                        full_img, (0, pad_w_img, 0, pad_h_img), value=0
                    ).unsqueeze(0).to(compute_device)
                else:
                    batch = full_img.unsqueeze(0).to(compute_device)
                with torch.amp.autocast('cuda', enabled=self.fp16):
                    full_p4 = self.backbone(batch)["p4"][0]
                if self.fp16 and full_p4.dtype != torch.float16:
                    full_p4 = full_p4.half()
                self._store_tile_p4s(full_p4, tiles, pts, storage_device)
                del batch, full_p4
            else:
                # 大图: 分块 backbone forward + 按 tile 裁剪
                # Large image: chunked backbone + per-tile crop
                # FastSAM 要求输入尺寸是 32 的倍数 → 对每个 chunk pad
                # FastSAM requires input dims be multiples of 32 → pad each chunk
                for cy0 in range(0, h - ts + 1, MAX_CHUNK - CHUNK_OVERLAP):
                    cy1 = min(cy0 + MAX_CHUNK, h)
                    for cx0 in range(0, w - ts + 1, MAX_CHUNK - CHUNK_OVERLAP):
                        cx1 = min(cx0 + MAX_CHUNK, w)

                        # 裁剪 + pad 到 32 的倍数 | Crop + pad to multiple of 32
                        chunk_raw = full_img[:, cy0:cy1, cx0:cx1]
                        _, cH, cW = chunk_raw.shape
                        pad_h = (32 - cH % 32) % 32
                        pad_w = (32 - cW % 32) % 32
                        chunk = torch.nn.functional.pad(
                            chunk_raw, (0, pad_w, 0, pad_h), value=0
                        ).unsqueeze(0).to(compute_device)

                        with torch.amp.autocast('cuda', enabled=self.fp16):
                            chunk_p4 = self.backbone(chunk)["p4"][0]

                        if self.fp16 and chunk_p4.dtype != torch.float16:
                            chunk_p4 = chunk_p4.half()

                        # 裁剪 tile (用未 pad 坐标) | Crop tiles (un-padded coords)
                        for tile_idx, x0, y0 in tiles:
                            if (x0 >= cx0 and y0 >= cy0 and
                                x0 + ts <= cx1 and y0 + ts <= cy1):
                                px0 = (x0 - cx0) // 16
                                py0 = (y0 - cy0) // 16
                                tile_p4 = chunk_p4[:, py0:py0 + pts, px0:px0 + pts]
                                self._store_one(tile_idx, tile_p4, storage_device)

                        del chunk, chunk_p4

            del full_img
            # 定期清理 GPU 缓存 | Periodic GPU cache cleanup
            if img_idx % 10 == 9 and compute_device.type == "cuda":
                torch.cuda.empty_cache()

        dt = time.perf_counter() - t0
        self._built = True
        self.total_size_gb = sum(
            t.numel() * t.element_size() for t in self._cache.values()
        ) / 1e9

        logger.log_info(
            "p4_cache",
            f"P4 cache (fast) built: {len(self._cache)} tiles from {n_images} images, "
            f"{self.total_size_gb:.1f} GB, {dt:.0f}s "
            f"({n_images / dt:.1f} img/s, {n_tiles / dt:.1f} tiles/s)",
        )
        logger.log_metric("p4_cache_build_time_s", dt,
                         tags=["p4_cache_fast", f"n_img={n_images}", f"n_tiles={n_tiles}"])
        logger.log_metric("p4_cache_size_gb", self.total_size_gb,
                         tags=["p4_cache_fast", f"fp16={self.fp16}"])

        return self

    # ── 内部存储辅助 | Internal Storage Helpers ────────────────────────

    def _store_one(self, tile_idx: int, tile_p4: torch.Tensor,
                   storage_device: torch.device) -> None:
        """存储单个 tile P4 到目标设备 | Store single tile P4 to target device."""
        if storage_device.type == "cpu" and self.pin_memory:
            feat_cpu = torch.empty(
                tile_p4.shape, dtype=tile_p4.dtype,
                device="cpu", pin_memory=True,
            )
            feat_cpu.copy_(tile_p4, non_blocking=True)
            self._cache[tile_idx] = feat_cpu
        elif storage_device.type == "cpu":
            self._cache[tile_idx] = tile_p4.cpu()
        else:
            self._cache[tile_idx] = tile_p4.to(storage_device)

    def _store_tile_p4s(self, full_p4: torch.Tensor,
                        tiles: list, pts: int,
                        storage_device: torch.device) -> None:
        """从小图/全图 P4 批量裁剪并存储 tile P4 | Batch crop & store from small/full P4."""
        for tile_idx, x0, y0 in tiles:
            px0, py0 = x0 // 16, y0 // 16
            tile_p4 = full_p4[:, py0:py0 + pts, px0:px0 + pts]
            self._store_one(tile_idx, tile_p4, storage_device)

    # ── 访问接口 | Access Interface ──────────────────────────────────────

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取预计算的 P4 特征 → [C, h, w] on storage device。
        Get pre-computed P4 feature → [C, h, w] on storage device.

        如果存储在 CPU，调用者负责 .to(device)。
        Caller is responsible for .to(device) if stored on CPU.

        :param idx: 样本索引 | Sample index
        :return: P4 feature tensor
        """
        if not self._built:
            raise RuntimeError("P4Cache not built. Call .build() first.")
        return self._cache[idx]

    def get_batch(self, indices: list[int], target_device: str = None) -> torch.Tensor:
        """
        批量获取 P4 特征，可选传输到目标设备。
        Batch-get P4 features, optionally transfer to target device.

        :param indices: 样本索引列表 | List of sample indices
        :param target_device: 目标设备 (None = 保持存储设备) | Target device
        :return: [B, C, h, w] on target device
        """
        feats = torch.stack([self._cache[i] for i in indices])
        if target_device is not None and str(feats.device) != target_device:
            feats = feats.to(target_device, non_blocking=True)
        return feats

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, idx: int) -> bool:
        return idx in self._cache

    @property
    def is_built(self) -> bool:
        """缓存是否已构建 | Whether cache has been built."""
        return self._built

    # ── 持久化 (可选) | Persistence (optional) ──────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """
        保存缓存到磁盘 (.pt 文件)。
        Save cache to disk (.pt file).

        :param path: 输出文件路径 | Output file path
        """
        if not self._built:
            raise RuntimeError("P4Cache not built. Nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "feat_dim": self.feat_dim,
            "spatial_size": self.spatial_size,
            "fp16": self.fp16,
            "n_samples": len(self._cache),
            "features": self._cache,  # dict[int, Tensor]
        }
        torch.save(save_dict, path)
        logger.log_info(
            "p4_cache",
            f"Cache saved to {path} ({self.total_size_gb:.1f} GB)",
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        dataset=None,
        device: str = "cuda",
    ) -> "P4Cache":
        """
        从磁盘加载缓存。
        Load cache from disk.

        :param path: .pt 文件路径 | .pt file path
        :param dataset: 可选的 dataset 引用 (用于元信息)
        :param device: 加载到的设备 | Target device for loaded tensors
        :return: P4Cache instance with loaded features
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        cache = cls.__new__(cls)
        cache.dataset = dataset
        cache.backbone = None
        cache.device = device
        cache.fp16 = data["fp16"]
        cache.batch_size = 0
        cache.pin_memory = False
        cache.feat_dim = data["feat_dim"]
        cache.spatial_size = data["spatial_size"]

        # 迁移特征到目标设备 | Move features to target device
        cache._cache = {}
        for idx, feat in data["features"].items():
            cache._cache[int(idx)] = feat.to(device, non_blocking=True)

        cache._built = True
        cache.total_size_gb = sum(
            t.numel() * t.element_size() for t in cache._cache.values()
        ) / 1e9
        logger.log_info(
            "p4_cache",
            f"Cache loaded from {path}: {len(cache._cache)} samples, "
            f"{cache.total_size_gb:.1f} GB on {device}",
        )
        return cache


# ═══════════════════════════════════════════════════════════════════
# 便捷函数 | Convenience Functions
# ═══════════════════════════════════════════════════════════════════


def auto_build_p4_cache(
    dataset,
    backbone,
    device: str = "cuda",
    fp16: bool = True,
    batch_size: int = 8,
    pin_memory: bool = True,
    cache_dir: str = None,
    cache_name: str = "p4_cache",
    num_workers: int = 4,
    tile_wrapper=None,  # 如果传入则使用 build_fast (全图级预计算) | use build_fast if provided
) -> P4Cache:
    """
    自动构建或加载 P4 缓存 | Auto-build or load P4 cache.

    检查磁盘缓存是否存在 → 存在则加载，否则构建 + 保存。
    Checks disk cache existence → loads if exists, otherwise builds + saves.

    :param dataset: Duck-typed dataset with load_image(idx)
    :param backbone: Frozen FastSAM backbone
    :param device: 存储设备 | Storage device
    :param fp16: 是否使用 fp16 | Use fp16
    :param batch_size: 预计算 batch 大小 (仅 build() 使用)
    :param pin_memory: CPU 缓存是否使用 pinned memory
    :param cache_dir: 缓存目录 | Cache directory (None = no disk cache)
    :param cache_name: 缓存文件名前缀 | Cache filename prefix
    :param num_workers: 并行 I/O 线程数 (仅 build() 使用)
    :param tile_wrapper: ISAIDTileWrapper 实例 → 使用 build_fast (全图级, ~167× 加速)
    :return: P4Cache instance (built)
    """
    # 尝试从磁盘加载 | Try loading from disk
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{cache_name}.pt"
        if cache_path.exists():
            logger.log_info("p4_cache", f"Loading cached P4 from {cache_path}")
            try:
                return P4Cache.load(cache_path, dataset=dataset, device=device)
            except Exception as e:
                logger.log_info(
                    "p4_cache", f"Failed to load cache ({e}), rebuilding..."
                )

    # 构建新缓存 | Build new cache
    cache = P4Cache(
        dataset,
        backbone,
        device=device,
        fp16=fp16,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    if tile_wrapper is not None:
        # 全图级预计算: backbone 每张源图像仅跑一次 → ~167× 加速
        # Full-image precomputation: backbone once per source image
        cache.build_fast(tile_wrapper)
    else:
        cache.build()

    # 保存到磁盘 | Save to disk
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{cache_name}.pt"
        cache.save(cache_path)

    return cache
