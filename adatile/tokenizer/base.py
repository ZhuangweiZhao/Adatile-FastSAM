"""Dynamic tile tokenizer implementations.

Adaptive Tile Allocation:
    Given an importance map, determines tile sizes and positions.
    High-importance regions → small tiles (high resolution).
    Low-importance regions → large tiles or skipped.

Classes:
    - DynamicTileTokenizerImpl: full adaptive tokenizer (planner + generator + global)
    - UniformTileTokenizer: fixed-grid baseline for ablation
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import DynamicTileTokenizer, TileInfo
from adatile.registry import TOKENIZER
from adatile.tokenizer.tile_planner import TilePlanner, TilePlan, TileSpec
from adatile.tokenizer.token_generator import TokenGenerator


@TOKENIZER.register()
class DynamicTileTokenizerImpl(DynamicTileTokenizer):
    """Adaptive tile tokenizer: importance-guided partitioning + token extraction.

    Pipeline:
        1. TilePlanner: importance map → TilePlan (where & at what resolution)
        2. TokenGenerator: extract patches → conv stem → tokens + pos encoding
        3. GlobalThumbnailBranch (optional): global context fusion

    Args:
        tile_sizes: Available tile sizes in pixels [384, 768, 1536, 3072].
        strides: Overlap stride per tile size (fraction of tile_size).
        importance_threshold: Skip regions below this importance.
        max_tokens: Token budget per image.
        overlap_ratio: Overlap fraction between adjacent tiles.
        patch_size: Common resolution tiles are resized to for encoding.
        embed_dim: Output token embedding dimension.
        stem_width: Hidden channels in conv stem.
        use_global_branch: Whether to fuse global thumbnail context.
        use_overlap: Whether to generate overlap tiles.
    """

    def __init__(
        self,
        tile_sizes: Optional[List[int]] = None,
        strides: Optional[List[float]] = None,
        importance_threshold: float = 0.3,
        max_tokens: int = 4096,
        overlap_ratio: float = 0.25,
        patch_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 256,
        stem_width: int = 128,
        num_frequencies: int = 32,
        use_global_branch: bool = True,
        use_overlap: bool = False,
        skip_mode: str = "threshold",
        hard_skip_multiplier: float = 1.0,
    ):
        super().__init__()
        self.tile_sizes = tile_sizes or [384, 768, 1536, 3072]
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.use_global_branch = use_global_branch
        self.use_overlap = use_overlap
        self.skip_mode = skip_mode

        # Core components
        self.planner = TilePlanner(
            tile_sizes=self.tile_sizes,
            strides=strides,
            importance_threshold=importance_threshold,
            max_tokens=max_tokens,
            overlap_ratio=overlap_ratio,
            skip_mode=skip_mode,
            hard_skip_multiplier=hard_skip_multiplier,
        )
        self.generator = TokenGenerator(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            stem_width=stem_width,
            num_frequencies=num_frequencies,
        )

        # Global context branch (lazy import to avoid circular deps)
        self.global_branch: Optional[nn.Module] = None
        if use_global_branch:
            from adatile.tokenizer.global_branch import GlobalThumbnailBranch
            self.global_branch = GlobalThumbnailBranch(
                thumbnail_size=512,
                in_channels=in_channels,
                embed_dim=embed_dim,
            )

        # Register the global branch as submodule
        if self.global_branch is not None:
            self.add_module("global_branch", self.global_branch)

        self._last_plan: Optional[TilePlan] = None

    def forward(
        self,
        image: Tensor,
        features: Optional[Dict[str, Tensor]] = None,
        importance: Optional[Tensor] = None,
        granularity_hard: Optional[Tensor] = None,
    ) -> Tuple[List[TileInfo], Tensor]:
        """Partition image into tiles and extract token features.

        Args:
            image: [B, 3, H, W] input image (B=1 for single-image mode typical).
            features: Optional backbone feature maps (unused by default, for API compat).
            importance: Optional [B, 1, H_s, W_s] importance map from Ada-SPM.
            granularity_hard: Optional [B, 1, H_s, W_s] tile-size index per cell
                from Ada-SPM's GranularityHead. When provided, uses learned
                tile-size assignments instead of the hard-coded heuristic.

        Returns:
            tiles: List of TileInfo for each generated tile.
            tokens: [B, N_tiles, embed_dim] token features.
        """
        B, C, H, W = image.shape

        # If no importance provided, use uniform (all tiles at moderate resolution)
        if importance is None:
            H_s, W_s = H // 32, W // 32  # approximate spatial grid
            importance = torch.ones(B, 1, H_s, W_s, device=image.device)

        all_tile_infos: List[List[TileInfo]] = []
        all_tokens: List[Tensor] = []

        for b in range(B):
            imp_b = importance[b:b + 1] if importance.dim() == 4 else importance
            if imp_b.dim() == 4:
                imp_b = imp_b.squeeze(0)

            # Step 1: Plan tile allocation — use learned granularity
            # from Ada-SPM when available, else fall back to heuristic
            gr_hard_b = None
            if granularity_hard is not None:
                if granularity_hard.dim() == 4 and granularity_hard.shape[0] > b:
                    gr_hard_b = granularity_hard[b:b+1]

            plan = self.planner.plan(
                importance=imp_b,
                image_size=(H, W),
                granularity_hard=gr_hard_b,
                image_id=f"batch_{b}",
            )
            self._last_plan = plan

            # Step 2: Optional overlap augmentation
            specs = plan.specs
            if self.use_overlap:
                specs = self.planner.add_overlap(specs, self.planner.overlap_ratio)
                # Re-sort by priority after adding overlap tiles
                specs.sort(key=lambda t: t.priority, reverse=True)
                specs = specs[:self.max_tokens]

            # Step 3: Generate token embeddings
            img_b = image[b:b + 1]  # [1, C, H, W]
            tokens, centers, scale_ids = self.generator(img_b, specs)

            # Step 4: Optional global context fusion
            if self.global_branch is not None and tokens.shape[0] > 0:
                global_embed, _ = self.global_branch(img_b)
                # Fuse: add global context to each token
                tokens = tokens + global_embed.squeeze(0).unsqueeze(0).expand_as(tokens)

            # Step 5: Convert specs to TileInfo
            tile_infos = []
            for i, spec in enumerate(specs):
                tile_infos.append(spec.to_tile_info(
                    image_id=f"batch_{b}",
                    idx=i,
                ))

            all_tile_infos.append(tile_infos)
            all_tokens.append(tokens)  # [N_tiles, C]

        # Pad tokens to a common size across batch
        if B > 1:
            max_n = max(t.shape[0] for t in all_tokens)
            padded_tokens = torch.zeros(B, max_n, self.embed_dim, device=image.device)
            for b, t in enumerate(all_tokens):
                if t.shape[0] > 0:
                    padded_tokens[b, :t.shape[0]] = t
            return all_tile_infos, padded_tokens

        # B=1: return as [1, N, C]
        if all_tokens[0].shape[0] > 0:
            tokens_out = all_tokens[0].unsqueeze(0)  # [1, N, C]
        else:
            tokens_out = torch.zeros(1, 0, self.embed_dim, device=image.device)

        return all_tile_infos, tokens_out

    def prepare_cache(
        self,
        image: Tensor,
        tile_size: int,
        stride: int,
        save_dir: str,
    ) -> None:
        """Pre-compute and cache tile features for an image.

        Extracts uniform grid tiles at the given size/stride,
        encodes them through the generator, and saves to disk.

        Args:
            image: [3, H, W] single input image.
            tile_size: Tile side length.
            stride: Sliding window stride.
            save_dir: Output cache directory.
        """
        import os
        import pickle

        C, H, W = image.shape
        image_b = image.unsqueeze(0)  # [1, C, H, W]

        # Generate uniform grid specs
        specs = []
        idx = 0
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                specs.append(TileSpec(
                    x1=x, y1=y,
                    x2=min(x + tile_size, W),
                    y2=min(y + tile_size, H),
                    tile_size=tile_size,
                    stride=stride,
                    importance=1.0,
                    priority=1.0,
                    scale_level=0,
                ))
                idx += 1

        if not specs:
            raise ValueError(f"No tiles generated for image {H}x{W} with "
                             f"tile_size={tile_size}, stride={stride}")

        # Extract and encode tiles
        tokens, centers, scale_ids = self.generator(image_b, specs)

        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, f"tiles_{tile_size}_{stride}.pkl")
        cache_data = {
            "tiles": specs,
            "tokens": tokens.cpu(),
            "centers": centers.cpu(),
            "scale_ids": scale_ids.cpu(),
            "image_shape": (H, W),
            "tile_size": tile_size,
            "stride": stride,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    @property
    def last_plan(self) -> Optional[TilePlan]:
        """Return the most recent tile plan (for debugging/analysis)."""
        return self._last_plan

    def get_budget_stats(self) -> Dict[str, float]:
        """Quick budget stats from the last plan."""
        if self._last_plan is None:
            return {}
        return {
            "active_tiles": self._last_plan.active_tiles,
            "skipped_regions": self._last_plan.skipped_regions,
            "skip_ratio": self._last_plan.skip_ratio,
            "budget_used": self._last_plan.token_budget_used,
            "budget_max": self._last_plan.token_budget_max,
            "budget_util": self._last_plan.budget_utilization,
        }

    def set_importance_threshold(self, threshold: float) -> None:
        """Dynamically adjust the importance threshold."""
        self.planner.set_threshold(threshold)


@TOKENIZER.register()
class UniformTileTokenizer(DynamicTileTokenizer):
    """Uniform tiling baseline — fixed-size sliding window with no adaptation.

    Useful for ablation studies: comparing adaptive vs. uniform tiling
    under the same token budget.

    Args:
        tile_size: Fixed tile size in pixels.
        stride: Sliding window stride.
        max_tokens: Token budget per image.
        patch_size: Resize resolution for patch embedding.
        embed_dim: Output token dimension.
        stem_width: Conv stem hidden channels.
    """

    def __init__(
        self,
        tile_size: int = 768,
        stride: int = 384,
        max_tokens: int = 4096,
        patch_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 256,
        stem_width: int = 128,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.stride = stride
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.generator = TokenGenerator(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            stem_width=stem_width,
        )

    def forward(
        self,
        image: Tensor,
        features: Optional[Dict[str, Tensor]] = None,
        importance: Optional[Tensor] = None,
        granularity_hard: Optional[Tensor] = None,
    ) -> Tuple[List[TileInfo], Tensor]:
        """Partition image via uniform sliding window.

        Args:
            image: [B, 3, H, W].
            features: Unused.
            importance: Unused (uniform ignores importance).
            granularity_hard: Unused (uniform uses fixed tile size).

        Returns:
            tiles: List of TileInfo per image.
            tokens: [B, N_tiles, embed_dim].
        """
        B, C, H, W = image.shape

        all_tile_infos: List[List[TileInfo]] = []
        all_tokens: List[Tensor] = []

        for b in range(B):
            # Generate uniform grid
            specs = []
            idx = 0
            for y in range(0, H - self.tile_size + 1, self.stride):
                for x in range(0, W - self.tile_size + 1, self.stride):
                    specs.append(TileSpec(
                        x1=x, y1=y,
                        x2=min(x + self.tile_size, W),
                        y2=min(y + self.tile_size, H),
                        tile_size=self.tile_size,
                        stride=self.stride,
                        importance=1.0,
                        priority=1.0,
                        scale_level=0,
                    ))
                    idx += 1

            # Budget enforcement: truncate if too many
            if len(specs) > self.max_tokens:
                # Keep center tiles (highest "priority" for uniform)
                specs.sort(
                    key=lambda s: abs(s.x1 + s.x2 - W) + abs(s.y1 + s.y2 - H),
                )
                specs = specs[:self.max_tokens]

            img_b = image[b:b + 1]
            tokens, _, _ = self.generator(img_b, specs)

            tile_infos = []
            for i, spec in enumerate(specs):
                tile_infos.append(spec.to_tile_info(
                    image_id=f"batch_{b}",
                    idx=i,
                ))

            all_tile_infos.append(tile_infos)
            all_tokens.append(tokens)

        # Pad to batch
        if B > 1:
            max_n = max(t.shape[0] for t in all_tokens)
            padded = torch.zeros(B, max_n, self.embed_dim, device=image.device)
            for b, t in enumerate(all_tokens):
                if t.shape[0] > 0:
                    padded[b, :t.shape[0]] = t
            return all_tile_infos, padded

        tokens_out = (
            all_tokens[0].unsqueeze(0)
            if all_tokens[0].shape[0] > 0
            else torch.zeros(1, 0, self.embed_dim, device=image.device)
        )
        return all_tile_infos, tokens_out

    def prepare_cache(
        self, image: Tensor, tile_size: int, stride: int, save_dir: str
    ) -> None:
        """Pre-compute uniform tile features and cache to disk."""
        import os
        import pickle

        C, H, W = image.shape
        image_b = image.unsqueeze(0)

        specs = []
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                specs.append(TileSpec(
                    x1=x, y1=y,
                    x2=min(x + tile_size, W),
                    y2=min(y + tile_size, H),
                    tile_size=tile_size,
                    stride=stride,
                    importance=1.0,
                    priority=1.0,
                    scale_level=0,
                ))

        tokens, centers, scale_ids = self.generator(image_b, specs)

        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, f"uniform_{tile_size}_{stride}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "tiles": specs,
                "tokens": tokens.cpu(),
                "centers": centers.cpu(),
                "scale_ids": scale_ids.cpu(),
                "image_shape": (H, W),
            }, f)
