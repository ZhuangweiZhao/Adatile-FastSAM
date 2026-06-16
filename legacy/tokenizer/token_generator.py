"""Token generator: patch extraction, conv stem, positional encoding.

Converts raw image tiles (variable sizes, arbitrary positions) into
fixed-dimensional token embeddings for downstream processing.

Components:
    - PatchEmbed: lightweight conv stem + adaptive pooling
    - PosEmbed2D: sinusoidal 2D encoding of tile center + scale
    - TokenGenerator: orchestrates extraction → embedding → encoding
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.tokenizer.tile_planner import TileSpec


# ── Patch Embedding ───────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """Lightweight conv stem that maps variable-size tiles to token embeddings.

    Architecture:
        tile (resized to patch_size) → Conv2d → BN → ReLU → Conv2d → BN → ReLU
        → AdaptiveAvgPool2d → token [C]

    Args:
        patch_size: All tiles are resized to this square before encoding.
        in_channels: Input channels (3 for RGB, or feature dim).
        embed_dim: Output token dimension.
        stem_width: Hidden dimension in conv stem.
    """

    def __init__(
        self,
        patch_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 256,
        stem_width: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, tiles: Tensor) -> Tensor:
        """Encode a batch of tiles into token embeddings.

        Args:
            tiles: [N, C, H_tile, W_tile] variable-size tile patches.

        Returns:
            tokens: [N, embed_dim] token embeddings.
        """
        N = tiles.shape[0]
        # Resize all tiles to common patch_size
        if tiles.shape[-2:] != (self.patch_size, self.patch_size):
            tiles = F.interpolate(
                tiles, size=(self.patch_size, self.patch_size),
                mode="bilinear", align_corners=False,
            )
        x = self.stem(tiles)       # [N, embed_dim, H', W']
        x = self.pool(x)          # [N, embed_dim, 1, 1]
        return x.view(N, self.embed_dim)


# ── 2D Positional Encoding ─────────────────────────────────────────────


class PosEmbed2D(nn.Module):
    """2D sinusoidal positional encoding for tiles at arbitrary coordinates.

    Encodes (cx, cy, scale) for each tile using multi-frequency sine/cosine
    features, then projects through a small MLP.

    The encoding captures:
        - Absolute 2D position of the tile center in the full image.
        - Tile scale (to distinguish fine vs. coarse tokens).

    Args:
        embed_dim: Output encoding dimension.
        num_frequencies: Number of sinusoidal frequency bands per coordinate.
        temperature: Base frequency scaling factor.
        learnable: If True, use a learnable scale factor.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_frequencies: int = 32,
        temperature: float = 10000.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        self.temperature = temperature

        # Learnable temperature scale
        if learnable:
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("scale", torch.tensor(1.0))

        # Raw encoding dim: 3 coords × 2 (sin/cos) × num_frequencies
        raw_dim = 3 * 2 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, centers: Tensor, scales: Tensor) -> Tensor:
        """Encode tile positions.

        Args:
            centers: [N, 2] normalized tile centers in [0, 1] (cx, cy).
            scales: [N] tile scale in [0, 1] (0=finest, 1=coarsest).

        Returns:
            pos_embed: [N, embed_dim] positional encodings.
        """
        N = centers.shape[0]
        device = centers.device

        # Build coordinate tensor: [N, 3] = (cx, cy, scale)
        coords = torch.cat([
            centers,
            scales.view(N, 1).clamp(0.0, 1.0),
        ], dim=1)  # [N, 3]

        # Multi-frequency encoding
        freq_bands = self.scale * self.temperature ** (
            -torch.linspace(0, 1, self.num_frequencies, device=device)
        )  # [F]

        # [N, 3, F]
        angles = coords.unsqueeze(-1) * freq_bands.view(1, 1, -1) * (2 * math.pi)
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)

        raw = torch.cat([sin_part, cos_part], dim=1)  # [N, 6, F]
        raw = raw.view(N, -1)  # [N, 6*F]

        return self.proj(raw)


# ── Token Generator ────────────────────────────────────────────────────


class TokenGenerator(nn.Module):
    """Generates token embeddings from image tiles + positional info.

    Orchestrates:
        1. Extract tile patches from the full-resolution image (via F.grid_sample
           or direct crop sampling).
        2. Encode each tile through PatchEmbed → token features.
        3. Add 2D positional encoding based on tile location + scale.

    Args:
        patch_size: Common size tiles are resized to before encoding.
        embed_dim: Output token embedding dimension.
        stem_width: Hidden channels in conv stem.
        num_frequencies: Positional encoding frequency bands.
        use_features: If True, expects pre-computed feature maps instead of raw image.
    """

    def __init__(
        self,
        patch_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 256,
        stem_width: int = 128,
        num_frequencies: int = 32,
        use_features: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_features = use_features

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            stem_width=stem_width,
        )
        self.pos_embed = PosEmbed2D(
            embed_dim=embed_dim,
            num_frequencies=num_frequencies,
        )

    def forward(
        self,
        image: Tensor,
        specs: List[TileSpec],
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract tiles and produce token embeddings.

        Args:
            image: [B, C, H, W] input image (single image per batch item).
            features: Optional [B, C_f, H_f, W_f] feature map at stride S;
                when provided and use_features=True, tile from features.
            specs: List of TileSpec defining tile positions and sizes.

        Returns:
            tokens: [N, embed_dim] token embeddings.
            centers: [N, 2] tile centers in normalized [0,1] coords.
            scale_ids: [N] scale level indices.
        """
        if len(specs) == 0:
            device = image.device
            return (
                torch.empty(0, self.embed_dim, device=device),
                torch.empty(0, 2, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        B, C, H, W = image.shape
        tiles = self._extract_tiles(image, specs)  # [N, C, patch_size, patch_size]
        tokens = self.patch_embed(tiles)           # [N, embed_dim]

        # Build position info
        centers = torch.tensor(
            [[(s.x1 + s.x2) / (2 * W), (s.y1 + s.y2) / (2 * H)] for s in specs],
            device=image.device, dtype=torch.float32,
        )
        max_scale = max(1, max(s.scale_level for s in specs))
        scales = torch.tensor(
            [s.scale_level / max_scale for s in specs],
            device=image.device, dtype=torch.float32,
        )
        scale_ids = torch.tensor(
            [s.scale_level for s in specs],
            device=image.device, dtype=torch.long,
        )

        pos = self.pos_embed(centers, scales)
        tokens = tokens + pos

        return tokens, centers, scale_ids

    def _extract_tiles(
        self,
        image: Tensor,
        specs: List[TileSpec],
    ) -> Tensor:
        """Extract tile patches from the image using bilinear sampling.

        Uses F.grid_sample with normalized coordinates for efficient
        batched extraction of arbitrarily-positioned, variable-size tiles.

        Args:
            image: [B, C, H, W] image tensor (B=1 for single-image mode).
            specs: Tile specifications.

        Returns:
            tiles: [N, C, patch_size, patch_size].
        """
        B, C, H, W = image.shape
        device = image.device
        N = len(specs)
        ps = self.patch_size

        # Build sampling grid for each tile
        # grid_sample uses normalized coordinates [-1, 1]
        theta = torch.zeros(N, 2, 3, device=device)

        for i, s in enumerate(specs):
            tw = s.x2 - s.x1
            th = s.y2 - s.y1
            sx = tw / W
            sy = th / H
            cx = (s.x1 + s.x2) / (2 * W)
            cy = (s.y1 + s.y2) / (2 * H)
            theta[i, 0, 0] = sx
            theta[i, 1, 1] = sy
            theta[i, 0, 2] = cx * 2 - 1
            theta[i, 1, 2] = cy * 2 - 1

        # Batch tile extraction to avoid OOM on large images.
        # image.expand(N, ...) would create [N, C, H, W] = e.g. [128, 3, 5502, 3875]
        # which is ~8 GB in fp32 — fatal on 6 GB GPUs.
        # Process in chunks to keep peak memory low.
        CHUNK = 16  # tiles per batch; <=16 keeps expand() under ~1 GB
        all_tiles = []
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            n_chunk = end - start
            grid = F.affine_grid(
                theta[start:end], (n_chunk, C, ps, ps), align_corners=False,
            )
            image_n = image.expand(n_chunk, -1, -1, -1)
            chunk_tiles = F.grid_sample(
                image_n, grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )
            all_tiles.append(chunk_tiles)

        return torch.cat(all_tiles, dim=0) if len(all_tiles) > 1 else all_tiles[0]

    def extract_tiles_raw(
        self,
        image: Tensor,
        specs: List[TileSpec],
    ) -> List[Tensor]:
        """Extract raw tile patches at native resolution (no resize).

        Useful for visualization and debugging.

        Args:
            image: [1, C, H, W] single image.
            specs: Tile specifications.

        Returns:
            List of [C, H_t, W_t] tile tensors at native resolution.
        """
        img = image[0]  # [C, H, W]
        tiles = []
        for s in specs:
            tile = img[:, s.y1:s.y2, s.x1:s.x2]
            tiles.append(tile)
        return tiles
