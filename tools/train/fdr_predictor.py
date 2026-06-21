"""
FDRPredictor ? Wrapper for Foreground Density Router with MV3 backbone.

Extracted from train_b04.py (2026-06-21).
Combines frozen MobileNetV3 backbone + trainable DensityHead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from adatile.sparse.spatial_router import DensityHead

TILE_SIZE = 1024


class FDRPredictor(nn.Module):
    """
    FDR (Foreground Density Router).

    Frozen MV3 backbone => DensityHead => tile importance scores.
    Training target: predict fg_ratio per tile.
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.density_head = DensityHead(576, 128)

    def forward(self, x):
        return self.density_head(self.backbone(x))

    @torch.no_grad()
    def predict_tile_scores(self, img_np, device):
        """
        Predict importance scores for all tiles on an image.

        Flow:
        1. Resize image to fixed size => forward => importance map
        2. Divide importance map by tile grid => per-tile mean = score
        """
        from PIL import Image
        H, W = img_np.shape[:2]
        scale = 2048 / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        img_r = np.array(Image.fromarray(img_np).resize((nW, nH), Image.BILINEAR))
        ph, pw = (32 - nH % 32) % 32, (32 - nW % 32) % 32
        if ph > 0 or pw > 0:
            img_r = np.pad(img_r, ((0, ph), (0, pw), (0, 0)), mode="constant")
        img_t = torch.from_numpy(img_r.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)
        imp = self.forward(img_t)
        hp, wp = imp.shape[2], imp.shape[3]
        n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W + TILE_SIZE - 1) // TILE_SIZE
        scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0 = int(ty * TILE_SIZE * scale / 2048 * hp)
                y1 = int(min(ty * TILE_SIZE + TILE_SIZE, H) * scale / 2048 * hp)
                x0 = int(tx * TILE_SIZE * scale / 2048 * wp)
                x1 = int(min(tx * TILE_SIZE + TILE_SIZE, W) * scale / 2048 * wp)
                y0, y1 = max(0, min(y0, hp - 1)), max(y0 + 1, min(y1, hp))
                x0, x1 = max(0, min(x0, wp - 1)), max(x0 + 1, min(x1, wp))
                scores[ty, tx] = float(imp[0, 0, y0:y1, x0:x1].mean())
        return scores
