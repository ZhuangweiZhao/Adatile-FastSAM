"""Segmentation loss: binary and multi-class.

Binary:  Dice + Focal  (K=1)
Multi:   CrossEntropy + per-class Dice  (K>1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinarySegLoss(nn.Module):
    """Dice + Focal loss for binary segmentation."""

    def forward(self, logits, gt):
        """logits [B,1,H,W], gt [B,1,H,W] — returns loss, metrics dict."""
        eps = 1e-8
        t = gt.reshape(-1)
        p = logits.sigmoid().reshape(-1)
        inter = (p * t).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
        bce = F.binary_cross_entropy_with_logits(logits.reshape(-1), t, reduction="none")
        pt = torch.exp(-bce)
        at = 0.25 * t + 0.75 * (1 - t)
        loss = dice * 5 + (at * (1 - pt) ** 2 * bce).mean()

        pb = (logits.sigmoid() > 0.5).float()
        i = (pb * gt).sum()
        u = (pb + gt).clamp(0, 1).sum()
        metrics = {
            "iou": (i / (u + eps)).item(),
            "dice": (2 * i / (pb.sum() + gt.sum() + eps)).item(),
        }
        return loss, metrics


class MultiClassSegLoss(nn.Module):
    """CrossEntropy + per-class Dice for multi-class segmentation."""

    def forward(self, logits, gt_long):
        """logits [B,K,H,W], gt_long [B,H,W] with class indices."""
        eps = 1e-8
        ce = F.cross_entropy(logits, gt_long, reduction="mean")
        probs = logits.softmax(dim=1)
        B, K = logits.shape[0], logits.shape[1]

        dice_loss = 0.0
        all_ious, all_dices = [], []
        for b in range(B):
            pred_b = probs[b].argmax(dim=0)
            for k in range(1, K):  # skip bg
                tk = (gt_long[b] == k).float()
                if tk.sum() < 0.5:
                    continue
                pk = probs[b, k]
                inter_k = (pk * tk).sum()
                dice_loss += (1 - (2 * inter_k + 1) / (pk.sum() + tk.sum() + 1))
                pred_k = (pred_b == k).float()
                inter_h = (pred_k * tk).sum()
                union_h = (pred_k + tk).clamp(0, 1).sum()
                all_ious.append((inter_h / (union_h + eps)).item())
                all_dices.append((2 * inter_h / (pred_k.sum() + tk.sum() + eps)).item())

        n = max(1, len(all_ious))
        loss = ce + (dice_loss / n) * 2.0
        metrics = {
            "iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "dice": float(np.mean(all_dices)) if all_dices else 0.0,
        }
        return loss, metrics


class SegLoss(nn.Module):
    """Auto-dispatch binary vs multi-class segmentation loss."""

    def forward(self, logits, gt):
        """logits [B,K,H,W], gt [B,H,W] or [B,1,H,W] — auto-detects K."""
        K = logits.shape[1]
        if K == 1:
            t = gt.float()
            # Fix: ensure correct [B, 1, H, W] shape regardless of input dims
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
            elif t.dim() == 3:
                t = t.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
            # (4D already correct)
            t = F.interpolate(t, size=logits.shape[-2:], mode="nearest")
            return BinarySegLoss().forward(logits, t)
        else:
            gt = gt.squeeze(1).long() if gt.dim() == 4 else gt.long()
            gt = F.interpolate(gt.float().unsqueeze(1), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
            return MultiClassSegLoss().forward(logits, gt)
