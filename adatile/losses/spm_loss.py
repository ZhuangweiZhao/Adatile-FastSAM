"""SPM importance map losses.

DensityLoss  — absolute density regression (legacy baseline)
TopKLoss     — per-image top-K ranking (Ada-SPM core)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityLoss(nn.Module):
    """Absolute density regression: SPM output → fixed targets.

    Targets: foreground=0.90, background=0.01.
    Plus a binary BCE for coverage boost.
    """

    def forward(self, importance, gt_binary):
        """importance [B,1,H,W], gt_binary [B,1,H,W] (0/1 foreground mask)."""
        gt_d = 0.01 + 0.89 * gt_binary.clamp(0, 1)
        logit_imp = (importance.clamp(1e-7, 1 - 1e-7).log()
                     - (1 - importance.clamp(1e-7, 1 - 1e-7)).log())
        d_bce = F.binary_cross_entropy_with_logits(
            logit_imp.reshape(-1), gt_d.reshape(-1), reduction="none")
        d_pt = torch.exp(-d_bce)
        loss = (0.75 * (1 - d_pt) ** 2 * d_bce).mean()

        # Coverage boost: BCE on binarized importance
        gt_cov = (gt_d > 0.3).float().detach()
        loss = loss + F.binary_cross_entropy(
            importance.clamp(1e-7, 1 - 1e-7), gt_cov)
        return loss


class TopKLoss(nn.Module):
    """Per-image top-K ranking supervision.

    For each image: rank tiles by GT density, label top K% as 1, rest as 0.
    Uses Focal BCE for class imbalance.
    """

    def forward(self, importance, gt_binary, keep_ratio):
        """importance [B,1,H,W], gt_binary [B,1,H,W], keep_ratio in (0,1)."""
        B, _, H, W = importance.shape
        gt_d_flat = gt_binary.view(B, -1)
        N = gt_d_flat.shape[1]
        k = max(1, int(N * keep_ratio))
        topk_vals = gt_d_flat.topk(k, dim=1).values[:, -1:]

        gt_spm = (gt_d_flat >= topk_vals).float().view(B, 1, H, W)
        imp_c = importance.clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(imp_c, gt_spm, reduction="none")
        pt = torch.where(gt_spm > 0.5, imp_c, 1 - imp_c)
        loss = ((1 - pt) ** 2 * bce).mean()
        return loss
