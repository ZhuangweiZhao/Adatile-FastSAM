"""Unified loss composition: Seg + SPM + Budget.

Combines the clean sub-modules (seg_loss, spm_loss, budget_loss) into
a single callable that returns (loss, metrics_dict).

Usage:
    loss_fn = UnifiedLoss(use_spm=True, num_classes=1, spm_mode="topk")
    loss, metrics = loss_fn(logits, gt, importance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.losses.seg_loss import SegLoss
from adatile.losses.spm_loss import DensityLoss, TopKLoss
from adatile.losses.budget_loss import FixedBudget, LearnableBudget


class UnifiedLoss(nn.Module):
    """Composed segmentation + SPM + budget loss.

    L_total = L_seg + λ_spm × L_spm + L_budget
    """

    def __init__(
        self,
        use_spm: bool = False,
        num_classes: int = 1,
        spm_mode: str = "topk",
        budget_target: float = 0.15,
        lambda_spm: float = 1.0,
        lambda_budget: float = 5.0,
        budget_mode: str = "ratio",
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.use_spm = use_spm
        self.num_classes = num_classes
        self.spm_mode = spm_mode
        self.lambda_spm = lambda_spm
        self.budget_mode = budget_mode
        self._debug_step = 0

        self.seg_loss = SegLoss()
        self.spm_fn = TopKLoss() if spm_mode == "topk" else DensityLoss()

        if budget_mode == "learnable":
            self.budget_fn = LearnableBudget(budget_target, lambda_budget, lambda_reg)
        elif budget_mode == "entropy":
            self.budget_fn = None  # handled inline
        else:
            self.budget_fn = FixedBudget(budget_target)

        # For entropy mode, store fixed lambda
        self._fixed_lambda = lambda_budget

    def forward(self, logits, gt, importance=None):
        """Compute unified loss.

        Args:
            logits: [B, K, H/4, W/4] decoder output logits.
            gt: [B, H, W] or [B, 1, H, W] ground truth masks.
            importance: [B, 1, H_s, W_s] SPM importance map, or SparsePrediction
                        with .importance attribute. None if use_spm=False.

        Returns:
            loss: scalar tensor.
            metrics: dict with iou, dice, imp_mean, coverage (if SPM enabled).
        """
        # Support SparsePrediction from AdaSPM variants
        if importance is not None and hasattr(importance, 'importance'):
            importance = importance.importance

        loss, metrics = self.seg_loss(logits, gt)

        if self.use_spm and importance is not None:
            importance = torch.nan_to_num(importance, nan=0.5).clamp(1e-7, 1 - 1e-7)
            K = logits.shape[1]
            B_s, _, H_imp, W_imp = importance.shape

            # Binary foreground at SPM resolution
            if K == 1:
                # Fix: correctly handle [H,W] (2D) vs [B,H,W] (3D) vs [B,1,H,W] (4D)
                gt_raw = gt.float()
                if gt_raw.dim() == 2:
                    gt_raw = gt_raw.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
                elif gt_raw.dim() == 3:
                    gt_raw = gt_raw.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
                gt_fg = F.interpolate(gt_raw, size=(H_imp, W_imp), mode="area")
            else:
                gt_long = gt.squeeze(1).long() if gt.dim() == 4 else gt.long()
                gt_fg = F.interpolate(
                    (gt_long > 0).float().unsqueeze(1),
                    size=(H_imp, W_imp), mode="area")
            gt_fg = gt_fg.clamp(0, 1)

            # Pre-debug: save seg loss
            loss_seg = loss.detach().clone()

            # ── SPM ranking loss ──────────────────────────
            if self.spm_mode == "topk":
                # Top-K supervision uses learnable target if applicable
                target = (
                    self.budget_fn.target
                    if isinstance(self.budget_fn, FixedBudget)
                    else (
                        self.budget_fn.logit_target.sigmoid()
                        if isinstance(self.budget_fn, LearnableBudget)
                        else 0.15
                    )
                )
                loss_spm = self.spm_fn(importance, gt_fg, target)
                loss = loss + self.lambda_spm * loss_spm

                # ── Budget loss ────────────────────────────
                if self.budget_mode == "entropy":
                    imp_c = importance.clamp(1e-7, 1 - 1e-7)
                    H_val = -(imp_c * imp_c.log() + (1 - imp_c) * (1 - imp_c).log()).mean()
                    loss = loss - self._fixed_lambda * H_val
                    lb_raw = -H_val.item()
                    lambda_used = self._fixed_lambda
                    ratio_val = imp_c.mean().item()
                elif isinstance(self.budget_fn, LearnableBudget):
                    loss_b, imp_val, target_val, lambda_val = self.budget_fn(importance)
                    loss = loss + loss_b
                    lb_raw = loss_b.item()
                    lambda_used = lambda_val
                    ratio_val = imp_val
                else:
                    loss_budget, imp_val = self.budget_fn(importance)
                    loss = loss + loss_budget
                    lb_raw = loss_budget.item()
                    lambda_used = self._fixed_lambda
                    ratio_val = imp_val
            else:
                # Density regression (legacy)
                loss_spm = self.spm_fn(importance, gt_fg)
                loss = loss + self.lambda_spm * loss_spm
                if isinstance(self.budget_fn, FixedBudget):
                    loss_budget, imp_val = self.budget_fn(importance)
                    loss = loss + self._fixed_lambda * loss_budget
                    lambda_used = self._fixed_lambda
                    lb_raw = loss_budget.item()
                    ratio_val = imp_val
                else:
                    lambda_used = 5.0
                    lb_raw = 0.0
                    ratio_val = importance.mean().item()

            # Debug logging
            if self._debug_step < 100:
                print(
                    f"  [DEBUG loss#{self._debug_step}] "
                    f"L_seg={loss_seg.item():.4f}  "
                    f"L_spm_raw={loss_spm.item():.4f} "
                    f"λ*L_spm={self.lambda_spm * loss_spm.item():.4f}  "
                    f"L_budget_raw={lb_raw:.6f} "
                    f"λ*L_budget={lambda_used * lb_raw:.6f}  "
                    f"ratio={ratio_val:.3f} imp_m={importance.mean().item():.3f}"
                )
                self._debug_step += 1

            # Coverage metric
            imp_flat = importance[0, 0].reshape(-1)
            n_total = imp_flat.shape[0]
            target_cov = (
                self.budget_fn.target
                if isinstance(self.budget_fn, FixedBudget)
                else (
                    self.budget_fn.logit_target.sigmoid()
                    if isinstance(self.budget_fn, LearnableBudget)
                    else 0.15
                )
            )
            k_cov = max(1, int(n_total * target_cov))
            _, idx = imp_flat.topk(k_cov)
            keep_mask = torch.zeros(n_total, dtype=torch.bool, device=importance.device)
            keep_mask[idx] = True
            gt_in_kept = gt_fg[0, 0].reshape(-1)[keep_mask].sum()
            gt_total = gt_fg[0, 0].sum()
            coverage = (gt_in_kept / (gt_total + 1e-8)).item() if gt_total > 0 else 1.0
            metrics["imp_mean"] = importance.mean().item()
            metrics["coverage"] = coverage

            # ── Loss decomposition for logging ─────────────
            metrics["loss_seg"] = loss_seg.item()
            metrics["loss_seg_raw"] = loss_seg.item()
            metrics["loss_spm"] = self.lambda_spm * loss_spm.item()
            metrics["loss_spm_raw"] = loss_spm.item()
            if isinstance(self.budget_fn, LearnableBudget):
                metrics["loss_budget"] = loss_b.item() if 'loss_b' in dir() else lb_raw
                metrics["loss_budget_raw"] = lb_raw
                metrics["lambda_used"] = lambda_used
            elif self.budget_mode == "entropy":
                metrics["loss_budget"] = -self._fixed_lambda * H_val.item() if 'H_val' in dir() else 0.0
                metrics["loss_budget_raw"] = lb_raw
                metrics["lambda_used"] = self._fixed_lambda
            else:
                metrics["loss_budget"] = loss_budget.item() if 'loss_budget' in dir() else lb_raw
                metrics["loss_budget_raw"] = lb_raw
                metrics["lambda_used"] = self._fixed_lambda

        return loss, metrics
