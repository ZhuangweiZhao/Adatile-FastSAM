"""Full segmentation pipeline and loss functions.

AdaTileFastSAM:
    End-to-end model: backbone → SPM → tokenizer → router → decoder.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import (
    SegmentationOutput,
    TileInfo,
    LossFunction,
)
from adatile.registry import SEGMENTATION, LOSS


# ── Loss Functions ───────────────────────────────────────────────────


@LOSS.register()
class DiceLoss(LossFunction):
    """Dice loss for mask supervision."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Apply sigmoid internally so callers don't need to remember
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


@LOSS.register()
class FocalLoss(LossFunction):
    """Focal loss for classification / mask scoring."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


@LOSS.register()
class SegmentationLoss(LossFunction):
    """Composite segmentation loss: Dice + BCE + Focal."""

    def __init__(
        self,
        dice_weight: float = 5.0,
        focal_weight: float = 1.0,
        iou_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(
        self,
        pred_masks: Tensor,
        target_masks: Tensor,
        pred_iou: Optional[Tensor] = None,
        target_iou: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        loss_dice = self.dice(pred_masks, target_masks)
        loss_focal = self.focal(pred_masks, target_masks)
        total = self.dice_weight * loss_dice + self.focal_weight * loss_focal

        losses = {"loss_dice": loss_dice, "loss_focal": loss_focal, "loss_mask": total}

        if pred_iou is not None and target_iou is not None:
            loss_iou = F.mse_loss(pred_iou, target_iou)
            losses["loss_iou"] = loss_iou
            losses["loss_mask"] = total + self.iou_weight * loss_iou

        return losses


# ── Training Loss ──────────────────────────────────────────────────────


@LOSS.register()
class SimpleSegLoss(nn.Module):
    """Loss for simple binary mask datasets (no COCO annotations).

    v2: GT-driven planning + Focal density + Budget loss.
    Designed to prevent importance collapse on imbalanced data.

    Key fixes vs v1:
      - Density: Focal loss replaces MSE (handles 95% bg / 5% fg imbalance)
      - Planning: GT-driven (coverage target from GT mask, not planner output)
      - Sparsity: Budget loss replaces mean-target L1 (matches data distribution)
      - base_loss: deleted from pipeline
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        density_weight: float = 3.0,
        planning_weight: float = 1.0,
        budget_weight: float = 0.1,
        routing_weight: float = 0.0,
        full_mask_weight: float = 0.0,
        budget_target_keep: float = 0.15,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.density_weight = density_weight
        self.planning_weight = planning_weight
        self.budget_weight = budget_weight
        self.routing_weight = routing_weight
        self.full_mask_weight = full_mask_weight
        self.budget_target_keep = budget_target_keep
        self.proto_weight = 0.1  # prototype contrastive loss weight

        self.dice_loss = DiceLoss(smooth=1.0)
        # Focal loss for density supervision (alpha=0.75: emphasize object regions)
        self.density_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        self.register_buffer("_step_count", torch.zeros(1))

    def forward(
        self,
        output,
        batch: Dict,
        aux: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if aux is not None and "density" in aux:
            device = aux["density"].device
        losses: Dict[str, Tensor] = {}
        self._step_count += 1

        gt_mask = batch.get("masks")
        images = batch.get("images", batch.get("image"))

        # ── 1. Focal Density Loss ────────────────────────────────
        if self.density_weight > 0 and aux is not None and "density" in aux:
            density = aux["density"]
            gt_density = self._build_gt_density_from_mask(
                density.shape, gt_mask, images.shape[-2:], device
            )
            # Focal loss: γ=2 automatically down-weights easy background cells
            # α=0.75 emphasizes object regions (foreground)
            losses["loss_density"] = self.density_weight * self._focal_density_loss(
                torch.nan_to_num(density, nan=0.5), gt_density
            )
        else:
            losses["loss_density"] = torch.tensor(0.0, device=device)

        # ── 2. GT-Driven Planning Alignment ──────────────────────
        if self.planning_weight > 0 and aux is not None and "importance" in aux:
            importance = aux["importance"]
            gt_density = self._build_gt_density_from_mask(
                importance.shape, gt_mask, images.shape[-2:], device
            )
            # GT coverage: where should tiles go? (gt_density > 0.3 → object cells)
            gt_coverage = (gt_density > 0.3).float().detach()
            # BCE(importance, gt_coverage): encourage high imp at objects, low imp elsewhere
            # Disable autocast — BCE with sigmoid-activated importance is safe in fp32
            with torch.amp.autocast("cuda", enabled=False):
                losses["loss_planning"] = self.planning_weight * F.binary_cross_entropy(
                    importance.float().clamp(1e-7, 1 - 1e-7),
                    gt_coverage.float(),
                )
        else:
            losses["loss_planning"] = torch.tensor(0.0, device=device)

        # ── 3. Budget Loss (replaces sparsity L1) ─────────────────
        if self.budget_weight > 0 and aux is not None and "importance" in aux:
            importance = aux["importance"]
            # keep_ratio: fraction of cells with imp > 0.3
            keep_ratio = (importance > 0.3).float().mean()
            # Budget: penalize deviation from target keep ratio (e.g., 15%)
            losses["loss_sparse"] = self.budget_weight * (
                (keep_ratio - self.budget_target_keep) ** 2
            )
        else:
            losses["loss_sparse"] = torch.tensor(0.0, device=device)

        # ── 4. Routing Aux Loss ──────────────────────────────────
        if self.routing_weight > 0 and aux is not None and "routing_weights" in aux:
            routing_weights = aux["routing_weights"]
            w = torch.nan_to_num(routing_weights.squeeze(-1).float(), nan=0.5)
            skip_frac = (w < 0.15).float().mean()
            skip_shortfall = F.relu(0.35 - skip_frac)
            all_high_penalty = F.relu(w.mean() - 0.7)
            losses["loss_routing"] = self.routing_weight * (
                skip_shortfall * 2.0 + all_high_penalty
            )
        else:
            losses["loss_routing"] = torch.tensor(0.0, device=device)

        # ── 5a. Score regularization (keeps det_head active) ────
        if output is not None and output.scores.numel() > 0:
            # Push mean score toward 0.5 (prevents score collapse)
            losses["loss_score"] = 0.01 * ((output.scores.mean() - 0.5) ** 2)
        else:
            losses["loss_score"] = torch.tensor(0.0, device=device)

        # ── 5. Full-Image Mask Loss (optional) ───────────────────
        if self.full_mask_weight > 0 and gt_mask is not None and output.masks.numel() > 0:
            loss_mask = self._compute_full_mask_loss(output, gt_mask, images, device)
            losses["loss_mask"] = self.full_mask_weight * loss_mask
        else:
            losses["loss_mask"] = torch.tensor(0.0, device=device)

        # ── 6. Prototype Contrastive Loss ────────────────────────
        if self.proto_weight > 0 and aux is not None and "routed_tokens" in aux:
            prototypes = aux.get("prototypes") if aux else None
            routed = aux["routed_tokens"]
            if prototypes is not None and len(prototypes) > 0 and routed is not None and routed.numel() > 0:
                losses["loss_proto"] = self.proto_weight * self._proto_contrastive_loss(
                    routed, prototypes
                )
            else:
                losses["loss_proto"] = torch.tensor(0.0, device=device)
        else:
            losses["loss_proto"] = torch.tensor(0.0, device=device)

        # ── Total ────────────────────────────────────────────────
        total = (
            losses["loss_mask"]
            + losses["loss_score"]
            + losses["loss_density"]
            + losses["loss_sparse"]
            + losses["loss_routing"]
            + losses["loss_planning"]
            + losses["loss_proto"]
        )
        losses["loss"] = total
        return losses

    def _proto_contrastive_loss(self, tokens: Tensor, prototypes: Dict) -> Tensor:
        """Contrastive loss: pull tokens toward their prototype, push from others.

        Args:
            tokens: [N, C] routed token features.
            prototypes: Dict of class_id → prototype [C].
        Returns:
            Scalar loss.
        """
        if not prototypes or tokens.numel() == 0:
            return torch.tensor(0.0, device=tokens.device)

        proto_list = [p for p in prototypes.values()]
        proto_stack = torch.stack(proto_list, dim=0)  # [K, C]

        # Cosine similarity: [N, K]
        sim = F.cosine_similarity(
            tokens.unsqueeze(1), proto_stack.unsqueeze(0), dim=-1
        )

        # For each token, the closest prototype is the "positive"
        max_sim, _ = sim.max(dim=-1)  # [N]

        # Mean similarity to all prototypes (encourages discrimination)
        mean_sim = sim.mean(dim=-1)  # [N]

        # Contrastive: reward high max_sim, penalize high mean_sim (encourages specificity)
        # L = -max_sim + mean_sim = mean_sim - max_sim
        # Low when token is close to ONE prototype but far from others
        loss = (mean_sim - max_sim).mean()
        # Also add entropy: encourage uniform prototype usage
        proto_usage = sim.softmax(dim=0).mean(dim=1)  # [K] — how much each proto is used
        entropy = -(proto_usage * (proto_usage + 1e-8).log()).sum()
        return loss - 0.01 * entropy  # small entropy bonus

    def _focal_density_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Focal loss for density supervision.

        pred: [B, 1, H, W] importance predictions (sigmoid-activated by Ada-SPM)
        target: [B, 1, H, W] GT density ∈ [0, 1]

        Uses the FocalLoss class: α·(1-p_t)^γ·BCE
        With α=0.75: object cells (target high) get 3× the weight of background cells.
        """
        return self.density_focal(pred, target)

    def _build_gt_density_from_mask(
        self, shape, gt_mask, image_size, device
    ) -> Tensor:
        """Build GT density map from binary mask."""
        B, _, H_s, W_s = shape
        H_img, W_img = image_size

        if gt_mask is None:
            return torch.full((B, 1, H_s, W_s), 0.01, device=device)

        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        gt_mask = gt_mask.float().to(device)

        gt_density = F.interpolate(gt_mask, size=(H_s, W_s), mode="area")
        # Wide separation: background 0.01, object 0.90
        gt_density = 0.01 + 0.89 * gt_density.clamp(0, 1)
        return gt_density

    def _compute_full_mask_loss(
        self, output, gt_mask, images, device
    ) -> Tensor:
        """Compute loss on decoder masks — no projection needed.

        Uses the raw decoder masks (padded) and resizes GT to match.
        This preserves gradient flow since all operations are differentiable.
        """
        pred_masks = output.masks  # [N, max_h, max_w] — raw logits
        if pred_masks.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Take top-K masks by score for loss computation
        scores = output.scores  # [N]
        if scores.shape[0] > 10:
            _, topk = scores.topk(min(10, scores.shape[0]))
            pred_masks = pred_masks[topk]

        # Resize GT to match pred mask size
        Hp, Wp = pred_masks.shape[-2:]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask[0]
        gt_resized = F.interpolate(
            gt_mask.float().unsqueeze(0).unsqueeze(0).to(device),
            size=(Hp, Wp), mode="area"
        ).squeeze(0).squeeze(0)  # [Hp, Wp]

        # Mean over instances: compute Dice+Focal per instance, then average
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        for i in range(pred_masks.shape[0]):
            mask_i = pred_masks[i]  # [Hp, Wp] raw logits
            gt_i = gt_resized

            # Skip if mask is all zeros (degenerate)
            if mask_i.abs().sum() < 1e-6:
                continue

            dice_l = self.dice_loss(mask_i.unsqueeze(0).unsqueeze(0),
                                     gt_i.unsqueeze(0).unsqueeze(0))
            focal_l = self.focal_loss(mask_i.unsqueeze(0).unsqueeze(0),
                                       gt_i.unsqueeze(0).unsqueeze(0))
            total_loss = total_loss + dice_l * 5.0 + focal_l
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_valid


@LOSS.register()
class TrainingLoss(nn.Module):
    """Full training loss: mask matching + density + sparsity + routing.

    Handles the mismatch between tile-level crop predictions and
    full-image ground-truth annotations:

    1. Extracts GT masks from COCO-format annotations
    2. Matches predicted instances to GT via bbox IoU (greedy)
    3. Crops GT masks to prediction bbox region for pixel-level loss
    4. Computes density/sparsity/routing auxiliary losses
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        density_weight: float = 0.5,
        sparsity_weight: float = 0.002,
        routing_weight: float = 0.1,
        match_iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.mask_weight = mask_weight
        self.density_weight = density_weight
        self.sparsity_weight = sparsity_weight
        self.routing_weight = routing_weight
        self.match_iou_threshold = match_iou_threshold

        self.dice_loss = DiceLoss(smooth=1.0)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Track training progress for dynamic sparsity target
        self.register_buffer("_step_count", torch.zeros(1))

    def forward(
        self,
        output: SegmentationOutput,
        batch: Dict,
        aux: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        """Compute full training loss.

        Args:
            output: Model's SegmentationOutput.
            batch: Training batch with "annotations" and "images" keys.
            aux: Pipeline auxiliary outputs (importance, density, routing_weights).

        Returns:
            Dict with "loss" (total), "loss_mask", "loss_density",
            "loss_sparse", "loss_routing".
        """
        # Get device from aux tensors (always on GPU) rather than output
        # (which may have 0 instances → cpu fallback)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if aux is not None and "density" in aux:
            device = aux["density"].device
        losses: Dict[str, Tensor] = {}
        self._step_count += 1

        # ── 1. Mask Loss ──────────────────────────────────────────
        annotations = batch.get("annotations", [])
        images = batch.get("images", batch.get("image"))
        if images is not None and annotations and output.masks.numel() > 0:
            loss_mask = self._compute_mask_loss(output, annotations, images, device)
            losses["loss_mask"] = self.mask_weight * loss_mask
        else:
            losses["loss_mask"] = torch.tensor(0.0, device=device)

        # ── 2. Density Loss with GT density map ───────────────────
        if self.density_weight > 0 and aux is not None and "density" in aux:
            density = aux["density"]
            # Build GT density target from annotations (cheap heuristic)
            gt_density = self._build_gt_density(density.shape, annotations,
                                                 images.shape[-2:], device)
            losses["loss_density"] = self.density_weight * F.mse_loss(
                torch.nan_to_num(density, nan=0.5), gt_density
            )
        else:
            losses["loss_density"] = torch.tensor(0.0, device=device)

        # ── 3. Sparsity Loss (very weak, just a nudge) ─────────────
        if self.sparsity_weight > 0 and aux is not None and "importance" in aux:
            importance = aux["importance"]
            # Gentle push toward 0.4 sparsity — not strong enough to collapse
            losses["loss_sparse"] = self.sparsity_weight * F.l1_loss(
                torch.nan_to_num(importance, nan=0.5),
                torch.full_like(importance, 0.4),
            )
        else:
            losses["loss_sparse"] = torch.tensor(0.0, device=device)

        # ── 4. Routing Aux Loss (encourage sparsity) ───────────────
        if self.routing_weight > 0 and aux is not None and "routing_weights" in aux:
            routing_weights = aux["routing_weights"]
            w = torch.nan_to_num(routing_weights.squeeze(-1).float(), nan=0.5)
            # Skip ratio: how many tokens have weight < 0.15 (should be 40-60%)
            skip_frac = (w < 0.15).float().mean()
            # Penalize if skip ratio is too low (< 30%) — encourage sparsity
            skip_shortfall = F.relu(0.35 - skip_frac)
            # Also penalize if weights are all high (>0.8)
            all_high_penalty = F.relu(w.mean() - 0.7)
            losses["loss_routing"] = self.routing_weight * (
                skip_shortfall * 2.0 + all_high_penalty
            )
        else:
            losses["loss_routing"] = torch.tensor(0.0, device=device)

        # ── 5. Planning Alignment ─────────────────────────────────
        planning_loss = aux.get("planning_alignment_loss") if aux else None
        if planning_loss is not None and isinstance(planning_loss, Tensor):
            losses["loss_planning"] = planning_loss
        else:
            losses["loss_planning"] = torch.tensor(0.0, device=device)

        # ── Total ─────────────────────────────────────────────────
        total = (
            losses["loss_mask"]
            + losses["loss_density"]
            + losses["loss_sparse"]
            + losses["loss_routing"]
            + losses["loss_planning"]
        )
        losses["loss"] = total
        return losses

    def _build_gt_density(
        self,
        shape: tuple,
        annotations: List[List[Dict]],
        image_size: tuple,
        device: torch.device,
    ) -> Tensor:
        """Build GT density map from COCO annotations.

        Object regions → 0.8 density, background → 0.05 density.
        This gives AdaSPM a spatial target: learn to predict high
        importance near objects, low importance elsewhere.

        Args:
            shape: (B, 1, H_s, W_s) target density map shape.
            annotations: Per-image list of COCO annotation dicts.
            image_size: (H_img, W_img) original image dimensions.
            device: Target device.

        Returns:
            gt_density: [B, 1, H_s, W_s] float tensor.
        """
        B, _, H_s, W_s = shape
        H_img, W_img = image_size
        gt = torch.full((B, 1, H_s, W_s), 0.05, device=device)  # background

        cell_h = H_img / max(H_s, 1)
        cell_w = W_img / max(W_s, 1)

        for b in range(min(B, len(annotations))):
            for ann in annotations[b]:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                cx, cy = x + w / 2, y + h / 2
                sx = min(int(cx / cell_w), W_s - 1)
                sy = min(int(cy / cell_h), H_s - 1)
                # Set 3×3 region around instance center to high density
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = sy + dy, sx + dx
                        if 0 <= ny < H_s and 0 <= nx < W_s:
                            gt[b, 0, ny, nx] = 0.8
        return gt

    def _compute_mask_loss(
        self,
        output: SegmentationOutput,
        annotations: List[List[Dict]],
        images: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Match predictions to GT and compute Dice + Focal loss.

        Strategy:
            - Predictions have boxes in full-image coordinates
            - GT annotations have polygons → compute bboxes
            - Greedy IoU matching (no Hungarian for simplicity)
            - Crop GT mask to prediction bbox region, resize to pred mask size
        """
        B = images.shape[0] if images.dim() == 4 else 1
        H_img, W_img = images.shape[-2:]

        pred_boxes = output.boxes  # [N_inst, 4] in full-image coords
        pred_masks = output.masks  # [N_inst, max_h, max_w] crop-level
        pred_scores = output.scores  # [N_inst]

        if pred_boxes is None or pred_masks.numel() == 0:
            return torch.tensor(0.0, device=device)

        N_pred = pred_masks.shape[0]
        max_h, max_w = pred_masks.shape[-2:]

        # Collect all GT instances
        all_gt_boxes = []
        all_gt_masks_cropped = []
        for b in range(min(B, 1)):  # Only batch 0 for now (B=1 typical)
            anns = annotations[b] if b < len(annotations) else []
            for ann in anns:
                seg = ann.get("segmentation", [])
                bbox_gt = ann.get("bbox", None)  # COCO: [x, y, w, h]
                if not seg or bbox_gt is None:
                    continue
                x, y, w, h = bbox_gt
                gt_x1, gt_y1 = x, y
                gt_x2, gt_y2 = x + w, y + h
                all_gt_boxes.append([gt_x1, gt_y1, gt_x2, gt_y2])

                # Crop GT mask to prediction bbox area (done per match below)
                # Store raw mask data for cropping later
                all_gt_masks_cropped.append(seg)

        if not all_gt_boxes:
            return torch.tensor(0.0, device=device)

        gt_boxes_t = torch.tensor(all_gt_boxes, device=device, dtype=torch.float32)

        # ── Greedy matching via bbox IoU ──────────────────────────
        total_mask_loss = torch.tensor(0.0, device=device)
        matched = 0

        for i in range(N_pred):
            pred_box = pred_boxes[i]  # [4]: x1, y1, x2, y2
            # Compute IoU with all GT boxes
            inter_x1 = torch.max(pred_box[0], gt_boxes_t[:, 0])
            inter_y1 = torch.max(pred_box[1], gt_boxes_t[:, 1])
            inter_x2 = torch.min(pred_box[2], gt_boxes_t[:, 2])
            inter_y2 = torch.min(pred_box[3], gt_boxes_t[:, 3])
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h

            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_boxes_t[:, 2] - gt_boxes_t[:, 0]) * (
                gt_boxes_t[:, 3] - gt_boxes_t[:, 1]
            )
            union_area = pred_area + gt_area - inter_area
            ious = inter_area / (union_area + 1e-8)

            best_iou, best_j = ious.max(dim=0)
            if best_iou < self.match_iou_threshold:
                continue

            matched += 1
            # Get GT mask for this annotation
            seg = all_gt_masks_cropped[best_j.item()]
            try:
                from pycocotools.mask import frPyObjects, decode
                rles = frPyObjects(seg, H_img, W_img)
                if isinstance(rles, dict):
                    rles = [rles]
                gt_full_mask = torch.from_numpy(decode(rles)).float().to(device)
                if gt_full_mask.dim() == 3:
                    gt_full_mask = gt_full_mask.max(dim=2)[0]  # merge multiple objects
            except Exception:
                continue

            # Crop GT mask to prediction bbox
            x1 = int(pred_box[0].item())
            y1 = int(pred_box[1].item())
            x2 = int(pred_box[2].item())
            y2 = int(pred_box[3].item())
            x1 = max(0, min(x1, W_img - 1))
            y1 = max(0, min(y1, H_img - 1))
            x2 = max(x1 + 1, min(x2, W_img))
            y2 = max(y1 + 1, min(y2, H_img))
            gt_crop = gt_full_mask[y1:y2, x1:x2]

            # Resize to prediction mask size
            if gt_crop.numel() > 0:
                gt_crop = F.interpolate(
                    gt_crop.unsqueeze(0).unsqueeze(0),
                    size=(max_h, max_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0).clamp(0, 1)
            else:
                continue

            # Dice + Focal on this matched pair
            pred_mask_single = pred_masks[i:i + 1]  # [1, max_h, max_w]
            gt_mask_single = gt_crop.unsqueeze(0)  # [1, max_h, max_w]
            dice_l = self.dice_loss(pred_mask_single, gt_mask_single)
            focal_l = self.focal_loss(pred_mask_single, gt_mask_single)
            total_mask_loss = total_mask_loss + dice_l * 5.0 + focal_l

        if matched > 0:
            return total_mask_loss / matched
        return torch.tensor(0.0, device=device)


# ── Pipeline Placeholder ─────────────────────────────────────────────


@SEGMENTATION.register()
class AdaTileFastSAMPipeline(nn.Module):
    """Full AdaTile-FastSAM instance segmentation pipeline.

    Components:
        1. Backbone: image → multi-scale features
        2. Ada-SPM: features → importance map + density estimation
        3. DynamicTileTokenizer: image + importance → tiles + tile tokens
        4. BaseRouter: tokens → routed (expert-assigned) tokens
        5. PrototypeMemory (few-shot): support → class prototypes
        6. SegmentationDecoder: tile tokens + prototypes → masks

    Training:
        - End-to-end with auxiliary losses (density, entropy, sparsity)
        - Few-shot support/query dual forward
        - Sparse skip with token budget control
    """

    def __init__(
        self,
        backbone: nn.Module,
        sparse_predictor: nn.Module,
        tokenizer: nn.Module,
        router: nn.Module,
        decoder: nn.Module,
        prototype_memory: Optional[nn.Module] = None,
        global_context: Optional[nn.Module] = None,
        cat_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.sparse_predictor = sparse_predictor
        self.tokenizer = tokenizer
        self.router = router
        self.decoder = decoder
        self.prototype_memory = prototype_memory
        self.global_context = global_context
        self.cat_module = cat_module  # CAT-SAM style conditional tuning

    def forward_standard(
        self, image: Tensor
    ) -> Tuple[SegmentationOutput, Dict[str, Tensor]]:
        """Standard (non-few-shot) forward pass.

        Architecture (Gemini-inspired Global+Local):
          1. Global Thumbnail (512×512) → Backbone → multi-scale features
          2. Ada-SPM on thumbnail features → importance map
          3. Upsample importance → map to original image scale
          4. Tile Planner uses importance to decide WHERE to extract tiles
          5. Tokenizer extracts tiles from ORIGINAL image (F.grid_sample)
          6. DTR-v2 Router → selective attention
          7. Decoder → instance masks

        The backbone NEVER sees the full-resolution image. Only tile
        patches are extracted at native resolution. This is the key
        to handling 4000×4000+ images on consumer GPUs.

        Args:
            image: [B, 3, H, W] original image at native resolution.

        Returns:
            output: SegmentationOutput with masks, scores, boxes.
            aux: Dict of auxiliary outputs (importance, density, routing weights).
        """
        B, C, H_orig, W_orig = image.shape

        # 1. Global Thumbnail — backbone only sees this
        thumbnail_size = 512
        thumbnail = F.interpolate(
            image,
            size=(thumbnail_size, thumbnail_size),
            mode='bilinear',
            align_corners=False,
        )
        # Force fp32 backbone — AMP fp16 causes NaN in timm ResNet features
        with torch.cuda.amp.autocast(enabled=False):
            features = self.backbone(thumbnail.float())

        # ── CAT: Apply FPN adapters to backbone features ──────────
        if self.cat_module is not None:
            features = self.cat_module.adapt_fpn(features)

        # 2. Ada-SPM on thumbnail features → importance at thumbnail scale
        spm_output = self.sparse_predictor(features)

        # ── CAT: Apply PromptBridge bias to Ada-SPM features ──────
        if self.cat_module is not None:
            spm_bias = self.cat_module.get_spm_bias(B)
            # Inject bias into the fused features before density head
            # The bias is added to Ada-SPM's internal representation
            if hasattr(self.sparse_predictor, 'set_spm_bias'):
                self.sparse_predictor.set_spm_bias(spm_bias)
            # Store CAT token for decoder use
            cat_token = self.cat_module.get_cat_token(B)
        if isinstance(spm_output, tuple):
            spm_output, _spm_aux = spm_output
        importance_thumb = spm_output.importance  # [B, 1, H_t/32, W_t/32]
        density_thumb = spm_output.density
        granularity_hard_thumb = spm_output.granularity_hard
        granularity_soft_thumb = spm_output.granularity_soft

        # 3. Upsample importance to original image scale for tile planning.
        #    The tile planner divides the original image into a grid of cells.
        #    We need importance at each cell — interpolate from thumbnail scale.
        H_s = max(H_orig // 32, 1)
        W_s = max(W_orig // 32, 1)
        importance = F.interpolate(
            importance_thumb,
            size=(H_s, W_s),
            mode='bilinear',
            align_corners=False,
        )
        density = F.interpolate(
            density_thumb,
            size=(H_s, W_s),
            mode='bilinear',
            align_corners=False,
        )
        # granularity_hard: use nearest-neighbor to preserve integer indices
        granularity_hard = None
        if granularity_hard_thumb is not None:
            granularity_hard = F.interpolate(
                granularity_hard_thumb.float(),
                size=(H_s, W_s),
                mode='nearest',
            ).long()

        # 4. Free thumbnail tensors before tile extraction
        del thumbnail, features, spm_output
        torch.cuda.empty_cache()

        # ── NaN guard: detect + replace NaN importance before tile planning ──
        import logging
        _log = logging.getLogger("adatile.segmentation")
        nan_mask = torch.isnan(importance)
        if nan_mask.any():
            nan_pct = nan_mask.float().mean().item() * 100
            _log.warning(
                "[Pipeline] ⚠ importance has %.1f%% NaN after upsampling! "
                "Falling back to uniform 0.5. Image value range: [%.4f, %.4f]",
                nan_pct, image.min().item(), image.max().item(),
            )
            importance = torch.where(
                nan_mask,
                torch.full_like(importance, 0.5),
                importance,
            )
        # Also clamp density for safety (NaN in density would cascade)
        if torch.isnan(density).any():
            density = torch.nan_to_num(density, nan=0.5)

        # 5. Dynamic tile tokenizer — extracts tiles from ORIGINAL image.
        #    TRAINING: uniform tiles (importance=None) → decoder always has full signal.
        #              Importance is trained via GT-driven losses in parallel.
        #    INFERENCE: importance-based sparse tiling.
        importance_for_tiles = None if self.training else importance
        tile_infos, tile_tokens = self.tokenizer(
            image,
            features=None,
            importance=importance_for_tiles,
            granularity_hard=granularity_hard,
        )

        # Log importance stats for debugging
        threshold = getattr(self.sparse_predictor, 'importance_threshold', 0.3)
        n_tiles = sum(len(t) for t in tile_infos) if tile_infos else 0
        _log.info(
            "[AdaSPM] importance min=%.3f mean=%.3f max=%.3f "
            "above_thresh=%.1f%% tiles=%d",
            importance.min().item(), importance.mean().item(),
            importance.max().item(),
            (importance > threshold).float().mean().item() * 100,
            n_tiles,
        )
        if n_tiles == 0:
            _log.error(
                "[Pipeline] ⚠ ZERO tiles generated! importance_threshold=%.3f. "
                "All importance values below threshold → no tiles → no gradient. "
                "Lower cfg.sparse.importance_threshold (currently %.3f). "
                "→ FALLING BACK to uniform coarse tiling (16 tiles).",
                threshold, threshold,
            )
            # Fallback: force uniform grid of coarse tiles so gradients flow
            from adatile.tokenizer.tile_planner import TileSpec
            H_s2, W_s2 = importance.shape[-2:]
            fallback_specs: list = []
            grid_h, grid_w = max(H_s2 // 4, 1), max(W_s2 // 4, 1)
            cell_h, cell_w = H_orig // grid_h, W_orig // grid_w
            for gy in range(grid_h):
                for gx in range(grid_w):
                    fallback_specs.append(TileSpec(
                        x1=gx * cell_w, y1=gy * cell_h,
                        x2=min((gx + 1) * cell_w, W_orig),
                        y2=min((gy + 1) * cell_h, H_orig),
                        tile_size=max(cell_h, cell_w),
                        stride=max(cell_h // 2, cell_w // 2, 1),
                        importance=0.5, priority=1.0, density=0.5, scale_level=3,
                    ))
            # Re-generate tokenizer output from fallback specs
            img_b = image[0:1]  # [1, 3, H, W]
            from adatile.tokenizer.token_generator import TokenGenerator
            # Get the tokenizer's generator (already built)
            if hasattr(self.tokenizer, 'generator'):
                gen = self.tokenizer.generator
            else:
                gen = TokenGenerator(patch_size=224, in_channels=3, embed_dim=256)
            fb_tokens_raw, _, _ = gen(img_b, fallback_specs)
            if fb_tokens_raw.shape[0] > 0:
                tile_tokens = fb_tokens_raw.unsqueeze(0)  # [1, N, C]
            # Build TileInfo from fallback specs
            fallback_infos: list = []
            for i, s in enumerate(fallback_specs):
                fallback_infos.append(s.to_tile_info(f"batch_0", i))
            tile_infos = [fallback_infos]
            _log.info("[Pipeline] Fallback tiles=%d generated.", len(fallback_specs))

        # 6. Clean up intermediate tensors
        torch.cuda.empty_cache()

        # 7. Dynamic token router
        imp_for_router = self._extract_token_importance(
            tile_infos, importance, image.shape[-2:],
        )
        metadata = {"importance": imp_for_router} if imp_for_router is not None else None
        route_decision = self.router(tile_tokens, metadata)

        # 6. Decoder
        skipped_indices = (
            route_decision.skipped_mask.nonzero(as_tuple=True)[0]
            if route_decision.skipped_mask.any()
            else torch.empty(0, dtype=torch.long, device=route_decision.skipped_mask.device)
        )
        output = self.decoder(
            route_decision.routed_tokens,
            tile_infos,
            global_features=None,
            image_size=(H_orig, W_orig),
            skipped_indices=skipped_indices,
        )

        # ── GT-driven planning alignment (computed in loss, not here) ──
        # base_loss deleted — it unconditionally pushes importance down
        # planning_alignment_loss now computed from GT mask, not planner output
        aux = {
            "importance": importance,
            "density": density,
            "granularity_hard": granularity_hard,
            "granularity_soft": None,
            "routing_weights": route_decision.routing_weights,
            "routed_tokens": route_decision.routed_tokens,
            "skipped_indices": skipped_indices,
            "planner_stats": getattr(self.tokenizer, '_last_plan', None),
        }

        # ── CAT: Pass CAT-Token through decoder for mask generation ──
        if self.cat_module is not None:
            cat_token = self.cat_module.get_cat_token(B)
            aux["cat_token"] = cat_token
            # Use CAT mask head if available
            if hasattr(self.cat_module, 'mask_head') and route_decision.routed_tokens.numel() > 0:
                cat_mask = self.cat_module.mask_head(
                    route_decision.routed_tokens, cat_token
                )
                aux["cat_mask"] = cat_mask

        return output, aux

    def _extract_token_importance(
        self,
        tile_infos: List[List[TileInfo]],
        importance: Optional[Tensor],
        image_size: Tuple[int, int],
    ) -> Optional[Tensor]:
        """Extract per-tile importance from the spatial importance map.

        For each tile, samples the importance map at the tile center.
        If no importance map is available, returns None (router uses default).
        """
        if importance is None:
            return None
        H, W = image_size
        Hs, Ws = importance.shape[-2:]
        per_batch = []
        for batch_tiles in tile_infos:
            imps = []
            for info in batch_tiles:
                cx = (info.x1 + info.x2) / (2 * W)
                cy = (info.y1 + info.y2) / (2 * H)
                sx = min(int(cx * Ws), Ws - 1)
                sy = min(int(cy * Hs), Hs - 1)
                if importance.dim() == 4:
                    imps.append(importance[0, 0, sy, sx])
                elif importance.dim() == 3:
                    imps.append(importance[0, sy, sx])
                else:
                    imps.append(importance[sy, sx])
            per_batch.append(torch.stack(imps) if imps else torch.empty(0, device=importance.device))
        if per_batch:
            return torch.stack(per_batch, dim=0)  # [B, N]
        return None

    def forward_fewshot(
        self,
        support_images: Tensor,
        support_masks: Tensor,
        query_images: Tensor,
        class_ids: Optional[List[int]] = None,
    ) -> Tuple[SegmentationOutput, Dict[str, Tensor]]:
        """Few-shot forward pass: support set → prototypes → query prediction.

        Uses thumbnail-based backbone for both support and query branches
        to avoid OOM on high-resolution aerial imagery.

        Args:
            support_images: [B_s, 3, H, W] support images.
            support_masks: [B_s, H, W] binary support masks.
            query_images: [B_q, 3, H, W] query images.
            class_ids: Optional class labels for support samples.

        Returns:
            output: SegmentationOutput for query images.
            aux: Dict of auxiliary outputs.
        """
        if self.prototype_memory is None:
            raise ValueError("PrototypeMemory is required for few-shot inference.")

        thumbnail_size = 512

        # Support branch: thumbnail → features → prototypes
        support_thumb = F.interpolate(
            support_images, size=(thumbnail_size, thumbnail_size),
            mode='bilinear', align_corners=False,
        )
        with torch.cuda.amp.autocast(enabled=False):
            support_features = self.backbone(support_thumb.float())
        prototypes = self.prototype_memory(support_features, support_masks, class_ids)

        # Query branch: thumbnail → features → Ada-SPM → importance
        _, _, Hq, Wq = query_images.shape
        query_thumb = F.interpolate(
            query_images, size=(thumbnail_size, thumbnail_size),
            mode='bilinear', align_corners=False,
        )
        with torch.cuda.amp.autocast(enabled=False):
            features = self.backbone(query_thumb.float())
        spm_output = self.sparse_predictor(features)
        if isinstance(spm_output, tuple):
            spm_output, _spm_aux = spm_output
        importance_thumb = spm_output.importance
        density_thumb = spm_output.density
        granularity_hard_thumb = spm_output.granularity_hard

        # Upsample importance to query image scale
        H_s = max(Hq // 32, 1)
        W_s = max(Wq // 32, 1)
        importance = F.interpolate(
            importance_thumb, size=(H_s, W_s),
            mode='bilinear', align_corners=False,
        )
        density = F.interpolate(
            density_thumb, size=(H_s, W_s),
            mode='bilinear', align_corners=False,
        )
        granularity_hard = None
        if granularity_hard_thumb is not None:
            granularity_hard = F.interpolate(
                granularity_hard_thumb.float(),
                size=(H_s, W_s), mode='nearest',
            ).long()

        # ── NaN guard (few-shot) ──
        import logging
        _log = logging.getLogger("adatile.segmentation")
        if torch.isnan(importance).any():
            _log.warning("[Pipeline-fs] ⚠ NaN importance, falling back to uniform 0.5")
            importance = torch.where(
                torch.isnan(importance),
                torch.full_like(importance, 0.5),
                importance,
            )
        if torch.isnan(density).any():
            density = torch.nan_to_num(density, nan=0.5)

        # Tile tokenizer on ORIGINAL query images
        importance_for_tiles = None if self.training else importance
        tile_infos, tile_tokens = self.tokenizer(
            query_images, features=None, importance=importance_for_tiles,
            granularity_hard=granularity_hard,
        )

        imp_for_router = self._extract_token_importance(
            tile_infos, importance, query_images.shape[-2:],
        )
        metadata = {
            "importance": imp_for_router if imp_for_router is not None else None,
            "prototypes": prototypes,
        }
        route_decision = self.router(tile_tokens, metadata)

        skipped_indices = (
            route_decision.skipped_mask.nonzero(as_tuple=True)[0]
            if route_decision.skipped_mask.any()
            else torch.empty(0, dtype=torch.long, device=route_decision.skipped_mask.device)
        )
        output = self.decoder(
            route_decision.routed_tokens,
            tile_infos,
            prototypes=prototypes,
            image_size=(Hq, Wq),
            skipped_indices=skipped_indices,
        )

        aux = {
            "importance": importance,
            "density": density,
            "granularity_hard": granularity_hard,
            "prototypes": prototypes,
            "routing_weights": route_decision.routing_weights,
            "routed_tokens": route_decision.routed_tokens,
            "planner_stats": getattr(self.tokenizer, '_last_plan', None),
        }
        return output, aux

    def forward(
        self,
        image: Tensor,
        support_images: Optional[Tensor] = None,
        support_masks: Optional[Tensor] = None,
        class_ids: Optional[List[int]] = None,
    ) -> Tuple[SegmentationOutput, Dict[str, Tensor]]:
        """Unified forward: auto-switches between standard and few-shot.

        If support_images/support_masks are provided, runs few-shot mode.
        """
        if support_images is not None and support_masks is not None:
            return self.forward_fewshot(
                support_images, support_masks, image, class_ids
            )
        return self.forward_standard(image)
