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
        loss_dice = self.dice(pred_masks.sigmoid(), target_masks)
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
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

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
        if aux is not None and "density" in aux:
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
        if aux is not None and "importance" in aux:
            importance = aux["importance"]
            # Gentle push toward 0.4 sparsity — not strong enough to collapse
            losses["loss_sparse"] = self.sparsity_weight * F.l1_loss(
                torch.nan_to_num(importance, nan=0.5),
                torch.full_like(importance, 0.4),
            )
        else:
            losses["loss_sparse"] = torch.tensor(0.0, device=device)

        # ── 4. Routing Aux Loss (encourage sparsity) ───────────────
        routing_weights = aux.get("routing_weights") if aux else None
        if routing_weights is not None and routing_weights.numel() > 0:
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
            dice_l = self.dice_loss(pred_mask_single.sigmoid(), gt_mask_single)
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
    ):
        super().__init__()
        self.backbone = backbone
        self.sparse_predictor = sparse_predictor
        self.tokenizer = tokenizer
        self.router = router
        self.decoder = decoder
        self.prototype_memory = prototype_memory
        self.global_context = global_context

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

        # 2. Ada-SPM on thumbnail features → importance at thumbnail scale
        spm_output = self.sparse_predictor(features)
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

        # 5. Dynamic tile tokenizer — extracts tiles from ORIGINAL image
        #    using importance from the thumbnail-scale backbone/Ada-SPM.
        tile_infos, tile_tokens = self.tokenizer(
            image,             # ← original image for F.grid_sample tile extraction
            features=None,     # not used by tokenizer
            importance=importance,
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

        # 7. Base training loss from Ada-SPM density (always has gradient)
        #    + planning alignment loss when available
        base_loss = density.mean() * 0.01  # small density regularization signal

        # Only compute planning_loss if importance is clean (no NaN).
        # NaN in importance → NaN BCE gradient → corrupted backbone weights.
        planning_loss = base_loss
        if (hasattr(self.tokenizer, 'planner') and hasattr(self.tokenizer, '_last_plan')
                and not torch.isnan(importance).any()):
            plan = self.tokenizer._last_plan
            if plan is not None:
                planning_loss = base_loss + self.tokenizer.planner.compute_planning_alignment_loss(
                    importance[0], plan,
                )

        aux = {
            "importance": importance,
            "density": density,
            "granularity_hard": granularity_hard,
            "granularity_soft": None,  # upsampled soft would be misleading
            "routing_weights": route_decision.routing_weights,
            "routed_tokens": route_decision.routed_tokens,
            "skipped_indices": skipped_indices,
            "planner_stats": getattr(self.tokenizer, '_last_plan', None),
            "planning_alignment_loss": planning_loss,
        }
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
        tile_infos, tile_tokens = self.tokenizer(
            query_images, features=None, importance=importance,
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

        base_loss = density.mean() * 0.01
        # Only compute planning_loss if importance is clean (no NaN).
        # NaN in importance → NaN BCE gradient → corrupted backbone weights.
        planning_loss = base_loss
        if (hasattr(self.tokenizer, 'planner') and hasattr(self.tokenizer, '_last_plan')
                and not torch.isnan(importance).any()):
            plan = self.tokenizer._last_plan
            if plan is not None:
                planning_loss = base_loss + self.tokenizer.planner.compute_planning_alignment_loss(
                    importance[0], plan,
                )

        aux = {
            "importance": importance,
            "density": density,
            "granularity_hard": granularity_hard,
            "prototypes": prototypes,
            "routing_weights": route_decision.routing_weights,
            "routed_tokens": route_decision.routed_tokens,
            "planner_stats": getattr(self.tokenizer, '_last_plan', None),
            "planning_alignment_loss": planning_loss,
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
