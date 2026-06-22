#!/usr/bin/env python3
"""
B-09: FastSAM-FSS on NWPU-VHR-10 -- Few-Shot Instance Segmentation
===================================================================

NWPU-VHR-10 10-class geospatial object dataset.
Only bbox annotations available: bbox used as weak instance mask.

实验设计 | Design:
    - Frozen FastSAM backbone -> P4 features
    - Support: K images -> per-class prototype (bbox-masked P4 features)
    - Query: P4 -> cosine similarity with prototypes -> upsampled binary mask
    - Evaluation: binary IoU (bbox as GT mask)

用法 | Usage:
    python tools/paper_b/eval_b09_nwpu_fewshot.py \
        --data-root data/NWPU --shots 1,3,5 --episodes 100 --device cuda
"""

import sys, argparse, time, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.datasets.nwpu import NWPUDataset, CLASS_NAMES

NUM_CLASSES = 11  # 10 FG + background


def binary_iou(pred, gt):
    """Compute binary IoU for a single image."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


class FewShotPrototypeMatcher(nn.Module):
    """
    Non-parametric few-shot prototype matching for instance segmentation.

    Support: K images -> per-class prototype (bbox-masked P4 features)
    Query: P4 -> cosine similarity -> binary mask
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def compute_class_prototype(self, p4_features, masks, target_class):
        """
        Compute prototype for a specific class from support features.

        p4_features: list of [C, H_p4, W_p4] per support image
        masks:       list of [H_orig, W_orig] bbox masks for target_class
        target_class: class ID to compute prototype for

        Returns: [C] normalized prototype vector
        """
        feat_dim = p4_features[0].shape[0]
        p4_h, p4_w = p4_features[0].shape[1], p4_features[0].shape[2]

        all_feats = []
        for i in range(len(p4_features)):
            m = masks[i]
            if m.dim() == 3:
                m = m.squeeze(0)
            # Downsample mask to P4 spatial size
            mask_4d = m.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            mask_p4 = F.interpolate(mask_4d, size=(p4_h, p4_w),
                                     mode='nearest').squeeze(0)  # [1, H_p4, W_p4]
            if mask_p4.sum() > 0:
                # Mean feature within mask region
                weighted = (p4_features[i] * mask_p4).sum(dim=(1, 2)) / mask_p4.sum()
                all_feats.append(weighted)

        if not all_feats:
            return torch.zeros(feat_dim, device=p4_features[0].device)

        return F.normalize(torch.stack(all_feats).mean(dim=0), dim=0, p=2)

    def forward(self, query_p4, prototype, target_size=None):
        """
        Query P4 -> cosine similarity with class prototype -> binary logit map.
        """
        q_norm = F.normalize(query_p4, dim=1, p=2)
        sim = (q_norm * prototype.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        logit = sim / self.temperature  # [1, 1, H_p4, W_p4]

        if target_size is not None:
            logit = F.interpolate(logit, size=target_size, mode='bilinear',
                                  align_corners=False)
        return logit  # [1, 1, H_target, W_target]


@torch.no_grad()
def run_episode(support_samples, query_sample, model, backbone, target_class, device):
    """Single episode: support -> prototype -> query prediction."""

    # Support: extract features from each image
    support_p4s = []
    support_masks = []
    for img_t, mask_t in support_samples:
        feats = backbone(img_t.unsqueeze(0).to(device))
        support_p4s.append(feats['p4'][0])  # [C, H_p4, W_p4]
        support_masks.append(mask_t)

    # Compute class prototype from support bbox masks
    prototype = model.compute_class_prototype(support_p4s, support_masks, target_class)
    if prototype.sum() == 0:
        return 0.0

    # Query prediction
    query_img, query_gt_mask = query_sample
    feats = backbone(query_img.unsqueeze(0).to(device))
    target_h, target_w = query_gt_mask.shape
    logit = model(feats['p4'], prototype, target_size=(target_h, target_w))

    # Threshold prediction
    pred = (logit.squeeze().cpu() > 0).numpy()
    gt = query_gt_mask.cpu().numpy() > 0

    return binary_iou(pred, gt)


def evaluate_fewshot(model, backbone, train_ds, val_ds, device,
                     shot, n_episodes, logger, tag):
    """Run episodic evaluation."""

    # Build class-to-image index on train set
    class_to_images = {}
    for c in range(1, 11):
        class_to_images[c] = train_ds.class_to_images(c)
        logger.log_info(f'{tag}/classes',
                       f'  Class {c} ({CLASS_NAMES[c]}): {len(class_to_images[c])} train images')

    rng = np.random.RandomState(42)
    all_ious = []
    per_cls_ious = defaultdict(list)
    t0 = time.perf_counter()
    log_every = max(10, n_episodes // 10)

    for ep in tqdm(range(n_episodes), desc=f'  {shot}-shot'):
        # Pick a random class that has enough training images
        valid_classes = [c for c in range(1, 11)
                        if len(class_to_images[c]) >= shot]
        if not valid_classes:
            continue
        query_class = int(rng.choice(valid_classes))

        # Sample K support images
        candidates = class_to_images[query_class]
        support_idxs = rng.choice(candidates, shot, replace=False)

        # Build support samples: for each support image, create bbox mask for query_class
        support_samples = []
        for si in support_idxs:
            s = train_ds[si]
            img = s['image']
            # Build binary mask for query_class (union of all bbox masks for this class)
            mask = torch.zeros(img.shape[1], img.shape[2])
            for j, lbl in enumerate(s['labels']):
                if lbl.item() == query_class:
                    mask = (mask.bool() | s['masks'][j].bool()).float()
            support_samples.append((img, mask))

        # Sample query image from val (must contain query_class)
        val_candidates = val_ds.class_to_images(query_class)
        if not val_candidates:
            continue
        qi = int(rng.choice(val_candidates))
        q = val_ds[qi]

        # Query GT mask: union of all bbox masks for query_class
        query_mask = torch.zeros(q['image'].shape[1], q['image'].shape[2])
        for j, lbl in enumerate(q['labels']):
            if lbl.item() == query_class:
                query_mask = (query_mask.bool() | q['masks'][j].bool()).float()

        query_sample = (q['image'], query_mask)

        try:
            iou = run_episode(support_samples, query_sample, model, backbone,
                            query_class, device)
            all_ious.append(iou)
            per_cls_ious[query_class].append(iou)
        except Exception as e:
            logger.log_warn(f'{tag}/warn', f'Ep {ep+1}: {type(e).__name__}: {e}')

        if (ep + 1) % log_every == 0 and all_ious:
            running = float(np.mean(all_ious[-log_every:]))
            total = float(np.mean(all_ious))
            logger.log_info(f'{tag}/progress',
                           f'  Ep {ep+1}/{n_episodes}: running_IoU={running:.4f} total_IoU={total:.4f}')

    dt = time.perf_counter() - t0

    # Per-class breakdown
    per_cls_avg = {}
    for c in sorted(per_cls_ious.keys()):
        avg = float(np.mean(per_cls_ious[c]))
        per_cls_avg[str(c)] = avg
        logger.log_info(f'{tag}/per_cls',
                       f'  {CLASS_NAMES[c]:<20} IoU={avg:.4f} ({len(per_cls_ious[c])} eps)')

    n_valid = len(all_ious)
    result = {
        'miou_mean': float(np.mean(all_ious)) if all_ious else 0.0,
        'miou_std': float(np.std(all_ious)) if all_ious else 0.0,
        'n_valid': n_valid,
        'per_class_iou': per_cls_avg,
    }
    logger.log_info(f'{tag}/done',
                   f'  {shot}-shot: {n_valid} episodes in {dt:.0f}s ({dt/max(n_valid,1):.2f}s/ep)')
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='data/NWPU')
    p.add_argument('--shots', type=str, default='1,3,5')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--output-dir', type=str, default='runs/b09_nwpu_fewshot')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    shots = [int(x.strip()) for x in args.shots.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger('b09')
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / 'b09.jsonl')))

    train_ds = NWPUDataset(args.data_root, split='train')
    val_ds = NWPUDataset(args.data_root, split='val')
    logger.log_info('b09/data', f'Train: {len(train_ds)} imgs | Val: {len(val_ds)} imgs')
    logger.log_info('b09/data', f'10-class object detection -> bbox-as-mask instance segmentation')

    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    logger.log_info('b09/model', 'FastSAM backbone (frozen)')
    logger.log_info('b09/method', 'Few-Shot Prototype Matching (bbox-masked)')

    model = FewShotPrototypeMatcher().to(device)
    model.eval()

    all_results = {}
    for shot in shots:
        tag = f'b09/{shot}shot'
        logger.log_info(f'{tag}/start',
                       f'\n{chr(9472)*50}\n  {shot}-Shot ({args.episodes} eps)\n{chr(9472)*50}')
        t0 = time.perf_counter()
        result = evaluate_fewshot(model, backbone, train_ds, val_ds, device,
                                  shot, args.episodes, logger, tag)
        dt = time.perf_counter() - t0
        all_results[f'{shot}shot'] = result
        logger.log_info(f'{tag}/result',
                       f'  IoU={result["miou_mean"]:.4f}+-{result["miou_std"]:.4f} '
                       f'({result["n_valid"]} eps, {dt:.0f}s)')
        logger.log_metric(f'b09/iou_{shot}shot', result['miou_mean'],
                         tags=['b09', 'nwpu', f'{shot}shot'])

    # Summary
    logger.log_info('b09/summary', '\n' + '=' * 65)
    logger.log_info('b09/summary', '  FastSAM Few-Shot -- NWPU-VHR-10 (Bbox-as-Mask)')
    logger.log_info('b09/summary', '=' * 65)
    header = f'  {"Shot":<8} {"IoU":>10} {"+-std":>8}'
    for c in range(1, 11):
        header += f' {CLASS_NAMES[c][:6]:>8}'
    logger.log_info('b09/summary', header)
    logger.log_info('b09/summary', '  ' + '-' * (10+10+8+10*8))
    for shot in shots:
        r = all_results[f'{shot}shot']
        line = f'  {shot:<8} {r["miou_mean"]*100:>9.2f}% {r["miou_std"]*100:>7.2f}%'
        for c in range(1, 11):
            pc = r['per_class_iou'].get(str(c), 0.0)
            line += f' {pc*100:>7.2f}%'
        logger.log_info('b09/summary', line)

    logger.log_info('done', f'Results saved to {output_dir}/')


if __name__ == '__main__':
    main()
