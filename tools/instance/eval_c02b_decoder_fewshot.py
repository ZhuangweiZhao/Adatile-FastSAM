#!/usr/bin/env python3
"""
C-02B: FastSAM Few-Shot Decoder — 验证 P4 信息能否被 Decoder 提取
=====================================================================
FastSAM Few-Shot Decoder — Verify P4 info can be extracted by a decoder.

回答 | Answer:
    C-02A 证明 Prototype Matching 失败 (mIoU≈0.3%)。
    但这不代表 P4 没有信息——B-04 Decoder val_fg5≈0.47 说明 P4 在语义分割上工作。
    C-02B 验证：加一个可训练的 Decoder，P4 能否做 Few-Shot 实例分割？

实验 | Design:
    - 冻结 FastSAM → P4 特征 | Frozen FastSAM → P4 features
    - Support: K images → FG prototype (masked mean P4)
    - Query: P4 → cosine_sim(prototype) → Trainable Refine CNN → binary mask
    - Episodic training: Focal + Dice loss, 30 epochs
    - 3 类 (ship, small_vehicle, storage_tank), 1/3/5 shot
    - 对比: C-02A (non-parametric baseline)

用法 | Usage::
    python tools/instance/eval_c02b_decoder_fewshot.py \
        --src-root data/iSAID_processed --device cuda \
        --shots 1,3,5 --epochs 30

C-02A (non-parametric) vs C-02B (trainable decoder):
    C-02A: proto = mean(P4[fg]) → cosine_sim → threshold   (0 params)
    C-02B: proto = mean(P4[fg]) → cosine_sim → Refine CNN   (~10K params, trained)
"""

import sys, argparse, time, json, datetime
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
from adatile.datasets.isaid_tile_wrapper import ISAIDTileWrapper

# ── 复用 C-02A 的数据集和 mask 渲染 | Reuse C-02A dataset + mask rendering ──
from tools.instance.eval_c02a_fastsam_fewshot import (
    ISAIDInstanceDataset, TARGET_CLASSES,
)

# ═══════════════════════════════════════════════════════════════════
# Few-Shot Decoders
# ═══════════════════════════════════════════════════════════════════

from adatile.utils.prototype import compute_fg_prototype


class ProtoRefineDecoder(nn.Module):
    """
    Baseline Decoder | Proto → cosine_sim → Refine CNN → mask.
    与 C-02A 的唯一区别是加了可训练的 Refine CNN (~10K params)。
    Only difference from C-02A: the trainable Refine CNN.
    """

    def __init__(self, feat_dim: int = 1280, temperature: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
        )
        n = sum(p.numel() for p in self.parameters())
        print(f"[ProtoRefineDecoder] Trainable: {n:,}")

    def compute_fg_prototype(self, p4_features, masks):
        return compute_fg_prototype(p4_features, masks, self.feat_dim)

    def forward(self, query_p4, fg_prototype, target_size=None):
        q_norm = F.normalize(query_p4, dim=1, p=2)
        sim = (q_norm * fg_prototype.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        sim = sim / self.temperature
        x = self.refine(sim)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear',
                            align_corners=False)
        return x

# ═══════════════════════════════════════════════════════════════════
# Binary IoU (same as B-09)
# ═══════════════════════════════════════════════════════════════════

def binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute binary IoU for a single prediction."""
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Episodic Training
# ═══════════════════════════════════════════════════════════════════

def train_episode(decoder, backbone, support_idxs, query_idx,
                  train_ds, val_ds, query_class, device, opt):
    """Single training episode: support → proto → query → loss."""

    # Support
    support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
    support_masks_raw = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]

    # Query
    query_img = val_ds.load_image(query_idx).unsqueeze(0).to(device)
    query_mask = val_ds.render_class_mask(query_idx, query_class)
    query_mask = query_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1024, 1024]

    # Forward
    support_feats = backbone(support_imgs)
    support_p4s = [support_feats['p4'][i] for i in range(len(support_idxs))]

    query_feats = backbone(query_img)
    query_p4 = query_feats['p4']

    fg_proto = decoder.compute_fg_prototype(support_p4s, support_masks_raw)
    if fg_proto.sum() == 0:
        return None  # empty prototype, skip

    logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape[2:]))

    # Focal + Dice loss
    ce = F.binary_cross_entropy_with_logits(logit, query_mask, reduction='none')
    focal_loss = ((1 - torch.exp(-ce)) ** 5.0 * ce).mean()

    prob = torch.sigmoid(logit)
    inter = (prob * query_mask).sum()
    union = prob.sum() + query_mask.sum() + 1e-8
    dice_loss = 1.0 - (2 * inter / union)

    loss = 0.5 * focal_loss + 0.5 * dice_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


@torch.no_grad()
def validate_episode(decoder, backbone, train_ds, val_ds, query_class,
                     shot, device, rng, n_val=100):
    """Validation: 固定 100 episodes 评估 mIoU | Fixed 100 episodes for mIoU."""

    train_candidates = train_ds.class_to_images(query_class)
    val_candidates = val_ds.class_to_images(query_class)
    if len(train_candidates) < shot or not val_candidates:
        return 0.0

    ious = []
    for _ in range(n_val):
        support_idxs = rng.choice(train_candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_feats = backbone(support_imgs)
        support_p4s = [support_feats['p4'][i] for i in range(len(support_idxs))]
        query_p4 = backbone(query_img)['p4']

        fg_proto = decoder.compute_fg_prototype(support_p4s, support_masks)
        if fg_proto.sum() == 0:
            continue

        logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape))
        pred = (logit.squeeze().cpu() > 0).numpy()
        gt = query_mask.cpu().numpy() > 0
        ious.append(binary_iou(torch.from_numpy(pred), torch.from_numpy(gt)))

    return float(np.mean(ious)) if ious else 0.0


# ═══════════════════════════════════════════════════════════════════
# Full evaluation (post-training)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_trained(decoder, backbone, train_ds, val_ds, device,
                     shot, n_episodes, target_classes, logger, tag):
    """
    在训练后评估 | Evaluate after training.

    与 validate_episode 类似但输出完整 per-class 分解。
    Similar to validate but with full per-class breakdown.
    """
    class_to_images = {c: train_ds.class_to_images(c) for c in target_classes}
    rng = np.random.RandomState(42)
    all_ious = []
    per_cls_ious = defaultdict(list)
    t0 = time.perf_counter()
    log_every = max(10, n_episodes // 10)

    for ep in tqdm(range(n_episodes), desc=f'  {shot}-shot eval'):
        valid_classes = [c for c in target_classes if len(class_to_images[c]) >= shot]
        if not valid_classes:
            continue
        query_class = int(rng.choice(valid_classes))

        candidates = class_to_images[query_class]
        val_candidates = val_ds.class_to_images(query_class)
        if not val_candidates:
            continue

        support_idxs = rng.choice(candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        support_imgs = torch.stack([train_ds.load_image(si) for si in support_idxs]).to(device)
        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        support_p4s = [backbone(support_imgs)['p4'][i] for i in range(len(support_idxs))]
        query_p4 = backbone(query_img)['p4']

        fg_proto = decoder.compute_fg_prototype(support_p4s, support_masks)
        if fg_proto.sum() == 0:
            continue

        logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape))
        pred = (logit.squeeze().cpu() > 0).numpy()
        gt = query_mask.cpu().numpy() > 0
        iou = binary_iou(torch.from_numpy(pred), torch.from_numpy(gt))
        all_ious.append(iou)
        per_cls_ious[query_class].append(iou)

        if (ep + 1) % log_every == 0 and all_ious:
            running = float(np.mean(all_ious[-log_every:]))
            total = float(np.mean(all_ious))
            logger.log_info(f'{tag}/progress',
                           f'  Ep {ep+1}/{n_episodes}: running={running:.4f} total={total:.4f}')

    dt = time.perf_counter() - t0
    per_cls_avg = {}
    for c in sorted(per_cls_ious.keys()):
        avg = float(np.mean(per_cls_ious[c]))
        per_cls_avg[str(c)] = avg
        logger.log_info(f'{tag}/per_cls',
                       f'  {target_classes[c]:<20} IoU={avg:.4f} ({len(per_cls_ious[c])} eps)')

    result = {
        'miou_mean': float(np.mean(all_ious)) if all_ious else 0.0,
        'miou_std': float(np.std(all_ious)) if all_ious else 0.0,
        'n_valid': len(all_ious),
        'per_class_iou': per_cls_avg,
    }
    logger.log_info(f'{tag}/done',
                   f'  {shot}-shot TRAINED: mIoU={result["miou_mean"]:.4f}±{result["miou_std"]:.4f} '
                   f'({len(all_ious)} eps, {dt:.0f}s)')
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src-root', type=str, default='data/iSAID_processed')
    p.add_argument('--shots', type=str, default='1,3,5')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--episodes-per-epoch', type=int, default=50)
    p.add_argument('--eval-episodes', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', type=str, default='runs/c02b_decoder')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tile', action='store_true',
                   help='Tile 模式: 切分全图为 896×896 tiles (stride=512)')
    p.add_argument('--tile-size', type=int, default=896)
    p.add_argument('--tile-stride', type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    shots = [int(x.strip()) for x in args.shots.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger('c02b')
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / 'c02b.jsonl')))

    # ── 加载数据 | Load data ──
    train_ds = ISAIDInstanceDataset(args.src_root, split='train')
    val_ds = ISAIDInstanceDataset(args.src_root, split='val')

    if args.tile:
        train_ds = ISAIDTileWrapper(train_ds, tile_size=args.tile_size,
                                     stride=args.tile_stride)
        val_ds = ISAIDTileWrapper(val_ds, tile_size=args.tile_size,
                                   stride=args.tile_stride)
        logger.log_info('c02b/data',
                       f'iSAID Tile: {len(train_ds)} train tiles, {len(val_ds)} val tiles '
                       f'({args.tile_size}×{args.tile_size}, stride={args.tile_stride})')
    else:
        logger.log_info('c02b/data',
                       f'iSAID: {len(train_ds)} train, {len(val_ds)} val images')
    logger.log_info('c02b/data',
                   f'Target: {[(c, TARGET_CLASSES[c]) for c in sorted(TARGET_CLASSES)]}')

    # ── Backbone (frozen, shared across all shots) ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval().to(device)
    logger.log_info('c02b/model', 'FastSAM backbone (frozen)')

    all_results = {}
    rng = np.random.RandomState(args.seed)

    for shot in shots:
        tag = f'c02b/{shot}shot'
        logger.log_info(f'{tag}/start',
                       f'\n{"─"*55}\n  C-02B Decoder Training: {shot}-Shot\n{"─"*55}')

        # ── 训练 | Training ──
        decoder = ProtoRefineDecoder().to(device)
        decoder.train()
        opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-6)

        best_val_iou = 0.0
        best_state = None
        metrics_path = output_dir / f'decoder_{shot}shot_metrics.jsonl'

        # 为每个类别单独训练 eval 记录 | Per-class training eval
        class_val_history = {c: [] for c in TARGET_CLASSES}

        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            decoder.train()
            total_loss, n = 0.0, 0

            for _ in tqdm(range(args.episodes_per_epoch),
                         desc=f'  E{epoch}/{args.epochs}', leave=False):
                # 采样 episode: 随机选类 → K support → 1 query
                query_class = int(rng.choice(list(TARGET_CLASSES.keys())))
                candidates = train_ds.class_to_images(query_class)
                val_candidates = val_ds.class_to_images(query_class)
                if len(candidates) < shot or not val_candidates:
                    continue

                support_idxs = rng.choice(candidates, shot, replace=False)
                qi = int(rng.choice(val_candidates))

                loss = train_episode(decoder, backbone, support_idxs, qi,
                                    train_ds, val_ds, query_class, device, opt)
                if loss is not None:
                    total_loss += loss
                    n += 1

            sch.step()
            avg_loss = total_loss / max(n, 1)

            # 验证 | Validation (per-class)
            decoder.eval()
            per_cls_val = {}
            for cls_id in TARGET_CLASSES:
                val_iou = validate_episode(
                    decoder, backbone, train_ds, val_ds,
                    cls_id, shot, device, rng, n_val=30)
                per_cls_val[cls_id] = val_iou
                class_val_history[cls_id].append(val_iou)

            mval = float(np.mean(list(per_cls_val.values())))
            logger.log_info(f'{tag}/train',
                           f'E{epoch:2d}/{args.epochs} loss={avg_loss:.4f} '
                           f'val_mIoU={mval:.4f} ('
                           + ', '.join(f'{TARGET_CLASSES[c][:8]}={per_cls_val[c]:.4f}'
                                      for c in sorted(TARGET_CLASSES)) + ')')

            epoch_metrics = {
                'epoch': epoch, 'loss': round(avg_loss, 6),
                'val_miou': round(mval, 6),
                'per_cls': {str(k): round(v, 6) for k, v in per_cls_val.items()},
            }
            with open(metrics_path, 'a') as mf:
                mf.write(json.dumps(epoch_metrics) + '\n')
                mf.flush()

            if mval > best_val_iou:
                best_val_iou = mval
                best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
                torch.save(best_state, str(output_dir / f'decoder_{shot}shot_best.pt'))

        # 恢复最佳模型 | Restore best model
        if best_state:
            decoder.load_state_dict(best_state)
        dt_train = time.perf_counter() - t0
        logger.log_info(f'{tag}/best',
                       f'Best val mIoU={best_val_iou:.4f} ({dt_train:.0f}s training)')

        # ── 评估 | Evaluation (200 episodes full eval) ──
        logger.log_info(f'{tag}/eval', 'Evaluating trained decoder...')
        result = evaluate_trained(decoder, backbone, train_ds, val_ds, device,
                                  shot, args.eval_episodes, TARGET_CLASSES,
                                  logger, f'{tag}/eval')
        all_results[f'{shot}shot'] = result

        # 对比 C-02A baseline | Compare with C-02A
        logger.log_info(f'{tag}/compare',
                       f'C-02A (non-param): loading from runs/c02a_proto/...')
        c02a_path = Path('runs/c02a_proto/c02a_results.json')
        if c02a_path.exists():
            c02a = json.loads(c02a_path.read_text())
            c02a_miou = c02a['results'][f'{shot}shot']['miou_mean']
            delta = result['miou_mean'] - c02a_miou
            logger.log_info(f'{tag}/compare',
                           f'C-02A={c02a_miou:.4f} → C-02B={result["miou_mean"]:.4f} '
                           f'(Δ={delta:+.4f}, {delta/max(c02a_miou,1e-8)*100:+.0f}% relative)')

    # ── Summary | 汇总 ──
    logger.log_info('c02b/summary', '\n' + '=' * 75)
    logger.log_info('c02b/summary',
                   '  C-02B: FastSAM + BinaryFewShotDecoder — 3-Class Few-Shot')
    logger.log_info('c02b/summary', '=' * 75)
    logger.log_info('c02b/summary',
                   f'  {"Method":<12} {"Shot":<8} {"mIoU":>10} {"±std":>8}  '
                   f'{"ship":>8} {"smallV":>8} {"s.tank":>8}')

    # 也加载 C-02A 做对比
    c02a_results = {}
    c02a_path = Path('runs/c02a_proto/c02a_results.json')
    if c02a_path.exists():
        c02a_data = json.loads(c02a_path.read_text())
        c02a_results = c02a_data['results']

    for shot in shots:
        r = all_results[f'{shot}shot']
        # C-02B
        line_b = (f'  {"C-02B Dec":<12} {shot:<8} '
                 f'{r["miou_mean"]*100:>9.2f}% {r["miou_std"]*100:>7.2f}%')
        for c in sorted(TARGET_CLASSES):
            pc = r['per_class_iou'].get(str(c), 0.0)
            line_b += f' {pc*100:>7.2f}%'
        logger.log_info('c02b/summary', line_b)

        # C-02A (baseline)
        if f'{shot}shot' in c02a_results:
            ra = c02a_results[f'{shot}shot']
            line_a = (f'  {"C-02A Proto":<12} {shot:<8} '
                     f'{ra["miou_mean"]*100:>9.2f}% {ra["miou_std"]*100:>7.2f}%')
            for c in sorted(TARGET_CLASSES):
                pc = ra['per_class_iou'].get(str(c), 0.0)
                line_a += f' {pc*100:>7.2f}%'
            logger.log_info('c02b/summary', line_a)

    # 保存 | Save
    summary = {
        'experiment': 'C-02B FastSAM + BinaryFewShotDecoder',
        'dataset': 'iSAID',
        'target_classes': {str(k): v for k, v in TARGET_CLASSES.items()},
        'timestamp': datetime.datetime.now().isoformat(),
        'shots': shots,
        'epochs': args.epochs,
        'lr': args.lr,
        'decoder_params': sum(p.numel() for p in ProtoRefineDecoder().parameters()),
        'results': {k: {'miou_mean': v['miou_mean'], 'miou_std': v['miou_std'],
                       'n_valid': v['n_valid'], 'per_class_iou': v['per_class_iou']}
                   for k, v in all_results.items()},
    }
    with open(output_dir / 'c02b_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.log_info('done', f'Results saved to {output_dir}/')


if __name__ == '__main__':
    main()
