#!/usr/bin/env python3
"""
C-03: CAT-SAM Lite — Cross-Attention Conditioned Decoder for Few-Shot Instance Segmentation
===========================================================================================================
CAT-SAM Lite: FastSAM + Cross-Attention Adapter + InstanceDecoder.

核心思想 | Core Idea (from CAT-SAM):
    C-02A 死了不是因为 P4 没信息，而是 Support 对 Query 没有"条件化"。
    C-02A died not because P4 lacks info, but because Support never conditions Query.

    CAT-SAM: Support → proto → MLP → proto_token → CrossAttn(Q=query_feat, K=proto, V=proto) → mask.
    冻结 FastSAM，只训练 ProtoMLP + CrossAttn + InstanceDecoder ≈ 1.1M params.
    Frozen FastSAM, train only ProtoMLP + CrossAttn + InstanceDecoder ≈ 1.1M params.

对比 C-02A / C-02B | vs C-02A / C-02B:
    C-02A: proto → cosine_sim → threshold          (0 params, non-parametric)
    C-02B: proto → cosine_sim → Refine CNN          (~10K params, no conditioning)
    C-03:  proto → CrossAttn → InstanceDecoder      (~1.1M params, CAT-SAM lite)

架构 | Architecture:
    Support Image(s)                    Query Image
         │                                  │
    Frozen FastSAM P4                 Frozen FastSAM P4
         │                                  │
    masked_mean(fg) → proto [1280]    Proj: 1×1 Conv → [B,256,H/16,W/16]
         │                                  │
    ProtoMLP: 1280→256→256                  Q: reshape → [B,N,256]
         │                                  │
    proto_token [256] ──→ K, V ──→ CrossAttention(Q,K,V) ──→ [B,256,H/16,W/16]
                                       │ (+) residual
                                       │
                                  Up1: 256→128 + ×4
                                  Up2: 128→64  + ×2
                                  Up3: 64→32   + ×2
                                  Head: 32→1
                                       │
                                  Binary Mask

Cross-Attention 细节 | CA Details:
    - Q = Query Feature [B, 256, H, W] → reshape [B, N, 256]
    - K = V = proto_token [256] → [B, 1, 256]
    - sigmoid(Q @ Kᵀ / √d) → per-position gating [B, N, 1]
    - 用 sigmoid 而非 softmax：单 token 下 softmax 恒为 1，sigmoid 保留空间差异
    - output = gate * V + residual(x)

参数 | Params: ~1.1M trainable (ProtoMLP 394K + CrossAttn 0 + Decoder 716K)

用法 | Usage::
    python tools/instance/eval_c03_catsam_fewshot.py \
        --src-root data/iSAID_processed --device cuda \
        --shots 1,3,5 --epochs 30
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

from tools.instance.eval_c02a_fastsam_fewshot import (
    ISAIDInstanceDataset, TARGET_CLASSES,
)

# ═══════════════════════════════════════════════════════════════════
# Prototype computation (shared)
# ═══════════════════════════════════════════════════════════════════

def compute_fg_prototype(p4_features: list, masks: list,
                          feat_dim: int = 1280) -> torch.Tensor:
    """masked_mean(P4) → L2-normalized prototype | 掩码平均 → L2 归一化原型."""
    p4_h, p4_w = p4_features[0].shape[1], p4_features[0].shape[2]
    all_feats = []
    for i in range(len(p4_features)):
        m = masks[i]
        if m.dim() == 3:
            m = m.squeeze(0)
        mask_4d = m.unsqueeze(0).unsqueeze(0).float()
        mask_p4 = F.interpolate(mask_4d, size=(p4_h, p4_w),
                                 mode='nearest').squeeze(0)
        if mask_p4.sum() > 0:
            weighted = (p4_features[i] * mask_p4).sum(dim=(1, 2)) / mask_p4.sum()
            all_feats.append(weighted)
    if not all_feats:
        return torch.zeros(feat_dim, device=p4_features[0].device)
    return F.normalize(torch.stack(all_feats).mean(dim=0), dim=0, p=2)


# ═══════════════════════════════════════════════════════════════════
# CAT-SAM Lite Decoder
# ═══════════════════════════════════════════════════════════════════

class CATFewShotDecoder(nn.Module):
    """
    CAT-SAM Lite: Cross-Attention Conditioned InstanceDecoder.

    Support → proto → ProtoMLP → proto_token [256]
    Query P4 → Proj → [B,256,H,W] → CrossAttn(Q=x, K=proto_token, V=proto_token)
    → residual(+) → InstanceDecoder → mask.

    ~1.1M trainable params. Frozen FastSAM backbone.
    Cross-Attention has 0 learned params (parameter-free operation).
    """

    def __init__(self, feat_dim: int = 1280, hidden_dim: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # ── Proto Projection | 原型投影 ──
        # proto [1280] → MLP → proto_token [256]
        # 将 L2-normalized prototype 投影到与 query feature 相同的 256 维空间
        self.proto_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),  # proto token
        )

        # ── Query Projection | 查询投影 ──
        # P4 [B, 1280, H/16, W/16] → 1×1 Conv → [B, 256, H/16, W/16]
        # 将 P4 特征投影到 Cross-Attention 的工作空间
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Instance Decoder | 实例解码器 ──
        # 与 InstanceDecoder 同架构：256→128→64→32→1
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(32, 1, 1, bias=True)

        n_proto = sum(p.numel() for p in self.proto_mlp.parameters())
        n_proj = sum(p.numel() for p in self.proj.parameters())
        n_dec = sum(p.numel() for p in [
            *self.up1.parameters(), *self.up2.parameters(),
            *self.up3.parameters(), *self.mask_head.parameters(),
        ])
        # Cross-Attention: 0 参数 (纯计算操作)
        print(f"[CATFewShotDecoder] Total: {n_proto+n_proj+n_dec:,} params "
              f"(ProtoMLP: {n_proto:,}, Proj: {n_proj:,}, "
              f"CrossAttn: 0, Decoder: {n_dec:,})")

    def _cross_attention(self, x: torch.Tensor,
                         proto_token: torch.Tensor) -> torch.Tensor:
        """
        单层 Cross-Attention: Q=query features, K=V=proto_token.

        :param x: [B, C, H, W] query features (C=hidden_dim=256)
        :param proto_token: [C] prototype token
        :return: [B, C, H, W] attended features

        细节 | Details:
            - Q: [B, C, H, W] → [B, N, C]  (N = H*W)
            - K, V: [C] → [B, 1, C]
            - gate = sigmoid(Q @ Kᵀ / √d)  [B, N, 1]
            - 用 sigmoid 而非 softmax：单 token 下 softmax 恒为 1，
              而 sigmoid 让每个空间位置独立决定注入多少 proto 信息
            - out = gate * V  [B, N, C] (broadcast)
            - out → reshape [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # Q: [B, C, H, W] → [B, N, C]
        Q = x.reshape(B, C, N).transpose(1, 2)  # [B, N, C]

        # K, V: [C] → [B, 1, C] (单 token)
        K = proto_token.view(1, 1, C).expand(B, 1, C)  # [B, 1, C]
        V = proto_token.view(1, 1, C).expand(B, 1, C)  # [B, 1, C]

        # Scaled dot-product attention → sigmoid gate
        # sigmoid 而非 softmax：单 key 时 softmax = 1，丧失空间差异性
        scale = C ** 0.5
        gate = torch.sigmoid(torch.bmm(Q, K.transpose(1, 2)) / scale)  # [B, N, 1]

        # Gate the proto token: 相关性越高，proto 信息注入越多
        # [B, N, 1] * [B, 1, C] → [B, N, C] (broadcast)
        out = gate * V  # [B, N, C]

        # Reshape back: [B, N, C] → [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

    def forward(self, query_p4: torch.Tensor, fg_prototype: torch.Tensor,
                target_size: tuple = None) -> torch.Tensor:
        """
        :param query_p4: [B, 1280, H/16, W/16] P4 features from frozen FastSAM
        :param fg_prototype: [1280] L2-normalized foreground prototype vector
        :param target_size: (H, W) to resize output to
        :return: [B, 1, H_target, W_target] logits
        """
        # Step 1: Query projection | 查询投影
        x = self.proj(query_p4)  # [B, 256, H/16, W/16]

        # Step 2: Cross-Attention conditioning with residual | 交叉注意力 + 残差
        if fg_prototype is not None and fg_prototype.sum() != 0:
            proto_token = self.proto_mlp(fg_prototype)  # [256]
            x = x + self._cross_attention(x, proto_token)  # residual: x + CA(x, proto)

        # Step 3: Instance Decoder | 解码器上采样
        x = self.up1(x)  # [B, 128, H/16, W/16]
        x = F.interpolate(x, scale_factor=4, mode='bilinear',
                         align_corners=False)  # [B, 128, H/4, W/4]
        x = self.up2(x)  # [B, 64, H/4, W/4]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                         align_corners=False)  # [B, 64, H/2, W/2]
        x = self.up3(x)  # [B, 32, H/2, W/2]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                         align_corners=False)  # [B, 32, H, W]
        x = self.mask_head(x)  # [B, 1, H, W]

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear',
                            align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════
# Binary IoU
# ═══════════════════════════════════════════════════════════════════

def binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# P4 Feature Cache (frozen backbone → precompute once)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_p4_cache(dataset, backbone, device, desc="P4 cache"):
    """预计算数据集中所有图像的 P4 特征，存入 CPU dict。
    Pre-compute P4 features for all images. Frozen backbone →每一张图只算一次。

    内存 | Memory: ~21 MB/img (1280×64×64×4 bytes). iSAID train 141 img ≈ 3 GB CPU RAM.
    """
    cache = {}
    for idx in tqdm(range(len(dataset)), desc=desc, leave=False):
        img = dataset.load_image(idx).unsqueeze(0).to(device)
        feat = backbone(img)['p4']  # [1, 1280, H/16, W/16]
        cache[idx] = feat.squeeze(0).cpu()  # [1280, H/16, W/16] on CPU
    return cache


# ═══════════════════════════════════════════════════════════════════
# Episodic Training
# ═══════════════════════════════════════════════════════════════════

def train_episode(decoder, support_idxs, query_idx,
                  train_ds, val_ds, query_class, device, opt,
                  p4_cache_train=None, p4_cache_val=None,
                  backbone=None):
    """Single training episode: support → proto → CrossAttn → decoder → loss.

    支持 P4 缓存: 如果传入 cache，跳过 backbone forward。
    Supports P4 cache: backbone is skipped when caches are provided.
    """

    shot = len(support_idxs)

    # ── Support P4 (cache first, then backbone fallback) ──
    if p4_cache_train is not None:
        support_p4s = [p4_cache_train[int(si)].unsqueeze(0).to(device)
                       for si in support_idxs]
    else:
        support_imgs = torch.stack([train_ds.load_image(int(si))
                                     for si in support_idxs]).to(device)
        support_feats = backbone(support_imgs)
        support_p4s = [support_feats['p4'][i] for i in range(shot)]

    # ── Query P4 (cache first, then backbone fallback) ──
    if p4_cache_val is not None:
        query_p4 = p4_cache_val[int(query_idx)].unsqueeze(0).to(device)
    else:
        query_img = val_ds.load_image(int(query_idx)).unsqueeze(0).to(device)
        query_p4 = backbone(query_img)['p4']

    # ── Masks (lazy render) ──
    support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                     for si in support_idxs]
    query_mask = val_ds.render_class_mask(query_idx, query_class)
    query_mask = query_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    # ── Prototype → CrossAttn → Decoder → Loss ──
    fg_proto = compute_fg_prototype(support_p4s, support_masks)
    if fg_proto.sum() == 0:
        return None

    logit = decoder(query_p4, fg_proto, target_size=tuple(query_mask.shape[2:]))

    # BCE + Dice loss (per spec: 标准二分类损失)
    bce_loss = F.binary_cross_entropy_with_logits(logit, query_mask)

    prob = torch.sigmoid(logit)
    inter = (prob * query_mask).sum()
    union = prob.sum() + query_mask.sum() + 1e-8
    dice_loss = 1.0 - (2 * inter / union)

    loss = 0.5 * bce_loss + 0.5 * dice_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def validate_episode(decoder, train_ds, val_ds, query_class,
                     shot, device, rng, n_val=30,
                     p4_cache_train=None, p4_cache_val=None,
                     backbone=None):
    """Validation: 固定 episodes 评估 mIoU. Supports P4 cache."""
    train_candidates = train_ds.class_to_images(query_class)
    val_candidates = val_ds.class_to_images(query_class)
    if len(train_candidates) < shot or not val_candidates:
        return 0.0

    ious = []
    for _ in range(n_val):
        support_idxs = rng.choice(train_candidates, shot, replace=False)
        qi = int(rng.choice(val_candidates))

        # Support P4
        if p4_cache_train is not None:
            support_p4s = [p4_cache_train[int(si)].unsqueeze(0).to(device)
                          for si in support_idxs]
        else:
            support_imgs = torch.stack([train_ds.load_image(int(si))
                                         for si in support_idxs]).to(device)
            support_p4s = [backbone(support_imgs)['p4'][i] for i in range(shot)]

        # Query P4
        if p4_cache_val is not None:
            query_p4 = p4_cache_val[qi].unsqueeze(0).to(device)
        else:
            query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
            query_p4 = backbone(query_img)['p4']

        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        fg_proto = compute_fg_prototype(support_p4s, support_masks)
        if fg_proto.sum() == 0:
            continue

        logit = decoder(query_p4, fg_proto,
                       target_size=tuple(query_mask.shape))
        pred = (logit.squeeze().cpu() > 0).numpy()
        gt = query_mask.cpu().numpy() > 0
        ious.append(binary_iou(torch.from_numpy(pred), torch.from_numpy(gt)))

    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate_trained(decoder, train_ds, val_ds, device,
                     shot, n_episodes, target_classes, logger, tag,
                     p4_cache_train=None, p4_cache_val=None,
                     backbone=None):
    """Full evaluation after training. Supports P4 cache."""
    class_to_images = {c: train_ds.class_to_images(c) for c in target_classes}
    rng = np.random.RandomState(42)
    all_ious, per_cls_ious = [], defaultdict(list)
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

        # Support P4
        if p4_cache_train is not None:
            support_p4s = [p4_cache_train[int(si)].unsqueeze(0).to(device)
                          for si in support_idxs]
        else:
            support_imgs = torch.stack([train_ds.load_image(int(si))
                                         for si in support_idxs]).to(device)
            support_p4s = [backbone(support_imgs)['p4'][i] for i in range(shot)]

        # Query P4
        if p4_cache_val is not None:
            query_p4 = p4_cache_val[qi].unsqueeze(0).to(device)
        else:
            query_img = val_ds.load_image(qi).unsqueeze(0).to(device)
            query_p4 = backbone(query_img)['p4']

        support_masks = [train_ds.render_class_mask(si, query_class).to(device)
                         for si in support_idxs]
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)

        fg_proto = compute_fg_prototype(support_p4s, support_masks)
        if fg_proto.sum() == 0:
            continue

        logit = decoder(query_p4, fg_proto,
                       target_size=tuple(query_mask.shape))
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
                   f'  {shot}-shot: mIoU={result["miou_mean"]:.4f}±{result["miou_std"]:.4f} '
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
    p.add_argument('--output-dir', type=str, default='runs/c03_catsam')
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

    logger = get_logger('c03')
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / 'c03.jsonl')))

    # ── Data ──
    train_ds = ISAIDInstanceDataset(args.src_root, split='train')
    val_ds = ISAIDInstanceDataset(args.src_root, split='val')
    logger.log_info('c03/data',
                   f'iSAID: {len(train_ds)} train, {len(val_ds)} val')
    logger.log_info('c03/data',
                   f'Target: {[(c, TARGET_CLASSES[c]) for c in sorted(TARGET_CLASSES)]}')

    # ── Backbone (frozen, shared) ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval().to(device)
    logger.log_info('c03/model', 'FastSAM backbone (frozen)')
    logger.log_info('c03/method', 'CAT-SAM Lite: Cross-Attention + InstanceDecoder')

    # ── P4 Precomputation Cache | P4 预计算缓存 ──
    # 冻结 backbone → 每张图的 P4 只算一次。训练/验证直接查表，不再重复 forward。
    # Frozen backbone → compute P4 once per image. ~3 GB CPU RAM for iSAID train set.
    t_cache = time.perf_counter()
    p4_train = precompute_p4_cache(train_ds, backbone, device, desc="P4 cache train")
    p4_val = precompute_p4_cache(val_ds, backbone, device, desc="P4 cache val")
    dt_cache = time.perf_counter() - t_cache
    logger.log_info('c03/cache',
                   f'P4 cache built: {len(p4_train)} train + {len(p4_val)} val '
                   f'({dt_cache:.0f}s, ~{sum(t.numel() for t in p4_train.values()) * 4 / 1e9:.1f} GB CPU)')

    # Backbone 移到 CPU 释放 GPU 显存 | Free GPU memory after caching
    backbone.cpu()
    torch.cuda.empty_cache()
    logger.log_info('c03/cache', 'Backbone offloaded to CPU, GPU memory freed')

    all_results = {}
    rng = np.random.RandomState(args.seed)

    for shot in shots:
        tag = f'c03/{shot}shot'
        logger.log_info(f'{tag}/start',
                       f'\n{"─"*55}\n  C-03 CAT-SAM Lite: {shot}-Shot\n{"─"*55}')

        # ── Training ──
        decoder = CATFewShotDecoder().to(device)
        decoder.train()
        opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-6)

        best_val_iou = 0.0
        best_state = None
        metrics_path = output_dir / f'catsam_{shot}shot_metrics.jsonl'

        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            decoder.train()
            total_loss, n = 0.0, 0

            for _ in tqdm(range(args.episodes_per_epoch),
                         desc=f'  E{epoch}/{args.epochs}', leave=False):
                query_class = int(rng.choice(list(TARGET_CLASSES.keys())))
                candidates = train_ds.class_to_images(query_class)
                val_candidates = val_ds.class_to_images(query_class)
                if len(candidates) < shot or not val_candidates:
                    continue

                support_idxs = rng.choice(candidates, shot, replace=False)
                qi = int(rng.choice(val_candidates))

                loss = train_episode(decoder, support_idxs, qi,
                                    train_ds, val_ds, query_class, device, opt,
                                    p4_cache_train=p4_train, p4_cache_val=p4_val)
                if loss is not None:
                    total_loss += loss
                    n += 1

            sch.step()
            avg_loss = total_loss / max(n, 1)

            # Validation
            decoder.eval()
            per_cls_val = {}
            for cls_id in TARGET_CLASSES:
                per_cls_val[cls_id] = validate_episode(
                    decoder, train_ds, val_ds,
                    cls_id, shot, device, rng, n_val=30,
                    p4_cache_train=p4_train, p4_cache_val=p4_val)

            mval = float(np.mean(list(per_cls_val.values())))
            logger.log_info(f'{tag}/train',
                           f'E{epoch:2d}/{args.epochs} loss={avg_loss:.4f} '
                           f'val_mIoU={mval:.4f} ('
                           + ', '.join(f'{TARGET_CLASSES[c][:8]}={per_cls_val[c]:.4f}'
                                      for c in sorted(TARGET_CLASSES)) + ')')

            epoch_metrics = {'epoch': epoch, 'loss': round(avg_loss, 6),
                           'val_miou': round(mval, 6),
                           'per_cls': {str(k): round(v, 6) for k, v in per_cls_val.items()}}
            with open(metrics_path, 'a') as mf:
                mf.write(json.dumps(epoch_metrics) + '\n'); mf.flush()

            if mval > best_val_iou:
                best_val_iou = mval
                best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
                torch.save(best_state, str(output_dir / f'catsam_{shot}shot_best.pt'))

        if best_state:
            decoder.load_state_dict(best_state)
        dt_train = time.perf_counter() - t0
        logger.log_info(f'{tag}/best',
                       f'Best val mIoU={best_val_iou:.4f} ({dt_train:.0f}s training)')

        # ── Evaluation ──
        logger.log_info(f'{tag}/eval', 'Evaluating trained decoder...')
        result = evaluate_trained(decoder, train_ds, val_ds, device,
                                  shot, args.eval_episodes, TARGET_CLASSES,
                                  logger, f'{tag}/eval',
                                  p4_cache_train=p4_train, p4_cache_val=p4_val)
        all_results[f'{shot}shot'] = result

        # Compare with baselines if available
        for baseline, path in [('C-02A Proto', 'runs/c02a_proto/c02a_results.json'),
                                ('C-02B Dec', 'runs/c02b_decoder/c02b_results.json')]:
            if Path(path).exists():
                bl = json.loads(Path(path).read_text())
                bl_miou = bl['results'][f'{shot}shot']['miou_mean']
                delta = result['miou_mean'] - bl_miou
                logger.log_info(f'{tag}/compare',
                               f'{baseline}={bl_miou:.4f} → C-03={result["miou_mean"]:.4f} '
                               f'(Δ={delta:+.4f})')

    # ── Summary ──
    logger.log_info('c03/summary', '\n' + '=' * 80)
    logger.log_info('c03/summary', '  C-03: CAT-SAM Lite — Cross-Attention + InstanceDecoder')
    logger.log_info('c03/summary', '=' * 80)
    logger.log_info('c03/summary',
                   f'  {"Method":<14} {"Shot":<8} {"mIoU":>10} {"±std":>8}  '
                   f'{"ship":>8} {"smallV":>8} {"s.tank":>8}')

    # Load baselines for comparison table
    baseline_results = {}
    for bl_name, bl_path in [('C-02A Proto', 'runs/c02a_proto/c02a_results.json'),
                              ('C-02B Dec', 'runs/c02b_decoder/c02b_results.json')]:
        if Path(bl_path).exists():
            baseline_results[bl_name] = json.loads(Path(bl_path).read_text())['results']

    for shot in shots:
        # C-03
        r = all_results[f'{shot}shot']
        line = f'  {"C-03 CAT-SAM":<14} {shot:<8} '
        line += f'{r["miou_mean"]*100:>9.2f}% {r["miou_std"]*100:>7.2f}%'
        for c in sorted(TARGET_CLASSES):
            line += f' {r["per_class_iou"].get(str(c), 0)*100:>7.2f}%'
        logger.log_info('c03/summary', line)

        # Baselines
        for bl_name in ['C-02B Dec', 'C-02A Proto']:
            if bl_name in baseline_results and f'{shot}shot' in baseline_results[bl_name]:
                ra = baseline_results[bl_name][f'{shot}shot']
                line = f'  {bl_name:<14} {shot:<8} '
                line += f'{ra["miou_mean"]*100:>9.2f}% {ra["miou_std"]*100:>7.2f}%'
                for c in sorted(TARGET_CLASSES):
                    line += f' {ra["per_class_iou"].get(str(c), 0)*100:>7.2f}%'
                logger.log_info('c03/summary', line)

    # Save
    summary = {
        'experiment': 'C-03 CAT-SAM Lite — Cross-Attention + InstanceDecoder',
        'dataset': 'iSAID',
        'target_classes': {str(k): v for k, v in TARGET_CLASSES.items()},
        'timestamp': datetime.datetime.now().isoformat(),
        'shots': shots, 'epochs': args.epochs, 'lr': args.lr,
        'decoder': 'CATFewShotDecoder',
        'decoder_params': sum(p.numel() for p in CATFewShotDecoder().parameters()),
        'results': {k: {'miou_mean': v['miou_mean'], 'miou_std': v['miou_std'],
                       'n_valid': v['n_valid'], 'per_class_iou': v['per_class_iou']}
                   for k, v in all_results.items()},
    }
    with open(output_dir / 'c03_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.log_info('done', f'Results saved to {output_dir}/')


if __name__ == '__main__':
    main()
