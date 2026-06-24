#!/usr/bin/env python3
"""
C-02A: FastSAM + Prototype Matching — 验证 Few-Shot 可迁移性
==============================================================
FastSAM + Prototype Matching — Verify Few-Shot Transferability.

回答最基础的问题 | Answer the foundational question:
    冻结的 FastSAM P4 Feature 能否支持遥感实例分割的 Few-Shot Prototype Matching？
    Can frozen FastSAM P4 features support few-shot prototype matching for RS instance segmentation?

实验设计 | Design:
    - 冻结 FastSAM → P4 特征 | Frozen FastSAM → P4 features
    - Support: K-shot images → per-class prototype (masked mean P4)
    - Query: P4 → cosine similarity → upsample → threshold → binary mask
    - 数据集: iSAID, 3 类 (ship, small_vehicle, storage_tank)
    - K = 1, 3, 5 shot, 200 episodes/shot

直接复用 B-09 的 FewShotPrototypeMatcher + evaluate_fewshot。
Reuses B-09's FewShotPrototypeMatcher + evaluate_fewshot.

用法 | Usage::
    python tools/instance/eval_c02a_fastsam_fewshot.py \
        --src-root data/iSAID_processed --output-dir runs/c02a_proto
"""

import sys, argparse, time, json, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
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

# ── 复用 B-09 的核心类 | Reuse B-09 core classes ──
from tools.paper_b.eval_b09_nwpu_fewshot import (
    FewShotPrototypeMatcher, run_episode, evaluate_fewshot,
)

# ── 测试的 3 个类别: 难 / 中 / 易 | 3 target classes: hard / medium / easy ──
TARGET_CLASSES = {
    5: "ship",            # 难 | hard — 细长目标，多样朝向
    1: "small_vehicle",   # 中 | medium — 密集小目标
    4: "storage_tank",    # 易 | easy — 规则圆形，特征一致
}


# ═══════════════════════════════════════════════════════════════════
# ISAID Instance Dataset — COCO JSON → few-shot episode 接口
# ═══════════════════════════════════════════════════════════════════

def _render_gt_mask(ann, H, W):
    """
    从 COCO annotation 渲染单个实例的 binary mask。
    Render single GT instance binary mask from COCO annotation.

    支持 Polygon 和 RLE 两种分割格式。
    Supports both Polygon and RLE segmentation formats.

    :param ann: COCO annotation dict with "segmentation" key
    :param H: mask height in pixels
    :param W: mask width in pixels
    :return: boolean mask array of shape (H, W)
    """
    seg = ann.get("segmentation", [])

    # RLE 格式 | RLE format
    if isinstance(seg, dict):
        try:
            from pycocotools import mask as coco_mask
            rle = coco_mask.frPyObjects(seg, H, W)
            if isinstance(rle, list):
                rle = coco_mask.merge(rle)
            return coco_mask.decode(rle).astype(bool)
        except ImportError:
            pass

    mask = np.zeros((H, W), dtype=np.uint8)

    # Polygon 格式 | Polygon format
    if seg and not isinstance(seg, dict):
        polys = seg if isinstance(seg[0], list) else [seg]
        batch_polys = []
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            np.clip(pts[:, :, 0], 0, W - 1, out=pts[:, :, 0])
            np.clip(pts[:, :, 1], 0, H - 1, out=pts[:, :, 1])
            if len(pts) >= 3:
                batch_polys.append(pts)
        if batch_polys:
            cv2.fillPoly(mask, batch_polys, 1)
    else:
        # Bbox 回退 | Bbox fallback
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x, y, bw, bh = bbox
        x1, y1 = int(max(0, x)), int(max(0, y))
        x2, y2 = int(min(W, x + bw)), int(min(H, y + bh))
        mask[y1:y2, x1:x2] = 1

    return mask.astype(bool)


class ISAIDInstanceDataset:
    """
    iSAID 实例分割数据集 — Few-Shot 接口 (懒渲染版)。
    iSAID instance segmentation dataset — few-shot interface (lazy rendering).

    关键设计 | Key Design:
        - __getitem__ 只返回图像 tensor + 原始标注，不预渲染 mask
          __getitem__ returns only image tensor + raw annotations, NO pre-rendered masks.
        - render_class_mask(idx, class_id) 按需渲染单个类别的 union mask
          render_class_mask(idx, class_id) renders union mask for one class on demand.
        - 避免单张图 400+ 实例 × 4MB/mask = 1.8GB 的 OOM
          Avoids OOM from 400+ instances × 4MB/mask = 1.8GB per image.

    Interface:
        __len__() → int
        load_image(idx) → tensor [3, 1024, 1024]
        render_class_mask(idx, class_id) → tensor [1024, 1024]  0/1 binary
        class_to_images(class_id) → list[int]
    """

    def __init__(self, src_root: str, split: str, imgsz: int = 1024):
        self.src_root = Path(src_root)
        self.split = split
        self.imgsz = imgsz

        # 加载 COCO JSON | Load COCO JSON
        ann_file = self.src_root / split / "annotations" / f"instances_{split}.json"
        with open(ann_file) as f:
            coco = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        self._img_infos = []           # list of COCO image dicts
        self._img_anns = {}            # image_id → list of raw annotations
        self._class_to_img_idx = defaultdict(set)  # category_id → set of image indices

        # 过滤：只保留有有效标注的图像 | Filter: only images with valid annotations
        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            if cat_id in cat_id_to_name:
                self._img_anns.setdefault(ann["image_id"], []).append(ann)

        for img in coco["images"]:
            if img["id"] in self._img_anns:
                idx = len(self._img_infos)
                self._img_infos.append(img)
                for ann in self._img_anns[img["id"]]:
                    self._class_to_img_idx[ann["category_id"]].add(idx)

        # 转为 list 用于抽样 | Convert to list for sampling
        self._class_to_img_idx = {k: list(v) for k, v in self._class_to_img_idx.items()}

        # 预计算每个图像的 resize 参数 (避免重复计算) | Precompute resize params for each image
        self._resize_params = {}  # idx → (h_new, w_new, scale)

        # 简单图像缓存：避免同一 episode 内重复从磁盘加载
        # Simple image cache: avoid re-reading from disk within same episode
        self._img_cache = {}  # idx → tensor
        self._cache_max_size = 16

        # 日志 | Log
        print(f"[ISAIDInstanceDataset] {split}: {len(self._img_infos)} images, "
              f"{sum(len(v) for v in self._class_to_img_idx.values())} class-image pairs")
        for cid in TARGET_CLASSES:
            n = len(self._class_to_img_idx.get(cid, []))
            name = TARGET_CLASSES[cid]
            print(f"  Class {cid} ({name}): {n} images")

    def class_to_images(self, class_id: int) -> list:
        """返回包含指定类别的图像索引列表 | Return indices of images containing class_id."""
        return self._class_to_img_idx.get(class_id, [])

    def __len__(self) -> int:
        return len(self._img_infos)

    # ── 图像加载 (懒加载 + 小缓存) | Image loading (lazy + small cache) ──

    def load_image(self, idx: int) -> torch.Tensor:
        """
        加载并预处理单张图像 → [3, 1024, 1024] tensor, 值域 [0,1]。
        Load and preprocess a single image → [3, 1024, 1024] tensor in [0,1].

        带简单 LRU 缓存：最近 16 张图不重复从磁盘读取。
        With simple LRU cache: last 16 images avoid re-reading from disk.
        """
        if idx in self._img_cache:
            return self._img_cache[idx]

        img_info = self._img_infos[idx]
        img_path = str(self.src_root / self.split / "images" / img_info["file_name"])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize → letterbox → 1024×1024
        h_orig, w_orig = img.shape[:2]
        scale = self.imgsz / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        img_padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        img_padded[:h_new, :w_new] = img_resized

        img_t = torch.from_numpy(img_padded.astype(np.float32) / 255.0).permute(2, 0, 1)

        # 缓存管理 | Cache management
        if len(self._img_cache) >= self._cache_max_size:
            # 删除最早的条目 | Remove oldest entry
            oldest = next(iter(self._img_cache))
            del self._img_cache[oldest]
        self._img_cache[idx] = img_t
        self._resize_params[idx] = (h_new, w_new, scale)

        return img_t

    # ── Mask 懒渲染 | Lazy Mask Rendering ──

    def render_class_mask(self, idx: int, class_id: int) -> torch.Tensor:
        """
        按需渲染单个类别的 union mask → [1024, 1024] binary tensor。
        Render union mask for a single class on demand → [1024, 1024] binary.

        只渲染目标类别的实例 mask 并取并集，不碰其他类别。
        Only renders instances of the target class and unions them.

        :param idx: 图像索引 | Image index
        :param class_id: 目标类别 ID | Target category ID
        :return: [1024, 1024] float tensor, 值为 0.0 或 1.0
        """
        img_info = self._img_infos[idx]

        # 获取/计算 resize 参数 | Get/compute resize params
        if idx not in self._resize_params:
            # 从图像文件读取原始尺寸 | Read original size from image file
            img_path = str(self.src_root / self.split / "images" / img_info["file_name"])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {img_path}")
            h_orig, w_orig = img.shape[:2]
            scale = self.imgsz / max(h_orig, w_orig)
            h_new, w_new = int(h_orig * scale), int(w_orig * scale)
            self._resize_params[idx] = (h_new, w_new, scale)
        h_new, w_new, _ = self._resize_params[idx]

        anns = self._img_anns.get(img_info["id"], [])

        # 渲染目标类所有实例的并集 | Render union of all instances of target class
        mask_union = np.zeros((h_new, w_new), dtype=np.uint8)
        for ann in anns:
            if ann["category_id"] != class_id:
                continue
            inst_mask = _render_gt_mask(ann, h_new, w_new)
            if inst_mask.sum() > 0:
                mask_union = np.maximum(mask_union, inst_mask.astype(np.uint8))

        # Pad 到 1024×1024
        mask_padded = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
        mask_padded[:h_new, :w_new] = mask_union

        return torch.from_numpy(mask_padded).float()


# ═══════════════════════════════════════════════════════════════════
# 针对 3 类的定制评估 | Custom evaluation for 3 target classes
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_3class(model, backbone, train_ds, val_ds, device,
                    shot, n_episodes, target_classes, logger, tag):
    """
    在 3 个目标类别上运行 episode 评估。
    Run episodic evaluation on 3 target classes only.

    在 B-09 evaluate_fewshot 基础上精简：固定类别池，per-class 输出。
    Simplified from B-09: fixed class pool, per-class output.
    """
    # 建立类别索引 | Build class indices
    class_to_images = {}
    for c in target_classes:
        imgs = train_ds.class_to_images(c)
        class_to_images[c] = imgs
        logger.log_info(f'{tag}/classes',
                       f'  Class {c} ({target_classes[c]}): {len(imgs)} train images')

    rng = np.random.RandomState(42)
    all_ious = []
    per_cls_ious = defaultdict(list)
    t0 = time.perf_counter()
    log_every = max(10, n_episodes // 10)

    for ep in tqdm(range(n_episodes), desc=f'  {shot}-shot'):
        # 从 3 类中随机选 | Randomly pick from 3 target classes
        valid_classes = [c for c in target_classes if len(class_to_images[c]) >= shot]
        if not valid_classes:
            continue
        query_class = int(rng.choice(valid_classes))

        # K support | K support images
        candidates = class_to_images[query_class]
        support_idxs = rng.choice(candidates, shot, replace=False)

        # 构建 support samples: 懒渲染目标类 mask → 移到 GPU
        # Build support: lazy-render target class mask → move to GPU
        support_samples = []
        for si in support_idxs:
            img = train_ds.load_image(si).to(device)
            mask = train_ds.render_class_mask(si, query_class).to(device)
            support_samples.append((img, mask))

        # Query (val 集中含目标类的图像) | Query image containing target class
        val_candidates = val_ds.class_to_images(query_class)
        if not val_candidates:
            continue
        qi = int(rng.choice(val_candidates))

        query_img = val_ds.load_image(qi).to(device)
        query_mask = val_ds.render_class_mask(qi, query_class).to(device)
        query_sample = (query_img, query_mask)

        try:
            iou = run_episode(support_samples, query_sample, model, backbone,
                             query_class, device)
            all_ious.append(iou)
            per_cls_ious[query_class].append(iou)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.log_warn(f'{tag}/oom', f'CUDA OOM at ep {ep+1}, skipping')
                torch.cuda.empty_cache()
            else:
                raise

        if (ep + 1) % log_every == 0 and all_ious:
            running = float(np.mean(all_ious[-log_every:]))
            total = float(np.mean(all_ious))
            logger.log_info(f'{tag}/progress',
                           f'  Ep {ep+1}/{n_episodes}: running={running:.4f} total={total:.4f}')

    dt = time.perf_counter() - t0
    n_valid = len(all_ious)

    # Per-class | 逐类
    per_cls_avg = {}
    for c in sorted(per_cls_ious.keys()):
        avg = float(np.mean(per_cls_ious[c]))
        per_cls_avg[str(c)] = avg
        logger.log_info(f'{tag}/per_cls',
                       f'  {target_classes[c]:<20} IoU={avg:.4f} ({len(per_cls_ious[c])} eps)')

    result = {
        'miou_mean': float(np.mean(all_ious)) if all_ious else 0.0,
        'miou_std': float(np.std(all_ious)) if all_ious else 0.0,
        'n_valid': n_valid,
        'per_class_iou': per_cls_avg,
    }
    logger.log_info(f'{tag}/done',
                   f'  {shot}-shot: mIoU={result["miou_mean"]:.4f}±{result["miou_std"]:.4f} '
                   f'({n_valid} eps, {dt:.0f}s, {dt/max(n_valid,1):.2f}s/ep)')
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src-root', type=str, default='data/iSAID_processed')
    p.add_argument('--shots', type=str, default='1,3,5')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--output-dir', type=str, default='runs/c02a_proto')
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

    logger = get_logger('c02a')
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / 'c02a.jsonl')))

    # ── 加载数据 | Load data ──
    train_ds = ISAIDInstanceDataset(args.src_root, split='train')
    val_ds = ISAIDInstanceDataset(args.src_root, split='val')

    logger.log_info('c02a/data',
                   f'iSAID: {len(train_ds)} train, {len(val_ds)} val images')
    logger.log_info('c02a/data',
                   f'Target classes: {[(c, TARGET_CLASSES[c]) for c in sorted(TARGET_CLASSES)]}')

    # ── 加载 Backbone | Load backbone ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    logger.log_info('c02a/model', 'FastSAM backbone (frozen) — 0 trainable params')
    logger.log_info('c02a/method', 'Proto Baseline: masked mean P4 → cosine sim → threshold')

    # ── 非参数 Prototype Matcher (复用 B-09) ──
    model = FewShotPrototypeMatcher().to(device)
    model.eval()

    # ── 运行 | Run ──
    all_results = {}
    for shot in shots:
        tag = f'c02a/{shot}shot'
        logger.log_info(f'{tag}/start',
                       f'\n{"─"*50}\n  C-02A Proto Baseline: {shot}-Shot\n{"─"*50}')
        t0 = time.perf_counter()
        result = evaluate_3class(model, backbone, train_ds, val_ds, device,
                                 shot, args.episodes, TARGET_CLASSES, logger, tag)
        dt = time.perf_counter() - t0
        all_results[f'{shot}shot'] = result
        logger.log_info(f'{tag}/result',
                       f'  {shot}-shot: mIoU={result["miou_mean"]:.4f}±{result["miou_std"]:.4f} '
                       f'({result["n_valid"]} eps, {dt:.0f}s)')

    # ── Summary | 汇总 ──
    logger.log_info('c02a/summary', '\n' + '=' * 70)
    logger.log_info('c02a/summary', '  C-02A: FastSAM + Prototype Matching — 3-Class Pilot')
    logger.log_info('c02a/summary', '=' * 70)

    # Header
    cls_names = [TARGET_CLASSES[int(c)] for c in sorted(TARGET_CLASSES)]
    header = f"  {'Shot':<8} {'mIoU':>10} {'±std':>8}"
    for name in cls_names:
        header += f' {name:>16}'
    logger.log_info('c02a/summary', header)
    logger.log_info('c02a/summary', f"  {'─'*8} {'─'*10} {'─'*8}" + f" {'─'*16}" * len(cls_names))

    for shot in shots:
        r = all_results[f'{shot}shot']
        line = f"  {shot:<8} {r['miou_mean']*100:>9.2f}% {r['miou_std']*100:>7.2f}%"
        for c in sorted(TARGET_CLASSES):
            pc = r['per_class_iou'].get(str(c), 0.0)
            line += f' {pc*100:>15.2f}%'
        logger.log_info('c02a/summary', line)

    # SES | 样本效率评分
    if 1 in shots and 5 in shots:
        miou_1 = all_results['1shot']['miou_mean']
        miou_5 = all_results['5shot']['miou_mean']
        if miou_5 > 0:
            ses = miou_1 / miou_5
            logger.log_info('c02a/ses',
                           f'  SES(1-shot/5-shot) = {ses:.3f} → 1-shot retains {ses*100:.0f}% of 5-shot mIoU')

    # 保存 | Save
    summary = {
        'experiment': 'C-02A FastSAM + Prototype Matching',
        'dataset': 'iSAID',
        'target_classes': {str(k): v for k, v in TARGET_CLASSES.items()},
        'timestamp': datetime.datetime.now().isoformat(),
        'shots': shots,
        'episodes_per_shot': args.episodes,
        'results': {k: {'miou_mean': v['miou_mean'], 'miou_std': v['miou_std'],
                       'n_valid': v['n_valid'], 'per_class_iou': v['per_class_iou']}
                   for k, v in all_results.items()},
    }
    with open(output_dir / 'c02a_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.log_info('done', f'Results saved to {output_dir}/')


if __name__ == '__main__':
    main()
