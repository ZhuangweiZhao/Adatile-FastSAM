#!/usr/bin/env python3
"""
B-05: FDR Input Resolution Ablation — 输入分辨率消融
=====================================================

FDR 用 MobileNetV3-Small (冻结) + DensityHead (75K 可训) 预测 tile 前景密度。
当前的 image_size=2048 可能是严重过量的——FDR 不需要看到语义细节，只需感知"哪儿有东西"。

核心假设 | Core hypothesis:
    FDR only requires low-resolution global context.
    降低输入分辨率 → 大幅减少 FDR 计算量 → tile ranking 精度几乎不降。

实验设计 | Design:
    对每个分辨率 [512, 768, 1024, 1280, 1536, 2048]:
        1. 训练 FDR (20 epoch, MSE loss vs GT fg_ratio)
        2. 评估 Spearman r (tile ranking quality)
        3. 计算 Top-K% FG Retention (前景保留率)
        4. 可选: 动态选择 + Decoder → mIoU

指标 | Metrics:
    - Spearman r: 预测 tile score 与 GT fg_ratio 的排序相关性
    - FG Retention @ K%: Top-K% tile 捕获的前景像素比例
    - (可选) Dynamic mIoU: 加载 decoder checkpoint 做端到端验证

输出 | Output:
    runs/b05_fdr_resolution/
    ├── b05_resolution.json       # 汇总: per-resolution Spearman r, FG retention
    ├── b05_resolution.png        # 3-panel figure
    ├── b05_resolution.jsonl      # Per-epoch 训练指标
    └── checkpoints/
        ├── fdr_0512.pt ...       # Per-resolution FDR 权重

用法 | Usage:
    # 快速版 (仅 Spearman r + FG retention, 无需 decoder)
    python tools/paper_b/eval_b05_fdr_resolution.py \
        --src-root data/iSAID_processed \
        --resolutions 512,768,1024,2048 --epochs 20

    # 完整版 (含 decoder 动态选择)
    python tools/paper_b/eval_b05_fdr_resolution.py \
        --src-root data/iSAID_processed \
        --decoder-ckpt runs/b04_v3/decoder_best.pt \
        --resolutions 512,768,1024,1536,2048 --epochs 20
"""

import sys, argparse, json, datetime, os, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.sparse.spatial_router import DensityHead

# ═══════════════════════════════════════════════════════════════════
# 全局常量 | Global Constants
# ═══════════════════════════════════════════════════════════════════

TILE_SIZE = 1024  # tile 尺寸 (用于计算 GT fg_ratio) | tile size for GT fg_ratio
MV3_STRIDE = 32   # MobileNetV3-Small 特征 stride | MV3-Small feature stride
DEFAULT_RESOLUTIONS = [512, 768, 1024, 1280, 1536, 2048]
K_VALUES = [10, 20, 30, 40, 50]  # FG retention K% 值 | K% values for FG retention


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="B-05: FDR Input Resolution Ablation | FDR 输入分辨率消融")
    p.add_argument("--src-root", type=str, default="data/iSAID_processed",
                   help="iSAID 处理后数据目录 | iSAID processed data directory")
    p.add_argument("--split", type=str, default="train",
                   help="数据集分割 | Dataset split")
    p.add_argument("--resolutions", type=str,
                   default="512,768,1024,1280,1536,2048",
                   help="测试的输入分辨率 (逗号分隔) | Input resolutions to test")
    p.add_argument("--train-images", type=int, default=200,
                   help="训练图片数 | Number of training images")
    p.add_argument("--eval-images", type=int, default=100,
                   help="评估图片数 | Number of evaluation images")
    p.add_argument("--epochs", type=int, default=20,
                   help="每个分辨率的训练轮数 | Training epochs per resolution")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率 | Learning rate")
    p.add_argument("--batch-size", type=int, default=8,
                   help="批次大小 | Batch size")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 工作线程数 | DataLoader worker count")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="预处理缓存目录 | Preprocess cache directory")
    p.add_argument("--decoder-ckpt", type=str, default=None,
                   help="Decoder 权重路径 (可选, 用于完整动态评估) | Decoder checkpoint path")
    p.add_argument("--output-dir", type=str, default="runs/b05_fdr_resolution",
                   help="输出目录 | Output directory")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 | Device")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 语义掩码渲染 | Semantic Mask Rendering
# ═══════════════════════════════════════════════════════════════════

def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """
    渲染语义掩码 [H,W] uint8 (0-15) | Render semantic mask.

    直接使用 ann["category_id"]——映射已在 prep_isaid.py 中完成。
    Directly uses ann["category_id"] — mapping done in prep_isaid.py.
    """
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0:
            continue

        seg = ann.get("segmentation", [])
        if not seg:
            # bbox fallback | bbox 回退
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            sem[y1:y2, x1:x2] = cat_id
            continue
        if isinstance(seg, dict):
            continue

        # 多边形填充 | Polygon fill
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
            cv2.fillPoly(sem, [pts], int(cat_id))

    return sem


# ═══════════════════════════════════════════════════════════════════
# 预处理缓存 (只做一次: GT tile scores 是分辨率无关的)
# Preprocess Cache (done once: GT tile scores are resolution-independent)
# ═══════════════════════════════════════════════════════════════════

def compute_gt_tile_scores(mask: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    计算全图 mask 上每个 tile 的 GT fg_ratio | Compute GT fg_ratio per tile on full mask.

    Args:
        mask: [H, W] uint8 语义掩码 | semantic mask
        tile_size: tile 尺寸 | tile size (default 1024)

    Returns:
        gt_scores: [n_ty, n_tx] float32, 每个 tile 的 fg_ratio
    """
    H, W = mask.shape
    n_ty = (H + tile_size - 1) // tile_size
    n_tx = (W + tile_size - 1) // tile_size
    scores = np.zeros((n_ty, n_tx), dtype=np.float32)

    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * tile_size, min(ty * tile_size + tile_size, H)
            x0, x1 = tx * tile_size, min(tx * tile_size + tile_size, W)
            tile_mask = mask[y0:y1, x0:x1]
            total_px = (y1 - y0) * (x1 - x0)
            fg_px = int((tile_mask > 0).sum())
            scores[ty, tx] = fg_px / max(total_px, 1)

    return scores


def build_cache(src_root: Path, split: str, image_items: list, cache_dir: Path):
    """
    构建预处理缓存：对每张图渲染 mask → 计算 GT tile scores。
    Build preprocess cache: render mask → compute GT tile scores per image.

    缓存内容 | Cache content (per-image .npz):
        - gt_scores: [n_ty, n_tx] float32 GT fg_ratio grid
        - n_ty, n_tx: tile grid dimensions
        - orig_h, orig_w: original image dimensions
    """
    import cv2
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    img_dir = src_root / split / "images"

    logger = get_logger("b05_cache")
    logger.log_info("b05/cache", f"Building cache for {len(image_items)} images... | 构建缓存")

    for img_id, img_path, anns in tqdm(image_items, desc="  Precompute GT"):
        cache_path = cache_dir / f"{img_id}.npz"
        if cache_path.exists():
            continue

        # 加载图像获取尺寸 | Load image to get dimensions
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # 渲染全分辨率语义掩码 → 计算 GT tile scores
        # Render full-resolution semantic mask → compute GT tile scores
        mask = render_semantic_mask(anns, H, W)
        gt_scores = compute_gt_tile_scores(mask, TILE_SIZE)

        np.savez_compressed(
            cache_path,
            gt_scores=gt_scores,
            n_ty=gt_scores.shape[0],
            n_tx=gt_scores.shape[1],
            orig_h=H,
            orig_w=W,
        )

    logger.log_info("b05/cache", f"Cache ready: {cache_dir} | 缓存就绪")


# ═══════════════════════════════════════════════════════════════════
# FDR 模型 | FDR Model
# ═══════════════════════════════════════════════════════════════════

class FDRModel(nn.Module):
    """
    FDR: MobileNetV3-Small (冻结) + DensityHead (可训练) | FDR: Frozen MV3 + Trainable DensityHead.

    输入任意分辨率 RGB 图像 → 输出重要性图 [1, 1, h, w]。
    Input arbitrary resolution RGB image → output importance map [1, 1, h, w].
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features
        # 冻结 backbone | Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # DensityHead: MV3 stride-32 输出 576 通道 | MV3 stride-32 outputs 576 channels
        self.density_head = DensityHead(576, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        Args:
            x: [B, 3, H, W] RGB 图像 (归一化到 [0,1]) | RGB image (normalized to [0,1])

        Returns:
            importance: [B, 1, H/32, W/32] 重要性图 | importance map
        """
        features = self.backbone(x)       # [B, 576, H/32, W/32]
        importance = self.density_head(features)  # [B, 1, H/32, W/32]
        return importance


# ═══════════════════════════════════════════════════════════════════
# 数据集 (加载图像 → resize → 返回 tensor) | Dataset
# ═══════════════════════════════════════════════════════════════════

class ResolutionDataset(Dataset):
    """
    分辨率可变数据集 | Variable-resolution dataset.

    每次 epoch 按指定的 target_size resize 图像。
    如果 cache_dir 存在，GT scores 从缓存加载。
    Images are resized to target_size each epoch.
    GT scores loaded from cache if available.
    """

    def __init__(self, image_items: list, cache_dir: Path,
                 target_size: int, img_dir: Path):
        """
        Args:
            image_items: [(img_id, img_path, annotations), ...]
            cache_dir: GT scores 缓存目录 | GT scores cache directory
            target_size: 目标长边尺寸 | Target long-edge size
            img_dir: 图像目录 (仅用于无缓存时 fallback) | Image directory
        """
        self.items = image_items
        self.cache_dir = cache_dir
        self.target_size = target_size
        self.img_dir = img_dir
        self.stride = MV3_STRIDE

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        img_id, img_path, anns = self.items[idx]

        # 加载图像 | Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # Resize 到目标分辨率 | Resize to target resolution
        scale = self.target_size / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        img_r = np.array(Image.fromarray(img).resize((nW, nH), Image.BILINEAR))

        # Pad 到 32 的倍数 (MV3 stride 要求) | Pad to multiple of 32
        ph, pw = (self.stride - nH % self.stride) % self.stride, \
                 (self.stride - nW % self.stride) % self.stride
        if ph > 0 or pw > 0:
            img_r = np.pad(img_r, ((0, ph), (0, pw), (0, 0)), mode="constant")

        # 转 tensor (CHW, float32, [0,1]) | Convert to tensor
        img_t = torch.from_numpy(img_r.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1)  # HWC → CHW

        # 加载 GT tile scores (分辨率无关，始终在原生分辨率计算)
        # Load GT tile scores (resolution-independent, computed at native resolution)
        cache_path = self.cache_dir / f"{img_id}.npz"
        cached = np.load(cache_path)
        gt_scores = torch.from_numpy(cached["gt_scores"])  # [n_ty, n_tx]
        n_ty, n_tx = int(cached["n_ty"]), int(cached["n_tx"])
        orig_h, orig_w = int(cached["orig_h"]), int(cached["orig_w"])

        return {
            "image": img_t,           # [3, H_pad, W_pad] 归一化 RGB
            "gt_scores": gt_scores,   # [n_ty, n_tx] GT fg_ratio
            "n_ty": n_ty,
            "n_tx": n_tx,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "img_id": img_id,
        }


# ═══════════════════════════════════════════════════════════════════
# FDR 训练 (单分辨率) | FDR Training (single resolution)
# ═══════════════════════════════════════════════════════════════════

def train_fdr_one_resolution(model, train_loader, device, epochs, lr, logger, tag):
    """
    在单一分辨率下训练 FDR | Train FDR at a single resolution.

    Args:
        model: FDRModel
        train_loader: DataLoader
        device: torch device
        epochs: 训练轮数 | training epochs
        lr: 学习率 | learning rate
        logger: adatile logger
        tag: 日志标签前缀 (如 "b05/2048") | log tag prefix

    Returns:
        model: 训练后的模型 (best loss) | trained model (best loss)
        metrics: [(epoch, loss), ...] per-epoch metrics
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    best_state = None
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in train_loader:
            images = batch["image"].to(device)      # [B, 3, H_pad, W_pad]
            gt_scores = batch["gt_scores"].to(device)  # [B, n_ty, n_tx]

            # 前向 → 重要性图 | Forward → importance map
            imp = model(images)  # [B, 1, H_feat, W_feat]
            _, _, Hf, Wf = imp.shape

            # 每个样本: 将重要性图按 tile grid 池化为 tile scores
            # Per-sample: pool importance map by tile grid → tile scores
            losses = []
            for i in range(images.size(0)):
                n_ty_i = int(batch["n_ty"][i])
                n_tx_i = int(batch["n_tx"][i])
                imp_i = imp[i, 0]  # [Hf, Wf]

                # 每个 tile 的重要性 = 对应特征区域的均值 | Per-tile importance = mean of feature region
                preds = []
                for ty in range(n_ty_i):
                    for tx in range(n_tx_i):
                        y0 = int(ty * Hf / n_ty_i)
                        y1 = int((ty + 1) * Hf / n_ty_i)
                        x0 = int(tx * Wf / n_tx_i)
                        x1 = int((tx + 1) * Wf / n_tx_i)
                        y0, y1 = max(0, min(y0, Hf - 1)), max(y0 + 1, min(y1, Hf))
                        x0, x1 = max(0, min(x0, Wf - 1)), max(x0 + 1, min(x1, Wf))
                        preds.append(imp_i[y0:y1, x0:x1].mean())

                pred_t = torch.stack(preds).reshape(n_ty_i, n_tx_i)
                gt_i = gt_scores[i, :n_ty_i, :n_tx_i]
                losses.append(F.mse_loss(pred_t, gt_i))

            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)
        epoch_metrics.append((epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.log_info(tag, f"E{epoch:2d}/{epochs} loss={avg_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    logger.log_info(tag, f"Best loss={best_loss:.6f} | 最佳损失")

    return model, epoch_metrics


# ═══════════════════════════════════════════════════════════════════
# 评估 (Spearman r + FG Retention) | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_fdr(model, eval_loader, device) -> dict:
    """
    评估 FDR: Spearman r + Top-K% FG Retention | Evaluate FDR: Spearman r + FG retention.

    Returns:
        results: {
            "spearman_r": float,
            "pearson_r": float,
            "fg_retention": {K%: float},   # 该 K% 捕获的前景比例
            "n_tiles_total": int,
        }
    """
    model.eval()
    all_preds = []
    all_gts = []
    all_fg_pixels = []   # 每个 tile 的 FG 像素数 (用于 retention)
    all_pred_scores = []  # 每个 tile 的预测分数

    for batch in eval_loader:
        images = batch["image"].to(device)
        gt_scores = batch["gt_scores"]  # [B, n_ty, n_tx]

        imp = model(images)
        _, _, Hf, Wf = imp.shape

        for i in range(images.size(0)):
            n_ty_i = int(batch["n_ty"][i])
            n_tx_i = int(batch["n_tx"][i])
            imp_i = imp[i, 0].cpu().numpy()
            gt_i = gt_scores[i, :n_ty_i, :n_tx_i].numpy()

            # Pool importance map → per-tile scores | 池化重要性图 → 每 tile 分数
            preds = np.zeros((n_ty_i, n_tx_i), dtype=np.float32)
            for ty in range(n_ty_i):
                for tx in range(n_tx_i):
                    y0 = int(ty * Hf / n_ty_i)
                    y1 = int((ty + 1) * Hf / n_ty_i)
                    x0 = int(tx * Wf / n_tx_i)
                    x1 = int((tx + 1) * Wf / n_tx_i)
                    y0, y1 = max(0, min(y0, Hf - 1)), max(y0 + 1, min(y1, Hf))
                    x0, x1 = max(0, min(x0, Wf - 1)), max(x0 + 1, min(x1, Wf))
                    preds[ty, tx] = imp_i[y0:y1, x0:x1].mean()

            # 展平收集 | Flatten and collect
            all_preds.extend(preds.flatten().tolist())
            all_gts.extend(gt_i.flatten().tolist())
            all_pred_scores.extend(preds.flatten().tolist())
            # FG pixels = gt_fg_ratio * tile_area (用 1.0 做近似归一化) | approximate by fg_ratio
            all_fg_pixels.extend(gt_i.flatten().tolist())

    # Spearman / Pearson 相关性 | Correlation
    sr, _ = spearmanr(all_preds, all_gts)
    from scipy.stats import pearsonr
    pr, _ = pearsonr(all_preds, all_gts)

    # Top-K% FG Retention | 按预测分数排序 → 累积前景
    sorted_idx = np.argsort(all_preds)[::-1]  # 降序 | descending
    fg_arr = np.array(all_fg_pixels)
    cum_fg = np.cumsum(fg_arr[sorted_idx])
    total_fg = fg_arr.sum()

    retention = {}
    n_total = len(all_preds)
    for k in K_VALUES:
        nk = max(1, int(n_total * k / 100))
        retained = cum_fg[nk - 1] / max(total_fg, 1e-8)
        retention[k] = float(retained)

    return {
        "spearman_r": float(sr),
        "pearson_r": float(pr),
        "fg_retention": retention,
        "n_tiles_total": n_total,
    }


# ═══════════════════════════════════════════════════════════════════
# 可选: 完整动态选择评估 | Optional: Full Dynamic Selection Eval
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_dynamic_miou(model, decoder, backbone, eval_images,
                          device, K_LIST=None):
    """
    端到端动态选择评估 | End-to-end dynamic selection evaluation.

    对每张图: FDR 预测 → Top-K% tile → Decoder → mIoU。
    需要 Decoder checkpoint。
    """
    if K_LIST is None:
        K_LIST = [10, 25, 50, 100]

    from PIL import Image
    decoder.eval()
    backbone.eval()

    results = {k: {"mious": [], "n_tiles": []} for k in K_LIST}
    MAX_DECODE_BATCH = 16  # 子批次上限防 OOM | sub-batch limit

    for img_id, img_path, anns in tqdm(eval_images, desc="  Dynamic eval"):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_np.shape[:2]
        gt_mask = render_semantic_mask(anns, H, W)

        # FDR 预测 tile scores | FDR predicts tile scores
        n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W + TILE_SIZE - 1) // TILE_SIZE

        # Resize + forward
        scale = model.target_size / max(H, W)  # 需要知道 FDR 训练时的分辨率
        nH, nW = int(H * scale), int(W * scale)
        img_r = np.array(Image.fromarray(img_np).resize((nW, nH), Image.BILINEAR))
        stride = MV3_STRIDE
        ph, pw = (stride - nH % stride) % stride, (stride - nW % stride) % stride
        if ph > 0 or pw > 0:
            img_r = np.pad(img_r, ((0, ph), (0, pw), (0, 0)), mode="constant")

        img_t = torch.from_numpy(img_r.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)

        imp = model(img_t)
        _, _, Hf, Wf = imp.shape
        imp_np = imp[0, 0].cpu().numpy()

        # Pool to tile scores | 池化为 tile scores
        tile_scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0 = int(ty * Hf / n_ty)
                y1 = int((ty + 1) * Hf / n_ty)
                x0 = int(tx * Wf / n_tx)
                x1 = int((tx + 1) * Wf / n_tx)
                y0, y1 = max(0, min(y0, Hf - 1)), max(y0 + 1, min(y1, Hf))
                x0, x1 = max(0, min(x0, Wf - 1)), max(x0 + 1, min(x1, Wf))
                tile_scores[ty, tx] = imp_np[y0:y1, x0:x1].mean()

        # 预提取所有 tile | Pre-extract all tiles
        all_tiles_info = []
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W)
                tile = img_np[y0:y1, x0:x1]
                th, tw = tile.shape[:2]
                if th < TILE_SIZE or tw < TILE_SIZE:
                    p = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    p[:th, :tw] = tile
                    tile = p
                tile_t = torch.from_numpy(tile.astype(np.float32) / 255.0)
                tile_t = tile_t.permute(2, 0, 1)
                all_tiles_info.append((tile_t, y0, y1, x0, x1, th, tw))

        for K in K_LIST:
            if K >= 100:
                sel = np.ones(n_ty * n_tx, dtype=bool)
            else:
                nk = max(1, int(n_ty * n_tx * K / 100))
                idx = np.argsort(tile_scores.flatten())[::-1][:nk]
                sel = np.zeros(n_ty * n_tx, dtype=bool)
                sel[idx] = True

            selected, selected_pos = [], []
            for i, (tile_t, y0, y1, x0, x1, th, tw) in enumerate(all_tiles_info):
                if sel[i]:
                    selected.append(tile_t)
                    selected_pos.append((y0, y1, x0, x1, th, tw))

            pred_full = np.zeros((H, W), dtype=np.int64)
            if selected:
                for sb_start in range(0, len(selected), MAX_DECODE_BATCH):
                    sb_end = min(sb_start + MAX_DECODE_BATCH, len(selected))
                    batch = torch.stack(selected[sb_start:sb_end]).to(device)
                    feats = backbone(batch)
                    logits = decoder(feats["p4"], target_size=(TILE_SIZE, TILE_SIZE))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    for j in range(sb_end - sb_start):
                        y0, y1, x0, x1, th, tw = selected_pos[sb_start + j]
                        pred_full[y0:y0 + min(th, TILE_SIZE),
                                  x0:x0 + min(tw, TILE_SIZE)] = preds[j][:th, :tw]

            # mIoU (15 前景类) | mIoU (15 foreground classes)
            miou_v, valid = 0.0, 0
            for c in range(1, 16):
                pc = (pred_full == c)
                tc = (gt_mask == c)
                inter = (pc & tc).sum()
                union = (pc | tc).sum()
                if union > 0:
                    miou_v += inter / union
                    valid += 1

            results[K]["mious"].append(miou_v / max(valid, 1))
            results[K]["n_tiles"].append(len(selected))

    return results


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_results(resolutions, all_results, all_epoch_metrics, output_path):
    """
    绘制三面板图 | Plot 3-panel figure.

    Panel 1: Spearman r vs Resolution
    Panel 2: FG Retention @ K% vs Resolution (per K line)
    Panel 3: Training loss curves (per resolution)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Spearman r vs Resolution | Spearman r vs 分辨率
    ax = axes[0]
    sr_vals = [all_results[r]["spearman_r"] for r in resolutions]
    pr_vals = [all_results[r]["pearson_r"] for r in resolutions]
    x = np.arange(len(resolutions))
    width = 0.35
    bars1 = ax.bar(x - width / 2, sr_vals, width, label="Spearman r",
                   color="#3498DB", edgecolor="white")
    bars2 = ax.bar(x + width / 2, pr_vals, width, label="Pearson r",
                   color="#2ECC71", edgecolor="white")
    # 数值标注 | Value labels
    for bar, v in zip(bars1, sr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    for bar, v in zip(bars2, pr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in resolutions], fontsize=10)
    ax.set_ylabel("Correlation", fontsize=11)
    ax.set_title("FDR Tile Ranking Quality vs Input Resolution\n"
                 "Tile 排序质量 vs 输入分辨率", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    # 高亮最佳 | Highlight best
    best_idx = np.argmax(sr_vals)
    ax.annotate(f"Best: {resolutions[best_idx]}px\nSpearman r={sr_vals[best_idx]:.3f}",
                xy=(best_idx, sr_vals[best_idx]),
                xytext=(best_idx + 1.5, sr_vals[best_idx] - 0.15),
                arrowprops=dict(arrowstyle="->", color="#E74C3C"),
                fontsize=9, color="#E74C3C", fontweight="bold")

    # Panel 2: FG Retention vs Resolution (per K%) | FG 保留率 vs 分辨率
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(K_VALUES)))
    for ki, k in enumerate(K_VALUES):
        k_vals = [all_results[r]["fg_retention"][k] for r in resolutions]
        ax.plot([str(r) for r in resolutions], k_vals, "o-",
                color=colors[ki], linewidth=2, markersize=8, label=f"K={k}%")
    ax.set_ylabel("FG Retention (% of total FG)", fontsize=11)
    ax.set_title("Top-K% Foreground Retention vs Input Resolution\n"
                 "Top-K% 前景保留率 vs 输入分辨率", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: Training loss curves | 训练 loss 曲线
    ax = axes[2]
    res_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(resolutions)))
    for ri, r in enumerate(resolutions):
        metrics = all_epoch_metrics.get(r, [])
        if metrics:
            epochs, losses = zip(*metrics)
            ax.plot(epochs, losses, color=res_colors[ri], linewidth=1.5,
                    label=f"{r}px", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("FDR Training Loss by Resolution\n"
                 "各分辨率训练 Loss", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("B-05: FDR Input Resolution Ablation — "
                 "Lower Resolution = Comparable Ranking + Less Compute",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "b05_resolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    resolutions = [int(x.strip()) for x in args.resolutions.split(",")]
    src_root = Path(args.src_root)
    split = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志 | Logging
    logger = get_logger("b05")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b05_resolution.jsonl")))

    logger.log_info("b05/start",
                    f"B-05 FDR Resolution Ablation | {len(resolutions)} resolutions: {resolutions}")

    # ═══ 1. 加载数据 | Load Data ═══
    logger.log_info("b05/data", "Loading COCO annotations... | 加载标注")
    ann_file = src_root / split / "annotations" / f"instances_{split}.json"
    with open(ann_file) as f:
        coco = json.load(f)

    # Build image → annotations index
    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_id_to_anns[ann["image_id"]].append(ann)

    img_dir = src_root / split / "images"
    all_images = []
    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if img_path.exists():
            anns = img_id_to_anns.get(img_info["id"], [])
            all_images.append((img_info["file_name"], str(img_path), anns))

    logger.log_info("b05/data", f"Available images: {len(all_images)}")

    # Shuffle + split train/eval
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_images)
    n_train = min(args.train_images, len(all_images))
    n_eval = min(args.eval_images, len(all_images) - n_train)
    train_images = all_images[:n_train]
    eval_images = all_images[n_train:n_train + n_eval]

    logger.log_info("b05/data",
                    f"Train: {len(train_images)}, Eval: {len(eval_images)}")

    # ═══ 2. 构建缓存 (GT tile scores, 只做一次) | Build Cache ═══
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    build_cache(src_root, split, train_images + eval_images, cache_dir)

    # ═══ 3. 逐分辨率训练 + 评估 | Train + Evaluate per Resolution ═══
    all_results = {}
    all_epoch_metrics = {}
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ri, target_size in enumerate(resolutions):
        logger.log_info("b05/train",
                        f"\n{'='*60}\n"
                        f"  Resolution {ri+1}/{len(resolutions)}: {target_size}px\n"
                        f"  {'='*60}")

        # 创建数据集 | Build dataset
        train_ds = ResolutionDataset(train_images, cache_dir,
                                     target_size, img_dir)
        eval_ds = ResolutionDataset(eval_images, cache_dir,
                                    target_size, img_dir)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=False)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=min(2, args.num_workers),
                                 pin_memory=True)

        # 构建模型 | Build model
        model = FDRModel().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log_info("b05/model",
                        f"FDR@{target_size}px: {n_params:,} trainable params")

        # 训练 | Train
        tag = f"b05/{target_size}"
        model, epoch_metrics = train_fdr_one_resolution(
            model, train_loader, device, args.epochs, args.lr, logger, tag
        )
        all_epoch_metrics[target_size] = epoch_metrics

        # 保存 checkpoint | Save checkpoint
        ckpt_path = ckpt_dir / f"fdr_{target_size:04d}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # 评估 | Evaluate
        logger.log_info(tag, "Evaluating Spearman r + FG retention... | 评估排序质量")
        results = evaluate_fdr(model, eval_loader, device)
        all_results[target_size] = results

        logger.log_info(tag,
                        f"Spearman r={results['spearman_r']:.4f}, "
                        f"Pearson r={results['pearson_r']:.4f}, "
                        f"n_tiles={results['n_tiles_total']}")
        for k in K_VALUES:
            logger.log_info(tag,
                            f"  FG@{k}% = {results['fg_retention'][k]*100:.1f}%")

    # ═══ 4. 可视化 | Visualization ═══
    logger.log_info("b05/viz", "Generating plots... | 生成图表")
    plot_results(resolutions, all_results, all_epoch_metrics, output_dir)

    # ═══ 5. 可选: 端到端动态评估 | Optional: End-to-End Dynamic Eval ═══
    if args.decoder_ckpt:
        logger.log_info("b05/dynamic",
                        "Running full dynamic selection evaluation... | 运行端到端评估")

        from adatile.backbone import FastSAMBackbone

        # 加载 Decoder (内联 LightDecoder，与 train_b04.py 一致) | Load decoder
        # Reuse the same LightDecoder class from train_b04
        class LightDecoderEval(nn.Module):
            """FastSAM P4 → 16-class segmentation (mirrors train_b04.py LightDecoder)."""

            def __init__(self, in_channels=1280, num_classes=16):
                super().__init__()
                self.stage1 = nn.Sequential(
                    nn.Conv2d(in_channels, 256, 1, bias=False),
                    nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                )
                self.stage2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                )
                self.stage3 = nn.Sequential(
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                )
                self.head = nn.Conv2d(32, num_classes, 1, bias=True)

            def forward(self, p4, target_size=None):
                x = self.stage1(p4)
                x = F.interpolate(x, scale_factor=2, mode="bilinear",
                                  align_corners=False)
                x = self.stage2(x)
                x = F.interpolate(x, scale_factor=2, mode="bilinear",
                                  align_corners=False)
                x = self.stage3(x)
                x = self.head(x)
                if target_size is not None:
                    x = F.interpolate(x, size=target_size, mode="bilinear",
                                      align_corners=False)
                return x

        backbone = FastSAMBackbone(freeze_backbone=True).eval()
        decoder = LightDecoderEval(1280, 16).to(device)
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
        logger.log_info("b05/dynamic",
                        f"Loaded decoder from {args.decoder_ckpt}")

        # 对每个分辨率: 动态选择评估 | Per resolution: dynamic selection
        K_LIST = [10, 25, 50, 100]
        dynamic_results = {}
        # 只用前 20 张 eval 图做端到端 (省时间) | Use first 20 eval images for speed
        dynamic_images = eval_images[:20]

        for target_size in resolutions:
            logger.log_info("b05/dynamic",
                            f"Dynamic eval @ {target_size}px... | 动态评估")

            model = FDRModel().to(device)
            ckpt_path = ckpt_dir / f"fdr_{target_size:04d}.pt"
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.target_size = target_size  # 记录用于 resize

            results = evaluate_dynamic_miou(
                model, decoder, backbone, dynamic_images, device, K_LIST
            )
            dynamic_results[target_size] = results

            # 打印对比 | Print comparison
            logger.log_info("b05/dynamic",
                            f"  {target_size}px: "
                            f"K=100% mIoU={np.mean(results[100]['mious'])*100:.2f}%, "
                            f"K=50% mIoU={np.mean(results[50]['mious'])*100:.2f}%, "
                            f"K=25% mIoU={np.mean(results[25]['mious'])*100:.2f}%")

        # 保存动态评估结果 | Save dynamic results
        dyn_summary = {}
        for r in resolutions:
            dyn_summary[str(r)] = {
                str(k): {
                    "miou_mean": float(np.mean(dynamic_results[r][k]["mious"])),
                    "n_tiles_mean": float(np.mean(dynamic_results[r][k]["n_tiles"])),
                }
                for k in K_LIST
            }

        with open(output_dir / "b05_dynamic.json", "w") as f:
            json.dump(dyn_summary, f, indent=2)

    # ═══ 6. 保存汇总 | Save Summary ═══
    summary = {
        "experiment": "B-05 FDR Input Resolution Ablation",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "resolutions": resolutions,
            "train_images": len(train_images),
            "eval_images": len(eval_images),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "tile_size": TILE_SIZE,
        },
        "results": {
            str(r): {
                "spearman_r": all_results[r]["spearman_r"],
                "pearson_r": all_results[r]["pearson_r"],
                "fg_retention": {str(k): v for k, v in
                                 all_results[r]["fg_retention"].items()},
                "n_tiles_total": all_results[r]["n_tiles_total"],
            }
            for r in resolutions
        },
    }

    summary_path = output_dir / "b05_resolution.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b05/done",
                    f"Results saved to {output_dir}/ | 结果已保存")
    logger.log_info("b05/done",
                    f"  - {summary_path}")
    logger.log_info("b05/done",
                    f"  - {output_dir / 'b05_resolution.png'}")

    # 打印最终对比表 | Print final comparison table
    logger.log_info("b05/table",
                    f"\n{'='*80}\n"
                    f"  B-05 FINAL: Spearman r vs Input Resolution\n"
                    f"  {'='*80}")
    logger.log_info("b05/table",
                    f"  {'Resolution':<12} {'Spearman r':>12} {'Pearson r':>12} "
                    f"{'FG@50%':>10} {'FLOPs(rel)':>12}")
    logger.log_info("b05/table",
                    f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*12}")

    base_flops = 2048 ** 2  # FLOPs 正比于像素数 | FLOPs ∝ pixel count
    for r in resolutions:
        flops_rel = (r ** 2) / base_flops
        logger.log_info("b05/table",
                        f"  {r:>6}px     {all_results[r]['spearman_r']:>10.4f}   "
                        f"{all_results[r]['pearson_r']:>10.4f}   "
                        f"{all_results[r]['fg_retention'][50]*100:>8.1f}%   "
                        f"{flops_rel:>10.2f}×")

    best_r = min(resolutions, key=lambda r: all_results[r]["spearman_r"]
                 if all_results[r]["spearman_r"] >= 0.80 else -1)
    logger.log_info("b05/conclusion",
                    f"\nConclusion | 结论:\n"
                    f"  If Spearman r drops <5% from 2048 → {best_r}px is sufficient\n"
                    f"  → FDR FLOPs reduced by {(1 - (best_r/2048)**2)*100:.0f}%\n"
                    f"  → Paper B contribution: 'FDR only requires low-resolution context'")


if __name__ == "__main__":
    main()
