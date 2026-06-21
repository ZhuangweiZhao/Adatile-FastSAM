# Oracle Tile Importance Analysis — 回答"什么才算重要"
# ==========================================================
#
# Context:
#   B-04 FDR@2048 动态选择: K=50% → mIoU 保留率仅 68%。
#   问题是 FDR 学得不准，还是 fg_ratio 本身就不是正确的 importance 定义？
#
# This script tests FOUR ranking strategies on the SAME images/tiles/decoder:
#
#   L1: Random          — 下界 | lower bound
#   L2: Oracle fg_ratio  — GT 前景像素占比 (FDR 的训练目标)
#   L3: Oracle IoU       — GT per-tile IoU (每 tile 单独过 decoder)
#   L4: Oracle Contrib   — 边际贡献 | marginal contribution
#                          importance_i = mIoU_full - mIoU_without_tile_i
#
# 输出:
#   - 一张表: 四种 ranking × 五个 K% → mIoU 保留率
#   - Figure: mIoU vs K% curves (四种策略对比)
#   - Figure: fg_ratio vs IoU vs Contribution 散点图 (验证它们是否等价)
#
# 关键假设验证:
#   H0: fg_ratio ≈ per-tile IoU ≈ contribution → FDR 当前目标正确
#   H1: fg_ratio ≠ per-tile IoU → 存在更好的 importance 定义
#   H2: per-tile IoU ≠ contribution → tile 间存在交互, 需 submodular selection
#
# 用法:
#   python tools/paper_b/eval_b05_oracle_importance.py \
#       --src-root /root/autodl-tmp/iSAID_processed \
#       --decoder-ckpt runs/b04_v3/decoder_best.pt \
#       --n-images 20 --output-dir runs/b05_oracle

import sys, argparse, json, datetime
from pathlib import Path
from itertools import combinations
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone

TILE_SIZE = 1024
NUM_CLASSES = 15
NUM_OUT_CH = 16
K_VALUES = [10, 20, 30, 40, 50, 70, 100]

# ═══════════════════════════════════════════════════════════════════
# LightDecoder (与 train_b04.py 内联版一致)
# ═══════════════════════════════════════════════════════════════════

class LightDecoder(nn.Module):
    """FastSAM P4 → 16类分割 (mirrors train_b04.py)."""
    def __init__(self, in_channels=1280, num_classes=16):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, 1, bias=True)

    def forward(self, p4, target_size=None):
        x = self.stage1(p4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════
# 语义掩码渲染
# ═══════════════════════════════════════════════════════════════════

def render_semantic_mask(annotations, h, w):
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0: continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            sem[max(0, int(bbox[1])):min(h, int(bbox[1]+bbox[3])),
                max(0, int(bbox[0])):min(w, int(bbox[0]+bbox[2]))] = cat_id
            continue
        if isinstance(seg, dict): continue
        for poly in (seg if isinstance(seg[0], list) else [seg]):
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w-1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h-1)
            cv2.fillPoly(sem, [pts], int(cat_id))
    return sem


# ═══════════════════════════════════════════════════════════════════
# mIoU 计算 (15 前景类)
# ═══════════════════════════════════════════════════════════════════

def compute_miou(pred_mask, gt_mask):
    miou_v, valid = 0.0, 0
    for c in range(1, NUM_OUT_CH):
        pc = (pred_mask == c); tc = (gt_mask == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0: miou_v += inter / union; valid += 1
    return miou_v / max(valid, 1)


# ═══════════════════════════════════════════════════════════════════
# 提取所有 tile (只做一次) | Extract all tiles (done once)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_tiles_from_image(img_np, gt_mask, backbone, decoder, device):
    """提取全图的所有 tile → 每个 tile 的 tensor + 坐标 + 单独的 pred。"""
    from PIL import Image
    H, W = img_np.shape[:2]
    n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
    n_tx = (W + TILE_SIZE - 1) // TILE_SIZE

    tiles_info = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H)
            x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W)
            tile = img_np[y0:y1, x0:x1]
            th, tw = tile.shape[:2]
            if th < TILE_SIZE or tw < TILE_SIZE:
                p = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                p[:th, :tw] = tile; tile = p
            tile_t = torch.from_numpy(tile.astype(np.float32)/255.0).permute(2, 0, 1)

            # Per-tile decoder forward (单独跑 → 算 IoU)
            tile_batch = tile_t.unsqueeze(0).to(device)
            feats = backbone(tile_batch)
            logit = decoder(feats["p4"], target_size=(TILE_SIZE, TILE_SIZE))
            pred_tile = logit.argmax(dim=1).cpu().numpy()[0]
            gt_tile = gt_mask[y0:y0+th, x0:x0+tw]

            tile_iou = compute_miou(pred_tile[:th, :tw], gt_tile)

            tiles_info.append({
                "tensor": tile_t,        # [3, 1024, 1024]
                "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "th": th, "tw": tw,
                "pred": pred_tile,       # [1024, 1024] 完整 tile 预测
                "tile_iou": tile_iou,
                "fg_ratio": float((gt_tile > 0).sum() / max(th*tw, 1)),
            })

    return tiles_info, n_ty, n_tx


# ═══════════════════════════════════════════════════════════════════
# Ranking 策略
# ═══════════════════════════════════════════════════════════════════

def get_ranked_indices(tiles_info, strategy, rng=None, full_miou=None):
    """
    按指定策略排序 → 返回 tile 索引 (从最重要到最不重要).

    Args:
        tiles_info: list of per-tile dicts
        strategy: "random" | "fg_ratio" | "tile_iou" | "contribution"
        rng: np.random.RandomState (for "random")
        full_miou: float, 全图 mIoU (仅 "contribution" 需要)

    Returns:
        ranked_indices: [idx_most_important, ..., idx_least_important]
        importances: importance score per tile (for analysis/plotting)
    """
    n = len(tiles_info)

    if strategy == "random":
        order = rng.permutation(n)
        return order.tolist(), np.zeros(n)

    elif strategy == "fg_ratio":
        scores = np.array([t["fg_ratio"] for t in tiles_info])
        order = np.argsort(scores)[::-1]
        return order.tolist(), scores

    elif strategy == "tile_iou":
        scores = np.array([t["tile_iou"] for t in tiles_info])
        order = np.argsort(scores)[::-1]
        return order.tolist(), scores

    elif strategy == "contribution":
        # importance_i = mIoU_full - mIoU_without_i
        importances = np.zeros(n)
        for i in range(n):
            # 用所有 EXCEPT tile i 的预测拼接全图 → 算 mIoU
            # 注意: 跳过的 tile 区域 pred=0 (BG)
            importances[i] = full_miou  # 稍后在调用处填充
        # 实际计算在外部完成 (需要 decoder)
        # 这里返回占位
        raise NotImplementedError("Contribution importances computed externally")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════
# 按 K% 选择 tile → 拼接全图 → 算 mIoU
# ═══════════════════════════════════════════════════════════════════

def evaluate_at_k(ranked_indices, tiles_info, gt_mask, K_pct):
    """按 ranked_indices 的前 K% tile 拼接全图 → 算 mIoU."""
    n = len(ranked_indices)
    nk = max(1, int(n * K_pct / 100))
    selected = set(ranked_indices[:nk])

    H, W = gt_mask.shape
    pred_full = np.zeros((H, W), dtype=np.int64)

    for i, tinfo in enumerate(tiles_info):
        if i not in selected:
            continue
        y0, y1 = tinfo["y0"], tinfo["y1"]
        x0, x1 = tinfo["x0"], tinfo["x1"]
        th, tw = tinfo["th"], tinfo["tw"]
        pred_full[y0:y0+th, x0:x0+tw] = tinfo["pred"][:th, :tw]

    return compute_miou(pred_full, gt_mask)


# ═══════════════════════════════════════════════════════════════════
# L4: Oracle Contribution (Leave-One-Out)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_contribution_importances(tiles_info, n_ty, n_tx, gt_mask,
                                     backbone, decoder, device):
    """
    对每个 tile i: importance_i = mIoU_full - mIoU_without_i

    逐个 leave-one-out 计算真实边际贡献。
    """
    # 先算全图 mIoU (所有 tile 拼接) | Full-image mIoU
    H, W = gt_mask.shape
    pred_full = np.zeros((H, W), dtype=np.int64)
    for tinfo in tiles_info:
        y0, y1 = tinfo["y0"], tinfo["y1"]
        x0, x1 = tinfo["x0"], tinfo["x1"]
        th, tw = tinfo["th"], tinfo["tw"]
        pred_full[y0:y0+th, x0:x0+tw] = tinfo["pred"][:th, :tw]
    full_miou = compute_miou(pred_full, gt_mask)

    # Leave-One-Out per tile
    n = len(tiles_info)
    importances = np.zeros(n)

    for i in tqdm(range(n), desc="  L4 Contribution", leave=False):
        pred_loo = np.zeros((H, W), dtype=np.int64)
        for j, tinfo in enumerate(tiles_info):
            if j == i:
                continue  # 跳过 tile i | Leave tile i out
            y0, y1 = tinfo["y0"], tinfo["y1"]
            x0, x1 = tinfo["x0"], tinfo["x1"]
            th, tw = tinfo["th"], tinfo["tw"]
            pred_loo[y0:y0+th, x0:x0+tw] = tinfo["pred"][:th, :tw]

        miou_loo = compute_miou(pred_loo, gt_mask)
        importances[i] = full_miou - miou_loo

    order = np.argsort(importances)[::-1]
    return order.tolist(), importances, full_miou


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════

def plot_oracle_results(all_results, all_scatter_data, output_path):
    """双面板图 | 2-panel figure."""
    strategies = list(all_results.keys())
    colors = {"random": "#95A5A6", "fg_ratio": "#E74C3C",
              "tile_iou": "#3498DB", "contribution": "#27AE60"}
    labels = {"random": "Random", "fg_ratio": "Oracle fg_ratio",
              "tile_iou": "Oracle IoU", "contribution": "Oracle Contrib"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: mIoU Retention vs K% (每种策略一条曲线)
    ax = axes[0]
    for strategy in strategies:
        avgs = [all_results[strategy][k]["miou_mean"] for k in K_VALUES]
        ax.plot(K_VALUES, [a * 100 for a in avgs], "o-",
                color=colors.get(strategy, "#333"),
                linewidth=2.5 if strategy == "contribution" else 2,
                markersize=8, label=labels.get(strategy, strategy),
                alpha=0.9)

    ax.set_xlabel("K% (Tiles Selected)", fontsize=12)
    ax.set_ylabel("FG-mIoU (%)", fontsize=12)
    ax.set_title("Oracle Importance Analysis: mIoU vs K%\n"
                 "四种 Tile Importance 定义对比", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # 标注 50% 关键点
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    for strategy in strategies:
        miou_50 = all_results[strategy][50]["miou_mean"] * 100
        ax.annotate(f"{miou_50:.1f}%",
                    xy=(50, miou_50),
                    textcoords="offset points", xytext=(5, -10),
                    fontsize=8, color=colors.get(strategy, "#333"))

    # Panel 2: fg_ratio vs Per-tile IoU vs Contribution 散点图
    ax = axes[1]
    if all_scatter_data:
        fg_ratios = all_scatter_data["fg_ratio"]
        tile_ious = all_scatter_data["tile_iou"]
        contribs = all_scatter_data.get("contribution", None)

        ax.scatter(fg_ratios, tile_ious, c="#3498DB", alpha=0.5, s=20,
                   label="fg_ratio vs IoU")
        # 拟合线
        from scipy.stats import spearmanr
        sr, _ = spearmanr(fg_ratios, tile_ious)
        ax.text(0.95, 0.05, f"Spearman r={sr:.3f}",
                transform=ax.transAxes, fontsize=11, ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("GT fg_ratio (Foreground Density)", fontsize=12)
        ax.set_ylabel("Per-tile IoU", fontsize=12)
        ax.set_title("Is Foreground Density a Good Importance Proxy?\n"
                     f"前景密度 vs 分割质量 (Spearman r={sr:.3f})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("B-05 Oracle: What Makes a Tile Important?",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "oracle_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--decoder-ckpt", type=str, required=True)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--skip-contribution", action="store_true",
                   help="跳过 L4 Contribution (慢，每个 tile 一次推理)")
    p.add_argument("--output-dir", type=str, default="runs/b05_oracle")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    rng = np.random.RandomState(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b05_oracle")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "oracle.jsonl")))

    # ── 加载数据 | Load Data ──
    src_root = Path(args.src_root)
    with open(src_root / "train" / "annotations" / "instances_train.json") as f:
        coco = json.load(f)

    img_id_to_anns = {}
    for ann in coco["annotations"]:
       _img_id = ann["image_id"]
       img_id_to_anns.setdefault(_img_id, []).append(ann)

    img_dir = src_root / "train" / "images"
    images = []
    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if img_path.exists():
            anns = img_id_to_anns.get(img_info["id"], [])
            if anns:
                images.append((img_info, str(img_path), anns))

    rng.shuffle(images)
    images = images[:args.n_images]
    logger.log_info("b05/data", f"Test images: {len(images)}")

    # ── 加载模型 | Load Models ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, 16).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
    decoder.eval()
    logger.log_info("b05/model", f"Loaded decoder from {args.decoder_ckpt}")

    # ── 逐图提取 tile + 预计算 per-tile IoU ──
    logger.log_info("b05/extract", "Extracting tiles + computing per-tile IoU...")
    all_image_data = []
    all_scatter_fg, all_scatter_iou, all_scatter_contrib = [], [], []

    for img_info, img_path, anns in tqdm(images, desc="  Extract tiles"):
        from PIL import Image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_np.shape[:2]
        gt_mask = render_semantic_mask(anns, H, W)

        tiles_info, n_ty, n_tx = extract_tiles_from_image(
            img_np, gt_mask, backbone, decoder, device
        )

        # L4: Contribution (Leave-One-Out)
        if not args.skip_contribution:
            contrib_order, contrib_imp, full_miou = compute_contribution_importances(
                tiles_info, n_ty, n_tx, gt_mask, backbone, decoder, device
            )
        else:
            # Placeholder — 跳过
            # 先算 full_miou
            pred_full = np.zeros((H, W), dtype=np.int64)
            for tinfo in tiles_info:
                y0, y1 = tinfo["y0"], tinfo["y1"]
                x0, x1 = tinfo["x0"], tinfo["x1"]
                th, tw = tinfo["th"], tinfo["tw"]
                pred_full[y0:y0+th, x0:x0+tw] = tinfo["pred"][:th, :tw]
            full_miou = compute_miou(pred_full, gt_mask)
            # fg_ratio 作为 contribution 的近似 (仅用于散点图)
            contrib_imp = np.array([t["fg_ratio"] for t in tiles_info])
            contrib_order = np.argsort(contrib_imp)[::-1].tolist()

        # 收集散点数据 | Collect scatter data
        for t in tiles_info:
            all_scatter_fg.append(t["fg_ratio"])
            all_scatter_iou.append(t["tile_iou"])
        all_scatter_contrib.extend(contrib_imp.tolist() if hasattr(contrib_imp, 'tolist')
                                   else list(contrib_imp))

        all_image_data.append({
            "tiles_info": tiles_info,
            "gt_mask": gt_mask,
            "full_miou": full_miou,
            "n_tiles": len(tiles_info),
        })

    logger.log_info("b05/extract",
                    f"Total tiles: {sum(d['n_tiles'] for d in all_image_data)}")

    # ── 对每种 ranking 策略 × 每个 K% → 算 mIoU ──
    strategies = {
        "random": ("L1 Random", None),
        "fg_ratio": ("L2 Oracle fg_ratio", None),
        "tile_iou": ("L3 Oracle IoU", None),
    }
    if not args.skip_contribution:
        strategies["contribution"] = ("L4 Oracle Contrib", None)

    all_results = {s: {k: {"mious": [], "retentions": []} for k in K_VALUES}
                   for s in strategies}

    logger.log_info("b05/eval", "Evaluating ranking strategies...")

    for img_idx, img_data in enumerate(tqdm(all_image_data, desc="  Ranking eval")):
        tiles_info = img_data["tiles_info"]
        gt_mask = img_data["gt_mask"]
        full_miou = img_data["full_miou"]
        n_tiles = len(tiles_info)

        for strategy in strategies:
            if strategy == "random":
                order, _ = get_ranked_indices(tiles_info, "random", rng=rng)
            elif strategy == "fg_ratio":
                order, _ = get_ranked_indices(tiles_info, "fg_ratio")
            elif strategy == "tile_iou":
                order, _ = get_ranked_indices(tiles_info, "tile_iou")
            elif strategy == "contribution":
                contrib_order, contrib_imp, _ = compute_contribution_importances(
                    tiles_info,
                    (img_data["gt_mask"].shape[0] + TILE_SIZE - 1) // TILE_SIZE,
                    (img_data["gt_mask"].shape[1] + TILE_SIZE - 1) // TILE_SIZE,
                    gt_mask, backbone, decoder, device
                )
                order = contrib_order
            else:
                continue

            for k in K_VALUES:
                miou_k = evaluate_at_k(order, tiles_info, gt_mask, k)
                all_results[strategy][k]["mious"].append(miou_k)
                all_results[strategy][k]["retentions"].append(
                    miou_k / max(full_miou, 1e-8))

    # ── 汇总 | Aggregate ──
    logger.log_info("b05/table",
                    f"\n{'='*85}\n"
                    f"  ORACLE TILE IMPORTANCE ANALYSIS\n"
                    f"  {'='*85}")
    logger.log_info("b05/table",
                    f"  {'Strategy':<22} {'K=100%':>8} {'K=50%':>9} "
                    f"{'K=30%':>9} {'K=20%':>9} {'K=10%':>9}"
                    f"  {'Ret@50%':>9}")
    logger.log_info("b05/table",
                    f"  {'─'*22} {'─'*8} {'─'*9} {'─'*9} {'─'*9} {'─'*9}"
                    f"  {'─'*9}")

    final_summary = {}
    for strategy in strategies:
        avgs = {}
        for k in K_VALUES:
            avgs[k] = {
                "miou_mean": float(np.mean(all_results[strategy][k]["mious"])),
                "miou_std": float(np.std(all_results[strategy][k]["mious"])),
                "retention_mean": float(np.mean(all_results[strategy][k]["retentions"])),
            }

        logger.log_info("b05/table",
                        f"  {strategies[strategy][0]:<22} "
                        f"{avgs[100]['miou_mean']*100:>7.2f}% "
                        f"{avgs[50]['miou_mean']*100:>8.2f}% "
                        f"{avgs[30]['miou_mean']*100:>8.2f}% "
                        f"{avgs[20]['miou_mean']*100:>8.2f}% "
                        f"{avgs[10]['miou_mean']*100:>8.2f}% "
                        f" {avgs[50]['retention_mean']*100:>8.1f}%")

        final_summary[strategy] = avgs

    # ── 关键结论 | Key Conclusions ──
    logger.log_info("b05/conclusion", f"\n{'─'*60}")
    logger.log_info("b05/conclusion", "KEY FINDINGS | 关键发现:")

    fg_ret50 = final_summary["fg_ratio"][50]["retention_mean"] * 100
    iou_ret50 = final_summary["tile_iou"][50]["retention_mean"] * 100
    logger.log_info("b05/conclusion",
                    f"  Oracle fg_ratio @ K=50%: {fg_ret50:.1f}% retention")
    logger.log_info("b05/conclusion",
                    f"  Oracle IoU    @ K=50%: {iou_ret50:.1f}% retention")

    if "contribution" in final_summary:
        contrib_ret50 = final_summary["contribution"][50]["retention_mean"] * 100
        logger.log_info("b05/conclusion",
                        f"  Oracle Contrib @ K=50%: {contrib_ret50:.1f}% retention  ← 理论上界")

    # 诊断 | Diagnosis
    if iou_ret50 - fg_ret50 > 15:
        logger.log_info("b05/conclusion",
                        "\n  ⚠️  Oracle IoU >> Oracle fg_ratio\n"
                        "  → fg_ratio is a SUBOPTIMAL importance proxy\n"
                        "  → Crucial research gap identified: 'What makes a tile important?'")
    elif iou_ret50 - fg_ret50 > 5:
        logger.log_info("b05/conclusion",
                        "\n  △  Oracle IoU > Oracle fg_ratio (moderate gap)\n"
                        "  → fg_ratio is acceptable but not optimal")
    else:
        logger.log_info("b05/conclusion",
                        "\n  ✅ Oracle IoU ≈ Oracle fg_ratio\n"
                        "  → fg_ratio is a valid importance proxy\n"
                        "  → Current FDR target is well-justified")

    if "contribution" in final_summary:
        gap = contrib_ret50 - iou_ret50
        if gap > 5:
            logger.log_info("b05/conclusion",
                            f"\n  ⚠️  Oracle Contrib - Oracle IoU = {gap:.1f}%\n"
                            f"  → Tile interactions exist (submodularity matters)\n"
                            f"  → Per-tile IoU alone insufficient; marginal contribution is the true target")

    # ── 可视化 | Visualization ──
    scatter_data = {
        "fg_ratio": all_scatter_fg,
        "tile_iou": all_scatter_iou,
        "contribution": all_scatter_contrib,
    }
    plot_oracle_results(final_summary, scatter_data, output_dir)

    # ── 保存 | Save ──
    summary_json = {
        "experiment": "B-05 Oracle Tile Importance",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {"n_images": len(images), "tile_size": TILE_SIZE},
        "results": {
            s: {str(k): {"miou_mean": v[k]["miou_mean"],
                         "retention_mean": v[k]["retention_mean"]}
                for k in K_VALUES}
            for s, v in final_summary.items()
        },
    }
    with open(output_dir / "oracle_results.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.log_info("b05/done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
