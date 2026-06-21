#!/usr/bin/env python3
"""
B-04: Dynamic Tile Selection — FDR + Decoder 端到端验证
==========================================================

Paper B 最终实验: FDR 预测 tile 重要性 → Top-K 选择 → Decoder 只处理选中 tile → 测 Accuracy vs Compute.

诊断结论 (2026-06-20):
    Exp1: FG>5% 过滤 → FG-mIoU 0.801 (vs FG>1% 的 0.0005). 根因 = 数据采样, 非架构.
    Exp2: Binary segmentation → ? (pending, 预期 >0.8).

已应用修复:
    - FG>5% 过滤 (而非 FG>1%)
    - Focal γ=5.0
    - 稀有类过采样 (plane/pool/soccer ×5)
    - 使用全部 FG>5% tile (非采样)

用法:
    python tools/eval_b04_dynamic_tile_selection.py
    python tools/eval_b04_dynamic_tile_selection.py --decoder-epochs 50 --batch-size 8
"""

import sys, argparse, json, datetime, os, time
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.datasets.isaid_tiles import FastISAIDTileDataset

TILE_SIZE = 1024
NUM_CLASSES = 15
NUM_OUT_CH = 16
RARE_CLASSES = {12: 5, 14: 5, 15: 5}  # pool/soccer/plane ×5 oversample
MAX_DECODE_BATCH = 16  # max tiles in one decode batch (OOM guard)


def _render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """渲染语义掩码 [H,W] uint8 | Render semantic mask.
    直接使用 ann["category_id"]（映射已在预处理中完成 | mapping done in preprocessing）."""
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0: continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0,0,0,0])
            sem[max(0,int(bbox[1])):min(h,int(bbox[1]+bbox[3])),
                max(0,int(bbox[0])):min(w,int(bbox[0]+bbox[2]))] = cat_id
            continue
        if isinstance(seg, dict): continue
        for poly in (seg if isinstance(seg[0], list) else [seg]):
            pts = np.array(poly, dtype=np.int32).reshape(-1,1,2)
            pts[:,:,0] = np.clip(pts[:,:,0], 0, w-1)
            pts[:,:,1] = np.clip(pts[:,:,1], 0, h-1)
            cv2.fillPoly(sem, [pts], cat_id)
    return sem


def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="B-04 Dynamic Tile Selection | B-04 动态瓦片选择")
    p.add_argument("--src-root", type=str, default="data/iSAID_processed",
                   help="iSAID 处理后数据目录 | iSAID processed data directory")
    p.add_argument("--tile-root", type=str, default="data/iSAID_tiles",
                   help="瓦片数据目录 | Tile data directory")
    p.add_argument("--decoder-epochs", type=int, default=50,
                   help="Decoder 训练轮数 | Decoder training epochs")
    p.add_argument("--fdr-epochs", type=int, default=20,
                   help="FDR 训练轮数 | FDR training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率 | Learning rate")
    p.add_argument("--batch-size", type=int, default=8,
                   help="批次大小 | Batch size")
    p.add_argument("--image-size", type=int, default=2048,
                   help="全图缩放尺寸 | Full image resize dimension")
    p.add_argument("--top-k-list", type=str, default="10,20,30,40,50,70,100",
                   help="K% 列表 | Comma-separated K% values")
    p.add_argument("--output-dir", type=str, default="runs/b04_dynamic_tile",
                   help="输出目录 | Output directory for logs + checkpoints")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 | Device to run on")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Light Decoder (渐进上采样, Exp1 验证有效)
# ═══════════════════════════════════════════════════════════════════

class LightDecoder(nn.Module):
    """
    FastSAM P4 → 三步渐进上采样 → 16类分割 | FastSAM P4 → three-step progressive upsampling → 16-class segmentation.

    P4 [B,1280,H/16,W/16] → Conv 1280→64 → Upsample×2 → Conv 64→64 → Upsample×2
    → Conv 64→32 → Upsample×2 → Conv 32→32 → Upsample → Conv 32→C.

    ~800K 参数, 设计用于大尺寸 tile (1024×1024) 的快速推理。
    ~800K params, designed for fast inference on large tiles (1024×1024).
    """

    def __init__(self, in_channels=1280, num_classes=16):
        """初始化渐进上采样解码器 | Initialize progressive upsampling decoder."""
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
        """前向传播: 三步上采样 → 分割 logits | Forward: three-step upsample → segmentation logits."""
        # Stage 1: 1280→256→128, 上采样 ×2 | 1280→256→128, upsample ×2
        x = self.stage1(p4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Stage 2: 128→64, 上采样 ×2 | 128→64, upsample ×2
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Stage 3: 64→32, 上采样 ×2 → head | 64→32, upsample ×2 → head
        x = self.stage3(x)
        x = self.head(x)
        # 最终上采样到目标分辨率 | Final upsample to target resolution
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════
# Tile Dataset (FG>5% 过滤 + 稀有类过采样)
# ═══════════════════════════════════════════════════════════════════

def build_train_dataset(tile_root, log):
    """构建训练数据集：FG>5% 过滤 + 稀有类过采样 | Build training dataset: FG>5% filter + rare class oversampling.

    两步处理 | Two-step processing:
    1. FG>5% 过滤: 丢弃前景占比 ≤5% 的 tile (主要是 BG 为主的 tile)
       FG>5% filter: discard tiles with ≤5% foreground (mostly BG-dominated tiles)
    2. 稀有类过采样: pool/soccer/plane 等少数类 ×5 复制
       Rare class oversampling: pool/soccer/plane ×5 replication
    """
    from PIL import Image

    ds = FastISAIDTileDataset(tile_root, split="train", semantic=True)
    log("b04/data", f"All train tiles: {len(ds)}")

    # 过滤 FG>5% | Filter FG>5%
    fg5_tiles, fg5_class_info = [], []
    for fname in tqdm(ds._tiles, desc="  Filter FG>5%"):
        mask = np.array(Image.open(ds._mask_dir / fname))
        fg_r = (mask > 0).sum() / mask.size
        if fg_r > 0.05:
            fg5_tiles.append(fname)
            # 记录包含哪些稀有类 | Track which rare classes present
            has_rare = [c for c in RARE_CLASSES if (mask == c).sum() > 0]
            fg5_class_info.append(has_rare)

    log("b04/data",
        f"FG>5% tiles: {len(fg5_tiles)} ({len(fg5_tiles)/len(ds)*100:.1f}%)")

    # 稀有类过采样 | Rare class oversampling
    for c, factor in RARE_CLASSES.items():
        tiles_with_c = [(t, info) for t, info in zip(fg5_tiles, fg5_class_info)
                        if c in info]
        n_orig = len(tiles_with_c)
        for _ in range(n_orig * (factor - 1)):
            fg5_tiles.append(tiles_with_c[_ % n_orig][0])
        log("b04/data",
            f"  Class{c}: {n_orig} tiles → {n_orig * factor} (×{factor})")

    ds._tiles = fg5_tiles
    log("b04/data", f"Final train tiles: {len(ds)} (with oversampling)")
    return ds


# ═══════════════════════════════════════════════════════════════════
# FDR (B-03 主线)
# ═══════════════════════════════════════════════════════════════════

class FDRPredictor(nn.Module):
    """
    FDR (Foreground Density Router) | 前景密度路由器.

    冻结 MV3 backbone → DensityHead → 输出 tile 重要性分数。
    训练目标：预测每个 tile 的 fg_ratio (前景像素占比)。
    Frozen MV3 backbone → DensityHead → outputs tile importance scores.
    Training target: predict fg_ratio (foreground pixel fraction) per tile.
    """

    def __init__(self):
        """初始化 FDR: 冻结 MV3 + 可训练 DensityHead | Init FDR: frozen MV3 + trainable DensityHead."""
        super().__init__()
        import torchvision.models as models
        mnet = models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = mnet.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        from adatile.sparse.spatial_router import DensityHead
        self.density_head = DensityHead(576, 128)

    def forward(self, x):
        """前向传播：backbone → density head → 重要性图 | Forward: backbone → density head → importance map."""
        return self.density_head(self.backbone(x))

    @torch.no_grad()
    def predict_tile_scores(self, img_np, device):
        """预测每张图像上所有 tile 的重要性分数 | Predict importance scores for all tiles on an image.

        流程 | Flow:
        1. 缩放图像到固定尺寸 → 前向 → 得到重要性图
           Resize image to fixed size → forward → get importance map
        2. 将重要性图按 tile grid 分块 → 每块取均值作为 tile score
           Divide importance map by tile grid → per-tile mean = score
        """
        from PIL import Image
        H, W = img_np.shape[:2]
        scale = 2048 / max(H, W)
        nH, nW = int(H*scale), int(W*scale)
        img_r = np.array(Image.fromarray(img_np).resize((nW, nH), Image.BILINEAR))
        ph, pw = (32-nH%32)%32, (32-nW%32)%32
        if ph>0 or pw>0:
            img_r = np.pad(img_r, ((0,ph),(0,pw),(0,0)), mode="constant")
        img_t = torch.from_numpy(img_r.astype(np.float32)/255.0)
        img_t = img_t.permute(2,0,1).unsqueeze(0).to(device)
        imp = self.forward(img_t)
        hp, wp = imp.shape[2], imp.shape[3]
        n_ty = (H+TILE_SIZE-1)//TILE_SIZE
        n_tx = (W+TILE_SIZE-1)//TILE_SIZE
        scores = np.zeros((n_ty, n_tx), dtype=np.float32)
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0 = int(ty*TILE_SIZE*scale/2048*hp)
                y1 = int(min(ty*TILE_SIZE+TILE_SIZE, H)*scale/2048*hp)
                x0 = int(tx*TILE_SIZE*scale/2048*wp)
                x1 = int(min(tx*TILE_SIZE+TILE_SIZE, W)*scale/2048*wp)
                y0, y1 = max(0,min(y0,hp-1)), max(y0+1,min(y1,hp))
                x0, x1 = max(0,min(x0,wp-1)), max(x0+1,min(x1,wp))
                scores[ty,tx] = float(imp[0,0,y0:y1,x0:x1].mean())
        return scores


# ═══════════════════════════════════════════════════════════════════
# Step 1: Train Decoder
# ═══════════════════════════════════════════════════════════════════

def train_decoder(args, device, log):
    """Step 1: 训练 LightDecoder | Step 1: Train LightDecoder.

    使用 FG>5% 过滤 + 稀有类过采样的 tile 训练解码器。
    Loss = 0.5 * Focal(γ=5) + 0.5 * Dice。
    验证集：train FG>5%, val all, val FG>5% 三组指标。
    Trains decoder on FG>5% filtered + rare-class oversampled tiles.
    Loss = 0.5 * Focal(γ=5) + 0.5 * Dice.
    Validation: three sets — train FG>5%, val all, val FG>5%.
    """
    log("b04/decoder", f"{'='*50}")
    log("b04/decoder", f"Step 1: Train LightDecoder ({args.decoder_epochs} epochs) | 训练解码器")

    train_ds = build_train_dataset(args.tile_root, log)
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)
    log("b04/decoder", f"Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 过滤 val 为 FG>5% 用于训练监控 | Filter val to FG>5% for training monitor
    from PIL import Image
    val_fg5_tiles = []
    for fname in tqdm(val_ds._tiles, desc="  Filter Val FG>5%"):
        mask = np.array(Image.open(val_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.05:
            val_fg5_tiles.append(fname)
    log("b04/decoder", f"Val FG>5% tiles: {len(val_fg5_tiles)}/{len(val_ds)} "
                       f"({len(val_fg5_tiles)/len(val_ds)*100:.1f}%)")
    val_fg5_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)
    val_fg5_ds._tiles = val_fg5_tiles
    val_fg5_loader = DataLoader(val_fg5_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, NUM_OUT_CH).to(device)
    n_p = sum(p.numel() for p in decoder.parameters())
    log("b04/decoder", f"Decoder: {n_p:,} params")

    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.decoder_epochs, eta_min=1e-6)
    best_miou, best_state = 0.0, None

    # 每 epoch 指标记录 | Per-epoch metrics log
    metrics_path = Path(args.output_dir) / "decoder_metrics.jsonl"

    for epoch in range(1, args.decoder_epochs + 1):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"  Dec E{epoch}", leave=False):
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            feats = backbone(img)
            logit = decoder(feats["p4"], target_size=tgt.shape[1:])

            # Focal γ=5.0 + Dice 组合损失 | Focal γ=5.0 + Dice combined loss
            # Focal: 对难例加权，缓解类别不平衡 | Focal: hard-example weighting for class imbalance
            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            focal_loss = ((1 - torch.exp(-ce)) ** 5.0 * ce).mean()

            # Dice: 逐前景类计算，忽略 BG | Dice: per foreground class, ignore BG (c=0)
            probs = F.softmax(logit, dim=1)
            dice_sum, vd = 0.0, 0
            for c in range(1, NUM_OUT_CH):
                p_c = probs[:, c]; t_c = (tgt == c).float()
                inter = (p_c * t_c).sum()
                union = p_c.sum() + t_c.sum() + 1e-8
                if t_c.sum() > 0: dice_sum += (2*inter/union); vd += 1
            dice_loss = 1.0 - (dice_sum / max(vd, 1))

            # 组合损失 (1:1 权重) | Combined loss (1:1 weight)
            loss = 0.5 * focal_loss + 0.5 * dice_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)

        # Validation (3 sets: train FG>5% + val full + val FG>5%) | 验证 (三组: train FG>5% + val 全量 + val FG>5%)
        decoder.eval()
        per_cls = {
            "train": (torch.zeros(NUM_OUT_CH, device=device),
                      torch.zeros(NUM_OUT_CH, device=device)),
            "val_all": (torch.zeros(NUM_OUT_CH, device=device),
                        torch.zeros(NUM_OUT_CH, device=device)),
            "val_fg5": (torch.zeros(NUM_OUT_CH, device=device),
                        torch.zeros(NUM_OUT_CH, device=device)),
        }
        with torch.no_grad():
            for key, loader in [("train", train_loader), ("val_all", val_loader),
                                ("val_fg5", val_fg5_loader)]:
                inter, union = per_cls[key]
                for batch in loader:
                    img = batch["image"].to(device)
                    tgt = batch["mask"].to(device)
                    feats = backbone(img)
                    logit = decoder(feats["p4"], target_size=tgt.shape[1:])
                    pred = logit.argmax(dim=1)
                    for c in range(1, NUM_OUT_CH):
                        pc = (pred == c); tc = (tgt == c)
                        inter[c] += (pc & tc).sum().float()
                        union[c] += (pc | tc).sum().float()

        def _calc_miou(inter, union):
            s, v = 0.0, 0
            for c in range(1, NUM_OUT_CH):
                if union[c] > 0: s += (inter[c] / union[c]).item(); v += 1
            return s / max(v, 1), int(v)

        miou_train, valid_train = _calc_miou(*per_cls["train"])
        miou_all, valid_all = _calc_miou(*per_cls["val_all"])
        miou_fg5, valid_fg5 = _calc_miou(*per_cls["val_fg5"])

        # 每 epoch 诊断 | Per-epoch diagnostics
        epoch_metrics = {
            "epoch": epoch, "loss": round(avg_loss, 6),
            "miou_train": round(miou_train, 6),
            "miou_val_all": round(miou_all, 6),
            "miou_val_fg5": round(miou_fg5, 6),
        }
        # 每 5 epoch 打印 per-class IoU + pred 分布 | Every 5 epochs: per-class IoU + pred distribution
        if epoch == 1 or epoch % 5 == 0 or epoch == args.decoder_epochs:
            # pred 类别分布 | Pred class distribution
            inter_fg5, union_fg5 = per_cls["val_fg5"]
            per_cls_iou = {}
            for c in range(1, NUM_OUT_CH):
                if union_fg5[c] > 0:
                    iou_c = (inter_fg5[c] / union_fg5[c]).item()
                    per_cls_iou[c] = round(iou_c, 4)
                    epoch_metrics[f"iou_val_fg5_cls{c}"] = round(iou_c, 6)
            log("b04/diag",
                f"E{epoch:2d} pred_dist: bg={1-miou_train:.3f} "
                f"(train_mIoU={miou_train:.3f}) "
                f"val_fg5 IoU: {per_cls_iou}")
        # per-class IoU already saved from per_cls_iou above

        # 终端 + FileBackend | Console + FileBackend
        log("b04/decoder",
            f"E{epoch:2d}/{args.decoder_epochs} loss={avg_loss:.4f} "
            f"train={miou_train:.4f} val={miou_all:.4f}/{miou_fg5:.4f} "
            f"(all/FG>5%)")
        # 指标 JSONL (增量) | Metrics JSONL (append)
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(epoch_metrics) + "\n")
            mf.flush()

        if miou_fg5 > best_miou:
            best_miou = miou_fg5
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            torch.save(best_state, str(Path(args.output_dir) / "decoder_best.pt"))

    if best_state:
        decoder.load_state_dict(best_state)
    log("b04/decoder", f"Best FG>5%-mIoU={best_miou:.4f}")
    return decoder, backbone, best_miou


# ═══════════════════════════════════════════════════════════════════
# Step 2: Train FDR
# ═══════════════════════════════════════════════════════════════════

def train_fdr(args, device, log):
    """Step 2: 训练 FDR (前景密度路由器) | Step 2: Train FDR (Foreground Density Router).

    训练目标：预测每张全图上每个 tile 的 fg_ratio (不带类别标签)。
    使用 MV3 冻结 backbone → DensityHead，MSE loss。
    在无标注的全图上也可泛化 (category-agnostic)。
    Training target: predict fg_ratio per tile on each full image (no class labels).
    Uses frozen MV3 backbone → DensityHead, MSE loss.
    Generalizes to unannotated images (category-agnostic).
    """
    log("b04/fdr", f"{'='*50}")
    log("b04/fdr", f"Step 2: Train FDR ({args.fdr_epochs} epochs) | 训练 FDR")

    src_root = Path(args.src_root)
    with open(src_root/"train"/"annotations"/"instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_dir = src_root/"train"/"images"
    fdr_images = []
    for img_info in coco["images"][:300]:
        anns = img_id_to_anns.get(img_info["id"], [])
        if anns and (img_dir/img_info["file_name"]).exists():
            fdr_images.append((img_info["file_name"], str(img_dir/img_info["file_name"]), anns))

    log("b04/fdr", f"FDR training images: {len(fdr_images)}")

    fdr = FDRPredictor().to(device)
    n_p = sum(p.numel() for p in fdr.parameters() if p.requires_grad)
    log("b04/fdr", f"FDR trainable: {n_p:,}")

    opt = torch.optim.Adam(fdr.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.fdr_epochs, eta_min=1e-6)

    FDR_FEAT = TILE_SIZE // 32
    fdr_metrics_path = Path(args.output_dir) / "fdr_metrics.jsonl"

    for epoch in range(1, args.fdr_epochs + 1):
        fdr.train()
        total_loss, n = 0.0, 0
        for img_id, img_path, anns in tqdm(fdr_images, desc=f"  FDR E{epoch}", leave=False):
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
            H, W = img.shape[:2]
            mask = _render_semantic_mask(anns, H, W)

            scale = args.image_size / max(H, W)
            nH, nW = int(H*scale), int(W*scale)
            img_r = np.array(Image.fromarray(img).resize((nW,nH), Image.BILINEAR))
            mask_r = np.array(Image.fromarray(mask).resize((nW,nH), Image.NEAREST))
            ph, pw = (32-nH%32)%32, (32-nW%32)%32
            if ph>0 or pw>0:
                img_r = np.pad(img_r, ((0,ph),(0,pw),(0,0)), mode="constant")
                mask_r = np.pad(mask_r, ((0,ph),(0,pw)), mode="constant", constant_values=255)

            img_t = torch.from_numpy(img_r.astype(np.float32)/255.0)
            img_t = img_t.permute(2,0,1).unsqueeze(0).to(device)

            H2, W2 = mask_r.shape
            n_ty = (H2+TILE_SIZE-1)//TILE_SIZE; n_tx = (W2+TILE_SIZE-1)//TILE_SIZE
            # 计算每个 tile 的 GT fg_ratio | Compute GT fg_ratio per tile
            gts_list = []
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty*TILE_SIZE, min(ty*TILE_SIZE+TILE_SIZE, H2)
                    x0, x1 = tx*TILE_SIZE, min(tx*TILE_SIZE+TILE_SIZE, W2)
                    tm = mask_r[y0:y1, x0:x1]
                    gts_list.append(float((tm>0).sum())/max((y1-y0)*(x1-x0),1))
            gt_scores = torch.tensor(gts_list, dtype=torch.float32, device=device).reshape(n_ty, n_tx)

            # 前向 → 重要性图 | Forward → importance map
            imp = fdr(img_t)
            _, _, hp, wp = imp.shape
            # 将重要性图按 tile grid 池化为 tile scores | Pool importance map into tile scores by grid
            preds, gts = [], []
            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0 = ty*FDR_FEAT; y1 = min(y0+FDR_FEAT, hp)
                    x0 = tx*FDR_FEAT; x1 = min(x0+FDR_FEAT, wp)
                    if y1>y0 and x1>x0:
                        preds.append(imp[0,0,y0:y1,x0:x1].mean())
                        gts.append(gt_scores[ty,tx])
            if preds:
                # MSE loss: 预测 tile score vs GT fg_ratio
                loss = F.mse_loss(torch.stack(preds), torch.stack(gts))
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n += 1

        sch.step()
        # 每 epoch 打印 + 日志 + JSONL | Print + log + JSONL every epoch
        fdr_epoch_loss = round(total_loss/max(n, 1), 6)
        log("b04/fdr", f"FDR E{epoch:2d}/{args.fdr_epochs} loss={fdr_epoch_loss:.4f}")
        with open(fdr_metrics_path, "a") as mf:
            mf.write(json.dumps({"epoch": epoch, "loss": fdr_epoch_loss}) + "\n")
            mf.flush()

    torch.save(fdr.state_dict(), str(Path(args.output_dir) / "fdr_best.pt"))
    return fdr


# ═══════════════════════════════════════════════════════════════════
# Step 3: Dynamic Selection Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_dynamic(args, fdr, decoder, backbone, device, log):
    """Step 3: 动态瓦片选择评估 | Step 3: Dynamic Tile Selection Evaluation.

    对每张全图：
    1. FDR 预测所有 tile 的重要性分数
    2. 按 K% 选择 Top-K tile
    3. Decoder 仅处理选中 tile → 合并成完整预测图
    4. 计算 mIoU vs 全量 (100%) 基线
    Report: FG-mIoU, decoder time, speedup at each K%.

    For each full image:
    1. FDR predicts importance scores for all tiles
    2. Select Top-K% tiles
    3. Decoder processes only selected tiles → merge into full prediction map
    4. Compute mIoU vs full (100%) baseline
    """
    log("b04/eval", f"{'='*50}")
    log("b04/eval", "Step 3: Dynamic Tile Selection Evaluation | 动态瓦片选择评估")

    src_root = Path(args.src_root)
    with open(src_root/"train"/"annotations"/"instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_dir = src_root/"train"/"images"
    val_images = []
    for img_info in coco["images"][:20]:
        anns = img_id_to_anns.get(img_info["id"], [])
        if anns and (img_dir/img_info["file_name"]).exists():
            val_images.append((img_info, str(img_dir/img_info["file_name"]), anns))

    log("b04/eval", f"Test images: {len(val_images)}")

    K_LIST = [int(k) for k in args.top_k_list.split(",")]
    decoder.eval(); backbone.eval()

    results = {k: {"mious": [], "times_s": [], "n_tiles": []} for k in K_LIST}

    for img_info, img_path, anns in tqdm(val_images, desc="  Dynamic eval"):
        from PIL import Image
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_np.shape[:2]
        gt_mask = _render_semantic_mask(anns, H, W)

        # FDR 预测 tile 重要性 | FDR predicts tile importance
        tile_scores = fdr.predict_tile_scores(img_np, device)
        n_ty, n_tx = tile_scores.shape

        # 预提取所有 tile 为 tensor | Pre-extract all tiles as tensors
        all_tiles = []
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty*TILE_SIZE, min(ty*TILE_SIZE+TILE_SIZE, H)
                x0, x1 = tx*TILE_SIZE, min(tx*TILE_SIZE+TILE_SIZE, W)
                tile = img_np[y0:y1, x0:x1]
                th, tw = tile.shape[:2]
                if th < TILE_SIZE or tw < TILE_SIZE:
                    p = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    p[:th, :tw] = tile; tile = p
                tile_t = torch.from_numpy(tile.astype(np.float32)/255.0).permute(2, 0, 1)
                all_tiles.append((tile_t, y0, y1, x0, x1, th, tw))

        for K in K_LIST:
            # 按 FDR 分数选择 Top-K% tile | Select Top-K% tiles by FDR scores
            if K >= 100:
                # 100% = 所有 tile | all tiles
                sel = np.ones(n_ty * n_tx, dtype=bool)
            else:
                nk = max(1, int(n_ty * n_tx * K / 100))
                idx = np.argsort(tile_scores.flatten())[::-1][:nk]
                sel = np.zeros(n_ty * n_tx, dtype=bool); sel[idx] = True

            # 收集选中的 tile | Collect selected tiles
            selected_tensors, selected_pos = [], []
            for i, (tile_t, y0, y1, x0, x1, th, tw) in enumerate(all_tiles):
                if sel[i]:
                    selected_tensors.append(tile_t)
                    selected_pos.append((y0, y1, x0, x1, th, tw))

            pred_full = np.zeros((H, W), dtype=np.int64)
            t_dec_total = 0.0

            if selected_tensors:
                # 子批次推理：避免一次性处理全部 tile 导致 OOM | Sub-batched inference: avoid OOM from processing all tiles at once
                for sb_start in range(0, len(selected_tensors), MAX_DECODE_BATCH):
                    sb_end = min(sb_start + MAX_DECODE_BATCH, len(selected_tensors))
                    batch = torch.stack(selected_tensors[sb_start:sb_end]).to(device)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    feats = backbone(batch)
                    logits = decoder(feats["p4"], target_size=(TILE_SIZE, TILE_SIZE))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    torch.cuda.synchronize()
                    t_dec_total += time.perf_counter() - t0

                    for j in range(sb_end - sb_start):
                        y0, y1, x0, x1, th, tw = selected_pos[sb_start + j]
                        # 将 tile 预测写回全图坐标 | Write tile prediction back to full-image coordinates
                        pred_full[y0:y0+min(th, TILE_SIZE),
                                  x0:x0+min(tw, TILE_SIZE)] = preds[j][:th, :tw]

            # 计算 mIoU (仅前景类, BG 不计) | Compute mIoU (foreground classes only, exclude BG)
            miou_v, valid = 0.0, 0
            for c in range(1, NUM_CLASSES + 1):
                pc = (pred_full == c); tc = (gt_mask == c)
                inter = (pc & tc).sum(); union = (pc | tc).sum()
                if union > 0: miou_v += inter / union; valid += 1

            results[K]["mious"].append(miou_v / max(valid, 1))
            results[K]["times_s"].append(t_dec_total)
            results[K]["n_tiles"].append(len(selected_tensors))

    # Summary: Accuracy vs Compute 权衡表 | Summary: Accuracy vs Compute trade-off table
    log("b04/summary", f"  {'K%':<6} {'FG-mIoU':>9} {'ΔmIoU':>8} {'Tiles':>7} {'Time(ms)':>9} {'Speedup':>8}")
    base_miou = np.mean(results[100]["mious"])
    base_time = np.mean(results[100]["times_s"])
    base_tiles = int(np.mean(results[100]["n_tiles"]))

    log("b04/summary",
        f"  {'100%':<6} {base_miou*100:>8.2f}% {'-':>8} "
        f"{base_tiles:>7} {base_time*1000:>8.1f}ms {'1.00×':>8}")

    for K in sorted(K_LIST):
        if K == 100: continue
        miou = np.mean(results[K]["mious"])
        dt = np.mean(results[K]["times_s"])
        nt = int(np.mean(results[K]["n_tiles"]))
        dmiou = (miou - base_miou) * 100
        sp = base_time / max(dt, 1e-8)
        log("b04/summary",
            f"  {K:>4}%  {miou*100:>8.2f}% {dmiou:>+7.2f}% "
            f"{nt:>7} {dt*1000:>8.1f}ms {sp:>7.2f}×")

    return results, base_miou


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    """B-04 主入口：三阶段端到端验证 | B-04 main entry: three-stage end-to-end validation.

    阶段 | Stages:
    1. 训练 LightDecoder (FG>5% 过滤 + 稀有类过采样)
       Train LightDecoder (FG>5% filtered + rare-class oversampled)
    2. 训练 FDR (预测 tile fg_ratio, category-agnostic)
       Train FDR (predicts tile fg_ratio, category-agnostic)
    3. 动态选择评估 (Top-K% tile → Decoder → mIoU vs Speedup)
       Dynamic selection eval (Top-K% tiles → Decoder → mIoU vs Speedup)
    """
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志系统：Console + FileBackend (崩溃安全) | Logging: Console + FileBackend (crash-safe)
    logger = get_logger("b04")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir/"b04.jsonl")))

    def log(k, m):
        logger.log_info(k, m)
        print(f"  {m}")

    log("b04/start", f"B-04 Dynamic Tile Selection | device={device}")

    # Step 1: Train Decoder | 训练解码器
    decoder, backbone, dec_miou = train_decoder(args, device, log)

    # Step 2: Train FDR | 训练 FDR
    fdr = train_fdr(args, device, log)

    # Step 3: Dynamic Selection Evaluation | 动态选择评估
    results, base_miou = evaluate_dynamic(args, fdr, decoder, backbone, device, log)

    # 保存结果 | Save results
    summary = {
        "experiment": "B-04 Dynamic Tile Selection",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "decoder_val_miou": float(dec_miou),
        "per_k": {str(k): {"miou": float(np.mean(results[k]["mious"])),
                            "time_ms": float(np.mean(results[k]["times_s"])*1000)}
                  for k in [int(x) for x in args.top_k_list.split(",")]},
    }
    with open(output_dir/"results.json","w") as f:
        json.dump(summary, f, indent=2)

    log("b04/done", f"Saved: {output_dir}/ | 保存完成: {output_dir}/")


if __name__ == "__main__":
    main()
