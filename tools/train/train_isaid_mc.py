#!/usr/bin/env python3
"""
iSAID 多类别 Proto 实验 | Multi-Class Proto Experiment on iSAID
=================================================================

基于 E007-B 的双头对比 (Embedding vs Proto)，适配 iSAID 15 类分割。
E007-B based head-to-head comparison adapted for iSAID 15-class segmentation.

与 MassBuildings 实验的关键差异 | Key differences from MassBuildings:
    - 输出 | Output: 15 类 (vs 二值 | vs binary)
    - Loss: CrossEntropy (vs BCE)
    - 指标 | Metrics: mIoU + per-class IoU (vs Dice)
    - 数据 | Data: ISAIDDataset (vs MassachusettsBuildingsDataset)

日志先行 | Logging first:
    所有关键值通过 adatile.logging 输出，无 bare print()。
    All key values routed through adatile.logging, no bare print().

用法 | Usage:
    python tools/train_isaid_multiclass.py
    python tools/train_isaid_multiclass.py --n-protos 12 --epochs 30
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.logging import get_logger
from adatile.utils.seed import set_seed

logger = get_logger("isaid_mc")


def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(description="iSAID Multi-Class Proto Experiment | iSAID 多类别 Proto 实验")
    p.add_argument("--data-root", type=str, default="data/iSAID_tiles",
                   help="瓦片数据目录 | Tile data directory")
    p.add_argument("--epochs", type=int, default=30,
                   help="训练轮数 | Training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率 | Learning rate")
    p.add_argument("--embed-dim", type=int, default=128,
                   help="嵌入维度 | Embedding dimension")
    p.add_argument("--n-protos", type=int, default=12,
                   help="原型数量 | Number of prototypes")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="余弦相似度温度系数 | Temperature for cosine similarity")
    p.add_argument("--num-classes", type=int, default=15,
                   help="iSAID 类别数 | Number of iSAID classes (15)")
    p.add_argument("--output-dir", type=str, default="runs",
                   help="输出目录 | Output directory")
    p.add_argument("--name", type=str, default="isaid_mc",
                   help="实验名称 | Experiment name")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 | Device to run on")
    p.add_argument("--tile-size", type=int, default=1024,
                   help="瓦片尺寸 (iSAID 图像太大需切分) | Tile size for large iSAID images")
    p.add_argument("--max-images", type=int, default=0,
                   help="限制加载图片数 (0=全部, 调试用) | Limit number of images (0=all)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 多类别 ProtoHead | Multi-Class Proto Head
# ═══════════════════════════════════════════════════════════════════

class MultiClassProtoHead(nn.Module):
    """
    多类别 ProtoHead | Multi-Class Proto Head.

    P4 [B, 1280, H/16, W/16]
        → Conv(1280→D, 1×1) → ReLU → Embedding [B, D, H/16, W/16]
        → CosineSim(N 个可学习 Proto 向量) → sim_maps [B, N, H/16, W/16]
        → Conv(N→C, 1×1) → C-class logits [B, C, H/16, W/16]

    与二值 ProtoHead 的唯一区别：head 输出通道从 1 变为 C。
    Only difference from binary ProtoHead: head output channels from 1 to C.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128,
                 n_protos: int = 8, num_classes: int = 15):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos
        self.num_classes = num_classes

        # 特征投影 | Feature projection: 1280 → D (共享于所有 Proto 实验)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 可学习原型向量 | Learnable prototype vectors
        # 随机初始化, BCE 驱动下自组织形成语义 | Random init, self-organize via BCE
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)

        # 多类别分割头 | Multi-class segmentation head: Proto responses → C classes
        self.head = nn.Conv2d(n_protos, num_classes, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        logger.log_info("model/protohead_init",
                        f"MultiClassProtoHead: {n_params:,} params, "
                        f"embed_dim={embed_dim}, n_protos={n_protos}, "
                        f"num_classes={num_classes}")

    def forward(self, p4: torch.Tensor, temperature: float = 0.1
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward pass.

        Args:
            p4:          [B, 1280, H/16, W/16] FastSAM P4 特征
            temperature: 余弦相似度的温度系数 | temperature for cosine similarity

        Returns:
            embedding: [B, D, H, W]   低维嵌入 | Low-dim embedding
            sim_maps:  [B, N, H, W]   Proto 相似度图 | Proto similarity maps
            logit:     [B, C, H, W]   多类别 logits | Multi-class logits
        """
        # 投影 → 嵌入 | Project → embedding
        embedding = self.project(p4)  # [B, D, H/16, W/16]

        # L2 归一化后计算余弦相似度 | L2 normalize then cosine similarity
        emb_norm = F.normalize(embedding, dim=1, p=2)           # [B, D, H, W]
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)   # [N, D]
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm)
        sim_maps = sim_maps / temperature                       # [B, N, H, W]

        # 多类别 logits | Multi-class logits
        logit = self.head(sim_maps)  # [B, C, H, W]

        return embedding, sim_maps, logit


# ═══════════════════════════════════════════════════════════════════
# 多类别 EmbeddingHead (Baseline, 无 Proto 约束)
# Multi-Class Embedding Head (Baseline, no Proto constraint)
# ═══════════════════════════════════════════════════════════════════

class MultiClassEmbedHead(nn.Module):
    """
    嵌入头 (Baseline) | Embedding Head (Baseline).

    P4 → Conv(1280→D)→ReLU → Conv(D→C) → C-class logits.
    无 Proto 结构约束, 嵌入直接映射到类别 logits。
    No Proto constraint; embedding maps directly to class logits.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128,
                 num_classes: int = 15):
        super().__init__()
        self.embed_dim = embed_dim

        # 特征投影 (与 ProtoHead 相同) | Feature projection (same as ProtoHead)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 直接分类头 | Direct classification head (no Proto bottleneck)
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        logger.log_info("model/embedhead_init",
                        f"MultiClassEmbedHead: {n_params:,} params, "
                        f"embed_dim={embed_dim}, num_classes={num_classes}")

    def forward(self, p4: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p4: [B, 1280, H/16, W/16]

        Returns:
            embedding: [B, D, H, W]  低维嵌入 | Low-dim embedding
            logit:     [B, C, H, W]  多类别 logits | Multi-class logits
        """
        embedding = self.project(p4)  # [B, D, H, W]
        logit = self.head(embedding)  # [B, C, H, W]
        return embedding, logit


# ═══════════════════════════════════════════════════════════════════
# 指标 | Metrics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_miou(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int) -> dict:
    """
    计算 mIoU + per-class IoU | Compute mIoU + per-class IoU.

    Args:
        pred:   [B, H, W] 预测标签 | predicted class labels
                or [B, C, H, W] logits (会取 argmax)
        target: [B, H, W] GT 语义标签 | ground truth semantic labels

    Returns:
        {"miou": float, "iou_cls_0": float, ...}

    注意 | Note:
        使用 sum+reshape 避免 v1 的 unsqueeze(0) 广播爆炸问题。
        Uses sum+reshape to avoid v1's unsqueeze(0) broadcast bug.
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)  # [B, C, H, W] → [B, H, W]

    results = {}
    ious = []
    for c in range(num_classes):
        # 逐类别计算 IoU | Per-class IoU
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        iou = (inter + 1e-8) / (union + 1e-8)
        ious.append(iou.item())
        results[f"iou_cls_{c}"] = iou.item()

    miou = float(np.mean(ious))
    results["miou"] = miou
    return results


# ═══════════════════════════════════════════════════════════════════
# 训练 | Training
# ═══════════════════════════════════════════════════════════════════

def train_head(head: nn.Module, backbone: nn.Module,
               train_ds, val_ds, args, device: str,
               recorder: ExperimentRecorder,
               head_name: str, is_proto: bool = False) -> float:
    """
    训练一个 Head 变体 | Train one head variant.

    固定 Backbone (frozen), 只训练 Head。
    Frozen backbone, train head only.

    Args:
        head:      MultiClassProtoHead or MultiClassEmbedHead
        head_name: 用于日志标记 | used as log tag
        is_proto:  True=ProtoHead (需 temperature), False=EmbedHead

    Returns:
        best_miou: 最佳验证 mIoU | best validation mIoU
    """
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou: float = 0.0
    best_state: dict | None = None
    best_epoch: int = 0

    logger.log_info("train/start",
                    f"[{head_name}] {args.epochs} epochs, lr={args.lr}, "
                    f"CosineLR, num_classes={args.num_classes}")

    for epoch in range(1, args.epochs + 1):
        # ── 训练阶段 | Training phase ──
        head.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(range(len(train_ds)),
                    desc=f"  [{head_name}] Epoch {epoch}/{args.epochs}",
                    leave=False)
        for idx in pbar:
            # 逐样本加载 (few-shot 风格, 不 shuffle) | Per-sample loading (few-shot style, no shuffle)
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
            target = sample["mask"].unsqueeze(0).to(device)   # [1, H, W] semantic

            # 冻结 Backbone 提取特征 | Frozen backbone feature extraction
            with torch.no_grad():
                features = backbone(image)
            p4 = features["p4"]

            # 前向传播 | Forward pass
            if is_proto:
                _, _, logit = head(p4, temperature=args.temperature)
            else:
                _, logit = head(p4)

            # 上采样 logit 到 GT 分辨率 | Upsample logit to GT resolution
            logit_up = F.interpolate(logit, size=target.shape[1:],
                                     mode="bilinear", align_corners=False)

            # 多类别交叉熵 (ignore_index=255 排除无效像素) | Multi-class CE (ignore_index=255 for invalid pixels)
            loss = F.cross_entropy(logit_up, target, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 记录训练 loss | Log training loss
        logger.log_loss(f"ce/{head_name.lower()}", avg_loss, step=epoch)

        # ── 验证阶段 | Validation phase ──
        head.eval()
        mious: list[float] = []

        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                target = sample["mask"].unsqueeze(0).to(device)  # [1, H, W] semantic

                # 冻结 Backbone 提取特征 | Frozen backbone feature extraction
                features = backbone(image)
                p4 = features["p4"]

                if is_proto:
                    _, _, logit = head(p4, temperature=args.temperature)
                else:
                    _, logit = head(p4)

                # 上采样 + 取 argmax 预测 | Upsample + argmax to predict
                logit_up = F.interpolate(logit, size=target.shape[1:],
                                         mode="bilinear", align_corners=False)
                pred = logit_up.argmax(dim=1)  # [1, H, W], 类别索引 | class indices

                metrics = compute_miou(pred, target, args.num_classes)
                mious.append(metrics["miou"])

        miou_mean = float(np.mean(mious))
        is_best = miou_mean > best_miou

        if is_best:
            best_miou = miou_mean
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_epoch = epoch

        # 日志记录 | Log metrics
        marker = " *" if is_best else ""
        logger.log_metric(f"miou/{head_name.lower()}", miou_mean, step=epoch,
                          tags=["val", head_name])
        logger.log_info("train/epoch",
                        f"[{head_name}] Epoch {epoch:2d}/{args.epochs}  "
                        f"loss={avg_loss:.4f}  mIoU={miou_mean:.4f}{marker}")

        # 同时记录到 ExperimentRecorder | Also record to ExperimentRecorder
        recorder.record_metric(f"loss/train", avg_loss, step=epoch,
                               phase="train", tags=[head_name])
        recorder.record_metric(f"miou/val", miou_mean, step=epoch,
                               phase="val", tags=[head_name])

    # 恢复最佳权重 | Restore best weights
    if best_state is not None:
        head.load_state_dict(best_state)

    logger.log_info("train/best",
                    f"[{head_name}] Best mIoU: {best_miou:.4f} "
                    f"(epoch {best_epoch}/{args.epochs})")
    return best_miou


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    """iSAID 多类别 Proto 实验主入口 | iSAID multi-class Proto experiment main entry.

    三阶段流水线 | Three-stage pipeline:
    [1] 加载冻结 FastSAM Backbone | Load frozen FastSAM backbone
    [2] 加载 iSAID tile 数据集 (语义模式) | Load iSAID tile dataset (semantic mode)
    [3] 训练 MultiClassProtoHead | Train MultiClassProtoHead
    """
    args = parse_args()
    device = args.device

    # 记录实验启动配置 | Log experiment start config
    logger.log_info("exp/start",
                    f"iSAID Multi-Class Proto Experiment | "
                    f"N_protos={args.n_protos}, D={args.embed_dim}, "
                    f"C={args.num_classes}, Epochs={args.epochs}")
    logger.log_info("exp/config",
                    f"data_root={args.data_root}, lr={args.lr}, "
                    f"seed={args.seed}, device={device}")

    # 固定随机种子 (可复现性) | Fix random seed (reproducibility)
    set_seed(args.seed)

    # ── 实验管理 | Experiment management ──
    exp_id = generate_exp_id(name=args.name)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="iSAID", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)
    logger.log_info("exp/output", f"Results: {output_path} | 结果目录: {output_path}")

    # ── [1] Backbone (冻结) | Frozen Backbone ──
    # FastSAM backbone 保持 eval mode，只做特征提取，不参与梯度更新
    # FastSAM backbone stays in eval mode, feature extraction only, no gradient updates
    logger.log_info("phase", "[1/3] Loading frozen FastSAM backbone | 加载冻结 backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    logger.log_info("model/backbone", "FastSAM backbone loaded (frozen, eval mode) | backbone 已加载 (冻结, eval 模式)")

    # ── [2] 数据 | Data ──
    # 使用预切 tile 数据集 (semantic=True: mask 为语义标签而非实例)
    # Uses pre-cut tile dataset (semantic=True: mask is semantic label, not instance)
    logger.log_info("phase", "[2/3] Loading iSAID data (semantic mode) | 加载 iSAID 数据 (语义模式)")
    train_ds = FastISAIDTileDataset(root_dir=args.data_root, split="train", semantic=True)
    val_ds = FastISAIDTileDataset(root_dir=args.data_root, split="val", semantic=True)

    # 限制 tile 数 (调试/快速验证) | Limit tiles (debug/quick validation)
    if args.max_images > 0:
        train_ds._tiles = train_ds._tiles[:args.max_images]
        val_ds._tiles = val_ds._tiles[:max(1, args.max_images // 4)]

    logger.log_info("data/isaid",
                    f"Train: {len(train_ds)} tiles, "
                    f"Val: {len(val_ds)} tiles, "
                    f"Classes: {args.num_classes}, "
                    f"Mode: semantic, pre-cut (~20ms/sample)" +
                    (f" [LIMITED]" if args.max_images > 0 else ""))

    # ── [3] 训练 ProtoHead | Train ProtoHead ──
    logger.log_info("phase", "[3/3] Training MultiClassProtoHead | 训练多类别 ProtoHead")

    # 多类别 ProtoHead (主要实验变体) | Multi-class Proto Head (main experiment variant)
    proto_head = MultiClassProtoHead(
        in_channels=1280, embed_dim=args.embed_dim,
        n_protos=args.n_protos, num_classes=args.num_classes
    ).to(device)

    best_miou = train_head(
        proto_head, backbone, train_ds, val_ds,
        args, device, recorder,
        head_name="Proto", is_proto=True
    )

    # ── 记录最终结果 | Record final result ──
    recorder.record_metric("best_miou", best_miou,
                           phase="val", tags=["isaid", "mc", "summary"])
    logger.log_metric("best_miou", best_miou, tags=["isaid", "summary"])

    # 保存模型检查点 | Save model checkpoint
    ckpt_path = output_path / "multiclass_proto_head.pt"
    torch.save(proto_head.state_dict(), ckpt_path)
    logger.log_info("model/saved", f"Checkpoint: {ckpt_path} | 检查点: {ckpt_path}")

    recorder.finalize()
    recorder.close()

    logger.log_info("exp/done",
                    f"Best mIoU: {best_miou:.4f} | Results: {output_path}")


if __name__ == "__main__":
    main()
