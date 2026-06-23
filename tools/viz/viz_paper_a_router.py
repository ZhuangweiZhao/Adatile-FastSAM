#!/usr/bin/env python3
"""
E009-C: Router Visualization — Router 学会了什么路由策略？
===========================================================
E009-C: Router Visualization — What routing strategy did the Router learn?

核心问题 | Core question:
    conv3x3 Router 在不同的空间区域（建筑内部/边缘/背景）
    分别选择哪些 Proto？路由是否具有空间语义？
    In different spatial regions (building interior/edge/background),
    which Protos does the conv3x3 Router select? Does routing have spatial semantics?

验证假设 | Hypotheses:
    H1: 建筑内部 → 特定 Proto 组合 (Building Body protos)
        Interior regions → specific Proto combinations (Building Body protos)
    H2: 建筑边缘 → 不同 Proto 组合 (Edge/Detail protos)
        Edge regions → different Proto combinations (Edge/Detail protos)
    H3: 背景     → 抑制 Proto 组合 (Background protos)
        Background  → suppressed Proto combinations (Background protos)
    H4: 路由具有空间连续性（相邻像素路由相似）
        Routing has spatial continuity (adjacent pixels have similar routing)

与 E007.5 的关系 | Relation to E007.5:
    E007.5: 每个 Proto 看什么区域？（Proto → Region）
             What region does each Proto look at? (Proto → Region)
    E009-C: 每个区域用什么 Proto？（Region → Proto）
             What Protos are used in each region? (Region → Proto)
    两者互补，形成完整的可解释性论证。
    The two are complementary, forming a complete interpretability argument.

用法 | Usage::
    python tools/viz_e009_router.py --checkpoint runs/.../spm_head_s2.pt

输出文件 | Output files:
    router_viz_img{i}.png      — 单图可视化 | Per-image visualization (12-panel grid)
    region_proto_selection.png — 跨图像区域Proto选择汇总 | Cross-image region Proto summary
"""

# ═══════════════════════════════════════════════════════════════════
# 导入模块 | Imports
# ═══════════════════════════════════════════════════════════════════

import sys, argparse, glob as _glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免GUI依赖 | Non-interactive backend, avoids GUI dependency
import matplotlib.pyplot as plt
from scipy.ndimage import sobel  # Sobel 算子用于边缘检测 | Sobel operator for edge detection
from collections import Counter  # 统计 Proto 组合频率 | Count Proto combination frequencies

# ═══════════════════════════════════════════════════════════════════
# 项目路径设置 | Project path setup
# ═══════════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 上溯两级到项目根目录 | Go up 2 levels to project root
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
# 从 E009 实验脚本导入 ProtoHead 和 SPMHead | Import ProtoHead and SPMHead from E009 experiment script
from eval_e009_spm_router import ProtoHead, SPMHead


def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="E009-C Router Visualization — 可视化Router的空间路由决策 | "
                    "Visualize Router's spatial routing decisions"
    )
    # 模型检查点路径（支持通配符） | Model checkpoint path (supports glob wildcards)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="检查点路径，支持 * 通配符 | Checkpoint path, supports * wildcard")
    # 数据集根目录 | Dataset root directory
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings",
                   help="Massachusetts Buildings 数据集根目录 | Dataset root directory")
    # Proto 数量 | Number of prototypes
    p.add_argument("--n-protos", type=int, default=8,
                   help="Proto 总数量 | Total number of prototypes")
    # 嵌入维度 | Embedding dimension
    p.add_argument("--embed-dim", type=int, default=128,
                   help="Proto 嵌入向量维度 | Proto embedding vector dimension")
    # Router 选择的 Top-K 数量 | Number of Top-K protos selected by Router
    p.add_argument("--router-k", type=int, default=4,
                   help="Router 每个像素选择的 Proto 数量 | Number of Protos Router selects per pixel")
    # 可视化图像数量 | Number of images to visualize
    p.add_argument("--n-images", type=int, default=4,
                   help="可视化的测试图像数量 | Number of test images to visualize")
    # 数据集划分 | Dataset split
    p.add_argument("--split", type=str, default="test",
                   help="数据集划分 (test 图像更多) | Dataset split to use (test has more images)")
    # 输出目录 | Output directory
    p.add_argument("--output-dir", type=str, default="runs/router_viz",
                   help="可视化结果保存目录 | Directory to save visualization results")
    # 运行设备 | Compute device
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 (cuda/cpu) | Compute device (cuda/cpu)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def compute_edge(mask: np.ndarray) -> np.ndarray:
    """
    Sobel 边缘检测 | Sobel edge detection.

    计算二值掩码的边缘图 | Compute edge map from binary mask.

    原理 | Principle:
        Sobel 算子分别在水平和垂直方向计算梯度，然后合成梯度幅值:
        Sobel operator computes gradients in horizontal and vertical directions,
        then combines them into gradient magnitude:

            gy = ∂I/∂y (垂直梯度 | vertical gradient)
            gx = ∂I/∂x (水平梯度 | horizontal gradient)
            edge = sqrt(gx² + gy²)  →  梯度幅值 | gradient magnitude

    阈值 | Threshold:
        0.1: 低阈值，捕获所有从 0→1 的掩码边界。
        0.1: A low threshold that captures all mask boundaries from 0→1.
        对于二值掩码，任何边界像素的梯度都会 >= 1.0（8邻域），
        所以 0.1 足以检测所有边缘。
        For binary masks, any boundary pixel has gradient >= 1.0 (8-connectivity),
        so 0.1 is sufficient to detect all edges.

    :param mask: 二值分割掩码 [H, W] | Binary segmentation mask [H, W]
    :type mask: np.ndarray

    :return: 二值边缘图 [H, W] dtype=uint8 | Binary edge map [H, W] dtype=uint8
    :rtype: np.ndarray
    """
    # 水平方向梯度 | Gradient along vertical axis (changes along rows)
    gy = sobel(mask.astype(float), axis=0)
    # 垂直方向梯度 | Gradient along horizontal axis (changes along columns)
    gx = sobel(mask.astype(float), axis=1)
    # 梯度幅值 = L2范数 | Gradient magnitude = L2 norm of (gx, gy)
    edge = np.sqrt(gx**2 + gy**2)
    # 二值化：梯度 > 0.1 视为边缘 | Binarize: gradient > 0.1 classified as edge
    return (edge > 0.1).astype(np.uint8)


def region_mask(gt: np.ndarray) -> dict:
    """
    三分区掩码分解 | Three-region mask decomposition.

    将 GT 掩码分解为三个互斥的空间区域 | Decompose GT mask into three mutually
    exclusive spatial regions:

        ┌─────────────────────────────────────────────────────────────┐
        │ interior (建筑内部):  GT=1 AND edge=0                       │
        │   Building interior pixels — 远离边界的建筑体区域          │
        │   Far from object boundaries, represents building body area │
        │                                                             │
        │ edge (建筑边缘):      GT=1 AND edge=1                       │
        │   Building edge pixels — 建筑轮廓/边界过渡区域              │
        │   Boundary transition zone, detail-rich regions             │
        │                                                             │
        │ bg (背景):            GT=0                                  │
        │   Background pixels — 非建筑区域                             │
        │   Non-building regions                                      │
        └─────────────────────────────────────────────────────────────┘

    空间语义分离的意义 | Purpose of spatial semantic separation:
        如果 Router 学会了对不同区域选择不同的 Proto 组合，说明它获得了
        空间感知能力（而非简单的逐像素模式匹配）。
        If Router selects different Proto combinations for different regions,
        it demonstrates spatial awareness (not just per-pixel pattern matching).

    :param gt: 二值 GT 掩码 [H, W] | Binary ground-truth mask [H, W]
    :type gt: np.ndarray

    :return: (regions_dict, edge_map): tuple of - regions_dict: 三个布尔掩码的字典 | Dict of three boolean masks - edge_map:     二值边缘图 [H, W] | Binary edge map [H, W]
    :rtype: dict
    """
    # 计算 GT 的边缘图 | Compute edge map from GT mask
    edge_map = compute_edge(gt)

    # 三分区定义 | Three-region definitions:
    # interior: 建筑像素 AND 非边缘 | building pixels AND not on edge
    # edge:     建筑像素 AND 边缘上  | building pixels AND on edge
    # bg:       非建筑像素           | non-building pixels
    return {
        "interior": (gt == 1) & (edge_map == 0),
        "edge":     (gt == 1) & (edge_map == 1),
        "bg":       gt == 0,
    }, edge_map


@torch.no_grad()
def analyze_routing(spm_head, backbone, dataset, device, args, output_dir):
    """
    可视化 Router 的空间路由决策 | Visualize Router's spatial routing decisions.

    核心分析流程 | Core analysis pipeline:
        1. 选择包含建筑的测试图像 | Select test images with buildings (>3% FG)
        2. 对每张图像执行完整推理 | Run full inference on each image
        3. 提取 Router 的 Top-K Proto 选择 | Extract Router's Top-K Proto selection per pixel
        4. 将像素按区域分解为 Interior/Edge/Background | Decompose pixels into three regions
        5. 统计每个区域对不同 Proto 的偏好 | Count per-region Proto selection frequencies
        6. 生成 12 面板可视化 + 跨图像汇总图 | Generate 12-panel visualization + cross-image summary

    :param spm_head: SPMHead 模块（含 ProtoHead + Router） | SPMHead module (ProtoHead + Router)

    :param backbone: FastSAM 特征提取骨干 | FastSAM feature extraction backbone

    :param dataset: Massachusetts Buildings 数据集 | Massachusetts Buildings dataset

    :param device: 计算设备 | Compute device

    :param args: 命令行参数 | Command-line arguments

    :param output_dir: 输出目录 | Output directory for saving figures

    :return: (per_region, region_pixel_counts): tuple of - per_region:           每个区域的 Proto 选择次数 | Per-region Proto selection counts - region_pixel_counts:  每个区域的总像素数 | Per-region total pixel counts
    """
    # 设置为评估模式 | Set to evaluation mode
    spm_head.eval()

    # 创建输出目录 | Create output directory
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    k = args.router_k         # Router 每像素选择的 Proto 数量 | Proto count per pixel
    n_protos = args.n_protos  # 总 Proto 数量 | Total Proto count

    # ═══════════════════════════════════════════════════════════════
    # 统计累加器初始化 | Statistics Accumulator Initialization
    # ═══════════════════════════════════════════════════════════════
    # per_region[region_name][proto_idx] = 该 Proto 在该区域被选中的像素数
    # per_region[region_name][proto_idx] = number of pixels where this proto was selected in this region
    per_region = {r: np.zeros(n_protos) for r in ["interior", "edge", "bg"]}
    # 每个区域的像素总数（用于归一化）| Total pixel count per region (for normalization)
    region_pixel_counts = {r: 0 for r in ["interior", "edge", "bg"]}

    # Proto 组合计数器：记录 (区域, Proto组合) → 出现次数
    # Proto combination counter: records (region, Proto combo) → frequency
    # 用于发现每个区域最常用的 Proto 组合模式 | Used to discover dominant combo patterns per region
    combo_counter = Counter()

    # ═══════════════════════════════════════════════════════════════
    # 图像选择：过滤建筑占比 > 3% 的图像 | Image Selection: filter images with >3% building coverage
    # ═══════════════════════════════════════════════════════════════
    # 阈值 3%：仅选择有足够建筑区域进行有意义分析的图像
    # Threshold 3%: only select images with enough building area for meaningful analysis
    img_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["masks"]
        # 确保 mask 为 2D [H, W] | Ensure mask is 2D [H, W]
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        building_pct = mask.mean().item()  # 建筑像素占比 | Building pixel ratio
        if building_pct > 0.03:  # 建筑占比 > 3% | Building coverage > 3%
            img_indices.append(i)
        if len(img_indices) >= args.n_images:
            break

    print(f"  Selected {len(img_indices)} images from {args.split} split")
    print(f"  Analyzing router decisions (K={k}/{n_protos})...")

    # ═══════════════════════════════════════════════════════════════
    # 逐图像分析循环 | Per-Image Analysis Loop
    # ═══════════════════════════════════════════════════════════════
    for vis_idx, ds_idx in enumerate(tqdm(img_indices, desc="  Router viz")):
        # ── 1. 数据加载 | Data Loading ──
        sample = dataset[ds_idx]
        image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        gt_mask = sample["masks"]
        # 统一 mask 形状为 [H, W] | Normalize mask shape to [H, W]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        # ── 2. 特征提取 | Feature Extraction ──
        features = backbone(image)
        p4 = features["p4"]  # P4 特征 [1, 1280, H/16, W/16] | P4 features at stride 16

        # ── 3. 双前向传播 | Dual Forward Passes ──
        # (a) 完整前向：获取分割预测和相似度图（Decoder 使用所有 Proto）
        #     Full forward: get segmentation prediction and similarity maps (Decoder uses all Protos)
        logit_full, sim_maps, embedding = spm_head.forward_full(p4)
        # (b) 路由前向：获取 Router 的 per-pixel logits（用于分析路由决策）
        #     Routed forward: get Router's per-pixel logits (for analyzing routing decisions)
        _, _, _, router_logits = spm_head.forward_routed(p4, mode="learned", k=k)
        # router_logits: [1, N_protos, H/16, W/16] — 每个像素对每个 Proto 的路由分数
        # router_logits: [1, N_protos, H/16, W/16] — routing score per pixel per Proto

        # ── 4. Router Top-K 提取 | Router Top-K Extraction ──
        # 将 router_logits 展平并在每个像素取 Top-K Proto 索引
        # Flatten router_logits and take Top-K Proto indices per pixel
        router_flat = router_logits.squeeze(0).reshape(n_protos, -1)  # [N, H/16*W/16]
        _, topk_idx = router_flat.topk(k, dim=0)  # [K, H/16*W/16] — 每个像素的 Top-K Proto ID
        topk_idx = topk_idx.cpu().numpy()         # 转为 NumPy | Convert to NumPy

        # ── 5. 路由掩码上采样到 GT 分辨率 | Upsample Routing Masks to GT Resolution ──
        # 路由在特征分辨率 (H/16, W/16) 计算，需要上采样到原始图像分辨率进行逐像素分析
        # Routing is computed at feature resolution (H/16, W/16), need to upsample
        # to original image resolution for per-pixel analysis
        H_emb, W_emb = router_logits.shape[2], router_logits.shape[3]  # 特征图空间尺寸 | Feature map spatial dims
        H_gt, W_gt = int(gt_mask.shape[0]), int(gt_mask.shape[1])       # GT 原始尺寸 | GT original size

        # 对每个 Proto，创建二值路由掩码（该 Proto 是否在 Top-K）并最近邻上采样
        # For each Proto, create binary routing mask (is this Proto in Top-K) and nearest-neighbor upsample
        routing_up = np.zeros((n_protos, H_gt, W_gt), dtype=bool)
        for p in range(n_protos):
            # 检查 Proto p 在哪些特征像素的 Top-K 中 | Check which feature pixels have Proto p in Top-K
            mask_p = (topk_idx == p).any(axis=0).reshape(H_emb, W_emb)  # [H/16, W/16]
            mask_p_t = torch.from_numpy(mask_p).float().unsqueeze(0).unsqueeze(0).to(device)
            # 最近邻上采样保持二值特性 | Nearest-neighbor upsample preserves binary nature
            mask_up = F.interpolate(mask_p_t, size=(H_gt, W_gt), mode="nearest")
            routing_up[p] = mask_up.squeeze().cpu().numpy() > 0.5

        # ── 6. 区域分解 | Region Decomposition ──
        # 将 GT mask 分解为 interior / edge / bg 三个区域（在 GT 分辨率）
        # Decompose GT mask into interior / edge / bg regions (at GT resolution)
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        regions, edge_np = region_mask(gt_np)

        # ── 7. GT 降采样到特征分辨率用于组合统计 | GT Downsampling for Combo Statistics ──
        # 路由组合统计在特征分辨率进行（避免上采样引入的伪影）
        # Routing combo statistics are computed at feature resolution (avoids upsampling artifacts)
        gt_down = F.interpolate(
            torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_emb, W_emb), mode="nearest"
        ).squeeze().cpu().numpy() > 0.5
        edge_down = compute_edge(gt_down.astype(np.uint8))
        # 在特征分辨率的三区域定义 | Three-region definitions at feature resolution
        regions_down = {
            "interior": (gt_down == 1) & (edge_down == 0),
            "edge":     (gt_down == 1) & (edge_down == 1),
            "bg":       gt_down == 0,
        }

        # ── 8. 区域 Proto 统计 | Per-Region Proto Statistics ──
        for region_name, region_m in regions_down.items():
            if region_m.sum() == 0:
                continue  # 跳过空区域 | Skip empty regions
            region_pixel_counts[region_name] += region_m.sum()
            # 统计每个 Proto 在该区域被选中的像素数 | Count pixels where each Proto is selected
            for p in range(n_protos):
                proto_present = (topk_idx == p).any(axis=0)  # [H/16*W/16] bool
                per_region[region_name][p] += proto_present[region_m.flatten()].sum()

            # 统计该区域内的 Proto 组合 | Count Proto combinations in this region
            # 每个像素的组合 = Top-K Proto IDs 排序后的元组
            # Per-pixel combo = sorted tuple of Top-K Proto IDs
            for pixel_i in np.where(region_m.flatten())[0]:
                combo = tuple(sorted(topk_idx[:, pixel_i].tolist()))
                combo_counter[(region_name, combo)] += 1

        # ═══════════════════════════════════════════════════════════
        # 9. 单图可视化 (12 面板) | Per-Image Visualization (12 panels)
        # ═══════════════════════════════════════════════════════════
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img_np = np.clip(img_np, 0, 1)  # 限制像素值范围 | Clamp pixel values to [0,1]

        # 分割预测：将 logit 上采样到 GT 分辨率并二值化
        # Segmentation prediction: upsample logit to GT resolution and binarize
        logit_up = F.interpolate(logit_full, size=(H_gt, W_gt),
                                 mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze().cpu().numpy()

        # ── 创建 3x4 面板图 | Create 3×4 panel figure ──
        fig = plt.figure(figsize=(20, 14))

        # --- Row 1 (panels 1-4): 输入与基准 | Input & Baselines ---
        # Panel 1: 原始 RGB 图像 | Original RGB image
        ax = plt.subplot(3, 4, 1)
        ax.imshow(img_np); ax.set_title("Image\n图像", fontsize=9); ax.axis("off")

        # Panel 2: GT 二值掩码 | Binary ground-truth mask
        ax = plt.subplot(3, 4, 2)
        ax.imshow(gt_np, cmap="gray"); ax.set_title("GT Mask\nGT掩码", fontsize=9); ax.axis("off")

        # Panel 3: 模型分割预测（使用全部 Proto）| Model prediction (all Protos)
        ax = plt.subplot(3, 4, 3)
        ax.imshow(pred, cmap="gray"); ax.set_title("Prediction (Full)\n预测(全量)", fontsize=9); ax.axis("off")

        # Panel 4: GT 边缘图（Sobel 检测）| GT edge map (Sobel detection)
        ax = plt.subplot(3, 4, 4)
        ax.imshow(edge_np, cmap="hot"); ax.set_title("GT Edges\nGT边缘", fontsize=9); ax.axis("off")

        # --- Row 2 (panels 5-6): 路由空间模式 | Routing Spatial Patterns ---
        # Panel 5: 每个像素的 dominant Proto（被选中的 Proto 中 argmax 路由分数）
        # Panel 5: Dominant Proto per pixel (argmax routing score among selected Protos)
        # 对每个像素，找出所有选中 Proto 在路由输出中得分最高的那个
        # For each pixel, find the Proto with highest routing score among all selected Protos
        dominant = np.argmax(
            np.array([routing_up[p].astype(float) for p in range(n_protos)]), axis=0
        )
        ax = plt.subplot(3, 4, 5)
        ax.imshow(img_np)
        im = ax.imshow(dominant, alpha=0.5, cmap="tab10", vmin=0, vmax=9)
        ax.set_title("Dominant Proto per Pixel\n(color = proto index)\n主导Proto(颜色=Proto序号)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, ticks=range(n_protos))

        # Panel 6: 路由组合哈希图 — 相同 Proto 组合的像素用相同颜色
        # Panel 6: Routing combination hash map — same color = same Proto combo
        # 将每个像素的 Top-K Proto 组合哈希为整数 | Hash each pixel's Top-K Proto combo to integer
        combo_hash = np.zeros((H_emb, W_emb), dtype=int)
        for h in range(H_emb):
            for w in range(W_emb):
                combo_hash[h, w] = hash(tuple(sorted(topk_idx[:, h * W_emb + w].tolist()))) % 1000
        # 上采样到 GT 分辨率 | Upsample to GT resolution
        combo_hash_up = F.interpolate(
            torch.from_numpy(combo_hash).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_gt, W_gt), mode="nearest"
        ).squeeze().cpu().numpy()

        ax = plt.subplot(3, 4, 6)
        ax.imshow(img_np)
        ax.imshow(combo_hash_up, alpha=0.4, cmap="tab20")
        ax.set_title(f"Routing Combination Map\n路由组合图"
                     f"(unique combos = {len(set(combo_hash.flatten()))})\n"
                     f"(唯一组合数={len(set(combo_hash.flatten()))})", fontsize=9)
        ax.axis("off")

        # --- Row 3 (panels 7-9): 每区域 Proto 激活率 | Per-Region Proto Activation Rate ---
        # 统计单图内每个区域的 Proto 激活频率 | Compute per-region Proto activation frequency for this image
        img_per_region = {r: np.zeros(n_protos) for r in ["interior", "edge", "bg"]}
        for r in img_per_region:
            if regions[r].sum() > 0:
                for p in range(n_protos):
                    # 区域 r 中 Proto p 被选中的像素比例 | Fraction of region r pixels where Proto p is selected
                    img_per_region[r][p] = routing_up[p][regions[r]].mean()

        colors_bar = plt.cm.tab10(np.linspace(0, 1, n_protos))
        for i, (r_name, r_label) in enumerate([("interior", "Interior\n建筑内部"),
                                                ("edge", "Edge\n建筑边缘"),
                                                ("bg", "Background\n背景")]):
            ax = plt.subplot(3, 4, 7 + i)
            ax.bar(range(n_protos), img_per_region[r_name], color=colors_bar)
            ax.set_title(f"{r_label} — Proto Activation Rate\n区域Proto激活率", fontsize=9)
            ax.set_xlabel("Proto", fontsize=8)
            ax.set_ylabel("Activation Rate\n激活率", fontsize=8)
            ax.set_xticks(range(n_protos))
            ax.set_xticklabels([f"P{p}" for p in range(n_protos)], fontsize=7)
            ax.set_ylim(0, 1)  # 激活率范围 [0, 1] | Activation rate range [0, 1]

        # --- Panel 10: 空间连贯性与路由统计 | Spatial Coherence & Routing Statistics ---
        # 计算相邻像素共享 Proto 选择的 Jaccard 相似度
        # Compute Jaccard similarity of Proto selection between adjacent pixels
        # 高相似度 = 路由在空间上保持一致 | High similarity = routing is spatially consistent
        routing_onehot = np.stack([routing_up[p].astype(float) for p in range(n_protos)], axis=0)
        # 水平方向连贯性 | Horizontal coherence: compare pixel (h,w) with (h,w+1)
        jaccard_h = (routing_onehot[:, :-1, :] * routing_onehot[:, 1:, :]).sum(axis=0)
        union_h = (routing_onehot[:, :-1, :] + routing_onehot[:, 1:, :]).clip(0, 1).sum(axis=0)
        coherence_h = (jaccard_h / (union_h + 1e-8)).mean()

        # 垂直方向连贯性 | Vertical coherence: compare pixel (h,w) with (h+1,w)
        jaccard_w = (routing_onehot[:, :, :-1] * routing_onehot[:, :, 1:]).sum(axis=0)
        union_w = (routing_onehot[:, :, :-1] + routing_onehot[:, :, 1:]).clip(0, 1).sum(axis=0)
        coherence_w = (jaccard_w / (union_w + 1e-8)).mean()

        ax = plt.subplot(3, 4, 10)
        ax.text(0.5, 0.5, f"Spatial Coherence\n空间连贯性\n\n"
                f"Horizontal (水平): {coherence_h:.3f}\n"
                f"Vertical   (垂直): {coherence_w:.3f}\n\n"
                f"Top Region Combos:\n区域Top组合:\n"
                + "\n".join(f"  {r}: {tuple(c)} ({cnt}x)"
                           for (r, c), cnt in combo_counter.most_common(6)),
                transform=ax.transAxes, fontsize=8, va="center", ha="center",
                family="monospace")
        ax.set_title("Routing Statistics\n路由统计", fontsize=9)
        ax.axis("off")

        # --- Panels 11-12: Learned vs Fixed 路由对比 | Learned vs Fixed Routing Comparison ---
        # 固定路由：按 |w * sim| 绝对值排序 Top-K（无学习，纯相似度加权）
        # Fixed routing: Top-K by |w * sim| absolute value (no learning, pure similarity weighting)

        # 获取 learned routing 的 Top-K（复用之前的 router_logits）
        # Get learned routing Top-K (reuse router_logits from earlier)
        router_flat_l = router_logits.squeeze(0).reshape(n_protos, -1)
        _, topk_l = router_flat_l.topk(k, dim=0)

        # 计算 fixed routing 的 Top-K：|head_weight * similarity_map|
        # Compute fixed routing Top-K: |head_weight * similarity_map|
        head_w = spm_head.proto_head.head.weight.squeeze()  # Proto 分类权重 | Proto classification weights
        sim_flat = sim_maps.squeeze(0).reshape(n_protos, -1)  # 相似度图展平 | Flattened similarity map
        importance = (sim_flat * head_w.unsqueeze(1)).abs()   # |w_i * sim_i| 重要性分数
        _, topk_f = importance.topk(k, dim=0)  # Fixed Top-K

        # 计算 learned 和 fixed 之间的一致率（Top-K 交集 / K）
        # Compute agreement rate between learned and fixed (intersection / K)
        agree_mask = np.zeros(H_emb * W_emb, dtype=float)
        for pi in range(H_emb * W_emb):
            agree_mask[pi] = len(set(topk_l[:, pi].cpu().numpy()) &
                                 set(topk_f[:, pi].cpu().numpy())) / k
        agree_map = agree_mask.reshape(H_emb, W_emb)
        # 上采样一致率图 | Upsample agreement map
        agree_up = F.interpolate(
            torch.from_numpy(agree_map).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_gt, W_gt), mode="nearest"
        ).squeeze().cpu().numpy()

        # Panel 11: Learned vs Fixed 一致率热力图 | Agreement heatmap
        ax = plt.subplot(3, 4, 11)
        ax.imshow(img_np)
        im = ax.imshow(agree_up, alpha=0.5, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"Learned vs Fixed Agreement\n学习vs固定路由一致性\n"
                     f"(mean={agree_mask.mean():.2f})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Panel 12: 不一致区域红色覆盖 | Disagreement regions in red overlay
        ax = plt.subplot(3, 4, 12)
        ax.imshow(img_np)
        # 不一致 = 一致率 < 50%（至少一半的 Proto 选择不同）
        # Disagreement = agreement < 50% (at least half of Proto selections differ)
        disagreement = agree_up < 0.5
        ax.imshow(disagreement, alpha=0.3, cmap="Reds")
        ax.set_title("Regions where Router ≠ |w·sim|\nRouter与|w·sim|不一致区域\n(红色覆盖 | red overlay)", fontsize=9)
        ax.axis("off")

        # ── 保存单图可视化 | Save Per-Image Visualization ──
        fig.suptitle(f"E009-C Router Visualization — Image {ds_idx} "
                     f"(K={k}/{n_protos}, arch=conv3x3, Building={gt_np.mean():.1%})"
                     f"\nE009-C 路由可视化",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / f"router_viz_img{vis_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════
    # 10. 跨图像汇总 | Cross-Image Summary
    # ═══════════════════════════════════════════════════════════════

    # ── 10a. 区域 Proto 选择率表格 | Per-Region Proto Selection Rate Table ──
    # 打印每个 Proto 在三个区域的选中频率 | Print each Proto's selection frequency across three regions
    # 频率 = 该 Proto 在该区域被选中的像素数 / 该区域总像素数
    # Rate = pixels where Proto selected in region / total pixels in region
    print(f"\n  ── Cross-Image Proto Selection by Region ──")
    print(f"  ── 跨图像区域Proto选择汇总 ──")
    print(f"  {'Proto':<8} {'Interior (内)':>12} {'Edge (边)':>12} {'Background (背)':>12}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12}")
    for p in range(n_protos):
        ir = per_region["interior"][p] / max(region_pixel_counts["interior"], 1)
        er = per_region["edge"][p] / max(region_pixel_counts["edge"], 1)
        br = per_region["bg"][p] / max(region_pixel_counts["bg"], 1)
        print(f"  P{p:<7} {ir:>11.1%} {er:>11.1%} {br:>11.1%}")

    # ── 10b. 区域 Proto 选择率条形图 | Per-Region Proto Selection Bar Chart ──
    # 生成 1x3 面板的条形图，显示每个区域对不同 Proto 的偏好
    # Generate 1×3 bar chart showing each region's Proto preference
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (r_name, r_label) in enumerate([("interior", "Interior (建筑内部)"),
                                            ("edge", "Edge (建筑边缘)"),
                                            ("bg", "Background (背景)")]):
        rates = per_region[r_name] / max(region_pixel_counts[r_name], 1)
        # 高激活 Proto 红色标记 (>=50%), 低激活蓝色 | High-activation red (>=50%), low blue
        colors = ["tab:red" if v >= 0.5 else "tab:blue" for v in rates]
        axes[i].bar(range(n_protos), rates, color=colors)
        # 均匀选择基线：如果 Router 随机选择，每个 Proto 的期望选中率 = k/n_protos
        # Uniform selection baseline: if Router selects randomly, expected rate = k/n_protos
        axes[i].axhline(y=k/n_protos, color="gray", linestyle="--", alpha=0.5,
                        label=f"Uniform (均匀) ({k/n_protos:.1%})")
        axes[i].set_title(f"{r_label} — Proto Selection Rate\n区域Proto选择率\n"
                          f"({region_pixel_counts[r_name]:,} px)")
        axes[i].set_xticks(range(n_protos))
        axes[i].set_xticklabels([f"P{p}" for p in range(n_protos)])
        axes[i].set_ylabel("Selection Rate\n选择率")
        axes[i].set_ylim(0, 1)
        axes[i].legend(fontsize=7)

    fig.suptitle(f"E009-C: Per-Region Proto Selection (K={k}/{n_protos})\n"
                 f"E009-C: 各区域Proto选择率",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "region_proto_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 10c. Top Proto 组合输出 | Top Routing Combinations Printout ──
    # 打印每个区域最频繁的 Proto 组合（有助于发现区域特定的路由模式）
    # Print most frequent Proto combos per region (helps discover region-specific routing patterns)
    print(f"\n  ── Top Routing Combinations by Region ──")
    print(f"  ── 各区域Top路由组合 ──")
    for r_name in ["interior", "edge", "bg"]:
        r_combos = [(c, cnt) for (r, c), cnt in combo_counter.items() if r == r_name]
        r_combos.sort(key=lambda x: -x[1])  # 按频率降序排列 | Sort by descending frequency
        print(f"  [{r_name:9s}]")
        for combo, cnt in r_combos[:5]:
            pct = cnt / max(sum(c for _, c in r_combos), 1)
            print(f"    {combo}  ({cnt:6d} px, {pct:.1%})")

    print(f"\n  Visualizations saved to: {save_dir}/")
    print(f"  可视化结果已保存至: {save_dir}/")
    return per_region, region_pixel_counts


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main Function
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    主入口：加载模型检查点、初始化模块并运行路由可视化 |
    Main entry: load model checkpoint, initialize modules, and run routing visualization.

    流程 | Pipeline:
        1. 解析命令行参数 | Parse command-line arguments
        2. 加载检查点（支持 glob 通配符）| Load checkpoint (supports glob wildcards)
        3. 构建 ProtoHead 并加载预训练权重 | Build ProtoHead and load pretrained weights
        4. 构建 SPMHead（含 Router）并加载权重 | Build SPMHead (with Router) and load weights
        5. 冻结 FastSAM 骨干网络 | Freeze FastSAM backbone
        6. 实例化数据集并运行分析 | Instantiate dataset and run analysis
    """
    args = parse_args()
    device = args.device

    # ── 1. 加载检查点（支持通配符匹配）| Load checkpoint (supports wildcard matching) ──
    ckpt_path = args.checkpoint
    # 如果路径包含 *，使用 glob 匹配第一个文件 | If path contains *, glob-match the first file
    if "*" in ckpt_path:
        matches = _glob.glob(ckpt_path)
        ckpt_path = matches[0] if matches else ckpt_path
    print(f"Loading checkpoint (加载检查点): {ckpt_path}")

    # 加载检查点字典（weights_only=True 为了安全）| Load checkpoint dict (weights_only=True for safety)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # 获取 Router 架构类型（默认 conv3x3）| Get Router architecture type (default conv3x3)
    router_arch = ckpt.get("router_arch", "conv3x3")
    print(f"  Router arch: {router_arch}")

    # ── 2. 构建 ProtoHead（原型嵌入层）| Build ProtoHead (prototype embedding layer) ──
    # ProtoHead: 将 P4 特征 [B,1280,H,W] 投影到 Proto 嵌入空间 [B,N,embed_dim,H,W]
    # ProtoHead: projects P4 features [B,1280,H,W] to Proto embedding space [B,N,embed_dim,H,W]
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)
    proto_head.load_state_dict(ckpt["proto_head"])

    # ── 3. 构建 SPMHead（稀疏感知模块）| Build SPMHead (Sparse Perception Module) ──
    # SPMHead = ProtoHead + Router + Decoder 的集成体
    # SPMHead = integration of ProtoHead + Router + Decoder
    spm_head = SPMHead(proto_head, n_protos=args.n_protos,
                        router_k=args.router_k,
                        router_arch=router_arch).to(device)
    # 加载 Router 权重 | Load Router weights
    spm_head.router.load_state_dict(ckpt["router"])
    spm_head.eval()  # 推理模式 | Inference mode

    # ── 4. 初始化 FastSAM 骨干网络（冻结）| Initialize FastSAM backbone (frozen) ──
    # 骨干仅用于特征提取，不参与训练 | Backbone is for feature extraction only, not trained
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── 5. 加载数据集 | Load dataset ──
    dataset = MassachusettsBuildingsDataset(root_dir=args.data_root, split=args.split)
    print(f"Dataset (数据集): {args.split} ({len(dataset)} images)")

    # ── 6. 运行路由分析 | Run routing analysis ──
    analyze_routing(spm_head, backbone, dataset, device, args, args.output_dir)


if __name__ == "__main__":
    main()
