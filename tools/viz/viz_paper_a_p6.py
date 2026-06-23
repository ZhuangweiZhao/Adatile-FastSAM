#!/usr/bin/env python3
"""
E007.5: Proto Activation Analysis — 每个 Proto 到底在看什么？
===============================================================
Proto 激活模式可解释性分析与可视化 | Proto activation interpretability analysis & visualization.

诊断实验 | Diagnostic experiment.
用于验证 Proto 机制是否自发形成了语义上有意义的划分：
——建筑主体 (P2)、建筑边缘 (P6)、背景 (P8)。
Validates whether the Proto mechanism spontaneously forms semantically meaningful
partitions: building interior (P2), building edge (P6), background (P8).

核心问题 | Core question:
    P2 (83.9% building / 70K px)  → 建筑主体？| Building interior?
    P6 (93.7% building / 4K px)   → 建筑边缘？| Building edge?
    P8 (0.8% building / 3.1M px)  → 纯背景？  | Pure background?

验证方法 | Verification method:
    1. 计算每个 Proto 高激活区与 GT 边缘/内部/背景的 overlap
       Compute each Proto's high-activation region overlap with GT edge/interior/background
    2. 边缘 = Sobel(GT), 内部 = GT ∧ ¬edge, 背景 = ¬GT
       Edge = Sobel(GT), interior = GT AND NOT edge, background = NOT GT
    3. 多图平均统计 + 可视化 | Multi-image aggregate statistics + visualization

如果 P2≈内部、P6≈边缘、P8≈背景 → Proto 不仅是有效的，而是可解释的。
If P2≈interior, P6≈edge, P8≈background → Protos are not just effective, but interpretable.

核心可视化输出 (3行×4列面板) | Core visualization output (3 rows × 4 columns):
    Row 1: Image | GT Mask | GT Edges (Sobel) | Prediction
    Row 2: Building-interior Proto activation | Building-detail Proto activation |
           Background Proto activation | Top-20% activation overlay on image
    Row 3: Precision/Recall vs Edge bar chart | Interior Proto activation histogram |
           Detail Proto activation histogram | Per-Prototype Building Ratio bar chart

用法 | Usage::
    python tools/viz/viz_paper_a_p6.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 标准库导入 | Standard Library Imports
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 数值与深度学习库 | Numerical & Deep Learning Libraries
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# 可视化库 (非交互式后端) | Visualization (non-interactive backend)
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")  # 无头渲染，不弹出窗口 | Headless rendering, no GUI window
import matplotlib.pyplot as plt
from scipy.ndimage import sobel  # Sobel 边缘检测 | Sobel edge detection

# ═══════════════════════════════════════════════════════════════════════════════
# 项目路径设置 | Project Path Setup
# ═══════════════════════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # tools/viz/ → tools/ → 根目录
sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# AdaTile 项目模块 | AdaTile Project Modules
# ═══════════════════════════════════════════════════════════════════════════════
from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice, format_param_count


# ═══════════════════════════════════════════════════════════════════════════════
# 边缘检测工具函数 | Edge Detection Utility
# ═══════════════════════════════════════════════════════════════════════════════

def compute_edge(mask: np.ndarray) -> np.ndarray:
    """
    用 Sobel 算子从二值掩码中提取建筑边缘 | Extract building edges from binary mask via Sobel operator.

    Sobel 算子原理 | Sobel operator principle:
        - 对掩码分别在垂直 (axis=0) 和水平 (axis=1) 方向应用 Sobel 卷积核，
          计算每个像素的梯度幅值: G = sqrt(Gx² + Gy²)
        - Apply Sobel convolution kernels along vertical (axis=0) and horizontal (axis=1)
          directions, then compute gradient magnitude: G = sqrt(Gx² + Gy²)
        - 在建筑内部 (均匀=1) 梯度≈0，在建筑边界 (1↔0) 梯度大
          Building interior (uniform=1) → gradient≈0; building boundary (1↔0) → gradient large

    阈值 0.1 含义 | Threshold 0.1 meaning:
        - Sober 梯度幅值为连续值 [0, ~1.4] (sqrt(1²+1²))
          Sobel gradient magnitude is continuous [0, ~1.4] (sqrt(1²+1²))
        - 阈值 0.1: 只要检测到微弱边界即标记为边缘 (低阈值 = 敏感检测)
          Threshold 0.1: mark even weak boundaries as edges (low threshold = sensitive detection)
        - 目的是捕获所有可能的边缘像素，宁可多检不可漏检
          Goal: capture all potential edge pixels — better over-detect than miss

    :param mask: 二值分割掩码 [H, W]，1=建筑，0=背景 Binary segmentation mask [H, W], 1=building, 0=background
    :type mask: np.ndarray

    :return: edge_mask: 二值边缘掩码 [H, W], dtype=np.uint8, 1=边缘像素, 0=非边缘 Binary edge mask [H, W], dtype=np.uint8, 1=edge pixel, 0=non-edge
    :rtype: np.ndarray
    """
    # 垂直方向 Sobel 梯度 (检测水平边缘) | Vertical Sobel gradient (detects horizontal edges)
    gy = sobel(mask.astype(float), axis=0)
    # 水平方向 Sobel 梯度 (检测垂直边缘) | Horizontal Sobel gradient (detects vertical edges)
    gx = sobel(mask.astype(float), axis=1)
    # 梯度幅值: 综合两个方向的边缘强度 | Gradient magnitude: combined edge strength from both axes
    edge = np.sqrt(gx**2 + gy**2)
    # 二值化: >0.1 即为边缘 | Binarize: >0.1 threshold → edge pixel
    return (edge > 0.1).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 核心分析函数：Proto 激活模式可视化 | Core Analysis: Proto Activation Visualization
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_p6(backbone, proto_module, dataset, device, output_dir):
    """
    P6 激活模式分析与可视化主函数 | Main function for P6 activation pattern analysis & visualization.

    对选定的建筑相关 Proto (最高建筑率、次高建筑率、最低建筑率) 进行全维度分析：
    Comprehensive multi-dimensional analysis of selected building-related protos
    (highest building ratio, second-highest, lowest):

        1. 激活图可视化: 每个 Proto 的相似度图 (similarity map) 上采样至原图分辨率
           Activation visualization: upsample each Proto's similarity map to original resolution
        2. 边缘重叠分析: Proto 高激活区 (Top-20%) 与 GT Sobel 边缘的 Precision/Recall
           Edge overlap analysis: Precision/Recall of Proto top-20% activations vs GT Sobel edges
        3. 激活分布: 建筑区域 vs 背景区域的激活值直方图，验证类别分化
           Activation distribution: building vs background histogram — verify class differentiation
        4. Proto 语义分析: 所有 Proto 的建筑像素占比柱状图
           Proto semantic analysis: per-prototype building pixel ratio bar chart

    输出: 每张图一个 3×4 面板 PNG，保存到 output_dir
    Output: one 3×4 panel PNG per image, saved to output_dir

    :param backbone: FastSAMBackbone, 冻结的 P4 特征提取器 | Frozen P4 feature extractor

    :param proto_module: ProtoModule, 原型计算模块 | Prototype computation module

    :param dataset: MassachusettsBuildingsDataset (val split)

    :param device: torch device

    :param output_dir: 可视化输出目录 | Visualization output directory
    """
    # ── 输出目录创建 | Create output directory ──
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 快速分析每个 Proto 的建筑像素占比 | Quick per-proto building ratio analysis ──
    # 必须先跑一次 _quick_analyze 来确定哪些 Proto 偏向建筑、哪些偏向背景
    # Must run _quick_analyze first to determine which protos lean building vs background
    proto_module.eval()
    proto_build_pct, proto_activate_count = _quick_analyze(backbone, proto_module, dataset, device)

    # ── Proto 选择策略 | Proto Selection Strategy ──
    # 策略: 按 building ratio 降序/升序排列，选出三类有代表性的 Proto
    # Strategy: sort by building ratio asc/desc, select three representative protos

    # 建筑率降序 → 最"建筑"的 Proto 排前面 | Descending → most "building-like" protos first
    sorted_by_build = sorted(range(len(proto_build_pct)),
                             key=lambda p: proto_build_pct[p], reverse=True)
    # 建筑率升序 → 最"背景"的 Proto 排前面 | Ascending → most "background-like" protos first
    sorted_by_bg = sorted(range(len(proto_build_pct)),
                          key=lambda p: proto_build_pct[p])

    # 三类代表性 Proto | Three representative Proto categories:
    p_build_main = sorted_by_build[0]       # 主建筑 Proto (建筑主体/内部) | Main building Proto (building interior/body)
    p_build_detail = sorted_by_build[1] if len(sorted_by_build) > 1 else sorted_by_build[0]  # 细节建筑 Proto (建筑边缘) | Detail building Proto (building edges)
    p_bg = sorted_by_bg[0]                  # 纯背景 Proto | Pure background Proto

    # 按语义倾向分组 | Group by semantic leaning:
    # building-leaning: >50% 激活像素落在建筑区域 | >50% activated pixels fall in building regions
    build_protos = [p for p in range(len(proto_build_pct)) if proto_build_pct[p] > 0.5]
    # background-leaning: <30% 激活像素落在建筑区域 | <30% activated pixels fall in building regions
    bg_protos = [p for p in range(len(proto_build_pct)) if proto_build_pct[p] < 0.3]

    # 打印 Proto 分配结果 | Print proto assignment results
    print(f"  Building-leaning protos (>50%): {build_protos}")
    print(f"  Background-leaning protos (<30%): {bg_protos}")
    print(f"  Selected: P{p_build_main}=highest-build, P{p_build_detail}=2nd-build, P{p_bg}=lowest-build")
    for p in sorted_by_build:
        print(f"    P{p}: build%={proto_build_pct[p]:.1%}, px={proto_activate_count[p]:,.0f}")

    # ── Step 2: 选择可视化图像 (建筑占比 >5%) | Select visualization images (building area >5%) ──
    # 只选有足够建筑的图,否则边缘分析无意义
    # Only select images with sufficient buildings — otherwise edge analysis is meaningless
    val_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["masks"]
        # 确保掩码为 2D [H, W] | Ensure mask is 2D [H, W]
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        building_pct = mask.mean().item()  # 建筑像素占比 | Building pixel ratio
        if building_pct > 0.05:  # 至少 5% 建筑区域 | At least 5% building area
            val_indices.append(i)
        if len(val_indices) >= 4:  # 选 4 张 | Pick 4 images
            break

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: 逐图详细分析 | Per-Image Detailed Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    # 对每张选中的图:
    #   1) 提取 P4 特征 → 通过 ProtoModule 计算 similarity maps
    #   2) 上采样 similarity maps 至原图分辨率
    #   3) 绘制 3×4 面板: 原图/GT/边缘/预测 + 三个Proto激活 + 定量分析
    # For each selected image:
    #   1) Extract P4 features → compute similarity maps via ProtoModule
    #   2) Upsample similarity maps to original resolution
    #   3) Draw 3×4 panel: Image/GT/Edge/Pred + three Proto activations + quantitative analysis
    for vis_idx, ds_idx in enumerate(val_indices):
        # ── 加载单张图像与 GT | Load single image and GT ──
        sample = dataset[ds_idx]
        image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        gt_mask = sample["masks"]
        # 将 GT mask 统一规整为 2D [H, W] | Normalize GT mask to 2D [H, W]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        # ── 前向传播：特征提取 → Proto 计算 | Forward: feature extraction → Proto computation ──
        # 流程: Backbone(P4 features) → ProtoModule → similarity_maps + logit
        # Flow: Backbone(P4 features) → ProtoModule → similarity_maps + logit
        features = backbone(image)
        embedding, sim_maps, logit = proto_module(features["p4"], temperature=0.1)
        # embedding: [1, embed_dim, H/16, W/16] — 特征嵌入 | Feature embedding
        # sim_maps:  [1, n_protos, H/16, W/16] — 每个像素与每个 Proto 的余弦相似度 | Cosine similarity per pixel per proto
        # logit:     [1, 1, H/16, W/16] — 分割 logit (由 sim_maps 加权组合得到) | Segmentation logit (weighted combination of sim_maps)

        # ── 上采样 similarity maps 至原图分辨率 | Upsample similarity maps to original resolution ──
        # 双线性插值: 从 [H/16, W/16] → [H, W]，保持空间连续性
        # Bilinear interpolation: from [H/16, W/16] → [H, W], preserving spatial continuity
        sim_up = F.interpolate(sim_maps, size=tuple(gt_mask.shape),
                               mode="bilinear", align_corners=False)  # [1, N, H, W]
        sim_np = sim_up.squeeze(0).cpu().numpy()  # [N, H, W] — 逐 Proto 激活图 | Per-proto activation map

        # ── 准备可视化数据 (numpy 格式) | Prepare visualization data (numpy format) ──
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img_np = np.clip(img_np, 0, 1)  # 确保像素值在 [0,1] | Ensure pixel values in [0,1]
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)  # [H, W] 二值 GT | Binary GT
        edge_np = compute_edge(gt_np)  # [H, W] GT 边缘 | GT edges via Sobel

        # ═══════════════════════════════════════════════════════════════════════
        # 创建 3×4 可视化面板 | Create 3×4 Visualization Panel
        # ═══════════════════════════════════════════════════════════════════════
        # Row 1: 输入/GT 信息 | Input/GT Information
        # Row 2: Proto 激活图 | Proto Activation Maps
        # Row 3: 定量分析 | Quantitative Analysis
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))

        # ── Row 1: Image, GT, GT Edge, Prediction ──
        # Col 0: 原始 RGB 图像 | Original RGB image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Image", fontsize=9)
        axes[0, 0].axis("off")

        # Col 1: 二值 GT 掩码 (白=建筑, 黑=背景) | Binary GT mask (white=building, black=background)
        axes[0, 1].imshow(gt_np, cmap="gray")
        axes[0, 1].set_title("GT Mask", fontsize=9)
        axes[0, 1].axis("off")

        # Col 2: Sobel 边缘检测结果 (热力图, 亮=边缘) | Sobel edge detection result (heatmap, bright=edge)
        axes[0, 2].imshow(edge_np, cmap="hot")
        axes[0, 2].set_title("GT Edges (Sobel)", fontsize=9)
        axes[0, 2].axis("off")

        # Col 3: 模型预测 (二值, sigmoid>0.5) | Model prediction (binary, sigmoid>0.5)
        logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                 mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze().cpu().numpy()
        axes[0, 3].imshow(pred, cmap="gray")
        axes[0, 3].set_title("Prediction", fontsize=9)
        axes[0, 3].axis("off")

        # ═══════════════════════════════════════════════════════════════════════
        # Row 2: Proto 激活图 | Proto Activation Maps
        # ═══════════════════════════════════════════════════════════════════════
        # 三个代表性 Proto 的 similarity map (余弦相似度, 范围 [-1, 1])
        # Similarity maps of three representative protos (cosine similarity, range [-1, 1])
        #   正值 (红/亮) = 该像素与 Proto 方向一致 | Positive (red/bright) = pixel aligned with Proto direction
        #   负值 (蓝/暗) = 该像素与 Proto 方向相反 | Negative (blue/dark) = pixel opposed to Proto direction

        def plot_proto(ax, proto_idx, title, cmap="hot"):
            """
            绘制单个 Proto 的激活图 | Plot a single Proto's activation map.

            使用统一色阶 vmin=-1.0, vmax=1.0 (余弦相似度理论范围) 确保跨图可比较。
            Uses fixed color range vmin=-1.0, vmax=1.0 (cosine similarity theoretical range)
            to ensure comparability across images.

            :param ax: matplotlib axis

            :param proto_idx: Proto 索引 | Proto index

            :param title: 子图标题 | Subplot title

            :param cmap: 颜色映射 | Colormap (default "hot")
            """
            vmin, vmax = -1.0, 1.0
            im = ax.imshow(sim_np[proto_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            return im

        # Col 0: 主建筑 Proto — 预期激活建筑主体/内部 | Main building Proto — expected to activate building interior/body
        plot_proto(axes[1, 0], p_build_main,
                   f"P{p_build_main} (Building Interior)\n"
                   f"Build%={proto_build_pct[p_build_main]:.1%}")
        # Col 1: 细节建筑 Proto — 预期激活建筑边缘/细节 | Detail building Proto — expected to activate building edges/details
        plot_proto(axes[1, 1], p_build_detail,
                   f"P{p_build_detail} (Building Detail)\n"
                   f"Build%={proto_build_pct[p_build_detail]:.1%}, "
                   f"{proto_activate_count[p_build_detail]:,.0f} px")
        # Col 2: 背景 Proto — 预期激活非建筑区域 | Background Proto — expected to activate non-building regions
        plot_proto(axes[1, 2], p_bg,
                   f"P{p_bg} (Background)\n"
                   f"Build%={proto_build_pct[p_bg]:.1%}")

        # ═══════════════════════════════════════════════════════════════════════
        # Col 3: Top-20% 高激活区叠加显示 | Top-20% High-Activation Region Overlay
        # ═══════════════════════════════════════════════════════════════════════
        # 目的: 直观展示细节建筑 Proto 的"最关注"区域在哪
        # Purpose: intuitively show WHERE the detail-building Proto "focuses" most
        # Top-20% 阈值: 取该 Proto 所有像素相似度的第 80 百分位数
        # Top-20% threshold: 80th percentile of all pixel similarities for this Proto
        ax = axes[1, 3]
        ax.imshow(img_np)  # 底层显示原图 | Background: original image
        p_detail = sim_np[p_build_detail]
        threshold = np.percentile(p_detail, 80)  # 80 分位 → 高激活阈值 | 80th percentile → high-activation threshold
        high_act = p_detail > threshold  # 二值高激活掩码 | Binary high-activation mask
        ax.imshow(high_act, alpha=0.5, cmap="Reds")  # 半透明红色叠加 | Semi-transparent red overlay
        ax.set_title(f"P{p_build_detail} Top-20% Activation\noverlayed on image", fontsize=9)
        ax.axis("off")

        # ═══════════════════════════════════════════════════════════════════════
        # Row 3: 定量分析 | Quantitative Analysis
        # ═══════════════════════════════════════════════════════════════════════

        # ── Col 0: 边缘重叠 Precision/Recall 柱状图 | Edge Overlap Precision/Recall Bar Chart ──
        # 检验细节建筑 Proto (p_build_detail) 的高激活区是否与 GT 边缘重合
        # Test whether detail-building Proto's high-activation regions coincide with GT edges
        p_act_binary = (p_detail > threshold).astype(np.uint8)  # Top-20% 高激活区二值化 | Binarize top-20% activations
        edge_flat = edge_np.flatten()  # 展平 GT 边缘图 | Flatten GT edge map
        act_flat = p_act_binary.flatten()  # 展平激活图 | Flatten activation map

        # Precision (精确率): 在被激活的像素中，有多少比例是真正的边缘像素？
        # Precision: of all activated pixels, what fraction are true edge pixels?
        # → 衡量"该 Proto 是否专门关注边缘" | Measures "does this Proto specifically focus on edges?"
        edge_in_act = edge_flat[act_flat == 1]  # 激活区内的边缘像素 | Edge pixels within activated region
        prec = edge_in_act.mean() if len(edge_in_act) > 0 else 0

        # Recall (召回率): 在所有边缘像素中，有多少比例被该 Proto 激活？
        # Recall: of all edge pixels, what fraction are activated by this Proto?
        # → 衡量"该 Proto 覆盖了多少边缘" | Measures "how many edges does this Proto cover?"
        act_in_edge = act_flat[edge_flat == 1]  # 边缘上的激活像素 | Activated pixels on edges
        rec = act_in_edge.mean() if len(act_in_edge) > 0 else 0

        # 随机 baseline: 如果随机选取像素，Precision ≈ 全图边缘占比
        # Random baseline: if pixels are chosen randomly, Precision ≈ global edge ratio
        # 只有当 Prec >> edge_total 时，才说明 Proto 主动"寻找"边缘
        # Only when Prec >> edge_total does the Proto actively "seek" edges
        edge_total = edge_flat.mean()  # 全图边缘像素比例 | Global edge pixel ratio
        prec_random = edge_total  # 随机猜测的期望 Precision | Expected Precision if random

        # Bar chart: Precision, Recall vs random
        ax = axes[2, 0]
        ax.bar(["Precision", "Recall", "Edge% (random)"],
               [prec, rec, edge_total],
               color=["tab:red", "tab:blue", "gray"])
        ax.set_ylim(0, max(1.0, prec + 0.1))
        ax.set_title(f"P{p_build_detail} vs GT Edge Overlap\n"
                     f"Prec={prec:.3f} Rec={rec:.3f}", fontsize=9)
        ax.axhline(y=edge_total, color="gray", linestyle="--", alpha=0.3)  # 随机基线参考线 | Random baseline reference line

        # ── Col 1: 主建筑 Proto 激活分布直方图 | Main Building Proto Activation Distribution Histogram ──
        # 比较建筑区域 vs 背景区域的激活值分布
        # Compare activation value distributions: building regions vs background regions
        #   如果两个分布显著分离 → Proto 具有语义判别力
        #   If two distributions are significantly separated → Proto has semantic discriminative power
        ax = axes[2, 1]
        p_main_vals = sim_np[p_build_main].flatten()  # 该 Proto 所有像素的相似度 | Similarities of all pixels for this Proto
        ax.hist(p_main_vals[gt_np.flatten() == 1], bins=50, alpha=0.5,
                density=True, color="tab:red", label="Building")  # 建筑区域 | Building regions
        ax.hist(p_main_vals[gt_np.flatten() == 0], bins=50, alpha=0.5,
                density=True, color="tab:blue", label="Background")  # 背景区域 | Background regions
        ax.set_title(f"P{p_build_main} Activation Distribution\n"
                     f"Building vs Background", fontsize=9)
        ax.legend(fontsize=7)

        # ── Col 2: 细节建筑 Proto 激活分布直方图 | Detail Building Proto Activation Distribution Histogram ──
        # 同上，针对细节建筑 Proto；预期建筑/背景分布分离程度不同
        # Same as above, for detail-building Proto; expected separation differs from main building Proto
        ax = axes[2, 2]
        p_detail_vals = p_detail.flatten()
        ax.hist(p_detail_vals[gt_np.flatten() == 1], bins=50, alpha=0.5,
                density=True, color="tab:red", label="Building")
        ax.hist(p_detail_vals[gt_np.flatten() == 0], bins=50, alpha=0.5,
                density=True, color="tab:blue", label="Background")
        ax.set_title(f"P{p_build_detail} Activation Distribution\n"
                     f"Building vs Background", fontsize=9)
        ax.legend(fontsize=7)

        # ── Col 3: 所有 Proto 建筑像素占比柱状图 | Per-Prototype Building Ratio Bar Chart ──
        # 展示每个 Proto 的"语义偏好": 红色 (>50%) = 偏建筑, 蓝色 (<50%) = 偏背景
        # Shows each Proto's "semantic preference": red (>50%) = building-leaning, blue (<50%) = background-leaning
        # 虚线 y=0.5 为随机分界线 | Dashed line y=0.5 is the random baseline
        ax = axes[2, 3]
        x = np.arange(len(proto_build_pct))  # Proto 索引 | Proto indices
        width = 0.35
        ax.bar(x, proto_build_pct, width, color=[
            "tab:red" if v >= 0.5 else "tab:blue" for v in proto_build_pct  # 红=建筑倾向, 蓝=背景倾向
        ])
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i}" for i in range(len(proto_build_pct))], fontsize=7)
        ax.set_ylabel("Building Ratio", fontsize=8)
        ax.set_title("Per-Prototype Building Ratio", fontsize=9)
        ax.set_ylim(0, 1)

        # ── 保存整张图 | Save the whole figure ──
        fig.suptitle(f"P6 Analysis — Image {ds_idx} (Building Area = {gt_np.mean():.1%})",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / f"p6_img{vis_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)  # 释放内存 | Release memory

        # 打印该图的边缘重叠统计 | Print edge overlap statistics for this image
        print(f"  img{vis_idx} (idx={ds_idx}): "
              f"P{p_build_detail} Precision(edge)={prec:.3f}, Recall(edge)={rec:.3f}, "
              f"Random={edge_total:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: 多图聚合统计 | Multi-Image Aggregate Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    # 汇总所有 Proto 的语义属性 (基于前 20 张图的累积统计)
    # Summarize semantic properties of all Protos (based on cumulative stats from first 20 images)
    print(f"\n  All building protos: {build_protos}")
    for p in build_protos:
        print(f"    P{p}: build%={proto_build_pct[p]:.1%}, "
              f"pixels={proto_activate_count[p]:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 快速分析：每个 Proto 的建筑像素占比 | Quick Analysis: Per-Proto Building Pixel Ratio
# ═══════════════════════════════════════════════════════════════════════════════

def _quick_analyze(backbone, proto_module, dataset, device):
    """
    快速统计每个 Proto 的建筑像素占比 (复用 E007 逻辑) | Quick per-proto building pixel ratio (reuses E007 logic).

    工作流程 | Workflow:
        1. 取前 20 张图 → 运行 Backbone 提取 P4 特征
           Take first 20 images → run Backbone to extract P4 features
        2. 通过 ProtoModule.get_hard_assignment() 获取硬分配 (每个像素归属哪个 Proto)
           Get hard assignment via ProtoModule.get_hard_assignment() (which Proto each pixel belongs to)
        3. 最近邻上采样 hard assignment 至原图分辨率 (mode="nearest", 保持整数标签)
           Upsample hard assignment to original resolution via nearest-neighbor (preserve integer labels)
        4. 对每个 Proto 统计: 总激活像素数, 其中落在建筑区域的像素数
           For each Proto: count total activated pixels and those falling in building regions
        5. 计算 building ratio = 建筑像素数 / 总激活像素数
           Compute building ratio = building pixels / total activated pixels

    硬分配 (Hard Assignment) 原理 | Hard assignment principle:
        - get_hard_assignment 返回每个像素与哪个 Proto 的余弦相似度最大
          get_hard_assignment returns which Proto each pixel has the highest cosine similarity with
        - 输出: [1, H/16, W/16], dtype=long, 值范围 [0, n_protos-1]
          Output: [1, H/16, W/16], dtype=long, value range [0, n_protos-1]
        - 这是"赢者通吃"策略: 每个像素被唯一分配给一个 Proto
          This is a "winner-take-all" strategy: each pixel is uniquely assigned to one Proto

    :param backbone: FastSAMBackbone, 冻结的 P4 特征提取器 | Frozen P4 feature extractor

    :param proto_module: ProtoModule with get_hard_assignment method

    :param dataset: MassachusettsBuildingsDataset

    :param device: torch device

    :return: proto_build_pct:     np.ndarray [n_protos], 每个 Proto 的建筑像素占比 (累积归一化) Per-proto building pixel ratio (accumulated, normalized) proto_activate_count: np.ndarray [n_protos], 每个 Proto 的累积激活像素总数 Per-proto accumulated activated pixel count
    """
    n_protos = proto_module.n_protos
    # 累积统计: 分子=建筑像素, 分母=总激活像素 | Cumulative stats: numerator=building pixels, denominator=total activated
    proto_build_pct = np.zeros(n_protos)
    proto_activate_count = np.zeros(n_protos)

    # 遍历前 20 张图 (足够获得稳定的统计) | Iterate first 20 images (enough for stable statistics)
    for idx in range(min(20, len(dataset))):
        # ── 加载图像和 GT | Load image and GT ──
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
        gt_mask = sample["masks"].to(device)
        # 统一 GT mask 为 2D [H, W] | Normalize GT mask to 2D [H, W]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        # ── 前向传播：获取硬分配 | Forward: get hard assignment ──
        features = backbone(image)
        hard_assign = proto_module.get_hard_assignment(features["p4"])  # [1, H/16, W/16], long

        # ── 上采样硬分配至原图分辨率 | Upsample hard assignment to original resolution ──
        # 使用 nearest 插值保持离散标签 (不能用 bilinear, 会产生非整数)
        # Use nearest interpolation to preserve discrete labels (bilinear would produce non-integers)
        hard_up = F.interpolate(
            hard_assign.unsqueeze(1).float(),  # [1, 1, H/16, W/16]
            size=tuple(gt_mask.shape), mode="nearest",  # nearest = 不引入新值 | no new values introduced
        ).squeeze(1).long().squeeze(0)  # → [H, W], dtype=long

        # ── 逐 Proto 累积统计 | Per-Proto cumulative statistics ──
        for p in range(n_protos):
            proto_mask = (hard_up == p)  # 该 Proto 的所有激活像素 | All pixels assigned to this Proto
            n_pixels = proto_mask.sum().item()  # 激活像素总数 | Total activated pixels
            if n_pixels > 0:
                n_building = (gt_mask[proto_mask] == 1).sum().item()  # 其中建筑像素数 | Building pixels among them
                proto_build_pct[p] += n_building  # 累加建筑像素 | Accumulate building pixels
                proto_activate_count[p] += n_pixels  # 累加总像素 | Accumulate total pixels

    # ── 归一化: 建筑占比 = 累积建筑像素 / 累积总激活像素 | Normalize: building ratio = cum building / cum total ──
    for p in range(n_protos):
        if proto_activate_count[p] > 0:
            proto_build_pct[p] /= proto_activate_count[p]
        # 如果某 Proto 从未被激活, 占比保持 0 | If a Proto was never activated, ratio stays 0

    return proto_build_pct, proto_activate_count


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口 | Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    P6 激活分析完整流程 | Complete P6 activation analysis pipeline.

    三阶段流程 | Three-stage pipeline:
        [1/3] 加载冻结的 FastSAM Backbone | Load frozen FastSAM Backbone
        [2/3] 快速训练 ProtoModule (20 epochs, Adam+CosineLR) | Quick-train ProtoModule
        [3/3] 运行 P6 分析与可视化 | Run P6 analysis & visualization
    """
    # ── 设备选择 | Device selection ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "runs/p6_analysis"

    print("=" * 60)
    print("  P6 Activation Map Analysis")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage [1/3]: 加载 Backbone | Load Backbone
    # ═══════════════════════════════════════════════════════════════════════════
    # Backbone 完全冻结 — 只作为 P4 特征提取器，不参与训练
    # Backbone fully frozen — serves only as P4 feature extractor, no training
    print("\n[1/3] Load Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()  # YOLOv8 必须保持 eval 模式, 否则 detect head 崩溃 | Must keep eval mode for YOLOv8

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage [2/3]: 快速训练 ProtoModule | Quick-Train ProtoModule
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/3] Train Proto Module (quick)")
    # 导入 ProtoModule (来自 E007 实验) | Import ProtoModule (from E007 experiment)
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
    from eval_e007_proto_module import ProtoModule

    # ProtoModule 架构 | Architecture:
    #   in_channels=1280 (FastSAM P4 层通道数 | FastSAM P4 layer channels)
    #   embed_dim=128   (嵌入空间维度 | Embedding space dimension)
    #   n_protos=12     (原型数量 | Number of prototypes)
    proto_module = ProtoModule(in_channels=1280, embed_dim=128, n_protos=12).to(device)

    # ── 数据集加载 | Dataset loading ──
    train_ds = MassachusettsBuildingsDataset(root_dir="data/Massachusetts_Buildings", split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir="data/Massachusetts_Buildings", split="val")

    # ── 优化器与调度器 | Optimizer & Scheduler ──
    # Adam: 适合小模块快速收敛 | Good for fast convergence of small modules
    # CosineAnnealingLR: 从 lr=1e-3 余弦衰减至 eta_min=1e-6, 周期=20 epoch
    # CosineAnnealingLR: cosine decay from lr=1e-3 to eta_min=1e-6, period=20 epochs
    proto_module.train()
    optimizer = torch.optim.Adam(proto_module.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    best_dice = 0.0  # 跟踪最佳模型 (用于模型选择, 此处未保存) | Track best model (for selection, not saved here)

    # ═══════════════════════════════════════════════════════════════════════════
    # 训练循环 (20 Epochs) | Training Loop (20 Epochs)
    # ═══════════════════════════════════════════════════════════════════════════
    print("  Training 20 epochs (lr=1e-3, CosineLR)...")
    for epoch in range(1, 21):
        # ── 训练阶段 | Train Phase ──
        proto_module.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/20 [train]", leave=False)
        for idx in pbar:
            # 加载单张图像 (batch_size=1, 逐样本训练) | Load single image (batch_size=1, per-sample training)
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
            gt_mask = sample["masks"].to(device)
            # 统一 GT mask 为 2D [H, W] | Normalize GT mask to 2D [H, W]
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            elif gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(0).squeeze(0)

            # 前向传播 | Forward pass:
            #   backbone(image) → P4 features
            #   proto_module(P4) → similarity_maps + logit (只使用 logit)
            with torch.no_grad():
                features = backbone(image)
            _, _, logit = proto_module(features["p4"], temperature=0.1)

            # 损失计算 | Loss computation:
            #   BCEWithLogitsLoss: 直接在 logit 上计算, 数值稳定
            #   BCEWithLogitsLoss: computed directly on logits, numerically stable
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)

            # 反向传播与优化 | Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})

        # 学习率调度: 每个 epoch 后步进一次 | LR scheduling: step once per epoch
        scheduler.step()

        # ── 验证阶段 | Validation Phase ──
        proto_module.eval()
        dices = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(0).squeeze(0)

                features = backbone(image)
                _, _, logit = proto_module(features["p4"], temperature=0.1)
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                # sigmoid 二值化: >0.5 → 建筑 | sigmoid binarization: >0.5 → building
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)

                # 统一维度: pred=[1,H,W], gt=[H,W] → 各加 batch 维后用 compute_dice
                # Align dimensions: ensure both have batch dim before compute_dice
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                dices.append(compute_dice(pred, gt_mask).item())

        # ── 汇总验证指标、标记最佳模型 | Aggregate validation metrics, mark best model ──
        dice_mean = float(np.mean(dices))
        is_best = dice_mean > best_dice
        if is_best:
            best_dice = dice_mean
        marker = " *" if is_best else ""  # * 标记最佳 epoch | * marks best epoch
        print(f"    Epoch {epoch:2d}/20  loss={total_loss/len(train_ds):.4f}  "
              f"Dice(val)={dice_mean:.4f}{marker}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage [3/3]: P6 分析与可视化 | P6 Analysis & Visualization
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/3] P6 Analysis & Visualization")
    analyze_p6(backbone, proto_module, val_ds, device, output_dir)
    print(f"\n  Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
