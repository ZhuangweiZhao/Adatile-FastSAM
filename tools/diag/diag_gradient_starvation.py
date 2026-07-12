#!/usr/bin/env python
"""
诊断: 原型分支梯度饥饿探针 | Diagnostic: Prototype-branch Gradient-Starvation Probe.
================================================================================

**问题背景 | Background**
`AdaptiveSparseDecoder` 用**加法**融合两支 (adatile/decoder/adaptive_sparse_decoder.py:249):
    final_logit = refined_logit_up(query P4 CNN)  +  proto_mask(support prototype)
已有证据 ([[prototype-functional-death]]): Zero/Random/Normal prototype 输出逐位相等,
CoeffPredictor 系数跨类 cosine=1.0000 (塌缩为常数)。假说: **Query-only 捷径 / 梯度饥饿** ——
高容量的 query CNN 支独立把 loss 降到底, 原型支得不到保持类判别性的梯度压力, 退化成常数。

**本探针做什么 | What this probe does**
在**已训练好的 checkpoint** 上 (无需重训、不改训练流程), 跑若干 episode 反传, 测量两支各自
收到的梯度大小:
  - coeff 支 (被怀疑饿死): 参数名前缀 `coeff_predictor.`
  - query/refine 支:        `feat_proj.` / `feat_refine.` / `mask_head.`
并做**决定性反事实 detach-refine**: 把 query 支 `.detach()` 后再反传, 若 coeff 支梯度暴涨 →
证明 query 支是在**竞争性吸走**信号 (饿死), 而非原型天生无用。

**为什么静态探针成立 | Why a static probe is valid**
bs=1 逐 episode 的梯度**非零** (只有整个数据集的期望梯度在极小值处才≈0); 两支梯度的**比值**是
有效量。同时报告原始范数与**每参数 RMS** (norm/√#params), 消除两支参数量差异带来的假象。

用法 | Usage::

    # 单 checkpoint (normal + detach 两种条件)
    python tools/diag/diag_gradient_starvation.py \
        --checkpoint "runs/.../train_fewshot_allcls_K1_adaptive_uf8_0711_0837/best_model.pt" \
        --k-shot 1 --per-class 3 --device cuda

    # 解冻扫描 (uf0/uf5/uf8/uf12) → 梯度比 vs 解冻深度轨迹
    python tools/diag/diag_gradient_starvation.py --sweep --per-class 3 --device cuda

输出 | Output: `<output-dir>/gradient_starvation.json` + 图 + `eval.jsonl` (adatile.logging)。
"""

from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from adatile.utils.seed import set_seed
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend

# 复用训练脚本的数据/特征 helper (与 diag_prototype_analysis.py 相同来源)
# Reuse the trainer's data/feature helpers (same import surface as diag_prototype_analysis.py)
from tools.train.train_fewshot_allclass import (
    _build_class_index_instance,
    _resolve_paths,
    load_instance_tile_and_mask,
    extract_features,
    compute_support_prototype,
    semantic_mask_to_binary,
    _normalize_mask_to_4d,
    CATEGORY_NAMES,
)

# ── 分支参数分组 | Branch parameter groups (by name prefix on decoder.named_parameters()) ──
BRANCH_GROUPS: dict[str, tuple[str, ...]] = {
    "coeff": ("coeff_predictor.",),                       # 原型→系数 MLP (被怀疑饿死)
    "refine": ("feat_proj.", "feat_refine.", "mask_head."),  # query-only P4 精炼支
}

# ── 解冻扫描默认 checkpoint (K=1, adaptive; 相对项目根) | Default sweep checkpoints ──
SWEEP_CHECKPOINTS: list[tuple[int, str]] = [
    (0,  "runs/云服务器/runs (1)/runs/train_fewshot_allcls_K1_adaptive_0710_1841/best_model.pt"),
    (5,  "runs/云服务器/runs (1)/runs/train_fewshot_allcls_K1_adaptive_uf5_0711_0353/best_model.pt"),
    (8,  "runs/云服务器/runs (1)/runs/train_fewshot_allcls_K1_adaptive_uf8_0711_0837/best_model.pt"),
    (12, "runs/云服务器/runs (1)/runs/train_fewshot_allcls_K1_adaptive_uf12_0711_1807/best_model.pt"),
]


# ═══════════════════════════════════════════════════════════════════
# 纯函数: 梯度分组 (可单测) | Pure function: grad grouping (unit-testable)
# ═══════════════════════════════════════════════════════════════════

def group_grad_norms(named_parameters, groups: dict[str, tuple[str, ...]]) -> dict[str, dict]:
    """按名称前缀把参数梯度聚合成每支的 L2 范数 | Aggregate param-grad L2 norm per branch by name prefix.

    :param named_parameters: 可迭代 (name, param), param.grad 已填充 | iterable of (name, param) with .grad set.
    :param groups: {branch_name: (prefix, ...)} 前缀元组 (str.startswith 接受元组).
    :return: {branch: {"norm": float, "n_params": int, "rms": float}}
             norm = sqrt(Σ g²);  rms = norm / sqrt(#params) (消除参数量差异 | neutralize size gap)。
    """
    acc = {g: {"sq": 0.0, "n": 0} for g in groups}
    for name, p in named_parameters:
        if p.grad is None:
            continue
        for g, prefixes in groups.items():
            if name.startswith(prefixes):  # startswith 接受前缀元组 | startswith accepts a tuple
                acc[g]["sq"] += float(p.grad.detach().double().pow(2).sum().item())
                acc[g]["n"] += int(p.numel())
                break
    out = {}
    for g, a in acc.items():
        norm = a["sq"] ** 0.5
        n = a["n"]
        out[g] = {"norm": norm, "n_params": n, "rms": (norm / (n ** 0.5)) if n > 0 else 0.0}
    return out


# ═══════════════════════════════════════════════════════════════════
# 前向 + loss (内联复刻 decoder.forward, 支持 detach-refine)
# Forward + loss (inline replica of decoder.forward, supports detach-refine)
# ═══════════════════════════════════════════════════════════════════

def forward_and_loss(decoder, p4, proto_masks, support_proto, query_gt: np.ndarray,
                     device: str, detach_refine: bool = False, temp: float = 1.0):
    """复刻 AdaptiveSparseDecoder.forward (use_fdr=False) + 训练 loss (BCE+Dice)。

    内联复刻是为了能在 detach-refine 条件下切断 query 支梯度, 同时**完全不改动** decoder 模块。
    Mirrors adaptive_sparse_decoder.py:203-250 and the trainer loss (train_fewshot_allclass.py:828-835),
    so the query branch can be detached without touching the module.

    :param temp: 干预实验用的 sigmoid 温度 | de-saturation temperature: proto_mask = sigmoid(proto_logit / temp)。
                 temp>1 解除饱和 (softer) — 用于检验"解除饱和是否复活梯度"。
    :return: (loss, coeffs, diag) — coeffs 已 retain_grad, 反传后可读 coeffs.grad;
             diag 含饱和度 + 幅值分解 {proto_presig_absmax(raw), proto_mask_sat_frac,
             coeff_absmax, coeff_l2, proto_basis_l2_max, proto_basis_l2_mean, proto_basis_absmax}。
    """
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)                # [1,32,H4,W4] → [32,H4,W4]
    if support_proto.dim() == 1:
        support_proto = support_proto.unsqueeze(0)          # [C] → [1,C]

    # 应用 decoder 的 proto 归一化 (若有), 使探针如实反映该 checkpoint 的行为 | honor decoder normalization
    proto_masks = decoder._normalize_proto(proto_masks)

    # Step 1: 原型 → 系数 → 粗掩码 | prototype → coeffs → coarse proto mask
    #   内联复刻 generate_mask, 以便同时捕获 pre-sigmoid 幅值 (判断是否饱和死区)
    #   Inline replica of generate_mask so we can also capture the pre-sigmoid magnitude
    #   (to test the "saturated dead zone" hypothesis).
    coeffs = decoder.coeff_predictor(support_proto)         # [1, proto_dim]
    coeffs.retain_grad()                                    # 记录 ‖∂L/∂coeffs‖ | probe grad at coeffs
    proto_flat = proto_masks.reshape(proto_masks.shape[0], -1)   # [proto_dim, H4*W4]
    proto_logit = coeffs @ proto_flat                      # [1, H4*W4] pre-sigmoid (raw)
    proto_mask = torch.sigmoid(proto_logit / temp).view(1, *proto_masks.shape[1:])  # [1, H4, W4]

    # 诊断: proto 支是否落入 sigmoid 饱和死区 + 幅值分解 (归因 coeff vs proto_basis)
    # is the proto branch in a saturated dead zone? + magnitude attribution (coeff vs basis)
    with torch.no_grad():
        pm = proto_mask.detach()
        basis_l2 = proto_flat.norm(dim=1)                  # [proto_dim] 每个基的 L2 范数
        diag = {
            "proto_presig_absmax": float(proto_logit.abs().max().item()),  # raw, 未除 temp
            "proto_mask_sat_frac": float(((pm < 1e-6) | (pm > 1 - 1e-6)).float().mean().item()),
            "coeff_absmax": float(coeffs.detach().abs().max().item()),
            "coeff_l2": float(coeffs.detach().norm().item()),
            "proto_basis_l2_max": float(basis_l2.max().item()),
            "proto_basis_l2_mean": float(basis_l2.mean().item()),
            "proto_basis_absmax": float(proto_flat.abs().max().item()),
        }

    # Step 2: P4 精炼支 (query-only) | P4 refinement branch (query-only)
    feat_proj = decoder.feat_proj(p4)
    feat_refined = decoder.feat_refine(feat_proj)
    refined_logit = decoder.mask_head(feat_refined)        # [B,1,H16,W16]
    refined_logit_up = F.interpolate(
        refined_logit, size=proto_mask.shape[1:], mode="bilinear", align_corners=False
    )                                                       # [B,1,H4,W4]

    # 反事实: 切断 query 支梯度 | counterfactual: cut query-branch gradient
    if detach_refine:
        refined_logit_up = refined_logit_up.detach()

    # Step 3: 加法融合 (与 decoder.forward:249 一致) | additive fusion
    final_logit = refined_logit_up.squeeze(1) + proto_mask.squeeze(0)   # [1,H4,W4]
    final_mask = torch.sigmoid(final_logit)

    # 上采样到 GT + BCE+Dice (复刻 train_fewshot_allclass.py:820-835)
    mask_s4 = _normalize_mask_to_4d(final_mask)            # [1,1,H4,W4]
    H_gt, W_gt = query_gt.shape
    mask_pred = F.interpolate(
        mask_s4, size=(H_gt, W_gt), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)                                # [H_gt, W_gt]

    gt_tensor = torch.from_numpy(query_gt).float().to(device)
    bce = F.binary_cross_entropy(mask_pred.clamp(1e-7, 1 - 1e-7), gt_tensor)
    inter = (mask_pred * gt_tensor).sum()
    union = mask_pred.sum() + gt_tensor.sum()
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    loss = bce + (1.0 - dice)
    return loss, coeffs, diag


# ═══════════════════════════════════════════════════════════════════
# 模型 / decoder 加载 | Model / decoder loading
# ═══════════════════════════════════════════════════════════════════

def load_backbone(checkpoint: str, device: str):
    """加载 FastSAM 并恢复 checkpoint 里的解冻 backbone 层 | Load FastSAM + restore unfrozen backbone."""
    from ultralytics import FastSAM
    model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    model = FastSAM(str(model_path))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(checkpoint, map_location=device)
    unfreeze_layers = int(ckpt.get("unfreeze_layers", 0))
    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        print(f"  Backbone: restored {unfreeze_layers} unfrozen layers")
    return model, ckpt, unfreeze_layers


def build_decoder(ckpt: dict, device: str) -> AdaptiveSparseDecoder:
    """从 checkpoint 的 decoder state 重建 AdaptiveSparseDecoder (use_fdr=False) | Rebuild the full decoder."""
    dstate = ckpt["decoder"]
    w0 = dstate["coeff_predictor.mlp.0.weight"]   # [hidden, feat]
    hidden_dim, feat_dim = int(w0.shape[0]), int(w0.shape[1])
    proto_dim = int(dstate["coeff_predictor.mlp.4.weight"].shape[0])
    normalize_proto = ckpt.get("normalize_proto", "none")  # 复原训练时的归一化 | restore proto-norm
    decoder = AdaptiveSparseDecoder(
        in_channels=feat_dim, proto_dim=proto_dim, hidden_dim=hidden_dim, use_fdr=False,
        normalize_proto=normalize_proto,
    ).to(device)
    decoder.load_state_dict(dstate)               # strict: 训练用 use_fdr=False, 键完全匹配
    decoder.eval()                                # InstanceNorm 在 train/eval 行为一致 | IN same in train/eval
    for p in decoder.parameters():
        p.requires_grad_(True)
    print(f"  Decoder: feat={feat_dim}, hidden={hidden_dim}, proto={proto_dim}, normalize_proto={normalize_proto}")
    return decoder


# ═══════════════════════════════════════════════════════════════════
# Episode 采样 (复刻训练协议: 0% 源图重叠) | Episode sampling (0% source overlap)
# ═══════════════════════════════════════════════════════════════════

def sample_episodes(class_index: dict, k_shot: int, per_class: int,
                    max_classes: int, max_support_tiles: int, rng: random.Random):
    """为每个类采样最多 per_class 个 episode: support=K 源图全 tile, query=第 K+1 源图 1 tile。

    :return: list of dict{cls_id, support_stems, query_stem}
    """
    episodes = []
    cls_ids = sorted(class_index.keys())
    if max_classes > 0:
        cls_ids = cls_ids[:max_classes]
    for cls_id in cls_ids:
        src_to_tiles = class_index[cls_id]
        sources = list(src_to_tiles.keys())
        if len(sources) < k_shot + 1:
            continue
        for _ in range(per_class):
            picked = rng.sample(sources, k_shot + 1)
            support_stems = []
            for s in picked[:k_shot]:
                tiles = src_to_tiles[s]
                if max_support_tiles > 0 and len(tiles) > max_support_tiles:
                    tiles = rng.sample(tiles, max_support_tiles)
                support_stems.extend(tiles)
            query_stem = rng.choice(src_to_tiles[picked[k_shot]])
            episodes.append({"cls_id": cls_id, "support_stems": support_stems,
                             "query_stem": query_stem})
    return episodes


# ═══════════════════════════════════════════════════════════════════
# 单 checkpoint 探针 | Per-checkpoint probe
# ═══════════════════════════════════════════════════════════════════

def probe_checkpoint(ckpt_path: str, args, logger) -> dict:
    """在一个 checkpoint 上测量 normal / detach_refine 两条件下的分支梯度 | Measure branch grads."""
    device = args.device
    data_root, _, _, val_split = _resolve_paths(args)
    split = val_split if args.split == "val" else args.split

    model, ckpt, unfreeze_layers = load_backbone(ckpt_path, device)
    decoder = build_decoder(ckpt, device)
    class_index = _build_class_index_instance(data_root, split)

    rng = random.Random(args.seed)
    episodes = sample_episodes(
        class_index, args.k_shot, args.per_class, args.max_classes,
        args.max_support_tiles, rng,
    )
    print(f"  Episodes sampled: {len(episodes)}")

    # 每个条件独立累积 | accumulate per condition
    conditions = ["normal", "detach_refine"]
    agg = {c: {"coeff_norm": [], "refine_norm": [], "coeff_rms": [],
               "refine_rms": [], "grad_at_coeffs": []} for c in conditions}
    # 饱和诊断 + 幅值分解 (归因) | saturation diag + magnitude attribution
    sat_diag = {k: [] for k in ("proto_presig_absmax", "proto_mask_sat_frac", "coeff_absmax",
                                "coeff_l2", "proto_basis_l2_max", "proto_basis_l2_mean",
                                "proto_basis_absmax")}
    # 干预实验: 解除饱和温度 → 梯度是否复活 | intervention: de-saturation temps → grad revival
    temps = [float(t) for t in args.desat_temps.split(",")] if args.desat_temps else []
    interv = {t: {"grad_at_coeffs": [], "sat_frac": []} for t in temps}
    # 功能恢复检查: coeff 塌缩 (off-diag cos) + Normal-vs-Zero 输出散度 | function-recovery accumulators
    fn_div, fn_coeffs = [], {}

    for epi in episodes:
        cls_id = epi["cls_id"]
        # 加载 support / query (target_class_id=cls_id, 与训练一致) | load support/query
        support_imgs = []
        for stem in epi["support_stems"]:
            img, _ = load_instance_tile_and_mask(stem, split, data_root, target_class_id=cls_id)
            support_imgs.append(img)
        query_img, query_mask = load_instance_tile_and_mask(
            epi["query_stem"], split, data_root, target_class_id=cls_id)
        query_gt = semantic_mask_to_binary(query_mask, is_tile=True)   # [H,W] float32
        if query_gt.sum() < 1:
            continue  # 空 query 无监督信号 | empty query carries no signal

        # 特征提取 (no_grad: 我们只需 ∂L/∂decoder) | features detached; only decoder grads needed
        support_feats = extract_features(model, support_imgs, device, no_grad=True)
        query_feats = extract_features(model, [query_img], device, no_grad=True)[0]
        support_proto = compute_support_prototype(support_feats, source=args.proto_source)
        p4 = query_feats["p4"].to(device)
        proto_masks = query_feats["proto"].to(device)

        for cond in conditions:
            decoder.zero_grad(set_to_none=False)
            loss, coeffs, diag = forward_and_loss(
                decoder, p4, proto_masks, support_proto, query_gt, device,
                detach_refine=(cond == "detach_refine"),
            )
            loss.backward()
            norms = group_grad_norms(decoder.named_parameters(), BRANCH_GROUPS)
            gac = float(coeffs.grad.detach().double().pow(2).sum().sqrt().item()) \
                if coeffs.grad is not None else 0.0
            agg[cond]["coeff_norm"].append(norms["coeff"]["norm"])
            agg[cond]["refine_norm"].append(norms["refine"]["norm"])
            agg[cond]["coeff_rms"].append(norms["coeff"]["rms"])
            agg[cond]["refine_rms"].append(norms["refine"]["rms"])
            agg[cond]["grad_at_coeffs"].append(gac)
            if cond == "normal":  # 饱和度/分解与条件无关, 记一次 | condition-independent, record once
                for k in sat_diag:
                    sat_diag[k].append(diag[k])

        # 干预: 对每个温度重新前向+反传, 看梯度是否复活 | intervention: does grad revive when de-saturated?
        for T in temps:
            decoder.zero_grad(set_to_none=False)
            loss_t, coeffs_t, diag_t = forward_and_loss(
                decoder, p4, proto_masks, support_proto, query_gt, device,
                detach_refine=False, temp=T)
            loss_t.backward()
            gac_t = float(coeffs_t.grad.detach().double().pow(2).sum().sqrt().item()) \
                if coeffs_t.grad is not None else 0.0
            interv[T]["grad_at_coeffs"].append(gac_t)
            interv[T]["sat_frac"].append(diag_t["proto_mask_sat_frac"])

        # 功能检查: prototype 是否真的影响输出 (Normal vs Zero) + coeff 塌缩 | function check
        if args.function_check:
            with torch.no_grad():
                m_n = decoder(p4, proto_masks, support_proto)
                m_z = decoder(p4, proto_masks, torch.zeros_like(support_proto))
                fn_div.append(float((m_n - m_z).abs().mean().item()))
                c = decoder.coeff_predictor(support_proto).squeeze(0)
                fn_coeffs.setdefault(cls_id, []).append(c.detach().cpu().numpy())

    # 聚合 (episode 均值) | aggregate (mean over episodes)
    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    result = {"checkpoint": ckpt_path, "unfreeze_layers": unfreeze_layers,
              "n_episodes": len(agg["normal"]["coeff_norm"]),
              "proto_presig_absmax": _mean(sat_diag["proto_presig_absmax"]),
              "proto_mask_sat_frac": _mean(sat_diag["proto_mask_sat_frac"]),
              # 幅值分解 (归因 9e8 → coeff 爆炸 vs proto_basis 爆炸) | magnitude attribution
              "coeff_absmax": _mean(sat_diag["coeff_absmax"]),
              "coeff_l2": _mean(sat_diag["coeff_l2"]),
              "proto_basis_l2_max": _mean(sat_diag["proto_basis_l2_max"]),
              "proto_basis_l2_mean": _mean(sat_diag["proto_basis_l2_mean"]),
              "proto_basis_absmax": _mean(sat_diag["proto_basis_absmax"]),
              "conditions": {}}
    if temps:  # 干预结果 | intervention results
        result["intervention"] = {
            str(T): {"grad_at_coeffs": _mean(interv[T]["grad_at_coeffs"]),
                     "sat_frac": _mean(interv[T]["sat_frac"])}
            for T in temps}
    if args.function_check:  # 功能恢复 | function recovery
        cls_means = {c: np.mean(np.stack(v), axis=0) for c, v in fn_coeffs.items() if v}
        cids = sorted(cls_means)
        offs = []
        for i in range(len(cids)):
            for j in range(len(cids)):
                if i == j:
                    continue
                a, b = cls_means[cids[i]], cls_means[cids[j]]
                na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                if na > 0 and nb > 0:
                    offs.append(float(np.dot(a, b) / (na * nb)))
        result["function"] = {
            "coeff_offdiag_cosine": float(np.mean(offs)) if offs else 0.0,
            "normal_vs_zero_divergence": float(np.mean(fn_div)) if fn_div else 0.0,
            "n_classes": len(cids),
        }
    logger.log_metric(f"uf{unfreeze_layers}/proto_presig_absmax",
                      result["proto_presig_absmax"], step=unfreeze_layers, tags=["grad-starvation"])
    logger.log_metric(f"uf{unfreeze_layers}/proto_mask_sat_frac",
                      result["proto_mask_sat_frac"], step=unfreeze_layers, tags=["grad-starvation"])
    logger.log_metric(f"uf{unfreeze_layers}/coeff_l2", result["coeff_l2"],
                      step=unfreeze_layers, tags=["grad-starvation"])
    logger.log_metric(f"uf{unfreeze_layers}/proto_basis_l2_max", result["proto_basis_l2_max"],
                      step=unfreeze_layers, tags=["grad-starvation"])
    for cond in conditions:
        a = agg[cond]
        cn, rn = _mean(a["coeff_norm"]), _mean(a["refine_norm"])
        crms, rrms = _mean(a["coeff_rms"]), _mean(a["refine_rms"])
        block = {
            "grad_norm_coeff": cn, "grad_norm_refine": rn,
            "grad_rms_coeff": crms, "grad_rms_refine": rrms,
            "ratio_raw_coeff_over_refine": (cn / rn) if rn > 0 else 0.0,
            "ratio_rms_coeff_over_refine": (crms / rrms) if rrms > 0 else 0.0,
            "grad_at_coeffs": _mean(a["grad_at_coeffs"]),
        }
        result["conditions"][cond] = block
        # 结构化日志 | structured logging
        for k, v in block.items():
            logger.log_metric(f"uf{unfreeze_layers}/{cond}/{k}", v,
                              step=unfreeze_layers, tags=["grad-starvation", cond])

    return result


# ═══════════════════════════════════════════════════════════════════
# 绘图 | Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_results(results: list[dict], out_dir: Path):
    """画: (a) 梯度比 vs 解冻深度; (b) normal vs detach 的 ‖∂L/∂coeffs‖。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # 缺 matplotlib 不致命 | plotting is optional
        print(f"  [SKIP] plotting: {e}")
        return

    ufs = [r["unfreeze_layers"] for r in results]
    ratio_raw = [r["conditions"]["normal"]["ratio_raw_coeff_over_refine"] for r in results]
    ratio_rms = [r["conditions"]["normal"]["ratio_rms_coeff_over_refine"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(results) > 1:
        axes[0].plot(ufs, ratio_raw, "o-", label="raw norm ratio")
        axes[0].plot(ufs, ratio_rms, "s--", label="per-param RMS ratio")
        axes[0].set_xlabel("Unfreeze depth (layers)")
    else:
        axes[0].bar(["raw", "rms"], [ratio_raw[0], ratio_rms[0]])
    axes[0].set_ylabel("coeff / refine gradient ratio")
    axes[0].set_title("Prototype-branch starvation vs unfreeze depth")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # normal vs detach: grad_at_coeffs (决定性反事实)
    x = np.arange(len(results)); w = 0.35
    gac_n = [r["conditions"]["normal"]["grad_at_coeffs"] for r in results]
    gac_d = [r["conditions"]["detach_refine"]["grad_at_coeffs"] for r in results]
    axes[1].bar(x - w / 2, gac_n, w, label="normal")
    axes[1].bar(x + w / 2, gac_d, w, label="detach-refine")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"uf{u}" for u in ufs])
    axes[1].set_ylabel("‖∂L/∂coeffs‖ (mean)")
    axes[1].set_title("Counterfactual: detaching query branch restores coeff signal")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "gradient_starvation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════════════════
# Prototype Audit — Information Flow (Stage 1 信息 + Stage 2 影响)
# 回答: "prototype 本身有没有类别信息" + "decoder 有没有利用它", 先于 30-epoch 重训。
# Answers "does the prototype itself carry class info" + "does the decoder use it",
# BEFORE committing GPU-hours to a 30-epoch retrain. Pure-logic core is unit-tested.
# ═══════════════════════════════════════════════════════════════════

def _cat_name(cls_id) -> str:
    """类别 id → 名称 (兼容 dict / list / 缺失) | category id → name (dict/list/missing safe)."""
    try:
        if isinstance(CATEGORY_NAMES, dict):
            return str(CATEGORY_NAMES.get(cls_id, cls_id))
        return str(CATEGORY_NAMES[int(cls_id)])
    except Exception:
        return str(cls_id)


def audit_prototypes(protos_by_class: dict) -> dict:
    """Stage 1 信息: 支持原型本身是否携带类别信息 | does the support prototype carry class identity?

    :param protos_by_class: {cls_id: [np.ndarray(D), ...]} 每类若干 episode 的原型向量。
    :return: L2 幅值分布 + 类间/类内 cosine + 判别间隔 + 轮廓系数 (cosine)。
             - inter_class_cos → 1: 类均值原型互相平行 (塌缩, 无区分)。
             - intra_class_cos → 1: 同类跨 episode 稳定 (support 未改变原型)。
             - discriminability_gap = intra − inter → 0: 原型无类别信息 (即便修好梯度也救不了 AP)。
             - silhouette_cosine ≤ 0: 无类别聚类。
    """
    cids = sorted(protos_by_class)
    all_norms = [float(np.linalg.norm(v)) for c in cids for v in protos_by_class[c]]
    means = {c: np.stack(protos_by_class[c]).mean(0) for c in cids}

    # 类间 off-diagonal cosine (类均值原型两两) | inter-class off-diag cosine over class means
    inter = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a, b = means[cids[i]], means[cids[j]]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na > 0 and nb > 0:
                inter.append(float(np.dot(a, b) / (na * nb)))

    # 类内 cosine (每 episode 原型 vs 该类均值) | intra-class cosine (each episode vs its class mean)
    intra = []
    for c in cids:
        mu = means[c]; nmu = float(np.linalg.norm(mu))
        for v in protos_by_class[c]:
            nv = float(np.linalg.norm(v))
            if nv > 0 and nmu > 0:
                intra.append(float(np.dot(v, mu) / (nv * nmu)))

    inter_mean = float(np.mean(inter)) if inter else float("nan")
    intra_mean = float(np.mean(intra)) if intra else float("nan")

    # 轮廓系数 (cosine 距离) — 复用 E007-B 的判据 | silhouette (cosine), reusing the E007-B idiom
    sil = None
    try:
        from sklearn.metrics import silhouette_score
        X = np.stack([v for c in cids for v in protos_by_class[c]])
        y = np.array([c for c in cids for _ in protos_by_class[c]])
        if len(set(y.tolist())) >= 2 and len(y) > len(set(y.tolist())):
            sil = float(silhouette_score(X, y, metric="cosine"))
    except Exception as e:  # sklearn 缺失或退化输入不致命 | non-fatal
        print(f"    [SKIP] silhouette: {e}")

    gap = (intra_mean - inter_mean) if (inter and intra) else float("nan")
    return {
        "n_classes": len(cids),
        "proto_l2_mean": float(np.mean(all_norms)) if all_norms else 0.0,
        "proto_l2_std": float(np.std(all_norms)) if all_norms else 0.0,
        "inter_class_cos": inter_mean,
        "intra_class_cos": intra_mean,
        "discriminability_gap": gap,
        "silhouette_cosine": sil,
    }


def _matched_random(sp: torch.Tensor) -> torch.Tensor:
    """与 support 原型等 L2 范数的随机向量 | random vector matched to the support prototype's L2 norm."""
    r = torch.randn_like(sp)
    scale = sp.norm() / (r.norm() + 1e-8)
    return r * scale


def _2d(v: torch.Tensor) -> torch.Tensor:
    """[C] → [1,C] (coeff_predictor 期望批维) | ensure a batch dim for coeff_predictor."""
    return v if v.dim() == 2 else v.unsqueeze(0)


def measure_influence(decoder, p4, proto_masks, normal_proto: torch.Tensor,
                      sources: dict) -> dict:
    """Stage 2 影响: 替换 prototype → 观察 coeff 与 mask 的变化量 | swap prototype, measure Δcoeff, Δmask.

    coeff_div_rel(src) ≈ 0 ⇒ CoeffPredictor 对 "喂哪个类" 无反应 (类身份信息已丢)。
    mask_div(src)      ≈ 0 ⇒ decoder 根本不用 prototype (refinement 支主导)。
    两者定位断点: Support→Prototype→Coeff→Mask 到底哪一级失效。
    """
    with torch.no_grad():
        normal_coeff = decoder.coeff_predictor(_2d(normal_proto))     # [1, proto_dim]
        m_normal = decoder(p4, proto_masks, normal_proto)             # final mask
        nc = float(normal_coeff.norm().item())
        out = {}
        for name, src in sources.items():
            c_src = decoder.coeff_predictor(_2d(src))
            coeff_div = float((c_src - normal_coeff).norm().item() / (nc + 1e-8))
            m_src = decoder(p4, proto_masks, src)
            mask_div = float((m_src - m_normal).abs().mean().item())
            out[name] = {"coeff_div_rel": coeff_div, "mask_div": mask_div}
    return out


def plot_prototype_scatter(protos_by_class: dict, uf: int, out_dir: Path):
    """PCA + t-SNE 原型散点 (按类着色) — 直观判断是否有类别聚类 | prototype scatter, colored by class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except Exception as e:
        print(f"    [SKIP] scatter: {e}")
        return
    cids = sorted(protos_by_class)
    X = np.stack([v for c in cids for v in protos_by_class[c]]).astype(np.float64)
    y = np.array([c for c in cids for _ in protos_by_class[c]])
    n = X.shape[0]
    if n < 3:
        print("    [SKIP] scatter: <3 prototypes")
        return

    pca = PCA(n_components=2).fit_transform(X)
    tsne = None
    try:
        perp = max(2, min(30, (n - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=42).fit_transform(X)
    except Exception as e:
        print(f"    [SKIP] t-SNE (PCA still drawn): {e}")

    ncol = 2 if tsne is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5.5), squeeze=False)
    cmap = plt.get_cmap("tab20")
    panels = [("PCA", pca)] + ([("t-SNE", tsne)] if tsne is not None else [])
    for ax, (title, emb) in zip(axes[0], panels):
        for k, c in enumerate(cids):
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=36, color=cmap(k % 20),
                       label=_cat_name(c), alpha=0.85, edgecolors="none")
        ax.set_title(f"{title} — prototypes @ uf{uf}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="best", ncol=2, framealpha=0.6)
    fig.tight_layout()
    out = out_dir / f"prototype_scatter_uf{uf}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"    [OK] {out}")


def audit_checkpoint(ckpt_path: str, args, logger, out_dir: Path) -> dict:
    """在一个 checkpoint 上跑完整 Prototype Audit (Stage 1 信息 + Stage 2 影响 + 散点图)。

    Pass A: 抽 support 特征 → 原型向量 (Stage 1 + 类均值/全局均值)。
    Pass B: 抽 query 特征 → 用 {normal,zero,random,shuffle,global} 过 decoder (Stage 2)。
    support 特征只抽一次 (Pass A 缓存原型张量), 无重训。
    """
    device = args.device
    data_root, _, _, val_split = _resolve_paths(args)
    split = val_split if args.split == "val" else args.split

    model, ckpt, unfreeze_layers = load_backbone(ckpt_path, device)
    decoder = build_decoder(ckpt, device)
    class_index = _build_class_index_instance(data_root, split)

    rng = random.Random(args.seed)
    episodes = sample_episodes(
        class_index, args.k_shot, args.per_class, args.max_classes,
        args.max_support_tiles, rng,
    )
    print(f"  Episodes sampled: {len(episodes)}")

    # ── Pass A: 收集原型 (Stage 1) | collect support prototypes ──
    protos_by_class: dict = {}
    epi_protos = []  # 与 episodes 对齐: (cls_id, tensor[C] on device) | aligned with episodes
    for epi in episodes:
        cls_id = epi["cls_id"]
        support_imgs = []
        for stem in epi["support_stems"]:
            img, _ = load_instance_tile_and_mask(stem, split, data_root, target_class_id=cls_id)
            support_imgs.append(img)
        support_feats = extract_features(model, support_imgs, device, no_grad=True)
        sp = compute_support_prototype(support_feats, source=args.proto_source)  # [C] tensor
        protos_by_class.setdefault(cls_id, []).append(
            sp.detach().float().cpu().numpy().reshape(-1))
        epi_protos.append((cls_id, sp.detach()))
    stage1 = audit_prototypes(protos_by_class)
    plot_prototype_scatter(protos_by_class, unfreeze_layers, out_dir)

    # 类均值 / 全局均值张量 (Stage 2 的 shuffle / global 源) | class-mean & global-mean tensors
    cids = sorted(protos_by_class)
    ref = epi_protos[0][1]
    def _as_proto(np_vec):
        return torch.from_numpy(np_vec.astype(np.float32)).to(ref.device).reshape(ref.shape)
    class_mean_t = {c: _as_proto(np.stack(protos_by_class[c]).mean(0)) for c in cids}
    global_mean_t = _as_proto(
        np.stack([np.stack(protos_by_class[c]).mean(0) for c in cids]).mean(0))

    # ── Pass B: 影响度 (Stage 2) | prototype influence ──
    src_names = ["zero", "random", "shuffle", "global"]
    infl = {s: {"coeff_div_rel": [], "mask_div": []} for s in src_names}
    n_used = 0
    for epi, (cls_id, sp) in zip(episodes, epi_protos):
        query_img, query_mask = load_instance_tile_and_mask(
            epi["query_stem"], split, data_root, target_class_id=cls_id)
        query_gt = semantic_mask_to_binary(query_mask, is_tile=True)
        if query_gt.sum() < 1:
            continue  # 空 query 无意义 | empty query is meaningless
        qf = extract_features(model, [query_img], device, no_grad=True)[0]
        p4 = qf["p4"].to(device)
        proto_masks = qf["proto"].to(device)

        idx = cids.index(cls_id)
        shuffle_src = class_mean_t[cids[(idx + 1) % len(cids)]] if len(cids) >= 2 else global_mean_t
        sources = {
            "zero": torch.zeros_like(sp),
            "random": _matched_random(sp),
            "shuffle": shuffle_src,        # 错误类的均值原型 | a *wrong* class's mean prototype
            "global": global_mean_t,       # 类无关全局均值 | class-agnostic global mean
        }
        blk = measure_influence(decoder, p4, proto_masks, sp, sources)
        for name in src_names:
            infl[name]["coeff_div_rel"].append(blk[name]["coeff_div_rel"])
            infl[name]["mask_div"].append(blk[name]["mask_div"])
        n_used += 1

    stage2 = {s: {"coeff_div_rel": float(np.mean(v["coeff_div_rel"])) if v["coeff_div_rel"] else 0.0,
                  "mask_div": float(np.mean(v["mask_div"])) if v["mask_div"] else 0.0}
              for s, v in infl.items()}

    result = {"checkpoint": ckpt_path, "unfreeze_layers": unfreeze_layers,
              "n_episodes": len(episodes), "n_query_used": n_used,
              "stage1_information": stage1, "stage2_influence": stage2}

    # 结构化日志 (规则 1) | structured logging
    for k, v in stage1.items():
        if isinstance(v, (int, float)) and v == v:  # 跳过 None/NaN
            logger.log_metric(f"uf{unfreeze_layers}/stage1/{k}", float(v),
                              step=unfreeze_layers, tags=["proto-audit", "information"])
    for s, blk in stage2.items():
        logger.log_metric(f"uf{unfreeze_layers}/stage2/{s}/mask_div", blk["mask_div"],
                          step=unfreeze_layers, tags=["proto-audit", "influence"])
    return result


def run_prototype_audit(ckpt_list: list, args, logger, out_dir: Path):
    """跑 Prototype Audit 并打印 uf 对比表 (uf0 健康参照 vs uf8 死亡) | run audit + comparison tables."""
    print("=" * 78)
    print("PROTOTYPE AUDIT | 原型信息流审计 (Stage 1 信息 + Stage 2 影响)")
    print(f"  Checkpoints: {len(ckpt_list)} | k-shot={args.k_shot} | per-class={args.per_class}")
    print("=" * 78)

    results = []
    for cp in ckpt_list:
        print(f"\n▶ {cp}")
        results.append(audit_checkpoint(cp, args, logger, out_dir))
    results.sort(key=lambda r: r["unfreeze_layers"])

    out_json = out_dir / "prototype_audit.json"
    out_json.write_text(json.dumps(
        {"probe": "prototype_audit_v1", "k_shot": args.k_shot,
         "per_class": args.per_class, "seed": args.seed,
         "proto_source": args.proto_source, "results": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {out_json}")

    # ── Stage 1: 原型信息 | prototype information ──
    print("\n" + "=" * 78)
    print("STAGE 1  INFORMATION  (does the support prototype carry class identity?)")
    print(f"{'uf':>4} {'#cls':>5} {'protoL2':>10} {'interCos':>9} {'intraCos':>9} "
          f"{'gap':>8} {'silhou':>8}")
    for r in results:
        s = r["stage1_information"]
        sil = s["silhouette_cosine"]
        print(f"{r['unfreeze_layers']:>4} {s['n_classes']:>5} "
              f"{s['proto_l2_mean']:>10.2e} {s['inter_class_cos']:>9.4f} "
              f"{s['intra_class_cos']:>9.4f} {s['discriminability_gap']:>8.4f} "
              f"{(sil if sil is not None else float('nan')):>8.4f}")
    print("  读法 | Read: interCos→1 且 gap→0 且 silhou≤0 → 原型本身无类别信息 (上游即已死);")
    print("        interCos 低 & gap 大 & silhou>0 → 原型健康, 问题在下游 (梯度/融合)。")

    # ── Stage 2: 原型影响 | prototype influence ──
    print("\n" + "=" * 78)
    print("STAGE 2  INFLUENCE  (does the decoder respond to *which* prototype?)")
    print("  相对 Normal 的变化量; src=zero/random/shuffle(wrong-class)/global(class-agnostic)")
    print(f"{'uf':>4} {'src':>8} {'coeff_div_rel':>14} {'mask_div':>12}")
    for r in results:
        for s in ["zero", "random", "shuffle", "global"]:
            b = r["stage2_influence"][s]
            print(f"{r['unfreeze_layers']:>4} {s:>8} {b['coeff_div_rel']:>14.3e} "
                  f"{b['mask_div']:>12.3e}")
    print("  读法 | Read: coeff_div_rel(shuffle)≈0 → CoeffPredictor 无视类身份;")
    print("        mask_div(全部)≈0 → decoder 完全不用 prototype (refinement 支独占)。")
    print("=" * 78)
    logger.log_info("summary", f"prototype-audit over {len(results)} checkpoints → {out_json}")


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main():
    # Windows 控制台默认 GBK 无法编码 ▶/‖ 等字符 → 强制 UTF-8 | force UTF-8 stdout (Windows GBK safe)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Prototype-branch gradient-starvation probe")
    parser.add_argument("--checkpoint", type=str, default=None, help="单 checkpoint 路径 | single checkpoint")
    parser.add_argument("--sweep", action="store_true", help="扫描 uf0/5/8/12 | sweep unfreeze depths")
    parser.add_argument("--sweep-checkpoints", type=str, default=None,
                        help="逗号分隔的 uf:path 覆盖默认扫描 | comma-sep uf:path overrides")
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=3, help="每类 episode 数 | episodes per class")
    parser.add_argument("--max-classes", type=int, default=0, help="限制类别数 (0=全部) | cap #classes")
    parser.add_argument("--max-support-tiles", type=int, default=12,
                        help="每源图 support tile 上限 (0=不限, 控成本) | cap support tiles per source")
    parser.add_argument("--proto-source", type=str, default="p4", choices=["p4", "p8"])
    parser.add_argument("--desat-temps", type=str, default=None,
                        help="干预: 逗号分隔的 sigmoid 温度, 解除饱和看梯度是否复活 (如 '1,10,100,1000')")
    parser.add_argument("--function-check", action="store_true",
                        help="功能恢复: coeff off-diag cosine + Normal-vs-Zero 输出散度 (prototype 是否生效)")
    parser.add_argument("--prototype-audit", action="store_true",
                        help="信息流审计: Stage1 原型是否携带类别信息 + Stage2 decoder 是否利用它 (先于重训)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--data-format", type=str, default="isaid_instance")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)  # 可复现 (CLAUDE 规则 5) | reproducibility

    out_dir = Path(args.output_dir) if args.output_dir else \
        _PROJECT_ROOT / "runs" / "diag" / "grad_starvation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志 (CLAUDE 规则 1: 所有可观测值走 adatile.logging) | logging
    logger = get_logger("grad_starvation")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "eval.jsonl")))

    # 组装 checkpoint 列表 | assemble checkpoint list
    if args.sweep or args.sweep_checkpoints:
        if args.sweep_checkpoints:
            items = []
            for tok in args.sweep_checkpoints.split(","):
                uf, path = tok.split(":", 1)
                items.append((int(uf), path.strip()))
        else:
            items = SWEEP_CHECKPOINTS
        ckpt_list = []
        for uf, rel in items:
            p = (_PROJECT_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
            if p.exists():
                ckpt_list.append(str(p))
            else:
                print(f"  [SKIP] missing checkpoint (uf{uf}): {p}")
        if not ckpt_list:
            raise SystemExit("No sweep checkpoints found on disk.")
    elif args.checkpoint:
        ckpt_list = [args.checkpoint]
    else:
        raise SystemExit("Provide --checkpoint or --sweep.")

    # 信息流审计模式 (Stage 1 + Stage 2), 先于 30-epoch 重训 | prototype audit mode
    if args.prototype_audit:
        run_prototype_audit(ckpt_list, args, logger, out_dir)
        return

    print("=" * 70)
    print("Gradient-Starvation Probe | 原型分支梯度饥饿探针")
    print(f"  Checkpoints: {len(ckpt_list)} | k-shot={args.k_shot} | per-class={args.per_class}")
    print("=" * 70)

    results = []
    for cp in ckpt_list:
        print(f"\n▶ {cp}")
        results.append(probe_checkpoint(cp, args, logger))

    # 排序 + 落盘 | sort by unfreeze depth + dump
    results.sort(key=lambda r: r["unfreeze_layers"])
    out_json = out_dir / "gradient_starvation.json"
    payload = {
        "probe": "gradient_starvation_v1",
        "k_shot": args.k_shot, "per_class": args.per_class, "seed": args.seed,
        "proto_source": args.proto_source, "branch_groups": BRANCH_GROUPS,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] wrote {out_json}")

    plot_results(results, out_dir)

    # ── 摘要 | Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY  (normal condition; coeff = prototype branch)")
    print(f"{'uf':>4} {'ratio_raw':>10} {'ratio_rms':>10} {'gAc_norm':>10} {'gAc_detach':>11} {'sat_frac':>9} {'presig':>9}")
    for r in results:
        n = r["conditions"]["normal"]; d = r["conditions"]["detach_refine"]
        print(f"{r['unfreeze_layers']:>4} "
              f"{n['ratio_raw_coeff_over_refine']:>10.4f} "
              f"{n['ratio_rms_coeff_over_refine']:>10.4f} "
              f"{n['grad_at_coeffs']:>10.2e} {d['grad_at_coeffs']:>11.2e} "
              f"{r['proto_mask_sat_frac']:>9.3f} {r['proto_presig_absmax']:>9.1f}")
    print("  读法 | Read: gAc≈0 且 sat_frac≈1 → proto 支落入 sigmoid 饱和死区 (梯度精确为 0)。")
    print("  若 detach 列 ≫ normal 列 → 竞争性饿死; 若两列都≈0 → 饱和死锁 (detach 救不回)。")
    print("=" * 70)

    # ── 幅值分解 (归因 pre-sigmoid 爆炸) | magnitude attribution ──
    print("\nDECOMPOSITION  (why pre-sigmoid is huge: coeff vs proto_basis)")
    print(f"{'uf':>4} {'presig':>12} {'coeff_l2':>10} {'coeff_max':>10} {'basis_l2max':>12} {'basis_max':>10}")
    for r in results:
        print(f"{r['unfreeze_layers']:>4} {r['proto_presig_absmax']:>12.2e} "
              f"{r['coeff_l2']:>10.2e} {r['coeff_absmax']:>10.2e} "
              f"{r['proto_basis_l2_max']:>12.2e} {r['proto_basis_absmax']:>10.2e}")
    print("  读法 | Read: coeff_l2/max 大 → 系数爆炸 (CoeffPredictor); basis 大 → 特征爆炸 (backbone)。")

    # ── 干预结果 | intervention ──
    if any("intervention" in r for r in results):
        print("\nINTERVENTION  (de-saturate: proto_mask = sigmoid(pre_sigmoid / T))")
        for r in results:
            if "intervention" not in r:
                continue
            print(f"  uf{r['unfreeze_layers']}:")
            print(f"    {'T':>8} {'sat_frac':>10} {'grad_at_coeffs':>16}")
            for T, blk in r["intervention"].items():
                print(f"    {T:>8} {blk['sat_frac']:>10.3f} {blk['grad_at_coeffs']:>16.3e}")
        print("  读法 | Read: 若增大 T 使 sat_frac↓ 且 grad_at_coeffs↑ → 饱和确为梯度归零的直接原因 (因果)。")
        print("  注意 | Caveat: 静态干预只复活**梯度**; MLP 已塌缩为常数, 不会让 Normal≠Zero (那需重训)。")

    # ── 功能恢复 | function recovery ──
    if any("function" in r for r in results):
        print("\nFUNCTION  (does the prototype actually affect the output?)")
        print(f"{'uf':>4} {'coeff_offdiag_cos':>18} {'Normal-vs-Zero_div':>19}")
        for r in results:
            if "function" not in r:
                continue
            f = r["function"]
            print(f"{r['unfreeze_layers']:>4} {f['coeff_offdiag_cosine']:>18.4f} "
                  f"{f['normal_vs_zero_divergence']:>19.3e}")
        print("  读法 | Read: cos≈1 且 div≈0 → prototype 仍死; cos<1 且 div>0 → prototype 已生效 (功能恢复)。")
    print("=" * 70)
    logger.log_info("summary", f"probed {len(results)} checkpoints → {out_json}")


if __name__ == "__main__":
    main()
