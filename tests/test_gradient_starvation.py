"""
梯度饥饿探针单测 | Gradient-Starvation Probe unit tests.
=====================================================

验证纯逻辑, 无需 GPU / 数据集 / FastSAM backbone:
  1. group_grad_norms 按前缀正确分组、求和、算 per-parameter RMS。
  2. forward_and_loss 在 detach_refine 下: query 支梯度精确为 0, coeff 支梯度 > 0。
     (这正是"竞争性饿死"反事实的机械验证。)

Pure-logic tests only (no GPU/dataset/backbone):
  1. group_grad_norms groups by prefix, sums, and computes per-param RMS correctly.
  2. Under detach_refine, the query/refine branch gets exactly zero gradient while the coeff
     branch gets > 0 — the mechanical core of the counterfactual.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from tools.diag.diag_gradient_starvation import (
    group_grad_norms,
    forward_and_loss,
    BRANCH_GROUPS,
    audit_prototypes,
    _matched_random,
    measure_influence,
)
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder


# ═══════════════════════════════════════════════════════════════════
# group_grad_norms | 分组梯度范数
# ═══════════════════════════════════════════════════════════════════

def _param_with_grad(numel: int, grad_val: float):
    """构造一个 .grad 已填充的叶子参数 | Build a leaf param with a filled .grad."""
    p = nn.Parameter(torch.zeros(numel))
    p.grad = torch.full((numel,), grad_val)
    return p


def test_group_grad_norms_prefix_and_sums():
    """按前缀分组 + L2 范数 + RMS 计算正确, 不匹配前缀被忽略。"""
    named = [
        ("coeff_predictor.mlp.0.weight", _param_with_grad(4, 1.0)),   # Σg²=4
        ("feat_refine.0.weight", _param_with_grad(9, 2.0)),           # Σg²=36
        ("mask_head.0.weight", _param_with_grad(1, 0.0)),             # Σg²=0
        ("backbone.model.21.weight", _param_with_grad(5, 9.0)),       # 不属任何支 → 忽略
    ]
    out = group_grad_norms(named, BRANCH_GROUPS)

    # coeff: norm=sqrt(4)=2, n=4, rms=2/sqrt(4)=1
    assert math.isclose(out["coeff"]["norm"], 2.0, rel_tol=1e-6)
    assert out["coeff"]["n_params"] == 4
    assert math.isclose(out["coeff"]["rms"], 1.0, rel_tol=1e-6)

    # refine: norm=sqrt(36+0)=6, n=9+1=10, rms=6/sqrt(10)
    assert math.isclose(out["refine"]["norm"], 6.0, rel_tol=1e-6)
    assert out["refine"]["n_params"] == 10
    assert math.isclose(out["refine"]["rms"], 6.0 / math.sqrt(10), rel_tol=1e-6)


def test_group_grad_norms_skips_none_grad():
    """param.grad is None 的参数被跳过 (detach 支典型情形)。"""
    p_none = nn.Parameter(torch.zeros(3))  # grad 保持 None
    named = [
        ("coeff_predictor.mlp.0.weight", _param_with_grad(4, 1.0)),
        ("feat_refine.0.weight", p_none),
    ]
    out = group_grad_norms(named, BRANCH_GROUPS)
    assert out["refine"]["norm"] == 0.0
    assert out["refine"]["n_params"] == 0
    assert out["refine"]["rms"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# forward_and_loss detach 反事实 | detach counterfactual
# ═══════════════════════════════════════════════════════════════════

def _tiny_decoder():
    """极小 AdaptiveSparseDecoder (CPU) | Tiny decoder for CPU tests."""
    torch.manual_seed(0)
    return AdaptiveSparseDecoder(in_channels=8, proto_dim=4, hidden_dim=8, use_fdr=False)


def _tiny_inputs():
    torch.manual_seed(1)
    p4 = torch.randn(1, 8, 4, 4)
    proto_masks = torch.randn(1, 4, 16, 16)
    support_proto = torch.randn(1, 8)
    query_gt = np.zeros((16, 16), dtype=np.float32)
    query_gt[4:10, 4:10] = 1.0  # 一块前景 | a foreground blob
    return p4, proto_masks, support_proto, query_gt


def test_detach_refine_zeroes_query_branch_grad():
    """detach_refine=True: refine 支梯度精确为 0, coeff 支梯度 > 0。"""
    decoder = _tiny_decoder()
    p4, proto_masks, support_proto, query_gt = _tiny_inputs()

    decoder.zero_grad(set_to_none=True)
    loss, coeffs, diag = forward_and_loss(
        decoder, p4, proto_masks, support_proto, query_gt, "cpu", detach_refine=True)
    loss.backward()
    out = group_grad_norms(decoder.named_parameters(), BRANCH_GROUPS)

    assert out["refine"]["norm"] == 0.0, "query 支被 detach, 不应有梯度"
    assert out["coeff"]["norm"] > 0.0, "coeff 支仍应通过 proto_mask 收到梯度"
    assert coeffs.grad is not None and float(coeffs.grad.abs().sum()) > 0.0
    assert "proto_mask_sat_frac" in diag and "proto_presig_absmax" in diag


def test_normal_trains_both_branches():
    """normal: 两支都有梯度 (refine > 0)。"""
    decoder = _tiny_decoder()
    p4, proto_masks, support_proto, query_gt = _tiny_inputs()

    decoder.zero_grad(set_to_none=True)
    loss, _, _ = forward_and_loss(
        decoder, p4, proto_masks, support_proto, query_gt, "cpu", detach_refine=False)
    loss.backward()
    out = group_grad_norms(decoder.named_parameters(), BRANCH_GROUPS)

    assert out["refine"]["norm"] > 0.0
    assert out["coeff"]["norm"] > 0.0


def test_loss_is_finite_scalar():
    """loss 为有限标量 (BCE+Dice 数值稳定)。"""
    decoder = _tiny_decoder()
    p4, proto_masks, support_proto, query_gt = _tiny_inputs()
    loss, _, _ = forward_and_loss(
        decoder, p4, proto_masks, support_proto, query_gt, "cpu", detach_refine=False)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


# ═══════════════════════════════════════════════════════════════════
# normalize_proto — 修复机制 | fix mechanism
# ═══════════════════════════════════════════════════════════════════

def _decoder(mode):
    torch.manual_seed(0)
    return AdaptiveSparseDecoder(in_channels=8, proto_dim=4, hidden_dim=8,
                                 use_fdr=False, normalize_proto=mode)


def test_normalize_proto_modes_shapes_and_bounds():
    """l2 → 每 basis 单位 L2; layernorm → ~零均值; scale → 缩放; none → 恒等。"""
    torch.manual_seed(2)
    pm = torch.randn(4, 16, 16) * 1e6  # 巨幅 basis | huge-magnitude basis

    d_none = _decoder("none")
    assert torch.equal(d_none._normalize_proto(pm), pm)  # 恒等 | identity

    d_l2 = _decoder("l2")
    out = d_l2._normalize_proto(pm)
    per_basis_norm = out.reshape(4, -1).norm(dim=1)
    assert torch.allclose(per_basis_norm, torch.ones(4), atol=1e-4)  # 每 basis 单位 L2

    d_ln = _decoder("layernorm")
    out_ln = d_ln._normalize_proto(pm)
    assert abs(float(out_ln.reshape(4, -1).mean())) < 1e-2          # ~零均值 | ~zero-mean

    d_sc = _decoder("scale")
    assert torch.allclose(d_sc._normalize_proto(pm), pm / (16 * 16) ** 0.5)


def test_normalize_proto_desaturates_and_revives_gradient():
    """机制恢复: 巨幅 basis 下 none 饱和/零梯度, l2/layernorm 脱饱和/梯度>0。

    Mechanism recovery: with a huge basis, `none` saturates (∂L/∂coeffs≈0); `l2`/`layernorm`
    de-saturate and the coefficient branch regains gradient.
    """
    torch.manual_seed(3)
    p4 = torch.randn(1, 8, 4, 4)
    proto_masks = torch.randn(1, 4, 16, 16) * 1e6   # 触发饱和 | trigger saturation
    support_proto = torch.randn(1, 8)
    query_gt = np.zeros((16, 16), dtype=np.float32)
    query_gt[4:10, 4:10] = 1.0

    def _run(mode):
        d = _decoder(mode)
        d.collect_stats = True
        d.zero_grad(set_to_none=True)
        loss, coeffs, diag = forward_and_loss(
            d, p4, proto_masks, support_proto, query_gt, "cpu", detach_refine=False)
        loss.backward()
        gac = float(coeffs.grad.abs().sum()) if coeffs.grad is not None else 0.0
        return diag["proto_mask_sat_frac"], gac

    sat_none, g_none = _run("none")
    sat_l2, g_l2 = _run("l2")
    sat_ln, g_ln = _run("layernorm")

    assert sat_none > 0.99 and g_none < 1e-6          # none: 饱和死区, 梯度归零
    assert sat_l2 < 0.5 and g_l2 > g_none              # l2: 脱饱和, 梯度复活
    assert sat_ln < 0.5 and g_ln > g_none              # layernorm: 同上


def test_normalize_none_is_backward_compatible():
    """默认 none 不增参数, 与旧 decoder state_dict 键一致 (向后兼容)。"""
    d_none = AdaptiveSparseDecoder(in_channels=8, proto_dim=4, hidden_dim=8, use_fdr=False)
    d_default = AdaptiveSparseDecoder(in_channels=8, proto_dim=4, hidden_dim=8, use_fdr=False,
                                      normalize_proto="none")
    assert set(d_none.state_dict()) == set(d_default.state_dict())
    assert d_none.proto_norm is None                   # none 模式无归一化模块 | no extra params


# ═══════════════════════════════════════════════════════════════════
# Prototype Audit — Stage 1 信息 + Stage 2 影响 | information + influence
# ═══════════════════════════════════════════════════════════════════

def test_audit_prototypes_separated_vs_collapsed():
    """判别性: 分离原型 gap>0/silhouette>0; 塌缩原型 interCos≈1/gap≈0。

    Discriminability: well-separated class prototypes give a positive gap and silhouette;
    collapsed (near-parallel) prototypes give inter-class cosine ≈ 1 and gap ≈ 0.
    """
    rng = np.random.RandomState(0)

    # (a) 分离: 三类沿正交基, 类内小抖动 | separated: 3 classes on orthogonal axes
    sep = {}
    for c, axis in zip((1, 2, 3), np.eye(3, 16)):
        sep[c] = [axis * 5.0 + rng.randn(16) * 0.05 for _ in range(6)]
    a_sep = audit_prototypes(sep)
    assert a_sep["inter_class_cos"] < 0.3          # 类均值近正交 | near-orthogonal means
    assert a_sep["intra_class_cos"] > 0.9          # 类内稳定 | tight within class
    assert a_sep["discriminability_gap"] > 0.5     # 明确类别信息 | clear class info
    assert a_sep["silhouette_cosine"] is not None and a_sep["silhouette_cosine"] > 0.3

    # (b) 塌缩: 三类几乎同一方向 (uf8 现象) | collapsed: near-parallel across classes
    base = rng.randn(16) * 5.0
    col = {c: [base + rng.randn(16) * 0.02 for _ in range(6)] for c in (1, 2, 3)}
    a_col = audit_prototypes(col)
    assert a_col["inter_class_cos"] > 0.99         # 类均值几乎平行 | collapsed
    assert abs(a_col["discriminability_gap"]) < 0.02   # 无可区分间隔 | no gap → no class info


def test_matched_random_matches_norm():
    """_matched_random 的 L2 范数与源原型一致 | matched-norm random equals source L2 norm."""
    sp = torch.randn(1280) * 3.7
    r = _matched_random(sp)
    assert r.shape == sp.shape
    assert torch.allclose(r.norm(), sp.norm(), rtol=1e-4)


def test_measure_influence_zero_vs_normal_moves_output():
    """Stage 2: 未塌缩的小 decoder 对不同 prototype 有响应 (mask_div>0, coeff_div_rel>0)。

    On a fresh (non-collapsed) tiny decoder, swapping the prototype changes both the
    coefficients and the mask — the audit correctly detects a *live* prototype pathway.
    """
    torch.manual_seed(0)
    decoder = AdaptiveSparseDecoder(in_channels=8, proto_dim=4, hidden_dim=8, use_fdr=False)
    decoder.eval()
    p4 = torch.randn(1, 8, 4, 4)
    proto_masks = torch.randn(1, 4, 16, 16)
    normal_proto = torch.randn(8)
    sources = {
        "zero": torch.zeros_like(normal_proto),
        "random": _matched_random(normal_proto),
        "global": torch.randn(8),
    }
    out = measure_influence(decoder, p4, proto_masks, normal_proto, sources)
    assert set(out) == {"zero", "random", "global"}
    # zero≠normal ⇒ 系数与掩码都应变化 | zero differs from normal ⇒ coeff & mask move
    assert out["zero"]["coeff_div_rel"] > 0.0
    assert out["zero"]["mask_div"] > 0.0
    for s in out.values():
        assert math.isfinite(s["coeff_div_rel"]) and math.isfinite(s["mask_div"])

