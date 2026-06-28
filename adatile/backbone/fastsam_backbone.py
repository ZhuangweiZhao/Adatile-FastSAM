"""
FastSAMBackbone — 基于 thirdLibrary/FastSAM 的特征提取骨架。
=================================================================
FastSAMBackbone: feature extraction backbone based on thirdLibrary/FastSAM.

加载 FastSAM 模型，通过前向钩子（forward hooks）提取多尺度中间特征图。
Loads FastSAM model, extracts multi-scale intermediate feature maps via forward hooks.

V1 关键教训（必须遵守）| V1 critical lessons (must follow):
    1. model.train() 会崩溃 YOLOv8 的 Detect 头 → 始终保持 eval 模式
       model.train() crashes YOLOv8 Detect head → always keep eval mode
    2. 通过 requires_grad 控制选择性微调，不调用 .train()
       Use requires_grad for selective fine-tuning, never call .train()
    3. 钩子位置：stride ≈ 16 和 stride ≈ 32 的层
       Hook locations: layers with stride ≈ 16 and stride ≈ 32

Usage::
    >>> backbone = FastSAMBackbone()
    >>> features = backbone(image_tensor)  # image: [B, 3, H, W]
    >>> print(features["p4"].shape)  # [B, C, H/16, W/16]
    >>> print(features["p8"].shape)  # [B, C, H/32, W/32]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from adatile.logging import get_logger

# ── 确保 thirdLibrary 在 Python 路径中 | Ensure thirdLibrary on Python path ──
_THIRD_LIB = Path(__file__).resolve().parents[2] / "thirdLibrary" / "FastSAM"
if str(_THIRD_LIB) not in sys.path:
    sys.path.insert(0, str(_THIRD_LIB))


class FastSAMBackbone(nn.Module):
    """
    FastSAM 骨干网络，带特征钩子 | FastSAM backbone with feature hooks.

    从 thirdLibrary/FastSAM 加载 FastSAM-x 模型，
    注册前向钩子提取 P4 (stride≈16) 和 P8 (stride≈32) 中间特征图。
    Loads FastSAM-x from thirdLibrary/FastSAM,
    registers forward hooks to extract intermediate feature maps
    at P4 (stride≈16) and P8 (stride≈32).

    ----------
    checkpoint : str | None
        FastSAM 权重路径。None → 自动使用 'FastSAM-x.pt'。
        Path to FastSAM checkpoint. None → auto-use 'FastSAM-x.pt'.
    freeze_backbone : bool
        是否冻结骨干参数。默认 True（只训练 decoder/SPM）。
        Whether to freeze backbone params. Default True (only train decoder/SPM).
    device : str | None
        设备。None → 自动检测 CUDA 或 CPU。
        Device. None → auto-detect CUDA or CPU.

    钩子探测 | Hook Probing:
        首次 forward 时会自动探测所有层的输出步长，
        选择 stride ≈ 16 和 stride ≈ 32 的层作为特征提取点。
        On first forward, auto-probes all layer output strides,
        selects layers with stride ≈ 16 and stride ≈ 32 as extraction points.
    """

    # ── 候选步长范围 | Candidate stride ranges ──
    TARGET_STRIDE_8  = (7, 9)     # stride 8 的容许范围 | tolerance for stride 8
    TARGET_STRIDE_16 = (14, 18)   # stride 16 的容许范围 | tolerance for stride 16
    TARGET_STRIDE_32 = (28, 36)   # stride 32 的容许范围 | tolerance for stride 32

    def __init__(
        self,
        checkpoint: str | None = None,
        freeze_backbone: bool = True,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.logger = get_logger("backbone")
        self._freeze_backbone = freeze_backbone
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 特征缓存 | Feature cache
        self._features: dict[str, torch.Tensor] = {}

        # 钩子层索引（首次 forward 时自动探测）| Hooked layer indices (auto-detected on first forward)
        self._hook_p3_idx: int | None = None
        self._hook_p4_idx: int | None = None
        self._hook_p8_idx: int | None = None
        self._hook_handles: list = []  # 钩子句柄 | Hook handles

        # 加载 FastSAM 模型 | Load FastSAM model
        # 默认使用 thirdLibrary 中的权重文件 | Default to weight file in thirdLibrary
        if checkpoint is None:
            checkpoint = str(_THIRD_LIB / "weights" / "FastSAM-x.pt")
        self._checkpoint = checkpoint
        self.model = self._load_fastsam()

        # 始终保持在 eval 模式（V1 教训 | V1 lesson）
        # model.train() 会触发 YOLOv8 Detect 头代码路径，导致 crash
        # model.train() triggers YOLOv8 Detect head paths → crash
        self._force_eval_mode()

        # 冻结参数 | Freeze parameters
        if freeze_backbone:
            self._apply_freeze()
        else:
            # 确保参数可训练（某些模型加载后默认 requires_grad=False）
            # Ensure params are trainable (some models default to requires_grad=False after loading)
            self._unfreeze()

        self.logger.log_info(
            "backbone/init",
            f"FastSAMBackbone loaded from {self._checkpoint}, "
            f"freeze={freeze_backbone}, device={self._device}",
        )

    # ── 模型加载 | Model Loading ──────────────────────────────

    def _load_fastsam(self):
        """
        从 thirdLibrary/FastSAM 加载 FastSAM 模型。
        Load FastSAM model from thirdLibrary/FastSAM.

        :return: FastSAM 实例 | FastSAM instance.
        """
        # 导入 thirdLibrary 中的 FastSAM | Import FastSAM from thirdLibrary
        from fastsam import FastSAM  # type: ignore[import-not-found]

        model = FastSAM(self._checkpoint)

        # FastSAM 内部可能在加载时自动将权重迁移到 CUDA（即使无 GPU 仍标记为 cuda tensor）
        # FastSAM may auto-move weights to CUDA during load (even if no GPU, marked as cuda tensor)
        # 注意: FastSAM 不是 nn.Module，实际模型在 model.model (YOLO wrapper)
        # Note: FastSAM is NOT nn.Module; real model is model.model (YOLO wrapper)
        target = torch.device(self._device)
        model.model.to(target)

        # 额外安全: 显式检查并迁移残余 CUDA 参数 (处理 .to() 可能遗漏的边缘情况)
        # Extra safety: explicitly check & migrate residual CUDA params
        # FastSAM 非标准 __getattr__ → 必须直接访问 model.model
        yolo = model.model
        for p in yolo.parameters():
            if p.device != target:
                p.data = p.data.to(target)
        for b in yolo.buffers():
            if b.device != target:
                b.data = b.data.to(target)

        self.logger.log_info(
            "backbone/load",
            f"FastSAM loaded: {self._checkpoint} → {self._device}"
        )
        return model

    # ── 钩子管理 | Hook Management ────────────────────────────

    def _register_probe_hooks(self) -> None:
        """
        注册探测钩子：在所有子层上注册钩子，用于首次前向时探测步长。
        Register probe hooks on all child layers for stride detection on first forward.

        只注册钩子到 Sequential 模块的直接子层。
        Only registers hooks on direct children of the Sequential module.
        """
        # 获取 YOLO Sequential 模型 | Get YOLO Sequential model
        sequential = self.model.model.model

        for idx, layer in enumerate(sequential):
            handle = layer.register_forward_hook(self._make_hook(idx))
            self._hook_handles.append(handle)

    def _make_hook(self, idx: int):
        """
        创建钩子函数（闭包捕获 idx）| Create hook function (closure captures idx).

        :param idx: 层索引 | Layer index.
        :type idx: int

        :return: callable: 钩子函数 | Hook function.
        """

        def hook(module, input, output):
            # 只缓存 Tensor 输出（跳过 list/tuple 等） | Only cache Tensor outputs (skip list/tuple etc.)
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                self._features[str(idx)] = output

        return hook

    def _remove_all_hooks(self) -> None:
        """移除所有已注册的钩子 | Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _probe_strides(self, x: torch.Tensor) -> None:
        """
        探测模式：运行一次前向，找到 stride≈16 和 stride≈32 的层。
        Probe mode: run one forward pass, find layers with stride≈16 and stride≈32.

        :param x: 输入张量 [B, 3, H, W] | Input tensor.
        :type x: torch.Tensor
        """
        _, _, h_in, w_in = x.shape
        self._features.clear()

        # 注册所有层的钩子 | Register hooks on all layers
        self._register_probe_hooks()

        # 执行前向（通过 YOLO Sequential，绕过 predict 的预处理）
        # Forward through YOLO Sequential (bypasses predict's preprocessing)
        with torch.no_grad():
            # 使用 _predict_once 处理多输入层路由 | Use _predict_once for multi-input routing
            self.model.model._predict_once(x)

        # 分析各层输出的步长 | Analyze strides of each layer's output
        candidates_8  = []  # stride-8 candidates
        candidates_16 = []  # stride-16 candidates
        candidates_32 = []  # stride-32 candidates

        for key, feat in self._features.items():
            _, _, h_out, w_out = feat.shape
            stride_h = h_in / h_out
            stride_w = w_in / w_out
            avg_stride = (stride_h + stride_w) / 2

            if self.TARGET_STRIDE_8[0] <= avg_stride <= self.TARGET_STRIDE_8[1]:
                candidates_8.append((int(key), avg_stride, feat.shape[1]))
            if self.TARGET_STRIDE_16[0] <= avg_stride <= self.TARGET_STRIDE_16[1]:
                candidates_16.append((int(key), avg_stride, feat.shape[1]))
            if self.TARGET_STRIDE_32[0] <= avg_stride <= self.TARGET_STRIDE_32[1]:
                candidates_32.append((int(key), avg_stride, feat.shape[1]))

        # P3 (stride≈8) — 按通道数降序选择最佳匹配 | pick best by channel count
        if candidates_8:
            candidates_8.sort(key=lambda t: -t[2])
            self._hook_p3_idx = candidates_8[0][0]
            self.logger.log_info(
                "backbone/probe",
                f"P3 hook: layer {self._hook_p3_idx}, "
                f"stride={candidates_8[0][1]:.1f}, "
                f"channels={candidates_8[0][2]}",
            )

        # 选择最佳匹配：优先选择通道数多的（通常更有信息量）
        # Select best match: prefer layers with more channels (usually more informative)
        if candidates_16:
            candidates_16.sort(key=lambda t: -t[2])  # 按通道数降序 | sort by channels desc
            self._hook_p4_idx = candidates_16[0][0]
            self.logger.log_info(
                "backbone/probe",
                f"P4 hook: layer {self._hook_p4_idx}, "
                f"stride={candidates_16[0][1]:.1f}, "
                f"channels={candidates_16[0][2]}",
            )
        else:
            # 如果没有匹配，回退到猜测的索引 | If no match, fallback to guessed index
            self._hook_p4_idx = self._guess_p4_index()
            self.logger.log_warn(
                "backbone/probe",
                f"No stride-16 layer found, using fallback index {self._hook_p4_idx}",
            )

        if candidates_32:
            candidates_32.sort(key=lambda t: -t[2])  # 按通道数降序 | sort by channels desc
            self._hook_p8_idx = candidates_32[0][0]
            self.logger.log_info(
                "backbone/probe",
                f"P8 hook: layer {self._hook_p8_idx}, "
                f"stride={candidates_32[0][1]:.1f}, "
                f"channels={candidates_32[0][2]}",
            )
        else:
            self._hook_p8_idx = self._guess_p8_index()
            self.logger.log_warn(
                "backbone/probe",
                f"No stride-32 layer found, using fallback index {self._hook_p8_idx}",
            )

        # 移除所有探测钩子 | Remove all probe hooks
        self._remove_all_hooks()
        self._features.clear()

    def _register_final_hooks(self) -> None:
        """
        注册最终的 P4/P8 特征提取钩子 | Register final P4/P8 feature extraction hooks.
        仅在探测完成后调用。| Only called after probing is complete.
        """
        sequential = self.model.model.model

        # P3 钩子 | P3 hook (stride≈8)
        if self._hook_p3_idx is not None:
            handle = sequential[self._hook_p3_idx].register_forward_hook(
                self._make_feature_hook("p3")
            )
            self._hook_handles.append(handle)

        # P4 钩子 | P4 hook
        if self._hook_p4_idx is not None:
            handle = sequential[self._hook_p4_idx].register_forward_hook(
                self._make_feature_hook("p4")
            )
            self._hook_handles.append(handle)

        # P8 钩子 | P8 hook
        if self._hook_p8_idx is not None:
            handle = sequential[self._hook_p8_idx].register_forward_hook(
                self._make_feature_hook("p8")
            )
            self._hook_handles.append(handle)

    def _make_feature_hook(self, name: str):
        """
        创建特征提取钩子 | Create feature extraction hook.

        :param name: 特征名（"p4" 或 "p8"）| Feature name.
        :type name: str

        :return: callable: 钩子函数 | Hook function.
        """

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._features[name] = output
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # 某些层输出是 (tensor, ...) 的 tuple | Some layers output tuple of (tensor, ...)
                if isinstance(output[0], torch.Tensor):
                    self._features[name] = output[0]

        return hook

    def _guess_p4_index(self) -> int:
        """推测 P4 (stride≈16) 层索引 | Guess P4 (stride≈16) layer index."""
        # YOLOv8-x 的典型结构：第 6-8 层附近 | Typical YOLOv8-x structure: around layer 6-8
        return 6

    def _guess_p8_index(self) -> int:
        """推测 P8 (stride≈32) 层索引 | Guess P8 (stride≈32) layer index."""
        # YOLOv8-x 的典型结构：SPPF 在第 9 层附近 | Typical YOLOv8-x structure: SPPF around layer 9
        return 9

    # ── 参数冻结 | Parameter Freezing ─────────────────────────

    def _apply_freeze(self) -> None:
        """
        冻结骨干网络的所有参数 | Freeze all backbone parameters.
        设置 requires_grad=False 而非调用 .eval()。
        Sets requires_grad=False rather than calling .eval().

        V1 教训：.train() 不可用，但 .eval() 中的 requires_grad=False 是安全的。
        V1 lesson: .train() is unusable, but requires_grad=False in .eval() is safe.
        """
        for param in self.model.model.parameters():
            param.requires_grad = False
        self.logger.log_info("backbone/freeze", "Backbone parameters frozen")

    def _unfreeze(self) -> None:
        """
        解冻骨干网络的所有参数 | Unfreeze all backbone parameters.
        设置 requires_grad=True 但不调用 .train()（V1 教训）。
        Sets requires_grad=True without calling .train() (V1 lesson).
        """
        for param in self.model.model.parameters():
            param.requires_grad = True
        self.logger.log_info("backbone/unfreeze", "Backbone parameters unfrozen")

    # ── Eval Mode 强制 | Eval Mode Enforcement ────────────────

    def _force_eval_mode(self) -> None:
        """
        强制底层 YOLO 模型保持 eval 模式（V1 核心教训）。
        Force underlying YOLO model to stay in eval mode (V1 core lesson).

        model.train() 会修改 Detect head 内部状态，
        触发不兼容的代码路径导致 crash。
        model.train() changes Detect head internal state,
        triggering incompatible code paths → crash.
        """
        self.model.model.eval()
        # 双重保险：也设置 training 标志 | Double safety: also set training flag
        self.model.model.training = False

    def train(self, mode: bool = True) -> "FastSAMBackbone":
        """
        重写 train()——阻止进入训练模式。| Override train() — prevent training mode.

        V1 教训 | V1 lesson：
            model.train() → YOLOv8 Detect head crash
            替代方案：用 requires_grad 控制梯度流。
            Alternative: use requires_grad to control gradient flow.

        :raises RuntimeError: 总是抛出，因为 train() 不安全 | Always raises, train() is unsafe.
        """
        raise RuntimeError(
            "FastSAMBackbone.train() is FORBIDDEN.\n"
            "原因 | Reason: model.train() 会崩溃 YOLOv8 的 Detect 头 | crashes YOLOv8 Detect head.\n"
            "替代 | Alternative: 使用 requires_grad=True 选择性解冻参数 | Use requires_grad=True to unfreeze.\n"
            "调用 backbone.unfreeze() 解冻，backbone.freeze() 冻结。| Call backbone.unfreeze() / freeze()."
        )

    def eval(self) -> "FastSAMBackbone":
        """
        eval() 可安全调用 — 绕过被禁止的 train(False)。
        eval() is safe — bypasses forbidden train(False).

        直接设置 training 标志而不通过 train() 方法，
        因为 train(False) 与我们的禁止逻辑冲突。
        Sets training flag directly to avoid conflict with our train() override.
        """
        # 直接设置 training 标志，绕过禁止的 train(False) | Set directly, bypass forbidden train(False)
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def unfreeze(self) -> None:
        """安全解冻：设置 requires_grad=True 但不调 .train() | Safe unfreeze: requires_grad=True, no .train()."""
        self._unfreeze()

    def freeze(self) -> None:
        """安全冻结：设置 requires_grad=False | Safe freeze: requires_grad=False."""
        self._apply_freeze()

    def __del__(self) -> None:
        """清理钩子 | Clean up hooks."""
        self._remove_all_hooks()

    # ── LoRA | 低秩适配 ───────────────────────────────────────

    def apply_lora(self, rank: int = 4) -> int:
        """
        在冻结 backbone 的 P3/P4 输出特征后添加 Feature LoRA 适配器。
        Add Feature LoRA adapters after frozen P3/P4 outputs.

        安全：不修改 YOLOv8 内部结构，只在特征提取后添加可训练的轻量适配器。
        Safe: does NOT modify YOLOv8 internals, only adds trainable adapters
        after feature extraction points.

        :param rank: LoRA 秩 | LoRA rank (default 4).
        :return: 添加的参数数量 | Number of parameters added.
        """
        import torch.nn as nn

        # P3: stride=8, 960 channels → add LoRA
        # P4: stride=16, 1280 channels → add LoRA
        self._lora_p3 = nn.Sequential(
            nn.Conv2d(960, rank, 1, bias=False),
            nn.Conv2d(rank, 960, 1, bias=False),
        )
        self._lora_p4 = nn.Sequential(
            nn.Conv2d(1280, rank, 1, bias=False),
            nn.Conv2d(rank, 1280, 1, bias=False),
        )
        # 初始化: 第二层权重为 0 → LoRA 初始不改变特征
        nn.init.kaiming_uniform_(self._lora_p3[0].weight)
        nn.init.zeros_(self._lora_p3[1].weight)
        nn.init.kaiming_uniform_(self._lora_p4[0].weight)
        nn.init.zeros_(self._lora_p4[1].weight)

        self._lora_p3.to(torch.device(self._device))
        self._lora_p4.to(torch.device(self._device))
        self._has_lora = True

        n_params = sum(p.numel() for p in list(self._lora_p3.parameters()) +
                       list(self._lora_p4.parameters()))
        self.logger.log_info(
            "backbone/lora",
            f"Feature LoRA applied: rank={rank}, +{n_params:,} params "
            f"(P3: 960→{rank}→960, P4: 1280→{rank}→1280)",
        )
        return n_params

    # ── 前向传播 | Forward Pass ───────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播，返回多尺度特征图 | Forward pass, returns multi-scale feature maps."""
        if self._hook_p4_idx is None or self._hook_p8_idx is None or self._hook_p3_idx is None:
            self._probe_strides(x)
            self._register_final_hooks()

        if x.device != torch.device(self._device):
            x = x.to(self._device)

        self._features.clear()

        with torch.set_grad_enabled(not self._freeze_backbone):
            # 使用 _predict_once 而非 Sequential 直调：
            # YOLOv8 neck 中的 C2f 层需要多尺度特征路由 (m.f 索引)，
            # nn.Sequential 直调会绕过这个路由，导致通道不匹配崩溃。
            # Use _predict_once (NOT Sequential direct call):
            # C2f layers in YOLOv8 neck need multi-scale feature routing (m.f indices).
            # Direct Sequential call bypasses this → channel mismatch crash.
            self.model.model._predict_once(x)

        result: dict[str, torch.Tensor] = {}
        if "p3" in self._features:
            f = self._features["p3"]
            if getattr(self, '_has_lora', False):
                f = f + self._lora_p3(f)
            result["p3"] = f
        if "p4" in self._features:
            f = self._features["p4"]
            if getattr(self, '_has_lora', False):
                f = f + self._lora_p4(f)
            result["p4"] = f
        if "p8" in self._features:
            result["p8"] = self._features["p8"]
        return result


class ConvLoRA(nn.Module):
    """
    Conv2d LoRA 适配器 | Conv2d LoRA Adapter.

    y = W*x + (alpha/r) * B(A(x))
    原始权重 W 冻结，仅训练 A 和 B。
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.conv = conv  # 原始冻结 Conv | Frozen original Conv
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # 冻结原始权重 | Freeze original weights
        for p in conv.parameters():
            p.requires_grad = False

        # LoRA: 1×1 down → 1×1 up
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        self.lora_down = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_ch, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        self.lora_params = sum(p.numel() for p in [self.lora_down.weight, self.lora_up.weight])

    def forward(self, x):
        """前向传播: y = W*x + (alpha/r) * B(A(x))"""
        y = self.conv(x)
        lora_y = self.lora_up(self.lora_down(x))
        return y + self.scale * lora_y



# ═══════════════════════════════════════════════════════════════
# ConvLoRA: 独立模块类 | Standalone module class
# ═══════════════════════════════════════════════════════════════


# ── 工厂函数 | Factory Function ───────────────────────────────


def build_backbone(name: str = "FastSAM-x", **kwargs) -> FastSAMBackbone:
    """
    根据名称构建骨干网络 | Build backbone by name.

    当前支持的骨干 | Currently supported backbones:
        - "FastSAM-x": FastSAM 基于 YOLOv8-x | FastSAM on YOLOv8-x
        - "FastSAM-s": FastSAM 基于 YOLOv8-s (TODO) | FastSAM on YOLOv8-s (TODO)

    :param name: 骨干名称 | Backbone name. **kwargs: 传递给 FastSAMBackbone 的参数 | Args forwarded to FastSAMBackbone.
    :type name: str

    :return: FastSAMBackbone 实例 | FastSAMBackbone instance.
    :rtype: FastSAMBackbone

    :raises ValueError: 未知的骨干名称 | Unknown backbone name.
    """
    supported = {"FastSAM-x", "FastSAM-s"}
    if name not in supported:
        raise ValueError(
            f"未知骨干名称 | Unknown backbone: {name!r}. "
            f"支持 | Supported: {sorted(supported)}"
        )

    checkpoint_map = {
        "FastSAM-x": str(_THIRD_LIB / "weights" / "FastSAM-x.pt"),
        "FastSAM-s": str(_THIRD_LIB / "weights" / "FastSAM-s.pt"),
    }
    kwargs.setdefault("checkpoint", checkpoint_map[name])
    return FastSAMBackbone(**kwargs)
