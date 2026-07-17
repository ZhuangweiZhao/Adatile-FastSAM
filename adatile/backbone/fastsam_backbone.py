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
    TARGET_STRIDE_4  = (3, 5)     # stride 4 的容许范围 | tolerance for stride 4
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
        self._hook_p2_idx: int | None = None
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
            # 跳过 Detect head，只跑 Backbone+Neck | Skip Detect head, only Backbone+Neck
            self._forward_features(x)

        # 分析各层输出的步长 | Analyze strides of each layer's output
        candidates_4  = []  # stride-4 candidates
        candidates_8  = []  # stride-8 candidates
        candidates_16 = []  # stride-16 candidates
        candidates_32 = []  # stride-32 candidates

        for key, feat in self._features.items():
            _, _, h_out, w_out = feat.shape
            stride_h = h_in / h_out
            stride_w = w_in / w_out
            avg_stride = (stride_h + stride_w) / 2

            if self.TARGET_STRIDE_4[0] <= avg_stride <= self.TARGET_STRIDE_4[1]:
                candidates_4.append((int(key), avg_stride, feat.shape[1]))
            if self.TARGET_STRIDE_8[0] <= avg_stride <= self.TARGET_STRIDE_8[1]:
                candidates_8.append((int(key), avg_stride, feat.shape[1]))
            if self.TARGET_STRIDE_16[0] <= avg_stride <= self.TARGET_STRIDE_16[1]:
                candidates_16.append((int(key), avg_stride, feat.shape[1]))
            if self.TARGET_STRIDE_32[0] <= avg_stride <= self.TARGET_STRIDE_32[1]:
                candidates_32.append((int(key), avg_stride, feat.shape[1]))

        # P2 (stride≈4) — 按通道数降序选择最佳匹配 | pick best by channel count
        if candidates_4:
            candidates_4.sort(key=lambda t: -t[2])
            self._hook_p2_idx = candidates_4[0][0]
            self._p2_channels = candidates_4[0][2]
            self.logger.log_info(
                "backbone/probe",
                f"P2 hook: layer {self._hook_p2_idx}, "
                f"stride={candidates_4[0][1]:.1f}, "
                f"channels={candidates_4[0][2]}",
            )

        # P3 (stride≈8) — 按通道数降序选择最佳匹配 | pick best by channel count
        if candidates_8:
            candidates_8.sort(key=lambda t: -t[2])
            self._hook_p3_idx = candidates_8[0][0]
            self._p3_channels = candidates_8[0][2]
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
            self._p4_channels = candidates_16[0][2]
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
            self._p8_channels = candidates_32[0][2]
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

        # P2 钩子 | P2 hook (stride≈4)
        if self._hook_p2_idx is not None:
            handle = sequential[self._hook_p2_idx].register_forward_hook(
                self._make_feature_hook("p2")
            )
            self._hook_handles.append(handle)

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

    def unfreeze_last_n_layers(self, n: int = 0) -> int:
        """
        部分解冻: 仅解冻 backbone 最后 n 层 (neck/FPT), 前面保持冻结。
        Partial unfreeze: only unfreeze last n layers (neck/FPT), keep earlier frozen.

        适合 Freeze vs Partial Fine-tune 对比实验:
        - n=0: 完全冻结 (等同 freeze)
        - n=5-10: 仅解冻 neck 层 (与 P3/P4 特征直接相关)
        - n=-1: 全部解冻 (等同 unfreeze)

        Suitable for Freeze vs Partial Fine-tune comparison:
        - n=0: fully frozen (same as freeze)
        - n=5-10: only unfreeze neck layers (directly related to P3/P4 features)
        - n=-1: unfreeze all (same as unfreeze)

        :param n: 解冻最后 N 层 (0=全部冻结, -1=全部解冻)
                  Number of last layers to unfreeze (0=all frozen, -1=all unfrozen).
        :return: 解冻的参数数量 | Number of unfrozen parameters.
        """
        if n == -1:
            self._unfreeze()
            n_params = sum(p.numel() for p in self.model.model.parameters())
            self.logger.log_info(
                "backbone/partial_unfreeze",
                f"All layers unfrozen: {n_params:,} params",
            )
            self._freeze_backbone = False
            return n_params

        if n == 0:
            self._apply_freeze()
            self.logger.log_info(
                "backbone/partial_unfreeze",
                "All layers frozen (n=0)",
            )
            self._freeze_backbone = True
            return 0

        # 先全部冻结, 再选择性解冻最后 n 层
        # Freeze all first, then selectively unfreeze last n layers
        self._apply_freeze()

        sequential = self.model.model.model
        total_layers = len(sequential)
        unfreeze_start = max(0, total_layers - n)

        unfrozen_params = 0
        for idx in range(unfreeze_start, total_layers):
            layer = sequential[idx]
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        # 同时需要把 hook 关联的层之外检测头的参数情况
        # Also handle params outside the hooked layers if needed
        self._freeze_backbone = False  # 部分解冻时允许梯度流
        self.logger.log_info(
            "backbone/partial_unfreeze",
            f"Unfrozen last {n}/{total_layers} layers (idx {unfreeze_start}-{total_layers-1}), "
            f"{unfrozen_params:,} trainable params",
        )
        return unfrozen_params

    def freeze(self) -> None:
        """安全冻结：设置 requires_grad=False | Safe freeze: requires_grad=False."""
        self._apply_freeze()

    @property
    def channels(self) -> dict[str, int]:
        """
        探测到的各层通道数 | Detected channel counts for each feature level.

        首次 forward 后可用 | Available after first forward.
        返回 | Returns: {"p2": int, "p3": int, "p4": int, "p8": int}
        """
        return {
            "p2": getattr(self, "_p2_channels", 0),
            "p3": getattr(self, "_p3_channels", 0),
            "p4": getattr(self, "_p4_channels", 0),
            "p8": getattr(self, "_p8_channels", 0),
        }

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

    def apply_conv_lora(self, rank: int = 4, alpha: float = 1.0,
                        target_layers: list[int] | None = None,
                        target_layer_names: list[str] | None = None) -> int:
        """
        将 ConvLoRA 注入 YOLOv8 backbone 的 Conv2d 层 (真正的 Backbone LoRA)。
        Inject ConvLoRA into YOLOv8 backbone Conv2d layers (True Backbone LoRA).

        与 apply_lora() 的区别 | Difference from apply_lora():
            apply_lora(): 在 P3/P4 输出后添加特征空间适配器 (backbone 外部)
            apply_conv_lora(): 替换 backbone 内部的 Conv2d 为 ConvLoRA (backbone 内部)

        默认目标: neck 区域 (最后 16 层的前 14 层, 即 P3/P4 特征形成区域)。
        Default target: neck region (first 14 of last 16 layers, P3/P4 formation zone).

        原理 | Principle:
            冻结 SA-1B 训练的原始卷积权重, 注入低秩可训练旁路。
            旁路初始化为零 → 训练从原始行为开始逐步偏离 → 适应工业纹理域。
            Freeze SA-1B original conv weights, inject low-rank trainable bypass.
            Bypass initialized to zero → starts from original behavior → adapts to industrial textures.

        :param rank: LoRA 秩 | LoRA rank (建议 4-8 | recommend 4-8).
        :param alpha: LoRA 缩放因子 | LoRA scaling factor (default 1.0).
        :param target_layers: 目标层索引列表 (None=自动选择 neck 区域).
                              Target layer indices (None=auto-select neck region).
        :param target_layer_names: 按层名称过滤 (如 ["C2f", "Conv"]), None=不过滤.
                                   Filter by layer class name (e.g. ["C2f", "Conv"]).
        :return: 添加的 LoRA 参数数量 | Number of LoRA parameters added.
        """
        sequential = self.model.model.model
        total_layers = len(sequential)

        if target_layers is None:
            # 默认: neck 区域, 最后 16 层中排除 Detect head (最后 2 层)
            # Default: neck region, last 16 layers excluding Detect head (last 2)
            target_layers = list(range(max(0, total_layers - 16), total_layers - 1))

        lora_params = 0
        injected_layers = []

        for idx in target_layers:
            if idx >= total_layers:
                continue
            layer = sequential[idx]
            layer_name = layer.__class__.__name__

            # 按名称过滤 | Filter by name
            if target_layer_names and layer_name not in target_layer_names:
                continue

            # 注入 ConvLoRA 到该层的所有 Conv2d 子模块
            n = _inject_conv_lora(layer, rank=rank, alpha=alpha,
                                  path=f"layer_{idx}({layer_name})")
            if n > 0:
                lora_params += n
                injected_layers.append(f"{idx}({layer_name})")

        self._has_conv_lora = True
        self._lora_rank = rank

        self.logger.log_info(
            "backbone/conv_lora",
            f"ConvLoRA injected: rank={rank}, alpha={alpha}, "
            f"layers={injected_layers}, +{lora_params:,} trainable params "
            f"({lora_params/1e3:.1f}K)",
        )
        return lora_params

    def get_lora_parameters(self) -> list[nn.Parameter]:
        """
        获取所有 LoRA 可训练参数 (ConvLoRA + Feature LoRA).
        Get all LoRA trainable parameters (ConvLoRA + Feature LoRA).

        :return: 可训练参数列表 | List of trainable parameters.
        """
        params = []
        # ConvLoRA 参数 (backbone 内部)
        if getattr(self, '_has_conv_lora', False):
            for module in self.model.model.modules():
                if isinstance(module, ConvLoRA):
                    params.extend(module.lora_down.parameters())
                    params.extend(module.lora_up.parameters())
        # Feature LoRA 参数 (backbone 外部, P3/P4 适配器)
        if getattr(self, '_has_lora', False):
            params.extend(self._lora_p3.parameters())
            params.extend(self._lora_p4.parameters())
        return params

    # ── 前向传播 | Forward Pass ───────────────────────────────

    def _forward_features(self, x: torch.Tensor) -> None:
        """
        仅运行 Backbone+Neck，跳过 Detect/Segment head.
        Run only Backbone+Neck, skip Detect/Segment head.

        复刻 _predict_once 的多输入层路由 (m.f 索引)，
        但在遇到 Detect/Segment 层时提前终止。
        Replicates _predict_once's multi-input routing (m.f indices),
        but breaks early when Detect/Segment layers are reached.

        跳过 head 可节省 ~30-40% 前向时间 + 显存，
        因为我们只需要中间特征 (P3/P4/P8)，不需要 box/mask 输出。
        Skipping head saves ~30-40% forward time + memory,
        since we only need intermediate features, not box/mask outputs.
        """
        detection_model = self.model.model
        y = []
        for m in detection_model.model:
            # ── 多输入路由 | Multi-input routing ──
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]
            # ── 跳过 Detect/Segment head | Skip Detect/Segment head ──
            cls_name = m.__class__.__name__
            if cls_name in ("Detect", "Segment", "DetectSegment"):
                # ── 保存 Segment head 的输入特征 | Save Segment head input features ──
                # 这些特征有正确的通道数，用于 proto mask 提取
                # These features have the correct channel count for proto mask extraction
                # x = [P3, P4, P5] at the channels the Segment head expects
                if isinstance(x, list) and len(x) >= 1:
                    self._segment_inputs = x
                break
            x = m(x)
            y.append(x if m.i in getattr(detection_model, 'save', []) else None)

    @staticmethod
    def _pad_to_32(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        将输入填充到 32 的倍数（FastSAM 要求）| Pad input to multiple of 32 (FastSAM requirement).

        YOLOv8/FastSAM 的 neck 在 Concat 时要求所有特征图尺寸一致，
        非 32 倍数的输入会导致 stride 取整不一致 → 尺寸不匹配 → crash。
        YOLOv8/FastSAM's neck requires consistent feature map sizes at Concat;
        non-32-multiple input causes stride rounding mismatch → size mismatch → crash.

        :param x: 输入张量 [B, C, H, W] | Input tensor.
        :return: (padded_tensor, (pad_h, pad_w, orig_h, orig_w))
        """
        B, C, H, W = x.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w, H, W)

    def forward(self, x: torch.Tensor, extract_proto: bool = False) -> dict[str, torch.Tensor]:
        """
        前向传播，返回多尺度特征图 | Forward pass, returns multi-scale feature maps.

        自动将输入填充到 32 的倍数（FastSAM 的 Concat 层要求）。
        Auto-pads input to multiple of 32 (required by FastSAM's Concat layers).

        :param x: 输入图像张量 [B, 3, H, W] | Input image tensor.
        :param extract_proto: 是否提取 FastSAM proto masks [B, 32, H/4, W/4]。
            这些 proto masks 是预训练的基函数，可通过线性组合生成实例掩码：
            mask = sigmoid(coefficients @ proto_masks)。
            Whether to extract FastSAM proto masks. These are pretrained basis functions
            that generate instance masks via linear combination.
        :return: dict with keys "p2", "p3", "p4", "p8", and optionally "proto".
        """
        # ── 自动填充到 32 的倍数 | Auto-pad to 32× multiple ──
        x, self._pad_info = self._pad_to_32(x)

        if self._hook_p4_idx is None or self._hook_p8_idx is None or self._hook_p3_idx is None or self._hook_p2_idx is None:
            self._probe_strides(x)
            self._register_final_hooks()

        if x.device != torch.device(self._device):
            x = x.to(self._device)

        self._features.clear()

        # ── 梯度控制: LoRA 激活时允许梯度流 (即使 backbone 其他参数冻结) ──
        # Gradient control: enable grad flow when LoRA is active (even if backbone is frozen)
        _need_grad = (not self._freeze_backbone) or getattr(self, '_has_conv_lora', False)
        with torch.set_grad_enabled(_need_grad):
            self._forward_features(x)

        result: dict[str, torch.Tensor] = {}
        if "p2" in self._features:
            result["p2"] = self._features["p2"]
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

        # ── Proto Mask 提取 | Proto Mask Extraction ──
        # FastSAM 的 Segmentation head 包含 Proto 模块，将 P3 特征映射为 32 个基掩码。
        # 这些 proto masks 可通过线性组合生成任意实例掩码：
        #     instance_mask = sigmoid(coefficients @ proto_masks)
        # FastSAM's Segmentation head contains a Proto module that maps P3 features
        # to 32 basis masks. These can linearly combine to form any instance mask.
        if extract_proto and hasattr(self, '_segment_inputs') and self._segment_inputs:
            try:
                detection_model = self.model.model
                segment_head = detection_model.model[-1]

                if hasattr(segment_head, 'proto'):
                    # 使用 Segment head 接收的原始 P3 特征（正确的通道数）
                    # Use the original P3 features that the Segment head receives (correct channels)
                    # _segment_inputs[0] = P3 at Segment head's expected channels (e.g. 320)
                    p3_seg = self._segment_inputs[0]  # [B, C_seg, H/8, W/8]
                    proto_masks = segment_head.proto(p3_seg)  # [B, 32, H/4, W/4]
                    result["proto"] = proto_masks
            except Exception as e:
                self.logger.log_warn(
                    "backbone/proto",
                    f"Proto extraction failed: {e}",
                )

        # ── 应用 Adapter (CAT-SAM 迁移) | Apply Adapters (CAT-SAM port) ──
        if getattr(self, '_adapters', None) is not None:
            adapted = self._adapters(
                p3=result.get("p3"),
                p4=result.get("p4"),
                p8=result.get("p8"),
            )
            result.update(adapted)

        return result

    def set_adapters(self, adapters):
        """
        附加多尺度 ConvAdapter | Attach multi-scale ConvAdapters.

        设置后，每次 forward 会自动对 P3/P4/P8 应用 adapter。
        Once set, adapters are automatically applied to P3/P4/P8 on each forward.

        :param adapters: MultiScaleAdapter 实例 | MultiScaleAdapter instance.
        """
        self._adapters = adapters


class ConvLoRA(nn.Module):
    """
    Conv2d LoRA 适配器 | Conv2d LoRA Adapter.

    y = W*x + (alpha/r) * B(A(x))
    原始权重 W 冻结，仅训练 A 和 B。
    Original weight W frozen, only A and B trained.

    用于注入 YOLOv8 backbone 的 Conv2d 层，
    使冻结的 SA-1B 特征适应工业纹理域。
    Injects into YOLOv8 backbone Conv2d layers,
    adapting frozen SA-1B features to industrial texture domain.
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

        # LoRA: 1×1 down → 1×1 up (低秩分解 | Low-rank decomposition)
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        groups = conv.groups

        # LoRA A: C_in × rank (降维 | Down-projection)
        self.lora_down = nn.Conv2d(in_ch, rank, 1, bias=False)
        # LoRA B: rank × C_out (升维 | Up-projection)
        self.lora_up = nn.Conv2d(rank, out_ch, kernel_size=kernel_size,
                                 stride=stride, padding=padding,
                                 dilation=dilation, groups=1, bias=False)

        # 初始化 | Initialization
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        self.lora_params = sum(p.numel() for p in [self.lora_down.weight, self.lora_up.weight])

    def forward(self, x):
        """前向传播: y = W*x + (alpha/r) * B(A(x))"""
        y = self.conv(x)
        lora_y = self.lora_up(self.lora_down(x))
        return y + self.scale * lora_y


def _inject_conv_lora(module: nn.Module, rank: int = 4, alpha: float = 1.0,
                       path: str = "") -> int:
    """
    递归替换模块树中的 Conv2d 为 ConvLoRA | Recursively replace Conv2d with ConvLoRA.

    遍历 module 的所有子模块，将 nn.Conv2d 替换为 ConvLoRA 包装器。
    Walk through all children of module, replace nn.Conv2d with ConvLoRA wrappers.

    跳过 1×1 Conv (已经是低秩), 跳过 depthwise Conv (groups>1)。
    Skip 1×1 Conv (already low-rank), skip depthwise Conv (groups>1).

    :param module: 根模块 | Root module.
    :param rank: LoRA 秩 | LoRA rank.
    :param alpha: LoRA 缩放因子 | LoRA scaling factor.
    :param path: 当前模块路径 (调试用) | Current module path (for debugging).
    :return: 添加的 LoRA 参数数量 | Number of LoRA parameters added.
    """
    n_added = 0
    for name, child in module.named_children():
        child_path = f"{path}.{name}" if path else name
        if isinstance(child, nn.Conv2d):
            # 跳过 1×1 Conv (已低秩) 和 depthwise Conv | Skip 1×1 and depthwise
            if child.kernel_size == (1, 1) or child.groups > 1:
                continue
            # 替换为 ConvLoRA | Replace with ConvLoRA
            lora_conv = ConvLoRA(child, rank=rank, alpha=alpha)
            setattr(module, name, lora_conv)
            n_added += lora_conv.lora_params
        elif isinstance(child, ConvLoRA):
            # 已经注入过，跳过 | Already injected, skip
            continue
        else:
            n_added += _inject_conv_lora(child, rank, alpha, child_path)
    return n_added


def _collect_lora_modules(module: nn.Module) -> list["ConvLoRA"]:
    """
    收集模块树中的所有 ConvLoRA 模块 | Collect all ConvLoRA modules in module tree.

    :param module: 根模块 | Root module.
    :return: ConvLoRA 模块列表 | List of ConvLoRA modules.
    """
    lora_modules = []
    for child in module.modules():
        if isinstance(child, ConvLoRA):
            lora_modules.append(child)
    return lora_modules



# ═══════════════════════════════════════════════════════════════
# ConvLoRA: 独立模块类 | Standalone module class
# ═══════════════════════════════════════════════════════════════


# ── 工厂函数 | Factory Function ───────────────────────────────


def build_backbone(name: str = "FastSAM-x", **kwargs) -> FastSAMBackbone:
    """
    根据名称构建骨干网络 | Build backbone by name.

    当前支持的骨干 | Currently supported backbones:
        - "FastSAM-x": YOLOv8x-based, 68M params, P2(160)/P3(960)/P4(1280)/P8(1280)
        - "FastSAM-s": YOLOv8s-based, ~14M params, P2(64)/P3(128)/P4(256)/P8(512)

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
