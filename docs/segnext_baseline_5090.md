# SegNeXt Baseline on RTX 5090（REPRODUCE.md）

纯 PyTorch standalone 实现，零 mmcv 依赖，适配 RTX 5090 (Blackwell sm_120, CUDA 12.8+)。

训练配方 **对齐官方 mmseg 实现思想**：
- 数据增广: RandomResize(0.5~2.0) → RandomCrop(cat_max_ratio=0.75) → HFlip(0.5) → PhotoMetricDistortion
- 归一化: ImageNet mean/std（配合 IN-1K 预训练权重）
- 损失: 纯 CrossEntropy (ignore_index=255)
- 优化器: AdamW lr=6e-5, head lr×10, norm 参数 weight_decay=0
- 调度: linear warmup 1500 iters → poly power=1.0

## 环境搭建（5090，纯 PyTorch）

```bash
conda create -n segnext python=3.10 -y
source activate segnext          # 或 conda activate segnext

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy opencv-python tqdm

# 自检
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 代码与数据上传

```bash
# ── 上传内容 ──
# adatile/                        ← 完整包（backbone/mscan.py, decoder/ham_head.py,
#                                     datasets/neu_seg.py, logging/, utils/seed.py）
# tools/train/train_segnext.py    ← 官方配方训练入口
# tools/eval/eval_segnext.py      ← 评估入口（自动读取 ckpt 的归一化设置）
# pretrained/mscan_t.pth          ← IN-1K 预训练 MSCAN-T
#                                   从 https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/ 下载

# ── 数据（服务器上已有，软链即可）──
mkdir -p data
ln -s /root/SegNeXt/data/NEU_Seg data/NEU_Seg

# 验证结构
ls data/NEU_Seg/images/training/ | head -5    # 3630 JPG
ls data/NEU_Seg/annotations/test/ | head -5   # 840 PNG
```

## 预训练权重下载

从[清华云盘](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/)下载对应模型大小的权重：

| 模型 | 文件 | 参数量 |
|------|------|--------|
| MSCAN-T | `mscan_t.pth` | 3.9M |
| MSCAN-S | `mscan_s.pth` | 13.8M |
| MSCAN-B | `mscan_b.pth` | 27.6M |
| MSCAN-L | `mscan_l.pth` | 48.8M |

```bash
mkdir -p pretrained
# 上传或下载后放到 pretrained/ 目录下
```

**注意**: 首次训练时 `_load_pretrained` 会打印 missing/unexpected keys。
预期：missing≈0，unexpected 仅分类头(`head.*`)——这是正常的。

## 训练（官方配方）

```bash
# 完整训练（40k iters = 200 epochs × 200 steps/epoch）
python tools/train/train_segnext.py \
    --model-size tiny \
    --pretrained pretrained/mscan_t.pth \
    --batch-size 16 \
    --device cuda

# SegNeXt-Small
python tools/train/train_segnext.py \
    --model-size small \
    --pretrained pretrained/mscan_s.pth \
    --batch-size 16

# 快速 smoke test（500 步，验证 loss 下降、无 NaN、显存占用）
python tools/train/train_segnext.py \
    --epochs 3 --steps-per-epoch 167 --batch-size 16 \
    --device cuda
```

**关键参数**（默认值即官方配置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 6e-5 | 基础学习率（head 自动 ×10） |
| `--batch-size` | 16 | 200² 小图显存充裕，5090 可加更大 |
| `--warmup-iters` | 1500 | 线性 warmup |
| `--loss` | `ce` | 纯 CrossEntropy（官方）；`ce_dice`=旧配方 |
| `--img-norm` | `imagenet` | 官方 mean/std；`unit`=[0,1]（旧行为） |
| `--augment` | 默认开启 | `--no-augment` 关闭 |
| `--class-weights` | `none` | `balanced` / `inverse` 备选 |

**向后兼容旧配方**（若需复现旧 run）：
```bash
python tools/train/train_segnext.py --loss ce_dice --no-augment --img-norm unit --lr 6e-4
```

训练输出在 `runs/segnext_{size}_pt_NEUSeg_{MMDD_HHMM}/`：
- `train.jsonl` — 每 20 步一次标量日志
- `best_model.pt` — 最佳 mIoU checkpoint
- `last_model.pt` — 最终 epoch checkpoint
- `results.json` — 汇总指标

## 评估

```bash
# 自动从 ckpt args 读取 img_norm/ham_channels/md_r 等设置
python tools/eval/eval_segnext.py \
    --checkpoint runs/segnext_tiny_pt_NEUSeg_*/best_model.pt

# 输出 eval_results.json → per-class IoU, mIoU, pixel_accuracy, sample_mIoU
```

## 解决常见问题

### 1. `tools/dist_train.sh: line 7: $'\r': command not found`
Windows 打包导致的 CRLF 换行。运行：
```bash
find . -name "*.sh" -print0 | xargs -0 sed -i 's/\r$//'
```

### 2. 预训练权重键名不匹配
如果 `_load_pretrained` 输出大量 missing keys —— 检查你是不是用了 mmseg 而非分类器的权重。
清华云盘的 `mscan_t.pth` 是 IN-1K 分类权重，键名包含 `backbone.` 前缀，`_load_pretrained` 会自动 strip。
若仍有差异，查看 `adatile/backbone/mscan.py:350-376` 的加载逻辑。

### 3. 3090 / A100 / V100 (sm_86 以下)
无需改造，但需切换到 CUDA 11.x 工具链：
```bash
conda create -n segnext python=3.8 -y && conda activate segnext
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy opencv-python tqdm
```
其余步骤不变。

### 4. 旧 checkpoint 评估
旧 ckpt 的 `args` 字典中无 `img_norm` 字段，eval 脚本自动 fallback 到 `unit`（即 [0,1] 输入），向后兼容。

## 与 mmseg 官方实现的对照

| 官方 mmseg 组件 | standalone 对应 |
|---|---|
| `img_norm_cfg dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True)` | `normalize_img(img, "imagenet")` — 等效的 [0,1] 域归一化 |
| `MSCAN(embed_dims=[32,64,160,256], depths=[3,3,5,2])` | `MSCAN(model_size="tiny")` — 参数完全一致 |
| `LightHamHead(in_channels=[64,160,256], num_classes=4)` | `LightHamHead(in_channels=[64,160,256], num_classes=4)` — 完全一致 |
| `paramwise_cfg(custom_keys={'head': lr×10, 'norm': decay_mult=0})` | `build_official_param_groups()` — 子串匹配语义一致 |
| `lr_config(policy='poly', warmup='linear', warmup_iters=1500)` | `LambdaLR(poly_lambda)` + warmup — 数值一致 |
| `RandomResize + RandomCrop + RandomFlip + PhotoMetricDistortion` | `OfficialAugment` — 每步语义独立实现，数值级镜像 |
| `loss_decode(type='CrossEntropyLoss')` | `compute_loss(loss_type="ce")` — NLL(log(softmax+eps)) 等效于 CE |
