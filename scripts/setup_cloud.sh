#!/bin/bash
# ============================================================
# AdaTile-FastSAM Cloud Server Setup
# Run once after cloning the repo on a fresh cloud GPU instance.
#
# Usage:
#   bash scripts/setup_cloud.sh
# ============================================================
set -e

echo "=== AdaTile-FastSAM Cloud Setup ==="

# ── 1. Verify GPU ─────────────────────────────────────────
echo ""
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. This must run on a GPU instance."
    exit 1
}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME ($GPU_MEM)"

# ── 2. Install PyTorch if not present ──────────────────────
echo ""
echo "[2/4] Checking PyTorch..."
python -c "import torch; print(f'  PyTorch {torch.__version__} + CUDA {torch.version.cuda}')" 2>/dev/null || {
    echo "  PyTorch not found. Installing PyTorch 2.4 + CUDA 12.1..."
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
}

# ── 3. Install project dependencies ────────────────────────
echo ""
echo "[3/4] Installing AdaTile-FastSAM dependencies..."
cd "$(dirname "$0")/.."

# Core deps (skip torch/torchvision if already installed)
pip install --quiet \
    numpy opencv-python pycocotools shapely scikit-image \
    albumentations timm xformers fvcore \
    tensorboard rich tqdm matplotlib seaborn Pillow \
    pytest

# Install adatile in dev mode
pip install -e . --no-deps 2>/dev/null || pip install -e .

echo "  Dependencies installed."

# ── 4. Verify installation ─────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python -c "
import torch
from adatile.config import Config
from adatile.modeling import build_adatile_fastsam

cfg = Config()
cfg.backbone.name = 'ResNet50Backbone'
cfg.backbone.pretrained = False
cfg.sparse.importance_threshold = 0.15
cfg.router.name = 'DTRv2Router'

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB')

model = build_adatile_fastsam(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Model params: {n_params/1e6:.1f}M')
print(f'  Build OK')
" && echo "" && echo "=== Setup Complete ==="
