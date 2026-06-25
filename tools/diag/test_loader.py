#!/usr/bin/env python3
"""
验证 ISAIDDataset 加载与 tile 分片 | ISAIDDataset Loading & Tiling Validation
===============================================================================

测试内容 | Tests performed:
    - train/val/test 三 split 数据加载 | Three-split data loading
    - tile 模式下的 image/mask 形状验证 | Image/mask shape verification under tile mode
    - dense vs instance 模式切换      | Dense vs instance mode switching
    - 大尺寸图像 (up to 5500×3875) 兼容  | Large image (up to 5500×3875) compatibility

用途 | Purpose:
    快速验证数据预处理后的数据加载链路是否正常。
    Quick validation that the data loading pipeline works after preprocessing.

用法 | Usage::
    python tools/diag/test_loader.py
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid import ISAIDDataset

# 数据根目录 | Data root directory
DATA = "data/iSAID_processed"

for split in ["train", "val", "test"]:
    try:
        # iSAID 图像大 (up to 5500×3875), 需 tile 模式 | Large images need tile mode
        # test 无标注, 使用 instance 模式 | test has no labels, use instance mode
        use_dense_labels = (split != "test")
        ds = ISAIDDataset(root_dir=DATA, split=split, tile_size=1024,
                          dense_labels=use_dense_labels)
        s = ds[0]
        key = "mask" if use_dense_labels else "masks"
        print(f"[{split:5s}] {len(ds):5d} tiles  "
              f"sample: image={list(s['image'].shape)}  "
              f"{key}={list(s[key].shape)}  "
              f"id={s.get('image_id', 'N/A')}")
    except Exception as e:
        print(f"[{split:5s}] ERROR: {type(e).__name__}: {e}")
        print(f"         常见原因 | Common cause: data/iSAID_processed 目录不存在或未预处理")
