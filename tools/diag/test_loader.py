#!/usr/bin/env python3
"""
验证 ISAIDDataset 加载 | Quick ISAIDDataset loading test.

用法 | Usage:
    python tools/test_isaid_loader.py
"""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid import ISAIDDataset

DATA = "data/iSAID_processed"

for split in ["train", "val", "test"]:
    try:
        # iSAID 图像大 (up to 5500×3875), 需 tile 模式 | Large images need tile mode
        # test 无标注, 使用 instance 模式 | test has no labels, use instance mode
        use_semantic = (split != "test")
        ds = ISAIDDataset(root_dir=DATA, split=split, tile_size=1024,
                          semantic=use_semantic)
        s = ds[0]
        key = "mask" if use_semantic else "masks"
        print(f"[{split:5s}] {len(ds):5d} tiles  "
              f"sample: image={list(s['image'].shape)}  "
              f"{key}={list(s[key].shape)}  "
              f"id={s.get('image_id', 'N/A')}")
    except Exception as e:
        print(f"[{split:5s}] ERROR: {type(e).__name__}: {e}")
