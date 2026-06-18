"""
Pytest 共享 fixtures | Shared pytest fixtures.
===============================================
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mock_isaid_dir() -> str:
    """
    创建模拟 iSAID 数据集目录 | Create mock iSAID dataset directory.

    用于 datasets 测试和集成测试。
    Used by datasets tests and integration tests.
    """
    tmpdir = tempfile.mkdtemp(prefix="mock_isaid_")

    for split in ["train", "val"]:
        img_dir = Path(tmpdir) / split / "images"
        ann_dir = Path(tmpdir) / split / "annotations"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

    # 生成合成图像 | Generate synthetic images
    _make_dummy_image(Path(tmpdir) / "train" / "images" / "000001.png", 256, 256)
    _make_dummy_image(Path(tmpdir) / "train" / "images" / "000002.png", 512, 512)
    _make_dummy_image(Path(tmpdir) / "val" / "images" / "000003.png", 256, 256)

    # 生成 COCO 标注 | Generate COCO annotations
    categories = [
        {"id": 1, "name": "small_vehicle"},
        {"id": 2, "name": "large_vehicle"},
        {"id": 3, "name": "plane"},
        {"id": 4, "name": "storage_tank"},
        {"id": 5, "name": "ship"},
        {"id": 6, "name": "harbor"},
        {"id": 7, "name": "ground_track_field"},
        {"id": 8, "name": "soccer_ball_field"},
        {"id": 9, "name": "tennis_court"},
        {"id": 10, "name": "swimming_pool"},
        {"id": 11, "name": "road"},
        {"id": 12, "name": "basketball_court"},
        {"id": 13, "name": "bridge"},
        {"id": 14, "name": "helicopter"},
        {"id": 15, "name": "roundabout"},
    ]

    train_coco = {
        "images": [
            {"id": 1, "file_name": "000001.png", "width": 256, "height": 256},
            {"id": 2, "file_name": "000002.png", "width": 512, "height": 512},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [[10, 10, 100, 10, 100, 100, 10, 100]],
             "bbox": [10, 10, 90, 90], "area": 8100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2,
             "segmentation": [[150, 150, 200, 150, 200, 200, 150, 200]],
             "bbox": [150, 150, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 5,
             "segmentation": [[200, 200, 400, 200, 400, 400, 200, 400]],
             "bbox": [200, 200, 200, 200], "area": 40000, "iscrowd": 0},
        ],
        "categories": categories,
    }

    val_coco = {
        "images": [{"id": 3, "file_name": "000003.png", "width": 256, "height": 256}],
        "annotations": [],
        "categories": categories,
    }

    with open(Path(tmpdir) / "train" / "annotations" / "instances_train.json", "w") as f:
        json.dump(train_coco, f)
    with open(Path(tmpdir) / "val" / "annotations" / "instances_val.json", "w") as f:
        json.dump(val_coco, f)

    yield tmpdir

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make_dummy_image(path: Path, h: int, w: int) -> None:
    """创建随机 RGB 图像 | Create random RGB image."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)
