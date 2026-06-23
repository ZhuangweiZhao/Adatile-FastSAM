"""
NWPU-VHR-10 数据集 | NWPU-VHR-10 Dataset Adapter
================================================

NWPU-VHR-10 is a 10-class geospatial object detection dataset.
Only bounding box annotations available (no instance segmentation masks).

Classes: airplane(1), ship(2), storage tank(3), baseball diamond(4),
         tennis court(5), basketball court(6), ground track field(7),
         harbor(8), bridge(9), vehicle(10)

Key design: bboxes used as weak masks for few-shot instance segmentation.
Images padded to multiples of 32 for FastSAM compatibility.
"""

from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from adatile.logging import get_logger

logger = get_logger('nwpu')

CLASS_NAMES = {
    1: 'airplane', 2: 'ship', 3: 'storage tank',
    4: 'baseball diamond', 5: 'tennis court',
    6: 'basketball court', 7: 'ground track field',
    8: 'harbor', 9: 'bridge', 10: 'vehicle'
}

class NWPUDataset(Dataset):
    """
    Returns per-image: {image, masks, labels, bboxes, image_id, orig_size}
    Where masks = bbox-based binary masks [N, H_pad, W_pad] for each instance.

    With bbox-only annotations, each instance uses its bbox rectangle as mask.
    FastSAM pretrained mask quality can optionally refine these masks (TODO).
    """

    def __init__(
        self,
        root_dir: str = 'data/NWPU',
        split: str = 'train',
        pad_to_32: bool = True,
    ):
        self.root = Path(root_dir)
        self.split = split
        self.pad_to_32 = pad_to_32

        self._pos_dir = self.root / 'positive image set'
        self._gt_dir = self.root / 'ground truth'
        self._neg_dir = self.root / 'negative image set'

        # parse ground truth
        self._samples = self._parse_annotations()

        # train/val split: file-based (uses same ID mapping as standard NWPU splits)
        self._train_ids = self._load_split()
        if split == 'train':
            self._samples = [s for s in self._samples if s['stem'] in self._train_ids]
        elif split == 'val':
            self._samples = [s for s in self._samples if s['stem'] not in self._train_ids]

        logger.log_info('nwpu/init',
            f'NWPU {split}: {len(self)} images, {sum(len(s["bboxes"]) for s in self._samples)} instances')

    def _parse_annotations(self) -> list[dict]:
        """Parse all ground truth TXT files into sample dicts."""
        samples = []
        for txt_file in sorted(self._gt_dir.glob('*.txt')):
            stem = txt_file.stem  # e.g., '001'
            img_path = self._pos_dir / f'{stem}.jpg'
            if not img_path.exists():
                continue

            with open(txt_file) as f:
                bboxes = []
                labels = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r'\((\d+),(\d+)\),\((\d+),(\d+)\),(\d+)', line)
                    if m:
                        x1, y1, x2, y2, cls = int(m[1]),int(m[2]),int(m[3]),int(m[4]),int(m[5])
                        bboxes.append([x1, y1, x2-x1, y2-y1])  # xywh format
                        labels.append(cls)

            samples.append({
                'stem': stem,
                'img_path': str(img_path),
                'bboxes': bboxes,
                'labels': labels,
            })
        return samples

    def _load_split(self) -> set:
        """Load train split. Default: first 70% of images per class (stratified)."""
        # Class-stratified split
        class_to_stems = defaultdict(list)
        for s in self._samples:
            for lbl in set(s['labels']):
                class_to_stems[lbl].append(s['stem'])

        train_ids = set()
        for stems in class_to_stems.values():
            unique = sorted(set(stems))
            n_train = max(1, int(len(unique) * 0.7))
            train_ids.update(unique[:n_train])

        return train_ids

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        sample = self._samples[index]

        # load image
        img = Image.open(sample['img_path'])
        img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        H_orig, W_orig = img_np.shape[:2]

        # pad to multiple of 32 for FastSAM
        if self.pad_to_32:
            pad_h = (32 - H_orig % 32) % 32
            pad_w = (32 - W_orig % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img_np = np.pad(img_np, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
        else:
            pad_h, pad_w = 0, 0

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [C, H, W]

        # bbox-based instance masks (each instance = bbox rectangle)
        masks_list = []
        for (x, y, w, h), lbl in zip(sample['bboxes'], sample['labels']):
            mask = torch.zeros(img_np.shape[0], img_np.shape[1])
            mask[y:y+h, x:x+w] = 1.0
            masks_list.append(mask)

        if masks_list:
            masks_t = torch.stack(masks_list, dim=0)  # [N, H, W]
        else:
            masks_t = torch.zeros(0, img_np.shape[0], img_np.shape[1])

        return {
            'image': img_t,                          # [3, H, W] padded
            'masks': masks_t.float(),                # [N, H, W] binary
            'labels': torch.tensor(sample['labels'], dtype=torch.long),  # [N]
            'bboxes': torch.tensor(sample['bboxes'], dtype=torch.float32),  # [N, 4] xywh
            'image_id': sample['stem'],
            'orig_size': (H_orig, W_orig),
        }

    def class_to_images(self, class_id: int) -> list[int]:
        """Return indices of images containing a specific class."""
        return [i for i, s in enumerate(self._samples)
                if class_id in s['labels']]

    @property
    def num_classes(self):
        return 11  # 10 FG classes + background

    @property
    def class_names(self):
        return {0: 'background', **CLASS_NAMES}
