"""COCO-style UPAR loader with leakage-free, image-level deterministic splits."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

from .geometry import crop_params, crop_image, to_crop, as_tensor, heatmap_targets, FLIP_INDEX


LABEL_GROUPS = [[0, 1, 2], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]


def stable_split(image_identity, fraction=0.1, seed=2027):
    key = f'{seed}:{image_identity}'.encode('utf-8')
    fraction_hash = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / float(2**64)
    return 'val' if fraction_hash < fraction else 'train'


def resolve_image(root, domain, file_name):
    """Resolve official layout; fail clearly on missing files, never silently drop."""
    file_name = str(file_name).replace('\\', '/')
    p = Path(file_name)
    candidates = [p, root / p, root / domain / p, root / domain / p.name,
                  root / domain / 'images' / p, root / domain / 'images' / p.name,
                  root / domain / 'bounding_box_train' / p.name,
                  root / domain / 'bounding_box_test' / p.name,
                  root / domain / 'query' / p.name]
    # Annotation file names sometimes include an 'images/' prefix twice.
    if file_name.startswith('images/'):
        candidates += [root / domain / file_name[7:]]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f'Missing image for domain={domain}, file={file_name}. Tried: {candidates[:6]}')


def read_records(annotation_dir, domains):
    records = []
    for domain in domains:
        path = Path(annotation_dir) / f'train_{domain}.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing annotations: {path}')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or 'images' not in data or 'annotations' not in data:
            raise ValueError(f'{path}: expected COCO-style dict with images and annotations')
        images = {int(x['id']): x for x in data['images']}
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            image = images.get(int(ann['image_id']))
            if image is None:
                continue
            box = ann.get('bbox')
            joints = ann.get('keypoints', [])
            if not box or len(box) != 4 or len(joints) != 51:
                continue
            if box[2] <= 0 or box[3] <= 0:
                continue
            k = np.asarray(joints, np.float32).reshape(17, 3)
            valid = (k[:, 2] > 0)
            valid[3:5] = False
            if not valid.any():
                continue
            records.append({'domain': domain, 'image_id': int(ann['image_id']),
                            'file_name': image['file_name'], 'bbox': np.asarray(box, np.float32),
                            'keypoints': k[:, :2], 'visible': valid.astype(np.float32),
                            'annotation_id': int(ann['id'])})
    if not records:
        raise ValueError('No annotated person instances found; inspect COCO annotations and paths')
    return records


class PoseDataset(Dataset):
    def __init__(self, annotation_dir, image_root, domains, input_hw=(256, 192),
                 split='train', val_fraction=0.1, seed=2027, train_domains=None,
                 augment=True, sigma=2.0, require_images=True, aug=None):
        self.root = Path(image_root)
        self.annotation_dir = Path(annotation_dir)
        self.hw = tuple(input_hw)
        self.augment = augment and split in ('train', 'all')
        self.sigma = sigma
        # Defaults reproduce the original augmentation exactly.
        self.aug = {'scale': [.85, 1.15], 'shift': .08, 'rot': 0., 'rot_prob': 0.5, 'label_offset': 0., **(aug or {})}
        records = read_records(annotation_dir, domains)
        self.records = []
        for rec in records:
            # Domain+image key prevents collisions, and keeps all persons in one image together.
            fold = stable_split(f"{rec['domain']}:{rec['image_id']}", val_fraction, seed)
            if split == 'train' and (rec['domain'] not in (train_domains or domains) or fold == 'val'):
                continue
            if split == 'val' and fold != 'val':
                continue
            if split == 'all':
                pass
            elif split not in ('val', 'train'):
                raise ValueError('split must be train, val, or all')
            rec = dict(rec)
            if require_images:
                rec['image_path'] = resolve_image(self.root, rec['domain'], rec['file_name'])
            self.records.append(rec)
        if not self.records:
            raise ValueError(f'Empty dataset: split={split}, domains={domains}; change split fraction or inspect data')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        im_path = r.get('image_path') or resolve_image(self.root, r['domain'], r['file_name'])
        with Image.open(im_path) as fp:
            image = fp.convert('RGB')
        box = r['bbox']
        sh = self.aug['shift']
        shift = (random.uniform(-sh, sh), random.uniform(-sh, sh)) if self.augment else (0., 0.)
        scale = random.uniform(*self.aug['scale']) if self.augment else 1.
        params = crop_params(box, self.hw, pad=1.25, shift=shift, scale_jitter=scale)
        crop = crop_image(image, params, self.hw)
        points = to_crop(r['keypoints'], params)
        visible = r['visible'].copy()
        if self.augment:
            if self.aug['rot'] > 0 and random.random() < self.aug['rot_prob']:
                crop, points = rotate_crop(crop, points, random.uniform(-self.aug['rot'], self.aug['rot']))
            if self.aug['label_offset'] > 0:
                # Experimental: per-instance random offset per joint group (same for left/right), mimicking domain-specific
                # annotation conventions seen in LODO (scripts/bias_study.py). Targets only; image untouched.
                sig = self.aug['label_offset'] * self.hw[0]
                points = points.copy()
                for grp in LABEL_GROUPS:
                    points[grp, 1] += random.gauss(0., sig); points[grp, 0] += random.gauss(0., sig * .5)
            if random.random() < 0.5:
                crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                points = points[FLIP_INDEX].copy()
                points[:, 0] = self.hw[1] - 1 - points[:, 0]
                visible = visible[FLIP_INDEX].copy()
            if random.random() < 0.65:
                crop = ImageEnhance.Brightness(crop).enhance(random.uniform(.7, 1.3))
                crop = ImageEnhance.Contrast(crop).enhance(random.uniform(.7, 1.3))
            if random.random() < 0.25:
                crop = crop.filter(ImageFilter.GaussianBlur(random.uniform(.2, 1.4)))
            if random.random() < 0.3:
                # Coarse input degradation; annotations remain geometrically unchanged.
                h, w = self.hw
                factor = random.uniform(.4, .85)
                small = crop.resize((max(8, int(w*factor)), max(8, int(h*factor))), Image.Resampling.BILINEAR)
                crop = small.resize((w, h), Image.Resampling.BILINEAR)
        target, weights = heatmap_targets(points, visible, self.hw, sigma=self.sigma)
        return {'image': torch.from_numpy(as_tensor(crop).copy()),
                'target': torch.from_numpy(target), 'weights': torch.from_numpy(weights),
                'original_points': torch.from_numpy(r['keypoints'].copy()),
                'original_visible': torch.from_numpy(r['visible'].copy()),
                'crop_params': torch.tensor(params, dtype=torch.float32),
                'box': torch.from_numpy(box.copy()), 'domain': r['domain'],
                'image_id': r['image_id'], 'annotation_id': r['annotation_id']}


def rotate_crop(crop, points, degrees):
    """Rotate a crop counter-clockwise by `degrees` about its centre; move keypoints with it."""
    w, h = crop.size
    cx, cy = (w - 1) / 2., (h - 1) / 2.
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    out = crop.rotate(degrees, resample=Image.Resampling.BILINEAR, center=(cx, cy), fillcolor=(0, 0, 0))
    pts = np.asarray(points, np.float32).copy()
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    pts[:, 0] = cx + dx * c + dy * s
    pts[:, 1] = cy - dx * s + dy * c
    return out, pts


def balanced_sampler(dataset):
    counts = Counter(r['domain'] for r in dataset.records)
    weights = torch.tensor([1.0 / counts[r['domain']] for r in dataset.records], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
