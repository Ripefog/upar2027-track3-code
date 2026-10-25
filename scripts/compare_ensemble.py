#!/usr/bin/env python3
"""Ensemble / multi-scale TTA study on SOURCE validation only (never hidden data).

For each checkpoint caches plain+flipped heatmaps (flip shift 1 cell) at the native scale and
optionally at extra input scales (image resized, heatmap resized back), then evaluates single models
and heatmap-averaged combinations with DARK decoding (kernel 7, sigma 1.0).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import decode_dark, flip_heatmaps_back, shift_heatmap_x, from_crop, SCORED_INDEX, calibrated_joint_confidence
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model


@torch.no_grad()
def run_model(path, ds, dev, scales):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    m = build_model(obj['config']); m.load_state_dict(obj['state_dict']); m.to(dev).eval()
    out = {s: [] for s in scales}; meta = {'dom': [], 'ids': [], 'cps': []}
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        x = b['image'].to(dev)
        for s in scales:
            xs = x if s == 1.0 else F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
            h = m(xs).float().cpu().numpy(); hf = flip_heatmaps_back(m(xs.flip(dims=[3])).float().cpu().numpy())
            hm = (h + shift_heatmap_x(hf, 1)) * .5
            if s != 1.0:  # back to the native heatmap grid
                hm = F.interpolate(torch.from_numpy(hm), size=(64, 48), mode='bilinear', align_corners=False).numpy()
            out[s].append(hm.astype(np.float16))
        meta['dom'] += list(b['domain']); meta['ids'] += [int(i) for i in b['image_id']]; meta['cps'].append(b['crop_params'].numpy())
    del m
    return {s: np.concatenate(v) for s, v in out.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--scales', nargs='+', type=float, default=[1.0])
    ap.add_argument('--annotation-dir', default='data/annotations/specialization')
    ap.add_argument('--image-root', default='data')
    ap.add_argument('--domains', nargs='+', default=['market', 'pa100k', 'peta'])
    ap.add_argument('--val-fraction', type=float, default=.1)
    ap.add_argument('--seed', type=int, default=2027)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = torch.load(a.checkpoints[0], map_location='cpu', weights_only=False)['config']
    ds = PoseDataset(a.annotation_dir, a.image_root, a.domains, input_hw=cfg['data']['input_size'],
                     split='val', val_fraction=a.val_fraction, seed=a.seed, augment=False)
    cache = {}
    for c in a.checkpoints:
        cache[c], meta = run_model(c, ds, dev, a.scales)
    dom = np.array(meta['dom']); ids = np.array(meta['ids']); cps = np.concatenate(meta['cps'])

    def evaluate(hms):
        aps = []
        for d in a.domains:
            sel = np.where(dom == d)[0]; preds = []
            for s in range(0, len(sel), 256):
                ix = sel[s:s + 256]
                hm = np.mean([h[ix].astype(np.float32) for h in hms], axis=0)
                pts, peaks = decode_dark(hm, kernel=7, sigma=1.0)
                score = calibrated_joint_confidence(peaks)[:, SCORED_INDEX].mean(1)
                for j, i in enumerate(ix):
                    p = from_crop(pts[j], cps[i])
                    preds.append({'image_id': int(ids[i]), 'category_id': 1, 'score': float(score[j]),
                                  'keypoints': [v for q in p for v in (float(q[0]), float(q[1]), 1.0)]})
            r = coco_ap_for_domain(Path(a.annotation_dir) / f'train_{d}.json', preds, set(int(ids[i]) for i in sel))
            aps.append((r['AP'], r['AP75']))
        return np.mean([x[0] for x in aps]), np.mean([x[1] for x in aps])

    def show(name, hms):
        ap_, ap75 = evaluate(hms); print(f'{name:60s} AP={ap_:.4f} AP75={ap75:.4f}', flush=True)

    for c in a.checkpoints:
        for s in a.scales: show(f'{Path(c).parent.name} scale {s}', [cache[c][s]])
        if len(a.scales) > 1: show(f'{Path(c).parent.name} all scales', [cache[c][s] for s in a.scales])
    import itertools
    for c1, c2 in itertools.combinations(a.checkpoints, 2):
        n = f'{Path(c1).parent.name}+{Path(c2).parent.name}'
        show(f'pair {n} (scale 1.0)', [cache[c1][1.0], cache[c2][1.0]])
        if len(a.scales) > 1: show(f'pair {n} x all scales', [cache[c][s] for c in (c1, c2) for s in a.scales])
    if len(a.checkpoints) > 1:
        show('ensemble of all checkpoints (scale 1.0)', [cache[c][1.0] for c in a.checkpoints])
        if len(a.scales) > 1:
            show('ensemble all checkpoints x all scales', [cache[c][s] for c in a.checkpoints for s in a.scales])


if __name__ == '__main__':
    main()
