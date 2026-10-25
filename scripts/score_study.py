#!/usr/bin/env python3
"""Instance-score study on a LODO checkpoint (trained WITHOUT the held-out domain). Source data only; hidden test is never used.

Step 1 (collect): run the model with the SAME inference as run.py (flip TTA shift 1 + DARK), cache per-joint peaks, flip
disagreement, heatmap spread, crop-scale and GT OKS per instance.
Step 2 (score): compare instance score functions by COCO AP on image-hash 2-fold cross-fitting (fit on one half, evaluate on the other).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import (decode_dark, flip_heatmaps_back, shift_heatmap_x, from_crop, SCORED_INDEX,
                               calibrated_joint_confidence)
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model

SIG = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])  # COCO slots


def oks(pred, gt_kp, area):
    g = np.asarray(gt_kp, np.float64).reshape(17, 3)
    vis = g[:, 2] > 0
    vis[3:5] = False
    if not vis.any():
        return 0.
    d2 = ((pred[:, :2] - g[:, :2]) ** 2).sum(1)
    e = d2 / ((2 * SIG) ** 2) / (area + np.spacing(1)) / 2
    return float(np.exp(-e[vis]).mean())


@torch.no_grad()
def collect(a, dev):
    obj = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = obj['config']; model = build_model(cfg); model.load_state_dict(obj['state_dict']); model.to(dev).eval()
    ds = PoseDataset(a.annotation_dir, a.image_root, [a.domain], input_hw=cfg['data']['input_size'], split=a.split,
                     val_fraction=.1, seed=2027, augment=False)
    raw = json.loads((Path(a.annotation_dir) / f'train_{a.domain}.json').read_text())
    ann = {int(x['image_id']): x for x in raw['annotations']}
    loader = DataLoader(ds, batch_size=64, num_workers=6)
    F = {k: [] for k in ('ids', 'peaks', 'disagree', 'spread', 'oks', 'pts', 'scale')}
    for b in loader:
        x = b['image'].to(dev)
        h1 = model(x).float().cpu().numpy()
        h2 = shift_heatmap_x(flip_heatmaps_back(model(x.flip(dims=[3])).float().cpu().numpy()), 1)
        hm = (h1 + h2) * .5
        pts, peaks = decode_dark(hm, 7, 1.0)
        p1, _ = decode_dark(h1, 7, 1.0); p2, _ = decode_dark(h2, 7, 1.0)
        dis = np.linalg.norm(p1 - p2, axis=-1)                       # crop pixels (heatmap cell = 4)
        flat = hm.reshape(hm.shape[0], 17, -1)
        pk = flat.max(-1)
        spread = np.clip(flat, 0, None).sum(-1) / np.maximum(pk, 1e-6)  # heatmap mass / peak (blob area proxy)
        cp = b['crop_params'].numpy()
        for i in range(len(pts)):
            iid = int(b['image_id'][i]); op = from_crop(pts[i], cp[i])
            g = ann[iid]
            F['ids'].append(iid); F['peaks'].append(peaks[i]); F['disagree'].append(dis[i]); F['spread'].append(spread[i])
            F['pts'].append(op); F['scale'].append(cp[i][2]); F['oks'].append(oks(op, g['keypoints'], float(g.get('area', g['bbox'][2] * g['bbox'][3]))))
    return {k: np.asarray(v) for k, v in F.items()}


def ap_of(a, F, idx, score):
    preds = [{'image_id': int(F['ids'][i]), 'category_id': 1, 'score': float(s),
              'keypoints': [v for p in F['pts'][i] for v in (float(p[0]), float(p[1]), 1.0)]} for i, s in zip(idx, score)]
    return coco_ap_for_domain(Path(a.annotation_dir) / f'train_{a.domain}.json', preds, set(int(F['ids'][i]) for i in idx))['AP']


def feats(F):
    pk = F['peaks'][:, SCORED_INDEX]; dis = F['disagree'][:, SCORED_INDEX]; sp = F['spread'][:, SCORED_INDEX]
    sc = F['scale'][:, None]
    return np.concatenate([pk, np.log1p(dis), np.log(sp + 1e-3), np.log(sc)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--domain', default='pa100k')
    ap.add_argument('--annotation-dir', default='/home/ubuntu/workspace/upar_official/mix_ann')
    ap.add_argument('--image-root', default='data')
    ap.add_argument('--split', default='all', help="'all' for a LODO held-out domain, 'val' for the hash val split of a trained domain")
    ap.add_argument('--cache', default='outputs/score_study_cache.npz')
    a = ap.parse_args()
    cache = Path(a.cache)
    if cache.exists():
        F = dict(np.load(cache))
    else:
        F = collect(a, torch.device('cuda')); np.savez(cache, **F)
    n = len(F['ids']); print('instances', n, 'mean GT-OKS', F['oks'].mean())
    half = np.array([int(hashlib.sha256(str(i).encode()).hexdigest(), 16) % 2 for i in F['ids']])
    X = feats(F); y = F['oks']
    base = lambda idx: calibrated_joint_confidence(F['peaks'][idx], .25, .15)[:, SCORED_INDEX].mean(1)
    pk = F['peaks'][:, SCORED_INDEX]
    cands = {
        'baseline mean-sigmoid(0.25,0.15)': base,
        'mean peak': lambda idx: pk[idx].mean(1),
        'min peak': lambda idx: pk[idx].min(1),
        'geo-mean peak': lambda idx: np.exp(np.log(np.clip(pk[idx], 1e-3, 1)).mean(1)),
        'mean sigmoid(0.4,0.15)': lambda idx: calibrated_joint_confidence(F['peaks'][idx], .4, .15)[:, SCORED_INDEX].mean(1),
        'mean peak - 0.05*mean flip-disagree': lambda idx: pk[idx].mean(1) - .05 * F['disagree'][idx][:, SCORED_INDEX].mean(1),
    }
    res = {k: [] for k in list(cands) + ['ridge(all feats)', 'ridge(peaks only)']}
    for fit in (0, 1):
        tr = np.where(half == fit)[0]; te = np.where(half != fit)[0]
        for k, f in cands.items():
            res[k].append(ap_of(a, F, te, f(te)))
        for name, cols in (('ridge(all feats)', slice(None)), ('ridge(peaks only)', slice(0, 15))):
            mu = X[tr][:, cols].mean(0); sd = X[tr][:, cols].std(0) + 1e-6
            Z = (X[tr][:, cols] - mu) / sd
            w = np.linalg.solve(Z.T @ Z + 10 * np.eye(Z.shape[1]), Z.T @ (y[tr] - y[tr].mean()))
            res[name].append(ap_of(a, F, te, ((X[te][:, cols] - mu) / sd) @ w + y[tr].mean()))
    for k, v in res.items():
        print(f'{k:42s} AP fold1={v[0]:.4f} fold2={v[1]:.4f} mean={np.mean(v):.4f}', flush=True)
    print('upper bound (score = true OKS, 2-fold avg):', np.mean([ap_of(a, F, np.where(half != f)[0], y[np.where(half != f)[0]]) for f in (0, 1)]))


if __name__ == '__main__':
    main()
