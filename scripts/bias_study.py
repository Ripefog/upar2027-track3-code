#!/usr/bin/env python3
"""Systematic per-joint offset study on cached source predictions (score_study caches). Fits a per-joint (dx,dy) correction
in units of sqrt(GT area) on one image-hash half and measures AP on the other half. Source data only."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.geometry import SCORED_INDEX
from scripts.score_study import oks, SIG
from pose_src.metrics import coco_ap_for_domain

AD = Path('/home/ubuntu/workspace/upar_official/mix_ann')
for dom, cache in (('pa100k', 'outputs/score_study_cache.npz'), ('market', 'outputs/score_study_cache_market.npz'), ('peta', 'outputs/score_study_cache_peta.npz')):
    F = dict(np.load(cache)); raw = json.loads((AD / f'train_{dom}.json').read_text())
    ann = {int(a['image_id']): a for a in raw['annotations']}
    n = len(F['ids']); half = np.array([int(hashlib.sha256(str(i).encode()).hexdigest(), 16) % 2 for i in F['ids']])
    D = np.zeros((n, 17, 2)); V = np.zeros((n, 17), bool); S = np.zeros(n)
    for k, iid in enumerate(F['ids']):
        g = ann[int(iid)]; kp = np.asarray(g['keypoints'], float).reshape(17, 3)
        area = float(g.get('area', g['bbox'][2] * g['bbox'][3])); S[k] = np.sqrt(area)
        V[k] = kp[:, 2] > 0; V[k, 3:5] = False
        D[k] = (kp[:, :2] - F['pts'][k][:, :2]) / S[k]          # GT - pred, normalised
    print(f'== {dom} n={n}')
    names = ['nose', 'hbot', 'htop', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
    for j in SCORED_INDEX:
        m = V[:, j]; dx = D[m, j, 0]; dy = D[m, j, 1]
        print(f'  {names[j]:5s} mean dx={dx.mean():+.4f} dy={dy.mean():+.4f} (sd {dx.std():.3f}/{dy.std():.3f}) halfA dy={D[m&(half==0),j,1].mean():+.4f} halfB dy={D[m&(half==1),j,1].mean():+.4f}')
    res = []
    for fit in (0, 1):
        tr = half == fit; te = np.where(half != fit)[0]
        off = np.zeros((17, 2))
        for j in SCORED_INDEX:
            m = V[:, j] & tr
            if m.sum() > 50: off[j] = np.median(D[m, j], axis=0)   # robust
        sc = F['peaks'][:, SCORED_INDEX].mean(1)
        for tag, use in (('base', False), ('corrected', True)):
            preds = []
            for k in te:
                pts = F['pts'][k].copy()
                if use: pts[:, :2] = pts[:, :2] + off * S[k]
                preds.append({'image_id': int(F['ids'][k]), 'category_id': 1, 'score': float(sc[k]),
                              'keypoints': [v for p in pts for v in (float(p[0]), float(p[1]), 1.0)]})
            res.append((tag, coco_ap_for_domain(AD / f'train_{dom}.json', preds, set(int(F['ids'][k]) for k in te))['AP']))
    for tag in ('base', 'corrected'):
        print(f'  AP {tag}: {np.mean([a for t, a in res if t == tag]):.4f}')
