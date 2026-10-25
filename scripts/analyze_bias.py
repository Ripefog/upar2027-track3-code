#!/usr/bin/env python3
"""Error analysis on SOURCE validation only: systematic per-joint bias (px in crop space), error vs joint, DARK+flip decoded."""
import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import decode_dark, flip_heatmaps_back, shift_heatmap_x, to_crop, JOINTS
from pose_src.models import build_model

def main():
    ck = sys.argv[1]
    dev = torch.device('cuda')
    obj = torch.load(ck, map_location='cpu', weights_only=False); cfg = obj['config']
    m = build_model(cfg); m.load_state_dict(obj['state_dict']); m.to(dev).eval()
    ds = PoseDataset('data/annotations/specialization', 'data', ['market', 'pa100k', 'peta'], input_hw=cfg['data']['input_size'],
                     split='val', val_fraction=.1, seed=2027, augment=False)
    P, G, V, D = [], [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, num_workers=4):
            x = b['image'].to(dev)
            h = m(x).float().cpu().numpy(); hf = flip_heatmaps_back(m(x.flip(dims=[3])).float().cpu().numpy())
            pts, _ = decode_dark((h + shift_heatmap_x(hf, 1)) * .5, kernel=7, sigma=1.0)
            P.append(pts); cp = b['crop_params'].numpy(); G.append(np.stack([to_crop(g, c) for g, c in zip(b['original_points'].numpy(), cp)])); V.append(b['original_visible'].numpy()); D += list(b['domain'])
    P = np.concatenate(P); G = np.concatenate(G); V = np.concatenate(V) > 0; D = np.array(D)
    print('keys ok', P.shape, G.shape)
    err = P - G                      # crop-space px (256x192)
    for d in ['all', 'market', 'pa100k', 'peta']:
        sel = np.ones(len(D), bool) if d == 'all' else D == d
        print(f'--- {d} n={sel.sum()}')
        for j in (0, 1, 2, 5, 7, 9, 11, 13, 15):
            v = V[sel, j]; e = err[sel][v, j]
            print(f'{JOINTS[j]:15s} bias dx={e[:,0].mean():+.2f} dy={e[:,1].mean():+.2f}  mean|e|={np.linalg.norm(e,axis=1).mean():.2f}  p90={np.percentile(np.linalg.norm(e,axis=1),90):.2f}')


if __name__ == "__main__":
    main()
