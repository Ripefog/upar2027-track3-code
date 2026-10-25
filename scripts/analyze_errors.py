#!/usr/bin/env python3
"""Per-joint / per-domain / per-size error analysis on SOURCE validation only (never hidden data)."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import decode_dark, flip_heatmaps_back, shift_heatmap_x, from_crop, JOINTS
from pose_src.models import build_model

SIG = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])  # COCO sigmas by slot index (what COCOeval applies)
SCORED = [0, 1, 2] + list(range(5, 17))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--annotation-dir', default='/home/ubuntu/workspace/upar_official/mix_ann')
    ap.add_argument('--domains', nargs='+', default=['market', 'pa100k', 'peta', 'posetrack', 'coco'])
    a = ap.parse_args()
    dev = torch.device('cuda')
    obj = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    m = build_model(obj['config']); m.load_state_dict(obj['state_dict']); m.to(dev).eval()
    ds = PoseDataset(a.annotation_dir, 'data', a.domains, input_hw=obj['config']['data']['input_size'],
                     split='val', val_fraction=.1, seed=2027, augment=False)
    rec = []
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        x = b['image'].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = m(x).float().cpu().numpy(); hf = flip_heatmaps_back(m(x.flip(dims=[3])).float().cpu().numpy())
        hm = (h + shift_heatmap_x(hf, 1)) * .5
        pts, peaks = decode_dark(hm, kernel=7, sigma=1.0)
        gt = b['original_points'].numpy(); vis = b['original_visible'].numpy(); cp = b['crop_params'].numpy(); box = b['box'].numpy()
        for i in range(len(pts)):
            p = from_crop(pts[i], cp[i]); area = max(1., float(box[i][2] * box[i][3]))
            d2 = ((p - gt[i]) ** 2).sum(-1)
            e = d2 / (2 * area * (2 * SIG) ** 2 + 1e-9) * 1.0
            oks = np.exp(-e)
            rec.append((b['domain'][i], area, oks, vis[i].astype(bool), np.sqrt(d2) / np.sqrt(area)))
    out = {}
    def agg(sel):
        r = {}
        for k in SCORED:
            v = [(x[2][k], x[4][k]) for x in sel if x[3][k]]
            if v:
                o = np.array(v); r[JOINTS[k] if isinstance(JOINTS, (list, tuple)) else str(k)] = {
                    'n': len(v), 'mean_oks': round(float(o[:, 0].mean()), 4), 'acc_oks50': round(float((o[:, 0] >= .5).mean()), 4),
                    'p90_err_norm': round(float(np.percentile(o[:, 1], 90)), 4)}
        return r
    out['all'] = agg(rec)
    for d in a.domains: out['domain_' + d] = agg([x for x in rec if x[0] == d])
    areas = np.array([x[1] for x in rec]); q = np.quantile(areas, [.33, .66])
    for name, lo, hi in [('small', 0, q[0]), ('mid', q[0], q[1]), ('big', q[1], 1e12)]:
        out['size_' + name] = agg([x for x in rec if lo <= x[1] < hi])
    # person-level mean OKS over scored visible joints
    per = lambda sel: float(np.mean([x[2][[k for k in SCORED if x[3][k]]].mean() for x in sel if x[3][SCORED].any()]))
    out['person_mean_oks'] = {d: round(per([x for x in rec if x[0] == d]), 4) for d in a.domains}
    Path(a.output).write_text(json.dumps(out, indent=1)); print('done', len(rec))

if __name__ == '__main__':
    main()
