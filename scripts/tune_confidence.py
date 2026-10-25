#!/usr/bin/env python3
"""Grid-search confidence_bias / confidence_temperature on SOURCE validation only.

Runs the checkpoint once (flip TTA, same decode as submission/run.py), caches per-joint
heatmap peaks, then scores every (bias, temperature) with pycocotools. Never touches hidden data.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import decode_numpy, flip_heatmaps_back, from_crop, SCORED_INDEX, calibrated_joint_confidence
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model


@torch.no_grad()
def collect(ckpt, args, dev):
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = obj['config']; model = build_model(cfg); model.load_state_dict(obj['state_dict']); model.to(dev).eval()
    data = cfg['data']; domains = args.domains
    ds = PoseDataset(args.annotation_dir, args.image_root, domains, input_hw=data['input_size'],
                     split='all' if args.holdout_domain else 'val', val_fraction=args.val_fraction,
                     seed=args.seed, augment=False)
    loader = DataLoader(ds, batch_size=64, num_workers=6)
    rec = {d: {'peaks': [], 'pts': [], 'ids': []} for d in domains}
    for b in loader:
        x = b['image'].to(dev)
        hm = model(x).float().cpu().numpy()
        hm = (hm + flip_heatmaps_back(model(x.flip(dims=[3])).float().cpu().numpy())) * .5
        pts, peaks = decode_numpy(hm)
        cp = b['crop_params'].numpy()
        for i in range(len(pts)):
            r = rec[b['domain'][i]]
            r['peaks'].append(peaks[i]); r['pts'].append(from_crop(pts[i], cp[i])); r['ids'].append(int(b['image_id'][i]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--annotation-dir', default='data/annotations/specialization')
    ap.add_argument('--image-root', default='data')
    ap.add_argument('--domains', nargs='+', default=['market', 'pa100k', 'peta'])
    ap.add_argument('--holdout-domain', default=None, help='use a checkpoint trained WITHOUT this domain')
    ap.add_argument('--val-fraction', type=float, default=.1)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    if args.holdout_domain: args.domains = [args.holdout_domain]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rec = collect(args.checkpoint, args, dev)
    grid = [(b, t) for b in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5) for t in (0.05, 0.1, 0.15, 0.25, 0.4)]
    results = []
    for bias, temp in grid:
        aps = []
        for d, r in rec.items():
            peaks = np.stack(r['peaks']); score = calibrated_joint_confidence(peaks, bias, temp)[:, SCORED_INDEX].mean(1)
            preds = [{'image_id': i, 'category_id': 1, 'score': float(s),
                      'keypoints': [v for p in pts for v in (float(p[0]), float(p[1]), 1.0)]}
                     for i, s, pts in zip(r['ids'], score, r['pts'])]
            aps.append(coco_ap_for_domain(Path(args.annotation_dir) / f'train_{d}.json', preds, set(r['ids']))['AP'])
        results.append({'bias': bias, 'temperature': temp, 'mean_AP': float(np.mean(aps)), 'per_domain': aps})
        print(f'bias={bias:.2f} temp={temp:.2f} mean_AP={np.mean(aps):.4f}', flush=True)
    best = max(results, key=lambda x: x['mean_AP'])
    print('BEST', json.dumps(best)); print('default(0.25,0.15)', [r for r in results if r['bias'] == .25 and r['temperature'] == .15][0]['mean_AP'])
    if args.output: Path(args.output).write_text(json.dumps({'best': best, 'all': results}, indent=1))

if __name__ == '__main__':
    main()
