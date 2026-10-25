#!/usr/bin/env python3
"""Inference-time padding / box-jitter TTA study on a LODO checkpoint (held-out domain), source data only.

Each variant crops with (pad, shift), runs flip-TTA (shift 1) + DARK like run.py, maps coordinates back to the image and keeps
per-joint peaks. Combinations average coordinates over variants (score = mean peak over variants). Reports COCO AP per combo.
"""
import argparse, itertools, json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset, resolve_image
from pose_src.geometry import (crop_params, crop_image, as_tensor, from_crop, decode_dark, flip_heatmaps_back, shift_heatmap_x,
                               SCORED_INDEX)
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--domain', default='pa100k')
    ap.add_argument('--split', default='all')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--annotation-dir', default='/home/ubuntu/workspace/upar_official/mix_ann')
    ap.add_argument('--pads', type=float, nargs='+', default=None, help='only sweep these paddings (no shift/combos except pad pairs)')
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda')
    obj = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = obj['config']; hw = tuple(cfg['data']['input_size'])
    model = build_model(cfg); model.load_state_dict(obj['state_dict']); model.to(dev).eval()
    ds = PoseDataset(a.annotation_dir, 'data', [a.domain], input_hw=hw, split=a.split, val_fraction=.1, seed=2027, augment=False)
    recs = ds.records
    random.Random(0).shuffle(recs := list(recs)); recs = recs[:a.n]
    variants = {'p1.25': (1.25, (0, 0)), 'p1.10': (1.10, (0, 0)), 'p1.15': (1.15, (0, 0)), 'p1.35': (1.35, (0, 0)), 'p1.45': (1.45, (0, 0)),
                'p1.25 dx+': (1.25, (.04, 0)), 'p1.25 dx-': (1.25, (-.04, 0)), 'p1.25 dy+': (1.25, (0, .04)), 'p1.25 dy-': (1.25, (0, -.04))}
    if a.pads:
        variants = {f'p{p:g}': (p, (0, 0)) for p in a.pads}
    out = {k: {'pts': [], 'peaks': []} for k in variants}
    ids = [int(r['image_id']) for r in recs]
    B = 32
    with torch.no_grad():
        for s in range(0, len(recs), B):
            chunk = recs[s:s + B]
            imgs = []
            for r in chunk:
                with Image.open(r.get('image_path') or resolve_image(ds.root, r['domain'], r['file_name'])) as fp:
                    imgs.append(fp.convert('RGB'))
            for name, (pad, sh) in variants.items():
                params = [crop_params(r['bbox'], hw, pad=pad, shift=sh) for r in chunk]
                x = torch.from_numpy(np.stack([as_tensor(crop_image(im, p, hw)) for im, p in zip(imgs, params)]).copy()).to(dev)
                h1 = model(x).float().cpu().numpy()
                h2 = shift_heatmap_x(flip_heatmaps_back(model(x.flip(dims=[3])).float().cpu().numpy()), 1)
                pts, pk = decode_dark((h1 + h2) * .5, 7, 1.0)
                for i, p in enumerate(params):
                    out[name]['pts'].append(from_crop(pts[i], p)); out[name]['peaks'].append(pk[i])
            if (s // B) % 20 == 0:
                print('batch', s // B, flush=True)
    for k in out:
        out[k]['pts'] = np.stack(out[k]['pts']); out[k]['peaks'] = np.stack(out[k]['peaks'])
    annp = Path(a.annotation_dir) / f'train_{a.domain}.json'

    def ap_of(names):
        pts = np.mean([out[k]['pts'] for k in names], 0)
        pk = np.mean([out[k]['peaks'][:, SCORED_INDEX].mean(1) for k in names], 0)
        preds = [{'image_id': ids[i], 'category_id': 1, 'score': float(pk[i]),
                  'keypoints': [v for p in pts[i] for v in (float(p[0]), float(p[1]), 1.0)]} for i in range(len(ids))]
        return coco_ap_for_domain(annp, preds, set(ids))['AP']

    res = {}
    for k in variants:
        res[k] = ap_of([k]); print(f'{k:28s} {res[k]:.4f}', flush=True)
    combos = {} if a.pads else {'pad 1.15+1.25+1.35': ['p1.15', 'p1.25', 'p1.35'], 'pad 1.25+1.45': ['p1.25', 'p1.45'], 'pad 1.15+1.25': ['p1.15', 'p1.25'],
              'pad 1.25+1.35': ['p1.25', 'p1.35'], 'shift x3 (dx)': ['p1.25', 'p1.25 dx+', 'p1.25 dx-'], 'shift x3 (dy)': ['p1.25', 'p1.25 dy+', 'p1.25 dy-'],
              'shift x5': ['p1.25', 'p1.25 dx+', 'p1.25 dx-', 'p1.25 dy+', 'p1.25 dy-'], 'all 9': list(variants)}
    for k, v in combos.items():
        res[k] = ap_of(v); print(f'{k:28s} {res[k]:.4f}', flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
