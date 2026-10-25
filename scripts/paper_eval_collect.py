#!/usr/bin/env python3
"""Stage 1 for the paper evaluation: per-instance OKS and scores for several inference variants on the held-out 10% validation
split (hash split, seed 2027) of market / pa100k / peta. Source data only; hidden test never touched.

Variants (decode x flip x shift x score) are all computed from the same forward passes. Output: outputs/paper_eval/<name>.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import (decode_numpy, decode_dark, flip_heatmaps_back, shift_heatmap_x, from_crop, SCORED_INDEX,
                               calibrated_joint_confidence)
from pose_src.models import build_model
from scripts.score_study import oks

# name -> (flip, shift, decoder)
GEOM = {'noflip_argmax': (False, 0, 'argmax'), 'noflip_dark': (False, 0, 'dark'), 'flip0_dark': (True, 0, 'dark'),
        'flip1_argmax': (True, 1, 'argmax'), 'flip1_dark': (True, 1, 'dark')}


@torch.no_grad()
def run(ckpt, domains, ann_dir, dev, batch=64):
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = obj['config']; model = build_model(cfg); model.load_state_dict(obj['state_dict']); model.to(dev).eval()
    out = {}
    for dom in domains:
        ds = PoseDataset(ann_dir, 'data', [dom], input_hw=cfg['data']['input_size'], split='val', val_fraction=.1, seed=2027, augment=False)
        raw = json.loads((Path(ann_dir) / f'train_{dom}.json').read_text())
        ann = {int(a['image_id']): a for a in raw['annotations']}
        rec = {k: [] for k in ('ids', 'dom')}
        for g in GEOM:
            rec[g + '_oks'] = []; rec[g + '_sig'] = []; rec[g + '_mp'] = []
        rec['pts'] = []; rec['box'] = []
        for b in DataLoader(ds, batch_size=batch, num_workers=4):
            x = b['image'].to(dev)
            h1 = model(x).float().cpu().numpy()
            h2 = flip_heatmaps_back(model(x.flip(dims=[3])).float().cpu().numpy())
            cp = b['crop_params'].numpy()
            for g, (flip, shift, dec) in GEOM.items():
                hm = h1 if not flip else (h1 + shift_heatmap_x(h2, shift)) * .5
                pts, pk = decode_dark(hm, 7, 1.0) if dec == 'dark' else decode_numpy(hm)
                sig = calibrated_joint_confidence(pk, .25, .15)[:, SCORED_INDEX].mean(1); mp = pk[:, SCORED_INDEX].mean(1)
                for i in range(len(pts)):
                    iid = int(b['image_id'][i]); a = ann[iid]
                    op = from_crop(pts[i], cp[i])
                    rec[g + '_oks'].append(oks(op, a['keypoints'], float(a.get('area', a['bbox'][2] * a['bbox'][3]))))
                    rec[g + '_sig'].append(float(sig[i])); rec[g + '_mp'].append(float(mp[i]))
                    if g == 'flip1_dark':
                        rec['pts'].append(op); rec['box'].append(np.asarray(a['bbox'], np.float32))
            rec['ids'] += [int(v) for v in b['image_id']]; rec['dom'] += [dom] * len(b['image_id'])
        print(ckpt, dom, len(rec['ids']), flush=True)
        for k, v in rec.items():
            out.setdefault(k, []).extend(v)
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--domains', nargs='+', default=['market', 'pa100k', 'peta'])
    ap.add_argument('--annotation-dir', default='/home/ubuntu/workspace/upar_official/mix_ann')
    a = ap.parse_args()
    Path('outputs/paper_eval').mkdir(parents=True, exist_ok=True)
    np.savez(f'outputs/paper_eval/{a.name}.npz', **run(a.checkpoint, a.domains, a.annotation_dir, torch.device('cuda')))
    print('saved', a.name)


if __name__ == '__main__':
    main()
