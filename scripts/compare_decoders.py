#!/usr/bin/env python3
"""Compare heatmap decoders / flip-shift on SOURCE validation (never hidden data).

Runs the checkpoint once (plain + flipped input), caches heatmaps, then evaluates every variant with
pycocotools so differences are purely due to decoding.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import decode_numpy, decode_dark, flip_heatmaps_back, from_crop, SCORED_INDEX, calibrated_joint_confidence
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model


def shift_right(hm, n):
    if n == int(n): n = int(n)
    if n == 0: return hm
    if n != int(n):  # fractional shift = linear interpolation between integer shifts
        lo = int(np.floor(n)); a = n - lo
        return (1 - a) * shift_right(hm, lo) + a * shift_right(hm, lo + 1)
    out = np.zeros_like(hm)
    if n > 0: out[..., n:] = hm[..., :-n]; out[..., :n] = hm[..., :1]
    else: out[..., :n] = hm[..., -n:]; out[..., n:] = hm[..., -1:]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--annotation-dir', default='data/annotations/specialization')
    ap.add_argument('--image-root', default='data')
    ap.add_argument('--domains', nargs='+', default=['market', 'pa100k', 'peta'])
    ap.add_argument('--holdout-domain', default=None)
    ap.add_argument('--val-fraction', type=float, default=.1)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    if a.holdout_domain: a.domains = [a.holdout_domain]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = obj['config']; m = build_model(cfg); m.load_state_dict(obj['state_dict']); m.to(dev).eval()
    ds = PoseDataset(a.annotation_dir, a.image_root, a.domains, input_hw=cfg['data']['input_size'],
                     split='all' if a.holdout_domain else 'val', val_fraction=a.val_fraction, seed=a.seed, augment=False)
    dom, ids, cps, plain, flipped = [], [], [], [], []
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        x = b['image'].to(dev)
        plain.append(m(x).float().cpu().numpy().astype(np.float16))
        flipped.append(flip_heatmaps_back(m(x.flip(dims=[3])).float().cpu().numpy()).astype(np.float16))
        dom += list(b['domain']); ids += [int(i) for i in b['image_id']]; cps.append(b['crop_params'].numpy())
    plain = np.concatenate(plain); flipped = np.concatenate(flipped); cps = np.concatenate(cps)
    dom = np.array(dom); ids = np.array(ids)

    def evaluate(decoder, shift, use_flip=True, **kw):
        aps = []
        for d in a.domains:
            sel = np.where(dom == d)[0]
            preds = []
            for s in range(0, len(sel), 256):
                ix = sel[s:s + 256]
                hm = plain[ix].astype(np.float32)
                if use_flip: hm = (hm + shift_right(flipped[ix].astype(np.float32), shift)) * .5
                pts, peaks = (decode_dark(hm, **kw) if decoder == 'dark' else decode_numpy(hm))
                score = calibrated_joint_confidence(peaks)[:, SCORED_INDEX].mean(1)
                for j, i in enumerate(ix):
                    p = from_crop(pts[j], cps[i])
                    preds.append({'image_id': int(ids[i]), 'category_id': 1, 'score': float(score[j]),
                                  'keypoints': [v for q in p for v in (float(q[0]), float(q[1]), 1.0)]})
            r = coco_ap_for_domain(Path(a.annotation_dir) / f'train_{d}.json', preds, set(int(ids[i]) for i in sel))
            aps.append((r['AP'], r['AP75'], r['AP50']))
        return [float(np.mean([x[i] for x in aps])) for i in range(3)]

    variants = []
    for sh in (0.875, 1.0, 1.25):
        variants.append((f'old, flip shift {sh}', 'old', sh, True, {}))
        variants.append((f'DARK k=7 s=1.0, flip shift {sh}', 'dark', sh, True, {'kernel': 7, 'sigma': 1.0}))
    res = []
    for name, dec, sh, fl, kw in variants:
        ap_, ap75, ap50 = evaluate(dec, sh, fl, **kw)
        res.append({'name': name, 'AP': ap_, 'AP75': ap75, 'AP50': ap50})
        print(f'{name:42s} AP={ap_:.4f} AP75={ap75:.4f} AP50={ap50:.4f}', flush=True)
    if a.output: Path(a.output).write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
