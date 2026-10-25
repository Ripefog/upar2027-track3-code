#!/usr/bin/env python3
"""Stage 2 for the paper evaluation (CPU): paired-bootstrap confidence intervals, inference-component ablation, training-recipe table,
efficiency table and figures (bias by domain, qualitative). Source validation data only. Reads outputs/paper_eval/*.npz."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path('outputs/paper_eval'); FIG = Path('docs/figures'); FIG.mkdir(parents=True, exist_ok=True)
DOMS = ['market', 'pa100k', 'peta']
THR = np.arange(.5, .951, .05); REC = np.linspace(0, 1, 101)


def ap_fast(oks, score):
    """COCO-style keypoint AP for one prediction per GT instance (matches COCOeval for this setting; validated below)."""
    n = len(oks); o = oks[np.argsort(-score, kind='mergesort')]; aps = []
    for t in THR:
        tp = (o >= t).astype(np.float64); tpc = np.cumsum(tp); fpc = np.cumsum(1 - tp)
        rc = tpc / n; pr = tpc / (tpc + fpc + np.spacing(1)); pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, REC, side='left'); q = np.zeros(101); ok = inds < len(pr); q[ok] = pr[inds[ok]]; aps.append(q.mean())
    return float(np.mean(aps))


def load(name):
    z = np.load(OUT / f'{name}.npz', allow_pickle=True); return {k: z[k] for k in z.files}


def stat(d, key_oks, key_sc, idx_by_dom):
    per = {k: ap_fast(d[key_oks][i], d[key_sc][i]) for k, i in idx_by_dom.items()}
    w = sum(per[k] * len(idx_by_dom[k]) for k in per) / sum(len(i) for i in idx_by_dom.values())
    return per, w


def boot(variants, n_boot=1000, seed=0):
    """variants: {label: (d, oks_key, score_key)}; same resampled indices for all (paired). Returns point, samples."""
    d0 = next(iter(variants.values()))[0]
    idx = {k: np.where(d0['dom'] == k)[0] for k in DOMS}
    point = {lab: stat(d, ko, ks, idx) for lab, (d, ko, ks) in variants.items()}
    rng = np.random.default_rng(seed); S = {lab: [] for lab in variants}
    for _ in range(n_boot):
        bi = {k: rng.choice(i, len(i), replace=True) for k, i in idx.items()}
        for lab, (d, ko, ks) in variants.items():
            S[lab].append(stat(d, ko, ks, bi)[1])
    return point, {k: np.asarray(v) for k, v in S.items()}


def ci(a): return np.percentile(a, [2.5, 97.5])


def main():
    md = []
    soup, mh, ll, base = (load(n) for n in ('soup', 'maskhead', 'llrd_dp', 'baseline'))
    n_by = {k: int((soup['dom'] == k).sum()) for k in DOMS}
    md.append(f"Held-out validation instances: {n_by} (total {sum(n_by.values())}); AP = COCO keypoint AP 0.50:0.95; 'weighted' = image-weighted mean over domains; 95% CI = paired bootstrap over images (1,000 resamples).\n")

    # validate ap_fast vs pycocotools (pa100k, final config)
    try:
        from pose_src.metrics import coco_ap_for_domain
        i = np.where(soup['dom'] == 'pa100k')[0]
        preds = [{'image_id': int(soup['ids'][k]), 'category_id': 1, 'score': float(soup['flip1_dark_mp'][k]),
                  'keypoints': [v for p in soup['pts'][k] for v in (float(p[0]), float(p[1]), 1.0)]} for k in i]
        ref = coco_ap_for_domain('/home/ubuntu/workspace/upar_official/mix_ann/train_pa100k.json', preds, set(int(soup['ids'][k]) for k in i))['AP']
        mine = ap_fast(soup['flip1_dark_oks'][i], soup['flip1_dark_mp'][i])
        md.append(f"ap_fast validation on pa100k val: pycocotools {ref:.4f} vs ap_fast {mine:.4f}\n")
    except Exception as e:
        md.append(f"ap_fast validation skipped: {e}\n")

    # T1: inference ablation on the soup
    steps = [('no flip, argmax, sigmoid', 'noflip_argmax', 'sig'), ('no flip, DARK, sigmoid', 'noflip_dark', 'sig'),
             ('flip (no shift), DARK, sigmoid', 'flip0_dark', 'sig'), ('flip + 1-cell shift, argmax, sigmoid', 'flip1_argmax', 'sig'),
             ('flip + 1-cell shift, DARK, sigmoid', 'flip1_dark', 'sig'), ('flip + 1-cell shift, DARK, mean-peak (final)', 'flip1_dark', 'mp')]
    var = {lab: (soup, f'{g}_oks', f'{g}_{s}') for lab, g, s in steps}
    point, S = boot(var)
    md.append("### Table 1. Inference components (final soup model)\n\n| Variant | market | pa100k | peta | weighted | 95% CI |\n|---|---:|---:|---:|---:|---|")
    for lab in var:
        per, w = point[lab]; lo, hi = ci(S[lab])
        md.append(f"| {lab} | {per['market']:.4f} | {per['pa100k']:.4f} | {per['peta']:.4f} | {w:.4f} | [{lo:.4f}, {hi:.4f}] |")
    final = 'flip + 1-cell shift, DARK, mean-peak (final)'
    md.append("\nPaired differences vs the final configuration (final - variant): mean [95% CI]\n")
    for lab in list(var)[:-1]:
        dlt = S[final] - S[lab]; lo, hi = ci(dlt); md.append(f"- vs {lab}: {dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # T2: training recipes and soup under both scores
    md.append("\n### Table 2. Training recipe, soup and instance score (flip + shift, DARK)\n\n| Model | score | market | pa100k | peta | weighted | 95% CI |\n|---|---|---:|---:|---:|---:|---|")
    models = [('plain fine-tune (baseline)', base), ('+ head-mask (A)', mh), ('+ LLRD + drop-path (B)', ll), ('soup(A, B)', soup)]
    var2 = {}
    for nm, d in models:
        for sname, sk in (('sigmoid', 'sig'), ('mean-peak', 'mp')):
            var2[(nm, sname)] = (d, 'flip1_dark_oks', f'flip1_dark_{sk}')
    point2, S2 = boot(var2)
    for (nm, sname), v in var2.items():
        per, w = point2[(nm, sname)]; lo, hi = ci(S2[(nm, sname)])
        md.append(f"| {nm} | {sname} | {per['market']:.4f} | {per['pa100k']:.4f} | {per['peta']:.4f} | {w:.4f} | [{lo:.4f}, {hi:.4f}] |")
    md.append("\nPaired differences (mean [95% CI])\n")
    pairs = [(('+ head-mask (A)', 'mean-peak'), ('plain fine-tune (baseline)', 'mean-peak')),
             (('+ LLRD + drop-path (B)', 'mean-peak'), ('+ head-mask (A)', 'mean-peak')),
             (('soup(A, B)', 'mean-peak'), ('+ head-mask (A)', 'mean-peak')), (('soup(A, B)', 'mean-peak'), ('+ LLRD + drop-path (B)', 'mean-peak')),
             (('soup(A, B)', 'mean-peak'), ('soup(A, B)', 'sigmoid')), (('+ head-mask (A)', 'mean-peak'), ('+ head-mask (A)', 'sigmoid')),
             (('+ LLRD + drop-path (B)', 'mean-peak'), ('+ LLRD + drop-path (B)', 'sigmoid'))]
    for a, b in pairs:
        dlt = S2[a] - S2[b]; lo, hi = ci(dlt); md.append(f"- {a[0]} [{a[1]}] - {b[0]} [{b[1]}]: {dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Efficiency
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        from pose_src.models import build_model
        obj = torch.load('outputs/vith_cocoonly_soup_llrd_mh/best.pth', map_location='cpu', weights_only=False)
        m = build_model(obj['config']).eval(); params = sum(p.numel() for p in m.parameters())
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 3, 256, 192))
        flops = fc.get_total_flops()
        md.append(f"\n### Table 3. Efficiency\n\n- Parameters: {params/1e6:.1f} M; forward FLOPs per 256x192 crop: {flops/1e9:.1f} G (counted by torch FlopCounterMode, multiply-adds counted as 2); flip TTA = 2 forward passes per box.\n"
                  "- Reported inference time of the submission (scoring program): 2,385.13 s for 30,245 boxes = 0.0789 s/box; ZIP size 2.37 GB (fp32 weights).")
    except Exception as e:
        md.append(f"\nEfficiency table skipped: {e}")
    (OUT / 'summary.md').write_text('\n'.join(md)); print('\n'.join(md))

    # Figure: bias by domain (from score_study caches)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from pose_src.geometry import SCORED_INDEX
    AD = Path('/home/ubuntu/workspace/upar_official/mix_ann')
    names = ['nose', 'head_bot', 'head_top', None, None, 'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri', 'r_wri', 'l_hip', 'r_hip', 'l_kne', 'r_kne', 'l_ank', 'r_ank']
    fig, ax = plt.subplots(figsize=(10, 3.4)); width = .27
    for di, (dom, cache, lab) in enumerate((('pa100k', 'outputs/score_study_cache.npz', 'pa100k (held-out domain)'),
                                            ('market', 'outputs/score_study_cache_market.npz', 'market (val)'), ('peta', 'outputs/score_study_cache_peta.npz', 'peta (val)'))):
        F = dict(np.load(cache)); raw = json.loads((AD / f'train_{dom}.json').read_text()); ann = {int(a['image_id']): a for a in raw['annotations']}
        vals = []
        for j in SCORED_INDEX:
            dys = []
            for k, iid in enumerate(F['ids']):
                g = ann[int(iid)]; kp = np.asarray(g['keypoints'], float).reshape(17, 3)
                if kp[j, 2] > 0: dys.append((kp[j, 1] - F['pts'][k][j, 1]) / np.sqrt(float(g.get('area', g['bbox'][2] * g['bbox'][3]))))
            vals.append(np.mean(dys))
        ax.bar(np.arange(len(SCORED_INDEX)) + (di - 1) * width, vals, width, label=lab)
    ax.set_xticks(np.arange(len(SCORED_INDEX))); ax.set_xticklabels([names[j] for j in SCORED_INDEX], rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=.6); ax.set_ylabel('mean vertical error\n(GT - pred) / sqrt(area)'); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / 'bias_by_domain.pdf'); fig.savefig(FIG / 'bias_by_domain.png', dpi=200); plt.close(fig)

    # Figure: qualitative (best / median / worst per domain, final soup predictions)
    from PIL import Image
    from pose_src.data import read_records, resolve_image
    EDGES = [(1, 2), (0, 1), (1, 5), (1, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (5, 6), (11, 13), (13, 15), (12, 14), (14, 16)]
    recs = {}
    for dom in DOMS:
        for r in read_records(str(AD), [dom]): recs[(dom, int(r['image_id']))] = r
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 6, figsize=(12, 7.2))
    for r_i, dom in enumerate(DOMS):
        i = np.where(soup['dom'] == dom)[0]; o = soup['flip1_dark_oks'][i]; order = np.argsort(o)
        picks = [order[-1], order[-len(order) // 3], order[len(order) // 2], order[len(order) // 20], order[3], order[0]]
        tags = ['best', 'good', 'median', 'poor (5th pct)', 'poor', 'worst']
        for c, (p, tg) in enumerate(zip(picks, tags)):
            k = i[p]; rec = recs[(dom, int(soup['ids'][k]))]
            img = Image.open(rec.get('image_path') or resolve_image(Path('data'), dom, rec['file_name'])).convert('RGB')
            x, y, w, h = soup['box'][k]; pad = .15
            l, t, rr, bb = int(x - pad * w), int(y - pad * h), int(x + (1 + pad) * w), int(y + (1 + pad) * h)
            axx = axes[r_i, c]; axx.imshow(img.crop((l, t, rr, bb)), extent=(l, rr, bb, t))
            pts = soup['pts'][k]
            for a_, b_ in EDGES: axx.plot([pts[a_, 0], pts[b_, 0]], [pts[a_, 1], pts[b_, 1]], '-', color='#00e0ff', lw=1.4)
            axx.scatter(pts[SCORED_INDEX, 0], pts[SCORED_INDEX, 1], s=9, c='#ff3b30', zorder=3)
            axx.set_title(f'{dom}: {tg}\nOKS {soup["flip1_dark_oks"][k]:.2f}', fontsize=8); axx.axis('off')
    fig.tight_layout(); fig.savefig(FIG / 'qualitative.pdf'); fig.savefig(FIG / 'qualitative.png', dpi=170); plt.close(fig)
    print('figures saved')


if __name__ == '__main__':
    main()
