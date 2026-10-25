#!/usr/bin/env python3
"""Failure analysis figures on the held-out source validation split (final soup model, flip+shift, DARK):
fig_failure_quant: (a) OKS vs person height (resolution), (b) per-joint accuracy for visible (v=2) vs occluded (v=1) GT joints, (c) OKS vs number of occluded joints
fig_failure_cases: examples of head, wrist, occlusion and low-resolution failures (GT green, prediction red, offending joints ringed)
Source validation data only; hidden test is never used."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import read_records, resolve_image
from scripts.score_study import SIG

AD = Path('/home/ubuntu/workspace/upar_official/mix_ann'); FIG = Path('docs/figures')
NAMES = ['nose', 'head_bottom', 'head_top', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
SC = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
EDGES = [(1, 2), (0, 1), (1, 5), (1, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (5, 6), (11, 13), (13, 15), (12, 14), (14, 16)]

z = np.load('outputs/paper_eval/soup.npz', allow_pickle=True)
ids, dom, pts, box, oks = z['ids'], z['dom'], z['pts'], z['box'], z['flip1_dark_oks']
n = len(ids)
gt = np.zeros((n, 17, 3)); area = np.zeros(n); comp = np.full((n, 17), np.nan)
ann = {d: {int(a['image_id']): a for a in json.loads((AD / f'train_{d}.json').read_text())['annotations']} for d in ('market', 'pa100k', 'peta')}
for i in range(n):
    a = ann[str(dom[i])][int(ids[i])]; gt[i] = np.asarray(a['keypoints'], float).reshape(17, 3)
    area[i] = float(a.get('area', a['bbox'][2] * a['bbox'][3]))
    d2 = ((pts[i][:, :2] - gt[i][:, :2]) ** 2).sum(1)
    c = np.exp(-(d2 / ((2 * SIG) ** 2) / (area[i] + np.spacing(1)) / 2))
    lab = gt[i][:, 2] > 0; lab[3:5] = False
    comp[i, lab] = c[lab]
height = box[:, 3]
nocc = ((gt[:, :, 2] == 1)[:, SC]).sum(1)

# ---------------- quantitative figure
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
bins = [0, 80, 110, 140, 200, 1e9]; lab = ['<80', '80-110', '110-140', '140-200', '>=200']
m = [oks[(height >= bins[k]) & (height < bins[k + 1])].mean() for k in range(5)]
c = [int(((height >= bins[k]) & (height < bins[k + 1])).sum()) for k in range(5)]
ax[0].bar(range(5), m, color='#4c78a8'); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f'{l}\n(n={k})' for l, k in zip(lab, c)], fontsize=8)
ax[0].set_ylim(.8, 1); ax[0].set_ylabel('mean OKS'); ax[0].set_xlabel('person box height (px)'); ax[0].set_title('(a) resolution')
for k, v in enumerate(m): ax[0].text(k, v + .003, f'{v:.3f}', ha='center', fontsize=8)
vis_m, occ_m, vis_n, occ_n = [], [], [], []
for j in SC:
    v2 = gt[:, j, 2] == 2; v1 = gt[:, j, 2] == 1
    vis_m.append(np.nanmean(comp[v2, j])); occ_m.append(np.nanmean(comp[v1, j])); vis_n.append(int(v2.sum())); occ_n.append(int(v1.sum()))
x = np.arange(len(SC)); ax[1].bar(x - .2, vis_m, .4, label='visible (v=2)', color='#4c78a8'); ax[1].bar(x + .2, occ_m, .4, label='occluded (v=1)', color='#e45756')
ax[1].set_xticks(x); ax[1].set_xticklabels([NAMES[j] for j in SC], rotation=60, ha='right', fontsize=7); ax[1].set_ylim(.7, 1); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_ylabel('mean per-joint OKS component'); ax[1].set_title('(b) visible vs occluded joints')
ob = [(0, 0), (1, 2), (3, 5), (6, 99)]; ol = ['0', '1-2', '3-5', '>=6']
mo = [oks[(nocc >= lo) & (nocc <= hi)].mean() for lo, hi in ob]; co = [int(((nocc >= lo) & (nocc <= hi)).sum()) for lo, hi in ob]
ax[2].bar(range(4), mo, color='#72b7b2'); ax[2].set_xticks(range(4)); ax[2].set_xticklabels([f'{l}\n(n={k})' for l, k in zip(ol, co)], fontsize=8)
ax[2].set_ylim(.8, 1); ax[2].set_xlabel('# occluded (v=1) scored joints per person'); ax[2].set_ylabel('mean OKS'); ax[2].set_title('(c) occlusion')
for k, v in enumerate(mo): ax[2].text(k, v + .003, f'{v:.3f}', ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig_failure_quant.pdf'); fig.savefig(FIG / 'fig_failure_quant.png', dpi=180); plt.close(fig)
stats = {'height_bins': dict(zip(lab, [(round(float(a), 4), b) for a, b in zip(m, c)])), 'occ_bins': dict(zip(ol, [(round(float(a), 4), b) for a, b in zip(mo, co)])),
         'per_joint_visible_vs_occluded': {NAMES[j]: (round(float(a), 4), vn, round(float(b), 4), on) for j, a, b, vn, on in zip(SC, vis_m, occ_m, vis_n, occ_n)}}
Path('outputs/paper_eval/failure_stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))

# ---------------- qualitative failure cases
recs = {}
for d in ('market', 'pa100k', 'peta'):
    for r in read_records(str(AD), [d]): recs[(d, int(r['image_id']))] = r
rest = [j for j in SC if j not in (0, 1, 2)]
head = np.nanmean(comp[:, [0, 1, 2]], 1); body = np.nanmean(comp[:, rest], 1); wr = np.nanmean(comp[:, [9, 10]], 1); body_nw = np.nanmean(comp[:, [j for j in rest if j not in (9, 10)]], 1)
groups = {
    'Head error': (np.where((body > .85) & (head < .6))[0], lambda i: head[i], [0, 1, 2]),
    'Wrist error': (np.where((body_nw > .85) & (wr < .5))[0], lambda i: wr[i], [9, 10]),
    'Occlusion (>=5 occluded joints)': (np.where(nocc >= 5)[0], lambda i: oks[i], None),
    'Low resolution (box height < 80 px)': (np.where(height < 80)[0], lambda i: oks[i], None),
}
fig, axes = plt.subplots(4, 4, figsize=(11, 13)); rng = np.random.default_rng(1)
for r, (gname, (cand, key, joints)) in enumerate(groups.items()):
    order = cand[np.argsort([key(i) for i in cand])]
    take = order[:max(4, len(order) // 3)]; pick = take[np.linspace(0, len(take) - 1, 4).astype(int)] if len(take) >= 4 else take
    for c_ in range(4):
        ax_ = axes[r, c_]; ax_.axis('off')
        if c_ >= len(pick): continue
        i = int(pick[c_]); rec = recs[(str(dom[i]), int(ids[i]))]
        img = Image.open(rec.get('image_path') or resolve_image(Path('data'), str(dom[i]), rec['file_name'])).convert('RGB')
        x_, y_, w_, h_ = box[i]; pad = .2
        l, t, rr, bb = int(x_ - pad * w_), int(y_ - pad * h_), int(x_ + (1 + pad) * w_), int(y_ + (1 + pad) * h_)
        ax_.imshow(img.crop((l, t, rr, bb)), extent=(l, rr, bb, t))
        g_, p_ = gt[i], pts[i]
        for a, b in EDGES:
            if g_[a, 2] > 0 and g_[b, 2] > 0: ax_.plot([g_[a, 0], g_[b, 0]], [g_[a, 1], g_[b, 1]], '-', color='#00dd66', lw=1.3)
            ax_.plot([p_[a, 0], p_[b, 0]], [p_[a, 1], p_[b, 1]], '-', color='#ff3b30', lw=1.1, alpha=.9)
        sc = [j for j in SC if g_[j, 2] > 0]
        ax_.scatter(g_[sc, 0], g_[sc, 1], s=10, c='#00dd66', zorder=3); ax_.scatter(p_[sc, 0], p_[sc, 1], s=8, c='#ff3b30', zorder=3)
        for j in (joints or []):
            if g_[j, 2] > 0: ax_.scatter([p_[j, 0]], [p_[j, 1]], s=120, facecolors='none', edgecolors='yellow', linewidths=1.6, zorder=4)
        ax_.set_title(f'{dom[i]}  OKS {oks[i]:.2f}\nh={int(height[i])}px  occluded={int(nocc[i])}', fontsize=8)
        if c_ == 0: ax_.text(-0.02, 1.16, gname, transform=ax_.transAxes, fontsize=11, fontweight='bold', va='bottom')
fig.text(.5, .003, 'green: ground truth (labelled joints)   red: prediction   yellow ring: joints named in the category', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .012, 1, 1)); fig.savefig(FIG / 'fig_failure_cases.pdf'); fig.savefig(FIG / 'fig_failure_cases.png', dpi=170); print('saved')
