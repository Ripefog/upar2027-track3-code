#!/usr/bin/env python3
"""Paper figures that follow an image through the final model (source validation images only):
fig_stages: crop -> ViT token features (PCA) at blocks 8/16/24/32 -> heatmaps (flip+shift averaged) -> DARK keypoints
fig_joint_heatmaps: 15 scored-joint heatmaps for one instance
fig_dark_zoom: argmax vs DARK sub-cell refinement vs GT on heatmap cells
fig_flip_alignment: horizontal heatmap profiles with / without the 1-cell shift
fig_score_scatter: mean-peak score vs OKS
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.data import PoseDataset
from pose_src.geometry import (decode_dark, flip_heatmaps_back, shift_heatmap_x, to_crop, MEAN, STD, SCORED_INDEX)
from pose_src.models import build_model

FIG = Path('docs/figures'); FIG.mkdir(parents=True, exist_ok=True)
CK = 'outputs/vith_cocoonly_soup_llrd_mh/best.pth'; AD = '/home/ubuntu/workspace/upar_official/mix_ann'
NAMES = ['nose', 'head_bottom', 'head_top', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
         'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
EDGES = [(1, 2), (0, 1), (1, 5), (1, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (5, 6), (11, 13), (13, 15), (12, 14), (14, 16)]


def load():
    obj = torch.load(CK, map_location='cpu', weights_only=False)
    m = build_model(obj['config']); m.load_state_dict(obj['state_dict']); return m.cuda().eval(), obj['config']


@torch.no_grad()
def forward_all(model, x):
    feats = {}
    hooks = [model.backbone.blocks[i - 1].register_forward_hook(lambda mod, inp, out, i=i: feats.__setitem__(i, out.detach().float().cpu()))
             for i in (8, 16, 24, 32)]
    h1 = model(x).float().cpu().numpy(); [h.remove() for h in hooks]
    hf = model(x.flip(dims=[3])).float().cpu().numpy()
    back = flip_heatmaps_back(hf); shifted = shift_heatmap_x(back, 1)
    return h1, back, shifted, (h1 + shifted) * .5, feats


def pca_rgb(tokens, hp=16, wp=12):
    t = tokens[0]; t = t - t.mean(0, keepdim=True)
    u, s, v = torch.pca_lowrank(t, q=3, center=False)
    z = (t @ v[:, :3]).numpy(); z = (z - np.percentile(z, 2, 0)) / (np.percentile(z, 98, 0) - np.percentile(z, 2, 0) + 1e-9)
    return np.clip(z, 0, 1).reshape(hp, wp, 3)


def get_instance(dom, pick):
    z = np.load('outputs/paper_eval/soup.npz', allow_pickle=True)
    idx = np.where(z['dom'] == dom)[0]; o = z['flip1_dark_oks'][idx]; order = np.argsort(o)
    iid = int(z['ids'][idx[order[pick]]])
    ds = PoseDataset(AD, 'data', [dom], input_hw=(256, 192), split='val', val_fraction=.1, seed=2027, augment=False)
    k = [i for i, r in enumerate(ds.records) if int(r['image_id']) == iid][0]
    return ds[k], float(o[order[pick]])


def crop_rgb(item):
    return np.clip((item['image'].numpy() * STD + MEAN).transpose(1, 2, 0), 0, 1)


def main():
    model, cfg = load()
    plt.rcParams.update({'font.size': 8})
    # ---------- stages figure
    picks = [('market', 'median', 0.5), ('pa100k', 'median', 0.5), ('peta', 'good', 0.8)]
    fig, axes = plt.subplots(3, 7, figsize=(13.5, 6.6))
    saved = {}
    for r, (dom, tag, q) in enumerate(picks):
        z = np.load('outputs/paper_eval/soup.npz', allow_pickle=True); n = int((z['dom'] == dom).sum())
        item, oksv = get_instance(dom, int(q * (n - 1)))
        x = item['image'][None].cuda(); h1, back, shifted, avg, feats = forward_all(model, x); saved[dom] = (item, h1, back, shifted, avg)
        img = crop_rgb(item); H, W = img.shape[:2]
        pts, pk = decode_dark(avg, 7, 1.0)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f'{dom}: input crop 256x192')
        for c, L in enumerate((8, 16, 24, 32)):
            rgb = pca_rgb(feats[L]); axes[r, 1 + c].imshow(rgb, interpolation='nearest', extent=(0, W, H, 0)); axes[r, 1 + c].set_title(f'ViT block {L} (PCA)')
        mx = avg[0][SCORED_INDEX].max(0)
        axes[r, 5].imshow(img); axes[r, 5].imshow(mx, cmap='jet', alpha=.55, extent=(0, W, H, 0), vmin=0, vmax=1); axes[r, 5].set_title('heatmaps (max over joints)')
        axes[r, 6].imshow(img)
        p = pts[0]
        for a, b in EDGES: axes[r, 6].plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], '-', color='#00e0ff', lw=1.4)
        axes[r, 6].scatter(p[SCORED_INDEX, 0], p[SCORED_INDEX, 1], s=10, c='#ff3b30', zorder=3); axes[r, 6].set_title('DARK keypoints')
        for c in range(7): axes[r, c].axis('off')
        print(f'{dom}: instance score {pk[0][SCORED_INDEX].mean():.3f}, OKS vs GT {oksv:.3f}')
    fig.tight_layout(); fig.savefig(FIG / 'fig_stages.pdf'); fig.savefig(FIG / 'fig_stages.png', dpi=170); plt.close(fig)

    item, h1, back, shifted, avg = saved['market']
    img = crop_rgb(item); H, W = img.shape[:2]
    # ---------- per-joint heatmaps
    fig, axes = plt.subplots(3, 5, figsize=(10, 7.6))
    for a, j in zip(axes.ravel(), SCORED_INDEX):
        a.imshow(img); a.imshow(avg[0, j], cmap='jet', alpha=.6, extent=(0, W, H, 0), vmin=0, vmax=1)
        a.set_title(f'{NAMES[j]}  (peak {avg[0, j].max():.2f})', fontsize=8); a.axis('off')
    fig.tight_layout(); fig.savefig(FIG / 'fig_joint_heatmaps.pdf'); fig.savefig(FIG / 'fig_joint_heatmaps.png', dpi=170); plt.close(fig)

    # ---------- DARK zoom: argmax vs DARK vs GT on heatmap cells
    pts, pk = decode_dark(avg, 7, 1.0)
    gt = to_crop(item['original_points'].numpy(), item['crop_params'].numpy()) / 4.
    vis = item['original_visible'].numpy() > 0
    joints = [j for j in (9, 2, 13, 7) if vis[j]][:4]
    fig, axes = plt.subplots(1, len(joints), figsize=(3.2 * len(joints), 3.4))
    for a, j in zip(np.atleast_1d(axes), joints):
        hm = avg[0, j]; iy, ix = np.unravel_index(hm.argmax(), hm.shape); r = 5
        y0, y1, x0, x1 = max(0, iy - r), min(hm.shape[0], iy + r + 1), max(0, ix - r), min(hm.shape[1], ix + r + 1)
        a.imshow(hm[y0:y1, x0:x1], cmap='viridis', extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5), interpolation='nearest')
        a.plot(ix, iy, 's', mfc='none', mec='white', ms=11, label='argmax cell')
        a.plot(pts[0, j, 0] / 4., pts[0, j, 1] / 4., 'r+', ms=13, mew=2, label='DARK')
        a.plot(gt[j, 0], gt[j, 1], 'o', mfc='none', mec='#00ff66', ms=9, mew=1.8, label='GT')
        a.set_title(f'{NAMES[j]}', fontsize=9); a.set_xlabel('heatmap x (cells)'); a.set_ylabel('heatmap y (cells)')
    np.atleast_1d(axes)[0].legend(fontsize=7, loc='lower left', frameon=True)
    fig.tight_layout(); fig.savefig(FIG / 'fig_dark_zoom.pdf'); fig.savefig(FIG / 'fig_dark_zoom.png', dpi=170); plt.close(fig)

    # ---------- flip alignment: horizontal profile through the peak row
    joints = [j for j in (9, 13, 5) if vis[j]][:3]
    fig, axes = plt.subplots(1, len(joints), figsize=(3.6 * len(joints), 2.8))
    for a, j in zip(np.atleast_1d(axes), joints):
        iy = int(np.unravel_index(avg[0, j].argmax(), avg[0, j].shape)[0])
        xs = np.arange(h1.shape[-1])
        a.plot(xs, h1[0, j, iy], '--', lw=2.2, label='original'); a.plot(xs, back[0, j, iy], label='flipped back (no shift)')
        a.plot(xs, shifted[0, j, iy], label='flipped back + 1-cell shift'); a.axvline(gt[j, 0], color='#00aa44', ls='--', lw=1, label='GT x')
        a.set_title(f'{NAMES[j]} (row {iy})', fontsize=9); a.set_xlabel('heatmap x (cells)'); a.set_ylabel('response')
    np.atleast_1d(axes)[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / 'fig_flip_alignment.pdf'); fig.savefig(FIG / 'fig_flip_alignment.png', dpi=170); plt.close(fig)

    # ---------- score vs OKS
    z = np.load('outputs/paper_eval/soup.npz', allow_pickle=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for d, c in zip(('market', 'pa100k', 'peta'), ('#1f77b4', '#ff7f0e', '#2ca02c')):
        i = z['dom'] == d; ax.scatter(z['flip1_dark_mp'][i], z['flip1_dark_oks'][i], s=4, alpha=.35, c=c, label=d)
    from scipy.stats import spearmanr
    rho = spearmanr(z['flip1_dark_mp'], z['flip1_dark_oks']).correlation
    ax.set_xlabel('instance score (mean raw heatmap peak)'); ax.set_ylabel('OKS vs ground truth'); ax.set_title(f'Spearman rho = {rho:.3f} (n={len(z["ids"])})', fontsize=9)
    ax.legend(markerscale=3, fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(FIG / 'fig_score_scatter.pdf'); fig.savefig(FIG / 'fig_score_scatter.png', dpi=200)
    print('saved')


if __name__ == '__main__':
    main()
