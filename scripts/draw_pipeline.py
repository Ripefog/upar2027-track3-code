#!/usr/bin/env python3
"""Draws the pipeline figure (training -> weight averaging -> inference) with matplotlib. Output: docs/figures/pipeline.{pdf,png,svg}"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 150); ax.set_ylim(0, 82); ax.axis('off')
C = {'data': '#dbe9f6', 'proc': '#e8e8e8', 'model': '#fde9c9', 'loss': '#f9d6d5', 'avg': '#d9ead3', 'out': '#e4d9f3', 'edge': '#4a4a4a'}

def box(x, y, w, h, text, fc, fs=9.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=C['edge'], lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)

def arrow(x1, y1, x2, y2, style='-|>', rad=0., ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, lw=1.3, color=C['edge'],
                                 connectionstyle=f'arc3,rad={rad}', linestyle=ls))

# ---- Training row
ax.text(1, 79, '(a) Training', fontsize=12, fontweight='bold')
box(1, 55, 22, 20, 'UPAR train\nmarket / pa100k / peta\n+ PoseTrack18\n+ COCO train2017\n(299,667 instances)', C['data'])
box(27, 58, 18, 14, 'Domain-balanced\nsampler\n+ crop (pad 1.25)\n+ augmentation', C['proc'])
box(49, 51, 38, 26, '', C['model'])
ax.text(68, 74.5, 'ViTPose-H  (256×192 input)', ha='center', fontsize=10, fontweight='bold')
box(51, 58, 10, 12, 'Patch\nembed\n16×16', '#ffffff', 8.8)
box(63, 58, 11, 12, '32 ViT\nblocks\n(dim 1280)', '#ffffff', 8.8)
box(76, 58, 10, 12, 'Deconv\nhead\n17×64×48', '#ffffff', 8.8)
ax.text(68, 54.5, 'init: official COCO-only checkpoint\n(head rows of slots 1–4 re-initialised)', ha='center', fontsize=8.4, style='italic')
box(91, 58, 22, 14, 'Heatmap MSE loss\nmask slots 1,2 (head)\nfor PoseTrack samples', C['loss'])
arrow(23, 65, 27, 65); arrow(45, 65, 51, 65); arrow(61, 64, 63, 64); arrow(74, 64, 76, 64); arrow(86, 65, 91, 65)
box(117, 57, 31, 16, 'Recipe A: base LR\nRecipe B: LLRD 0.9 +\ndrop-path 0.2 (LR 1e-4)\n2 epochs, EMA 0.999', C['proc'], 8.8)
arrow(113, 65, 117, 65, ls='--')

# ---- Weight averaging
box(26, 34, 46, 10, 'Model soup: uniform weight average\nof checkpoints A and B (same seed / same init)', C['avg'], 9.0, True)
arrow(56, 51, 52, 44.3); ax.text(42, 47, 'checkpoint A', fontsize=8.4, style='italic', ha='center')
arrow(128, 56.7, 70, 40, rad=0.22); ax.text(103, 45.5, 'checkpoint B', fontsize=8.4, style='italic')
arrow(49, 33.7, 49, 22.5)

# ---- Inference row
ax.text(1, 29, '(b) Inference', fontsize=12, fontweight='bold'); ax.text(1, 25.6, 'single model, 2 passes per box', fontsize=8.6, style='italic')
box(1, 8, 16, 14, 'Image +\nprovided\nperson box', C['data'])
box(21, 8, 15, 14, 'Crop 256×192\n(pad 1.25)', C['proc'])
box(40, 11, 18, 11, 'Soup model\nimage', C['model'], 9.0)
box(40, 4, 18, 5.5, 'Soup model\nflipped image', C['model'], 8.0)
box(62, 6, 20, 14, 'Flip back,\nswap L/R joints,\nshift 1 cell,\naverage heatmaps', C['proc'], 8.8)
box(86, 6, 18, 14, 'DARK decode\n(kernel 7, σ 1.0)\nmap to image\ncoordinates', C['proc'], 8.8)
box(108, 11, 20, 9.5, 'Keypoints\n(17 slots)', C['out'], 9.3, True)
box(108, 2.5, 20, 7, 'Score = mean raw\nheatmap peak\n(15 scored joints)', C['out'], 8.6, True)
box(132, 4, 16, 14, 'Prediction\nper box:\nkeypoints +\nscore', C['out'], 8.8, True)
arrow(17, 15, 21, 15); arrow(36, 15, 40, 16); arrow(36, 13, 40, 7, rad=-0.15)
arrow(58, 16.5, 62, 15); arrow(58, 7, 62, 10)
arrow(82, 13, 86, 13); arrow(104, 16, 108, 16); arrow(104, 9, 108, 6)
arrow(128, 15, 132, 13); arrow(128, 6, 132, 8)
fig.savefig('docs/figures/pipeline.pdf', bbox_inches='tight'); fig.savefig('docs/figures/pipeline.png', dpi=200, bbox_inches='tight')
fig.savefig('docs/figures/pipeline.svg', bbox_inches='tight')
print('saved')
