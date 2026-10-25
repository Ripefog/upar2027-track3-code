"""Fail-closed checkpoint loader: provenance and coverage explicitly required."""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn.functional as F


def _extract(obj):
    if not isinstance(obj, dict):
        raise ValueError('Checkpoint must be a dictionary containing weights')
    for key in ('state_dict', 'model', 'model_state_dict'):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def _normalize_name(name):
    for prefix in ('module.', 'model.', 'pose_estimator.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # ViTPose old official checkpoints share these names.
    name = name.replace('backbone.patch_embed.projection.', 'backbone.patch_embed.proj.')
    name = name.replace('backbone.patch_embed.patch_embeddings.projection.', 'backbone.patch_embed.proj.')
    name = name.replace('backbone.norm.', 'backbone.last_norm.')
    name = name.replace('backbone.layers.', 'backbone.blocks.')
    name = name.replace('.ln1.', '.norm1.').replace('.ln2.', '.norm2.')
    name = name.replace('.attention.attention.', '.attn.')
    name = name.replace('.attention.output.dense.', '.attn.proj.')
    name = name.replace('.intermediate.dense.', '.mlp.fc1.').replace('.output.dense.', '.mlp.fc2.')
    return name


def _resize_pos_embed(source, target):
    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
        return None
    old_n, new_n, c = source.shape[1], target.shape[1], target.shape[2]
    if old_n == new_n:
        return source
    # Both default input sizes have aspect ratio 4:3.
    oh, nh = int(round(math.sqrt(old_n*4/3))), int(round(math.sqrt(new_n*4/3)))
    ow, nw = old_n // oh, new_n // nh
    if oh*ow != old_n or nh*nw != new_n:
        return None
    z = source.transpose(1, 2).reshape(1, c, oh, ow)
    return F.interpolate(z, (nh, nw), mode='bicubic', align_corners=False).flatten(2).transpose(1, 2)


def load_pretrained(model, path, source_topology, allow_low_coverage=False):
    """Load compatible upstream tensors, protect PoseTrack head from COCO eyes.

    The caller is responsible for verifying the legal training provenance.
    Coverage is parameter-element-based; default requires >=90% of backbone.
    """
    if source_topology not in ('coco', 'posetrack18'):
        raise ValueError('Explicit --pretrained-topology must be coco or posetrack18')
    state = _extract(torch.load(path, map_location='cpu', weights_only=False))
    target = model.state_dict()
    incoming = {}
    is_hrnet = type(model).__name__ == 'HRNetW48'
    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        key = _normalize_name(key)
        if is_hrnet:
            # MMPose HRNet uses normN for BatchNorm; our standalone model uses bnN.
            for n in (1,2,3):
                key = key.replace(f'.norm{n}.', f'.bn{n}.')
            if key.startswith('head.final_layer.'):
                key = key.replace('head.final_layer.', 'head.')
        # Accept top-level backbone names from standalone pretrained backbones.
        if key not in target and ('backbone.' + key) in target:
            key = 'backbone.' + key
        if key not in target:
            continue
        dst = target[key]
        if key.endswith('pos_embed') and tensor.shape != dst.shape:
            tensor = _resize_pos_embed(tensor, dst)
            if tensor is None:
                continue
        if dst.shape != tensor.shape:
            continue
        if source_topology == 'coco' and key.startswith('head.') and dst.ndim >= 1 and dst.shape[0] == 17:
            tensor = tensor.clone()
            # PoseTrack indices 1 and 2 are head_bottom/head_top, NOT COCO eyes.
            tensor[1:3] = dst[1:3]
            tensor[3:5] = dst[3:5]   # ears are not supervised
        incoming[key] = tensor
    # Everything outside the head is backbone (HRNet has no 'backbone.' prefix).
    backbone = [(k, p) for k, p in target.items() if not k.startswith('head.') and p.is_floating_point()]
    covered = sum(p.numel() for k, p in backbone if k in incoming)
    total = sum(p.numel() for _, p in backbone)
    ratio = covered / max(1, total)
    if ratio < .9 and not allow_low_coverage:
        raise RuntimeError(f'Pretrained backbone coverage {ratio:.1%} below 90%; '
                           f'only {len(incoming)} tensors matched. Correct adapter or checkpoint required.')
    model.load_state_dict(incoming, strict=False)
    return {'backbone_coverage': ratio, 'loaded_tensors': len(incoming),
            'source_topology': source_topology, 'checkpoint': str(path)}


def load_trained(model, path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' not in obj or 'config' not in obj:
        raise ValueError('Expected native training checkpoint with state_dict and config')
    model.load_state_dict(obj['state_dict'], strict=True)
    return obj
