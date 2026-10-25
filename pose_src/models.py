"""Standalone PyTorch ViTPose-style and HRNet models; no timm/MMCV at inference.

All models return raw (B,17,H/4,W/4) heatmaps. Model architectures are
self-contained implementations, not byte-for-byte copies of upstream repos.
Only use pretrained parameters whose entire training provenance is permitted.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, drop_path: float = 0.):
        super().__init__()
        self.drop_path = float(drop_path)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def _dp(self, y):
        # Stochastic depth (training only); identity at inference so exported weights/ZIP are unaffected.
        if self.drop_path > 0. and self.training:
            keep = 1. - self.drop_path
            y = y * (torch.rand(y.shape[0], 1, 1, device=y.device) < keep).to(y.dtype) / keep
        return y

    def forward(self, x):
        x = x + self._dp(self.attn(self.norm1(x)))
        return x + self._dp(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size, padding=2)  # ViTPose official uses padding 2

    def forward(self, x):
        return self.proj(x)


class VitBackbone(nn.Module):
    def __init__(self, dim=1024, depth=24, heads=16, input_size=(256, 192), drop_path=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        hp, wp = input_size[0] // 16, input_size[1] // 16
        self.pos_embed = nn.Parameter(torch.zeros(1, hp * wp, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path * i / max(1, depth - 1)) for i in range(depth)])
        self.last_norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        feat = self.patch_embed(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        if self.pos_embed.shape[1] != h * w:
            old_n = self.pos_embed.shape[1]
            # The base input is always rectangular 4:3 (H/W); fail rather than
            # silently interpolate an ambiguous pretrained positional embedding.
            old_h = int(round(math.sqrt(old_n * 4 / 3)))
            old_w = old_n // old_h
            if old_h * old_w != old_n:
                raise ValueError('Incompatible positional embedding shape')
            pos = F.interpolate(self.pos_embed.transpose(1, 2).reshape(1, c, old_h, old_w),
                                (h, w), mode='bicubic', align_corners=False).flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos
        for layer in self.blocks:
            tokens = layer(tokens)
        tokens = self.last_norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class HeatmapHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 256):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, mid_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, mid_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(mid_ch, 17, 1)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class HeatmapHeadBilinear(nn.Module):
    """Experimental decoder (not RefinePose's Fusion Decoder): bilinear x2 upsample + 3x3 conv instead of transposed conv."""
    def __init__(self, in_ch: int, mid_ch: int = 256):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(mid_ch, 17, 1)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class ViTPose(nn.Module):
    def __init__(self, variant='l', input_size=(256, 192), drop_path=0., head='deconv', **kw):
        super().__init__()
        presets = {'b': (768, 12, 12), 'l': (1024, 24, 16), 'h': (1280, 32, 16), 'tiny': (96, 2, 4)}
        if variant not in presets:
            raise ValueError(f'Invalid ViTPose variant: {variant}')
        dim, depth, heads = presets[variant]
        self.backbone = VitBackbone(dim, depth, heads, input_size, drop_path)
        head_cls = {'deconv': HeatmapHead, 'bilinear': HeatmapHeadBilinear}[head]
        self.head = head_cls(dim, 64 if variant == 'tiny' else 256)

    def forward(self, x):
        return self.head(self.backbone(x))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, c, out):
        super().__init__()
        self.conv1 = nn.Conv2d(c, out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = (nn.Sequential(nn.Conv2d(c, out, 1, bias=False), nn.BatchNorm2d(out))
                           if c != out else nn.Identity())
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(y)) + self.downsample(x), inplace=True)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c, planes):
        super().__init__()
        out = 4 * planes
        self.conv1 = nn.Conv2d(c, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.downsample = (nn.Sequential(nn.Conv2d(c, out, 1, bias=False), nn.BatchNorm2d(out))
                           if c != out else nn.Identity())
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        return F.relu(self.bn3(self.conv3(y)) + self.downsample(x), inplace=True)


class HRModule(nn.Module):
    """Parallel HRNet stages with explicit cross-scale fusion."""
    def __init__(self, channels, blocks_per_branch=4):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(c, c) for _ in range(blocks_per_branch)]) for c in channels
        ])
        self.fuse_layers = nn.ModuleList()
        for i, c_out in enumerate(channels):
            row = nn.ModuleList()
            for j, c_in in enumerate(channels):
                if j == i:
                    row.append(nn.Identity())
                elif j > i:
                    row.append(nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out)))
                else:
                    layers = []
                    for k in range(i - j):
                        last = (k == i - j - 1)
                        nxt = c_out if last else c_in
                        layers += [nn.Conv2d(c_in if k == 0 else c_in, nxt, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(nxt)]
                        if not last:
                            layers.append(nn.ReLU(inplace=True))
                    row.append(nn.Sequential(*layers))
            self.fuse_layers.append(row)

    def forward(self, x):
        x = [branch(v) for branch, v in zip(self.branches, x)]
        fused = []
        for i, row in enumerate(self.fuse_layers):
            target = x[i].shape[-2:]
            ys = []
            for j, op in enumerate(row):
                z = op(x[j])
                if j > i:
                    z = F.interpolate(z, size=target, mode='nearest')
                ys.append(z)
            fused.append(F.relu(sum(ys), inplace=False))
        return fused


class HRNetW48(nn.Module):
    """Four-resolution HRNet-W48 style implementation with standard 4/4/3 modules.

    Names/layout are not guaranteed to exactly match any external MMPose
    checkpoint; loading MUST pass coverage validation before training.
    """
    def __init__(self, channels=48, **kw):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(Bottleneck(64, 64), *[Bottleneck(256, 64) for _ in range(3)])
        c = channels
        self.transition1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(256, 2*c, 3, 2, 1, bias=False), nn.BatchNorm2d(2*c), nn.ReLU(True)),
        ])
        self.stage2 = nn.Sequential(HRModule([c,2*c]))
        self.transition2 = nn.Sequential(nn.Conv2d(2*c, 4*c, 3, 2, 1, bias=False), nn.BatchNorm2d(4*c), nn.ReLU(True))
        self.stage3 = nn.Sequential(*[HRModule([c,2*c,4*c]) for _ in range(4)])
        self.transition3 = nn.Sequential(nn.Conv2d(4*c, 8*c, 3, 2, 1, bias=False), nn.BatchNorm2d(8*c), nn.ReLU(True))
        self.stage4 = nn.Sequential(*[HRModule([c,2*c,4*c,8*c]) for _ in range(3)])
        self.head = nn.Conv2d(c, 17, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.layer1(x)
        x = [op(x) for op in self.transition1]
        x = self.stage2(x)
        x = self.stage3(x + [self.transition2(x[-1])])
        x = self.stage4(x + [self.transition3(x[-1])])
        return self.head(x[0])


def build_model(cfg):
    name = cfg['model']['name']
    if name == 'vitpose':
        return ViTPose(variant=cfg['model'].get('variant', 'l'),
                       input_size=tuple(cfg['data']['input_size']),
                       drop_path=float(cfg['model'].get('drop_path', 0.)),
                       head=cfg['model'].get('head', 'deconv'))
    if name in ('hrnet_w48', 'hrnet_w32'):
        return HRNetW48(channels=int(cfg['model'].get('channels', 48 if name == 'hrnet_w48' else 32)))
    raise ValueError(f'Unknown model: {name}')
