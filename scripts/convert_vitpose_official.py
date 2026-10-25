#!/usr/bin/env python3
"""Convert an official ViTPose (mmpose-style, COCO single-dataset) checkpoint to the repo-native ViTPose state dict.
Loaded with weights_only=True. Usage: convert_vitpose_official.py <in.pth> <out.pth>"""
import sys, torch

def convert(sd):
    out = {}
    for k, v in sd.items():
        if k == 'backbone.pos_embed':
            v = v[:, 1:] + v[:, :1]          # cls slot + patch slots, as in the official forward
        out[k.replace('keypoint_head.', 'head.')] = v
    return out

if __name__ == '__main__':
    sd = torch.load(sys.argv[1], map_location='cpu', weights_only=True)['state_dict']
    torch.save({'state_dict': convert(sd)}, sys.argv[2]); print('saved', sys.argv[2])
