#!/usr/bin/env python3
"""Audit upstream model key coverage without silently accepting incompatible checkpoints."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from pose_src.models import build_model
from pose_src.checkpoints import load_pretrained


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--topology',choices=['coco','posetrack18'],required=True)
    p.add_argument('--show-keys',type=int,default=10)
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    model=build_model(cfg)
    data=torch.load(a.weights,map_location='cpu',weights_only=False)
    state=data.get('state_dict',data.get('model',data))
    print('First source keys:')
    for k in list(state)[:a.show_keys]:print(' ',k)
    print('First expected keys:')
    for k in list(model.state_dict())[:a.show_keys]:print(' ',k)
    result=load_pretrained(model,a.weights,a.topology)
    print(result)

if __name__=='__main__':main()
