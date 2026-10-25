#!/usr/bin/env python3
"""Build official Codabench ZIP with run.py at archive ROOT."""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import numpy  # noqa: F401  mkl-service: numpy must load before torch/libgomp

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoints',nargs='+',required=True,help='One or more trained best.pth files')
    ap.add_argument('--output',default='dist/upar_track3_submission.zip')
    ap.add_argument('--flip-tta',action='store_true')
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--flip-shift',type=int,default=1,help='cells to shift the flipped heatmap (validated: 1)')
    ap.add_argument('--decoder',choices=['argmax','dark'],default='dark')
    ap.add_argument('--dark-kernel',type=int,default=7)
    ap.add_argument('--dark-sigma',type=float,default=1.0)
    ap.add_argument('--score-mode',choices=['sigmoid','mean_peak'],default='sigmoid',help='instance score; mean_peak chosen on source LODO/val (scripts/score_study.py)')
    ap.add_argument('--padding',type=float,default=1.25)
    ap.add_argument('--confidence-bias',type=float,default=0.25)
    ap.add_argument('--confidence-temperature',type=float,default=0.15)
    args=ap.parse_args()
    if not 1 <= len(args.checkpoints) <= 3:
        ap.error('Use 1-3 checkpoints')
    if args.batch_size <= 0:
        ap.error('batch-size must be positive')
    if args.confidence_temperature <= 0:
        ap.error('confidence-temperature must be positive')
    import torch
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)
        shutil.copy2(ROOT/'submission'/'run.py',stage/'run.py')
        shutil.copy2(ROOT/'submission'/'metadata.yaml',stage/'metadata.yaml')
        shutil.copytree(ROOT/'pose_src',stage/'pose_src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (stage/'weights').mkdir()
        entries=[]
        for index, filename in enumerate(args.checkpoints):
            ckpt=torch.load(filename,map_location='cpu',weights_only=False)
            if 'state_dict' not in ckpt or 'config' not in ckpt:
                raise RuntimeError(f'Invalid trained checkpoint: {filename}')
            cfg=ckpt['config']
            weight_file=stage/'weights'/f'model_{index}.pth'
            # Include ONLY inference weights/config, strip optimizer/history.
            torch.save({'state_dict':ckpt['state_dict']},weight_file)
            entries.append({'weights':f'weights/model_{index}.pth','config':cfg,
                            'ensemble_weight':1.,'flip_tta':args.flip_tta,
                            'flip_shift':args.flip_shift,'decoder':args.decoder,
                            'dark_kernel':args.dark_kernel,'dark_sigma':args.dark_sigma,
                            'batch_size':args.batch_size,'padding':args.padding,
                            'score_mode':args.score_mode,
                            'confidence_bias':args.confidence_bias,
                            'confidence_temperature':args.confidence_temperature})
        (stage/'models.json').write_text(json.dumps({'models':entries},indent=2))
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in stage.rglob('*'):
                if path.is_file(): z.write(path,path.relative_to(stage).as_posix())
    with zipfile.ZipFile(out) as z:
        names=z.namelist()
        assert 'run.py' in names and 'metadata.yaml' in names and 'models.json' in names
        assert any(x.startswith('weights/') for x in names)
    print(f'Created: {out} ({out.stat().st_size/2**20:.1f} MiB)')

if __name__=='__main__':main()
