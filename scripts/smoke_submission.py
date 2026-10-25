#!/usr/bin/env python3
"""Quick offline smoke test of unpacked or zipped Codabench submission."""
import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy  # noqa: F401  mkl-service: numpy must load before torch/libgomp
from PIL import Image


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--submission',required=True,help='ZIP or folder containing root-level run.py')
    ap.add_argument('--image',default=None)
    args=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(args.submission)
        if root.is_file():
            with zipfile.ZipFile(root) as z:z.extractall(tmp)
            root=Path(tmp)
        assert (root/'run.py').is_file() and (root/'metadata.yaml').is_file()
        sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location('run',root/'run.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        image=Path(args.image) if args.image else (root/'smoke.png')
        if not args.image: Image.new('RGB',(100,150),(90,80,70)).save(image)
        output=mod.predict_image({'image_path':str(image),'boxes':[[10,10,60,120],[0,0,0,0]]})
        assert set(output)=={'keypoints','scores'}
        assert len(output['keypoints'])==len(output['scores'])==2
        assert all(len(person)==17 and all(len(j)==3 for j in person) for person in output['keypoints'])
        assert all(isinstance(s,(int,float)) and 0 <= s <= 1 for s in output['scores'])
        print(json.dumps({'valid':True,'predicted_people':2,'scores':output['scores']},indent=2))

if __name__=='__main__':main()
