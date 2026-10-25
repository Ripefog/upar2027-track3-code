#!/usr/bin/env python3
"""Report domain counts, joints, visibility and image path consistency."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_src.data import read_records, resolve_image
from pose_src.geometry import JOINTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--annotations',default='data/annotations/specialization')
    p.add_argument('--image-root',default='data')
    p.add_argument('--verify-paths',action='store_true')
    args=p.parse_args()
    records=read_records(args.annotations,['market','pa100k','peta'])
    by_domain=Counter(r['domain'] for r in records)
    by_images=Counter((r['domain'],r['image_id']) for r in records)
    joints=np.sum([r['visible'] for r in records],axis=0)
    print('Person instances:',len(records))
    print('Unique domain/image pairs:',len(by_images))
    print('Instances by domain:',dict(by_domain))
    print('Visible keypoints:',dict(zip(JOINTS,joints.astype(int).tolist())))
    if args.verify_paths:
        missing=[]
        for domain,img in sorted(by_images):
            r=next(r for r in records if r['domain']==domain and r['image_id']==img)
            try:resolve_image(Path(args.image_root),domain,r['file_name'])
            except FileNotFoundError as e:missing.append(str(e))
        print('Missing unique images:',len(missing))
        for text in missing[:20]:print(text)
        if missing:raise SystemExit(1)

if __name__=='__main__':main()
