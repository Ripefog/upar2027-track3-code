#!/usr/bin/env python3
"""Run a packaged submission against PERMITTED SOURCE validation data.

This calls the same predict_image(sample) shape as the challenge; uses
pycocotools locally for COCO OKS AP. Do not run on hidden/test images.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_src.data import stable_split, resolve_image
from pose_src.metrics import coco_ap_for_domain


def load_submission(folder):
    if not (folder/'run.py').is_file() or not (folder/'metadata.yaml').is_file():
        raise RuntimeError('run.py and metadata.yaml must be in archive root')
    sys.path.insert(0,str(folder))
    spec = importlib.util.spec_from_file_location('submission_eval_run',folder/'run.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module,'load_model'):
        module.load_model()
    if not callable(getattr(module,'predict_image',None)):
        raise ValueError('Submission is missing predict_image(sample)')
    return module


def validate_prediction(out, number):
    import numpy as np
    if isinstance(out,dict):
        people,scores=out['keypoints'],out['scores']
    else:
        people=out
        scores=[1.0]*len(people)
    if len(people)!=number or len(scores)!=number:
        raise RuntimeError('Submission must return one pose + score per box in the original order')
    cleaned=[]
    for kp,s in zip(people,scores):
        data=np.asarray(kp,dtype=float)
        if data.size != 51 or not np.isfinite(data).all() or not np.isfinite(float(s)):
            raise RuntimeError('Pose must contain 17x3 finite numbers and a finite score')
        cleaned.append((data.reshape(17,3).flatten().tolist(),float(s)))
    return cleaned


def evaluate_module(mod,args):
    domains=[args.holdout_domain] if args.holdout_domain else args.domains
    summary={}
    for domain in domains:
        annot=Path(args.annotation_dir)/f'train_{domain}.json'
        if not annot.is_file():
            raise FileNotFoundError(annot)
        data=json.loads(annot.read_text(encoding='utf-8'))
        imgs={int(img['id']):img for img in data['images']}
        ann_by_image=defaultdict(list)
        for ann in data['annotations']:
            if int(ann.get('category_id',1))!=1:
                continue
            box=ann.get('bbox',[])
            if len(box)!=4: continue
            ann_by_image[int(ann['image_id'])].append(ann)
        ids = [im_id for im_id in sorted(imgs) if
               (args.holdout_domain or stable_split(f'{domain}:{im_id}',args.val_fraction,args.seed)=='val')]
        preds=[];total=0;elapsed=0.
        for im_id in ids:
            anns=ann_by_image[im_id]
            if not anns:
                continue
            image=imgs[im_id]
            sample={
                'image_id':im_id, 'file_name':image['file_name'],
                'image_path':str(resolve_image(Path(args.image_root),domain,image['file_name'])),
                'width':int(image.get('width',0)), 'height':int(image.get('height',0)),
                'boxes':[ann['bbox'] for ann in anns],
                'box_ids':[int(ann['id']) for ann in anns]}
            t=time.perf_counter()
            output=mod.predict_image(sample)
            elapsed+=time.perf_counter()-t
            for (kpts,score),ann in zip(validate_prediction(output,len(anns)),anns):
                preds.append({'image_id':im_id,'category_id':1,'keypoints':kpts,'score':score})
            total+=1
        if not preds:
            raise ValueError(f'No selected validation boxes for {domain}')
        metric=coco_ap_for_domain(annot,preds,ids)
        metric['inference_images']=total
        metric['seconds_per_image']=elapsed/max(total,1)
        summary[domain]=metric
        print(f"{domain}: AP={metric['AP']:.4f}, AP50={metric['AP50']:.4f}, "
              f"AP_medium={metric['AP_medium']}, AP_large={metric['AP_large']}, "
              f"s/image={metric['seconds_per_image']:.3f}",flush=True)
    aps=[x['AP'] for x in summary.values() if x['AP'] is not None]
    if len(aps)!=len(summary):
        raise RuntimeError('Some domain AP values are unavailable')
    result={'domains':summary,'mean_SOURCE_domain_AP':statistics.mean(aps),
            'selection':'holdout-domain' if args.holdout_domain else 'image-level validation',
            'note':'Not the four-hidden-subset ranking metric; COCOeval uses standard keypoint sigmas.'}
    print(json.dumps(result,indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--submission',required=True, help='Packaged ZIP or extracted submission folder')
    p.add_argument('--annotation-dir',default='data/annotations/specialization')
    p.add_argument('--image-root',default='data')
    p.add_argument('--domains',nargs='+',choices=['market','pa100k','peta'],default=['market','pa100k','peta'])
    p.add_argument('--holdout-domain',choices=['market','pa100k','peta'],default=None)
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    path=Path(args.submission).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        base=path
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                for item in archive.namelist():
                    if '..' in Path(item).parts or item.startswith('/'):
                        raise ValueError('Unsafe ZIP path')
                archive.extractall(tmp)
            base=Path(tmp)
        mod=load_submission(base)
        evaluate_module(mod,args)

if __name__=='__main__':main()
