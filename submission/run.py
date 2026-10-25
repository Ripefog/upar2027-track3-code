"""UPAR Track 3 Codabench contract: load_model() and predict_image(sample).

Everything this module imports is in the zip. No network or MMCV needed.
Models are loaded once, not on each image.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from pose_src.geometry import (crop_params, crop_image, as_tensor, from_crop, decode_numpy, decode_dark, flip_heatmaps_back,
                               shift_heatmap_x, SCORED_INDEX, calibrated_joint_confidence)
from pose_src.models import build_model

BASE = Path(__file__).resolve().parent
_MODELS = []
_DEVICE = None


def load_model():
    global _MODELS, _DEVICE
    if _MODELS:
        return
    manifest_path = BASE / 'models.json'
    if not manifest_path.exists():
        raise FileNotFoundError('models.json missing. Run scripts/pack_submission.py after training.')
    info = json.loads(manifest_path.read_text())
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for entry in info['models']:
        cfg = entry['config']
        model = build_model(cfg)
        weight_path = BASE / entry['weights']
        if not weight_path.is_file():
            raise FileNotFoundError(f'Missing model weights {weight_path}')
        ckpt = torch.load(weight_path,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['state_dict'],strict=True)
        model.to(_DEVICE).eval()
        _MODELS.append({'model':model,'cfg':cfg,'ensemble_weight':float(entry.get('ensemble_weight',1.)),
                        'flip_tta':bool(entry.get('flip_tta',False)),
                        # Defaults keep the behaviour of older ZIPs (no shift, argmax decoding).
                        'flip_shift':int(entry.get('flip_shift',0)),
                        'decoder':str(entry.get('decoder','argmax')),
                        'dark_kernel':int(entry.get('dark_kernel',7)),
                        'dark_sigma':float(entry.get('dark_sigma',1.0)),
                        'pad':float(entry.get('padding',1.25)),
                        'batch_size':int(entry.get('batch_size',8)),
                        'score_mode':str(entry.get('score_mode','sigmoid')),
                        'confidence_bias':float(entry.get('confidence_bias',0.25)),
                        'confidence_temperature':float(entry.get('confidence_temperature',0.15))})
    if not _MODELS:
        raise ValueError('models.json has no models')


def _image_path(sample):
    p = Path(str(sample.get('image_path','')))
    if str(p) not in ('', '.') and p.is_file():
        return p
    if str(p) not in ('', '.'):
        s=str(p)
        for name in (s.replace('/images/images/','/images/'),s.replace('\\images\\images\\','\\images\\')):
            if Path(name).is_file(): return Path(name)
    name=str(sample.get('file_name',''))
    for cand in [BASE/name,Path(name),BASE/name.removeprefix('images/')]:
        if cand.is_file(): return cand
    raise FileNotFoundError(f'Could not locate image: {sample.get("image_path")} / {name}')


def _fallback(box):
    x,y,w,h=[float(z) for z in box[:4]]
    return [[x+w*.5,y+h*.5,0.] for _ in range(17)]


@torch.inference_mode()
def predict_image(sample):
    if not _MODELS:
        load_model()
    boxes=sample['boxes']
    if not boxes:
        return {'keypoints':[],'scores':[]}
    image=Image.open(_image_path(sample)).convert('RGB')
    n=len(boxes)
    coord_sum=np.zeros((n,17,2),np.float64)
    weight_sum=np.zeros((n,17),np.float64)
    result=[None]*n; score=[0.]*n
    score_sum=np.zeros((n,),np.float64)
    model_weight_total=sum(e["ensemble_weight"] for e in _MODELS)
    usable=[i for i,b in enumerate(boxes) if float(b[2])>0 and float(b[3])>0]
    for i in set(range(n))-set(usable):
        result[i]=_fallback(boxes[i]); score[i]=0.01
    for entry in _MODELS:
        model=entry['model']
        hw=tuple(entry['cfg']['data']['input_size'])
        for s in range(0,len(usable),entry['batch_size']):
            chunk=usable[s:s+entry['batch_size']]
            params=[crop_params(boxes[i],hw,pad=entry['pad']) for i in chunk]
            crops=[as_tensor(crop_image(image,p,hw)) for p in params]
            x=torch.from_numpy(np.stack(crops).copy()).to(_DEVICE)
            hmap=model(x).float()
            if entry['flip_tta']:
                hmap_flip=model(x.flip(dims=[3])).float().cpu().numpy()
                hmap=(hmap.cpu().numpy()+shift_heatmap_x(flip_heatmaps_back(hmap_flip),entry['flip_shift']))*.5
            else:
                hmap=hmap.cpu().numpy()
            if entry['decoder']=='dark':
                coords, confidences=decode_dark(hmap,entry['dark_kernel'],entry['dark_sigma'])
            else:
                coords, confidences=decode_numpy(hmap)
            for row,idx in enumerate(chunk):
                orig=from_crop(coords[row],params[row])
                # Per-joint normalized scores; positive weighting for weak joints.
                joint_conf=calibrated_joint_confidence(
                    confidences[row],entry['confidence_bias'],entry['confidence_temperature'])
                conf=np.maximum(joint_conf,.001) * entry['ensemble_weight']
                if entry['score_mode']=='mean_peak':   # raw mean heatmap peak over the 15 scored joints (no sigmoid)
                    inst=float(np.mean(confidences[row][SCORED_INDEX]))
                else:
                    inst=float(np.mean(joint_conf[SCORED_INDEX]))
                score_sum[idx] += inst * entry['ensemble_weight']
                conf[3:5]=0. # ears ignored, not used in scoring
                coord_sum[idx] += orig*conf[:,None]
                weight_sum[idx] += conf
    for i in usable:
        pts=coord_sum[i]/np.maximum(weight_sum[i,:,None],1e-9)
        # Placeholder ear coordinates: unscored joints, maintained in format.
        for ear in (3,4):
            pts[ear]=pts[0]
        result[i]=[[float(p[0]),float(p[1]),1.0] for p in pts]
        score[i]=float(np.clip(score_sum[i]/max(model_weight_total,1e-9),0.,1.))
    return {'keypoints':result,'scores':score}
