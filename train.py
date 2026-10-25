#!/usr/bin/env python3
"""Single-GPU 96 GB Blackwell training. Local PyTorch >=2.7 with cu128 recommended."""
from __future__ import annotations
import argparse
import copy
import json
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pose_src.checkpoints import load_pretrained
from pose_src.data import PoseDataset, balanced_sampler
from pose_src.geometry import from_crop, decode_numpy, SCORED_INDEX, person_confidence
from pose_src.metrics import coco_ap_for_domain
from pose_src.models import build_model


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def heatmap_loss(pred, targets, weights):
    if pred.shape != targets.shape:
        raise RuntimeError(f'Heatmap size mismatch: predicted {pred.shape}, GT {targets.shape}')
    diff = ((pred.float() - targets.float()) ** 2).mean(dim=(-1,-2))
    return (diff * weights).sum() / weights.sum().clamp(min=1)


def _layer_scale(name, depth, decay):
    """Layer-wise LR decay for ViT backbones: embeddings get decay^(depth+1), the last block decay^1, head 1."""
    if name.startswith('head.'):
        return 1.
    if name.startswith('backbone.blocks.'):
        layer = int(name.split('.')[2]) + 1
    elif name.startswith('backbone.last_norm'):
        layer = depth + 1
    else:
        layer = 0
    return decay ** (depth + 1 - layer)


def create_optimizer(model, cfg):
    decay_rate = cfg['training'].get('layer_decay')
    if decay_rate and hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
        depth = len(model.backbone.blocks)
        buckets = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            head = n.startswith('head.')
            lr = cfg['training']['head_lr'] if head else cfg['training']['backbone_lr'] * _layer_scale(n, depth, decay_rate)
            wd = 0. if (p.ndim <= 1 or n.endswith('pos_embed')) else cfg['training']['weight_decay']
            buckets.setdefault((round(lr, 12), wd), []).append(p)
        return torch.optim.AdamW([{'params': ps, 'lr': lr, 'initial_lr': lr, 'weight_decay': wd}
                                  for (lr, wd), ps in buckets.items()])
    groups = []
    # Every non-head parameter is "backbone" (HRNet has no 'backbone.' prefix).
    for group_name, lr in [('backbone', cfg['training']['backbone_lr']),
                           ('head', cfg['training']['head_lr'])]:
        for decay in (0., cfg['training']['weight_decay']):
            selected = [(n,p) for n,p in model.named_parameters()
                        if (n.startswith('head.') == (group_name == 'head')) and p.requires_grad
                        and ((p.ndim <= 1 or n.endswith('pos_embed')) == (decay == 0.))]
            if selected:
                groups.append({'params': [p for _,p in selected], 'lr': lr,
                               'initial_lr': lr, 'weight_decay': decay})
    if not groups:
        raise RuntimeError('No trainable parameters')
    return torch.optim.AdamW(groups)


@torch.no_grad()
def evaluate(model, loader, device, selection='oks_ap', confidence_bias=.25,
             confidence_temperature=.15):
    """Source-domain validation: OKS AP (preferred) and diagnostic PCK.

    OKS AP requires pycocotools installed IN TRAINING ENVIRONMENT ONLY.
    Source validation cannot replicate scores on the four hidden domains.
    """
    model.eval()
    sums = defaultdict(lambda: [0,0,0,0])
    preds_by_domain = defaultdict(list)
    imgids_by_domain = defaultdict(set)
    for batch in loader:
        pred = model(batch['image'].to(device)).float().cpu().numpy()
        crop_pts, peaks = decode_numpy(pred)
        original = batch['original_points'].numpy()
        visible = batch['original_visible'].numpy()
        cp = batch['crop_params'].numpy()
        boxes = batch['box'].numpy()
        sample_scores = person_confidence(peaks, confidence_bias, confidence_temperature)
        for i in range(len(crop_pts)):
            pts = from_crop(crop_pts[i], cp[i])
            domain = batch['domain'][i]
            image_id = int(batch['image_id'][i])
            imgids_by_domain[domain].add(image_id)
            kpts = [[float(x),float(y),1.0] for x,y in pts]
            # Ear predictions remain present for compatibility; GT ignores them.
            preds_by_domain[domain].append({
                'image_id': image_id, 'category_id': 1,
                'keypoints': [x for kp in kpts for x in kp],
                'score': float(sample_scores[i])})
            dist = np.linalg.norm(pts - original[i], axis=-1)
            denom = max(1., boxes[i][2], boxes[i][3])
            for k in SCORED_INDEX:
                if not visible[i, k]: continue
                accum = sums[domain]
                accum[0] += int(dist[k] <= .1 * denom)
                accum[1] += int(dist[k] <= .2 * denom)
                accum[2] += 1
                accum[3] += dist[k] / denom
    per_domain = {d: {'pck_010': v[0] / max(v[2], 1),
                      'pck_020': v[1] / max(v[2], 1),
                      'mean_normalized_error': float(v[3] / max(v[2], 1)),
                      'valid_joints': v[2]}
                  for d,v in sums.items()}
    global_v = [sum(v[j] for v in sums.values()) for j in range(4)]
    overall = {'pck_010': global_v[0] / max(global_v[2],1),
               'pck_020': global_v[1] / max(global_v[2],1),
               'mean_normalized_error': float(global_v[3] / max(global_v[2],1)),
               'valid_joints': global_v[2]}
    output = {'overall': overall, 'domains': per_domain, 'selection': selection}
    if selection == 'oks_ap':
        domain_ap = {}
        annotation_dir = loader.dataset.annotation_dir
        for domain in sorted(preds_by_domain):
            domain_ap[domain] = coco_ap_for_domain(
                annotation_dir / f'train_{domain}.json',
                preds_by_domain[domain], imgids_by_domain[domain])
        valid_aps = [v['AP'] for v in domain_ap.values() if v['AP'] is not None]
        if len(valid_aps) != len(domain_ap) or not valid_aps:
            raise RuntimeError('Missing/invalid source-domain AP; inspect GT and pycocotools setup')
        output['oks_ap'] = {'domains':domain_ap,
                            'mean_source_domain_ap':float(np.mean(valid_aps)),
                            'min_source_domain_ap':float(np.min(valid_aps))}
    else:
        output['warning'] = 'PCK proxy selected: not the official ranking AP'
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--pretrained', default=None, help='Permitted upstream checkpoint')
    ap.add_argument('--pretrained-topology', choices=['coco', 'posetrack18'], default=None)
    ap.add_argument('--provenance', default=None, help='Local textual statement of allowed pretraining')
    ap.add_argument('--scratch', action='store_true', help='Explicitly choose random weights')
    ap.add_argument('--domains', nargs='+', default=['market','pa100k','peta'],
                    help='Annotation domains (train_<domain>.json); coco is used for COCO pretraining')
    ap.add_argument('--holdout-domain', default=None)
    ap.add_argument('--train-all', action='store_true', help='Final training on every permitted source record; no validation/checkpoint selection')
    ap.add_argument('--mask-head-domains', nargs='+', default=[],
                    help='Domains whose head_bottom/head_top (slots 1,2) labels are excluded from the loss')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--output', default='outputs/experiment')
    ap.add_argument('--selection', choices=['oks_ap','pck_proxy'], default='oks_ap',
                    help='Use oks_ap for real experiments; pck_proxy is for smoke testing only')
    ap.add_argument('--confidence-bias', type=float, default=.25)
    ap.add_argument('--confidence-temperature', type=float, default=.15)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if bool(args.pretrained) == bool(args.scratch):
        ap.error('Choose exactly one: --pretrained PATH or --scratch')
    if args.train_all and args.holdout_domain:
        ap.error('--train-all and --holdout-domain cannot be used together')
    if args.pretrained and (not args.pretrained_topology or not args.provenance):
        ap.error('Pretrained weights require --pretrained-topology and --provenance')
    if args.pretrained and not Path(args.provenance).is_file():
        ap.error(f'Provenance document not found: {args.provenance}')
    if args.confidence_temperature <= 0:
        ap.error('--confidence-temperature must be positive')
    if args.epochs: cfg['training']['epochs'] = args.epochs
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.workers is not None: cfg['training']['workers'] = args.workers
    cfg['experiment'] = {'holdout_domain': args.holdout_domain, 'train_all':args.train_all,
                         'pretrained': args.pretrained, 'provenance': args.provenance,
                         'source_topology': args.pretrained_topology,
                         'selection':args.selection,
                         'confidence_bias':args.confidence_bias,
                         'confidence_temperature':args.confidence_temperature,
                         'mask_head_domains': args.mask_head_domains}
    seed_everything(cfg['training'].get('seed',2027))
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config_used.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    if args.provenance:
        (output_dir / 'PRETRAINING_PROVENANCE.txt').write_text(Path(args.provenance).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = cfg['data']; tr = cfg['training']; domains = args.domains
    train_domains = [d for d in domains if d != args.holdout_domain]
    train_set = PoseDataset(data['annotation_dir'], data['image_root'], train_domains,
                            input_hw=data['input_size'], split='all' if args.train_all else 'train',
                            val_fraction=data['val_fraction'], seed=tr.get('split_seed', tr['seed']),
                            augment=True, sigma=data.get('sigma',2.), aug=data.get('aug'))
    val_domains = [args.holdout_domain] if args.holdout_domain else domains
    val_set = None if args.train_all else PoseDataset(
        data['annotation_dir'], data['image_root'], val_domains, input_hw=data['input_size'],
        split='all' if args.holdout_domain else 'val',
        val_fraction=data['val_fraction'], seed=tr.get('split_seed', tr['seed']), augment=False)
    print('train',len(train_set),'val',len(val_set) if val_set else 'disabled (final train-all)',
          'device',dev,'domains',train_domains,flush=True)
    sampler = balanced_sampler(train_set) if tr.get('balanced', True) else None
    train_loader = DataLoader(train_set, batch_size=tr['batch_size'], sampler=sampler,
                              shuffle=sampler is None, num_workers=tr['workers'],
                              pin_memory=dev.type=='cuda', drop_last=False, persistent_workers=tr['workers']>0)
    val_loader = (DataLoader(val_set, batch_size=tr['batch_size'], shuffle=False,
                            num_workers=tr['workers'], pin_memory=dev.type=='cuda')
                  if val_set is not None else None)
    model = build_model(cfg)
    provenance = None
    if args.pretrained:
        provenance = load_pretrained(model, args.pretrained, args.pretrained_topology)
        print('pretrained coverage', provenance, flush=True)
    model = model.to(dev)
    opt = create_optimizer(model,cfg)
    epochs = tr['epochs']
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(total_steps * tr.get('warmup_ratio',.05)))
    base_lrs = [group['lr'] for group in opt.param_groups]
    use_bf16 = dev.type=='cuda' and torch.cuda.is_bf16_supported()
    ema = copy.deepcopy(model).eval() if tr.get('ema',True) else None
    if ema:
        for p in ema.parameters(): p.requires_grad_(False)
    best = -float('inf'); step=0
    history = []
    for epoch in range(1, epochs+1):
        model.train(); running=0.; count=0; begin=time.time()
        for batch in train_loader:
            step += 1
            if step < warmup_steps:
                factor = max(.01,step/max(1,warmup_steps))
            else:
                import math
                factor = .5*(1+math.cos(math.pi * (step-warmup_steps) / max(1,total_steps-warmup_steps)))
            for group, base in zip(opt.param_groups,base_lrs): group['lr']=base*factor
            imgs=batch['image'].to(dev,non_blocking=True)
            tgt=batch['target'].to(dev,non_blocking=True)
            mask=batch['weights'].to(dev,non_blocking=True)
            if args.mask_head_domains:
                drop = torch.tensor([d in args.mask_head_domains for d in batch['domain']], device=dev)
                mask = mask.clone(); mask[drop, 1:3] = 0
            opt.zero_grad(set_to_none=True)
            context = (torch.autocast(device_type='cuda',dtype=torch.bfloat16) if use_bf16 else nullcontext())
            with context:
                pred = model(imgs)
                loss = heatmap_loss(pred,tgt,mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch={epoch} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get('grad_clip',1.))
            opt.step()
            if ema:
                rate=tr.get('ema_decay',.999)
                with torch.no_grad():
                    current=dict(model.named_parameters())
                    for name,p in ema.named_parameters(): p.lerp_(current[name],1-rate)
                    current_buffers=dict(model.named_buffers())
                    for name,b in ema.named_buffers():
                        b.copy_(current_buffers[name])
            running += float(loss.detach()); count+=1
        active=ema if ema else model
        metrics=(evaluate(active,val_loader,dev,selection=args.selection,
                          confidence_bias=args.confidence_bias,
                          confidence_temperature=args.confidence_temperature)
                 if val_loader is not None else None)
        record={'epoch':epoch,'train_loss':running/max(1,count),'val':metrics,
                'seconds':round(time.time()-begin,1)}
        history.append(record)
        (output_dir/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(record),flush=True)
        state={k:v.detach().cpu().contiguous() for k,v in active.state_dict().items()}
        if args.train_all:
            torch.save({'state_dict':state,'config':cfg,'epoch':epoch,
                        'provenance':provenance},output_dir/'last.pth')
        else:
            val_metric=(metrics['oks_ap']['mean_source_domain_ap']
                        if args.selection=='oks_ap' else metrics['overall']['pck_010'])
            if val_metric > best:
                best=val_metric
                torch.save({'state_dict':state,'config':cfg,'epoch':epoch,
                            'selection_metric':args.selection, 'selection_value':best,
                            'provenance':provenance},output_dir/'best.pth')
        if dev.type=='cuda':
            print('peak VRAM GiB:', round(torch.cuda.max_memory_allocated()/2**30,2),flush=True)
            torch.cuda.reset_peak_memory_stats()
    print('Saved checkpoint:', output_dir/('last.pth' if args.train_all else 'best.pth'))

if __name__=='__main__':
    main()
