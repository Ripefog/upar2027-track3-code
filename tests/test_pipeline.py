import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_src.data import PoseDataset, stable_split, balanced_sampler
from pose_src.geometry import JOINTS, SCORED_INDEX, FLIP_INDEX, crop_params, crop_image, from_crop, to_crop, heatmap_targets, decode_numpy
from pose_src.models import build_model, HRNetW48
from pose_src.checkpoints import load_pretrained
from train import heatmap_loss


def test_topology():
    assert len(JOINTS)==17 and len(SCORED_INDEX)==15
    assert JOINTS[1:3]==('head_bottom','head_top')
    assert np.array_equal(FLIP_INDEX[FLIP_INDEX],np.arange(17))


def test_geometry_round_trip():
    params=crop_params([20,10,60,100],(256,192))
    pts=np.array([[35.1,55.7],[29.,42.]],np.float32)
    np.testing.assert_allclose(from_crop(to_crop(pts,params),params),pts,atol=1e-4)
    cropped=crop_image(Image.new('RGB',(100,180),'white'),params,(256,192))
    assert cropped.size==(192,256)


def test_heatmaps_and_loss():
    pts=np.ones((17,2),np.float32)*20
    tgt,w=heatmap_targets(pts,np.ones(17),(64,48))
    assert tgt.shape==(17,16,12)
    assert w[3]==w[4]==0
    assert float(tgt[0].max())>0.9
    x=torch.from_numpy(tgt)[None]
    assert torch.isclose(heatmap_loss(x,x,torch.from_numpy(w)[None]),torch.tensor(0.))
    decoded,_=decode_numpy(tgt)
    assert decoded.shape==(1,17,2)


def test_small_vit_forward_backward():
    cfg={'model':{'name':'vitpose','variant':'tiny'},'data':{'input_size':[64,48]}}
    m=build_model(cfg).train()
    x=torch.rand(2,3,64,48)
    hm=m(x)
    assert hm.shape==(2,17,16,12)
    hm.mean().backward()
    assert m.backbone.patch_embed.proj.weight.grad is not None


def test_hrnet_forward():
    m=HRNetW48(channels=8).eval()
    with torch.no_grad():
        out=m(torch.zeros(1,3,64,48))
    assert out.shape==(1,17,16,12)


def test_dataset_and_path(tmp_path):
    root=tmp_path/'data'
    (root/'market'/'images').mkdir(parents=True)
    im=Image.new('RGB',(100,160),'white')
    images=[];ann=[]
    for i in range(30):
        im.save(root/'market'/'images'/f'{i}.png')
        images.append({'id':i,'file_name':f'{i}.png'})
        ann.append({'id':i,'image_id':i,'bbox':[10,10,40,130],
                    'keypoints':sum(([20.,40.,2.] for _ in range(17)),[])})
    ann_dir=root/'annotations'/'specialization';ann_dir.mkdir(parents=True)
    (ann_dir/'train_market.json').write_text(json.dumps({'images':images,'annotations':ann}))
    ds=PoseDataset(ann_dir,root,['market'],input_hw=(64,48),split='train',
                   val_fraction=.25,seed=2027,augment=False)
    v=PoseDataset(ann_dir,root,['market'],input_hw=(64,48),split='val',
                   val_fraction=.25,seed=2027,augment=False)
    assert len(ds)+len(v)==30
    sample=ds[0]
    assert sample['image'].shape==(3,64,48)
    assert sample['target'].shape==(17,16,12)
    assert stable_split('market:1',.25)==stable_split('market:1',.25)


def test_fail_closed_checkpoint(tmp_path):
    cfg={'model':{'name':'vitpose','variant':'tiny'},'data':{'input_size':[64,48]}}
    m=build_model(cfg)
    wrong=tmp_path/'partial.pth'
    torch.save({'state_dict':{'head.final_layer.bias':torch.zeros(17)}},wrong)
    with pytest.raises(RuntimeError,match='coverage'):
        load_pretrained(m,wrong,'coco')


def test_optimizer_covers_all_hrnet_params():
    from train import create_optimizer
    cfg={'model':{'name':'hrnet_w32','channels':8},'data':{'input_size':[64,48]},
         'training':{'backbone_lr':1e-3,'head_lr':1e-3,'weight_decay':0.01}}
    m=build_model(cfg)
    opt=create_optimizer(m,cfg)
    in_opt=sum(p.numel() for g in opt.param_groups for p in g['params'])
    assert in_opt==sum(p.numel() for p in m.parameters())


def test_hrnet_checkpoint_coverage(tmp_path):
    cfg={'model':{'name':'hrnet_w32','channels':8},'data':{'input_size':[64,48]}}
    src=build_model(cfg)
    ck=tmp_path/'own.pth'
    torch.save({'state_dict':src.state_dict(),'config':cfg},ck)
    res=load_pretrained(build_model(cfg),ck,'posetrack18')
    assert res['backbone_coverage']>0.99


def test_rotate_crop_moves_keypoints_with_image():
    from pose_src.data import rotate_crop
    im=Image.new('RGB',(61,101),'black')
    x,y=40,30
    im.putpixel((x,y),(255,255,255))
    for deg in (-20,10,25):
        out,pts=rotate_crop(im,np.array([[x,y]],np.float32),deg)
        arr=np.asarray(out).sum(-1)
        yy,xx=np.unravel_index(arr.argmax(),arr.shape)
        assert abs(xx-pts[0,0])<=1.5 and abs(yy-pts[0,1])<=1.5


def test_dark_decodes_subpixel_gaussian_better_than_sign_shift():
    from pose_src.geometry import decode_dark
    yy,xx=np.mgrid[:64,:48]
    errs_old=[];errs_dark=[]
    rng=np.random.default_rng(0)
    for _ in range(20):
        cx,cy=rng.uniform(8,40),rng.uniform(8,56)
        hm=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*2.0**2)).astype(np.float32)
        hm=np.repeat(hm[None,None],17,1)
        o,_=decode_numpy(hm); d,_=decode_dark(hm)
        errs_old.append(np.abs(o[0,0]/4-[cx,cy]).max()); errs_dark.append(np.abs(d[0,0]/4-[cx,cy]).max())
    assert np.mean(errs_dark)<0.05 and np.mean(errs_dark)<np.mean(errs_old)


def test_shift_heatmap_x_moves_peak_right():
    from pose_src.geometry import shift_heatmap_x
    hm=np.zeros((1,17,8,10),np.float32); hm[...,3,4]=1.
    out=shift_heatmap_x(hm,1)
    assert out[0,0,3,5]==1. and out[0,0,3,4]==0.
    assert np.array_equal(shift_heatmap_x(hm,0),hm)
