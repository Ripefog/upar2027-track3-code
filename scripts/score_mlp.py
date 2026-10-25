#!/usr/bin/env python3
"""Does a small MLP predicting instance OKS from cached features beat mean-peak? 2-fold by image hash, source data only."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_src.geometry import SCORED_INDEX
from pose_src.metrics import coco_ap_for_domain
AD = Path('/home/ubuntu/workspace/upar_official/mix_ann')

def ap_of(dom, F, idx, score):
    preds = [{'image_id': int(F['ids'][i]), 'category_id': 1, 'score': float(s),
              'keypoints': [v for p in F['pts'][i] for v in (float(p[0]), float(p[1]), 1.0)]} for i, s in zip(idx, score)]
    return coco_ap_for_domain(AD / f'train_{dom}.json', preds, set(int(F['ids'][i]) for i in idx))['AP']

def feats(F):
    pk = F['peaks'][:, SCORED_INDEX]; dis = F['disagree'][:, SCORED_INDEX]; sp = F['spread'][:, SCORED_INDEX]
    return np.concatenate([pk, np.log1p(dis), np.log(sp + 1e-3), np.log(F['scale'])[:, None], pk.mean(1, keepdims=True), pk.min(1, keepdims=True)], 1).astype(np.float32)

def fit_predict(Xtr, ytr, Xte, seed=0, epochs=60):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(ytr, dtype=torch.float32)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]; loss = ((net(Xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd)).numpy()[:, 0]

if __name__ == "__main__":
  for dom, cache in (('pa100k', 'outputs/score_study_cache.npz'), ('market', 'outputs/score_study_cache_market.npz'), ('peta', 'outputs/score_study_cache_peta.npz')):
      F = dict(np.load(cache)); X = feats(F); y = F['oks']
      half = np.array([int(hashlib.sha256(str(i).encode()).hexdigest(), 16) % 2 for i in F['ids']])
      r = {'mean-peak': [], 'mlp(oks)': [], 'mlp(oks)+mean-peak avg': []}
      for fit in (0, 1):
          tr = np.where(half == fit)[0]; te = np.where(half != fit)[0]
          mp = F['peaks'][te][:, SCORED_INDEX].mean(1)
          pred = np.mean([fit_predict(X[tr], y[tr], X[te], seed=s) for s in range(3)], 0)
          r['mean-peak'].append(ap_of(dom, F, te, mp)); r['mlp(oks)'].append(ap_of(dom, F, te, pred))
          z = lambda v: (v - v.mean()) / v.std()
          r['mlp(oks)+mean-peak avg'].append(ap_of(dom, F, te, z(pred) + z(mp)))
      print(dom, ' '.join(f'{k}={np.mean(v):.4f}' for k, v in r.items()), flush=True)
