"""Uniform weight average of checkpoints with identical architecture. Usage: make_soup.py OUT.pth CK1 CK2 ..."""
import sys, os, torch
out, cks = sys.argv[1], sys.argv[2:]
cs = [torch.load(p, map_location='cpu', weights_only=False) for p in cks]
sd = {k: sum(c['state_dict'][k].float() for c in cs) / len(cs) for k in cs[0]['state_dict']}
res = dict(cs[0]); res['state_dict'] = sd
res['provenance'] = f"uniform soup of: {', '.join(cks)}\n" + str(cs[0].get('provenance', ''))
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(res, out)
print('saved', out, 'from', len(cks))
