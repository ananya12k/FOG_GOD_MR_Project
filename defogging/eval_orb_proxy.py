"""ORB matching check on the same frames, for clean, foggy and defogged images. It is NOT ORB-SLAM:
it counts how many ORB matches survive a geometric (RANSAC) check between frames STRIDE apart, a fast
in-house signal to read alongside the real ORB-SLAM results. Writes per-clip rows and a summary by MOR."""
import argparse, csv, json, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd())
import importlib.util
# Load by path: student_train_inference/inference.py would shadow the inference/ folder.
_spec = importlib.util.spec_from_file_location('eval_edfog_test', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_edfog_test.py'))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
load_model, list_images, load_stack = _mod.load_model, _mod.list_images, _mod.load_stack

def inliers(a, b, orb, bf):
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return len(ka), 0, 0
    m = bf.match(da, db)
    if len(m) < 8: return len(ka), len(m), 0
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.99)
    return len(ka), len(m), 0 if mask is None else int(mask.sum())

@torch.no_grad()
def restore(model, haze_t, c, seq, dev, amp):
    h, n = seq // 2, len(haze_t); H, W = haze_t.shape[1:3]
    x = haze_t[[min(max(c + i, 0), n - 1) for i in range(-h, h + 1)]].to(dev).permute(0, 3, 1, 2).float().div_(255)
    ph, pw = (-H) % 16, (-W) % 16
    if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    with torch.autocast('cuda', enabled=amp): out = model(x)[0]
    return (out[h, :, :H, :W].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

a = argparse.ArgumentParser()
for k, v in dict(ckpt=None, root=None, out=None).items(): a.add_argument('--' + k, required=True)
a.add_argument('--split', default='test'); a.add_argument('--pairs', type=int, default=100); a.add_argument('--stride', type=int, default=5)
a.add_argument('--seq-len', type=int, default=7); a.add_argument('--gpu', type=int, default=0)
a = a.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); model = load_model(a.ckpt, a.seq_len, dev)
orb = cv2.ORB_create(1500); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True); gray = lambda i: cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
root = os.path.join(a.root, a.split); rows = []
for clip in sorted(d for d in os.listdir(f'{root}/Haze') if os.path.isdir(f'{root}/Haze/{d}')):
    hp, gp = f'{root}/Haze/{clip}', f'{root}/GT/{clip.split("_")[0]}'; meta = json.load(open(f'{hp}/meta.json'))
    hf, gf = list_images(hp), list_images(gp); n = min(len(hf), len(gf)); haze, gt = load_stack(hp, hf[:n]), load_stack(gp, gf[:n]); ht = torch.from_numpy(haze)
    tot = {k: np.zeros(3) for k in ('clean', 'foggy', 'defog')}; idx = np.unique(np.linspace(0, n - 1 - a.stride, a.pairs).round().astype(int))
    for c in idx:
        c = int(c); d = c + a.stride
        for k, (x, y) in dict(clean=(gt[c], gt[d]), foggy=(haze[c], haze[d]), defog=(restore(model, ht, c, a.seq_len, dev, True), restore(model, ht, d, a.seq_len, dev, True))).items():
            tot[k] += np.array(inliers(gray(x), gray(y), orb, bf))
    r = dict(clip=clip, mor=meta.get('mor', ''), pairs=len(idx))
    for k in tot: r[f'{k}_keypoints'], r[f'{k}_matches'], r[f'{k}_inliers'] = tot[k] / len(idx)
    rows.append(r); print(clip, {k: round(r[f'{k}_inliers'], 1) for k in tot}, flush=True)
os.makedirs(a.out, exist_ok=True)
w = csv.DictWriter(open(f'{a.out}/per_clip.csv', 'w', newline=''), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f'{a.out}/summary.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['group', 'clean_inliers', 'foggy_inliers', 'defog_inliers', 'clean_keypoints', 'foggy_keypoints', 'defog_keypoints'])
    groups = [('overall', rows)] + [(f'MOR {m:g} m', [r for r in rows if float(r['mor']) == m]) for m in sorted({float(r['mor']) for r in rows})]
    for g, s in groups:
        w.writerow([g] + [round(np.mean([r[f'{k}_{v}'] for r in s]), 2) for v in ('inliers', 'keypoints') for k in ('clean', 'foggy', 'defog')])
