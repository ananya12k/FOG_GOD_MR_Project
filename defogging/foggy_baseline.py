"""PSNR/SSIM of foggy input vs clean on the same test frames as eval_edfog_test.py, same formula and SSIM args.
Also rescores the saved defogged PNGs, to check this script reproduces the reported numbers."""
import csv, glob, os, sys
from multiprocessing import Pool
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
R = "/scratch/Ananya_Kulkarni"; RES = f"{R}/ft_eow_mf_results"; OUT = f"{R}/mid_eval/foggy_baseline.csv"
def rd(p): return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
def score(a):
    pred, gt = rd(a[0]), rd(a[1]); mse = max(float(((pred - gt) ** 2).mean()), 1e-10)
    return 10 * np.log10(1 / mse), ssim(gt, pred, data_range=1.0, channel_axis=2)
rows = []
with Pool(14) as P:
    for dom in ("endofworld", "multifloor"):
        for c in sorted(glob.glob(f"{RES}/1_before_finetune/{dom}/images/*")):
            clip = os.path.basename(c); names = sorted(os.listdir(c)); gt = f"{R}/Defog_eow_mf/train/GT/{dom}"
            for lab, d in (("foggy", f"{R}/Defog_eow_mf/train/Haze/{clip}"), ("defog_before", c), ("defog_after", c.replace("1_before_finetune", "2_after_finetune"))):
                r = np.array(P.map(score, [(f"{d}/{n}", f"{gt}/{n}") for n in names], chunksize=16))
                rows.append(dict(domain=dom, clip=clip, input=lab, n=len(names), psnr=r[:, 0].mean(), ssim=r[:, 1].mean()))
                print(dom, clip, lab, len(names), f"{r[:,0].mean():.3f} {r[:,1].mean():.4f}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
print("BASELINE_DONE")
