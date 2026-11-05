#!/usr/bin/env python
"""Full DROID-SLAM matrix for the mid eval. Safe to re-run: finished runs (metrics.json) are skipped.
results/<domain>/<condition>/<mor>/run<k>/  conditions: clean, foggy, defog_before, defog_after. Workers: one per GPU."""
import json, os, queue, subprocess, threading
R = '/scratch/Ananya_Kulkarni'; RES = f'{R}/ft_eow_mf_results'; S = f'{R}/slam'; OUT = f'{S}/results'
PY = f'{S}/droid_env/bin/python'; RUNS = 3; GPUS = [0, 1, 2, 3]
MOR = {'cfg0_beta0.0400': 75, 'cfg1_beta0.0300': 100, 'cfg2_beta0.0200': 150, 'cfg3_beta0.0600': 50}
jobs = []
for k in range(1, RUNS + 1):
    for dom in ('endofworld', 'multifloor'):
        ref = f'{RES}/1_before_finetune/{dom}/images/{dom}_cfg0_beta0.0400'
        jobs.append((dom, 'clean', 'none', k, f'{R}/Defog_eow_mf/train/GT/{dom}', ref))
        for cfg, m in sorted(MOR.items(), key=lambda x: x[1]):
            c = f'{dom}_{cfg}'
            jobs.append((dom, 'foggy', m, k, f'{R}/Defog_eow_mf/train/Haze/{c}', f'{RES}/1_before_finetune/{dom}/images/{c}'))
            jobs.append((dom, 'defog_before', m, k, f'{RES}/1_before_finetune/{dom}/images/{c}', None))
            jobs.append((dom, 'defog_after', m, k, f'{RES}/2_after_finetune/{dom}/images/{c}', None))
q = queue.Queue(); [q.put(j) for j in jobs]; lock = threading.Lock(); done = [0]
def work(g):
    while True:
        try: dom, cond, m, k, imgs, ref = q.get_nowait()
        except queue.Empty: return
        o = f'{OUT}/{dom}/{cond}/{m}/run{k}'; os.makedirs(o, exist_ok=True)
        if os.path.exists(f'{o}/metrics.json'): st = 'skip'
        else:
            cmd = [PY, f'{S}/droid_clip.py', '--images', imgs, '--domain', dom, '--out', o] + (['--frames-from', ref] if ref else [])
            with open(f'{o}/run.log', 'w') as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=S, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
            st = 'ok' if os.path.exists(f'{o}/metrics.json') else f'FAILED rc={rc}'
        with lock:
            done[0] += 1; print(f'[{done[0]}/{len(jobs)}] gpu{g} {dom} {cond} {m} run{k}: {st}', flush=True)
ts = [threading.Thread(target=work, args=(g,)) for g in GPUS]; [t.start() for t in ts]; [t.join() for t in ts]
print('ALL_DONE', flush=True)
