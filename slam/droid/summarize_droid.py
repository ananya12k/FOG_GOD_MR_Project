#!/usr/bin/env python
"""Collect results/**/metrics.json into droid_runs.csv and droid_summary.csv (median ATE over runs)."""
import csv, glob, json, statistics as st
O = '/scratch/Ananya_Kulkarni/slam/results'; rows = []
for f in sorted(glob.glob(f'{O}/*/*/*/run*/')):
    p = f.rstrip('/').split('/'); r = dict(domain=p[-4], condition=p[-3], mor=p[-2], run=p[-1])
    try: m = json.load(open(f + 'metrics.json')); r.update(ok=1, **{k: m[k] for k in ('ate_rmse_m', 'keyframes', 'frames', 'gt_matched', 'gt_path_m', 'sim3_scale', 'fps', 'peak_gpu_gb')})
    except FileNotFoundError: r.update(ok=0)
    rows.append(r)
keys = ['domain', 'condition', 'mor', 'run', 'ok', 'ate_rmse_m', 'keyframes', 'frames', 'gt_matched', 'gt_path_m', 'sim3_scale', 'fps', 'peak_gpu_gb']
w = csv.DictWriter(open(f'{O}/droid_runs.csv', 'w', newline=''), keys); w.writeheader(); w.writerows(rows)
groups = {}
for r in rows: groups.setdefault((r['domain'], r['condition'], r['mor']), []).append(r)
S = []
for (d, c, m), rs in sorted(groups.items()):
    a = [x['ate_rmse_m'] for x in rs if x['ok']]
    S.append(dict(domain=d, condition=c, mor=m, runs_ok=len(a), runs_total=len(rs), ate_rmse_median_m=round(st.median(a), 4) if a else '', ate_min_m=round(min(a), 4) if a else '', ate_max_m=round(max(a), 4) if a else '', gt_path_m=next((x['gt_path_m'] for x in rs if x['ok']), '')))
w = csv.DictWriter(open(f'{O}/droid_summary.csv', 'w', newline=''), S[0].keys()); w.writeheader(); w.writerows(S)
for s in S: print(s)
