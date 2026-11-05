"""Link view for fine-tuning on new domains with replay of the old driving data. Copies no images.
  <out>/train  = new domain train frames + REPLAY old clips per scene (spread across all MORs)
  <out>/val    = new domain val frames   + every old val clip
  <out>/test   = new domain test frames only (old test is evaluated from its own dataset)
Held-out frames never enter train; gaps between the parts stop near-duplicates leaking across."""
import csv, json, os, shutil, sys
old, new, out, domains, replay = sys.argv[1:6]; domains = domains.split(','); replay = int(replay)
# Test parts are long enough for trajectory evaluation; gaps stop near-duplicates leaking across.
RANGES = {'endofworld': dict(train=(1, 425), val=(451, 500), test=(526, 708), total=708),
          'multifloor': dict(train=(1, 5000), val=(5251, 5500), test=(5698, 10017), total=10017)}
if os.path.exists(out): shutil.rmtree(out)
L = lambda t, n: (os.makedirs(os.path.dirname(n), exist_ok=True), os.symlink(t, n))
hz = f'{new}/train/Haze'; counts = {}; nnew = 0
# Optional: EXCLUDE_MORS=20,30 leaves those fog levels of the new domains out of the view.
exclude = {float(m) for m in os.environ.get('EXCLUDE_MORS', '').split(',') if m.strip()}
for d in domains:
    clips = sorted(c for c in os.listdir(hz) if c.startswith(d + '_'))
    if len(clips) != 6: sys.exit(f'!!! {d}: expected 6 clips in {hz}, found {len(clips)}')
    for c in clips:
        n = len([f for f in os.listdir(f'{hz}/{c}') if f.endswith('.png')])
        if n != RANGES[d]['total'] or not os.path.isfile(f'{hz}/{c}/meta.json'):
            sys.exit(f'!!! {c}: {n} frames, expected {RANGES[d]["total"]}: generation or transfer unfinished')
    clips = [c for c in clips if float(json.load(open(f'{hz}/{c}/meta.json'))['mor']) not in exclude]
    if not clips: sys.exit(f'!!! {d}: EXCLUDE_MORS removed every clip')
    nnew += len(clips)
    for split in ('train', 'val', 'test'):
        a, b = RANGES[d][split]; names = [f'{i:06d}.png' for i in range(a, b + 1)]
        for f in names: L(f'{new}/train/GT/{d}/{f}', f'{out}/{split}/GT/{d}/{f}')
        for c in clips:
            for f in names: L(f'{hz}/{c}/{f}', f'{out}/{split}/Haze/{c}/{f}')
            shutil.copy(f'{hz}/{c}/meta.json', f'{out}/{split}/Haze/{c}/meta.json')
        counts[f'{split}/{d}'] = f'{len(clips)} clips x {len(names)} frames'
rows = list(csv.DictReader(open(f'{old}/manifest.csv')))
clip = lambda r: os.path.basename(r['haze_path'])
scenes = sorted({r['scene'] for r in rows if r['split'] == 'train'}); kept = []
for i, sc in enumerate(scenes):
    rs = sorted((r for r in rows if r['split'] == 'train' and r['scene'] == sc), key=lambda r: float(r['mor']))
    step = len(rs) / max(1, replay)
    kept += [rs[(int(k * step) + i) % len(rs)] for k in range(min(replay, len(rs)))]
    L(f'{old}/train/GT/{sc}', f'{out}/train/GT/{sc}')
for r in kept: L(f'{old}/train/Haze/{clip(r)}', f'{out}/train/Haze/{clip(r)}')
for sc in sorted({r['scene'] for r in rows if r['split'] == 'val'}): L(f'{old}/val/GT/{sc}', f'{out}/val/GT/{sc}')
vals = [r for r in rows if r['split'] == 'val']
for r in vals: L(f'{old}/val/Haze/{clip(r)}', f'{out}/val/Haze/{clip(r)}')
print(json.dumps(counts, indent=1))
print(f'replay train clips: {len(kept)}, MOR counts:', dict(sorted({m: sum(float(r["mor"]) == m for r in kept) for m in {float(r["mor"]) for r in kept}}.items())))
print(f'train clips: {nnew} new + {len(kept)} old = new-domain share {100 * nnew / (nnew + len(kept)):.0f}%; val clips: {nnew} new + {len(vals)} old')
