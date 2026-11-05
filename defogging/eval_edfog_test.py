"""
eval_edfog_test.py  —  Full-reference test-set evaluation for FogPromptNetV2 (DriveDefog)
=========================================================================================
Runs the V2 model (DCN temporal alignment + 19-class SPAIR) over every Haze clip of
<dataset-root>/test with a centred sliding window of --seq-len frames (edge frames
replicate-padded, same as VideoDataset), and scores the centre prediction of each
window against its GT frame at the dataset's native resolution.

Input is reflect-padded to a multiple of 16 (the encoder's total stride) and the
output cropped back, so no resizing touches the scored pixels.

Outputs in --out-dir:
    per_clip.csv   clip, scene, mor, heterogeneity, airlight, fog_layer_m, n_frames, psnr, ssim
    summary.csv    overall + per-MOR means (frame-weighted)

Usage:
    python student_train_inference/inference/eval_edfog_test.py \
        --ckpt  checkpoints/drivedefog_edfog/best_model.pth \
        --dataset-root /scratch/Ananya_Kulkarni/EDFog \
        --out-dir      checkpoints/drivedefog_edfog/test_eval
"""
import os, sys, csv, json, argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from student_train_inference.models.msbdn_rdff_dual_p3d_fogprompt_v2 import Net as FogPromptNetV2
from student_train_inference.training.video_dataset import natural_sort_key


def list_images(d):
    return sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))],
                  key=natural_sort_key)


def load_stack(d, files):
    return np.stack([np.asarray(Image.open(os.path.join(d, f)).convert('RGB')) for f in files])


def load_model(ckpt_path, seq_len, device):
    model = FogPromptNetV2(res_blocks=18, num_classes=19, T=seq_len).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f'[ERROR] checkpoint is missing {len(missing)} V2 keys, e.g. {missing[:5]} '
                         f'— not a FogPromptNetV2 checkpoint?')
    if unexpected:
        print(f'[WARN] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:5]}')
    model.eval()
    print(f'[Model] {ckpt_path}  epoch={ckpt.get("epoch")}  best_val_psnr={ckpt.get("best_psnr")}')
    return model


@torch.no_grad()
def eval_clip(model, haze, gt, seq_len, device, amp, frames=None, save_dir=None, names=None, save_ext='png'):
    """haze, gt: uint8 [N, H, W, 3]. Returns per-frame (psnr, ssim) arrays."""
    n, H, W, _ = haze.shape
    pad_h, pad_w = (-H) % 16, (-W) % 16
    h = seq_len // 2
    psnrs, ssims = [], []
    haze_t = torch.from_numpy(haze)   # stays on CPU: long clips do not fit in GPU memory
    for c in (range(n) if frames is None else frames):
        idxs = [min(max(c + i, 0), n - 1) for i in range(-h, h + 1)]
        x = haze_t[idxs].to(device).permute(0, 3, 1, 2).float().div_(255)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        with torch.autocast('cuda', enabled=amp):
            out, _, _, _ = model(x)
        pred = out[h, :, :H, :W].float().clamp(0, 1)
        if save_dir:
            arr = pred.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            kw = dict(quality=95, subsampling=0) if save_ext == 'jpg' else {}
            Image.fromarray(arr).save(os.path.join(save_dir, os.path.splitext(names[c])[0] + '.' + save_ext), **kw)
        g    = torch.from_numpy(gt[c]).to(device).permute(2, 0, 1).float().div(255)
        mse  = (pred - g).pow(2).mean().clamp_min(1e-10)
        psnrs.append((10 * torch.log10(1.0 / mse)).item())
        ssims.append(sk_ssim(g.permute(1, 2, 0).cpu().numpy(), pred.permute(1, 2, 0).cpu().numpy(),
                             data_range=1.0, channel_axis=2))
    return np.array(psnrs), np.array(ssims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seq-len', type=int, default=7)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--max-samples', type=int, default=0,
                   help='total frames to score, split equally across all clips and evenly spaced; 0 = every frame')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1,
                   help='split clips across processes (one per GPU); merge with --merge-only')
    p.add_argument('--save-dir', default=None, help='also write each scored restored frame to <save-dir>/<clip>/')
    p.add_argument('--save-ext', default='png', choices=['png', 'jpg'])
    p.add_argument('--clip-prefix', default=None, help='only clips whose name starts with this, e.g. endofworld_')
    p.add_argument('--merge-only', action='store_true',
                   help='combine per_clip.shard*.csv from --out-dir into per_clip.csv + summary.csv')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_only:
        rows = []
        for f in sorted(os.listdir(args.out_dir)):
            if f.startswith('per_clip.shard') and f.endswith('.csv'):
                with open(os.path.join(args.out_dir, f)) as fh:
                    rows += list(csv.DictReader(fh))
        if not rows:
            raise SystemExit(f'no per_clip.shard*.csv in {args.out_dir}')
        for r in rows:
            r['n_frames'], r['psnr'], r['ssim'] = int(r['n_frames']), float(r['psnr']), float(r['ssim'])
        write_results(sorted(rows, key=lambda r: r['clip']), args.out_dir)
        return

    device = torch.device(f'cuda:{args.gpu}')
    model  = load_model(args.ckpt, args.seq_len, device)
    root   = os.path.join(args.dataset_root, args.split)
    clips  = sorted(d for d in os.listdir(os.path.join(root, 'Haze'))
                    if os.path.isdir(os.path.join(root, 'Haze', d)))
    if args.clip_prefix:
        clips = [c for c in clips if c.startswith(args.clip_prefix)]
    if not clips:
        raise SystemExit(f'[ERROR] no clips in {root}/Haze match prefix {args.clip_prefix!r}')
    per_clip = args.max_samples // len(clips) if args.max_samples > 0 else 0
    clips  = clips[args.shard_index::args.num_shards]

    rows = []
    for clip in tqdm(clips, desc='Clips'):
        hp = os.path.join(root, 'Haze', clip)
        scene = clip.split('_')[0]
        gp = os.path.join(root, 'GT', scene)
        meta = {}
        if os.path.isfile(os.path.join(hp, 'meta.json')):
            with open(os.path.join(hp, 'meta.json')) as f:
                meta = json.load(f)
        hf, gf = list_images(hp), list_images(gp)
        n = min(len(hf), len(gf))
        frames = (np.unique(np.linspace(0, n - 1, per_clip).round().astype(int)).tolist()
                  if 0 < per_clip < n else None)
        clip_save = os.path.join(args.save_dir, clip) if args.save_dir else None
        if clip_save:
            os.makedirs(clip_save, exist_ok=True)
        psnr, ssim = eval_clip(model, load_stack(hp, hf[:n]), load_stack(gp, gf[:n]),
                               args.seq_len, device, args.amp, frames, clip_save, hf, args.save_ext)
        rows.append(dict(clip=clip, scene=scene, mor=meta.get('mor', ''),
                         heterogeneity=meta.get('heterogeneity', ''),
                         airlight=meta.get('airlight_color', ''),
                         fog_layer_m=meta.get('fog_layer_m', ''),
                         n_frames=len(psnr), psnr=psnr.mean(), ssim=ssim.mean()))
        print(f'  {clip}: MOR={rows[-1]["mor"]}  PSNR={psnr.mean():.3f}  SSIM={ssim.mean():.4f}  ({len(psnr)} of {n} frames)',
              flush=True)

    if args.num_shards > 1:
        with open(os.path.join(args.out_dir, f'per_clip.shard{args.shard_index}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return
    write_results(rows, args.out_dir)


def write_results(rows, out_dir):
    with open(os.path.join(out_dir, 'per_clip.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def weighted(sel):
        nf = sum(r['n_frames'] for r in sel)
        return (sum(r['psnr'] * r['n_frames'] for r in sel) / nf,
                sum(r['ssim'] * r['n_frames'] for r in sel) / nf, nf)

    summary = [('overall',) + weighted(rows)]
    for mor in sorted({float(r['mor']) for r in rows if r['mor'] != ''}):
        summary.append((f'MOR {mor:g} m',) + weighted([r for r in rows if r['mor'] != '' and float(r['mor']) == mor]))
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'psnr', 'ssim', 'n_frames'])
        w.writerows(summary)

    print('\n=== Test results ===')
    for g, ps, ss, nf in summary:
        print(f'  {g:12s}  PSNR={ps:.3f}  SSIM={ss:.4f}  ({nf} frames)')
    print(f'\nSaved -> {out_dir}/per_clip.csv, summary.csv')


if __name__ == '__main__':
    main()
