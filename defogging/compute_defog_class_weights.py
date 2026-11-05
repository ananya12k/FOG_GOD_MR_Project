"""
compute_defog_class_weights.py

Per-Cityscapes-class inverse-frequency weights for the SD decoder's segmentation
CrossEntropyLoss in train_fogprompt_exp.py (--class-weights-path), computed from
the SegFormer pseudo-label .npy maps under <sem-root>/<scene>/*.npy.

Weighting (--method):
    inverse  total_pixels / (num_classes * count_c), clipped to [0.1, 50]
             (same as compute_night_class_weights.py / sklearn 'balanced')
    sqrt     sqrt(median_freq / freq_c), clipped to [0.1, --clip-max]; the gentler
             default, so one rare class cannot dominate the seg gradient

Usage:
    python compute_defog_class_weights.py \
        --sem-root /scratch/Ananya_Kulkarni/Defog_Dataset/train/Semantic \
        --out      /scratch/Ananya_Kulkarni/VISIA/class_weights_defog.pt
"""
import os
import argparse
from multiprocessing import Pool

import numpy as np
import torch

NUM_CLASSES = 19
CLIP_MIN = 0.1

CITYSCAPES_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


def count_file(path):
    labels = np.load(path)
    valid = labels[labels < NUM_CLASSES]
    return np.bincount(valid.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sem-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stride', type=int, default=10,
                   help='use every Nth frame per scene (neighbouring frames are near-identical)')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--method', choices=['sqrt', 'inverse'], default='sqrt')
    p.add_argument('--clip-max', type=float, default=None,
                   help='upper weight limit (default: 10 for sqrt, 50 for inverse)')
    args = p.parse_args()

    files = []
    for scene in sorted(os.listdir(args.sem_root)):
        d = os.path.join(args.sem_root, scene)
        if os.path.isdir(d):
            npys = sorted(f for f in os.listdir(d) if f.endswith('.npy'))
            files += [os.path.join(d, f) for f in npys[::args.stride]]
    if not files:
        raise SystemExit(f'no .npy label maps under {args.sem_root}')
    print(f'Counting {len(files)} label maps with {args.workers} workers ...')

    with Pool(args.workers) as pool:
        counts = np.sum(pool.map(count_file, files, chunksize=16), axis=0).astype(np.int64)

    total   = counts.sum()
    freq    = counts / max(total, 1)
    if args.method == 'sqrt':
        weights  = np.sqrt(np.median(freq[counts > 0]) / np.maximum(freq, 1e-12))
        clip_max = 10.0 if args.clip_max is None else args.clip_max
    else:
        weights  = total / (NUM_CLASSES * np.maximum(counts, 1))
        clip_max = 50.0 if args.clip_max is None else args.clip_max
    weights = np.clip(weights, CLIP_MIN, clip_max).astype(np.float32)

    print('\nPer-class pixel counts / frequency / weight:')
    for i, name in enumerate(CITYSCAPES_NAMES):
        print(f'  {i:2d} {name:14s} count={counts[i]:>14,d}  freq={freq[i]:.6f}  '
              f'weight={weights[i]:.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(torch.from_numpy(weights), args.out)
    print(f'\nSaved class weights -> {args.out}')


if __name__ == '__main__':
    main()
