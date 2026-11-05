#!/bin/bash
#SBATCH -A mobility_arfs
#SBATCH -c 48
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --partition=ihub
#SBATCH --nodelist=gnode102
#SBATCH --chdir=/scratch/Ananya_Kulkarni/VISIA
#SBATCH --output=/scratch/Ananya_Kulkarni/VISIA/ft_eow_mf_all_%j.txt
#SBATCH --job-name=ft_eow_mf_all
#SBATCH --signal=B:USR1@900
#SBATCH --requeue

# One job: EOW+MF inference with the best 10k checkpoint, fine-tune, EOW+MF inference after.
# Re-running skips finished steps.
set -Eeuo pipefail
echo "Running on $(hostname)  job=${SLURM_JOB_ID:-none}  restarts=${SLURM_RESTART_COUNT:-0}"
echo "Job started at $(date)"

R=/scratch/Ananya_Kulkarni
VISIA=$R/VISIA
PY="$HOME/miniconda3/envs/visia/bin/python"
TORCHRUN="$(dirname "$PY")/torchrun"
EVAL=student_train_inference/inference/eval_edfog_test.py
BASE_CKPT="$VISIA/checkpoints/drivedefog_6mor/best_model.pth"   # 10k-sample base run, epoch 12
OLD_DATA="$R/Defog_fog_6mor"
NEW_DATA="$R/Defog_eow_mf"
DATA="$R/Defog_ft_eowmf_6mor"
DOMAINS=endofworld,multifloor
export EXCLUDE_MORS=20,30
REPLAY_PER_SCENE=2
SAVE="$HOME/ft_eow_mf_6mor"
OUT="$R/ft_eow_mf_results"
FT_EPOCHS=5
FT_LR=2e-5
START_AT=20
END_AT=$(( START_AT + FT_EPOCHS ))
TRAIN_SAMPLES=10000
VAL_SAMPLES=4000
TEST_SAMPLES=10000
ORB_PAIRS=100
echo "BASE_CKPT=$BASE_CKPT  DATA=$DATA  SAVE=$SAVE  OUT=$OUT  EXCLUDE_MORS=$EXCLUDE_MORS  REPLAY=$REPLAY_PER_SCENE  EPOCHS=$FT_EPOCHS"

cd "$VISIA"
[[ -f "$BASE_CKPT" ]] || { echo "!!! no $BASE_CKPT"; exit 1; }
FREE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
(( FREE_GB >= 40 )) || { echo "!!! only ${FREE_GB} GB free on /scratch, need 40"; exit 1; }
mkdir -p "$SAVE" "$OUT"

NGPU=$(nvidia-smi -L | wc -l)
NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NUM_WORKERS=$(( NCPU / NGPU - 2 )); (( NUM_WORKERS < 1 )) && NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MASTER_PORT=$(( 31500 + ${SLURM_JOB_ID:-0} % 1000 ))
echo "Resources: $NGPU GPU(s), $NCPU CPU(s) -> $NUM_WORKERS loader workers per GPU"

CHILD=0
trap 'echo "[$(date)] time limit near -> requeue"; kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
run_bg() { "$@" & CHILD=$!; wait $CHILD; }

infer() {   # label ckpt root out_dir images_dir_or_- clip_prefix_or_-
    local label=$1 ckpt=$2 root=$3 out=$4 images=$5 prefix=$6 g PIDS=() extra=() p
    [[ -f "$out/summary.csv" ]] && { echo "  $label: done, skipping"; return; }
    echo "--- $label  ($(date))"
    mkdir -p "$out"; rm -f "$out"/per_clip.shard*.csv
    [[ "$images" != - ]] && extra+=(--save-dir "$images")
    [[ "$prefix" != - ]] && extra+=(--clip-prefix "$prefix")
    for ((g=0; g<NGPU; g++)); do
        "$PY" "$EVAL" --ckpt "$ckpt" --dataset-root "$root" --split test --out-dir "$out" --seq-len 7 \
            --gpu $g --shard-index $g --num-shards "$NGPU" --max-samples "$TEST_SAMPLES" "${extra[@]}" > "$out/shard$g.log" 2>&1 &
        PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || { echo "!!! $label failed, see $out/shard*.log"; exit 1; }; done
    "$PY" "$EVAL" --ckpt "$ckpt" --dataset-root "$root" --out-dir "$out" --merge-only
}

echo "=== [1/4] data view ==="
if [[ -f "$DATA/.view_complete" && "$(cat "$DATA/.view_complete")" == "$DOMAINS $REPLAY_PER_SCENE $EXCLUDE_MORS" ]]; then
    echo "  exists, skipping"
else
    "$PY" build_ft_domains.py "$OLD_DATA" "$NEW_DATA" "$DATA" "$DOMAINS" "$REPLAY_PER_SCENE"
    echo "$DOMAINS $REPLAY_PER_SCENE $EXCLUDE_MORS" > "$DATA/.view_complete"
fi

echo "=== [2/4] EOW and MF inference, before fine-tuning ==="
infer "EOW before" "$BASE_CKPT" "$DATA" "$OUT/1_before_finetune/endofworld/metrics" "$OUT/1_before_finetune/endofworld/images" endofworld_
infer "MF before"  "$BASE_CKPT" "$DATA" "$OUT/1_before_finetune/multifloor/metrics" "$OUT/1_before_finetune/multifloor/images" multifloor_
[[ -f "$OUT/orb_before/summary.csv" ]] || "$PY" student_train_inference/inference/eval_orb_proxy.py --ckpt "$BASE_CKPT" --root "$DATA" --out "$OUT/orb_before" --pairs "$ORB_PAIRS" --gpu 0 || echo "WARNING: ORB check before failed, continuing"

echo "=== [3/4] fine-tune: epochs $((START_AT+1))-$END_AT, starting LR $FT_LR ==="
last_epoch() {
    [[ -f "$SAVE/latest.pth" ]] || { echo 0; return; }
    "$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch', 0))" "$SAVE/latest.pth"
}
INIT="$SAVE/init_from_base.pth"
if [[ -f "$SAVE/latest.pth" ]]; then
    RESUME="$SAVE/latest.pth"; echo "  continuing from $RESUME"
else
    "$PY" - "$BASE_CKPT" "$INIT" "$START_AT" <<'PYEOF'
import os, sys, torch
src, dst, e = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = torch.load(src, map_location='cpu', weights_only=False)
print(f"  base: epoch={c.get('epoch')} best_psnr={c.get('best_psnr')}")
torch.save({'epoch': e, 'state_dict': c['state_dict'], 'best_psnr': 0.0}, dst + '.tmp'); os.replace(dst + '.tmp', dst)
PYEOF
    echo 'epoch,total,charbonnier,seg,perceptual,loc,sky_colour,sky_temporal,fog_w_calib' > "$SAVE/train_loss.csv"
    echo 'epoch,psnr,ssim' > "$SAVE/val_metrics.csv"
    RESUME="$INIT"
fi
RAW_LR=$(CUDA_VISIBLE_DEVICES= "$PY" ft_lr.py "$FT_LR" "$START_AT" "$FT_EPOCHS" 1 2>/dev/null)
if (( $(last_epoch) >= END_AT )); then
    echo "  already at epoch $END_AT, skipping"
else
    run_bg "$TORCHRUN" --nproc_per_node="$NGPU" --master_port "$MASTER_PORT" \
        student_train_inference/training/train_fogprompt_exp.py \
        --save-dir "$SAVE" --dataset-root "$DATA" --resume "$RESUME" \
        --max-epoch "$END_AT" --batch-size 1 --grad-accum-steps 1 --seq-len 7 \
        --img-size 256 256 --num-workers "$NUM_WORKERS" \
        --max-train-samples "$TRAIN_SAMPLES" --max-val-samples "$VAL_SAMPLES" --balanced-subset \
        --val-freq 1 --lr "$RAW_LR" --lr-min 1e-7 --warmup-epochs 1 \
        --perceptual-weight 0.2 --seg-loss-weight 0 --sd-lr-mult 0 --grad-clip 1.0 \
        --temporal-loss-weight 0.3 --structure-loss-weight 0.2 --distortion-alpha 4.0 --distortion-loc-weight 0.1 \
        --distortion-threshold 0.08 --sign-boost 3.0 --fog-w-calib-weight 0.2 \
        --use-augmentation --aug-photometric --aug-noise --aug-spatial --amp
    (( $(last_epoch) >= END_AT )) || { echo "!!! fine-tune stopped before epoch $END_AT"; exit 1; }
fi

echo "=== [4/4] EOW and MF inference, after fine-tuning ==="
BEST="$SAVE/best_model.pth"
infer "EOW after" "$BEST" "$DATA" "$OUT/2_after_finetune/endofworld/metrics" "$OUT/2_after_finetune/endofworld/images" endofworld_
infer "MF after"  "$BEST" "$DATA" "$OUT/2_after_finetune/multifloor/metrics" "$OUT/2_after_finetune/multifloor/images" multifloor_
[[ -f "$OUT/orb_after/summary.csv" ]] || "$PY" student_train_inference/inference/eval_orb_proxy.py --ckpt "$BEST" --root "$DATA" --out "$OUT/orb_after" --pairs "$ORB_PAIRS" --gpu 0 || echo "WARNING: ORB check after failed, continuing"

"$PY" - "$OUT" <<'PYEOF' | tee "$OUT/before_after_report.txt" || echo "WARNING: report failed, summaries are still in $OUT"
import csv, sys
out = sys.argv[1]
read = lambda p: {r['group']: r for r in csv.DictReader(open(f'{out}/{p}/summary.csv'))}
pairs = (('EOW', '1_before_finetune/endofworld/metrics', '2_after_finetune/endofworld/metrics'),
         ('MF', '1_before_finetune/multifloor/metrics', '2_after_finetune/multifloor/metrics'))
for name, a, b in pairs:
    x, y = read(a), read(b); print(f'--- {name}: before -> after')
    for g in x:
        px, py, sx, sy = float(x[g]['psnr']), float(y[g]['psnr']), float(x[g]['ssim']), float(y[g]['ssim'])
        print(f'  {g:12s} PSNR {px:7.3f} -> {py:7.3f} ({py-px:+.3f})   SSIM {sx:.4f} -> {sy:.4f} ({sy-sx:+.4f})   {x[g]["n_frames"]} frames')
PYEOF
echo "Job completed at $(date)"
echo "Results: $OUT   Checkpoints: $SAVE"
