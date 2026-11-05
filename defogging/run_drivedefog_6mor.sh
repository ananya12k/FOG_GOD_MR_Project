#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 46
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --nodelist=gnode102
#SBATCH --output=drivedefog_6mor_%j.txt
#SBATCH --job-name=drivedefog_6mor
#SBATCH --signal=B:USR1@900
#SBATCH --requeue

# One script, end to end:
#   [1] dataset   symlink view of $SRC_DATA keeping the paper's MOR ladder only
#                 (300 200 150 100 75 50 m; no 30 or 20 m in train, val or test) -> $DATA
#   [2] weights   seg-loss class weights from $DATA/train/Semantic
#   [3] train     DriveDefog (FogPromptNetV2) from scratch, 40 epochs, 10k train / 4k val, drawn as
#                 an equal, evenly spaced share of every fog clip (--balanced-subset)
#   [4] test      full-reference eval of best_model.pth on 10k evenly spaced test frames, one shard per GPU
#
# Run it either way:
#   sbatch run_drivedefog_6mor.sh                 # new job on gnode102
#   bash   run_drivedefog_6mor.sh 2>&1 | tee drivedefog_6mor.log   # inside an existing allocation
# GPUs and CPUs are taken from whatever the job/allocation actually has.
# Every stage skips work that is already done, so re-running just continues.

set -Eeuo pipefail
echo "Running on $(hostname)  job=${SLURM_JOB_ID:-none}  restarts=${SLURM_RESTART_COUNT:-0}"
echo "Job started at $(date)"

# Call the env's interpreter directly: the inherited .venv shadows conda's python.
PY="${PY:-$HOME/miniconda3/envs/visia/bin/python}"
TORCHRUN="$(dirname "$PY")/torchrun"

VISIA=/scratch/Ananya_Kulkarni/VISIA
JARVIS=/scratch/Ananya_Kulkarni/JarvisIR
SRC_DATA="${SRC_DATA:-/scratch/Ananya_Kulkarni/Defog_fog_8mor}"
DATA="${DATA:-/scratch/Ananya_Kulkarni/Defog_fog_6mor}"
MIN_MOR="${MIN_MOR:-50}"
SAVE="${SAVE:-$VISIA/checkpoints/drivedefog_6mor}"
CW="${CW:-$SAVE/class_weights_sqrt.pt}"
MAX_EPOCH="${MAX_EPOCH:-40}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-10000}"
VAL_SAMPLES="${VAL_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-10000}"

NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
NUM_WORKERS=$(( NCPU / NGPU - 2 ))          # loader workers per rank; rest for main procs
(( NUM_WORKERS < 1 )) && NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MASTER_PORT="${MASTER_PORT:-$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))}"
echo "Resources: $NGPU GPU(s), $NCPU CPU(s) -> $NUM_WORKERS loader workers per GPU"
mkdir -p "$SAVE"

# Only when this script is its own sbatch job: requeue on the pre-timeout signal and
# resume from latest.pth. (Never requeue a borrowed allocation such as an interactive job.)
CHILD=0
if [[ "${SLURM_JOB_NAME:-}" == "drivedefog_6mor" ]]; then
    trap 'echo "[$(date)] time limit near -> requeue"; kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
fi
run_bg() { "$@" & CHILD=$!; wait $CHILD; }

# ── [1/4] dataset ─────────────────────────────────────────────────────────────
echo "=== [1/4] dataset -> $DATA (MOR >= $MIN_MOR m, from $SRC_DATA) ==="
[[ -f "$SRC_DATA/.build_complete" ]] || { echo "!!! $SRC_DATA is not a complete build"; exit 1; }
# Links only, so the kept clips stay exactly the ones in $SRC_DATA. Safe to re-run.
"$PY" - "$SRC_DATA" "$DATA" "$MIN_MOR" <<'EOF'
import csv, json, os, sys
src, dst, min_mor = sys.argv[1], sys.argv[2], float(sys.argv[3])

def link(target, name):
    if not os.path.lexists(name):
        os.symlink(target, name)

for split in ('train', 'val', 'test'):
    os.makedirs(f'{dst}/{split}/Haze', exist_ok=True)
    link(f'{src}/{split}/GT', f'{dst}/{split}/GT')
    link(f'{src}/{split}/Semantic', f'{dst}/{split}/Semantic')
    for clip in sorted(os.listdir(f'{src}/{split}/Haze')):
        with open(f'{src}/{split}/Haze/{clip}/meta.json') as f:
            keep = float(json.load(f)['mor']) >= min_mor
        if keep:
            link(f'{src}/{split}/Haze/{clip}', f'{dst}/{split}/Haze/{clip}')
        elif os.path.lexists(f'{dst}/{split}/Haze/{clip}'):
            sys.exit(f'!!! {dst}/{split}/Haze/{clip} is below {min_mor} m')

with open(f'{src}/manifest.csv', newline='') as f:
    rows = list(csv.DictReader(f))
with open(f'{dst}/manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(r for r in rows if float(r['mor']) >= min_mor)
EOF
for s in train val test; do
    echo "  $s: $(ls "$DATA/$s/Haze" | wc -l) fog clips over $(ls "$DATA/$s/GT" | wc -l) scenes"
done

cd "$VISIA"

# ── [2/4] class weights ──────────────────────────────────────────────────────
echo "=== [2/4] class weights -> $CW ==="
if [[ -s "$CW" ]]; then
    echo "  exists, skipping"
else
    "$PY" compute_defog_class_weights.py --sem-root "$DATA/train/Semantic" --out "$CW" --workers "$NCPU"
fi

last_epoch() {
    [[ -f "$SAVE/latest.pth" ]] || { echo 0; return; }
    "$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('epoch', 0))" "$SAVE/latest.pth"
}

# ── [3/4] training ───────────────────────────────────────────────────────────
echo "=== [3/4] training: $NGPU GPU(s), $MAX_EPOCH epochs -> $SAVE ==="
if (( $(last_epoch) >= MAX_EPOCH )); then
    echo "  already at epoch $MAX_EPOCH, skipping"
else
    run_bg "$TORCHRUN" \
        --nproc_per_node="$NGPU" \
        --nnodes="${NNODES:-1}" \
        --node_rank="${NODE_RANK:-0}" \
        --master_addr="${MASTER_ADDR:-127.0.0.1}" \
        --master_port "$MASTER_PORT" \
        student_train_inference/training/train_fogprompt_exp.py \
        --save-dir              "$SAVE"     \
        --dataset-root          "$DATA"     \
        --class-weights-path    "$CW"       \
        --resume                "$SAVE/latest.pth" \
        --max-epoch             "$MAX_EPOCH" \
        --batch-size            1           \
        --grad-accum-steps      "${GRAD_ACCUM:-1}" \
        --seq-len               7           \
        --img-size              256 256     \
        --num-workers           "$NUM_WORKERS" \
        --max-train-samples     "$TRAIN_SAMPLES" \
        --max-val-samples       "$VAL_SAMPLES" \
        --balanced-subset                   \
        --val-freq              2           \
        --lr                    2e-4        \
        --lr-min                1e-7        \
        --warmup-epochs         4           \
        --perceptual-weight     0.2         \
        --seg-loss-weight       0.03        \
        --seg-ramp-epochs       12          \
        --seg-clamp             30.0        \
        --sky-start-epoch       15          \
        --sky-temporal-ramp-epochs 20       \
        --grad-clip             1.0         \
        --sd-lr-mult            0.1         \
        --log-sd-peak                       \
        --temporal-loss-weight  0.1         \
        --distortion-alpha      4.0         \
        --distortion-loc-weight 0.1         \
        --distortion-threshold  0.08        \
        --sign-boost            3.0         \
        --fog-w-calib-weight    0.2         \
        --use-augmentation --aug-photometric --aug-noise --aug-spatial \
        --amp
    (( $(last_epoch) >= MAX_EPOCH )) || { echo "!!! training stopped before epoch $MAX_EPOCH"; exit 1; }
fi

# ── [4/4] test ───────────────────────────────────────────────────────────────
# Multi-node runs: only node 0 does the test evaluation.
if (( ${NODE_RANK:-0} != 0 )); then echo "node_rank ${NODE_RANK} done, test eval runs on node 0"; exit 0; fi
echo "=== [4/4] test-set evaluation of $SAVE/best_model.pth ==="
OUT_EVAL="$SAVE/test_eval"
mkdir -p "$OUT_EVAL"
rm -f "$OUT_EVAL"/per_clip.shard*.csv
PIDS=()
for ((g=0; g<NGPU; g++)); do
    "$PY" student_train_inference/inference/eval_edfog_test.py \
        --ckpt "$SAVE/best_model.pth" --dataset-root "$DATA" --split test \
        --out-dir "$OUT_EVAL" --seq-len 7 --gpu $g \
        --shard-index $g --num-shards "$NGPU" --max-samples "$TEST_SAMPLES" > "$OUT_EVAL/shard$g.log" 2>&1 &
    PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
"$PY" student_train_inference/inference/eval_edfog_test.py \
    --ckpt "$SAVE/best_model.pth" --dataset-root "$DATA" --out-dir "$OUT_EVAL" --merge-only

echo "Job completed at $(date)"
echo "Checkpoints: $SAVE/{best_model,latest}.pth   Test metrics: $OUT_EVAL/summary.csv"
