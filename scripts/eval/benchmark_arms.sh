#!/bin/bash
#BSUB -J benchmark_arms
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/benchmark_arms_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/benchmark_arms_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# This file was once a copy of the sizing-sweep launcher and still carried its
# knobs -- ARMS, BATCH_SIZES, LENGTHS, BLOCK_SIZE, CHECKPOINT_MODES, WARMUP --
# none of which ever reached argv, while the banner printed them as if they had.
# Only the variables below are real. Everything echoed is built from what is
# actually passed.
ARMS_JSON=${ARMS_JSON:-$REPO/results/benchmark_arms.json}
BENCH_OUT=${BENCH_OUT:-$REPO/results/benchmark_table.json}
BATCH_SIZE=${BATCH_SIZE:-4}
# 0 = score the WHOLE validation cache. The previous default of 32 batches was
# 1,048,448 of 76.9M held-out nt (1.36%) and moved uSSM-AR's val NLL by 0.0084,
# which is 1.5x the architecture difference this table exists to report.
VAL_BATCHES=${VAL_BATCHES:-0}
MC_SAMPLES=${MC_SAMPLES:-8}
WARMUP=${WARMUP:-2}
ITERS=${ITERS:-5}

mkdir -p "$(dirname "$BENCH_OUT")" logs

echo "[$(date)] benchmark_arms | arms_json=$ARMS_JSON out=$BENCH_OUT"
echo "  batch=$BATCH_SIZE val_batches=$VAL_BATCHES (0=all) mc=$MC_SAMPLES warmup=$WARMUP iters=$ITERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/eval/benchmark_arms.py \
  --arms "$ARMS_JSON" \
  --output "$BENCH_OUT" \
  --batch-size "$BATCH_SIZE" \
  --val-batches "$VAL_BATCHES" \
  --mc-samples "$MC_SAMPLES" \
  --warmup "$WARMUP" \
  --iters "$ITERS"
