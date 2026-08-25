#!/bin/bash
#BSUB -J bd_step_breakdown
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 4:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bd_step_breakdown_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/bd_step_breakdown_%J.err
#
# Where the wall clock of ONE training step actually goes: phase split
# (prefill / active / head / backward / optimizer), CUDA kernel count and busy
# time, cudaLaunchKernel count and the CPU time inside it, CPU-GPU sync count,
# and the headline (wall - kernel_time)/wall.
#
# Default grid is the one the scaling table was measured on: arms
# ussm-ar,ussm,bissm,dit at L in {2048, 8192, 32768}, micro batch 2. With
# CHECKPOINT_PREFILL left at `both`, the two SSM BD arms run twice and the
# paired off/on delta is printed -- that delta is the prefill recompute,
# measured rather than modelled.
#
# Each (arm, length, mode) case runs in a fresh subprocess (--isolate, the
# default). That is mandatory for the `dit` arm: flex attention compiles with
# static shapes on the first length it sees and a second length in the same
# process dies inside Inductor's autotuner (see sizing_sweep.py's docstring).
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
# /nfs/team361 is full; a Triton cache under $HOME dies with Errno 28.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" logs results/sizing

# `bsub -env` splits its own argument on commas, so list-valued variables are
# passed with '+' as the separator and translated back here. Both spellings
# work when the script is run directly.
ARMS=${ARMS:-ussm-ar,ussm,bissm,dit}
LENGTHS=${LENGTHS:-2048,8192,32768}
ARMS=${ARMS//+/,}
LENGTHS=${LENGTHS//+/,}
BATCH_SIZE=${BATCH_SIZE:-2}
BLOCK_SIZE=${BLOCK_SIZE:-256}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-5}
TOP=${TOP:-12}
# both | on | off. `both` runs the checkpoint A/B for bissm and ussm; the
# other arms have no such flag and are always run once.
CHECKPOINT_PREFILL=${CHECKPOINT_PREFILL:-both}
LABEL=${LABEL:-bd_step_breakdown}
OUTPUT=${OUTPUT:-$REPO/results/sizing/$LABEL.json}
# Subprocess shards land here; TMPDIR on the compute node is fine and small.
export TMPDIR=${TMPDIR:-/tmp}

case "$CHECKPOINT_PREFILL" in
  both) CKPT_FLAG="" ;;
  on)   CKPT_FLAG="--checkpoint-prefill" ;;
  off)  CKPT_FLAG="--no-checkpoint-prefill" ;;
  *) echo "CHECKPOINT_PREFILL must be both|on|off, got '$CHECKPOINT_PREFILL'" >&2
     exit 2 ;;
esac

mkdir -p "$(dirname "$OUTPUT")"

echo "[$(date)] bd step breakdown | arms=$ARMS lengths=$LENGTHS batch=$BATCH_SIZE block=$BLOCK_SIZE ckpt=$CHECKPOINT_PREFILL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Cheap guard (~1 s): if the accounting helpers or the printers are broken the
# GPU hours are wasted. `--self-test-model` is the deeper CPU check (it builds
# a real tiny model per arm) but takes minutes, so it is not run here -- run it
# by hand on the head node after touching the instrumentation.
"$PYTHON" -u scripts/smoke/bd_step_breakdown.py --self-test

"$PYTHON" -u scripts/smoke/bd_step_breakdown.py \
  --arms "$ARMS" \
  --lengths "$LENGTHS" \
  --batch-size "$BATCH_SIZE" \
  --block-size "$BLOCK_SIZE" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --top "$TOP" \
  ${CKPT_FLAG:+$CKPT_FLAG} \
  --isolate \
  --json "$OUTPUT"

echo "[$(date)] done -> $OUTPUT"
