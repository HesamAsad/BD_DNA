#!/bin/bash
#BSUB -J flop_ktrace
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flop_ktrace_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/flop_ktrace_%J.err
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

ARM=${ARM:-bissm}
BATCH=${BATCH:-1}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
WARMUP=${WARMUP:-2}
TOP=${TOP:-30}
LABEL=${LABEL:-flop_kernel_trace_${ARM}_L${LENGTH}}
# STACK=off drops with_stack; the trace shrinks by roughly an order of
# magnitude but the folded-stack export goes away with it.
STACK=${STACK:-on}
STACK_FLAG=$([ "$STACK" = off ] && echo --no-stack || echo --stack)
# default | on | off. `measured_flops_sweep` turns prefill checkpointing ON for
# bissm, which runs the prefill forward twice; `off` isolates that term.
CHECKPOINT_PREFILL=${CHECKPOINT_PREFILL:-default}

echo "[$(date)] flop kernel trace | arm=$ARM batch=$BATCH len=$LENGTH block=$BLOCK_SIZE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PYTHON" -u scripts/smoke/flop_kernel_trace.py \
  --arm "$ARM" \
  --batch "$BATCH" \
  --length "$LENGTH" \
  --block-size "$BLOCK_SIZE" \
  --warmup "$WARMUP" \
  --top "$TOP" \
  --checkpoint-prefill "$CHECKPOINT_PREFILL" \
  "$STACK_FLAG" \
  --label "$LABEL" \
  --output-dir "$REPO/results/sizing"

echo "[$(date)] done"
