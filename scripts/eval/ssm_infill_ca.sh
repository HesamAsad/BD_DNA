#!/usr/bin/env bash
#BSUB -J ssm_infill_ca
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_infill_ca_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_infill_ca_%J.err
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"
CKPT=${CKPT:?set CKPT}
LABEL=${LABEL:?set LABEL}
NUM_BATCHES=${NUM_BATCHES:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
MC_SAMPLES=${MC_SAMPLES:-16}
# `bsub -env` splits its own argument on commas, so pass lists with '+'
# (e.g. RIGHT_BLOCKS=1+2+4+all). Commas still work when run directly.
RIGHT_NT=${RIGHT_NT:-all}
RIGHT_NT=${RIGHT_NT//+/,}
SEED=${SEED:-1}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/infill_ca/${LABEL}-${RUN_TAG}}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$OUTPUT_DIR"
"$PYTHON" -u scripts/eval/ssm_infill_ca.py \
  --checkpoint "$CKPT" --output-dir "$OUTPUT_DIR" --label "$LABEL" \
  --num-batches "$NUM_BATCHES" --batch-size "$BATCH_SIZE" \
  --mc-samples "$MC_SAMPLES" --seed "$SEED" \
  --right-nt "$RIGHT_NT" ${MISMATCH:+--mismatch-control}
