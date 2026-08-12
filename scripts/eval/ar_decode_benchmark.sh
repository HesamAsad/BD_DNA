#!/usr/bin/env bash
#BSUB -J ar_decode
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 8:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ar_decode_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ar_decode_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT}
LABEL=${LABEL:?set LABEL}
PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
GENERATION_LENGTH=${GENERATION_LENGTH:-512}
SEED=${SEED:-1}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/ar_decode/${LABEL}-${RUN_TAG}}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" logs
"$PYTHON" -u scripts/eval/ar_decode_benchmark.py \
  --checkpoint "$CKPT" --output-dir "$OUTPUT_DIR" --label "$LABEL" \
  --prompt-length "$PROMPT_LENGTH" --generation-length "$GENERATION_LENGTH" \
  --seed "$SEED"
