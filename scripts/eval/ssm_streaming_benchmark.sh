#!/usr/bin/env bash
#BSUB -J ssm_stream
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_stream_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ssm_stream_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.ckpt}
LABEL=${LABEL:?set LABEL=ussm-ar, ussm-bd, or bissm-bd}
GENERATION_LENGTH=${GENERATION_LENGTH:-2048}
# model_length = max(chunk_size, prompt_length) in the script, so CHUNK_SIZE is
# what sets the context the model is BUILT at -- and therefore how big the
# Transformer's KV cache is allocated. Sweeping it is the only way to measure
# the cache's scaling rather than derive it from an assumed layout.
CHUNK_SIZE=${CHUNK_SIZE:-8192}
PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-64}
SEED=${SEED:-1}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/streaming/${LABEL}-${RUN_TAG}}

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" logs
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u scripts/eval/ssm_streaming_benchmark.py \
  --checkpoint "$CKPT" --output-dir "$OUTPUT_DIR" --label "$LABEL" \
  --prefix-lengths ${PREFIX_LENGTHS:-8192 65536 262144 1048576} \
  --chunk-size "$CHUNK_SIZE" \
  --prompt-length "$PROMPT_LENGTH" \
  --generation-length "$GENERATION_LENGTH" \
  --diffusion-steps "$DIFFUSION_STEPS" --seed "$SEED"
