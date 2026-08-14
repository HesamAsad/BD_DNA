#!/usr/bin/env bash
#BSUB -J deg_score
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 48:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/deg_score_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/deg_score_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/absolute/path/to/checkpoint.ckpt}
LABEL=${LABEL:?set LABEL=ussm-ar or LABEL=bissm-bd ...}
DATA=${DATA:-$REPO/data_cache/dnahnet/deg_essentiality.jsonl.gz}
BATCH_SIZE=${BATCH_SIZE:-8}
# 128, not 8: the MaveDB MC curve (0.0966/0.1242/0.1318/0.1360 at 8/32/64/128)
# puts diminishing returns near 128. AR checkpoints ignore this (exact NLL).
MC_SAMPLES=${MC_SAMPLES:-128}
MODEL_LENGTH=${MODEL_LENGTH:-8192}
LOCAL_RADIUS=${LOCAL_RADIUS:-128}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}  # batches between partial-result flushes
SEED=${SEED:-1}
MAX_GENES=${MAX_GENES:-}
SAMPLE_GENES=${SAMPLE_GENES:-}   # seeded uniform subset; prefer over MAX_GENES
SAMPLE_SEED=${SAMPLE_SEED:-0}
ORGANISM=${ORGANISM:-}        # space-separated organism_id list, optional
REVERSE_OFF=${REVERSE_OFF:-0} # 1 = ablate the in-block reverse scan
BASELINES_ONLY=${BASELINES_ONLY:-0}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/dnahnet/deg/${LABEL}-${RUN_TAG}}

# A missing PYTHONPATH export has killed a job on this cluster before.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" logs

EXTRA_ARGS=()
[[ "$REVERSE_OFF" == "1" ]] && EXTRA_ARGS+=(--reverse-off)
[[ "$BASELINES_ONLY" == "1" ]] && EXTRA_ARGS+=(--baselines-only)
[ -n "$MAX_GENES" ] && EXTRA_ARGS+=(--max-genes "$MAX_GENES")
[ -n "$SAMPLE_GENES" ] && EXTRA_ARGS+=(--sample-genes "$SAMPLE_GENES" --sample-seed "$SAMPLE_SEED")
for ORG in $ORGANISM; do EXTRA_ARGS+=(--organism "$ORG"); done

echo "[$(date)] DEG | label=$LABEL | checkpoint=$CKPT | pairs=$BATCH_SIZE | MC=$MC_SAMPLES | seed=$SEED | L=$MODEL_LENGTH | output=$OUTPUT_DIR"

if [[ "$BASELINES_ONLY" != "1" ]]; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  "$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); \
    print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); \
    sys.exit(0 if ok else 3)" || { echo 'FATAL: torch sees no GPU.'; exit 3; }
fi

"$PYTHON" -u scripts/eval/dnahnet/score_deg.py \
  --checkpoint "$CKPT" \
  --data "$DATA" \
  --output-dir "$OUTPUT_DIR" \
  --label "$LABEL" \
  --batch-size "$BATCH_SIZE" \
  --mc-samples "$MC_SAMPLES" \
  --model-length "$MODEL_LENGTH" \
  --local-radius "$LOCAL_RADIUS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$SEED" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "[$(date)] DEG scoring exited"
