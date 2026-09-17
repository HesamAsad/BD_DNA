#!/usr/bin/env bash
#BSUB -J mavedb_score
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
# GPU MODEL: unconstrained since 2026-09-13. Pinning H200 queued every MaveDB
# job behind the `iclr_2026` advance reservation (active to 9/30) while A100,
# L40S and H100 hosts sat idle. Nothing here needs it: peak HOST memory across
# this harness's own logs is 2.59-3.32 GB against a 128 GB request, and Score I
# (--score-mode infill_codon) is a single deterministic forward pass per
# variant. Re-pin only for a job whose memory you have actually measured.
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/mavedb_score_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/mavedb_score_%J.err
set -euo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.ckpt}
LABEL=${LABEL:?set LABEL=bissm or LABEL=transformer}
DATA=${DATA:-$REPO/data_cache/dnahnet/mavedb_ecoli_k12_21250.jsonl.gz}
BATCH_SIZE=${BATCH_SIZE:-16}
# 128, not 8. The old default was measured wrong and left in place: the MC curve
# on BiSSM-BD is 0.0966 (8) -> 0.1242 (32) -> 0.1318 (64) -> 0.1360 (128), so the
# 8-sample default understated every block-diffusion arm by ~0.04 -- about a
# third of its signal -- and the first published comparison was computed at it.
# Same failure shape as GB_MAX_TRAIN=20000 on the Caduceus harness. Score I
# (--score-mode infill_codon) needs no Monte Carlo at all and is the better
# answer where it applies.
MC_SAMPLES=${MC_SAMPLES:-128}
MODEL_LENGTH=${MODEL_LENGTH:-256}
SEED=${SEED:-1}
MAX_VARIANTS=${MAX_VARIANTS:-}
SCORE_MODE=${SCORE_MODE:-nelbo}   # nelbo | pll | infill_nt | infill_codon | infill_nsyn | state_displacement
GENOMIC_PREFIX=${GENOMIC_PREFIX:-}   # path to the urn->prefix JSON
REVERSE_OFF=${REVERSE_OFF:-0}   # 1 = ablate the in-block reverse scan
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/dnahnet/mavedb/${LABEL}-${RUN_TAG}}

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR" logs

EXTRA_ARGS=()
[[ "$REVERSE_OFF" == "1" ]] && EXTRA_ARGS+=(--reverse-off)
EXTRA_ARGS+=(--score-mode "$SCORE_MODE")
[ -n "${INFILL_PAD_SIDE:-}" ] && EXTRA_ARGS+=(--infill-pad-side "$INFILL_PAD_SIDE")
[ -n "$GENOMIC_PREFIX" ] && EXTRA_ARGS+=(--genomic-prefix "$GENOMIC_PREFIX")
[ -n "$MAX_VARIANTS" ] && EXTRA_ARGS+=(--max-variants "$MAX_VARIANTS")

echo "[$(date)] MaveDB | label=$LABEL | checkpoint=$CKPT | pairs=$BATCH_SIZE | MC=$MC_SAMPLES | seed=$SEED | L=$MODEL_LENGTH | output=$OUTPUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -u scripts/eval/dnahnet/score_mavedb.py \
  --checkpoint "$CKPT" \
  --data "$DATA" \
  --output-dir "$OUTPUT_DIR" \
  --label "$LABEL" \
  --batch-size "$BATCH_SIZE" \
  --mc-samples "$MC_SAMPLES" \
  --model-length "$MODEL_LENGTH" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo "[$(date)] MaveDB scoring exited"
