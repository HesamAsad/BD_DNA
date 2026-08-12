#!/usr/bin/env bash
#BSUB -J aggregate_mavedb
#BSUB -G s10396
#BSUB -q training-normal
#BSUB -n 2
#BSUB -W 1:00
#BSUB -R "select[mem>16000]"
#BSUB -R "rusage[mem=16000]"
#BSUB -M 16000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/aggregate_mavedb_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/aggregate_mavedb_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
PREDICTION_A=${PREDICTION_A:?set PREDICTION_A}
PREDICTION_B=${PREDICTION_B:?set PREDICTION_B}
LABEL=${LABEL:?set LABEL}
RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/results/dnahnet/mavedb_aggregated/${LABEL}-${RUN_TAG}}
mkdir -p "$OUTPUT_DIR" logs
"$PYTHON" -u scripts/eval/dnahnet/aggregate_mavedb.py \
  --prediction "$PREDICTION_A" --prediction "$PREDICTION_B" \
  --output-dir "$OUTPUT_DIR" --label "$LABEL"
