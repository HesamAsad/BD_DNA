#!/usr/bin/env bash
#BSUB -J gb_probe
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 6:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
# GPU MODEL: unconstrained, deliberately, since 2026-09-13. This line used to
# read `...:gmodel=NVIDIAH200`, which made every GenomicBenchmarks evaluation
# queue behind the `iclr_2026` advance reservation on the farm-gpu050x hosts
# (window 8/22-9/30) while nine non-H200 hosts sat completely idle. Nothing on
# this path needs an H200: measured host peak across the historical
# gb_ft_*/gb_probe_*.out logs is 2.4-2.7 GB for the fine-tune and 0.3-6.5 GB for
# the probe, against 80 GB of device. Put the constraint back only for a job
# whose memory you have actually measured as needing it.
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_probe_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_probe_%J.err
set -euo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface/datasets
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" logs

CKPT=${CKPT:?set CKPT}
LABEL=${LABEL:?set LABEL}
TASKS=${TASKS:-all}; TASKS=${TASKS//+/,}
POOLING=${POOLING:-mean}
BATCH_SIZE=${BATCH_SIZE:-32}
EXTRA=()
# NB: do not call this WINDOW -- GNU screen exports WINDOW=<n> and
# `bsub -env all` carries it in, which silently passed --window 0.
[ -n "${GB_WINDOW:-}" ] && EXTRA+=(--window "$GB_WINDOW")
# Readout selection. READOUT=recurrent switches from pooled hidden states (C) to
# the SSM recurrent summary (A); see embed.py:recurrent_summary.
[ -n "${READOUT:-}" ]           && EXTRA+=(--readout "$READOUT")
[ -n "${RECURRENT_LAYERS:-}" ]  && EXTRA+=(--recurrent-layers "$RECURRENT_LAYERS")
[ -n "${RECURRENT_REDUCE:-}" ]  && EXTRA+=(--recurrent-reduce "$RECURRENT_REDUCE")
[ -n "${SIGMA:-}" ]             && EXTRA+=(--sigma "$SIGMA")
# Subsampling is opt-in twice over, matching finetune.sh. GB_MAX_TRAIN/GB_MAX_TEST
# arriving through `bsub -env all` are IGNORED unless GB_ALLOW_CAPS=1 says you
# meant it. Every published probe result before 2026-08-25 was silently capped at
# 20000/8000 by exactly this path -- on human_ensembl_regulatory that is 8.6% of
# the training data available.
if [ -n "${GB_MAX_TRAIN:-}${GB_MAX_TEST:-}" ]; then
  if [ "${GB_ALLOW_CAPS:-0}" = "1" ]; then
    echo "WARNING: subsampling ON -- max_train=${GB_MAX_TRAIN:-none} max_test=${GB_MAX_TEST:-none}"
    [ -n "${GB_MAX_TRAIN:-}" ] && EXTRA+=(--max-train "$GB_MAX_TRAIN")
    [ -n "${GB_MAX_TEST:-}" ] && EXTRA+=(--max-test "$GB_MAX_TEST")
  else
    echo "IGNORING inherited GB_MAX_TRAIN=${GB_MAX_TRAIN:-} GB_MAX_TEST=${GB_MAX_TEST:-} (set GB_ALLOW_CAPS=1 to mean it)"
  fi
fi

echo "[$(date)] GenomicBenchmarks probe | label=$LABEL | ckpt=$CKPT | tasks=$TASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u scripts/eval/caduceus/genomic_benchmarks.py \
  --checkpoint "$CKPT" --label "$LABEL" --tasks "$TASKS" \
  --pooling "$POOLING" --batch-size "$BATCH_SIZE" ${EXTRA[@]+"${EXTRA[@]}"}
