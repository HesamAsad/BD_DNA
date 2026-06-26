#!/usr/bin/env bash
#BSUB -J dump_longctx_io
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/dump_longctx_io_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/dump_longctx_io_%J.err
#
# Dump real input / tokens / decoded predictions of the dual BD3-LM forward pass
# at a SHORT (control) and a LONG (~1M) context, so the long-context output can
# be manually verified as valid (not just trusted via NLL). Writes, per length,
# report.txt / windows.txt / decoded_seq0.txt / arrays.npz under IO_DUMP_DIR.
set -uo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:-$REPO/outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt}
# 1x control (in-distribution) + 10x ~= 1M (the long-context validity check).
LENGTHS=${LENGTHS:-"98496 984960"}
BLOCK_SIZE=${BLOCK_SIZE:-18}

export BD3LM_COMPILE_MASK=1
export BD3LM_FLEX_COMPILE_MODE=${FLEX_MODE:-default}
export BD3LM_DATA_NUM_PROC=${BD3LM_DATA_NUM_PROC:-1}   # avoid the 128-way grouping deadlock
export IO_DUMP_FRACS=${IO_DUMP_FRACS:-0.15,0.5}
export IO_DUMP_NSEQ=${IO_DUMP_NSEQ:-1}                 # 1 long seq (batch>1 OOMs at ~1M)
export IO_DUMP_BINS=${IO_DUMP_BINS:-20}
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs logs logs/eval

RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
echo "[`date`] IO dump | host=$(hostname) | LSF=${LSB_JOBID:-local} | lengths=$LENGTHS | fracs=$IO_DUMP_FRACS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 4; }

for L in $LENGTHS; do
  OUTDIR="$REPO/logs/eval/io_dump_L${L}_${RUN_TAG}"
  LOG="logs/eval/dump_longctx_io_${RUN_TAG}_L${L}.log"
  echo "======================================================================"
  echo "[`date`] length=$L -> dir $OUTDIR"
  IO_DUMP_DIR="$OUTDIR" "$PYTHON" -u main.py \
      mode=io_dump \
      model=small_dual \
      algo=bd3lm \
      algo.backbone=dit_dual \
      data=carbon-prokaryote \
      data.dna_num_files=1 \
      model.length=$L \
      block_size=$BLOCK_SIZE \
      model.attn_backend=flex \
      loader.eval_global_batch_size=1 \
      loader.eval_batch_size=1 \
      eval.checkpoint_path="$CKPT" \
      eval.disable_ema=False \
      > "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "[`date`] length=$L OK. report.txt:"; sed -n '1,40p' "$OUTDIR/report.txt" 2>/dev/null
  else
    echo "[`date`] length=$L FAILED rc=$RC (see $LOG)"; tail -15 "$LOG"
  fi
done
echo "[`date`] IO dump done."
