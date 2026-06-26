#!/usr/bin/env bash
#BSUB -J sweep_longctx_nll
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/sweep_longctx_nll_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/sweep_longctx_nll_%J.err
#
# Test-time LENGTH-EXTRAPOLATION sweep for the dual-stream BD3-LM.
# ---------------------------------------------------------------------------
# Loads the L=98,496 dual checkpoint into a model REBUILT at increasing
# model.length (1x .. 6x) and runs the validated teacher-forced NLL path
# (mode=ppl_eval) on held-out prokaryote at each length, on a SINGLE H200.
#
# Why this is a clean extrapolation probe (see models/dit_dual.py):
#   - fine self-attention is windowed-local (+/-8 blocks = 144 nt) -> length
#     INVARIANT; it never attends beyond the window.
#   - fine->coarse cross-attention is block-causal -> no positional break.
#   - the ONLY length-sensitive component is the coarse stream's ROTARY
#     (Rotary is parameter-free, computed at runtime from seq_len), so this
#     sweep isolates how far the coarse rotary stretches (16,416 trained
#     coarse positions -> ~98,496 at 6x) plus numerical/memory feasibility.
#
# Weights are length-independent (rotary is a dim-based buffer; the flex
# BlockMasks are rebuilt at __init__ and are NOT in the state_dict), so
# load_from_checkpoint(config=<L-overridden>, strict=False) rebuilds the masks
# at the new length and loads the 98k weights cleanly.
#
# Each length runs in its own process (clean OOM isolation + avoids the flex
# dynamo dynamic-shape error between lengths). A length that OOMs is recorded
# and the sweep continues to the next.
#
# Tunables (env): CKPT, LENGTHS, LIMIT, PYTHON, FLEX_MODE.
set -uo pipefail   # NOT -e: we want to survive a per-length OOM and continue.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT=${CKPT:-$REPO/outputs/carbon-prokaryote/2026.06.19/030312/checkpoints/7-52500.ckpt}
# 1x..6x of 98496. Every entry is divisible by block_size=18 AND k_coarse=6.
LENGTHS=${LENGTHS:-"98496 196992 295488 393984 492480 590976"}
BLOCK_SIZE=${BLOCK_SIZE:-18}
LIMIT=${LIMIT:-16}          # val batches (eval_batch_size=1 -> sequences) per length
DNA_NUM_FILES=${DNA_NUM_FILES:-1}

export BD3LM_COMPILE_MASK=1                                # block-wise mask build (required at long L)
# Cap datasets.map() workers for any per-length re-wrap. The node exposes 128
# cores; multi-worker grouping of million-element chunk tensors DEADLOCKS at long
# L (fork + huge Python-list slicing) — observed hanging at "Grouping 0%" for
# hours at BOTH num_proc=128 AND 8. num_proc=1 (no fork, incremental Arrow write)
# is deadlock-proof. Caches are normally pre-built (scripts/eval/pregen_longctx_
# caches.py at num_proc=1); this only bites if a length's cache is missing.
export BD3LM_DATA_NUM_PROC=${BD3LM_DATA_NUM_PROC:-1}
# 'default' (not max-autotune): avoids the ~21-min autotune compile and its
# compile-time scratch allocation; we only need a correct forward here.
export BD3LM_FLEX_COMPILE_MODE=${FLEX_MODE:-default}
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
SUMMARY="logs/eval/sweep_longctx_nll_${RUN_TAG}.tsv"
echo -e "length\tmultiple\tstatus\tval_nll\tval_ppl\tpeak_gib\twall_s" > "$SUMMARY"

echo "[`date`] DUAL long-ctx NLL sweep | host=$(hostname) | LSF=${LSB_JOBID:-local} | ckpt=$CKPT"
echo "[`date`] lengths: $LENGTHS | limit_val_batches=$LIMIT | flex_mode=$BD3LM_FLEX_COMPILE_MODE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); sys.exit(0 if ok else 3)" \
  || { echo 'FATAL: torch sees no GPU.'; exit 3; }
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 4; }

L1=98496
for L in $LENGTHS; do
  MULT=$(awk "BEGIN{printf \"%.2f\", $L/$L1}")
  LOG="logs/eval/sweep_longctx_nll_${RUN_TAG}_L${L}.log"
  echo "======================================================================"
  echo "[`date`] length=$L (${MULT}x) -> $LOG"

  # Sample GPU memory in the background; report the peak for this length.
  MEMFILE="$(mktemp)"
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
      sleep 3
    done ) > "$MEMFILE" &
  MEMPID=$!

  START=$(date +%s)
  "$PYTHON" -u main.py \
      mode=ppl_eval \
      model=small_dual \
      algo=bd3lm \
      algo.backbone=dit_dual \
      data=carbon-prokaryote \
      data.dna_num_files=$DNA_NUM_FILES \
      model.length=$L \
      block_size=$BLOCK_SIZE \
      model.attn_backend=flex \
      loader.eval_global_batch_size=1 \
      loader.eval_batch_size=1 \
      trainer.limit_val_batches=$LIMIT \
      eval.checkpoint_path="$CKPT" \
      eval.disable_ema=False \
      > "$LOG" 2>&1
      # NB: do NOT override wandb here — main.py sets `config.wandb = None`
      # itself in ppl_eval mode (main.py:236), which routes _ppl_eval to the
      # CSVLogger. Passing `~wandb` deletes the key and crashes that line.
  RC=$?
  END=$(date +%s)
  WALL=$((END - START))

  kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
  PEAK_MIB=$(sort -n "$MEMFILE" | tail -1); rm -f "$MEMFILE"
  PEAK_GIB=$(awk "BEGIN{printf \"%.1f\", ${PEAK_MIB:-0}/1024}")

  # Lightning prints the validate metrics as a box table; grab the float on the
  # val/nll and val/ppl rows.
  VAL_NLL=$(grep -E 'val/nll' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  VAL_PPL=$(grep -E 'val/ppl' "$LOG" | grep -oE '[0-9]+\.[0-9]+' | tail -1)

  if [ $RC -eq 0 ] && [ -n "$VAL_NLL" ]; then
    STATUS=ok
  elif grep -qiE 'out of memory|CUDA error: out of memory' "$LOG"; then
    STATUS=OOM
  else
    STATUS="fail(rc=$RC)"
  fi

  echo -e "${L}\t${MULT}x\t${STATUS}\t${VAL_NLL:-NA}\t${VAL_PPL:-NA}\t${PEAK_GIB}\t${WALL}" | tee -a "$SUMMARY"

  # Stop early once we hit the single-GPU wall (the chosen scope: "up to
  # feasible on 1xH200"). Comment out to force every length to be attempted.
  if [ "$STATUS" = "OOM" ]; then
    echo "[`date`] OOM at length=$L -> single-H200 ceiling reached; stopping sweep."
    break
  fi
done

echo "======================================================================"
echo "[`date`] sweep complete. Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
