#!/usr/bin/env bash
#BSUB -J smoke_prok_arms
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 16
#BSUB -W 3:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_prok_arms_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/smoke_prok_arms_%J.err
#
# Sizes both arms of the prokaryote perplexity comparison on one H200: for each
# (backbone, micro batch) it runs a few real optimizer steps in a child process,
# samples peak GPU memory, and reports throughput, so the 4-GPU runs launch with
# a micro batch that is known to fit.
#
# Submit: bsub -env "all, LENGTH=8192, MICRO_BATCHES=2 4 8" < scripts/smoke/smoke_prok_arms.sh
set -uo pipefail   # not -e: an OOM in one cell must not kill the sweep

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
MICRO_BATCHES=${MICRO_BATCHES:-"2 4 8"}
ARMS=${ARMS:-"bissm xf"}
STEPS=${STEPS:-6}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-120000}
ACTIVE_BLOCKS=${ACTIVE_BLOCKS:-all}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs logs/eval

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
TSV="logs/eval/smoke_prok_arms_${RUN_TAG}.tsv"
printf 'arm\tmicro_batch\tstatus\tpeak_mib\twall_s\tit_per_s\ttok_per_s\tval_nll\n' > "$TSV"

echo "[$(date)] prokaryote arm sizing | host=$(hostname) | LSF=${LSB_JOBID:-local} | L=$LENGTH | block=$BLOCK_SIZE | arms=$ARMS | micro=$MICRO_BATCHES | active_blocks=$ACTIVE_BLOCKS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for ARM in $ARMS; do
  for MB in $MICRO_BATCHES; do
    LOG="logs/eval/smoke_prok_${ARM}_mb${MB}_${RUN_TAG}.log"
    if [ "$ARM" = "bissm" ]; then
      ARM_ARGS=( model=small_bissm algo=bd3lm_bissm
                 "model.active_blocks=$ACTIVE_BLOCKS" )
    else
      ARM_ARGS=( model=small algo=bd3lm model.dropout=0.0
                 model.attn_backend=flex )
    fi

    MEMFILE="$(mktemp)"
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
        sleep 2
      done ) > "$MEMFILE" &
    MEMPID=$!

    START=$(date +%s)
    "$PYTHON" -u main.py \
      "${ARM_ARGS[@]}" \
      data=carbon-prokaryote \
      data.dna_num_files="$DNA_NUM_FILES" \
      data.dna_max_rows="$DNA_MAX_ROWS" \
      model.length="$LENGTH" \
      block_size="$BLOCK_SIZE" \
      loader.global_batch_size="$MB" \
      loader.eval_global_batch_size="$MB" \
      loader.batch_size="$MB" \
      loader.eval_batch_size="$MB" \
      loader.num_workers=4 \
      trainer.max_steps="$STEPS" \
      trainer.log_every_n_steps=1 \
      trainer.val_check_interval="$STEPS" \
      trainer.limit_val_batches=4 \
      trainer.num_sanity_val_steps=0 \
      training.from_pretrained=null \
      wandb=null \
      mode=train > "$LOG" 2>&1
    RC=$?
    WALL=$(( $(date +%s) - START ))

    kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
    PEAK_MIB=$(sort -n "$MEMFILE" | tail -1); rm -f "$MEMFILE"

    IT_PER_S=$(tr '\r' '\n' < "$LOG" | grep -oE '[0-9.]+it/s' | tail -1 | tr -d 'it/s')
    VAL_NLL=$(tr '\r' '\n' < "$LOG" | grep -oE "'val/nll' reached [0-9.]+" | tail -1 | grep -oE '[0-9.]+$')
    TOK_PER_S=''
    if [ -n "$IT_PER_S" ]; then
      TOK_PER_S=$("$PYTHON" -c "print(int(float('$IT_PER_S')*$MB*$LENGTH))" 2>/dev/null)
    fi
    if   grep -qiE 'out of memory' "$LOG";                then STATUS=OOM
    elif [ $RC -eq 0 ];                                   then STATUS=ok
    else                                                       STATUS="fail(rc=$RC)"; fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$ARM" "$MB" "$STATUS" "${PEAK_MIB:-NA}" "$WALL" "${IT_PER_S:-NA}" \
      "${TOK_PER_S:-NA}" "${VAL_NLL:-NA}" >> "$TSV"
    echo "[$(date)] $ARM mb=$MB -> $STATUS peak=${PEAK_MIB}MiB wall=${WALL}s it/s=${IT_PER_S:-NA} tok/s=${TOK_PER_S:-NA}"
  done
done

echo "=== summary ($TSV) ==="
cat "$TSV"
echo "[$(date)] sizing sweep exited"
