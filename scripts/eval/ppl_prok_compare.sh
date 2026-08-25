#!/usr/bin/env bash
#BSUB -J ppl_prok_compare
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_prok_compare_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_prok_compare_%J.err
#
# Final held-out perplexity for the prokaryote backbone comparison. Both arms
# are scored on the same validation cache, same length, same block size, same
# batching and the same number of batches, so the only difference is the
# checkpoint's backbone.
#
# Submit: bsub -env "all, CKPT_BISSM=/path/last.ckpt, CKPT_XF=/path/last.ckpt" \
#           < scripts/eval/ppl_prok_compare.sh
set -uo pipefail
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms; cd "$REPO"

PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
CKPT_BISSM=${CKPT_BISSM:?set CKPT_BISSM=/path/to/bissm.ckpt}
CKPT_XF=${CKPT_XF:?set CKPT_XF=/path/to/transformer.ckpt}
L=${L:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
EVAL_BATCH=${EVAL_BATCH:-4}
# LIMIT=0 means the FULL validation cache, matching configs/config.yaml:77
# (`limit_val_batches: 1.0`, "validate on full dataset"). The old default of
# 512 scored 512 of 2,347 batches -- 21.8% -- and the launcher silently
# contradicted the config it loaded. NB `LIMIT` is a bare, collidable name
# that `bsub -env all` will happily import from the submitting shell.
LIMIT=${LIMIT:-1.0}                    # 1.0 = all 2,347 batches = 76.9M validation nt
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}
ACTIVE_BLOCKS=${ACTIVE_BLOCKS:-all}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p logs logs/eval

RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
TSV="logs/eval/ppl_prok_compare_${RUN_TAG}.tsv"
printf 'arm\tval_nll_nats\tval_ppl\tbits_per_nt\tcheckpoint\n' > "$TSV"

echo "[$(date)] prokaryote perplexity comparison | host=$(hostname) | LSF=${LSB_JOBID:-local} | L=$L block=$BLOCK_SIZE limit=$LIMIT eval_batch=$EVAL_BATCH"

for ARM in bissm xf; do
  if [ "$ARM" = "bissm" ]; then
    CKPT="$CKPT_BISSM"
    ARM_ARGS=( model=small_bissm algo=bd3lm_bissm
               "model.active_blocks=$ACTIVE_BLOCKS" )
  else
    CKPT="$CKPT_XF"
    ARM_ARGS=( model=small algo=bd3lm model.dropout=0.0
               model.attn_backend=flex )
  fi
  LOG="logs/eval/ppl_prok_${ARM}_${RUN_TAG}.log"

  "$PYTHON" -u main.py mode=ppl_eval \
    "${ARM_ARGS[@]}" \
    data=carbon-prokaryote \
    data.dna_num_files="$DNA_NUM_FILES" \
    data.dna_max_rows="$DNA_MAX_ROWS" \
    model.length="$L" \
    block_size="$BLOCK_SIZE" \
    loader.eval_global_batch_size="$EVAL_BATCH" \
    loader.eval_batch_size="$EVAL_BATCH" \
    loader.num_workers=4 \
    trainer.limit_val_batches="$LIMIT" \
    eval.checkpoint_path="$CKPT" \
    eval.disable_ema=False > "$LOG" 2>&1
  RC=$?

  NLL=$(tr '\r' '\n' < "$LOG" | grep -E 'val/nll' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  if [ -n "$NLL" ]; then
    read PPL BITS <<< "$("$PYTHON" -c "import math; n=float('$NLL'); print(f'{math.exp(n):.4f} {n/math.log(2):.4f}')")"
  else
    PPL=NA; BITS=NA; NLL="fail(rc=$RC)"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$ARM" "$NLL" "$PPL" "$BITS" "$CKPT" >> "$TSV"
  echo "[$(date)] $ARM -> val/nll=${NLL} ppl=${PPL} bits/nt=${BITS}"
done

echo "=== summary ($TSV) ==="
cat "$TSV"
