#!/usr/bin/env bash
#BSUB -J ppl_ema_ablation
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>96000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=96000]"
#BSUB -M 96000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_ema_ablation_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/ppl_ema_ablation_%J.err
#
# Quantifies how much of the published Transformer BD3-LM lead is EMA smoothing
# rather than the backbone.
#
# The recorded comparison scored Transformer 96604 on EMA-smoothed weights (its
# checkpoint carries an `ema` key; diffusion.py:531-539 swaps EMA in for
# validation) against BiSSM/uSSM checkpoints that carry no EMA at all and were
# therefore scored raw. This re-scores the SAME Transformer checkpoint both
# ways on the SAME 512 validation batches. The delta is the EMA advantage.
set -uo pipefail

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
cd "$REPO"

CKPT=${CKPT:-outputs/carbon-prokaryote/2026.08.03/162751/checkpoints/0-8000.ckpt}
# Hydra chdir's into its run directory, so a relative checkpoint path would be
# resolved against that instead of the repo. Always hand main.py an absolute one.
CKPT=$(readlink -f "$CKPT")
L=${L:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-256}
EVAL_BATCH=${EVAL_BATCH:-4}
# LIMIT=0 means the FULL validation cache, matching configs/config.yaml:77
# (`limit_val_batches: 1.0`, "validate on full dataset"). The old default of
# 512 scored 512 of 2,347 batches -- 21.8% -- and the launcher silently
# contradicted the config it loaded. NB `LIMIT` is a bare, collidable name
# that `bsub -env all` will happily import from the submitting shell.
LIMIT=${LIMIT:-1.0}
DNA_NUM_FILES=${DNA_NUM_FILES:-1}
DNA_MAX_ROWS=${DNA_MAX_ROWS:-400000}

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
# Triton ignores XDG_CACHE_HOME and defaults under $HOME, which is on the
# full /nfs/team361 volume. Keep compiled kernels on scratch.
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=0 TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/eval

RUN_TAG=${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}
TSV="logs/eval/ppl_ema_ablation_${RUN_TAG}.tsv"
printf 'arm\tema_applied\tval_nll_nats\tval_ppl\tbits_per_nt\tcheckpoint\n' > "$TSV"

echo "checkpoint: $CKPT"
"$PYTHON" -c "
import torch,sys
c=torch.load('$CKPT',map_location='cpu',weights_only=False)
print('carries ema state:', 'ema' in c)
sys.exit(0 if 'ema' in c else 4)" || {
  echo "FATAL: checkpoint has no EMA state; the ablation is meaningless."; exit 4; }

for DISABLE in False True; do
  APPLIED=$([[ "$DISABLE" == "False" ]] && echo yes || echo no)
  LOG="logs/eval/ppl_ema_${APPLIED}_${RUN_TAG}.log"
  echo "=== scoring with EMA applied=${APPLIED} ==="
  "$PYTHON" -u main.py mode=ppl_eval \
    model=small algo=bd3lm model.dropout=0.0 \
    model.attn_backend=flex \
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
    "eval.disable_ema=$DISABLE" > "$LOG" 2>&1
  NLL=$(tr '\r' '\n' < "$LOG" | grep -E 'val/nll' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  if [[ -n "$NLL" ]]; then
    read PPL BITS <<< "$("$PYTHON" -c "import math; n=float('$NLL'); print(f'{math.exp(n):.4f} {n/math.log(2):.4f}')")"
    printf 'xf_bd\t%s\t%s\t%s\t%s\t%s\n' "$APPLIED" "$NLL" "$PPL" "$BITS" "$CKPT" >> "$TSV"
  else
    printf 'xf_bd\t%s\tFAILED\t-\t-\t%s\n' "$APPLIED" "$CKPT" >> "$TSV"
    echo "WARNING: no val/nll parsed from $LOG"
  fi
done

echo
column -t "$TSV"
if grep -q 'FAILED' "$TSV"; then
  echo "FATAL: at least one arm produced no val/nll -- see logs/eval/ppl_ema_*_${RUN_TAG}.log"
  exit 5
fi
"$PYTHON" - "$TSV" <<'EOF'
import sys, csv
rows = {r['ema_applied']: r['val_nll_nats'] for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')}
try:
    d = float(rows['no']) - float(rows['yes'])
    print(f"\nEMA advantage on the SAME weights: {d:+.5f} nats "
          f"(raw {rows['no']} vs EMA {rows['yes']})")
    print("For scale, the published Transformer-vs-BiSSM margin was 0.00148 nats.")
    ref = 1.2457695007324219   # logs/eval/ppl_prok_xf_98003.log, same ckpt+settings
    got = float(rows['yes'])
    if abs(got - ref) > 5e-3:
        print(f"\nINVALID: the EMA arm scored {got:.5f} but this checkpoint and "
              f"these settings are known to give {ref:.5f}. The harness is not "
              f"reproducing the reference, so the delta above is meaningless.")
        sys.exit(6)
    print(f"reference check OK: EMA arm {got:.5f} vs known {ref:.5f}")
except (KeyError, ValueError):
    print("\nCould not compute the delta; see the TSV above.")
EOF
