#!/bin/bash
#BSUB -J gb_smoke_probe
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 4
#BSUB -W 2:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>64000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_smoke_probe_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_smoke_probe_%J.err
set -uo pipefail

# Companion to smoke_all_arms.sh, for the PROBE path rather than the fine-tune
# path: scripts/eval/caduceus/genomic_benchmarks.py -> embed.py.
#
# embed.py had two defects of its own, fixed 2026-08-26:
#   * it called `layers[i].scan_active(...)` unconditionally, which is
#     BIDIRECTIONAL, so a unidirectional (AR) checkpoint was probed with a
#     reverse scan it never trained with -- silently, because UnidirectionalSSM
#     subclasses BidirectionalSSM and its layers do expose scan_active.
#   * it raised TypeError on any DiT, so neither Transformer arm could be
#     probed at all. (It at least failed LOUDLY, unlike the fine-tune path.)
# Only the branch SELECTION could be checked without a GPU -- the DiT tap needs
# CUDA because flash_attn's rotary is a Triton kernel. This job is that check.
#
# Correctness smoke, not a benchmark: one small task. The accuracies mean
# nothing; what matters is that each arm produces embeddings at all.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
mkdir -p logs results/caduceus/smoke_probe

P=outputs/carbon-prokaryote
ARMS=(
  "bissm_bd|$P/2026.08.06/dna-bd3lm-bi-mamba2-lr1e-3-b20.95-wd0.1-L8192-99658/checkpoints/best.ckpt"
  "ussm_ar|$P/2026.08.06/dna-ar-uni-mamba2-lr3e-4-b20.95-wd0.1-L8192-99653/checkpoints/best.ckpt"
  "xf_bd|$P/xf-bd-tuned-20260809-v1/full/checkpoints/best.ckpt"
  "xf_ar|$P/transformer-ar-20260808-v1/full/checkpoints/best.ckpt"
)

rc_total=0
for entry in "${ARMS[@]}"; do
  label="${entry%%|*}"; ckpt="${entry#*|}"
  echo; echo "=============================================================="
  echo "[$(date)] probe $label"
  if [ ! -f "$ckpt" ]; then
    echo "  MISSING $ckpt"; rc_total=$((rc_total + 1)); continue
  fi
  # Caps go on the command line here. GB_MAX_TRAIN/GB_MAX_TEST are read by the
  # genomic_benchmarks.sh WRAPPER, not by the python; setting them as env vars
  # around a direct python call does nothing at all -- the same shape of bug as
  # the DATA_TRAIN that two training launchers accepted and ignored.
  "$PYTHON" -u scripts/eval/caduceus/genomic_benchmarks.py \
    --checkpoint "$ckpt" \
    --label "probe_$label" \
    --output-dir results/caduceus/smoke_probe \
    --tasks dummy_mouse_enhancers_ensembl \
    --pooling mean \
    --batch-size 8 \
    --max-train 256 \
    --max-test 128
  rc=$?; echo "  exit=$rc"; rc_total=$((rc_total + rc))
done

echo; echo "=============================================================="
echo "[$(date)] summed exit codes: $rc_total  (0 = every arm embedded)"
exit $rc_total
