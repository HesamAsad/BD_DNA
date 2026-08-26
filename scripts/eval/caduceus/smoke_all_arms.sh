#!/bin/bash
#BSUB -J gb_smoke_arms
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
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_smoke_arms_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gb_smoke_arms_%J.err
set -uo pipefail

# Does the fine-tuning harness actually run on all FOUR backbone kinds?
#
# Until 2026-08-26 `Classifier.forward` called
#   h = b.layers[i].scan_active(h, left.states[i], right.states[i])
# unconditionally. That is SSM-only (a DiT has `blocks`, no `scan_active`) AND
# bidirectional (an AR checkpoint got a reverse scan it never trained with).
# Every GenomicBenchmarks number we hold is `backbone: bissm` for that reason.
#
# This is a CORRECTNESS smoke, not a benchmark: one small task, one epoch, a
# few hundred examples. The accuracies it prints mean nothing. What it proves
# is that each arm loads, taps its blocks, produces a pooled representation,
# takes a gradient, and scores -- and that the readout it gets matches the way
# it was trained (causal for AR, full for BD).
#
# Prokaryote checkpoints, because the hg38 arms are still queued behind the
# corpus build. The backbone dispatch does not depend on the corpus.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
mkdir -p logs results/caduceus/smoke_arms

P=outputs/carbon-prokaryote
ARMS=(
  "bissm_bd|$P/2026.08.06/dna-bd3lm-bi-mamba2-lr1e-3-b20.95-wd0.1-L8192-99658/checkpoints/best.ckpt"
  "ussm_ar|$P/2026.08.06/dna-ar-uni-mamba2-lr3e-4-b20.95-wd0.1-L8192-99653/checkpoints/best.ckpt"
  "xf_bd|$P/xf-bd-tuned-20260809-v1/full/checkpoints/best.ckpt"
  "xf_ar|$P/transformer-ar-20260808-v1/full/checkpoints/best.ckpt"
)

rc_total=0
for entry in "${ARMS[@]}"; do
  label="${entry%%|*}"
  ckpt="${entry#*|}"
  echo
  echo "=============================================================="
  echo "[$(date)] $label"
  echo "  $ckpt"
  if [ ! -f "$ckpt" ]; then
    echo "  MISSING -- skipping (does not count as a pass)"
    rc_total=$((rc_total + 1))
    continue
  fi
  # Caps are deliberate here and declared, per the GB_ALLOW_CAPS guard: this is
  # a smoke, and a full task would take an hour per arm for no extra evidence.
  GB_ALLOW_CAPS=1 "$PYTHON" -u scripts/eval/caduceus/finetune.py \
    --checkpoint "$ckpt" \
    --label "smoke_$label" \
    --output-dir results/caduceus/smoke_arms \
    --tasks dummy_mouse_enhancers_ensembl \
    --preset v2 \
    --epochs 1 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --max-train 256 \
    --max-test 128
  rc=$?
  echo "  exit=$rc"
  rc_total=$((rc_total + rc))
done

echo
echo "=============================================================="
echo "[$(date)] summed exit codes: $rc_total  (0 = every arm ran)"
$PYTHON - <<'PYEOF'
import glob, json, os
rows = []
for path in sorted(glob.glob("results/caduceus/smoke_arms/*.json")):
    try:
        payload = json.load(open(path))
    except Exception as exc:
        rows.append((os.path.basename(path), f"unreadable: {exc}"))
        continue
    label = payload.get("label", os.path.basename(path))
    tasks = payload.get("tasks", payload.get("results", {}))
    acc = None
    if isinstance(tasks, dict):
        for value in tasks.values():
            if isinstance(value, dict) and "test_accuracy" in value:
                acc = value["test_accuracy"]
                break
    rows.append((label, f"test_acc={acc}"))
print("\nsmoke results")
for label, detail in rows:
    print(f"  {label:<24} {detail}")
PYEOF
echo "[$(date)] done"
exit $rc_total
