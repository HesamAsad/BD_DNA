#!/usr/bin/env bash
#BSUB -J fwd_matched
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fwd_matched_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/fwd_matched_%J.err
set -uo pipefail

# Re-measure the forward-pass figure with the Transformer arms PARAMETER-MATCHED
# to the SSM arms, and with the AR Transformer included.
#
# WHAT WAS WRONG WITH THE PUBLISHED FIGURE (results/figures/inference_forward.png,
# measured 2026-08-25):
#
#   curve            config                geometry   params        vs SSM
#   BiSSM/uSSM/uSSM-AR small_bissm|ussm    768/12    100,685,664   reference
#   Transformer-BD   small                 768/12     85,019,917   -15.6%
#   Transformer-AR   small_ar_transformer  832/13     99,772,621   NOT MEASURED
#
#   1. Transformer-BD was measured on a model 15.6% SMALLER than the SSM arms
#      it is plotted against, because a Mamba-2 layer carries more parameters
#      than an attention layer of the same width. The asymptotics on that
#      figure are unaffected -- quadratic is quadratic -- but every constant
#      favours the Transformer.
#   2. Transformer-AR was absent, and it is the most informative curve on the
#      page: the only parameter-matched Transformer, and the clean
#      architecture comparison against uSSM-AR with no block-diffusion
#      machinery on either side.
#
# ALL FIVE ARMS ARE RE-MEASURED, not just the two Transformers, so the figure
# comes from one run on one device in one state rather than being spliced.
#
# EXPECT Transformer-BD TO STOP EARLY, and for a boring reason. dit.py:773
# builds a DENSE (2L x 2L) block-diffusion mask under the sdpa backend, which
# is O(L^2) BYTES -- at 2^17 that alone is ~68 GB. The AR Transformer builds no
# mask at all (gen_mask is called only when algo.cross_attn, dit.py:749), so it
# should run the full range. That asymmetry is a property of our block-mask
# implementation, not of attention, and should not be read as one.
#
# Writes a NEW json rather than overwriting the 2026-08-25 file, so the old
# measurement survives for comparison and nothing is lost if this run fails.

REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PYTHON=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export TRITON_CACHE_DIR=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/triton
export TOKENIZERS_PARALLELISM=false USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
mkdir -p logs results/sizing results/figures

OUT=${OUT:-results/sizing/forward_pass_matched.json}
ARMS=${ARMS:-bissm,ussm,ussm-ar,dit,dit-ar}
LENGTHS=${LENGTHS:-1024,2048,4096,8192,16384,32768,65536,131072,262144,524288}

echo "[$(date)] forward-pass sweep, parameter-matched Transformer arms"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u -c "
import sys; sys.path.insert(0,'$REPO')
from scripts.smoke.sizing_sweep import ARMS
for k,v in ARMS.items(): print(f'  {k:<9} model={v[0]:<18} algo={v[1]}')"

"$PYTHON" -u scripts/eval/forward_pass_bench.py \
  --arms "$ARMS" \
  --lengths "$LENGTHS" \
  --batch 1 \
  --block-size 256 \
  --warmup 5 \
  --iters 15 \
  --output "$OUT"
rc=$?
echo "[$(date)] bench exit=$rc"

if [ "$rc" = "0" ]; then
  echo "[$(date)] regenerating the figure from $OUT"
  "$PYTHON" -u scripts/eval/inference_curves.py \
    --forward "$OUT" --outdir results/figures
  rc=$((rc + $?))
  "$PYTHON" - <<PYEOF
import json
d = json.load(open("$OUT"))
rows = d["rows"]
arms = sorted({r["arm"] for r in rows})
print()
print(f"{'arm':<9}{'measured':>9}{'oom':>5}{'failed':>8}{'max L':>10}")
print("-" * 41)
for a in arms:
    got = [r for r in rows if r["arm"] == a and r.get("tokens_per_second")]
    oom = [r for r in rows if r["arm"] == a and r.get("oom")]
    err = [r for r in rows if r["arm"] == a and r.get("error")]
    top = max((r["length"] for r in got), default=0)
    print(f"{a:<9}{len(got):>9}{len(oom):>5}{len(err):>8}{top:>10,}")
    for r in err[:1]:
        print(f"         first failure at L={r['length']:,}: {r['error'][:70]}")
PYEOF
fi
echo "[$(date)] done rc=$rc"
exit $rc
