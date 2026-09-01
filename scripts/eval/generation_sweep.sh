#!/usr/bin/env bash
#BSUB -J gen_sweep
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gen_sweep_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/gen_sweep_%J.err
set -uo pipefail

# END-TO-END GENERATION COST: wall-clock and memory to produce N tokens, with
# the denoising-step count T swept for the block-diffusion arms.
#
# WHY THIS IS NEEDED, stated plainly. Neither existing computational figure
# measures generation cost honestly:
#
#   inference_forward.png  times ONE _loss forward. For a BD arm that is a
#                          single noise-and-denoise pass, so it charges BD for
#                          one denoising step when generation needs T of them.
#                          It therefore UNDERSTATES BD's generation cost.
#   inference_state.png    is real generation but reports MEMORY only, and only
#                          for two arms.
#
# The arithmetic the figures leave out: to emit L tokens,
#     AR  does  L                forward passes, 1 token each
#     BD  does  (L/b) * T        forward passes, b tokens each in parallel
# so BD performs L*T token-updates against AR's L. Parallelism within the block
# claws back most of that in wall-clock, but not all -- and how much is exactly
# what T controls. Sweeping T also traces the quality/cost knob BD has and AR
# does not: fewer steps is faster and worse, and nothing in our figures has
# shown where that curve sits.
#
# T is meaningless for the AR arms (one token per step, no refinement), so they
# are measured once and drawn as a horizontal reference across the T axis.
#
# MEASURED RESULT (2026-09-01, 69 runs). The framing above anticipated BD being
# the expensive one. It is not, at any T this sweep reached. The decisive count
# is FORWARD PASSES, not token-updates: AR does L of them, BD does (L/b)*T. With
# b=256, BD does FEWER forwards than AR whenever T < 256, and the sweep tops out
# at T=64. So BD leads almost everywhere, and the true break-even sits at T = b =
# 256, outside this grid.
#
#   fastest AR is Transformer-AR, flat at ~125 tok/s for every N (sliding KV
#   cache => constant per-token cost). uSSM-AR is roughly half that, 45-64 tok/s,
#   rising with N as its fixed per-step overhead amortises -- the launch-bound
#   signature, not a bandwidth limit.
#
#   BD stays ahead of the fastest AR up to:
#     N=1,024   uSSM-BD T<=32   BiSSM-BD T<=16   Transformer-BD T<=64
#     N=4,096   uSSM-BD T<=64   BiSSM-BD T<=32   Transformer-BD T<=64
#     N=16,384  all three arms T<=64 (never crossed in range)
#
#   Memory is the cleaner architectural story: the SSM arms are FLAT in N at
#   0.72-0.73 GB with a 4.7 MiB cache, while Transformer-BD grows 1.19->1.37 GB
#   and carries a 468 MiB cache -- 100x the SSM state.
#
# CAVEAT: T=1 runs first in each (arm, N) group and absorbs warm-up, so several
# T=1 points read slower than their T=2 neighbours (uSSM-BD at N=1,024: 137.7 vs
# 174.0 tok/s). Treat T=1 as contaminated; the T>=2 curve is the trustworthy one.
#
# Outputs one JSON per (arm, N, T) under results/generation/, which
# scripts/eval/generation_curves.py turns into the figure.

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
mkdir -p logs results/generation

# OUTPUT PATHS. Both harnesses write "<output-dir>/summary.json" unconditionally
# -- --label goes INSIDE the payload, it does not name the file. A shared
# --output-dir therefore means all 69 runs overwrite one file and the sweep
# still exits 0. Each run gets its own directory; generation_curves.py globs
# recursively and reads the label from the payload.
R=$REPO/outputs/hg38-caduceus
OUT=results/generation
LENGTHS=${LENGTHS:-"1024 4096 16384"}
# T values. 1 is the degenerate one-shot case (worst quality, cheapest); 64 is
# the harness default. Block size is 256, so T=256 would denoise one token per
# step and recover an AR-like factorisation at AR-like cost.
TSTEPS=${TSTEPS:-"1 2 4 8 16 32 64"}
PROMPT=${PROMPT:-1024}

# WHICH HARNESS HANDLES WHICH ARM. ar_decode_benchmark.py walks
# backbone.blocks and block.kv_cache, so it is TRANSFORMER-ONLY -- it would
# raise on an SSM, which has neither. ssm_streaming_benchmark.py dispatches
# internally on config.algo.backbone and model.parameterization, so it covers
# the SSM AR arm (generate_ar) and every BD arm including the DiT
# (generate_diffusion, whose cache accounting explicitly supports both).
#
# arm | run dir
SSM_AR="ussm_ar:hg_ussm_ar"            # ssm_streaming -> generate_ar
XF_AR="xf_ar:hg_xf_ar"                 # ar_decode -> generate_transformer
BD_ARMS="ussm_bd:hg_ussm_bd bissm_bd:hg_bissm_bd xf_bd:hg_xf_bd"

echo "[$(date)] generation sweep: N in {$LENGTHS}, T in {$TSTEPS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rc=0
# ---- AR arms: one measurement per N (T does not apply) ----------------------
arm="${SSM_AR%%:*}"; dir="${SSM_AR##*:}"; ck="$R/$dir/checkpoints/best.ckpt"
if [ -f "$ck" ]; then
  for N in $LENGTHS; do
    echo; echo "[$(date)] AR $arm  N=$N  (ssm_streaming -> generate_ar)"
    "$PYTHON" -u scripts/eval/ssm_streaming_benchmark.py \
      --checkpoint "$ck" --output-dir "$OUT/gen_${arm}_N${N}" \
      --label "gen_${arm}_N${N}" \
      --prompt-length "$PROMPT" --generation-length "$N" \
      --generation-batch-size 1 --prefix-lengths "$N" || rc=$((rc+1))
  done
else echo "  MISSING $ck"; rc=$((rc+1)); fi

arm="${XF_AR%%:*}"; dir="${XF_AR##*:}"; ck="$R/$dir/checkpoints/best.ckpt"
if [ -f "$ck" ]; then
  for N in $LENGTHS; do
    echo; echo "[$(date)] AR $arm  N=$N  (ar_decode -> generate_transformer)"
    "$PYTHON" -u scripts/eval/ar_decode_benchmark.py \
      --checkpoint "$ck" --output-dir "$OUT/gen_${arm}_N${N}" \
      --label "gen_${arm}_N${N}" \
      --prompt-length "$PROMPT" --generation-length "$N" || rc=$((rc+1))
  done
else echo "  MISSING $ck"; rc=$((rc+1)); fi

# ---- BD arms: full (N, T) grid ---------------------------------------------
for spec in $BD_ARMS; do
  arm="${spec%%:*}"; dir="${spec##*:}"
  ck="$R/$dir/checkpoints/best.ckpt"
  [ -f "$ck" ] || { echo "  MISSING $ck"; rc=$((rc+1)); continue; }
  for N in $LENGTHS; do
    for T in $TSTEPS; do
      echo; echo "[$(date)] BD $arm  N=$N  T=$T"
      "$PYTHON" -u scripts/eval/ssm_streaming_benchmark.py \
        --checkpoint "$ck" --output-dir "$OUT/gen_${arm}_N${N}_T${T}" \
        --label "gen_${arm}_N${N}_T${T}" \
        --prompt-length "$PROMPT" --generation-length "$N" \
        --generation-batch-size 1 --diffusion-steps "$T" \
        --prefix-lengths "$N" || rc=$((rc+1))
    done
  done
done

echo; echo "[$(date)] sweep exit=$rc"
"$PYTHON" -u scripts/eval/generation_curves.py --indir "$OUT" \
  --outdir results/figures || rc=$((rc+1))
echo "[$(date)] done rc=$rc"
exit $rc
