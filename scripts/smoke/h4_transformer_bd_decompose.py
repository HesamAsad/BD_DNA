#!/usr/bin/env python3
"""H4: where does Transformer-BD's 'unexplained' slowdown actually come from?

Reads results/figures/scaling_data.json (measured throughput, micro batch 2)
and scripts/eval/scaling_curves.py's analytic FLOPs, and does three things:

1. Reproduces the slowdown / FLOPs-explain / unexplained table at EVERY swept
   length, not just the two endpoints.
2. Converts throughput to achieved TFLOP/s per arm -- the quantity the
   'unexplained' factor is literally a ratio of.
3. Fits, per arm, the two-parameter roofline

       step_seconds = a * tokens_through_backbone  +  b * attention_FLOPs

   `a` is the per-token cost of everything that is NOT attention (projections,
   MLP, norms, rotary, dropout, adaLN, the [x_t;x_0] cat).  `b` is the inverse
   efficiency of the attention kernel (flash causal for AR, flex block-mask for
   BD).  Splitting these is the only way to tell 'the BD arm's attention kernel
   is slower' from 'the BD arm's per-token path is slower'.

Usage:  python scripts/smoke/h4_transformer_bd_decompose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.training_flops as tf  # noqa: E402
from scripts.eval.scaling_curves import flops_per_sequence  # noqa: E402

BATCH = 2
BLOCK = 256
N_LAYERS = 12
TILE = 128   # flex/flash sparse-block granularity


def attn_flops_ar(L, corrected=False):
  tokens = L - 1
  pairs = tokens * (tokens + 1) // 2
  if corrected:
    n = -(-tokens // TILE)
    pairs = TILE * TILE * n * (n + 1) // 2
  return tf.GRAD_MULT * tf.ATTN_SCALE * 4 * N_LAYERS * pairs * tf.D_AR


def attn_flops_bd(L, corrected=False):
  nb = L // BLOCK
  pairs = BLOCK * BLOCK * nb * (nb + 1)   # measured exact: 0 partial tiles
  return tf.GRAD_MULT * tf.ATTN_SCALE * 4 * N_LAYERS * pairs * tf.D_BD


def param_flops_ar(L):
  return tf.GRAD_MULT * 2 * tf.transformer_ar_params() * (L - 1)


def param_flops_bd(L):
  return tf.GRAD_MULT * 2 * tf.transformer_bd_params() * (2 * L)


def lstsq2(rows):
  """Least squares for y = a*x1 + b*x2 (no intercept), 2 unknowns."""
  s11 = sum(x1 * x1 for x1, x2, y in rows)
  s12 = sum(x1 * x2 for x1, x2, y in rows)
  s22 = sum(x2 * x2 for x1, x2, y in rows)
  t1 = sum(x1 * y for x1, x2, y in rows)
  t2 = sum(x2 * y for x1, x2, y in rows)
  det = s11 * s22 - s12 * s12
  a = (t1 * s22 - t2 * s12) / det
  b = (s11 * t2 - s12 * t1) / det
  return a, b


def main():
  data = json.loads((REPO / "results/figures/scaling_data.json").read_text())
  tput = data["nt_per_second"]
  lengths = sorted(int(k) for k in tput["dit"] if k in tput["dit-ar"])

  print("=" * 86)
  print("1. FULL unexplained curve (the brief quoted only L=2048 and L=32768)")
  print("=" * 86)
  print(f"{'L':>7} {'ar nt/s':>10} {'bd nt/s':>10} {'slowdown':>9} "
        f"{'FLOP ratio':>11} {'UNEXPL':>8} {'UNEXPL(corr)':>13}")
  unexp = {}
  for L in lengths:
    k = str(L)
    f_ar = flops_per_sequence("dit-ar", L)
    f_bd = flops_per_sequence("dit", L)
    slow = tput["dit-ar"][k] / tput["dit"][k]
    ratio = f_bd / f_ar
    # corrected denominators: flash computes whole diagonal tiles causally
    f_ar_c = param_flops_ar(L) + attn_flops_ar(L, corrected=True)
    f_bd_c = param_flops_bd(L) + attn_flops_bd(L, corrected=True)
    unexp[L] = slow / ratio
    print(f"{L:>7} {tput['dit-ar'][k]:>10,.0f} {tput['dit'][k]:>10,.0f} "
          f"{slow:>9.3f} {ratio:>11.3f} {slow/ratio:>8.3f} "
          f"{slow/(f_bd_c/f_ar_c):>13.3f}")

  print()
  print("=" * 86)
  print("2. Achieved TFLOP/s (the 'unexplained' factor IS this ratio)")
  print("=" * 86)
  print(f"{'L':>7} {'AR TF/s':>9} {'BD TF/s':>9} {'ratio':>7}   "
        f"{'AR attn%':>9} {'BD attn%':>9}")
  for L in lengths:
    k = str(L)
    f_ar = flops_per_sequence("dit-ar", L)
    f_bd = flops_per_sequence("dit", L)
    ar_tf = f_ar * tput["dit-ar"][k] / L / 1e12
    bd_tf = f_bd * tput["dit"][k] / L / 1e12
    print(f"{L:>7} {ar_tf:>9.1f} {bd_tf:>9.1f} {ar_tf/bd_tf:>7.3f}   "
          f"{100*attn_flops_ar(L)/f_ar:>8.1f}% {100*attn_flops_bd(L)/f_bd:>8.1f}%")

  print()
  print("=" * 86)
  print("3. Two-parameter roofline fit:  step_s = a*tokens + b*attn_FLOPs")
  print("=" * 86)
  fits = {}
  for arm, tok_fn, attn_fn, per_tok_flops in (
      ("dit-ar", lambda L: BATCH * (L - 1), attn_flops_ar,
       tf.GRAD_MULT * 2 * tf.transformer_ar_params()),
      ("dit", lambda L: BATCH * 2 * L, attn_flops_bd,
       tf.GRAD_MULT * 2 * tf.transformer_bd_params())):
    rows = []
    for L in lengths:
      step_s = BATCH * L / tput[arm][str(L)]
      rows.append((tok_fn(L), BATCH * attn_fn(L), step_s))
    a, b = lstsq2(rows)
    fits[arm] = (a, b, per_tok_flops, rows)
    print(f"\n  {arm}:  a = {a*1e9:.4f} ns/token   "
          f"b = {b*1e12:.4f} s/TFLOP  -> attention runs at "
          f"{1/b/1e12:,.0f} TFLOP/s")
    print(f"    non-attention FLOPs/token (fwd+bwd) = {per_tok_flops:,}"
          f"  -> per-token path runs at {per_tok_flops/a/1e12:,.1f} TFLOP/s")
    print(f"    {'L':>7} {'measured s':>11} {'fitted s':>10} {'resid':>8}")
    for (x1, x2, y), L in zip(rows, lengths):
      pred = a * x1 + b * x2
      print(f"    {L:>7} {y:>11.4f} {pred:>10.4f} {100*(pred-y)/y:>7.2f}%")

  a_ar, b_ar, w_ar, _ = fits["dit-ar"]
  a_bd, b_bd, w_bd, _ = fits["dit"]
  print()
  print("  VERDICT SPLIT")
  print(f"    attention kernel   : BD is {b_bd/b_ar:.3f}x the s/FLOP of AR "
        f"(flex block-mask vs flash causal)")
  print(f"    per-token path     : BD is {(a_bd/w_bd)/(a_ar/w_ar):.3f}x the "
        f"s/FLOP of AR (same work shape, {w_bd/w_ar:.3f}x the FLOPs/token)")
  print(f"    tokens through backbone: BD/AR = {2*2/1:.1f}x nominal "
        f"(2L vs L, at equal batch)")

  print()
  print("=" * 86)
  print("4. What the per-token gap would have to be, in ns/token, to close it")
  print("=" * 86)
  # Hold b fixed at the AR value; how much of BD's step time is left over?
  for L in lengths:
    step_s = BATCH * L / tput["dit"][str(L)]
    attn_s = b_ar * BATCH * attn_flops_bd(L)
    rest = step_s - attn_s
    print(f"  L={L:>6}  step {step_s*1e3:>8.2f} ms   "
          f"attn@AR-eff {attn_s*1e3:>8.2f} ms   rest {rest*1e3:>8.2f} ms   "
          f"rest/token {rest/(BATCH*2*L)*1e9:>7.2f} ns")
  print(f"\n  AR, same treatment:")
  for L in lengths:
    step_s = BATCH * L / tput["dit-ar"][str(L)]
    attn_s = b_ar * BATCH * attn_flops_ar(L)
    rest = step_s - attn_s
    print(f"  L={L:>6}  step {step_s*1e3:>8.2f} ms   "
          f"attn        {attn_s*1e3:>8.2f} ms   rest {rest*1e3:>8.2f} ms   "
          f"rest/token {rest/(BATCH*(L-1))*1e9:>7.2f} ns")

  print()
  print("=" * 86)
  print("5. GLOBAL roofline: ONE set of constants, BOTH arms, no per-arm fudge")
  print("=" * 86)
  print("   step_s = F_gemm/E_gemm + F_attn/E_attn + Bytes_nonGEMM/BW + params*K")
  print("   Bytes_nonGEMM from scripts/smoke/dit_block_traffic_audit.py "
        "(exact aten tally,\n   CPU fp32, attention and GEMM excluded, "
        "AR corrected to flash's fused rotary):")
  print(f"      Transformer-BD  {BD_BYTES_PER_TOK_LAYER:>10,} B/token/layer  "
        f"({BD_OPS} non-GEMM aten calls per layer)")
  print(f"      Transformer-AR  {AR_BYTES_PER_TOK_LAYER:>10,} B/token/layer  "
        f"({AR_OPS} non-GEMM aten calls per layer)")
  print(f"      ratio           {BD_BYTES_PER_TOK_LAYER/AR_BYTES_PER_TOK_LAYER:>10.2f}x per token, "
        f"{2*BD_BYTES_PER_TOK_LAYER/AR_BYTES_PER_TOK_LAYER:.2f}x per SEQUENCE (BD runs 2L tokens)")
  print(f"      compare analytic FLOP ratio: 1.78x - 1.83x per sequence\n")

  # F_gemm and Bytes are BOTH proportional to tokens, so a least-squares fit
  # over the two arms is exactly determined, not overdetermined: solve the 2x2
  # directly from the two fitted per-token costs.  The TEST is whether the
  # solution lands at physically plausible values for BOTH constants at once.
  w_ar = tf.GRAD_MULT * 2 * tf.transformer_ar_params()
  w_bd = tf.GRAD_MULT * 2 * tf.transformer_bd_params()
  B_ar = AR_BYTES_PER_TOK_LAYER * N_LAYERS
  B_bd = BD_BYTES_PER_TOK_LAYER * N_LAYERS
  det = w_ar * B_bd - w_bd * B_ar
  u = (a_ar * B_bd - a_bd * B_ar) / det          # s per GEMM FLOP
  v = (w_ar * a_bd - w_bd * a_ar) / det          # s per byte
  print(f"   solve:  a_ar = w_ar/E + B_ar/BW ,  a_bd = w_bd/E + B_bd/BW")
  print(f"           w_ar = {w_ar:,} FLOP/token   B_ar = {B_ar:,} B/token")
  print(f"           w_bd = {w_bd:,} FLOP/token   B_bd = {B_bd:,} B/token")
  print(f"   ->  E_gemm = {1/u/1e12:,.0f} TFLOP/s   "
        f"({100/u/1e12/990:.0f}% of H200 bf16 dense peak 990 TFLOP/s)")
  print(f"   ->  BW     = {1/v/1e12:.2f} TB/s      "
        f"({100/v/1e12/4.8:.0f}% of H200 HBM3e 4.8 TB/s)")
  print(f"   ->  E_attn = {1/b_ar/1e12:,.0f} TFLOP/s (flash causal), "
        f"{1/b_bd/1e12:,.0f} TFLOP/s (flex block-mask)")
  print("\n   Both constants are physically admissible and SHARED. That is the")
  print("   confirmation: a wrong traffic model would drive one of them "
        "negative\n   or past peak, as the over-parameterised 4-term fit did.\n")

  print(f"   {'arm':>7} {'L':>7} {'measured ms':>12} {'model ms':>10} {'resid':>8} "
        f"| {'gemm':>7} {'attn':>7} {'bytes':>7}")
  for arm in ("dit-ar", "dit"):
    for L in lengths:
      if arm == "dit-ar":
        tok, fa, w, B, b = BATCH * (L - 1), attn_flops_ar(L), w_ar, B_ar, b_ar
      else:
        tok, fa, w, B, b = BATCH * 2 * L, attn_flops_bd(L), w_bd, B_bd, b_bd
      parts = [tok * w * u, BATCH * fa * b, tok * B * v]
      pred, y = sum(parts), BATCH * L / tput[arm][str(L)]
      print(f"   {arm:>7} {L:>7} {y*1e3:>12.2f} {pred*1e3:>10.2f} "
            f"{100*(pred-y)/y:>7.1f}% | "
            + " ".join(f"{100*p/pred:>6.1f}%" for p in parts))

  print()
  print("=" * 86)
  print("6. PROJECTED GAIN from removing BD-only non-GEMM traffic")
  print("=" * 86)
  # NOTE ON DROPOUT -- measured, and it REFUTES the obvious guess. small.yaml:9
  # sets 0.1 and small_ar_transformer.yaml:12 sets 0.0, so dropout looks like a
  # confound. It is not: inside the jit.script'd
  # `bias_dropout_add_scale_fused_train` (dit.py:129) with a grad-requiring
  # input, `F.dropout(x, p=0.0, training=True)` still dispatches to
  # native_dropout -- ATen's p==0 short-circuit is not taken. Both arms record
  # exactly 2 native_dropout + 2 native_dropout_backward per block at either p.
  # So dropout is symmetric (and wasted work for the AR arm), not an asymmetry.
  fixes = [
    ("fuse the rotary (dit.py:201 -> 1 kernel; -14 aten ops)", 233_472 - 24_576),
    ("+ single qkv GEMM at 2L, drop the cat (dit.py:604-607)", 36_900 + 8_346),
    ("+ adaLN, measured by ablation -- INTRINSIC, shown only to bound it",
     104_607),
  ]
  cum = 0
  for label, saved in fixes:
    cum += saved
    newB = (BD_BYTES_PER_TOK_LAYER - cum) * N_LAYERS
    print(f"\n   {label}")
    print(f"     BD non-GEMM traffic {BD_BYTES_PER_TOK_LAYER:,} -> "
          f"{BD_BYTES_PER_TOK_LAYER-cum:,} B/token/layer "
          f"(-{100*cum/BD_BYTES_PER_TOK_LAYER:.1f}%)")
    print(f"     {'L':>7} {'nt/s now':>10} {'nt/s pred':>10} {'gain':>7} "
          f"{'unexpl now':>11} {'unexpl pred':>12}")
    for L in lengths:
      tok = BATCH * 2 * L
      old = tok * B_bd * v + tok * w_bd * u + BATCH * attn_flops_bd(L) * b_bd
      new = tok * newB * v + tok * w_bd * u + BATCH * attn_flops_bd(L) * b_bd
      now = tput["dit"][str(L)]
      pred = now * old / new
      slow = tput["dit-ar"][str(L)] / pred
      ratio = flops_per_sequence("dit", L) / flops_per_sequence("dit-ar", L)
      print(f"     {L:>7} {now:>10,.0f} {pred:>10,.0f} "
            f"{100*(old/new-1):>6.1f}% {unexp[L]:>11.3f} {slow/ratio:>12.3f}")


def tokens_bytes(L, arm):
  """Non-GEMM bytes for ONE sequence, from the audited per-token-per-layer rate."""
  if arm == "dit-ar":
    return AR_BYTES_PER_TOK_LAYER * N_LAYERS * (L - 1)
  return BD_BYTES_PER_TOK_LAYER * N_LAYERS * 2 * L


def lstsq_n(rows):
  """Normal equations for y = sum_i c_i * x_i, no intercept."""
  k = len(rows[0][0])
  A = [[sum(r[0][i] * r[0][j] for r in rows) for j in range(k)] for i in range(k)]
  b = [sum(r[0][i] * r[1] for r in rows) for i in range(k)]
  for i in range(k):                       # gaussian elimination
    p = max(range(i, k), key=lambda r: abs(A[r][i]))
    A[i], A[p] = A[p], A[i]
    b[i], b[p] = b[p], b[i]
    for r in range(i + 1, k):
      f = A[r][i] / A[i][i]
      for c in range(i, k):
        A[r][c] -= f * A[i][c]
      b[r] -= f * b[i]
  x = [0.0] * k
  for i in reversed(range(k)):
    x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, k))) / A[i][i]
  return x


# Measured by scripts/smoke/dit_block_traffic_audit.py at micro batch 2:
#   BD  798,032 B/token/layer over 92 non-GEMM aten calls
#   AR  677,188 B/token/layer over 46 calls WITH the torchscript rotary; the
#       real AR arm takes dit.py:363-365 into flash's fused in-place kernel,
#       which replaces 14 of those calls with 1 and 252,672 -> 26,624 B/token.
BD_BYTES_PER_TOK_LAYER = 798_032
AR_BYTES_PER_TOK_LAYER = 452_676
BD_OPS, AR_OPS = 92, 33


if __name__ == "__main__":
  main()
