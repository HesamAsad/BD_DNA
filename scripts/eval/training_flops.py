#!/usr/bin/env python3
"""Exact training FLOPs per arm -- replaces the wrong `trainer/total_pflop`.

`diffusion.py:_log_train_telemetry` estimates compute from only
(n_params, L, n_layers, d, cross_attn). That is architecture-blind in two ways
that matter:

  1. It adds a QUADRATIC ATTENTION term to every arm, including the SSM arms,
     which contain no attention at all. This is why all three SSM runs logged a
     bit-identical 4433.47 PFLOP.
  2. It never sees the extra backbone invocations that block diffusion on an
     SSM backbone actually performs -- the clean boundary prefill, its doubled
     scan, the reverse scan, and the C-a right-flank pass.

This script computes the honest number from the real forward paths. Every
structural claim below was verified against the code:

  * the boundary prefill is GRAD-ENABLED -- `prefill_left_boundaries_stacked` /
    `prefill_right_boundaries_stacked` are called at diffusion.py:1174,1183
    under no `no_grad` context, so it is charged 3x (fwd+bwd) like everything
    else, not 1x.
  * `ngroups=1`: B and C are rearranged to "b l 1 n"
    (mamba2_segment.py:372-373), so ONE [chunk, chunk] score matrix is shared
    by all `nheads` heads. Charging it per-head overstates BiSSM-BD by 3.4%.
  * `scan_with_block_boundaries` calls the fused kernel TWICE over the prefix
    (mamba2_segment.py ~377 and ~382): once contiguously for the true outputs,
    once folded per block for the local final states. Hence `2 * SCAN`.
  * the bidirectional active pass runs the mixer twice (forward + flipped
    reverse scan), bidirectional_ssm.py:199-201. Hence `2 * MIXER`.

Conventions: an (m x k) @ (k x n) matmul costs 2*m*k*n. Backward costs 2x
forward, so training = 3x forward. Elementwise ops, norms and softmax are
ignored (each below 2% of total). The SSD selective scan is LINEAR in sequence
length; it is charged densely per 128-token chunk because
`mamba_chunk_scan_combined` materialises the full [chunk, chunk] tile and
multiplies by a decay mask -- it does not skip the masked triangle, unlike
flash/flex attention, which do skip fully-masked tiles and are therefore
charged only the permitted pairs.

**THESE NUMBERS ARE A MODULE-LEVEL LOWER BOUND FOR THE SSM ARMS.** Validated
against `torch.utils.flop_counter.FlopCounterMode` on an H200 (LSF 116338,
116373; results/sizing/measured_flops.json, flop_breakdown.json):

  Transformer arms  EXACT. The counter is blind to flash attention, and its
                    shortfall equals this file's attention term to the decimal
                    at every length: 1.48 / 8.92 / 83.92 TFLOP per sequence at
                    L = 2048 / 8192 / 32768. The formula is correct.

  SSM arms          UNDERCOUNT by a constant 1.35x (bissm) to 1.37x (ussm-ar),
                    identical at every length from 2k to 32k. Every PER-MODULE
                    term here is nonetheless exactly right: summed over leaves,
                    in_proj + conv1d + out_proj + mlp + scan + head = 1294.60
                    GFLOP against 1294.61 predicted. The gap is 486 GFLOP --
                    27% of the dispatched total, 19.8M FLOP per token per layer
                    -- that the counter attributes to NO module, i.e. work
                    outside nn.Module boundaries which a module-by-module
                    derivation cannot see by construction. It has not been
                    traced to a specific operation.

So: quote these figures for the Transformer arms, and quote MEASURED FLOPs for
the SSM arms (scripts/eval/measured_flops_sweep.py). The direction of the error
is against us -- the SSM arms cost MORE than this file reports, which weakens
rather than strengthens any efficiency claim built on them. In particular the
"our compute is 0.33x dnaHNet's smallest budget" line for uSSM-AR is optimistic
by roughly 37%.

Run:  python scripts/eval/training_flops.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- run budget (identical for every arm) --------------------------------
STEPS = 8000
GLOBAL_BATCH = 64          # sequences per optimizer step
LENGTH = 8192              # nucleotides per sequence
BLOCK = 256
NUM_BLOCKS = LENGTH // BLOCK          # 32
PREFIX = (NUM_BLOCKS - 1) * BLOCK     # 7936 tokens carry a boundary prefill
N_LAYERS = 12
VOCAB = 13
GRAD_MULT = 3              # fwd + bwd
SEQUENCES = GRAD_MULT * STEPS * GLOBAL_BATCH   # 1.536e6

# ---- SSM (Mamba-2) geometry ----------------------------------------------
D_MODEL = 768
D_INNER = 1536
NHEADS = 24
HEADDIM = 64
D_STATE = 64
CHUNK = 128
CONV_DIM = 1664
D_CONV = 4
PROJ_DIM = 3224
MLP_HIDDEN = 4 * D_MODEL   # 3072

# ---- Transformer geometry -------------------------------------------------
D_AR = 832                 # Transformer-AR runs a wider model
D_BD = 768


def ssm_terms():
  """Per-token, per-layer FLOPs for each piece of a Mamba-2 layer."""
  in_proj = 2 * D_MODEL * PROJ_DIM
  conv = 2 * CONV_DIM * D_CONV
  out_proj = 2 * D_INNER * D_MODEL
  mlp = 2 * (D_MODEL * MLP_HIDDEN + MLP_HIDDEN * D_MODEL)
  # SSD chunked scan, per token: the shared [chunk, chunk] score matrix
  # (ngroups=1) plus, per head, the chunk-local output, the state update and
  # the state-to-output read; plus amortised inter-chunk state passing.
  scan = (2 * CHUNK * D_STATE
          + NHEADS * (2 * CHUNK * HEADDIM
                      + 2 * D_STATE * HEADDIM
                      + 2 * D_STATE * HEADDIM)
          + 2 * NHEADS * HEADDIM * D_STATE // CHUNK)
  head = 2 * D_MODEL * VOCAB
  mixer = in_proj + conv + scan + out_proj
  # `_block_state_passing`: a masked [num_seg, num_seg] decay matmul over the
  # prefix, amortised to a per-token cost.
  bsp = (2 * NUM_BLOCKS * (NUM_BLOCKS - 1)
         * NHEADS * HEADDIM * D_STATE) // PREFIX
  return dict(in_proj=in_proj, conv=conv, out_proj=out_proj, mlp=mlp,
              scan=scan, head=head, mixer=mixer, bsp=bsp)


def ssm_passes():
  """FLOPs per sequence for each composite SSM pass (forward only)."""
  t = ssm_terms()
  clean = N_LAYERS * PREFIX * (
    t["in_proj"] + t["conv"] + 2 * t["scan"] + t["out_proj"]
    + t["mlp"] + t["bsp"])
  act_uni = (N_LAYERS * LENGTH * (t["mixer"] + t["mlp"])
             + LENGTH * t["head"])
  act_bi = (N_LAYERS * LENGTH * (2 * t["mixer"] + t["mlp"])
            + LENGTH * t["head"])
  # AR shifts by one token (diffusion.py:1016), so it sees L-1.
  ar = (N_LAYERS * (LENGTH - 1) * (t["mixer"] + t["mlp"])
        + (LENGTH - 1) * t["head"])
  return dict(clean=clean, act_uni=act_uni, act_bi=act_bi, ar=ar)


def transformer_ar_params(d=D_AR):
  """Matmul-bearing parameters: qkv, out proj, and the 4x MLP, plus the head."""
  per_layer = 3 * d * d + d * d + 2 * d * (4 * d)
  return N_LAYERS * per_layer + d * VOCAB


def transformer_bd_params(d=D_BD):
  per_layer = d * (3 * d) + d * d + 2 * d * (4 * d)
  return N_LAYERS * per_layer + VOCAB * d


def arms():
  p = ssm_passes()
  tokens_ar = LENGTH - 1

  # Transformer-AR: dense causal attention over L-1 tokens.
  xf_ar_pairs = tokens_ar * (tokens_ar + 1) // 2
  xf_ar = (2 * transformer_ar_params() * tokens_ar
           + 4 * N_LAYERS * xf_ar_pairs * D_AR)

  # Transformer-BD: cross_attn concatenates [x_t; x_0] to 2L. The block
  # diffusion mask permits block^2 * nb * (nb+1) pairs, NOT (2L)^2 and not the
  # telemetry's L*L + 2*L*block.
  xf_bd_pairs = BLOCK * BLOCK * NUM_BLOCKS * (NUM_BLOCKS + 1)
  xf_bd = (2 * transformer_bd_params() * (2 * LENGTH)
           + 4 * N_LAYERS * xf_bd_pairs * D_BD)

  return [
    ("uSSM-AR", p["ar"], "one grad pass, no attention, linear scan"),
    ("Transformer-AR", xf_ar, f"causal attention, d={D_AR}"),
    ("uSSM-BD", p["clean"] + p["act_uni"], "clean prefill + unidirectional active"),
    ("BiSSM-BD", p["clean"] + p["act_bi"], "clean prefill + bidirectional active"),
    ("BiSSM-Ca", 1.5 * p["clean"] + p["act_bi"], "1.5x prefill (rfp=0.5) + bidirectional"),
    ("Transformer-BD", xf_bd, f"cross_attn [x_t;x_0] at 2L, {xf_bd_pairs:,} pairs"),
  ]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  rows = []
  for name, per_seq, note in arms():
    pflop = per_seq * SEQUENCES / 1e15
    rows.append({"arm": name, "pflop": pflop, "flops": pflop * 1e15,
                 "per_sequence_forward_flops": per_seq, "note": note})

  base = next(r["pflop"] for r in rows if r["arm"] == "uSSM-AR")
  six_nd = 6 * 100.70e6 * (STEPS * GLOBAL_BATCH * LENGTH) / 1e15

  print(f"budget: {STEPS} steps x {GLOBAL_BATCH} seq x {LENGTH} nt "
        f"= {STEPS*GLOBAL_BATCH*LENGTH/1e9:.4f}e9 nt\n")
  print(f"{'arm':<16} {'PFLOP':>10} {'rel':>7}   note")
  print("-" * 78)
  for r in sorted(rows, key=lambda x: x["pflop"]):
    print(f"{r['arm']:<16} {r['pflop']:>10.1f} {r['pflop']/base:>6.2f}x   {r['note']}")
  print("-" * 78)
  print(f"{'6ND reference':<16} {six_nd:>10.1f} {six_nd/base:>6.2f}x   "
        f"6*N*D with N=100.70M (uSSM-AR should sit just above this)")
  print(f"\nwandb trainer/total_pflop logged 4433.47 for ALL FOUR SSM arms, "
        f"4569.40 for Transformer-AR and 8687.00 for Transformer-BD "
        f"-- do not quote it.")

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
      {"budget": {"steps": STEPS, "global_batch": GLOBAL_BATCH,
                  "length": LENGTH, "tokens": STEPS * GLOBAL_BATCH * LENGTH},
       "six_nd_reference_pflop": six_nd,
       "arms": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main()
