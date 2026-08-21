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

    THIS FILE COSTS THE TRAINED RUNS, AND `2 * MIXER` IS RIGHT FOR THEM. The
    BiSSM checkpoints are from 2026-08-10, when scan_active called
    mixer.scan_segment twice -- forward and flipped -- each a complete mixer
    including in_proj and out_proj. Commit 5dad03c (2026-08-20) replaced that
    with a shared-projection scan_bidirectional: mamba2_segment.py:635 calls
    in_proj once and :666 calls out_proj once, so only conv and scan are now
    doubled. Do NOT "correct" this term -- it would misstate what the runs in
    the results table actually cost.

    Anything projecting the cost of the CURRENT code must instead charge
    `MIXER + CONV + SCAN`. scaling_curves.py:flops_per_sequence does exactly
    that, and the two files disagree by design: in_proj + out_proj = 7,311,360
    FLOP per token per layer, 28.5% of the active term, 16.8% of the arm.

Conventions: an (m x k) @ (k x n) matmul costs 2*m*k*n. Backward costs 2x
forward, so training = 3x forward. Elementwise ops, norms and softmax are
ignored (each below 2% of total). The SSD selective scan is LINEAR in sequence
length; it is charged densely per 128-token chunk because
`mamba_chunk_scan_combined` materialises the full [chunk, chunk] tile and
multiplies by a decay mask -- it does not skip the masked triangle, unlike
flash/flex attention, which do skip fully-masked tiles and are therefore
charged only the permitted pairs.

**THESE NUMBERS ARE CORRECT. QUOTE THEM FOR EVERY ARM.** Validated against
`torch.utils.flop_counter.FlopCounterMode` on an H200 (LSF 116338, 116373,
116638, 116639; results/sizing/measured_flops*.json, flop_breakdown.json):

  Transformer arms  EXACT. The counter is blind to flash attention, and its
                    shortfall equals this file's attention term to the decimal
                    at every length: 1.48 / 8.92 / 83.92 TFLOP per sequence at
                    L = 2048 / 8192 / 32768.

  SSM arms          EXACT for every aten-visible term. The counter reads
                    1.35x (bissm) to 1.37x (ussm-ar) ABOVE this file, flat at
                    every length from 2k to 128k. That is the COUNTER
                    overcounting, not this file undercounting.

RESOLVED -- what the 1.35-1.37x actually was. Two independent facts compound:

  1. ATTRIBUTION. mamba2_segment.py never calls self.conv1d(...). It reads
     self.conv1d.weight / .bias (:191) and calls functional F.conv1d (:197,
     :209, :210, :264, :270, :271, :492). nn.Conv1d.__call__ therefore never
     fires, ModuleTracker never opens a bucket, and every conv FLOP is billed
     to "Global" and to no module -- which is why flop_breakdown.json has no
     conv1d row while this file has a conv term.

  2. MAGNITUDE. torch.utils.flop_counter.conv_backward_flop takes `_groups` and
     never uses it; its docstring (flop_counter.py:83) admits "there are also
     some details involving transpose of the batch/channel dimensions and
     groups, but I skip those for the sake of brevity". The grad_weight branch
     transposes dims 0 and 1, after which batch_size becomes the channel count
     and c_out*c_in becomes conv_dim*1 -- so a DEPTHWISE conv is billed at the
     DENSE price. Here that is conv_dim = 1664x on grad_weight, 555x on the
     conv as a whole (verified standalone on CPU: 181,724,827,648 counted
     against 327,235,584 true, for this model's exact conv geometry).

The accounting closes exactly. uSSM-AR, L=8192, one sequence, fwd+bwd, TFLOP:

    analytic (this file)                    5.180
  + phantom depthwise-conv grad_weight      2.177
  - Triton SSD scan, invisible to counter   0.237
  = predicted                               7.120
    measured                                7.120     residual 0.0%

The flatness of the ratio across a 64x span in length (1.376 -> 1.374 for
ussm-ar, 1.341 -> 1.349 for bissm; results/figures/scaling_data.json,
flop_counter_ratio) is independent confirmation: a constant ratio is what a
strictly per-token phantom produces, and the conv is per-token.

ONE HONEST LIMIT. The above confirms every term this file derives EXCEPT the
scan. `scan` (4.6% of the uSSM-AR total) runs in a Triton kernel outside aten
and is invisible to the counter, so it remains a derivation that no measurement
has checked. Everything else is now measured-confirmed.

So the "our compute is 0.33x dnaHNet's smallest budget" line for uSSM-AR STANDS.
An earlier revision of this docstring told you to inflate it by 37%; that was
wrong and is retracted.

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
