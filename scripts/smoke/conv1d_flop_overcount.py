#!/usr/bin/env python
"""The SSM 1.35-1.37x "unattributed residual" is a FlopCounterMode conv bug.

`training_flops.py` reports the SSM arms undercounting the counter by a
constant factor, with 486 GFLOP at L=2048 that FlopCounterMode attributes to no
nn.Module. This script shows that residual is not missing arithmetic at all:

  * `models/mamba2_segment.py:191,197,209-212` calls `F.conv1d` on
    `self.conv1d.weight` instead of calling `self.conv1d(...)`, so the
    `nn.Conv1d` module never enters `ModuleTracker.parents` and every conv FLOP
    lands in "Global" and in no module. That is why `flop_breakdown.json` has no
    `conv1d` row.
  * `torch.utils.flop_counter.conv_backward_flop` (flop_counter.py:155-257)
    IGNORES its `_groups` argument. For the grad_weight branch it swaps batch
    and channel (`t()`, line 170-171) and then charges `c_out * c_in` as if the
    convolution were dense, so a DEPTHWISE conv over `conv_dim` channels is
    billed `conv_dim` times its real cost.

The counter therefore OVER-counts, and `training_flops.py`'s conv term is
right. Runs on CPU in a second; no GPU, no model.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[2]

# uSSM-AR at L=2048, micro batch 1 -- the geometry of results/sizing/flop_breakdown.json.
BATCH = 1
SEQLEN = 2047          # diffusion.py:1016 shifts the AR objective by one token
CONV_DIM = 1664        # d_inner 1536 + 2 * d_state 64 (mamba2_segment.py:100)
D_CONV = 4
N_LAYERS = 12
OBSERVED_RESIDUAL = 546_638_217_216   # counted_total - the four counted leaves


def counted_conv_flops(seqlen):
  """FLOPs FlopCounterMode bills for one layer's `_causal_conv`, fwd+bwd."""
  weight = torch.randn(CONV_DIM, 1, D_CONV, requires_grad=True)
  bias = torch.randn(CONV_DIM, requires_grad=True)
  raw = torch.randn(BATCH, CONV_DIM, seqlen, requires_grad=True)
  state = torch.zeros(BATCH, CONV_DIM, D_CONV)
  pad = D_CONV - 1
  counter = FlopCounterMode(display=False)
  with counter:
    # mamba2_segment.py:209 -- the full-length zero-padded convolution.
    full = F.conv1d(raw, weight, bias, groups=CONV_DIM, padding=pad)
    # mamba2_segment.py:210-212 -- the `pad`-wide boundary fixup.
    head = F.conv1d(
      torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
      weight, bias, groups=CONV_DIM).narrow(-1, 1, pad)
    (full.sum() + head.sum()).backward()
  return counter.get_total_flops()


def main():
  per_layer = counted_conv_flops(SEQLEN)
  total = per_layer * N_LAYERS
  # The real cost of a depthwise causal conv: 2 * conv_dim * d_conv per token,
  # times 3 for fwd+bwd -- exactly `training_flops.ssm_terms()["conv"]`.
  honest = 3 * N_LAYERS * BATCH * SEQLEN * 2 * CONV_DIM * D_CONV

  print(f"uSSM-AR L=2048 batch=1: {N_LAYERS} layers x depthwise conv1d "
        f"({CONV_DIM} groups, k={D_CONV})\n")
  print(f"{'counted by FlopCounterMode, per layer':<44}{per_layer/1e9:>12.3f} GFLOP")
  print(f"{'counted, all layers':<44}{total/1e9:>12.3f} GFLOP")
  print(f"{'observed unattributed residual':<44}"
        f"{OBSERVED_RESIDUAL/1e9:>12.3f} GFLOP")
  print(f"{'match':<44}{'EXACT' if total == OBSERVED_RESIDUAL else 'NO':>12}")
  print()
  print(f"{'honest depthwise cost (2*C*k per token, 3x)':<44}"
        f"{honest/1e9:>12.3f} GFLOP")
  print(f"{'counter over-count factor':<44}{total/honest:>12.1f}x")
  print(f"\nconv_dim = {CONV_DIM}; the over-count is the c_out*c_in product a "
        f"grouped\nconvolution does not pay (flop_counter.py:146, reached from "
        f"line 255).")
  return 0 if total == OBSERVED_RESIDUAL else 1


if __name__ == "__main__":
  sys.exit(main())
