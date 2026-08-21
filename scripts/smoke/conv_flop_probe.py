#!/usr/bin/env python
"""Does the depthwise conv1d explain the SSM arms' unattributed FLOPs?

`results/sizing/flop_breakdown.json` (uSSM-AR, L=2048, batch=1) leaves
546.638 GFLOP that FlopCounterMode charges to `Global` and to no
weight-bearing leaf module. Two independent code facts predict exactly that:

  1. `mamba2_segment.py:209,211` call `F.conv1d(...)` -- the functional, not
     `self.conv1d(...)`. `nn.Conv1d.__call__` therefore never fires, so
     `ModuleTracker` never pushes a bucket and the op lands in `Global` only.
     Consistent with `counted_by_module` having no `conv1d` key at all.

  2. `torch/utils/flop_counter.py:155` (`conv_backward_flop`) takes `_groups`
     and ignores it. The grad-weight branch calls
     `conv_flop_count(t(x_shape), grad_out_shape, grad_weight_shape)`, whose
     body (line 145) is
         prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
     With the operands permuted the way the backward branch permutes them,
     `c_in` becomes the FULL channel count (1664) instead of
     channels-per-group (1), so a depthwise conv's weight gradient is charged
     `conv_dim` = 1664x its true cost.

This script replays only the conv calls, at the exact shapes each entry point
uses, and reports what the counter charges. CPU only.

Run:  python scripts/smoke/conv_flop_probe.py
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

CONV_DIM = 1664
D_CONV = 4
N_LAYERS = 12


def count(fn):
  counter = FlopCounterMode(display=False)
  with counter:
    fn()
  return counter.get_total_flops()


def causal_conv(batch, seqlen, weight, bias):
  """`mamba2_segment.py:189-225`, the seqlen >= d_conv branch."""
  raw = torch.randn(batch, CONV_DIM, seqlen, requires_grad=True)
  state = torch.zeros(batch, CONV_DIM, D_CONV)
  pad = D_CONV - 1
  full = F.conv1d(raw, weight, bias, groups=CONV_DIM, padding=pad)
  head = F.conv1d(
    torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
    weight, bias, groups=CONV_DIM).narrow(-1, 1, pad)
  full.narrow(-1, 0, pad).copy_(head)
  full.sum().backward()


def boundary_conv(batch, seqlen, weight, bias):
  """`mamba2_segment.py:491-497`: one padded conv, no boundary fixup."""
  raw = torch.randn(batch, CONV_DIM, seqlen, requires_grad=True)
  F.conv1d(raw, weight, bias, groups=CONV_DIM,
           padding=D_CONV - 1).sum().backward()


def reverse_causal_conv(batch, seqlen, weight, bias):
  """`mamba2_segment.py:254-279`, the seqlen >= d_conv branch."""
  raw = torch.randn(batch, CONV_DIM, seqlen, requires_grad=True)
  state = torch.zeros(batch, CONV_DIM, D_CONV)
  flipped = weight.flip(-1)
  pad = D_CONV - 1
  full = F.conv1d(raw, flipped, bias, groups=CONV_DIM, padding=pad)
  tail = F.conv1d(
    torch.cat((raw.narrow(-1, seqlen - pad, pad), state.flip(-1)), dim=-1),
    flipped, bias, groups=CONV_DIM).narrow(-1, 0, pad)
  full.narrow(-1, seqlen, pad).copy_(tail)
  full.sum().backward()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  weight = torch.randn(CONV_DIM, 1, D_CONV, requires_grad=True)
  bias = torch.randn(CONV_DIM, requires_grad=True)

  rows = []

  # --- the measured point: uSSM-AR, L=2048, batch=1, 12 layers -------------
  per_layer = count(lambda: causal_conv(1, 2048, weight, bias))
  ussm_ar = N_LAYERS * per_layer
  rows.append(("ussm-ar L=2048 b=1  causal_conv x12", ussm_ar))

  # --- BiSSM-BD, L=8192, block 256, batch 2 -------------------------------
  batch, length, block = 2, 8192, 256
  num_blocks = length // block
  prefix = (num_blocks - 1) * block
  folded = batch * num_blocks
  bd = N_LAYERS * (
    count(lambda: boundary_conv(batch, prefix, weight, bias))
    + count(lambda: causal_conv(folded, block, weight, bias))
    + count(lambda: reverse_causal_conv(folded, block, weight, bias)))
  rows.append(("bissm  L=8192 b=2  prefill+fwd+rev x12", bd))

  print(f"{'case':<44}{'GFLOP':>14}{'FLOP/tok/layer':>18}")
  print("-" * 76)
  print(f"{rows[0][0]:<44}{rows[0][1]/1e9:>14.3f}"
        f"{rows[0][1]/(2047*N_LAYERS):>18,.0f}")
  tokens = batch * length
  print(f"{rows[1][0]:<44}{rows[1][1]/1e9:>14.3f}"
        f"{rows[1][1]/(tokens*N_LAYERS):>18,.0f}")
  print()
  print("flop_breakdown.json unattributed residual: "
        f"{546_638_217_216/1e9:.3f} GFLOP "
        f"({546_638_217_216/(2047*N_LAYERS):,.0f} FLOP/tok/layer)")
  print(f"conv predicts / residual = {ussm_ar/546_638_217_216:.4f}")

  if args.json:
    print(json.dumps(dict(rows), indent=2))


if __name__ == "__main__":
  main()
