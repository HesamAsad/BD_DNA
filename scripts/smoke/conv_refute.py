#!/usr/bin/env python
"""Independent A/B of the padded-conv change: restore the old cat() form by
monkeypatch and re-run the same production step sizing_sweep.py runs."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch, torch.nn.functional as F

import sizing_sweep as ss  # same directory
from models.mamba2_segment import SegmentMamba2


def old_causal_conv(self, xBC, initial_state, return_state=True):
  raw = xBC.transpose(1, 2)
  history = torch.cat((initial_state.to(dtype=raw.dtype), raw), dim=-1)
  convolved = F.conv1d(history, self.conv1d.weight, self.conv1d.bias,
                       groups=self.conv_dim)
  convolved = convolved.narrow(-1, 1, raw.shape[-1]).transpose(1, 2)
  out = F.silu(convolved)
  if not return_state:
    return out, None
  return out, history.narrow(-1, history.shape[-1] - self.d_conv, self.d_conv)


def old_reverse_causal_conv(self, xBC, initial_state):
  """The pre-fix spelling of the reverse conv: flip the projection, run the
  ordinary causal `cat` convolution over it, keep the result in *reversed*
  order.

  Returning reversed order is what `scan_bidirectional` consumes -- the flip
  back to original order happens after the scan. An earlier version of this
  control flipped here as well, which silently fed the scan original-order
  operands; the memory numbers were unaffected but the loss was not, so the
  ordering matters for any row that compares losses.
  """
  raw = xBC.transpose(1, 2)
  seqlen = raw.shape[-1]
  history = torch.cat(
    (initial_state.to(dtype=raw.dtype), torch.flip(raw, dims=(-1,))), dim=-1)
  convolved = F.conv1d(history, self.conv1d.weight, self.conv1d.bias,
                       groups=self.conv_dim)
  return F.silu(convolved.narrow(-1, 1, seqlen).transpose(1, 2))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--arm", default="ussm-ar")
  p.add_argument("--variant", choices=["new", "old"], default="new")
  p.add_argument("--batch-size", type=int, default=4)
  p.add_argument("--length", type=int, default=8192)
  p.add_argument("--block-size", type=int, default=256)
  p.add_argument("--output", type=Path, required=True)
  a = p.parse_args()
  if a.variant == "old":
    SegmentMamba2._causal_conv = old_causal_conv
    SegmentMamba2._reverse_causal_conv = old_reverse_causal_conv
  device = torch.device("cuda")
  row = ss.run_case(a.arm, a.length, a.block_size, a.batch_size, False,
                    2, 5, device)
  row["variant"] = a.variant
  a.output.parent.mkdir(parents=True, exist_ok=True)
  a.output.write_text(json.dumps(row, indent=2))
  print(json.dumps(row, indent=2))
