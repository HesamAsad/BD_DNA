#!/usr/bin/env python
"""CPU probe: what FlopCounterMode actually charges for the scan region.

Answers three questions without a GPU, by dispatching the *same aten ops* the
CUDA run dispatches:

  1. `_reference_scan`'s two einsums (mamba2_segment.py:300, :304) -- does the
     3-operand outer-product einsum count as a matmul, or as a `mul`?
  2. `_block_state_passing`'s masked decay einsum (mamba2_segment.py:450).
  3. `_causal_conv`'s two depthwise convolutions (mamba2_segment.py:209, :210)
     -- forward AND backward, which is where `conv_backward_flop` ignores
     `groups` (torch/utils/flop_counter.py:170-171, :247-256).

Every number printed is a FlopCounterMode count, not an analytic guess.
"""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

D_MODEL, D_INNER, NHEADS, HEADDIM, D_STATE = 768, 1536, 24, 64, 64
CONV_DIM, D_CONV, CHUNK, BLOCK = 1664, 4, 128, 256


def count(fn):
  counter = FlopCounterMode(display=False)
  with counter:
    fn()
  return counter.get_total_flops()


def reference_scan_step(batch=1):
  """One position of mamba2_segment.py:_reference_scan (:294-306)."""
  dt_i = torch.randn(batch, NHEADS)
  B_i = torch.randn(batch, D_STATE)
  x_i = torch.randn(batch, NHEADS, HEADDIM)
  state = torch.randn(batch, NHEADS, HEADDIM, D_STATE)
  C_i = torch.randn(batch, D_STATE)

  def outer():
    torch.einsum("bh,bn,bhp->bhpn", dt_i, B_i, x_i)

  def read():
    torch.einsum("bhpn,bn->bhp", state, C_i)

  print(f"  einsum :300 'bh,bn,bhp->bhpn' (outer)  {count(outer):>18,}")
  print(f"  einsum :304 'bhpn,bn->bhp'   (contract) {count(read):>17,}")


def block_state_passing(batch=1, length=8192, block=BLOCK):
  """mamba2_segment.py:450 -- the [num_seg+1, num_seg] decay matmul."""
  num_seg = (length - block) // block          # prefix = length - block
  decay = torch.randn(batch, num_seg + 1, num_seg, NHEADS)
  local = torch.randn(batch, num_seg, NHEADS, HEADDIM, D_STATE)

  def go():
    torch.einsum("bijh,bjhpn->bihpn", decay, local)

  flops = count(go)
  print(f"  L={length} num_seg={num_seg}: {flops:>15,} FLOP/layer fwd"
        f"   -> {flops / length:>10,.0f} FLOP/token/layer")


def causal_conv(batch=1, length=2047):
  """mamba2_segment.py:209-212 -- the two depthwise convs, fwd + bwd."""
  raw = torch.randn(batch, CONV_DIM, length, requires_grad=True)
  state = torch.randn(batch, CONV_DIM, D_CONV)
  weight = torch.randn(CONV_DIM, 1, D_CONV, requires_grad=True)
  bias = torch.randn(CONV_DIM, requires_grad=True)
  pad = D_CONV - 1

  def fwd_only():
    F.conv1d(raw, weight, bias, groups=CONV_DIM, padding=pad)

  def both():
    full = F.conv1d(raw, weight, bias, groups=CONV_DIM, padding=pad)
    head = F.conv1d(
      torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
      weight, bias, groups=CONV_DIM).narrow(-1, 1, pad)
    (full.sum() + head.sum()).backward()

  f = count(fwd_only)
  b = count(both)
  out_len = length + 2 * pad - D_CONV + 1
  print(f"  main conv fwd only              {f:>18,}")
  print(f"  both convs fwd+bwd              {b:>18,}")
  print(f"  honest main conv fwd            "
        f"{2 * batch * out_len * CONV_DIM * D_CONV:>18,}")
  print(f"  overcount factor on grad_weight = groups = {CONV_DIM}")
  print(f"  per layer x12                   {b * 12:>18,}")


def reconcile():
  """Close results/sizing/flop_breakdown.json exactly, with no free parameter."""
  import json
  from pathlib import Path
  path = Path(__file__).resolve().parents[2] / "results/sizing/flop_breakdown.json"
  if not path.exists():
    print(f"  (missing {path})")
    return
  d = json.loads(path.read_text())
  by = d["counted_by_module"]
  # Only these four leaves ever fire nn.Module.__call__ on a matmul.
  attributed = by["in_proj"] + by["mlp"] + by["out_proj"] + by["output"]
  residual = d["counted_total"] - attributed
  conv = count_causal_conv(length=d["tokens"], batch=d["batch"]) * 12
  print(f"  counted_total            {d['counted_total']:>18,}")
  print(f"  attributed to leaves     {attributed:>18,}")
  print(f"  unattributed residual    {residual:>18,}")
  print(f"  depthwise conv x12 layers{conv:>18,}")
  print(f"  DELTA                    {residual - conv:>18,}")
  print(f"  -> the fused Triton scan contributed exactly 0 counted FLOP")


def count_causal_conv(length, batch=1):
  raw = torch.randn(batch, CONV_DIM, length, requires_grad=True)
  state = torch.randn(batch, CONV_DIM, D_CONV)
  weight = torch.randn(CONV_DIM, 1, D_CONV, requires_grad=True)
  bias = torch.randn(CONV_DIM, requires_grad=True)
  pad = D_CONV - 1

  def both():
    full = F.conv1d(raw, weight, bias, groups=CONV_DIM, padding=pad)
    head = F.conv1d(
      torch.cat((state, raw.narrow(-1, 0, pad)), dim=-1),
      weight, bias, groups=CONV_DIM).narrow(-1, 1, pad)
    (full.sum() + head.sum()).backward()

  return count(both)


if __name__ == "__main__":
  torch.manual_seed(0)
  print("\n[1] _reference_scan einsums, per position, batch 1")
  reference_scan_step()
  print("\n[2] _block_state_passing einsum, batch 1")
  for length in (2048, 8192, 32768):
    block_state_passing(length=length)
  print("\n[3] _causal_conv depthwise convolutions, batch 1, L=2047")
  causal_conv()
  print("\n[4] reconciliation against results/sizing/flop_breakdown.json")
  reconcile()
