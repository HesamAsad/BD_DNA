#!/usr/bin/env python
"""Adversarial re-test of round-2 lead A(i): checkpointing `_causal_conv`.

The round-1/round-2 tables measured `ckpt` and `pad+ckpt` on the 12-layer AR
stack only, and against a `base` that was the *old* cat spelling.  Two claims
were then extrapolated rather than measured:

  1. "if you checkpoint, wrap the cat form, not the landed padded form"
     -- evidenced by pad+ckpt, which is the *discarded* `pad` variant (the one
     that rebuilds a full-width cat), not the landed `pad2`.
  2. "an additional -1.22 GiB on AR, -3.7 GiB on BiSSM"
     -- but BiSSM runs three conv sites per layer (`_causal_conv` in
     scan_bidirectional, `_reverse_causal_conv`, and the conv inlined in
     scan_with_block_boundaries) and the proposed fix wraps only the first.

This script measures both directly, on the *landed* code, using the production
`Diffusion._loss` step for the real-model rows (one process per row so absolute
peaks are comparable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/smoke"))

import conv_variants as cv  # noqa: E402
from models import mamba2_segment as m2  # noqa: E402


def old_causal_conv(self, xBC, initial_state, return_state=True):
  """The pre-fix cat spelling (the round-1 `base`)."""
  raw = xBC.transpose(1, 2)
  history = torch.cat((initial_state.to(dtype=raw.dtype), raw), dim=-1)
  convolved = F.conv1d(history, self.conv1d.weight, self.conv1d.bias,
                       groups=self.conv_dim)
  convolved = convolved.narrow(-1, 1, raw.shape[-1]).transpose(1, 2)
  out = F.silu(convolved)
  if not return_state:
    return out, None
  return out, history.narrow(
    -1, history.shape[-1] - self.d_conv, self.d_conv).contiguous()


def old_reverse_causal_conv(self, xBC, initial_state):
  raw = xBC.transpose(1, 2)
  seqlen = raw.shape[-1]
  history = torch.cat(
    (initial_state.to(dtype=raw.dtype), torch.flip(raw, dims=(-1,))), dim=-1)
  convolved = F.conv1d(history, self.conv1d.weight, self.conv1d.bias,
                       groups=self.conv_dim)
  return F.silu(convolved.narrow(-1, 1, seqlen).transpose(1, 2))


def checkpointed(self, xBC, initial_state, return_state=True):
  """Like conv_variants.causal_conv_checkpointed but accepts `return_state`,
  which scan_bidirectional (mamba2_segment.py:639) passes; the published arm
  does not, so it raises TypeError on the BiSSM path."""
  return torch.utils.checkpoint.checkpoint(
    self._causal_conv_inner, xBC, initial_state, return_state,
    use_reentrant=False)


def reverse_checkpointed(self, xBC, initial_state):
  return torch.utils.checkpoint.checkpoint(
    self._reverse_causal_conv_inner, xBC, initial_state, use_reentrant=False)


def patch(variant):
  """variant is a '+'-joined subset of {cat, ckpt, rckpt}; '' is the repo."""
  cls = m2.SegmentMamba2
  parts = set(p for p in variant.split("+") if p)
  cls._causal_conv_base = cls._causal_conv
  cls._reverse_causal_conv_base = cls._reverse_causal_conv
  inner = cls._causal_conv
  rinner = cls._reverse_causal_conv
  if "cat" in parts:
    inner, rinner = old_causal_conv, old_reverse_causal_conv
  cls._causal_conv_inner = inner
  cls._reverse_causal_conv_inner = rinner
  cls._causal_conv = checkpointed if "ckpt" in parts else inner
  cls._reverse_causal_conv = (
    reverse_checkpointed if "rckpt" in parts else rinner)
  return parts


def unpatch():
  cls = m2.SegmentMamba2
  cls._causal_conv = cls._causal_conv_base
  cls._reverse_causal_conv = cls._reverse_causal_conv_base
  for name in ("_causal_conv_base", "_reverse_causal_conv_base",
               "_causal_conv_inner", "_reverse_causal_conv_inner"):
    delattr(cls, name)


def equivalence(device):
  """fp32 max-abs diff of output/grads: landed vs cat, landed vs ckpt."""
  torch.manual_seed(0)
  mixer = m2.SegmentMamba2(d_model=768).to(device).float()
  xBC = torch.randn(2, 1024, mixer.conv_dim, device=device,
                    requires_grad=True)
  state = torch.randn(2, mixer.conv_dim, mixer.d_conv, device=device,
                      requires_grad=True)
  cotan = torch.randn(2, 1024, mixer.conv_dim, device=device)

  def run(variant):
    patch(variant)
    try:
      xBC.grad = state.grad = None
      mixer.zero_grad(set_to_none=True)
      out, final = mixer._causal_conv(xBC, state)
      out.backward(cotan)
      return (out.detach().clone(), final.detach().clone(),
              xBC.grad.clone(), state.grad.clone(),
              mixer.conv1d.weight.grad.clone())
    finally:
      unpatch()

  ref = run("")
  res = {}
  for variant in ("cat", "ckpt", "cat+ckpt"):
    got = run(variant)
    res[variant] = [float((a - b).abs().max()) for a, b in zip(ref, got)]
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=["stack", "real", "equiv"],
                  default="stack")
  ap.add_argument("--variant", default="")
  ap.add_argument("--arm", default="ussm-ar")
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--length", type=int, default=8192)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--output", type=Path, required=True)
  a = ap.parse_args()
  device = torch.device("cuda")
  a.output.parent.mkdir(parents=True, exist_ok=True)

  if a.mode == "equiv":
    row = {"mode": "equiv", "diffs": equivalence(device)}
  elif a.mode == "stack":
    rows = []
    for variant in a.variant.split(","):
      label = variant or "landed"
      if label == "attn":
        row = cv.run_stack("attn", a.batch, a.length, a.warmup, a.iters)
      else:
        patch(variant)
        try:
          row = cv.run_stack(label, a.batch, a.length, a.warmup, a.iters)
        except torch.cuda.OutOfMemoryError:
          row = {"arm": label, "oom": True}
        finally:
          unpatch()
          torch.cuda.empty_cache()
      rows.append(row)
      print(json.dumps(row), flush=True)
    row = {"mode": "stack", "rows": rows}
  else:
    import sizing_sweep as ss
    patch(a.variant)
    try:
      row = ss.run_case(a.arm, a.length, 256, a.batch, False,
                        a.warmup, a.iters, device)
    finally:
      unpatch()
    row["variant"] = a.variant or "landed"
  a.output.write_text(json.dumps(row, indent=2))
  print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
  main()
