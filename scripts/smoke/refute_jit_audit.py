#!/usr/bin/env python
"""Decompose the `nojit` win into its two independent halves.

`nojit` (scripts/smoke/dit_audit.py:140) does two things at once:

  1. it stops routing `aten::dropout` through TorchScript's autodiff, which
     retains a full-shape bool mask even at prob == 0.0;
  2. it stops TorchScript's autocast pass from wrapping `aten::mul` in
     `aten::_autocast_to_full_precision`, so the gate multiply drops from fp32
     to bf16 -- a precision change, not a wasted allocation.

Only (1) is free. This script adds two surgical variants so the two halves can
be attributed separately:

  `nomask`   keep @torch.jit.script (so the fp32 mul promotion survives) but
             route the dropout==0 call sites to a scripted gate+residual add
             with no dropout node at all.  Bit-identical by construction.
  `bf16mul`  keep the jit helpers exactly as they are, but cast the gate to
             fp32 ... not needed; instead this is the complement of `nomask`:
             `nojit` minus `nomask` == the precision change.
"""
from __future__ import annotations

import sys
import typing
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import dit_audit as da  # noqa: E402  (registers the round-2 dit patches)
import saved_tensor_audit as sta  # noqa: E402


@torch.jit.script
def _gate_residual_add_fused(x: torch.Tensor,
                             bias: typing.Optional[torch.Tensor],
                             scale: torch.Tensor,
                             residual: typing.Optional[torch.Tensor],
                             prob: float) -> torch.Tensor:
  """`bias_dropout_add_scale` specialised to prob == 0.0.

  Still @torch.jit.script, so TorchScript's autocast pass still inserts
  `aten::_autocast_to_full_precision` around the multiply exactly as the
  production helper does.  The only difference is that no `aten::dropout`
  node exists, so autodiff cannot emit `aten::native_dropout` and no bool
  mask is retained.
  """
  if bias is not None:
    x = x + bias
  out = scale * x
  if residual is not None:
    out = residual + out
  return out


def patch_nomask():
  """Drop the prob==0 dropout node, keep everything else (incl. the fp32 mul)."""
  from models import dit

  orig_train = dit.bias_dropout_add_scale_fused_train
  orig_infer = dit.bias_dropout_add_scale_fused_inference

  def train_fn(x, bias, scale, residual, prob):
    if prob == 0.0:
      return _gate_residual_add_fused(x, bias, scale, residual, prob)
    return orig_train(x, bias, scale, residual, prob)

  def infer_fn(x, bias, scale, residual, prob):
    if prob == 0.0:
      return _gate_residual_add_fused(x, bias, scale, residual, prob)
    return orig_infer(x, bias, scale, residual, prob)

  dit.bias_dropout_add_scale_fused_train = train_fn
  dit.bias_dropout_add_scale_fused_inference = infer_fn


sta.PATCHES.update({'nomask': patch_nomask})
sta.ORDER = list(sta.ORDER) + ['nomask']

if __name__ == '__main__':
  sta.main_cli()
