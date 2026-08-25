#!/usr/bin/env python3
"""Correctness gate for fusing `apply_rotary_pos_emb_torchscript` (dit.py:201).

The proposed change replaces 14 eager aten kernels with one compiled pointwise
region.  Numerics are FROZEN in this project, so the change ships only if this
test passes.  Two candidates are checked, both against the CURRENT function as
reference:

  A. compile-only      torch.compile(fullgraph=True) over the identical
                       expression.  Claim: BITWISE identical, forward and
                       gradient.  Inductor fuses pointwise chains without
                       reassociating them -- every output element is produced by
                       the same op sequence -- so this is expected to hold
                       exactly, and this test is the proof obligation.

  B. compile + skip v  additionally leaves the v slot untouched.  Rotary.forward
                       fills cos[:,:,2,:,:] = 1 and sin[:,:,2,:,:] = 0
                       (dit.py:175-176), so the current code computes
                       v*1 + rotate_half(v)*0 == v.  Claim: bitwise identical
                       for all finite v EXCEPT v == -0.0, where
                       (-0.0) + (+0.0) == +0.0 under round-to-nearest.  The test
                       reports that case separately rather than hiding it.

Bitwise comparison is done on the integer reinterpretation of the buffer, not
with allclose, because "provably equivalent" here means the trained checkpoint
must produce the SAME BITS.

Runs on CPU (bf16 and fp32) and, if a GPU is present, on CUDA as well.

Usage:  python scripts/smoke/test_rotary_fusion_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.dit import Rotary, rotate_half, apply_rotary_pos_emb_torchscript  # noqa: E402

INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
            torch.float32: torch.int32}


# ---- candidate A ----------------------------------------------------------
@torch.compile(fullgraph=True, dynamic=False)
def rope_compiled(qkv, cos, sin):
  return (qkv * cos) + (rotate_half(qkv) * sin)


# ---- candidate B ----------------------------------------------------------
@torch.compile(fullgraph=True, dynamic=False)
def rope_compiled_qk(qkv, cos, sin):
  qk = qkv[:, :, :2]
  out = (qk * cos[:, :, :2]) + (rotate_half(qk) * sin[:, :, :2])
  return torch.cat((out, qkv[:, :, 2:]), dim=2)


def bitwise_equal(a, b):
  if a.shape != b.shape or a.dtype != b.dtype:
    return False, "shape/dtype mismatch"
  iv = INT_VIEW[a.dtype]
  ia, ib = a.contiguous().view(iv), b.contiguous().view(iv)
  same = torch.equal(ia, ib)
  if same:
    return True, "bitwise identical"
  n = int((ia != ib).sum())
  ulp = int((ia.to(torch.int64) - ib.to(torch.int64)).abs().max())
  return False, f"{n}/{ia.numel()} elements differ, max {ulp} ULP"


def check(fn, name, device, dtype, B=2, L=256, heads=12, hd=64, seed=0,
          neg_zero=False):
  torch.manual_seed(seed)
  rot = Rotary(hd).to(device)
  probe = torch.zeros(B, L, heads * hd, device=device)
  cos, sin = rot(probe)
  cos, sin = cos.to(dtype), sin.to(dtype)

  base = torch.randn(B, L, 3, heads, hd, device=device, dtype=dtype)
  if neg_zero:
    base[:, :, 2, :, :hd // 4] = -0.0            # exercise the v == -0.0 case
  qkv_ref = base.clone().requires_grad_(True)
  qkv_new = base.clone().requires_grad_(True)
  g = torch.randn_like(base)

  ref = apply_rotary_pos_emb_torchscript(qkv_ref, cos, sin)
  ref.backward(g)
  new = fn(qkv_new, cos, sin)
  new.backward(g)

  ok_f, msg_f = bitwise_equal(ref.detach(), new.detach())
  ok_b, msg_b = bitwise_equal(qkv_ref.grad, qkv_new.grad)
  tag = f"{name:<18} {device:<5} {str(dtype).replace('torch.',''):<9}"
  suffix = "  [v = -0.0 probe]" if neg_zero else ""
  print(f"  {tag} fwd: {'PASS' if ok_f else 'FAIL'} ({msg_f}){suffix}")
  print(f"  {' '*len(tag)} bwd: {'PASS' if ok_b else 'FAIL'} ({msg_b})")
  return ok_f and ok_b


def main():
  devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
  results = []
  for device in devices:
    dtypes = [torch.float32] + ([torch.bfloat16] if device == "cuda" else
                                [torch.bfloat16])
    for dtype in dtypes:
      print(f"\n--- {device} / {dtype} ---")
      results.append(("A", check(rope_compiled, "A compile-only", device, dtype)))
      results.append(("B", check(rope_compiled_qk, "B compile+skip-v",
                                 device, dtype)))
      print("  (negative-zero probe: candidate B's one documented exception)")
      check(rope_compiled_qk, "B compile+skip-v", device, dtype, neg_zero=True)

  print("\n" + "=" * 70)
  a_ok = all(ok for tag, ok in results if tag == "A")
  b_ok = all(ok for tag, ok in results if tag == "B")
  print(f"  candidate A (compile-only)     : {'BITWISE SAFE' if a_ok else 'NOT SAFE'}")
  print(f"  candidate B (compile + skip v) : {'BITWISE SAFE' if b_ok else 'NOT SAFE'}"
        f"  (modulo v == -0.0, reported above)")
  print("\n  A trained checkpoint is unaffected either way: this function holds")
  print("  no parameters and no buffers, so nothing in the state_dict changes.")
  print("  Bitwise-identical forward + gradient is the full proof obligation.")
  print("\n  NOT PROVEN HERE: this must be re-run on the H200 before landing.")
  print("  CPU inductor and CUDA inductor emit different kernels; only the CUDA")
  print("  result licenses the change for the training runs.")
  return 0 if a_ok else 1


if __name__ == "__main__":
  sys.exit(main())
