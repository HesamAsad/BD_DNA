#!/usr/bin/env python
"""Independent refutation check for the dit.py LayerNorm fused-weight claim.

Claim under test (models/dit.py:229-237):

    def forward(self, x):
      with torch.amp.autocast('cuda', enabled=False):
        x = F.layer_norm(x.float(), [self.dim])
      return x * self.weight[None, None, :]

  keeps TWO full-size fp32 [B, L, D] tensors alive to backward:
    * native_layer_norm's _saved_input  (the x.float() copy)
    * MulBackward0's  _saved_self       (the normalised output)
  and folding the weight into F.layer_norm removes the second one, bit-
  identically.

Three parts, each independently checkable:
  A. saved-tensor census on a single LayerNorm at the real shapes
  B. numerical equivalence (forward bits, grad_x, grad_weight)
  C. end-to-end peak memory + step time on the real training step,
     baseline vs patched, for the dit-ar and dit arms
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import models.dit as dit_mod  # noqa: E402


BASELINE_FORWARD = dit_mod.LayerNorm.forward


def patched_forward(self, x):
  with torch.amp.autocast('cuda', enabled=False):
    return F.layer_norm(x.float(), [self.dim], weight=self.weight)


# --------------------------------------------------------------------------
# A. saved-tensor census
# --------------------------------------------------------------------------
def census(forward_fn, shape, in_dtype, device):
  """Bytes retained for backward by one LayerNorm call, deduped by storage."""
  ln = dit_mod.LayerNorm(shape[-1]).to(device)
  x = torch.randn(*shape, device=device, dtype=in_dtype, requires_grad=True)
  seen = {}

  def pack(t):
    if isinstance(t, torch.Tensor) and t.is_cuda:
      seen[t.untyped_storage().data_ptr()] = (
        tuple(t.shape), str(t.dtype), t.untyped_storage().nbytes())
    return t

  with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      y = forward_fn(ln, x)
    y.sum().backward()

  groups = collections.Counter()
  total = 0
  for shp, dt, nbytes in seen.values():
    groups[(shp, dt)] += nbytes
    total += nbytes
  return total, dict(groups), y.dtype


# --------------------------------------------------------------------------
# B. numerical equivalence
# --------------------------------------------------------------------------
def equivalence(shape, in_dtype, device):
  torch.manual_seed(0)
  w = torch.randn(shape[-1], device=device)
  x = torch.randn(*shape, device=device, dtype=in_dtype)

  g_ref = torch.randn(*shape, device=device, dtype=torch.float32)

  out = {}
  grads = {}
  for tag, fn in (("base", BASELINE_FORWARD), ("patched", patched_forward)):
    ln = dit_mod.LayerNorm(shape[-1]).to(device)
    with torch.no_grad():
      ln.weight.copy_(w)
    xi = x.clone().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      y = fn(ln, xi)
    y.backward(g_ref.to(y.dtype))
    out[tag] = y.detach()
    grads[tag] = (xi.grad.detach(), ln.weight.grad.detach())

  dy = (out["base"] - out["patched"]).abs().max().item()
  bit_identical = bool(torch.equal(out["base"], out["patched"]))
  gx = ((grads["base"][0] - grads["patched"][0]).norm()
        / grads["base"][0].norm().clamp_min(1e-30)).item()
  gw = ((grads["base"][1] - grads["patched"][1]).norm()
        / grads["base"][1].norm().clamp_min(1e-30)).item()
  return {
    "in_dtype": str(in_dtype), "out_dtype": str(out["base"].dtype),
    "max_abs_dy": dy, "forward_bit_identical": bit_identical,
    "grad_x_rel": gx, "grad_weight_rel": gw,
  }


# --------------------------------------------------------------------------
# C. end-to-end
# --------------------------------------------------------------------------
def end_to_end(arms, length, batch_size, block_size, warmup, iters, device):
  sys.path.insert(0, str(REPO / "scripts" / "smoke"))
  import sizing_sweep

  rows = []
  for variant in ("base", "patched"):
    dit_mod.LayerNorm.forward = (
      BASELINE_FORWARD if variant == "base" else patched_forward)
    for arm in arms:
      row = sizing_sweep.run_case(
        arm, length, block_size, batch_size, False, warmup, iters, device)
      row["variant"] = variant
      rows.append(row)
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats(device)
      print(f"{variant:<8}{arm:<9}{row.get('peak_gib') or -1:>10.2f}"
            f"{row.get('step_seconds') or -1:>10.4f}"
            f"{row.get('loss') or float('nan'):>12.5f}", flush=True)
  dit_mod.LayerNorm.forward = BASELINE_FORWARD
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--arms", default="dit-ar,dit")
  p.add_argument("--length", type=int, default=8192)
  p.add_argument("--batch-size", type=int, default=4)
  p.add_argument("--block-size", type=int, default=256)
  p.add_argument("--warmup", type=int, default=2)
  p.add_argument("--iters", type=int, default=5)
  p.add_argument("--skip-e2e", action="store_true")
  p.add_argument("--output", type=Path, default=None)
  args = p.parse_args()

  device = torch.device("cuda")
  print(f"device: {torch.cuda.get_device_name(device)}  torch {torch.__version__}\n")

  report = {}

  print("=== A. saved-tensor census, one LayerNorm call ===")
  for tag, shape in (("AR  (4, 8191, 832)", (4, 8191, 832)),
                     ("BD  (4, 16384, 768)", (4, 16384, 768))):
    for in_dtype in (torch.bfloat16, torch.float32):
      base_tot, base_g, base_dt = census(
        BASELINE_FORWARD, shape, in_dtype, device)
      pat_tot, pat_g, pat_dt = census(
        patched_forward, shape, in_dtype, device)
      print(f"{tag}  in={str(in_dtype).split('.')[-1]:<9} "
            f"out={str(base_dt).split('.')[-1]}")
      print(f"    base    {base_tot / 2**30:8.4f} GiB  {base_g}")
      print(f"    patched {pat_tot / 2**30:8.4f} GiB  {pat_g}")
      print(f"    delta   {(base_tot - pat_tot) / 2**30:8.4f} GiB/call"
            f"   x25 calls = {(base_tot - pat_tot) * 25 / 2**30:.3f} GiB")
      report[f"census/{tag}/{in_dtype}"] = {
        "base_bytes": base_tot, "patched_bytes": pat_tot,
        "base_groups": {str(k): v for k, v in base_g.items()},
        "patched_groups": {str(k): v for k, v in pat_g.items()},
      }
    print()

  print("=== B. numerical equivalence ===")
  for shape in ((4, 8191, 832), (4, 16384, 768)):
    for in_dtype in (torch.bfloat16, torch.float32):
      r = equivalence(shape, in_dtype, device)
      print(f"  {shape} {r['in_dtype']:<16} -> out {r['out_dtype']:<16} "
            f"bit-identical={r['forward_bit_identical']}  "
            f"max|dy|={r['max_abs_dy']:.3e}  "
            f"grad_x_rel={r['grad_x_rel']:.3e}  "
            f"grad_w_rel={r['grad_weight_rel']:.3e}")
      report[f"equiv/{shape}/{in_dtype}"] = r
  print()

  if not args.skip_e2e:
    print("=== C. end-to-end real training step ===")
    print(f"{'variant':<8}{'arm':<9}{'peak GiB':>10}{'step s':>10}{'loss':>12}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    report["e2e"] = end_to_end(
      arms, args.length, args.batch_size, args.block_size,
      args.warmup, args.iters, device)

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main()
