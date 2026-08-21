#!/usr/bin/env python
"""Is the fused gated-RMSNorm tail numerically the same function, and what does
it buy on the real training path?

Three parts:

1. EQUIVALENCE. `SegmentMamba2._gated_output` (models/mamba2_segment.py:258)
   is exactly mamba_ssm's `rmsnorm_fn(..., norm_before_gate=False)`: both
   compute out_proj(RMSNorm(y * silu(z))). Compare outputs and every parameter
   gradient, in bf16 and in fp64-reference terms, before believing any memory
   number.

2. GRAPH WALK. Label every tensor the base mixer retains by the autograd node
   that saved it, so the census line items have names.

3. END TO END. Re-run the real `sizing_sweep` rows (Diffusion._loss -> backward
   -> AdamW) with and without the patch, so the saving is measured on the same
   code path that produced 27.19 GiB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from einops import rearrange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "smoke"))

from models import mamba2_segment as m2  # noqa: E402


def gated_output_fused(self, y, z):
  from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
  y = rearrange(y, "b l h p -> b l (h p)")
  y = rmsnorm_fn(y, self.norm_weight, None, z=z, eps=1e-5,
                 norm_before_gate=False)
  return self.out_proj(y)


# ------------------------------------------------------------- equivalence


def equivalence(batch=2, length=1024):
  """Both paths run exactly as production does: bf16 activations inside the
  bf16 autocast the backbone opens (models/bidirectional_ssm.py:312), and an
  fp32 no-autocast control."""
  import contextlib
  torch.manual_seed(0)
  mixer = m2.SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                           headdim=64, chunk_size=128).cuda()
  torch.nn.init.normal_(mixer.norm_weight, mean=1.0, std=0.05)
  results = {}
  for dtype in (torch.bfloat16, torch.float32):
    ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
           if dtype is torch.bfloat16 else contextlib.nullcontext())
    y0 = torch.randn(batch, length, 24, 64, device="cuda", dtype=dtype)
    z0 = torch.randn(batch, length, 1536, device="cuda", dtype=dtype)
    outs, grads = [], []
    for fn in (m2.SegmentMamba2._gated_output, gated_output_fused):
      y = y0.clone().requires_grad_(True)
      z = z0.clone().requires_grad_(True)
      mixer.zero_grad(set_to_none=True)
      with ctx:
        out = fn(mixer, y, z)
      out.float().pow(2).mean().backward()
      outs.append(out.detach().float())
      grads.append((y.grad.float(), z.grad.float(),
                    mixer.norm_weight.grad.float().clone(),
                    mixer.out_proj.weight.grad.float().clone()))
    # fp64 reference for the same function, so "they differ" can be scored
    # as "which one is closer to the truth".
    with torch.no_grad():
      yr = rearrange(y0.double(), "b l h p -> b l (h p)")
      zr = z0.double()
      t = yr * torch.nn.functional.silu(zr)
      t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)
      ref = torch.nn.functional.linear(
        t * mixer.norm_weight.double(), mixer.out_proj.weight.double()).float()
    scale = outs[0].abs().mean().item()
    row = {"out_max_abs_diff": (outs[0] - outs[1]).abs().max().item(),
           "out_mean_abs": scale,
           "base_vs_fp64_max": (outs[0] - ref).abs().max().item(),
           "fused_vs_fp64_max": (outs[1] - ref).abs().max().item()}
    for name, a, b in zip(("dy", "dz", "dnorm_weight", "dout_proj"),
                          grads[0], grads[1]):
      row[f"{name}_max_abs_diff"] = (a - b).abs().max().item()
      row[f"{name}_mean_abs"] = a.abs().mean().item()
    results[str(dtype)] = row
  del mixer
  torch.cuda.empty_cache()
  return results


# -------------------------------------------------------------- graph walk


def graph_walk(batch=4, length=8192):
  torch.manual_seed(0)
  mixer = m2.SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                           headdim=64, chunk_size=128).cuda()
  params = {p.untyped_storage().data_ptr() for p in mixer.parameters()}
  u = torch.randn(batch, length, 768, device="cuda", dtype=torch.bfloat16,
                  requires_grad=True)
  params.add(u.untyped_storage().data_ptr())
  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out, _ = mixer.scan_segment(u)

  rows, seen_nodes, seen_storage = [], set(), set()
  stack = [out.grad_fn]
  while stack:
    node = stack.pop()
    if node is None or node in seen_nodes:
      continue
    seen_nodes.add(node)
    for attr in dir(node):
      if not attr.startswith("_saved"):
        continue
      try:
        value = getattr(node, attr)
      except Exception:
        continue
      tensors = value if isinstance(value, tuple) else (value,)
      for t in tensors:
        if not torch.is_tensor(t) or not t.is_cuda or t.numel() <= 4096:
          continue
        key = t.untyped_storage().data_ptr()
        if key in params or key in seen_storage:
          continue
        seen_storage.add(key)
        rows.append({
          "node": type(node).__name__, "attr": attr,
          "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
          "storage_MiB": round(t.untyped_storage().nbytes() / 2**20, 1)})
    stack.extend(child for child, _ in node.next_functions)
  rows.sort(key=lambda r: -r["storage_MiB"])
  del out, u, mixer
  torch.cuda.empty_cache()
  return rows


# --------------------------------------------------------------- end to end


def end_to_end(arms, batch, length, block_size, warmup, iters, patched):
  import sizing_sweep
  if patched:
    m2.SegmentMamba2._gated_output = gated_output_fused
  device = torch.device("cuda")
  rows = []
  for arm in arms:
    row = sizing_sweep.run_case(arm, length, block_size, batch, False,
                                warmup, iters, device)
    row["patched"] = patched
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in row.items()}), flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--length", type=int, default=8192)
  ap.add_argument("--block-size", type=int, default=256)
  ap.add_argument("--arms", default="ussm-ar,bissm")
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--output",
                  default=str(REPO / "results/sizing/gate_verify.json"))
  args = ap.parse_args()

  out = {"config": vars(args)}
  out["equivalence"] = equivalence()
  print("=== equivalence ===")
  print(json.dumps(out["equivalence"], indent=1), flush=True)

  out["graph_walk"] = graph_walk(args.batch, args.length)
  print("\n=== graph walk (base mixer, b=%d L=%d) ===" % (args.batch,
                                                          args.length))
  for r in out["graph_walk"]:
    print(f"  {r['storage_MiB']:8.1f} MiB  {r['dtype']:9s} "
          f"{str(r['shape']):22s} {r['node']}.{r['attr']}", flush=True)

  arms = [a.strip() for a in args.arms.split(",") if a.strip()]
  out["end_to_end"] = []
  print("\n=== end to end: baseline ===", flush=True)
  out["end_to_end"] += end_to_end(arms, args.batch, args.length,
                                  args.block_size, args.warmup, args.iters,
                                  patched=False)
  print("\n=== end to end: fused gated RMSNorm ===", flush=True)
  out["end_to_end"] += end_to_end(arms, args.batch, args.length,
                                  args.block_size, args.warmup, args.iters,
                                  patched=True)

  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  Path(args.output).write_text(json.dumps(out, indent=2))
  print("wrote", args.output)


if __name__ == "__main__":
  main()
