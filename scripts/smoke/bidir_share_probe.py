#!/usr/bin/env python
"""Cost of running the active block's two directions as two full mixer calls.

`BiMambaLayer.scan_active` used to call `scan_segment` once per direction on
the same normalized input. Only two of the mixer's four stages are actually
direction-dependent:

* `in_proj` is per-position, so the reverse direction's projection is a flip
  of the forward direction's -- computing it twice costs a second
  768 x 3224 GEMM and retains a second full-width projection for backward,
  plus the flipped copy of the layer input that `in_proj` saves;
* the depthwise convolution IS direction-dependent (forward output j spans
  raw[j-3..j], reverse spans raw[j..j+3]) but an anti-causal convolution is a
  causal one with a reversed kernel, so it can still read the shared
  projection;
* the scan is genuinely direction-dependent and cannot be shared: the SSD
  kernel is causal in memory order, so the reverse operands must be
  materialised reversed;
* `out_proj` is linear, so the two directions can be summed before it instead
  of after -- one retained input and one GEMM instead of two.

This measures a single `BiMambaLayer` at the real active-block geometry under
both `bidirectional_impl` settings: exact saved-tensor bytes (via an identity
`saved_tensors_hooks` pack hook, deduplicated by storage) and fwd/bwd time.
It also checks that the two spellings agree, in fp32 and under the bf16
autocast the real path runs in.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.bidirectional_ssm import BiMambaLayer  # noqa: E402
from models.mamba2_segment import Mamba2State  # noqa: E402

GEOMETRY = dict(
  dim=768, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=128,
  mlp_ratio=4.0, dropout=0.0, backend="auto")


def make_layer(impl, device, dtype=torch.float32):
  torch.manual_seed(0)
  return BiMambaLayer(bidirectional_impl=impl, **GEOMETRY).to(device, dtype)


def make_inputs(rows, length, device, dtype, layer):
  torch.manual_seed(1)
  x = torch.randn(rows, length, GEOMETRY["dim"], device=device, dtype=dtype)
  mixer = layer.mixer
  states = []
  for _ in range(2):
    states.append(Mamba2State(
      torch.randn(rows, mixer.conv_dim, mixer.d_conv,
                  device=device, dtype=dtype) * 0.1,
      torch.randn(rows, mixer.nheads, mixer.headdim, mixer.d_state,
                  device=device, dtype=dtype) * 0.1))
  return x, states[0], states[1]


class Audit(torch.autograd.graph.saved_tensors_hooks):
  """Identity pack hook recording retained storages, deduplicated by pointer."""

  def __init__(self, param_ptrs):
    self.records = {}
    self.param_ptrs = param_ptrs
    super().__init__(self._pack, lambda t: t)

  def _pack(self, t):
    storage = t.untyped_storage()
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if ptr in self.records:
      self.records[ptr]["hits"] += 1
      return t
    self.records[ptr] = {
      "bytes": nbytes, "dtype": str(t.dtype).replace("torch.", ""),
      "shape": tuple(t.shape), "site": call_site(),
      "is_param": ptr in self.param_ptrs, "hits": 1}
    return t


def call_site():
  frame = sys._getframe(2)
  while frame is not None:
    name = frame.f_code.co_filename
    if "bd3lms/models" in name or "mamba_ssm" in name:
      return f"{Path(name).name}:{frame.f_lineno} ({frame.f_code.co_name})"
    frame = frame.f_back
  return "<other>"


def audit_layer(impl, rows, length, device):
  layer = make_layer(impl, device)
  x, left, right = make_inputs(rows, length, device, torch.float32, layer)
  x.requires_grad_(True)
  ptrs = {p.untyped_storage().data_ptr() for p in layer.parameters()}
  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    layer.scan_active(x, left, right).sum().backward()  # warm up triton
  layer.zero_grad(set_to_none=True)
  torch.cuda.empty_cache()

  audit = Audit(ptrs)
  with audit, torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = layer.scan_active(x, left, right)
  torch.cuda.synchronize()
  live = [r for r in audit.records.values() if not r["is_param"]]
  total = sum(r["bytes"] for r in live)
  groups = collections.defaultdict(lambda: {"bytes": 0, "count": 0})
  for r in live:
    key = (r["site"], r["dtype"], r["shape"])
    groups[key]["bytes"] += r["bytes"]
    groups[key]["count"] += 1
  rows_out = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
  print(f"\n--- saved tensors, one BiMambaLayer.scan_active [{impl}] ---")
  print(f"retained: {total / 1024**2:.1f} MiB over {len(live)} storages")
  print(f"{'MiB':>9} {'n':>3}  {'dtype':<9} {'shape':<26} site")
  for (site, dtype, shape), g in rows_out:
    if g["bytes"] < 1024 ** 2:
      continue
    print(f"{g['bytes']/1024**2:9.1f} {g['count']:>3}  {dtype:<9} "
          f"{str(shape):<26} {site}")
  del out, layer
  torch.cuda.empty_cache()
  return {"impl": impl, "retained_bytes": total,
          "rows": [{"site": k[0], "dtype": k[1], "shape": list(k[2]),
                    "bytes": v["bytes"], "count": v["count"]}
                   for k, v in rows_out]}


def time_layer(impl, rows, length, device, warmup, iters):
  layer = make_layer(impl, device)
  x, left, right = make_inputs(rows, length, device, torch.float32, layer)
  x.requires_grad_(True)
  times, peaks = [], []
  for step in range(warmup + iters):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    layer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      out = layer.scan_active(x, left, right)
    out.float().square().sum().backward()
    torch.cuda.synchronize()
    if step >= warmup:
      times.append(time.perf_counter() - start)
      peaks.append(torch.cuda.max_memory_allocated(device))
    del out
  median = statistics.median(times)
  print(f"[{impl}] layer fwd+bwd {median*1000:8.3f} ms   "
        f"peak {statistics.median(peaks)/1024**2:8.1f} MiB")
  del layer
  torch.cuda.empty_cache()
  return {"impl": impl, "seconds": median,
          "peak_bytes": statistics.median(peaks)}


def equivalence(rows, length, device, dtype, autocast):
  split = make_layer("split", device, dtype)
  fused = make_layer("fused", device, dtype)
  fused.load_state_dict(split.state_dict())
  x, left, right = make_inputs(rows, length, device, dtype, split)
  outs, grads = [], []
  for layer in (split, fused):
    xi = x.detach().clone().requires_grad_(True)
    context = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if autocast
               else torch.amp.autocast("cuda", enabled=False))
    with context:
      out = layer.scan_active(xi, left, right)
    out.float().square().sum().backward()
    outs.append(out.detach().float())
    grads.append({n: p.grad.detach().float().clone()
                  for n, p in layer.named_parameters()} | {"x": xi.grad.float()})
  scale = outs[0].abs().max().item()
  dv = (outs[0] - outs[1]).abs().max().item()
  worst = max(((grads[0][n] - grads[1][n]).abs().max().item()
               / max(grads[0][n].abs().max().item(), 1e-30), n)
              for n in grads[0])
  tag = "bf16 autocast" if autocast else str(dtype).replace("torch.", "")
  print(f"[equivalence {tag:<14}] max|d out| {dv:.3e} (scale {scale:.3f}, "
        f"rel {dv/scale:.2e})   worst rel dgrad {worst[0]:.3e} @ {worst[1]}")
  del split, fused
  torch.cuda.empty_cache()
  return {"mode": tag, "abs_delta": dv, "scale": scale,
          "rel_delta": dv / scale, "worst_grad_rel": worst[0],
          "worst_grad_name": worst[1]}


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iters", type=int, default=10)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  device = torch.device("cuda")
  rows = args.batch_size * (args.length // args.block_size)
  print(f"device: {torch.cuda.get_device_name(device)}")
  print(f"active-block geometry: batch {args.batch_size} x L {args.length} "
        f"/ block {args.block_size} -> folded [{rows}, {args.block_size}], "
        f"{args.batch_size * args.length} tokens\n")

  results = {"rows": rows, "block_size": args.block_size,
             "tokens": args.batch_size * args.length}
  results["equivalence"] = [
    equivalence(rows, args.block_size, device, torch.float32, False),
    equivalence(rows, args.block_size, device, torch.float32, True),
  ]
  results["audit"] = [audit_layer(i, rows, args.block_size, device)
                      for i in ("split", "fused")]
  print()
  results["timing"] = [
    time_layer(i, rows, args.block_size, device, args.warmup, args.iters)
    for i in ("split", "fused")]

  a, b = results["audit"]
  ta, tb = results["timing"]
  saved = (a["retained_bytes"] - b["retained_bytes"]) / 1024 ** 2
  print(f"\nper layer: retained {a['retained_bytes']/1024**2:.1f} -> "
        f"{b['retained_bytes']/1024**2:.1f} MiB  ({saved:+.1f} MiB, "
        f"{-100*saved/(a['retained_bytes']/1024**2):+.1f}%)")
  print(f"12-layer stack: {12 * saved / 1024:+.3f} GiB")
  print(f"per layer fwd+bwd: {ta['seconds']*1000:.3f} -> "
        f"{tb['seconds']*1000:.3f} ms "
        f"({100*(ta['seconds']/tb['seconds'] - 1):+.1f}% faster)")

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
