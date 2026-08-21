#!/usr/bin/env python
"""Which retained tensors make a Mamba-2 layer cost 1.69x a causal attention layer?

Two measurements, both on one GPU:

1. A saved-tensor census. `torch.autograd.graph.saved_tensors_hooks` intercepts
   every tensor the graph retains during a forward pass of ONE mixer, keyed by
   storage pointer so views are not double counted and parameters are excluded.
   This turns "the SSM keeps more" into a line-item list.

2. Peak allocated memory + step time for a 12-layer stack, fwd+bwd+AdamW, under
   variants that change only how the fused SSD kernel is wrapped:
     base        - the repo as-is
     gate        - _gated_output via mamba_ssm's fused gated RMSNorm kernel
     conv        - the causal conv recomputed in backward instead of stored
     cs256       - chunk_size 256 instead of 128
     nostate     - initial_states=None instead of an explicit zero tensor
     gate+conv+nostate - all of the above that help
   plus `attn`, a causal flash-attention block of the parameter-matched width
   (d=832, 13 heads) for calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import models.bidirectional_ssm as bs  # noqa: E402
from models import mamba2_segment as m2  # noqa: E402


# ---------------------------------------------------------------- variants


def gated_output_fused(self, y, z):
  from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
  y = rearrange(y, "b l h p -> b l (h p)")
  y = rmsnorm_fn(y, self.norm_weight, None, z=z, eps=1e-5,
                 norm_before_gate=False)
  return self.out_proj(y)


def causal_conv_checkpointed(self, xBC, initial_state):
  return torch.utils.checkpoint.checkpoint(
    self._causal_conv_raw, xBC, initial_state, use_reentrant=False)


def fused_scan_nostate(self, x, dt, B, C, initial_state):
  if initial_state is not None and not initial_state.any():
    initial_state = None
  return m2.mamba_chunk_scan_combined(
    x, dt, -torch.exp(self.A_log.float()), B, C,
    chunk_size=self.chunk_size, D=self.D, dt_bias=self.dt_bias,
    initial_states=initial_state, dt_softplus=True, return_final_states=True)


def patch_class(variant):
  """Patch SegmentMamba2 in place; call restore_class() afterwards."""
  cls = m2.SegmentMamba2
  parts = set(variant.split("+"))
  if "gate" in parts:
    cls._gated_output_orig = cls._gated_output
    cls._gated_output = gated_output_fused
  if "conv" in parts:
    cls._causal_conv_raw = cls._causal_conv
    cls._causal_conv = causal_conv_checkpointed
  if "nostate" in parts:
    cls._fused_scan_orig = cls._fused_scan
    cls._fused_scan = fused_scan_nostate
  return parts


def restore_class():
  cls = m2.SegmentMamba2
  for patched, original in (("_gated_output", "_gated_output_orig"),
                            ("_causal_conv", "_causal_conv_raw"),
                            ("_fused_scan", "_fused_scan_orig")):
    if hasattr(cls, original):
      setattr(cls, patched, getattr(cls, original))
      delattr(cls, original)


def set_chunk_size(module, variant):
  if "cs256" in variant.split("+"):
    for sub in module.modules():
      if isinstance(sub, m2.SegmentMamba2):
        sub.chunk_size = 256


# ---------------------------------------------------------------- models


class SSMStack(nn.Module):
  def __init__(self, n_layers=12, dim=768, vocab=13):
    super().__init__()
    self.embed = nn.Embedding(vocab, dim)
    self.layers = nn.ModuleList([
      bs.BiMambaLayer(dim=dim, d_state=64, d_conv=4, expand=2, headdim=64,
                      chunk_size=128, mlp_ratio=4.0, dropout=0.0,
                      backend="auto")
      for _ in range(n_layers)])
    self.norm = bs.RMSNorm(dim)
    self.head = nn.Linear(dim, vocab, bias=False)

  def forward(self, idx):
    x = self.embed(idx)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      for layer in self.layers:
        x, _ = layer.scan_clean(x, None)
      x = self.norm(x)
    return self.head(x.float())


class AttnBlock(nn.Module):
  def __init__(self, dim, n_heads):
    super().__init__()
    self.n_heads = n_heads
    self.norm1 = bs.RMSNorm(dim)
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.proj = nn.Linear(dim, dim, bias=False)
    self.norm2 = bs.RMSNorm(dim)
    self.mlp = bs.FeedForward(dim, 4.0, 0.0)

  def forward(self, x):
    import flash_attn
    qkv = self.qkv(self.norm1(x))
    qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3,
                    h=self.n_heads)
    y = flash_attn.flash_attn_qkvpacked_func(qkv, causal=True)
    x = x + self.proj(rearrange(y, "b s h d -> b s (h d)"))
    return x + self.mlp(self.norm2(x))


class AttnStack(nn.Module):
  def __init__(self, n_layers=12, dim=832, n_heads=13, vocab=13):
    super().__init__()
    self.embed = nn.Embedding(vocab, dim)
    self.layers = nn.ModuleList(
      [AttnBlock(dim, n_heads) for _ in range(n_layers)])
    self.norm = bs.RMSNorm(dim)
    self.head = nn.Linear(dim, vocab, bias=False)

  def forward(self, idx):
    x = self.embed(idx)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      for layer in self.layers:
        x = layer(x)
      x = self.norm(x)
    return self.head(x.float())


# ---------------------------------------------------------------- census


def census(batch, length, variant, kind="mixer"):
  """Per-tensor list of everything one layer's forward retains."""
  torch.manual_seed(0)
  if kind == "mixer":
    module = m2.SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                              headdim=64, chunk_size=128).cuda()
    dim = 768
  else:
    module = AttnBlock(832, 13).cuda()
    dim = 832
  set_chunk_size(module, variant)
  params = {p.untyped_storage().data_ptr() for p in module.parameters()}
  u = torch.randn(batch, length, dim, device="cuda", dtype=torch.bfloat16,
                  requires_grad=True)
  params.add(u.untyped_storage().data_ptr())
  seen, order = {}, []

  def pack(t):
    if t.is_cuda and t.numel() > 4096:
      key = t.untyped_storage().data_ptr()
      if key not in seen and key not in params:
        nbytes = t.untyped_storage().nbytes()
        seen[key] = nbytes
        order.append((tuple(t.shape), str(t.dtype).replace("torch.", ""),
                      nbytes / 2**20))
    return t

  torch.cuda.synchronize()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  base = torch.cuda.memory_allocated()
  with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      if kind == "mixer":
        out, _ = module.scan_segment(u)
      else:
        out = module(u)
  torch.cuda.synchronize()
  live = (torch.cuda.memory_allocated() - base) / 2**20
  transient_peak = (torch.cuda.max_memory_allocated() - base) / 2**20
  result = {
    "unique_saved_MiB": round(sum(seen.values()) / 2**20, 1),
    "live_after_forward_MiB": round(live, 1),
    "forward_peak_MiB": round(transient_peak, 1),
    "tensors": [{"shape": list(s), "dtype": d, "MiB": round(m, 1)}
                for s, d, m in order],
  }
  del out, u, module
  torch.cuda.empty_cache()
  return result


# ---------------------------------------------------------------- step loop


def run_stack(arm, batch, length, warmup, iters):
  torch.manual_seed(0)
  if arm == "attn":
    model = AttnStack().cuda()
  else:
    model = SSMStack().cuda()
    set_chunk_size(model, arm)
  opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
  idx = torch.randint(0, 13, (batch, length), device="cuda")
  target = torch.randint(0, 13, (batch, length), device="cuda")
  times = []
  torch.cuda.empty_cache()
  for i in range(warmup + iters):
    if i == warmup:
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, 13), target.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if i >= warmup:
      times.append(time.time() - t0)
  peak = torch.cuda.max_memory_allocated() / 2**30
  n_params = sum(p.numel() for p in model.parameters())
  del model, opt, logits, loss
  torch.cuda.empty_cache()
  times.sort()
  median = times[len(times) // 2]
  return {"arm": arm, "params_M": round(n_params / 1e6, 2),
          "peak_GiB": round(peak, 3), "step_s": round(median, 4),
          "tok_per_s": round(batch * length / median)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--length", type=int, default=8192)
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=3)
  ap.add_argument("--arms", default="base,gate,conv,cs256,nostate,"
                                    "gate+conv+nostate,attn")
  ap.add_argument("--census", default="base,gate+conv,attn")
  ap.add_argument("--output",
                  default=str(REPO / "results/sizing/argprobe.json"))
  args = ap.parse_args()

  print(torch.cuda.get_device_name(0), flush=True)
  print("fused available:", m2.fused_mamba2_available(), flush=True)

  out = {"config": vars(args), "census": {}, "stack": []}
  for variant in args.census.split(","):
    if not variant:
      continue
    restore_class()
    kind = "attn" if variant == "attn" else "mixer"
    if kind == "mixer":
      patch_class(variant)
    out["census"][variant] = census(args.batch, args.length, variant, kind)
    print(f"\n=== census {variant} ===", flush=True)
    print(json.dumps(out["census"][variant], indent=1), flush=True)
  restore_class()

  for arm in args.arms.split(","):
    if not arm:
      continue
    restore_class()
    if arm != "attn":
      patch_class(arm)
    try:
      row = run_stack(arm, args.batch, args.length, args.warmup, args.iters)
    except torch.cuda.OutOfMemoryError:
      row = {"arm": arm, "oom": True}
    torch.cuda.empty_cache()
    out["stack"].append(row)
    print(json.dumps(row), flush=True)
  restore_class()

  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  Path(args.output).write_text(json.dumps(out, indent=2))
  print("wrote", args.output)


if __name__ == "__main__":
  main()
