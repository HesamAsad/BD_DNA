#!/usr/bin/env python
"""Enumerate every tensor the autograd graph retains in one training forward.

Uses ``torch.autograd.graph.saved_tensors_hooks`` with an identity pack hook,
so the numerics are untouched, and records for each *storage* (deduplicated by
data_ptr, so views count once) its size, dtype, shape and the innermost
project/library call site that saved it.  That gives an exact, attributed
breakdown of activation memory -- which is what peak memory is made of once
parameters, gradients and optimizer state are subtracted.

Also supports patching in candidate memory fixes (``--variants``) and
reporting the real end-to-end peak of a full fwd+bwd+AdamW step for each, so
the accounting can be checked against ``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

import main  # noqa: F401 - registers OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion
from mamba_ssm.ops.triton.ssd_combined import (
  mamba_chunk_scan_combined as _ORIGINAL_SCAN)

REPO = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO))

ARMS = {
  "ussm-ar": ("small_ussm", "ar"),
  "dit-ar": ("small_ar_transformer", "ar"),
  "bissm": ("small_bissm", "bd3lm_bissm"),
  "dit": ("small", "bd3lm"),
}


def build(arm, length, batch_size, block_size, extra=()):
  model_cfg, algo_cfg = ARMS[arm]
  is_ar = algo_cfg == "ar"
  overrides = [
    f"model={model_cfg}",
    f"algo={algo_cfg}",
    "data=carbon-prokaryote",
    f"model.length={length}",
    f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch_size}",
    f"loader.eval_batch_size={batch_size}",
    "loader.global_batch_size=64",
    "training.ema=0",
    "trainer.accumulate_grad_batches=1",
  ]
  if arm == "ussm-ar":
    overrides.append("algo.backbone=ussm")
  if arm == "dit":
    overrides.append("model.attn_backend=flex")
  if arm in {"bissm"}:
    overrides.append("model.active_blocks=all")
  overrides.extend(extra)
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=overrides)


INTERESTING = ("/models/", "/diffusion.py", "mamba_ssm",
               "flash_attn", "torch/nn/functional")


def call_site():
  """Innermost frame belonging to the model code, as 'file.py:line'."""
  frame = sys._getframe(2)
  best = None
  while frame is not None:
    name = frame.f_code.co_filename
    if any(tag in name for tag in INTERESTING):
      best = f"{Path(name).name}:{frame.f_lineno} ({frame.f_code.co_name})"
      break
    frame = frame.f_back
  return best or "<other>"


class Audit(torch.autograd.graph.saved_tensors_hooks):
  """Identity hook that records what gets saved, deduplicated by storage."""

  def __init__(self, param_ptrs):
    self.records = {}
    self.param_ptrs = param_ptrs
    super().__init__(self._pack, lambda t: t)

  def _pack(self, t):
    try:
      storage = t.untyped_storage()
      ptr, nbytes = storage.data_ptr(), storage.nbytes()
    except Exception:
      ptr, nbytes = t.data_ptr(), t.numel() * t.element_size()
    if ptr in self.records:
      self.records[ptr]["hits"] += 1
      return t
    self.records[ptr] = {
      "bytes": nbytes,
      "dtype": str(t.dtype).replace("torch.", ""),
      "shape": tuple(t.shape),
      "site": call_site(),
      "is_param": ptr in self.param_ptrs,
      "hits": 1,
    }
    return t


def summarize(records, label, top=40):
  live = [r for r in records.values() if not r["is_param"]]
  total = sum(r["bytes"] for r in live)
  by_dtype = collections.Counter()
  for r in live:
    by_dtype[r["dtype"]] += r["bytes"]
  groups = collections.defaultdict(
    lambda: {"bytes": 0, "count": 0, "dtype": "", "shape": ()})
  for r in live:
    key = (r["site"], r["dtype"], r["shape"])
    g = groups[key]
    g["bytes"] += r["bytes"]
    g["count"] += 1
    g["dtype"] = r["dtype"]
    g["shape"] = r["shape"]
  rows = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
  print(f"\n=== saved-tensor audit: {label} ===")
  print(f"total retained activations: {total / 1024**3:.3f} GiB "
        f"({len(live)} distinct storages)")
  for dt, b in by_dtype.most_common():
    print(f"  {dt:>10}: {b / 1024**3:8.3f} GiB  ({100*b/max(total,1):5.1f}%)")
  print(f"\n{'GiB':>8} {'n':>4}  {'dtype':<9} {'shape':<28} site")
  for (site, dtype, shape), g in rows[:top]:
    print(f"{g['bytes']/1024**3:8.3f} {g['count']:>4}  {dtype:<9} "
          f"{str(shape):<28} {site}")
  return {"total_bytes": total,
          "by_dtype": dict(by_dtype),
          "rows": [{"site": k[0], "dtype": k[1], "shape": list(k[2]),
                    "bytes": v["bytes"], "count": v["count"]}
                   for k, v in rows]}


# --------------------------------------------------------------------------
# Candidate fixes.  Each is a monkeypatch applied before the model is built.
# --------------------------------------------------------------------------

def patch_rmsnorm_checkpoint():
  """Recompute RMSNorm in backward instead of retaining its fp32 copy."""
  from models import bidirectional_ssm as bs
  original = bs.RMSNorm.forward

  def forward(self, x):
    if torch.is_grad_enabled() and x.requires_grad:
      return torch.utils.checkpoint.checkpoint(
        original, self, x, use_reentrant=False)
    return original(self, x)

  bs.RMSNorm.forward = forward


def patch_gated_checkpoint():
  """Recompute the SiLU-gate + gated RMSNorm tail in backward."""
  from models import mamba2_segment as ms
  original = ms.SegmentMamba2._gated_output

  def _gated_output(self, y, z):
    if torch.is_grad_enabled() and y.requires_grad:
      return torch.utils.checkpoint.checkpoint(
        original, self, y, z, use_reentrant=False)
    return original(self, y, z)

  ms.SegmentMamba2._gated_output = _gated_output


def patch_conv_checkpoint():
  """Recompute the depthwise causal conv (and its history cat) in backward."""
  from models import mamba2_segment as ms
  original = ms.SegmentMamba2._causal_conv

  def _causal_conv(self, xBC, initial_state):
    if torch.is_grad_enabled() and xBC.requires_grad:
      return torch.utils.checkpoint.checkpoint(
        original, self, xBC, initial_state, use_reentrant=False)
    return original(self, xBC, initial_state)

  ms.SegmentMamba2._causal_conv = _causal_conv


def patch_rsqrt_dtype():
  """Undo autocast's silent fp32 promotion of ``torch.rsqrt``.

  Under a bf16 autocast ``torch.rsqrt`` is on the fp32 promotion list, so it
  returns fp32 even though both call sites explicitly downcast the variance to
  the activation dtype first.  The fp32 result then promotes the normalized
  activation to fp32, doubling it.  Running the rsqrt with autocast disabled
  restores what ``variance.to(dtype=x.dtype)`` plainly intends.
  """
  from models import bidirectional_ssm as bs
  from models import mamba2_segment as ms

  def _rsqrt(t):
    if t.is_cuda:
      with torch.amp.autocast("cuda", enabled=False):
        return torch.rsqrt(t)
    return torch.rsqrt(t)

  def forward(self, x):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * _rsqrt(variance.to(dtype=x.dtype) + self.eps)
    return normalized * self.weight.to(dtype=x.dtype)

  def _gated_output(self, y, z):
    y = rearrange(y, "b l h p -> b l (h p)")
    y = y * F.silu(z)
    variance = y.float().pow(2).mean(dim=-1, keepdim=True)
    y = y * _rsqrt(variance.to(dtype=y.dtype) + 1e-5)
    y = y * self.norm_weight.to(dtype=y.dtype)
    return self.out_proj(y)

  bs.RMSNorm.forward = forward
  ms.SegmentMamba2._gated_output = _gated_output


def patch_split_proj():
  """Stop ``z``/``dt`` views from pinning the whole in_proj output alive.

  ``z`` and ``dt`` are views of the 3224-wide projection, and both are saved
  for backward, so the 1664-wide ``xBC`` slice -- dead once the convolution has
  copied it -- stays resident for the whole step.  Making the two saved slices
  their own storages lets the projection be freed.
  """
  from models import mamba2_segment as ms

  def scan_segment(self, u, initial_state=None):
    batch_size = u.shape[0]
    if initial_state is None:
      initial_state = self.zero_state(
        batch_size, device=u.device, dtype=u.dtype)
    zxbcdt = self.in_proj(u)
    z, xBC, dt = torch.split(
      zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1)
    z, dt = z.contiguous(), dt.contiguous()
    del zxbcdt
    xBC, final_conv = self._causal_conv(xBC, initial_state.conv)
    x, B, C = torch.split(
      xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
    B = rearrange(B, "b l n -> b l 1 n")
    C = rearrange(C, "b l n -> b l 1 n")
    y, final_ssm = self._scan(
      x, dt, B, C, initial_state.ssm, self._select_backend(u))
    return self._gated_output(y, z), ms.Mamba2State(final_conv, final_ssm)

  ms.SegmentMamba2.scan_segment = scan_segment


def patch_xbc_contiguous():
  """Give ``xBC`` its own storage before it is retained.

  Must be applied *after* the conv patch: with the convolution recomputed, the
  checkpoint retains its input ``xBC``, which is a 1664-wide view and therefore
  still pins the whole 3224-wide projection.
  """
  from models import mamba2_segment as ms
  original = ms.SegmentMamba2._causal_conv

  def _causal_conv(self, xBC, initial_state):
    return original(self, xBC.contiguous(), initial_state)

  ms.SegmentMamba2._causal_conv = _causal_conv


def dtype_probe(device):
  """Confirm that autocast promotes ``torch.rsqrt`` to fp32."""
  x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
  with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    v = x.float().pow(2).mean(-1, keepdim=True)
    r = torch.rsqrt(v.to(dtype=x.dtype) + 1e-5)
    n = x * r
    with torch.amp.autocast("cuda", enabled=False):
      r2 = torch.rsqrt(v.to(dtype=x.dtype) + 1e-5)
  print(f"[dtype probe] x={x.dtype} variance.to(x.dtype)={v.to(x.dtype).dtype} "
        f"rsqrt(bf16)={r.dtype} -> x*r={n.dtype} | "
        f"rsqrt with autocast disabled={r2.dtype}")


def patch_dead_saves():
  """Lead C: stop the SSD scan retaining tensors its backward never reads.

  ``MambaChunkScanCombinedFn.forward`` (ssd_combined.py:606) saves
  ``out if z is None else out_x`` and ``dA_cumsum``.  When ``z is None`` --
  which is every call this repo makes, because the gate lives outside the
  scan -- the backward never touches either: ``out`` only reaches
  ``outz = out`` (ssd_combined.py:454) which is returned only under
  ``recompute_output``, false on this path, and ``dA_cumsum`` is recomputed
  from scratch at ssd_combined.py:441 and the saved copy is never passed to
  ``_mamba_chunk_scan_combined_bwd`` at all (ssd_combined.py:624).
  """
  from mamba_ssm.ops.triton import ssd_combined as sc

  fwd = sc._mamba_chunk_scan_combined_fwd
  bwd = sc._mamba_chunk_scan_combined_bwd

  class Lean(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None, cu_seqlens=None,
                dt_softplus=False, dt_limit=(0.0, float("inf")),
                return_final_states=False, return_varlen_states=False,
                state_dtype=None):
      ctx.dt_dtype = dt.dtype
      assert not return_varlen_states
      cu_seqlens = None
      out, out_x, _dt, dA_cumsum, _states, final_states = fwd(
        x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus, dt_limit=dt_limit, state_dtype=state_dtype)
      # A zero-element tensor stands in for each dead save so the unpack in
      # backward keeps its arity.
      dead = out.new_empty(0)
      ctx.save_for_backward(
        dead if z is None else out_x, x, dt, dead if z is None else dA_cumsum,
        A, B, C, D, z, dt_bias, initial_states, seq_idx)
      ctx.dt_softplus = dt_softplus
      ctx.chunk_size = chunk_size
      ctx.dt_limit = dt_limit
      ctx.return_final_states = return_final_states
      ctx.state_dtype = state_dtype
      return out if not return_final_states else (out, final_states)

    @staticmethod
    def backward(ctx, dout, *args):
      (out, x, dt, _dA_cumsum, A, B, C, D, z, dt_bias, initial_states,
       seq_idx) = ctx.saved_tensors
      dfinal_states = args[0] if ctx.return_final_states else None
      (dx, ddt, dA, dB, dC, dD, dz, ddt_bias,
       dinitial_states) = bwd(
        dout, x, dt, A, B, C, x if z is None else out, ctx.chunk_size,
        D=D, z=z, dt_bias=dt_bias,
        initial_states=initial_states, dfinal_states=dfinal_states,
        seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit,
        state_dtype=ctx.state_dtype)
      return (dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states,
              None, None, None, None, None, None, None)

  def lean_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None, cu_seqlens=None,
                dt_softplus=False, dt_limit=(0.0, float("inf")),
                return_final_states=False, return_varlen_states=False,
                state_dtype=None):
    return Lean.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias,
                      initial_states, seq_idx, cu_seqlens, dt_softplus,
                      dt_limit, return_final_states, return_varlen_states,
                      state_dtype)

  sc.mamba_chunk_scan_combined = lean_scan
  from models import mamba2_segment as ms
  ms.mamba_chunk_scan_combined = lean_scan


def patch_fused_rmsnorm():
  """Route the plain RMSNorm through mamba_ssm's fused kernel.

  The eager body (bidirectional_ssm.py:34-37) retains two fp32 copies of the
  activation per call: ``x.float()`` is kept by ``PowBackward0`` and the
  ``x * rsqrt(...)`` product is kept by the weight multiply.  ``rmsnorm_fn``
  with ``z=None`` keeps only the bf16 input and the per-row ``rstd``, and
  recomputes the rest in its backward.  This is the round-1 gate fix applied
  to the two ungated norms per layer plus the final norm.
  """
  from models import bidirectional_ssm as bs
  from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rms

  def forward(self, x):
    if x.is_cuda:
      return _rms(x, self.weight, None, eps=self.eps)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
    return normalized * self.weight.to(dtype=x.dtype)

  bs.RMSNorm.forward = forward


def patch_rsqrt_only():
  """Cheapest half of the norm fix: keep the eager body, drop the promotion.

  ``torch.rsqrt`` is on autocast's fp32 promotion list, so under a bf16
  autocast it returns fp32 even though the call site downcasts the variance
  first, and the fp32 result then promotes the normalized activation.  Unlike
  ``patch_rsqrt_dtype`` this touches only ``RMSNorm``, leaving the already
  fused gated output alone.
  """
  from models import bidirectional_ssm as bs

  def forward(self, x):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    scale = variance.to(dtype=x.dtype) + self.eps
    if scale.is_cuda:
      with torch.amp.autocast("cuda", enabled=False):
        scale = torch.rsqrt(scale)
    else:
      scale = torch.rsqrt(scale)
    return (x * scale) * self.weight.to(dtype=x.dtype)

  bs.RMSNorm.forward = forward


def patch_norm_input_cast():
  """Retain a bf16 copy of the norm input instead of pinning the fp32 stream.

  The residual stream is fp32 -- ``token_embedding`` runs under
  ``Diffusion.forward``'s fp32 autocast and ``x + mixed`` then promotes -- so
  the norm's saved input is a 96 MiB fp32 tensor per call.  Casting only the
  norm's argument leaves the residual accumulation in fp32 and lets the fp32
  tensor die as soon as the next add consumes it.
  """
  from models import bidirectional_ssm as bs
  from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rms

  def forward(self, x):
    if x.is_cuda:
      if x.dtype == torch.float32:
        x = x.to(torch.bfloat16)
      return _rms(x, self.weight, None, eps=self.eps)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
    return normalized * self.weight.to(dtype=x.dtype)

  bs.RMSNorm.forward = forward


def patch_bf16_residual():
  """Carry the whole residual stream in bf16 rather than fp32."""
  from models import bidirectional_ssm as bs

  def wrap(original):
    def entry(self, x, *args, **kwargs):
      if x.is_cuda and x.dtype == torch.float32:
        x = x.to(torch.bfloat16)
      return original(self, x, *args, **kwargs)
    return entry

  for name in ("scan_clean", "scan_clean_with_boundaries", "scan_active"):
    setattr(bs.BiMambaLayer, name, wrap(getattr(bs.BiMambaLayer, name)))


def reset_library_patches():
  """Undo patches that live outside ``models``/``diffusion``.

  The variant loop reimports the project modules but not ``mamba_ssm``, so a
  library-level monkeypatch would leak into every later variant.
  """
  from mamba_ssm.ops.triton import ssd_combined as sc
  sc.mamba_chunk_scan_combined = _ORIGINAL_SCAN


PATCHES = {
  "deadsave": patch_dead_saves,
  "fusedrms": patch_fused_rmsnorm,
  "normcast": patch_norm_input_cast,
  "bf16res": patch_bf16_residual,
  "rsqrtonly": patch_rsqrt_only,
  "rmsnorm": patch_rmsnorm_checkpoint,
  "gated": patch_gated_checkpoint,
  "conv": patch_conv_checkpoint,
  "rsqrt": patch_rsqrt_dtype,
  "splitproj": patch_split_proj,
  "xbc": patch_xbc_contiguous,
}

# Patches mutate the same methods, so composition order matters: dtype fix
# first, then the recompute wrappers, then the storage-splitting ones.
ORDER = ["rsqrt", "rsqrtonly", "fusedrms", "normcast", "bf16res",
         "rmsnorm", "gated", "conv", "xbc", "splitproj", "deadsave"]


def run_step_peak(config, device, batch_size, length, warmup, iters):
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  x0 = torch.randint(8, 12, (batch_size, length), device=device)
  attention_mask = torch.ones_like(x0)
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device)
  times, loss = [], None
  for step in range(warmup + iters):
    if step == warmup:
      torch.cuda.synchronize(device)
      torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model._loss(x0, attention_mask)
    loss.loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    if step >= warmup:
      times.append(time.perf_counter() - start)
  peak = torch.cuda.max_memory_allocated(device)
  out = {"peak_gib": peak / 1024 ** 3,
         "step_seconds": statistics.median(times),
         "loss": float(loss.loss.detach())}
  del model, optimizer, loss
  torch.cuda.empty_cache()
  return out


def run_audit(config, device, batch_size, length, label, top):
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
  x0 = torch.randint(8, 12, (batch_size, length), device=device)
  attention_mask = torch.ones_like(x0)
  # Warm up outside the hook so cudnn/triton autotune allocations do not
  # appear in the accounting.
  loss = model._loss(x0, attention_mask)
  loss.loss.backward()
  model.zero_grad(set_to_none=True)
  del loss
  torch.cuda.empty_cache()

  audit = Audit(param_ptrs)
  before = torch.cuda.memory_allocated(device)
  with audit:
    loss = model._loss(x0, attention_mask)
  torch.cuda.synchronize(device)
  after = torch.cuda.memory_allocated(device)
  print(f"\n[{label}] allocator delta across forward: "
        f"{(after - before) / 1024**3:.3f} GiB")
  summary = summarize(audit.records, label, top=top)
  summary["allocator_forward_delta_gib"] = (after - before) / 1024 ** 3
  loss.loss.backward()
  del model, loss
  torch.cuda.empty_cache()
  return summary


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", default="ussm-ar")
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--variants", default="none",
                      help="comma-separated; each is a '+'-joined patch set "
                           "from {none,rmsnorm,gated,conv,all}")
  parser.add_argument("--audit", action="store_true")
  parser.add_argument("--peak", action="store_true")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--top", type=int, default=40)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  device = torch.device("cuda")
  print(f"device: {torch.cuda.get_device_name(device)}  "
        f"total {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GiB")

  dtype_probe(device)
  results = {}
  for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
    # Patches mutate module-level classes, so run each variant in a fresh
    # import state.
    for name in list(sys.modules):
      if name.startswith("models") or name == "diffusion":
        del sys.modules[name]
    reset_library_patches()
    import diffusion as _d  # noqa: F401
    globals()["Diffusion"] = _d.Diffusion
    names = [] if variant == "none" else variant.split("+")
    if names == ["all"]:
      names = ["rmsnorm", "gated", "conv", "splitproj"]
    if names == ["full"]:
      names = list(ORDER)
    names = sorted(names, key=ORDER.index)
    for name in names:
      PATCHES[name]()
    config = build(args.arm, args.length, args.batch_size, args.block_size)
    label = f"{args.arm} [{variant}]"
    entry = {}
    if args.audit:
      entry["audit"] = run_audit(
        config, device, args.batch_size, args.length, label, args.top)
    if args.peak:
      entry["step"] = run_step_peak(
        config, device, args.batch_size, args.length,
        args.warmup, args.iters)
      print(f"[{label}] peak {entry['step']['peak_gib']:.2f} GiB  "
            f"step {entry['step']['step_seconds']*1000:.1f} ms  "
            f"loss {entry['step']['loss']:.4f}")
    results[variant] = entry

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
