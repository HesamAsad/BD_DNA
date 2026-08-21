#!/usr/bin/env python
"""Lens B: is `mamba_split_conv1d_scan_combined` usable here, and where?

The fully fused entry point (mamba_ssm/ops/triton/ssd_combined.py:988) runs
in_proj-output -> causal conv -> SSD scan -> gated RMSNorm -> out_proj inside a
single autograd Function, so its backward recomputes the conv, the norm and the
output projection instead of retaining them. It needs `causal_conv1d`, which is
not installed here, and it hard-codes `None` for the convolution's
initial/final boundary state (ssd_combined.py:845-846), which is exactly what
BiSSM's block-boundary continuation depends on.

This script answers three things on one GPU:

1. Does the prebuilt causal_conv1d wheel work in this env (no nvcc needed)?
2. Is the fused split path numerically identical to `SegmentMamba2.scan_segment`
   on the zero-conv-state (AR) path, and what does it save?
3. Can the conv boundary state be plumbed through a thin local fork of the
   Function, unlocking the BiSSM active path too?  `causal_conv1d`'s own
   fwd/bwd C++ entry points DO take `initial_states` / `dinitial_states`
   (causal_conv1d/cpp_functions.py:92-155); only mamba_ssm declines to pass
   them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main  # noqa: F401,E402 - registers OmegaConf resolvers
from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402

DT_LIMIT = (0.0, float("inf"))


# --------------------------------------------------------------------------
# Fused entry points
# --------------------------------------------------------------------------

def _imports():
  from mamba_ssm.ops.triton.ssd_combined import (
    mamba_split_conv1d_scan_combined,
    _mamba_chunk_scan_combined_fwd,
    _mamba_chunk_scan_combined_bwd,
    ensure_stride,
    causal_conv1d_fwd_function,
    causal_conv1d_bwd_function,
  )
  from mamba_ssm.ops.triton.layernorm_gated import _layer_norm_fwd, _layer_norm_bwd
  return dict(
    split=mamba_split_conv1d_scan_combined,
    fwd=_mamba_chunk_scan_combined_fwd,
    bwd=_mamba_chunk_scan_combined_bwd,
    ensure_stride=ensure_stride,
    conv_fwd=causal_conv1d_fwd_function,
    conv_bwd=causal_conv1d_bwd_function,
    ln_fwd=_layer_norm_fwd,
    ln_bwd=_layer_norm_bwd,
  )


class SplitConvStateFn(torch.autograd.Function):
  """`MambaSplitConv1dScanCombinedFn` with the conv boundary state plumbed.

  Specialised to the geometry this repo actually uses -- d_nonssm == 0,
  rmsnorm_weight and outproj_weight both present, no outproj bias, no seq_idx,
  default dt_limit -- which removes every branch the upstream version carries
  for other configurations. The only substantive change is that
  `conv_initial_states` reaches `causal_conv1d_fwd_function` (upstream passes
  `None`) and that its gradient comes back out of `causal_conv1d_bwd_function`
  (upstream passes `return_dinitial_states=False`).
  """

  @staticmethod
  @torch.amp.custom_fwd(device_type="cuda")
  def forward(ctx, zxbcdt, conv_weight, conv_bias, dt_bias, A, D, chunk_size,
              conv_initial_states, ssm_initial_states, rmsnorm_weight,
              rmsnorm_eps, outproj_weight, headdim, norm_before_gate):
    fns = _imports()
    nheads, = D.shape
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (conv_weight.shape[0] - dim) // 2
    z, xBC, dt = torch.split(
      zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xBC_conv = rearrange(
      fns["conv_fwd"](
        rearrange(fns["ensure_stride"](xBC), "b s d -> b d s"),
        conv_weight, conv_bias, None, conv_initial_states, None, True),
      "b d s -> b s d")
    x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=1)
    C = rearrange(C, "b l (g n) -> b l g n", g=1)
    out_x, _, dt_out, dA_cumsum, states, final_states = fns["fwd"](
      x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias,
      initial_states=ssm_initial_states, seq_idx=None, dt_softplus=True,
      dt_limit=DT_LIMIT)
    x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
    z_rms = rearrange(z, "b s d -> (b s) d")
    rmsnorm_weight = rmsnorm_weight.contiguous()
    out, _, rstd = fns["ln_fwd"](
      x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=None,
      group_size=dim, norm_before_gate=norm_before_gate, is_rms_norm=True)
    out = rearrange(out, "(b s) d -> b s d", b=batch)
    if torch.is_autocast_enabled():
      dtype = torch.get_autocast_dtype("cuda")
      out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
    out = F.linear(out, outproj_weight, None)
    ctx.save_for_backward(zxbcdt, conv_weight, conv_bias, out_x, A, D, dt_bias,
                          conv_initial_states, ssm_initial_states,
                          rmsnorm_weight, rstd, outproj_weight)
    ctx.chunk_size = chunk_size
    ctx.headdim = headdim
    ctx.rmsnorm_eps = rmsnorm_eps
    ctx.norm_before_gate = norm_before_gate
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type="cuda")
  def backward(ctx, dout):
    fns = _imports()
    (zxbcdt, conv_weight, conv_bias, out, A, D, dt_bias, conv_initial_states,
     ssm_initial_states, rmsnorm_weight, rstd, outproj_weight) = ctx.saved_tensors
    headdim = ctx.headdim
    nheads, = D.shape
    dim = nheads * headdim
    dstate = (conv_weight.shape[0] - dim) // 2
    batch, seqlen = out.shape[:2]

    out_recompute = torch.empty(
      batch, seqlen, dim, device=out.device, dtype=out.dtype)
    z, xBC, dt = torch.split(
      zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xBC_conv = rearrange(
      fns["conv_fwd"](
        rearrange(fns["ensure_stride"](xBC), "b s d -> b d s"),
        conv_weight, conv_bias, None, conv_initial_states, None, True),
      "b d s -> b s d")
    x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=1)
    C = rearrange(C, "b l (g n) -> b l g n", g=1)

    dzxbcdt = torch.empty_like(zxbcdt)
    dz, dxBC_given, ddt_given = torch.split(
      dzxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    dxBC = torch.empty_like(xBC)
    dx, dB, dC = torch.split(dxBC, [dim, dstate, dstate], dim=-1)
    dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
    dB = rearrange(dB, "b l (g n) -> b l g n", g=1)
    dC = rearrange(dC, "b l (g n) -> b l g n", g=1)

    dout_og = dout
    dout = F.linear(dout, outproj_weight.t())

    dy_rms = rearrange(dout, "b s d -> (b s) d")
    dz_rms = rearrange(dz, "b l d -> (b l) d")
    x_rms = rearrange(out, "b s h p -> (b s) (h p)")
    z_rms = rearrange(z, "b s d -> (b s) d")
    out1 = rearrange(out_recompute, "b s d -> (b s) d")
    dout, drmsnorm_weight, _, dz_out, *rest = fns["ln_bwd"](
      dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms,
      group_size=dim, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True,
      recompute_output=True, dz=dz_rms, out=out1)
    dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
    (dx, ddt, dA, dB, dC, dD, _, ddt_bias,
     dssm_initial_states) = fns["bwd"](
      dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias,
      initial_states=ssm_initial_states, dfinal_states=None, seq_idx=None,
      dt_softplus=True, dt_limit=DT_LIMIT, dx=dx, ddt=ddt_given, dB=dB, dC=dC)

    doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_recompute)

    dxBC_given_update, dconv_weight, dconv_bias, dconv_initial = fns["conv_bwd"](
      rearrange(fns["ensure_stride"](xBC), "b s d -> b d s"),
      conv_weight, conv_bias,
      rearrange(fns["ensure_stride"](dxBC), "b s d -> b d s"),
      None, conv_initial_states, None,
      rearrange(fns["ensure_stride"](dxBC_given), "b s d -> b d s"),
      conv_initial_states is not None, True)
    dxBC_given_update = rearrange(dxBC_given_update, "b d s -> b s d")
    if dxBC_given.stride() != dxBC_given_update.stride():
      dxBC_given.copy_(dxBC_given_update)
    return (dzxbcdt, dconv_weight, dconv_bias, ddt_bias, dA, dD, None,
            dconv_initial, dssm_initial_states, drmsnorm_weight, None,
            doutproj_weight, None, None)


def conv_state_for_kernel(conv_state, d_conv, dtype):
  """Our `Mamba2State.conv` in the layout and dtype `causal_conv1d` demands.

  Three adjustments, all of them free:

  * the cache keeps `d_conv` raw projected inputs but only the `d_conv - 1`
    most recent are ever read -- our own convolution drops its first output
    (models/mamba2_segment.py:183), so the oldest slot is dead;
  * the kernel requires the *channel* axis to be the contiguous one
    (`initial_states.stride(1) == 1`), matching the layout
    `causal_conv1d_bwd_function` allocates for `dinitial_states`
    (causal_conv1d/cpp_functions.py:137);
  * the kernel requires `initial_states.scalar_type() == input_type`, and the
    caches are built in the FP32 outer autocast while `xBC` is bf16 -- the same
    cast `_causal_conv` already performs at models/mamba2_segment.py:175.
  """
  state = conv_state[..., -(d_conv - 1):].to(dtype)
  return state.transpose(1, 2).contiguous().transpose(1, 2)


def _final_conv_slice(mixer, zxbcdt):
  """The conv boundary state, for free, without touching the fused kernel.

  A block's convolution boundary state is the last `d_conv` *raw* projected
  inputs -- it is a slice of the `in_proj` output that the fused call already
  consumes, not a function of the convolution result -- so the entry point's
  hard-coded `final_states_out=None` costs us nothing.
  """
  xBC = zxbcdt[..., mixer.d_inner:mixer.d_inner + mixer.conv_dim]
  tail = xBC[:, -mixer.d_conv:].transpose(1, 2)
  if tail.shape[-1] < mixer.d_conv:
    tail = F.pad(tail, (mixer.d_conv - tail.shape[-1], 0))
  return tail


def split_scan(mixer, u, conv_initial=None, ssm_initial=None, fork=False,
               return_final_conv=False):
  """One `scan_segment`-equivalent forward through a fused entry point."""
  fns = _imports()
  zxbcdt = mixer.in_proj(u)
  A = -torch.exp(mixer.A_log.float())
  conv_weight = rearrange(mixer.conv1d.weight, "d 1 w -> d w")
  if conv_initial is not None:
    # `conv_initial` arrives as a full `Mamba2State.conv`; the dtype cast has
    # to happen before the layout fix, or `.to()` silently re-contiguifies.
    conv_initial = conv_state_for_kernel(
      conv_initial, mixer.d_conv, zxbcdt.dtype)
  if fork or conv_initial is not None:
    out = SplitConvStateFn.apply(
      zxbcdt, conv_weight, mixer.conv1d.bias, mixer.dt_bias, A, mixer.D,
      mixer.chunk_size, conv_initial, ssm_initial, mixer.norm_weight, 1e-5,
      mixer.out_proj.weight, mixer.headdim, False)
  else:
    out = fns["split"](
      zxbcdt, conv_weight, mixer.conv1d.bias, mixer.dt_bias, A, mixer.D,
      mixer.chunk_size, ssm_initial, None, DT_LIMIT, False, "silu",
      mixer.norm_weight, 1e-5, mixer.out_proj.weight, None, mixer.headdim,
      1, False, None)
  if return_final_conv:
    return out, _final_conv_slice(mixer, zxbcdt)
  return out


# --------------------------------------------------------------------------
# 1. Numerical checks
# --------------------------------------------------------------------------

def _relerr(a, b):
  return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


def check_equivalence(device, batch=2, seqlen=512, dtype=torch.float32):
  from models.mamba2_segment import SegmentMamba2, Mamba2State

  results = {}
  torch.manual_seed(0)
  mixer = SegmentMamba2(d_model=768, d_state=64, d_conv=4, expand=2,
                        headdim=64, chunk_size=128, backend="fused").to(
                          device=device, dtype=dtype)
  u = torch.randn(batch, seqlen, 768, device=device, dtype=dtype,
                  requires_grad=True)

  def run(fn):
    for p in mixer.parameters():
      p.grad = None
    if u.grad is not None:
      u.grad = None
    out = fn()
    out.float().pow(2).mean().backward()
    return (out.detach().float().clone(),
            u.grad.detach().float().clone(),
            {n: p.grad.detach().float().clone()
             for n, p in mixer.named_parameters()})

  def record(name, ref_fn, got_fn):
    try:
      results[name] = _compare(run(ref_fn), run(got_fn))
    except Exception as exc:
      results[name] = {"error": f"{type(exc).__name__}: {exc}"}
      import traceback
      traceback.print_exc()
    print(f"  {name}: {results[name]}", flush=True)

  print("\n=== numerical equivalence (fp32, b=2 L=512) ===", flush=True)
  # (a) zero conv boundary state == the AR path.
  record("zero_conv_state",
         lambda: mixer.scan_segment(u)[0],
         lambda: split_scan(mixer, u))
  record("zero_conv_state_fork",
         lambda: mixer.scan_segment(u)[0],
         lambda: split_scan(mixer, u, fork=True))

  # (b) non-zero conv + ssm boundary state == the BiSSM active path.
  torch.manual_seed(1)
  prefix = torch.randn(batch, 256, 768, device=device, dtype=dtype)
  with torch.no_grad():
    _, boundary = mixer.scan_segment(prefix)
  conv0 = boundary.conv.detach().clone()
  ssm0 = boundary.ssm.detach().clone()

  # causal_conv1d wants the (width - 1) most recent raw inputs; our cache keeps
  # d_conv of them and its oldest entry is never read (mamba2_segment.py:183
  # drops the first convolution output).
  record("nonzero_conv_state_fork",
         lambda: mixer.scan_segment(u, Mamba2State(conv0, ssm0))[0],
         lambda: split_scan(
           mixer, u, conv_initial=conv0, ssm_initial=ssm0, fork=True))
  # An SSM-only boundary state through the STOCK entry point, to isolate which
  # of the two boundary states the upstream Function actually supports.
  record("nonzero_ssm_state_stock",
         lambda: mixer.scan_segment(
           u, Mamba2State(torch.zeros_like(conv0), ssm0))[0],
         lambda: split_scan(mixer, u, ssm_initial=ssm0))

  # (b2) gradients *into* the boundary state. BiSSM's caches are not detached
  # (models/bidirectional_ssm.py:248-253), so the block denoiser's loss must
  # push gradient back through both boundary states into the clean prefill.
  # That is the one path the end-to-end run exercises but the checks above do
  # not: it depends on `causal_conv1d_bwd_function`'s `dinitial_states`, which
  # upstream never asks for.
  def state_grads(fn, conv_leaf, ssm_leaf):
    for leaf in (conv_leaf, ssm_leaf):
      leaf.grad = None
    fn().float().pow(2).mean().backward()
    return (conv_leaf.grad.detach().float().clone(),
            ssm_leaf.grad.detach().float().clone())

  try:
    conv_ref = conv0.clone().requires_grad_(True)
    ssm_ref = ssm0.clone().requires_grad_(True)
    gc_ref, gs_ref = state_grads(
      lambda: mixer.scan_segment(u, Mamba2State(conv_ref, ssm_ref))[0],
      conv_ref, ssm_ref)
    conv_got = conv0.clone().requires_grad_(True)
    ssm_got = ssm0.clone().requires_grad_(True)
    gc_got, gs_got = state_grads(
      lambda: split_scan(mixer, u, conv_initial=conv_got,
                         ssm_initial=ssm_got, fork=True),
      conv_got, ssm_got)
    results["boundary_state_grads_fork"] = {
      "dconv_relerr": _relerr(gc_got, gc_ref),
      "dssm_relerr": _relerr(gs_got, gs_ref),
      # The oldest cache slot feeds only the convolution output our own
      # implementation discards, so its gradient must be exactly zero in both.
      "dconv_dead_slot_ref": gc_ref[..., 0].abs().max().item(),
      "dconv_dead_slot_got": gc_got[..., 0].abs().max().item(),
    }
  except Exception as exc:
    results["boundary_state_grads_fork"] = {"error": f"{type(exc).__name__}: {exc}"}
    import traceback
    traceback.print_exc()
  print(f"  boundary_state_grads_fork: {results['boundary_state_grads_fork']}",
        flush=True)

  # (c) the final conv boundary state read straight off in_proj.
  try:
    with torch.no_grad():
      _, state = mixer.scan_segment(u.detach())
      _, free = split_scan(mixer, u.detach(), return_final_conv=True)
    results["final_conv_free_slice"] = {
      "max_abs_diff": (state.conv[..., 1:] - free[..., 1:]).abs().max().item()}
  except Exception as exc:
    results["final_conv_free_slice"] = {"error": f"{type(exc).__name__}: {exc}"}
  print(f"  final_conv_free_slice: {results['final_conv_free_slice']}",
        flush=True)
  return results


def _compare(ref, got):
  out_ref, gu_ref, gp_ref = ref
  out_got, gu_got, gp_got = got
  worst_param, worst = None, 0.0
  for name in gp_ref:
    e = _relerr(gp_got[name], gp_ref[name])
    if e > worst:
      worst, worst_param = e, name
  return {
    "out_relerr": _relerr(out_got, out_ref),
    "dinput_relerr": _relerr(gu_got, gu_ref),
    "worst_param_relerr": worst,
    "worst_param": worst_param,
  }


# --------------------------------------------------------------------------
# 2. Monkeypatches for the end-to-end memory / throughput measurement
# --------------------------------------------------------------------------

def _patch_scan_clean(fork):
  """`BiMambaLayer.scan_clean` routed through a fused entry point.

  On the AR training path `initial_state` is always the zero state built by
  `_empty_cache` and the returned final state is discarded by
  `forward_active`, so the stock entry point (`fork=False`) suffices; the
  final SSM state is not recoverable from it, which is exactly the AR-only
  restriction this probe is measuring.
  """
  from models import bidirectional_ssm as bs
  from models import mamba2_segment as ms

  def scan_clean(self, x, initial_state):
    normalized = self.mixer_norm(x)
    conv_initial = initial_state.conv if fork else None
    mixed, final_conv = split_scan(
      self.mixer, normalized, conv_initial=conv_initial,
      ssm_initial=initial_state.ssm if fork else None, fork=fork,
      return_final_conv=True)
    state = ms.Mamba2State(
      final_conv,
      torch.zeros(x.shape[0], self.mixer.nheads, self.mixer.headdim,
                  self.mixer.d_state, device=x.device, dtype=x.dtype))
    x = x + self.dropout(mixed)
    x = x + self.dropout(self.mlp(self.mlp_norm(x)))
    return x, state

  bs.BiMambaLayer.scan_clean = scan_clean


def patch_ar_split():
  _patch_scan_clean(fork=False)


def patch_active_split():
  """Route the BiSSM/uSSM active block through the forked Function so the
  non-zero conv boundary state survives."""
  from models import bidirectional_ssm as bs

  def scan_active(self, x, left_state, right_state):
    normalized = self.mixer_norm(x)
    forward = split_scan(
      self.mixer, normalized,
      conv_initial=left_state.conv,
      ssm_initial=left_state.ssm, fork=True)
    reverse = split_scan(
      self.mixer, torch.flip(normalized, dims=(1,)).contiguous(),
      conv_initial=right_state.conv,
      ssm_initial=right_state.ssm, fork=True)
    reverse = torch.flip(reverse, dims=(1,))
    x = x + self.dropout(forward + reverse)
    x = x + self.dropout(self.mlp(self.mlp_norm(x)))
    return x

  bs.BiMambaLayer.scan_active = scan_active


PATCHES = {
  "arsplit": patch_ar_split,
  "arfork": lambda: _patch_scan_clean(fork=True),
  "activesplit": patch_active_split,
}


# --------------------------------------------------------------------------
# 3. End-to-end step measurement (copied from saved_tensor_audit.py)
# --------------------------------------------------------------------------

ARMS = {
  "ussm-ar": ("small_ussm", "ar"),
  "dit-ar": ("small_ar_transformer", "ar"),
  "bissm": ("small_bissm", "bd3lm_bissm"),
}


def build(arm, length, batch_size, block_size):
  model_cfg, algo_cfg = ARMS[arm]
  is_ar = algo_cfg == "ar"
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}", f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch_size}",
    f"loader.eval_batch_size={batch_size}",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
  ]
  if arm == "ussm-ar":
    overrides.append("algo.backbone=ussm")
  if arm == "bissm":
    overrides.append("model.active_blocks=all")
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=overrides)


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
  median = statistics.median(times)
  out = {"peak_gib": peak / 1024 ** 3,
         "step_seconds": median,
         "tokens_per_s": batch_size * length / median,
         "loss": float(loss.loss.detach())}
  del model, optimizer, loss
  torch.cuda.empty_cache()
  return out


def run_audit(config, device, batch_size, length, label, top=18):
  """Saved-tensor attribution, same method as scripts/smoke/saved_tensor_audit.py."""
  import collections

  interesting = ("bd3lms/models", "bd3lms/diffusion.py", "bd3lms/scripts",
                 "mamba_ssm", "flash_attn")

  def call_site():
    frame = sys._getframe(2)
    while frame is not None:
      name = frame.f_code.co_filename
      if any(tag in name for tag in interesting):
        return f"{Path(name).name}:{frame.f_lineno} ({frame.f_code.co_name})"
      frame = frame.f_back
    return "<other>"

  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
  x0 = torch.randint(8, 12, (batch_size, length), device=device)
  attention_mask = torch.ones_like(x0)
  loss = model._loss(x0, attention_mask)
  loss.loss.backward()
  model.zero_grad(set_to_none=True)
  del loss
  torch.cuda.empty_cache()

  records = {}

  def pack(t):
    try:
      storage = t.untyped_storage()
      ptr, nbytes = storage.data_ptr(), storage.nbytes()
    except Exception:
      ptr, nbytes = t.data_ptr(), t.numel() * t.element_size()
    if ptr not in records:
      records[ptr] = {"bytes": nbytes, "dtype": str(t.dtype).replace("torch.", ""),
                      "shape": tuple(t.shape), "site": call_site(),
                      "is_param": ptr in param_ptrs}
    return t

  with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    loss = model._loss(x0, attention_mask)
  torch.cuda.synchronize(device)
  live = [r for r in records.values() if not r["is_param"]]
  total = sum(r["bytes"] for r in live)
  groups = collections.defaultdict(lambda: {"bytes": 0, "count": 0})
  for r in live:
    g = groups[(r["site"], r["dtype"], r["shape"])]
    g["bytes"] += r["bytes"]
    g["count"] += 1
  rows = sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])
  print(f"\n=== saved-tensor audit: {label} ===")
  print(f"total retained activations: {total / 1024**3:.3f} GiB")
  print(f"{'GiB':>8} {'n':>4}  {'dtype':<9} {'shape':<28} site")
  for (site, dtype, shape), g in rows[:top]:
    print(f"{g['bytes']/1024**3:8.3f} {g['count']:>4}  {dtype:<9} "
          f"{str(shape):<28} {site}", flush=True)
  loss.loss.backward()
  del model, loss
  torch.cuda.empty_cache()
  return {"total_gib": total / 1024 ** 3,
          "rows": [{"site": k[0], "dtype": k[1], "shape": list(k[2]),
                    "gib": v["bytes"] / 1024 ** 3, "count": v["count"]}
                   for k, v in rows[:top]]}


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", default="ussm-ar")
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--variants", default="none,arsplit")
  parser.add_argument("--check", action="store_true")
  parser.add_argument("--peak", action="store_true")
  parser.add_argument("--audit", action="store_true")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  device = torch.device("cuda")
  print(f"device: {torch.cuda.get_device_name(device)}", flush=True)
  import causal_conv1d
  print(f"causal_conv1d {causal_conv1d.__version__} at {causal_conv1d.__file__}",
        flush=True)
  from mamba_ssm.ops.triton import ssd_combined
  print(f"causal_conv1d_fwd_function bound: "
        f"{ssd_combined.causal_conv1d_fwd_function is not None}", flush=True)

  results = {}
  if args.check:
    results["check"] = check_equivalence(device)
    print("\n=== numerical equivalence (fp32, b=2 L=512) ===")
    print(json.dumps(results["check"], indent=2), flush=True)

  for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
    for name in list(sys.modules):
      if name.startswith("models") or name == "diffusion":
        del sys.modules[name]
    import diffusion as _d  # noqa: F401
    globals()["Diffusion"] = _d.Diffusion
    for name in ([] if variant == "none" else variant.split("+")):
      PATCHES[name]()
    config = build(args.arm, args.length, args.batch_size, args.block_size)
    label = f"{args.arm} [{variant}]"
    entry = {}
    try:
      if args.audit:
        entry["audit"] = run_audit(
          config, device, args.batch_size, args.length, label)
      if args.peak:
        entry["step"] = run_step_peak(
          config, device, args.batch_size, args.length,
          args.warmup, args.iters)
        s = entry["step"]
        print(f"[{label}] peak {s['peak_gib']:.3f} GiB  "
              f"step {s['step_seconds']*1000:.1f} ms  "
              f"{s['tokens_per_s']:,.0f} tok/s  loss {s['loss']:.5f}",
              flush=True)
    except torch.cuda.OutOfMemoryError:
      entry["oom"] = True
      torch.cuda.empty_cache()
      print(f"[{label}] OOM", flush=True)
    except Exception as exc:  # a broken variant must not kill the sweep
      entry["error"] = f"{type(exc).__name__}: {exc}"
      torch.cuda.empty_cache()
      print(f"[{label}] FAILED {entry['error']}", flush=True)
      import traceback
      traceback.print_exc()
    results[variant] = entry

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
