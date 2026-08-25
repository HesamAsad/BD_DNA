#!/usr/bin/env python
"""Count CPU-side operator dispatches per training step, per arm and length.

WHY. The measured throughput table has a signature that only one class of
cause produces: for the SSM arms the step TIME is flat while the length grows
4x (BiSSM 0.2579 / 0.2590 / 0.2630 s at L = 2048 / 4096 / 8192). A step whose
wall clock does not move when its arithmetic quadruples is not compute-bound;
it is bound by something with a fixed cost. The candidate is the CPU-side cost
of issuing the step's kernels, which in eager PyTorch is a fixed number of
dispatches whose cost does not depend on tensor size.

WHAT THIS MEASURES. Every aten dispatch of a real `Diffusion._loss` ->
`backward()` on CPU, split into forward and backward, via `TorchDispatchMode`.
Dispatch COUNT is independent of tensor size, so a tiny CPU model counts the
same operators an H200 step launches. Two things need forcing, because both
have a `.is_cuda` branch that a CPU run would take the wrong way:

  * the Mamba-2 scan (`mamba_chunk_scan_combined`) -- on CPU the code falls
    back to `_reference_scan`, a Python loop over positions, which is not the
    operator sequence a GPU step issues. It is replaced with an opaque stub
    whose own ops are excluded from the count and whose CALL count is reported
    separately, to be multiplied by the Triton kernel count read off
    mamba_ssm 2.3.2 (5 kernels forward, 14 backward with z=None).
  * `rmsnorm_fn` (mamba_ssm's fused gated RMSNorm) -- likewise stubbed and
    counted separately (1 kernel forward, 2 backward).

CAVEAT, stated up front: this UNDERCOUNTS a CUDA step. Autocast inserts
`_to_copy` weight casts that a CPU run does not, the caching allocator's
`cudaMalloc`/free are not dispatches, and one aten op can be more than one
kernel. It is a lower bound, and the conclusion only needs a lower bound.

Usage:
  python scripts/smoke/launch_count_probe.py --lengths 2048,4096,8192,32768
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import models.mamba2_segment as m2  # noqa: E402
import models.bidirectional_ssm as bissm  # noqa: E402

# Triton kernels launched per call, read off mamba_ssm 2.3.2's source:
#   fwd  ssd_combined.py:374-387  -> _chunk_cumsum_fwd, _chunk_state_fwd,
#                                    _state_passing_fwd, _bmm_chunk_fwd,
#                                    _chunk_scan_fwd
#   bwd  ssd_combined.py:441-509  -> the four fwd recomputes plus
#                                    _chunk_scan_bwd_dstates, _state_passing_bwd,
#                                    _chunk_scan_chunk_state_bwd_dx,
#                                    _chunk_state_bwd_db, _chunk_scan_bwd_dC,
#                                    _chunk_scan_bwd_dcb, _bmm_chunk_bwd x2,
#                                    _chunk_scan_bwd_ddAcs_stable,
#                                    _chunk_cumsum_bwd
# (each helper launches exactly one kernel; verified by grep for `_kernel[grid`)
SCAN_KERNELS_FWD = 5
SCAN_KERNELS_BWD = 14
NORM_KERNELS_FWD = 1
NORM_KERNELS_BWD = 2


class Counter(TorchDispatchMode):
  def __init__(self):
    super().__init__()
    self.n = 0
    self.by_op = {}
    self.suspend = 0
    self.scan_fwd = 0
    self.scan_bwd = 0
    self.norm_fwd = 0
    self.norm_bwd = 0

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if not self.suspend:
      self.n += 1
      name = str(func)
      self.by_op[name] = self.by_op.get(name, 0) + 1
    return func(*args, **(kwargs or {}))

  @contextlib.contextmanager
  def paused(self):
    self.suspend += 1
    try:
      yield
    finally:
      self.suspend -= 1


COUNTER = Counter()


class _OpaqueScan(torch.autograd.Function):
  """Stands in for `mamba_chunk_scan_combined`; its own ops are not counted."""

  @staticmethod
  def forward(ctx, x, dt, A, B, C, D, dt_bias, init):
    COUNTER.scan_fwd += 1
    ctx.save_for_backward(x, dt, A, B, C, D, dt_bias)
    ctx.had_init = init is not None
    with COUNTER.paused():
      scale = (B * C).sum(-1)
      y = x * (1.0 + dt.unsqueeze(-1)) + scale[..., None] * (
        D.view(1, 1, -1, 1) + A.view(1, 1, -1, 1) + dt_bias.view(1, 1, -1, 1))
      final = x.sum(1)[..., None] * B.sum((1, 2))[:, None, None, :]
      if init is not None:
        final = final + init
    return y, final

  @staticmethod
  def backward(ctx, dy, dfinal):
    COUNTER.scan_bwd += 1
    x, dt, A, B, C, D, dt_bias = ctx.saved_tensors
    with COUNTER.paused():
      grads = (torch.zeros_like(x), torch.zeros_like(dt),
               torch.zeros_like(A), torch.zeros_like(B),
               torch.zeros_like(C), torch.zeros_like(D),
               torch.zeros_like(dt_bias))
      init_grad = torch.zeros_like(dfinal) if ctx.had_init else None
    return grads + (init_grad,)


def stub_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
              initial_states=None, dt_softplus=False, return_final_states=False,
              **kwargs):
  y, final = _OpaqueScan.apply(x, dt, A, B, C, D, dt_bias, initial_states)
  return (y, final) if return_final_states else y


class _OpaqueNorm(torch.autograd.Function):
  @staticmethod
  def forward(ctx, y, weight, z):
    COUNTER.norm_fwd += 1
    ctx.save_for_backward(y, weight, z)
    with COUNTER.paused():
      out = y * weight if z is None else y * weight * z
    return out

  @staticmethod
  def backward(ctx, dout):
    COUNTER.norm_bwd += 1
    y, weight, z = ctx.saved_tensors
    with COUNTER.paused():
      grads = (torch.zeros_like(y), torch.zeros_like(weight),
               None if z is None else torch.zeros_like(z))
    return grads


def install_gpu_branches():
  """Force every `.is_cuda` branch to the path a GPU step actually takes."""
  m2.mamba_chunk_scan_combined = stub_scan
  m2.SegmentMamba2._select_backend = lambda self, x: "fused"

  def gated_norm(self, y, z):
    y = y.reshape(y.shape[0], y.shape[1], -1)
    return _OpaqueNorm.apply(y, self.norm_weight, z)

  m2.SegmentMamba2._gated_norm = gated_norm
  bissm.RMSNorm.forward = lambda self, x: _OpaqueNorm.apply(
    x, self.weight, None)


def build(arm, length, block_size, batch_size, checkpoint_prefill=True):
  import hydra
  import main  # noqa: F401  registers resolvers
  from dataloader import DNATokenizer
  from diffusion import Diffusion

  model_cfg, algo_cfg = {
    "bissm": ("small_bissm", "bd3lm_bissm"),
    "ussm": ("small_ussm", "bd3lm_ussm"),
    "ussm-ar": ("small_ussm", "ar"),
  }[arm]
  is_ar = arm == "ussm-ar"
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}",
    f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch_size}",
    f"loader.eval_batch_size={batch_size}",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
    # Width does not change the dispatch COUNT, only the tensor sizes, so run
    # the CPU probe narrow. Depth does, so n_blocks stays at the real 12.
    "model.hidden_size=128", "model.ssm_head_dim=64",
  ]
  if not is_ar:
    overrides += ["model.active_blocks=all",
                  "model.checkpoint_boundary_prefill="
                  f"{str(bool(checkpoint_prefill)).lower()}"]
  else:
    overrides += ["algo.backbone=ussm"]
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO / "configs")):
    config = hydra.compose(config_name="config", overrides=overrides)
  return Diffusion(config, DNATokenizer())


def run(arm, length, block_size, batch_size, checkpoint_prefill=True):
  torch.manual_seed(0)
  model = build(arm, length, block_size, batch_size, checkpoint_prefill)
  model.train()
  x0 = torch.randint(8, 12, (batch_size, length))
  mask = torch.ones_like(x0)

  COUNTER.n = 0
  COUNTER.by_op = {}
  COUNTER.scan_fwd = COUNTER.scan_bwd = 0
  COUNTER.norm_fwd = COUNTER.norm_bwd = 0
  with COUNTER:
    loss = model._loss(x0, mask)
    fwd = COUNTER.n
    fwd_scan, fwd_norm = COUNTER.scan_fwd, COUNTER.norm_fwd
    loss.loss.backward()
    total = COUNTER.n
  bwd = total - fwd
  n_layers = int(model.config.model.n_blocks)
  num_blocks = max(length // block_size, 1)
  kernels = (total
             + fwd_scan * (SCAN_KERNELS_FWD - 1)
             + COUNTER.scan_bwd * (SCAN_KERNELS_BWD - 1)
             + fwd_norm * (NORM_KERNELS_FWD - 1)
             + COUNTER.norm_bwd * (NORM_KERNELS_BWD - 1))
  return {
    "arm": arm, "length": length, "num_blocks": num_blocks,
    "n_layers": n_layers, "checkpoint_prefill": bool(checkpoint_prefill),
    "aten_fwd": fwd, "aten_bwd": bwd, "aten_total": total,
    "scan_calls_fwd": fwd_scan, "scan_calls_bwd": COUNTER.scan_bwd,
    "norm_calls_fwd": fwd_norm, "norm_calls_bwd": COUNTER.norm_bwd,
    "kernels_est": kernels,
    "top_ops": sorted(COUNTER.by_op.items(), key=lambda kv: -kv[1])[:12],
    "by_op": dict(COUNTER.by_op),
  }


def main_cli():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--arms", default="ussm-ar,ussm,bissm")
  ap.add_argument("--lengths", default="2048,4096,8192")
  ap.add_argument("--block-size", type=int, default=256)
  ap.add_argument("--batch-size", type=int, default=2)
  ap.add_argument("--checkpoint-modes", default="on")
  ap.add_argument("--output", type=Path, default=None)
  args = ap.parse_args()

  install_gpu_branches()
  rows = []
  head = (f"{'arm':<8}{'L':>7}{'nb':>5}{'aten_f':>9}{'aten_b':>9}"
          f"{'aten':>9}{'scan_f':>8}{'scan_b':>8}{'kernels':>9}")
  print(head)
  print("-" * len(head))
  modes = [m.strip() == "on" for m in args.checkpoint_modes.split(",")]
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
     for ck in modes:
      row = run(arm, length, args.block_size, args.batch_size, ck)
      rows.append(row)
      print(f"{row['arm']+('/ck' if ck else '/no'):<8}{row['length']:>7}{row['num_blocks']:>5}"
            f"{row['aten_fwd']:>9}{row['aten_bwd']:>9}{row['aten_total']:>9}"
            f"{row['scan_calls_fwd']:>8}{row['scan_calls_bwd']:>8}"
            f"{row['kernels_est']:>9}")
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
