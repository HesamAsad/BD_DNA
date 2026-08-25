#!/usr/bin/env python
"""Decompose ONE training step into phases, kernels, launches and syncs.

Every number in `results/figures/scaling_data.json` is a single wall clock per
step. Everything said about it since -- that a fixed per-step cost dominates
the SSM arms below L~8192, that the prefill's `torch.utils.checkpoint`
recompute is worth ~1.16x, that the residual is CUDA launch overhead -- has
been *inferred* from how that one number moves. This script measures the parts
directly.

WHAT IT REPORTS, per (arm, length, checkpoint mode)
--------------------------------------------------
1. **Wall clock**, median over `--iters` un-profiled steps, measured exactly
   the way `scripts/smoke/sizing_sweep.py:118-138` measures it (zero_grad ->
   `Diffusion._loss` -> backward -> `AdamW.step` -> `torch.cuda.synchronize`),
   so the total is directly comparable to the published table.

2. **A phase split of that wall clock.** The phases are the architecture's own
   parts, instrumented by wrapping bound methods on the live model -- nothing
   under `models/` is touched:

     prefill    `BidirectionalSSM.prefill_left_boundaries_stacked`
                (models/bidirectional_ssm.py:612) and its right-flank twin
                (:620), called from `Diffusion._forward_pass_bissm_all_blocks`
                (diffusion.py:1220-1232). Zero for the Transformer and AR
                arms, which never call it.
     active     `BidirectionalSSM.forward_active`
                (models/bidirectional_ssm.py:643), or plain `backbone.forward`
                for the arms that have no active/prefill split (`dit`,
                `dit-ar`, `ussm-ar`).
     head       `Diffusion._subs_parameterization`, the logit tail
                (diffusion.py:509, :1244).
     fwd-other  the rest of `_loss`: `_sample_t`, `q_xt`, the gather, the
                masked mean. Derived, not timed directly.
     backward   derived (see below).
     optimizer  `AdamW.step`.
     tail-sync  the closing `torch.cuda.synchronize()`. **This column is a
                diagnostic in its own right**: it is the time the CPU spends
                waiting for a GPU it has already run away from. Large
                tail-sync => GPU-bound. Near-zero tail-sync with the phase CPU
                times summing to the wall clock => the CPU is the critical
                path, i.e. launch-bound.

   Each phase is reported three ways:
     cpu_ms     CPU wall inside the region, no synchronisation added
     gpu_ms     the span of the region on the CUDA stream, from a pair of
                `torch.cuda.Event`s -- includes idle gaps, so it is the
                region's share of the GPU *timeline*
     kern_ms    the sum of the durations of the CUDA kernels launched inside
                the region -- excludes idle gaps, so it is the region's share
                of the GPU *work*

3. **Kernel count and total kernel time**, from `torch.profiler` with
   `ProfilerActivity.CUDA`. Kernel time is reported two ways: the plain sum
   over kernel events, and the union of their [start, end) intervals (which is
   what "GPU busy" means when streams overlap). The union is the one used for
   the headline.

4. **The headline number, printed on its own line:**

       gpu_idle_fraction = (wall - kernel_time) / wall

   This is the direct answer to H1. If a fixed per-step issue cost dominates
   at short L, this fraction must be LARGE at L=2048 and must SHRINK with
   length for the SSM arms, tracking the "UNEXPLAINED" column of the brief. If
   instead the arms are simply doing badly-shaped arithmetic, the GPU is busy
   and this fraction is small at every length.

   `cudaLaunchKernel` calls are counted separately, and the CPU time spent
   inside them is summed, which converts "launch-bound" from an inference into
   a measured quantity: `launch_api_ms` is literally how long the CPU spent in
   the launch API.

5. **Top kernels by time and by count.**

6. **CPU-GPU synchronisations**, counted three independent ways because no one
   of them is complete:
     a. `torch.cuda.set_sync_debug_mode('warn')` over one extra step, with
        `warnings.catch_warnings(record=True)`. This is the mechanism-level
        count -- ATen itself reporting each implicit sync -- and is the one
        printed as the headline. It does NOT see explicit
        `torch.cuda.synchronize()` or allocator syncs.
     b. runtime API calls in the trace whose name is a blocking sync
        (`cudaDeviceSynchronize`, `cudaStreamSynchronize`,
        `cudaEventSynchronize`, `cudaMemcpy`).
     c. ATen ops that can only be implemented by draining the queue
        (`aten::_local_scalar_dense`, `aten::nonzero`, ...), plus
        device-to-host memcpys. This is an UPPER bound: the same op names
        appear for calls on CPU tensors, which never touch the device.
   The five per-step syncs claimed for `diffusion.py:406, :1014 (x2), :1096,
   :1281` should show up in (a) and (c).

THE CHECKPOINT A/B
------------------
`--checkpoint-prefill` / `--no-checkpoint-prefill` set
`model.checkpoint_boundary_prefill` (models/bidirectional_ssm.py:493-508).
With NEITHER flag given -- the default -- both modes are run for the arms that
support it (`bissm`, `ussm`; see `sizing_sweep.ARMS`) and a paired delta table
is printed. That delta is the recompute cost, measured rather than modelled:
`on` runs the prefill's forward a second time inside the backward, so the
difference should appear almost entirely in the *backward* row, and its share
of the step should be flat in L if H2's analytic prediction (1.156 -> 1.166) is
right.

The flag is a no-op for `dit`, `dit-ar` and `ussm-ar` -- `ussm-ar` never calls
the prefill at all (`sizing_sweep.py:81` forces `block_size=1`) -- so those
arms are run once and reported with mode `n/a`.

ISOLATION
---------
Each (arm, length, mode) case runs in a FRESH subprocess by default. This is
not tidiness: `sizing_sweep.py`'s docstring records that the `dit` arm's flex
attention compiles with static shapes on the first length it sees and a second
length in the same process dies inside Inductor's autotuner. `--no-isolate`
runs everything in one process, which is fine for the SSM arms only.

WHAT THIS DOES NOT MEASURE
--------------------------
The backward pass runs on the autograd engine's own thread, and
`torch.autograd.profiler`'s tree is built per thread
(torch/autograd/profiler_util.py:83-91), so a `record_function` opened on the
main thread around `.backward()` does not become the parent of the backward's
kernels. The backward row is therefore a RESIDUAL:

    backward = total - forward-subtree - optimizer

which is exact by construction (the three partition the step) but cannot be
broken down further into prefill-backward vs active-backward. The checkpoint
A/B is the tool for that: the recompute lives entirely in the backward row.

The profiler inflates CPU time, so `gpu_idle_fraction` is computed against the
UN-profiled median wall clock while kernel time comes from the profiled step.
Kernel durations are measured on the device and are insensitive to CPU-side
profiler overhead; the CPU-side columns of the profiled step are not, and are
reported separately as `prof_wall_ms` so the inflation is visible.

Examples
--------
  # the full grid the brief asks for, both checkpoint modes, one process each
  python scripts/smoke/bd_step_breakdown.py \
      --arms ussm-ar,ussm,bissm,dit --lengths 2048,8192,32768 \
      --batch-size 2 --json results/sizing/bd_step_breakdown.json

  # one case, in-process, recompute off
  python scripts/smoke/bd_step_breakdown.py --arms bissm --lengths 2048 \
      --no-checkpoint-prefill --no-isolate

  # CPU-only validation of the pure helpers (no GPU needed, ~1 s)
  python scripts/smoke/bd_step_breakdown.py --self-test

  # ...plus a real tiny model per arm: checks the right regions fire, that the
  # instrumentation is bitwise non-invasive, and that the PhaseClock and the
  # profiler agree on the same region (no GPU needed, a few minutes)
  python scripts/smoke/bd_step_breakdown.py --self-test-model
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE = Path(__file__).resolve().parent
for _path in (str(REPO), str(SMOKE)):
  if _path not in sys.path:
    sys.path.insert(0, _path)

import torch  # noqa: E402

REGION_PREFIX = "bd3lm::"
# Order matters: this is the print order and the accounting order.
LEAF_PHASES = ("prefill", "active", "head")
DERIVED_PHASES = ("fwd-other", "backward")
TIMED_PHASES = ("forward", "backward", "optimizer", "tail-sync")
PRINT_PHASES = ("prefill", "active", "head", "fwd-other", "backward",
                "optimizer", "tail-sync")

# Blocking CUDA runtime entry points. `cudaMemcpyAsync` is deliberately NOT
# here: it is only a sync when the destination is pageable host memory, and
# that case shows up as a `Memcpy DtoH` device event instead.
SYNC_RUNTIME_NAMES = frozenset({
  "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
  "cudaMemcpy", "cudaStreamQuery", "cudaEventQuery",
})
# `cudaStreamQuery`/`cudaEventQuery` do not block, but the allocator polls with
# them; kept separate in the report.
BLOCKING_RUNTIME_NAMES = frozenset({
  "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
  "cudaMemcpy",
})
LAUNCH_RUNTIME_NAMES = frozenset({
  "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
  "cuLaunchKernel", "cudaGraphLaunch",
})
# ATen ops that cannot be implemented without draining the launch queue.
#
# `aten::item` is DELIBERATELY absent: it always dispatches to
# `aten::_local_scalar_dense`, so both records appear in the trace for a single
# sync and counting the pair doubles the total. Measured on a CPU run of the
# `bissm` arm: 37 `aten::item` against 37 `aten::_local_scalar_dense`.
#
# This count is an UPPER BOUND on device syncs: the same op names appear for
# calls on CPU tensors, which do not touch the device at all. It corroborates
# the ATen sync detector, it does not replace it.
SYNC_ATEN_NAMES = frozenset({
  "aten::_local_scalar_dense", "aten::nonzero",
  "aten::equal", "aten::allclose", "aten::_assert_async", "aten::masked_select",
})


# --------------------------------------------------------------------------
# Pure helpers. Everything below this line up to `PhaseClock` is testable on
# CPU with no torch.cuda and no profiler; `--self-test` exercises it.
# --------------------------------------------------------------------------

def interval_union_us(intervals):
  """Total length covered by [start, end) pairs, counting overlap once.

  Summing kernel durations double counts whenever two streams run at the same
  time; the union is what "the GPU was busy" actually means. With a single
  stream the two agree, and the report prints both so the gap is visible.
  """
  ordered = sorted((float(s), float(e)) for s, e in intervals if e > s)
  total = 0.0
  cur_start = cur_end = None
  for start, end in ordered:
    if cur_end is None:
      cur_start, cur_end = start, end
    elif start > cur_end:
      total += cur_end - cur_start
      cur_start, cur_end = start, end
    elif end > cur_end:
      cur_end = end
  if cur_end is not None:
    total += cur_end - cur_start
  return total


def gpu_idle_fraction(wall_s, kernel_s):
  """(wall - kernel_time) / wall, clamped to [0, 1].

  The single number H1 lives or dies by. 0 means the GPU never went idle
  inside the step; 0.9 means nine tenths of the step is the device waiting for
  work to be issued.
  """
  if wall_s <= 0:
    return float("nan")
  return max(0.0, min(1.0, (wall_s - kernel_s) / wall_s))


def classify_device_event(name):
  """kernel / memcpy / memset, for a CUDA-side profiler event name."""
  lowered = name.lower()
  if lowered.startswith("memcpy") or " memcpy" in lowered:
    return "memcpy"
  if lowered.startswith("memset") or " memset" in lowered:
    return "memset"
  return "kernel"


def plan_cases(arms, lengths, ckpt_mode, supports):
  """Expand the CLI grid into (arm, length, mode) cases.

  `supports` maps arm -> whether `model.checkpoint_boundary_prefill` exists
  for it. Arms without the flag are emitted once with mode 'n/a' so the table
  never implies an A/B that was not run.
  """
  cases = []
  for arm in arms:
    if not supports.get(arm, False):
      modes = ["n/a"]
    elif ckpt_mode == "both":
      modes = ["off", "on"]
    else:
      modes = [ckpt_mode]
    for length in lengths:
      for mode in modes:
        cases.append((arm, int(length), mode))
  return cases


def derive_phase_table(cpu_s, gpu_ms, kern_ms, wall_s, total_kernel_ms,
                       launches=None, total_launches=0):
  """Turn the raw region accumulators into the printed, additive phase table.

  `fwd-other` and `backward` are residuals so that the phases partition the
  step exactly:

      fwd-other = forward            - (prefill + active + head)
      backward  = total_kernel_time  - forward - optimizer      [kernel col]
      backward  = total_launches     - forward - optimizer      [launch col]
      backward  = measured                                      [cpu col]

  The kernel and launch columns' backward is a residual because the autograd
  engine runs on its own thread and a main-thread `record_function` cannot
  parent it (torch/autograd/profiler_util.py:83-91).

  `total_kernel_ms` must be the SUM over kernels, not the interval union, or
  the column stops being additive. The union is the right number for "how long
  was the GPU busy" and is used for `gpu_idle_fraction`; the two differ only
  when kernels overlap on separate streams, and `run_case` reports both.
  """
  launches = launches or {}
  rows = {}
  leaf_cpu = sum(cpu_s.get(p, 0.0) for p in LEAF_PHASES)
  leaf_kern = sum(kern_ms.get(p, 0.0) for p in LEAF_PHASES)
  leaf_gpu = sum(gpu_ms.get(p, 0.0) for p in LEAF_PHASES)
  leaf_launch = sum(launches.get(p, 0) for p in LEAF_PHASES)
  for phase in LEAF_PHASES:
    rows[phase] = {
      "cpu_ms": cpu_s.get(phase, 0.0) * 1e3,
      "gpu_ms": gpu_ms.get(phase, 0.0),
      "kern_ms": kern_ms.get(phase, 0.0),
      "launches": launches.get(phase, 0),
      "calls": cpu_s.get(phase + "::calls", 0),
    }
  rows["fwd-other"] = {
    "cpu_ms": max(0.0, cpu_s.get("forward", 0.0) - leaf_cpu) * 1e3,
    "gpu_ms": max(0.0, gpu_ms.get("forward", 0.0) - leaf_gpu),
    "kern_ms": max(0.0, kern_ms.get("forward", 0.0) - leaf_kern),
    "launches": max(0, launches.get("forward", 0) - leaf_launch),
    "calls": 1,
  }
  rows["backward"] = {
    "cpu_ms": cpu_s.get("backward", 0.0) * 1e3,
    "gpu_ms": gpu_ms.get("backward", 0.0),
    "kern_ms": max(0.0, total_kernel_ms - kern_ms.get("forward", 0.0)
                   - kern_ms.get("optimizer", 0.0)),
    "launches": max(0, total_launches - launches.get("forward", 0)
                    - launches.get("optimizer", 0)),
    "calls": 1,
  }
  for phase in ("optimizer", "tail-sync"):
    rows[phase] = {
      "cpu_ms": cpu_s.get(phase, 0.0) * 1e3,
      "gpu_ms": gpu_ms.get(phase, 0.0),
      "kern_ms": kern_ms.get(phase, 0.0),
      "launches": launches.get(phase, 0),
      "calls": 1,
    }
  accounted = sum(cpu_s.get(p, 0.0) for p in TIMED_PHASES)
  rows["step-other"] = {
    "cpu_ms": max(0.0, wall_s - accounted) * 1e3,
    "gpu_ms": 0.0,
    "kern_ms": 0.0,
    "launches": 0,
    "calls": 1,
  }
  return rows


def fmt_ms(value):
  if value is None:
    return "--"
  return f"{value:.2f}"


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

class PhaseClock:
  """Accumulates CPU wall, CUDA-stream span and profiler scope per phase.

  The CPU timer never synchronises, so `cpu_ms` is the time the CPU spent
  issuing that phase -- which under a launch-bound regime IS the phase's cost,
  and under a GPU-bound regime is much smaller than it. The CUDA event pair
  gives the same region's span on the device timeline. Reading both apart is
  the whole point.
  """

  def __init__(self, device=None, record=False):
    self.device = device
    self.record = record
    self.cpu = defaultdict(float)
    self.calls = defaultdict(int)
    self._pairs = defaultdict(list)

  def reset(self):
    self.cpu.clear()
    self.calls.clear()
    self._pairs.clear()

  @contextlib.contextmanager
  def region(self, name):
    self.calls[name] += 1
    start_event = end_event = None
    if self.device is not None:
      start_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
    scope = None
    if self.record:
      from torch.profiler import record_function
      scope = record_function(REGION_PREFIX + name)
      scope.__enter__()
    started = time.perf_counter()
    try:
      yield
    finally:
      self.cpu[name] += time.perf_counter() - started
      if scope is not None:
        scope.__exit__(None, None, None)
      if start_event is not None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._pairs[name].append((start_event, end_event))

  def gpu_ms(self):
    """Device-timeline span per phase.

    Synchronises first. `cudaEventElapsedTime` returns `cudaErrorNotReady` --
    which PyTorch raises as a RuntimeError -- if EITHER event has not
    completed, and the last region's closing event is recorded after
    `run_step`'s own `torch.cuda.synchronize`, so without this the tail-sync
    row would intermittently vanish from the table rather than read wrong.
    """
    if self.device is None:
      return {}
    torch.cuda.synchronize(self.device)
    out = {}
    for name, pairs in self._pairs.items():
      total = 0.0
      for start_event, end_event in pairs:
        try:
          total += start_event.elapsed_time(end_event)
        except RuntimeError:
          # A region that raised before recording its closing event; skip it
          # rather than poison the whole table.
          continue
      out[name] = total
    return out


def patch_regions(model, clock):
  """Wrap the phase entry points on the live model. Returns an undo callable.

  Instance-attribute assignment on an `nn.Module` lands in the instance
  `__dict__` (plain functions are not Parameters/Modules, so
  `nn.Module.__setattr__` falls through to `object.__setattr__`), and
  `nn.Module._call_impl` looks up `self.forward` -- so overriding `forward`
  here is picked up by `__call__`. Nothing under `models/` is modified.
  """
  saved = []

  def wrap(obj, attr, phase):
    if not hasattr(obj, attr):
      return
    original = getattr(obj, attr)
    had_own = attr in obj.__dict__

    def wrapper(*args, **kwargs):
      with clock.region(phase):
        return original(*args, **kwargs)

    wrapper.__name__ = f"{phase}::{attr}"
    setattr(obj, attr, wrapper)
    saved.append((obj, attr, original, had_own))

  backbone = model.backbone
  for attr in ("prefill_left_boundaries_stacked",
               "prefill_right_boundaries_stacked",
               "prefill_left", "prefill_right"):
    wrap(backbone, attr, "prefill")
  # `forward_active` is the BD path (models/bidirectional_ssm.py:643);
  # `forward` is what the Transformer and both AR arms take instead
  # (diffusion.py:491-500). Exactly one of the two fires per arm.
  wrap(backbone, "forward_active", "active")
  wrap(backbone, "forward", "active")
  wrap(model, "_subs_parameterization", "head")

  def undo():
    for obj, attr, original, had_own in reversed(saved):
      if had_own:
        setattr(obj, attr, original)
      else:
        try:
          delattr(obj, attr)
        except AttributeError:
          setattr(obj, attr, original)

  return undo


def run_step(model, optimizer, x0, mask, clock, device):
  """One training step, phase-timed, matching sizing_sweep.py:118-138."""
  optimizer.zero_grad(set_to_none=True)
  with clock.region("forward"):
    outputs = model._loss(x0, mask)
  with clock.region("backward"):
    outputs.loss.backward()
  with clock.region("optimizer"):
    optimizer.step()
  with clock.region("tail-sync"):
    torch.cuda.synchronize(device)
  return float(outputs.loss.detach())


# --------------------------------------------------------------------------
# Profiler analysis
# --------------------------------------------------------------------------

def subtree_totals(event):
  """Kernels launched anywhere under a CPU record, and the launch API calls.

  `FunctionEvent.kernels` holds only the kernels a record launched DIRECTLY --
  the correlation walk appends each kernel to exactly one frontend op
  (torch/autograd/profiler.py:655-671) -- so summing over the subtree counts
  every kernel once and no kernel twice.
  """
  kernel_count = 0
  kernel_us = 0.0
  launches = 0
  syncs = 0
  stack = [event]
  while stack:
    node = stack.pop()
    for kernel in node.kernels:
      kernel_count += 1
      kernel_us += float(kernel.duration)
    if node.name in LAUNCH_RUNTIME_NAMES:
      launches += 1
    if node.name in BLOCKING_RUNTIME_NAMES or node.name in SYNC_ATEN_NAMES:
      syncs += 1
    stack.extend(node.cpu_children)
  return kernel_count, kernel_us, launches, syncs


def analyse_profile(prof):
  """Everything the trace can say about one step, as a plain dict."""
  from torch.autograd import DeviceType

  events = prof.events()
  device_rows = defaultdict(lambda: {"count": 0, "us": 0.0})
  intervals = []
  totals = {"kernel": {"count": 0, "us": 0.0},
            "memcpy": {"count": 0, "us": 0.0},
            "memset": {"count": 0, "us": 0.0}}
  memcpy_d2h = 0
  for event in events:
    if event.device_type != DeviceType.CUDA:
      continue
    kind = classify_device_event(event.name)
    duration = float(event.time_range.end - event.time_range.start)
    totals[kind]["count"] += 1
    totals[kind]["us"] += duration
    if kind == "kernel":
      device_rows[event.name]["count"] += 1
      device_rows[event.name]["us"] += duration
      intervals.append((event.time_range.start, event.time_range.end))
    elif kind == "memcpy" and "dtoh" in event.name.lower():
      memcpy_d2h += 1

  launch_calls = 0
  launch_us = 0.0
  runtime_syncs = defaultdict(int)
  aten_syncs = defaultdict(int)
  regions = {}
  for event in events:
    if event.device_type != DeviceType.CUDA:
      name = event.name
      if name in LAUNCH_RUNTIME_NAMES:
        launch_calls += 1
        launch_us += float(event.self_cpu_time_total)
      elif name in SYNC_RUNTIME_NAMES:
        runtime_syncs[name] += 1
      elif name in SYNC_ATEN_NAMES:
        aten_syncs[name] += 1
      if name.startswith(REGION_PREFIX):
        phase = name[len(REGION_PREFIX):]
        kernel_count, kernel_us, launches, syncs = subtree_totals(event)
        row = regions.setdefault(
          phase, {"count": 0, "kernels": 0, "kernel_us": 0.0,
                  "launches": 0, "syncs": 0, "cpu_us": 0.0})
        row["count"] += 1
        row["kernels"] += kernel_count
        row["kernel_us"] += kernel_us
        row["launches"] += launches
        row["syncs"] += syncs
        row["cpu_us"] += float(event.cpu_time_total)

  union_us = interval_union_us(intervals)
  return {
    "device_rows": {k: dict(v) for k, v in device_rows.items()},
    "kernel_count": totals["kernel"]["count"],
    "kernel_us": totals["kernel"]["us"],
    "kernel_union_us": union_us,
    "memcpy_count": totals["memcpy"]["count"],
    "memcpy_us": totals["memcpy"]["us"],
    "memcpy_d2h_count": memcpy_d2h,
    "memset_count": totals["memset"]["count"],
    "memset_us": totals["memset"]["us"],
    "launch_api_calls": launch_calls,
    "launch_api_us": launch_us,
    "runtime_syncs": dict(runtime_syncs),
    "aten_syncs": dict(aten_syncs),
    "regions": regions,
  }


def count_implicit_syncs(fn):
  """Run `fn` with ATen's sync detector on and count what it reports.

  `torch.cuda.set_sync_debug_mode('warn')` makes every implicit
  device->host synchronisation emit a warning from C++
  (torch/cuda/__init__.py:1126-1150). `simplefilter('always')` defeats
  Python's per-site deduplication so the count is per occurrence, not per site.

  Blind spots, stated so the number is not over-read: it does not see explicit
  `torch.cuda.synchronize()`, allocator-internal syncs, or NCCL.
  """
  try:
    torch.cuda.set_sync_debug_mode("warn")
  except Exception as error:  # pragma: no cover - older/newer torch
    return None, [f"set_sync_debug_mode unavailable: {error}"]
  try:
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      fn()
    messages = [str(w.message) for w in caught
                if "synchron" in str(w.message).lower()]
  finally:
    torch.cuda.set_sync_debug_mode("default")
  return len(messages), messages


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------

def run_case(arm, length, mode, args, device):
  import sizing_sweep as ss
  from dataloader import DNATokenizer
  from diffusion import Diffusion
  from torch.profiler import ProfilerActivity, profile

  batch = args.batch_size
  checkpoint = mode == "on"
  config = ss.build(arm, length, args.block_size, batch, checkpoint)
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  x0 = torch.randint(8, 12, (batch, length), device=device)
  mask = torch.ones_like(x0)

  clock = PhaseClock(device=device, record=False)
  undo = patch_regions(model, clock)
  try:
    # --- warmup: Triton autotuning, flex-attention compilation, allocator
    for _ in range(args.warmup):
      clock.reset()
      run_step(model, optimizer, x0, mask, clock, device)
    torch.cuda.reset_peak_memory_stats(device)

    # --- timing pass: no profiler, no record_function, CUDA events only
    walls = []
    per_phase_cpu = defaultdict(list)
    per_phase_gpu = defaultdict(list)
    loss_value = None
    for _ in range(args.iters):
      clock.reset()
      started = time.perf_counter()
      loss_value = run_step(model, optimizer, x0, mask, clock, device)
      walls.append(time.perf_counter() - started)
      gpu = clock.gpu_ms()
      for name, seconds in clock.cpu.items():
        per_phase_cpu[name].append(seconds)
      for name, milliseconds in gpu.items():
        per_phase_gpu[name].append(milliseconds)
    wall_s = statistics.median(walls)
    cpu_s = {name: statistics.median(values)
             for name, values in per_phase_cpu.items()}
    gpu_ms = {name: statistics.median(values)
              for name, values in per_phase_gpu.items()}
    for name, count in clock.calls.items():
      cpu_s[name + "::calls"] = count
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    # --- sync count, un-profiled, with ATen's own detector. The phase
    # instrumentation stays installed so the counted step is the same step the
    # rest of the report describes; the clock's accumulators are discarded.
    sync_count, sync_messages = count_implicit_syncs(
      lambda: run_step(model, optimizer, x0, mask, clock, device))

    # --- profiled pass: record_function scopes on, CUDA activities on
    clock.record = True
    clock.reset()
    torch.cuda.synchronize(device)
    profile_started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False,
                 with_flops=False) as prof:
      run_step(model, optimizer, x0, mask, clock, device)
    prof_wall_s = time.perf_counter() - profile_started
    clock.record = False
    trace = analyse_profile(prof)
    del prof
  finally:
    undo()

  kernel_ms = {phase: row["kernel_us"] / 1e3
               for phase, row in trace["regions"].items()}
  region_launches = {phase: row["launches"]
                     for phase, row in trace["regions"].items()}
  # The phase table is built on the SUM so its columns stay additive; the
  # headline idle fraction uses the interval UNION, which is what "the GPU was
  # busy" means when kernels overlap. `kernel_overlap_ratio` shows the gap.
  phases = derive_phase_table(
    cpu_s, gpu_ms, kernel_ms, wall_s, trace["kernel_us"] / 1e3,
    launches=region_launches, total_launches=trace["launch_api_calls"])

  row = {
    "arm": arm,
    "length": length,
    "batch_size": batch,
    "block_size": args.block_size,
    "checkpoint_boundary_prefill": mode,
    "loss": loss_value,
    "peak_gib": peak_gib,
    "wall_s": wall_s,
    "wall_ms": wall_s * 1e3,
    "nt_per_second": batch * length / wall_s,
    "prof_wall_ms": prof_wall_s * 1e3,
    "kernel_count": trace["kernel_count"],
    "kernel_ms_sum": trace["kernel_us"] / 1e3,
    "kernel_ms_union": trace["kernel_union_us"] / 1e3,
    "kernel_overlap_ratio": (trace["kernel_us"] / trace["kernel_union_us"]
                             if trace["kernel_union_us"] else float("nan")),
    "memcpy_count": trace["memcpy_count"],
    "memcpy_d2h_count": trace["memcpy_d2h_count"],
    "memset_count": trace["memset_count"],
    "launch_api_calls": trace["launch_api_calls"],
    "launch_api_ms": trace["launch_api_us"] / 1e3,
    "gpu_idle_fraction": gpu_idle_fraction(
      wall_s, trace["kernel_union_us"] / 1e6),
    "gpu_idle_fraction_profiled": gpu_idle_fraction(
      prof_wall_s, trace["kernel_union_us"] / 1e6),
    "implicit_sync_count": sync_count,
    "implicit_sync_messages": sync_messages[:20] if sync_messages else [],
    "runtime_syncs": trace["runtime_syncs"],
    "aten_syncs": trace["aten_syncs"],
    "phases": phases,
    "regions": trace["regions"],
    "top_kernels_by_time": sorted(
      ({"name": name, **stats} for name, stats in
       trace["device_rows"].items()),
      key=lambda item: -item["us"])[:args.top],
    "top_kernels_by_count": sorted(
      ({"name": name, **stats} for name, stats in
       trace["device_rows"].items()),
      key=lambda item: -item["count"])[:args.top],
  }
  return row


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def short(name, width=64):
  return name if len(name) <= width else name[:width - 1] + "…"


def print_case(row):
  title = (f"{row['arm']}  L={row['length']}  batch={row['batch_size']}  "
           f"ckpt={row['checkpoint_boundary_prefill']}")
  print()
  print("=" * 84)
  print(title)
  print("=" * 84)
  print(f"wall (median of un-profiled steps) : {row['wall_ms']:.2f} ms   "
        f"({row['nt_per_second']:.0f} nt/s, peak {row['peak_gib']:.2f} GiB)")
  print(f"wall (profiled step, inflated)     : {row['prof_wall_ms']:.2f} ms")
  print(f"CUDA kernels                       : {row['kernel_count']} "
        f"kernels, {row['kernel_ms_union']:.2f} ms GPU-busy "
        f"(sum {row['kernel_ms_sum']:.2f} ms, overlap "
        f"{row['kernel_overlap_ratio']:.3f}x)")
  print(f"memcpy / memset                    : {row['memcpy_count']} "
        f"({row['memcpy_d2h_count']} D2H) / {row['memset_count']}")
  print(f"cudaLaunchKernel calls             : "
        f"{row['launch_api_calls']}, {row['launch_api_ms']:.2f} ms of CPU "
        f"inside the launch API")
  print()
  print(f">>> gpu_idle_fraction = (wall - kernel_time) / wall = "
        f"{row['gpu_idle_fraction']:.4f}   "
        f"<-- H1: launch-bound iff this is large and shrinks with L")
  print(f"    same against the profiled wall               = "
        f"{row['gpu_idle_fraction_profiled']:.4f}")
  print()

  header = (f"{'phase':<12}{'calls':>6}{'cpu ms':>10}{'gpu ms':>10}"
            f"{'kern ms':>10}{'launches':>10}{'us/launch':>10}"
            f"{'% wall':>8}{'% kern':>8}")
  print(header)
  print("-" * len(header))
  wall_ms = row["wall_ms"]
  kern_total = row["kernel_ms_sum"] or 1.0
  for phase in PRINT_PHASES + ("step-other",):
    stats = row["phases"].get(phase)
    if stats is None:
      continue
    per_launch = (stats["kern_ms"] * 1e3 / stats["launches"]
                  if stats["launches"] else 0.0)
    print(f"{phase:<12}{stats['calls']:>6}{fmt_ms(stats['cpu_ms']):>10}"
          f"{fmt_ms(stats['gpu_ms']):>10}{fmt_ms(stats['kern_ms']):>10}"
          f"{stats['launches']:>10}{per_launch:>10.1f}"
          f"{100.0 * stats['cpu_ms'] / wall_ms:>8.2f}"
          f"{100.0 * stats['kern_ms'] / kern_total:>8.2f}")
  print("-" * len(header))
  cpu_sum = sum(row["phases"][p]["cpu_ms"] for p in PRINT_PHASES
                if p in row["phases"])
  cpu_sum += row["phases"].get("step-other", {}).get("cpu_ms", 0.0)
  kern_sum = sum(row["phases"][p]["kern_ms"] for p in PRINT_PHASES
                 if p in row["phases"])
  launch_sum = sum(row["phases"][p]["launches"] for p in PRINT_PHASES
                   if p in row["phases"])
  print(f"{'TOTAL':<12}{'':>6}{fmt_ms(cpu_sum):>10}{'':>10}"
        f"{fmt_ms(kern_sum):>10}{launch_sum:>10}{'':>10}"
        f"{100.0 * cpu_sum / wall_ms:>8.2f}"
        f"{100.0 * kern_sum / kern_total:>8.2f}")
  print("cpu ms  = CPU wall inside the region, no sync added; a large "
        "tail-sync means the CPU ran ahead (GPU-bound)")
  print("gpu ms  = CUDA-event span of the region, idle gaps included; "
        "kern ms = kernel time inside it")
  print("us/launch = kern_ms/launches. A launch-bound phase shows MANY "
        "launches at FEW us each.")

  print(f"\ntop {len(row['top_kernels_by_time'])} kernels by device time")
  print(f"{'#':>3} {'ms':>9} {'%':>6} {'count':>7}  kernel")
  print("-" * 96)
  for index, kernel in enumerate(row["top_kernels_by_time"], start=1):
    print(f"{index:>3} {kernel['us'] / 1e3:>9.3f} "
          f"{100.0 * kernel['us'] / 1e3 / kern_total:>6.2f} "
          f"{kernel['count']:>7}  {short(kernel['name'])}")

  print(f"\ntop {len(row['top_kernels_by_count'])} kernels by launch count")
  print(f"{'#':>3} {'count':>7} {'ms':>9} {'us/call':>9}  kernel")
  print("-" * 96)
  for index, kernel in enumerate(row["top_kernels_by_count"], start=1):
    per_call = kernel["us"] / max(1, kernel["count"])
    print(f"{index:>3} {kernel['count']:>7} {kernel['us'] / 1e3:>9.3f} "
          f"{per_call:>9.2f}  {short(kernel['name'])}")

  print("\nCPU-GPU synchronisations")
  count = row["implicit_sync_count"]
  print(f"  a. ATen sync detector (set_sync_debug_mode='warn') : "
        f"{'unavailable' if count is None else count}")
  runtime_total = sum(v for k, v in row["runtime_syncs"].items()
                      if k in BLOCKING_RUNTIME_NAMES)
  print(f"  b. blocking CUDA runtime calls in the trace        : "
        f"{runtime_total}  {row['runtime_syncs'] or '{}'}")
  aten_total = sum(row["aten_syncs"].values())
  print(f"  c. queue-draining ATen ops (UPPER bound)           : "
        f"{aten_total}  {row['aten_syncs'] or '{}'}")
  print(f"     (+ {row['memcpy_d2h_count']} device-to-host memcpys)")
  if row["implicit_sync_messages"]:
    seen = {}
    for message in row["implicit_sync_messages"]:
      seen[message] = seen.get(message, 0) + 1
    for message, times in seen.items():
      print(f"       x{times}  {short(message, 88)}")


def print_summary(rows):
  header = (f"{'arm':<8}{'L':>7}{'ckpt':>6}{'wall ms':>10}{'kern ms':>9}"
            f"{'IDLE':>8}{'kernels':>9}{'launch ms':>10}{'syncs':>7}"
            f"{'prefill%':>9}{'active%':>9}{'bwd%':>7}")
  print()
  print("=" * len(header))
  print("SUMMARY -- one row per case")
  print("=" * len(header))
  print(header)
  print("-" * len(header))
  for row in rows:
    if row.get("error"):
      print(f"{row['arm']:<8}{row['length']:>7}"
            f"{row['checkpoint_boundary_prefill']:>6}  ERROR "
            f"{short(row['error'], 70)}")
      continue
    phases = row["phases"]
    wall = row["wall_ms"]
    syncs = row["implicit_sync_count"]
    print(f"{row['arm']:<8}{row['length']:>7}"
          f"{row['checkpoint_boundary_prefill']:>6}{wall:>10.2f}"
          f"{row['kernel_ms_union']:>9.2f}"
          f"{row['gpu_idle_fraction']:>8.3f}{row['kernel_count']:>9}"
          f"{row['launch_api_ms']:>10.2f}"
          f"{('--' if syncs is None else syncs):>7}"
          f"{100.0 * phases['prefill']['cpu_ms'] / wall:>9.2f}"
          f"{100.0 * phases['active']['cpu_ms'] / wall:>9.2f}"
          f"{100.0 * phases['backward']['cpu_ms'] / wall:>7.2f}")
  print("-" * len(header))
  print("IDLE = (wall - kernel_time)/wall. H1 predicts this is large at "
        "L=2048 and falls with L for the SSM arms.")


def print_checkpoint_ab(rows):
  """Paired off/on delta -- the recompute hypothesis, measured."""
  index = {(row["arm"], row["length"], row["checkpoint_boundary_prefill"]): row
           for row in rows if not row.get("error")}
  pairs = [(arm, length) for (arm, length, mode) in index if mode == "on"
           if (arm, length, "off") in index]
  if not pairs:
    return
  header = (f"{'arm':<8}{'L':>7}{'wall off':>10}{'wall on':>10}"
            f"{'on/off':>8}{'d bwd ms':>10}{'d kern ms':>11}"
            f"{'d kernels':>11}{'d peak GiB':>12}")
  print()
  print("=" * 92)
  print("CHECKPOINT PREFILL A/B -- model.checkpoint_boundary_prefill "
        "(models/bidirectional_ssm.py:493-508)")
  print("=" * 92)
  print(header)
  print("-" * len(header))
  for arm, length in sorted(set(pairs)):
    off = index[(arm, length, "off")]
    on = index[(arm, length, "on")]
    print(f"{arm:<8}{length:>7}{off['wall_ms']:>10.2f}{on['wall_ms']:>10.2f}"
          f"{on['wall_ms'] / off['wall_ms']:>8.3f}"
          f"{on['phases']['backward']['cpu_ms'] - off['phases']['backward']['cpu_ms']:>10.2f}"
          f"{on['kernel_ms_union'] - off['kernel_ms_union']:>11.2f}"
          f"{on['kernel_count'] - off['kernel_count']:>11}"
          f"{on['peak_gib'] - off['peak_gib']:>12.2f}")
  print("-" * len(header))
  print("The recompute is one extra forward of the prefix through every "
        "layer, executed inside backward,")
  print("so `d bwd ms` should carry essentially all of `wall on - wall off`, "
        "and `on/off` should be FLAT in L")
  print("(H2's analytic prediction is 1.156 at L=2048 rising to 1.166 at "
        "L=32768).")


# --------------------------------------------------------------------------
# Self test (CPU, no GPU, no profiler)
# --------------------------------------------------------------------------

class _FakeKernel:
  def __init__(self, name, duration):
    self.name = name
    self.duration = duration


class _FakeEvent:
  def __init__(self, name, kernels=(), children=()):
    self.name = name
    self.kernels = list(kernels)
    self.cpu_children = list(children)


def _fake_row(arm, length, mode, wall_ms, kernels):
  """A structurally complete result row, for exercising the printers."""
  kernel_ms = wall_ms * 0.45
  cpu_s = {"forward": wall_ms * 0.3e-3, "prefill": wall_ms * 0.12e-3,
           "active": wall_ms * 0.15e-3, "head": wall_ms * 0.01e-3,
           "backward": wall_ms * 0.6e-3, "optimizer": wall_ms * 0.05e-3,
           "tail-sync": wall_ms * 0.02e-3,
           "prefill::calls": 1, "active::calls": 1, "head::calls": 1}
  if arm == "dit":  # no prefill phase at all
    cpu_s["prefill"] = 0.0
    cpu_s["prefill::calls"] = 0
  phases = derive_phase_table(
    cpu_s, {"forward": wall_ms * 0.3, "prefill": wall_ms * 0.12},
    {"forward": kernel_ms * 0.4, "prefill": kernel_ms * 0.2,
     "active": kernel_ms * 0.15, "head": kernel_ms * 0.01,
     "optimizer": kernel_ms * 0.05},
    wall_ms * 1e-3, kernel_ms,
    launches={"forward": kernels // 3, "prefill": kernels // 6,
              "active": kernels // 8, "optimizer": kernels // 20},
    total_launches=kernels)
  return {
    "arm": arm, "length": length, "batch_size": 2, "block_size": 256,
    "checkpoint_boundary_prefill": mode, "loss": 1.4, "peak_gib": 40.0,
    "wall_s": wall_ms * 1e-3, "wall_ms": wall_ms,
    "nt_per_second": 2 * length / (wall_ms * 1e-3),
    "prof_wall_ms": wall_ms * 1.3,
    "kernel_count": kernels, "kernel_ms_sum": kernel_ms,
    "kernel_ms_union": kernel_ms * 0.98, "kernel_overlap_ratio": 1.02,
    "memcpy_count": 4, "memcpy_d2h_count": 2, "memset_count": 1,
    "launch_api_calls": kernels, "launch_api_ms": kernels * 0.006,
    "gpu_idle_fraction": gpu_idle_fraction(wall_ms * 1e-3,
                                           kernel_ms * 0.98e-3),
    "gpu_idle_fraction_profiled": 0.5,
    "implicit_sync_count": 5,
    "implicit_sync_messages": ["called a synchronizing CUDA operation"] * 5,
    "runtime_syncs": {"cudaStreamSynchronize": 3},
    "aten_syncs": {"aten::_local_scalar_dense": 5},
    "phases": phases, "regions": {},
    "top_kernels_by_time": [
      {"name": "_chunk_scan_fwd_kernel", "count": 24, "us": 12000.0},
      {"name": "void cutlass::Kernel<...>" * 4, "count": 96, "us": 8000.0}],
    "top_kernels_by_count": [
      {"name": "elementwise_kernel", "count": 900, "us": 3000.0},
      {"name": "_chunk_scan_fwd_kernel", "count": 24, "us": 12000.0}],
  }


def self_test():
  checks = []

  def check(label, condition):
    checks.append((label, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

  print("interval_union_us")
  check("empty -> 0", interval_union_us([]) == 0.0)
  check("single 10", interval_union_us([(0, 10)]) == 10.0)
  check("disjoint 10+5", interval_union_us([(0, 10), (20, 25)]) == 15.0)
  check("overlapping counted once",
        interval_union_us([(0, 10), (5, 15)]) == 15.0)
  check("nested counted once",
        interval_union_us([(0, 10), (2, 4)]) == 10.0)
  check("touching merge", interval_union_us([(0, 10), (10, 20)]) == 20.0)
  check("zero-length dropped", interval_union_us([(3, 3), (0, 4)]) == 4.0)
  check("union <= sum on overlap",
        interval_union_us([(0, 10), (5, 15)]) < 10.0 + 10.0)

  print("gpu_idle_fraction")
  check("gpu busy the whole step -> 0", gpu_idle_fraction(1.0, 1.0) == 0.0)
  check("gpu busy 10% -> 0.9",
        abs(gpu_idle_fraction(1.0, 0.1) - 0.9) < 1e-12)
  check("over-attributed kernel time clamps to 0",
        gpu_idle_fraction(1.0, 1.5) == 0.0)
  check("zero wall -> nan", gpu_idle_fraction(0.0, 0.0) != gpu_idle_fraction(0.0, 0.0))

  print("classify_device_event")
  check("triton kernel", classify_device_event("_chunk_scan_fwd_kernel")
        == "kernel")
  check("memcpy DtoH", classify_device_event("Memcpy DtoH (Device -> Pageable)")
        == "memcpy")
  check("memset", classify_device_event("Memset (Device)") == "memset")
  check("cutlass gemm",
        classify_device_event("void cutlass::Kernel<...>") == "kernel")

  print("subtree_totals")
  tree = _FakeEvent("bd3lm::prefill", kernels=[_FakeKernel("k0", 100.0)],
                    children=[
                      _FakeEvent("aten::mm",
                                 kernels=[_FakeKernel("gemm", 50.0)],
                                 children=[_FakeEvent("cudaLaunchKernel")]),
                      _FakeEvent("aten::_local_scalar_dense"),
                      _FakeEvent("cudaLaunchKernel"),
                      _FakeEvent("cudaStreamSynchronize"),
                    ])
  kernels, kernel_us, launches, syncs = subtree_totals(tree)
  check("counts kernels across the whole subtree", kernels == 2)
  check("sums kernel durations", kernel_us == 150.0)
  check("counts cudaLaunchKernel at any depth", launches == 2)
  check("counts both sync flavours", syncs == 2)
  check("a leaf with no kernels is 0",
        subtree_totals(_FakeEvent("x")) == (0, 0.0, 0, 0))

  print("plan_cases")
  supports = {"bissm": True, "ussm": True, "dit": False, "ussm-ar": False}
  both = plan_cases(["bissm", "dit"], [2048, 8192], "both", supports)
  check("A/B doubles only the supporting arm",
        both == [("bissm", 2048, "off"), ("bissm", 2048, "on"),
                 ("bissm", 8192, "off"), ("bissm", 8192, "on"),
                 ("dit", 2048, "n/a"), ("dit", 8192, "n/a")])
  forced = plan_cases(["bissm", "dit"], [2048], "on", supports)
  check("explicit --checkpoint-prefill runs one mode",
        forced == [("bissm", 2048, "on"), ("dit", 2048, "n/a")])
  check("unsupported arm is never given a mode it cannot honour",
        all(mode == "n/a" for arm, _, mode in
            plan_cases(["dit", "ussm-ar"], [2048], "on", supports)))

  print("derive_phase_table")
  cpu_s = {"forward": 0.100, "prefill": 0.040, "active": 0.030,
           "head": 0.005, "backward": 0.200, "optimizer": 0.010,
           "tail-sync": 0.001,
           "prefill::calls": 1, "active::calls": 1, "head::calls": 1}
  gpu_ms = {"forward": 90.0, "prefill": 40.0, "active": 30.0, "head": 5.0,
            "backward": 190.0, "optimizer": 9.0, "tail-sync": 0.5}
  kern_ms = {"forward": 60.0, "prefill": 25.0, "active": 20.0, "head": 4.0,
             "optimizer": 8.0}
  launches = {"forward": 900, "prefill": 400, "active": 300, "head": 40,
              "optimizer": 60}
  rows = derive_phase_table(cpu_s, gpu_ms, kern_ms, 0.320, 200.0,
                            launches=launches, total_launches=3000)
  check("fwd-other is forward minus its leaves",
        abs(rows["fwd-other"]["cpu_ms"] - 25.0) < 1e-9)
  check("fwd-other kernel time is forward minus its leaves",
        abs(rows["fwd-other"]["kern_ms"] - 11.0) < 1e-9)
  check("fwd-other launches are forward minus its leaves",
        rows["fwd-other"]["launches"] == 900 - 740)
  check("backward kernel time is the residual",
        abs(rows["backward"]["kern_ms"] - (200.0 - 60.0 - 8.0)) < 1e-9)
  check("backward launches are the residual",
        rows["backward"]["launches"] == 3000 - 900 - 60)
  check("step-other closes the cpu column against the wall",
        abs(rows["step-other"]["cpu_ms"] - (320.0 - 311.0)) < 1e-9)
  check("cpu column sums to the wall clock",
        abs(sum(rows[p]["cpu_ms"] for p in
                PRINT_PHASES + ("step-other",)) - 320.0) < 1e-6)
  check("kernel column sums to the profiled total",
        abs(sum(rows[p]["kern_ms"] for p in PRINT_PHASES) - 200.0) < 1e-6)
  check("launch column sums to the profiled total",
        sum(rows[p]["launches"] for p in PRINT_PHASES) == 3000)
  negative = derive_phase_table(
    {"forward": 0.010, "prefill": 0.040}, {}, {"forward": 1.0}, 0.001, 0.5,
    launches={"forward": 10, "prefill": 40}, total_launches=1)
  check("residuals never go negative",
        all(row["cpu_ms"] >= 0 and row["kern_ms"] >= 0
            and row["launches"] >= 0 for row in negative.values()))
  check("launches default to zero when the trace has none",
        all(row["launches"] == 0 for row in
            derive_phase_table(cpu_s, gpu_ms, kern_ms, 0.320, 200.0).values()))

  # A bug here silently runs the wrong geometry for a whole GPU job, so the
  # child command line is checked rather than trusted.
  print("child_argv")
  parsed = build_parser().parse_args([])
  parsed.batch_size, parsed.block_size = 2, 256
  parsed.warmup, parsed.iters, parsed.top = 3, 5, 12
  on = child_argv("bissm", 8192, "on", parsed)
  off = child_argv("bissm", 8192, "off", parsed)
  na = child_argv("dit", 2048, "n/a", parsed)
  check("child gets exactly one arm and one length",
        on[on.index("--arms") + 1] == "bissm"
        and on[on.index("--lengths") + 1] == "8192")
  check("mode 'on' passes --checkpoint-prefill",
        "--checkpoint-prefill" in on and "--no-checkpoint-prefill" not in on)
  check("mode 'off' passes --no-checkpoint-prefill",
        "--no-checkpoint-prefill" in off and "--checkpoint-prefill" not in off)
  check("mode 'n/a' passes neither, so the arm's own default stands",
        "--checkpoint-prefill" not in na
        and "--no-checkpoint-prefill" not in na)
  check("child never recurses into its own subprocess fan-out",
        "--no-isolate" in on and "--isolate" not in on)
  check("child inherits batch/block/warmup/iters",
        on[on.index("--batch-size") + 1] == "2"
        and on[on.index("--block-size") + 1] == "256"
        and on[on.index("--warmup") + 1] == "3"
        and on[on.index("--iters") + 1] == "5")
  check("child re-parses without error",
        build_parser().parse_args(on[3:]).arms == "bissm")
  check("round trip: parsed child mode matches the requested mode",
        build_parser().parse_args(on[3:]).ckpt == "on"
        and build_parser().parse_args(off[3:]).ckpt == "off"
        and build_parser().parse_args(na[3:]).ckpt is None)

  print("formatting")
  check("fmt_ms rounds", fmt_ms(1.23456) == "1.23")
  check("fmt_ms handles None", fmt_ms(None) == "--")
  check("short truncates with an ellipsis", short("a" * 80, 10).endswith("…")
        and len(short("a" * 80, 10)) == 10)
  check("short leaves a fitting name alone", short("abc", 10) == "abc")

  # The report is printed AFTER the GPU work is done. A formatting bug here
  # would throw away the whole job, so the printers are exercised on synthetic
  # rows with output swallowed.
  print("printers")
  synthetic = [
    _fake_row("bissm", 2048, "off", wall_ms=210.0, kernels=4787),
    _fake_row("bissm", 2048, "on", wall_ms=257.9, kernels=7360),
    _fake_row("dit", 32768, "n/a", wall_ms=1432.5, kernels=3100),
    {"arm": "ussm", "length": 32768, "checkpoint_boundary_prefill": "off",
     "error": "OOM: tried to allocate 40 GiB"},
  ]
  import io
  try:
    with contextlib.redirect_stdout(io.StringIO()) as sink:
      for row in synthetic[:3]:
        print_case(row)
      print_summary(synthetic)
      print_checkpoint_ab(synthetic)
    text = sink.getvalue()
    printed = True
  except Exception as error:  # pragma: no cover - this is the check
    text = ""
    printed = False
    print(f"       printer raised {type(error).__name__}: {error}")
  check("print_case / print_summary / print_checkpoint_ab all run", printed)
  check("the A/B table pairs off with on", "CHECKPOINT PREFILL A/B" in text)
  check("an error row is reported, not crashed on", "ERROR" in text)
  check("the headline idle fraction is printed",
        "gpu_idle_fraction" in text)
  check("a case with no prefill still prints a prefill row",
        text.count("prefill") >= 3)

  print("guards")
  check("PhaseClock with no device records no cuda events",
        PhaseClock(device=None).gpu_ms() == {})
  clock = PhaseClock(device=None)
  with clock.region("prefill"):
    pass
  check("PhaseClock times a region on CPU",
        clock.calls["prefill"] == 1 and clock.cpu["prefill"] >= 0.0)

  failed = [label for label, ok in checks if not ok]
  print()
  print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
  if failed:
    for label in failed:
      print(f"  FAILED: {label}")
    return 1
  return 0


# Shrunk-down geometry so a real `Diffusion` fits in a head node's RAM. Width,
# depth and state size do not change WHICH regions fire or how the profiler
# nests them, which is all this test is about.
_TINY_SSM = ("model.hidden_size=64", "model.n_blocks=2", "model.n_heads=2",
             "model.cond_dim=32", "model.ssm_state_size=8",
             "model.ssm_head_dim=16", "model.ssm_chunk_size=64")
_TINY_DIT = ("model.hidden_size=64", "model.n_blocks=2", "model.n_heads=2",
             "model.cond_dim=32", "model.attn_backend=sdpa")
# arm -> (hydra overrides, the regions that must fire exactly once)
SELF_TEST_MODELS = {
  # BD SSM arms take prefill -> active -> head.
  "bissm": (_TINY_SSM, {"prefill", "active", "head"}),
  "ussm": (_TINY_SSM, {"prefill", "active", "head"}),
  # The AR objective never calls the prefill and returns raw log-softmax
  # logits, so `_subs_parameterization` never runs (diffusion.py:494-509).
  "ussm-ar": (_TINY_SSM, {"active"}),
  # The Transformer has no prefill/active split; `flex` needs a GPU to build
  # its BlockMask, so the CPU test runs the `sdpa` spelling of the same graph.
  "dit": (_TINY_DIT, {"active", "head"}),
}


def self_test_model(arms=("bissm", "ussm", "ussm-ar", "dit"), length=512,
                    block_size=256):
  """CPU-only check of the parts `self_test`'s fakes cannot reach.

  Builds a real (tiny) `Diffusion` per arm and asserts four things that the
  whole report rests on:

    1. the right regions fire, exactly once each, for that arm's code path;
    2. `patch_regions` is bitwise NON-INVASIVE -- same seed, same inputs, the
       loss with instrumentation installed is bit-identical to the loss after
       it is removed;
    3. `undo()` leaves no wrapper behind in any instance `__dict__`;
    4. the two independent clocks agree -- `PhaseClock`'s `perf_counter` and
       the profiler's own `cpu_time_total` for the same `record_function`
       scope -- and the profiler nests prefill/active/head inside forward, so
       `derive_phase_table`'s subtraction is sound.

  Needs no GPU. Slow (a minute or two per arm) because the SSM runs its CPU
  reference scan, which is why it is not part of `--self-test`.
  """
  import sizing_sweep as ss
  from dataloader import DNATokenizer
  from diffusion import Diffusion
  from torch.profiler import ProfilerActivity, profile

  checks = []

  def check(label, condition):
    checks.append((label, bool(condition)))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

  for arm in arms:
    extra, expected = SELF_TEST_MODELS[arm]
    print(f"\n{arm}  (L={length}, block={block_size}, tiny width/depth)")
    try:
      config = ss.build(arm, length, block_size, 1, False, extra=extra)
      torch.manual_seed(0)
      model = Diffusion(config, DNATokenizer()).to("cpu")
      model.train()
    except Exception as error:  # pragma: no cover - environment dependent
      check(f"{arm}: model builds on CPU ({type(error).__name__}: "
            f"{short(str(error), 60)})", False)
      continue

    x0 = torch.randint(8, 12, (1, length))
    mask = torch.ones_like(x0)
    clock = PhaseClock(device=None, record=True)
    undo = patch_regions(model, clock)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
      with clock.region("forward"):
        torch.manual_seed(1)
        patched = model._loss(x0, mask)
      with clock.region("backward"):
        patched.loss.backward()
    undo()

    check(f"{arm}: regions fired = {sorted(expected)}",
          {name for name, count in clock.calls.items()
           if name not in TIMED_PHASES} == expected)
    check(f"{arm}: each region fired exactly once",
          all(count == 1 for name, count in clock.calls.items()
              if name in expected))
    leftovers = [attr for attr in
                 ("forward", "forward_active", "prefill_left",
                  "prefill_right", "prefill_left_boundaries_stacked",
                  "prefill_right_boundaries_stacked")
                 if attr in model.backbone.__dict__]
    leftovers += ["_subs_parameterization"
                  if "_subs_parameterization" in model.__dict__ else ""]
    check(f"{arm}: undo() leaves no wrapper behind",
          not [item for item in leftovers if item])

    after_clock = PhaseClock(device=None)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    clean = model._loss(x0, mask)
    check(f"{arm}: no region fires after undo()", dict(after_clock.calls) == {})
    check(f"{arm}: instrumentation is bitwise non-invasive",
          torch.equal(patched.loss.detach(), clean.loss.detach()))

    trace = analyse_profile(prof)
    regions = trace["regions"]
    check(f"{arm}: profiler sees every region the clock did",
          set(regions) >= (expected | {"forward", "backward"}))
    leaves = [name for name in LEAF_PHASES if name in regions]
    check(f"{arm}: forward nests its leaves "
          f"(profiler subtree >= prefill+active+head)",
          regions["forward"]["cpu_us"]
          >= sum(regions[name]["cpu_us"] for name in leaves) - 1.0)
    # `PhaseClock`'s perf_counter window sits strictly INSIDE the
    # `record_function` scope (see `PhaseClock.region`), so the profiler's
    # figure must be the larger of the two at any region size. That is the
    # structural invariant and it is checked unconditionally.
    contained = []
    worst = 0.0
    worst_region = "-"
    for name in leaves + ["forward"]:
      clocked_us = clock.cpu[name] * 1e6
      profiled_us = regions[name]["cpu_us"]
      contained.append(profiled_us >= clocked_us - 1.0)
      # The gap is the profiler's own per-record overhead, which is roughly
      # constant per region and so is only negligible relative to a region
      # that is itself big. Below 10 ms the ratio measures profiler overhead,
      # not disagreement, so only the invariant above applies there.
      if clocked_us > 1e4:
        error = abs(clocked_us - profiled_us) / clocked_us
        if error > worst:
          worst, worst_region = error, name
    check(f"{arm}: profiler scope contains the clock window in every region",
          all(contained))
    # 5%, not 1%: this runs the SSM's CPU *reference* scan, which issues a
    # very large number of very small ATen ops, so per-record profiler
    # overhead is at its worst here (measured 2.0% on `ussm-ar`'s forward).
    # On the GPU the same regions are far fewer, far larger dispatches. The
    # point of the check is that the two clocks measure the SAME region, not
    # that the profiler is free.
    check(f"{arm}: the two clocks agree to <5% on regions >10 ms "
          f"(worst {worst * 100:.2f}% on {worst_region})",
          worst < 0.05)
    check(f"{arm}: a CPU run reports zero CUDA kernels and zero launches",
          trace["kernel_count"] == 0 and trace["launch_api_calls"] == 0)

  failed = [label for label, ok in checks if not ok]
  print()
  print(f"{len(checks) - len(failed)}/{len(checks)} model checks passed")
  if failed:
    for label in failed:
      print(f"  FAILED: {label}")
    return 1
  return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--arms", default="ussm-ar,ussm,bissm,dit",
                      help="comma separated; keys of sizing_sweep.ARMS")
  parser.add_argument("--lengths", default="2048,8192,32768")
  parser.add_argument("--batch-size", type=int, default=2,
                      help="micro batch; the published scaling table is 2")
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--checkpoint-prefill", dest="ckpt",
                      action="store_const", const="on", default=None,
                      help="force model.checkpoint_boundary_prefill=true")
  parser.add_argument("--no-checkpoint-prefill", dest="ckpt",
                      action="store_const", const="off",
                      help="force model.checkpoint_boundary_prefill=false; "
                           "with neither flag BOTH modes run and the paired "
                           "delta is printed")
  parser.add_argument("--warmup", type=int, default=3,
                      help="un-measured steps; the first pays Triton "
                           "autotuning and, for `dit`, flex-attention "
                           "compilation")
  parser.add_argument("--iters", type=int, default=5,
                      help="measured un-profiled steps; the median is the "
                           "reported wall clock")
  parser.add_argument("--top", type=int, default=12,
                      help="rows in the top-kernel tables")
  parser.add_argument("--json", dest="json_path", type=Path, default=None)
  parser.add_argument("--isolate", dest="isolate", action="store_true",
                      default=True,
                      help="one subprocess per case (default). Required for "
                           "`dit`: flex attention compiles static shapes and "
                           "a second length in the same process dies inside "
                           "Inductor (sizing_sweep.py docstring)")
  parser.add_argument("--no-isolate", dest="isolate", action="store_false")
  parser.add_argument("--self-test", action="store_true",
                      help="run the CPU-only checks on the pure helpers and "
                           "exit; needs no GPU, takes a second")
  parser.add_argument("--self-test-model", action="store_true",
                      help="also build a real (tiny) Diffusion per arm on CPU "
                           "and check that the right regions fire, that the "
                           "instrumentation is bitwise non-invasive, and that "
                           "the clock and the profiler agree; needs no GPU "
                           "but takes a few minutes")
  return parser


def child_argv(arm, length, mode, args):
  argv = [sys.executable, "-u", str(Path(__file__).resolve()),
          "--arms", arm, "--lengths", str(length),
          "--batch-size", str(args.batch_size),
          "--block-size", str(args.block_size),
          "--warmup", str(args.warmup), "--iters", str(args.iters),
          "--top", str(args.top), "--no-isolate"]
  if mode == "on":
    argv.append("--checkpoint-prefill")
  elif mode == "off":
    argv.append("--no-checkpoint-prefill")
  return argv


def main_cli(argv=None):
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.self_test or args.self_test_model:
    status = self_test()
    if args.self_test_model:
      requested = [a.strip() for a in args.arms.split(",") if a.strip()]
      covered = [a for a in requested if a in SELF_TEST_MODELS]
      skipped = [a for a in requested if a not in SELF_TEST_MODELS]
      if skipped:
        print(f"\nno CPU-runnable spelling for {skipped}; skipping "
              f"(dit-ar needs flash-attn, which is CUDA only)")
      status |= self_test_model(covered or tuple(SELF_TEST_MODELS))
    return status

  arms = [a.strip() for a in args.arms.split(",") if a.strip()]
  lengths = [int(v) for v in args.lengths.split(",") if v.strip()]

  import sizing_sweep as ss
  unknown = [a for a in arms if a not in ss.ARMS]
  if unknown:
    parser.error(f"unknown arms {unknown}; choose from {sorted(ss.ARMS)}")
  supports = {arm: ss.ARMS[arm][2] for arm in ss.ARMS}
  cases = plan_cases(arms, lengths, args.ckpt or "both", supports)

  if not torch.cuda.is_available():
    raise SystemExit(
      "bd_step_breakdown needs a CUDA GPU: it profiles CUDA kernels and "
      "counts CUDA launches, neither of which exists on a head node. Submit "
      "scripts/smoke/bd_step_breakdown.sh instead, or run --self-test to "
      "exercise the CPU-only helpers.")

  device = torch.device("cuda")
  total_gib = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
  print(f"device: {torch.cuda.get_device_name(device)}  "
        f"total {total_gib:.2f} GiB   torch {torch.__version__}")
  print(f"cases: {len(cases)}  isolate={args.isolate}")

  rows = []
  if args.isolate and len(cases) > 1:
    scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    for arm, length, mode in cases:
      shard = scratch / f"bd_step_breakdown_{arm}_{length}_{mode}_{os.getpid()}.json"
      argv_child = child_argv(arm, length, mode, args) + [
        "--json", str(shard)]
      print(f"\n$ {' '.join(argv_child)}", flush=True)
      completed = subprocess.run(argv_child, cwd=str(REPO))
      if shard.exists():
        rows.extend(json.loads(shard.read_text())["rows"])
        shard.unlink()
      else:
        rows.append({"arm": arm, "length": length,
                     "checkpoint_boundary_prefill": mode,
                     "error": f"child exited {completed.returncode}"})
  else:
    for arm, length, mode in cases:
      try:
        row = run_case(arm, length, mode, args, device)
      except torch.cuda.OutOfMemoryError as error:
        row = {"arm": arm, "length": length,
               "checkpoint_boundary_prefill": mode,
               "error": f"OOM: {error}"}
      else:
        print_case(row)
      rows.append(row)
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats(device)

  if len(rows) > 1:
    print_summary(rows)
    print_checkpoint_ab(rows)

  if args.json_path:
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(
      {"device": torch.cuda.get_device_name(device),
       "total_gib": total_gib,
       "torch": torch.__version__,
       "batch_size": args.batch_size,
       "block_size": args.block_size,
       "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.json_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main_cli())
