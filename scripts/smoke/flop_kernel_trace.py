#!/usr/bin/env python
"""A second, independent view on the SSM FLOP residual, via torch.profiler.

`scripts/eval/training_flops.py` undercounts the SSM arms by a constant 1.35x
(bissm) to 1.37x (ussm-ar) against `FlopCounterMode`, at every length from 2k
to 32k. Every per-module term is nonetheless right: summed over leaves,
in_proj + conv1d + out_proj + mlp + scan + head reproduces the analytic total
to five figures (results/sizing/flop_breakdown.json). The gap -- 486 GFLOP at
L=2048 batch 1, 27% of the dispatched total -- is attributed by the counter to
NO nn.Module, and is larger than the whole MLP term.

`FlopCounterMode` and `torch.profiler` are blind in opposite directions, which
is the point of running both:

  FlopCounterMode  sees every aten dispatch and knows its FLOP cost, but never
                   sees a Triton kernel (mamba-ssm's SSD scan, the gated
                   RMSNorm) because those bypass the dispatcher entirely. It
                   also loses MODULE ownership for anything called as a plain
                   method rather than through `nn.Module.__call__` --
                   `prefill_left_boundaries_stacked` (diffusion.py:1174),
                   `prefill_right_boundaries_stacked` (diffusion.py:1183) and
                   `forward_active` (diffusion.py:1145, :1195) are all method
                   calls, so raw tensor math in their bodies lands in the
                   "Global" bucket and in no module. Child `nn.Linear` /
                   `nn.Conv1d` calls inside them still fire their own hooks and
                   ARE attributed, which is exactly why every per-module term
                   can be right while the total is not.

  torch.profiler  sees every launched CUDA kernel, Triton included, but knows
                  nothing about FLOPs beyond a handful of hard-coded shape
                  formulas. What it does carry is `record_shapes`, so the FLOPs
                  can be re-derived from the shapes with the same 2*m*k*n
                  convention `training_flops.py` uses.

So this script profiles one forward+backward and reports:

  1. the top kernels by self CUDA time, split into Triton and aten kernels with
     a subtotal for each -- the Triton subtotal is the share of runtime the
     dispatch counter is structurally blind to;
  2. which CPU-side record launched each kernel, split into `aten::*` owners
     and everything else (a custom autograd Function body, i.e. Triton);
  3. the top aten ops by self CUDA time with their recorded input shapes;
  4. the profiler's own `with_flops` estimate wherever it has one; and
  5. a matmul census: for every LEAF matmul-like op, the implied 2*m*k*n from
     the recorded shapes, ranked, cumulative, and normalised per token per
     layer, so a row can be matched against a claimed residual by eye.

"Leaf" means an op that launched a kernel itself. `aten::linear` and its child
`aten::addmm` both appear in the trace with matmul-shaped inputs; only the
child launches the GEMM, so counting both would double the census. Filtering
on "has its own kernels" resolves that exactly and is reported.

Optionally (default on) it then reruns the same step under `FlopCounterMode` so
the dispatched total, the shape-implied total and the analytic total are
printed side by side from ONE invocation.

  python scripts/smoke/flop_kernel_trace.py --arm bissm --length 8192 --batch 1

Needs a GPU. Writes results/sizing/flop_kernel_trace_<arm>_L<length>.json,
a chrome trace next to it, and (with --stack) a folded-stack file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

# `measured_flops_sweep` owns the arm -> hydra-override mapping, and importing
# it also registers the OmegaConf resolvers via its own `import main`. Building
# the model the same way is the whole point: the two views must be of the same
# graph, not of two configs that merely look alike.
import scripts.eval.measured_flops_sweep as mfs  # noqa: E402
import scripts.eval.training_flops as tf  # noqa: E402

# Kernels mamba-ssm and causal-conv1d launch, by the bare Python function name
# Triton gives them. `causal_conv1d` is not installed in this environment (the
# conv path falls back to `F.conv1d`, mamba2_segment.py:190), so its rows are
# expected to be absent rather than zero.
TRITON_HINTS = (
  "chunk_scan", "chunk_state", "chunk_cumsum", "bmm_chunk", "state_passing",
  "layer_norm", "layernorm", "swiglu", "selective_scan", "causal_conv1d",
  "mamba", "softplus",
)
GEMM_HINTS = ("gemm", "cutlass", "cublas", "nvjet", "xmma", "s16816", "sgemm",
              "dot_kernel", "splitk")
MEMORY_HINTS = ("memcpy", "memset")

MATMUL_OPS = ("aten::mm", "aten::bmm", "aten::matmul", "aten::addmm",
              "aten::baddbmm", "aten::addbmm", "aten::linear", "aten::mv",
              "aten::_scaled_mm", "aten::einsum")
CONV_OPS = ("aten::conv1d", "aten::conv2d", "aten::convolution",
            "aten::_convolution", "aten::cudnn_convolution",
            "aten::convolution_backward")


def prod(values):
  out = 1
  for value in values:
    out *= value
  return out


def pair_flops(left, right):
  """2*m*k*n for a batched (m x k) @ (k x n) product, or None if not one."""
  if len(left) < 2 or len(right) < 2:
    return None
  m, k = left[-2], left[-1]
  k_other, n = right[-2], right[-1]
  if k != k_other:
    return None
  batch = max(prod(left[:-2]), prod(right[:-2]))
  return 2 * batch * m * k * n


def implied_flops(name, shapes):
  """FLOPs implied by an op's recorded input shapes.

  Returns ``(flops, kind, note)`` with ``flops=None`` when the op is not
  matmul-like or its shapes were not recorded. The convention matches
  `training_flops.py`: an (m x k) @ (k x n) product costs 2*m*k*n, one multiply
  and one add per MAC.
  """
  if not shapes:
    return None, "", ""
  tensors = [tuple(s) for s in shapes if isinstance(s, (list, tuple)) and s]
  mats = [s for s in tensors if len(s) >= 2]
  base = name.split(".")[0]

  if base == "aten::linear" and len(mats) >= 2:
    # weight is [out, in]; input is [..., in].
    inp, weight = mats[0], mats[1]
    if inp[-1] == weight[-1]:
      return 2 * prod(inp[:-1]) * inp[-1] * weight[0], "matmul", ""
    return None, "matmul", "unparsed"

  if base in MATMUL_OPS:
    if base == "aten::einsum":
      # einsum records no tensor shapes; its bmm/mm child carries them.
      return None, "matmul", "no shapes"
    if len(mats) >= 2:
      flops = pair_flops(mats[-2], mats[-1])
      if flops is not None:
        return flops, "matmul", ""
    return None, "matmul", "unparsed"

  if base in CONV_OPS:
    # [batch, c_in, l] * [c_out, c_in/groups, k]. The output length is taken as
    # the input length, which is exact for the padded full-width convolution
    # and overstates the d_conv-wide boundary fix-up by a few columns. Grouping
    # needs no special case: the weight's second axis is already c_in/groups,
    # which is 1 for this depthwise convolution.
    backward = base == "aten::convolution_backward"
    # convolution_backward is called (grad_output, input, weight, ...).
    offset = 1 if backward else 0
    if len(tensors) < 2 + offset:
      return None, "conv", "unparsed"
    inp, weight = tensors[offset], tensors[offset + 1]
    if len(inp) < 3 or len(weight) < 3:
      return None, "conv", "unparsed"
    flops = 2 * inp[0] * inp[-1] * weight[0] * weight[1] * prod(weight[2:])
    # The backward computes grad_input and grad_weight, each one
    # forward-sized convolution.
    return (2 * flops if backward else flops), "conv", "~approx"

  return None, "", ""


def gemm_label(shapes):
  """Name the model GEMM a pair of shapes belongs to, forward or backward."""
  mats = [tuple(s) for s in shapes
          if isinstance(s, (list, tuple)) and len(s) >= 2]
  if len(mats) < 2:
    return "?"
  known = {
    (tf.D_MODEL, tf.PROJ_DIM): "in_proj",
    (tf.D_INNER, tf.D_MODEL): "out_proj",
    (tf.D_MODEL, tf.MLP_HIDDEN): "mlp",
    (tf.MLP_HIDDEN, tf.D_MODEL): "mlp",
    (tf.D_MODEL, tf.VOCAB): "head",
  }
  dims = {mats[-2][-2], mats[-2][-1], mats[-1][-2], mats[-1][-1]}
  for (a, b), label in known.items():
    if a in dims and b in dims:
      return label
  return "?"


def classify_kernel(name):
  """Triton kernels enter the trace under a bare Python identifier.

  aten kernels arrive as demangled C++ (``void at::native::...``,
  ``void cutlass::Kernel<...>``), so the absence of ``::``/``<``/``void`` is a
  reliable separator, and the mamba-ssm hint list then says which Triton.
  """
  lowered = name.lower()
  looks_triton = ("::" not in name and "<" not in name
                  and not name.startswith("void "))
  if looks_triton:
    if any(hint in lowered for hint in TRITON_HINTS):
      return "triton-mamba"
    if lowered.startswith("triton_") or lowered.startswith("_"):
      return "triton-other"
  if any(hint in lowered for hint in MEMORY_HINTS):
    return "memory"
  if any(hint in lowered for hint in GEMM_HINTS):
    return "aten-gemm"
  if looks_triton:
    return "triton-other"
  return "aten-other"


def fmt_shapes(shapes, width=44):
  if not shapes:
    return ""
  parts = []
  for shape in shapes:
    if isinstance(shape, (list, tuple)) and shape:
      parts.append("x".join(str(int(v)) for v in shape))
  text = ",".join(parts)
  return text if len(text) <= width else text[:width - 1] + "…"


def fmt_name(name, width=62):
  return name if len(name) <= width else name[:width - 1] + "…"


def build_batch(length, batch, device):
  x0 = torch.randint(8, 12, (batch, length), device=device)
  return x0, torch.ones_like(x0)


def run_step(model, x0, mask):
  model.zero_grad(set_to_none=True)
  outputs = model._loss(x0, mask)
  outputs.loss.backward()
  return float(outputs.loss.detach())


def collect_kernels(events):
  """Aggregate CUDA kernel events by name.

  Kernel events are separate `FunctionEvent`s with ``device_type == CUDA``
  (torch/autograd/profiler.py:600-641); for those, ``self_device_time_total``
  is the kernel's own wall duration (profiler_util.py:624-637).
  """
  rows = defaultdict(lambda: {"count": 0, "us": 0.0})
  for event in events:
    if event.device_type != DeviceType.CUDA:
      continue
    row = rows[event.name]
    row["count"] += 1
    row["us"] += float(event.self_device_time_total)
  return rows


def collect_ownership(events):
  """Join each kernel to the CPU record that launched it.

  ``FunctionEvent.kernels`` holds only the kernels a record launched DIRECTLY:
  the correlation walk appends a kernel to exactly one frontend op
  (torch/autograd/profiler.py:655-671), and ``device_time_total`` adds the
  children's on top only afterwards (profiler_util.py:594-606). So summing
  ``.kernels`` over every CPU record double counts nothing.
  """
  rows = defaultdict(lambda: {"count": 0, "us": 0.0})
  for event in events:
    if event.device_type != DeviceType.CPU:
      continue
    for kernel in event.kernels:
      row = rows[(kernel.name, event.name)]
      row["count"] += 1
      row["us"] += float(kernel.duration)
  return rows


def collect_ops(events):
  """Aggregate CPU-side op records by (name, input shapes).

  Keeps `self_device_time_total` (kernel time this record launched itself),
  the profiler's `with_flops` estimate, whether the record launched a kernel
  at all, and the FLOPs implied by its shapes.
  """
  rows = {}
  for event in events:
    if event.device_type != DeviceType.CPU:
      continue
    key = (event.name, json.dumps(event.input_shapes))
    row = rows.get(key)
    if row is None:
      flops, kind, note = implied_flops(event.name, event.input_shapes)
      row = rows[key] = {
        "name": event.name, "shapes": event.input_shapes, "count": 0,
        "self_device_us": 0.0, "cpu_us": 0.0, "profiler_flops": 0,
        "kernel_launches": 0, "implied_per_call": flops, "kind": kind,
        "note": note,
        "label": gemm_label(event.input_shapes) if kind == "matmul" else "",
      }
    row["count"] += 1
    row["self_device_us"] += float(event.self_device_time_total)
    row["cpu_us"] += float(event.self_cpu_time_total)
    row["profiler_flops"] += int(event.flops or 0)
    row["kernel_launches"] += len(event.kernels)
  for row in rows.values():
    per_call = row["implied_per_call"]
    row["implied_total"] = (
      None if per_call is None else per_call * row["count"])
  return list(rows.values())


def analytic_reference(arm, length, block, batch, checkpoint_prefill):
  """`training_flops.py`'s own numbers for this point, or None.

  That module hardcodes the run geometry as constants (LENGTH=8192, BLOCK=256),
  so quoting it at any other point would silently mix geometries.

  Three corrections separate its number from what a dispatch counter should
  report, and keeping them apart is the whole point of the table:

  * `scan_flops` is carried by the Triton SSD kernel, which never dispatches,
    so a counter cannot see it however right the formula is;
  * `checkpoint_prefill_flops` is the boundary prefill's forward run a SECOND
    time, because `checkpoint_boundary_prefill` recomputes each layer during
    backward (bidirectional_ssm.py:466-477). `training_flops.py` charges the
    prefill 3x; with checkpointing on it actually runs 4x. This applies to
    bissm only -- the AR arm never calls the prefill;
  * `expected_counter_flops` is the sum a complete formula would predict for
    `FlopCounterMode`, and is what the residual should be measured against.
  """
  if length != tf.LENGTH or block != tf.BLOCK:
    return None
  terms, passes = tf.ssm_terms(), tf.ssm_passes()
  clean_scan = tf.N_LAYERS * tf.PREFIX * 2 * terms["scan"]
  extra = extra_visible = 0
  if arm == "bissm":
    forward = passes["clean"] + passes["act_bi"]
    # clean carries 2*scan per prefix token; act_bi carries 2*scan per token
    # (two mixers), both per layer.
    scan = clean_scan + tf.N_LAYERS * tf.LENGTH * 2 * terms["scan"]
    if checkpoint_prefill:
      extra = passes["clean"]
      extra_visible = passes["clean"] - clean_scan
  elif arm == "ussm-ar":
    forward = passes["ar"]
    scan = tf.N_LAYERS * (tf.LENGTH - 1) * terms["scan"]
  else:
    return None
  visible = tf.GRAD_MULT * (forward - scan) * batch
  return {
    "forward_flops": forward * batch,
    "training_flops": tf.GRAD_MULT * forward * batch,
    "scan_flops": tf.GRAD_MULT * scan * batch,
    "visible_flops": visible,
    "checkpoint_prefill": bool(checkpoint_prefill),
    "checkpoint_prefill_flops": extra * batch,
    "checkpoint_prefill_visible_flops": extra_visible * batch,
    "expected_counter_flops": visible + extra_visible * batch,
  }


def print_kernels(kernels, top, total_us):
  print(f"\n=== 1. top {top} CUDA kernels by self device time "
        f"({len(kernels)} distinct) ===")
  print(f"{'#':>3} {'class':<13} {'count':>7} {'ms':>9} {'%':>6}  kernel")
  print("-" * 118)
  ranked = sorted(kernels.items(), key=lambda kv: -kv[1]["us"])
  for index, (name, row) in enumerate(ranked[:top], start=1):
    share = 100.0 * row["us"] / total_us if total_us else 0.0
    print(f"{index:>3} {classify_kernel(name):<13} {row['count']:>7} "
          f"{row['us'] / 1e3:>9.3f} {share:>6.2f}  {fmt_name(name)}")


def print_kernel_classes(kernels, total_us):
  print("\n=== 2. kernel class subtotals -- what a dispatch counter cannot "
        "see ===")
  buckets = defaultdict(lambda: {"count": 0, "us": 0.0, "kinds": 0})
  for name, row in kernels.items():
    bucket = buckets[classify_kernel(name)]
    bucket["count"] += row["count"]
    bucket["us"] += row["us"]
    bucket["kinds"] += 1
  print(f"{'class':<15} {'kinds':>6} {'launches':>10} {'ms':>10} {'%':>7}")
  print("-" * 52)
  for name, bucket in sorted(buckets.items(), key=lambda kv: -kv[1]["us"]):
    share = 100.0 * bucket["us"] / total_us if total_us else 0.0
    print(f"{name:<15} {bucket['kinds']:>6} {bucket['count']:>10} "
          f"{bucket['us'] / 1e3:>10.3f} {share:>7.2f}")
  triton = sum(v["us"] for k, v in buckets.items() if k.startswith("triton"))
  share = 100.0 * triton / total_us if total_us else 0.0
  print("-" * 52)
  print(f"{'TRITON TOTAL':<15} {'':>6} {'':>10} {triton / 1e3:>10.3f} "
        f"{share:>7.2f}   <- invisible to FlopCounterMode")
  print(f"{'ALL KERNELS':<15} {'':>6} {'':>10} {total_us / 1e3:>10.3f} "
        f"{100.0:>7.2f}")
  return {name: dict(bucket) for name, bucket in buckets.items()}


def print_ownership(ownership, top, total_us):
  print(f"\n=== 3. top {top} (kernel, launching CPU record) pairs ===")
  print("A kernel launched from an `aten::*` record is one the dispatch")
  print("counter saw; anything else is a custom autograd Function body, whose")
  print("raw tensor math the counter attributes to no module.")
  print(f"{'#':>3} {'ms':>9} {'count':>7}  {'owner':<40} kernel")
  print("-" * 118)
  ranked = sorted(ownership.items(), key=lambda kv: -kv[1]["us"])
  for index, ((kernel, owner), row) in enumerate(ranked[:top], start=1):
    print(f"{index:>3} {row['us'] / 1e3:>9.3f} {row['count']:>7}  "
          f"{fmt_name(owner, 40):<40} {fmt_name(kernel, 56)}")
  aten_us = sum(v["us"] for (_, owner), v in ownership.items()
                if owner.startswith("aten::"))
  other_us = sum(v["us"] for (_, owner), v in ownership.items()
                 if not owner.startswith("aten::"))
  owned = aten_us + other_us
  print("-" * 118)
  print(f"  owned by aten:: records   {aten_us / 1e3:>10.3f} ms  "
        f"{100.0 * aten_us / total_us if total_us else 0:>6.2f}%")
  print(f"  owned by other records    {other_us / 1e3:>10.3f} ms  "
        f"{100.0 * other_us / total_us if total_us else 0:>6.2f}%")
  print(f"  launched from no record   {(total_us - owned) / 1e3:>10.3f} ms  "
        f"{100.0 * (total_us - owned) / total_us if total_us else 0:>6.2f}%")
  return {"aten_us": aten_us, "other_us": other_us,
          "unowned_us": total_us - owned}


def print_ops(ops, top):
  print(f"\n=== 4. top {top} aten ops by self device time, with shapes ===")
  print("`prof flops` is the profiler's own with_flops estimate (0 where it "
        "has no formula).")
  print("`2mkn` is this script's, from the recorded shapes -- the two should "
        "agree where both exist.")
  print(f"{'#':>3} {'ms':>8} {'count':>7} {'prof GF':>9} {'2mkn GF':>9} "
        f"{'op':<30} shapes")
  print("-" * 130)
  ranked = sorted(ops, key=lambda row: -row["self_device_us"])
  for index, row in enumerate(ranked[:top], start=1):
    prof = row["profiler_flops"] / 1e9
    implied = row["implied_total"]
    implied_text = "-" if implied is None else f"{implied / 1e9:9.2f}"
    print(f"{index:>3} {row['self_device_us'] / 1e3:>8.3f} {row['count']:>7} "
          f"{prof:>9.2f} {implied_text:>9} {fmt_name(row['name'], 30):<30} "
          f"{fmt_shapes(row['shapes'], 46)}")


def print_matmul_census(ops, tokens, n_layers, top):
  """Every leaf matmul-like op, ranked by the FLOPs its shapes imply.

  Only records that launched a kernel themselves are counted: `aten::linear`
  and its child `aten::addmm` both carry matmul-shaped inputs, but only the
  child issues the GEMM, so counting both would double the census.
  """
  leaves = [row for row in ops
            if row["implied_total"] and row["kernel_launches"] > 0]
  skipped = [row for row in ops
             if row["implied_total"] and row["kernel_launches"] == 0]
  total = sum(row["implied_total"] for row in leaves)
  print("\n=== 5. matmul/conv census: FLOPs implied by recorded shapes ===")
  print(f"{len(leaves)} leaf rows (launched a kernel), "
        f"{len(skipped)} non-leaf rows excluded to avoid double counting")
  print("Covers the GEMM and convolution families, which is FlopCounterMode's"
        " whole")
  print("registry for these arms; the sdpa family is NOT shape-derived here, "
        "so this")
  print("census is comparable to the counter for the SSM arms only.")
  print(f"{'#':>3} {'GFLOP':>10} {'cum%':>6} {'/tok/layer':>11} {'count':>7} "
        f"{'label':<9} {'op':<26} shapes")
  print("-" * 138)
  ranked = sorted(leaves, key=lambda row: -row["implied_total"])
  running = 0.0
  for index, row in enumerate(ranked[:top], start=1):
    running += row["implied_total"]
    per_token = row["implied_total"] / (tokens * n_layers) if tokens else 0.0
    print(f"{index:>3} {row['implied_total'] / 1e9:>10.2f} "
          f"{100.0 * running / total if total else 0:>6.1f} "
          f"{per_token:>11,.0f} {row['count']:>7} "
          f"{row['label'] + row['note']:<9} {fmt_name(row['name'], 26):<26} "
          f"{fmt_shapes(row['shapes'], 40)}")
  print("-" * 138)
  print(f"{'':>3} {total / 1e9:>10.2f} {100.0:>6.1f} "
        f"{total / (tokens * n_layers) if tokens else 0:>11,.0f}  TOTAL "
        f"implied by shapes")
  if skipped:
    print("\nexcluded non-leaf rows (their child op carries the same work):")
    for row in sorted(skipped, key=lambda r: -r["implied_total"])[:8]:
      print(f"    {row['implied_total'] / 1e9:>10.2f} GFLOP  "
            f"{row['name']:<24} {fmt_shapes(row['shapes'], 40)}")
  return total, leaves


def print_reconciliation(implied_total, counted_total, analytic, tokens,
                         n_layers):
  print("\n=== 6. reconciliation ===")
  print(f"{'quantity':<44}{'GFLOP':>12}{'/tok/layer':>13}")
  print("-" * 69)

  def row(label, value):
    if value is None:
      print(f"{label:<44}{'n/a':>12}{'':>13}")
      return
    per_token = value / (tokens * n_layers) if tokens else 0.0
    print(f"{label:<44}{value / 1e9:>12.2f}{per_token:>13,.0f}")

  row("A. implied by profiler shapes (leaf ops)", implied_total)
  row("B. FlopCounterMode dispatched total", counted_total)
  if analytic:
    row("C. training_flops.py training total", analytic["training_flops"])
    row("D.   of which Triton scan (invisible)", analytic["scan_flops"])
    row("E. C - D, the dispatchable part", analytic["visible_flops"])
    row("F. + prefill recompute, dispatchable",
        analytic["checkpoint_prefill_visible_flops"])
    row("G. E + F, what a counter should see",
        analytic["expected_counter_flops"])
  print("-" * 69)
  if counted_total and implied_total:
    delta = implied_total - counted_total
    print(f"A - B = {delta / 1e9:>10.2f} GFLOP "
          f"({100.0 * delta / counted_total:+.2f}%)  "
          "-- shape re-derivation vs the counter; near zero means")
    print("        every dispatched FLOP is accounted for by a named op and "
          "shape.")
  if analytic and counted_total:
    expected = analytic["expected_counter_flops"]
    residual = counted_total - expected
    per_token = residual / (tokens * n_layers) if tokens else 0.0
    ratio = residual / expected if expected else 0.0
    print(f"B - G = {residual / 1e9:>10.2f} GFLOP "
          f"({ratio:+.3f}x of G, {per_token:,.0f} FLOP/token/layer)")
    print("        -- THE RESIDUAL. Find the section 5 census rows that sum "
          "to it.")
    if not analytic["checkpoint_prefill"]:
      print("        (checkpointing is OFF here, so F is zero; rerun with "
            "--checkpoint-prefill on")
      print("         to see how much of a bissm residual is prefill "
            "recompute)")


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", default="bissm", choices=sorted(mfs.ARMS))
  parser.add_argument("--length", type=int, default=8192)
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--n-layers", type=int, default=tf.N_LAYERS,
                      help="only used to normalise FLOPs per token per layer")
  parser.add_argument("--warmup", type=int, default=2,
                      help="steps outside the profiler; the first step pays "
                           "Triton autotuning and allocator growth, which "
                           "would otherwise dominate every kernel row")
  parser.add_argument("--top", type=int, default=30)
  parser.add_argument("--checkpoint-prefill", default="default",
                      choices=("default", "on", "off"),
                      help="override model.checkpoint_boundary_prefill. "
                           "`measured_flops_sweep` turns it ON for bissm, "
                           "which makes the prefill forward run twice; "
                           "`off` isolates that term")
  parser.add_argument("--stack", dest="stack", action="store_true",
                      default=True)
  parser.add_argument("--no-stack", dest="stack", action="store_false",
                      help="with_stack costs real overhead and inflates the "
                           "trace; drop it if the trace is unmanageable")
  parser.add_argument("--flop-counter", dest="flop_counter",
                      action="store_true", default=True)
  parser.add_argument("--no-flop-counter", dest="flop_counter",
                      action="store_false")
  parser.add_argument("--label", default=None)
  parser.add_argument("--output-dir", type=Path,
                      default=REPO / "results" / "sizing")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU; head nodes have none")
  device = torch.device("cuda")
  label = args.label or f"flop_kernel_trace_{args.arm}_L{args.length}"
  args.output_dir.mkdir(parents=True, exist_ok=True)

  print(f"arm={args.arm} length={args.length} batch={args.batch} "
        f"block={args.block_size}")
  print(f"{torch.cuda.get_device_name(0)}  torch {torch.__version__}")

  torch.manual_seed(0)
  config = mfs.build(args.arm, args.length, args.block_size, args.batch)
  if args.checkpoint_prefill != "default":
    if "checkpoint_boundary_prefill" not in config.model:
      raise ValueError(
        f"arm {args.arm!r} has no model.checkpoint_boundary_prefill to set")
    config.model.checkpoint_boundary_prefill = args.checkpoint_prefill == "on"
  checkpoint_prefill = bool(
    config.model.get("checkpoint_boundary_prefill", False))
  print(f"checkpoint_boundary_prefill={checkpoint_prefill}"
        + ("  <- the boundary prefill's forward runs TWICE, once in forward "
           "and once recomputed in backward (bidirectional_ssm.py:466-477); "
           "training_flops.py charges it 3x, not 4x"
           if checkpoint_prefill and args.arm == "bissm" else ""))
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  x0, mask = build_batch(args.length, args.batch, device)

  for _ in range(args.warmup):
    run_step(model, x0, mask)
  torch.cuda.synchronize()

  start = time.perf_counter()
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
               record_shapes=True, with_stack=args.stack,
               with_flops=True) as prof:
    loss_value = run_step(model, x0, mask)
    torch.cuda.synchronize()
  wall = time.perf_counter() - start
  print(f"profiled step: loss={loss_value:.4f}  wall={wall * 1e3:.1f} ms "
        f"(profiler overhead included)")

  events = prof.events()
  kernels = collect_kernels(events)
  total_us = sum(row["us"] for row in kernels.values())
  ownership = collect_ownership(events)
  ops = collect_ops(events)

  # AR shifts the target by one token (diffusion.py:1016); the block-diffusion
  # arms supervise the full length.
  is_ar = mfs.ARMS[args.arm][1] == "ar"
  tokens = args.batch * (args.length - 1 if is_ar else args.length)
  print(f"tokens={tokens} layers={args.n_layers}  "
        f"total CUDA kernel time={total_us / 1e3:.3f} ms")

  print_kernels(kernels, args.top, total_us)
  classes = print_kernel_classes(kernels, total_us)
  owner_totals = print_ownership(ownership, args.top, total_us)
  print_ops(ops, args.top)
  implied_total, leaves = print_matmul_census(ops, tokens, args.n_layers,
                                              args.top)

  counted_total = None
  if args.flop_counter:
    # `prof` stays alive -- the chrome trace is exported from it below. Only
    # the materialised event list is dropped; every table above is already
    # aggregated.
    del events
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    counter = FlopCounterMode(display=False)
    with counter:
      run_step(model, x0, mask)
    counted_total = counter.get_total_flops()

  analytic = analytic_reference(args.arm, args.length, args.block_size,
                                args.batch, checkpoint_prefill)
  print_reconciliation(implied_total, counted_total, analytic, tokens,
                       args.n_layers)
  if analytic is None:
    print(f"\n(no analytic reference: training_flops.py is written for "
          f"L={tf.LENGTH} block={tf.BLOCK}, arms bissm/ussm-ar only)")

  trace_path = args.output_dir / f"{label}.json.gz"
  prof.export_chrome_trace(str(trace_path))
  print(f"\nwrote chrome trace {trace_path}")
  print("  open at https://ui.perfetto.dev (reads .gz), or gunzip for "
        "chrome://tracing")
  stacks_path = None
  if args.stack:
    stacks_path = args.output_dir / f"{label}.stacks.txt"
    try:
      prof.export_stacks(str(stacks_path), "self_cuda_time_total")
      print(f"wrote folded stacks {stacks_path}")
      print("  semicolon-separated python frames + microseconds of CUDA time; "
            "this is what")
      print("  maps kernel time back to diffusion.py / "
            "bidirectional_ssm.py line numbers")
    except Exception as exc:  # noqa: BLE001
      print(f"export_stacks failed ({type(exc).__name__}: {exc})")
      stacks_path = None

  # Every CPU record is a row, including thousands of zero-time ones; the
  # chrome trace holds the full detail, so the JSON keeps the ops that used
  # device time plus the whole matmul census.
  ranked_ops = sorted(ops, key=lambda row: -row["self_device_us"])
  payload = {
    "arm": args.arm, "length": args.length, "batch": args.batch,
    "block_size": args.block_size, "tokens": tokens,
    "n_layers": args.n_layers, "torch": torch.__version__,
    "device": torch.cuda.get_device_name(0),
    "checkpoint_boundary_prefill": checkpoint_prefill,
    "wall_ms": wall * 1e3, "with_stack": args.stack,
    "total_kernel_us": total_us,
    "op_rows_total": len(ops), "op_rows_kept": min(len(ops), 500),
    "kernel_classes": classes,
    "kernels": [{"name": name, "class": classify_kernel(name), **row}
                for name, row in sorted(kernels.items(),
                                        key=lambda kv: -kv[1]["us"])],
    "ownership": [{"kernel": kernel, "owner": owner, **row}
                  for (kernel, owner), row in sorted(
                    ownership.items(), key=lambda kv: -kv[1]["us"])],
    "ownership_totals": owner_totals,
    "ops": ranked_ops[:500],
    "matmul_census": sorted(leaves, key=lambda row: -row["implied_total"]),
    "implied_total_flops": implied_total,
    "flop_counter_total_flops": counted_total,
    "analytic": analytic,
    "trace": str(trace_path),
    "stacks": str(stacks_path) if stacks_path else None,
  }
  output = args.output_dir / f"{label}.json"
  with tempfile.NamedTemporaryFile("w", dir=args.output_dir,
                                   delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, output)
  print(f"wrote {output}")


if __name__ == "__main__":
  main_cli()
