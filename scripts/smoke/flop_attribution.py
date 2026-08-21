#!/usr/bin/env python3
"""Name the operation behind the SSM FLOP residual, by aten op and by owner.

`training_flops.py` reproduces the Transformer arms exactly and undercounts the
SSM arms by a constant 1.35x (bissm) to 1.37x (ussm-ar) at every length. Every
PER-MODULE term is nonetheless right, so 486 GFLOP at L=2048 batch 1 -- 27% of
the dispatched total, ~19.8M FLOP per token per layer -- belongs to no leaf
module. This resolves that residual by aten operator (which needs no module
attribution at all and is therefore airtight), by owning module (which needs
care, see below), and against three different baselines, because "residual"
turns out to mean three different things here.


WHY THE PER-OP TABLE IS THE PRIMARY RESULT

`FlopCounterMode` only ever counts ops in `flop_registry`
(flop_counter.py:556-575): mm, addmm, bmm, baddbmm, _scaled_mm, convolution,
_convolution, convolution_backward, and the sdpa family. Nothing else can
contribute a single FLOP. So the Global total partitions exactly into
GEMM + CONV + ATTENTION with no attribution machinery involved, and whichever
of those three the residual lands in is a fact about the dispatch stream rather
than about nn.Module bookkeeping. This script asserts that partition closes.


THE DE-NESTING RULE, AND WHY THE OBVIOUS ONE IS WRONG

`get_flop_counts()` buckets NEST: `_count_flops` credits each op to *every*
entry of `mod_tracker.parents` (flop_counter.py:752-754), so an op inside
`...mixer.in_proj` is added to the leaf, to `...mixer`, to the layer, to the
root and to "Global" alike. Summing all non-"Global" keys therefore multiply
counts.

Worse, the bucket totals are not even a tree. `ModuleTracker` reconstructs
module ownership during BACKWARD from multi-grad hooks: the module name is
re-added when its output grad fires (module_tracker.py:142-147) and removed
when its input grad fires (module_tracker.py:131-136). A module whose inputs do
not all receive grads never gets popped and keeps owning the rest of the
backward pass, until the end-of-backward callback resets `parents`
(module_tracker.py:75-79). That is not hypothetical here: in
results/sizing/flop_breakdown.json the `dropout` buckets hold 8899 GFLOP and
`net.2` holds 8667 GFLOP against a Global total of 1781 GFLOP. Any rule built on
bucket totals inherits that leak.

So this script does NOT de-nest bucket totals. It records ownership at dispatch
time and builds a DISJOINT partition instead:

  RULE. Every counted op is assigned to exactly one bucket: the deepest fqn in
  `mod_tracker.parents` at the moment it dispatched (most dotted components,
  ties broken lexicographically), or "<unowned>" when `parents` is exactly
  {"Global"}, i.e. no nn.Module forward is open at all.

  A bucket is a LEAF when its fqn resolves to a module with no children in the
  live model; when an fqn cannot be resolved (see the naming caveat below) the
  fallback is the observed-prefix rule -- a key is interior iff some other
  observed key extends it with a dot.

  residual := Global - (sum over leaf buckets)
            = (sum over interior buckets) + (unowned)

By construction leaf + interior + unowned == Global exactly; the script asserts
it. Because it is a partition, not a sum of overlapping totals, it is immune to
the double counting above -- but the backward-window leak still misplaces
BACKWARD ops between buckets, so every module-side number is reported twice:
once over all ops and once over forward-only ops, where `ModuleTracker` is
exact. Read the forward-only column; the per-op table needs neither.


THREE RESIDUALS, BECAUSE "RESIDUAL" IS AMBIGUOUS

R1  vs the analytic model. Global minus what `training_flops.py` predicts --
    the 486 GFLOP / 19.8M-per-token-per-layer quantity in its docstring. Split
    by op FAMILY (GEMM / convolution / attention), because the analytic side
    has no per-aten-op resolution and inventing one would be fabrication. The
    frozen `ssm_terms`/`ssm_passes` are reused under rebound geometry rather
    than reimplemented, so this cannot drift from the file under test.

R2  vs real arithmetic. Global minus the physically-correct cost of the same
    dispatch stream. Needs no module attribution and no analytic model, so it
    is immune to both the leak and to any error in `training_flops.py`. Per
    aten op, and this is the table that names the operation.

R3  vs leaf modules. Global minus the leaf buckets of the partition above --
    the definition in the task. Reported, and reported honestly: because the
    backward window leaks, an op can be credited to a stale leaf and vanish
    from R3 even though no module legitimately owns it. R3 is a lower bound on
    the unattributed work; R2 is not.

To make the fqns usable the model's name table is pre-seeded via
`mod_tracker._get_mod_name(model)` before counting. Without it, names are rooted
at the class of whichever module is entered FIRST (module_tracker.py:89-98), and
because diffusion.py calls `prefill_left_boundaries_stacked` (diffusion.py:1174)
and `forward_active` (diffusion.py:1145,1195) as plain METHODS, no BiSSM module
ever fires its own `__call__` -- so every `nn.Linear` in the model would collapse
into one bucket literally named "Linear". Seeding only fills the name cache; it
adds nothing to `parents` and changes no count.


WHAT ELSE IS RECORDED

For each op the argument signature (tensor shapes plus scalar args such as
`groups`) is accumulated, and for a bounded number of records the repo frames of
the Python stack, so a residual op can be traced to a source line. Backward ops
have no user frames -- the stack is the autograd engine -- and are marked as
such.

Convolutions additionally carry a physically-correct recount.
`conv_flop_count` (flop_counter.py:110-145) charges `c_out * c_in` and
`conv_backward_flop` explicitly skips groups ("I skip those for the sake of
brevity", flop_counter.py:238-240). The forward and grad-input branches survive
that because `w_shape[1]` is already `in_channels / groups`; the grad-weight
branch does not -- it charges `in_channels * out_channels` where the grouped
convolution does `out_channels * in_channels / groups`. `conv_corrected` divides
that one branch by `groups` and leaves everything else alone.

No GPU work beyond one forward+backward per (arm, length). Batch 1.

Run:  python scripts/smoke/flop_attribution.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main  # noqa: F401,E402 - registers the OmegaConf resolvers
from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import scripts.eval.training_flops as tf  # noqa: E402
from scripts.eval.measured_flops_sweep import build  # noqa: E402

try:
  from torch.utils.flop_counter import conv_flop_count as _conv_flop_count
except ImportError:  # pragma: no cover - only if torch reorganises the module
  _conv_flop_count = None

GEMM_OPS = {"aten.mm", "aten.addmm", "aten.bmm", "aten.baddbmm",
            "aten._scaled_mm"}
CONV_OPS = {"aten.convolution", "aten._convolution",
            "aten.convolution_backward"}
UNOWNED = "<unowned>"


def shape_of(value):
  if isinstance(value, torch.Tensor):
    return tuple(int(v) for v in value.shape)
  return None


def arg_signature(args, limit=12):
  """Compact, hashable description of a dispatch's arguments."""
  parts = []
  for value in args[:limit]:
    if isinstance(value, torch.Tensor):
      parts.append("x".join(str(v) for v in value.shape) or "scalar")
    elif isinstance(value, bool):
      parts.append("T" if value else "F")
    elif isinstance(value, int):
      parts.append(str(value))
    elif isinstance(value, (list, tuple)):
      inner = []
      for item in value[:6]:
        if isinstance(item, torch.Tensor):
          inner.append("x".join(str(v) for v in item.shape))
        elif isinstance(item, (bool, int)):
          inner.append("T" if item is True else
                       "F" if item is False else str(item))
        else:
          inner.append(type(item).__name__)
      parts.append("[" + ",".join(inner) + "]")
    elif value is None:
      parts.append("None")
    else:
      parts.append(type(value).__name__)
  return " ".join(parts)


def conv_corrected_flops(op_name, args, out):
  """Torch's own conv cost with `groups` honoured in the grad-weight branch.

  Returns None when the recount cannot be made, so the caller can fall back to
  the counted value rather than silently reporting a wrong one.
  """
  if _conv_flop_count is None:
    return None
  try:
    if op_name in ("aten.convolution", "aten._convolution"):
      # forward already divides the input channels by groups via w_shape[1]
      return None
    if op_name != "aten.convolution_backward":
      return None
    grad_out_shape = shape_of(args[0])
    x_shape = shape_of(args[1])
    w_shape = shape_of(args[2])
    transposed = bool(args[7])
    groups = int(args[9])
    output_mask = args[10]
    if grad_out_shape is None or x_shape is None or w_shape is None:
      return None

    def swap(shape):
      return [shape[1], shape[0]] + list(shape[2:])

    total = 0
    if output_mask[0]:
      grad_input_shape = shape_of(out[0])
      if grad_input_shape is None:
        return None
      total += _conv_flop_count(list(grad_out_shape), list(w_shape),
                                list(grad_input_shape), not transposed)
    if output_mask[1]:
      grad_weight_shape = shape_of(out[1])
      if grad_weight_shape is None:
        return None
      if transposed:
        raw = _conv_flop_count(swap(grad_out_shape), swap(x_shape),
                               swap(grad_weight_shape), False)
      else:
        raw = _conv_flop_count(swap(x_shape), swap(grad_out_shape),
                               swap(grad_weight_shape), False)
      total += raw // max(groups, 1)
    return total
  except Exception:  # noqa: BLE001 - a diagnostic must never break the probe
    return None


class attributing_counter(FlopCounterMode):
  """`FlopCounterMode` that also records who owned each dispatched op.

  Overrides `_count_flops`, the single funnel every counted op passes through
  (flop_counter.py:748-756), reads the flop delta straight off the counter's
  own "Global" bucket so the two can never disagree, and snapshots
  `mod_tracker.parents` before it is unwound.
  """

  def __init__(self, max_traces=3, trace_budget=400):
    super().__init__(display=False)
    self.records = defaultdict(int)        # (op, is_bw, owner) -> flops
    self.signatures = defaultdict(int)     # (op, is_bw, signature) -> flops
    self.corrected = defaultdict(int)      # op -> corrected flops
    self.corrected_seen = defaultdict(int)  # op -> counted flops we recounted
    self.traces = defaultdict(list)        # (op, is_bw) -> [frame strings]
    self.max_traces = max_traces
    self.trace_budget = trace_budget
    self.owner_sets = defaultdict(int)     # (op, is_bw, owners) -> flops

  def _count_flops(self, func_packet, out, args, kwargs):
    before = self.flop_counts["Global"].get(func_packet, 0)
    result = super()._count_flops(func_packet, out, args, kwargs)
    delta = self.flop_counts["Global"].get(func_packet, 0) - before
    if delta:
      op_name = str(func_packet)
      is_bw = bool(self.mod_tracker.is_bw)
      owners = tuple(sorted(p for p in self.mod_tracker.parents
                            if p != "Global"))
      owner = max(owners, key=lambda n: (n.count("."), n)) if owners \
          else UNOWNED
      self.records[(op_name, is_bw, owner)] += delta
      self.owner_sets[(op_name, is_bw, owners)] += delta
      self.signatures[(op_name, is_bw, arg_signature(args))] += delta
      if op_name in CONV_OPS:
        corrected = conv_corrected_flops(op_name, args, out)
        if corrected is not None:
          self.corrected[op_name] += corrected
          self.corrected_seen[op_name] += delta
      key = (op_name, is_bw)
      if len(self.traces[key]) < self.max_traces and self.trace_budget > 0:
        self.trace_budget -= 1
        frames = [f"{Path(f.filename).name}:{f.lineno} {f.name}"
                  for f in traceback.extract_stack()
                  if f.filename.startswith(str(REPO))
                  and f.filename != __file__]
        if frames and frames[-6:] not in self.traces[key]:
          self.traces[key].append(frames[-6:])
    return result


def leaf_table(model, owner_keys):
  """Which owner fqns are leaves. Returns (is_leaf, rule) per key.

  Primary rule: resolve the fqn against the live model and ask whether that
  module has children. The seeded names are `type(model).__name__` followed by
  the `named_modules` path, so the mapping is exact when seeding worked.
  Fallback: a key is interior iff another observed key extends it with a dot.
  """
  root = type(model).__name__
  by_path = {}
  for path, module in model.named_modules():
    key = root if path == "" else f"{root}.{path}"
    by_path[key] = module
  is_leaf, rule = {}, {}
  for key in owner_keys:
    if key == UNOWNED:
      is_leaf[key], rule[key] = False, "unowned"
      continue
    module = by_path.get(key)
    if module is not None:
      is_leaf[key] = not any(True for _ in module.children())
      rule[key] = "model"
    else:
      is_leaf[key] = not any(other != key and other.startswith(key + ".")
                             for other in owner_keys)
      rule[key] = "prefix"
  return is_leaf, rule


def run_one(arm, length, block_size, batch, device, max_traces):
  config = build(arm, length, block_size, batch)
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  x0 = torch.randint(8, 12, (batch, length), device=device)
  attention_mask = torch.ones_like(x0)

  counter = attributing_counter(max_traces=max_traces)
  seeded = False
  try:
    counter.mod_tracker._get_mod_name(model)
    seeded = True
  except Exception:  # noqa: BLE001 - fall back to class-rooted names
    seeded = False

  with counter:
    output = model._loss(x0, attention_mask)
    output.loss.backward()

  counts = counter.get_flop_counts()
  global_total = counter.get_total_flops()

  # ---- per aten op (no module attribution involved) -----------------------
  by_op = defaultdict(lambda: {"total": 0, "forward": 0, "backward": 0})
  for (op_name, is_bw, _owner), flops in counter.records.items():
    entry = by_op[op_name]
    entry["total"] += flops
    entry["backward" if is_bw else "forward"] += flops
  recorded_total = sum(e["total"] for e in by_op.values())
  assert recorded_total == global_total, (
    f"instrumented records {recorded_total} != Global {global_total}")

  gemm_total = sum(e["total"] for op, e in by_op.items() if op in GEMM_OPS)
  conv_total = sum(e["total"] for op, e in by_op.items() if op in CONV_OPS)
  other_total = global_total - gemm_total - conv_total

  # ---- disjoint owner partition -------------------------------------------
  owner_keys = {owner for (_op, _bw, owner) in counter.records}
  is_leaf, leaf_rule = leaf_table(model, owner_keys)
  by_owner = defaultdict(lambda: {"total": 0, "forward": 0})
  leaf_sum = leaf_forward = 0
  residual_by_op = defaultdict(int)
  residual_by_op_forward = defaultdict(int)
  residual_by_owner = defaultdict(int)
  unowned_total = unowned_forward = 0
  forward_total = sum(e["forward"] for e in by_op.values())
  for (op_name, is_bw, owner), flops in counter.records.items():
    entry = by_owner[owner]
    entry["total"] += flops
    if not is_bw:
      entry["forward"] += flops
    if is_leaf[owner]:
      leaf_sum += flops
      if not is_bw:
        leaf_forward += flops
    else:
      residual_by_op[op_name] += flops
      residual_by_owner[owner] += flops
      if not is_bw:
        residual_by_op_forward[op_name] += flops
    if owner == UNOWNED:
      unowned_total += flops
      if not is_bw:
        unowned_forward += flops
  residual_total = global_total - leaf_sum
  assert leaf_sum + residual_total == global_total
  assert residual_total == sum(residual_by_op.values()), (
    "residual per-op breakdown does not close")

  # ---- conv recount, with `groups` honoured -------------------------------
  conv_recounted = sum(counter.corrected_seen.values())
  conv_corrected = sum(counter.corrected.values())
  conv_overcount = conv_recounted - conv_corrected
  conv_real = conv_total - conv_overcount

  # ---- normalisation -------------------------------------------------------
  n_layers = int(config.model.n_blocks)
  denominator = float(batch * length * n_layers)

  # ---- R2: counted minus physically-correct arithmetic --------------------
  # Real work per op: GEMMs and attention are counted correctly; convolutions
  # are counted correctly except for the grouped grad-weight branch.
  real_by_op, residual_real_by_op = {}, {}
  for op, entry in by_op.items():
    if op == "aten.convolution_backward" and counter.corrected_seen.get(op):
      real = entry["total"] - (counter.corrected_seen[op]
                               - counter.corrected[op])
    else:
      real = entry["total"]
    real_by_op[op] = real
    if entry["total"] != real:
      residual_real_by_op[op] = entry["total"] - real
  real_total = sum(real_by_op.values())
  assert real_total + sum(residual_real_by_op.values()) == global_total

  # ---- R1: counted minus the module-level analytic prediction -------------
  mismatch = geometry_mismatch(config)
  analytic = None if mismatch else analytic_prediction(
    arm, length, block_size, batch, n_layers)
  residual_analytic = None
  if analytic is not None:
    gap = global_total - analytic["total"]
    residual_analytic = {
      "analytic_total": analytic["total"],
      "analytic_terms": analytic["terms"],
      "total": gap,
      "counted_over_analytic": global_total / analytic["total"],
      "fraction_of_global": gap / global_total,
      "flop_per_token_per_layer": gap / denominator,
      # split by op FAMILY: the analytic side has no per-aten-op resolution,
      # so inventing one would be fabrication. Within a family the counted
      # by_op table names the operation.
      "by_family": {
        "gemm": gemm_total - analytic["gemm"],
        "convolution": conv_total - analytic["conv"],
        "attention": other_total,
      },
      "counted_by_family": {"gemm": gemm_total, "convolution": conv_total,
                            "attention": other_total},
      "analytic_by_family": {"gemm": analytic["gemm"],
                             "convolution": analytic["conv"],
                             "attention": 0},
      "caveats": analytic["caveats"],
    }

  # ---- raw nested bucket totals, for reference only ------------------------
  bucket_totals = {key: sum(ops.values()) for key, ops in counts.items()}
  naive_leaf_sum = sum(
    total for key, total in bucket_totals.items()
    if key != "Global" and is_leaf.get(key, not any(
      other != key and other.startswith(key + ".")
      for other in bucket_totals)))

  record = {
    "arm": arm,
    "length": length,
    "batch": batch,
    "block_size": block_size,
    "n_layers": n_layers,
    "names_seeded": seeded,
    "analytic_geometry_mismatch": mismatch,
    "global_total": global_total,
    "forward_total": forward_total,
    "backward_total": global_total - forward_total,
    "by_op": {op: dict(entry) for op, entry in by_op.items()},
    "gemm_total": gemm_total,
    "conv_total": conv_total,
    "other_total": other_total,
    "conv_recounted_counted": conv_recounted,
    "conv_corrected": conv_corrected,
    "conv_overcount": conv_overcount,
    "conv_real": conv_real,
    "conv_overcount_per_token_per_layer": conv_overcount / denominator,
    "residuals": {
      # R1: the 486 GFLOP / 19.8M-per-token-per-layer gap in the docstring of
      # scripts/eval/training_flops.py.
      "vs_analytic": residual_analytic,
      # R2: dispatched count minus physically-correct arithmetic. Needs no
      # module attribution and is immune to the ModuleTracker backward leak.
      "vs_real_work": {
        "real_total": real_total,
        "total": global_total - real_total,
        "fraction_of_global": (global_total - real_total) / global_total,
        "flop_per_token_per_layer": (global_total - real_total) / denominator,
        "by_op": residual_real_by_op,
        "real_by_op": real_by_op,
      },
      # R3: Global minus the disjoint leaf-owner partition. This is the
      # quantity the task defines, but see the caveat: the tracker's backward
      # window leaks, so backward ops can be credited to a stale leaf and
      # disappear from this residual. Read `forward` here, not `total`.
      "vs_leaf_modules": {
        "leaf_sum": leaf_sum,
        "leaf_sum_forward": leaf_forward,
        "total": residual_total,
        "fraction_of_global": residual_total / global_total,
        "flop_per_token_per_layer": residual_total / denominator,
        "by_op": dict(residual_by_op),
        "by_op_forward": dict(residual_by_op_forward),
        "by_owner": dict(residual_by_owner),
        "unowned_total": unowned_total,
        "unowned_forward": unowned_forward,
        "unowned_fraction_of_global": unowned_total / global_total,
        "unowned_flop_per_token_per_layer": unowned_total / denominator,
        "caveat": ("ModuleTracker re-enters a module during backward on its "
                   "output-grad hook and leaves on its input-grad hook "
                   "(module_tracker.py:127-147); a module whose inputs never "
                   "all receive grads is never popped and owns the remainder "
                   "of the backward pass. Forward-side numbers are exact."),
      },
    },
    "owner_partition": {owner: dict(entry)
                        for owner, entry in by_owner.items()},
    "owner_is_leaf": {owner: bool(is_leaf[owner]) for owner in owner_keys},
    "owner_leaf_rule": {owner: leaf_rule[owner] for owner in owner_keys},
    "nested_bucket_totals": bucket_totals,
    "nested_leaf_sum_over_global": naive_leaf_sum / global_total,
    "signatures": {
      f"{op}|{'bwd' if is_bw else 'fwd'}|{sig}": flops
      for (op, is_bw, sig), flops in counter.signatures.items()},
    "owner_sets": {
      f"{op}|{'bwd' if is_bw else 'fwd'}|{'+'.join(owners) or UNOWNED}": flops
      for (op, is_bw, owners), flops in counter.owner_sets.items()},
    "traces": {f"{op}|{'bwd' if is_bw else 'fwd'}": frames
               for (op, is_bw), frames in counter.traces.items()},
  }

  del model, x0, attention_mask, output, counter
  torch.cuda.empty_cache()
  return record


class tf_geometry:
  """Re-point `training_flops.py`'s module globals at this run's geometry.

  `training_flops.py` is frozen at LENGTH=8192 / BLOCK=256 / 12 layers. Rather
  than reimplement `ssm_terms`/`ssm_passes` here -- which would be free to
  drift from the file whose predictions we are testing -- this rebinds the
  handful of globals they read and restores them afterwards.
  """

  fields = ("LENGTH", "BLOCK", "NUM_BLOCKS", "PREFIX", "N_LAYERS")

  def __init__(self, length, block_size, n_layers):
    num_blocks = max(length // block_size, 1)
    self.values = {
      "LENGTH": length,
      "BLOCK": block_size,
      "NUM_BLOCKS": num_blocks,
      "PREFIX": max((num_blocks - 1) * block_size, 1),
      "N_LAYERS": n_layers,
    }

  def __enter__(self):
    self.saved = {name: getattr(tf, name) for name in self.fields}
    for name, value in self.values.items():
      setattr(tf, name, value)
    return self

  def __exit__(self, *exc):
    for name, value in self.saved.items():
      setattr(tf, name, value)
    return False


def geometry_mismatch(config):
  """Where the live model disagrees with `training_flops.py`'s constants.

  R1 compares a measured count against a file whose geometry is hardcoded at
  the production width. Run a different width and the comparison is
  meaningless, so it is refused rather than reported.
  """
  model = config.model
  d_model = int(model.hidden_size)
  d_inner = int(model.ssm_expand) * d_model
  d_state = int(model.ssm_state_size)
  observed = {
    "d_model": (d_model, tf.D_MODEL),
    "d_inner": (d_inner, tf.D_INNER),
    "d_state": (d_state, tf.D_STATE),
    "headdim": (int(model.ssm_head_dim), tf.HEADDIM),
    "nheads": (d_inner // int(model.ssm_head_dim), tf.NHEADS),
    "chunk": (int(model.ssm_chunk_size), tf.CHUNK),
    "d_conv": (int(model.ssm_conv_size), tf.D_CONV),
    "conv_dim": (d_inner + 2 * d_state, tf.CONV_DIM),
    "mlp_hidden": (int(float(model.mlp_ratio) * d_model), tf.MLP_HIDDEN),
  }
  return {name: {"live": live, "training_flops": frozen}
          for name, (live, frozen) in observed.items() if live != frozen}


def analytic_prediction(arm, length, block_size, batch, n_layers):
  """What a per-module derivation predicts for this arm, and per aten op.

  For `ussm-ar` the per-term dict is exactly the construction in
  scripts/smoke/flop_breakdown.py: the AR path shifts by one token
  (diffusion.py:1016), so it sees L-1. For `bissm` the composite passes come
  straight from `tf.ssm_passes()` under the rebound geometry, which is what
  `tf.arms()` sums for "BiSSM-BD".

  `by_op` maps each term onto the aten op that would carry it, so the residual
  can be split the same way the counter's totals are:
    GEMM family  <- in_proj, out_proj, mlp, head
    conv family  <- the depthwise convolution term
    (the SSD scan dispatches through Triton and is invisible to the counter,
     so it is listed but attributed to no aten op)
  """
  with tf_geometry(length, block_size, n_layers):
    terms = tf.ssm_terms()
    passes = tf.ssm_passes()
    grad = tf.GRAD_MULT
    if arm == "ussm-ar":
      tokens = batch * (length - 1)
      per_term = {
        "in_proj": grad * n_layers * tokens * terms["in_proj"],
        "conv1d": grad * n_layers * tokens * terms["conv"],
        "out_proj": grad * n_layers * tokens * terms["out_proj"],
        "mlp": grad * n_layers * tokens * terms["mlp"],
        "scan": grad * n_layers * tokens * terms["scan"],
        "head": grad * tokens * terms["head"],
      }
      total = sum(per_term.values())
      gemm = per_term["in_proj"] + per_term["out_proj"] + per_term["mlp"] \
          + per_term["head"]
      conv = per_term["conv1d"]
      caveats = []
    elif arm == "bissm":
      total = grad * batch * (passes["clean"] + passes["act_bi"])
      prefix = tf.PREFIX
      tokens_conv = batch * (prefix + 2 * length)   # prefill + both directions
      conv = grad * n_layers * tokens_conv * terms["conv"]
      scan = grad * n_layers * batch * (
        2 * prefix + 2 * length) * terms["scan"]
      per_term = {"composite_clean": grad * batch * passes["clean"],
                  "composite_active_bidirectional":
                      grad * batch * passes["act_bi"],
                  "conv1d_within": conv, "scan_within": scan}
      gemm = total - conv - scan
      caveats = [
        "checkpoint_boundary_prefill=true recomputes the clean prefill during "
        "backward; training_flops.py does not model that extra forward, so "
        "the analytic total is low by about one prefill forward pass",
        "the conv/scan split of the composite total is this script's reading "
        "of tf.ssm_passes(), not a term tf itself exposes"]
    else:
      return None
  return {"total": total, "terms": per_term, "gemm": gemm, "conv": conv,
          "caveats": caveats}


def print_record(record):
  giga = 1e9
  print(f"\n{'=' * 74}")
  print(f"{record['arm']}  L={record['length']}  batch={record['batch']}  "
        f"layers={record['n_layers']}  names_seeded={record['names_seeded']}")
  print("=" * 74)
  print(f"Global total            {record['global_total'] / giga:>12.2f} GFLOP")
  print(f"  forward               {record['forward_total'] / giga:>12.2f}")
  print(f"  backward              {record['backward_total'] / giga:>12.2f}")

  print(f"\n{'aten op':<32}{'GFLOP':>12}{'fwd':>12}{'bwd':>12}{'%':>7}")
  print("-" * 75)
  for op, entry in sorted(record["by_op"].items(),
                          key=lambda kv: -kv[1]["total"]):
    share = 100 * entry["total"] / record["global_total"]
    print(f"{op:<32}{entry['total'] / giga:>12.2f}"
          f"{entry['forward'] / giga:>12.2f}{entry['backward'] / giga:>12.2f}"
          f"{share:>6.1f}%")
  print("-" * 75)
  print(f"{'GEMM family':<32}{record['gemm_total'] / giga:>12.2f}")
  print(f"{'convolution family':<32}{record['conv_total'] / giga:>12.2f}")
  print(f"{'other (sdpa)':<32}{record['other_total'] / giga:>12.2f}")

  if record["conv_recounted_counted"]:
    print(f"\nconvolution recount (groups honoured)")
    print(f"  counter charges       "
          f"{record['conv_recounted_counted'] / giga:>12.2f} GFLOP")
    print(f"  physically correct    "
          f"{record['conv_corrected'] / giga:>12.2f}")
    print(f"  overcount             "
          f"{record['conv_overcount'] / giga:>12.2f}"
          f"   = {record['conv_overcount_per_token_per_layer']:>13,.0f} "
          f"FLOP/token/layer")

  residuals = record["residuals"]

  analytic = residuals["vs_analytic"]
  if analytic:
    print(f"\nR1  residual vs the training_flops.py module-level prediction")
    print(f"  analytic total        "
          f"{analytic['analytic_total'] / giga:>12.2f} GFLOP")
    print(f"  counted / analytic    "
          f"{analytic['counted_over_analytic']:>12.3f}x")
    print(f"  residual              {analytic['total'] / giga:>12.2f}"
          f"   = {100 * analytic['fraction_of_global']:>5.1f}% of Global,"
          f" {analytic['flop_per_token_per_layer']:,.0f} FLOP/token/layer")
    print(f"  {'by op family':<22}{'counted':>12}{'analytic':>12}"
          f"{'residual':>12}")
    for family, gap in sorted(analytic["by_family"].items(),
                              key=lambda kv: -kv[1]):
      print(f"  {family:<22}"
            f"{analytic['counted_by_family'][family] / giga:>12.2f}"
            f"{analytic['analytic_by_family'][family] / giga:>12.2f}"
            f"{gap / giga:>12.2f}")
    for caveat in analytic["caveats"]:
      print(f"  caveat: {caveat}")
  else:
    reason = record["analytic_geometry_mismatch"] or "arm has no analytic model"
    print(f"\nR1  skipped: {reason}")

  real = residuals["vs_real_work"]
  print(f"\nR2  residual vs physically-correct arithmetic "
        f"(no module attribution)")
  print(f"  real work             {real['real_total'] / giga:>12.2f} GFLOP")
  print(f"  residual              {real['total'] / giga:>12.2f}"
        f"   = {100 * real['fraction_of_global']:>5.1f}% of Global,"
        f" {real['flop_per_token_per_layer']:,.0f} FLOP/token/layer")
  print(f"\n  {'residual by aten op':<32}{'GFLOP':>12}")
  print("  " + "-" * 44)
  for op, flops in sorted(real["by_op"].items(), key=lambda kv: -kv[1]):
    print(f"  {op:<32}{flops / giga:>12.2f}")
  if not real["by_op"]:
    print("  (none: every counted op is real arithmetic)")

  leafres = residuals["vs_leaf_modules"]
  print(f"\nR3  residual = Global - (disjoint leaf-owner partition)")
  print(f"  leaf-owned            {leafres['leaf_sum'] / giga:>12.2f} GFLOP")
  print(f"  residual              {leafres['total'] / giga:>12.2f}"
        f"   = {100 * leafres['fraction_of_global']:>5.1f}% of Global,"
        f" {leafres['flop_per_token_per_layer']:,.0f} FLOP/token/layer")
  print(f"  of which unowned      {leafres['unowned_total'] / giga:>12.2f}"
        f"   ({100 * leafres['unowned_fraction_of_global']:.1f}% of Global,"
        f" {leafres['unowned_flop_per_token_per_layer']:,.0f} FLOP/tok/layer)")
  print(f"  NOTE backward attribution leaks; read the fwd-only column.")
  print(f"\n  {'residual by aten op':<32}{'GFLOP':>12}{'fwd only':>12}")
  print("  " + "-" * 56)
  for op, flops in sorted(leafres["by_op"].items(), key=lambda kv: -kv[1]):
    forward = leafres["by_op_forward"].get(op, 0)
    print(f"  {op:<32}{flops / giga:>12.2f}{forward / giga:>12.2f}")

  print(f"\n  {'residual by owning module':<48}{'GFLOP':>12}")
  print("  " + "-" * 60)
  for owner, flops in sorted(leafres["by_owner"].items(),
                             key=lambda kv: -kv[1])[:12]:
    print(f"  {owner[-47:]:<48}{flops / giga:>12.2f}")

  print(f"\n{'owner partition (top 16, disjoint)':<48}"
        f"{'GFLOP':>12}{'fwd':>10}{'leaf':>6}")
  print("-" * 76)
  for owner, entry in sorted(record["owner_partition"].items(),
                             key=lambda kv: -kv[1]["total"])[:16]:
    print(f"{owner[-47:]:<48}{entry['total'] / giga:>12.2f}"
          f"{entry['forward'] / giga:>10.2f}"
          f"{'yes' if record['owner_is_leaf'][owner] else 'no':>6}")

  print(f"\nnested bucket totals (OVERLAPPING, reference only; naive leaf sum "
        f"is {record['nested_leaf_sum_over_global']:.2f}x Global)")
  print(f"{'module bucket':<52}{'GFLOP':>12}")
  print("-" * 64)
  for key, total in sorted(record["nested_bucket_totals"].items(),
                           key=lambda kv: -kv[1])[:14]:
    print(f"{key[-51:]:<52}{total / giga:>12.2f}")

  print(f"\ntop argument signatures")
  print(f"{'op|dir|args':<62}{'GFLOP':>12}")
  print("-" * 74)
  for key, flops in sorted(record["signatures"].items(),
                           key=lambda kv: -kv[1])[:12]:
    print(f"{key[:61]:<62}{flops / giga:>12.2f}")

  residual_ops = set(residuals["vs_real_work"]["by_op"]) \
      | set(residuals["vs_leaf_modules"]["by_op"])
  shown = 0
  for key, frames in record["traces"].items():
    if key.split("|")[0] not in residual_ops or shown >= 4:
      continue
    shown += 1
    print(f"\ncall site  {key}")
    if not frames:
      print("    (no repo frame: dispatched from inside the autograd engine)")
    for frame_list in frames[:2]:
      print("    " + " <- ".join(reversed(frame_list)))


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", default="bissm,ussm-ar")
  parser.add_argument("--lengths", default="2048,8192,16384")
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--max-traces", type=int, default=3)
  parser.add_argument("--output", type=Path,
                      default=REPO / "results" / "sizing"
                      / "flop_attribution.json")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")

  rows = []
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
      try:
        record = run_one(arm, length, args.block_size, args.batch, device,
                         args.max_traces)
        rows.append(record)
        print_record(record)
      except Exception as exc:  # noqa: BLE001
        rows.append({"arm": arm, "length": length,
                     "error": f"{type(exc).__name__}: {exc}"})
        print(f"\n{arm} L={length} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
        torch.cuda.empty_cache()

  print(f"\n{'=' * 92}")
  print("is the residual linear in length? (constant FLOP/token/layer => yes)")
  print("=" * 92)
  print(f"{'arm':<10}{'L':>7}{'global GF':>12}{'R1 GF':>10}{'R1/tok/lyr':>13}"
        f"{'R2 GF':>10}{'R2/tok/lyr':>13}{'R3 GF':>10}{'R3/tok/lyr':>13}")
  print("-" * 92)
  for row in rows:
    if "error" in row:
      continue
    residuals = row["residuals"]
    r1 = residuals["vs_analytic"]
    r2 = residuals["vs_real_work"]
    r3 = residuals["vs_leaf_modules"]
    r1_total = r1["total"] / 1e9 if r1 else float("nan")
    r1_norm = r1["flop_per_token_per_layer"] if r1 else float("nan")
    print(f"{row['arm']:<10}{row['length']:>7}"
          f"{row['global_total'] / 1e9:>12.2f}"
          f"{r1_total:>10.2f}{r1_norm:>13,.0f}"
          f"{r2['total'] / 1e9:>10.2f}{r2['flop_per_token_per_layer']:>13,.0f}"
          f"{r3['total'] / 1e9:>10.2f}"
          f"{r3['flop_per_token_per_layer']:>13,.0f}")
  print("R1 = vs training_flops.py analytic   R2 = vs real arithmetic   "
        "R3 = vs leaf-owner partition")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", dir=args.output.parent,
                                   delete=False) as handle:
    json.dump({
      "batch": args.batch,
      "block_size": args.block_size,
      "torch": torch.__version__,
      "note": ("residuals.vs_analytic (R1) is Global minus the "
               "training_flops.py module-level prediction; "
               "residuals.vs_real_work (R2) is Global minus the physically "
               "correct cost of the same dispatch stream; "
               "residuals.vs_leaf_modules (R3) is Global minus the leaf "
               "buckets of the disjoint deepest-owner partition and is a "
               "LOWER bound, because ModuleTracker's backward window leaks. "
               "nested_bucket_totals are FlopCounterMode's own OVERLAPPING "
               "buckets, kept for reference only -- do not sum them. by_op "
               "needs no module attribution and is the primary result."),
      "rows": rows}, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, args.output)
  print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
