#!/usr/bin/env python3
"""FLOPs actually dispatched, per arm, across sequence lengths.

`training_flops.py` derives FLOPs from hand-written formulas. An earlier
revision of this docstring claimed the counter showed those formulas WRONG for
every SSM arm and that "the FLOPs panel cannot be built from arithmetic".
BOTH CLAIMS ARE RETRACTED. The 1.35-1.37x excess was torch's flop_counter
overcounting depthwise convolutions by `groups` (= 1664), not the formulas
undercounting; see training_flops.py's docstring for the closed accounting.
scaling_curves.py now builds every curve from arithmetic, correctly.

This measures instead. One forward+backward per point under
`torch.utils.flop_counter.FlopCounterMode`, no timing loop, so it is cheap.

Two things the counter cannot do, both handled explicitly rather than silently:

* It is blind to flash/flex attention and to mamba-ssm's Triton SSD kernel,
  which dispatch outside aten. For the Transformer the missing piece is exactly
  the attention term and can be added back analytically; that reconstruction is
  reported separately from the raw count so the two never get confused.
* Transformer-BD's flex path fails under the counter outright
  ("Attempted to call function marked as skipped" -- flex is compiled). This
  falls back to `attn_backend=sdpa` for that arm, which computes the same
  attention with an aten op the counter can see. Different kernel, same
  arithmetic.

The output file ACCUMULATES across runs. One sweep cannot cover every point:
the long lengths only fit at micro batch 1, and one length per job is the only
way to avoid allocating a second model against a fragmented pool. So a run
merges its rows into whatever is already on disk, replacing by (arm, length)
and leaving every other row alone. Because of that a single file can mix
batches, each row records the `batch` and `block_size` it was measured at, and
the top-level copies describe only the most recent run. Consumers must divide
by the ROW's batch. `--no-merge` restores the old clobber-everything write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import hydra
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main  # noqa: F401,E402 - registers the OmegaConf resolvers
from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import scripts.eval.training_flops as tf  # noqa: E402

ARMS = {
  "bissm":   ("small_bissm", "bd3lm_bissm", False),
  "ussm-ar": ("small_ussm", "ar", False),
  "dit":     ("small", "bd3lm", True),
  "dit-ar":  ("small_ar_transformer", "ar", False),
}

NOTE = (
  "counted_tflop is what FlopCounterMode dispatched (blind to flash/flex "
  "attention and the Triton SSD scan). analytic_attention_tflop adds back the "
  "attention term for the Transformer arms only; it is zero for the SSM arms, "
  "whose scan remains uncounted, so their total is a LOWER bound. Rows "
  "accumulate across runs, keyed by (arm, length), so the file can mix sweeps "
  "taken at different micro batches: divide counted_tflop by the ROW's batch, "
  "not by the top-level one, which describes only the most recent run."
)


def row_key(row):
  """Identity of a measurement: one point of one curve."""
  return (row.get("arm"), row.get("length"))


def merge_rows(existing, fresh):
  """`fresh` overrides `existing` per (arm, length); everything else survives.

  A replaced row keeps its original position, so re-measuring one point does
  not reshuffle the file; genuinely new points append in the order measured.
  """
  merged, order = {}, []
  for row in list(existing) + list(fresh):
    key = row_key(row)
    if key not in merged:
      order.append(key)
    merged[key] = row
  return [merged[key] for key in order]


def write_payload(output, rows, batch, block_size, merge=True):
  """Atomically write `rows`, folded into any payload already at `output`."""
  output.parent.mkdir(parents=True, exist_ok=True)
  if merge and output.exists():
    try:
      previous = json.loads(output.read_text()).get("rows", [])
    except (OSError, ValueError) as exc:
      # Never clobber a file we failed to read, and never throw away GPU hours:
      # park the fresh rows next to it and let a human reconcile them.
      rescue = output.with_name(output.stem + ".unmerged.json")
      rescue.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
      raise RuntimeError(f"cannot parse {output} to merge into ({exc}); "
                         f"fresh rows written to {rescue}") from exc
    rows = merge_rows(previous, rows)
  batches = sorted({r["batch"] for r in rows if isinstance(r.get("batch"), int)})
  payload = {"batch": batch, "batches": batches, "block_size": block_size,
             "note": NOTE, "rows": rows}
  with tempfile.NamedTemporaryFile("w", dir=output.parent,
                                   delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, output)
  return payload


def build(arm, length, block_size, batch, checkpoint_prefill=True):
  """Compose one arm's config.

  `checkpoint_prefill` applies to bissm only. It defaults True, which is what
  the memory/throughput sweeps use and what every already-published measured
  number was taken under -- do not change the default without re-running them.
  Set it False to match what training_flops.py models: recomputing the boundary
  prefill dispatches its forward an extra time, so a checkpointed run counts
  ~13% more FLOPs than the analytic file predicts, and comparing the two under
  checkpointing measures the recompute rather than the discrepancy of interest.
  """
  model_cfg, algo_cfg, needs_sdpa = ARMS[arm]
  is_ar = algo_cfg == "ar"
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}", f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch}", f"loader.eval_batch_size={batch}",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
  ]
  if arm == "ussm-ar":
    overrides.append("algo.backbone=ussm")
  if arm == "bissm":
    overrides.append("model.active_blocks=all")
    overrides.append("model.checkpoint_boundary_prefill="
                     f"{str(bool(checkpoint_prefill)).lower()}")
  if needs_sdpa:
    # flex is compiled and the counter cannot enter it; sdpa is the same
    # attention through an aten op it can see.
    overrides.append("model.attn_backend=sdpa")
  with hydra.initialize_config_dir(version_base=None,
                                   config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=overrides)


def analytic_attention_tflop(arm, length, batch, n_layers=12):
  """The term the counter is structurally blind to, for the Transformer arms."""
  if arm == "dit-ar":
    tokens = length - 1
    pairs = tokens * (tokens + 1) // 2
    return tf.GRAD_MULT * 4 * n_layers * pairs * tf.D_AR * batch / 1e12
  if arm == "dit":
    # ZERO, deliberately. `dit` is forced onto attn_backend=sdpa here because
    # flex is compiled and the counter cannot enter it. But sdpa dispatches
    # aten ops the counter CAN see, so dit's attention is ALREADY inside
    # counted_tflop -- adding it back double-counts. Earlier revisions did add
    # it, inflating every dit total_tflop by up to 3.96x at L=16384.
    #
    # Note the counted value is also not comparable to the analytic formula:
    # sdpa evaluates the full (2L)^2 attention, while flex (and the formula)
    # evaluate only the mask-permitted block^2*nb*(nb+1) pairs, ~3.6-3.9x fewer.
    # So dit's counted number measures a DIFFERENT algorithm and validates only
    # the dense/param term.
    return 0.0
  return 0.0


def run(arm, length, block, batch, device):
  config = build(arm, length, block, batch)
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer()).to(device)
  model.train()
  x0 = torch.randint(8, 12, (batch, length), device=device)
  mask = torch.ones_like(x0)
  counter = FlopCounterMode(display=False)
  with counter:
    loss = model._loss(x0, mask)
    loss.loss.backward()
  counted = counter.get_total_flops() / 1e12
  del model, x0, mask, loss
  torch.cuda.empty_cache()
  return counted


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", default="bissm,dit,ussm-ar,dit-ar")
  parser.add_argument("--lengths", default="2048,4096,8192,16384,32768")
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--output", type=Path,
                      default=REPO / "results" / "sizing" / "measured_flops.json")
  parser.add_argument("--merge", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="fold these rows into an existing output file, "
                           "replacing by (arm, length); --no-merge overwrites "
                           "the file wholesale, as this script used to")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")
  rows = []
  print(f"{'arm':<9}{'L':>8}{'counted TF':>12}{'+attn TF':>10}{'total TF':>10}")
  print("-" * 49)
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
      try:
        counted = run(arm, length, args.block_size, args.batch, device)
        attn = analytic_attention_tflop(arm, length, args.batch)
        rows.append({"arm": arm, "length": length, "batch": args.batch,
                     "block_size": args.block_size,
                     "counted_tflop": counted,
                     "analytic_attention_tflop": attn,
                     "total_tflop": counted + attn})
        print(f"{arm:<9}{length:>8}{counted:>12.2f}{attn:>10.2f}"
              f"{counted + attn:>10.2f}")
      except Exception as exc:  # noqa: BLE001
        rows.append({"arm": arm, "length": length, "batch": args.batch,
                     "block_size": args.block_size,
                     "error": f"{type(exc).__name__}: {exc}"})
        print(f"{arm:<9}{length:>8}   FAILED {type(exc).__name__}: "
              f"{str(exc)[:44]}")
        torch.cuda.empty_cache()

  payload = write_payload(args.output, rows, args.batch, args.block_size,
                          merge=args.merge)
  verb = "merged into" if args.merge else "overwrote"
  print(f"\n{verb} {args.output} "
        f"({len(rows)} new, {len(payload['rows'])} total, "
        f"batches {payload['batches']})")


if __name__ == "__main__":
  main_cli()
