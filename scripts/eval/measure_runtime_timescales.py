#!/usr/bin/env python3
"""Per-head state half-life at RUNTIME, from the input-dependent dt.

WHY THIS EXISTS. `measure_timescales.py` computes tau from `dt_bias` alone:

    tau = ln2 / (A * softplus(dt_bias)),   A = exp(A_log)

That is a parameter-level estimate, and for the frozen-dt arm it is the
quantity the freeze actually pins. But dt is INPUT-DEPENDENT at runtime --
`in_proj` emits a per-position, per-head dt that is added to the bias before
the softplus:

    dt_runtime = softplus(dt_proj(u) + dt_bias)

`freeze_dt` sets `dt_bias.requires_grad_(False)`; it does NOT freeze the dt
slice of `in_proj`. So the frozen arm can in principle learn to supply large dt
through the projection and undo its own schedule, leaving a long *parameter*
timescale and a short *effective* one.

THAT IS A LIVE CONFOUND, not a hypothetical. At step 41,000 the two arms sat at
val/nll 1.0779 (learned) and 1.0773 (frozen) -- statistically the same -- while
their bias-derived tau medians differed 738x (1.8 nt vs 1,328 nt). Two readings
fit those numbers equally well:

  (a) Long timescales genuinely do not help on this corpus, consistent with the
      full-attention oracle finding all context value within +-256 nt.
  (b) The freeze never took effect at runtime: dt_proj compensated, both arms
      really ran at short timescales, and the likelihoods match because the
      models are effectively identical.

Reading (a) is a result. Reading (b) is a bug. They are distinguished only by
measuring dt under real data, which is what this script does.

HOW. One forward hook per mixer on `in_proj`, whose output is split
`[z (d_inner), xBC (conv_dim), dt (nheads)]` in every one of the three forward
paths -- so the last `nheads` channels are the raw dt regardless of which path
runs. Add the bias, softplus, and average over batch and position to get the
operating-point dt per head. Then compare tau computed both ways.

Usage:
  python scripts/eval/measure_runtime_timescales.py \
      --checkpoint outputs/hg38-caduceus/tau_frozen/checkpoints/last.ckpt \
      --checkpoint outputs/hg38-caduceus/tau_learned/checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402


def mixers(model):
  """Every SegmentMamba2 in the backbone, in layer order.

  Identified structurally (it owns dt_bias, A_log and in_proj) rather than by
  class name, so a subclass or a rename does not silently yield zero mixers.
  """
  found = []
  for name, module in model.named_modules():
    if (hasattr(module, "dt_bias") and hasattr(module, "A_log")
        and hasattr(module, "in_proj")):
      found.append((name, module))
  return found


def probe(checkpoint, args, device):
  model, tokenizer, config, step = load_checkpoint_model(
    checkpoint, args.sequence_length, args.batch_size, device)
  layers = mixers(model)
  if not layers:
    raise ValueError(f"{checkpoint}: no SSM mixers (not an SSM checkpoint)")

  # running sum/count of dt per (layer, head), accumulated across every
  # in_proj call the forward makes
  totals = [torch.zeros(m.nheads, dtype=torch.float64, device=device)
            for _, m in layers]
  counts = [0 for _ in layers]
  handles = []

  def make_hook(index, mixer):
    def hook(_module, _inputs, output):
      # in_proj -> [z | xBC | dt]; dt is the last nheads channels in all paths.
      raw = output[..., -mixer.nheads:]
      dt = F.softplus(raw.float() + mixer.dt_bias.float())
      flat = dt.reshape(-1, mixer.nheads)
      totals[index] += flat.sum(dim=0).double()
      counts[index] += flat.shape[0]
    return hook

  for index, (_, mixer) in enumerate(layers):
    handles.append(mixer.in_proj.register_forward_hook(make_hook(index, mixer)))

  config.loader.batch_size = args.batch_size
  config.loader.eval_batch_size = args.batch_size
  # get_dataloaders validates the (unused) training global batch even under
  # skip_train=True, and the checkpoint stores it as an unresolved
  # device_count interpolation.
  config.trainer.accumulate_grad_batches = 1
  world = torch.cuda.device_count() * int(config.trainer.num_nodes)
  config.loader.global_batch_size = args.batch_size * world
  config.loader.eval_global_batch_size = args.batch_size * world
  config.loader.num_workers = min(int(config.loader.num_workers), 8)
  _, valid_loader = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=args.seed)

  seen = 0
  with torch.inference_mode():
    for batch in valid_loader:
      if seen >= args.num_batches:
        break
      x0 = batch["input_ids"].to(device)
      mask = batch["attention_mask"].to(device)
      model._loss(x0, mask)
      seen += 1
  for handle in handles:
    handle.remove()
  if seen == 0 or not any(counts):
    raise RuntimeError("no batches consumed / hooks never fired")

  rows = []
  for index, (name, mixer) in enumerate(layers):
    A = torch.exp(mixer.A_log.float()).detach().cpu().numpy()
    dt_bias = F.softplus(mixer.dt_bias.float()).detach().cpu().numpy()
    dt_run = (totals[index] / max(counts[index], 1)).cpu().numpy()
    rows.append({
      "layer": index,
      "module": name,
      "tau_bias": (math.log(2) / (A * dt_bias)).tolist(),
      "tau_runtime": (math.log(2) / (A * dt_run)).tolist(),
      "dt_bias": dt_bias.tolist(),
      "dt_runtime": dt_run.tolist(),
    })
  return {"checkpoint": str(checkpoint), "global_step": step,
          "batches": seen, "layers": rows}


def report(result):
  tb = np.array([t for r in result["layers"] for t in r["tau_bias"]])
  tr = np.array([t for r in result["layers"] for t in r["tau_runtime"]])
  db = np.array([d for r in result["layers"] for d in r["dt_bias"]])
  dr = np.array([d for r in result["layers"] for d in r["dt_runtime"]])
  name = Path(result["checkpoint"]).parts[-3]
  step = result["global_step"]
  print(f"\n=== {name}  step {step:,}  ({result['batches']} val batches, "
        f"{tb.size} heads)")
  print(f"  dt  from bias only : median {np.median(db):>12.6f}")
  print(f"  dt  at runtime     : median {np.median(dr):>12.6f}"
        f"   ({np.median(dr) / max(np.median(db), 1e-12):>8.1f}x the bias)")
  print(f"  tau from bias only : median {np.median(tb):>12,.1f} nt   "
        f"max {tb.max():>12,.1f}   >16nt {int((tb > 16).sum())}/{tb.size}")
  print(f"  tau at RUNTIME     : median {np.median(tr):>12,.1f} nt   "
        f"max {tr.max():>12,.1f}   >16nt {int((tr > 16).sum())}/{tr.size}")
  print(f"    {'layer':>6}{'tau_bias':>14}{'tau_runtime':>14}")
  for r in result["layers"]:
    print(f"    {r['layer']:>6}{np.median(r['tau_bias']):>14,.1f}"
          f"{np.median(r['tau_runtime']):>14,.1f}")
  return np.median(tr)


def main():
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", action="append", required=True,
                      type=Path, help="repeatable")
  parser.add_argument("--sequence-length", type=int, default=8192)
  parser.add_argument("--batch-size", type=int, default=2)
  parser.add_argument("--num-batches", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", type=Path,
                      default=REPO / "results" / "runtime_timescales.json")
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")

  results, medians = [], {}
  for checkpoint in args.checkpoint:
    result = probe(checkpoint, args, device)
    results.append(result)
    medians[Path(checkpoint).parts[-3]] = report(result)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=2))
  print(f"\nwrote {args.out}")

  # The verdict below compares a FROZEN arm against its LEARNED control, where
  # a small spread means the freeze failed. Across unrelated arms a small
  # spread means nothing of the kind, so only print it for that pair -- an
  # automatic verdict on the wrong comparison is worse than none.
  pair = len(medians) == 2 and all(n.startswith("tau_") for n in medians)
  if len(medians) > 1:
    print("\n--- runtime tau by arm ---")
    for name, median in medians.items():
      print(f"  {name:<16} runtime tau median {median:>12,.1f} nt")
  if pair:
    lo, hi = min(medians.values()), max(medians.values())
    ratio = hi / max(lo, 1e-12)
    print(f"\n  runtime spread, frozen vs learned: {ratio:,.1f}x")
    if ratio < 2:
      print("  -> the arms run at the SAME effective timescale. The frozen "
            "schedule\n     did NOT survive into runtime: dt_proj compensated. "
            "The matched\n     likelihoods say nothing about long-range value.")
    else:
      print("  -> the arms genuinely run at different timescales, so the "
            "matched\n     likelihoods ARE evidence that long timescales do "
            "not help here.")


if __name__ == "__main__":
  main()
