#!/usr/bin/env python3
"""One table: memory, throughput, FLOPs, NLL and perplexity, per arm.

Every number in this project has so far come from a different harness -- peak
memory from `scripts/smoke/sizing_sweep.py`, FLOPs from
`scripts/eval/training_flops.py`, NLL from training logs, throughput from a
third place. That made several comparisons quietly unfair: the wall clocks were
not validation-matched, the FLOP telemetry was architecture-blind, and the
memory numbers came from block-diffusion arms only. This script produces all of
them together, from one definition, so a reviewer can read one table.

What each column is, and what it is not:

  peak_gib      torch.cuda.max_memory_allocated across a real
                forward + backward + AdamW step at the stated geometry, after
                warmup steps are discarded. Includes optimizer state. NOT a
                forward-only microbenchmark.
  tokens_per_s  median over post-warmup steps of the same loop, so it is
                comparable to peak_gib by construction.
  train_pflop   from scripts/eval/training_flops.py, derived from the real
                forward paths rather than the wandb telemetry, which is
                architecture-blind and wrong by -44% to +67% depending on arm.
  val_nll       recomputed HERE on a fixed held-out slice, not scraped from
                training logs -- logs differ in validation cadence between
                arms, which is exactly the confound that made earlier wall
                clocks incomparable.
  val_ppl       exp(val_nll). For the AR arms this is a true perplexity. For
                the block-diffusion arms val_nll is a NELBO, an UPPER bound, so
                the perplexity is an upper bound too and is not comparable
                across objectives. The column reports it because reviewers ask
                for it; the caveat travels with it in the JSON.

Arms are declared by checkpoint, so the table always describes models that
actually exist rather than configurations that could be built.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
from diffusion import Diffusion  # noqa: E402

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)


def load(checkpoint, batch_size, device):
  raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
  config = OmegaConf.create(raw["hyper_parameters"]["config"])
  config.loader.batch_size = batch_size
  config.loader.eval_batch_size = batch_size
  config.trainer.accumulate_grad_batches = 1
  if OmegaConf.select(config, "training.ema") not in (None, 0):
    config.training.ema = 0
  tokenizer = dataloader.DNATokenizer()
  model = Diffusion(config, tokenizer)
  state = {k[len("backbone."):]: v for k, v in raw["state_dict"].items()
           if k.startswith("backbone.")}
  model.backbone.load_state_dict(state, strict=False)
  step = raw.get("global_step")
  del raw
  return model.to(device), tokenizer, config, step


def measure_step(model, length, batch_size, device, warmup=2, iters=5):
  """Peak memory and median step time for a real training step."""
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  x0 = torch.randint(8, 12, (batch_size, length), device=device)
  attention_mask = torch.ones_like(x0)
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(device)
  times = []
  for step in range(warmup + iters):
    if step == warmup:
      torch.cuda.synchronize(device)
      torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model._loss(x0, attention_mask)
    loss.loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    if step >= warmup:
      times.append(time.perf_counter() - started)
  peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
  median = statistics.median(times)
  del optimizer, x0, attention_mask, loss
  return peak, median, batch_size * length / median


def measure_nll(model, config, tokenizer, batches, seed, device):
  """Val NLL on a fixed held-out slice, recomputed rather than scraped."""
  _, valid_loader = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  model.eval()
  total, tokens = 0.0, 0
  with torch.inference_mode():
    for index, batch in enumerate(valid_loader):
      if index >= batches:
        break
      x0 = batch["input_ids"].to(device)
      mask = batch.get("attention_mask")
      mask = torch.ones_like(x0) if mask is None else mask.to(device)
      out = model._loss(x0, mask)
      # `_loss` returns a summed NLL and its token count; use them rather than
      # the mean, so short final batches do not get equal weight.
      total += float(out.nlls.sum())
      tokens += int(out.token_mask.sum()) if hasattr(out, "token_mask") \
          else int(out.nlls.numel())
  return total / max(tokens, 1), tokens


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", type=Path, required=True,
                      help="JSON: [{label, checkpoint, batch_size?}, ...]")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--val-batches", type=int, default=32)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--skip-nll", action="store_true")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("benchmark_arms needs a CUDA GPU")
  device = torch.device("cuda")

  # FLOPs come from the dedicated script so there is exactly one definition.
  flops = {}
  flops_path = REPO / "results" / "training_flops.json"
  if flops_path.exists():
    flops = {r["arm"]: r["pflop"]
             for r in json.loads(flops_path.read_text())["arms"]}

  arms = json.loads(args.arms.read_text())
  print(f"device {torch.cuda.get_device_name(device)}  "
        f"{torch.cuda.get_device_properties(device).total_memory/1024**3:.1f} GiB\n")
  header = (f"{'arm':<16}{'L':>7}{'bs':>4}{'peak GiB':>10}{'tok/s':>10}"
            f"{'PFLOP':>9}{'val NLL':>10}{'val PPL':>9}")
  print(header)
  print("-" * len(header))

  rows = []
  for spec in arms:
    label = spec["label"]
    batch = int(spec.get("batch_size", args.batch_size))
    row = {"label": label, "checkpoint": spec["checkpoint"], "batch_size": batch}
    try:
      model, tokenizer, config, step = load(
        Path(spec["checkpoint"]), batch, device)
      length = int(config.model.length)
      peak, seconds, tps = measure_step(
        model, length, batch, device, args.warmup, args.iters)
      row.update({
        "length": length, "backbone": str(config.algo.backbone),
        "objective": str(config.algo.name), "checkpoint_global_step": step,
        "peak_gib": peak, "step_seconds": seconds, "tokens_per_s": tps,
        "train_pflop": flops.get(spec.get("flops_key", label)),
        "pretraining_data": str(OmegaConf.select(config, "data.train")),
      })
      if not args.skip_nll:
        nll, counted = measure_nll(
          model, config, tokenizer, args.val_batches, args.seed, device)
        row.update({
          "val_nll": nll, "val_ppl": math.exp(nll), "val_tokens": counted,
          "nll_is_upper_bound": str(config.algo.name) != "ar",
        })
      print(f"{label:<16}{length:>7}{batch:>4}{peak:>10.2f}{tps:>10.0f}"
            f"{(row['train_pflop'] or float('nan')):>9.0f}"
            f"{row.get('val_nll', float('nan')):>10.5f}"
            f"{row.get('val_ppl', float('nan')):>9.4f}")
    except torch.cuda.OutOfMemoryError:
      row["error"] = "OOM"
      print(f"{label:<16}{'':>7}{batch:>4}{'OOM':>10}")
    except Exception as exc:  # noqa: BLE001
      row["error"] = f"{type(exc).__name__}: {exc}"
      print(f"{label:<16}  FAILED {type(exc).__name__}: {str(exc)[:60]}")
    finally:
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats(device)
    rows.append(row)

  summary = {
    "device": torch.cuda.get_device_name(device),
    "protocol": {
      "peak_gib": "max_memory_allocated over a real fwd+bwd+AdamW step, "
                  "post-warmup, includes optimizer state",
      "tokens_per_s": "median post-warmup step of the same loop",
      "train_pflop": "scripts/eval/training_flops.py, from the real forward "
                     "paths; the wandb telemetry is architecture-blind",
      "val_nll": "recomputed here on a fixed held-out slice, not scraped from "
                 "training logs whose validation cadence differs by arm",
      "val_ppl": "exp(val_nll); an UPPER bound for block-diffusion arms, whose "
                 "val_nll is a NELBO. Not comparable across objectives.",
    },
    "arms": rows,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile("w", dir=args.output.parent,
                                   delete=False) as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, args.output)
  print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main()
