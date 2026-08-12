#!/usr/bin/env python3
"""Measure whether recurrent DNA SSMs use context beyond a fixed local window.

The target is the final diffusion block.  All conditions keep its immediately
preceding local tokens identical.  Only the recurrent state entering that
local window changes: true distal prefix, an empty state, or a state produced
by another sequence's distal prefix.  Positive NLL deltas relative to ``true``
are direct evidence that the model uses information beyond the local radius.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402


CONDITIONS = ("true", "zero_distal", "shuffled_distal")


def _condition_cache(backbone, sequence, donor, boundary, radius, condition):
  local_start = max(0, boundary - radius)
  if condition == "true":
    return backbone.prefill_left(sequence[:, :boundary], detach=True)
  cache = None
  if condition == "shuffled_distal" and local_start:
    cache = backbone.prefill_left(donor[:, :local_start], detach=True)
  if local_start < boundary:
    cache = backbone.prefill_left(
      sequence[:, local_start:boundary], cache=cache, detach=True)
  return cache


def _ar_target_losses(model, sequence, donor, target_start, radius, condition):
  # A next-token logit for x[target_start] is emitted while consuming the
  # preceding token, so the recurrent cache ends one token before the target.
  boundary = target_start - 1
  cache = _condition_cache(
    model.backbone, sequence, donor, boundary, radius, condition)
  inputs = sequence[:, boundary:-1]
  targets = sequence[:, target_start:]
  logits, _ = model.backbone._scan_active(inputs, None, cache)
  logits[..., model.mask_index] = model.neg_infinity
  log_probs = logits.log_softmax(dim=-1)
  return -torch.gather(
    log_probs, dim=-1, index=targets[..., None]).squeeze(-1)


def _diffusion_target_losses(
    model, sequence, target_start, cache, t, uniform):
  clean_target = sequence[:, target_start:]
  loss_scale, p = model.noise(t)
  sigma = model._sigma_from_p(p[:, :1]).squeeze(-1)
  noisy_target = torch.where(
    uniform <= p[:, :1], model.mask_index, clean_target)
  logits = model.backbone.forward_active(
    noisy_target, sigma, left_cache=cache, right_cache=None)
  log_scores = model._subs_parameterization(logits, noisy_target)
  log_p = torch.gather(
    log_scores, dim=-1, index=clean_target[..., None]).squeeze(-1)
  return loss_scale[:, :1] * log_p


def _atomic_json(path, value):
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def _atomic_csv(path, rows):
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", newline="", dir=path.parent,
      delete=False) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = Path(handle.name)
  os.replace(temporary, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--sequence-length", type=int, default=8192)
  parser.add_argument("--target-length", type=int, default=256)
  parser.add_argument("--radii", type=int, nargs="+", default=[256, 1024, 4096])
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--num-sequences", type=int, default=64)
  parser.add_argument("--mc-samples", type=int, default=16)
  parser.add_argument("--epsilon", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("Prefix intervention requires a CUDA GPU")
  if args.target_length >= args.sequence_length:
    parser.error("target-length must be smaller than sequence-length")
  if min(args.radii) < 0 or max(args.radii) >= args.sequence_length:
    parser.error("radii must lie in [0, sequence-length)")

  device = torch.device("cuda")
  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, args.sequence_length, args.batch_size, device)
  if str(config.algo.backbone) not in {"ussm", "bissm"}:
    raise ValueError("This intervention is defined for recurrent SSM checkpoints")
  if str(model.parameterization) == "ar" and str(config.algo.backbone) != "ussm":
    raise ValueError("AR recurrent scoring currently requires backbone=ussm")

  config.loader.batch_size = args.batch_size
  config.loader.eval_batch_size = args.batch_size
  # get_dataloaders validates the (unused) training global batch even when
  # skip_train=True. The raw checkpoint stores this field as an unresolved
  # device_count interpolation, so replace it with the evaluation-safe value.
  config.trainer.accumulate_grad_batches = 1
  world_size = torch.cuda.device_count() * int(config.trainer.num_nodes)
  config.loader.global_batch_size = args.batch_size * world_size
  config.loader.eval_global_batch_size = args.batch_size * world_size
  config.loader.num_workers = min(int(config.loader.num_workers), 8)
  _, valid_loader = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=args.seed)

  target_start = args.sequence_length - args.target_length
  generator = torch.Generator(device="cpu").manual_seed(args.seed)
  values = {(radius, condition): []
            for radius in args.radii for condition in CONDITIONS}
  seen = 0
  with torch.inference_mode():
    for batch in valid_loader:
      if seen >= args.num_sequences:
        break
      sequence = batch["input_ids"][:args.num_sequences - seen].to(device)
      attention = batch["attention_mask"][:sequence.shape[0]].to(device).bool()
      if sequence.shape[0] < 2:
        continue
      donor = sequence.roll(1, dims=0)
      target_mask = attention[:, target_start:]

      if str(model.parameterization) == "ar":
        for radius in args.radii:
          for condition in CONDITIONS:
            losses = _ar_target_losses(
              model, sequence, donor, target_start, radius, condition)
            per_sequence = (
              (losses.double() * target_mask).sum(-1)
              / target_mask.sum(-1).clamp_min(1))
            values[(radius, condition)].extend(per_sequence.cpu().tolist())
      else:
        caches = {
          (radius, condition): _condition_cache(
            model.backbone, sequence, donor, target_start, radius, condition)
          for radius in args.radii for condition in CONDITIONS
        }
        totals = {(radius, condition): torch.zeros(
          sequence.shape[0], dtype=torch.float64, device=device)
          for radius in args.radii for condition in CONDITIONS}
        for _ in range(args.mc_samples):
          t = torch.rand((sequence.shape[0], 1), generator=generator)
          t = (t * (1.0 - args.epsilon) + args.epsilon).to(device)
          uniform = torch.rand(
            (sequence.shape[0], args.target_length), generator=generator).to(device)
          for radius in args.radii:
            for condition in CONDITIONS:
              losses = _diffusion_target_losses(
                model, sequence, target_start, caches[(radius, condition)],
                t, uniform)
              totals[(radius, condition)] += (
                (losses.double() * target_mask).sum(-1)
                / target_mask.sum(-1).clamp_min(1))
        for key, total in totals.items():
          values[key].extend((total / args.mc_samples).cpu().tolist())
      seen += sequence.shape[0]
      print(f"[{args.label}] evaluated {seen}/{args.num_sequences}", flush=True)

  rows = []
  for radius in args.radii:
    true = torch.tensor(values[(radius, "true")], dtype=torch.float64)
    for condition in CONDITIONS:
      sample = torch.tensor(values[(radius, condition)], dtype=torch.float64)
      delta = sample - true
      rows.append({
        "label": args.label,
        "radius": radius,
        "condition": condition,
        "num_sequences": sample.numel(),
        "mean_nll_per_token": sample.mean().item(),
        "mean_delta_nll_vs_true": delta.mean().item(),
        "delta_standard_error": (
          delta.std(unbiased=True).item() / max(delta.numel(), 1) ** 0.5
          if delta.numel() > 1 else 0.0),
      })

  summary = {
    "label": args.label,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "parameterization": str(model.parameterization),
    "sequence_length": args.sequence_length,
    "target_length": args.target_length,
    "radii": args.radii,
    "num_sequences": seen,
    "mc_samples": 1 if str(model.parameterization) == "ar" else args.mc_samples,
    "seed": args.seed,
    "intervention": (
      "identical local prefix and target; replace only recurrent state entering "
      "the local window"),
    "rows": rows,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_csv(args.output_dir / "prefix_intervention.csv", rows)
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
