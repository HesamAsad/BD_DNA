#!/usr/bin/env python3
"""Profile checkpoint forward inference across dnaHNet context lengths.

dnaHNet Appendix A.5 reports single-GPU BF16 forward-pass throughput, peak
memory, and latency from 2^10 through 2^19 nucleotides. This harness measures
the analogous diffusion likelihood forward: one fixed noisy draw is scored at
batch size one, with no backward pass or sampling loop.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.score_mavedb import (  # noqa: E402
  _loss_from_fixed_corruption,
  load_checkpoint_model,
)


DEFAULT_LENGTHS = tuple(2 ** exponent for exponent in range(10, 20))


def parse_lengths(value: str) -> tuple[int, ...]:
  fields = value.replace(",", " ").split()
  if not fields:
    raise ValueError("At least one context length is required")
  lengths = tuple(int(field) for field in fields)
  if any(length <= 0 for length in lengths):
    raise ValueError("Context lengths must be positive")
  if len(set(lengths)) != len(lengths):
    raise ValueError("Context lengths must be unique")
  return lengths


def _atomic_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def _random_dna(tokenizer, length: int, device: torch.device, seed: int):
  base_ids = torch.tensor(
    [tokenizer.convert_tokens_to_ids(base) for base in "ACGT"],
    device=device,
    dtype=torch.long)
  generator = torch.Generator(device=device).manual_seed(seed)
  choices = torch.randint(
    len(base_ids), (1, length), generator=generator, device=device)
  x0 = base_ids[choices]
  common_uniform = torch.rand(
    (1, length), generator=generator, device=device)
  return x0, common_uniform


def profile_length(
    checkpoint: Path,
    length: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict:
  device = torch.device("cuda")
  model, tokenizer, config, global_step = load_checkpoint_model(
    checkpoint, length, eval_batch_size=1, device=device)
  if length % int(config.block_size):
    raise ValueError(
      f"length={length} is not divisible by block_size={config.block_size}")

  # Lightning normally supplies mixed precision around the forward. Explicitly
  # cast the backbone because this standalone profiler bypasses the Trainer.
  model.backbone.to(dtype=torch.bfloat16)
  x0, common_uniform = _random_dna(tokenizer, length, device, seed)
  t = torch.full((1, 1), 0.5, device=device, dtype=torch.float32)

  def forward_once():
    return _loss_from_fixed_corruption(model, x0, t, common_uniform)

  with torch.inference_mode():
    for _ in range(warmups):
      losses = forward_once()
      del losses
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    checksum = None
    for _ in range(repeats):
      torch.cuda.synchronize()
      start = time.perf_counter()
      losses = forward_once()
      torch.cuda.synchronize()
      latencies.append(time.perf_counter() - start)
      checksum = float(losses.float().mean().cpu())
      del losses

  median_latency = float(statistics.median(latencies))
  return {
    "status": "ok",
    "length": length,
    "working_nucleotides": length,
    "latency_seconds": median_latency,
    "latency_repeats_seconds": latencies,
    "throughput_nt_per_second": length / median_latency,
    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
    "allocated_memory_gib": torch.cuda.memory_allocated(device) / 2 ** 30,
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "block_size": int(config.block_size),
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "loss_checksum": checksum,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument(
    "--lengths",
    default=",".join(str(length) for length in DEFAULT_LENGTHS))
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.warmups < 0 or args.repeats <= 0:
    parser.error("warmups must be non-negative and repeats must be positive")
  try:
    lengths = parse_lengths(args.lengths)
  except ValueError as error:
    parser.error(str(error))
  if not torch.cuda.is_available():
    raise RuntimeError("Forward profiling requires a CUDA GPU")
  if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Forward profiling requires BF16 support")

  summary = {
    "label": args.label,
    "checkpoint": str(args.checkpoint.resolve()),
    "protocol": "batch-1 BF16 diffusion likelihood forward",
    "noise_time": 0.5,
    "warmups": args.warmups,
    "repeats": args.repeats,
    "seed": args.seed,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_memory_gib": (
      torch.cuda.get_device_properties(0).total_memory / 2 ** 30),
    "torch_version": torch.__version__,
    "records": [],
  }
  _atomic_json(args.output, summary)

  for length in lengths:
    print(f"[{args.label}] profiling length={length}", flush=True)
    try:
      record = profile_length(
        args.checkpoint, length, args.warmups, args.repeats,
        args.seed + length)
      print(
        f"[{args.label}] length={length} "
        f"throughput={record['throughput_nt_per_second']:.2f} nt/s "
        f"latency={record['latency_seconds']:.4f}s "
        f"peak={record['peak_memory_gib']:.2f}GiB",
        flush=True)
    except torch.cuda.OutOfMemoryError as error:
      record = {
        "status": "oom", "length": length,
        "error": str(error).splitlines()[0]}
      print(f"[{args.label}] length={length} OOM", flush=True)
    summary["records"].append(record)
    _atomic_json(args.output, summary)
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    if record["status"] != "ok":
      break

  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
