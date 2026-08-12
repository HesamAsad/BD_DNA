#!/usr/bin/env python3
"""Benchmark recurrent-cache scaling and native generation for DNA SSMs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402


def _dna_ids(tokenizer):
  ids = [tokenizer.convert_tokens_to_ids(base) for base in "ACGT"]
  if len(set(ids)) != 4 or any(value == tokenizer.unk_token_id for value in ids):
    raise ValueError("Tokenizer does not expose four distinct A/C/G/T IDs")
  return ids


def sequence_diagnostics(sequence, tokenizer):
  tokens = tokenizer.convert_ids_to_tokens(sequence.tolist())
  bases = [token for token in tokens if token in "ACGT"]
  counts = Counter(bases)
  total = max(len(bases), 1)
  probabilities = [counts[base] / total for base in "ACGT" if counts[base]]
  entropy_bits = -sum(p * math.log2(p) for p in probabilities)
  longest = current = 0
  previous = None
  for base in bases:
    current = current + 1 if base == previous else 1
    longest = max(longest, current)
    previous = base
  kmers = ["".join(bases[index:index + 6])
           for index in range(max(0, len(bases) - 5))]
  return {
    "length": len(tokens),
    "acgt_fraction": len(bases) / max(len(tokens), 1),
    "gc_fraction": (counts["G"] + counts["C"]) / total,
    "base_entropy_bits": entropy_bits,
    "longest_homopolymer": longest,
    "unique_6mer_fraction": len(set(kmers)) / max(len(kmers), 1),
    "base_frequencies": {base: counts[base] / total for base in "ACGT"},
  }


def _random_dna(batch, length, ids, device, generator):
  choices = torch.randint(
    0, 4, (batch, length), generator=generator, device="cpu")
  vocabulary = torch.tensor(ids, dtype=torch.long)
  return vocabulary[choices].to(device)


def benchmark_prefill(model, tokenizer, lengths, chunk_size, seed):
  generator = torch.Generator(device="cpu").manual_seed(seed)
  ids = _dna_ids(tokenizer)
  rows = []
  for length in lengths:
    model.backbone.reset_kv_cache()
    sequence = _random_dna(1, length, ids, model.device, generator)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    cache = None
    for start in range(0, length, chunk_size):
      cache = model.backbone.prefill_left(
        sequence[:, start:start + chunk_size], cache=cache, detach=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows.append({
      "prefix_length": length,
      "seconds": elapsed,
      "tokens_per_second": length / elapsed,
      "cache_bytes": cache.nbytes,
      "cache_length": cache.length,
      "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    del sequence, cache
  return rows


def generate_ar(model, tokenizer, prompt_length, generate_length, batch_size, seed):
  generator = torch.Generator(device="cpu").manual_seed(seed)
  ids = _dna_ids(tokenizer)
  prompt = _random_dna(batch_size, prompt_length, ids, model.device, generator)
  prompt[:, 0] = tokenizer.bos_token_id
  cache = model.backbone.prefill_left(prompt[:, :-1], detach=True)
  current = prompt[:, -1:]
  generated = [prompt]
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  started = time.perf_counter()
  for _ in range(generate_length):
    logits, cache = model.backbone._scan_active(current, None, cache)
    logits[..., model.mask_index] = model.neg_infinity
    probabilities = logits[:, -1].softmax(-1)
    current = torch.multinomial(probabilities, num_samples=1)
    generated.append(current)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - started
  sequence = torch.cat(generated, dim=1)
  return sequence, {
    "mode": "exact_autoregressive",
    "prompt_length": prompt_length,
    "generated_tokens": generate_length * batch_size,
    "seconds": elapsed,
    "tokens_per_second": generate_length * batch_size / elapsed,
    "cache_bytes": cache.nbytes,
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
  }


def generate_diffusion(
    model, tokenizer, generation_length, batch_size, num_steps):
  if generation_length % model.block_size:
    raise ValueError("generation-length must be divisible by checkpoint block size")
  model.backbone.reset_kv_cache()
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()
  started = time.perf_counter()
  blocks = []
  forward_evaluations = 0
  for block_index in range(generation_length // model.block_size):
    active = model._sample_prior(batch_size, model.block_size).to(model.device)
    if block_index == 0:
      active[:, 0] = tokenizer.bos_token_id
    p_x0 = None
    dt = 1.0 / num_steps
    for step in range(num_steps):
      if model.mask_index not in active:
        break
      t = torch.full(
        (batch_size, 1), 1.0 - step * dt,
        device=model.device, dtype=model.dtype)
      cached_before = p_x0 is not None
      p_x0, active = model._ddpm_caching_update(
        x=active, t=t, dt=dt, p_x0=p_x0, first_hitting=False)
      if not cached_before:
        forward_evaluations += 1
    if model.mask_index in active:
      raise RuntimeError("Diffusion block retained masks after the fixed grid")
    if model.backbone._sampling_left_cache is None:
      raise RuntimeError("Denoised block was not committed to the recurrent cache")
    blocks.append(active)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - started
  sequence = torch.cat(blocks, dim=1)
  return sequence, {
    "mode": "block_diffusion_fixed_grid",
    "block_size": model.block_size,
    "num_steps": num_steps,
    "generated_tokens": generation_length * batch_size,
    "forward_evaluations": forward_evaluations,
    "seconds": elapsed,
    "tokens_per_second": generation_length * batch_size / elapsed,
    "cache_bytes": model.backbone.sampling_cache_nbytes,
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
  }


def _atomic_json(path, value):
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--prefix-lengths", type=int, nargs="+",
                      default=[8192, 65536, 262144, 1048576])
  parser.add_argument("--chunk-size", type=int, default=8192)
  parser.add_argument("--prompt-length", type=int, default=1024)
  parser.add_argument("--generation-length", type=int, default=2048)
  parser.add_argument("--generation-batch-size", type=int, default=1)
  parser.add_argument("--diffusion-steps", type=int, default=64)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("Streaming benchmark requires a CUDA GPU")
  model_length = max(args.chunk_size, args.prompt_length)
  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, model_length, args.generation_batch_size,
    torch.device("cuda"))
  if str(config.algo.backbone) not in {"ussm", "bissm"}:
    raise ValueError("Streaming benchmark requires a recurrent SSM checkpoint")

  with torch.inference_mode():
    prefill = benchmark_prefill(
      model, tokenizer, args.prefix_lengths, args.chunk_size, args.seed)
    if str(model.parameterization) == "ar":
      samples, generation = generate_ar(
        model, tokenizer, args.prompt_length, args.generation_length,
        args.generation_batch_size, args.seed + 1)
    else:
      samples, generation = generate_diffusion(
        model, tokenizer, args.generation_length,
        args.generation_batch_size, args.diffusion_steps)

  summary = {
    "label": args.label,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "parameterization": str(model.parameterization),
    "prefill": prefill,
    "generation": generation,
    "sample_diagnostics": [
      sequence_diagnostics(row.cpu(), tokenizer) for row in samples],
    "seed": args.seed,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
