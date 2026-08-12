#!/usr/bin/env python3
"""Benchmark cached autoregressive decoding for Transformer or recurrent AR."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402
from scripts.eval.ssm_streaming_benchmark import (  # noqa: E402
  _dna_ids,
  _random_dna,
  generate_ar,
  sequence_diagnostics,
)


def generate_transformer(model, tokenizer, prompt_length, generate_length, seed):
  generator = torch.Generator(device="cpu").manual_seed(seed)
  prompt = _random_dna(
    1, prompt_length, _dna_ids(tokenizer), model.device, generator)
  prompt[:, 0] = tokenizer.bos_token_id
  context = prompt
  samples = [prompt]
  model.backbone.reset_kv_cache()
  torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():
    log_probs = model.forward(context, None, store_kv=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(generate_length):
      token = torch.multinomial(log_probs[:, -1].exp(), num_samples=1)
      samples.append(token)
      context = torch.cat((context, token), dim=1)[:, -prompt_length:]
      log_probs = model.forward(context, None, store_kv=True)
    torch.cuda.synchronize()
  elapsed = time.perf_counter() - started
  cache_bytes = sum(
    block.kv_cache.numel() * block.kv_cache.element_size()
    for block in model.backbone.blocks if block.kv_cache is not None)
  return torch.cat(samples, dim=1), {
    "mode": "transformer_sliding_kv_cache",
    "prompt_length": prompt_length,
    "generated_tokens": generate_length,
    "seconds": elapsed,
    "tokens_per_second": generate_length / elapsed,
    "cache_bytes": cache_bytes,
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
  parser.add_argument("--prompt-length", type=int, default=1024)
  parser.add_argument("--generation-length", type=int, default=512)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("AR decode benchmark requires a CUDA GPU")

  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, args.prompt_length, 1, torch.device("cuda"))
  if str(model.parameterization) != "ar":
    raise ValueError("Checkpoint must use the AR parameterization")
  if str(config.algo.backbone) == "ussm":
    sample, benchmark = generate_ar(
      model, tokenizer, args.prompt_length, args.generation_length, 1,
      args.seed)
  elif str(config.algo.backbone) == "dit":
    sample, benchmark = generate_transformer(
      model, tokenizer, args.prompt_length, args.generation_length, args.seed)
  else:
    raise ValueError(f"Unsupported AR backbone: {config.algo.backbone}")

  result = {
    "label": args.label,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "benchmark": benchmark,
    "sample_diagnostics": sequence_diagnostics(sample[0].cpu(), tokenizer),
    "seed": args.seed,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_json(args.output_dir / "summary.json", result)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
