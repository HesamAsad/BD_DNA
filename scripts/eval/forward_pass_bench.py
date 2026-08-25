#!/usr/bin/env python3
"""Forward-pass throughput, memory and latency -- the dnaHNet Figure 7 protocol.

WHY A SEPARATE SCRIPT. Our three published panels measure a TRAINING step:
forward + backward + optimizer. dnaHNet's Figure 7 measures a FORWARD PASS
only, and the distinction is not cosmetic -- it is the whole reason the
architectures were built.

  training   every position is processed in parallel. There is no KV cache and
             no recurrent state to reuse, so you pay for activations. A Mamba-2
             layer is WIDER per token than an attention layer (in_proj expands
             768 -> 3224, d_inner 1536 flows through the conv, and the scan
             saves chunk states), so the SSM arms use MORE memory here. Our
             measurements show exactly that, and it is not a defect.

  inference  the Transformer must retain a key and value for every token it has
             seen -- a KV cache linear in context. The SSM retains one
             fixed-size state, constant in context. This is the property SSMs
             exist for, and NONE of our training panels can show it.

So a page that shows only training panels, under titles that read as general
architectural claims, tells half the story and the less flattering half for the
SSM. This script measures the other half on the same axes as a published
baseline, so the two can be compared directly rather than by assertion.

MATCHING THE REFERENCE. dnaHNet Figure 7 sweeps 2^10 to 2^19 nucleotides and
reports (left) wall-clock throughput in tokens/s, (middle) peak GPU memory in
GB, (right) forward-pass latency in ms on a log axis. Their throughput rises,
peaks near 2^16, then declines, which they attribute to memory pressure rather
than algorithmic cost. Their latency panel is FLAT from 2^10 to 2^12-2^13 --
the same fixed per-step cost we measured directly (step time invariant to an 8x
batch change, LSF 119837), visible in their published data and unremarked.

Reported under `torch.no_grad()` and `model.eval()`, so this is the inference
path: no activations retained, no optimizer state, no dropout.

Usage:
  python scripts/eval/forward_pass_bench.py --arms bissm,ussm-ar,dit,dit-ar \
      --lengths 1024,...,524288 --batch 1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main  # noqa: F401,E402 - registers the OmegaConf resolvers
from dataloader import DNATokenizer  # noqa: E402
from diffusion import Diffusion  # noqa: E402
from scripts.eval.provenance import write_json  # noqa: E402
from scripts.smoke.sizing_sweep import ARMS, build  # noqa: E402


def run_case(arm, length, block_size, batch, warmup, iters, device):
  row = {"arm": arm, "length": length, "batch_size": batch,
         "block_size": block_size}
  model = None
  try:
    supports_ckpt = ARMS[arm][2]
    config = build(arm, length, block_size, batch,
                   False if supports_ckpt else None, ())
    torch.manual_seed(0)
    model = Diffusion(config, DNATokenizer()).to(device)
    # eval(), not train(): dropout off, and nothing retained for a backward
    # that never happens. This is the inference path.
    model.eval()
    x0 = torch.randint(8, 12, (batch, length), device=device)
    attention_mask = torch.ones_like(x0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    with torch.no_grad():
      for step in range(warmup + iters):
        if step == warmup:
          # The first pass pays Triton autotune and allocator growth -- 377 s
          # against a ~2 s steady state in real training. Discard it, and reset
          # the peak so the reported memory is steady state rather than warmup.
          torch.cuda.synchronize(device)
          torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model._loss(x0, attention_mask)
        torch.cuda.synchronize(device)
        if step >= warmup:
          times.append(time.perf_counter() - start)

    latency = statistics.median(times)
    row.update({
      "oom": False,
      "peak_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
      "peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
      "forward_seconds": latency,
      "forward_ms": latency * 1e3,
      "tokens_per_second": batch * length / latency,
      "n_iters": len(times),
    })
  except torch.cuda.OutOfMemoryError:
    row.update({"oom": True, "peak_gib": None, "peak_gb": None,
                "forward_seconds": None, "forward_ms": None,
                "tokens_per_second": None})
  finally:
    del model
    torch.cuda.empty_cache()
  return row


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", default="bissm,ussm,ussm-ar,dit,dit-ar")
  # 2^10 .. 2^19, the dnaHNet Figure 7 range.
  parser.add_argument("--lengths",
                      default=",".join(str(2 ** k) for k in range(10, 20)))
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--iters", type=int, default=15)
  parser.add_argument("--output", type=Path,
                      default=REPO / "results" / "sizing" / "forward_pass.json")
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")
  rows = []
  print(f"{'arm':<9}{'L':>9}{'ms':>10}{'tok/s':>13}{'GB':>9}")
  print("-" * 50)
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
      # A block-diffusion arm needs length divisible by block_size; below that
      # the geometry is undefined rather than slow, so skip rather than fake it.
      if arm in ("bissm", "ussm", "dit") and length % args.block_size:
        continue
      row = run_case(arm, length, args.block_size, args.batch,
                     args.warmup, args.iters, device)
      rows.append(row)
      if row.get("oom"):
        print(f"{arm:<9}{length:>9}{'OOM':>10}")
      else:
        print(f"{arm:<9}{length:>9}{row['forward_ms']:>10.2f}"
              f"{row['tokens_per_second']:>13,.0f}{row['peak_gb']:>9.2f}")
  write_json(args.output, {"device": torch.cuda.get_device_name(0),
                           "protocol": "forward only, no_grad, eval mode; "
                                       "median of post-warmup iterations; "
                                       "matches dnaHNet Figure 7 axes",
                           "rows": rows}, args)
  print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
