#!/usr/bin/env python
"""Peak memory and step time for a real training step, per arm and geometry.

Answers two questions that the FLOP accounting cannot:

1. **Does recomputing the clean boundary prefill buy back enough memory to
   train the SSM at the Transformer's micro batch?** The SSM arms run micro
   batch 4 against the Transformer's 8, not by choice: at L=8192 BiSSM peaks at
   75.1 GiB against the Transformer's 41.6 GiB, and batch 8 needs 137.45 of
   143.77 GiB before optimizer state. Memory, not arithmetic, is the binding
   constraint, and `model.checkpoint_boundary_prefill` is the lever.

2. **Where does linear scaling actually overtake attention?** At L=8192
   attention is only ~48% of Transformer-BD's forward FLOPs, so the SSM's
   linear cost has not yet paid for its lower arithmetic intensity. Sweeping
   the length locates the crossover, which is what the architecture claim
   rests on.

This runs the real `Diffusion._loss` -> backward -> AdamW step, not a forward
microbenchmark, because optimizer state is part of what tips batch 8 over.
Peak memory is `torch.cuda.max_memory_allocated` measured across the whole
step, and the reported step time is the median of the post-warmup steps.

An OOM is a result, not a crash: the row is recorded with `oom: true` so the
sweep keeps going and the boundary shows up in the output.

**Run ONE length per process for the `dit` arm.** flex attention compiles with
static shapes on the first length it sees; a second length in the same process
triggers a dynamic-shape recompile that fails inside Inductor's autotuner
(`'Symbol' object has no attribute 'get_device'`). Observed twice, always on
the second length and never the first (LSF 112599 cleared 8192 then died at
16384; 112602 cleared 16384 then died at 32768). Multiple lengths in one
invocation are fine for the SSM arms, which do not compile a mask.

Example:
  python scripts/smoke/sizing_sweep.py --arms bissm,dit \
      --batch-sizes 4,8 --lengths 8192 --checkpoint-modes off,on
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import hydra
import torch

import main  # noqa: F401 - registers the project's OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion

REPO = Path(__file__).resolve().parents[1].parent

ARMS = {
  # name: (model config, algo config, whether the prefill flag applies)
  "bissm": ("small_bissm", "bd3lm_bissm", True),
  "ussm": ("small_ussm", "bd3lm_ussm", True),
  "dit": ("small", "bd3lm", False),
  # The AR arms are the honest apples-to-apples memory comparison: neither does
  # a boundary prefill (diffusion.py routes only bd3lm through
  # _forward_pass_bissm) and neither doubles the mixer, so what is left is the
  # per-token cost of a Mamba-2 layer against a causal attention layer.
  "ussm-ar": ("small_ussm", "ar", False),
  "dit-ar": ("small_ar_transformer", "ar", False),
}
AR_ARMS = {"ussm-ar", "dit-ar"}


def build(arm, length, block_size, batch_size, checkpoint_prefill):
  model_cfg, algo_cfg, supports_checkpoint = ARMS[arm]
  is_ar = arm in AR_ARMS
  overrides = [
    f"model={model_cfg}",
    f"algo={algo_cfg}",
    "data=carbon-prokaryote",
    f"model.length={length}",
    f"block_size={1 if is_ar else block_size}",
    f"loader.batch_size={batch_size}",
    f"loader.eval_batch_size={batch_size}",
    "loader.global_batch_size=64",
    "training.ema=0",
    "trainer.accumulate_grad_batches=1",
  ]
  if supports_checkpoint:
    overrides.append("model.active_blocks=all")
    overrides.append(
      f"model.checkpoint_boundary_prefill={str(bool(checkpoint_prefill)).lower()}")
  elif arm == "dit":
    # `configs/model/small.yaml` ships `attn_backend: flash_attn`, which
    # `DIT.gen_mask` does not accept -- the production launcher overrides it to
    # `flex` (scripts/train/train_dna_bd3lm_prok_tuned.sh:50). Match that, or
    # the arm dies before it measures anything.
    overrides.append("model.attn_backend=flex")
  if arm == "ussm-ar":
    overrides.append("algo.backbone=ussm")
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(REPO / "configs")):
    config = hydra.compose(config_name="config", overrides=overrides)
  return config


def run_case(arm, length, block_size, batch_size, checkpoint_prefill,
             warmup, iters, device):
  row = {
    "arm": arm, "length": length, "block_size": block_size,
    "batch_size": batch_size,
    "checkpoint_boundary_prefill": bool(checkpoint_prefill),
  }
  try:
    config = build(arm, length, block_size, batch_size, checkpoint_prefill)
    torch.manual_seed(0)
    model = Diffusion(config, DNATokenizer()).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x0 = torch.randint(8, 12, (batch_size, length), device=device)
    attention_mask = torch.ones_like(x0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for step in range(warmup + iters):
      if step == warmup:
        # Discard warmup allocations so the reported peak is steady state.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
      start = time.perf_counter()
      optimizer.zero_grad(set_to_none=True)
      loss = model._loss(x0, attention_mask)
      loss.loss.backward()
      optimizer.step()
      torch.cuda.synchronize(device)
      if step >= warmup:
        times.append(time.perf_counter() - start)

    peak = torch.cuda.max_memory_allocated(device)
    row.update({
      "oom": False,
      "peak_gib": peak / 1024 ** 3,
      "step_seconds": statistics.median(times),
      "nt_per_second": batch_size * length / statistics.median(times),
      "loss": float(loss.loss.detach()),
    })
  except torch.cuda.OutOfMemoryError:
    row.update({"oom": True, "peak_gib": None, "step_seconds": None,
                "nt_per_second": None, "loss": None})
  return row


def main_cli():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arms", default="bissm,dit")
  parser.add_argument("--batch-sizes", default="4,8")
  parser.add_argument("--lengths", default="8192")
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--checkpoint-modes", default="off,on",
                      help="off, on, or both; ignored for the dit arm")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("sizing_sweep needs a CUDA GPU")
  device = torch.device("cuda")
  total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
  print(f"device: {torch.cuda.get_device_name(device)}  "
        f"total {total:.2f} GiB\n")

  modes = [m.strip() for m in args.checkpoint_modes.split(",") if m.strip()]
  rows = []
  header = (f"{'arm':<7}{'L':>7}{'batch':>6}{'ckpt':>6}"
            f"{'peak GiB':>10}{'step s':>9}{'nt/s':>11}")
  print(header)
  print("-" * len(header))
  for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
    for length in [int(v) for v in args.lengths.split(",")]:
      for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
        arm_modes = modes if ARMS[arm][2] else ["off"]
        for mode in arm_modes:
          row = run_case(arm, length, args.block_size, batch_size,
                         mode == "on", args.warmup, args.iters, device)
          rows.append(row)
          # `run_case`'s model, optimizer and activations are freed when it
          # returns; release the caching allocator's blocks too, so the next
          # case's peak is not inflated by this one's leftovers.
          torch.cuda.empty_cache()
          torch.cuda.reset_peak_memory_stats(device)
          if row["oom"]:
            print(f"{arm:<7}{length:>7}{batch_size:>6}{mode:>6}"
                  f"{'OOM':>10}{'--':>9}{'--':>11}")
          else:
            print(f"{arm:<7}{length:>7}{batch_size:>6}{mode:>6}"
                  f"{row['peak_gib']:>10.2f}{row['step_seconds']:>9.3f}"
                  f"{row['nt_per_second']:>11.0f}")

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
      {"device": torch.cuda.get_device_name(device),
       "total_gib": total, "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
  main_cli()
