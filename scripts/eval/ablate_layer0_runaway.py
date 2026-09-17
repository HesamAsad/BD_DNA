#!/usr/bin/env python3
"""Is layer 0's runaway-tau head cluster (15/24 heads, dt~4.4e-6, tau in the
tens of millions of nt at step 40,000) wasted capacity or real signal?

Paired ablation on the SAME checkpoint, SAME val batches (fixed seed): force
those 15 heads' dt back up to the layer's own normal-head median (dt~0.0102,
tau~10nt) via a forward hook on layer 0's in_proj, and compare val NLL against
the untouched model. If NLL is unchanged/better under ablation, those heads
are dead weight. If NLL gets worse, they are doing real work.

Head indices and the ablation target come from
results/runtime_tau_traj/step_40000.json (already measured): runaway =
[2,5,6,11,12,13,14,15,16,17,18,19,20,21,23], normal-head median
dt=0.010198863962339976.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402

CKPT = str(REPO / "outputs/hg38-caduceus/hg_bissm_cos/checkpoints/0-40000.ckpt")
RUNAWAY = [2, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
NORMAL_MEDIAN_DT = 0.010198863962339976
NUM_BATCHES = 64
BATCH_SIZE = 4
SEED = 0


def inv_softplus(y: float) -> float:
  return math.log(math.expm1(y))


def main():
  device = torch.device("cuda")
  model, tokenizer, config, step = load_checkpoint_model(CKPT, 8192, BATCH_SIZE, device)
  print(f"loaded step={step}")

  mixer0 = None
  mixer0_name = None
  for name, module in model.named_modules():
    if hasattr(module, "dt_bias") and hasattr(module, "A_log") and hasattr(module, "in_proj"):
      mixer0 = module
      mixer0_name = name
      break
  assert mixer0 is not None
  print(f"layer-0 mixer: {mixer0_name}  nheads={mixer0.nheads}")

  target_pre_softplus = inv_softplus(NORMAL_MEDIAN_DT)
  runaway_idx = torch.tensor(RUNAWAY, device=device)

  ablate = {"on": False}

  def hook(_module, _inputs, output):
    if not ablate["on"]:
      return None
    out = output.clone()
    nh = mixer0.nheads
    out[..., -nh:][..., runaway_idx] = target_pre_softplus
    return out

  handle = mixer0.in_proj.register_forward_hook(hook)

  config.loader.batch_size = BATCH_SIZE
  config.loader.eval_batch_size = BATCH_SIZE
  config.trainer.accumulate_grad_batches = 1
  world = torch.cuda.device_count() * int(config.trainer.num_nodes)
  config.loader.global_batch_size = BATCH_SIZE * world
  config.loader.eval_global_batch_size = BATCH_SIZE * world
  config.loader.num_workers = min(int(config.loader.num_workers), 8)
  _, valid_loader = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=SEED)

  batches = []
  for i, batch in enumerate(valid_loader):
    if i >= NUM_BATCHES:
      break
    batches.append((batch["input_ids"].to(device), batch["attention_mask"].to(device)))
  print(f"collected {len(batches)} batches")

  def run(ablate_on):
    ablate["on"] = ablate_on
    losses = []
    torch.manual_seed(SEED)
    with torch.inference_mode():
      for x0, mask in batches:
        loss = model._loss(x0, mask)
        val = loss.loss if hasattr(loss, "loss") else loss
        losses.append(float(val.mean() if hasattr(val, "mean") else val))
    return losses

  base = run(False)
  abl = run(True)
  handle.remove()

  diffs = [a - b for a, b in zip(abl, base)]
  print(f"\nbaseline  mean nll = {statistics.mean(base):.6f}  median = {statistics.median(base):.6f}")
  print(f"ablated   mean nll = {statistics.mean(abl):.6f}  median = {statistics.median(abl):.6f}")
  print(f"paired (ablated - baseline): mean {statistics.mean(diffs):+.6f}  median {statistics.median(diffs):+.6f}  "
        f"stdev {statistics.pstdev(diffs):.6f}  n={len(diffs)}")
  worse = sum(1 for d in diffs if d > 0)
  print(f"ablation WORSE than baseline on {worse}/{len(diffs)} batches")
  if statistics.mean(diffs) > 0.01:
    print("-> ablation clearly HURTS: the runaway heads are doing real work.")
  elif statistics.mean(diffs) < -0.01:
    print("-> ablation HELPS: the runaway heads were actively harmful.")
  else:
    print("-> ablation is within noise of baseline: the runaway heads look like wasted capacity.")


if __name__ == "__main__":
  main()
