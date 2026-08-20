#!/usr/bin/env python
"""Which SSM scan actually runs, and what does each cost?

`model.ssm_backend: auto` resolves in `SegmentMamba2._select_backend` to the
fused upstream SSD kernel on CUDA, or a pure-torch reference scan otherwise.
Nothing logs which one ran. If the torch path is silently in use -- for example
because importing `mamba_ssm` failed and left `mamba_chunk_scan_combined` as
None -- it would explain the SSM's memory footprint on its own, because the
reference scan materialises per-position state instead of working in chunks.

This settles it by measurement: report the resolved backend, then run the same
real training step under each backend and compare peak memory and step time.
"""
import argparse, json, os, statistics, time
from pathlib import Path

import hydra, torch
import main  # noqa: F401 - registers the OmegaConf resolvers
from dataloader import DNATokenizer
from diffusion import Diffusion
from models import mamba2_segment as m2

REPO = Path(__file__).resolve().parents[2]


def build(backend, length, batch):
  with hydra.initialize_config_dir(version_base=None,
                                   config_dir=str(REPO / "configs")):
    return hydra.compose(config_name="config", overrides=[
      "model=small_ussm", "algo=ar", "algo.backbone=ussm", "block_size=1",
      "data=carbon-prokaryote", f"model.length={length}",
      f"loader.batch_size={batch}", f"loader.eval_batch_size={batch}",
      "loader.global_batch_size=64", "training.ema=0",
      "trainer.accumulate_grad_batches=1",
      f"model.ssm_backend={backend}"])


def run(backend, length, batch, device, iters=4):
  model = Diffusion(build(backend, length, batch), DNATokenizer()).to(device)
  model.train()
  opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
  x0 = torch.randint(8, 12, (batch, length), device=device)
  am = torch.ones_like(x0)
  # what the model itself resolves for a CUDA tensor
  probe = model.backbone.layers[0].mixer
  resolved = probe._select_backend(torch.zeros(1, device=device))
  torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
  ts = []
  for step in range(2 + iters):
    if step == 2:
      torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss = model._loss(x0, am); loss.loss.backward(); opt.step()
    torch.cuda.synchronize(device)
    if step >= 2: ts.append(time.perf_counter() - t0)
  peak = torch.cuda.max_memory_allocated(device) / 1024**3
  out = {"requested": backend, "resolved": resolved, "peak_gib": peak,
         "step_s": statistics.median(ts), "loss": float(loss.loss.detach())}
  del model, opt, x0, am, loss
  torch.cuda.empty_cache()
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--length", type=int, default=8192)
  ap.add_argument("--batch", type=int, default=4)
  ap.add_argument("--output", type=Path, default=None)
  a = ap.parse_args()
  dev = torch.device("cuda")
  print(f"mamba_chunk_scan_combined importable: "
        f"{m2.mamba_chunk_scan_combined is not None}")
  print(f"fused_mamba2_available(): {m2.fused_mamba2_available()}\n")
  rows = []
  print(f"{'requested':>10}{'resolved':>10}{'peak GiB':>11}{'step s':>9}{'loss':>10}")
  print("-" * 50)
  for b in ("auto", "fused", "torch"):
    try:
      r = run(b, a.length, a.batch, dev); rows.append(r)
      print(f"{r['requested']:>10}{r['resolved']:>10}{r['peak_gib']:>11.2f}"
            f"{r['step_s']:>9.3f}{r['loss']:>10.4f}")
    except Exception as exc:
      print(f"{b:>10}{'-':>10}   FAILED {type(exc).__name__}: {str(exc)[:60]}")
      rows.append({"requested": b, "error": f"{type(exc).__name__}: {exc}"})
  if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
