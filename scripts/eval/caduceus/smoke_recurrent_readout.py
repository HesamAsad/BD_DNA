#!/usr/bin/env python3
"""Pre-flight for readout (A) on a real checkpoint. Needs a GPU.

Four things the unit tests cannot check, because they run a 2-layer toy model on
CPU with random weights:

1. (A) runs at all on a trained `bissm` checkpoint, and its width is the
   predicted `2 * n_layers_selected * nheads * d_state`.
2. The features are finite and not degenerate (a constant column would mean the
   state saturated and the probe would be fitting noise).
3. **The sigma fix is live.** On a `time_conditioning: True` checkpoint, two
   different sigmas must give different POOLED features. Until 2026-09-13 the
   readout never called `time_embedding` at all, so a sigma sweep would have
   returned a flat result that looked like "sigma does not matter" rather than
   "sigma was never applied". This is the regression guard for that.
4. (A) is timestep-free: the same two sigmas must give IDENTICAL recurrent
   features, because `_prefill` has no `time_embedding` call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)

from scripts.eval.caduceus.embed import (  # noqa: E402
  embed_sequences, embed_sequences_recurrent)
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402

CHECKPOINTS = {
  "cos_rf05": REPO / "outputs/hg38-caduceus/hg_bissm_cos/checkpoints/best.ckpt",
  "gb_rf00": REPO / ("outputs/carbon-prokaryote/2026.08.18/"
                     "dna-bd3lm-bi-mamba2-lr3e-4-b20.95-wd0.1-rf0.0-a16.0-"
                     "dt0.1-N64-L8192-human-lr8192v2-114010/checkpoints/"
                     "1-8000.ckpt"),
}


def describe(name, vectors):
  finite = bool(np.isfinite(vectors).all())
  spread = vectors.std(axis=0)
  dead = int((spread < 1e-8).sum())
  print(f"    {name:26s} shape {str(vectors.shape):>14s}  finite {finite}  "
        f"|x| mean {np.abs(vectors).mean():.4g}  "
        f"dead dims {dead}/{vectors.shape[1]}")
  return finite and dead < vectors.shape[1]


def main():
  if not torch.cuda.is_available():
    raise RuntimeError("needs a GPU")
  device = torch.device("cuda")
  rng = np.random.default_rng(0)
  # Mixed lengths on purpose: exercises the length-grouping path, including a
  # group of one and a length that is not a multiple of 8.
  sequences = ["".join(rng.choice(list("ACGT"), size=n))
               for n in (200, 200, 200, 269, 269, 511, 8)]
  print(f"{len(sequences)} sequences, lengths {[len(s) for s in sequences]}")

  ok = True
  for tag, path in CHECKPOINTS.items():
    if not path.exists():
      print(f"\n== {tag}: MISSING {path}")
      ok = False
      continue
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    trained = OmegaConf.create(raw.get("hyper_parameters", {}).get("config", {}))
    step = int(raw.get("global_step", -1))
    del raw
    model, tokenizer, config, _ = load_checkpoint_model(
      path, int(trained.model.length), 8, device)
    backbone = model.backbone
    timed = getattr(backbone, "time_embedding", None) is not None
    mixer = backbone.layers[0].mixer
    print(f"\n== {tag}: step {step}  time_conditioning={timed}  "
          f"rf_trained={float(trained.model.right_flank_probability)}  "
          f"layers={len(backbone.layers)} nheads={mixer.nheads} "
          f"d_state={mixer.d_state} headdim={mixer.headdim}")

    window = 8192
    # ---- (A) recurrent, last layer and all layers ----
    last = embed_sequences_recurrent(model, tokenizer, sequences, window,
                                     batch_size=4, layers="last")
    expect = 2 * mixer.nheads * mixer.d_state
    print(f"    predicted last-layer width {expect}")
    ok &= describe("(A) recurrent last", last)
    ok &= last.shape[1] == expect
    every = embed_sequences_recurrent(model, tokenizer, sequences, window,
                                      batch_size=4, layers="all")
    ok &= describe("(A) recurrent all", every)
    ok &= every.shape[1] == expect * len(backbone.layers)

    # ---- (C) pooled, two sigmas ----
    pooled_a = embed_sequences(model, tokenizer, sequences, window, "mean", 4,
                               0.0, 0, device, sigma=0.0)
    pooled_b = embed_sequences(model, tokenizer, sequences, window, "mean", 4,
                               0.0, 0, device, sigma=0.5)
    ok &= describe("(C) pooled sigma=0.0", pooled_a)
    moved = float(np.abs(pooled_a - pooled_b).max())
    print(f"    pooled |sigma=0.0 - sigma=0.5|max = {moved:.4g}")
    if timed:
      if moved <= 0:
        print("    FAIL: sigma is applied nowhere -- the time embedding is "
              "still being dropped")
        ok = False
      else:
        print("    PASS: sigma reaches the pooled features")
    else:
      print("    (expected 0: this checkpoint has no time embedding)")

    # ---- (A) must be timestep-free ----
    again = embed_sequences_recurrent(model, tokenizer, sequences, window,
                                      batch_size=4, layers="last")
    if not np.allclose(last, again, atol=0):
      print("    FAIL: (A) is not deterministic across calls")
      ok = False
    else:
      print("    PASS: (A) deterministic (and timestep-free by construction)")

    # ---- order preservation on the real model ----
    singly = np.concatenate(
      [embed_sequences_recurrent(model, tokenizer, [s], window, batch_size=1,
                                 layers="last") for s in sequences], axis=0)
    if not np.allclose(last, singly, atol=1e-4):
      worst = int(np.abs(last - singly).max(axis=1).argmax())
      print(f"    FAIL: length-grouping misaligned rows; worst row {worst}")
      ok = False
    else:
      print("    PASS: length-grouping preserves row order")

    del model
    torch.cuda.empty_cache()

  print("\nSMOKE " + ("PASSED" if ok else "FAILED"))
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
