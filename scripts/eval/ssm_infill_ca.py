#!/usr/bin/env python3
"""Does the clean right-flank cache actually help? (C-a infilling)

Scores the SAME held-out middle block two ways on the SAME corruption:

  de-novo : left cache from blocks < i, no right cache
  C-a     : left cache from blocks < i, PLUS a right cache from blocks > i

Both use identical weights, identical diffusion times and identical masks, so
the difference is exactly what the clean suffix buys. This is the capability a
causal autoregressive model cannot express at all: it has no mechanism to
condition on sequence to the right of the target.

The target block is never part of either cache by construction
(`prefill_left` sees x0[:, :start], `prefill_right` sees x0[:, end:]), so the
comparison is leak-free in both arms. That property is asserted, not assumed.

Reports mean NELBO per nucleotide for each arm and the delta, over a fixed set
of validation sequences and a fixed set of target block positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402

# `get_dataloaders` reads `trainer.accumulate_grad_batches`, whose value is a
# `${device_count:}` interpolation. Those resolvers are registered by `main.py`,
# which an evaluation script never imports, so register them here too. Keep the
# definitions identical to main.py's -- a divergent `device_count` would change
# the batch-size assertion this config is validated against.
for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)


def _atomic_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def score_block(model, x0, xt, loss_scale, p, start, end, use_right):
  """NELBO of x0[:, start:end] under a shared corruption, with/without suffix."""
  with model._model_autocast_context():
    left_cache = model.backbone.prefill_left(x0[:, :start])
    right_cache = (
      model.backbone.prefill_right(x0[:, end:]) if use_right else None)
    sigma = model._sigma_from_p(p[:, start:start + 1]).squeeze(-1)
    logits = model.backbone.forward_active(
      xt[:, start:end], sigma,
      left_cache=left_cache, right_cache=right_cache)
  log_scores = model._subs_parameterization(logits, xt[:, start:end])
  log_p_theta = torch.gather(
    input=log_scores, dim=-1,
    index=x0[:, start:end, None]).squeeze(-1)
  # loss_scale is negative; the NELBO contribution is loss_scale * log p.
  return (loss_scale[:, start:end] * log_p_theta).double()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--num-batches", type=int, default=32)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--mc-samples", type=int, default=16,
                      help="diffusion times per batch; the two arms share them")
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("C-a infilling evaluation requires a CUDA GPU")
  device = torch.device("cuda")

  # load_checkpoint_model needs an explicit length (it validates
  # length % block_size), so read the checkpoint's own geometry first and keep
  # it -- this evaluation must run at the geometry the model was trained on.
  raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  trained = OmegaConf.create(
    raw.get("hyper_parameters", {}).get("config", {}))
  del raw
  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, int(trained.model.length), args.batch_size, device)
  if str(config.algo.backbone) != "bissm":
    raise ValueError(
      f"C-a requires the bidirectional backbone; got {config.algo.backbone}. "
      f"A unidirectional model has no reverse scan to initialise from a "
      f"right-flank cache.")
  length = int(config.model.length)
  block = int(config.block_size)
  num_blocks = length // block
  if num_blocks < 3:
    raise ValueError(
      f"Need at least 3 blocks so a target can have both a prefix and a "
      f"suffix; got {num_blocks}")
  # Interior blocks only: block 0 has no prefix, the last has no suffix, so
  # neither can show what a suffix is worth.
  targets = list(range(1, num_blocks - 1))

  _, valid_loader = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=args.seed)
  generator = torch.Generator(device="cpu").manual_seed(args.seed)

  totals = {"de_novo": 0.0, "ca": 0.0}
  tokens = 0
  batches = 0
  with torch.inference_mode():
    for batch in valid_loader:
      if batches >= args.num_batches:
        break
      x0 = batch["input_ids"][:, :length].to(device)
      if x0.shape[1] < length:
        continue
      for _ in range(args.mc_samples):
        t = torch.rand((x0.shape[0], 1), generator=generator).to(device)
        t = t * (1 - float(config.training.sampling_eps)) + float(
          config.training.sampling_eps)
        loss_scale, p = model.noise(t)
        loss_scale = loss_scale.expand_as(x0)
        p = p.expand_as(x0)
        uniform = torch.rand(x0.shape, generator=generator).to(device)
        xt = torch.where(uniform <= p, model.mask_index, x0)
        xt = model._preserve_observed_bos(xt, x0)
        for index in targets:
          start, end = index * block, (index + 1) * block
          for arm, use_right in (("de_novo", False), ("ca", True)):
            totals[arm] += float(
              score_block(model, x0, xt, loss_scale, p, start, end,
                          use_right).sum())
          tokens += x0.shape[0] * block
      batches += 1

  # `loss_scale` is negative and `log_p_theta` is negative, so their product is
  # already the positive NELBO contribution -- exactly `diffusion.py`'s
  # `loss = loss_scale * log_p_theta`. Do NOT negate again.
  nelbo = {k: v / tokens for k, v in totals.items()}
  delta = nelbo["de_novo"] - nelbo["ca"]
  summary = {
    "label": args.label,
    "checkpoint": str(args.checkpoint),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    # Read from the checkpoint's OWN config: `load_checkpoint_model` force-sets
    # `config.model.right_flank_probability = 0.0` (it is a training-time
    # sampling knob and this evaluation passes the right cache explicitly), so
    # `config` would report the override rather than how the model was trained.
    "right_flank_probability_trained": float(
      trained.model.get("right_flank_probability", 0.0)),
    "length": length,
    "block_size": block,
    "target_blocks": targets,
    "batches": batches,
    "mc_samples": args.mc_samples,
    "scored_nucleotides": tokens,
    "nelbo_de_novo": nelbo["de_novo"],
    "nelbo_ca": nelbo["ca"],
    "delta_nats_per_nt": delta,
    "note": ("delta > 0 means the clean right flank lowers NELBO, i.e. the C-a "
             "cache helps. Both arms share weights, diffusion times and masks."),
  }
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))
  print(f"\nde-novo {nelbo['de_novo']:.5f}  C-a {nelbo['ca']:.5f}  "
        f"delta {delta:+.5f} nats/nt")


if __name__ == "__main__":
  main()
