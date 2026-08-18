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


def score_block(model, x0, xt, loss_scale, p, start, end, left_cache,
                use_right, right_nt=None, mismatch=False):
  """Per-position NELBO of x0[:, start:end] under a shared corruption.

  `left_cache` is passed in because it does not depend on the arm and is the
  dominant O(L) cost -- rebuilding it per arm made a 12-arm sweep 12x too slow.

  `right_nt` truncates the cache to that many NUCLEOTIDES after the block
  (None = whole suffix). Nucleotide granularity matters: the per-head decay
  half-lives in this checkpoint are a few tokens, so a block-granular sweep
  saturates at its very first point and shows nothing.

  `mismatch` builds the cache from a DIFFERENT sequence's suffix. If the gain
  survives that, it is not real information transfer but a generic effect of
  having any non-zero state, which would invalidate the whole C-a claim.

  Returns [block] summed over the batch, so the caller keeps the position axis.
  """
  with model._model_autocast_context():
    if not use_right:
      right_cache = None
    else:
      suffix = x0[:, end:]
      if right_nt is not None:
        suffix = suffix[:, :right_nt]
      if mismatch:
        suffix = torch.roll(suffix, shifts=1, dims=0)
      right_cache = model.backbone.prefill_right(suffix)
    sigma = model._sigma_from_p(p[:, start:start + 1]).squeeze(-1)
    logits = model.backbone.forward_active(
      xt[:, start:end], sigma,
      left_cache=left_cache, right_cache=right_cache)
  log_scores = model._subs_parameterization(logits, xt[:, start:end])
  log_p_theta = torch.gather(
    input=log_scores, dim=-1,
    index=x0[:, start:end, None]).squeeze(-1)
  # loss_scale is negative; the NELBO contribution is loss_scale * log p.
  return (loss_scale[:, start:end] * log_p_theta).double().sum(dim=0)


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
  parser.add_argument(
    "--right-nt", default="all",
    help="comma-separated NUCLEOTIDE widths for the right cache, e.g. "
         "'4,16,64,256,all'. 'all' uses the whole clean suffix (the headline "
         "setting). Every width is scored on the SAME corruption, so the curve "
         "is paired. Nucleotide granularity is required: the decay half-lives "
         "in these checkpoints are a few tokens, so a block-granular sweep "
         "saturates at its first point.")
  parser.add_argument(
    "--mismatch-control", action="store_true",
    help="add an arm whose cache is built from a DIFFERENT sequence's suffix. "
         "If the gain survives, it is not information transfer.")
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

  sweep = []
  for token in str(args.right_nt).split(","):
    token = token.strip()
    if not token:
      continue
    sweep.append(None if token == "all" else int(token))
  if not sweep:
    raise ValueError("--right-nt produced no values")

  # (name, use_right, right_nt, mismatch)
  arms = [("de_novo", False, None, False)]
  arms += [(f"ca_{'all' if k is None else k}", True, k, False) for k in sweep]
  if args.mismatch_control:
    arms.append(("ca_mismatch", True, None, True))
  totals = {name: torch.zeros(block, dtype=torch.float64, device=device)
            for name, _, _, _ in arms}
  rows = 0
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
          with model._model_autocast_context():
            left_cache = model.backbone.prefill_left(x0[:, :start])
          for arm, use_right, k, mismatch in arms:
            totals[arm] += score_block(
              model, x0, xt, loss_scale, p, start, end, left_cache,
              use_right, right_nt=k, mismatch=mismatch)
          rows += x0.shape[0]
      batches += 1

  # `loss_scale` is negative and `log_p_theta` is negative, so their product is
  # already the positive NELBO contribution -- exactly `diffusion.py`'s
  # `loss = loss_scale * log_p_theta`. Do NOT negate again.
  # Per-position NELBO, then the block mean.
  per_position = {k: (v / rows).tolist() for k, v in totals.items()}
  nelbo = {k: float(sum(v) / len(v)) for k, v in per_position.items()}
  headline = "ca_all" if "ca_all" in nelbo else f"ca_{sweep[-1]}"
  delta = nelbo["de_novo"] - nelbo[headline]

  base = per_position["de_novo"]

  def gain_vector(name):
    return [base[j] - per_position[name][j] for j in range(block)]

  # CURVE 1 -- the DATA's value function, with transmission removed. Position
  # block-1 sits one step from the cache, so the model's own decay barely
  # attenuates it; how this grows with width is the data's correlation scale.
  value_curve = [
    {"right_nt": ("all" if k is None else k),
     "gain_at_nearest_position": gain_vector(f"ca_{'all' if k is None else k}")[-1],
     "block_mean_gain": nelbo["de_novo"] - nelbo[f"ca_{'all' if k is None else k}"]}
    for k in sweep]

  # CURVE 2 -- the REALIZED profile: how the full-suffix gain decays with
  # distance from the block's right edge. Compared with curve 1 this says
  # whether we are limited by the data or by the model's ability to carry it.
  profile = [
    {"distance_from_right_edge": block - 1 - j, "gain": gain_vector(headline)[j]}
    for j in range(block)]

  summary = {
    "label": args.label,
    "checkpoint": str(args.checkpoint),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "right_flank_probability_trained": float(
      trained.model.get("right_flank_probability", 0.0)),
    "length": length,
    "block_size": block,
    "target_blocks": targets,
    "batches": batches,
    "mc_samples": args.mc_samples,
    "scored_rows": rows,
    "scored_nucleotides": rows * block,
    "nelbo_de_novo": nelbo["de_novo"],
    "nelbo_ca": nelbo[headline],
    "delta_nats_per_nt": delta,
    "information_per_block_nats": delta * block,
    "value_curve": value_curve,
    "realized_profile": profile,
    "per_position_nelbo": per_position,
    "mismatch_control": (
      {"nelbo": nelbo["ca_mismatch"],
       "delta_vs_de_novo": nelbo["de_novo"] - nelbo["ca_mismatch"]}
      if "ca_mismatch" in nelbo else None),
    "note": ("delta > 0 means the clean right flank lowers NELBO. All arms share "
             "weights, diffusion times and masks; the target block is in neither "
             "cache. value_curve isolates the data's correlation scale, "
             "realized_profile the model's transmission range."),
  }
  _atomic_json(args.output_dir / "summary.json", summary)

  print(f"\nde-novo {nelbo['de_novo']:.5f}   full-suffix {nelbo[headline]:.5f}"
        f"   delta {delta:+.5f} nats/nt  ({delta*block:.3f} nats/block)")
  print("\ncache width -> gain (nearest position | block mean)")
  for row in value_curve:
    print(f"  {str(row['right_nt']):>5} nt   {row['gain_at_nearest_position']:+.5f}"
          f"   {row['block_mean_gain']:+.5f}")
  print("\ngain vs distance from the block's right edge")
  for d in (0, 1, 2, 4, 8, 16, 32, 64, 128, block - 1):
    if d < block:
      print(f"  d={d:>4}   {profile[block - 1 - d]['gain']:+.5f}")
  if summary["mismatch_control"] is not None:
    m = summary["mismatch_control"]
    print(f"\nmismatched-suffix control: delta {m['delta_vs_de_novo']:+.5f} "
          f"(should be ~0; a large value would invalidate the result)")


if __name__ == "__main__":
  main()
