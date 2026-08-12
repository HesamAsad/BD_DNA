#!/usr/bin/env python3
"""Score the pinned dnaHNet MaveDB set with the checkpoint's own objective.

AR checkpoints use deterministic exact next-token negative log-likelihood.
For a diffusion checkpoint, WT and mutant members of each pair receive the
same sampled time and corruption mask. Common random numbers make their NELBO
difference substantially lower variance than two independent estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
import diffusion  # noqa: E402
from scripts.eval.dnahnet.mavedb import (  # noqa: E402
  read_jsonl_gz,
  summarize_predictions,
)


DEFAULT_DATA = REPO / "data_cache/dnahnet/mavedb_ecoli_k12_21250.jsonl.gz"


def load_checkpoint_model(
    checkpoint_path: Path,
    model_length: int,
    eval_batch_size: int,
    device: torch.device,
):
  raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  hyperparameters = raw.get("hyper_parameters", {})
  config = OmegaConf.create(
    hyperparameters.get("config", hyperparameters))
  if model_length % int(config.block_size):
    raise ValueError(
      f"model_length={model_length} must be divisible by checkpoint "
      f"block_size={config.block_size}")
  config.model.length = model_length
  config.loader.eval_batch_size = eval_batch_size
  config.loader.eval_global_batch_size = eval_batch_size
  config.eval.checkpoint_path = str(checkpoint_path)
  if config.algo.backbone in {"bissm", "ussm"}:
    config.model.active_blocks = "all"
    config.model.right_flank_probability = 0.0

  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion.load_from_checkpoint(
    checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to(device)
  model.eval()
  model.backbone.eval()
  model.noise.eval()
  if getattr(model, "ema", None) is not None:
    model.ema.copy_to(model._get_parameters())
  return model, tokenizer, config, int(raw.get("global_step", -1))


def encode_dna(tokenizer, sequence: str, model_length: int):
  sequence = sequence.upper()
  if len(sequence) > model_length:
    raise ValueError(
      f"Sequence of length {len(sequence)} exceeds model length {model_length}")
  ids = tokenizer.convert_tokens_to_ids(list(sequence))
  if any(token_id == tokenizer.unk_token_id for token_id in ids):
    raise ValueError("Benchmark sequence contains a non-ACGT token")
  n_id = tokenizer.convert_tokens_to_ids("N")
  ids = ids + [n_id] * (model_length - len(ids))
  token_mask = [True] * len(sequence) + [False] * (model_length - len(sequence))
  return ids, token_mask


def build_pair_tensors(records, tokenizer, model_length: int):
  rows, masks = [], []
  for record in records:
    for key in ("wt_sequence", "mut_sequence"):
      ids, token_mask = encode_dna(
        tokenizer, record[key], model_length)
      rows.append(ids)
      masks.append(token_mask)
  return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _loss_from_fixed_corruption(model, x0, t, common_uniform):
  """Matches Diffusion._forward_pass_diffusion with externally fixed q(x_t)."""
  loss_scale, p = model.noise(t)
  sigma = model._sigma_from_p(p[:, 0].unsqueeze(-1))
  if model.mdlm_loss_scale:
    sigma = model.noise.total_noise(t)
    dsigma = model.noise.rate_noise(t)
    p = 1 - torch.exp(-sigma)
    loss_scale = -(dsigma / torch.expm1(sigma))
  xt = torch.where(common_uniform <= p, model.mask_index, x0)
  if hasattr(model, "_preserve_observed_bos"):
    xt = model._preserve_observed_bos(xt, x0)
  elif model.ignore_bos:
    # Compatibility for the lightweight test double; checkpoint-backed models
    # always use the token-aware method above.
    xt[:, 0] = x0[:, 0]

  if model.config.algo.backbone in {"bissm", "ussm"}:
    # The all-block path takes one noise value at each block boundary. A fixed
    # sequence-level time has shape [batch, 1], so materialize its broadcast
    # explicitly for sequences containing more than one block.
    return model._forward_pass_bissm(
      x0=x0,
      xt=xt,
      p=p.expand_as(x0),
      loss_scale=loss_scale.expand_as(x0))

  x_input = torch.cat((xt, x0), dim=-1) if model.cross_attn else xt
  log_scores = model.forward(x_input, sigma=sigma)
  log_p_theta = torch.gather(
    input=log_scores, dim=-1, index=x0[:, :, None]).squeeze(-1)
  return loss_scale * log_p_theta


def _exact_ar_losses(model, x0):
  """Per-position exact next-token NLL, aligned to ``x0`` positions.

  Position zero has no left context in this benchmark and is assigned zero;
  callers must exclude it from the scored-token mask. No corruption or Monte
  Carlo estimate is involved.
  """
  if x0.shape[1] < 2:
    raise ValueError("AR scoring requires sequences of at least two tokens")
  log_probs = model.forward(x0[:, :-1], sigma=None)
  target_nll = -torch.gather(
    input=log_probs, dim=-1, index=x0[:, 1:, None]).squeeze(-1)
  losses = torch.zeros(
    x0.shape, dtype=target_nll.dtype, device=target_nll.device)
  losses[:, 1:] = target_nll
  return losses


def score_batch(
    model,
    tokenizer,
    records,
    model_length: int,
    mc_samples: int,
    epsilon: float,
    generator: torch.Generator,
):
  x0_cpu, token_mask_cpu = build_pair_tensors(
    records, tokenizer, model_length)
  bos_rows = model._bos_rows(x0_cpu)
  if bos_rows.any():
    token_mask_cpu[bos_rows, 0] = False
  x0 = x0_cpu.to(model.device)
  token_mask = token_mask_cpu.to(model.device)
  totals = torch.zeros(x0.shape[0], dtype=torch.float64, device=model.device)

  is_ar = str(model.parameterization) == "ar"
  if is_ar:
    token_mask[:, 0] = False
    token_mask_cpu[:, 0] = False
    losses = _exact_ar_losses(model, x0)
    totals = (losses.double() * token_mask).sum(dim=-1)
  else:
    for _ in range(mc_samples):
      pair_t = torch.rand((len(records), 1), generator=generator)
      pair_t = pair_t * (1.0 - epsilon) + epsilon
      t = pair_t.repeat_interleave(2, dim=0).to(model.device)
      common_uniform = torch.rand(
        (len(records), model_length), generator=generator)
      common_uniform = common_uniform.repeat_interleave(2, dim=0).to(model.device)
      losses = _loss_from_fixed_corruption(model, x0, t, common_uniform)
      totals += (losses.double() * token_mask).sum(dim=-1)
    totals /= mc_samples

  totals = totals.cpu()
  objective = "exact_ar_nll" if is_ar else "paired_diffusion_nelbo"
  scored = []
  for index, record in enumerate(records):
    wt_loss = float(totals[2 * index])
    mut_loss = float(totals[2 * index + 1])
    wt_tokens = int(token_mask_cpu[2 * index].sum())
    mut_tokens = int(token_mask_cpu[2 * index + 1].sum())
    scored_record = dict(record)
    scored_record.update({
      "loss_type": objective,
      "wt_loss": wt_loss,
      "mut_loss": mut_loss,
      # Compatibility aliases retained for aggregate_mavedb.py and old runs.
      "wt_nelbo": wt_loss,
      "mut_nelbo": mut_loss,
      # log p(mut) - log p(WT) = loss(WT) - loss(mutant).
      "predicted_fitness": wt_loss - mut_loss,
      "wt_loss_per_nt": wt_loss / wt_tokens,
      "mut_loss_per_nt": mut_loss / mut_tokens,
      "wt_nelbo_per_nt": wt_loss / wt_tokens,
      "mut_nelbo_per_nt": mut_loss / mut_tokens,
      "predicted_fitness_per_nt": (
        wt_loss / wt_tokens - mut_loss / mut_tokens),
      "wt_scored_tokens": wt_tokens,
      "mut_scored_tokens": mut_tokens,
    })
    scored.append(scored_record)
  return scored


def _atomic_json(path: Path, value):
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def _atomic_csv(path: Path, records):
  fields = [
    "score_set_urn", "score_set_title", "target", "accession",
    "hgvs_nt", "hgvs_pro", "experimental_score", "predicted_fitness",
    "predicted_fitness_per_nt", "loss_type", "wt_loss", "mut_loss",
    "wt_loss_per_nt", "mut_loss_per_nt", "wt_nelbo", "mut_nelbo",
    "wt_nelbo_per_nt", "mut_nelbo_per_nt", "wt_scored_tokens",
    "mut_scored_tokens", "license", "source_url",
  ]
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", newline="", dir=path.parent,
      delete=False) as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    temporary = Path(handle.name)
  os.replace(temporary, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--batch-size", type=int, default=16,
                      help="Number of WT/mutant pairs per GPU batch")
  parser.add_argument("--mc-samples", type=int, default=8)
  parser.add_argument("--model-length", type=int, default=256)
  parser.add_argument("--epsilon", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--max-variants", type=int)
  args = parser.parse_args()

  if args.batch_size <= 0 or args.mc_samples <= 0:
    parser.error("batch-size and mc-samples must be positive")
  if not 0 < args.epsilon < 1:
    parser.error("epsilon must be in (0, 1)")
  records = list(read_jsonl_gz(args.data))
  if args.max_variants is not None:
    records = records[:args.max_variants]
  if not records:
    raise ValueError("No benchmark records were loaded")

  device = torch.device("cuda")
  if not torch.cuda.is_available():
    raise RuntimeError("MaveDB checkpoint scoring requires a CUDA GPU")
  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, args.model_length, args.batch_size * 2, device)
  generator = torch.Generator(device="cpu").manual_seed(args.seed)

  scored = []
  with torch.inference_mode():
    for start in range(0, len(records), args.batch_size):
      batch = records[start:start + args.batch_size]
      scored.extend(score_batch(
        model=model,
        tokenizer=tokenizer,
        records=batch,
        model_length=args.model_length,
        mc_samples=args.mc_samples,
        epsilon=args.epsilon,
        generator=generator))
      print(
        f"[{args.label}] scored {len(scored)}/{len(records)} variants",
        flush=True)

  summary = summarize_predictions(scored)
  is_ar = str(model.parameterization) == "ar"
  summary.update({
    "label": args.label,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_global_step": global_step,
    "backbone": str(config.algo.backbone),
    "model_length": args.model_length,
    "block_size": int(config.block_size),
    "parameterization": str(model.parameterization),
    "mc_samples": 1 if is_ar else args.mc_samples,
    "requested_mc_samples": args.mc_samples,
    "epsilon": args.epsilon,
    "seed": args.seed,
    "score_definition": (
      "exact NLL(WT) - exact NLL(mutant)"
      if is_ar else "paired NELBO(WT) - paired NELBO(mutant)"),
    "headline_metric": "macro mean absolute per-assay Spearman",
  })
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_csv(args.output_dir / "predictions.csv", scored)
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
