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

import math
import torch
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import dataloader  # noqa: E402
import diffusion  # noqa: E402
from scripts.eval.dnahnet.mavedb import (  # noqa: E402
  protein_event_baseline,
  read_jsonl_gz,
  stratified_summary,
  summarize_predictions,
)
from scripts.eval.provenance import stamp  # noqa: E402


DEFAULT_DATA = REPO / "data_cache/dnahnet/mavedb_ecoli_k12_21250.jsonl.gz"


# Human-readable provenance per --score-mode. Keyed explicitly rather than
# chained on if/elif: an `else` fallback stamped four different estimators
# (predictive_divergence, pll, pll_causal, state_displacement) with the NELBO
# definition, so a summary.json claimed to be something it was not. The
# completeness test asserts this dict covers every argparse choice.
_SCORE_DEFINITIONS = {
  "nelbo": "paired NELBO(WT) - paired NELBO(mutant), Monte Carlo bound",
  "pll": "pseudo-log-likelihood(mutant) - pseudo-log-likelihood(WT): "
         "sum_i log p(x_i | x_{-i}), bidirectional masked marginals",
  "block_marginal": "fully-masked one-step control: sum log q(x_i | clean "
                    "prefix, whole block masked, t=1); deterministic, no "
                    "within-block conditioning -- isolates what revealing adds",
  "seq_unmask": "sequential-unmasking log-likelihood(mutant) - ...(WT): "
                "within each block reveal true bases left to right, "
                "sum log q(x_j | clean prefix, revealed x_<j); deterministic, "
                "no 1/t, expresses within-block dependence",
  "pll_causal": "causal pseudo-log-likelihood(mutant) - ...(WT): AR "
                "factorisation sum_i log p(x_i | x_{<i}) via the BD model",
  "infill_nt": "Score I: log q(mutant) - log q(WT) at the masked variant "
               "site, one forward pass, per nucleotide",
  "infill_nt_bounded": "Score I per nucleotide, mapped to [0,1] by eq. (6)-(7)",
  "infill_codon": "Score I summed over the whole mutated codon",
  "infill_nsyn": "Score I over the codon, non-synonymous stratum only",
  "state_displacement": "cosine distance between the WT and mutant recurrent "
                        "states at the end of the scan",
  "predictive_divergence": "Score II: mean_j JSD(p_j^wt || p_j^mut) in bits "
                           "over downstream positions j > max(M), stored "
                           "negated so that higher = fitter",
}


def load_checkpoint_model(
    checkpoint_path: Path,
    model_length: int,
    eval_batch_size: int,
    device: torch.device,
    reverse_off: bool = False,
    right_flank: float | None = None,
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
    # Inference default stays 0.0 so every published number reproduces.
    # --right-flank turns on BiSSM's reverse CROSS-BLOCK pathway, which was
    # dead in every arm until 2026-09-20.
    # SEMANTICS: with a clean right cache each block also conditions on the
    # clean blocks AFTER it, so the score stops being a NELBO and becomes a
    # two-sided block PSEUDO-LIKELIHOOD. Legitimate -- Evo 2 prescribes
    # pseudolikelihood for masked models -- but a different quantity.
    config.model.right_flank_probability = (
      0.0 if right_flank is None else float(right_flank))
  if reverse_off:
    # Same ablation as the `bissm_reverse_off` arm of
    # scripts/eval/ppl_ssm_baselines.sh: build the UNIdirectional backbone from
    # the bidirectional checkpoint. UnidirectionalSSM subclasses
    # BidirectionalSSM and shares every parameter name (the two scan directions
    # share one SegmentMamba2), so the weights load unchanged and the
    # in-block reverse scan simply never runs. This isolates what the reverse
    # scan contributes, holding the weights fixed.
    if config.algo.backbone != "bissm":
      raise ValueError(
        f"--reverse-off applies to a bissm checkpoint, got "
        f"{config.algo.backbone}")
    config.algo.backbone = "ussm"

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


def build_pair_tensors(records, tokenizer, model_length: int, prefixes=None):
  """Encode WT/mutant pairs, optionally behind a real genomic prefix.

  With `prefixes`, each variant is placed in BLOCK 1 of a two-block window and
  block 0 is filled with 256 nt of native E. coli sequence immediately 5' of
  that assay's gene. That gives every scored token a fully populated recurrent
  prefix cache and, for the DiT, satisfies block_q > block_kv so the
  offset-block-causal cross-attention path actually fires -- neither of which
  happens in the default single-block geometry. The prefix is upstream of the
  target, identical for WT and mutant, so it cannot leak which is which.
  """
  rows, masks = [], []
  for record in records:
    prefix = None if prefixes is None else prefixes.get(record["score_set_urn"])
    for key in ("wt_sequence", "mut_sequence"):
      if prefix is None:
        ids, token_mask = encode_dna(tokenizer, record[key], model_length)
      else:
        block = len(prefix)
        pre_ids, _ = encode_dna(tokenizer, prefix, block)
        var_ids, var_mask = encode_dna(
          tokenizer, record[key], model_length - block)
        ids = pre_ids + var_ids
        token_mask = [False] * block + var_mask
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



def _block_marginal_totals(model, x0, token_mask):
  """Fully-masked one-step control for `seq_unmask`.

  Identical clean caches, identical starting time, but every real position is
  scored from ONE forward pass with its whole block masked -- nothing is ever
  revealed:

      l_marg(x) = sum_b sum_{i in b} log q(x_i | x_{<b}, MASK^B, t=1)

  This is literally `_sequential_unmask_totals`' first forward pass, read at
  every position instead of only at position 0.

  WHY IT IS NEEDED. Sequential unmasking differs from the high-noise NELBO
  score in two ways at once: it removes Monte Carlo corruption sampling, and
  it lets a block condition on its own already-revealed bases. Both this
  control and `seq_unmask` are deterministic, so the difference between THEM
  isolates the second effect. Without it, any sequential improvement could
  simply be the removal of estimator noise.

  It does NOT show that the recovered dependencies are codon dependencies;
  that needs the synonymous-encoding comparison.

  Returns positive NLL-like totals, matching the caller's convention.
  """
  block_size = int(model.config.block_size)
  batch_size, sequence_length = x0.shape
  if sequence_length % block_size:
    raise ValueError(
      f"sequence length {sequence_length} not divisible by block {block_size}")
  num_blocks = sequence_length // block_size

  with model._model_autocast_context():
    left_cache = model.backbone.prefill_left_boundaries_stacked(x0, block_size)
  if model._bissm_use_right_flank(x0.device):
    pass  # allowed: see the semantics note in load_checkpoint_model

  folded = (batch_size * num_blocks, block_size)
  clean = x0.reshape(*folded)
  current = torch.full_like(clean, model.mask_index)
  # t = 1.0 is seq_unmask's step-0 time, so the two share this forward exactly.
  p_step = torch.ones((batch_size * num_blocks, 1),
                      device=x0.device, dtype=torch.float32)
  sigma = model._sigma_from_p(p_step).reshape(batch_size * num_blocks)
  with model._model_autocast_context():
    logits = model.backbone.forward_active(
      current, sigma, left_cache=left_cache, right_cache=None)
  scores = model._subs_parameterization(logits, current)
  logp = scores.gather(-1, clean[:, :, None]).squeeze(-1).double()
  total = logp.reshape(batch_size, sequence_length)
  return -(total * token_mask).sum(dim=-1)


def _sequential_unmask_totals(model, x0, token_mask):
  """Exact log-likelihood under a fixed left-to-right unmasking order.

  WHY THIS EXISTS. The NELBO estimator draws t and masks each position
  independently. At the epsilon=0.9 setting that scored best, a block of 8
  holds 0.4 visible bases on average and 68% of blocks are masked ENTIRELY
  (E[t^8] over U(0.9,1) = 0.681). A fully masked block is scored as a product
  of independent per-position marginals, which cannot express any within-block
  dependency: if the only legal strings are AA and CC, the marginals are
  uniform at both positions and the impossible AC scores exactly like AA. The
  codon is the DNA analogue. The model can know the dependency perfectly and
  still be unable to express it through that estimator.

  This estimator removes that limitation. Within each block, reveal the true
  bases one at a time, left to right, and read each base's log-probability
  given the clean prefix AND the already-revealed bases of its own block:

      l_seq(x) = sum_b sum_j log q(x_{b,j} | x_{<b}, x_{b,<j}, MASK_{b,>=j})

  Properties: deterministic (no Monte Carlo), no 1/t weight, and it is an
  EXACT likelihood for the generative order it defines -- not a bound, though
  also not the likelihood of a stochastic-order diffusion sampler.

  Cost: block_size sequential steps, with all blocks batched inside each step,
  so 8 forward passes per sequence against the NELBO path's 128 draws.

  Returns positive NLL-like totals so the caller's `wt_loss - mut_loss`
  keeps its meaning.
  """
  block_size = int(model.config.block_size)
  batch_size, sequence_length = x0.shape
  if sequence_length % block_size:
    raise ValueError(
      f"sequence length {sequence_length} not divisible by block {block_size}")
  num_blocks = sequence_length // block_size

  with model._model_autocast_context():
    left_cache = model.backbone.prefill_left_boundaries_stacked(x0, block_size)
  if model._bissm_use_right_flank(x0.device):
    pass  # allowed: conditioning becomes two-sided, reveal order unchanged

  folded = (batch_size * num_blocks, block_size)
  clean = x0.reshape(*folded)
  current = torch.full_like(clean, model.mask_index)
  logp = torch.zeros(folded, dtype=torch.float64, device=x0.device)

  for step in range(block_size):
    # fraction of the block still hidden, used only if the net is time-conditioned
    remaining = (block_size - step) / block_size
    p_step = torch.full((batch_size * num_blocks, 1), remaining,
                        device=x0.device, dtype=torch.float32)
    sigma = model._sigma_from_p(p_step).reshape(batch_size * num_blocks)
    with model._model_autocast_context():
      logits = model.backbone.forward_active(
        current, sigma, left_cache=left_cache, right_cache=None)
    scores = model._subs_parameterization(logits, current)
    logp[:, step] = scores[:, step, :].gather(
      -1, clean[:, step, None]).squeeze(-1).double()
    current[:, step] = clean[:, step]          # reveal the true base, then continue

  total = logp.reshape(batch_size, sequence_length)
  return -(total * token_mask).sum(dim=-1)     # positive, NLL-like


def _pll_totals(model, x0, token_mask, chunk_size=128):
  """Pseudo-log-likelihood: sum_i -log p(x_i | x_{-i}), one masked position at a time.

  A block-diffusion model has no exact likelihood -- log p(x) factorises into
  per-block conditionals that the NELBO only bounds. The pseudo-likelihood
  replaces that bound with a sum of terms each of which the model computes
  EXACTLY: mask position i, leave every other position clean, read
  log p(x_i | x_{-i}). This is the standard estimator for masked language
  models (Salazar et al. 2020) and underlies ESM-1v's protein variant scoring.
  It is deterministic -- no Monte Carlo, no seed variance.

  Validity here rests on two facts, both asserted below. At
  `model_length == block_size` there is exactly one block, so `block_diff_mask`
  admits no x_t -> x_0 attention and the clean stream cannot leak the answer;
  and `time_conditioning` is False for every BD algo, so the output depends
  only on the masked input.

  Returns a positive NLL-like total, matching the sign convention of the
  NELBO path so `predicted_fitness = wt_loss - mut_loss` is unchanged.
  """
  # The leak this guards against is the CLEAN stream: with more than one block,
  # x_t could in principle read x_0 and the masked marginal would return the
  # answer. That stream only exists when `cross_attn` is set -- every estimator
  # here builds it as `cat(noisy, clean)` under `if model.cross_attn`, and
  # passes `noisy` alone otherwise. A state-space backbone therefore sees only
  # the noisy sequence, in which the scored position is masked, so there is
  # nothing to leak from at any model_length. Guarding it anyway made pll,
  # pll_causal and all four infill modes unrunnable on the b8/b32 checkpoints
  # -- the two best models in the study had 3 estimators against block 256's 9.
  if model.cross_attn and int(model.config.model.length) != int(model.config.block_size):
    raise ValueError(
      f"PLL scoring requires one block under cross-attention: model.length "
      f"({model.config.model.length}) must equal block_size "
      f"({model.config.block_size}), otherwise x_t can attend to the clean "
      f"stream and the masked marginal leaks the answer")
  if model.config.algo.time_conditioning:
    raise ValueError("PLL scoring assumes time_conditioning=False")

  batch_size, length = x0.shape
  totals = torch.zeros(batch_size, dtype=torch.float64, device=x0.device)
  for row in range(batch_size):
    positions = token_mask[row].nonzero(as_tuple=True)[0]
    sequence = x0[row]
    for start in range(0, positions.numel(), chunk_size):
      index = positions[start:start + chunk_size]
      count = index.numel()
      arange = torch.arange(count, device=x0.device)
      noisy = sequence.unsqueeze(0).repeat(count, 1)
      noisy[arange, index] = model.mask_index
      if model.cross_attn:
        clean = sequence.unsqueeze(0).repeat(count, 1)
        model_input = torch.cat((noisy, clean), dim=-1)
      else:
        model_input = noisy
      sigma = torch.zeros(count, 1, device=x0.device, dtype=torch.float32)
      log_scores = model.forward(model_input, sigma=sigma)
      totals[row] -= log_scores[
        arange, index, sequence[index]].double().sum()
  return totals


def _causal_pll_totals(model, x0, token_mask, chunk_size=128):
  """sum_i -log p(x_i | x_{<i}) -- the AUTOREGRESSIVE factorisation, computed
  with a block-diffusion model.

  Why this exists. The AR arms score a variant with the exact chain rule,
  conditioning position i on its LEFT context only. `_pll_totals` conditions on
  BOTH flanks. Those differ in two ways at once -- the training objective AND
  the conditioning -- so the 0.2547 (AR) vs 0.1031 (BD PLL) gap cannot be
  attributed to either. This estimator changes only the conditioning: mask
  position i and everything to its right, leave the left clean, read
  log p(x_i | x_{<i}). Summed over i that IS the autoregressive factorisation,
  evaluated by the BD checkpoint.

  If this recovers ~0.25 the BD objective is fine and the deficit was the
  bidirectional conditioning. If it stays near 0.10 the BD model's conditionals
  are genuinely weaker and the objective is the cause.

  Same one-block and time_conditioning requirements as `_pll_totals`, for the
  same anti-leak reason, and the same positive NLL-like sign convention.
  """
  # The leak this guards against is the CLEAN stream: with more than one block,
  # x_t could in principle read x_0 and the masked marginal would return the
  # answer. That stream only exists when `cross_attn` is set -- every estimator
  # here builds it as `cat(noisy, clean)` under `if model.cross_attn`, and
  # passes `noisy` alone otherwise. A state-space backbone therefore sees only
  # the noisy sequence, in which the scored position is masked, so there is
  # nothing to leak from at any model_length. Guarding it anyway made pll,
  # pll_causal and all four infill modes unrunnable on the b8/b32 checkpoints
  # -- the two best models in the study had 3 estimators against block 256's 9.
  if model.cross_attn and int(model.config.model.length) != int(model.config.block_size):
    raise ValueError(
      f"causal PLL requires one block under cross-attention: model.length "
      f"({model.config.model.length}) must equal block_size "
      f"({model.config.block_size})")
  if model.config.algo.time_conditioning:
    raise ValueError("causal PLL scoring assumes time_conditioning=False")

  batch_size, length = x0.shape
  totals = torch.zeros(batch_size, dtype=torch.float64, device=x0.device)
  positions_all = torch.arange(length, device=x0.device)
  for row in range(batch_size):
    positions = token_mask[row].nonzero(as_tuple=True)[0]
    sequence = x0[row]
    for start in range(0, positions.numel(), chunk_size):
      index = positions[start:start + chunk_size]
      count = index.numel()
      arange = torch.arange(count, device=x0.device)
      noisy = sequence.unsqueeze(0).repeat(count, 1)
      # Mask position i AND everything after it, so the model sees exactly the
      # left context an autoregressive factorisation would condition on.
      noisy[positions_all.unsqueeze(0) >= index.unsqueeze(1)] = model.mask_index
      if model.cross_attn:
        clean = sequence.unsqueeze(0).repeat(count, 1)
        model_input = torch.cat((noisy, clean), dim=-1)
      else:
        model_input = noisy
      sigma = torch.zeros(count, 1, device=x0.device, dtype=torch.float32)
      log_scores = model.forward(model_input, sigma=sigma)
      totals[row] -= log_scores[
        arange, index, sequence[index]].double().sum()
  return totals


# --------------------------------------------------------------------------
# Score I: context compatibility (masked infilling preference)
# --------------------------------------------------------------------------

_BASES = "TCAG"
_AA_TABLE = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
GENETIC_CODE = {
  b1 + b2 + b3: _AA_TABLE[i * 16 + j * 4 + k]
  for i, b1 in enumerate(_BASES)
  for j, b2 in enumerate(_BASES)
  for k, b3 in enumerate(_BASES)
}


def variant_positions(wt: str, mut: str):
  """Differing nucleotide indices, or None if the pair changes length.

  MEASURED, not assumed: only 91.05% of the 21,250 pairs preserve length. The
  other 8.95% are protein-level delins, for which "the mutated position" is not
  well defined, so they are excluded from Score I and reported separately rather
  than silently mis-scored.
  """
  if len(wt) != len(mut):
    return None
  return [i for i, (a, b) in enumerate(zip(wt, mut)) if a != b]


def _infill_logprobs(model, ids, mask_positions):
  """log p(. | everything unmasked) at each masked position. One forward pass.

  `ids` is [length]; `mask_positions` a LongTensor of indices to mask. Returns
  [len(mask_positions), vocab] log-probabilities. Deterministic -- no Monte
  Carlo, no seed. Both the wild-type and the mutant read-outs come from THIS
  single pass, so their difference carries exactly zero sampling variance.
  """
  noisy = ids.clone()
  noisy[mask_positions] = model.mask_index
  model_input = noisy.unsqueeze(0)
  if model.cross_attn:
    model_input = torch.cat((model_input, ids.unsqueeze(0)), dim=-1)
  sigma = torch.zeros(1, 1, device=ids.device, dtype=torch.float32)
  log_scores = model.forward(model_input, sigma=sigma)
  return log_scores[0, mask_positions]


def _assert_index_lands_on_variant(wt_ids, index, wt_string, positions, model):
  """Fail loudly if the absolute indices do not sit on the intended bases.

  `positions` are 0-based within the VARIANT string, while `wt_ids` is the
  padded model input. Those coincide only when nothing precedes the variant.
  With `--genomic-prefix` the prefix occupies `[0, len(prefix))` and the variant
  starts after it, but `score_batch` computes a non-zero `offset` only for
  `--infill-pad-side left` -- so an infill run behind a prefix would silently
  read positions INSIDE THE PREFIX and score the wrong nucleotides.

  No published number is affected: no run has ever combined an infill mode with
  a genomic prefix. It became REACHABLE on 2026-09-20, when the one-block guard
  was relaxed to fire on `cross_attn` only -- exactly the kind of hole a
  relaxation opens. Checking the bases rather than re-deriving the arithmetic
  catches any offset error, not only this one.
  """
  ids = wt_ids[index].tolist()
  want = [model.tokenizer.convert_tokens_to_ids(wt_string[p]) for p in positions]
  if ids != want:
    raise ValueError(
      f"Score I index does not land on the variant: read token ids {ids[:8]} "
      f"but the wild-type bases at those variant positions are {want[:8]}. A "
      f"genomic prefix shifts the variant; pass the prefix length as `offset`.")


def score_infill(model, wt_ids, mut_ids, wt_string, mut_string, unit="codon",
                 offset=0):
  """Score I. Returns (score, n_units, n_synonymous_units) or (nan, 0, 0).

  Higher = the model prefers the MUTANT, matching `predicted_fitness`'s existing
  sign convention (`wt_loss - mut_loss`), so nothing downstream changes.

  `unit="nt"` masks exactly the differing nucleotides and contrasts the two
  base sequences. `unit="codon"` masks every codon containing a difference and
  contrasts AMINO ACIDS, summing each codon's probability over its synonymous
  spellings before taking the log.

  WHY THE CODON FORM IS THE POINT HERE, measured on this dataset: the median
  variant changes 13 nucleotides across 11 codons but only **1** amino acid
  (max 2), leaving a median of **10 synonymous codon changes** per variant.
  Nucleotide-level scoring therefore spends roughly 91% of its signal on changes
  the assay cannot see. The counting evidence is stark -- macro per-assay
  Spearman against fitness is +0.337 for the non-synonymous codon count but only
  +0.093 for the synonymous count and +0.170 for the raw nucleotide count. A
  synonymous codon contributes EXACTLY zero to the codon score, by construction,
  because its wild-type and mutant amino acids are the same.

  APPROXIMATION, stated rather than buried: the model emits independent
  per-position marginals, so a codon's probability is taken as the product of
  its three positional probabilities. That is the standard masked-marginal
  factorisation (Salazar et al. 2020, and ESM-1v's variant scoring), not an
  exact joint over the codon -- the masked positions are masked jointly but read
  independently.
  """
  positions = variant_positions(wt_string, mut_string)
  if positions is None:
    return float("nan"), 0, 0
  if not positions:
    return 0.0, 0, 0          # identity variant: exactly zero, as for NELBO/PLL

  if unit not in ("nt", "nt_bounded", "codon", "codon_nsyn"):
    raise ValueError(
      f"unit must be nt|nt_bounded|codon|codon_nsyn, got {unit!r}")
  if unit in ("nt", "nt_bounded"):
    index = torch.tensor([p + offset for p in positions], device=wt_ids.device)
    _assert_index_lands_on_variant(wt_ids, index, wt_string, positions, model)
    logp = _infill_logprobs(model, wt_ids, index).double()
    arange = torch.arange(index.numel(), device=wt_ids.device)
    per_position = logp[arange, mut_ids[index]] - logp[arange, wt_ids[index]]
    del arange
    if unit == "nt":
      # S_raw = sum_i [log q_i(mut) - log q_i(wt)].
      return float(per_position.sum()), len(positions), 0
    # BOUNDED AGGREGATION. sigma() applied to the TOTAL is a monotone map of
    # S_raw and so leaves Spearman exactly unchanged (verified: 0.124532 both
    # ways). Applied PER POSITION and then averaged it is NOT monotone in the
    # sum, and it is a different estimator:
    #
    #   mean_i  q_i(mut) / (q_i(mut) + q_i(wt))   in [0, 1]
    #
    # The motivation is a measured property of this dataset: the median variant
    # differs at 13 nucleotides, so an unbounded SUM is dominated by its single
    # most extreme term. Bounding each position first caps any one site's
    # influence at 1/n. Same single forward pass, still zero-variance.
    return float(torch.sigmoid(per_position).mean()), len(positions), 0

  codons = sorted({i // 3 for i in positions})
  if unit == "codon_nsyn":
    # Mask ONLY the codons whose amino acid actually changes, leaving the
    # synonymous ones as observed clean context.
    #
    # WHY. The plain `codon` mode masks every differing codon -- a median of 33
    # nt, 18.1% of a 204-nt fragment -- then reads independent per-position
    # marginals out of a context it has just destroyed, in order to score a
    # median of TEN synonymous codon changes that carry no fitness signal at
    # all. Restricting the mask to the non-synonymous codons (median 3 nt, 1.5%
    # of the fragment) is an 11x reduction in context destruction and scores
    # only what the assay can see. Measured motivation: on the 13,838 variants
    # with exactly one amino-acid change -- the stratum where this benchmark is
    # genuinely "predict the effect of one substitution" -- a zero-parameter
    # BLOSUM62 lookup reaches +0.2384 (positive on 12/12 assays) while Score I
    # codon reaches 0.0031 and Score I nt reaches 0.0506.
    #
    # Note this is NOT the same as normalising by the number of changed
    # positions, which was measured HARMFUL (-0.021 on Score I nt). The gain is
    # in shrinking the scored SET, not in rescaling the sum.
    keep = []
    for codon in codons:
      start = 3 * codon
      wt_aa = GENETIC_CODE.get(wt_string[start:start + 3].upper())
      mut_aa = GENETIC_CODE.get(mut_string[start:start + 3].upper())
      if wt_aa is None or mut_aa is None or wt_aa != mut_aa:
        keep.append(codon)          # non-synonymous, or untranslatable
    synonymous_skipped = len(codons) - len(keep)
    if not keep:
      # Every change was synonymous: exactly zero, as in the `codon` mode.
      return 0.0, len(codons), synonymous_skipped
    codons = keep
  _assert_index_lands_on_variant(
    wt_ids, torch.tensor([p + offset for p in positions],
                         device=wt_ids.device), wt_string, positions, model)
  span = torch.tensor([3 * c + k + offset for c in codons for k in range(3)],
                      device=wt_ids.device)
  logp = _infill_logprobs(model, wt_ids, span).double()
  total, synonymous, skipped = 0.0, 0, 0
  for slot, codon in enumerate(codons):
    rows = logp[3 * slot:3 * slot + 3]                       # [3, vocab]
    start = 3 * codon
    wt_aa = GENETIC_CODE.get(wt_string[start:start + 3].upper())
    mut_aa = GENETIC_CODE.get(mut_string[start:start + 3].upper())
    if wt_aa is None or mut_aa is None:
      # Untranslatable (an N in the codon). Count it -- a partial sum over
      # fewer codons is not comparable to a full one, and silently returning
      # the partial total would place the variant mid-ranking as though the
      # model were indifferent. Refused below instead.
      skipped += 1
      continue
    if wt_aa == mut_aa:
      synonymous += 1
      continue                 # contributes exactly 0; skip the arithmetic
    buckets = {}
    for triplet, amino in GENETIC_CODE.items():
      if amino not in (wt_aa, mut_aa):
        continue
      ids = [model.tokenizer.convert_tokens_to_ids(b) for b in triplet]
      buckets.setdefault(amino, []).append(
        rows[0, ids[0]] + rows[1, ids[1]] + rows[2, ids[2]])
    if wt_aa not in buckets or mut_aa not in buckets:
      skipped += 1
      continue
    total += float(torch.logsumexp(torch.stack(buckets[mut_aa]), 0)
                   - torch.logsumexp(torch.stack(buckets[wt_aa]), 0))
  if skipped:
    # Refuse rather than return an incomplete sum. `summarize_predictions` drops
    # non-finite predictions and reports the count, so a refusal is visible;
    # a partial total would not be.
    return float("nan"), len(codons), synonymous
  return total, len(codons), synonymous


def score_predictive_divergence(model, wt_ids, mut_ids, positions, window=0):
  """Score II, eq. (4): mean Jensen-Shannon divergence over DOWNSTREAM positions.

      S_II = (1/|J|) sum_{j in J} JSD( p_j^wt || p_j^mut ),   J = { j > max(M) }

  `p_j = p_theta(x_j | context)` where the context is the prefix carrying either
  the wild-type or the mutant bases at the variant sites M. Every j in J is
  masked in the SAME pass, so each p_j conditions on the prefix and not on the
  other downstream positions -- which is what "the induced predictive
  distribution over the remaining sequence" means. Two forward passes total,
  deterministic, no Monte Carlo.

  NORMALISATION. JSD is computed in BITS (log base 2), for which
  0 <= JSD <= 1 for any pair of distributions, so the mean over J is in [0, 1]
  natively and needs no rescaling. 0 means "the variant changes nothing about
  what should follow".

  SIGN. S_II is a MAGNITUDE of perturbation: larger = more disruptive = LOWER
  fitness. This harness stores `predicted_fitness` as higher = FITTER
  throughout, so the caller negates before storing. Reporting -S_II against +f
  is identical to reporting S_II against rank(-f), which is the convention the
  design asks for; doing both would double-negate and invert the result.

  `window` > 0 truncates J to the next `window` positions after the variant;
  0 uses the whole remaining real sequence.
  """
  if not positions:
    return float("nan"), 0
  start = max(positions) + 1
  real = int((wt_ids != model.tokenizer.convert_tokens_to_ids("N")).sum())
  stop = wt_ids.shape[0] if real <= 0 else min(real, wt_ids.shape[0])
  if window > 0:
    stop = min(stop, start + window)
  if stop <= start:
    return float("nan"), 0
  index = torch.arange(start, stop, device=wt_ids.device)

  wt_log = _infill_logprobs(model, wt_ids, index).double()     # [|J|, vocab]
  mut_log = _infill_logprobs(model, mut_ids, index).double()
  p = wt_log.exp()
  q = mut_log.exp()
  m = 0.5 * (p + q)
  log_m = m.clamp_min(1e-30).log()
  kl_pm = (p * (wt_log - log_m)).sum(-1)
  kl_qm = (q * (mut_log - log_m)).sum(-1)
  jsd_nats = 0.5 * (kl_pm + kl_qm)
  jsd_bits = jsd_nats / math.log(2.0)                          # in [0, 1]
  return float(jsd_bits.mean()), int(index.numel())


def score_state_displacement(model, wt_ids, mut_ids, direction="right"):
  """Score II's cheap premise test: how far does the variant move the state?

  Compares the SSM recurrent summary after scanning wild-type against mutant.
  Returns 0.5 * (1 - cos) in [0, 1]; higher = larger displacement.

  DIRECTION MATTERS, AND THE OBVIOUS CHOICE IS THE WRONG ONE. `prefill_left`
  leaves its state at the 3' end, but the mutations in this benchmark sit at the
  5' end -- median 6% into the fragment, a median of 171 nt upstream of that
  state, with only 1.0% within 20 nt of it. At the measured 4.66 nt per-head
  half-life, a perturbation 171 nt back survives as roughly 2^-37 of itself. A
  left-prefill test would therefore return ~0 for almost every variant, and a
  null result would say nothing about whether the state carries functional
  information -- it would only restate the decay rate.

  `prefill_right` flips its input, so its final state sits at position 0, about
  12 nt from the median mutation: roughly 2^-2.6, or ~17%, survives. That is the
  direction that can actually see these variants. Both are computed so the
  asymmetry is measured rather than argued.
  """
  prefill = (model.backbone.prefill_right if direction == "right"
             else model.backbone.prefill_left)
  with model._model_autocast_context():
    wt = prefill(wt_ids.unsqueeze(0), detach=True)
    mut = prefill(mut_ids.unsqueeze(0), detach=True)
  index = len(wt.states) - 1
  a = wt.states[index].ssm.flatten(1).float()
  b = mut.states[index].ssm.flatten(1).float()
  cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
  return float(0.5 * (1.0 - cosine).item())


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
    score_mode: str = "nelbo",
    prefixes=None,
    infill_pad_side: str = "right",
    score2_window: int = 0,
):
  x0_cpu, token_mask_cpu = build_pair_tensors(
    records, tokenizer, model_length, prefixes)
  bos_rows = model._bos_rows(x0_cpu)
  if bos_rows.any():
    token_mask_cpu[bos_rows, 0] = False
  x0 = x0_cpu.to(model.device)
  token_mask = token_mask_cpu.to(model.device)
  totals = torch.zeros(x0.shape[0], dtype=torch.float64, device=model.device)

  is_ar = str(model.parameterization) == "ar"
  infill = None
  if (score_mode.startswith("infill")
      or score_mode in ("state_displacement", "predictive_divergence")):
    if is_ar:
      raise ValueError(
        f"--score-mode {score_mode} reads DOWNSTREAM context, which an "
        f"autoregressive model cannot see. Use --score-mode nelbo for AR "
        f"checkpoints; the contrast between the two is the point, not a "
        f"limitation to work round.")
    if score_mode == "predictive_divergence":
      pass          # needs only a downstream flank, which the AR check above covers
    elif score_mode == "state_displacement":
      # prefill_right is the direction that can see these variants, and only a
      # bidirectional backbone has an honest one. UnidirectionalSSM subclasses
      # BidirectionalSSM, so this would otherwise run out of distribution rather
      # than fail -- the same shape of bug that reached the Caduceus readout.
      if str(model.config.algo.backbone) != "bissm":
        raise ValueError(
          f"state displacement uses the reverse-scan summary; backbone is "
          f"{model.config.algo.backbone!r}, not 'bissm'")
    elif (model.cross_attn
          and int(model.config.model.length) != int(model.config.block_size)):
      raise ValueError(
        f"Score I requires one block so the clean stream cannot leak the "
        f"answer: model.length ({model.config.model.length}) must equal "
        f"block_size ({model.config.block_size})")
    if score_mode == "predictive_divergence":
      infill = []
      for i, record in enumerate(records):
        pos = variant_positions(record["wt_sequence"], record["mut_sequence"])
        value, n_j = score_predictive_divergence(
          model, x0[2 * i], x0[2 * i + 1], pos or [], window=score2_window)
        # NEGATED: S_II is higher = MORE DISRUPTIVE, but `predicted_fitness` is
        # higher = FITTER everywhere in this harness. Storing -S_II and
        # correlating against +f is exactly "S_II against rank(-f)".
        infill.append((-value, n_j, 0))
      objective = "predictive_divergence_jsd_bits"
    elif score_mode == "state_displacement":
      infill = []
      for i in range(len(records)):
        right = score_state_displacement(model, x0[2 * i], x0[2 * i + 1],
                                         "right")
        left = score_state_displacement(model, x0[2 * i], x0[2 * i + 1], "left")
        # Negated: `predicted_fitness` is higher = FITTER throughout this
        # harness, and a large displacement means a disruptive variant.
        infill.append((-right, left, right))
      objective = "state_displacement_right"
    else:
      # Explicit, not .get(..., "nt"): a default here would silently route an
      # unrecognised infill mode to plain nt scoring and report it under the
      # requested label.
      unit = {"infill_nt": "nt",
              "infill_nt_bounded": "nt_bounded",
              "infill_codon": "codon",
              "infill_nsyn": "codon_nsyn"}[score_mode]
      # PADDING SIDE. MaveDB fragments are 132-216 nt padded to a 256-nt block,
      # i.e. 15.6-48.4% filler, and the three 132-nt assays that carry the
      # ENTIRE block-diffusion macro number are the most padded at 48.4%. The
      # in-block reverse scan runs right-to-left, so with the default right
      # padding it consumes up to 124 N tokens -- a symbol it barely saw in DNA
      # pretraining -- before reaching any real base. The forward scan and the
      # AR baseline are structurally immune. A recovered probe measured Score I
      # swinging 6.64 (N pad) / 5.20 (real genomic pad) / 4.35 (shuffled pad) on
      # one variant, so what fills the block demonstrably matters; which side it
      # sits on is the cheap half of that question.
      infill = []
      for i, record in enumerate(records):
        wt_row, mut_row = x0[2 * i], x0[2 * i + 1]
        shift = 0
        if infill_pad_side == "left":
          real = int(token_mask[2 * i].sum())
          shift = wt_row.shape[0] - real
          wt_row = torch.roll(wt_row, shift, dims=0)
          mut_row = torch.roll(mut_row, shift, dims=0)
        infill.append(score_infill(
          model, wt_row, mut_row,
          record["wt_sequence"], record["mut_sequence"], unit, offset=shift))
      objective = f"infill_preference_{unit}"
  elif score_mode == "block_marginal" and not is_ar:
    totals = _block_marginal_totals(model, x0, token_mask)
    objective = "fully_masked_block_marginal"
  elif score_mode == "seq_unmask" and not is_ar:
    totals = _sequential_unmask_totals(model, x0, token_mask)
    objective = "sequential_unmasking_likelihood"
  elif score_mode == "pll" and not is_ar:
    totals = _pll_totals(model, x0, token_mask)
    objective = "pseudo_log_likelihood"
  elif score_mode == "pll_causal" and not is_ar:
    totals = _causal_pll_totals(model, x0, token_mask)
    # Distinct label: this is the AUTOREGRESSIVE factorisation, not the
    # bidirectional masked marginal, and conflating them in results would
    # hide the very comparison the mode exists to make.
    objective = "pseudo_log_likelihood_causal"
  elif is_ar:
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
  if infill is None and (
      score_mode not in ("pll", "pll_causal", "seq_unmask",
                         "block_marginal") or is_ar):
    objective = "exact_ar_nll" if is_ar else "paired_diffusion_nelbo"
  scored = []
  for index, record in enumerate(records):
    wt_loss = float(totals[2 * index])
    mut_loss = float(totals[2 * index + 1])
    wt_tokens = int(token_mask_cpu[2 * index].sum())
    mut_tokens = int(token_mask_cpu[2 * index + 1].sum())
    scored_record = dict(record)
    if infill is not None:
      # Score I reports a preference, not a likelihood: there is no per-sequence
      # loss to divide by length, and that is the point -- only the changed
      # codons contribute, so the length confound that `partial_corr.py` exists
      # to strip is absent by construction rather than removed afterwards.
      value, units, synonymous = infill[index]
      scored_record.update({
        "loss_type": objective,
        "predicted_fitness": value,
        "predicted_fitness_per_nt": value,
        "wt_loss": float("nan"), "mut_loss": float("nan"),
        "wt_nelbo": float("nan"), "mut_nelbo": float("nan"),
        "wt_loss_per_nt": float("nan"), "mut_loss_per_nt": float("nan"),
        "wt_nelbo_per_nt": float("nan"), "mut_nelbo_per_nt": float("nan"),
        "wt_scored_tokens": wt_tokens, "mut_scored_tokens": mut_tokens,
        "infill_units": units,
        "infill_synonymous_units": synonymous,
        # For state_displacement these two carry the left/right displacements
        # so the decay asymmetry is recoverable from the CSV.
        "state_displacement_left": units if objective.startswith("state") else "",
        "state_displacement_right": synonymous if objective.startswith("state") else "",
      })
      scored.append(scored_record)
      continue
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
    "predicted_fitness_per_nt", "loss_type", "infill_units",
    "infill_synonymous_units", "state_displacement_left",
    "state_displacement_right", "wt_loss", "mut_loss",
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
  parser.add_argument(
    "--right-flank", type=float, default=None,
    help="Probability of supplying the clean RIGHT boundary cache at scoring. "
         "Default off, matching every published number; 1.0 always supplies "
         "it. Turns on BiSSM's reverse cross-block pathway. NOTE this makes "
         "the score a two-sided block pseudo-likelihood, not a NELBO.")
  parser.add_argument(
    "--only-urn", default=None,
    help="Score ONLY this score_set_urn. Exists so each assay can be scored at "
         "its own --model-length: the 12 references run 132-216 nt, so a single "
         "L=256 pads sucB to 48.4%% N while FecA sits at 15.6%%. The BiSSM "
         "reverse "
         "scan starts at the padded end, so that handicap is real AND varies by "
         "assay, which biases the macro average unevenly rather than as a "
         "constant offset. Note the block COUNT changes too, and block count is "
         "what drives the epsilon effect -- measure, do not assume.")
  parser.add_argument(
    "--genomic-prefix", type=Path,
    help="JSON mapping score_set_urn -> a real 256-nt upstream genomic prefix. "
         "Places each variant in block 1 so the recurrent cache is populated "
         "and the DiT cross-block mask is live. Requires --model-length to be "
         "prefix length + one block.")
  parser.add_argument(
    "--score-mode",
    choices=("nelbo", "pll", "pll_causal", "seq_unmask", "block_marginal",
             "infill_nt",
             "infill_nt_bounded", "infill_codon", "infill_nsyn",
             "state_displacement", "predictive_divergence"), default="nelbo",
    help="nelbo: the training objective's paired Monte Carlo NELBO (a bound). "
         "pll: deterministic pseudo-log-likelihood, sum_i log p(x_i | x_{-i}), "
         "exact per term and directly comparable in spirit to an exact "
         "likelihood. Ignored for AR checkpoints, which already have one. "
         "infill_nt / infill_codon: Score I, the masked infilling preference -- "
         "mask what the variant changed and ask which spelling the model would "
         "write. One forward pass, deterministic, no Monte Carlo, and no length "
         "confound. infill_codon marginalises to amino acids, which makes "
         "synonymous changes contribute exactly zero; on this dataset the "
         "median variant carries 10 synonymous codon changes against 1 "
         "non-synonymous, so that is the dominant systematic error in "
         "nucleotide scoring rather than a niche correction. "
         "infill_nsyn masks ONLY the non-synonymous codons, leaving the "
         "synonymous ones as clean context -- an 11x reduction in how much of "
         "the fragment is destroyed before the marginals are read. BD only.")
  parser.add_argument(
    "--score2-window", type=int, default=0,
    help="Score II: truncate the downstream window J to this many positions "
         "after the variant. 0 = the whole remaining real sequence. The "
         "measured effective range of these models is ~128 nt, so a J that "
         "runs the full remainder averages over positions where no "
         "perturbation survives.")
  parser.add_argument(
    "--infill-pad-side", choices=("right", "left"), default="right",
    help="which side the N filler sits on for Score I. 'right' is the historical "
         "behaviour and makes the in-block REVERSE scan enter through up to 124 "
         "N tokens on the 132-nt assays; 'left' moves the filler to the forward "
         "scan's entry instead. Neither is free -- the point is to measure which "
         "costs less.")
  parser.add_argument(
    "--reverse-off", action="store_true",
    help="Ablate the in-block reverse scan: load a bissm checkpoint into the "
         "unidirectional backbone, holding weights fixed. Mirrors the "
         "bissm_reverse_off arm of scripts/eval/ppl_ssm_baselines.sh.")
  args = parser.parse_args()

  if args.batch_size <= 0 or args.mc_samples <= 0:
    parser.error("batch-size and mc-samples must be positive")
  if not 0 < args.epsilon < 1:
    parser.error("epsilon must be in (0, 1)")
  records = list(read_jsonl_gz(args.data))
  if args.only_urn is not None:
    records = [r for r in records if r["score_set_urn"] == args.only_urn]
    if not records:
      raise SystemExit(f"--only-urn {args.only_urn!r} matched no variants")
  if args.max_variants is not None:
    records = records[:args.max_variants]
  if not records:
    raise ValueError("No benchmark records were loaded")

  device = torch.device("cuda")
  if not torch.cuda.is_available():
    raise RuntimeError("MaveDB checkpoint scoring requires a CUDA GPU")
  # Read the TRAINING config off the checkpoint BEFORE load_checkpoint_model
  # rewrites it for inference. That function sets right_flank_probability=0 and
  # model.length to the scoring length -- both correct for scoring, both fatal
  # if read back later as provenance.
  _raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False,
                    mmap=True)
  _trained = OmegaConf.create(
    _raw.get("hyper_parameters", {}).get("config", {}))
  trained_right_flank = float(
    OmegaConf.select(_trained, "model.right_flank_probability", default=float("nan")))
  trained_length = int(
    OmegaConf.select(_trained, "model.length", default=-1))
  del _raw, _trained

  model, tokenizer, config, global_step = load_checkpoint_model(
    args.checkpoint, args.model_length, args.batch_size * 2, device,
    reverse_off=args.reverse_off, right_flank=args.right_flank)
  prefixes = (
    json.loads(args.genomic_prefix.read_text())
    if args.genomic_prefix else None)
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
        score_mode=args.score_mode,
        prefixes=prefixes,
        infill_pad_side=args.infill_pad_side,
        score2_window=args.score2_window,
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
    "reverse_off": bool(args.reverse_off),
    "score_mode": str(args.score_mode),
    "genomic_prefix": bool(args.genomic_prefix),
    "model_length": args.model_length,
    "block_size": int(config.block_size),
    "parameterization": str(model.parameterization),
    "mc_samples": 1 if is_ar else args.mc_samples,
    "requested_mc_samples": args.mc_samples,
    "epsilon": args.epsilon,
    "seed": args.seed,
    "score_definition": (
      "exact NLL(WT) - exact NLL(mutant)" if is_ar
      else _SCORE_DEFINITIONS[str(args.score_mode)]),
    # The right-flank gate is forced to 0 for scoring (load_checkpoint_model),
    # so the RUNTIME value says nothing about training. Record what the
    # CHECKPOINT was trained with, read before that override -- reporting the
    # inference setting as the training setting is exactly how an rf=0.0
    # checkpoint was read as "the suffix carries no information" for a whole
    # campaign.
    "trained_right_flank_probability": trained_right_flank,
    "trained_model_length": trained_length,
    # The zero-parameter baseline this benchmark has to be read against: it
    # scores +0.30931 and beats every model here.
    "protein_event_baseline": {
      k: v for k, v in protein_event_baseline(records).items() if k != "assays"},
    # The stratum where this benchmark genuinely asks "what does ONE amino-acid
    # substitution do", with the zero-parameter BLOSUM62 bar beside it. The
    # pooled macro number is substantially a 1-vs-2 mutation detector and must
    # not be quoted without this.
    "stratified": stratified_summary(scored),
    "headline_metric": "macro mean SIGNED per-assay Spearman "
                       "(macro_abs_spearman retained for comparability with "
                       "runs predating 2026-09-13)",
  })
  stamp(summary, args)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_csv(args.output_dir / "predictions.csv", scored)
  _atomic_json(args.output_dir / "summary.json", summary)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
