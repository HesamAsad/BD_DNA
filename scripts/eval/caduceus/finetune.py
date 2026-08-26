#!/usr/bin/env python3
"""GenomicBenchmarks by FINE-TUNING, with the protocol matched to Caduceus.

This replaces the first version of the harness, whose numbers in
`results/caduceus/genomic_benchmarks_ft/*.json` were produced under four
handicaps that are ours, not the model's:

* **`--max-train 20000 / --max-test 8000`** were passed from the environment by
  `finetune.sh` and never recorded anywhere. That is 8.6-16% of the training
  data on `human_ensembl_regulatory`, `human_ocr_ensembl` and
  `human_enhancers_ensembl` -- the three tasks that carry essentially the whole
  gap. Every knob that could hide is now written into the summary JSON.
* **4 epochs**, against Caduceus's 10, with `best_epoch == last` on 3/8 tasks.
* **backbone LR 1e-5** against their 1e-3, never swept, at batch 16 against
  their 128/256, with no schedule and no warmup.
* **one seed**, against their 5-seed mean.

LEAK AUDIT, stated loudly because it is the first thing anyone should ask. The
old code called `evaluate(test)` inside the epoch loop whenever validation
improved (old `finetune.py:144-148`). The test accuracy was *recorded* there,
never compared against anything, and `best_val` alone gated the update -- so it
was **not** a leak and the published numbers are **not** optimistic from this.
They are pessimistic for the four reasons above. The one genuine test-side
blemish was `num_classes = max(ytr.max(), yte.max()) + 1`, which read test
labels to size the head; it is now taken from the train split. This version
keeps the best-validation weights on the host and scores the test split exactly
once, at the end, and a `HeldOutTest` counter proves it in the JSON.

Everything new is behind a flag and the old behaviour is reachable, so each
change can be attributed instead of measured as one lump:

  --preset legacy   exactly the old harness (default)
  --preset v2       protocol match + the conditioning fixes, no new hypotheses
  --preset v2-rc    v2 plus reverse-complement augmentation and test-time RC

Design choices that did not change: mean pooling over unpadded positions, two
parameter groups (a random head needs a large rate, a pretrained backbone a
small one), model selection on a held-out slice of train, and both caches empty
because a benchmark sequence has no prefix and no clean suffix.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.caduceus.embed import pool  # noqa: E402
from scripts.eval.caduceus.genomic_benchmarks import (  # noqa: E402
  TASKS, load_task, reference, task_stats)
from scripts.eval.dnahnet.score_mavedb import (  # noqa: E402
  load_checkpoint_model)

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)


# --------------------------------------------------------------------------
# Encoding, padding, and reverse complement
# --------------------------------------------------------------------------

# Watson-Crick pairing. `N` is its own complement; the special tokens are left
# alone so the table is a well-defined involution over the whole vocabulary.
COMPLEMENT_PAIRS = (("A", "T"), ("C", "G"))


def build_char_table(tokenizer):
  """256-entry uint8 -> token id map, exactly reproducing `encode_dna`.

  `encode_dna` upper-cases and then calls `convert_tokens_to_ids` on a Python
  list of single characters, which is one dict lookup per nucleotide. At
  batch 128 x 768 nt that is 98k dict lookups per step in the training loop.
  This is the same map as a numpy take; `tests` below assert they agree
  bit-for-bit, including the non-ACGT rejection.
  """
  unk = tokenizer.convert_tokens_to_ids("[UNK]")
  table = np.full(256, unk, dtype=np.int64)
  for character in "ACGTN":
    table[ord(character)] = tokenizer.convert_tokens_to_ids(character)
  return table, int(unk)


def build_complement_table(tokenizer, vocab_size):
  """Token-id permutation implementing the base complement.

  Read straight off the tokenizer rather than hard-coded: `DNATokenizer` in
  `dataloader.py:228-238` places 8 special tokens first and then `ACGTN`, so
  A=8, C=9, G=10, T=11, N=12 -- but the offset is a property of that class, not
  of DNA, and hard-coding it would break silently if the special-token block
  ever changed length.
  """
  table = torch.arange(vocab_size, dtype=torch.long)
  for left, right in COMPLEMENT_PAIRS:
    a = tokenizer.convert_tokens_to_ids(left)
    b = tokenizer.convert_tokens_to_ids(right)
    table[a], table[b] = b, a
  # An involution by construction, but assert it: an RC that is not an
  # involution silently produces a third "strand" and every RC result would be
  # meaningless rather than merely wrong.
  if not torch.equal(table[table], torch.arange(vocab_size, dtype=torch.long)):
    raise AssertionError("complement table is not an involution")
  return table


def reverse_complement_ids(ids, mask, complement):
  """Reverse-complement the real run of each row, leaving padding in place.

  The subtlety a naive `flip(dim=1)` gets wrong: our rows are padded, so
  flipping the tensor moves the padding to the other end and shifts the
  sequence off its alignment. Instead each row's real span [first, last] is
  reversed within itself -- source position for output `p` is
  `first + last - p` -- and padded positions keep their original token.

  Holds for either padding side, since `first`/`last` are read from the mask.

  WHAT RC ACTUALLY BUYS US, measured rather than assumed. With both caches
  empty the bidirectional stack is EXACTLY equivariant to plain reversal --
  `flip(hidden(flip(x))) == hidden(x)` to 1.7e-15 in float64, because the two
  scan directions share one `SegmentMamba2` and every other op is
  per-position. All three of our poolings are permutation-invariant, so
  `pool(hidden(flip(x))) == pool(hidden(x))` to 3e-16. Therefore, on an
  unpadded batch, reverse-complement TTA on this architecture is exactly a
  COMPLEMENT-ONLY ensemble: the reversal half is already free and always has
  been. (Padding breaks the identity, because the pad run sits at the same end
  of the row before and after; measured mean-pool deviation 6.8e-2 at 17%
  padding, which is another reason to run --pad-to batch.)

  This lowers what to expect from `--rc-tta`. Caduceus's own w/o-equiv -> Ph
  increment is +0.0042 mean, and that is the value of the WHOLE mechanism on a
  model whose left padding breaks the reversal symmetry ours preserves. Half of
  it we already have. `tests/test_caduceus_finetune.py` pins all of this.
  """
  if ids.shape != mask.shape:
    raise ValueError("ids and mask must have the same shape")
  length = ids.shape[1]
  index = torch.arange(length, device=ids.device)
  # Contiguity of the real span is what makes `first + last - p` correct.
  # Padding is always a single run at one end here, so assert rather than
  # assume: `mask.sum` must equal `last - first + 1`.
  position = torch.where(mask, index, torch.full_like(index, length))
  first = position.min(dim=1).values
  position = torch.where(mask, index, torch.full_like(index, -1))
  last = position.max(dim=1).values
  if not torch.equal(last - first + 1, mask.sum(dim=1)):
    raise ValueError("padding is not a single contiguous run")
  source = (first + last)[:, None] - index[None, :]
  source = source.clamp(0, length - 1)
  flipped = torch.gather(ids, 1, source)
  return torch.where(mask, complement.to(ids.device)[flipped], ids)


class Encoder:
  """Sequences -> (ids, mask), with the padding policy as data.

  `--pad-to task` is the old behaviour: one window for the whole task, sized
  from the longest sequence and rounded up to the block size. On the three weak
  tasks that means 57-65% of every forward pass is `N` padding, and the reverse
  Mamba scan *starts inside it* and runs backward into the real sequence.
  `--pad-to batch` sorts by length and pads to the batch maximum instead, which
  removes most of that and cuts those tasks' compute 2.3-2.9x.
  """

  def __init__(self, tokenizer, task_window, pad_to="task", pad_multiple=1,
               pad_token="N", pad_side="right", fast=True):
    self.tokenizer = tokenizer
    self.task_window = int(task_window)
    self.pad_to = pad_to
    self.pad_multiple = max(1, int(pad_multiple))
    self.pad_side = pad_side
    self.fast = fast
    self.table, self.unk = build_char_table(tokenizer)
    self.pad_id = int(tokenizer.convert_tokens_to_ids(
      "N" if pad_token == "N" else "[PAD]"))
    self.padded_positions = 0
    self.total_positions = 0

  def window_for(self, sequences):
    if self.pad_to == "task":
      return self.task_window
    longest = max(len(s) for s in sequences)
    rounded = -(-longest // self.pad_multiple) * self.pad_multiple
    return min(max(rounded, self.pad_multiple), self.task_window)

  def encode(self, sequences, device):
    window = self.window_for(sequences)
    rows = np.full((len(sequences), window), self.pad_id, dtype=np.int64)
    mask = np.zeros((len(sequences), window), dtype=bool)
    for row, sequence in enumerate(sequences):
      sequence = sequence.upper()
      if len(sequence) > window:
        raise ValueError(
          f"sequence of length {len(sequence)} exceeds window {window}")
      if self.fast:
        ids = self.table[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
      else:
        ids = np.asarray(
          self.tokenizer.convert_tokens_to_ids(list(sequence)), dtype=np.int64)
      if (ids == self.unk).any():
        raise ValueError("benchmark sequence contains a non-ACGT token")
      start = 0 if self.pad_side == "right" else window - len(sequence)
      rows[row, start:start + len(sequence)] = ids
      mask[row, start:start + len(sequence)] = True
    self.padded_positions += int((~mask).sum())
    self.total_positions += mask.size
    return (torch.from_numpy(rows).to(device, non_blocking=True),
            torch.from_numpy(mask).to(device, non_blocking=True))

  @property
  def pad_fraction(self):
    return (self.padded_positions / self.total_positions
            if self.total_positions else 0.0)


# --------------------------------------------------------------------------
# The test-set guard
# --------------------------------------------------------------------------

class HeldOutTest:
  """The official test split behind an access counter.

  A sweep that selects on validation must not touch this at all, and the final
  run must touch it exactly once per seed. Rather than trusting a code reading,
  the counter is asserted against an expected budget and the access log is
  written into the summary JSON, so a future reader can check the claim without
  re-deriving it from the control flow.
  """

  def __init__(self, sequences, labels):
    self._sequences = sequences
    self._labels = labels
    self.log = []

  def __len__(self):
    return len(self._sequences)

  def take(self, reason):
    self.log.append(reason)
    return self._sequences, self._labels

  @property
  def accesses(self):
    return len(self.log)

  def require(self, expected):
    if self.accesses != expected:
      raise AssertionError(
        f"test split touched {self.accesses} times, expected {expected}: "
        f"{self.log}")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Classifier(nn.Module):
  """Pretrained backbone + pooled readout + linear head.

  `layer` taps an intermediate depth. The backbone ties its output projection
  to the 13-token input embedding (`small_bissm.yaml:16`), so the last hidden
  state is forced into token-embedding space and is maximally specialised to
  "which nucleotide is at position i" -- the strongest possible last-layer
  specialisation, and the reason intermediate layers are worth checking at all.
  Tapping below the top skips `final_norm` (which was fit to layer-12
  statistics), so a head LayerNorm is required there and is forced on.
  """

  def __init__(self, backbone, hidden, num_classes, pooling="mean",
               layer=-1, head_layernorm=False, log_length=False,
               length_scale=0.0):
    super().__init__()
    self.backbone = backbone
    self.pooling = pooling
    # Which readout path this backbone needs. Until 2026-08-26 the forward
    # below unconditionally called `layers[i].scan_active(h, left, right)`,
    # which is (a) SSM-only -- a DiT has `blocks`, not `layers`, and no
    # `scan_active`, so every Transformer arm raised AttributeError -- and
    # (b) BIDIRECTIONAL, so an autoregressive SSM checkpoint was read out with
    # a reverse scan it was never trained to use. Every GenomicBenchmarks
    # result we hold is `backbone: bissm` for exactly that reason.
    self.kind = ("dit" if hasattr(backbone, "blocks")
                 else "ssm-uni" if type(backbone).__name__ == "UnidirectionalSSM"
                 else "ssm-bi")
    depth = (backbone.blocks if self.kind == "dit" else backbone.layers)
    self.n_layers = len(depth) if layer < 0 else int(layer)
    if not 1 <= self.n_layers <= len(depth):
      raise ValueError(f"--layer must be in 1..{len(depth)}")
    # The SSM path ends at `final_norm`; the DiT's equivalent trailing norm
    # lives inside `output_layer`, which this readout deliberately skips (it is
    # tied to the 13-token vocabulary). So a DiT tap is never "the full stack",
    # and the head LayerNorm forced below is what normalises it instead.
    self.use_final_norm = (self.kind != "dit"
                           and self.n_layers == len(depth))
    if not self.use_final_norm and not head_layernorm:
      head_layernorm = True
    width = hidden * (2 if pooling == "meanmax" else 1)
    self.norm = nn.LayerNorm(width) if head_layernorm else nn.Identity()
    self.log_length = log_length
    self.length_scale = float(length_scale)
    self.head = nn.Linear(width + (1 if log_length else 0), num_classes)

  def trainable_backbone_parameters(self):
    """Only the parameters the forward pass actually reaches."""
    if self.kind == "dit":
      # vocab_embed + the tapped blocks, plus sigma_map when this checkpoint
      # has one (BD only -- see the note in forward). An AR DiT has no
      # sigma_map, so listing it unconditionally raised AttributeError before
      # a single batch was seen.
      modules = [self.backbone.vocab_embed]
      if getattr(self.backbone, "sigma_map", None) is not None:
        modules.append(self.backbone.sigma_map)
      modules += list(self.backbone.blocks[:self.n_layers])
    else:
      modules = [self.backbone.token_embedding]
      modules += list(self.backbone.layers[:self.n_layers])
      if self.use_final_norm:
        modules.append(self.backbone.final_norm)
    seen, out = set(), []
    for module in modules:
      for parameter in module.parameters():
        if id(parameter) not in seen:
          seen.add(id(parameter))
          out.append(parameter)
    return out

  def head_parameters(self):
    return list(self.norm.parameters()) + list(self.head.parameters())

  def forward(self, ids, attention_mask):
    b = self.backbone
    batch = ids.shape[0]
    if self.kind == "dit":
      # Plain encoder pass: no block-diffusion mask, so a BD checkpoint gets
      # full attention and an AR one stays causal (b.causal). This mirrors
      # what the SSM path does -- empty left/right caches make it a plain pass
      # over the sequence too -- rather than replaying the training-time block
      # structure, which a classification readout has no use for.
      h = b.vocab_embed(ids)
      rotary_cos_sin = b.rotary_emb(h)
      # Conditioning follows how this checkpoint was BUILT, not a fixed choice.
      # dit.py:679 sets adaLN = (not causal) or model.adaln, and :687-690 create
      # sigma_map only then -- so a BD DiT has one and an AR DiT does NOT, and
      # was trained with c=None throughout (diffusion.py passes sigma=None for
      # the AR parameterization). Assuming sigma_map exists raised
      # AttributeError on every Transformer-AR checkpoint.
      #
      # Where it does exist, feed sigma=0: the clean end of the schedule, which
      # is what a classification input is. The blocks tolerate c=None
      # (dit.py:425 falls back to unmodulated norms), but adaLN is a TRAINED
      # module and bypassing it silently discards it.
      sigma_map = getattr(b, "sigma_map", None)
      if sigma_map is None:
        t_cond = None
      else:
        sigma = torch.zeros(ids.shape[0], device=ids.device,
                            dtype=torch.float32)
        t_cond = F.silu(sigma_map(sigma))
      ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if h.is_cuda
             else contextlib.nullcontext())
      with ctx:
        for index in range(self.n_layers):
          h = b.blocks[index](h, rotary_cos_sin, c=t_cond, causal=b.causal,
                              sample_mode=False, mask=None, store_kv=False)
      h = h.float()
    else:
      h = b.token_embedding(ids)
      left = b._empty_cache(batch, h.device, h.dtype, "left")
      with b._compute_autocast(h):
        if self.kind == "ssm-uni":
          # Forward-only, matching how an AR checkpoint was trained. Its layers
          # DO expose scan_active (UnidirectionalSSM subclasses
          # BidirectionalSSM and overrides only backbone-level methods), so the
          # bidirectional call would have run without error and quietly used
          # the reverse scan out of distribution.
          for index in range(self.n_layers):
            h, _ = b.layers[index].scan_clean(h, left.states[index])
        else:
          right = b._empty_cache(batch, h.device, h.dtype, "right")
          for index in range(self.n_layers):
            h = b.layers[index].scan_active(
              h, left.states[index], right.states[index])
        if self.use_final_norm:
          h = b.final_norm(h)
    features = self.norm(pool(h.float(), attention_mask, self.pooling))
    if self.log_length:
      lengths = attention_mask.sum(dim=1).clamp(min=1).float()
      features = torch.cat(
        [features, torch.log(lengths)[:, None] - self.length_scale], dim=-1)
    return self.head(features)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def make_batches(sequences, indices, batch_size, bucket, rng=None,
                 pool_batches=50):
  """Index arrays for one pass. `bucket` sorts by length inside a pool.

  Sorting inside a pool of `pool_batches` batches rather than globally keeps
  the batch composition random -- global sorting would make every epoch see the
  same length-homogeneous batches in the same order, which correlates the
  gradient sequence with length.
  """
  indices = list(indices)
  if not bucket:
    batches = [indices[i:i + batch_size]
               for i in range(0, len(indices), batch_size)]
  else:
    batches = []
    span = batch_size * max(1, pool_batches)
    for start in range(0, len(indices), span):
      chunk = sorted(indices[start:start + span], key=lambda i: len(sequences[i]))
      batches.extend(chunk[i:i + batch_size]
                     for i in range(0, len(chunk), batch_size))
    if rng is not None:
      batches = [batches[i] for i in rng.permutation(len(batches))]
  return batches


def evaluate(classifier, encoder, sequences, labels, batch_size, device,
             complement=None, rc_average="prob"):
  """Top-1 accuracy. With `complement`, average over both strands."""
  classifier.eval()
  correct = 0
  order = make_batches(sequences, range(len(sequences)), batch_size,
                       bucket=encoder.pad_to == "batch", pool_batches=10**6)
  with torch.inference_mode():
    for chunk in order:
      ids, keep = encoder.encode([sequences[i] for i in chunk], device)
      logits = classifier(ids, keep)
      if complement is not None:
        other = classifier(reverse_complement_ids(ids, keep, complement), keep)
        if rc_average == "prob":
          # Average PROBABILITIES, not logits. The two strands are two views of
          # one input, and we want the mixture of the two predictive
          # distributions. Averaging logits is a geometric mean, so the view
          # with the larger logit magnitude dominates -- fine for Caduceus,
          # whose decoder averages logits, because it was PRETRAINED with RC
          # augmentation and the two views are calibrated against each other.
          # Ours has seen no RC in pretraining, so the RC view's confidence is
          # not on the same scale as the forward view's, and an arithmetic mean
          # of probabilities is the version that bounds each view at 1/2.
          # `--rc-average logit` restores their exact rule.
          probability = (logits.float().softmax(-1)
                         + other.float().softmax(-1)) / 2
        else:
          probability = (logits.float() + other.float()) / 2
      else:
        probability = logits
      target = torch.as_tensor(np.asarray(labels)[chunk], device=device)
      correct += int((probability.argmax(dim=-1) == target).sum())
  return correct / len(sequences)


# --------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------

def split_train_val(labels, fraction, seed, stratified):
  """Held-out slice of TRAIN for model selection. Never test."""
  rng = np.random.default_rng(seed)
  n = len(labels)
  cut = max(1, int(fraction * n))
  if not stratified:
    order = rng.permutation(n)
    return order[cut:], order[:cut]
  # Stratified: on `dummy_mouse_enhancers_ensembl` the val slice is ~96 rows,
  # where an unstratified draw can land several points off the class balance
  # and model selection becomes close to random.
  val = []
  for value in np.unique(labels):
    pool = np.flatnonzero(labels == value)
    pool = pool[rng.permutation(len(pool))]
    val.extend(pool[:max(1, int(round(fraction * len(pool))))])
  val = np.asarray(sorted(val))
  train = np.setdiff1d(np.arange(n), val)
  return rng.permutation(train), val


def build_schedule(config, total_steps):
  """Per-group LR multiplier. Group 0 is the backbone, group 1 the head."""
  warmup = max(1, int(config["warmup_frac"] * total_steps))
  freeze = int(config["head_warmup_steps"])

  def factor(step):
    if config["scheduler"] == "none":
      shape = 1.0
    else:
      if step < warmup:
        shape = (step + 1) / warmup
      else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        shape = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return shape

  return [lambda step: 0.0 if step < freeze else factor(step),
          factor]


def train_one(model, encoder, task, config, seed, device, complement=None,
              keep_state=True):
  """Fine-tune once. Returns best-validation accuracy and the weights at it.

  The test split is not visible from here -- `task` carries train/val only.
  `keep_state=False` skips the host copy of the best weights, which a sweep
  does not need: a sweep only ever reads `best["val"]`.
  """
  xtr, ytr, xva, yva, num_classes, length_scale = task
  if int(config["epochs"]) < 1 or not len(xtr):
    raise ValueError("need at least one epoch over a non-empty train split; "
                     "otherwise there is no best-validation checkpoint to score")
  torch.manual_seed(seed)
  model.backbone.load_state_dict(model.pristine)
  for module in model.backbone.modules():
    if isinstance(module, nn.Dropout):
      module.p = float(config["dropout"])

  classifier = Classifier(
    model.backbone, int(model.backbone.hidden_size), num_classes,
    config["pooling"], config["layer"], config["head_layernorm"],
    config["log_length"], length_scale).to(device)

  # The head is random and the backbone is pretrained; one shared learning rate
  # either leaves the head untrained or wrecks the backbone.
  backbone_params = classifier.trainable_backbone_parameters()
  if config["honour_no_weight_decay"]:
    # `dt_bias`, `A_log` and `D` set `_no_weight_decay` in mamba2_segment.py
    # (:119, :125, :128) and nothing in the repo has ever read the flag.
    plain = [p for p in backbone_params if not getattr(p, "_no_weight_decay", False)]
    special = [p for p in backbone_params if getattr(p, "_no_weight_decay", False)]
  else:
    plain, special = backbone_params, []
  groups = [{"params": plain, "lr": config["backbone_lr"]},
            {"params": classifier.head_parameters(), "lr": config["head_lr"]}]
  if special:
    groups.append({"params": special, "lr": config["backbone_lr"],
                   "weight_decay": 0.0})
  optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
  loss_fn = nn.CrossEntropyLoss()

  batch_size = int(config["batch_size"])
  steps_per_epoch = max(1, -(-len(xtr) // batch_size))
  total_steps = steps_per_epoch * int(config["epochs"])
  lambdas = build_schedule(config, total_steps)
  # The third group (no-weight-decay backbone params) follows the backbone.
  schedule = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambdas + ([lambdas[0]] if special else []))
  eval_every = (steps_per_epoch if not config["evals_per_epoch"]
                else max(1, steps_per_epoch // int(config["evals_per_epoch"])))

  rng = np.random.default_rng(seed)
  best = {"val": -1.0, "state": None, "step": -1, "epoch": -1}
  stale, step, stop = 0, 0, False
  for epoch in range(int(config["epochs"])):
    for chunk in make_batches(xtr, rng.permutation(len(xtr)), batch_size,
                              config["pad_to"] == "batch", rng):
      classifier.train()
      if config["head_warmup_steps"]:
        frozen = step < int(config["head_warmup_steps"])
        for parameter in backbone_params:
          parameter.requires_grad_(not frozen)
      ids, keep = encoder.encode([xtr[i] for i in chunk], device)
      if complement is not None and config["rc_aug"] > 0:
        coin = torch.rand(ids.shape[0], device=device) < config["rc_aug"]
        ids = torch.where(coin[:, None],
                          reverse_complement_ids(ids, keep, complement), ids)
      target = torch.as_tensor(ytr[chunk], device=device)
      optimizer.zero_grad(set_to_none=True)
      loss_fn(classifier(ids, keep), target).backward()
      if config["clip_mode"] == "per-group":
        # A randomly initialised head at lr 1e-3 dominates the global gradient
        # norm for the first few hundred steps, so a single clip over both
        # groups scales the backbone's already-tiny update down further.
        for group in optimizer.param_groups:
          torch.nn.utils.clip_grad_norm_(group["params"], config["clip"])
      else:
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), config["clip"])
      optimizer.step()
      schedule.step()
      step += 1
      if step % eval_every == 0 or step == total_steps:
        val = evaluate(classifier, encoder, xva, yva, config["eval_batch_size"],
                       device, complement if config["rc_tta"] else None,
                       config["rc_average"])
        if val > best["val"]:
          best = {"val": val, "step": step, "epoch": epoch,
                  "state": ({k: v.detach().to("cpu", copy=True)
                             for k, v in classifier.state_dict().items()}
                            if keep_state else None)}
          stale = 0
        else:
          stale += 1
          if config["patience"] and stale >= int(config["patience"]):
            stop = True
            break
    if stop:
      break
  for parameter in backbone_params:
    parameter.requires_grad_(True)
  return best, classifier


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

SWEEPABLE = ("backbone_lr", "head_lr", "batch_size", "epochs", "dropout",
             "weight_decay", "pooling", "layer", "rc_aug", "warmup_frac",
             "head_warmup_steps")


def parse_sweep(spec, base):
  """`"backbone_lr=1e-5,1e-4;head_lr=1e-3,3e-3"` -> list of config overrides."""
  if not spec:
    return [{}]
  axes = []
  for part in spec.split(";"):
    part = part.strip()
    if not part:
      continue
    key, _, values = part.partition("=")
    key = key.strip().replace("-", "_")
    if key not in SWEEPABLE:
      raise ValueError(f"{key} is not sweepable; pick from {SWEEPABLE}")
    cast = type(base[key])
    axes.append([(key, cast(v.strip())) for v in values.split(",") if v.strip()])
  return [dict(combo) for combo in itertools.product(*axes)]


def describe(overrides):
  return " ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "base"


# --------------------------------------------------------------------------
# One task
# --------------------------------------------------------------------------

def run_task(name, model, args, base_config, device, complement):
  stats = task_stats(name)
  xtr_all, ytr_all, xte, yte = load_task(
    name, args.max_train, args.max_test, args.seed)
  # Shape check on the label alphabet, not a measurement: the head is sized from
  # the TRAIN labels (the old code read `yte.max()`), so a class that only ever
  # appears in test would silently be unpredictable rather than loudly wrong.
  if int(np.max(yte)) >= stats["num_classes"]:
    raise ValueError(
      f"{name}: test labels reach {int(np.max(yte))} but the train split only "
      f"defines {stats['num_classes']} classes")
  guard = HeldOutTest(xte, yte)

  block = int(base_config["block_size"])
  if args.window != "auto":
    window = int(args.window)
  elif args.window_from == "full":
    # Size from the FULL splits, so the window stops silently depending on
    # --max-train the way the old code's did.
    window = min(-(-stats["max_length"] // block) * block, args.window_cap)
  else:
    longest = max(max(len(s) for s in xtr_all), max(len(s) for s in xte))
    window = min(-(-longest // block) * block, args.window_cap)

  encoder = Encoder(model.tokenizer, window, base_config["pad_to"],
                    args.pad_multiple, args.pad_token, args.pad_side,
                    not args.slow_encode)
  length_scale = float(np.log(max(1.0, stats["median_length"])))
  seeds = args.seed_list
  sweep_seeds = seeds[:max(1, args.sweep_seeds)]

  def task_for(seed):
    tr, va = split_train_val(ytr_all, args.val_fraction, seed,
                             base_config["stratified_val"])
    return ([xtr_all[i] for i in tr], ytr_all[tr],
            [xtr_all[i] for i in va], ytr_all[va],
            stats["num_classes"], length_scale)

  # ---- sweep: validation only, test untouched -----------------------------
  # A single-point grid skips this loop entirely: with nothing to select
  # between, a selection pass would just be a second copy of the final run.
  # When there IS a grid, the winner is retrained for every final seed rather
  # than reusing the sweep's weights -- more compute, but it keeps "trained on
  # this seed" and "selected by this seed" from silently becoming the same run.
  grid = parse_sweep(args.sweep, base_config)
  sweep_rows = None
  winner = grid[0]
  if len(grid) > 1:
    sweep_rows = []
    for overrides in grid:
      config = dict(base_config, **overrides)
      vals = []
      for seed in sweep_seeds:
        started = time.time()
        best, classifier = train_one(model, encoder, task_for(seed), config,
                                     seed, device, complement, keep_state=False)
        vals.append(best["val"])
        del classifier, best
        torch.cuda.empty_cache()
      print(f"    sweep {describe(overrides):<48} val {np.mean(vals):.4f}"
            f"  ({(time.time() - started) / 60:.1f} min/seed)", flush=True)
      sweep_rows.append({"config": overrides, "val_mean": float(np.mean(vals)),
                         "val_per_seed": vals})
    winner = max(sweep_rows, key=lambda r: r["val_mean"])["config"]
    print(f"    winner {describe(winner)}", flush=True)
  guard.require(0)  # nothing above this line has seen the test split
  config = dict(base_config, **winner)

  # ---- final: one test evaluation per seed --------------------------------
  accuracies, vals, epochs, steps = [], [], [], []
  for seed in seeds:
    started = time.time()
    best, classifier = train_one(model, encoder, task_for(seed), config, seed,
                                 device, complement)
    classifier.load_state_dict(best["state"])
    sequences, labels = guard.take(f"final seed={seed}")
    accuracies.append(evaluate(
      classifier, encoder, sequences, labels, config["eval_batch_size"], device,
      complement if config["rc_tta"] else None, config["rc_average"]))
    vals.append(best["val"])
    epochs.append(best["epoch"])
    steps.append(best["step"])
    print(f"    seed {seed}: val {best['val']:.4f} @ epoch {best['epoch']} "
          f"step {best['step']} -> test {accuracies[-1]:.4f} "
          f"({(time.time() - started) / 60:.1f} min)", flush=True)
    del classifier, best
    torch.cuda.empty_cache()
  guard.require(len(seeds))

  return {
    "task": name,
    "accuracy": float(np.mean(accuracies)),
    "accuracy_std": float(np.std(accuracies)),
    "accuracy_per_seed": accuracies,
    "val_accuracy": float(np.mean(vals)), "val_per_seed": vals,
    "best_epoch": epochs, "best_step": steps, "seeds": seeds,
    "config": {k: config[k] for k in sorted(SWEEPABLE)},
    "sweep": sweep_rows,
    "window": window, "pad_fraction": round(encoder.pad_fraction, 4),
    "n_train_used": len(xtr_all), "n_test_used": len(xte),
    "n_train_full": stats["n_train_full"], "n_test_full": stats["n_test_full"],
    "train_fraction": round(len(xtr_all) / stats["n_train_full"], 4),
    "num_classes": stats["num_classes"],
    "median_length": stats["median_length"], "max_length": stats["max_length"],
    "test_evaluations": guard.accesses, "test_access_log": guard.log,
    "caduceus_ph_published": reference(name, "ph"),
    "caduceus_ps_published": reference(name, "ps"),
    "caduceus_no_equiv_published": reference(name, "no_equiv"),
    "delta_ph": float(np.mean(accuracies)) - reference(name, "ph"),
    "delta_ps": float(np.mean(accuracies)) - reference(name, "ps"),
  }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

PRESETS = {
  "legacy": {},
  # Protocol match plus the conditioning fixes. Deliberately contains no new
  # modelling hypothesis: no dropout, no RC, no layer tap, no pooling change,
  # so `v2 - legacy` is attributable to protocol alone.
  "v2": {
    "epochs": 10, "batch_size": 128, "weight_decay": 0.1,
    "scheduler": "cosine", "warmup_frac": 0.05, "head_warmup_steps": 200,
    "clip_mode": "per-group", "head_layernorm": True,
    "pad_to": "batch", "window_from": "full", "stratified_val": True,
    "evals_per_epoch": 4, "patience": 8,
  },
}
# v2 plus the reverse-complement pair, kept separate so `v2-rc - v2` isolates
# what RC is worth on OUR checkpoint. Caduceus's own ablation column bounds it
# at +0.004 (Ph) to +0.007 (PS) mean, and ours -- unlike theirs -- was not
# pretrained with RC augmentation, so the sign is not guaranteed.
PRESETS["v2-rc"] = dict(PRESETS["v2"], rc_aug=0.5, rc_tta=True)


def build_parser():
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--output-dir", type=Path,
                      default=REPO / "results" / "caduceus" / "genomic_benchmarks_ft")
  parser.add_argument("--tasks", default="all")
  parser.add_argument("--preset", choices=sorted(PRESETS), default="legacy",
                      help="bundle of defaults; individual flags still win")

  group = parser.add_argument_group("optimisation")
  group.add_argument("--epochs", type=int, default=4)
  group.add_argument("--batch-size", type=int, default=16)
  group.add_argument("--eval-batch-size", type=int, default=32)
  group.add_argument("--backbone-lr", type=float, default=1e-5)
  group.add_argument("--head-lr", type=float, default=1e-3)
  group.add_argument("--weight-decay", type=float, default=0.01)
  group.add_argument("--dropout", type=float, default=0.0,
                     help="override every nn.Dropout in the backbone; the "
                          "checkpoint trained at 0.0 and the modules are live")
  group.add_argument("--clip", type=float, default=1.0)
  group.add_argument("--clip-mode", choices=("global", "per-group"),
                     default="global")
  group.add_argument("--scheduler", choices=("none", "cosine"), default="none")
  group.add_argument("--warmup-frac", type=float, default=0.05)
  group.add_argument("--head-warmup-steps", type=int, default=0,
                     help="LP-FT: hold the backbone frozen for N steps first")
  group.add_argument("--honour-no-weight-decay", action="store_true",
                     default=None,
                     help="exclude dt_bias/A_log/D from weight decay")

  group = parser.add_argument_group("selection")
  group.add_argument("--seed", type=int, default=0,
                     help="base seed; also seeds the --max-train/--max-test "
                          "subsample, which is held FIXED across --seeds")
  group.add_argument("--seeds", default=None,
                     help="comma-separated, e.g. '0,1,2,3,4'; default: --seed")
  group.add_argument("--val-fraction", type=float, default=0.1)
  group.add_argument("--stratified-val", action="store_true", default=None)
  group.add_argument("--evals-per-epoch", type=int, default=0,
                     help="0 = validate once per epoch (old behaviour)")
  group.add_argument("--patience", type=int, default=0,
                     help="stop after N validations without improvement")
  group.add_argument("--sweep", default="",
                     help="grid over validation, e.g. "
                          "'backbone_lr=1e-5,3e-5,1e-4;head_lr=1e-3,3e-3'")
  group.add_argument("--sweep-seeds", type=int, default=1,
                     help="how many of --seeds the sweep uses")

  group = parser.add_argument_group("readout")
  group.add_argument("--pooling", default="mean",
                     choices=("mean", "max", "meanmax"))
  group.add_argument("--layer", type=int, default=-1,
                     help="1-indexed tap depth; -1 = last (with final_norm)")
  group.add_argument("--head-layernorm", action="store_true", default=None)
  group.add_argument("--log-length", action="store_true", default=None,
                     help="append log(len) to the pooled feature; mean pooling "
                          "is blind to length and length is a real class "
                          "signal on human_ensembl_regulatory")

  group = parser.add_argument_group("windowing")
  group.add_argument("--window", default="auto")
  group.add_argument("--window-cap", type=int, default=8192)
  group.add_argument("--window-from", choices=("full", "subset"),
                     default="subset",
                     help="'subset' reproduces the old behaviour, where the "
                          "window silently depended on --max-train")
  group.add_argument("--pad-to", choices=("task", "batch"), default="task")
  group.add_argument("--pad-multiple", type=int, default=8)
  group.add_argument("--pad-token", choices=("N", "PAD"), default="N",
                     help="'N' is the nucleotide the old harness padded with; "
                          "'PAD' is the true [PAD] id, which Caduceus uses but "
                          "which this checkpoint never saw in DNA pretraining, "
                          "so its embedding is still at init")
  group.add_argument("--pad-side", choices=("right", "left"), default="right")
  group.add_argument("--slow-encode", action="store_true",
                     help="per-nucleotide tokenizer lookups, as before")

  group = parser.add_argument_group("reverse complement")
  group.add_argument("--rc-tta", action="store_true", default=None,
                     help="score both strands and average; OFF by default "
                          "because this checkpoint saw no RC in pretraining")
  group.add_argument("--rc-average", choices=("prob", "logit"), default="prob")
  group.add_argument("--rc-aug", type=float, default=0.0,
                     help="probability of replacing a training example with "
                          "its reverse complement")

  group = parser.add_argument_group("data")
  group.add_argument("--max-train", type=int, default=None)
  group.add_argument("--max-test", type=int, default=None)
  return parser


def resolve(args):
  """Apply the preset under any flag the user did not pass explicitly."""
  explicit = set()
  for token in sys.argv[1:]:
    if token.startswith("--"):
      explicit.add(token.split("=")[0].lstrip("-").replace("-", "_"))
  for key, value in PRESETS[args.preset].items():
    if not hasattr(args, key):
      raise KeyError(f"preset {args.preset!r} sets unknown option {key!r}")
    if key not in explicit:
      setattr(args, key, value)
  for key in ("stratified_val", "head_layernorm", "log_length", "rc_tta",
              "honour_no_weight_decay"):
    if getattr(args, key) is None:
      setattr(args, key, False)
  args.seed_list = ([args.seed] if not args.seeds
                    else [int(s) for s in args.seeds.split(",") if s.strip()])
  return args


def main():
  parser = build_parser()
  args = resolve(parser.parse_args())

  if not torch.cuda.is_available():
    raise RuntimeError("fine-tuning needs a CUDA GPU")
  device = torch.device("cuda")

  raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  trained = OmegaConf.create(raw.get("hyper_parameters", {}).get("config", {}))
  del raw

  # Load ONCE and keep a pristine host copy of the backbone. The old code
  # re-read the checkpoint per task to avoid tasks inheriting each other's
  # adaptation; with a sweep and multiple seeds that would be dozens of reads.
  # Restoring the snapshot is identical -- EMA is already applied at load
  # (score_mavedb.py:85-86) and a state_dict copy is exact.
  model, tokenizer, config, step = load_checkpoint_model(
    args.checkpoint, int(trained.model.length), args.eval_batch_size, device)
  model.tokenizer = tokenizer
  # The SSM backbones set self.hidden_size; the DiT keeps its width in a local
  # (`dim = config.model.hidden_size`, dit.py:683) and never stores it. Resolve
  # it here rather than editing models/dit.py, which is on the live training
  # path. Not a state_dict entry, so the pristine snapshot below is unaffected.
  if not hasattr(model.backbone, "hidden_size"):
    model.backbone.hidden_size = int(trained.model.hidden_size)
  model.pristine = copy.deepcopy(model.backbone.state_dict())
  complement = build_complement_table(tokenizer, model.backbone.vocab_size)

  base_config = {k: getattr(args, k) for k in (
    "epochs", "batch_size", "eval_batch_size", "backbone_lr", "head_lr",
    "weight_decay", "dropout", "clip", "clip_mode", "scheduler", "warmup_frac",
    "head_warmup_steps", "honour_no_weight_decay", "pooling", "layer",
    "head_layernorm", "log_length", "pad_to", "stratified_val",
    "evals_per_epoch", "patience", "rc_tta", "rc_average", "rc_aug")}
  base_config["block_size"] = int(trained.block_size)

  wanted = ([t for t, _ in TASKS] if args.tasks == "all"
            else [t.strip() for t in args.tasks.split(",") if t.strip()])

  print(f"{args.label} | FINE-TUNE | preset {args.preset} | "
        f"seeds {args.seed_list} | epochs {args.epochs} | "
        f"batch {args.batch_size} | backbone_lr {args.backbone_lr} | "
        f"head_lr {args.head_lr} | pad_to {args.pad_to} | "
        f"rc_tta {args.rc_tta} | rc_aug {args.rc_aug} | "
        f"max_train {args.max_train} | max_test {args.max_test}")
  if args.sweep:
    print(f"sweep: {args.sweep} on {args.sweep_seeds} seed(s), "
          f"selected on validation, test untouched until the winner\n")
  else:
    print()
  header = (f"{'task':<34}{'ours':>8}{'+-':>7}{'Ph':>8}{'dPh':>8}{'PS':>8}"
            f"{'dPS':>8}{'win':>7}{'pad':>6}{'ntr':>8}{'min':>7}")
  print(header)
  print("-" * len(header))

  rows = []
  for name in wanted:
    started = time.time()
    row = run_task(name, model, args, base_config, device, complement)
    row["minutes"] = (time.time() - started) / 60
    print(f"{name:<34}{row['accuracy']:>8.4f}{row['accuracy_std']:>7.4f}"
          f"{row['caduceus_ph_published']:>8.3f}{row['delta_ph']:>+8.4f}"
          f"{row['caduceus_ps_published']:>8.3f}{row['delta_ps']:>+8.4f}"
          f"{row['window']:>7}{row['pad_fraction']:>6.2f}"
          f"{row['n_train_used']:>8}{row['minutes']:>7.1f}", flush=True)
    rows.append(row)

  mean = float(np.mean([r["accuracy"] for r in rows]))
  ph = float(np.mean([r["caduceus_ph_published"] for r in rows]))
  ps = float(np.mean([r["caduceus_ps_published"] for r in rows]))
  print("-" * len(header))
  print(f"{'MEAN':<34}{mean:>8.4f}{'':>7}{ph:>8.3f}{mean - ph:>+8.4f}"
        f"{ps:>8.3f}{mean - ps:>+8.4f}")

  summary = {
    "label": args.label, "checkpoint": str(args.checkpoint),
    "checkpoint_global_step": step, "backbone": str(config.algo.backbone),
    "pretraining_data": str(OmegaConf.select(trained, "data.train")),
    "protocol": "full fine-tune, Caduceus Table 1 protocol",
    "preset": args.preset,
    "args": {k: (str(v) if isinstance(v, Path) else v)
             for k, v in sorted(vars(args).items())},
    "mean_accuracy": mean,
    "caduceus_ph_mean_published": ph,
    "caduceus_ps_mean_published": ps,
    "caduceus_no_equiv_mean_published": float(np.mean(
      [r["caduceus_no_equiv_published"] for r in rows])),
    "delta_ph": mean - ph, "delta_ps": mean - ps,
    "note": (
      "Published values are transcribed from arXiv:2403.03234 Table 1, not "
      "reproduced here. The GenomicBenchmarks row of that table is a 470K-"
      "parameter, 1k-context model (App. D.1; the released wrapper loads "
      "caduceus-ph_seqlen-1k_d_model-118_n_layer-4), so we are comparing a "
      "~100.7M-parameter 8k-context model against one 214x smaller -- the "
      "capacity advantage is OURS and no gap is 'expected' from scale or "
      "context. Quote the comparison against Caduceus-PS (0.8690 mean), the "
      "stronger of their two variants, not Caduceus-Ph (0.8660). Their own "
      "'w/o equivariance' column (0.8618) is the same backbone class as ours, "
      "which bounds everything reverse-complement handling can buy on this "
      "suite at +0.004 (Ph) to +0.007 (PS)."),
    "test_split_policy": (
      "The test split is held behind HeldOutTest and evaluated exactly once "
      "per seed, from the restored best-validation weights. Sweeps select on "
      "validation only and assert zero test accesses; per-task "
      "'test_evaluations' records the count."),
    "tasks": rows,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  path = args.output_dir / f"{args.label}.json"
  with tempfile.NamedTemporaryFile("w", dir=args.output_dir, delete=False) as fh:
    json.dump(summary, fh, indent=2, sort_keys=True, default=str)
    fh.write("\n")
    tmp = Path(fh.name)
  os.replace(tmp, path)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
