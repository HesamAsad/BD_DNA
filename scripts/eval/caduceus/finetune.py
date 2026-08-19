#!/usr/bin/env python3
"""GenomicBenchmarks by FINE-TUNING, matching how Caduceus reports the suite.

The companion `genomic_benchmarks.py` fits a logistic regression on frozen
features. That answers "are the representations good?" but it is not the
protocol Caduceus uses, so the published numbers are not a like-for-like
comparison against it. This script removes that confound: the whole backbone is
unfrozen and trained end-to-end with a classification head, which is what
arXiv:2403.03234 does for its Table 3.

Design choices, and why:

* **Mean pooling over unpadded positions**, the same readout the probe used, so
  the only thing that changes between the two scripts is whether the backbone
  moves. Any difference is attributable to fine-tuning and nothing else.
* **Two parameter groups.** The head is new and needs a large learning rate;
  the backbone is pretrained and needs a small one, or fine-tuning destroys it.
  A single shared rate is the usual way this comparison gets botched.
* **Model selection on a held-out slice of train**, never on test. The reported
  number is test accuracy at the best validation epoch, so a task that overfits
  early is not rewarded for it.
* **Both caches empty**, as in the probe: a benchmark sequence has no prefix and
  no clean suffix. The reverse scan still runs within the sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.caduceus.embed import pool  # noqa: E402
from scripts.eval.caduceus.genomic_benchmarks import (  # noqa: E402
  TASKS, PUBLISHED, load_task)
from scripts.eval.dnahnet.score_mavedb import (  # noqa: E402
  load_checkpoint_model, encode_dna)

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)


class Classifier(nn.Module):
  """Pretrained backbone + mean pool + linear head."""

  def __init__(self, backbone, hidden, num_classes, pooling="mean"):
    super().__init__()
    self.backbone = backbone
    self.pooling = pooling
    width = hidden * (2 if pooling == "meanmax" else 1)
    self.head = nn.Linear(width, num_classes)

  def forward(self, ids, attention_mask):
    b = self.backbone
    batch = ids.shape[0]
    h = b.token_embedding(ids)
    left = b._empty_cache(batch, h.device, h.dtype, "left")
    right = b._empty_cache(batch, h.device, h.dtype, "right")
    with b._compute_autocast(h):
      for index, layer in enumerate(b.layers):
        h = layer.scan_active(h, left.states[index], right.states[index])
      h = b.final_norm(h)
    return self.head(pool(h.float(), attention_mask, self.pooling))


def encode_batch(tokenizer, sequences, length, device):
  rows, masks = [], []
  for sequence in sequences:
    ids, keep = encode_dna(tokenizer, sequence, length)
    rows.append(ids)
    masks.append(keep)
  return (torch.tensor(rows, dtype=torch.long, device=device),
          torch.tensor(masks, dtype=torch.bool, device=device))


def evaluate(model, tokenizer, sequences, labels, length, batch_size, device):
  model.eval()
  correct = 0
  with torch.inference_mode():
    for start in range(0, len(sequences), batch_size):
      ids, keep = encode_batch(
        tokenizer, sequences[start:start + batch_size], length, device)
      logits = model(ids, keep)
      target = torch.tensor(
        labels[start:start + batch_size], device=device)
      correct += int((logits.argmax(dim=-1) == target).sum())
  return correct / len(sequences)


def finetune_task(model, tokenizer, data, length, args, device):
  xtr, ytr, xte, yte = data
  rng = np.random.default_rng(args.seed)
  order = rng.permutation(len(xtr))
  cut = max(1, int(0.1 * len(xtr)))
  val_idx, tr_idx = order[:cut], order[cut:]
  xva = [xtr[i] for i in val_idx]
  yva = ytr[val_idx]
  xtr = [xtr[i] for i in tr_idx]
  ytr = ytr[tr_idx]

  num_classes = int(max(ytr.max(), yte.max())) + 1
  classifier = Classifier(
    model.backbone, int(model.backbone.hidden_size), num_classes,
    args.pooling).to(device)
  # The head is random and the backbone is pretrained; one shared learning rate
  # either leaves the head untrained or wrecks the backbone.
  optimizer = torch.optim.AdamW([
    {"params": classifier.backbone.parameters(), "lr": args.backbone_lr},
    {"params": classifier.head.parameters(), "lr": args.head_lr},
  ], weight_decay=args.weight_decay)
  loss_fn = nn.CrossEntropyLoss()

  best_val, best_test, best_epoch = -1.0, 0.0, -1
  for epoch in range(args.epochs):
    classifier.train()
    perm = rng.permutation(len(xtr))
    for start in range(0, len(perm), args.batch_size):
      chunk = perm[start:start + args.batch_size]
      ids, keep = encode_batch(
        tokenizer, [xtr[i] for i in chunk], length, device)
      target = torch.tensor(ytr[chunk], device=device)
      optimizer.zero_grad(set_to_none=True)
      loss = loss_fn(classifier(ids, keep), target)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
      optimizer.step()
    val = evaluate(classifier, tokenizer, xva, yva, length,
                   args.eval_batch_size, device)
    if val > best_val:
      best_val = val
      best_test = evaluate(classifier, tokenizer, xte, yte, length,
                           args.eval_batch_size, device)
      best_epoch = epoch
  return best_test, best_val, best_epoch, num_classes


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--output-dir", type=Path,
                      default=REPO / "results" / "caduceus" / "genomic_benchmarks_ft")
  parser.add_argument("--tasks", default="all")
  parser.add_argument("--pooling", default="mean")
  parser.add_argument("--window", default="auto")
  parser.add_argument("--window-cap", type=int, default=8192)
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--eval-batch-size", type=int, default=32)
  parser.add_argument("--backbone-lr", type=float, default=1e-5)
  parser.add_argument("--head-lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=0.01)
  parser.add_argument("--max-train", type=int, default=None)
  parser.add_argument("--max-test", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("fine-tuning needs a CUDA GPU")
  device = torch.device("cuda")

  raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  trained = OmegaConf.create(raw.get("hyper_parameters", {}).get("config", {}))
  del raw
  block = int(trained.block_size)
  fixed_window = None if str(args.window) == "auto" else int(args.window)

  wanted = ([t for t, _ in TASKS] if args.tasks == "all"
            else [t.strip() for t in args.tasks.split(",") if t.strip()])

  print(f"{args.label} | FINE-TUNE | epochs {args.epochs} | "
        f"backbone_lr {args.backbone_lr} | head_lr {args.head_lr}\n")
  print(f"{'task':<34}{'ours':>8}{'Caduceus-Ph':>13}{'delta':>9}{'win':>7}"
        f"{'ep':>4}{'min':>7}")
  print("-" * 82)

  rows = []
  for name in wanted:
    started = time.time()
    data = load_task(name, args.max_train, args.max_test, args.seed)
    longest = max(max(len(s) for s in data[0]), max(len(s) for s in data[2]))
    window = fixed_window or min(-(-longest // block) * block, args.window_cap)
    # A fresh backbone per task: fine-tuning mutates the weights, so tasks must
    # not inherit each other's adaptation.
    model, tokenizer, config, step = load_checkpoint_model(
      args.checkpoint, int(trained.model.length), args.batch_size, device)
    accuracy, val, epoch, classes = finetune_task(
      model, tokenizer, data, window, args, device)
    reference = PUBLISHED.get(name)
    minutes = (time.time() - started) / 60
    print(f"{name:<34}{accuracy:>8.4f}{reference:>13.4f}"
          f"{accuracy - reference:>+9.4f}{window:>7}{epoch:>4}{minutes:>7.1f}")
    rows.append({"task": name, "accuracy": accuracy, "val_accuracy": val,
                 "best_epoch": epoch, "window": window, "num_classes": classes,
                 "caduceus_ph_published": reference,
                 "delta": accuracy - reference, "minutes": minutes})
    del model
    torch.cuda.empty_cache()

  mean = float(np.mean([r["accuracy"] for r in rows]))
  ref = float(np.mean([r["caduceus_ph_published"] for r in rows]))
  print("-" * 82)
  print(f"{'MEAN':<34}{mean:>8.4f}{ref:>13.4f}{mean - ref:>+9.4f}")

  summary = {
    "label": args.label, "checkpoint": str(args.checkpoint),
    "checkpoint_global_step": step, "backbone": str(config.algo.backbone),
    "pretraining_data": str(OmegaConf.select(trained, "data.train")),
    "protocol": "full fine-tune, matching Caduceus Table 3",
    "epochs": args.epochs, "backbone_lr": args.backbone_lr,
    "head_lr": args.head_lr, "pooling": args.pooling,
    "mean_accuracy": mean, "caduceus_ph_mean_published": ref,
    "note": ("Published values are transcribed from arXiv:2403.03234, not "
             "reproduced here. Caduceus is 131k-context and RC-equivariant; "
             "this is 8k-context and not RC-equivariant, so a residual gap is "
             "expected even with the protocol matched."),
    "tasks": rows,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  path = args.output_dir / f"{args.label}.json"
  with tempfile.NamedTemporaryFile("w", dir=args.output_dir, delete=False) as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
    tmp = Path(fh.name)
  os.replace(tmp, path)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
