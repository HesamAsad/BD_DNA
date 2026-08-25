#!/usr/bin/env python3
"""GenomicBenchmarks, the way Caduceus reports it -- but as a linear probe.

Caduceus (arXiv:2403.03234, Table 3) fine-tunes the whole backbone on each of
the 8 GenomicBenchmarks classification tasks. We run a **frozen linear probe**
instead, for a specific reason: the question is whether block-diffusion
pretraining produces competitive *representations*. Fine-tuning conflates the
quality of the representation with the model's capacity to adapt away from it,
so a probe is the sharper instrument and it is far cheaper. It also biases
AGAINST us, since the published numbers come from full fine-tuning -- read a
probe result as a floor, not a like-for-like score.

Reported metric is top-1 accuracy on the official test split, which is what
Caduceus reports for this suite.

Published Caduceus-Ph accuracies are carried in PUBLISHED below so every run
prints our number beside theirs. They are reference values transcribed from the
paper, not something this script computes -- treat them as a target, and check
the paper before quoting them anywhere.

Usage:
  python scripts/eval/caduceus/genomic_benchmarks.py \
      --checkpoint outputs/.../0-8000.ckpt --label bissm-prok
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.provenance import (  # noqa: E402
  assert_full_coverage, provenance)
from scripts.eval.caduceus.embed import embed_sequences  # noqa: E402
from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model  # noqa: E402

for _name, _resolver in (("cwd", os.getcwd),
                         ("device_count", torch.cuda.device_count),
                         ("eval", eval),
                         ("div_up", lambda x, y: (x + y - 1) // y)):
  OmegaConf.register_new_resolver(_name, _resolver, replace=True)

HF_PREFIX = "katarinagresova/Genomic_Benchmarks_"

# The 8 tasks Caduceus reports, with its published Caduceus-Ph top-1 accuracy.
# Transcribed from the paper for side-by-side printing; verify before quoting.
TASKS = [
  ("dummy_mouse_enhancers_ensembl", 0.754),
  ("demo_coding_vs_intergenomic_seqs", 0.915),
  ("demo_human_or_worm", 0.973),
  ("human_enhancers_cohn", 0.747),
  ("human_enhancers_ensembl", 0.893),
  ("human_ensembl_regulatory", 0.872),
  ("human_nontata_promoters", 0.946),
  ("human_ocr_ensembl", 0.828),
]
PUBLISHED = dict(TASKS)

# The full Table 1 of arXiv:2403.03234, every column. Kept here so a run can
# print itself against the strongest published variant rather than the softest.
#
# Caduceus-PS, not Caduceus-Ph, is the higher 8-task mean (0.8690 vs 0.8660), so
# PS is the bar to quote. The `no_equiv` column is the same RC-augmented BiMamba
# checkpoint evaluated WITHOUT reverse-complement conjoining -- i.e. our own
# architecture class -- and it already scores 0.8618. That bounds everything RC
# can buy on this suite at +0.0042 (Ph) / +0.0073 (PS).
#
# Every column here is 470K parameters at 1k context (paper App. D.1 and the
# released `wrapper_run_genomics.sh`, which loads
# `caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3`). Any claim that we are
# handicapped by context length or scale against this table is backwards.
REFERENCE_COLUMNS = ("cnn", "hyenadna", "mamba", "no_equiv", "ph", "ps")
REFERENCE = {
  # task:                             cnn    hyena  mamba  noeq   ph     ps
  "dummy_mouse_enhancers_ensembl":   (0.715, 0.780, 0.743, 0.770, 0.754, 0.793),
  "demo_coding_vs_intergenomic_seqs": (0.892, 0.904, 0.904, 0.908, 0.915, 0.910),
  "demo_human_or_worm":              (0.942, 0.964, 0.967, 0.970, 0.973, 0.968),
  "human_enhancers_cohn":            (0.702, 0.729, 0.732, 0.741, 0.747, 0.745),
  "human_enhancers_ensembl":         (0.744, 0.849, 0.862, 0.883, 0.893, 0.900),
  "human_ensembl_regulatory":        (0.872, 0.869, 0.814, 0.871, 0.872, 0.873),
  "human_nontata_promoters":         (0.861, 0.944, 0.933, 0.933, 0.946, 0.945),
  "human_ocr_ensembl":               (0.698, 0.783, 0.815, 0.818, 0.828, 0.818),
}


def reference(name, column):
  """Published accuracy for `name` under one Table 1 column, or None."""
  row = REFERENCE.get(name)
  return None if row is None else row[REFERENCE_COLUMNS.index(column)]


def _open_task(name):
  from datasets import load_dataset
  data = load_dataset(HF_PREFIX + name)
  splits = set(data.keys())
  test_key = "test" if "test" in splits else sorted(splits - {"train"})[0]
  return data, "train", test_key


def _sequence_column(rows):
  return "seq" if "seq" in rows.column_names else "sequence"


def task_stats(name):
  """Shape of the FULL official splits, before any --max-train/--max-test cap.

  Two uses. (1) The fine-tune harness sizes its window from the full data, so
  the window stops silently depending on the cap. (2) `num_classes` comes from
  the TRAIN labels only -- the old code read `yte.max()` to size the head, which
  is harmless but means the test split is touched before training, and the point
  of the rewrite is that the test split is touchable exactly once.
  """
  data, train_key, test_key = _open_task(name)
  train, test = data[train_key], data[test_key]
  train_lengths = np.asarray([len(s) for s in train[_sequence_column(train)]])
  test_lengths = np.asarray([len(s) for s in test[_sequence_column(test)]])
  labels = np.asarray(train["label"])
  return {
    "n_train_full": len(train), "n_test_full": len(test),
    # The window has to cover the test split too or encoding raises, so this is
    # the one number that reads the test side -- a shape constraint, not a
    # measurement. Everything a model could be selected on comes from train.
    "max_length": int(max(train_lengths.max(), test_lengths.max())),
    "median_length": float(np.median(train_lengths)),
    "mean_length": float(train_lengths.mean()),
    "num_classes": int(labels.max()) + 1,
  }


def load_task(name, max_train=None, max_test=None, seed=0):
  """Load one task. Returns the rows AND how many were available.

  The sizes are returned rather than discarded because every published probe
  result before 2026-08-25 was capped at 20000/8000 and the output recorded
  only the USED count, with nothing to compare it against. See
  scripts/eval/provenance.py.
  """
  data, train_key, test_key = _open_task(name)

  def take(split, cap):
    rows = data[split]
    available = len(rows)
    if cap and len(rows) > cap:
      rows = rows.shuffle(seed=seed).select(range(cap))
    return (list(rows[_sequence_column(rows)]), np.asarray(rows["label"]),
            available)

  xtr, ytr, n_train_full = take(train_key, max_train)
  xte, yte, n_test_full = take(test_key, max_test)
  assert_full_coverage(len(xtr), n_train_full, f"{name} train rows",
                       allow=max_train is not None)
  assert_full_coverage(len(xte), n_test_full, f"{name} test rows",
                       allow=max_test is not None)
  return xtr, ytr, xte, yte, n_train_full, n_test_full


def probe(train_x, train_y, test_x, test_y, seed=0):
  """Standardise, then logistic regression with C chosen on a held-out split."""
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split

  scaler = StandardScaler().fit(train_x)
  train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
  inner_tr, inner_va, inner_ytr, inner_yva = train_test_split(
    train_x, train_y, test_size=0.2, random_state=seed, stratify=train_y)

  best, best_c = -1.0, 1.0
  for c in (0.001, 0.01, 0.1, 1.0, 10.0):
    model = LogisticRegression(C=c, max_iter=2000, n_jobs=-1)
    model.fit(inner_tr, inner_ytr)
    score = model.score(inner_va, inner_yva)
    if score > best:
      best, best_c = score, c
  final = LogisticRegression(C=best_c, max_iter=2000, n_jobs=-1)
  final.fit(train_x, train_y)
  return float(final.score(test_x, test_y)), best_c


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--output-dir", type=Path,
                      default=REPO / "results" / "caduceus" / "genomic_benchmarks")
  parser.add_argument("--tasks", default="all",
                      help="comma-separated task names, or 'all'")
  parser.add_argument("--pooling", default="mean")
  parser.add_argument(
    "--window", default="auto",
    help="embedding window in nt: an integer, or 'auto' to size it per task "
         "from the data. 'auto' is the default because 5 of the 8 tasks carry "
         "sequences longer than the 256-nt block size, and silently truncating "
         "them would understate every one of those scores.")
  parser.add_argument(
    "--window-cap", type=int, default=8192,
    help="upper bound for 'auto'; the checkpoints were trained at L=8192.")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--max-train", type=int, default=None,
                      help="cap training sequences per task (speed)")
  parser.add_argument("--max-test", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("needs a CUDA GPU")
  device = torch.device("cuda")

  raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  trained = OmegaConf.create(raw.get("hyper_parameters", {}).get("config", {}))
  del raw
  fixed_window = None if str(args.window) == "auto" else int(args.window)
  model, tokenizer, config, step = load_checkpoint_model(
    args.checkpoint, int(trained.model.length), args.batch_size, device)

  wanted = ([t for t, _ in TASKS] if args.tasks == "all"
            else [t.strip() for t in args.tasks.split(",") if t.strip()])

  print(f"{args.label} | backbone {config.algo.backbone} | step {step} | "
        f"window {args.window} | pooling {args.pooling} | LINEAR PROBE\n")
  print(f"{'task':<34}{'ours':>8}{'Caduceus-Ph':>13}{'delta':>9}{'win':>7}{'n_tr':>8}")
  print("-" * 79)

  rows = []
  for name in wanted:
    try:
      xtr, ytr, xte, yte, n_train_full, n_test_full = load_task(
        name, args.max_train, args.max_test, args.seed)
    except Exception as exc:  # noqa: BLE001
      print(f"{name:<36}{'LOAD FAILED':>8}   {type(exc).__name__}: {exc}"[:110])
      rows.append({"task": name, "error": f"{type(exc).__name__}: {exc}"})
      continue
    if fixed_window is not None:
      window = fixed_window
    else:
      # Cover the longest sequence in the task, rounded up to a multiple of the
      # block size so the window is a whole number of trained blocks.
      block = int(trained.block_size)
      longest = max(max(len(s) for s in xtr), max(len(s) for s in xte))
      window = min(-(-longest // block) * block, args.window_cap)
    etr = embed_sequences(model, tokenizer, xtr, window, args.pooling,
                          args.batch_size, 0.0, args.seed, device)
    ete = embed_sequences(model, tokenizer, xte, window, args.pooling,
                          args.batch_size, 0.0, args.seed, device)
    accuracy, c = probe(etr, ytr, ete, yte, args.seed)
    reference = PUBLISHED.get(name)
    delta = accuracy - reference if reference else float("nan")
    print(f"{name:<34}{accuracy:>8.4f}{reference:>13.4f}{delta:>+9.4f}"
          f"{window:>7}{len(xtr):>8}")
    rows.append({"task": name, "accuracy": accuracy, "C": c, "window": window,
                 "caduceus_ph_published": reference, "delta": delta,
                 "n_train": len(xtr), "n_test": len(xte),
                 # The full split sizes, so a cap is visible in the output
                 # rather than only inferable from suspiciously round numbers.
                 "n_train_full": n_train_full, "n_test_full": n_test_full,
                 "train_fraction": len(xtr) / max(n_train_full, 1),
                 "dim": int(etr.shape[1])})

  scored = [r for r in rows if "accuracy" in r]
  mean = float(np.mean([r["accuracy"] for r in scored])) if scored else float("nan")
  ref_mean = float(np.mean([r["caduceus_ph_published"] for r in scored
                            if r["caduceus_ph_published"]])) if scored else float("nan")
  print("-" * 79)
  print(f"{'MEAN':<34}{mean:>8.4f}{ref_mean:>13.4f}{mean - ref_mean:>+9.4f}")

  summary = {
    "label": args.label, "checkpoint": str(args.checkpoint),
    "checkpoint_global_step": step, "backbone": str(config.algo.backbone),
    "pretraining_data": str(OmegaConf.select(trained, "data.train")),
    "window": str(args.window), "pooling": args.pooling,
    "protocol": "frozen linear probe",
    "note": ("Caduceus fine-tunes the whole backbone; this is a frozen probe, "
             "so it is a floor rather than a like-for-like comparison. "
             "Published values are transcribed from the paper."),
    "mean_accuracy": mean, "caduceus_ph_mean_published": ref_mean,
    "tasks": rows,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  path = args.output_dir / f"{args.label}.json"
  with tempfile.NamedTemporaryFile("w", dir=args.output_dir, delete=False) as fh:
    summary["_provenance"] = provenance(args)
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
    tmp = Path(fh.name)
  os.replace(tmp, path)
  print(f"\nwrote {path}")


if __name__ == "__main__":
  main()
