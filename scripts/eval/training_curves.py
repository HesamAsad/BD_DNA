#!/usr/bin/env python3
"""Training and validation curves for every arm, from the CSV logs.

WHY A SCRIPT AND NOT A NOTEBOOK. The five hg38 arms each write
`csv_logs/version_*/metrics.csv` under their own output directory, at two
different cadences in the same file: `trainer/*` rows every
`log_every_n_steps` (50) and `val/*` rows every `VAL_EVERY` (500). Lightning
writes each row with only the columns that fired, so a naive read gives a frame
that is mostly NaN and a plot that is mostly gaps. Every series here is
extracted by dropping NaN on that column alone.

WHAT THE CURVES DO AND DO NOT LET YOU CONCLUDE. The AR arms report an exact
token NLL. The BD arms report a NELBO, which is an UPPER bound on NLL -- so a
BD arm plotted above an AR arm may be worse, or may be an equally good model
with a looser bound, and this figure cannot tell you which. The gap is not a
constant either: BD3-LM's own analysis has the bound loosening as the block
length approaches the sequence length. Read AR-vs-AR and BD-vs-BD comparisons
as measurements and AR-vs-BD as an upper bound only. The axis labels say so.

Three x axes, because "better" depends on what you are holding fixed:
  step      what you watch while it runs; fair only at equal batch and length.
  tokens    data efficiency -- the axis that matters for a corpus-bound claim.
  PFLOP     compute efficiency -- the axis that matters for a budget-bound one.
Both `trainer/total_gtokens` and `trainer/total_pflop` are already cumulative
and cluster-scaled in the log (diffusion.py:_log_train_telemetry).

Usage:
  python scripts/eval/training_curves.py \
      --run "uSSM-AR=outputs/hg38-caduceus/<dir>" \
      --run "Transformer-BD=outputs/hg38-caduceus/<other>" \
      --outdir results/figures
  python scripts/eval/training_curves.py --glob "outputs/hg38-caduceus/*"
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.provenance import write_json  # noqa: E402

# Same identities as the scaling and inference figures, so a reader moving
# between them tracks one arm by one colour throughout.
STYLE = {
  "Transformer-BD": ("#1f4e9c", "s"),
  "Transformer-AR": ("#5b9bd5", "^"),
  "BiSSM-BD":       ("#c2503f", "o"),
  "uSSM-BD":        ("#8c6d1f", "v"),
  "uSSM-AR":        ("#e0a62e", "D"),
}
# Which arms report a NELBO rather than an exact NLL.
BOUNDED = {"Transformer-BD", "BiSSM-BD", "uSSM-BD"}

# Job-name fragments -> the label used in every figure. Keys are matched
# against the run directory path, longest first so `hg_ussm_bd` is not eaten
# by `hg_ussm`.
INFER = [
  ("hg_bissm_bd", "BiSSM-BD"), ("bissm-bd", "BiSSM-BD"), ("bi-mamba2", "BiSSM-BD"),
  ("hg_ussm_ar", "uSSM-AR"), ("ussm-ar", "uSSM-AR"),
  ("hg_ussm_bd", "uSSM-BD"), ("ussm-bd", "uSSM-BD"),
  ("hg_xf_ar", "Transformer-AR"), ("transformer-ar", "Transformer-AR"),
  ("hg_xf_bd", "Transformer-BD"), ("transformer-bd", "Transformer-BD"), ("xf-bd", "Transformer-BD"),
]


def infer_label(path: str) -> str:
  lowered = path.lower()
  for fragment, label in sorted(INFER, key=lambda kv: -len(kv[0])):
    if fragment in lowered:
      return label
  return os.path.basename(path.rstrip("/"))


def load_run(run_dir: Path) -> pd.DataFrame | None:
  """Concatenate every version_* under a run, ordered by step.

  A resumed run writes version_1 alongside version_0 rather than appending, so
  reading only version_0 silently truncates the curve at the crash point --
  which is exactly where a curve gets interesting. Duplicate steps (the tail of
  version_0 replayed by the resume) are dropped, keeping the later row.
  """
  paths = sorted(globmod.glob(str(run_dir / "csv_logs" / "version_*" / "metrics.csv")))
  if not paths:
    paths = sorted(globmod.glob(str(run_dir / "**" / "metrics.csv"), recursive=True))
  if not paths:
    return None
  frames = []
  for order, path in enumerate(paths):
    try:
      frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
      continue
    if "step" not in frame.columns or not len(frame):
      continue
    frame["_version"] = order
    frames.append(frame)
  if not frames:
    return None
  merged = pd.concat(frames, ignore_index=True)
  merged = merged.sort_values(["step", "_version"])
  return merged


def series(frame: pd.DataFrame, y: str, x: str = "step"):
  """The (x, y) pairs where y actually fired.

  Lightning writes one row per logging event with every other column NaN, so
  `dropna()` across the frame would return nothing at all. Dropping on the two
  columns in play is what makes a curve out of a sparse file. `x` is
  forward-filled first: a val row carries no `total_pflop`, but the compute
  spent by that step is the last training row's value.
  """
  if y not in frame.columns or x not in frame.columns:
    return None, None
  work = frame[[x, y]].copy()
  work[x] = work[x].ffill()
  work = work.dropna(subset=[x, y])
  if not len(work):
    return None, None
  work = work.drop_duplicates(subset=[x], keep="last").sort_values(x)
  return work[x].to_numpy(), work[y].to_numpy()


def smooth(values, window: int):
  if window <= 1 or len(values) < window:
    return values
  return pd.Series(values).rolling(window, min_periods=1, center=False).mean().to_numpy()


def panel(ax, runs, y, x, ylabel, title, logy=False, window=1, marker=True):
  drawn = 0
  for label, frame in runs.items():
    xs, ys = series(frame, y, x)
    if xs is None:
      continue
    colour, mark = STYLE.get(label, ("#555555", "o"))
    ys = smooth(ys, window)
    ax.plot(xs, ys, color=colour, linewidth=1.7,
            marker=mark if marker and len(xs) <= 60 else None,
            markersize=4.5,
            label=label + ("  (NELBO)" if label in BOUNDED else ""))
    drawn += 1
  ax.set_xlabel({"step": "Training step",
                 "trainer/total_gtokens": "Tokens seen (billions)",
                 "trainer/total_pflop": "Compute (PFLOP)"}.get(x, x))
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=10.5)
  if logy:
    ax.set_yscale("log")
  if x != "step":
    ax.set_xscale("log")
  ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
  if drawn:
    ax.legend(frameon=False, fontsize=8)
  return drawn


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--run", action="append", default=[],
                      help="LABEL=path/to/run_dir; repeatable")
  parser.add_argument("--glob", default=None,
                      help="glob of run dirs; labels inferred from the path")
  parser.add_argument("--smooth", type=int, default=20,
                      help="rolling window for the per-step train curves; the "
                           "validation curves are never smoothed")
  parser.add_argument("--outdir", type=Path, default=REPO / "results" / "figures")
  args = parser.parse_args()

  specs = []
  for item in args.run:
    if "=" not in item:
      sys.exit(f"--run needs LABEL=path, got {item!r}")
    label, path = item.split("=", 1)
    specs.append((label, Path(path)))
  if args.glob:
    for path in sorted(globmod.glob(args.glob)):
      if os.path.isdir(path):
        specs.append((infer_label(path), Path(path)))
  if not specs:
    sys.exit("nothing to plot: pass --run or --glob")

  runs, missing = {}, []
  for label, path in specs:
    frame = load_run(path if path.is_absolute() else REPO / path)
    if frame is None:
      missing.append((label, str(path)))
      continue
    # A label seen twice (two seeds, or a rerun) would silently overwrite.
    key, n = label, 2
    while key in runs:
      key, n = f"{label} #{n}", n + 1
    runs[key] = frame
  for label, path in missing:
    print(f"  no metrics.csv under {path}  ({label}) -- skipped", file=sys.stderr)
  if not runs:
    sys.exit("every run directory was empty of metrics.csv")

  args.outdir.mkdir(parents=True, exist_ok=True)
  written = []

  # ---- validation: the headline comparison, on all three x axes ------------
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=160)
  panel(axes[0], runs, "val/nll", "step",
        "Validation NLL (nats/token)", "Against steps")
  panel(axes[1], runs, "val/nll", "trainer/total_gtokens",
        "Validation NLL (nats/token)", "Against data")
  panel(axes[2], runs, "val/nll", "trainer/total_pflop",
        "Validation NLL (nats/token)", "Against compute")
  fig.suptitle("Validation NLL — AR arms report an exact NLL, BD arms a NELBO "
               "(an upper bound); AR-vs-BD gaps are bounds, not measurements",
               fontsize=11)
  fig.tight_layout()
  path = args.outdir / "training_val_nll.png"
  fig.savefig(path); plt.close(fig); written.append(path)

  # ---- perplexity and bits/base, the two rescalings people ask for ---------
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
  panel(axes[0], runs, "val/ppl", "step", "Validation perplexity",
        "Perplexity (exp of the NLL above)", logy=True)
  panel(axes[1], runs, "val/bpd", "step", "Validation bits per base",
        "Bits per base — uniform ACGT is 2.0")
  fig.tight_layout()
  path = args.outdir / "training_val_ppl.png"
  fig.savefig(path); plt.close(fig); written.append(path)

  # ---- train side: the curve that was not being logged at all until now ----
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
  drew = panel(axes[0], runs, "trainer/train_nll", "step",
               "Train NLL (nats/token)",
               f"Train NLL, rolling mean over {args.smooth} points",
               window=args.smooth, marker=False)
  if not drew:
    # Runs started before the train_nll logging landed still have the loss,
    # which is the same quantity under a different name.
    panel(axes[0], runs, "trainer/loss", "step", "Train loss (nats/token)",
          f"Train loss, rolling mean over {args.smooth} points",
          window=args.smooth, marker=False)
  panel(axes[1], runs, "trainer/tokens_per_s", "step", "Tokens / second",
        "Training throughput", window=args.smooth, marker=False)
  fig.tight_layout()
  path = args.outdir / "training_train_nll.png"
  fig.savefig(path); plt.close(fig); written.append(path)

  # ---- the tidy dump, so a plot never has to be re-derived from logs -------
  tidy = {}
  for label, frame in runs.items():
    entry = {}
    for column in ("val/nll", "val/ppl", "val/bpd", "trainer/train_nll",
                   "trainer/train_ppl", "trainer/loss", "trainer/train_bpb",
                   "trainer/tokens_per_s", "trainer/total_pflop",
                   "trainer/total_gtokens"):
      xs, ys = series(frame, column, "step")
      if xs is None:
        continue
      entry[column] = {"step": [int(v) for v in xs],
                       "value": [float(v) for v in ys]}
    best = entry.get("val/nll")
    if best:
      index = min(range(len(best["value"])), key=lambda i: best["value"][i])
      entry["best"] = {"val/nll": best["value"][index],
                       "step": best["step"][index],
                       "is_nelbo_upper_bound": label.split(" #")[0] in BOUNDED}
    tidy[label] = entry
  path = args.outdir / "training_curves.json"
  write_json(path, {"runs": tidy,
                    "note": "AR arms report an exact NLL; BD arms report a "
                            "NELBO, an upper bound. is_nelbo_upper_bound says "
                            "which is which."}, args)
  written.append(path)

  print()
  for label in runs:
    best = tidy[label].get("best")
    if best:
      flag = "  (NELBO upper bound)" if best["is_nelbo_upper_bound"] else ""
      print(f"  {label:<18} best val/nll {best['val/nll']:.4f} "
            f"@ step {best['step']:,}{flag}")
    else:
      print(f"  {label:<18} no val/nll logged yet")
  print()
  for item in written:
    print(f"wrote {item}")


if __name__ == "__main__":
  main()
