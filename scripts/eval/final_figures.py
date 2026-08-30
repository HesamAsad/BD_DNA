#!/usr/bin/env python3
"""Headline figures: a GenomicBenchmarks radar, and the likelihood metrics.

TWO FIGURES, and one caveat each that has to travel with them.

RADAR. Eight axes, one per GenomicBenchmarks task, five arms plus the two
Caduceus references. A radar makes per-task shape visible in a way a mean
hides -- and this suite badly needs that, because one axis is degenerate:
`human_ensembl_regulatory` is ~90% solvable from len(seq) alone
(scripts/data/audit_hg38_corpus.py's sibling analysis; a depth-6 decision tree
on sequence length scores 0.9004 against a 0.3704 majority), which beats every
published number on it. That axis is drawn with a dashed guide at the
length-only baseline so nobody reads a lead on it as a modelling result.

Axes are scaled per task rather than 0-1: the tasks span 0.68 to 0.97, so a
common radius would compress every real difference into the outer ring. Each
axis runs from a little below the worst score to a little above the best, and
the tick labels state the range, so shape is comparable but absolute position
is not over-read.

LIKELIHOOD. Validation NLL, perplexity and bits/nt. The AR arms report an
EXACT token NLL; the BD arms report a NELBO, an UPPER bound. Those are
different quantities and the bars are hatched accordingly -- a BD bar sitting
above an AR bar may be a worse model or an equally good one with a looser
bound, and this figure cannot distinguish them. Bits/nt has a reference line at
2.0, the uniform-ACGT value.

Usage:
  python scripts/eval/final_figures.py --outdir results/figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ARMS = [
  ("ussm_ar", "hg_ussm_ar", "uSSM-AR", "#e0a62e", False),
  ("xf_ar", "hg_xf_ar", "Transformer-AR", "#5b9bd5", False),
  ("ussm_bd", "hg_ussm_bd", "uSSM-BD", "#8c6d1f", True),
  ("bissm_bd", "hg_bissm_bd", "BiSSM-BD", "#c2503f", True),
  ("xf_bd", "hg_xf_bd", "Transformer-BD", "#1f4e9c", True),
]
TASKS = [
  ("dummy_mouse_enhancers_ensembl", "dummy_mouse\nenhancers", None),
  ("demo_coding_vs_intergenomic_seqs", "coding vs\nintergenomic", None),
  ("demo_human_or_worm", "human or\nworm", None),
  ("human_enhancers_cohn", "enhancers\ncohn", None),
  ("human_enhancers_ensembl", "enhancers\nensembl", None),
  # the length-only decision-tree baseline for the degenerate task
  ("human_ensembl_regulatory", "ensembl\nregulatory", 0.9004),
  ("human_nontata_promoters", "non-TATA\npromoters", None),
  ("human_ocr_ensembl", "OCR\nensembl", None),
]
FT_LR = 3e-5   # only results at the corrected fine-tuning LR count


def load_benchmarks(directory: Path):
  """(arm, task) -> accuracy, restricted to the corrected fine-tuning LR.

  Results at the old 1e-5 default are still on disk and are NOT comparable --
  that value left the backbone under-adapted and cost ~0.037 on the task where
  it was measured. Mixing the two would confound architecture with recipe.
  """
  out, published = {}, {}
  for path in glob.glob(str(directory / "*.json")):
    try:
      payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
      continue
    label = payload.get("label", "")
    for task in payload.get("tasks", []):
      config = task.get("config") or {}
      lr = config.get("backbone_lr", (payload.get("args") or {}).get("backbone_lr"))
      if lr is None or abs(float(lr) - FT_LR) > 1e-12:
        continue
      name = task.get("task")
      arm = next((k for k, *_ in ARMS
                  if label.startswith(f"lr35_{k}_") or label.startswith(f"hg38_{k}_")
                  or label.startswith(f"ocrfix_fix_{k}")), None)
      if arm and name:
        out[(arm, name)] = task["accuracy"]
        published[name] = (task.get("caduceus_ph_published"),
                           task.get("caduceus_ps_published"))
  return out, published


def best_val_nll(run_dir: Path):
  best = None
  for path in sorted(glob.glob(str(run_dir / "csv_logs" / "version_*" / "metrics.csv"))):
    try:
      for row in csv.DictReader(open(path)):
        value, step = row.get("val/nll"), row.get("step", "")
        if value and step.isdigit():
          try:
            value = float(value)
          except ValueError:
            continue
          if best is None or value < best[0]:
            best = (value, int(step))
    except OSError:
      continue
  return best


def load_full_split(pattern: str):
  """arm -> exact NLL over the whole validation split, if that job has run."""
  rows = {}
  for path in sorted(glob.glob(pattern)):
    with open(path) as handle:
      for row in csv.DictReader(handle, delimiter="\t"):
        try:
          rows[row["arm"]] = float(row["val_nll_nats"])
        except (KeyError, TypeError, ValueError):
          continue
  return rows


def radar(scores, published, path):
  labels = [d for _, d, _ in TASKS]
  n = len(TASKS)
  angles = [i / n * 2 * math.pi for i in range(n)] + [0.0]

  # Per-axis scaling. The suite spans 0.68-0.97; a shared radius would push
  # every arm into the same outer ring and hide the differences the figure
  # exists to show.
  lo, hi = [], []
  for key, _, _ in TASKS:
    vals = [scores[(a, key)] for a, *_ in ARMS if (a, key) in scores]
    vals += [v for v in published.get(key, ()) if v]
    span = max(vals) - min(vals)
    pad = max(span * 0.35, 0.02)
    lo.append(min(vals) - pad)
    hi.append(min(1.0, max(vals) + pad * 0.6))

  def norm(values):
    return [(v - lo[i]) / (hi[i] - lo[i]) for i, v in enumerate(values)]

  fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=160,
                         subplot_kw={"projection": "polar"})
  ax.set_theta_offset(math.pi / 2)
  ax.set_theta_direction(-1)

  for key, colour, style, width, alpha in [
      ("ph", "#7a7a7a", (0, (4, 2)), 1.6, 0.85),
      ("ps", "#b0b0b0", (0, (1, 2)), 1.6, 0.85)]:
    idx = 0 if key == "ph" else 1
    vals = [published[t][idx] for t, _, _ in TASKS]
    v = norm(vals) + [norm(vals)[0]]
    ax.plot(angles, v, color=colour, linestyle=style, linewidth=width,
            alpha=alpha, label=f"Caduceus-{key.upper()}", zorder=2)

  for arm, _, label, colour, bounded in ARMS:
    vals = [scores.get((arm, t)) for t, _, _ in TASKS]
    if any(v is None for v in vals):
      continue
    v = norm(vals) + [norm(vals)[0]]
    ax.plot(angles, v, color=colour, linewidth=2.0, label=label, zorder=3)
    ax.fill(angles, v, color=colour, alpha=0.06, zorder=1)

  # the degenerate axis, marked
  for i, (key, _, baseline) in enumerate(TASKS):
    if baseline is None:
      continue
    y = (baseline - lo[i]) / (hi[i] - lo[i])
    # Marker on the axis, caption parked in a free corner with a leader line.
    # Placing the text near the axis put it on top of that axis's own tick
    # label and rendered both unreadable.
    ax.plot([angles[i]], [y], marker="o", markersize=9, color="#c00000",
            markerfacecolor="none", markeredgewidth=2.2, zorder=6)
    ax.annotate("this axis is degenerate:\nlen(seq) alone scores 0.9004,\n"
                "above every model here",
                xy=(angles[i], y), xycoords="data",
                xytext=(0.015, 0.055), textcoords="figure fraction",
                color="#c00000", fontsize=8.2, ha="left", va="bottom",
                zorder=7,
                arrowprops=dict(arrowstyle="-", color="#c00000",
                                linewidth=1.0, alpha=0.75,
                                connectionstyle="arc3,rad=0.15"))

  ax.set_xticks(angles[:-1])
  ax.set_xticklabels(
    [f"{d}\n[{lo[i]:.2f}–{hi[i]:.2f}]" for i, (_, d, _) in enumerate(TASKS)],
    fontsize=8.2)
  ax.set_yticklabels([])
  ax.set_ylim(0, 1.05)
  ax.grid(color="#cccccc", linewidth=0.5)
  ax.spines["polar"].set_color("#cccccc")
  ax.set_title("GenomicBenchmarks — 8 tasks, full splits, 5 seeds, "
               "fine-tuned at backbone_lr 3e-5\n"
               "each axis scaled to its own range (shown in brackets); "
               "outward is better",
               fontsize=10.5, pad=26)
  ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False,
            fontsize=8.6)
  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)


def likelihood(curve_best, full_split, path):
  labels, nlls, colours, bounded_flags, sources = [], [], [], [], []
  for arm, run, label, colour, bounded in ARMS:
    alias = {"bissm_bd": "bissm"}.get(arm, arm)
    if alias in full_split:
      nlls.append(full_split[alias]); sources.append("full split")
    elif curve_best.get(arm):
      nlls.append(curve_best[arm][0]); sources.append("128-batch")
    else:
      continue
    labels.append(label); colours.append(colour); bounded_flags.append(bounded)

  ppl = [math.exp(v) for v in nlls]
  bits = [v / math.log(2) for v in nlls]
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), dpi=160)
  for ax, vals, title, ylab in [
      (axes[0], nlls, "Validation NLL", "nats / token"),
      (axes[1], ppl, "Perplexity", "exp(NLL)"),
      (axes[2], bits, "Bits per nucleotide", "bits")]:
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colours, width=0.66,
                  edgecolor="white", linewidth=1.1)
    for bar, bnd in zip(bars, bounded_flags):
      if bnd:
        bar.set_hatch("///")
    for xi, v in zip(x, vals):
      ax.text(xi, v, f"{v:.4f}" if ylab != "bits" else f"{v:.3f}",
              ha="center", va="bottom", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.6)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    lo_, hi_ = min(vals), max(vals)
    ax.set_ylim(lo_ - (hi_ - lo_) * 0.35, hi_ + (hi_ - lo_) * 0.22)
  axes[2].axhline(2.0, color="#c00000", linestyle="--", linewidth=1.2)
  axes[2].text(len(labels) - 0.5, 2.0, " uniform ACGT = 2.0", color="#c00000",
               fontsize=8, va="bottom", ha="right")
  fig.suptitle("Hatched bars are a NELBO — an UPPER bound on NLL, not the NLL. "
               "A hatched bar above a solid one may be a worse model, or an "
               "equally good one with a looser bound.", fontsize=9.6, y=1.02)
  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)
  return list(zip(labels, nlls, ppl, bits, sources))


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--gb-dir", default="results/caduceus/genomic_benchmarks_ft")
  parser.add_argument("--ppl-glob", default="logs/eval/ppl_ssm_baselines_123151.tsv")
  parser.add_argument("--runs", default="outputs/hg38-caduceus")
  parser.add_argument("--outdir", type=Path, default=REPO / "results" / "figures")
  args = parser.parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)

  scores, published = load_benchmarks(REPO / args.gb_dir)
  have = [a for a, *_ in ARMS if all((a, t) in scores for t, _, _ in TASKS)]
  print(f"benchmarks: {len(have)}/5 arms complete at backbone_lr {FT_LR:g}")
  radar(scores, published, args.outdir / "benchmark_radar.png")
  print(f"wrote {args.outdir / 'benchmark_radar.png'}")

  curve = {a: best_val_nll(REPO / args.runs / run) for a, run, *_ in ARMS}
  full = load_full_split(str(REPO / args.ppl_glob))
  rows = likelihood(curve, full, args.outdir / "likelihood_metrics.png")
  print(f"wrote {args.outdir / 'likelihood_metrics.png'}")
  print(f"\n  {'arm':<16}{'NLL':>9}{'ppl':>9}{'bits/nt':>9}   source")
  for label, n, p, b, src in rows:
    print(f"  {label:<16}{n:>9.4f}{p:>9.4f}{b:>9.4f}   {src}")


if __name__ == "__main__":
  main()
