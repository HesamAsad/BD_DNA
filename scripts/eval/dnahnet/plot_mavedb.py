#!/usr/bin/env python3
"""Plot our MaveDB results beside the values published by dnaHNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PUBLISHED = [
  ("dnaHNet", 0.3266, r"$6.4\times10^{19}$ FLOPs"),
  ("StripedHyena2", 0.3110, r"$7.11\times10^{19}$ FLOPs"),
  ("Transformer", 0.1555, r"$8.0\times10^{19}$ FLOPs"),
]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--result", action="append", default=[], metavar="LABEL=SUMMARY_JSON",
    help="May be repeated for our evaluated checkpoints")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  labels, values, notes, colors, hatches = [], [], [], [], []
  for item in args.result:
    if "=" not in item:
      parser.error(f"Expected LABEL=SUMMARY_JSON, received {item!r}")
    label, filename = item.split("=", 1)
    with Path(filename).open(encoding="utf-8") as handle:
      summary = json.load(handle)
    labels.append(label)
    values.append(float(summary["macro_abs_spearman"]))
    notes.append(f"ours, n={summary['num_variants']:,}")
    colors.append("#4c78a8")
    hatches.append("")
  for label, value, compute in PUBLISHED:
    labels.append(label)
    values.append(value)
    notes.append(f"published\n{compute}")
    colors.append("#f58518")
    hatches.append("//")

  figure, axis = plt.subplots(figsize=(max(7.0, 1.35 * len(labels)), 4.6))
  bars = axis.bar(range(len(labels)), values, color=colors, edgecolor="#333333")
  for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
  for index, (value, note) in enumerate(zip(values, notes)):
    axis.text(index, value + 0.007, f"{value:.3f}\n{note}",
              ha="center", va="bottom", fontsize=8)
  axis.set_xticks(range(len(labels)), labels)
  axis.set_ylabel("Absolute Spearman correlation")
  axis.set_title("Zero-shot E. coli K-12 variant-effect prediction")
  axis.set_ylim(0, max(values) * 1.35)
  axis.grid(axis="y", alpha=0.22)
  figure.text(
    0.01, 0.01,
    "Our values: macro mean across 12 pinned MaveDB assays. "
    "Published values: dnaHNet Table 5; training data/objectives differ.",
    fontsize=7)
  figure.tight_layout(rect=(0, 0.05, 1, 1))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=220)
  print(args.output)


if __name__ == "__main__":
  main()
