#!/usr/bin/env python3
"""Plot dnaHNet-aligned forward throughput, memory, and latency profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _length_label(length: int) -> str:
  if length >= 1024:
    return f"{length // 1024}K"
  return str(length)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--result", action="append", required=True, metavar="LABEL=PROFILE_JSON")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  profiles = []
  for item in args.result:
    if "=" not in item:
      parser.error(f"Expected LABEL=PROFILE_JSON, received {item!r}")
    label, filename = item.split("=", 1)
    with Path(filename).open(encoding="utf-8") as handle:
      summary = json.load(handle)
    records = [
      record for record in summary["records"] if record["status"] == "ok"]
    if not records:
      parser.error(f"Profile {filename!r} contains no successful records")
    profiles.append((label, summary, records))

  figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
  fields = (
    ("throughput_nt_per_second", "Throughput (nt/s)", True),
    ("peak_memory_gib", "Peak GPU memory (GiB)", False),
    ("latency_seconds", "Forward latency (s)", True),
  )
  all_lengths = sorted({
    int(record["length"])
    for _, _, records in profiles for record in records})
  for axis, (field, ylabel, log_y) in zip(axes, fields):
    for label, summary, records in profiles:
      line, = axis.plot(
        [record["length"] for record in records],
        [record[field] for record in records],
        marker="o", linewidth=2.2, label=label)
      for record in summary["records"]:
        if record["status"] != "ok":
          axis.axvline(
            record["length"], color=line.get_color(), linestyle=":",
            linewidth=1.5, alpha=0.75)
    axis.set_xscale("log", base=2)
    if log_y:
      axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Context length (nucleotides)")
    axis.grid(alpha=0.22, which="both")
    axis.set_xticks(all_lengths, [_length_label(length) for length in all_lengths],
                    rotation=45, ha="right")
  axes[0].legend(frameon=False)
  figure.suptitle("Single-GPU BF16 diffusion forward scaling (batch 1)")
  gpu_names = sorted({summary["gpu"] for _, summary, _ in profiles})
  figure.text(
    0.01, 0.01,
    "Fixed t=0.5 corruption; medians exclude one warm-up. "
    f"GPU: {', '.join(gpu_names)}. dnaHNet reports the same three metrics "
    "for autoregressive H100 forwards; objectives and hardware differ. "
    "Dotted endpoint: backend capacity limit.",
    fontsize=8)
  figure.tight_layout(rect=(0, 0.07, 1, 0.95))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=220)
  print(args.output)


if __name__ == "__main__":
  main()
