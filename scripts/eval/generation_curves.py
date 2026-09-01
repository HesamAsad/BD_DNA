#!/usr/bin/env python3
"""End-to-end generation cost: wall-clock and memory to emit N tokens vs T.

THE HONEST GENERATION COMPARISON, which neither existing computational figure
provides. `inference_forward.png` times ONE _loss forward -- a single
noise-and-denoise pass for a BD arm -- so it charges block diffusion for one
denoising step when generation needs T, and understates BD's cost.
`inference_state.png` is real generation but reports memory only, for two arms.

The arithmetic those figures omit: to emit L tokens,

    AR   L         forward passes, 1 token each
    BD   (L/b)*T   forward passes, b tokens each in parallel

BD performs L*T token-updates against AR's L. Within-block parallelism recovers
most of that in wall-clock but not all, and T is the dial. Sweeping it also
traces the cost/quality knob BD has and AR does not -- fewer steps is cheaper
and worse -- which no figure in this project has shown.

T does not apply to the AR arms (one token per step, no refinement), so they
are drawn as horizontal reference lines across the T axis rather than as
curves. That is the comparison: at which T, if any, does a BD arm generate as
fast as AR?


MEASURED RESULT (2026-09-01, 69 runs). The framing above anticipated BD being
the expensive one. It is not, at any T this sweep reached. The decisive count
is FORWARD PASSES, not token-updates: AR does L of them, BD does (L/b)*T. With
b=256, BD does FEWER forwards than AR whenever T < 256, and the sweep tops out
at T=64. So BD leads almost everywhere, and the true break-even sits at T = b =
256, outside this grid.

  fastest AR is Transformer-AR, flat at ~125 tok/s for every N (sliding KV
  cache => constant per-token cost). uSSM-AR is roughly half that, 45-64 tok/s,
  rising with N as its fixed per-step overhead amortises -- the launch-bound
  signature, not a bandwidth limit.

  BD stays ahead of the fastest AR up to:
    N=1,024   uSSM-BD T<=32   BiSSM-BD T<=16   Transformer-BD T<=64
    N=4,096   uSSM-BD T<=64   BiSSM-BD T<=32   Transformer-BD T<=64
    N=16,384  all three arms T<=64 (never crossed in range)

  Memory is the cleaner architectural story: the SSM arms are FLAT in N at
  0.72-0.73 GB with a 4.7 MiB cache, while Transformer-BD grows 1.19->1.37 GB
  and carries a 468 MiB cache -- 100x the SSM state.

CAVEAT: T=1 runs first in each (arm, N) group and absorbs warm-up, so several
T=1 points read slower than their T=2 neighbours (uSSM-BD at N=1,024: 137.7 vs
174.0 tok/s). Treat T=1 as contaminated; the T>=2 curve is the trustworthy one.
Usage:
  python scripts/eval/generation_curves.py --indir results/generation
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STYLE = {
  "ussm_ar": ("uSSM-AR", "#e0a62e", False),
  "xf_ar": ("Transformer-AR", "#5b9bd5", False),
  "ussm_bd": ("uSSM-BD", "#8c6d1f", True),
  "bissm_bd": ("BiSSM-BD", "#c2503f", True),
  "xf_bd": ("Transformer-BD", "#1f4e9c", True),
}
LABEL = re.compile(r"gen_(?P<arm>[a-z_]+?)_N(?P<n>\d+)(?:_T(?P<t>\d+))?$")


def load(indir: Path):
  """(arm, N, T) -> {tokens_per_second, peak_gpu_bytes, cache_bytes}.

  T is None for the AR arms. Reads whichever of the two harnesses wrote the
  file: ar_decode_benchmark stores its numbers at the top level, while
  ssm_streaming_benchmark nests the generation block under "generation".
  """
  out = {}
  # recursive: each run writes <indir>/<label>/summary.json, because both
  # harnesses hardcode the summary.json basename.
  for path in sorted(str(p) for p in indir.rglob("*.json")):
    try:
      payload = json.load(open(path))
    except (OSError, json.JSONDecodeError):
      continue
    stem = os.path.splitext(os.path.basename(path))[0]
    match = LABEL.search(payload.get("label", stem) or stem)
    if not match:
      continue
    # The two harnesses nest their numbers differently: ssm_streaming puts them
    # under "generation", ar_decode under "benchmark", and neither at the top
    # level. Checking only the first two silently drops every Transformer-AR
    # run -- an absent arm, not an error.
    body = (payload.get("generation") or payload.get("benchmark") or payload)
    tps = body.get("tokens_per_second")
    if tps is None:
      continue
    arm = match.group("arm")
    n = int(match.group("n"))
    t = int(match.group("t")) if match.group("t") else None
    out[(arm, n, t)] = {
      "tokens_per_second": float(tps),
      "peak_gb": (body.get("peak_gpu_bytes") or 0) / 1e9,
      "cache_mib": (body.get("cache_bytes") or 0) / 2 ** 20,
      "seconds": body.get("seconds"),
    }
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--indir", type=Path, default=REPO / "results" / "generation")
  parser.add_argument("--outdir", type=Path, default=REPO / "results" / "figures")
  args = parser.parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)

  data = load(args.indir)
  if not data:
    sys.exit(f"no generation results under {args.indir}")
  lengths = sorted({n for _, n, _ in data})
  steps = sorted({t for _, _, t in data if t is not None})
  print(f"loaded {len(data)} runs   N in {lengths}   T in {steps}")

  fig, axes = plt.subplots(2, len(lengths), figsize=(5.2 * len(lengths), 8.6),
                           dpi=160, squeeze=False)
  for col, n in enumerate(lengths):
    for row, (key, ylab, title) in enumerate([
        ("tokens_per_second", "Tokens / second", "Generation throughput"),
        ("peak_gb", "Peak GPU memory (GB)", "Generation memory")]):
      ax = axes[row][col]
      for arm, (label, colour, bounded) in STYLE.items():
        if bounded:
          xs = [t for t in steps if (arm, n, t) in data]
          if not xs:
            continue
          ax.plot(xs, [data[(arm, n, t)][key] for t in xs], marker="o",
                  color=colour, linewidth=1.9, markersize=5, label=label)
        else:
          entry = data.get((arm, n, None))
          if not entry:
            continue
          # AR has no T. Draw it as the level it sits at, which is the whole
          # point of the comparison: where does a BD curve cross it?
          ax.axhline(entry[key], color=colour, linestyle="--", linewidth=1.7,
                     label=f"{label} (no T)")
      ax.set_xscale("log", base=2)
      if key == "tokens_per_second":
        ax.set_yscale("log")
      ax.set_xlabel("Denoising steps per block, T")
      ax.set_ylabel(ylab)
      ax.set_title(f"{title} — N = {n:,} tokens", fontsize=10.5)
      ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
      if steps:
        ax.set_xticks(steps)
        ax.set_xticklabels([str(t) for t in steps], fontsize=8)
      if row == 0 and col == 0:
        ax.legend(frameon=False, fontsize=8)
  fig.suptitle("End-to-end generation cost. BD emits (L/b)*T forward passes "
               "against AR's L; dashed lines are the AR arms, which have no T.",
               fontsize=10.5, y=1.0)
  fig.tight_layout()
  path = args.outdir / "generation_cost.png"
  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)
  print(f"wrote {path}")

  # the number people will actually ask for
  print(f"\n{'arm':<16}{'N':>8}{'T':>5}{'tok/s':>12}{'peak GB':>10}{'cache MiB':>11}")
  print("-" * 62)
  for (arm, n, t), v in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2] or 0)):
    print(f"{STYLE.get(arm, (arm,))[0]:<16}{n:>8,}{(t if t else '-'):>5}"
          f"{v['tokens_per_second']:>12,.1f}{v['peak_gb']:>10.2f}{v['cache_mib']:>11.1f}")
  for n in lengths:
    ar = [(a, data[(a, n, None)]["tokens_per_second"])
          for a in STYLE if (a, n, None) in data]
    if not ar:
      continue
    fastest = max(ar, key=lambda kv: kv[1])
    print(f"\n  N={n:,}: fastest AR is {STYLE[fastest[0]][0]} at "
          f"{fastest[1]:,.0f} tok/s")
    for arm, (label, _, bounded) in STYLE.items():
      if not bounded:
        continue
      cross = [t for t in steps if (arm, n, t) in data
               and data[(arm, n, t)]["tokens_per_second"] >= fastest[1]]
      print(f"    {label:<16} matches or beats it at T <= "
            f"{max(cross) if cross else 'never in this sweep'}")


if __name__ == "__main__":
  main()
