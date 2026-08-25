#!/usr/bin/env python3
"""The two panels the training figures structurally cannot show.

Our published throughput/memory/FLOPs panels all measure a TRAINING step, where
every position is processed in parallel: there is no KV cache and no recurrent
state to reuse, so you simply pay for activations. A Mamba-2 layer is WIDER per
token than an attention layer, so the SSM arms use MORE memory there. That is
correct, expected, and exactly backwards from what an SSM is famous for.

The famous property is an INFERENCE one, and it needs its own measurement:

  forward   one parallel pass over a whole sequence, no cache. Matches the axes
            of dnaHNet Figure 7 so the two can be compared directly.
  state     the cache carried while GENERATING. The SSM keeps one fixed-size
            recurrent state; the Transformer keeps a key and value for every
            token it has seen. This is where the ordering inverts.

Usage:
  python scripts/eval/inference_curves.py --outdir results/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STYLE = {
  "dit":     ("Transformer-BD", "#1f4e9c", "s"),
  "dit-ar":  ("Transformer-AR", "#5b9bd5", "^"),
  "bissm":   ("BiSSM-BD",       "#c2503f", "o"),
  "ussm":    ("uSSM-BD",        "#8c6d1f", "v"),
  "ussm-ar": ("uSSM-AR",        "#e0a62e", "D"),
}
ORDER = ["dit", "dit-ar", "bissm", "ussm", "ussm-ar"]


def panel(ax, series, ylabel, title, logy=True):
  xs = sorted({L for d in series.values() for L in d})
  for arm in ORDER:
    d = series.get(arm)
    if not d:
      continue
    label, colour, marker = STYLE[arm]
    ls = sorted(d)
    ax.plot(ls, [d[L] for L in ls], marker=marker, color=colour, label=label,
            linewidth=1.9, markersize=5.5)
  ax.set_xscale("log", base=2)
  if logy:
    ax.set_yscale("log")
  ax.set_xlabel("Sequence length (nt)")
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=11)
  ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
  ax.set_xticks(xs)
  ax.set_xticklabels([f"$2^{{{int(math.log2(x))}}}$" for x in xs], fontsize=8)
  ax.legend(frameon=False, fontsize=8.5)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--forward", default="results/sizing/forward_pass.json")
  # fullctx-* are the runs taken AFTER models/dit.py's 1024-token window was
  # removed. The older infer-* runs measured the DiT discarding context, and
  # globbing them re-emits the stale 54.0 MiB constant.
  parser.add_argument("--streaming", default="results/streaming/fullctx-*")
  parser.add_argument("--outdir", type=Path, default=REPO / "results" / "figures")
  args = parser.parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)
  written = []

  # ---- forward pass: throughput, memory, latency on dnaHNet's axes ---------
  fwd = json.loads((REPO / args.forward).read_text())
  tput, mem, lat = {}, {}, {}
  for r in fwd["rows"]:
    if r.get("tokens_per_second") is None:
      continue
    tput.setdefault(r["arm"], {})[r["length"]] = r["tokens_per_second"]
    mem.setdefault(r["arm"], {})[r["length"]] = r["peak_gb"]
    lat.setdefault(r["arm"], {})[r["length"]] = r["forward_ms"]
  fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), dpi=160)
  panel(axes[0], tput, "Tokens / second", "Forward throughput")
  panel(axes[1], mem, "Peak GPU memory (GB)", "Forward memory")
  panel(axes[2], lat, "Latency (ms)", "Forward latency")
  fig.suptitle("Forward pass only, batch 1 — the axes dnaHNet Figure 7 uses",
               fontsize=12)
  fig.tight_layout()
  path = args.outdir / "inference_forward.png"
  fig.savefig(path)
  plt.close(fig)
  written.append(path)

  # ---- generation state: the panel where the ordering inverts --------------
  state = {}
  gen = {}
  for d in sorted(glob.glob(str(REPO / args.streaming))):
    summary = Path(d) / "summary.json"
    if not summary.exists():
      continue
    # Directory names are <prefix>-<arm>-<n>-<jobid>, where the prefix is
    # `fullctx` (post-fix) or `infer` (pre-fix, DiT capped at 1024). Parse both
    # rather than slicing a fixed prefix length off the front.
    name = os.path.basename(d)
    parts = name.split("-")
    if len(parts) < 4:
      continue
    arm = {"bissm": "bissm", "ussm": "ussm", "ussm_ar": "ussm-ar",
           "dit": "dit", "bissm_bd": "bissm",
           "transformer_bd": "dit", "transformer_ar": "dit-ar"}.get(parts[1])
    if arm is None:
      arm = {"bissm-bd": "bissm", "ussm-ar": "ussm-ar",
             "transformer-bd": "dit",
             "transformer-ar": "dit-ar"}.get("-".join(parts[1:3]))
    if arm is None:
      continue
    tokens = None
    for candidate in parts[2:]:
      if candidate.isdigit() and int(candidate) <= 2 ** 20:
        tokens = int(candidate)
        break
    payload = json.loads(summary.read_text())
    generation = payload.get("generation") or {}
    # The x axis here is GENERATED TOKENS, which is what sizes a KV cache --
    # not the model's context window, which was the wrong knob and gave three
    # flat sweeps before anyone read the allocation.
    if tokens and "cache_bytes" in generation:
      state.setdefault(arm, {})[tokens] = generation["cache_bytes"] / 2 ** 20
      gen[arm] = generation["cache_bytes"] / 2 ** 20

  if state:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    panel(ax, state, "Cache carried during generation (MiB)",
          "Generation cache vs tokens generated", logy=True)
    ax.set_xlabel("Tokens generated")
    # The Transformer has no prefill_left -- there is no state to prefill,
    # only a KV cache -- so its curve comes from the generation measurement and
    # is drawn as the level it sits at, annotated rather than faked as a sweep.
    for arm, mib in gen.items():
      if arm in state:
        continue
      label, colour, _ = STYLE[arm]
      ax.axhline(mib, color=colour, linestyle="--", linewidth=1.6)
      ax.text(ax.get_xlim()[1], mib, f"  {label} KV cache {mib:.1f} MiB",
              color=colour, fontsize=8, va="center", ha="right")
    fig.tight_layout()
    path = args.outdir / "inference_state.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

  for p in written:
    print(f"wrote {p}")
  summary_path = args.outdir / "inference_data.json"
  summary_path.write_text(json.dumps(
    {"forward_tokens_per_second": tput, "forward_peak_gb": mem,
     "forward_ms": lat, "generation_state_mib": state,
     "generation_cache_mib": gen,
     "note": fwd.get("protocol", "")}, indent=2, sort_keys=True) + "\n")
  print(f"wrote {summary_path}")


if __name__ == "__main__":
  main()
