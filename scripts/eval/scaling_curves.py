#!/usr/bin/env python3
"""Throughput, memory and FLOPs against sequence length, as three figures.

`training_flops.py` answers "what did each run cost" at one fixed geometry.
This answers the different question the architecture claim actually rests on:
how does each cost SCALE with context length?

The three panels are not the same kind of quantity, and conflating them is how
this project previously got the story wrong:

  FLOPs       analytic, per forward+backward sequence. Attention is quadratic
              here and the SSM is linear -- this is the panel where the
              textbook asymptotics are visible.
  throughput  measured. This is where the crossover actually lives, and it sits
              far from where FLOPs alone would predict, because a Mamba-2 scan
              achieves much lower arithmetic intensity than a flash-attention
              matmul.
  memory      measured. BOTH families are LINEAR in length: flash attention
              never materialises the L x L score matrix, so its quadratic cost
              is time, not space. The SSM's disadvantage here is a constant
              factor, not an asymptote, and no length fixes it.

FLOP formulas are imported from training_flops so there is exactly one
definition; only the length-dependent constants are recomputed per point.

Usage:
  python scripts/eval/scaling_curves.py \
      --measured 'results/sizing/scaling-*.json' --outdir results/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import scripts.eval.training_flops as tf  # noqa: E402

# Arm -> (display label, colour, marker). Colours separate the two families:
# blues for attention, warm for state space.
STYLE = {
  "dit":     ("Transformer-BD", "#1f4e9c", "s"),
  "dit-ar":  ("Transformer-AR", "#5b9bd5", "^"),
  "bissm":   ("BiSSM-BD",       "#c2503f", "o"),
  "ussm-ar": ("uSSM-AR",        "#e0a62e", "D"),
}
ORDER = ["dit", "dit-ar", "bissm", "ussm-ar"]


def flops_per_sequence(arm, length, block=256, n_layers=12):
  """Forward+backward FLOPs for ONE sequence of `length`, from tf's formulas."""
  num_blocks = max(length // block, 1)
  prefix = (num_blocks - 1) * block
  t = tf.ssm_terms()

  clean = n_layers * prefix * (
    t["in_proj"] + t["conv"] + 2 * t["scan"] + t["out_proj"] + t["mlp"]
    + (2 * num_blocks * (num_blocks - 1) * tf.NHEADS * tf.HEADDIM
       * tf.D_STATE) // max(prefix, 1))
  act_uni = n_layers * length * (t["mixer"] + t["mlp"]) + length * t["head"]
  act_bi = n_layers * length * (2 * t["mixer"] + t["mlp"]) + length * t["head"]
  ar = (n_layers * (length - 1) * (t["mixer"] + t["mlp"])
        + (length - 1) * t["head"])

  if arm == "ussm-ar":
    fwd = ar
  elif arm == "bissm":
    fwd = clean + act_bi
  elif arm == "dit-ar":
    tokens = length - 1
    pairs = tokens * (tokens + 1) // 2
    fwd = (2 * tf.transformer_ar_params() * tokens
           + 4 * n_layers * pairs * tf.D_AR)
  elif arm == "dit":
    # cross_attn concatenates [x_t; x_0] to 2L; the block-diffusion mask
    # permits block^2 * nb * (nb + 1) pairs, not (2L)^2.
    pairs = block * block * num_blocks * (num_blocks + 1)
    fwd = (2 * tf.transformer_bd_params() * (2 * length)
           + 4 * n_layers * pairs * tf.D_BD)
  else:
    raise KeyError(arm)
  return tf.GRAD_MULT * fwd


def load_measured(pattern):
  """{arm: {length: row}} from sizing_sweep outputs."""
  out = {}
  for path in sorted(glob.glob(pattern)):
    payload = json.loads(Path(path).read_text())
    for row in payload.get("rows", []):
      if row.get("oom") or row.get("peak_gib") is None:
        continue
      out.setdefault(row["arm"], {})[row["length"]] = row
  return out


def panel(ax, xs, series, ylabel, title, logy=True):
  for arm in ORDER:
    if arm not in series or not series[arm]:
      continue
    label, colour, marker = STYLE[arm]
    lengths = sorted(series[arm])
    ax.plot(lengths, [series[arm][L] for L in lengths],
            marker=marker, color=colour, label=label, linewidth=1.9,
            markersize=5.5)
  ax.set_xscale("log", base=2)
  if logy:
    ax.set_yscale("log")
  ax.set_xlabel("Sequence length (nt)")
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=11)
  ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
  ax.set_xticks(xs)
  ax.set_xticklabels([f"$2^{{{int(math.log2(x))}}}$" for x in xs])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--measured", default="results/sizing/scaling-*.json")
  parser.add_argument("--outdir", type=Path, default=REPO / "results" / "figures")
  parser.add_argument("--flop-lengths", default="2048,4096,8192,16384,32768,65536,131072")
  args = parser.parse_args()

  measured = load_measured(str(REPO / args.measured)
                           if not args.measured.startswith("/")
                           else args.measured)
  if not measured:
    print(f"no measured rows matched {args.measured}", file=sys.stderr)

  flop_lengths = [int(v) for v in args.flop_lengths.split(",")]
  flops = {arm: {L: flops_per_sequence(arm, L) / 1e12 for L in flop_lengths}
           for arm in ORDER}

  # Prefer MEASURED FLOPs where the analytic formula is known wrong.
  #
  # Running torch's FlopCounterMode against these formulas settled which to
  # trust, per arm:
  #   dit-ar  analytic VALIDATED exactly -- counted dense plus the analytic
  #           attention term reproduces the analytic total to the decimal at
  #           every length (1.48/8.92/83.92). Keep analytic.
  #   dit     cannot be counted: flex is compiled and the counter cannot enter
  #           it, and substituting sdpa measures a DIFFERENT algorithm, since
  #           sdpa evaluates the full (2L)^2 attention while flex evaluates only
  #           the mask-permitted pairs. Keep analytic, which the dit-ar result
  #           validates the machinery of.
  #   ussm-ar analytic UNDERCOUNTS by a constant 1.37x at every length.
  #   bissm   analytic UNDERCOUNTS by a constant 1.35x at every length.
  # For the two SSM arms the measured count is the better number even though it
  # excludes the Triton scan, so it is a LOWER bound on the truth -- and it is
  # already above the analytic figure.
  measured_path = REPO / "results" / "sizing" / "measured_flops.json"
  measured_arms = {"ussm-ar", "bissm"}
  flop_source = {arm: "analytic" for arm in ORDER}
  if measured_path.exists():
    payload = json.loads(measured_path.read_text())
    batch = payload.get("batch", 1)
    for row in payload.get("rows", []):
      if row.get("arm") in measured_arms and "counted_tflop" in row:
        flops[row["arm"]][row["length"]] = row["counted_tflop"] / batch
        flop_source[row["arm"]] = "measured (lower bound; scan uncounted)"
    # Measured points exist only at the swept lengths; drop analytic points
    # beyond them so a single curve never mixes the two sources.
    swept = {r["length"] for r in payload.get("rows", []) if "counted_tflop" in r}
    for arm in measured_arms:
      if flop_source[arm] != "analytic":
        flops[arm] = {L: v for L, v in flops[arm].items() if L in swept}
  mem = {arm: {L: r["peak_gib"] for L, r in d.items()}
         for arm, d in measured.items()}
  tput = {arm: {L: r["nt_per_second"] for L, r in d.items()}
          for arm, d in measured.items()}

  args.outdir.mkdir(parents=True, exist_ok=True)
  specs = [
    ("flops", flops, sorted({L for d in flops.values() for L in d}),
     "TFLOPs per sequence (fwd+bwd)",
     "Arithmetic: attention quadratic, scan linear (SSM = measured)", True),
    ("throughput", tput, sorted({L for d in tput.values() for L in d}),
     "Nucleotides / second", "Throughput: where the crossover really is", False),
    ("memory", mem, sorted({L for d in mem.values() for L in d}),
     "Peak GPU memory (GiB)", "Memory: both linear -- a constant factor, not an asymptote", False),
  ]
  written = []
  for name, series, xs, ylabel, title, logy in specs:
    if not xs:
      print(f"skipping {name}: no data")
      continue
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    panel(ax, xs, series, ylabel, title, logy)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = args.outdir / f"scaling_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    print(f"wrote {path}")

  # The crossover, from the measured throughput, is the number the architecture
  # claim rests on -- report it rather than leaving it to be eyeballed.
  if "bissm" in tput and "dit" in tput:
    shared = sorted(set(tput["bissm"]) & set(tput["dit"]))
    if len(shared) >= 2:
      xs = [math.log2(L) for L in shared]
      ys = [math.log(tput["dit"][L] / tput["bissm"][L]) for L in shared]
      n = len(xs)
      mx, my = sum(xs) / n, sum(ys) / n
      denom = sum((x - mx) ** 2 for x in xs)
      if denom > 0:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        crossover = 2 ** (-(my - slope * mx) / slope)
        print(f"\nmeasured throughput crossover, BiSSM vs Transformer-BD: "
              f"L ~= {crossover:,.0f} nt")
        for L in shared:
          r = tput["dit"][L] / tput["bissm"][L]
          who = "Transformer" if r > 1 else "BiSSM"
          print(f"  L={L:>6}  ratio {r:.3f}  ({who} ahead)")

  summary = args.outdir / "scaling_data.json"
  summary.write_text(json.dumps(
    {"flops_tflop_per_sequence": flops, "flop_source": flop_source,
     "peak_gib": mem, "nt_per_second": tput,
     "note": "FLOPs analytic from training_flops.py; memory and throughput "
             "measured with a real fwd+bwd+AdamW step at micro batch 2, "
             "SSM arms with checkpoint_boundary_prefill=on."},
    indent=2, sort_keys=True) + "\n")
  print(f"wrote {summary}")


if __name__ == "__main__":
  main()
