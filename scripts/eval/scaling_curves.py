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
  # NOT 2 * mixer. training_flops.py charges the full mixer twice because that
  # is what the TRAINED runs paid: before 5dad03c (2026-08-20) scan_active
  # called mixer.scan_segment twice, forward and flipped, each a complete mixer
  # including both projections. The BiSSM-BD checkpoint is from 2026-08-10 and
  # is correctly costed that way.
  #
  # This file projects the CURRENT code, where scan_bidirectional shares the
  # projections: mamba2_segment.py:635 calls in_proj once and :666 calls
  # out_proj once, and only _causal_conv/_reverse_causal_conv and the two _scan
  # calls are doubled. So the doubled part is conv + scan, not the whole mixer.
  # Charging 2 * mixer here overstates BiSSM by in_proj + out_proj per token per
  # layer = 7,311,360, which is 28.5% of the active term and 16.8% of the arm.
  act_bi = (n_layers * length * (t["mixer"] + t["conv"] + t["scan"] + t["mlp"])
            + length * t["head"])
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
           + tf.ATTN_SCALE * 4 * n_layers * pairs * tf.D_AR)
  elif arm == "dit":
    # cross_attn concatenates [x_t; x_0] to 2L; the block-diffusion mask
    # permits block^2 * nb * (nb + 1) pairs, not (2L)^2.
    pairs = block * block * num_blocks * (num_blocks + 1)
    fwd = (2 * tf.transformer_bd_params() * (2 * length)
           + tf.ATTN_SCALE * 4 * n_layers * pairs * tf.D_BD)
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

  # ANALYTIC for every arm. This reverses an earlier decision, and the reason is
  # a bug in torch, not in this repo.
  #
  # We used to prefer the runtime FlopCounterMode count for the two SSM arms,
  # because it exceeded the analytic formula by a constant 1.35-1.37x and we read
  # that as the formula undercounting. It is the counter overcounting.
  #
  #   torch.utils.flop_counter.conv_backward_flop takes `_groups` and never uses
  #   it -- its own docstring (flop_counter.py:83) says "there are also some
  #   details involving transpose of the batch/channel dimensions and groups, but
  #   I skip those for the sake of brevity". The grad_weight branch transposes
  #   dims 0 and 1, after which batch_size becomes the channel count and
  #   c_out*c_in becomes conv_dim*1, so a DEPTHWISE conv is billed at the DENSE
  #   price. For this model that is a factor of conv_dim = 1664 on the
  #   grad_weight term, 555x on the conv as a whole.
  #
  #   It lands in no module because mamba2_segment.py never calls self.conv1d(...)
  #   -- it reads self.conv1d.weight/.bias (:191) and calls functional F.conv1d
  #   (:197, :209, :210, :264, :270, :271, :492), so nn.Conv1d.__call__ never
  #   fires and no bucket is created.
  #
  # The accounting closes exactly (uSSM-AR, L=8192, one sequence, fwd+bwd, TFLOP):
  #   analytic 5.180 + phantom conv 2.177 - invisible Triton scan 0.237 = 7.120
  #   = measured, to three decimals, i.e. a residual of 0.0% of the gap.
  #
  # So the analytic formula is CONFIRMED for every aten-visible term. The scan
  # term (4.6% of the total) is Triton and stays invisible to the counter, so it
  # remains a derivation rather than a measurement -- state that, do not hide it.
  #
  # Analytic also evaluates at any length, so the SSM curves now reach 2^17
  # alongside the Transformer ones instead of stopping at the swept set.
  flop_source = {arm: "analytic" for arm in ORDER}

  # The measured file is still loaded, as a DIAGNOSTIC of the torch bug rather
  # than as curve data. Keeping the ratio in the output is what stops anyone
  # (including us, again) from re-reading the discrepancy as a modelling error.
  measured_path = REPO / "results" / "sizing" / "measured_flops.json"
  counter_ratio = {}
  for path in sorted(glob.glob(str(measured_path.parent / "measured_flops*.json"))):
    payload = json.loads(Path(path).read_text())
    for row in payload.get("rows", []):
      if "counted_tflop" not in row or row["arm"] not in flops:
        continue
      # Per ROW batch. These files accumulate across sweeps that do not share
      # one: everything past 32768 had to be counted at micro batch 1 because
      # batch 2 OOMs. Dividing a mixed file by one top-level number would halve
      # or double whichever half of the curve it did not describe.
      batch = row.get("batch") or payload.get("batch") or 1
      per_seq = row["counted_tflop"] / batch
      predicted = flops[row["arm"]].get(row["length"])
      if predicted:
        counter_ratio.setdefault(row["arm"], {})[row["length"]] = per_seq / predicted

  mem = {arm: {L: r["peak_gib"] for L, r in d.items()}
         for arm, d in measured.items()}
  tput = {arm: {L: r["nt_per_second"] for L, r in d.items()}
          for arm, d in measured.items()}

  args.outdir.mkdir(parents=True, exist_ok=True)
  specs = [
    ("flops", flops, sorted({L for d in flops.values() for L in d}),
     "TFLOPs per sequence (fwd+bwd)",
     "Arithmetic: attention quadratic, scan linear", True),
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
     "flop_counter_ratio": counter_ratio,
     "note": "FLOPs per SEQUENCE: analytic from training_flops.py for EVERY "
             "arm. An earlier version preferred the FlopCounterMode count for "
             "the SSM arms, believing the formulas undercut by 1.35-1.37x. That "
             "was the counter overcounting, not the formula undercounting: "
             "torch's conv_backward_flop ignores `groups` and bills this "
             "model's depthwise conv at the dense price (conv_dim = 1664x on "
             "the grad_weight term). Accounting closes exactly -- uSSM-AR at "
             "L=8192, per sequence fwd+bwd: analytic 5.180 + phantom conv 2.177 "
             "- invisible Triton scan 0.237 = 7.120 = measured. Every "
             "aten-visible term of the formula is thereby confirmed; the scan "
             "term (4.6% of total) is Triton and remains a derivation, not a "
             "measurement. flop_counter_ratio keeps the measured/analytic ratio "
             "as evidence of the torch bug, NOT as curve data. Memory and "
             "throughput measured with a real fwd+bwd+AdamW step at micro batch "
             "2 and are NOT per sequence, SSM arms with "
             "checkpoint_boundary_prefill=on."},
    indent=2, sort_keys=True) + "\n")
  print(f"wrote {summary}")


if __name__ == "__main__":
  main()
