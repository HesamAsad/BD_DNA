#!/usr/bin/env python3
"""Where BiSSM-BD stands on both benchmarks, with every known correction applied.

One place to read the answer, so a headline is never quoted without the caveats
that were measured alongside it:

  MaveDB  - the macro per-assay Spearman IS largely a 1-vs-2 mutation counter, so the
            nsyn==1 stratum and the zero-parameter BLOSUM62 bar are printed
            beside it. Per-token normalisation is applied where it matters
            (it fixes a real defect: wt_loss - mut_loss differences totals over
            different token counts whenever a variant changes length).
  GB      - the 8-task mean is quoted for comparability with the published
            table, the 6-task leak-free mean beside it, because
            human_enhancers_ensembl has 37.8% of its test set duplicated from
            train and human_nontata_promoters 47.5% near-duplicated.
"""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.eval.dnahnet.mavedb import (  # noqa: E402
  protein_substitutions, BLOSUM62, read_jsonl_gz)
from scripts.eval.dnahnet.partial_corr import (  # noqa: E402
  partial_spearman, _count_features, _protein_events)
from scripts.eval.caduceus.genomic_benchmarks import (  # noqa: E402
  LEAKY_TASKS, REFERENCE, REFERENCE_COLUMNS)
from scipy.stats import spearmanr  # noqa: E402

# Canonical run directory per arm. TWO rules, both learned the hard way:
#  1. AR arms MUST use a --genomic-prefix run. Without a left prefix a
#     left-to-right model is starved of context, which penalises the
#     Transformer specifically: the no-prefix runs read uSSM-AR 0.2640 vs
#     Transformer-AR 0.1880 (a spurious 1.4x gap), while at matched geometry
#     the two TIE at 0.2547 / 0.2540. This list pointed at the no-prefix runs
#     until 2026-09-20 and so reproduced that artifact on every invocation.
#     `_check_arm_geometry` below now refuses to print an unprefixed AR arm.
#  2. Estimators are not defined on the same variants (nelbo 21,250,
#     infill_* 19,349, predictive_divergence 15,889), so read DOWN a column,
#     not across rows, when comparing estimators.
MAVEDB_ARMS = [
  ("uSSM-AR", "exact AR", "ussm-ar-genomicprefix-*"),
  ("Transformer-AR", "exact AR", "transformer-ar-genomicprefix-*"),
  ("BiSSM-BD b8", "NELBO eps0.9", "b8-nelbo-eps0.9-*"),
  ("BiSSM-BD b32", "Score I nt", "b32-infill_nt-unlocked-*"),
  ("BiSSM-BD b8", "NELBO", "blk8-nelbo-*"),
  ("BiSSM-BD b32", "NELBO", "blk32-step8000-nelbo-*"),
  ("BiSSM-BD b256", "NELBO", "matched-b256-nelbo-eps0.001-*"),
  ("BiSSM-BD b256", "Score I nt", "scoreI-nt-padleft-*"),
  ("BiSSM-BD b256", "causal PLL", "bissm-pllcausal-*"),
  ("uSSM-BD b256", "causal PLL", "ussm-bd-pllcausal-*"),
]


def _check_arm_geometry(arm: str, predictions_path) -> str | None:
  """Return a warning string if this run is not a valid choice for this arm."""
  sidecar = Path(predictions_path).resolve().parent / "summary.json"
  if not sidecar.exists():
    return "no summary.json -- cannot verify geometry"
  payload = json.loads(sidecar.read_text())
  if payload.get("macro_signed_spearman") is None:
    return "pre-correction summary (no macro_signed_spearman)"
  is_ar = (str(payload.get("parameterization")) == "ar"
           or payload.get("block_size") == 1)
  if is_ar and not payload.get("genomic_prefix"):
    return "AR arm scored WITHOUT --genomic-prefix (starves left context)"
  return None


def _macro(rows, col, want=None, absolute=False):
  """Macro per-assay Spearman: correlate WITHIN each assay, then average.

  `absolute=True` averages |rho| instead of rho. That is not our headline --
  taking the modulus credits a model for ranking an assay backwards -- but it
  is what dnaHNet's Figure 4B reports, so it is the only like-for-like figure
  to put beside their 0.3266.
  """
  groups = defaultdict(list)
  for row in rows:
    if want is not None:
      subs = protein_substitutions(row.get("hgvs_pro"))
      if subs is None or not want(len(subs)):
        continue
    raw = row.get(col)
    if raw in (None, ""):
      return float("nan")
    value = float(raw)
    if not np.isfinite(value):
      continue
    groups[row["score_set_urn"]].append(
      (value, float(row["experimental_score"])))
  vals = [spearmanr([a for a, _ in v], [b for _, b in v]).statistic
          for v in groups.values() if len(v) >= 10]
  vals = [v for v in vals if np.isfinite(v)]
  if absolute:
    vals = [abs(v) for v in vals]
  return float(np.mean(vals)) if vals else float("nan")



def _macro_partial(rows):
  """Macro per-assay Spearman after removing every counting shortcut.

  Controls: nucleotides changed, length delta, number of edit events, and the
  protein-event count -- the last of which on its own outscores every model
  here. What survives is signal no counting feature supplies.
  """
  groups = defaultdict(list)
  for row in rows:
    raw = row.get("predicted_fitness")
    if raw in (None, ""):
      return float("nan")
    try:
      pred, exp = float(raw), float(row["experimental_score"])
    except (TypeError, ValueError):
      continue
    if not (np.isfinite(pred) and np.isfinite(exp)):
      continue
    n_nt, len_delta, n_events = _count_features(row.get("hgvs_nt"))
    groups[row["score_set_urn"]].append(
      (pred, exp, n_nt, len_delta, n_events, _protein_events(row.get("hgvs_pro"))))
  vals = []
  for v in groups.values():
    if len(v) < 10:
      continue
    cols = list(zip(*v))
    try:
      r = partial_spearman(np.array(cols[0]), np.array(cols[1]),
                           [np.array(c) for c in cols[2:]])
    except Exception:
      continue
    if np.isfinite(r):
      vals.append(r)
  return float(np.mean(vals)) if vals else float("nan")


def mavedb():
  print("=" * 96)
  print("MaveDB  -  macro per-assay signed Spearman, 21,250 E. coli variants")
  print("=" * 96)
  print(f"{'arm':16s} {'estimator':14s} {'macro':>8s} {'per-tok':>8s} "
        f"{'nsyn==1':>8s} {'nsyn>=2':>8s} {'partial':>8s} {'kept':>6s}")
  print("-" * 96)
  best = (None, -9)
  best_abs = (None, -9)
  for arm, est, pat in MAVEDB_ARMS:
    hits = glob.glob(str(REPO / f"results/dnahnet/mavedb/{pat}/predictions.csv"))
    if not hits:
      continue
    problem = _check_arm_geometry(arm, hits[0])
    if problem:
      print(f"{arm:16s} {est:14s}  SKIPPED: {problem}")
      continue
    rows = list(csv.DictReader(open(hits[0])))
    macro = _macro(rows, "predicted_fitness")
    pertok = _macro(rows, "predicted_fitness_per_nt")
    one = _macro(rows, "predicted_fitness", lambda k: k == 1)
    two = _macro(rows, "predicted_fitness", lambda k: k >= 2)
    # Headline is the raw-score macro, full stop. This used to be
    # `max(macro, pertok)`, which picked whichever of two DIFFERENT metrics
    # happened to score higher, independently per arm -- so the "best BiSSM
    # arm" could be selected on a length-normalised score while its
    # competitors were judged on the raw one. ProteinGym selects on a single
    # metric (performance_DMS_benchmarks.py: final_average.idxmax()), and the
    # per-token column stays visible beside it for inspection.
    headline = macro
    if arm.startswith("BiSSM-BD") and headline > best[1]:
      best = (f"{arm} / {est}", headline)
    macro_abs = _macro(rows, "predicted_fitness", absolute=True)
    if np.isfinite(macro_abs) and macro_abs > best_abs[1]:
      best_abs = (f"{arm}", macro_abs)
    partial = _macro_partial(rows)
    kept = (partial / macro) if (np.isfinite(partial) and macro > 1e-9) else float("nan")
    kept_s = f"{kept*100:5.0f}%" if np.isfinite(kept) else "     -"
    print(f"{arm:16s} {est:14s} {macro:8.4f} {pertok:8.4f} {one:8.4f} {two:8.4f} "
          f"{partial:8.4f} {kept_s:>6s}")
  # The bar that matters. Sourced from the CANONICAL DATA FILE, not from some
  # model's predictions.csv -- this used to read `ussm-ar-bf16-seed1`, a
  # superseded no-prefix run, so the zero-parameter bar was hostage to a
  # directory that nothing else references any more and would vanish with it.
  rows = list(read_jsonl_gz(
    REPO / "data_cache/dnahnet/mavedb_ecoli_k12_21250.jsonl.gz"))
  groups = defaultdict(list)
  for row in rows:
    subs = protein_substitutions(row.get("hgvs_pro"))
    if subs is None or len(subs) != 1:
      continue
    score = BLOSUM62.get(subs[0])
    if score is None:
      continue
    groups[row["score_set_urn"]].append(
      (float(score), float(row["experimental_score"])))
  vals = [spearmanr([a for a, _ in v], [b for _, b in v]).statistic
          for v in groups.values() if len(v) >= 10]
  print("-" * 96)
  print(f"{'BLOSUM62':16s} {'0 parameters':14s} {'':8s} {'':8s} "
        f"{np.mean(vals):8.4f}  <- the bar in the honest stratum "
        f"({sum(1 for v in vals if v > 0)}/{len(vals)} assays positive)")
  # dnaHNet Fig 4B reports ABSOLUTE Spearman, so it is not comparable to the
  # signed column above. Print our best |rho| beside it or the row is
  # apples-to-oranges; and their paper never says macro or pooled, so the
  # comparison stays unresolved however careful we are on our side.
  print(f"{'dnaHNet':16s} {'published':14s} {0.3266:8.4f}  "
        f"(ABSOLUTE rho, 6.4e19 FLOP; 0.2601 at 8.0e18)")
  if best_abs[0]:
    print(f"{'  our best |rho|':16s} {best_abs[0]:14s} {best_abs[1]:8.4f}  "
          f"(the like-for-like figure; macro vs pooled unresolved in their paper)")
  print(f"\n  best BiSSM-BD: {best[0]} at {best[1]:.4f} macro")


def gb():
  print("\n" + "=" * 96)
  print("GenomicBenchmarks  -  fine-tune top-1 accuracy")
  print("=" * 96)
  per = defaultdict(list)
  for path in glob.glob(str(REPO / "results/caduceus/genomic_benchmarks_ft/gb-A-legacy-fulldata*.json")):
    for row in json.load(open(path))["tasks"]:
      per[row["task"]].append(row["accuracy"])
  # Fold in single-knob probes, selected on VALIDATION only.
  #
  # This used to select the probe whose TEST accuracy beat the baseline, and
  # then take the max over probes on TEST -- which is the same test-selected
  # hyperparameter defect the harness audit flagged, reintroduced at the
  # reporting layer. It inflates our own headline. A probe is now adopted only
  # if it beats the baseline's mean VALIDATION accuracy, and among candidates
  # we pick the highest VALIDATION, then report whatever test it got.
  per_val = defaultdict(list)
  for path in glob.glob(str(REPO / "results/caduceus/genomic_benchmarks_ft/gb-A-legacy-fulldata*.json")):
    for row in json.load(open(path))["tasks"]:
      v = row.get("val_accuracy")
      if v is not None:
        per_val[row["task"]].append(float(v))

  # Features that exploit a benchmark artifact rather than model DNA.
  # human_ensembl_regulatory is ~90% solvable from len(seq) alone, so a gain
  # driven by an explicit length feature is not a modelling result.
  LENGTH_FEATURE_KEYS = ("log_length", "length_bins")

  overrides = {}
  probe_globs = ("q-*.json", "gb-ocr-lr3e5-5seed*.json", "ocr-lr1e4-warmup-4s.json",
                 "gb-all-lr1e4-warmup*.json")
  for path in sorted({p for g in probe_globs
                      for p in glob.glob(str(REPO / "results/caduceus/genomic_benchmarks_ft" / g))}):
    payload = json.load(open(path))
    args = payload.get("args", {}) or {}
    uses_len = any(bool(args.get(k)) for k in LENGTH_FEATURE_KEYS)
    for row in payload["tasks"]:
      base_val = np.mean(per_val.get(row["task"], [np.nan]))
      probe_val = row.get("val_accuracy")
      if probe_val is None or not np.isfinite(base_val):
        continue
      probe_val = float(probe_val)
      if probe_val <= base_val:
        continue
      prev = overrides.get(row["task"])
      if prev is None or probe_val > prev[3]:
        overrides[row["task"]] = (row["accuracy"], Path(path).stem,
                                  len(row.get("accuracy_per_seed", [1])),
                                  probe_val, uses_len)
  ix = {c: i for i, c in enumerate(REFERENCE_COLUMNS)}
  print(f"{'task':34s} {'5-seed':>8s} {'probe':>8s} {'used':>8s} {'PS':>7s} leak")
  print("-" * 96)
  used = {}
  for task in sorted(per):
    base = float(np.mean(per[task]))
    ov = overrides.get(task)
    pick = ov[0] if ov else base
    used[task] = pick
    ps = REFERENCE[task][ix["ps"]]
    print(f"{task:34s} {base:8.4f} {(ov[0] if ov else float('nan')):8.4f} "
          f"{pick:8.4f} {ps:7.4f} {'LEAKY' if task in LEAKY_TASKS else ''}")
  clean = [t for t in used if t not in LEAKY_TASKS]
  for label, keys in (("8-task mean", list(used)), ("6 leak-free", clean)):
    ours = float(np.mean([used[t] for t in keys]))
    ps = float(np.mean([REFERENCE[t][ix["ps"]] for t in keys]))
    ph = float(np.mean([REFERENCE[t][ix["ph"]] for t in keys]))
    print(f"\n  {label:14s} ours {ours:.4f} | Caduceus-Ph {ph:.4f} | "
          f"Caduceus-PS {ps:.4f} | vs PS {ours - ps:+.4f}")
  if overrides:
    print("\n  probes folded in, VAL-GATED (task <- run, seeds):")
    for task, (acc, stem, ns, pv, uses_len) in overrides.items():
      mark = "  [LENGTH FEATURE]" if uses_len else ""
      print(f"    {task}: test {acc:.4f} (val {pv:.4f}) <- {stem} "
            f"({ns} seed(s)){mark}")
    print("  NOTE any 1-seed entry is provisional; the 5-seed baseline noise "
          "floor on these tasks is 0.005-0.008.")
    if any(o[4] for o in overrides.values()):
      print("  WARNING a [LENGTH FEATURE] probe feeds sequence length to the "
            "classifier.\n           human_ensembl_regulatory is ~90% solvable "
            "from len(seq) alone, so\n           that gain exploits a dataset "
            "artifact and must not be reported as a\n           modelling "
            "result. Quote the mean without it as well.")
      # Keep the SAME task sets as above -- a length-feature task is reverted to
      # its baseline, never dropped, or the comparison silently changes its
      # denominator and stops being comparable to the published means.
      def _no_len(task):
        ov = overrides.get(task)
        if ov is not None and ov[4]:
          return float(np.mean(per[task]))
        return used[task]
      for label, keys in (("8-task, no len", list(used)),
                          ("6 leak-free, no len", clean)):
        ours = float(np.mean([_no_len(t) for t in keys]))
        ps = float(np.mean([REFERENCE[t][ix["ps"]] for t in keys]))
        print(f"  {label:22s} ours {ours:.4f} | Caduceus-PS {ps:.4f} | "
              f"vs PS {ours - ps:+.4f}  ({len(keys)} tasks)")


if __name__ == "__main__":
  mavedb()
  gb()
