#!/usr/bin/env python3
"""Score the pinned dnaHNet DEG essentiality set with the checkpoint's objective.

Sign convention (AUROC is sign-sensitive, unlike the |Spearman| used for
MaveDB, so this is stated once and asserted at startup):

    essentiality_score = loss(knockout) - loss(wild type)
                       = log p(wild type) - log p(knockout)

A LARGER score means the stop-codon knockout hurt the sequence likelihood
more, i.e. MORE essential. That is the direction AUROC is computed in, and a
random predictor lands at 0.5.

Model loading, the two likelihood objectives and the tensor plumbing are
imported from ``score_mavedb.py`` rather than reimplemented; only the record
shape, the region masks and the metric block are new here. In particular the
EMA handling is inherited: ``load_checkpoint_model`` copies EMA weights only
when the checkpoint actually has them, so ema=0 checkpoints load cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.dnahnet.deg import (  # noqa: E402
  STOP_CASSETTE,
  WINDOW_LENGTH,
  knockout_from_record,
  read_jsonl_gz,
  summarize_deg_predictions,
)

DEFAULT_DATA = REPO / "data_cache/dnahnet/deg_essentiality.jsonl.gz"

#: Diagnostic region half-width around the cassette, in nucleotides. The
#: headline predictor is the whole-window difference, as published; the local
#: and downstream variants cost nothing extra (same forward pass) and show
#: whether any signal is being drowned by 8192 positions of shared noise.
LOCAL_RADIUS = 128

SCORE_FIELDS = (
  "essentiality_score",
  "essentiality_score_local",
  "essentiality_score_downstream",
)

CSV_FIELDS = [
  "organism_id", "organism_name", "accession", "locus_tag", "gene_name",
  "protein_id", "gene_start", "gene_end", "strand", "gene_length",
  "essential", "match_route", "window_start", "window_gc", "cds_offset",
  "cassette_start", "loss_type", "wt_loss", "ko_loss",
  "essentiality_score", "essentiality_score_local",
  "essentiality_score_downstream", "wt_loss_per_nt", "ko_loss_per_nt",
  "scored_tokens",
]


def _assemble(record, wt_loss, ko_loss, wt_local, ko_local, wt_down, ko_down,
              tokens, objective):
  """Single place where the sign convention is applied."""
  return {
    "organism_id": record["organism_id"],
    "organism_name": record["organism_name"],
    "accession": record["accession"],
    "locus_tag": record["locus_tag"],
    "gene_name": record["gene_name"],
    "protein_id": record["protein_id"],
    "gene_start": record["gene_start"],
    "gene_end": record["gene_end"],
    "strand": record["strand"],
    "gene_length": record["gene_length"],
    "essential": int(record["essential"]),
    "match_route": record["match_route"],
    "window_start": record["window_start"],
    "window_gc": record["window_gc"],
    "cds_offset": record["cds_offset"],
    "cassette_start": record["cassette_start"],
    "loss_type": objective,
    "wt_loss": wt_loss,
    "ko_loss": ko_loss,
    # More essential => knockout is more surprising => larger score.
    "essentiality_score": ko_loss - wt_loss,
    "essentiality_score_local": ko_local - wt_local,
    "essentiality_score_downstream": ko_down - wt_down,
    "wt_loss_per_nt": wt_loss / tokens,
    "ko_loss_per_nt": ko_loss / tokens,
    "scored_tokens": tokens,
  }


def assert_sign_convention():
  """A knockout that costs more likelihood must yield a positive score."""
  record = {
    "organism_id": "x", "organism_name": "x", "accession": "x",
    "locus_tag": "x", "gene_name": "x", "protein_id": "x",
    "gene_start": 0, "gene_end": 30, "strand": 1, "gene_length": 30,
    "essential": 1, "match_route": "name", "window_start": 0,
    "window_gc": 0.5, "cds_offset": 0, "cassette_start": 12,
  }
  hurt = _assemble(record, 100.0, 140.0, 1.0, 5.0, 10.0, 20.0, 8192, "test")
  harmless = _assemble(record, 100.0, 100.5, 1.0, 1.1, 10.0, 10.2, 8192, "test")
  if not hurt["essentiality_score"] > harmless["essentiality_score"] > 0:
    raise AssertionError(
      "Sign convention is inverted: a costlier knockout must score higher")
  from scripts.eval.dnahnet.deg import roc_auc
  if roc_auc([0, 1], [0.0, 1.0]) != 1.0 or roc_auc([0, 1], [1.0, 0.0]) != 0.0:
    raise AssertionError("roc_auc is not sign-sensitive as expected")


def iter_records(path: Path, max_genes=None, organisms=None,
                 sample_genes=None, sample_seed=0, drop_sequence=False):
  """Streams benchmark records, optionally filtered and subsampled.

  ``--max-genes`` truncates in genome order, which is NOT representative:
  bacterial chromosomes put the ribosomal and replication machinery near the
  origin, so the first N genes are strongly enriched for essentials and share
  a distinctive neighbourhood. ``--sample-genes`` draws a seeded uniform
  sample instead and is what any subset run should use.
  """
  wanted = set(organisms) if organisms else None

  def filtered():
    for record in read_jsonl_gz(path):
      if wanted is None or record["organism_id"] in wanted:
        if drop_sequence:
          # The 8192-nt window is 90% of the record. Baseline-only runs never
          # touch it, and keeping 184k of them would need ~1.5 GB -- past the
          # head node's ~1.7 GB per-process cgroup.
          record.pop("window_sequence", None)
        yield record

  if sample_genes is not None:
    import random

    reservoir = []
    rng = random.Random(sample_seed)
    for index, record in enumerate(filtered()):
      if index < sample_genes:
        reservoir.append(record)
      else:
        position = rng.randrange(index + 1)
        if position < sample_genes:
          reservoir[position] = record
    # Restore genome order so runs remain comparable record by record.
    reservoir.sort(key=lambda row: (
      row["organism_id"], row["accession"], row["gene_start"]))
    yield from reservoir
    return

  taken = 0
  for record in filtered():
    yield record
    taken += 1
    if max_genes is not None and taken >= max_genes:
      return


def score_batch(model, tokenizer, records, model_length, mc_samples, epsilon,
                generator, local_radius=LOCAL_RADIUS):
  """Mirrors ``score_mavedb.score_batch`` with DEG's three region masks."""
  import torch

  from scripts.eval.dnahnet.score_mavedb import (
    _exact_ar_losses, _loss_from_fixed_corruption, build_pair_tensors)

  pairs = [{"wt_sequence": record["window_sequence"],
            "mut_sequence": knockout_from_record(record)}
           for record in records]
  x0_cpu, token_mask_cpu = build_pair_tensors(pairs, tokenizer, model_length)
  bos_rows = model._bos_rows(x0_cpu)
  if bos_rows.any():
    token_mask_cpu[bos_rows, 0] = False

  is_ar = str(model.parameterization) == "ar"
  if is_ar:
    token_mask_cpu[:, 0] = False

  local_cpu = torch.zeros_like(token_mask_cpu)
  downstream_cpu = torch.zeros_like(token_mask_cpu)
  for index, record in enumerate(records):
    start = int(record["cassette_start"])
    end = start + len(record.get("cassette", STOP_CASSETTE))
    low = max(0, start - local_radius)
    high = min(model_length, end + local_radius)
    for row in (2 * index, 2 * index + 1):
      local_cpu[row, low:high] = True
      downstream_cpu[row, start:] = True
  local_cpu &= token_mask_cpu
  downstream_cpu &= token_mask_cpu

  x0 = x0_cpu.to(model.device)
  token_mask = token_mask_cpu.to(model.device)
  local_mask = local_cpu.to(model.device)
  downstream_mask = downstream_cpu.to(model.device)

  totals = torch.zeros(x0.shape[0], dtype=torch.float64, device=model.device)
  local_totals = torch.zeros_like(totals)
  downstream_totals = torch.zeros_like(totals)

  if is_ar:
    losses = _exact_ar_losses(model, x0).double()
    totals = (losses * token_mask).sum(dim=-1)
    local_totals = (losses * local_mask).sum(dim=-1)
    downstream_totals = (losses * downstream_mask).sum(dim=-1)
  else:
    for _ in range(mc_samples):
      pair_t = torch.rand((len(records), 1), generator=generator)
      pair_t = pair_t * (1.0 - epsilon) + epsilon
      t = pair_t.repeat_interleave(2, dim=0).to(model.device)
      common_uniform = torch.rand(
        (len(records), model_length), generator=generator)
      common_uniform = common_uniform.repeat_interleave(
        2, dim=0).to(model.device)
      losses = _loss_from_fixed_corruption(
        model, x0, t, common_uniform).double()
      totals += (losses * token_mask).sum(dim=-1)
      local_totals += (losses * local_mask).sum(dim=-1)
      downstream_totals += (losses * downstream_mask).sum(dim=-1)
    totals /= mc_samples
    local_totals /= mc_samples
    downstream_totals /= mc_samples

  totals = totals.cpu()
  local_totals = local_totals.cpu()
  downstream_totals = downstream_totals.cpu()
  objective = "exact_ar_nll" if is_ar else "paired_diffusion_nelbo"

  scored = []
  for index, record in enumerate(records):
    wt, ko = 2 * index, 2 * index + 1
    scored.append(_assemble(
      record,
      float(totals[wt]), float(totals[ko]),
      float(local_totals[wt]), float(local_totals[ko]),
      float(downstream_totals[wt]), float(downstream_totals[ko]),
      int(token_mask_cpu[wt].sum()), objective))
  return scored


def baseline_records(records):
  """Model-free rows: every score variant is zero, so only baselines matter."""
  rows = []
  for record in records:
    rows.append(_assemble(
      record, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, WINDOW_LENGTH, "none"))
  return rows


def _atomic_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
  os.replace(temporary, path)


def _atomic_csv(path: Path, records):
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", newline="", dir=path.parent,
      delete=False) as handle:
    writer = csv.DictWriter(
      handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    temporary = Path(handle.name)
  os.replace(temporary, path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--batch-size", type=int, default=8,
                      help="Wild-type/knockout pairs per GPU batch")
  parser.add_argument("--mc-samples", type=int, default=128,
                      help="Diffusion Monte Carlo samples per pair. The "
                           "MaveDB MC curve (0.0966/0.1242/0.1318/0.1360 at "
                           "8/32/64/128) puts diminishing returns near 128. "
                           "Ignored by AR checkpoints, whose likelihood is "
                           "exact.")
  parser.add_argument("--model-length", type=int, default=WINDOW_LENGTH)
  parser.add_argument("--local-radius", type=int, default=LOCAL_RADIUS)
  parser.add_argument("--epsilon", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument(
    "--max-genes", type=int,
    help="Truncate in genome order. Not representative -- prefer "
         "--sample-genes for any subset you intend to interpret.")
  parser.add_argument(
    "--sample-genes", type=int,
    help="Seeded uniform sample of this many genes (reservoir sampling over "
         "the stream, then re-sorted into genome order)")
  parser.add_argument("--sample-seed", type=int, default=0)
  parser.add_argument("--organism", action="append")
  parser.add_argument("--baseline-seed", type=int, default=0)
  parser.add_argument(
    "--checkpoint-every", type=int, default=200,
    help="Flush predictions.partial.csv every N batches, so a long BD run "
         "that is killed at the wall limit does not lose everything")
  parser.add_argument(
    "--baselines-only", action="store_true",
    help="Compute only the model-free baselines and the random control. No "
         "GPU and no checkpoint required; runs on the head node.")
  parser.add_argument("--reverse-off", action="store_true",
                      help="Ablate a bissm checkpoint's in-block reverse scan")
  args = parser.parse_args()

  assert_sign_convention()

  if args.batch_size <= 0 or args.mc_samples <= 0:
    parser.error("batch-size and mc-samples must be positive")
  if args.checkpoint_every <= 0:
    parser.error("checkpoint-every must be positive")
  if not 0 < args.epsilon < 1:
    parser.error("epsilon must be in (0, 1)")
  if not args.baselines_only and args.checkpoint is None:
    parser.error("--checkpoint is required unless --baselines-only is set")

  if args.max_genes is not None and args.sample_genes is not None:
    parser.error("Use --max-genes or --sample-genes, not both")
  records = list(iter_records(
    args.data, args.max_genes, args.organism,
    args.sample_genes, args.sample_seed,
    drop_sequence=args.baselines_only))
  if not records:
    raise ValueError("No benchmark records were loaded")
  if not args.baselines_only:
    lengths = {len(record["window_sequence"]) for record in records}
    if lengths != {args.model_length}:
      raise ValueError(
        f"Windows are {sorted(lengths)} nt but model-length is "
        f"{args.model_length}")
  labels = {int(record["essential"]) for record in records}
  if labels != {0, 1}:
    raise ValueError(
      f"AUROC needs both classes present; found labels {sorted(labels)}. "
      "Widen --max-genes or drop --organism.")

  summary_extra = {}
  if args.baselines_only:
    scored = baseline_records(records)
    summary_extra = {
      "label": args.label,
      "checkpoint": None,
      "parameterization": "none",
      "score_definition": "no model; baselines and random control only",
    }
  else:
    import torch

    from scripts.eval.dnahnet.score_mavedb import load_checkpoint_model

    if not torch.cuda.is_available():
      raise RuntimeError("DEG checkpoint scoring requires a CUDA GPU")
    device = torch.device("cuda")
    model, tokenizer, config, global_step = load_checkpoint_model(
      args.checkpoint, args.model_length, args.batch_size * 2, device,
      reverse_off=args.reverse_off)
    block_size = int(config.block_size)
    if args.model_length % block_size:
      raise ValueError(
        f"model_length={args.model_length} is not divisible by "
        f"block_size={block_size}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    is_ar = str(model.parameterization) == "ar"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "predictions.partial.csv"
    started = time.monotonic()
    scored = []
    with torch.inference_mode():
      for index, start in enumerate(
          range(0, len(records), args.batch_size), start=1):
        scored.extend(score_batch(
          model=model,
          tokenizer=tokenizer,
          records=records[start:start + args.batch_size],
          model_length=args.model_length,
          mc_samples=args.mc_samples,
          epsilon=args.epsilon,
          generator=generator,
          local_radius=args.local_radius))
        elapsed = time.monotonic() - started
        rate = len(scored) / elapsed
        remaining = (len(records) - len(scored)) / rate if rate else float("nan")
        print(
          f"[{args.label}] scored {len(scored)}/{len(records)} genes "
          f"| {rate:.3f} genes/s | eta {remaining / 3600:.2f} h", flush=True)
        # A full BD arm is a multi-hundred-hour job. Flush partial results so
        # a wall-clock kill or a node failure does not discard all of it.
        if index % args.checkpoint_every == 0:
          _atomic_csv(partial_path, scored)
    partial_path.unlink(missing_ok=True)
    summary_extra = {
      "label": args.label,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_global_step": global_step,
      "backbone": str(config.algo.backbone),
      "reverse_off": bool(args.reverse_off),
      "block_size": block_size,
      "parameterization": str(model.parameterization),
      "mc_samples": 1 if is_ar else args.mc_samples,
      "requested_mc_samples": args.mc_samples,
      "score_definition": (
        "exact NLL(knockout) - exact NLL(wild type)" if is_ar else
        "paired NELBO(knockout) - paired NELBO(wild type)"),
    }

  summary = summarize_deg_predictions(
    scored, score_fields=SCORE_FIELDS,
    headline_field="essentiality_score",
    baseline_seed=args.baseline_seed)
  summary.update(summary_extra)
  summary.update({
    "data": str(args.data.resolve()),
    "model_length": args.model_length,
    "local_radius": args.local_radius,
    "epsilon": args.epsilon,
    "seed": args.seed,
    "subset": (
      "full" if args.max_genes is None and args.sample_genes is None else
      f"max_genes={args.max_genes}" if args.max_genes is not None else
      f"sample_genes={args.sample_genes} (seed {args.sample_seed})"),
    "organism_filter": args.organism,
    "stop_cassette": STOP_CASSETTE,
    "sign_convention": (
      "essentiality_score = loss(knockout) - loss(wild type); larger is "
      "more essential"),
  })
  metadata_path = Path(str(args.data) + ".metadata.json")
  if metadata_path.exists():
    with metadata_path.open(encoding="utf-8") as handle:
      prepared = json.load(handle)
    summary["data_sha256"] = prepared.get("output_sha256")
    summary["data_protocol"] = prepared.get("protocol")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  _atomic_csv(args.output_dir / "predictions.csv", scored)
  _atomic_json(args.output_dir / "summary.json", summary)
  compact = {key: summary[key] for key in (
    "label", "num_genes", "num_organisms", "num_essential", "base_rate",
    "macro_auroc", "macro_auroc_std", "pooled_auroc",
    "macro_auroc_if_sign_flipped", "score_variants", "baselines")}
  print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
