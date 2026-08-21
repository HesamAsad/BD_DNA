#!/usr/bin/env python3
"""CPU test for the accumulating write in `measured_flops_sweep`.

The sweep used to overwrite its output, so extending the curve past 32768 meant
destroying the 2048..32768 rows that cost GPU hours to produce. It now merges.
That makes the write the risky part rather than the arithmetic, and the risk is
silent: a bad merge loses rows that nothing else records, and the loss only
shows up as a hole in a figure much later.

So this asserts the three things a merge has to get right -- replace by
(arm, length), keep untouched rows, keep the file parseable -- plus the two
places the mixed-batch file can go wrong downstream: rows must carry their own
batch, and `scaling_curves` must divide by that one and not by the top-level
copy. No GPU: the payloads are synthetic, and the one real-data case reads
files already on disk.

  python scripts/smoke/test_flop_merge.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.measured_flops_sweep import (  # noqa: E402
  merge_rows, write_payload)

PASS, FAIL = [], []


def check(name, condition, detail=""):
  (PASS if condition else FAIL).append(name)
  print(f"  {'ok  ' if condition else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))


def row(arm, length, batch, tflop, **extra):
  return {"arm": arm, "length": length, "batch": batch, "block_size": 256,
          "counted_tflop": tflop, "analytic_attention_tflop": 0.0,
          "total_tflop": tflop, **extra}


def test_merge_rows():
  """The pure function: replace by key, preserve everything else."""
  print("\n[1] merge_rows: replace by (arm, length), keep the rest")
  existing = [row("bissm", 2048, 2, 8.26), row("bissm", 32768, 2, 140.04),
              row("dit", 2048, 2, 8.50),
              {"arm": "dit", "length": 32768, "batch": 2,
               "error": "OutOfMemoryError: CUDA out of memory."}]
  fresh = [row("bissm", 32768, 1, 70.02), row("bissm", 65536, 1, 140.60)]
  merged = merge_rows(existing, fresh)

  index = {(r["arm"], r["length"]): r for r in merged}
  check("no rows lost", len(merged) == 5, f"{len(merged)} rows")
  check("touched row replaced",
        index[("bissm", 32768)]["counted_tflop"] == 70.02
        and index[("bissm", 32768)]["batch"] == 1)
  check("untouched row of the same arm survives",
        index[("bissm", 2048)]["counted_tflop"] == 8.26)
  check("untouched row of another arm survives",
        index[("dit", 2048)]["counted_tflop"] == 8.50)
  check("failed row the new sweep did not rerun survives verbatim",
        index[("dit", 32768)]["error"].startswith("OutOfMemoryError"))
  check("new point appended", ("bissm", 65536) in index)
  check("replacement keeps its original position",
        [(r["arm"], r["length"]) for r in merged][:4]
        == [("bissm", 2048), ("bissm", 32768), ("dit", 2048), ("dit", 32768)])
  check("no aliasing: merged rows are the input objects, unmutated",
        existing[1]["counted_tflop"] == 140.04)

  again = merge_rows(merged, fresh)
  check("idempotent: merging the same rows twice changes nothing",
        again == merged)


def test_write_payload(tmp):
  """The on-disk half: merge by default, --no-merge still clobbers."""
  print("\n[2] write_payload: default merges, no-merge overwrites")
  out = tmp / "measured_flops.json"

  first = [row("bissm", 2048, 2, 8.26), row("ussm-ar", 2048, 2, 3.56)]
  write_payload(out, first, batch=2, block_size=256)
  check("writes a fresh file when none exists",
        json.loads(out.read_text())["rows"] == first)

  second = [row("bissm", 65536, 1, 140.60), row("ussm-ar", 65536, 1, 56.96)]
  payload = write_payload(out, second, batch=1, block_size=256)
  on_disk = json.loads(out.read_text())
  check("second run did not destroy the first", len(on_disk["rows"]) == 4)
  check("returned payload matches what landed on disk",
        payload["rows"] == on_disk["rows"])
  check("every row carries its own batch",
        all("batch" in r for r in on_disk["rows"]))
  check("mixed batches are declared at the top level",
        on_disk["batches"] == [1, 2], str(on_disk.get("batches")))
  check("top-level batch is the most recent run", on_disk["batch"] == 1)
  check("note warns the top-level batch is not the one to divide by",
        "ROW's batch" in on_disk["note"])

  write_payload(out, second, batch=1, block_size=256, merge=False)
  check("--no-merge still overwrites wholesale",
        json.loads(out.read_text())["rows"] == second)

  # A file we cannot parse must not be silently replaced, and the fresh rows
  # (GPU hours) must not evaporate with the error.
  broken = tmp / "broken.json"
  broken.write_text("{not json")
  try:
    write_payload(broken, first, batch=2, block_size=256)
    check("unparseable existing file raises", False)
  except RuntimeError:
    check("unparseable existing file raises", True)
  check("unparseable file left untouched", broken.read_text() == "{not json")
  rescue = tmp / "broken.unmerged.json"
  check("fresh rows parked next to it", rescue.exists()
        and json.loads(rescue.read_text())["rows"] == first)


def normalise(payload):
  """The FIX-2 read path from scaling_curves.py, isolated for testing."""
  out = {}
  for r in payload.get("rows", []):
    if "counted_tflop" not in r:
      continue
    batch = r.get("batch") or payload.get("batch") or 1
    out.setdefault(r["arm"], {})[r["length"]] = r["counted_tflop"] / batch
  return out


def test_per_row_batch():
  print("\n[3] per-row batch: the top-level value cannot scale a mixed file")
  payload = {"batch": 1, "rows": [row("bissm", 32768, 2, 140.037825748992),
                                  row("bissm", 65536, 1, 140.599706382336)]}
  per_seq = normalise(payload)["bissm"]
  check("batch-2 row divided by 2", abs(per_seq[32768] - 70.018912874496) < 1e-9)
  check("batch-1 row divided by 1", abs(per_seq[65536] - 140.599706382336) < 1e-9)
  check("doubling length doubles per-sequence FLOPs across the batch change",
        abs(per_seq[65536] / per_seq[32768] - 2.0) < 0.01,
        f"ratio {per_seq[65536] / per_seq[32768]:.4f}")

  legacy = {"batch": 2, "rows": [{"arm": "bissm", "length": 2048,
                                  "counted_tflop": 8.260539113472}]}
  check("falls back to top-level batch for a row that lacks one",
        abs(normalise(legacy)["bissm"][2048] - 4.130269556736) < 1e-9)


def test_real_files(tmp):
  """End to end on the actual measurements, without touching the repo copy."""
  print("\n[4] real payloads: 2048..32768 (batch 2) + 65536,131072 (batch 1)")
  base = REPO / "results" / "sizing" / "measured_flops.json"
  extras = [REPO / "results" / "sizing" / "measured_flops_L65536.json",
            REPO / "results" / "sizing" / "measured_flops_L131072.json"]
  if not base.exists() or not all(p.exists() for p in extras):
    print("  skip -- measured payloads not on disk")
    return

  out = tmp / "real_merged.json"
  out.write_text(base.read_text())
  before = json.loads(base.read_text())["rows"]
  for path in extras:
    side = json.loads(path.read_text())
    write_payload(out, side["rows"], batch=side["batch"],
                  block_size=side["block_size"])
  merged = json.loads(out.read_text())

  check("all 20 original rows still present",
        len(merged["rows"]) == len(before) + 4,
        f"{len(before)} + 4 -> {len(merged['rows'])}")
  check("the dit OOM row survived the merge",
        any(r.get("arm") == "dit" and "error" in r for r in merged["rows"]))
  check("both batches present", merged["batches"] == [1, 2])

  per_seq = normalise(merged)
  for arm in ("bissm", "ussm-ar"):
    lengths = sorted(per_seq[arm])
    check(f"{arm}: reaches 2^17", lengths[-1] == 131072, str(lengths))
    check(f"{arm}: 2^11..2^17 all on one curve", len(lengths) == 7)
    ratios = [per_seq[arm][b] / per_seq[arm][a]
              for a, b in zip(lengths, lengths[1:])]
    # Roughly 2x per doubling, but NOT exactly: bissm carries an
    # O(num_blocks^2) boundary-prefill term (scaling_curves.flops_per_sequence
    # line 69) that makes it mildly superlinear, and both arms carry O(1)
    # embedding work. A band, not an equality.
    check(f"{arm}: each doubling is near-2x", all(1.95 < r < 2.10 for r in ratios),
          " ".join(f"{r:.3f}" for r in ratios))
    # The seam: 32768 was counted at batch 2, 65536 at batch 1. If the batch
    # were mishandled this one ratio would jump to ~1 or ~4 while its
    # neighbours stayed near 2, so compare it to them rather than to a constant.
    seam = ratios[lengths.index(32768)]
    neighbours = (ratios[lengths.index(32768) - 1]
                  + ratios[lengths.index(32768) + 1]) / 2
    check(f"{arm}: no discontinuity where the micro batch changes",
          abs(seam - neighbours) < 0.02,
          f"seam {seam:.4f} vs neighbours {neighbours:.4f}")

  # The seam itself: 32768 was counted at batch 2, 65536 at batch 1. Under the
  # old single-top-level-batch read one of the two is off by exactly 2x.
  naive = {r["length"]: r["counted_tflop"] / merged["batch"]
           for r in merged["rows"] if r.get("arm") == "bissm"
           and "counted_tflop" in r}
  check("old code would have put a 2x step at the batch seam",
        abs(naive[65536] / naive[32768] - 2.0) > 0.9,
        f"naive ratio {naive[65536] / naive[32768]:.3f} vs true 2.008")


def main():
  with tempfile.TemporaryDirectory() as name:
    tmp = Path(name)
    test_merge_rows()
    test_write_payload(tmp)
    test_per_row_batch()
    test_real_files(tmp)
  print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
  for name in FAIL:
    print(f"  FAILED: {name}")
  return 1 if FAIL else 0


if __name__ == "__main__":
  raise SystemExit(main())
