#!/usr/bin/env python3
"""Make a result file say what produced it, and refuse to hide a truncation.

WHY THIS EXISTS. Every published GenomicBenchmarks number in this project was
produced with `GB_MAX_TRAIN=20000`, which used 8.6% of the training data on the
worst-affected task. It went unnoticed for weeks because of three properties,
and every sibling defect the audit later found shared all three:

  1. the value arrived through the ENVIRONMENT (`bsub -env all` inherits the
     submitting shell), so it appears in no command line anyone typed;
  2. it was NEVER WRITTEN to the output, so the result looked complete;
  3. nothing compared what was used against what was available.

Enumerating bad defaults one at a time does not work -- the same cap was fixed
in `finetune.sh` and missed in `genomic_benchmarks.sh` on the same day. These
two helpers attack properties (2) and (3) generically instead.

USAGE

    from scripts.eval.provenance import stamp, assert_full_coverage

    assert_full_coverage(used=len(train_rows), available=n_train_full,
                         what="train rows", allow=args.max_train is not None)
    payload = stamp({"tasks": rows}, args)

`stamp` records the resolved argument namespace, the raw argv, the environment
variables that could have influenced the run, the git commit, and the host.
`assert_full_coverage` raises unless the run consumed everything it was given,
or the caller passes `allow=True` to say the truncation was deliberate -- in
which case the shortfall is recorded rather than silently accepted.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Environment names that have influenced a run here, or plausibly could. Bare
# names are the dangerous ones: GNU screen exports WINDOW=<n>, `bsub -env all`
# carried it in, and it silently passed `--window 0` until the flags were
# namespaced GB_*.
WATCHED_ENV = (
  "GB_MAX_TRAIN", "GB_MAX_TEST", "GB_ALLOW_CAPS", "GB_WINDOW", "GB_TASKS",
  "LIMIT", "WINDOW", "BATCH", "BATCH_SIZE", "BATCH_SIZES", "SEED", "SEEDS",
  "STEPS", "EPOCHS", "LENGTH", "LENGTHS", "MC_SAMPLES", "VAL_BATCHES",
  "WARMUP", "ITERS", "MAX_TRAIN", "MAX_TEST", "DEBUG", "SMOKE",
  "BD3LM_DATA_NUM_PROC", "CUDA_VISIBLE_DEVICES",
)


class CoverageError(RuntimeError):
  """A run silently used less of its input than was available."""


def _git_sha():
  try:
    return subprocess.run(
      ["git", "-C", str(REPO), "rev-parse", "HEAD"],
      capture_output=True, text=True, timeout=10).stdout.strip() or None
  except Exception:  # noqa: BLE001 - provenance must never break a run
    return None


def _git_dirty():
  try:
    out = subprocess.run(
      ["git", "-C", str(REPO), "status", "--porcelain"],
      capture_output=True, text=True, timeout=10).stdout.strip()
    return bool(out)
  except Exception:  # noqa: BLE001
    return None


def provenance(args=None, **extra):
  """The block that makes a result reproducible and its limits visible."""
  resolved = None
  if args is not None:
    resolved = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()}
  return {
    "argv": list(sys.argv),
    "resolved_args": resolved,
    "watched_env": {k: os.environ[k] for k in WATCHED_ENV if k in os.environ},
    "git_sha": _git_sha(),
    "git_dirty": _git_dirty(),
    "host": socket.gethostname(),
    "lsf_job_id": os.environ.get("LSB_JOBID"),
    "python": sys.version.split()[0],
    **extra,
  }


def stamp(payload, args=None, **extra):
  """Attach a `_provenance` block to a result dict, in place, and return it."""
  payload["_provenance"] = provenance(args, **extra)
  return payload


def assert_full_coverage(used, available, what, allow=False, record=None):
  """Refuse to silently consume less input than was offered.

  `allow=True` marks a deliberate truncation -- the caller passed an explicit
  cap -- and the shortfall is then recorded in `record` instead of raising.
  Passing neither a full pass nor `allow` is the exact failure mode that
  produced our GenomicBenchmarks results, so it is a hard error.
  """
  used, available = int(used), int(available)
  if used == available:
    return True
  fraction = used / available if available else 0.0
  message = (f"{what}: used {used:,} of {available:,} available "
             f"({100 * fraction:.1f}%)")
  if not allow:
    raise CoverageError(
      message + " -- this run would silently report a truncated result. Pass "
      "an explicit cap flag if you meant it.")
  if record is not None:
    record.setdefault("truncations", []).append(
      {"what": what, "used": used, "available": available,
       "fraction": fraction})
  print(f"WARNING: {message} (explicitly requested)", file=sys.stderr)
  return False


def write_json(path, payload, args=None, **extra):
  """Stamp and write atomically, so a killed job never leaves a partial file."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  stamp(payload, args, **extra)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)
  return path


if __name__ == "__main__":
  print(json.dumps(provenance(), indent=2))
