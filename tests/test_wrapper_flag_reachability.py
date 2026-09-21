"""Every science lever a python entrypoint accepts must be reachable from its wrapper.

Three flags have now been found accepted by the python and un-emittable by the shell
that launches it in production:

  --epsilon      in mavedb_score.sh   (found 2026-09-19; forwarding it was worth
                                       +31% confound-stripped on MaveDB)
  --right-flank  in mavedb_score.sh   (found 2026-09-20)
  --epsilon      in deg_score.sh      (found 2026-09-20; DEG's noise band had
                                       therefore never been swept at all)

The failure is invisible: the run succeeds, the summary looks clean, and the setting
simply never reaches the process. This test is the ratchet. A flag may be unreachable
only if it is listed in ALLOWED below with a reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# wrapper -> {flag: why it is fine for this wrapper never to emit it}
ALLOWED: dict[str, dict[str, str]] = {
    "scripts/eval/dnahnet/mavedb_score.sh": {
        "--only-urn": "single-assay debug helper, run by hand",
    },
    "scripts/eval/caduceus/genomic_benchmarks.sh": {
        "--output-dir": "wrapper owns the results layout",
        "--window-cap": "probe path never truncates; window comes from the task",
    },
    "scripts/eval/caduceus/finetune.sh": {
        "--seed": "superseded by --seeds, which the wrapper does emit",
        "--output-dir": "wrapper owns the results layout",
        "--allow-window-mismatch": "an override that must stay deliberate",
        "--window-cap": "ditto",
        "--slow-encode": "debug fallback",
        "--pad-multiple": "not swept",
        "--head-layernorm": "not swept",
        "--head-warmup-steps": "not swept",
        "--honour-no-weight-decay": "not swept",
    },
    "scripts/eval/dnahnet/deg_score.sh": {
        "--baseline-seed": "derived from SEED inside the wrapper",
    },
}

# Only wrappers that produce published numbers are policed. Smoke and micro-benchmark
# scripts hardcode their arguments on purpose.
POLICED = sorted(ALLOWED)


def _argparse_flags(py: Path) -> set[str]:
    tree = ast.parse(py.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    out.add(arg.value)
    return out


@pytest.mark.parametrize("wrapper", POLICED)
def test_every_python_flag_is_reachable_from_its_wrapper(wrapper: str) -> None:
    sh = REPO / wrapper
    text = sh.read_text()
    targets = {m for m in re.findall(r"scripts/[\w/.-]+\.py", text)}
    assert targets, f"{wrapper} launches no python entrypoint"

    emitted = set(re.findall(r"--[a-z0-9][a-z0-9-]*", text))
    allowed = ALLOWED[wrapper]

    for target in sorted(targets):
        py = REPO / target
        if not py.exists():
            continue
        unreachable = sorted(_argparse_flags(py) - emitted - set(allowed))
        assert not unreachable, (
            f"{wrapper} can never pass {unreachable} to {target}. "
            "Forward it, or add it to ALLOWED with a reason. A flag the wrapper "
            "cannot emit is a setting that silently never takes effect."
        )


def test_the_three_regressions_that_motivated_this_test_stay_fixed() -> None:
    mavedb = (REPO / "scripts/eval/dnahnet/mavedb_score.sh").read_text()
    assert "--epsilon" in mavedb and "EPSILON" in mavedb
    assert "--right-flank" in mavedb and "RIGHT_FLANK" in mavedb
    deg = (REPO / "scripts/eval/dnahnet/deg_score.sh").read_text()
    assert "--epsilon" in deg and "EPSILON" in deg


def test_allowlist_entries_all_name_a_real_flag() -> None:
    """An ALLOWED entry for a flag that no longer exists hides a later regression."""
    for wrapper, entries in ALLOWED.items():
        text = (REPO / wrapper).read_text()
        known: set[str] = set()
        for target in re.findall(r"scripts/[\w/.-]+\.py", text):
            py = REPO / target
            if py.exists():
                known |= _argparse_flags(py)
        stale = sorted(set(entries) - known)
        assert not stale, f"{wrapper}: ALLOWED lists {stale}, which no entrypoint accepts"
