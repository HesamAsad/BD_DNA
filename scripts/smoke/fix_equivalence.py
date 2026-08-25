#!/usr/bin/env python
"""Numerics gate for the throughput fixes in `scripts/smoke/fixes/*.patch`.

Every fix in that directory claims to be a pure time/memory change. This script
is the proof obligation. It does NOT reimplement the fixes -- it builds a
shadow copy of the repository, applies the real patch files into it, and then
runs the SAME inputs through the unpatched tree and the patched tree in
**float64 on CPU**, comparing the loss and every parameter gradient.

float64 matters: at float32 an association-order change of the size these
fixes could plausibly introduce hides under the rounding of the comparison
itself. At float64 an exactly-equal path stays exactly equal, so the default
tolerance is ZERO -- `--tol 0` means bitwise, and any nonzero max-|delta| is a
finding, not noise.

Cases
-----
  f2f3-ssm   BiSSM end-to-end. Covers F2 (branchless BOS + host-mirrored eps
             bounds, diffusion.py) and F3 (cached carry mask,
             mamba2_segment.py) together, because both are on the SSM step's
             hot path. Expected: bitwise.
  f2f3-ussm  Same for the unidirectional SSM arm.
  f5f6-dit   Transformer-BD end-to-end with `attn_backend=sdpa`. Covers F6
             (trim the x_0 half before the head) and F5 (one packed qkv GEMM
             instead of two plus a `cat`). F6 is exact by row-independence;
             F5 is exact in exact arithmetic but its projection is ONE GEMM
             where the old path ran two, so its float error is a property of
             the BLAS, not of the algebra. That is the number this case
             measures. A nonzero result here at float64/CPU is a real
             reassociation; the same case must be re-run on the GPU in the
             training dtype before F5 is landed.
  f1-ckpt    `checkpoint_boundary_prefill` on vs off, both in the UNPATCHED
             tree, at float64. F1 only changes which of these two branches is
             chosen by default, so this is the whole of F1's numerics risk.
             `torch.utils.checkpoint` recompute is exact, so: bitwise.
  f4-graph   `graph_boundary_prefill` on vs off, both in the PATCHED tree.
             CUDA only -- a graph cannot be captured on CPU, and the claim
             being tested ("the replay issues the same kernels in the same
             order") is only meaningful on a device.

Usage
-----
  python scripts/smoke/fix_equivalence.py                  # all CPU cases
  python scripts/smoke/fix_equivalence.py --cases f5f6-dit
  python scripts/smoke/fix_equivalence.py --tol 1e-12      # loosen, and say so

The two GPU-only obligations, which a head node cannot discharge and which
must be run before F5 or F4 lands. Both need one GPU; neither trains anything:

  # F5: is one M=2bn GEMM bitwise equal to two M=bn GEMMs, in the training
  # dtype, on the actual BLAS? Nothing but a measurement can answer this.
  python scripts/smoke/fix_equivalence.py --cases f5f6-dit \
      --device cuda --dtype float32 --length 512 --block-size 256

  # F4: does the captured graph reproduce the eager step exactly?
  python scripts/smoke/fix_equivalence.py --cases f4-graph \
      --device cuda --dtype float32 --length 1024 --block-size 256

Defaults are CPU. Submits nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Two different roots, and the difference matters. `REPO` follows symlinks and
# is the real working tree -- what the DRIVER patches from. `TREE` does NOT
# resolve, so when this file is reached through the shadow tree's symlink it
# still names the shadow: what the WORKER must import and read configs from.
# Resolving here would silently make the "patched" worker import the unpatched
# modules and compare a tree against itself.
REPO = Path(__file__).resolve().parents[2]
TREE = Path(__file__).absolute().parents[2]
PATCH_DIR = REPO / "scripts" / "smoke" / "fixes"
if str(TREE) not in sys.path:
  sys.path.insert(0, str(TREE))

# Files any patch in `fixes/` touches. They become real copies in the shadow
# tree; everything else is symlinked, so the shadow costs nothing to build and
# cannot be written through into the working tree.
PATCHED_FILES = (
  "diffusion.py",
  "models/bidirectional_ssm.py",
  "models/mamba2_segment.py",
  "models/dit.py",
  "configs/model/small_bissm.yaml",
  "configs/model/small_ussm.yaml",
  "scripts/smoke/sizing_sweep.py",
  "scripts/smoke/flop_kernel_trace.py",
)

CASES = {
  # name: (model config, algo config, overrides applied on top)
  "f2f3-ssm": ("small_bissm", "bd3lm_bissm", ()),
  "f2f3-ussm": ("small_ussm", "bd3lm_ussm", ()),
  "f5f6-dit": ("small", "bd3lm", ("model.attn_backend=sdpa",)),
}


# --------------------------------------------------------------------------
# worker: runs inside whichever tree it was launched from
# --------------------------------------------------------------------------

def _build(model_cfg, algo_cfg, extra, length, block_size, batch_size,
           checkpoint_prefill, graph_prefill=None):
  import hydra
  import main  # noqa: F401  registers the project's OmegaConf resolvers
  import torch
  from dataloader import DNATokenizer
  from diffusion import Diffusion

  root = TREE  # the shadow tree's configs when running as the patched worker
  overrides = [
    f"model={model_cfg}", f"algo={algo_cfg}", "data=carbon-prokaryote",
    f"model.length={length}", f"block_size={block_size}",
    f"loader.batch_size={batch_size}", f"loader.eval_batch_size={batch_size}",
    "loader.global_batch_size=64", "training.ema=0",
    "trainer.accumulate_grad_batches=1",
    # Narrow and shallow: neither width nor depth changes WHICH operations
    # run, and a float64 CPU backward at production width is minutes per case.
    "model.hidden_size=64", "model.n_blocks=2",
  ]
  if model_cfg != "small":
    overrides += ["model.ssm_head_dim=32", "model.active_blocks=all"]
    if checkpoint_prefill is not None:
      overrides.append(
        "model.checkpoint_boundary_prefill="
        f"{str(bool(checkpoint_prefill)).lower()}")
    if graph_prefill is not None:
      overrides.append(
        f"model.graph_boundary_prefill={str(bool(graph_prefill)).lower()}")
  else:
    overrides += ["model.n_heads=2"]
  overrides.extend(extra)
  with hydra.initialize_config_dir(
      version_base=None, config_dir=str(root / "configs")):
    config = hydra.compose(config_name="config", overrides=overrides)
  torch.manual_seed(0)
  model = Diffusion(config, DNATokenizer())
  return model


def _inputs(model, batch_size, length, variant):
  """Token batch whose BOS pattern exercises all three mask populations.

  F2 replaces `if bos_rows.any(): tokens[bos_rows, 0] = ...` with an
  unconditional `torch.where`. The two spellings can only differ on rows the
  mask excludes, so the test has to contain rows of both kinds -- and the
  all-True / all-False extremes, which are the two cases where the old code
  took a different branch.
  """
  import torch

  x0 = torch.randint(8, 12, (batch_size, length))
  bos = model.tokenizer.bos_token_id
  if bos is not None:
    if variant == "all-bos":
      x0[:, 0] = bos
    elif variant == "mixed":
      x0[0, 0] = bos                    # first row has BOS
      x0[1:, 0] = (bos + 1) % 12        # the rest provably do not
    elif variant == "no-bos":
      x0[:, 0] = (bos + 1) % 12
  return x0


def worker(args):
  import torch

  torch.use_deterministic_algorithms(True, warn_only=True)
  dtype = {"float64": torch.float64, "float32": torch.float32,
           "bfloat16": torch.bfloat16}[args.dtype]
  # float64 is the default for a reason (see the module docstring). The other
  # dtypes exist only for the two fixes whose exactness is a property of a
  # KERNEL rather than of the algebra -- F5's single fused GEMM and F4's graph
  # replay -- and those must be checked on the GPU, in the training dtype,
  # because that is where the kernel choice actually happens.
  torch.set_default_dtype(dtype if dtype != torch.bfloat16 else torch.float32)
  device = torch.device(args.device)
  model_cfg, algo_cfg, extra = CASES[args.case] if args.case in CASES else (
    "small_bissm", "bd3lm_bissm", ())
  model = _build(model_cfg, algo_cfg, extra, args.length, args.block_size,
                 args.batch_size, args.checkpoint_prefill, args.graph_prefill)
  if dtype == torch.float64:
    model = model.double()
  model = model.to(device)
  model.train()

  out = {}
  for variant in ("mixed", "all-bos", "no-bos"):
    torch.manual_seed(1234)
    x0 = _inputs(model, args.batch_size, args.length, variant).to(device)
    attention_mask = torch.ones_like(x0)
    model.zero_grad(set_to_none=True)
    # Every stochastic draw inside `_loss` (the noise level `t`, and the noisy
    # tokens `x_t`) comes from the global generator, so re-seeding immediately
    # before the call is what makes the two trees see identical inputs.
    torch.manual_seed(4321)
    losses = model._loss(x0, attention_mask)
    losses.loss.backward()
    out[variant] = {
      "loss": losses.loss.detach().clone(),
      "nlls": losses.nlls.detach().clone(),
      "grads": {n: p.grad.detach().clone()
                for n, p in model.named_parameters() if p.grad is not None},
    }
  torch.save(out, args.dump)
  n_grads = len(out["mixed"]["grads"])
  import diffusion as _diffusion_module
  print(f"[worker] case={args.case} dtype=float64 variants=3 "
        f"grads/variant={n_grads} tree={TREE} "
        f"diffusion={_diffusion_module.__file__} -> {args.dump}")
  if not n_grads:
    raise SystemExit("no gradients were produced; the case is vacuous")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def make_shadow_tree(dest: Path) -> None:
  """Symlink the repo into `dest`, with the patch targets as real copies."""
  dest.mkdir(parents=True, exist_ok=True)
  real_dirs = {str(Path(f).parent) for f in PATCHED_FILES if "/" in f}
  # Every ancestor of a patched file must be a real directory, not a symlink,
  # or `patch` would write straight through into the working tree.
  for d in list(real_dirs):
    parts = Path(d).parts
    for i in range(1, len(parts)):
      real_dirs.add(str(Path(*parts[:i])))

  def populate(rel: Path):
    src_dir = REPO / rel if str(rel) != "." else REPO
    dst_dir = dest / rel if str(rel) != "." else dest
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_dir.iterdir():
      if entry.name in {".git", "__pycache__"}:
        continue
      child_rel = (rel / entry.name) if str(rel) != "." else Path(entry.name)
      if entry.is_dir() and str(child_rel) in real_dirs:
        populate(child_rel)
      else:
        link = dst_dir / entry.name
        if not link.exists() and not link.is_symlink():
          link.symlink_to(entry)

  populate(Path("."))
  for rel in PATCHED_FILES:
    target = dest / rel
    if target.is_symlink():
      target.unlink()
    shutil.copy2(REPO / rel, target)
    assert not target.is_symlink()


def _assert_no_symlink_targets(dest: Path, patches) -> None:
  """Refuse to run `patch` against a symlink.

  The shadow tree is mostly symlinks back into the working tree, so a patch
  naming a file that was NOT copied would be applied straight into the real
  repository. Fail loudly instead; the fix is to add the file to
  `PATCHED_FILES`.
  """
  import re

  named = set()
  for patch in patches:
    for line in patch.read_text().splitlines():
      m = re.match(r"^\+\+\+ b/(\S+)", line)
      if m:
        named.add(m.group(1))
  leaked = sorted(f for f in named
                  if (dest / f).is_symlink() or f not in PATCHED_FILES)
  if leaked:
    raise SystemExit(
      "these patch targets are not real copies in the shadow tree and would "
      f"be written into the working tree: {leaked}\n"
      "add them to PATCHED_FILES in this script.")


def apply_patches(dest: Path, patches) -> None:
  _assert_no_symlink_targets(dest, patches)
  for patch in patches:
    subprocess.run(
      ["patch", "-p1", "--no-backup-if-mismatch", "-i", str(patch)],
      cwd=dest, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"  applied {patch.name}")


def run_worker(tree: Path, case: str, dump: Path, checkpoint_prefill,
               length, block_size, batch_size, device="cpu",
               dtype="float64", graph_prefill=None) -> None:
  cmd = [sys.executable, str(tree / "scripts" / "smoke" / "fix_equivalence.py"),
         "--worker", "--case", case, "--dump", str(dump),
         "--length", str(length), "--block-size", str(block_size),
         "--batch-size", str(batch_size), "--device", device,
         "--dtype", dtype]
  if checkpoint_prefill is not None:
    cmd += ["--checkpoint-prefill", "1" if checkpoint_prefill else "0"]
  if graph_prefill is not None:
    cmd += ["--graph-prefill", "1" if graph_prefill else "0"]
  env = dict(os.environ, PYTHONPATH=str(tree))
  if device == "cpu":
    env["CUDA_VISIBLE_DEVICES"] = ""
  subprocess.run(cmd, cwd=tree, check=True, env=env)


def compare(path_a: Path, path_b: Path, tol: float, label_a: str,
            label_b: str):
  import torch

  a = torch.load(path_a, weights_only=False)
  b = torch.load(path_b, weights_only=False)
  worst = 0.0
  worst_where = "-"
  n_compared = 0
  assert set(a) == set(b), "variant sets differ"
  for variant in sorted(a):
    for key in ("loss", "nlls"):
      d = (a[variant][key].double() - b[variant][key].double()).abs().max().item()
      n_compared += 1
      if d > worst:
        worst, worst_where = d, f"{variant}/{key}"
    ga, gb = a[variant]["grads"], b[variant]["grads"]
    assert set(ga) == set(gb), (
      f"{variant}: gradient sets differ: "
      f"{sorted(set(ga) ^ set(gb))[:5]}")
    for name in sorted(ga):
      d = (ga[name].double() - gb[name].double()).abs().max().item()
      n_compared += 1
      if d > worst:
        worst, worst_where = d, f"{variant}/grad:{name}"
  ok = worst <= tol
  print(f"    {label_a} vs {label_b}: {n_compared} tensors compared, "
        f"max|delta| = {worst:.3e} at {worst_where}  "
        f"{'PASS' if ok else 'FAIL'} (tol {tol:g})")
  return ok, worst, worst_where


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  ap.add_argument("--case", default="f2f3-ssm")
  ap.add_argument("--dump", type=Path)
  ap.add_argument("--checkpoint-prefill", type=int, default=None)
  ap.add_argument("--graph-prefill", type=int, default=None)
  ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
  ap.add_argument("--dtype", default="float64",
                  choices=("float64", "float32", "bfloat16"))
  ap.add_argument("--cases", default="f2f3-ssm,f2f3-ussm,f5f6-dit,f1-ckpt")
  ap.add_argument("--patches", default=None,
                  help="comma-separated patch filenames; default = all")
  # Small on purpose. The CPU fallback scan (`mamba2_segment._reference_scan`)
  # is a Python loop over POSITIONS, so float64 runtime is linear in `length`
  # and dominates everything else; 384/128 gives three blocks -- enough for a
  # non-trivial boundary carry and a non-trivial folded scan -- in about 90 s
  # per worker. `--length 512` is the fuller run and has also been checked.
  ap.add_argument("--length", type=int, default=384)
  ap.add_argument("--block-size", type=int, default=128)
  ap.add_argument("--batch-size", type=int, default=3)
  ap.add_argument("--tol", type=float, default=0.0,
                  help="max permitted |delta|; 0 (default) means BITWISE")
  ap.add_argument("--keep", action="store_true",
                  help="keep the shadow tree for inspection")
  args = ap.parse_args()

  if args.worker:
    args.checkpoint_prefill = (None if args.checkpoint_prefill is None
                               else bool(args.checkpoint_prefill))
    args.graph_prefill = (None if args.graph_prefill is None
                          else bool(args.graph_prefill))
    return worker(args)

  # `*-REJECTED.patch` files are kept as evidence, not as proposals: they
  # FAILED this very test and are excluded unless named explicitly.
  patches = [p for p in sorted(PATCH_DIR.glob("*.patch"))
             if "REJECTED" not in p.name]
  if args.patches:
    wanted = {p.strip() for p in args.patches.split(",") if p.strip()}
    patches = [p for p in patches if p.name in wanted]
  if not patches:
    raise SystemExit(f"no patches found in {PATCH_DIR}")

  work = Path(tempfile.mkdtemp(prefix="fix-equiv-"))
  shadow = work / "patched"
  print(f"shadow tree: {shadow}")
  make_shadow_tree(shadow)
  apply_patches(shadow, patches)

  cases = [c.strip() for c in args.cases.split(",") if c.strip()]
  results = []
  try:
    for case in cases:
      print(f"\n[{case}]")
      if case == "f1-ckpt":
        # Both runs in the UNPATCHED tree: F1 only chooses between these two.
        on = work / "f1-on.pt"
        off = work / "f1-off.pt"
        run_worker(REPO, "f2f3-ssm", off, False, args.length,
                   args.block_size, args.batch_size, args.device, args.dtype)
        run_worker(REPO, "f2f3-ssm", on, True, args.length,
                   args.block_size, args.batch_size, args.device, args.dtype)
        results.append((case,) + compare(
          off, on, args.tol, "prefill ckpt off", "prefill ckpt on"))
        continue
      if case == "f4-graph":
        # Both runs in the PATCHED tree: F4 only chooses between these two.
        # CUDA-only -- a graph cannot be captured on CPU, and the whole claim
        # is about which kernels the replay issues.
        if args.device != "cuda":
          print("    SKIPPED: f4-graph needs --device cuda")
          continue
        eager = work / "f4-eager.pt"
        graph = work / "f4-graph.pt"
        run_worker(shadow, "f2f3-ssm", eager, False, args.length,
                   args.block_size, args.batch_size, args.device, args.dtype,
                   graph_prefill=False)
        run_worker(shadow, "f2f3-ssm", graph, False, args.length,
                   args.block_size, args.batch_size, args.device, args.dtype,
                   graph_prefill=True)
        results.append((case,) + compare(
          eager, graph, args.tol, "prefill eager", "prefill cuda-graph"))
        continue
      if case not in CASES:
        raise SystemExit(f"unknown case {case!r}; known: {sorted(CASES)}")
      base = work / f"{case}-base.pt"
      new = work / f"{case}-new.pt"
      run_worker(REPO, case, base, None, args.length, args.block_size,
                 args.batch_size, args.device, args.dtype)
      run_worker(shadow, case, new, None, args.length, args.block_size,
                 args.batch_size, args.device, args.dtype)
      results.append((case,) + compare(
        base, new, args.tol, "unpatched", "patched"))
  finally:
    if not args.keep:
      shutil.rmtree(work, ignore_errors=True)
    else:
      print(f"\nkept {work}")

  print("\n" + "=" * 72)
  print(f"{'case':<14}{'max|delta|':>14}  {'verdict':<8} where")
  print("-" * 72)
  failed = 0
  for case, ok, worst, where in results:
    failed += 0 if ok else 1
    print(f"{case:<14}{worst:>14.3e}  {'PASS' if ok else 'FAIL':<8} {where}")
  print("=" * 72)
  print(f"{args.dtype} / {args.device.upper()}, tolerance {args.tol:g}"
        f"{' (bitwise)' if args.tol == 0 else ''}")
  print(json.dumps({c: {"max_abs_delta": w, "pass": bool(o)}
                    for c, o, w, _ in results}))
  raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
  main()
