# Throughput fixes for the block-diffusion arms

Patches, not applied. Each is a standalone `patch -p1` against the tree at
`codex/bidirectional-ssm`, except **F4**, which is a diff on top of **F1**
(apply in filename order and they compose; verified).

The numerics gate for all of them is
`scripts/smoke/fix_equivalence.py` — it builds a shadow copy of the repo,
applies these very files into it, and compares loss + every gradient between
the unpatched and patched trees in **float64 on CPU at tolerance 0**.

```
python scripts/smoke/fix_equivalence.py                 # all cases
python scripts/smoke/fix_equivalence.py --cases f5f6-dit --keep
```

## Which problem each one attacks

The brief identifies two separate phenomena. They do not share a fix.

* **SSM arms** — the unexplained factor SHRINKS with length (2.08 → 1.44).
  That is a fixed per-step cost being amortised. `F1`–`F4` attack it. Their
  gain is concentrated at short `L` and goes to roughly zero by `L=32768`,
  which is the correct signature: a fix for a fixed cost cannot help where the
  fixed cost no longer binds.
* **Transformer arm** — the factor is flat-to-humped (1.20 / 1.48 / 1.26). That
  is per-token memory traffic. `F5`–`F7` attack it, and their gain is roughly
  proportional in `L`.

## The patches

| id | file(s) | what | numerics |
|----|---------|------|----------|
| F0 | `scripts/smoke/sizing_sweep.py` | Flag the asymmetric-checkpoint measurement confound in the output and the JSON | none — measurement only |
| F1 | `models/bidirectional_ssm.py`, both SSM yamls | `checkpoint_boundary_prefill: auto`, decided once from geometry | bitwise; both branches already asserted equal by `tests/test_bissm_diffusion_integration.py:180` |
| F2 | `diffusion.py` | Remove all 5 device→host syncs per step | bitwise |
| F3 | `models/mamba2_segment.py` | Cache the constant block-carry mask | bitwise |
| F4 | `models/bidirectional_ssm.py` | **Opt-in, experimental.** CUDA-graph the 12 prefill layers | bitwise *by construction* (same kernels, same order); engineering risk is real |
| F5 | `models/dit.py` | One packed qkv GEMM for `[x_t; x_0]` instead of two plus a `cat` | **measured bitwise** at float64/CPU; the GEMM reshape is still BLAS-dependent, so re-measure on GPU |
| ~~F6~~ | `models/dit.py` | Trim the `x_0` half before the output head | **REJECTED ON MEASUREMENT.** Forward bitwise, but `grad_weight` is a cross-row reduction and moving the slice regroups the blocked GEMM: 1.11e-16 on `output_layer.linear.weight`. <0.2% gain does not buy a training-trajectory change. Patch kept as `F6-...-REJECTED.patch`. |
| F7 | — | Fuse the eager rotary. **Not proposed as a diff**: `scripts/smoke/test_rotary_fusion_equivalence.py` passes at fp32 and FAILS at bf16 | blocked |

## Test results (float64, CPU, tolerance 0 = bitwise)

```
case              max|delta|  verdict  where
f2f3-ssm           0.000e+00  PASS     -
f2f3-ussm          0.000e+00  PASS     -
f1-ckpt            0.000e+00  PASS     -
F5 alone           0.000e+00  PASS     -
F5 + F6            1.110e-16  FAIL     grad: output_layer.linear.weight
```

96 tensors per case (loss, per-token nlls, and all 30 parameter gradients,
across three BOS populations: mixed, all-BOS, no-BOS). The F6 result is the
useful one — it refutes a claim I had made from a correct-but-incomplete
row-independence argument, which covered the forward and not the weight
gradient.

## Expected recovery

Micro batch 2, against the measured baseline in
`results/figures/scaling_data.json`. All SSM numbers are model-derived, not
measured: they come from `T = max(c, g + F/R)` fitted in the launch-overhead
diagnosis, with `c` scaled by the dispatch counts that
`scripts/smoke/launch_count_probe.py` measures directly.

| fix | arm | L=2048 | L=32768 |
|-----|-----|--------|---------|
| F1 | bissm | 15,879 → ~23,200 (**+46%**) | 0% as shipped (`auto` picks ON); +11% if the budget is raised after a memory check |
| F1 | ussm | 19,473 → 27,000–30,000 (**+37…54%**) | same caveat |
| F2 | both SSM | +1…5% | +0.5…2% |
| F3 | both SSM | +0.6% | ~0% |
| F4 | bissm | +28…84% *on top of F1* | ~0% |
| F5 | dit | +1…3% | +1…3% |
| ~~F6~~ | dit | <0.2% — **rejected, not bitwise** | — |
| F7 | dit | +22% (diagnosis estimate) | +8.5% | 

Note the shape: every SSM fix is worth a lot at L=2048 and nothing at L=32768,
and every Transformer fix is worth roughly the same at both. That is the
signature the brief asked for, and it is the strongest available check that
these fixes are aimed at the right causes.

## Order to land

1. **F0** — free, and it tells you the size of the confound you are about to
   remove.
2. **F1 + F3** — bitwise, and F1 is the single biggest item.
3. **F2** — bitwise; most of its value is the one sync at `diffusion.py:1281`,
   which sits between the forward and the backward and forces the CPU to wait
   for the whole forward before it can start issuing the backward.
4. **F5** (bitwise on CPU; still needs the GPU check, because the whole
   question is which kernel cuBLAS picks). **Not F6** — see the table.
5. **F4** only if the floor still matters after 1–3, and only behind its flag.
6. **F7** only after the CUDA run of the rotary test decides it, and — given
   `MEMORY.md`'s `backbone-precision-asymmetry` note — never silently.

## GPU jobs this needs (none submitted from here)

Two are proof obligations, two are measurements. In priority order:

```bash
# 1. RE-MEASURE without the confound. This is what F0 exists to point at, and
#    it alone moves the BD-SSM column of the scaling table. One job per length
#    (see docs/lsf_conventions.md for the -env spelling; '+' is the list
#    separator because bsub -env splits on commas).
bsub -env "all, ARMS=bissm+ussm+ussm-ar+dit+dit-ar, BATCH_SIZES=2, \
  LENGTHS=2048, CHECKPOINT_MODES=off, LABEL=scaling-off-2048" \
  < scripts/smoke/sizing_sweep.sh     # repeat for 4096 8192 16384 32768

# 2. CONFIRM the launch-bound model before anyone builds F4. Throughput should
#    scale ~8x linearly from batch 1 to 8 at fixed L=2048 if the floor is issue
#    cost, and stay flat if it is not. The two predictions are 4x apart.
python scripts/smoke/sizing_sweep.py --arms bissm \
  --batch-sizes 1,2,4,8,16 --lengths 2048 --checkpoint-modes off

# 3. F5's numerics gate (one GEMM vs two, in the training dtype, on cuBLAS).
python scripts/smoke/fix_equivalence.py --cases f5f6-dit \
  --device cuda --dtype float32 --length 512 --block-size 256

# 4. F4's numerics gate, only if (2) confirms the model.
python scripts/smoke/fix_equivalence.py --cases f4-graph \
  --device cuda --dtype float32 --length 1024 --block-size 256
```
