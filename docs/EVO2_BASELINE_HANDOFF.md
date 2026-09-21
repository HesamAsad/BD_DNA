# Running Evo 2 as a baseline on AG-LongGen Task 2 (infilling generation)

Everything you need to produce a number that is directly comparable to ours.
Written 2026-09-21.

The short version: **you generate sequences, we score them.** If your output
matches the JSON schema in §4, you run our scorer unchanged and the comparison
is exact. Please do not reimplement the metrics — §6 explains why they are not
the obvious ones.

---

## 0. Where everything lives

All paths in this document are relative to the repository root:

```
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
```

So `scripts/eval/...` means `$REPO/scripts/eval/...`. The absolute paths for the
things you need first:

| what | absolute path |
|---|---|
| repository root | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms` |
| python interpreter | `/software/cellgen/team361/ha11/envs/nichejepa/bin/python` |
| released loci + real/dinuc records | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/results/aglonggen/task2_gen_cosfinal_bissm.json` |
| the scorer | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/scripts/eval/aglonggen/task2_score.py` |
| the validator | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/scripts/eval/aglonggen/validate_task2_submission.py` |
| reference genome | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data/hg38/hg38.ml.fa` |

**Prerequisites, worth checking before you start:**

- Both `/lustre/scratch126` and `/software` are **shared cluster filesystems**,
  so this only works from a node on the same cluster (`tiger22`). There is no
  copy of this outside it.
- You need to be in the **`team361` unix group**. The repository itself is
  world-readable, but everything this document points you at is group-readable (`-rw-rw-r--`) rather than world-readable. Check with `id -nG | tr ' ' '\n' | grep team361`.
- Run from `$REPO`, not from a copy. Each script derives the repo root from
  its own location (`Path(__file__).resolve().parents[N]`, with N depending
  on how deep the script sits) and resolves sibling data paths from it, so
  moving a script out of its directory breaks those lookups silently.

## 1. The task

Take a real 16,384 nt human locus. Mask a contiguous interior span, commit both
flanks as ground truth, and have the model write the interior back. Then ask a
frozen functional oracle (AlphaGenome) how close the reconstruction is
*functionally* to the real locus — not how close the letters are.

Sequence-level metrics do not just miss this -- they can point the wrong way.
Across our own `denovo` runs, going from a 256 nt gap to a 4,096 nt gap raises
the functional error **29.5x** (median `mse_RNA_SEQ` 0.0478 -> 1.4099) while
median k-mer JS divergence **falls** to 0.39x of its value (0.9015 -> 0.3479
bits). A wider, functionally far worse reconstruction looks *better* by k-mer
composition, because a long generated span regresses to genomic background.
That is the whole reason this benchmark runs a functional oracle.

**Why 16,384 nt.** AlphaGenome accepts exactly {16384, 131072, 524288, 1048576}
and nothing else. 16,384 is the only one our 8,192-trained models can reach
without heavy extrapolation. If Evo 2's context makes a longer setting
attractive, that is a *different* benchmark cell and not comparable to the
numbers in §7 — but it would be a genuinely interesting addition, see §8.

## 2. The design grid

| axis | values |
|---|---|
| locus length | 16,384 nt, fixed |
| gap width | **256, 512, 1024, 4096 nt** (4 levels) |
| loci | 120, fixed list, chr8 + chr9 |
| conditions | 5, see §3 |
| total records | 120 x 4 x 5 = **2,400** |

The gap is centred by `left_nt = ((16384/256 - gap_blocks) // 2) * 256`. Note
this **floors**, so gap 256 is NOT symmetric. Read the flanks off the released
file rather than recomputing them:

| gap_nt | left_nt | right_nt | |
|---|---|---|---|
| 256 | **7,936** | **8,192** | asymmetric -- (64-1)//2 floors |
| 512 | 7,936 | 7,936 | symmetric |
| 1024 | 7,680 | 7,680 | symmetric |
| 4096 | 6,144 | 6,144 | symmetric |

**Loci are not random ACGT windows.** They are contiguous ACGT windows whose
centre — the span masked at the widest gap — carries at least one ENCODE cCRE.
Masking arbitrary intergenic sequence is not the benchmark's task and cannot
test regulatory reconstruction. Use the same locus list (§5) rather than
resampling, or the comparison is against a different set of loci.

## 3. The five conditions, and which ones Evo 2 can actually do

| condition | what it is | can Evo 2 do it? |
|---|---|---|
| `real` | the true interior; scores 0 by construction | n/a, already have it |
| `denovo` | infill from the **left flank only** | **yes — this is your cell** |
| `ca` | infill with the **true right flank** also committed | no, not natively |
| `mismatch` | right flank from a *different* locus (control for `ca`) | no |
| `dinuc` | dinucleotide-shuffled real interior (composition floor) | n/a, model-free |

**This is the main thing to get right.** Evo 2 is autoregressive, so it
conditions on the left only. `ca` and `mismatch` exist to test whether a model
can consume a committed *suffix*, which is the capability our bidirectional SSM
has and an AR model does not. So:

- **Evo 2 contributes `denovo`, and it is directly comparable to our `denovo`
  column.** That is a clean, honest, architecture-vs-architecture comparison at
  matched task, matched loci, matched metric.
- Comparing Evo 2's `denovo` against our `ca` is **not** clean — it confounds
  architecture with whether a right flank was available. Report it if you like,
  but label it.
- `real` and `dinuc` need no model; reuse ours so the floor and ceiling are
  identical.

If you want a bidirectional Evo 2 arm, §8 has a suggestion that would be a real
contribution rather than a confound.

## 4. The integration contract — the JSON we score

Emit one JSON file:

```json
{
  "checkpoint": "evo2-<size>-<revision>",
  "length": 16384,
  "n_loci": 120,
  "records": [
    {
      "chrom": "chr8",
      "start": 120161149,
      "gap_nt": 256,
      "left_nt": 7936,
      "condition": "denovo",
      "sequence": "<the FULL 16384 nt reconstruction, uppercase ACGT>",
      "interior": "<just the gap_nt generated bases>",
      "ccres_in_gap": []
    }
  ]
}
```

Hard requirements, all checked by the validator in §5:

- `len(sequence) == 16384` for every record
- `len(interior) == gap_nt` for every record
- `sequence[:left_nt]` and `sequence[left_nt+gap_nt:]` are **byte-identical to
  the real locus** — the flanks are committed, not regenerated
- `sequence[left_nt:left_nt+gap_nt] == interior`
- uppercase ACGT only, no N
- `(chrom, start, gap_nt)` matches the released locus list

Then:

```bash
export ALPHAGENOME_API_KEY=...
python scripts/eval/aglonggen/task2_score.py \
    --gen  your_evo2_generations.json \
    --out  results/aglonggen/task2_scores_evo2.json \
    --label evo2
```

## 5. Files you need from us

| what | path |
|---|---|
| the released locus list + real/dinuc records | `results/aglonggen/task2_gen_cosfinal_bissm.json` |
| the scorer (run unchanged) | `scripts/eval/aglonggen/task2_score.py` |
| our reference scores | `results/aglonggen/task2_scores_cosfinal_bissm.json` |
| the write-up of our run | `results/aglonggen/report/AG-LongGen_Task2_Report.pdf` |
| output validator | `scripts/eval/aglonggen/validate_task2_submission.py` |
| reference genome | `data/hg38/hg38.ml.fa` |

Extract the locus list and the model-free conditions straight from our
generation file — that guarantees identical loci:

```python
import json
recs = json.load(open("results/aglonggen/task2_gen_cosfinal_bissm.json"))["records"]
loci = sorted({(r["chrom"], r["start"], r["gap_nt"], r["left_nt"]) for r in recs})
# 480 (locus, gap) cells; real/dinuc/mismatch rows are reusable as-is
```

Validate before scoring — an AlphaGenome pass over 2,400 records is not free:

```bash
python scripts/eval/aglonggen/validate_task2_submission.py \
    --submission your_evo2_generations.json \
    --reference  results/aglonggen/task2_gen_cosfinal_bissm.json
```

A `denovo`-only submission is fine, but it must cover **all 480** (locus, gap)
cells. Partial coverage fails by default -- a partial file scores without
complaint and yields a mean over a different locus set than ours, which is
silently incomparable. `--allow-partial` overrides if that is deliberate.

The validator is fault-injection tested: it catches a regenerated flank, a
lower-cased sequence, a wrong length, a resampled locus, an `interior` that
disagrees with its own `sequence` slice, and partial coverage.

## 6. The metrics, and why they are not the obvious ones

Two deliberate departures from the AG-LongGen note. Both came out of a
receptive-field pre-flight (`scripts/eval/aglonggen/ag_receptive_field.py`) and
both matter for interpreting your numbers.

**Per track family, not a flat mean.** The note defines
`track-MSE = 1/T * sum_tracks ||AG(x) - y*||^2`. Measured, the families do not
behave alike: block-permuting a locus moves `RNA_SEQ` by 0.39 and
`CHIP_HISTONE` by 0.22 even far from any seam, while `ATAC`, `DNASE` and `CAGE`
decay to ~0.04 within about 1 kb and `SPLICE_SITES` is local and tiny.
**`RNA_SEQ` is where AlphaGenome's long-range sensitivity lives**, and averaging
it against four mostly-local families dilutes exactly the signal the benchmark
exists to measure. Every family is reported separately; **`mse_RNA_SEQ` is the
headline.**

**Shared normalisation.** Tracks differ in dynamic range by orders of magnitude.
Normalising each prediction by its *own* statistics would absorb a genuine
global shift into the z-scoring and manufacture agreement. Both sides are
standardised with **the real locus's** per-track mean and sd.

**Quote the median, not the mean.** The failure mode here is a minority of loci
going badly wrong while most are fine; means hide that and medians do not. Our
own earlier reporting was misleading on exactly this point.

AlphaGenome is deterministic — the same sequence twice gave a noise floor of
exactly 0.0 — so no repeat-call averaging is needed and any difference you see
is real.

## 7. The numbers to beat

Median `mse_RNA_SEQ`, our bidirectional SSM (`hg_bissm_cos/best.ckpt`), 2,400
records:

| gap nt | real | ca | mismatch | **denovo** | dinuc |
|---|---|---|---|---|---|
| 256 | 0.0000 | 0.0427 | 0.0499 | **0.0478** | 0.0293 |
| 512 | 0.0000 | 0.1507 | 0.1263 | **0.1313** | 0.0783 |
| 1024 | 0.0000 | 0.3536 | 0.3094 | **0.3103** | 0.1882 |
| 4096 | 0.0000 | 1.2593 | 1.7782 | **1.4099** | 1.0155 |

**`denovo` is your column.** Lower is better; `real` is the floor by
construction and `dinuc` is the composition-matched control.

Read `dinuc` before concluding anything. At gap 4096 it scores 1.0155 against
our `denovo` 1.4099 — **a dinucleotide shuffle beats our model at the widest
gap.** Any baseline that does not clear `dinuc` has not demonstrated it is
generating regulatory structure rather than plausible composition. That is a low
bar and we do not clear it at 4096 either.

Note also `ca` vs `mismatch` at 4096 (1.2593 vs 1.7782): a *wrong* right flank
is much worse than no right flank, which is why `mismatch` exists.

## 8. Gotchas, in the order they will bite you

1. **Training contamination is your problem too, and differently.** These loci
   are on chr8/chr9. For *our* checkpoints 99.4% of chr8/chr9 intervals are in
   the training split, so we are explicit that our numbers partly measure
   memorisation. Evo 2's pretraining corpus includes human genome data — please
   state whether these intervals were in it. If they were, both sides are
   contaminated and the comparison is still internally valid, but neither number
   is a clean generalisation claim and the write-up must say so.
2. **Commit the flanks; do not regenerate them.** The scorer compares the whole
   16,384 nt window. If Evo 2 rewrites any flank base the MSE moves for reasons
   that have nothing to do with infilling. The validator checks this.
3. **Uppercase, ACGT only.** hg38 is soft-masked; `hg38.ml.fa` lowercase means
   repeat, not unknown. Upper-case everything and reject any window containing N.
4. **Sampling temperature is a real knob and must be declared.** We generate
   with 64 diffusion steps; you will have a temperature / top-k. Greedy decoding
   will produce low-entropy sequence that scores oddly on composition metrics.
   Whatever you choose, report it, and consider a small sweep on a held-out
   subset of loci rather than tuning on all 120.
5. **2,400 AlphaGenome calls is the expensive step.** Validate first. If you
   only run `denovo`, that is 480 records, not 2,400.
6. **`PROCAP` is `nan` on many records** — expected, it is a sparse assay. The
   scorer handles it; do not treat it as a failure.

## 9. A genuinely interesting extension, if you have appetite

Evo 2 cannot consume a committed suffix, which is why it only fills the
`denovo` cell. But you could build a **bidirectional AR baseline**: generate the
interior left-to-right from the left flank, generate it again right-to-left from
the reverse complement of the right flank, and combine (simplest: average the
two per-position distributions, or take the higher-likelihood sequence).

That would give a fair `ca`-equivalent cell for an autoregressive model and
would make the bidirectionality question architecture-independent, which is
currently the weakest part of our own claim. It is the single most valuable
thing this baseline could add beyond a `denovo` number.

## 10. Contact points in the code

- task definition and generation: `scripts/eval/aglonggen/task2_generate.py`
  (the module docstring states the design and its known limitations)
- scoring: `scripts/eval/aglonggen/task2_score.py`
- the receptive-field pre-flight that justified the metric choices:
  `scripts/eval/aglonggen/ag_receptive_field.py`
- report generation: `scripts/eval/aglonggen/task2_report.py`
