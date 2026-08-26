# LSF job submission — conventions and tricks (Sanger farm, `tiger22` cluster)

Distilled from the launcher scripts in `scripts/` and the operational history of the
long-context / dual-stream / oracle experiments (jobs ~31k–83k). Everything here has
been exercised on this cluster; failure modes are recorded with the fix that worked.

---

## 1. Cluster basics

| Thing | Value |
|---|---|
| Repo (always submit with absolute paths) | `/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms` |
| User group (`-G`, **mandatory**) | `s10396` |
| GPU queue | `training-parallel` |
| CPU-ish queue | `training-normal` (still needs a `-gpu` request — see §6) |
| Alternate route | `ssh ha11@farm22-head1` for the `gpu-lotfollahi` / `gpu-lotfollahi-train` queues |
| Python | `/software/cellgen/team361/ha11/envs/nichejepa/bin/python` (path venv, **not** conda-registered) |
| Node naming | H200 = `farm-gpu05*` (`gmodel=NVIDIAH200`, 140 GB); H100 = `farm-gpu03*` (`gmodel=NVIDIAH10080GBH`, 80 GB) |
| Bad node | `farm-gpu0504` — `nvidia-smi` reports no devices; exclude it (see §5) |

Head nodes (e.g. `tiger22-head1`) have **no GPU** and a tight per-process memory cgroup
(~1.7 GB observed). Nothing that needs a GPU or more than ~1 GB of RSS runs there —
submit it.

---

## 2. The canonical header block

Every launcher in `scripts/` starts with the same shape. Copy it verbatim and change
the name / resources:

```bash
#!/usr/bin/env bash
#BSUB -J train_dna_bd3lm
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -n 32
#BSUB -W 168:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[mem>128000 && hname!='farm-gpu0504']"
#BSUB -R "rusage[mem=128000]"
#BSUB -M 128000
#BSUB -gpu "num=4:mode=exclusive_process:gmodel=NVIDIAH200"
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
#BSUB -o /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_%J.out
#BSUB -e /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/logs/train_dna_bd3lm_%J.err
```

Notes that matter:

- **Memory is MB** and appears three times (`select[mem>...]`, `rusage[mem=...]`, `-M`).
  Keep them equal; `select` is the placement filter, `rusage` the reservation, `-M` the kill limit.
- `-o` / `-e` must be **absolute** and the `logs/` directory must already exist —
  `%J` expands to the job id. Every script also does `mkdir -p ... logs logs/eval` defensively.
- `-cwd` is set even though the script `cd`s itself; belt and braces after the
  `#BSUB -cwd` + `REPO=...; cd "$REPO"` pattern.
- `mode=exclusive_process` is required for the DDP+flex path (`main.py` calls
  `torch.cuda.set_device(LOCAL_RANK)` early precisely because of it).
- `-W` is wall-clock `HH:MM`. Exceeding it kills the job with **exit code 140**
  (seen on job 56750, which hung in a data-prep deadlock and was reaped at its 2 h limit).

### Resource sizing actually used

| Job class | `-n` | `-W` | mem (MB) | GPUs |
|---|---|---|---|---|
| 4-GPU training (`train_dna_bd3lm`, `train_dna_longctx_dual`) | 32 | 168:00 | 128000 | 4 |
| 1-GPU training (oracle, synthLR) | 8 | 24:00–72:00 | 96000 | 1 |
| 1-GPU smoke | 16 | 1:00 | 64000 | 1 |
| Long eval / sweep | 16 | 8:00–48:00 | 128000 | 1 |
| Short eval (`synth_copy_one`, `oracle_delta`) | 4 | 2:00–3:00 | 64000 | 1 |
| Data/cache build | 4 | 3:00–4:00 | 32000–96000 | 1 (see §6) |

---

## 3. Submission patterns

**Standard — script on stdin** (`<`, not an argument; that is what makes the in-file
`#BSUB` directives take effect):

```bash
bsub < scripts/train/train_dna_bd3lm.sh
```

**Passing job parameters — `-env "all, VAR=val"`.** All launchers read their knobs from
the environment with `${VAR:-default}`, so one script covers a whole family of runs:

```bash
bsub -env "all, L_USE_WEIGHT=1.0, BATCH=4, MAX_STEPS=6000, LENGTH=12288, \
           BLOCK_SIZE=1536, DATA=synthDUPlong, WANDB=null" \
     < scripts/train/train_synthlr_incentive.sh
```

`all` inherits the submitting shell's environment; without it the job starts with a
near-empty env. The equivalent shorthand (var exported into `bsub`'s own environment) also works
and reads better in loops:

```bash
for DV in carbon-prok-lr983 carbon-prok-lr983sblk carbon-prok-lr983shuf; do
  MODEL=small_dual_bigblock BLOCK_SIZE=24576 L=983040 CKPT="$CKPT" DATA_VALID=$DV \
    bsub -env "all" -J "lr_bb_$DV" < scripts/eval/ppl_one.sh
done
```

**Command-line flags override the in-file `#BSUB` directives** — the cheapest way to
re-purpose a training launcher as a short smoke without editing it:

```bash
export BLOCK_SIZE=1536 LENGTH=6144 MAX_STEPS=60 GLOBAL_BATCH=4
bsub -env "all" -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200" \
     -J bigblock_smoke -W 2:00 < scripts/train/train_dna_bigblock_dual.sh
```

Use `-J` to give each variant a distinct name — `bjobs -w` becomes readable and
`bkill -J <name>` becomes usable.

**One-off command (no script file)** — used for cache/dataset builds:

```bash
bsub -q training-parallel -G s10396 -n 4 -W 4:00 \
  -R "span[hosts=1]" -R "select[mem>96000 && hname!='farm-gpu0504']" \
  -R "rusage[mem=96000]" -M 96000 \
  -gpu "num=1:mode=exclusive_process" \
  -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms \
  -o /lustre/.../logs/build_human_v2_%J.out -e /lustre/.../logs/build_human_v2_%J.err \
  "export USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3; /software/cellgen/team361/ha11/envs/nichejepa/bin/python scripts/eval/build_human_longrange.py --length 32768 --name human-lr32768v2 --val_chroms chr8,chr9"
```

The command string must carry its own `export`s — an inline command does not source a profile.

**Interactive** (rarely used, handy for debugging a node):

```bash
bsub -Is -q training-parallel -G s10396 -n 16 -gpu "num=1:mode=exclusive_process" \
     -R "span[hosts=1]" -M 64000 -R "rusage[mem=64000]" bash scripts/train/smoke_dna_bd3lm.sh
```

---

## 4. In-script conventions

Every launcher follows this skeleton — reproduce it in new ones:

```bash
set -euo pipefail                 # ... but `set -uo pipefail` (no -e) in sweeps that
                                  # must survive a per-item OOM and continue
REPO=/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
cd "$REPO"
PYTHON=${PYTHON:-/software/cellgen/team361/ha11/envs/nichejepa/bin/python}
LENGTH=${LENGTH:-98496}           # every knob env-overridable with a default
CKPT=${CKPT:?set CKPT=/path/to/ckpt.ckpt}     # `:?` for genuinely required inputs

export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/huggingface
export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/torch
export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/cache/xdg
export NCCL_NVLS_ENABLE=0                     # NVLS off on these nodes
export TOKENIZERS_PARALLELISM=false
export USE_TF=0                               # keep transformers/tensorboard off TF
export TF_CPP_MIN_LOG_LEVEL=3                 #   (TF/protobuf mismatch in this venv)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token || true
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" outputs watch_folder logs sample_logs

echo "[`date`] <job name> | host=$(hostname) | LSF=${LSB_JOBID:-local} | <key params>"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PYTHON" -c "import sys,torch; ok=torch.cuda.is_available(); \
  print('torch',torch.__version__,'| cuda',ok,'| devices',torch.cuda.device_count()); \
  sys.exit(0 if ok else 3)" || { echo 'FATAL: torch sees no GPU.'; exit 3; }
```

Why each piece earns its place:

- **The CUDA fail-fast.** On a bad node the run would otherwise die deep inside Hydra
  with a cryptic `ZeroDivisionError` (from `accumulate_grad_batches` interpolation).
  Exit 3 + a clear message instead.
- **The banner line** (`host=`, `LSF=${LSB_JOBID:-local}`) is what makes a log file
  attributable months later, and lets the same script run locally with no LSF.
- **`${LSB_JOBID}` as the run tag.** The Hydra config derives `wandb.id` from
  `${name}_${seed}`, so a constant `wandb.name` reused the *same* wandb id and runs
  overwrote each other in the UI. Stamp it once in bash (so all DDP ranks agree):

  ```bash
  RUN_TAG="${LSB_JOBID:-$(date +%Y%m%d-%H%M%S)}"
  WANDB_NAME="bd3lm-dna-prok-dual-len${LENGTH}-bs${GLOBAL_BATCH}-${RUN_TAG}"
  ```

  The same `RUN_TAG` names per-job artefacts: `logs/eval/sweep_longctx_nll_${RUN_TAG}.tsv`.
- **Conditional Hydra overrides** go through an array so an empty case stays clean under `set -u`:

  ```bash
  EXTRA_ARGS=()
  [ "$CROSS_MODE" != "attn" ] && EXTRA_ARGS+=( "++strategy.find_unused_parameters=true" )
  "$PYTHON" -u main.py ... ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  ```
- **`WANDB=null` → `wandb=null`** (CSVLogger) for smokes; never `logger=None`
  (Lightning then falls back to TensorBoard → imports TF → crashes this venv).
- **`"$PYTHON" -u`** always — unbuffered, so `bpeek`/`tail` on a running job shows progress.

### Multi-item sweeps inside one job

`scripts/eval/sweep_longctx_nll.sh` is the reference pattern for "N configurations,
one job": `set -uo pipefail` (not `-e`), one child process per item for clean OOM
isolation, a background `nvidia-smi` sampler for peak memory, per-item log file,
status classification from the log, and a TSV summary:

```bash
MEMFILE="$(mktemp)"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 3; done ) > "$MEMFILE" &
MEMPID=$!
START=$(date +%s); "$PYTHON" -u main.py ... > "$LOG" 2>&1; RC=$?; WALL=$(( $(date +%s) - START ))
kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
PEAK_MIB=$(sort -n "$MEMFILE" | tail -1); rm -f "$MEMFILE"
if   [ $RC -eq 0 ] && [ -n "$VAL_NLL" ];                       then STATUS=ok
elif grep -qiE 'out of memory|CUDA error: out of memory' "$LOG"; then STATUS=OOM
else STATUS="fail(rc=$RC)"; fi
```

---

## 5. Node-selection tricks

- **Pin the GPU model** when memory matters: `-gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200"`.
  Without it a 140 GB-sized job can land on an 80 GB H100 and OOM.
- **Exclude a bad node inside the same `select[]` clause as `mem`:**
  `-R "select[mem>128000 && hname!='farm-gpu0504']"`.
  A *separate* command-line `-R "select[hname!=...]"` replaces the mem reservation and
  the esub then rejects the job. One `select[]`, both conditions.
- Rule of thumb from measured ceilings: 1×H200 holds ~98 k dual-stream context or
  ~120 k single-stream at batch 1, without gradient checkpointing.

---

## 6. esub rules (the non-obvious gate)

The cluster's esub validates every submission and **rejects a job on the training
queues that does not request a GPU**:

```
To submit jobs to a gpu queue, you need to select a system which has gpus, eg -gpu -
Request aborted by esub. Job not submitted.
```

That was hit trying to run a pure-CPU cache builder on `training-normal`. So:

- Add `-gpu "num=1:mode=exclusive_process"` even to CPU-only helper jobs, or
- run the CPU work on the head node **only if it fits the ~1.7 GB per-process cgroup**
  (the `num_proc=1` incremental `datasets.map` path does; the `from_dict` bulk builder does not).

Also, since the esub rewrites the request, verify the submission actually happened —
`bsub` prints `Job <NNNNN> is submitted to queue <training-parallel>.` on success and
nothing useful on rejection, so always capture stderr (`2>&1`) at submit time.

---

## 7. Monitoring, polling, log forensics

**Capture the job id at submit time:**

```bash
OUT=$(bsub -env "all, ..." < scripts/train/train_synthlr_incentive.sh 2>&1); echo "$OUT"
J=$(echo "$OUT" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+')
```

**Compact status** (the three columns worth reading):

```bash
bjobs -w 2>/dev/null | awk '{print $1, $3, $7}' | head       # id, state, name
bjobs -noheader -o 'jobid stat run_time' 56750               # scriptable, single job
bjobs -noheader -o stat "$JID"                               # '' when the job is gone
```

**Poll to completion** (45 s cadence, bounded iterations — an unbounded `while` on a
pending job will outlive the shell's patience):

```bash
for i in $(seq 1 240); do
  STAT=$(bjobs -noheader -o stat $JOB 2>/dev/null | head -1)
  if [ -z "$STAT" ] || [ "$STAT" = "DONE" ] || [ "$STAT" = "EXIT" ]; then
    echo "[`date`] $JOB finished stat='${STAT:-gone}' after ~$i polls"; break
  fi
  sleep 45
done
```

Better than polling on state alone: **poll on the artefact you actually need** and exit
early when the evidence is sufficient (`scripts/_mon_fixcheck.sh` waits for a checkpoint
*and* a metric threshold, then stops). Waiting for a specific checkpoint:

```bash
while [ -z "$(ls $BASE/checkpoints/*-4000.ckpt 2>/dev/null)" ] && [ $i -lt 140 ]; do sleep 45; i=$((i+1)); done
```

**Reading the log** — Lightning's progress bar is carriage-return based, so a raw
`tail` shows one smeared line. Always translate first:

```bash
tr '\r' '\n' < logs/train_bigblock_dual_${JOB}.out | grep -ivE 'warn' \
  | grep -iE 'loss=|nll|global_step|Error|Traceback|out of memory|assert|Successfully completed|Exited' | tail -30
```

LSF appends its own job report to the `.out` file — `Successfully completed`,
`Exited with exit code N`, `TERM_*`, and a `Subject: Job NNNN: <name> ... Exited`
header. `grep -iE 'Successfully completed|Exited|TERM_'` is the quick verdict.
The `.err` file holds the Python traceback; the `.out` holds the banner + progress.

**Kill:** `bkill <id>` (`Job <id> is being terminated`); `bkill -J <name>` by name.

---

## 8. Failure modes seen, and the fix that worked

| Symptom | Cause | Fix |
|---|---|---|
| `Request aborted by esub. Job not submitted.` | CPU-only job on a training queue | add `-gpu "num=1:..."` (§6) |
| Job dies instantly, `cuda False` | landed on `farm-gpu0504` | exclude via `hname!=` inside `select[]` (§5) |
| Cryptic `ZeroDivisionError` in Hydra | no visible GPU | the up-front CUDA assert (§4) turns it into exit 3 |
| OOM **during compile/warm-up** | `max-autotune` benchmarks 29 kernel variants; scratch tips a near-full GPU over | `BD3LM_FLEX_COMPILE_MODE=default` |
| OOM **in steady state** | genuinely too big | lower `BATCH` (dual at 24 kb ≈ 22 GB per batch unit — batch 6 blew 140 GB, batch 2 fit) or trim `LENGTH` |
| Job hangs at `Grouping 0%` for hours, then exit 140 at the wall limit | `datasets.map(_group_texts, num_proc=128)` forking on million-element lists deadlocks (also at 8) | `BD3LM_DATA_NUM_PROC=1`, and pre-build caches with `scripts/eval/pregen_longctx_caches.py` before the GPU job |
| `ValueError: val_check_interval (2000) must be <= number of training batches (1024)` | val interval exceeds batches/epoch at that batch size | set `VAL_EVERY` ≤ `n_train/BATCH` (killed job 81050) |
| DDP: `parameters that were not used in producing the loss` | a module bypassed by an ablation probe | `++strategy.find_unused_parameters=true`, **only** for that probe (it costs throughput) |
| `~wandb` override crashes at config resolution | `main.py` already sets `config.wandb = None` in `ppl_eval` mode | don't override wandb in eval modes |
| NCCL watchdog timeout kills a multi-GPU run mid-training | intermittent; observed on the human 4-GPU runs | resubmit; resume needs §9 |
| wandb runs overwriting each other | `wandb.id` derived from a constant `name` | stamp `RUN_TAG=${LSB_JOBID}` into the name (§4) |

Cheap-failure discipline: smoke every new configuration first (`MAX_STEPS=10 BATCH=2 WANDB=null`
on 1 GPU, `-W 1:00`). Config-resolution failures surface in ~90 s and cost nothing; a
bad 4-GPU launch can idle four H200s for hours.

---

## 9. Resume semantics (read before resubmitting a crashed long run)

`configs/config.yaml` sets `hydra.run.dir: ./outputs/${data.train}/${now:%Y.%m.%d}/${now:%H%M%S}`
with `chdir: true`, and `checkpointing.save_dir: ${cwd:}` with
`resume_from_ckpt: true`, `resume_ckpt_path: ${.save_dir}/checkpoints/last.ckpt`.

Consequence: **a plain resubmission starts a fresh timestamped output directory and
therefore does not resume** — it finds no `last.ckpt` and trains from scratch. To
actually continue a crashed run, point Hydra at the original directory:

```bash
"$PYTHON" -u main.py ... hydra.run.dir=outputs/carbon-prokaryote/2026.06.19/030312
```

(or override `checkpointing.resume_ckpt_path` explicitly). Checkpoints live in
`<run dir>/checkpoints/` as `<epoch>-<step>.ckpt` plus `last.ckpt`, and the launched
config is preserved in `<run dir>/.hydra/{config,overrides}.yaml` — which is also how
you identify which output directory belongs to which job:

```bash
grep -h "l_use_weight" outputs/synthLR12k/*/*/.hydra/overrides.yaml
```

---

## 9b. Four things that cost real time on 2026-08-26

**The advance reservation blocks you unless you ask for it.** Every job sat in
PEND with `Not enough job slot(s) while advance reservation is active: 9
hosts`, which reads like a busy cluster and is not. `brsvs` showed the group's
`iclr_2026` reservation holding 8 farm-gpu050x hosts with **1100 of 1280 CPUs
idle**. A job that does not pass `-U` is not merely failing to use the
reservation, it is blocked *by* it: the reservation withdraws those hosts from
the general pool, so not asking is strictly worse than the reservation not
existing. Seven jobs went PEND -> RUN the instant `bmod -U iclr_2026 -G s10396`
was applied.

    bsub -U iclr_2026 -G s10396 ... < script      # always, for GPU work

**You cannot raise `-W` on a running job.** The esub refuses every `bmod` on a
RUNNING job -- `Request aborted by esub` -- with or without `-G`, with or
without a matching `-gpu` spec. A wall limit that turns out to be too short can
only be fixed by killing and resuming from `last.ckpt`. Set it correctly at
submission; over-estimating costs nothing.

**`bmod` wants `-G` when PENDING and rejects it when RUNNING.** On a pending job
`bmod -W ... -G s10396 <id>` works. On a running job the same command returns
`Only the following parameters can be used to modify a running job: -c, -M, -W,
...` -- `-G` is not in that list, so it fails before the esub even sees it.

**A single-GPU job fragments a host and starves a 4-GPU job.** Twelve
concurrent 1-GPU jobs scattered across the reservation left no contiguous block
of four, and a 4-GPU arm waited hours behind them. If a multi-GPU job will not
schedule while capacity looks free, count GPUs *per host*, not in total. The
same applies to a CPU-only job that the esub forced to request a GPU it never
uses -- it still occupies one.

**Throughput read shortly after a resume is not real.** Not LSF, but it belongs
next to the log forensics: Lightning fast-forwards through already completed
batches on resume, so the progress bar's cumulative average is meaningless
until the replay is past. A restarted arm showed **91.95 it/s at batch 2001**
against a true steady state of 0.56. Always measure the MARGINAL rate over a
window:

    n0=$(...); sleep 300; n1=$(...); echo "$(( (n1-n0) / 300 )) it/s"

## 10. Quick reference

```bash
# submit
bsub < scripts/train/train_dna_bd3lm.sh
bsub -env "all, BATCH=2, MAX_STEPS=40000, WANDB=null" < scripts/train/train_oracle_human.sh
bsub -env "all" -J my_variant -W 2:00 -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAH200" < script.sh

# watch
bjobs -w | awk '{print $1,$3,$7}'
bjobs -noheader -o 'jobid stat run_time' <id>
bpeek <id>                                   # live stdout of a running job
tr '\r' '\n' < logs/<name>_<id>.out | grep -iE 'loss|nll|Error|Exited' | tail -30

# stop
bkill <id>
bkill -J <name>

# cluster
bqueues | awk 'NR==1 || /training/'
```
