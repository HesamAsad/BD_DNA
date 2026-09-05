"""Fixed-offset DUPLICATION benchmark -- replaces the random-motif "echo" task.

WHY THE ECHO TASK FAILED (diagnosed 2026-07-20 after two dead runs):
  The echo task planted a random motif at s and copied it at s+g, with g drawn from a SET
  of gaps and s random per sequence. To reconstruct a masked target span the model must
  first work out WHERE to copy from -- but the target span is masked, so there is NO cue
  linking it to its source: the sequence around the target is unrelated random background,
  and g varies. Content-based retrieval is impossible when the query itself is hidden, so
  even a perfect model scores chance under the eval's full-span masking. Training only ever
  got a weak, ambiguous partial-mask signal, and val/nll never left the uniform floor
  (1.386 = ln 4) at EITHER 0.78% or 15.6% echo density.

THE FIX -- make the source location UNAMBIGUOUS by construction:
  x[i] = x[i - D] for all i >= D, with D FIXED. Every position past D is predictable by
  copying from exactly D back. No retrieval ambiguity: the model learns one fixed relative
  offset (easy for RoPE-relative attention). Density is maximal (the whole tail), and the
  eval needs no visible cue, so full-span masking is legitimate.

USE IT AS A LADDER:
  D <  block_size*window_blocks : SANITY -- reachable by the windowed fine attention.
                                  If this does not learn, the pipeline/model is broken.
  D >> block_size*window_blocks : THE TEST -- unreachable by fine attention, so it can ONLY
                                  be solved through the coarse cross-attention route.

Emits an echo-manifest compatible with `main.py mode=synth_copy_eval`, so the existing
metric works unchanged (probe spans in the duplicated tail, each with gap=D).
"""
import argparse
import json
import os
import shutil

import numpy as np
import datasets

NUC = np.array([8, 9, 10, 11], dtype=np.int32)  # A C G T

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, default=12288)
ap.add_argument('--offset', type=int, required=True, help='fixed copy offset D')
ap.add_argument('--n_train', type=int, default=8192)
ap.add_argument('--n_val', type=int, default=512)
ap.add_argument('--motif_len', type=int, default=24, help='probe span length for the eval')
ap.add_argument('--n_probe', type=int, default=40, help='eval probe spans per val sequence')
ap.add_argument('--name', required=True)
ap.add_argument('--cache_dir',
                default='/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon')
ap.add_argument('--direction', choices=('forward', 'backward'), default='forward',
                help="forward: x[i]=x[i-D], solvable from the LEFT cache. "
                     "backward: x[:D]=x[D:], solvable ONLY from the RIGHT "
                     "cache -- the bidirectional test. See make_seq.")
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
L, D, M = args.length, args.offset, args.motif_len
assert 0 < D < L, 'need 0 < offset < length'
assert D + M < L, 'need room for at least one probe span past the offset'
if args.direction == 'backward' and L != 2 * D:
  raise SystemExit(
    f'backward needs --length exactly 2*offset (got L={L}, D={D}). Any longer '
    'and the copy pattern repeats, which restores a leftward route and defeats '
    'the whole point of the variant.')


def make_seq(rng):
  """Build one sequence in the requested direction.

  forward:  x[:D] random, x[i] = x[i-D] thereafter (tiling if L > 2D).

  backward: x[D:] random, x[:D] = x[D:2D]. The target half is a copy of the
  half that comes AFTER it, so the only source is D positions to the RIGHT.

  WHY THE FORWARD TASK CANNOT TEST BIDIRECTIONALITY. Tiling makes the sequence
  PERIODIC with period D, so x[i] equals x[i-D] *and* x[i+D]: a left-only model
  solves it exactly as well as a bidirectional one. Every copy-gate number
  measured so far therefore says nothing about the right cache.

  HOW THE BACKWARD VARIANT DISCRIMINATES, and it is NOT that left-only scores
  chance -- copying is symmetric, so with two identical halves each half is
  reachable from one side:

      target in [D, 2D)  ->  source at i-D, to the LEFT   (any model)
      target in [0, D)   ->  source at i+D, to the RIGHT  (needs right cache)

  So a left-only model tops out at HALF the positions and lands near
  recovered = 0.5, while a model that reads its right cache goes to ~1.0. The
  50% ceiling is the signal.

  *** THIS BREAKS THE GATE'S DEFAULT PASS THRESHOLD. *** PASS at 0.50 is
  exactly the left-only ceiling, so on this variant it certifies the failure
  mode it is meant to detect. Read backward runs against ~0.5 (left-only) and
  ~1.0 (bidirectional); anything near 0.5 means the right cache went unused.

  This needs L = 2D, enforced below. Any L > 2D would let the pattern repeat
  and reintroduce a leftward route, which is precisely the flaw being avoided.
  The offset stays FIXED, so there is no retrieval ambiguity of the kind that
  made the old "echo" benchmark unsolvable even in principle.
  """
  x = np.empty(L, dtype=np.int32)
  if args.direction == 'backward':
    x[D:] = NUC[rng.integers(0, 4, size=L - D)]
    x[:D] = x[D:2 * D]
    return x
  x[:D] = NUC[rng.integers(0, 4, size=D)]
  for i in range(D, L, D):
    j = min(i + D, L)
    x[i:j] = x[i - D:i - D + (j - i)]
  return x


def probes(rng):
  """Probe spans inside the predictable region, which differs by direction."""
  out = []
  for _ in range(args.n_probe):
    if args.direction == 'backward':
      t = int(rng.integers(0, D - M))          # target is the LEADING copy
      out.append({'source': t + D, 'target': t, 'gap': D, 'motif_len': M})
    else:
      t = int(rng.integers(D, L - M))
      out.append({'source': t - D, 'target': t, 'gap': D, 'motif_len': M})
  return out


FEATS = datasets.Features({
  'input_ids': datasets.Sequence(datasets.Value('int32')),
  'attention_mask': datasets.Sequence(datasets.Value('float32'))})


def save(ds, mode):
  p = os.path.join(args.cache_dir, f'{args.name}_{mode}_bs{L}_wrapped_specialFalse.dat')
  tmp = p + '.tmp'
  shutil.rmtree(tmp, ignore_errors=True)
  ds.save_to_disk(tmp)
  shutil.rmtree(p, ignore_errors=True)
  os.rename(tmp, p)
  print(f'saved {mode}: rows={ds.num_rows} -> {p}', flush=True)


def train_gen():
  rng = np.random.default_rng(args.seed)
  for _ in range(args.n_train):
    yield {'input_ids': make_seq(rng), 'attention_mask': np.ones(L, np.float32)}


frac_pred = (L - D) / L
print(f'building {args.name}: L={L} offset={D} -> {100*frac_pred:.1f}% of positions '
      f'predictable by copy; ideal val/nll = {(D/L)*np.log(4):.3f} nats '
      f'(floor if unlearned = {np.log(4):.3f})', flush=True)

save(datasets.Dataset.from_generator(train_gen, features=FEATS,
                                     writer_batch_size=max(64, 250_000_000 // (L * 4))),
     'train')

rng = np.random.default_rng(args.seed + 1)
rows, mani = [], []
for _ in range(args.n_val):
  rows.append(make_seq(rng))
  mani.append(probes(rng))
save(datasets.Dataset.from_dict(
  {'input_ids': [r for r in rows],
   'attention_mask': [np.ones(L, np.float32) for _ in rows]}, features=FEATS), 'validation')
json.dump({'args': vars(args), 'echoes': mani},
          open(os.path.join(args.cache_dir, f'{args.name}_echo_manifest.json'), 'w'))
print(f'val probe spans: {sum(len(e) for e in mani)} (all gap={D})', flush=True)
print('DONE', flush=True)
