"""RUNG C -- associative recall (MQAR-style) over DNA tokens.

WHY THIS RUNG EXISTS. The ladder's other rungs cannot fairly evaluate a recurrent state:
  A `synthDUPshort` D=512  : local copy, pipeline control (PASSES, val/nll 1.386->0.047)
  B `synthDUPlong`  D=6144 : EXACT copy 4 blocks back. The coarse k-mer route has the
                             information (bijective 6-mers) yet never learns to use it.
                             B is a HIGH-BANDWIDTH EXACT-RETRIEVAL STRESS TEST -- keep it
                             for the SSM too, just not as the sole acceptance gate.
                             (CORRECTION to an earlier claim of mine: B is NOT
                             information-theoretically "unfair" to an SSM. The retained
                             information scales with the COPIED SPAN, not the source-target
                             distance. This task is full duplication, x[i]=x[i-D] for all
                             i>=D, so it does demand a ~D-wide rolling buffer ~= 12 kbit --
                             but a Mamba-2 state of 4x768x16 scalars in bf16 is ~786 kbit
                             raw, well above that. So failure would reflect addressing /
                             overwriting / readout precision in the recurrence, NOT a
                             capacity impossibility.)
  C (this file)            : the dependence is COMPRESSIBLE -- a small dictionary (256 nt),
                             not an arbitrary sequence -- so it isolates statistical/
                             associative memory from raw retrieval bandwidth.

  Diagnostic value of running BOTH: passes B and C = strong exact + statistical memory;
  fails B passes C = good compressed state, weak exact retrieval; fails both = the state
  pathway is ineffective.

TASK. Block 0 holds a dictionary of `n_pairs` (key, value) motifs. Later blocks contain
queries: a VISIBLE key immediately followed by its value, which is the prediction target.
To fill a masked value the model must retrieve that key's value from block 0.

WHY IT IS WELL-POSED (the lesson from the failed echo task): the key sits immediately
before the masked value, so the retrieval cue is VISIBLE and unambiguous. The echo task
masked the target with no cue and was therefore unsolvable by any model.

WHY IT NEEDS LONG RANGE: the dictionary appears ONLY in block 0, and queries are placed in
blocks >= `first_query_block`, beyond the fine stream's +/-window_blocks reach -- so the
mapping can only arrive via the long-range route.

WHY AN SSM CAN PASS: the state need only carry the dictionary (n_pairs*(key+val) nt),
which is small and fixed -- unlike rung B's arbitrary-sequence recall. This is the DNA
analogue of MQAR, the standard probe for associative recall in SSM evaluations.

Emits an echo-manifest (source = value-in-dictionary, target = value-at-query), so
`main.py mode=synth_copy_eval` scores it unchanged.
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
ap.add_argument('--block_size', type=int, default=1536)
ap.add_argument('--n_pairs', type=int, default=16, help='dictionary entries in block 0')
ap.add_argument('--key_len', type=int, default=8)
ap.add_argument('--val_len', type=int, default=8)
ap.add_argument('--n_queries', type=int, default=120, help='queries spread over later blocks')
ap.add_argument('--first_query_block', type=int, default=2,
                help='queries start here; must exceed window_blocks so fine attn cannot reach')
ap.add_argument('--n_train', type=int, default=8192)
ap.add_argument('--n_val', type=int, default=512)
ap.add_argument('--name', default='synthRECALL')
ap.add_argument('--cache_dir',
                default='/lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms/data_cache/carbon')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
L, B, K = args.length, args.block_size, args.n_pairs
KL, VL = args.key_len, args.val_len
PAIR = KL + VL
assert K * PAIR <= B, f'dictionary ({K*PAIR}) must fit in block 0 ({B})'
QSTART = args.first_query_block * B
assert QSTART + PAIR < L, 'no room for queries'


def make_seq(rng):
  x = NUC[rng.integers(0, 4, size=L)]
  # --- block 0: the dictionary (unique keys) ---
  keys, seen = [], set()
  while len(keys) < K:
    k = tuple(rng.integers(0, 4, size=KL))
    if k not in seen:
      seen.add(k)
      keys.append(np.array(k))
  vals = [NUC[rng.integers(0, 4, size=VL)] for _ in range(K)]
  dict_pos = []
  for j in range(K):
    off = j * PAIR
    x[off:off + KL] = NUC[keys[j]]
    x[off + KL:off + PAIR] = vals[j]
    dict_pos.append(off + KL)                       # where this value lives
  # --- later blocks: queries (visible key, then the value = target) ---
  slots = list(range(QSTART, L - PAIR, PAIR))
  rng.shuffle(slots)
  probes = []
  for s in slots[:args.n_queries]:
    j = int(rng.integers(0, K))
    x[s:s + KL] = NUC[keys[j]]
    x[s + KL:s + PAIR] = vals[j]
    probes.append({'source': dict_pos[j], 'target': s + KL,
                   'gap': (s + KL) - dict_pos[j], 'motif_len': VL})
  return x, probes


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
    x, _ = make_seq(rng)
    yield {'input_ids': x, 'attention_mask': np.ones(L, np.float32)}


pred_frac = args.n_queries * VL / L
print(f'building {args.name}: L={L} block={B} dict={K}x({KL}+{VL}) queries={args.n_queries} '
      f'from block {args.first_query_block} | {100*pred_frac:.1f}% of positions are '
      f'retrievable targets | state needed = {K*PAIR} nt (fixed, SSM-compatible)', flush=True)

save(datasets.Dataset.from_generator(train_gen, features=FEATS,
                                     writer_batch_size=max(64, 250_000_000 // (L * 4))),
     'train')

rng = np.random.default_rng(args.seed + 1)
rows, mani = [], []
for _ in range(args.n_val):
  x, pr = make_seq(rng)
  rows.append(x)
  mani.append(pr)
save(datasets.Dataset.from_dict(
  {'input_ids': [r for r in rows],
   'attention_mask': [np.ones(L, np.float32) for _ in rows]}, features=FEATS), 'validation')
json.dump({'args': vars(args), 'echoes': mani},
          open(os.path.join(args.cache_dir, f'{args.name}_echo_manifest.json'), 'w'))
gaps = [p['gap'] for e in mani for p in e]
print(f'val probes: {len(gaps)} | gap min={min(gaps)} max={max(gaps)} '
      f'(all cross-block: min gap spans {min(gaps)//B} blocks)', flush=True)
print('DONE', flush=True)
