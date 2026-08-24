"""Numerical proof that the RC-equivariant BiSSM satisfies f(rc(x)) == rc(f(x)).

CPU only, float64, tiny width -- runs on a head node in a few seconds. No CUDA,
no LSF. Run:

    /software/cellgen/team361/ha11/envs/nichejepa/bin/python \
      scripts/smoke/rc_equivariance.py

Tests follow the design's T0-T9. Every tolerance is stated and every measured
residual is printed next to the scale it is compared against, so a "pass" that
is really a degenerate zero is visible.

The load-bearing assertion is T3/T5:

    logits(rc(x)) == flip(logits(x), 1)[..., PI]

with PI the complement permutation on the 13-token DNA vocabulary. Note the
`[..., PI]`: Caduceus's alphabet is exactly {A,C,G,T} so *reversing* four
logits is complementation, but our vocabulary has eight specials before A and N
after T, so a channel flip would map [CLS] <-> N. T3 asserts the flip spelling
FAILS, so the shortcut cannot creep back in.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from models.bidirectional_ssm import BidirectionalSSM  # noqa: E402
from models.rc_equivariance import (  # noqa: E402
  DNA_COMPLEMENT_IDS, complement_permutation, rc_token_ids, swap_halves)

VOCAB = 13
BLOCK = 4
FAILURES: list[str] = []
_RESULTS: list[tuple[str, str]] = []


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def check(name: str, condition: bool, detail: str = ""):
  status = "PASS" if condition else "FAIL"
  print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
  if not condition:
    FAILURES.append(name)


def section(title: str):
  print(f"\n{title}\n" + "-" * len(title))


def config(*, rc: bool, time_conditioning: bool = True, impl: str = "fused"):
  return OmegaConf.create({
    "block_size": BLOCK,
    "algo": {"time_conditioning": time_conditioning},
    "model": {
      "hidden_size": 8,
      "cond_dim": 8,
      "n_blocks": 3,
      "dropout": 0.0,               # dropout breaks equivariance stochastically
      "tie_word_embeddings": True,
      "ssm_state_size": 3,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 4,
      "ssm_chunk_size": 4,
      "ssm_backend": "torch",       # fused SSD kernel is CUDA only
      "mlp_ratio": 2.0,
      "bidirectional_impl": impl,
      "rc_equivariant": rc,
    },
  })


def build(*, rc: bool, seed: int = 11, impl: str = "fused",
          time_conditioning: bool = True, double: bool = True, cls=None):
  torch.manual_seed(seed)
  cls = cls or BidirectionalSSM
  model = cls(
    config(rc=rc, impl=impl, time_conditioning=time_conditioning),
    vocab_size=VOCAB).eval()
  return model.double() if double else model


def layer_trace(model, ids, sigma=None, left=None, right=None):
  """Embedding output, every layer output, and the final norm.

  Mirrors `BidirectionalSSM.forward_active` (models/bidirectional_ssm.py:611)
  so a break can be localised to one sublayer instead of only showing up in the
  logits.
  """
  x = model.token_embedding(ids)
  if model.time_embedding is not None:
    x = x + model.time_embedding(sigma)[:, None, :]
  batch = ids.shape[0]
  if left is None:
    left = model._empty_cache(batch, x.device, x.dtype, "left")
  if right is None:
    right = model._empty_cache(batch, x.device, x.dtype, "right")
  trace = [("embedding", x)]
  for index, layer in enumerate(model.layers):
    x = layer.scan_active(x, left.states[index], right.states[index])
    trace.append((f"layer{index}", x))
  trace.append(("final_norm", model.final_norm(x)))
  return trace


def err(a, b):
  return (a - b).abs().max().item()


def scale(t):
  return t.abs().mean().item()


# ---------------------------------------------------------------------------
# T0 -- the flag-off path is byte identical to the pre-change code
# ---------------------------------------------------------------------------

def t0_flag_off_is_inert():
  section("T0  flag off == git HEAD, bit for bit")
  workdir = Path(tempfile.mkdtemp(prefix="rc_baseline_"))
  pkg = workdir / "models_baseline"
  pkg.mkdir()
  (pkg / "__init__.py").write_text("")
  for name in ("bidirectional_ssm.py", "mamba2_segment.py"):
    blob = subprocess.check_output(
      ["git", "show", f"HEAD:models/{name}"], cwd=REPO)
    (pkg / name).write_bytes(blob)
  sys.path.insert(0, str(workdir))
  try:
    baseline = importlib.import_module("models_baseline.bidirectional_ssm")
  finally:
    sys.path.remove(str(workdir))

  # promote_types is the identity for every dtype production uses, which is
  # what makes the RMSNorm/_gated_norm edits provably inert.
  for dtype in (torch.float32, torch.bfloat16, torch.float16):
    check(f"promote_types({dtype}, fp32) is fp32",
          torch.promote_types(dtype, torch.float32) is torch.float32)

  for impl in ("fused", "split"):
    old = build(rc=False, impl=impl, double=False, cls=baseline.BidirectionalSSM)
    new = build(rc=False, impl=impl, double=False)
    same_keys = list(old.state_dict()) == list(new.state_dict())
    check(f"state_dict keys unchanged ({impl})", same_keys,
          f"{len(list(new.state_dict()))} tensors")
    same_weights = all(
      torch.equal(old.state_dict()[k], new.state_dict()[k])
      for k in old.state_dict())
    check(f"initial weights identical ({impl})", same_weights)
    check(f"no parametrizations registered ({impl})",
          not any("parametrizations" in k for k in new.state_dict()))

    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
    sigma = torch.rand(2)
    with torch.no_grad():
      a = old.forward_active(ids, sigma)
      b = new.forward_active(ids, sigma)
      la = old.prefill_left_boundaries_stacked(ids, BLOCK)
      lb = new.prefill_left_boundaries_stacked(ids, BLOCK)
    check(f"fp32 logits bitwise identical ({impl})", torch.equal(a, b),
          f"max|d| = {err(a, b):.3e}")
    check(f"fp32 boundary caches bitwise identical ({impl})",
          all(torch.equal(x.conv, y.conv) and torch.equal(x.ssm, y.ssm)
              for x, y in zip(la.states, lb.states)))


# ---------------------------------------------------------------------------
# T1 -- the pre-existing plain-reversal symmetry (a DEFECT, documented)
# ---------------------------------------------------------------------------

def t1_plain_reversal_equivariance():
  section("T1  backbone is exactly equivariant to plain reversal (the defect)")
  for rc in (False, True):
    for impl in ("fused", "split"):
      model = build(rc=rc, impl=impl)
      ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
      sigma = torch.rand(2, dtype=torch.float64)
      with torch.no_grad():
        h = layer_trace(model, ids, sigma)[-1][1]
        h_flip = layer_trace(model, ids.flip(1), sigma)[-1][1]
      tag = "rc_equivariant" if rc else "baseline"
      check(f"h(flip x) == flip h(x)  [{tag}/{impl}]",
            err(h_flip, h.flip(1)) < 1e-12,
            f"max|d| = {err(h_flip, h.flip(1)):.3e}, mean|h| = {scale(h):.3f}")
      pooled, pooled_flip = h.mean(1), h_flip.mean(1)
      check(f"meanpool is reversal INVARIANT [{tag}/{impl}]",
            err(pooled, pooled_flip) < 1e-12,
            f"max|d| = {err(pooled, pooled_flip):.3e}")
  print("  NOTE: plain reversal is not a symmetry of DNA. The mean-pooled")
  print("        GenomicBenchmarks classifier therefore cannot tell TATAAA")
  print("        from AAATAT. The Z2 construction does NOT remove this.")


# ---------------------------------------------------------------------------
# T2 -- the complement permutation itself
# ---------------------------------------------------------------------------

def t2_complement_permutation():
  section("T2  complement permutation vs the string oracle")
  import dataloader
  tokenizer = dataloader.DNATokenizer()
  perm = complement_permutation(VOCAB, tokenizer=tokenizer)
  check("tokenizer.vocab_size == 13", tokenizer.vocab_size == VOCAB)
  check("pi is an involution",
        torch.equal(perm[perm], torch.arange(VOCAB)))
  check("A<->T, C<->G",
        perm[8].item() == 11 and perm[11].item() == 8
        and perm[9].item() == 10 and perm[10].item() == 9)
  check("N is a fixed point", perm[12].item() == 12)
  check("all eight specials are fixed points",
        all(perm[i].item() == i for i in range(8)))
  check("[MASK] is a fixed point (commutes with q_xt)",
        perm[tokenizer.mask_token_id].item() == tokenizer.mask_token_id)
  check("[UNK] is a fixed point", perm[tokenizer.unk_token_id].item()
        == tokenizer.unk_token_id)
  check("literal table matches", tuple(perm.tolist()) == DNA_COMPLEMENT_IDS)

  from scripts.eval.dnahnet.deg import reverse_complement
  torch.manual_seed(5)
  agree = True
  for _ in range(50):
    n = int(torch.randint(1, 40, ()).item())
    seq = "".join("ACGTN"[i] for i in torch.randint(0, 5, (n,)).tolist())
    ids = torch.tensor(
      [tokenizer.convert_tokens_to_ids(list(seq))], dtype=torch.long)
    oracle = torch.tensor(
      [tokenizer.convert_tokens_to_ids(list(reverse_complement(seq)))],
      dtype=torch.long)
    agree &= torch.equal(rc_token_ids(ids, perm), oracle)
  check("rc_token_ids == encode(deg.reverse_complement(s)) on 50 strings",
        agree)


# ---------------------------------------------------------------------------
# T3 -- the headline assertion
# ---------------------------------------------------------------------------

def t3_head_is_rc_equivariant():
  section("T3  logits(rc x) == flip(logits x)[..., PI]   (empty caches)")
  perm = complement_permutation(VOCAB)
  for impl in ("fused", "split"):
    for time_conditioning in (True, False):
      model = build(rc=True, impl=impl, time_conditioning=time_conditioning)
      ids = torch.randint(0, VOCAB, (3, 4 * BLOCK))
      sigma = (torch.rand(3, dtype=torch.float64)
               if time_conditioning else None)
      with torch.no_grad():
        logits = model.forward_active(ids, sigma)
        logits_rc = model.forward_active(rc_token_ids(ids, perm), sigma)
      target = logits.flip(1)[..., perm]
      residual = err(logits_rc, target)
      tag = f"{impl}/tcond={time_conditioning}"
      check(f"RC equivariant [{tag}]", residual < 1e-12,
            f"max|d| = {residual:.3e}, mean|logits| = {scale(logits):.4f}")
      # negative control: the Caduceus `flip_chan` shortcut must NOT work here
      naive = logits.flip(1).flip(-1)
      check(f"channel-flip shortcut is rejected [{tag}]",
            err(logits_rc, naive) > 1e-3 * scale(logits),
            f"max|d| = {err(logits_rc, naive):.3e}")

  # a plain baseline model must fail the same assertion, or T3 proves nothing
  model = build(rc=False)
  ids = torch.randint(0, VOCAB, (2, 2 * BLOCK))
  sigma = torch.rand(2, dtype=torch.float64)
  with torch.no_grad():
    logits = model.forward_active(ids, sigma)
    logits_rc = model.forward_active(rc_token_ids(ids, perm), sigma)
  residual = err(logits_rc, logits.flip(1)[..., perm])
  check("baseline (flag off) is NOT RC equivariant",
        residual > 1e-3 * scale(logits),
        f"max|d| = {residual:.3e}, mean|logits| = {scale(logits):.4f}")


# ---------------------------------------------------------------------------
# T4 -- localise a break to one sublayer
# ---------------------------------------------------------------------------

def t4_every_layer_is_rc_equivariant():
  section("T4  h_l(rc x) == rho(flip(h_l(x)))   per sublayer")
  perm = complement_permutation(VOCAB)
  model = build(rc=True)
  ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
  sigma = torch.rand(2, dtype=torch.float64)
  with torch.no_grad():
    plain = layer_trace(model, ids, sigma)
    rc = layer_trace(model, rc_token_ids(ids, perm), sigma)
  for (name, h), (_, h_rc) in zip(plain, rc):
    target = swap_halves(h.flip(1))
    check(f"{name}", err(h_rc, target) < 1e-12,
          f"max|d| = {err(h_rc, target):.3e}, mean|h| = {scale(h):.4f}")


# ---------------------------------------------------------------------------
# T5 -- block diffusion with real boundary caches
# ---------------------------------------------------------------------------

def t5_block_diffusion_with_real_caches():
  section("T5  block diffusion, real prefill caches, left<->right under rc")
  perm = complement_permutation(VOCAB)
  for impl in ("fused", "split"):
    for boundary in ("layer_major", "block_major"):
      torch.manual_seed(11)
      cfg = config(rc=True, impl=impl)
      cfg.model.boundary_impl = boundary
      model = BidirectionalSSM(cfg, vocab_size=VOCAB).eval().double()
      ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
      sigma = torch.rand(2, dtype=torch.float64)
      ids_rc = rc_token_ids(ids, perm)

      def denoise(tokens):
        left = model.prefill_left(tokens[:, :BLOCK])
        right = model.prefill_right(tokens[:, 2 * BLOCK:])
        return model.forward_active(
          tokens[:, BLOCK:2 * BLOCK], sigma, left, right)

      with torch.no_grad():
        logits = denoise(ids)
        logits_rc = denoise(ids_rc)
      target = logits.flip(1)[..., perm]
      residual = err(logits_rc, target)
      check(f"C-a infilling is RC equivariant [{impl}/{boundary}]",
            residual < 1e-12,
            f"max|d| = {residual:.3e}, mean|logits| = {scale(logits):.4f}")

      # the stacked all-blocks path, which is what training actually runs
      with torch.no_grad():
        left_all = model.prefill_left_boundaries_stacked(ids, BLOCK)
        right_all = model.prefill_right_boundaries_stacked(ids, BLOCK)
        blocks = ids.reshape(-1, BLOCK)
        sig_all = sigma.repeat_interleave(3)
        all_logits = model.forward_active(
          blocks, sig_all, left_all, right_all).reshape(2, 3 * BLOCK, VOCAB)

        left_rc = model.prefill_left_boundaries_stacked(ids_rc, BLOCK)
        right_rc = model.prefill_right_boundaries_stacked(ids_rc, BLOCK)
        all_rc = model.forward_active(
          ids_rc.reshape(-1, BLOCK), sig_all, left_rc, right_rc
        ).reshape(2, 3 * BLOCK, VOCAB)
      target = all_logits.flip(1)[..., perm]
      residual = err(all_rc, target)
      check(f"all-blocks stacked objective is RC equivariant "
            f"[{impl}/{boundary}]", residual < 1e-12,
            f"max|d| = {residual:.3e}, mean|logits| = {scale(all_logits):.4f}")


# ---------------------------------------------------------------------------
# T6 -- the de-novo objective stays directed (guards against over-claiming)
# ---------------------------------------------------------------------------

def t6_denovo_objective_is_directed():
  section("T6  de-novo objective (empty right cache) is NOT RC symmetric")
  perm = complement_permutation(VOCAB)
  model = build(rc=True)
  ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
  sigma = torch.rand(2, dtype=torch.float64)

  def denoise(tokens):
    left = model.prefill_left(tokens[:, :BLOCK])
    return model.forward_active(tokens[:, BLOCK:2 * BLOCK], sigma, left, None)

  with torch.no_grad():
    logits = denoise(ids)
    logits_rc = denoise(rc_token_ids(ids, perm))
  residual = err(logits_rc, logits.flip(1)[..., perm])
  check("right_flank_probability=0.0 keeps a direction",
        residual > 1e-3 * scale(logits),
        f"max|d| = {residual:.3e}, mean|logits| = {scale(logits):.4f}")
  print("  NOTE: this is a feature. BD3-LM factorises left to right, so RC")
  print("        augmentation supplies the reverse-order block factorisation")
  print("        that no forward pass on x can produce. Do NOT 'fix' it by")
  print("        symmetrising the loss.")


# ---------------------------------------------------------------------------
# T7 -- augmentation commutes with the absorbing corruption
# ---------------------------------------------------------------------------

def t7_augmentation_commutes_with_masking():
  section("T7  rc augmentation commutes with q_xt, and p=0 is inert")
  import dataloader
  import diffusion as diffusion_module
  tokenizer = dataloader.DNATokenizer()
  perm = complement_permutation(VOCAB, tokenizer=tokenizer)
  mask_id = tokenizer.mask_token_id
  check("pi fixes the mask id", perm[mask_id].item() == mask_id)

  torch.manual_seed(7)
  x0 = torch.randint(8, 13, (4, 16))
  keep = torch.rand(4, 16) > 0.3
  noisy = torch.where(keep, x0, torch.full_like(x0, mask_id))
  lhs = rc_token_ids(noisy, perm)
  rhs = torch.where(keep.flip(1), rc_token_ids(x0, perm),
                    torch.full_like(x0, mask_id))
  check("rc(q_xt(x0, M)) == q_xt(rc(x0), flip(M))", torch.equal(lhs, rhs))

  class _Stub:
    pass

  stub = _Stub()
  stub.training = True
  stub.rc_augment_probability = 1.0
  stub._complement_token_ids = perm
  mask = torch.ones(4, 16)
  mask[:, 12:] = 0.0
  out, out_mask = diffusion_module.Diffusion._maybe_rc_augment(stub, x0, mask)
  check("_maybe_rc_augment(p=1) == rc_token_ids",
        torch.equal(out, rc_token_ids(x0, perm)))
  check("_maybe_rc_augment flips the attention mask too",
        torch.equal(out_mask, mask.flip(1)))
  check("_maybe_rc_augment is an involution",
        torch.equal(
          diffusion_module.Diffusion._maybe_rc_augment(stub, out, out_mask)[0],
          x0))

  stub.rc_augment_probability = 0.0
  before = torch.random.get_rng_state()
  out, out_mask = diffusion_module.Diffusion._maybe_rc_augment(stub, x0, mask)
  check("p=0 returns the inputs unchanged",
        out is x0 and out_mask is mask)
  check("p=0 draws no RNG (existing runs keep their random stream)",
        torch.equal(before, torch.random.get_rng_state()))

  stub.rc_augment_probability = 1.0
  stub.training = False
  out, _ = diffusion_module.Diffusion._maybe_rc_augment(stub, x0, mask)
  check("validation is never augmented", out is x0)


# ---------------------------------------------------------------------------
# T8 -- the padding trap
# ---------------------------------------------------------------------------

def t8_padding_trap():
  section("T8  RC must be applied to the string, not to the padded tensor")
  import dataloader
  from scripts.eval.dnahnet.deg import reverse_complement
  from scripts.eval.dnahnet.score_mavedb import encode_dna
  tokenizer = dataloader.DNATokenizer()
  perm = complement_permutation(VOCAB, tokenizer=tokenizer)
  window = 24
  seq = "ACGTTTGCACGT"
  ids, _ = encode_dna(tokenizer, seq, window)
  ids = torch.tensor([ids])
  correct, _ = encode_dna(tokenizer, reverse_complement(seq), window)
  correct = torch.tensor([correct])
  wrong = rc_token_ids(ids, perm)
  check("encode(rc(s)) != rc_ids(encode(s)) for a padded window",
        not torch.equal(correct, wrong))
  n_id = tokenizer.convert_tokens_to_ids("N")
  check("the correct encoding keeps the N pad on the RIGHT",
        bool((correct[0, len(seq):] == n_id).all()))
  check("the tensor-level RC puts the N pad on the LEFT",
        bool((wrong[0, :window - len(seq)] == n_id).all()))

  # the two harnesses that actually do post-hoc conjoining must avoid the trap
  from scripts.eval.caduceus.finetune import (
    build_complement_table, reverse_complement_ids)
  keep = torch.zeros(1, window, dtype=torch.bool)
  keep[0, :len(seq)] = True
  harness = reverse_complement_ids(
    ids, keep, build_complement_table(tokenizer, VOCAB))
  check("finetune.reverse_complement_ids matches encode(rc(s)) on the span",
        torch.equal(harness[0, :len(seq)], correct[0, :len(seq)]))
  check("finetune.reverse_complement_ids leaves the pad in place",
        torch.equal(harness[0, len(seq):], ids[0, len(seq):]))
  import scripts.eval.caduceus.embed as embed_module

  def names(code):
    out = set(code.co_names)
    for const in code.co_consts:
      if hasattr(const, "co_names"):      # comprehensions are nested code objects
        out |= names(const)
    return out

  check("embed.embed_sequences takes rc_tta and builds RC from the string",
        'rc_tta' in embed_module.embed_sequences.__code__.co_varnames
        and 'reverse_complement' in names(embed_module.embed_sequences.__code__))
  print("  NOTE: the scan runs through the pad, so a naive tensor flip gives")
  print("        different hidden states. finetune.py reverses within the")
  print("        real span; embed.py reverse-complements the STRING.")


# ---------------------------------------------------------------------------
# T9 -- training keeps the constraint; parameter accounting
# ---------------------------------------------------------------------------

def t9_training_and_cost():
  section("T9  gradients flow, the constraint survives a step, param cost")
  perm = complement_permutation(VOCAB)
  model = build(rc=True)
  ids = torch.randint(0, VOCAB, (2, 3 * BLOCK))
  sigma = torch.rand(2, dtype=torch.float64)
  logits = model.forward_active(ids, sigma)
  logits.square().mean().backward()
  grads = [p for p in model.parameters() if p.grad is not None]
  check("every free parameter receives a gradient",
        len(grads) == len(list(model.parameters())),
        f"{len(grads)}/{len(list(model.parameters()))}")
  check("gradients are non-trivial",
        max(p.grad.abs().max().item() for p in grads) > 0)

  optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
  optimizer.step()
  with torch.no_grad():
    a = model.forward_active(ids, sigma)
    b = model.forward_active(rc_token_ids(ids, perm), sigma)
  residual = err(b, a.flip(1)[..., perm])
  check("still RC equivariant after an optimizer step", residual < 1e-12,
        f"max|d| = {residual:.3e}, mean|logits| = {scale(a):.4f}")

  free = sum(p.numel() for p in build(rc=True, double=False).parameters())
  full = sum(p.numel() for p in build(rc=False, double=False).parameters())
  check("free parameters are roughly halved", free < 0.65 * full,
        f"{free} vs {full} ({free / full:.3f}x) at hidden_size=8")
  _RESULTS.append(("tiny model free/full params", f"{free}/{full}"))


# ---------------------------------------------------------------------------
# T10 -- the production shapes, where rho has to be head aligned
# ---------------------------------------------------------------------------

def t10_production_shapes():
  section("T10  production widths (hidden 768, headdim 64, d_state 64)")
  perm = complement_permutation(VOCAB)
  cfg = OmegaConf.load(REPO / "configs" / "model" / "small_bissm_rc.yaml")
  # One layer and a short block keep the fp64 reference scan cheap; what this
  # test is for is the *shapes* -- hidden 768, expand 2, headdim 64, d_state 64
  # -- which are what decide whether rho can be head aligned at all.
  cfg.n_blocks = 1
  cfg.ssm_chunk_size = 8
  full = OmegaConf.create({
    "block_size": 8,
    "algo": {"time_conditioning": True},
    "model": cfg})
  torch.manual_seed(0)
  model = BidirectionalSSM(full, vocab_size=VOCAB).eval().double()
  mixer = model.layers[0].mixer
  check("rho is head aligned",
        (mixer.d_inner // 2) % mixer.headdim == 0 and mixer.nheads % 2 == 0,
        f"d_inner={mixer.d_inner} headdim={mixer.headdim} "
        f"nheads={mixer.nheads} -> {mixer.nheads // 2} heads per half")
  ids = torch.randint(0, VOCAB, (1, 24))
  sigma = torch.rand(1, dtype=torch.float64)
  with torch.no_grad():
    left = model.prefill_left(ids[:, :8])
    right = model.prefill_right(ids[:, 16:])
    logits = model.forward_active(ids[:, 8:16], sigma, left, right)
    ids_rc = rc_token_ids(ids, perm)
    left_rc = model.prefill_left(ids_rc[:, :8])
    right_rc = model.prefill_right(ids_rc[:, 16:])
    logits_rc = model.forward_active(ids_rc[:, 8:16], sigma, left_rc, right_rc)
  residual = err(logits_rc, logits.flip(1)[..., perm])
  check("RC equivariant at production width", residual < 1e-12,
        f"max|d| = {residual:.3e}, mean|logits| = {scale(logits):.4f}")

  # the number the report quotes, computed rather than asserted
  wide = OmegaConf.load(REPO / "configs" / "model" / "small_bissm_rc.yaml")
  plain = OmegaConf.load(REPO / "configs" / "model" / "small_bissm.yaml")
  counts = {}
  for name, model_cfg in (("rc", wide), ("baseline", plain)):
    torch.manual_seed(0)
    built = BidirectionalSSM(
      OmegaConf.create({"block_size": 256,
                        "algo": {"time_conditioning": True},
                        "model": model_cfg}),
      vocab_size=VOCAB)
    counts[name] = sum(p.numel() for p in built.parameters())
    del built
  _RESULTS.append((
    "hidden_size=768 free params",
    f"{counts['rc']:,} vs {counts['baseline']:,} "
    f"({counts['rc'] / counts['baseline']:.4f}x)"))
  check("hidden_size=768 free parameters halve",
        counts['rc'] < 0.55 * counts['baseline'],
        f"{counts['rc']:,} vs {counts['baseline']:,}")


def main():
  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
  torch.manual_seed(0)
  torch.use_deterministic_algorithms(False)
  print("RC equivariance smoke test -- CPU, float64, ssm_backend='torch'")
  print(f"torch {torch.__version__}; repo {REPO}")
  for test in (t0_flag_off_is_inert,
               t1_plain_reversal_equivariance,
               t2_complement_permutation,
               t3_head_is_rc_equivariant,
               t4_every_layer_is_rc_equivariant,
               t5_block_diffusion_with_real_caches,
               t6_denovo_objective_is_directed,
               t7_augmentation_commutes_with_masking,
               t8_padding_trap,
               t9_training_and_cost,
               t10_production_shapes):
    test()
  section("summary")
  for name, value in _RESULTS:
    print(f"  {name}: {value}")
  if FAILURES:
    print(f"  {len(FAILURES)} FAILED: {FAILURES}")
    return 1
  print("  all checks passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
