import torch
from omegaconf import OmegaConf

from models.unidirectional_ssm import UnidirectionalSSM
from scripts.eval.ssm_prefix_intervention import _ar_target_losses
from scripts.eval.ssm_streaming_benchmark import sequence_diagnostics


def _model():
  config = OmegaConf.create({
    "block_size": 4,
    "algo": {"parameterization": "ar", "time_conditioning": False},
    "model": {
      "hidden_size": 8,
      "cond_dim": 8,
      "n_blocks": 2,
      "dropout": 0.0,
      "tie_word_embeddings": True,
      "right_flank_probability": 0.0,
      "ssm_state_size": 3,
      "ssm_conv_size": 4,
      "ssm_expand": 2,
      "ssm_head_dim": 4,
      "ssm_chunk_size": 4,
      "ssm_backend": "torch",
      "mlp_ratio": 2.0,
    },
  })
  torch.manual_seed(9)
  return UnidirectionalSSM(config, vocab_size=13).eval()


def test_true_prefix_ar_intervention_matches_full_causal_scoring():
  model = _model()
  sequence = torch.randint(0, 12, (2, 12))
  target_start = 8
  actual = _ar_target_losses(
    type("Wrapper", (), {"backbone": model, "mask_index": 12,
                          "neg_infinity": -1_000_000.0})(),
    sequence, sequence.roll(1, 0), target_start, radius=2,
    condition="true")

  logits = model.forward_active(sequence[:, :-1], None)
  logits[..., 12] = -1_000_000.0
  expected = -torch.gather(
    logits.log_softmax(-1)[:, target_start - 1:], -1,
    sequence[:, target_start:, None]).squeeze(-1)
  torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_sequence_diagnostics_reports_composition_and_repeats():
  class Tokenizer:
    def convert_ids_to_tokens(self, ids):
      alphabet = ["A", "C", "G", "T", "N"]
      return [alphabet[index] for index in ids]

  result = sequence_diagnostics(
    torch.tensor([0, 0, 0, 1, 2, 3, 4]), Tokenizer())
  assert result["acgt_fraction"] == 6 / 7
  assert result["gc_fraction"] == 2 / 6
  assert result["longest_homopolymer"] == 3
