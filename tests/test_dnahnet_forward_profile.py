import pytest
import torch

from scripts.eval.dnahnet.profile_forward import DEFAULT_LENGTHS, parse_lengths
from scripts.eval.dnahnet.score_mavedb import _loss_from_fixed_corruption


def test_default_lengths_match_dnahnet_appendix_range():
  assert DEFAULT_LENGTHS == tuple(2 ** exponent for exponent in range(10, 20))


def test_parse_lengths_accepts_commas_and_spaces():
  assert parse_lengths("1024, 2048 4096") == (1024, 2048, 4096)


@pytest.mark.parametrize("value", ["", "0", "1024,1024", "-1"])
def test_parse_lengths_rejects_invalid_values(value):
  with pytest.raises(ValueError):
    parse_lengths(value)


def test_fixed_corruption_expands_sequence_time_for_multiblock_bissm():
  class Noise:
    sigma_max = 20.0

    def __call__(self, t):
      return torch.ones_like(t), torch.full_like(t, 0.5)

  class Model:
    noise = Noise()
    mdlm_loss_scale = False
    ignore_bos = False
    mask_index = 12
    config = type("Config", (), {
      "algo": type("Algo", (), {"backbone": "bissm"})()})()

    @staticmethod
    def _sigma_from_p(p):
      return -torch.log1p(-p)

    @staticmethod
    def _forward_pass_bissm(x0, xt, p, loss_scale):
      assert p.shape == x0.shape
      assert loss_scale.shape == x0.shape
      return torch.zeros_like(p)

  x0 = torch.ones((2, 8), dtype=torch.long)
  losses = _loss_from_fixed_corruption(
    Model(), x0, torch.full((2, 1), 0.5), torch.ones((2, 8)))
  assert losses.shape == x0.shape
