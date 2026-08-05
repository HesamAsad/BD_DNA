import pytest

from scripts.eval.dnahnet.profile_forward import DEFAULT_LENGTHS, parse_lengths


def test_default_lengths_match_dnahnet_appendix_range():
  assert DEFAULT_LENGTHS == tuple(2 ** exponent for exponent in range(10, 20))


def test_parse_lengths_accepts_commas_and_spaces():
  assert parse_lengths("1024, 2048 4096") == (1024, 2048, 4096)


@pytest.mark.parametrize("value", ["", "0", "1024,1024", "-1"])
def test_parse_lengths_rejects_invalid_values(value):
  with pytest.raises(ValueError):
    parse_lengths(value)
