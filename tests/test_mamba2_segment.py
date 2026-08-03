import torch

from models.mamba2_segment import SegmentMamba2


def _mixer():
  torch.manual_seed(7)
  return SegmentMamba2(
    d_model=8,
    d_state=3,
    d_conv=4,
    expand=2,
    headdim=4,
    chunk_size=4,
    backend="torch")


def test_segment_continuation_matches_one_shot_scan():
  mixer = _mixer()
  x = torch.randn(2, 11, 8)

  expected_y, expected_state = mixer.scan_segment(x)
  first_y, boundary = mixer.scan_segment(x[:, :4])
  second_y, actual_state = mixer.scan_segment(x[:, 4:], boundary)

  torch.testing.assert_close(
    torch.cat((first_y, second_y), dim=1), expected_y,
    atol=3e-6, rtol=3e-6)
  torch.testing.assert_close(actual_state.conv, expected_state.conv)
  torch.testing.assert_close(
    actual_state.ssm, expected_state.ssm, atol=3e-6, rtol=3e-6)


def test_continuation_does_not_mutate_boundary_state():
  mixer = _mixer()
  x = torch.randn(2, 8, 8)
  _, boundary = mixer.scan_segment(x[:, :4])
  snapshot = boundary.clone()

  mixer.scan_segment(x[:, 4:], boundary)

  torch.testing.assert_close(boundary.conv, snapshot.conv)
  torch.testing.assert_close(boundary.ssm, snapshot.ssm)


def test_active_loss_backpropagates_through_prefix_state():
  mixer = _mixer()
  prefix = torch.randn(2, 5, 8, requires_grad=True)
  active = torch.randn(2, 3, 8, requires_grad=True)

  _, boundary = mixer.scan_segment(prefix)
  output, _ = mixer.scan_segment(active, boundary)
  output.square().mean().backward()

  assert prefix.grad is not None
  assert prefix.grad.abs().sum() > 0
  assert active.grad is not None
  assert active.grad.abs().sum() > 0

