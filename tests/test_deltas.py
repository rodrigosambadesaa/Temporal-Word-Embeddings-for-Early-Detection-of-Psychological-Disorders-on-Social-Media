import pytest
import torch

from src.models.deltas import (
    f_generate_deltas,
    jensen_shannon_divergence,
    wasserstein_distance,
)


def test_minkowski_matches_torch_p3_norm():
    left = torch.tensor([[[1.0, 2.0, 3.0]]])
    right = torch.tensor([[[4.0, 6.0, 3.0]]])
    expected = torch.norm(left - right, p=3, dim=-1)
    actual = f_generate_deltas(left, right, "minkowski")
    assert torch.allclose(actual, expected)


def test_jensen_shannon_accepts_signed_embeddings_and_is_symmetric():
    left = torch.tensor([[[-1.0, 0.0, 2.0]]])
    right = torch.tensor([[[2.0, -2.0, 0.5]]])
    forward = jensen_shannon_divergence(left, right)
    backward = jensen_shannon_divergence(right, left)
    assert torch.isfinite(forward).all()
    assert (forward >= 0).all()
    assert torch.allclose(forward, backward, atol=1e-6)


def test_wasserstein_uses_normalized_empirical_cdfs():
    left = torch.tensor([[[0.0, 1.0]]])
    right = torch.tensor([[[1.0, 2.0]]])
    actual = wasserstein_distance(left, right)
    assert actual.item() == pytest.approx(1.0)


def test_unknown_distance_is_rejected():
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        f_generate_deltas(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), "unknown")
