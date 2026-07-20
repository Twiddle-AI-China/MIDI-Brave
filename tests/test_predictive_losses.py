from __future__ import annotations

import torch

from midibrave.predictive_losses import (LatentStatistics, horizon_weights,
                                         overlap_loss, prediction_loss)


def test_horizon_weights_are_normalized():
    weights = horizon_weights(8, 0.95, torch.device("cpu"), torch.float32)
    assert weights.shape == (8,)
    assert torch.all(weights[:-1] > weights[1:])
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_prediction_loss_is_zero_for_exact_future():
    history = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 8)
    statistics = LatentStatistics(torch.ones(4), torch.ones(4), torch.ones(4))
    result = prediction_loss(target, target, history, statistics)
    assert result.future.item() == 0.0
    assert result.delta.item() == 0.0
    assert result.acceleration.item() == 0.0


def test_channel_normalization_makes_scaled_problem_equivalent():
    history = torch.zeros(1, 2, 16)
    target = torch.zeros(1, 2, 8)
    prediction = torch.ones(1, 2, 8)
    unit = LatentStatistics(torch.ones(2), torch.ones(2), torch.ones(2))
    scaled = LatentStatistics(torch.full((2,), 2.0), torch.full((2,), 2.0),
                              torch.full((2,), 2.0))
    first = prediction_loss(prediction, target, history, unit)
    second = prediction_loss(prediction * 2, target * 2, history * 2, scaled)
    assert torch.allclose(first.future, second.future)
    assert torch.allclose(first.delta, second.delta)
    assert torch.allclose(first.acceleration, second.acceleration)


def test_overlap_compares_only_shared_horizons():
    previous = torch.randn(2, 3, 8)
    current = torch.randn(2, 3, 8)
    current[..., :4] = previous[..., 4:]
    statistics = LatentStatistics(torch.ones(3), torch.ones(3), torch.ones(3))
    assert overlap_loss(previous, current, 4, statistics).item() == 0.0
