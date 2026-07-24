from __future__ import annotations

import torch

from midibrave.zrave_config import ZraveModelConfig
from midibrave.zrave_evaluate import (
    acceptance_gate,
    linear_baseline,
    persistence_baseline,
    rollout_prediction,
)
from midibrave.zrave_model import ZraveStatistics, ZraveTransformer


def _model() -> ZraveTransformer:
    config = ZraveModelConfig(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
    )
    statistics = ZraveStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        acceleration_std=torch.ones(16),
    )
    return ZraveTransformer(config, statistics)


def test_rollout_and_baselines_have_exact_contract() -> None:
    history = torch.randn(3, 128, 16)

    persistence = persistence_baseline(history, 32)
    linear = linear_baseline(history, 32)
    rollout = rollout_prediction(_model(), history, 128)

    assert persistence.shape == (3, 32, 16)
    assert rollout.shape == (3, 128, 16)
    expected = history[:, -1:] + torch.arange(
        1,
        33,
        device=history.device,
    )[None, :, None] * (
        history[:, -1] - history[:, -2]
    )[:, None]
    torch.testing.assert_close(linear, expected)


def test_acceptance_requires_both_baselines_and_stable_variance() -> None:
    report = {
        "finite": True,
        "evaluation_contract": {
            "gate_horizons": [16, 64],
            "variance_horizon": 64,
        },
        "rollout": {
            "16": {
                "model": {"normalized_smooth_l1": 0.4},
                "persistence": {"normalized_smooth_l1": 0.5},
                "linear": {"normalized_smooth_l1": 0.6},
            },
            "64": {
                "model": {"normalized_smooth_l1": 0.7},
                "persistence": {"normalized_smooth_l1": 0.8},
                "linear": {"normalized_smooth_l1": 0.9},
            },
        },
        "prediction_variance_ratio": 1.1,
    }

    passed = acceptance_gate(report)
    report["prediction_variance_ratio"] = 0.2
    failed = acceptance_gate(report)

    assert passed["passed"] is True
    assert all(passed["checks"].values())
    assert failed["passed"] is False
    assert failed["checks"]["variance_ratio_64"] is False
