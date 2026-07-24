from __future__ import annotations

import numpy as np
import torch

from midibrave.zrave_config import ZraveModelConfig
from midibrave.zrave_evaluate import (
    _ErrorAccumulator,
    acceptance_gate,
    linear_baseline,
    persistence_baseline,
    rollout_prediction,
    summarize_distribution,
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


def test_distribution_summary_exposes_tail() -> None:
    report = summarize_distribution(np.arange(10, dtype=np.float64))

    assert report == {
        "median": 4.5,
        "p10": 0.9,
        "p90": 8.1,
        "worst_decile_mean": 9.0,
        "maximum": 9.0,
    }


def test_error_accumulator_reports_per_window_distribution() -> None:
    accumulator = _ErrorAccumulator()
    prediction = torch.zeros(3, 2, 2)
    target = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )

    accumulator.update(prediction, target, torch.ones(2))
    report = accumulator.result()

    assert set(report["window_normalized_smooth_l1"]) == {
        "median",
        "p10",
        "p90",
        "worst_decile_mean",
        "maximum",
    }
    assert report["window_normalized_smooth_l1"]["maximum"] == 1.5
