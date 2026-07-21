from __future__ import annotations

import torch

from midibrave.evaluate import (predictive_control_quality_gate,
                                predictive_rave_quality_gate,
                                predictive_rollout_report)
from midibrave.predictive_losses import LatentStatistics


def _inputs():
    frames = 128 * 4
    reference = torch.randn(1, 3, frames)
    predicted = reference + 0.05 * torch.randn_like(reference)
    reference_audio = torch.randn(1, 1, frames * 8)
    predicted_audio = reference_audio + 0.01 * torch.randn_like(reference_audio)
    statistics = LatentStatistics(torch.ones(3), torch.ones(3), torch.ones(3))
    return predicted, reference, predicted_audio, reference_audio, statistics


def test_predictive_report_emits_all_horizons_and_stride_contract():
    report = predictive_rollout_report(*_inputs(), stride_frames=4,
                                       samples_per_latent=8, elapsed_seconds=0.1)
    assert set(report["horizons"]) == {"1", "8", "32", "128"}
    for metrics in report["horizons"].values():
        assert {"normalized_latent_error", "normalized_delta_error",
                "variance_ratio", "stft", "f0_cents", "rms_error_db",
                "clap_cosine", "non_finite_count", "realtime_factor"} <= set(metrics)
    assert report["control_stride_frames"] == 4
    assert report["control_stride_samples"] == 32
    assert report["gate"]["passed"] is True


def test_collapsed_variance_and_nonfinite_fail_gate():
    predicted, reference, predicted_audio, reference_audio, statistics = _inputs()
    collapsed = predictive_rollout_report(
        torch.zeros_like(predicted), reference, predicted_audio, reference_audio,
        statistics, 4, 8, 0.1)
    assert collapsed["gate"]["passed"] is False
    assert "variance_ratio" in collapsed["gate"]["failures"]

    predicted[..., 0] = float("nan")
    invalid = predictive_rollout_report(
        predicted, reference, predicted_audio, reference_audio,
        statistics, 4, 8, 0.1)
    assert invalid["gate"]["passed"] is False
    assert "non_finite" in invalid["gate"]["failures"]


def test_predictive_rave_gate_requires_pitch_and_counterfactual_timbre_following():
    metrics = {
        "reconstruction_f0_absolute_cents": {"median": 30.0, "p90": 80.0},
        "swap_f0_absolute_cents": {"median": 60.0, "p90": 150.0},
        "midi_swap_following": {"mean": 0.95},
        "reconstruction_clap_cosine": {"median": 0.82},
        "clap_control_following": {"mean": 0.92},
    }
    passed = predictive_rave_quality_gate(metrics, nonfinite_count=0)
    assert passed["passed"] is True

    metrics["midi_swap_following"]["mean"] = 0.5
    failed = predictive_rave_quality_gate(metrics, nonfinite_count=1)
    assert failed["passed"] is False
    assert failed["failures"] == ["midi_swap_following", "non_finite"]

    metrics["midi_swap_following"]["mean"] = 0.95
    metrics["reconstruction_clap_cosine"]["median"] = 0.79
    metrics["clap_control_following"]["mean"] = 0.89
    failed = predictive_rave_quality_gate(metrics, nonfinite_count=0)
    assert failed["failures"] == [
        "clap_control_following", "reconstruction_clap_cosine"]


def test_predictor_control_gate_requires_causal_midi_clap_and_healthy_rollout():
    metrics = {
        "swap_f0_absolute_cents": {"p90": 180.0},
        "midi_swap_following": {"mean": 0.92},
        "midi_timbre_preservation_cosine": {"median": 0.84},
        "clap_control_following": {"mean": 0.91},
        "timbre_f0_absolute_cents": {"p90": 120.0},
        "timbre_path_monotonicity": {"mean": 0.80},
        "seed_washout_ratio": {"mean": 0.70},
        "rollout_variance_ratio": {"mean": 1.1},
    }

    passed = predictive_control_quality_gate(metrics, nonfinite_count=0)
    assert passed["passed"] is True
    assert passed["thresholds"]["midi_swap_following"] == 0.90
    assert passed["thresholds"]["swap_f0_p90_cents"] == 200.0
    assert passed["thresholds"]["clap_control_following"] == 0.90
    assert passed["thresholds"]["midi_timbre_preservation_cosine"] == 0.80
    assert passed["thresholds"]["timbre_path_monotonicity"] == 0.75
    assert passed["thresholds"]["seed_washout_ratio_max"] == 0.90

    metrics["swap_f0_absolute_cents"]["p90"] = 240.0
    metrics["midi_swap_following"]["mean"] = 0.80
    metrics["midi_timbre_preservation_cosine"]["median"] = 0.60
    metrics["clap_control_following"]["mean"] = 0.70
    metrics["timbre_f0_absolute_cents"]["p90"] = 250.0
    metrics["timbre_path_monotonicity"]["mean"] = 0.20
    metrics["seed_washout_ratio"]["mean"] = 1.10
    metrics["rollout_variance_ratio"]["mean"] = 0.01
    failed = predictive_control_quality_gate(metrics, nonfinite_count=3)
    assert failed["passed"] is False
    assert failed["failures"] == [
        "clap_control_following", "midi_swap_following",
        "midi_timbre_preservation_cosine", "non_finite",
        "rollout_variance_ratio", "seed_washout_ratio", "swap_f0_p90",
        "timbre_f0_p90", "timbre_path_monotonicity",
    ]
