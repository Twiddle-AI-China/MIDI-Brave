from __future__ import annotations

import torch

from midibrave.zrave_flow_evaluate import (
    GateEarlyStopState,
    flow_acceptance_gate,
    rollout_flow,
)
from midibrave.zrave_flow_model import FlowStatistics


def _statistics() -> FlowStatistics:
    return FlowStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(10.0),
    )


def _passing_report() -> dict[str, object]:
    return {
        "f0_median_cents": 50.0,
        "f0_p90_cents": 100.0,
        "voiced_following": 0.90,
        "swapped_f0_median_cents": 50.0,
        "swapped_f0_p90_cents": 100.0,
        "swapped_voiced_following": 0.90,
        "exact_reproduction": True,
        "nonfinite_renders": 0,
        "boundary_jump_over_real_p95": 1.0,
        "silence_fraction": 0.01,
        "maximum_short_cycle_autocorrelation": 0.95,
        "tail_delta_ratio": 0.25,
        "tail_seed_diversity_ratio": 0.50,
        "first_second_seed_diversity_ratio": 0.999,
        "nearest_training_nrmse": 0.0011,
        "latent_norm_violation_fraction": 0.02,
    }


def test_gate_fails_each_independent_hard_requirement() -> None:
    report = _passing_report()
    assert flow_acceptance_gate(report)["passed"]
    for field, bad in {
        "f0_median_cents": 50.1,
        "f0_p90_cents": 100.1,
        "voiced_following": 0.899,
        "swapped_f0_median_cents": 50.1,
        "swapped_f0_p90_cents": 100.1,
        "swapped_voiced_following": 0.899,
        "exact_reproduction": False,
        "nonfinite_renders": 1,
        "boundary_jump_over_real_p95": 1.001,
        "silence_fraction": 0.011,
        "maximum_short_cycle_autocorrelation": 0.951,
        "tail_delta_ratio": 0.249,
        "tail_seed_diversity_ratio": 0.499,
        "first_second_seed_diversity_ratio": 1.001,
        "nearest_training_nrmse": 0.001,
    }.items():
        changed = dict(report)
        changed[field] = bad
        assert not flow_acceptance_gate(changed)["passed"], field


class _CountingBlockSampler:
    def __init__(self) -> None:
        self.block_indices: list[int] = []

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        block_index = int(kwargs["block_index"])
        self.block_indices.append(block_index)
        history = args[2]
        assert isinstance(history, torch.Tensor)
        return torch.full(
            (history.shape[0], 64, 16),
            float(block_index),
        )


def test_rollout_uses_deterministic_block_indices() -> None:
    sampler = _CountingBlockSampler()
    output = rollout_flow(
        object(),
        _statistics(),
        torch.zeros(1, 32, 16),
        torch.tensor([60]),
        frames=320,
        generation_seed=17,
        temperature=1.0,
        wander_delay_frames=32,
        sample_block=sampler,
    )

    assert output.shape == (1, 320, 16)
    assert sampler.block_indices == [0, 1, 2, 3, 4]


def test_checkpoint_evaluation_controls_only_gate_based_early_stop() -> None:
    state = GateEarlyStopState(required_consecutive_passes=3)

    assert not state.update({"passed": False})
    assert not state.update({"passed": True})
    assert not state.update({"passed": True})
    assert state.update({"passed": True})
    assert state.consecutive_passes == 3
