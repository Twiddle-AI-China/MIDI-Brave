from __future__ import annotations

import pytest
import torch

from midibrave.zrave_flow_model import FlowStatistics


def _statistics(latent_dim: int = 4) -> FlowStatistics:
    return FlowStatistics(
        mean=torch.zeros(latent_dim),
        latent_std=torch.ones(latent_dim),
        delta_std=torch.ones(latent_dim),
        latent_norm_p01=torch.tensor(0.0),
        latent_norm_p99=torch.tensor(1000.0),
    )


def test_exploration_control_endpoints() -> None:
    from midibrave.zrave_flow_exploration import exploration_controls

    levels = torch.tensor([0.0, 0.5, 1.0])
    controls = exploration_controls(levels)

    assert controls.visible_history_frames.tolist() == [32, 20, 8]
    torch.testing.assert_close(
        controls.temperature,
        torch.tensor([0.7, 1.0, 1.3]),
    )
    torch.testing.assert_close(
        controls.wander_delay_frames,
        torch.tensor([48.0, 32.0, 16.0]),
    )


def test_trailing_history_mask_is_right_aligned() -> None:
    from midibrave.zrave_flow_exploration import trailing_history_mask

    mask = trailing_history_mask(torch.tensor([8, 16, 32]), 32)

    assert mask.shape == (3, 32)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [8, 16, 32]
    assert mask[0, -8:].all()
    assert not mask[0, :-8].any()


@pytest.mark.parametrize(
    "levels",
    [
        torch.tensor([-0.01]),
        torch.tensor([1.01]),
        torch.tensor([float("nan")]),
        torch.zeros(1, 1),
    ],
)
def test_exploration_controls_reject_invalid_levels(
    levels: torch.Tensor,
) -> None:
    from midibrave.zrave_flow_exploration import exploration_controls

    with pytest.raises(ValueError):
        exploration_controls(levels)


@pytest.mark.parametrize(
    "visible",
    [
        torch.tensor([7]),
        torch.tensor([33]),
        torch.zeros(1, 1, dtype=torch.long),
    ],
)
def test_trailing_history_mask_rejects_invalid_counts(
    visible: torch.Tensor,
) -> None:
    from midibrave.zrave_flow_exploration import trailing_history_mask

    with pytest.raises(ValueError):
        trailing_history_mask(visible, 32)


def test_candidate_score_rejects_static_and_exploding_prefixes() -> None:
    from midibrave.zrave_flow_exploration import (
        score_rollout_candidates,
    )

    history = torch.zeros(1, 32, 4)
    static = torch.zeros(1, 64, 4)
    valid = (
        torch.arange(1, 65, dtype=torch.float32)
        .view(1, 64, 1)
        .expand(-1, -1, 4)
    )
    exploding = valid * 10.0
    candidates = torch.stack((static, valid, exploding), dim=1)

    scores = score_rollout_candidates(
        candidates,
        history=history,
        statistics=_statistics(),
        exploration=torch.tensor([1.0]),
        commit_frames=16,
    )

    assert scores.shape == (1, 3)
    assert scores.argmax(dim=1).item() == 1


def test_candidate_score_breaks_exact_ties_by_lowest_index() -> None:
    from midibrave.zrave_flow_exploration import (
        score_rollout_candidates,
    )

    history = torch.zeros(1, 32, 4)
    candidate = (
        torch.arange(1, 65, dtype=torch.float32)
        .view(1, 64, 1)
        .expand(-1, -1, 4)
    )
    candidates = torch.stack((candidate, candidate), dim=1)

    scores = score_rollout_candidates(
        candidates,
        history=history,
        statistics=_statistics(),
        exploration=torch.tensor([0.5]),
        commit_frames=16,
    )

    assert scores.argmax(dim=1).item() == 0
