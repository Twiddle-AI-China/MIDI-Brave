from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional

if TYPE_CHECKING:
    from .zrave_flow_model import FlowStatistics


@dataclass(frozen=True)
class ExplorationControls:
    visible_history_frames: Tensor
    temperature: Tensor
    wander_delay_frames: Tensor


@dataclass(frozen=True)
class RolloutCandidateMetrics:
    scores: Tensor
    motion: Tensor
    boundary_rms: Tensor
    norm_violation: Tensor


def exploration_controls(levels: Tensor) -> ExplorationControls:
    values = torch.as_tensor(levels, dtype=torch.float32)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError(
            "exploration levels must be one finite vector"
        )
    if torch.any(values < 0.0) or torch.any(values > 1.0):
        raise ValueError("exploration levels must be in [0, 1]")
    return ExplorationControls(
        visible_history_frames=torch.round(
            32.0 - 24.0 * values
        ).long(),
        temperature=0.7 + 0.6 * values,
        wander_delay_frames=48.0 - 32.0 * values,
    )


def trailing_history_mask(
    visible_frames: Tensor,
    context_frames: int = 32,
) -> Tensor:
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    visible = torch.as_tensor(visible_frames, dtype=torch.long)
    if (
        visible.ndim != 1
        or torch.any(visible < 8)
        or torch.any(visible > context_frames)
    ):
        raise ValueError(
            "visible history must be in [8, context_frames]"
        )
    positions = torch.arange(
        context_frames,
        device=visible.device,
    )
    return positions.unsqueeze(0) >= (
        context_frames - visible
    ).unsqueeze(1)


def rollout_candidate_metrics(
    candidates: Tensor,
    *,
    history: Tensor,
    statistics: FlowStatistics,
    exploration: Tensor,
    commit_frames: int,
) -> RolloutCandidateMetrics:
    if candidates.ndim != 4:
        raise ValueError(
            "rollout candidates must be [batch, candidates, frames, latent]"
        )
    batch, candidate_count, frames, latent_dim = candidates.shape
    if candidate_count <= 0 or frames <= 1:
        raise ValueError("rollout candidates must be non-empty")
    if history.ndim != 3 or history.shape != (
        batch,
        history.shape[1],
        latent_dim,
    ):
        raise ValueError("rollout history shape does not match candidates")
    if not 2 <= commit_frames <= frames:
        raise ValueError("commit_frames must fit inside candidate frames")
    if not torch.isfinite(candidates).all() or not torch.isfinite(history).all():
        raise ValueError("rollout candidates and history must be finite")
    statistics.validate(latent_dim)
    levels = torch.as_tensor(
        exploration,
        device=candidates.device,
        dtype=candidates.dtype,
    )
    if levels.ndim == 0:
        levels = levels.expand(batch)
    if (
        levels.shape != (batch,)
        or not torch.isfinite(levels).all()
        or torch.any(levels < 0.0)
        or torch.any(levels > 1.0)
    ):
        raise ValueError("exploration must be in [0, 1] for each batch row")

    prefix = candidates[:, :, :commit_frames]
    delta_std = statistics.delta_std.to(
        device=candidates.device,
        dtype=candidates.dtype,
    ).view(1, 1, 1, latent_dim)
    normalized_delta = (prefix[:, :, 1:] - prefix[:, :, :-1]) / delta_std
    motion = normalized_delta.square().mean(dim=(2, 3)).add(1.0e-8).sqrt()
    boundary = (prefix[:, :, 0] - history[:, None, -1]) / delta_std.squeeze(2)
    boundary_rms = boundary.square().mean(dim=-1).add(1.0e-8).sqrt()

    norms = torch.linalg.vector_norm(prefix, dim=-1)
    lower = statistics.latent_norm_p01.to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    upper = statistics.latent_norm_p99.to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    norm_violation = ((norms < lower) | (norms > upper)).to(
        candidates.dtype
    ).mean(dim=-1)

    target_motion = (0.35 + 0.90 * levels).unsqueeze(1)
    motion_error = (
        motion.clamp_min(1.0e-6).log() - target_motion.log()
    ).abs()
    boundary_penalty = functional.relu(boundary_rms - 4.0)
    stagnation_penalty = functional.relu(
        0.25 + 0.25 * levels.unsqueeze(1) - motion
    )
    scores = -(
        motion_error
        + 2.0 * boundary_penalty
        + 2.0 * norm_violation
        + stagnation_penalty
    )
    return RolloutCandidateMetrics(
        scores=scores,
        motion=motion,
        boundary_rms=boundary_rms,
        norm_violation=norm_violation,
    )


def score_rollout_candidates(
    candidates: Tensor,
    *,
    history: Tensor,
    statistics: FlowStatistics,
    exploration: Tensor,
    commit_frames: int,
) -> Tensor:
    return rollout_candidate_metrics(
        candidates,
        history=history,
        statistics=statistics,
        exploration=exploration,
        commit_frames=commit_frames,
    ).scores
