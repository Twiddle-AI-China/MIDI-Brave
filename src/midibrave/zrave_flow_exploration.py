from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ExplorationControls:
    visible_history_frames: Tensor
    temperature: Tensor
    wander_delay_frames: Tensor


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
