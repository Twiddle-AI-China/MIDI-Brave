from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentStatistics:
    latent_std: Tensor
    delta_std: Tensor
    acceleration_std: Tensor
    floor: float = 0.0001

    def scales(self, channels: int, device: torch.device,
               dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        values = (self.latent_std, self.delta_std, self.acceleration_std)
        if any(value.ndim != 1 or value.shape[0] != channels for value in values):
            raise ValueError("latent statistics must be one vector per channel")
        return tuple(value.to(device=device, dtype=dtype).clamp_min(self.floor)
                     .view(1, channels, 1) for value in values)  # type: ignore[return-value]


@dataclass(frozen=True)
class PredictionLoss:
    future: Tensor
    delta: Tensor
    acceleration: Tensor


def horizon_weights(frames: int, discount: float, device: torch.device,
                    dtype: torch.dtype) -> Tensor:
    if frames <= 0 or not 0.0 < discount <= 1.0:
        raise ValueError("invalid horizon weighting parameters")
    weights = torch.pow(torch.tensor(discount, device=device, dtype=dtype),
                        torch.arange(frames, device=device, dtype=dtype))
    return weights / weights.sum()


def _weighted_horizon_loss(error: Tensor, discount: float) -> Tensor:
    weights = horizon_weights(error.shape[-1], discount, error.device, error.dtype)
    per_horizon = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    return (per_horizon.mean(dim=(0, 1)) * weights).sum()


def prediction_loss(prediction: Tensor, target: Tensor, history: Tensor,
                    statistics: LatentStatistics,
                    discount: float = 0.95) -> PredictionLoss:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [batch, channels, horizon]")
    if history.ndim != 3 or history.shape[:2] != prediction.shape[:2] or history.shape[-1] < 2:
        raise ValueError("history must align with prediction and contain at least two frames")
    latent_scale, delta_scale, acceleration_scale = statistics.scales(
        prediction.shape[1], prediction.device, prediction.dtype)
    future = _weighted_horizon_loss((prediction - target) / latent_scale, discount)
    prediction_with_anchor = torch.cat((history[..., -1:], prediction), dim=-1)
    target_with_anchor = torch.cat((history[..., -1:], target), dim=-1)
    delta_error = (prediction_with_anchor.diff(dim=-1)
                   - target_with_anchor.diff(dim=-1)) / delta_scale
    delta = _weighted_horizon_loss(delta_error, discount)
    prediction_with_context = torch.cat((history[..., -2:], prediction), dim=-1)
    target_with_context = torch.cat((history[..., -2:], target), dim=-1)
    acceleration_error = (prediction_with_context.diff(dim=-1).diff(dim=-1)
                          - target_with_context.diff(dim=-1).diff(dim=-1))
    acceleration = _weighted_horizon_loss(
        acceleration_error / acceleration_scale, discount)
    return PredictionLoss(future, delta, acceleration)


def overlap_loss(previous: Tensor, current: Tensor, stride_frames: int,
                 statistics: LatentStatistics) -> Tensor:
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("overlap predictions must share [batch, channels, horizon]")
    if not 0 < stride_frames < previous.shape[-1]:
        raise ValueError("overlap stride must be inside the prediction horizon")
    latent_scale, _, _ = statistics.scales(
        previous.shape[1], previous.device, previous.dtype)
    error = (previous[..., stride_frames:] - current[..., :-stride_frames]) / latent_scale
    return F.smooth_l1_loss(error, torch.zeros_like(error))
