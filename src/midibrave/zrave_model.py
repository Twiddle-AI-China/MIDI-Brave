from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .zrave_config import ZraveLossConfig, ZraveModelConfig


@dataclass(frozen=True)
class ZraveStatistics:
    mean: Tensor
    latent_std: Tensor
    delta_std: Tensor
    acceleration_std: Tensor

    def validate(self, latent_dim: int) -> None:
        for name in (
            "mean",
            "latent_std",
            "delta_std",
            "acceleration_std",
        ):
            value = getattr(self, name)
            if value.shape != (latent_dim,):
                raise ValueError(
                    f"statistics.{name} must have shape ({latent_dim},), "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"statistics.{name} contains non-finite values")
        for name in ("latent_std", "delta_std", "acceleration_std"):
            if not torch.all(getattr(self, name) > 0):
                raise ValueError(f"statistics.{name} must be strictly positive")


class ZravePrediction(NamedTuple):
    delta: Tensor
    latent: Tensor


@dataclass(frozen=True)
class ZraveLoss:
    total: Tensor
    components: dict[str, Tensor]


class ZraveTransformer(nn.Module):
    def __init__(
        self,
        config: ZraveModelConfig,
        statistics: ZraveStatistics,
    ) -> None:
        super().__init__()
        statistics.validate(config.latent_dim)
        self.config = config
        self.register_buffer("latent_mean", statistics.mean.float().clone())
        self.register_buffer(
            "latent_std",
            statistics.latent_std.float().clone(),
        )
        self.register_buffer(
            "delta_std",
            statistics.delta_std.float().clone(),
        )
        self.register_buffer(
            "acceleration_std",
            statistics.acceleration_std.float().clone(),
        )
        self.input_projection = nn.Linear(config.latent_dim, config.d_model)
        self.position = nn.Parameter(
            torch.empty(1, config.context_frames, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.output = nn.Linear(
            config.d_model,
            config.horizon_frames * config.latent_dim,
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def statistics(self) -> ZraveStatistics:
        return ZraveStatistics(
            mean=self.latent_mean,
            latent_std=self.latent_std,
            delta_std=self.delta_std,
            acceleration_std=self.acceleration_std,
        )

    def forward(self, history: Tensor) -> ZravePrediction:
        expected = (
            self.config.context_frames,
            self.config.latent_dim,
        )
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "history must have shape "
                f"[batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(history.shape)}"
            )
        normalized = (
            history.float() - self.latent_mean
        ) / self.latent_std
        hidden = self.input_projection(normalized)
        hidden = self.transformer(hidden + self.position)
        normalized_delta = self.output(hidden[:, -1]).reshape(
            history.shape[0],
            self.config.horizon_frames,
            self.config.latent_dim,
        )
        delta = normalized_delta.float() * self.delta_std
        latent = history[:, -1:].float() + delta.cumsum(dim=1)
        return ZravePrediction(delta=delta, latent=latent)


def _horizon_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    scale: Tensor,
    discount: float,
    floor: float,
) -> Tensor:
    horizon = prediction.shape[1]
    weights = torch.pow(
        prediction.new_tensor(discount, dtype=torch.float32),
        torch.arange(horizon, device=prediction.device, dtype=torch.float32),
    )
    weights = weights / weights.sum()
    safe_scale = scale.to(
        device=prediction.device,
        dtype=torch.float32,
    ).clamp_min(floor)
    element_loss = functional.smooth_l1_loss(
        prediction.float() / safe_scale,
        target.float() / safe_scale,
        reduction="none",
    )
    per_horizon = element_loss.mean(dim=(0, 2))
    return torch.sum(per_horizon * weights)


def zrave_prediction_loss(
    prediction: ZravePrediction,
    history: Tensor,
    target: Tensor,
    statistics: ZraveStatistics,
    config: ZraveLossConfig,
) -> ZraveLoss:
    if history.ndim != 3 or history.shape[1] < 2:
        raise ValueError("history must contain at least two latent frames")
    if target.shape != prediction.latent.shape:
        raise ValueError(
            "target and predicted latent shapes must match, got "
            f"{tuple(target.shape)} and {tuple(prediction.latent.shape)}"
        )
    if prediction.delta.shape != prediction.latent.shape:
        raise ValueError("predicted delta and latent shapes must match")

    target_delta = torch.diff(
        torch.cat([history[:, -1:].float(), target.float()], dim=1),
        dim=1,
    )
    observed_delta = history[:, -1].float() - history[:, -2].float()
    target_acceleration = torch.diff(
        torch.cat([observed_delta[:, None], target_delta], dim=1),
        dim=1,
    )
    predicted_acceleration = torch.diff(
        torch.cat(
            [observed_delta[:, None], prediction.delta.float()],
            dim=1,
        ),
        dim=1,
    )

    components = {
        "future": _horizon_smooth_l1(
            prediction.latent,
            target,
            statistics.latent_std,
            config.horizon_discount,
            config.statistic_floor,
        ),
        "delta": _horizon_smooth_l1(
            prediction.delta,
            target_delta,
            statistics.delta_std,
            config.horizon_discount,
            config.statistic_floor,
        ),
        "acceleration": _horizon_smooth_l1(
            predicted_acceleration,
            target_acceleration,
            statistics.acceleration_std,
            config.horizon_discount,
            config.statistic_floor,
        ),
    }
    total = (
        config.future * components["future"]
        + config.delta * components["delta"]
        + config.acceleration * components["acceleration"]
    )
    return ZraveLoss(total=total, components=components)
