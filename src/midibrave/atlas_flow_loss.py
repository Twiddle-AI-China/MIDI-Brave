from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .atlas_flow_config import AtlasLossConfig
from .atlas_flow_model import AtlasConditionedTrajectoryFlow, FlowBatch


@dataclass(frozen=True)
class AtlasFlowLossOutput:
    total: Tensor
    components: dict[str, Tensor]
    reconstructed: Tensor


def _motion(value: Tensor, order: int) -> Tensor:
    result = value
    for _ in range(order):
        result = result[:, 1:] - result[:, :-1]
    return result


def atlas_flow_matching_loss(
    model: AtlasConditionedTrajectoryFlow,
    batch: FlowBatch,
    config: AtlasLossConfig,
    *,
    generator: torch.Generator | None = None,
) -> AtlasFlowLossOutput:
    target_residual = batch.target - batch.anchor_path
    noise = torch.randn(
        target_residual.shape,
        device=target_residual.device,
        dtype=target_residual.dtype,
        generator=generator,
    )
    flow_time = torch.rand(
        target_residual.shape[0],
        device=target_residual.device,
        dtype=target_residual.dtype,
        generator=generator,
    )
    mixed = (1.0 - flow_time[:, None, None]) * noise + flow_time[:, None, None] * target_residual
    target_velocity = target_residual - noise
    predicted_velocity = model(
        mixed,
        flow_time,
        batch.history,
        batch.history_anchor,
        batch.anchor_path,
        batch.atlas_path,
        batch.lifecycle,
        batch.history_mask,
    )
    reconstructed_residual = mixed + (1.0 - flow_time[:, None, None]) * predicted_velocity
    reconstructed = batch.anchor_path + reconstructed_residual
    flow = F.mse_loss(predicted_velocity.float(), target_velocity.float())
    boundary = F.smooth_l1_loss(reconstructed[:, 0].float(), batch.history[:, -1].float())
    target_mean = batch.target.float().mean(1)
    prediction_mean = reconstructed.float().mean(1)
    target_std = batch.target.float().std(1, unbiased=False)
    prediction_std = reconstructed.float().std(1, unbiased=False)
    statistics = F.smooth_l1_loss(prediction_mean, target_mean) + F.smooth_l1_loss(
        prediction_std, target_std
    )
    temporal_motion = F.smooth_l1_loss(
        _motion(reconstructed.float(), 1), _motion(batch.target.float(), 1)
    ) + 0.5 * F.smooth_l1_loss(
        _motion(reconstructed.float(), 2), _motion(batch.target.float(), 2)
    )
    components = {
        "flow": flow,
        "boundary": boundary,
        "statistics": statistics,
        "temporal_motion": temporal_motion,
    }
    total = (
        config.flow * flow
        + config.boundary * boundary
        + config.statistics * statistics
        + config.temporal_motion * temporal_motion
    )
    return AtlasFlowLossOutput(total, components, reconstructed)


__all__ = ["AtlasFlowLossOutput", "atlas_flow_matching_loss"]
