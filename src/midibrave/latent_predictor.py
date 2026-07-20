from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import CausalConv1d, ChannelRMSNorm


@dataclass(frozen=True)
class LatentPrediction:
    delta: Tensor
    latent: Tensor


@dataclass(frozen=True)
class RolloutResult:
    latent: Tensor
    history: Tensor


class _PredictorResidual(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = ChannelRMSNorm()
        self.conv = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.projection = CausalConv1d(channels, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.projection(F.silu(self.conv(self.norm(value))))
        return (value + residual) * (2.0**-0.5)


class MultiHorizonPredictor(nn.Module):
    """Predict all future deltas in one causal, non-autoregressive forward."""

    def __init__(self, rave_dim: int, clap_dim: int, midi_dim: int, hidden_dim: int,
                 history_frames: int, horizon_frames: int):
        super().__init__()
        dimensions = (rave_dim, clap_dim, midi_dim, hidden_dim,
                      history_frames, horizon_frames)
        if any(value <= 0 for value in dimensions):
            raise ValueError("predictor dimensions must be positive")
        self.rave_dim = rave_dim
        self.clap_dim = clap_dim
        self.midi_dim = midi_dim
        self.history_frames = history_frames
        self.horizon_frames = horizon_frames
        self.history_input = CausalConv1d(rave_dim, hidden_dim, 3)
        self.history_tcn = nn.Sequential(*[
            _PredictorResidual(hidden_dim, dilation) for dilation in (1, 2, 4, 8)
        ])
        self.control_input = CausalConv1d(clap_dim + midi_dim, hidden_dim, 3)
        self.horizon_embedding = nn.Parameter(
            torch.zeros(1, hidden_dim, horizon_frames))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        self.future_tcn = nn.Sequential(*[
            _PredictorResidual(hidden_dim, dilation) for dilation in (1, 2, 4)
        ])
        self.output = nn.Conv1d(hidden_dim, rave_dim, 1)

    def _validate(self, history: Tensor, clap_future: Tensor, midi_future: Tensor) -> None:
        if history.ndim != 3 or history.shape[1] != self.rave_dim:
            raise ValueError("history must have shape [batch, rave_dim, history_frames]")
        if history.shape[-1] != self.history_frames:
            raise ValueError(f"history must contain exactly {self.history_frames} frames")
        expected_clap = (history.shape[0], self.clap_dim, self.horizon_frames)
        expected_midi = (history.shape[0], self.midi_dim, self.horizon_frames)
        if tuple(clap_future.shape) != expected_clap:
            raise ValueError(f"CLAP future must have shape {expected_clap}")
        if tuple(midi_future.shape) != expected_midi:
            raise ValueError(f"MIDI future must have shape {expected_midi}")

    def forward(self, history: Tensor, clap_future: Tensor,
                midi_future: Tensor) -> LatentPrediction:
        self._validate(history, clap_future, midi_future)
        history_features = self.history_tcn(F.silu(self.history_input(history)))
        context = history_features[..., -1:].expand(-1, -1, self.horizon_frames)
        controls = F.silu(self.control_input(torch.cat((clap_future, midi_future), dim=1)))
        features = self.future_tcn(context + controls + self.horizon_embedding)
        delta = self.output(F.silu(features))
        latent = history[..., -1:] + delta.cumsum(dim=-1)
        return LatentPrediction(delta, latent)


def _future_window(value: Tensor, start: int, frames: int) -> Tensor:
    window = value[..., start:start + frames]
    if window.shape[-1] == frames:
        return window
    if not window.shape[-1]:
        raise ValueError("rollout control trajectory is empty")
    padding = window[..., -1:].expand(-1, -1, frames - window.shape[-1])
    return torch.cat((window, padding), dim=-1)


def rollout_blocks(predictor: MultiHorizonPredictor, history: Tensor,
                   clap_future: Tensor, midi_future: Tensor,
                   stride_frames: int) -> RolloutResult:
    if clap_future.ndim != 3 or midi_future.ndim != 3:
        raise ValueError("rollout controls must be rank-three tensors")
    if clap_future.shape[0] != midi_future.shape[0] or clap_future.shape[-1] != midi_future.shape[-1]:
        raise ValueError("rollout CLAP and MIDI trajectories must align")
    if not 0 < stride_frames <= predictor.horizon_frames:
        raise ValueError("stride_frames must satisfy 0 < stride <= horizon")
    total_frames = clap_future.shape[-1]
    if total_frames <= 0 or total_frames % stride_frames:
        raise ValueError("rollout length must be a positive multiple of stride_frames")
    current = history
    chunks = []
    for start in range(0, total_frames, stride_frames):
        prediction = predictor(
            current,
            _future_window(clap_future, start, predictor.horizon_frames),
            _future_window(midi_future, start, predictor.horizon_frames),
        )
        consumed = prediction.latent[..., :stride_frames]
        chunks.append(consumed)
        current = torch.cat((current, consumed), dim=-1)[..., -predictor.history_frames:]
    return RolloutResult(torch.cat(chunks, dim=-1), current)
