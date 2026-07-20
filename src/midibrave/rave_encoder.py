from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import CausalConv1d, ChannelRMSNorm, PQMF


@dataclass(frozen=True)
class RavePosterior:
    mean: Tensor
    logvar: Tensor
    latent: Tensor
    kl: Tensor


class _EncoderResidual(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = ChannelRMSNorm()
        self.conv = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.projection = CausalConv1d(channels, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.projection(F.silu(self.conv(self.norm(value))))
        return (value + residual) * (2.0**-0.5)


class _EncoderStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, ratio: int):
        super().__init__()
        self.norm = ChannelRMSNorm()
        self.downsample = CausalConv1d(
            input_channels, output_channels, 2 * ratio + 1, stride=ratio)
        self.residuals = nn.Sequential(
            _EncoderResidual(output_channels, 1),
            _EncoderResidual(output_channels, 3),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = F.silu(self.downsample(self.norm(value)))
        return self.residuals(value)


class RaveEncoder(nn.Module):
    """Causal variational audio encoder with a fixed 128-sample latent hop."""

    def __init__(self, pqmf_bands: int, latent_dim: int, ratios: list[int],
                 capacity: int = 64, pqmf_taps: int = 256):
        super().__init__()
        if pqmf_bands <= 0 or latent_dim <= 0 or capacity <= 0:
            raise ValueError("encoder dimensions must be positive")
        hop = pqmf_bands
        for ratio in ratios:
            if ratio <= 0:
                raise ValueError("encoder ratios must be positive")
            hop *= ratio
        if hop != 128:
            raise ValueError(f"RAVE encoder hop must be 128 samples, got {hop}")
        self.samples_per_latent = hop
        self.pqmf = PQMF(pqmf_bands, pqmf_taps)
        self.input = CausalConv1d(pqmf_bands, capacity, 7)
        stages: list[nn.Module] = []
        input_channels = capacity
        for index, ratio in enumerate(ratios):
            output_channels = capacity * min(2 ** (index + 1), 8)
            stages.append(_EncoderStage(input_channels, output_channels, ratio))
            input_channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.output_norm = ChannelRMSNorm()
        self.mean = nn.Conv1d(input_channels, latent_dim, 1)
        self.logvar = nn.Conv1d(input_channels, latent_dim, 1)

    def _causal_pqmf_analysis(self, audio: Tensor) -> Tensor:
        padded = F.pad(audio, (self.pqmf.taps, 0))
        return F.conv1d(padded, self.pqmf.analysis_weight, stride=self.pqmf.bands)

    def forward(self, audio: Tensor, sample: bool = True) -> RavePosterior:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("RAVE encoder expects mono audio shaped [batch, 1, samples]")
        if audio.shape[-1] % self.samples_per_latent:
            raise ValueError("audio length must be divisible by the 128-sample latent hop")
        value = F.silu(self.input(self._causal_pqmf_analysis(audio)))
        value = self.output_norm(self.stages(value))
        mean = self.mean(value)
        logvar = self.logvar(value).clamp(-12.0, 4.0)
        kl = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).mean()
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return RavePosterior(mean, logvar, latent, kl)
