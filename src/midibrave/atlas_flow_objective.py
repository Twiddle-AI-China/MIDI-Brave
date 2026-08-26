from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .atlas_flow_config import AtlasLossConfig
from .atlas_flow_model import AtlasTrajectoryInstrument


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        ctx.scale = scale  # type: ignore[attr-defined]
        return value

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(ctx.scale) * gradient, None  # type: ignore[attr-defined]


def _stft_loss(prediction: Tensor, target: Tensor, sizes: tuple[int, ...]) -> Tensor:
    total = prediction.new_zeros((), dtype=torch.float32)
    prediction = prediction.float().flatten(0, 1)
    target = target.float().flatten(0, 1)
    for fft_size in sizes:
        window = torch.hann_window(fft_size, device=prediction.device)
        predicted = torch.stft(
            prediction, fft_size, fft_size // 4, fft_size, window,
            center=True, return_complex=True,
        ).abs().clamp_min(1.0e-6)
        reference = torch.stft(
            target, fft_size, fft_size // 4, fft_size, window,
            center=True, return_complex=True,
        ).abs().clamp_min(1.0e-6)
        convergence = torch.linalg.vector_norm((predicted - reference).flatten(1), dim=1)
        convergence = convergence / torch.linalg.vector_norm(reference.flatten(1), dim=1).clamp_min(1.0e-6)
        total = total + torch.log1p(convergence).mean() + F.l1_loss(predicted.log(), reference.log())
    return total / len(sizes)


def _envelope_loss(prediction: Tensor, target: Tensor) -> Tensor:
    total = prediction.new_zeros((), dtype=torch.float32)
    for width in (128, 512, 2_048, 8_192):
        predicted = F.avg_pool1d(prediction.float().abs(), width, max(1, width // 4), ceil_mode=True)
        reference = F.avg_pool1d(target.float().abs(), width, max(1, width // 4), ceil_mode=True)
        total = total + F.smooth_l1_loss(predicted, reference)
    return total / 4


def _rms_db(value: Tensor) -> Tensor:
    return 20.0 * torch.log10(value.float().square().mean(-1).sqrt().clamp_min(1.0e-7))


@dataclass(frozen=True)
class Stage1LossOutput:
    total: Tensor
    components: dict[str, Tensor]
    trajectory_a: Tensor
    trajectory_b: Tensor
    self_audio: Tensor
    cross_audio: Tensor


class AtlasStage1Objective(nn.Module):
    def __init__(self, config: AtlasLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        instrument: AtlasTrajectoryInstrument,
        batch: dict[str, Tensor],
    ) -> Stage1LossOutput:
        trajectory_a = instrument.encode(batch["features_a"])
        trajectory_b = instrument.encode(batch["features_b"])
        self_audio, _ = instrument.decode(
            trajectory_a, batch["note_a"], batch["audio_a"].shape[-1],
            window_start=batch["window_start"],
        )
        cross_audio, _ = instrument.decode(
            trajectory_a, batch["note_b"], batch["audio_b"].shape[-1],
            window_start=batch["window_start"],
        )
        self_stft = _stft_loss(self_audio, batch["audio_a"], (2_048, 1_024, 512, 256))
        cross_stft = _stft_loss(cross_audio, batch["audio_b"], (2_048, 1_024, 512, 256))
        # Shorter FFTs emphasize subband/transient reconstruction and are
        # reported separately from the broad MR-STFT objective.
        multiband = 0.5 * (
            _stft_loss(self_audio, batch["audio_a"], (512, 256, 128))
            + _stft_loss(cross_audio, batch["audio_b"], (512, 256, 128))
        )
        envelope = 0.5 * (
            _envelope_loss(self_audio, batch["audio_a"])
            + _envelope_loss(cross_audio, batch["audio_b"])
        )
        rms = 0.5 * (
            F.smooth_l1_loss((_rms_db(self_audio) - _rms_db(batch["audio_a"])) / 20.0, torch.zeros_like(_rms_db(self_audio)))
            + F.smooth_l1_loss((_rms_db(cross_audio) - _rms_db(batch["audio_b"])) / 20.0, torch.zeros_like(_rms_db(cross_audio)))
        )
        frames = min(trajectory_a.shape[-1], trajectory_b.shape[-1])
        same_preset = F.smooth_l1_loss(trajectory_a[..., :frames], trajectory_b[..., :frames])
        pooled = trajectory_a.float().mean(-1)
        pitch_logits = instrument.pitch_adversary(_ReverseGradient.apply(pooled, 1.0))
        pitch_adversary = F.cross_entropy(pitch_logits, batch["note_a"].long().clamp(0, 127))
        components = {
            "self_stft": self_stft,
            "cross_stft": cross_stft,
            "multiband": multiband,
            "envelope": envelope,
            "rms": rms,
            "same_preset": same_preset,
            "pitch_adversary": pitch_adversary,
        }
        total = (
            self.config.stft * 0.5 * (self_stft + cross_stft)
            + self.config.multiband * multiband
            + self.config.envelope * envelope
            + self.config.rms * rms
            + self.config.same_preset * same_preset
            + self.config.pitch_adversary * pitch_adversary
        )
        return Stage1LossOutput(
            total, components, trajectory_a, trajectory_b, self_audio, cross_audio
        )


__all__ = ["AtlasStage1Objective", "Stage1LossOutput"]
