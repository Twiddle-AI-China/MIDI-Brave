from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .atlas_flow_config import AtlasDataConfig, AtlasModelConfig
from .config import ModelConfig as BraveModelConfig
from .model import BraveDecoder, CausalConv1d


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation=dilation)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return value + residual


class TemporalTimbreEncoder(nn.Module):
    """Pitch-normalized 96D frames at hop 512 -> 128D trajectory at hop 2048."""

    def __init__(self, feature_dim: int = 96, channels: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.stem = CausalConv1d(feature_dim, channels, 7)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(channels, dilation) for dilation in (1, 3, 9, 27)
        )
        self.downsample0 = CausalConv1d(channels, channels, 4, stride=2)
        self.downsample1 = CausalConv1d(channels, channels, 4, stride=2)
        self.norm = nn.GroupNorm(1, channels)
        self.output = nn.Conv1d(channels, output_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 96:
            raise ValueError("features must be [batch,96,frames]")
        value = self.blocks[0](self.stem(features))
        value = self.downsample0(value)
        value = self.blocks[1](value)
        value = self.downsample1(value)
        value = self.blocks[2](value)
        value = self.blocks[3](value)
        return torch.tanh(self.output(F.silu(self.norm(value))))


class NoteOnlyConditioner(nn.Module):
    """MIDI pitch only; lifecycle and velocity cannot bypass the trajectory."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        if output_dim != 32:
            raise ValueError("note conditioner is fixed at 32D")
        self.note = nn.Embedding(128, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim)
        )

    def forward(self, note: Tensor, frames: int) -> Tensor:
        if note.ndim != 1 or frames <= 0:
            raise ValueError("note must be [batch] and frames must be positive")
        value = self.projection(self.note(note.long().clamp(0, 127)))
        return value.unsqueeze(-1).expand(-1, -1, frames)


class PhaseContinuousHarmonicExcitation(nn.Module):
    """Parameter-free MIDI excitation with an explicit per-voice phase clock."""

    def __init__(self, sample_rate: int, harmonics: int = 128, target_rms: float = 0.1) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.target_rms = float(target_rms)
        self.register_buffer("harmonics", torch.arange(1, harmonics + 1, dtype=torch.float32))

    @torch.no_grad()
    def forward(
        self, note: Tensor, samples: int, phase: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if note.ndim != 1 or samples <= 0:
            raise ValueError("note must be [batch] and samples must be positive")
        frequency = 440.0 * torch.pow(2.0, (note.float() - 69.0) / 12.0)
        if phase is None:
            phase = torch.zeros_like(frequency)
        if phase.shape != frequency.shape:
            raise ValueError("phase must have shape [batch]")
        step = 2.0 * math.pi * frequency / self.sample_rate
        positions = torch.arange(samples, device=note.device, dtype=torch.float32)
        fundamental = phase[:, None] + step[:, None] * positions[None]
        output = torch.zeros_like(fundamental)
        for harmonic_chunk in self.harmonics.split(16):
            harmonic = harmonic_chunk.to(note.device).view(1, -1, 1)
            keep = frequency[:, None, None] * harmonic <= self.sample_rate / 2.0
            output.add_(((torch.sin(fundamental[:, None] * harmonic) / harmonic) * keep).sum(1))
        measured = output.square().mean(-1, keepdim=True).sqrt().clamp_min(1.0e-6)
        output = output * (self.target_rms / measured)
        end_phase = torch.remainder(phase + step * samples, 2.0 * math.pi)
        return output.unsqueeze(1), end_phase


class AdaLayerNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, -1)
        return self.norm(value) * (1.0 + scale) + shift


class FutureFlowLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = AdaLayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_norm = AdaLayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = AdaLayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width), nn.Dropout(dropout),
        )

    def forward(self, value: Tensor, condition: Tensor, memory: Tensor, history_mask: Tensor) -> Tensor:
        query = self.self_norm(value, condition)
        value = value + self.self_attention(query, query, query, need_weights=False)[0]
        query = self.cross_norm(value, condition)
        value = value + self.cross_attention(
            query, memory, memory, key_padding_mask=~history_mask, need_weights=False
        )[0]
        return value + self.feedforward(self.ff_norm(value, condition))


class AtlasConditionedTrajectoryFlow(nn.Module):
    """Conditional flow over anchor-relative residual trajectories.

    MIDI is intentionally absent. The flow sees legal atlas coordinates,
    canonical anchor paths, lifecycle state, and acoustic history only.
    """

    def __init__(self, config: AtlasModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.trajectory_dim
        width = config.d_model
        self.history_projection = nn.Linear(2 * d, width)
        self.history_position = nn.Parameter(torch.empty(1, config.context_frames, width))
        context_layer = nn.TransformerEncoderLayer(
            width, config.heads, config.feedforward_dim, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            context_layer, config.context_layers, nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.future_projection = nn.Linear(d, width)
        self.future_position = nn.Parameter(torch.empty(1, config.future_frames, width))
        self.anchor_projection = nn.Linear(d, width)
        self.atlas_projection = nn.Sequential(nn.Linear(config.atlas_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.lifecycle_projection = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList(
            FutureFlowLayer(width, config.heads, config.feedforward_dim, config.dropout)
            for _ in range(config.future_layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.future_position, std=0.02)

    def forward(
        self,
        noisy_residual: Tensor,
        flow_time: Tensor,
        history: Tensor,
        history_anchor: Tensor,
        anchor_path: Tensor,
        atlas_path: Tensor,
        lifecycle: Tensor,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        cfg = self.config
        batch = noisy_residual.shape[0]
        expected_future = (batch, cfg.future_frames, cfg.trajectory_dim)
        expected_history = (batch, cfg.context_frames, cfg.trajectory_dim)
        if noisy_residual.shape != expected_future or anchor_path.shape != expected_future:
            raise ValueError("future residual and anchor path have invalid shape")
        if history.shape != expected_history or history_anchor.shape != expected_history:
            raise ValueError("history and history anchor have invalid shape")
        if atlas_path.shape != (batch, cfg.future_frames, cfg.atlas_dim):
            raise ValueError("atlas path has invalid shape")
        if lifecycle.shape != (batch, cfg.future_frames, 4):
            raise ValueError("lifecycle must be [batch,future,4]")
        if flow_time.shape != (batch,):
            raise ValueError("flow_time must be [batch]")
        if not all(torch.isfinite(value).all() for value in (
            noisy_residual, flow_time, history, history_anchor, anchor_path, atlas_path, lifecycle
        )):
            raise ValueError("flow inputs contain non-finite values")
        if history_mask is None:
            history_mask = torch.ones(batch, cfg.context_frames, dtype=torch.bool, device=history.device)
        if history_mask.shape != (batch, cfg.context_frames) or not history_mask.any(1).all():
            raise ValueError("history_mask must expose at least one frame per sample")
        memory_input = torch.cat((history - history_anchor, history_anchor), -1)
        memory = self.history_projection(memory_input) + self.history_position
        memory = self.context(memory, src_key_padding_mask=~history_mask)
        condition = (
            self.anchor_projection(anchor_path)
            + self.atlas_projection(atlas_path)
            + self.lifecycle_projection(lifecycle)
            + self.time_projection(flow_time[:, None, None].expand(-1, cfg.future_frames, -1))
        )
        value = self.future_projection(noisy_residual) + self.future_position + condition
        for layer in self.layers:
            value = layer(value, condition, memory, history_mask)
        return self.output(value)

    @torch.no_grad()
    def sample(
        self,
        history: Tensor,
        history_anchor: Tensor,
        anchor_path: Tensor,
        atlas_path: Tensor,
        lifecycle: Tensor,
        *,
        seed: int,
        temperature: float = 1.0,
        history_mask: Tensor | None = None,
        solver_steps: int | None = None,
    ) -> Tensor:
        if temperature < 0.0 or temperature > 1.5:
            raise ValueError("temperature must be in [0,1.5]")
        generator = torch.Generator(device=history.device)
        generator.manual_seed(int(seed))
        residual = torch.randn(
            anchor_path.shape, generator=generator, device=history.device, dtype=history.dtype
        ) * temperature
        steps = self.config.solver_steps if solver_steps is None else int(solver_steps)
        if steps < 1 or steps > self.config.solver_steps:
            raise ValueError(
                f"solver_steps must be in [1,{self.config.solver_steps}]"
            )
        for index in range(steps):
            t0 = index / steps
            t1 = (index + 1) / steps
            time0 = torch.full((history.shape[0],), t0, device=history.device, dtype=history.dtype)
            velocity0 = self(
                residual, time0, history, history_anchor, anchor_path, atlas_path, lifecycle, history_mask
            )
            estimate = residual + (t1 - t0) * velocity0
            if index + 1 < steps:
                time1 = torch.full_like(time0, t1)
                velocity1 = self(
                    estimate, time1, history, history_anchor, anchor_path, atlas_path, lifecycle, history_mask
                )
                residual = residual + 0.5 * (t1 - t0) * (velocity0 + velocity1)
            else:
                residual = estimate
        return anchor_path + residual


class AtlasTrajectoryInstrument(nn.Module):
    samples_per_decoder_frame = 128

    def __init__(self, model: AtlasModelConfig, data: AtlasDataConfig) -> None:
        super().__init__()
        self.model_config = model
        self.data_config = data
        self.encoder = TemporalTimbreEncoder(model.feature_dim, 128, model.trajectory_dim)
        self.adapter = nn.Sequential(
            nn.Conv1d(model.trajectory_dim, model.decoder_timbre_dim, 1), nn.SiLU(),
            nn.Conv1d(model.decoder_timbre_dim, model.decoder_timbre_dim, 1),
        )
        self.pitch_conditioner = NoteOnlyConditioner(model.midi_dim)
        brave = BraveModelConfig(
            clap_dim=512, timbre_dim=model.decoder_timbre_dim, midi_dim=model.midi_dim,
            capacity=model.decoder_capacity, pqmf_bands=model.pqmf_bands,
            ratios=[2, 2, 2, 1], pitch_backend="none", pqmf_taps=model.pqmf_taps,
            anti_image_taps=31, warmup_latent_frames=64, excitation_harmonics=128,
            excitation_rms=0.1, static_condition_fast_path=False,
            stochastic_excitation=False, decoder_fp32_tail=True, pqmf_dtype="fp32",
        )
        self.decoder = BraveDecoder(brave)
        self.excitation = PhaseContinuousHarmonicExcitation(data.sample_rate)
        self.output_gain = nn.Conv1d(model.trajectory_dim, 1, 1)
        self.pitch_adversary = nn.Sequential(
            nn.Linear(model.trajectory_dim, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        nn.init.zeros_(self.output_gain.weight)
        nn.init.zeros_(self.output_gain.bias)

    def encode(self, features: Tensor) -> Tensor:
        return self.encoder(features)

    def decode(
        self,
        trajectory: Tensor,
        note: Tensor,
        output_samples: int,
        *,
        phase: Tensor | None = None,
        window_start: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if trajectory.ndim != 3 or trajectory.shape[1] != self.model_config.trajectory_dim:
            raise ValueError("trajectory must be [batch,128,frames]")
        warmup_frames = 64
        output_frames = math.ceil(output_samples / self.samples_per_decoder_frame)
        total_frames = warmup_frames + output_frames
        full_frames = math.ceil(self.data_config.render_samples / self.samples_per_decoder_frame)
        expanded_full = F.interpolate(
            trajectory.float(), size=full_frames, mode="linear", align_corners=False
        )
        if window_start is None:
            expanded = F.interpolate(
                trajectory.float(), size=total_frames, mode="linear", align_corners=False
            )
        else:
            if window_start.shape != (trajectory.shape[0],):
                raise ValueError("window_start must be [batch]")
            base = torch.div(
                window_start.long(), self.samples_per_decoder_frame, rounding_mode="floor"
            ) - warmup_frames
            offset = torch.arange(total_frames, device=trajectory.device)
            indices = (base[:, None] + offset[None]).clamp(0, full_frames - 1)
            expanded = expanded_full.gather(
                2, indices[:, None].expand(-1, expanded_full.shape[1], -1)
            )
        timbre = self.adapter(expanded)
        midi = self.pitch_conditioner(note, total_frames)
        total_samples = total_frames * self.samples_per_decoder_frame
        if phase is None and window_start is not None:
            frequency = 440.0 * torch.pow(2.0, (note.float() - 69.0) / 12.0)
            excitation_start = (window_start.long() - warmup_frames * self.samples_per_decoder_frame).clamp_min(0)
            phase = torch.remainder(
                2.0 * math.pi * frequency * excitation_start.float() / self.data_config.sample_rate,
                2.0 * math.pi,
            )
        excitation, end_phase = self.excitation(note, total_samples, phase)
        with torch.autocast(device_type=trajectory.device.type, enabled=False):
            excitation_bands = self.decoder.pqmf.analysis(excitation.float())
        waveform = self.decoder(timbre, midi, excitation_bands, total_samples)
        waveform = waveform[..., warmup_frames * self.samples_per_decoder_frame:]
        waveform = waveform[..., :output_samples]
        gain = 2.0 * torch.sigmoid(self.output_gain(expanded[..., warmup_frames:]))
        gain = F.interpolate(gain, size=output_samples, mode="linear", align_corners=False)
        return waveform * gain.to(waveform.dtype), end_phase

    def parameter_report(self) -> dict[str, int]:
        groups: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "adapter": self.adapter,
            "decoder": self.decoder,
            "pitch_conditioner": self.pitch_conditioner,
            "excitation": self.excitation,
            "output_gain": self.output_gain,
            "pitch_adversary": self.pitch_adversary,
        }
        return {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}


class AtlasFlowSystem(nn.Module):
    def __init__(self, model: AtlasModelConfig, data: AtlasDataConfig) -> None:
        super().__init__()
        self.instrument = AtlasTrajectoryInstrument(model, data)
        self.flow = AtlasConditionedTrajectoryFlow(model)

    def parameter_report(self) -> dict[str, int]:
        report = self.instrument.parameter_report()
        report["flow"] = sum(parameter.numel() for parameter in self.flow.parameters())
        report["total"] = sum(parameter.numel() for parameter in self.parameters())
        return report


@dataclass(frozen=True)
class FlowBatch:
    history: Tensor
    history_anchor: Tensor
    target: Tensor
    anchor_path: Tensor
    atlas_path: Tensor
    lifecycle: Tensor
    history_mask: Tensor


__all__ = [
    "AtlasConditionedTrajectoryFlow", "AtlasFlowSystem", "AtlasTrajectoryInstrument",
    "FlowBatch", "NoteOnlyConditioner", "PhaseContinuousHarmonicExcitation",
    "TemporalTimbreEncoder",
]
