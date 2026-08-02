from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .zrave_flow_config import FlowModelConfig
from .zrave_flow_exploration import trailing_history_mask


@dataclass(frozen=True)
class FlowStatistics:
    mean: Tensor
    latent_std: Tensor
    delta_std: Tensor
    latent_norm_p01: Tensor
    latent_norm_p99: Tensor

    def validate(self, latent_dim: int) -> None:
        for name in ("mean", "latent_std", "delta_std"):
            value = getattr(self, name)
            if value.shape != (latent_dim,):
                raise ValueError(
                    f"statistics.{name} must have shape ({latent_dim},), "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"statistics.{name} is not finite")
        if not torch.all(self.latent_std > 0):
            raise ValueError("statistics.latent_std must be positive")
        if not torch.all(self.delta_std > 0):
            raise ValueError("statistics.delta_std must be positive")
        for name in ("latent_norm_p01", "latent_norm_p99"):
            value = getattr(self, name)
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(
                    f"statistics.{name} must be one finite value"
                )
        if self.latent_norm_p01.item() < 0:
            raise ValueError(
                "statistics.latent_norm_p01 must be non-negative"
            )
        if self.latent_norm_p99.item() <= self.latent_norm_p01.item():
            raise ValueError(
                "statistics latent norm percentiles are not ordered"
            )


class AdaLayerNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return (
            self.normalization(value) * (1.0 + scale.unsqueeze(1))
            + shift.unsqueeze(1)
        )


class _FutureFlowLayer(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.adaln_self = AdaLayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.adaln_cross = AdaLayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.adaln_feedforward = AdaLayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        value: Tensor,
        condition: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None,
        retention: Tensor,
        future_mask: Tensor,
    ) -> Tensor:
        conditioned = self.adaln_self(value, condition)
        attended = self.self_attention(
            conditioned,
            conditioned,
            conditioned,
            key_padding_mask=~future_mask,
            need_weights=False,
        )[0]
        value = value + attended
        query = self.adaln_cross(value, condition)
        cross = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=(
                ~memory_mask if memory_mask is not None else None
            ),
            need_weights=False,
        )[0]
        value = value + retention.unsqueeze(-1) * cross
        value = value + self.feedforward(
            self.adaln_feedforward(value, condition)
        )
        return value


def _resolve_model_dimensions(
    config: FlowModelConfig | None,
    overrides: dict[str, bool | int | float | None],
) -> dict[str, bool | int | float]:
    defaults: dict[str, bool | int | float] = {
        "latent_dim": 16,
        "context_frames": 32,
        "future_frames": 64,
        "d_model": 384,
        "context_layers": 4,
        "future_layers": 8,
        "heads": 8,
        "feedforward_dim": 1536,
        "dropout": 0.0,
        "pitch_conditioning": True,
        "note_min": 21,
        "note_max": 109,
    }
    if config is not None:
        if any(value is not None for value in overrides.values()):
            raise ValueError(
                "model dimension overrides cannot accompany config"
            )
        return {
            name: getattr(config, name)
            for name in defaults
        }
    return {
        name: defaults[name] if value is None else value
        for name, value in overrides.items()
    }


class ZraveFlowTransformer(nn.Module):
    def __init__(
        self,
        config: FlowModelConfig | None = None,
        statistics: FlowStatistics | None = None,
        *,
        latent_dim: int | None = None,
        context_frames: int | None = None,
        future_frames: int | None = None,
        d_model: int | None = None,
        context_layers: int | None = None,
        future_layers: int | None = None,
        heads: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float | None = None,
        pitch_conditioning: bool | None = None,
        note_min: int | None = None,
        note_max: int | None = None,
    ) -> None:
        super().__init__()
        dimensions = _resolve_model_dimensions(
            config,
            {
                "latent_dim": latent_dim,
                "context_frames": context_frames,
                "future_frames": future_frames,
                "d_model": d_model,
                "context_layers": context_layers,
                "future_layers": future_layers,
                "heads": heads,
                "feedforward_dim": feedforward_dim,
                "dropout": dropout,
                "pitch_conditioning": pitch_conditioning,
                "note_min": note_min,
                "note_max": note_max,
            },
        )
        self.latent_dim = int(dimensions["latent_dim"])
        self.context_frames = int(dimensions["context_frames"])
        self.future_frames = int(dimensions["future_frames"])
        self.d_model = int(dimensions["d_model"])
        self.pitch_conditioning = bool(
            dimensions["pitch_conditioning"]
        )
        self.note_min = int(dimensions["note_min"])
        self.note_max = int(dimensions["note_max"])
        context_layer_count = int(dimensions["context_layers"])
        future_layer_count = int(dimensions["future_layers"])
        attention_heads = int(dimensions["heads"])
        feedforward_width = int(dimensions["feedforward_dim"])
        dropout_probability = float(dimensions["dropout"])
        if statistics is None:
            raise ValueError("statistics are required")
        statistics.validate(self.latent_dim)
        if self.d_model % attention_heads:
            raise ValueError("d_model must be divisible by heads")
        if min(
            self.latent_dim,
            self.context_frames,
            self.future_frames,
            self.d_model,
            context_layer_count,
            future_layer_count,
            attention_heads,
            feedforward_width,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.note_max < self.note_min:
            raise ValueError("invalid MIDI note range")

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
            "latent_norm_p01",
            statistics.latent_norm_p01.float().reshape(()).clone(),
        )
        self.register_buffer(
            "latent_norm_p99",
            statistics.latent_norm_p99.float().reshape(()).clone(),
        )

        self.history_projection = nn.Linear(self.latent_dim, self.d_model)
        self.history_position = nn.Parameter(
            torch.empty(1, self.context_frames, self.d_model)
        )
        context_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_width,
            dropout=dropout_probability,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=context_layer_count,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        if self.pitch_conditioning:
            self.null_memory = nn.Parameter(
                torch.empty(1, self.context_frames, self.d_model)
            )
        else:
            self.register_parameter("null_memory", None)

        self.future_projection = nn.Linear(self.latent_dim, self.d_model)
        self.future_position = nn.Parameter(
            torch.empty(1, self.future_frames, self.d_model)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        note_count = self.note_max - self.note_min + 1
        self.null_note_index = note_count
        self.midi_embedding = (
            nn.Embedding(note_count + 1, self.d_model)
            if self.pitch_conditioning
            else None
        )
        self.future_layers = nn.ModuleList(
            _FutureFlowLayer(
                self.d_model,
                attention_heads,
                feedforward_width,
                dropout_probability,
            )
            for _ in range(future_layer_count)
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.velocity_projection = nn.Linear(
            self.d_model,
            self.latent_dim,
        )
        nn.init.normal_(self.history_position, mean=0.0, std=0.02)
        nn.init.normal_(self.future_position, mean=0.0, std=0.02)
        if self.null_memory is not None:
            nn.init.normal_(self.null_memory, mean=0.0, std=0.02)

    def statistics(self) -> FlowStatistics:
        return FlowStatistics(
            mean=self.latent_mean,
            latent_std=self.latent_std,
            delta_std=self.delta_std,
            latent_norm_p01=self.latent_norm_p01,
            latent_norm_p99=self.latent_norm_p99,
        )

    def _validate_core_forward(
        self,
        noisy_future: Tensor,
        flow_time: Tensor,
        history: Tensor,
        retention: Tensor,
        future_mask: Tensor | None,
        history_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch = noisy_future.shape[0] if noisy_future.ndim else 0
        future_shape = (batch, self.future_frames, self.latent_dim)
        if noisy_future.ndim != 3 or noisy_future.shape != future_shape:
            raise ValueError(
                f"noisy_future must have shape {future_shape}, "
                f"got {tuple(noisy_future.shape)}"
            )
        history_shape = (batch, self.context_frames, self.latent_dim)
        if history.shape != history_shape:
            raise ValueError(
                f"history must have shape {history_shape}, "
                f"got {tuple(history.shape)}"
            )
        if flow_time.shape != (batch,):
            raise ValueError(
                f"flow_time must have shape ({batch},), "
                f"got {tuple(flow_time.shape)}"
            )
        if retention.shape != (batch, self.future_frames):
            raise ValueError(
                "retention must have shape "
                f"({batch}, {self.future_frames})"
            )
        if not torch.isfinite(noisy_future).all():
            raise ValueError("noisy_future contains non-finite values")
        if not torch.isfinite(history).all():
            raise ValueError("history contains non-finite values")
        if not torch.isfinite(flow_time).all():
            raise ValueError("flow_time contains non-finite values")
        if torch.any(flow_time < 0) or torch.any(flow_time > 1):
            raise ValueError("flow_time must be in [0, 1]")
        if not torch.isfinite(retention).all():
            raise ValueError("retention contains non-finite values")
        device = noisy_future.device
        if future_mask is None:
            resolved_mask = torch.ones(
                batch,
                self.future_frames,
                device=device,
                dtype=torch.bool,
            )
        else:
            if future_mask.shape != (batch, self.future_frames):
                raise ValueError(
                    "future_mask must have shape "
                    f"({batch}, {self.future_frames})"
                )
            resolved_mask = future_mask.to(device=device, dtype=torch.bool)
        if not torch.all(resolved_mask.any(dim=1)):
            raise ValueError("every future_mask row needs one valid frame")
        if history_mask is None:
            resolved_history_mask = torch.ones(
                batch,
                self.context_frames,
                device=device,
                dtype=torch.bool,
            )
        else:
            if history_mask.shape != (batch, self.context_frames):
                raise ValueError(
                    "history_mask must have shape "
                    f"({batch}, {self.context_frames})"
                )
            resolved_history_mask = history_mask.to(
                device=device,
                dtype=torch.bool,
            )
        if not torch.all(resolved_history_mask.any(dim=1)):
            raise ValueError(
                "every sample needs at least one visible history frame"
            )
        return resolved_mask, resolved_history_mask

    def _validate_forward(
        self,
        noisy_future: Tensor,
        flow_time: Tensor,
        history: Tensor,
        midi_note: Tensor,
        retention: Tensor,
        future_mask: Tensor | None,
        context_present: Tensor | None,
        history_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        resolved_mask, resolved_history_mask = self._validate_core_forward(
            noisy_future,
            flow_time,
            history,
            retention,
            future_mask,
            history_mask,
        )
        batch = noisy_future.shape[0]
        if midi_note.shape != (batch,):
            raise ValueError(
                f"midi_note must have shape ({batch},), "
                f"got {tuple(midi_note.shape)}"
            )
        notes = midi_note.to(dtype=torch.long)
        valid_notes = (notes == -1) | (
            (notes >= self.note_min) & (notes <= self.note_max)
        )
        if not torch.all(valid_notes):
            raise ValueError(
                "midi_note must be -1 or in "
                f"[{self.note_min}, {self.note_max}]"
            )
        if midi_note.is_floating_point() and not torch.equal(
            midi_note,
            midi_note.round(),
        ):
            raise ValueError("midi_note must contain integer values")
        device = noisy_future.device
        if context_present is None:
            resolved_context = torch.ones(
                batch,
                device=device,
                dtype=torch.bool,
            )
        else:
            if context_present.shape != (batch,):
                raise ValueError(
                    f"context_present must have shape ({batch},)"
                )
            resolved_context = context_present.to(
                device=device,
                dtype=torch.bool,
            )
        return resolved_mask, resolved_context, resolved_history_mask

    def _encode_history(
        self,
        history: Tensor,
        history_mask: Tensor,
        context_present: Tensor | None = None,
    ) -> Tensor:
        normalized_history = (
            history.float() - self.latent_mean
        ) / self.latent_std
        normalized_history = normalized_history.masked_fill(
            ~history_mask.unsqueeze(-1),
            0.0,
        )
        memory = self.history_projection(normalized_history)
        memory = self.context_encoder(
            memory + self.history_position,
            src_key_padding_mask=~history_mask,
        )
        if context_present is not None:
            if self.null_memory is None:
                raise RuntimeError(
                    "context dropout requires pitch conditioning"
                )
            null_memory = self.null_memory.expand(
                memory.shape[0],
                -1,
                -1,
            )
            memory = torch.where(
                context_present[:, None, None],
                memory,
                null_memory,
            )
        return memory

    def _predict_velocity(
        self,
        noisy_future: Tensor,
        flow_time: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        retention: Tensor,
        future_mask: Tensor,
        condition_offset: Tensor | None = None,
    ) -> Tensor:
        device = noisy_future.device
        condition = self.time_embedding(
            flow_time.to(device=device, dtype=torch.float32).unsqueeze(-1)
        )
        if condition_offset is not None:
            condition = condition + condition_offset
        value = self.future_projection(noisy_future.float())
        value = value + self.future_position
        value = value.masked_fill(~future_mask.unsqueeze(-1), 0.0)
        retention = retention.to(device=device, dtype=value.dtype)
        for layer in self.future_layers:
            value = layer(
                value,
                condition,
                memory,
                memory_mask,
                retention,
                future_mask,
            )
        velocity = self.velocity_projection(self.output_norm(value))
        return velocity.masked_fill(
            ~future_mask.unsqueeze(-1),
            0.0,
        )

    def forward(
        self,
        noisy_future: Tensor,
        flow_time: Tensor,
        history: Tensor,
        midi_note: Tensor | None = None,
        retention: Tensor | None = None,
        future_mask: Tensor | None = None,
        context_present: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        if not self.pitch_conditioning:
            if midi_note is not None or context_present is not None:
                raise ValueError(
                    "pure flow forward does not accept MIDI/context dropout"
                )
            if retention is None:
                raise ValueError("retention is required")
            return self.forward_pure(
                noisy_future,
                flow_time,
                history,
                retention,
                future_mask=future_mask,
                history_mask=history_mask,
            )
        if midi_note is None or retention is None:
            raise ValueError(
                "conditional flow forward requires MIDI and retention"
            )
        if self.midi_embedding is None:
            raise RuntimeError("conditional model lacks MIDI embedding")
        future_mask, context_present, history_mask = self._validate_forward(
            noisy_future,
            flow_time,
            history,
            midi_note,
            retention,
            future_mask,
            context_present,
            history_mask,
        )
        device = noisy_future.device
        memory = self._encode_history(
            history,
            history_mask,
            context_present,
        )

        notes = midi_note.to(device=device, dtype=torch.long)
        note_indices = torch.where(
            notes == -1,
            torch.full_like(notes, self.null_note_index),
            notes - self.note_min,
        )
        return self._predict_velocity(
            noisy_future,
            flow_time,
            memory,
            history_mask,
            retention,
            future_mask,
            self.midi_embedding(note_indices),
        )

    def forward_pure(
        self,
        noisy_future: Tensor,
        flow_time: Tensor,
        history: Tensor,
        retention: Tensor,
        future_mask: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        resolved_mask, resolved_history_mask = self._validate_core_forward(
            noisy_future,
            flow_time,
            history,
            retention,
            future_mask,
            history_mask,
        )
        return self._predict_velocity(
            noisy_future,
            flow_time,
            self._encode_history(history, resolved_history_mask),
            resolved_history_mask,
            retention,
            resolved_mask,
        )


def _smoothstep(progress: Tensor) -> Tensor:
    progress = progress.clamp(0.0, 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _schedule_progress(
    delays: Tensor,
    frames: int,
    offset_frames: Tensor | int = 0,
) -> Tensor:
    if frames <= 1:
        raise ValueError("frames must be greater than one")
    if delays.ndim != 1 or delays.numel() == 0:
        raise ValueError("delays must be a non-empty vector")
    if not torch.isfinite(delays).all() or torch.any(delays <= 0):
        raise ValueError("delays must be finite and positive")
    offsets = torch.as_tensor(
        offset_frames,
        device=delays.device,
        dtype=torch.float32,
    )
    if offsets.ndim == 0:
        offsets = offsets.expand(delays.shape[0])
    if (
        offsets.shape != delays.shape
        or not torch.isfinite(offsets).all()
        or torch.any(offsets < 0)
    ):
        raise ValueError(
            "schedule offsets must be finite, non-negative, and match delays"
        )
    positions = offsets.unsqueeze(1) + torch.arange(
        frames,
        device=delays.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    progress = positions / delays.float().unsqueeze(1)
    return _smoothstep(progress)


def retention_curve(
    delays: Tensor,
    frames: int,
    *,
    offset_frames: Tensor | int = 0,
) -> Tensor:
    progress = _schedule_progress(delays, frames, offset_frames)
    return 1.0 - 0.85 * progress


def temperature_curve(
    temperature: Tensor,
    delays: Tensor,
    frames: int,
    *,
    offset_frames: Tensor | int = 0,
) -> Tensor:
    progress = _schedule_progress(delays, frames, offset_frames)
    temperature = torch.as_tensor(
        temperature,
        device=delays.device,
        dtype=torch.float32,
    )
    if temperature.ndim == 0:
        temperature = temperature.expand(delays.shape[0])
    if temperature.shape != delays.shape:
        raise ValueError("temperature and delays must have equal shape")
    if not torch.isfinite(temperature).all() or torch.any(temperature < 0):
        raise ValueError("temperature must be finite and non-negative")
    return temperature.unsqueeze(1) * (0.15 + 0.85 * progress)


def derive_block_seed(global_seed: int, block_index: int) -> int:
    if global_seed < 0:
        raise ValueError("global_seed must be non-negative")
    if block_index < 0:
        raise ValueError("block_index must be non-negative")
    digest = hashlib.sha256(
        f"{global_seed}:{block_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _guided_velocity(
    model: ZraveFlowTransformer,
    state: Tensor,
    flow_time: float,
    history: Tensor,
    midi_note: Tensor,
    retention: Tensor,
    pitch_guidance: float,
) -> Tensor:
    times = torch.full(
        (state.shape[0],),
        flow_time,
        device=state.device,
        dtype=torch.float32,
    )
    full = model(
        state,
        times,
        history,
        midi_note,
        retention,
    )
    no_pitch = model(
        state,
        times,
        history,
        torch.full_like(midi_note, -1),
        retention,
    )
    return no_pitch + pitch_guidance * (full - no_pitch)


def sample_flow_block(
    model: ZraveFlowTransformer,
    statistics: FlowStatistics,
    history: Tensor,
    midi_note: Tensor,
    *,
    generation_seed: int,
    block_index: int,
    temperature: float,
    wander_delay_frames: int,
    pitch_guidance: float,
    solver_steps: int,
) -> Tensor:
    if not torch.isfinite(torch.tensor(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if wander_delay_frames not in {16, 32, 48}:
        raise ValueError("wander_delay_frames must be 16, 32, or 48")
    if not 1.0 <= pitch_guidance <= 5.0:
        raise ValueError("pitch_guidance must be in [1, 5]")
    if solver_steps not in {4, 8, 12}:
        raise ValueError("solver_steps must be 4, 8, or 12")
    statistics.validate(model.latent_dim)
    device = next(model.parameters()).device
    history = history.to(device=device)
    midi_note = midi_note.to(device=device)
    batch = history.shape[0] if history.ndim else 0
    if midi_note.shape != (batch,):
        raise ValueError(f"midi_note must have shape ({batch},)")
    delay = torch.full(
        (batch,),
        wander_delay_frames,
        device=device,
        dtype=torch.long,
    )
    retention = retention_curve(delay, model.future_frames)
    temperatures = temperature_curve(
        torch.full(
            (batch,),
            temperature,
            device=device,
            dtype=torch.float32,
        ),
        delay,
        model.future_frames,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_block_seed(generation_seed, block_index))
    state = torch.randn(
        batch,
        model.future_frames,
        model.latent_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    state = state * temperatures.unsqueeze(-1)
    step = 1.0 / solver_steps
    with torch.no_grad():
        for index in range(solver_steps):
            t0 = index * step
            v0 = _guided_velocity(
                model,
                state,
                t0,
                history,
                midi_note,
                retention,
                pitch_guidance,
            )
            proposal = state + step * v0
            v1 = _guided_velocity(
                model,
                proposal,
                min(1.0, t0 + step),
                history,
                midi_note,
                retention,
                pitch_guidance,
            )
            state = state + 0.5 * step * (v0 + v1)
    mean = statistics.mean.to(device=device, dtype=state.dtype)
    latent_std = statistics.latent_std.to(
        device=device,
        dtype=state.dtype,
    )
    return state * latent_std + mean


def _pure_velocity(
    model: ZraveFlowTransformer,
    state: Tensor,
    flow_time: float,
    history: Tensor,
    retention: Tensor,
    history_mask: Tensor,
) -> Tensor:
    times = torch.full(
        (state.shape[0],),
        flow_time,
        device=state.device,
        dtype=torch.float32,
    )
    return model.forward_pure(
        state,
        times,
        history,
        retention,
        history_mask=history_mask,
    )


def sample_pure_flow_block(
    model: ZraveFlowTransformer,
    statistics: FlowStatistics,
    history: Tensor,
    *,
    generation_seed: int,
    block_index: int,
    temperature: float,
    wander_delay_frames: float,
    solver_steps: int,
    schedule_offset_frames: int = 0,
    visible_history_frames: int = 32,
) -> Tensor:
    if not torch.isfinite(torch.tensor(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if (
        not torch.isfinite(torch.tensor(wander_delay_frames))
        or not 16.0 <= wander_delay_frames <= 48.0
    ):
        raise ValueError(
            "wander_delay_frames must be finite and in [16, 48]"
        )
    if solver_steps not in {4, 8, 12}:
        raise ValueError("solver_steps must be 4, 8, or 12")
    if schedule_offset_frames < 0:
        raise ValueError("schedule_offset_frames must be non-negative")
    if not 8 <= visible_history_frames <= model.context_frames:
        raise ValueError(
            "visible_history_frames must be between 8 and context_frames"
        )
    statistics.validate(model.latent_dim)
    device = next(model.parameters()).device
    history = history.to(device=device)
    batch = history.shape[0] if history.ndim else 0
    delay = torch.full(
        (batch,),
        wander_delay_frames,
        device=device,
        dtype=torch.float32,
    )
    retention = retention_curve(
        delay,
        model.future_frames,
        offset_frames=schedule_offset_frames,
    )
    temperatures = temperature_curve(
        torch.full(
            (batch,),
            temperature,
            device=device,
            dtype=torch.float32,
        ),
        delay,
        model.future_frames,
        offset_frames=schedule_offset_frames,
    )
    history_mask = trailing_history_mask(
        torch.full(
            (batch,),
            visible_history_frames,
            device=device,
            dtype=torch.long,
        ),
        model.context_frames,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(
        derive_block_seed(generation_seed, block_index)
    )
    state = torch.randn(
        batch,
        model.future_frames,
        model.latent_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    state = state * temperatures.unsqueeze(-1)
    step = 1.0 / solver_steps
    with torch.no_grad():
        for index in range(solver_steps):
            t0 = index * step
            v0 = _pure_velocity(
                model,
                state,
                t0,
                history,
                retention,
                history_mask,
            )
            proposal = state + step * v0
            v1 = _pure_velocity(
                model,
                proposal,
                min(1.0, t0 + step),
                history,
                retention,
                history_mask,
            )
            state = state + 0.5 * step * (v0 + v1)
    mean = statistics.mean.to(device=device, dtype=state.dtype)
    latent_std = statistics.latent_std.to(
        device=device,
        dtype=state.dtype,
    )
    return state * latent_std + mean
