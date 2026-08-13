from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .zrave_flow_model import FlowStatistics, temperature_curve
from .zrave_pitch_probe import pitch_probe_loss


@dataclass(frozen=True)
class FlowTrainingPair:
    noisy_future: Tensor
    target_velocity: Tensor
    flow_time: Tensor
    normalized_future: Tensor
    clean_future: Tensor


@dataclass(frozen=True)
class FlowLossReport:
    total: Tensor
    components: dict[str, Tensor]


def _validate_future(
    future: Tensor,
    future_mask: Tensor,
    statistics: FlowStatistics,
) -> Tensor:
    if future.ndim != 3:
        raise ValueError("future must have shape [batch, frames, latent_dim]")
    batch, frames, latent_dim = future.shape
    if frames != 64:
        raise ValueError("future must contain 64 frames")
    statistics.validate(latent_dim)
    if future_mask.shape != (batch, frames):
        raise ValueError(f"future_mask must have shape ({batch}, {frames})")
    if not torch.isfinite(future).all():
        raise ValueError("future contains non-finite values")
    mask = future_mask.to(device=future.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every sample needs at least one valid future frame")
    return mask


def _batch_vector(
    value: Tensor | float | int,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(batch)
    if result.shape != (batch,):
        raise ValueError(f"{name} must be scalar or shape ({batch},)")
    return result


def make_flow_training_pair(
    future: Tensor,
    future_mask: Tensor,
    temperature: Tensor | float,
    wander_delay_frames: Tensor | int,
    statistics: FlowStatistics,
    generator: torch.Generator,
    schedule_offset_frames: Tensor | int = 0,
) -> FlowTrainingPair:
    mask = _validate_future(future, future_mask, statistics)
    batch, frames, latent_dim = future.shape
    device = future.device
    delays = _batch_vector(
        wander_delay_frames,
        batch=batch,
        device=device,
        dtype=torch.float32,
        name="wander_delay_frames",
    )
    if (
        not torch.isfinite(delays).all()
        or torch.any(delays < 16)
        or torch.any(delays > 48)
    ):
        raise ValueError("wander_delay_frames must be finite and in [16, 48]")
    temperatures = _batch_vector(
        temperature,
        batch=batch,
        device=device,
        dtype=torch.float32,
        name="temperature",
    )
    if not torch.isfinite(temperatures).all() or torch.any(temperatures < 0):
        raise ValueError("temperature must be finite and non-negative")
    mean = statistics.mean.to(device=device, dtype=torch.float32)
    latent_std = statistics.latent_std.to(
        device=device,
        dtype=torch.float32,
    )
    clean_future = future.float()
    normalized_future = (clean_future - mean) / latent_std
    noise = torch.randn(
        batch,
        frames,
        latent_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    noise = noise * temperature_curve(
        temperatures,
        delays,
        frames,
        offset_frames=schedule_offset_frames,
    ).unsqueeze(-1)
    flow_time = torch.rand(
        batch,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    interpolation = flow_time[:, None, None]
    noisy_future = (1.0 - interpolation) * noise + interpolation * normalized_future
    target_velocity = normalized_future - noise
    invalid = ~mask.unsqueeze(-1)
    return FlowTrainingPair(
        noisy_future=noisy_future.masked_fill(invalid, 0.0),
        target_velocity=target_velocity.masked_fill(invalid, 0.0),
        flow_time=flow_time,
        normalized_future=normalized_future.masked_fill(invalid, 0.0),
        clean_future=clean_future,
    )


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    if values.ndim == 0:
        return values if bool(expanded) else values * 0.0
    flattened_values = values.reshape(values.shape[0], -1)
    flattened_mask = expanded.reshape(values.shape[0], -1)
    counts = flattened_mask.sum(dim=1)
    valid = counts > 0
    if not torch.any(valid):
        return values.sum() * 0.0
    per_sample = (flattened_values * flattened_mask.to(values.dtype)).sum(
        dim=1
    ) / counts.clamp_min(1).to(values.dtype)
    return per_sample.masked_select(valid).mean()


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    element_loss = functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    return _masked_mean(element_loss, mask)


def _masked_channel_std(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum().clamp_min(1.0)
    mean = (values * weights).sum(dim=(0, 1)) / count
    variance = ((values - mean).square() * weights).sum(dim=(0, 1)) / count
    return variance.clamp_min(0.0).sqrt()


def _masked_delta_rms(values: Tensor, mask: Tensor) -> Tensor:
    delta_mask = mask[:, 1:] & mask[:, :-1]
    weights = delta_mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum().clamp_min(1.0)
    deltas = values[:, 1:] - values[:, :-1]
    mean_square = (deltas.square() * weights).sum(dim=(0, 1)) / count
    return (mean_square + 1.0e-12).sqrt()


def _per_sample_motion(
    values: Tensor,
    mask: Tensor,
    scale: Tensor,
    stride: int,
) -> Tensor:
    if stride <= 0 or stride >= values.shape[1]:
        raise ValueError("motion stride is outside the sequence")
    pair_mask = mask[:, stride:] & mask[:, :-stride]
    delta = (values[:, stride:] - values[:, :-stride]) / scale.to(
        device=values.device, dtype=values.dtype
    ).view(
        1,
        1,
        -1,
    )
    weights = pair_mask.to(dtype=delta.dtype).unsqueeze(-1)
    count = weights.sum(dim=(1, 2)).clamp_min(1.0)
    count = count * values.shape[-1]
    return ((delta.square() * weights).sum(dim=(1, 2)) / count + 1.0e-8).sqrt()


def _temporal_motion_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    delta_std: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("temporal prediction and target must share rank-three shape")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("temporal mask shape does not match values")
    if delta_std.shape != (prediction.shape[-1],):
        raise ValueError("temporal delta scale has the wrong shape")
    losses: list[Tensor] = []
    for stride, weight in ((1, 1.0), (4, 0.5)):
        predicted = _per_sample_motion(
            prediction,
            mask,
            delta_std,
            stride,
        )
        expected = _per_sample_motion(
            target,
            mask,
            delta_std,
            stride,
        )
        match = functional.smooth_l1_loss(
            predicted.log(),
            expected.log(),
        )
        moving = expected > 0.25
        floor = functional.relu(0.5 * expected - predicted)
        floor_loss = (
            floor.masked_select(moving).mean()
            if torch.any(moving)
            else floor.sum() * 0.0
        )
        losses.append(weight * (match + floor_loss))
    return (losses[0] + losses[1]) / 1.5


def _pitch_windows(
    clean_estimate: Tensor,
    future_mask: Tensor,
    midi_note: Tensor,
) -> tuple[Tensor | None, Tensor | None]:
    windows: list[Tensor] = []
    notes: list[Tensor] = []
    if midi_note.shape == (clean_estimate.shape[0],):
        note_sequence = midi_note[:, None].expand(-1, clean_estimate.shape[1])
    elif midi_note.shape == clean_estimate.shape[:2]:
        note_sequence = midi_note
    else:
        raise ValueError("midi_note must be [batch] or [batch, future_frames]")
    for sample in range(clean_estimate.shape[0]):
        valid_frames = int(future_mask[sample].sum().item())
        if valid_frames <= 0:
            continue
        sample_notes = note_sequence[sample, :valid_frames]
        run_start = 0
        while run_start < valid_frames:
            run_note = sample_notes[run_start]
            run_end = run_start + 1
            while run_end < valid_frames and bool(sample_notes[run_end] == run_note):
                run_end += 1
            run_length = run_end - run_start
            complete_starts = range(run_start, run_end - 15, 16)
            emitted = False
            for start in complete_starts:
                windows.append(
                    clean_estimate[
                        sample : sample + 1,
                        start : start + 16,
                    ]
                )
                notes.append(run_note.reshape(1))
                emitted = True
            if not emitted:
                partial = clean_estimate[sample, run_start:run_end]
                padding = partial[-1:].expand(16 - run_length, -1)
                windows.append(torch.cat((partial, padding), dim=0).unsqueeze(0))
                notes.append(run_note.reshape(1))
            run_start = run_end
    if not windows:
        return None, None
    return torch.cat(windows, dim=0), torch.cat(notes, dim=0)


def _last_visible_history(history: Tensor, history_mask: Tensor | None) -> Tensor:
    if history_mask is None:
        return history[:, -1]
    if history_mask.shape != history.shape[:2]:
        raise ValueError("history_mask shape does not match history")
    resolved = history_mask.to(device=history.device, dtype=torch.bool)
    if not torch.all(resolved.any(dim=1)):
        raise ValueError("every sample needs visible history")
    indices = torch.arange(
        history.shape[1],
        device=history.device,
    ).expand(history.shape[0], -1)
    last = indices.masked_fill(~resolved, -1).max(dim=1).values
    return history[torch.arange(history.shape[0], device=history.device), last]


def _boundary_loss(
    clean_estimate: Tensor,
    clean_future: Tensor,
    history: Tensor,
    future_mask: Tensor,
    history_mask: Tensor | None = None,
) -> Tensor:
    first_mask = future_mask[:, :8]
    frame_loss = _masked_smooth_l1(
        clean_estimate[:, :8],
        clean_future[:, :8],
        first_mask,
    )
    valid_boundary = future_mask[:, 0]
    history_boundary = _last_visible_history(history, history_mask)
    estimated_delta = clean_estimate[:, 0] - history_boundary
    target_delta = clean_future[:, 0] - history_boundary
    delta_loss = _masked_smooth_l1(
        estimated_delta,
        target_delta,
        valid_boundary,
    )
    return frame_loss + delta_loss


def _statistics_loss(
    clean_estimate: Tensor,
    clean_future: Tensor,
    future_mask: Tensor,
    statistics: FlowStatistics,
) -> Tensor:
    std_loss = functional.smooth_l1_loss(
        _masked_channel_std(clean_estimate, future_mask),
        _masked_channel_std(clean_future, future_mask),
    )
    delta_loss = functional.smooth_l1_loss(
        _masked_delta_rms(clean_estimate, future_mask),
        _masked_delta_rms(clean_future, future_mask),
    )
    norm = torch.linalg.vector_norm(clean_estimate, dim=-1)
    lower = statistics.latent_norm_p01.to(
        device=norm.device,
        dtype=norm.dtype,
    )
    upper = statistics.latent_norm_p99.to(
        device=norm.device,
        dtype=norm.dtype,
    )
    hinge = torch.relu(lower - norm) + torch.relu(norm - upper)
    norm_hinge = _masked_mean(hinge, future_mask)
    return std_loss + delta_loss + norm_hinge


def zrave_flow_loss(
    predicted_velocity: Tensor,
    pair: FlowTrainingPair,
    history: Tensor,
    future_mask: Tensor,
    midi_note: Tensor,
    pitch_probe: nn.Module,
    statistics: FlowStatistics,
    pitch_weight: float,
    history_mask: Tensor | None = None,
) -> FlowLossReport:
    mask = _validate_future(
        pair.clean_future,
        future_mask,
        statistics,
    )
    expected_shape = pair.clean_future.shape
    for name in (
        "noisy_future",
        "target_velocity",
        "normalized_future",
    ):
        value = getattr(pair, name)
        if value.shape != expected_shape:
            raise ValueError(f"pair.{name} must have shape {expected_shape}")
    if predicted_velocity.shape != expected_shape:
        raise ValueError(f"predicted_velocity must have shape {expected_shape}")
    batch, _frames, latent_dim = expected_shape
    if pair.flow_time.shape != (batch,):
        raise ValueError(f"pair.flow_time must have shape ({batch},)")
    if history.shape != (batch, 32, latent_dim):
        raise ValueError(f"history must have shape ({batch}, 32, {latent_dim})")
    if midi_note.shape not in {(batch,), (batch, expected_shape[1])}:
        raise ValueError(
            f"midi_note must have shape ({batch},) or ({batch}, {expected_shape[1]})"
        )
    if not math.isfinite(pitch_weight) or pitch_weight < 0:
        raise ValueError("pitch_weight must be finite and non-negative")
    if not torch.isfinite(predicted_velocity).all():
        raise ValueError("predicted_velocity contains non-finite values")

    flow = _masked_mean(
        (predicted_velocity.float() - pair.target_velocity).square(),
        mask,
    )
    clean_normalized = (
        pair.noisy_future
        + (1.0 - pair.flow_time[:, None, None]) * predicted_velocity.float()
    )
    mean = statistics.mean.to(
        device=predicted_velocity.device,
        dtype=torch.float32,
    )
    latent_std = statistics.latent_std.to(
        device=predicted_velocity.device,
        dtype=torch.float32,
    )
    clean_estimate = clean_normalized * latent_std + mean

    windows, window_notes = _pitch_windows(
        clean_estimate,
        mask,
        midi_note,
    )
    if windows is None or window_notes is None:
        pitch = predicted_velocity.sum() * 0.0
    else:
        output = pitch_probe(windows)
        pitch = pitch_probe_loss(
            output,
            window_notes,
            note_min=int(getattr(pitch_probe, "note_min", 21)),
            note_max=int(getattr(pitch_probe, "note_max", 109)),
        ).total
    boundary = _boundary_loss(
        clean_estimate,
        pair.clean_future,
        history,
        mask,
        history_mask,
    )
    statistics_loss = _statistics_loss(
        clean_estimate,
        pair.clean_future,
        mask,
        statistics,
    )
    total = flow + pitch_weight * pitch + 0.10 * boundary + 0.02 * statistics_loss
    return FlowLossReport(
        total=total,
        components={
            "flow": flow,
            "pitch": pitch,
            "boundary": boundary,
            "statistics": statistics_loss,
        },
    )


def zrave_pure_flow_loss(
    predicted_velocity: Tensor,
    pair: FlowTrainingPair,
    history: Tensor,
    future_mask: Tensor,
    statistics: FlowStatistics,
    temporal_weight: float = 0.0,
    history_mask: Tensor | None = None,
) -> FlowLossReport:
    mask = _validate_future(
        pair.clean_future,
        future_mask,
        statistics,
    )
    expected_shape = pair.clean_future.shape
    for name in (
        "noisy_future",
        "target_velocity",
        "normalized_future",
    ):
        value = getattr(pair, name)
        if value.shape != expected_shape:
            raise ValueError(f"pair.{name} must have shape {expected_shape}")
    if predicted_velocity.shape != expected_shape:
        raise ValueError(f"predicted_velocity must have shape {expected_shape}")
    batch, _frames, latent_dim = expected_shape
    if pair.flow_time.shape != (batch,):
        raise ValueError(f"pair.flow_time must have shape ({batch},)")
    if history.shape != (batch, 32, latent_dim):
        raise ValueError(f"history must have shape ({batch}, 32, {latent_dim})")
    if not torch.isfinite(predicted_velocity).all():
        raise ValueError("predicted_velocity contains non-finite values")
    if not math.isfinite(temporal_weight) or temporal_weight < 0.0:
        raise ValueError("temporal_weight must be finite and non-negative")

    flow = _masked_mean(
        (predicted_velocity.float() - pair.target_velocity).square(),
        mask,
    )
    clean_normalized = (
        pair.noisy_future
        + (1.0 - pair.flow_time[:, None, None]) * predicted_velocity.float()
    )
    mean = statistics.mean.to(
        device=predicted_velocity.device,
        dtype=torch.float32,
    )
    latent_std = statistics.latent_std.to(
        device=predicted_velocity.device,
        dtype=torch.float32,
    )
    clean_estimate = clean_normalized * latent_std + mean
    boundary = _boundary_loss(
        clean_estimate,
        pair.clean_future,
        history,
        mask,
        history_mask,
    )
    statistics_loss = _statistics_loss(
        clean_estimate,
        pair.clean_future,
        mask,
        statistics,
    )
    total = flow + 0.10 * boundary + 0.02 * statistics_loss
    components = {
        "flow": flow,
        "boundary": boundary,
        "statistics": statistics_loss,
    }
    if temporal_weight > 0.0:
        temporal = _temporal_motion_loss(
            clean_estimate,
            pair.clean_future,
            mask,
            statistics.delta_std,
        )
        total = total + temporal_weight * temporal
        components["temporal"] = temporal
    return FlowLossReport(
        total=total,
        components=components,
    )


def distributed_gradient_l2_norm(
    loss: Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    retain_graph: bool = True,
) -> float:
    selected = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not selected:
        raise ValueError("at least one trainable parameter is required")
    gradients = torch.autograd.grad(
        loss,
        selected,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros(
        (),
        device=loss.device,
        dtype=torch.float64,
    )
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().double().square().sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            squared,
            op=torch.distributed.ReduceOp.SUM,
        )
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("gradient norm is not finite")
    return float(norm.item())


class PitchWeightController:
    def __init__(
        self,
        initial: float = 0.30,
        minimum: float = 0.10,
        maximum: float = 1.00,
        target_minimum: float = 0.20,
        target_maximum: float = 0.35,
        ema_decay: float = 0.90,
        warmup_updates: int = 1000,
    ) -> None:
        values = (
            initial,
            minimum,
            maximum,
            target_minimum,
            target_maximum,
            ema_decay,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("controller values must be finite")
        if not 0 < minimum <= initial <= maximum:
            raise ValueError("pitch weight bounds are invalid")
        if not 0 < target_minimum <= target_maximum:
            raise ValueError("gradient ratio targets are invalid")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if warmup_updates <= 0:
            raise ValueError("warmup_updates must be positive")
        self.initial = initial
        self.minimum = minimum
        self.maximum = maximum
        self.target_minimum = target_minimum
        self.target_maximum = target_maximum
        self.ema_decay = ema_decay
        self.warmup_updates = warmup_updates
        self.value = initial
        self.ema_flow_norm = 0.0
        self.ema_pitch_norm = 0.0
        self.ema_initialized = False

    def update(
        self,
        flow_norm: float,
        pitch_norm: float,
        update: int,
    ) -> float:
        if (
            not math.isfinite(flow_norm)
            or not math.isfinite(pitch_norm)
            or flow_norm < 0
            or pitch_norm < 0
        ):
            raise FloatingPointError("gradient norms must be finite")
        if update < 0:
            raise ValueError("update must be non-negative")
        if not self.ema_initialized:
            self.ema_flow_norm = flow_norm
            self.ema_pitch_norm = pitch_norm
            self.ema_initialized = True
        else:
            keep = self.ema_decay
            self.ema_flow_norm = keep * self.ema_flow_norm + (1.0 - keep) * flow_norm
            self.ema_pitch_norm = keep * self.ema_pitch_norm + (1.0 - keep) * pitch_norm
        observed = self.ema_pitch_norm * self.value / max(self.ema_flow_norm, 1.0e-12)
        candidate = self.value
        if observed > self.target_maximum:
            candidate *= self.target_maximum / max(observed, 1.0e-12)
        elif observed < self.target_minimum:
            candidate *= self.target_minimum / max(observed, 1.0e-12)
        candidate = min(self.maximum, max(self.minimum, candidate))
        if update < self.warmup_updates:
            warmup_cap = self.initial * update / self.warmup_updates
            candidate = min(candidate, warmup_cap)
        self.value = candidate
        return self.value

    def distributed_update(
        self,
        flow_norm: float,
        pitch_norm: float,
        update: int,
        *,
        source_rank: int = 0,
    ) -> float:
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not distributed:
            return self.update(flow_norm, pitch_norm, update)
        rank = torch.distributed.get_rank()
        if rank == source_rank:
            self.update(flow_norm, pitch_norm, update)
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.distributed.get_backend() == "nccl"
            else torch.device("cpu")
        )
        state = torch.tensor(
            [
                self.value,
                self.ema_flow_norm,
                self.ema_pitch_norm,
                float(self.ema_initialized),
            ],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.broadcast(state, src=source_rank)
        self.value = float(state[0].item())
        self.ema_flow_norm = float(state[1].item())
        self.ema_pitch_norm = float(state[2].item())
        self.ema_initialized = bool(state[3].item())
        return self.value

    def state_dict(self) -> dict[str, float | bool]:
        return {
            "value": self.value,
            "ema_flow_norm": self.ema_flow_norm,
            "ema_pitch_norm": self.ema_pitch_norm,
            "ema_initialized": self.ema_initialized,
        }

    def load_state_dict(
        self,
        state: dict[str, float | bool],
    ) -> None:
        required = {
            "value",
            "ema_flow_norm",
            "ema_pitch_norm",
            "ema_initialized",
        }
        if set(state) != required:
            raise ValueError("invalid pitch weight controller state")
        value = float(state["value"])
        flow_norm = float(state["ema_flow_norm"])
        pitch_norm = float(state["ema_pitch_norm"])
        initialized = bool(state["ema_initialized"])
        if not all(math.isfinite(item) for item in (value, flow_norm, pitch_norm)):
            raise ValueError("controller state contains non-finite values")
        if not 0.0 <= value <= self.maximum:
            raise ValueError("controller state weight is out of range")
        if flow_norm < 0 or pitch_norm < 0:
            raise ValueError("controller state norm is negative")
        self.value = value
        self.ema_flow_norm = flow_norm
        self.ema_pitch_norm = pitch_norm
        self.ema_initialized = initialized
