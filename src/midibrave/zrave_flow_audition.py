from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor

from .zrave_codec import decode_with_seed
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_exploration import (
    exploration_controls,
    rollout_candidate_metrics,
)
from .zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    sample_midi_sequence_flow_block,
    sample_pure_flow_block,
)


@dataclass(frozen=True)
class ExplorationRolloutResult:
    generated: Tensor
    selected_candidate_indices: Tensor
    candidate_scores: Tensor
    candidate_motion: Tensor
    candidate_boundary_rms: Tensor
    candidate_norm_violation: Tensor


def _candidate_seed(
    generation_seed: int,
    commit_index: int,
    candidate_index: int,
) -> int:
    if generation_seed < 0 or commit_index < 0 or candidate_index < 0:
        raise ValueError("rollout seed indices must be non-negative")
    digest = hashlib.sha256(
        f"{generation_seed}:{commit_index}:{candidate_index}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sample_rollout_candidates(
    model: Any,
    statistics: FlowStatistics,
    history: Tensor,
    *,
    generation_seed: int,
    commit_index: int,
    candidate_count: int,
    temperature: float,
    wander_delay_frames: float,
    solver_steps: int,
    schedule_offset_frames: int,
    visible_history_frames: int,
) -> Tensor:
    candidates = [
        sample_pure_flow_block(
            model,
            statistics,
            history,
            generation_seed=_candidate_seed(
                generation_seed,
                commit_index,
                candidate_index,
            ),
            block_index=0,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            solver_steps=solver_steps,
            schedule_offset_frames=schedule_offset_frames,
            visible_history_frames=visible_history_frames,
        )
        for candidate_index in range(candidate_count)
    ]
    return torch.stack(candidates, dim=1)


def rollout_exploration_flow(
    model: Any,
    statistics: FlowStatistics,
    history: Tensor,
    frames: int,
    *,
    generation_seed: int,
    exploration: float,
    candidate_count: int = 4,
    stride_frames: int = 16,
    solver_steps: int = 8,
    sample_candidates: Callable[..., Tensor] = _sample_rollout_candidates,
) -> ExplorationRolloutResult:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")
    if candidate_count not in {1, 2, 4}:
        raise ValueError("candidate_count must be 1, 2, or 4")
    if stride_frames != 16:
        raise ValueError("exploration rollout stride must be 16 frames")
    if not math.isfinite(exploration) or not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be finite and in [0, 1]")
    expected_history = (
        history.shape[0] if history.ndim else 0,
        int(model.context_frames),
        int(model.latent_dim),
    )
    if history.ndim != 3 or tuple(history.shape) != expected_history:
        raise ValueError(
            f"history must have shape {expected_history}, "
            f"got {tuple(history.shape)}"
        )
    statistics.validate(int(model.latent_dim))
    control = exploration_controls(
        torch.tensor([exploration], device=history.device)
    )
    temperature = float(control.temperature[0].item())
    wander_delay = float(control.wander_delay_frames[0].item())
    visible_history = int(control.visible_history_frames[0].item())
    levels = torch.full(
        (history.shape[0],),
        exploration,
        device=history.device,
        dtype=history.dtype,
    )

    chunks: list[Tensor] = []
    selected_indices: list[Tensor] = []
    score_steps: list[Tensor] = []
    motion_steps: list[Tensor] = []
    boundary_steps: list[Tensor] = []
    norm_steps: list[Tensor] = []
    current = history
    committed = 0
    commit_index = 0
    while committed < frames:
        candidates = sample_candidates(
            model,
            statistics,
            current,
            generation_seed=generation_seed,
            commit_index=commit_index,
            candidate_count=candidate_count,
            temperature=temperature,
            wander_delay_frames=wander_delay,
            solver_steps=solver_steps,
            schedule_offset_frames=committed,
            visible_history_frames=visible_history,
        )
        expected_candidates = (
            history.shape[0],
            candidate_count,
            int(model.future_frames),
            int(model.latent_dim),
        )
        if tuple(candidates.shape) != expected_candidates:
            raise ValueError(
                f"candidate sampler must return {expected_candidates}, "
                f"got {tuple(candidates.shape)}"
            )
        metrics = rollout_candidate_metrics(
            candidates,
            history=current,
            statistics=statistics,
            exploration=levels,
            commit_frames=stride_frames,
        )
        selected = metrics.scores.argmax(dim=1)
        batch_indices = torch.arange(
            history.shape[0],
            device=candidates.device,
        )
        selected_block = candidates[batch_indices, selected]
        take = min(stride_frames, frames - committed)
        chunk = selected_block[:, :take]
        chunks.append(chunk)
        selected_indices.append(selected)
        score_steps.append(metrics.scores)
        motion_steps.append(metrics.motion)
        boundary_steps.append(metrics.boundary_rms)
        norm_steps.append(metrics.norm_violation)
        current = torch.cat((current, chunk), dim=1)[
            :, -int(model.context_frames) :
        ].detach()
        committed += take
        commit_index += 1

    return ExplorationRolloutResult(
        generated=torch.cat(chunks, dim=1),
        selected_candidate_indices=torch.stack(selected_indices, dim=1),
        candidate_scores=torch.stack(score_steps, dim=1),
        candidate_motion=torch.stack(motion_steps, dim=1),
        candidate_boundary_rms=torch.stack(boundary_steps, dim=1),
        candidate_norm_violation=torch.stack(norm_steps, dim=1),
    )


def rollout_pure_flow(
    model: Any,
    statistics: FlowStatistics | object,
    history: Tensor,
    frames: int,
    *,
    generation_seed: int,
    temperature: float,
    wander_delay_frames: int,
    solver_steps: int,
    sample_block: Callable[..., Tensor] = sample_pure_flow_block,
) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    expected_history = (
        history.shape[0] if history.ndim else 0,
        int(model.context_frames),
        int(model.latent_dim),
    )
    if history.ndim != 3 or tuple(history.shape) != expected_history:
        raise ValueError(
            f"history must have shape {expected_history}, "
            f"got {tuple(history.shape)}"
        )
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    block_index = 0
    while remaining:
        generated = sample_block(
            model,
            statistics,
            current,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            solver_steps=solver_steps,
        )
        expected_block = (
            history.shape[0],
            int(model.future_frames),
            int(model.latent_dim),
        )
        if tuple(generated.shape) != expected_block:
            raise ValueError(
                f"flow sampler must return {expected_block}, "
                f"got {tuple(generated.shape)}"
            )
        take = min(remaining, int(model.future_frames))
        chunk = generated[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[
            :, -int(model.context_frames) :
        ].detach()
        remaining -= take
        block_index += 1
    return torch.cat(chunks, dim=1)


def _validate_midi_control_values(
    name: str,
    value: Tensor,
    *,
    note_min: int | None = None,
    note_max: int | None = None,
) -> None:
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if note_min is not None and note_max is not None:
        if value.is_floating_point() and not torch.equal(
            value,
            value.round(),
        ):
            raise ValueError(f"{name} must contain integer MIDI notes")
        if torch.any(value < note_min) or torch.any(value > note_max):
            raise ValueError(
                f"{name} must be in [{note_min}, {note_max}]"
            )
    elif torch.any(value < 0) or torch.any(value > 127):
        raise ValueError(f"{name} must be in [0, 127]")


def _padded_control_block(
    value: Tensor,
    start: int,
    block_frames: int,
) -> Tensor:
    block = value[:, start : start + block_frames]
    if block.shape[1] < block_frames:
        block = torch.cat(
            (
                block,
                block[:, -1:].expand(-1, block_frames - block.shape[1]),
            ),
            dim=1,
        )
    return block


def rollout_midi_sequence_flow(
    model: Any,
    statistics: FlowStatistics | object,
    history: Tensor,
    history_midi_note: Tensor,
    history_velocity: Tensor,
    future_midi_note: Tensor,
    future_velocity: Tensor,
    frames: int,
    *,
    generation_seed: int,
    temperature: float,
    wander_delay_frames: int,
    pitch_guidance: float,
    solver_steps: int,
    history_mask: Tensor | None = None,
    sample_block: Callable[..., Tensor] = sample_midi_sequence_flow_block,
) -> Tensor:
    """Roll out frame-aligned MIDI control without desynchronizing context.

    Future note and velocity tensors cover the complete requested horizon.
    They are sliced into model-sized blocks; a partial final block repeats its
    last requested control only for sampler padding.  Only committed controls
    enter the next history.
    """

    if frames <= 0:
        raise ValueError("frames must be positive")
    if generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")
    if not bool(getattr(model, "midi_sequence_conditioning", False)):
        raise ValueError("model must use MIDI sequence conditioning")
    context_frames = int(model.context_frames)
    future_frames = int(model.future_frames)
    latent_dim = int(model.latent_dim)
    batch = history.shape[0] if history.ndim else 0
    expected_history = (batch, context_frames, latent_dim)
    if history.ndim != 3 or tuple(history.shape) != expected_history:
        raise ValueError(
            f"history must have shape {expected_history}, "
            f"got {tuple(history.shape)}"
        )
    expected_history_control = (batch, context_frames)
    if tuple(history_midi_note.shape) != expected_history_control:
        raise ValueError(
            "history_midi_note must have shape "
            f"{expected_history_control}"
        )
    if tuple(history_velocity.shape) != expected_history_control:
        raise ValueError(
            f"history_velocity must have shape {expected_history_control}"
        )
    expected_future_control = (batch, frames)
    if tuple(future_midi_note.shape) != expected_future_control:
        raise ValueError(
            f"future_midi_note must have shape {expected_future_control}"
        )
    if tuple(future_velocity.shape) != expected_future_control:
        raise ValueError(
            f"future_velocity must have shape {expected_future_control}"
        )
    resolved_history_mask = (
        torch.ones(
            batch,
            context_frames,
            device=history.device,
            dtype=torch.bool,
        )
        if history_mask is None
        else history_mask.to(device=history.device, dtype=torch.bool)
    )
    if tuple(resolved_history_mask.shape) != expected_history_control:
        raise ValueError(
            f"history_mask must have shape {expected_history_control}"
        )
    if not torch.all(resolved_history_mask.any(dim=1)):
        raise ValueError("every sample needs visible MIDI history")
    note_min = int(model.note_min)
    note_max = int(model.note_max)
    _validate_midi_control_values(
        "history_midi_note",
        history_midi_note,
        note_min=note_min,
        note_max=note_max,
    )
    _validate_midi_control_values(
        "future_midi_note",
        future_midi_note,
        note_min=note_min,
        note_max=note_max,
    )
    _validate_midi_control_values("history_velocity", history_velocity)
    _validate_midi_control_values("future_velocity", future_velocity)

    resolved_future_note = future_midi_note.to(device=history.device)
    resolved_future_velocity = future_velocity.to(device=history.device)
    chunks: list[Tensor] = []
    current = history
    current_note = history_midi_note.to(device=history.device)
    current_velocity = history_velocity.to(device=history.device)
    current_mask = resolved_history_mask
    committed = 0
    block_index = 0
    while committed < frames:
        note_block = _padded_control_block(
            resolved_future_note,
            committed,
            future_frames,
        )
        velocity_block = _padded_control_block(
            resolved_future_velocity,
            committed,
            future_frames,
        )
        generated = sample_block(
            model,
            statistics,
            current,
            current_note,
            current_velocity,
            note_block,
            velocity_block,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            pitch_guidance=pitch_guidance,
            solver_steps=solver_steps,
            history_mask=current_mask,
            schedule_offset_frames=committed,
        )
        expected_block = (batch, future_frames, latent_dim)
        if tuple(generated.shape) != expected_block:
            raise ValueError(
                f"MIDI flow sampler must return {expected_block}, "
                f"got {tuple(generated.shape)}"
            )
        take = min(future_frames, frames - committed)
        chunk = generated[:, :take]
        committed_note = note_block[:, :take]
        committed_velocity = velocity_block[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[
            :, -context_frames:
        ].detach()
        current_note = torch.cat(
            (current_note, committed_note),
            dim=1,
        )[:, -context_frames:]
        current_velocity = torch.cat(
            (current_velocity, committed_velocity),
            dim=1,
        )[:, -context_frames:]
        current_mask = torch.cat(
            (
                current_mask,
                torch.ones(
                    batch,
                    take,
                    device=current_mask.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )[:, -context_frames:]
        committed += take
        block_index += 1
    return torch.cat(chunks, dim=1)


def build_midi_audition_control_sequences(
    recorded_notes: Tensor,
    recorded_velocities: Tensor,
    frames: int,
    *,
    note_min: int,
    note_max: int,
    available_notes: Sequence[int] | Tensor | None = None,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Build matched and observed-note-swapped controls for audition rows."""

    if recorded_notes.ndim != 1 or recorded_velocities.shape != (
        recorded_notes.shape[0],
    ):
        raise ValueError("recorded MIDI controls must have shape [batch]")
    if frames <= 0:
        raise ValueError("frames must be positive")
    _validate_midi_control_values(
        "recorded_notes",
        recorded_notes,
        note_min=note_min,
        note_max=note_max,
    )
    _validate_midi_control_values(
        "recorded_velocities",
        recorded_velocities,
    )
    note_pool = torch.as_tensor(
        recorded_notes if available_notes is None else available_notes,
        device=recorded_notes.device,
        dtype=recorded_notes.dtype,
    ).flatten()
    _validate_midi_control_values(
        "available_notes",
        note_pool,
        note_min=note_min,
        note_max=note_max,
    )
    note_pool = torch.unique(note_pool, sorted=True)
    if note_pool.numel() < 2:
        raise ValueError("note swap requires at least two observed notes")
    matches = recorded_notes[:, None] == note_pool[None, :]
    if not torch.all(matches.any(dim=1)):
        raise ValueError("every recorded note must occur in available_notes")
    indices = matches.to(dtype=torch.long).argmax(dim=1)
    swapped = note_pool[(indices + 1) % note_pool.numel()]
    if torch.any(swapped == recorded_notes):
        raise ValueError("note swap must change every requested note")
    matched_notes = recorded_notes[:, None].expand(-1, frames).clone()
    swapped_notes = swapped[:, None].expand(-1, frames).clone()
    velocities = recorded_velocities[:, None].expand(-1, frames).clone()
    return {
        "matched": (matched_notes, velocities.clone()),
        "note_swap": (swapped_notes, velocities.clone()),
    }


def select_audition_rows(
    rows: Iterable[dict[str, Any]],
    *,
    categories: Sequence[str],
    split: str,
    seed: int,
    minimum_active_frames: int,
) -> list[dict[str, Any]]:
    if minimum_active_frames <= 0:
        raise ValueError("minimum_active_frames must be positive")
    available = list(rows)
    selected: list[dict[str, Any]] = []
    for category in categories:
        candidates = [
            row
            for row in available
            if str(row.get("category", "")).casefold()
            == category.casefold()
            and str(row.get("split", "")) == split
            and int(row.get("active_frames", 0))
            >= minimum_active_frames
        ]
        if not candidates:
            raise ValueError(
                f"no {split} {category} row has at least "
                f"{minimum_active_frames} active frames"
            )
        candidates.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:serum128-flow-audition:{category}:"
                    f"{row['sample_id']}"
                ).encode("utf-8")
            ).digest()
        )
        selected.append(candidates[0])
    return selected


def match_rms(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    peak_limit: float = 0.98,
) -> np.ndarray:
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if not np.isfinite(reference).all():
        raise ValueError("reference audio is not finite")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate audio is not finite")
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate))))
    if reference_rms <= 1.0e-12 or candidate_rms <= 1.0e-12:
        return np.zeros_like(candidate, dtype=np.float32)
    matched = candidate * np.float32(reference_rms / candidate_rms)
    peak = float(np.max(np.abs(matched)))
    if peak > peak_limit:
        matched *= np.float32(peak_limit / peak)
    return matched.astype(np.float32, copy=False)


def validate_pure_checkpoint_payload(
    payload: object,
    *,
    latent_dim: int,
    context_frames: int,
    future_frames: int,
    config_sha256: str,
    pack_index_sha256: str,
    statistics_sha256: str,
    exploration_enabled: bool | None = None,
) -> int:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("checkpoint must use format 1")
    expected_architecture = (
        "zrave_pure_flow_transformer_v2"
        if exploration_enabled is True
        else "zrave_pure_flow_transformer_v1"
    )
    if payload.get("architecture") != expected_architecture:
        raise ValueError("checkpoint architecture is not pure flow")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no contract")
    expected: dict[str, object] = {
        "latent_dim": latent_dim,
        "context_frames": context_frames,
        "future_frames": future_frames,
        "pitch_conditioning": False,
        "config_sha256": config_sha256,
        "pack_index_sha256": pack_index_sha256,
        "statistics_sha256": statistics_sha256,
    }
    if exploration_enabled is not None:
        expected["exploration_enabled"] = exploration_enabled
    for name, value in expected.items():
        actual = (
            contract.get(name, False)
            if name == "exploration_enabled"
            else contract.get(name)
        )
        if actual != value:
            raise ValueError(f"checkpoint {name} contract mismatch")
    update = payload.get("update")
    if not isinstance(update, int) or isinstance(update, bool) or update <= 0:
        raise ValueError("checkpoint update must be positive")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model state")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint model state is malformed")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor is non-finite: {name}")
    return update


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIDI_SEQUENCE_ARCHITECTURE = (
    "zrave_midi_sequence_flow_transformer_v2"
)
_PURE_INITIALIZER_ARCHITECTURES = {
    "zrave_pure_flow_transformer_v1",
    "zrave_pure_flow_transformer_v2",
}


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_contract_integer(
    contract: Mapping[str, object],
    name: str,
) -> int:
    value = contract.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"checkpoint {name} must be positive")
    return value


def validate_midi_sequence_checkpoint_payload(
    payload: object,
    *,
    latent_dim: int,
    context_frames: int,
    future_frames: int,
    config_sha256: str,
    pack_index_sha256: str,
    statistics_sha256: str,
    pitch_checkpoint_sha256: str,
    pitch_qualification_sha256: str,
    maximum_updates: int,
    segment_sampling: bool,
    expected_initializer_sha256: str,
    expected_initializer_update: int = 85000,
) -> int:
    """Validate the complete inference contract for a G3 MIDI checkpoint."""

    expected_hashes = {
        "config_sha256": config_sha256,
        "pack_index_sha256": pack_index_sha256,
        "statistics_sha256": statistics_sha256,
        "pitch_checkpoint_sha256": pitch_checkpoint_sha256,
        "pitch_qualification_sha256": pitch_qualification_sha256,
    }
    for name, value in expected_hashes.items():
        _require_sha256(name, value)
    _require_sha256(
        "expected_initializer_sha256",
        expected_initializer_sha256,
    )
    if maximum_updates <= 0:
        raise ValueError("maximum_updates must be positive")
    if expected_initializer_update <= 0:
        raise ValueError("expected_initializer_update must be positive")
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("checkpoint must use format 1")
    if payload.get("architecture") != _MIDI_SEQUENCE_ARCHITECTURE:
        raise ValueError("checkpoint architecture is not MIDI sequence flow")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no contract")
    expected: dict[str, object] = {
        "latent_dim": latent_dim,
        "context_frames": context_frames,
        "future_frames": future_frames,
        "pitch_conditioning": True,
        "midi_sequence_conditioning": True,
        "exploration_enabled": False,
        "segment_sampling": segment_sampling,
        "maximum_updates": maximum_updates,
        **expected_hashes,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"checkpoint {name} contract mismatch")
    for name in expected_hashes:
        _require_sha256(f"checkpoint {name}", contract.get(name))
    _positive_contract_integer(contract, "world_size")
    _positive_contract_integer(contract, "batch_per_gpu")

    update = payload.get("update")
    if (
        not isinstance(update, int)
        or isinstance(update, bool)
        or not 0 < update <= maximum_updates
    ):
        raise ValueError(
            "checkpoint update must be positive and within maximum_updates"
        )
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model state")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint model state is malformed")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor is non-finite: {name}")

    initialization = payload.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("MIDI checkpoint has no initialization lineage")
    source_update = initialization.get("source_update")
    if (
        not isinstance(source_update, int)
        or isinstance(source_update, bool)
        or source_update != expected_initializer_update
    ):
        raise ValueError("checkpoint initializer update mismatch")
    if (
        initialization.get("source_architecture")
        not in _PURE_INITIALIZER_ARCHITECTURES
    ):
        raise ValueError("checkpoint initializer architecture is not pure flow")
    initializer_hash = _require_sha256(
        "checkpoint initializer checkpoint_sha256",
        initialization.get("checkpoint_sha256"),
    )
    if initializer_hash != expected_initializer_sha256:
        raise ValueError("checkpoint initializer SHA-256 mismatch")
    return update


def _validated_pitch_qualification(
    path: str | Path,
    *,
    expected_pack_hash: str,
    latent_dim: int,
    note_min: int,
    note_max: int,
) -> dict[str, Any]:
    """Resolve and verify the exact qualified pitch artifact pair."""

    from .zrave_pitch_train import load_qualified_pitch_probe

    _require_sha256("expected pitch pack hash", expected_pack_hash)
    candidate = Path(path)
    if candidate.is_dir():
        qualification_path = candidate / "qualification.json"
    elif candidate.suffix == ".json":
        qualification_path = candidate
    else:
        qualification_path = candidate.parent.parent / "qualification.json"
    if not qualification_path.is_file():
        raise ValueError("pitch probe qualification is missing")
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    if not isinstance(qualification, dict):
        raise ValueError("pitch qualification must be a mapping")
    if qualification.get("passed") is not True:
        raise ValueError("pitch probe did not pass qualification")
    if qualification.get("pack_index_sha256") != expected_pack_hash:
        raise ValueError("pitch probe pack hash mismatch")
    _require_sha256(
        "pitch qualification config_sha256",
        qualification.get("config_sha256"),
    )
    qualification_update = qualification.get("update")
    if (
        not isinstance(qualification_update, int)
        or isinstance(qualification_update, bool)
        or qualification_update <= 0
    ):
        raise ValueError("pitch qualification update must be positive")
    gates = qualification.get("gates")
    if not isinstance(gates, dict) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise ValueError("pitch qualification gates are incomplete")
    metrics = qualification.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("pitch qualification metrics are missing")
    required_metrics = {
        "accuracy",
        "median_absolute_cents",
        "p90_absolute_cents",
        "finite",
    }
    if required_metrics - set(metrics) or metrics.get("finite") is not True:
        raise ValueError("pitch qualification metrics are incomplete")
    numeric_metrics = (
        metrics["accuracy"],
        metrics["median_absolute_cents"],
        metrics["p90_absolute_cents"],
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_metrics
    ):
        raise ValueError("pitch qualification metrics must be finite")

    checkpoint_value = qualification.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("pitch qualification lacks checkpoint")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = qualification_path.parent / checkpoint_path
    if not checkpoint_path.is_file():
        raise ValueError("qualified pitch checkpoint is missing")
    probe = load_qualified_pitch_probe(
        qualification_path,
        expected_pack_hash,
    )
    if (
        probe.latent_dim != latent_dim
        or probe.note_min != note_min
        or probe.note_max != note_max
    ):
        raise ValueError("qualified pitch probe architecture mismatch")
    return {
        "qualification_path": qualification_path.resolve(),
        "qualification_sha256": _sha256_file(qualification_path),
        "checkpoint_path": checkpoint_path.resolve(),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "update": qualification_update,
        "gates": gates,
        "metrics": metrics,
    }


def _audio_cell(title: str, payload: dict[str, Any]) -> str:
    path = escape(str(payload["matched_wav"]), quote=True)
    label = escape(title)
    return (
        '<div class="take">'
        f'<div class="take__label">{label}</div>'
        f'<audio controls preload="none" src="{path}"></audio>'
        "</div>"
    )


def render_index_html(manifest: dict[str, Any]) -> str:
    checkpoint = manifest["checkpoint"]
    is_midi = manifest.get("conditioning") == "midi_sequence"
    page_title = (
        "Serum128 MIDI-flow audition"
        if is_midi
        else "Serum128 flow audition"
    )
    eyebrow = (
        "MIDI Z-RAVE / matched vs note swap"
        if is_midi
        else "Pure Z-RAVE / listening proof"
    )
    headline = (
        "Does the latent follow control?"
        if is_midi
        else "Does the latent keep moving?"
    )
    description = (
        "Compare matched MIDI control with the same velocity under an "
        "next observed-vocabulary note."
        if is_midi
        else "Compare the source, codec ceiling, then stochastic continuation."
    )
    generated_frames = int(manifest["generated_frames"])
    commit_stride = int(manifest.get("commit_stride_frames", 64))
    commit_count = math.ceil(generated_frames / commit_stride)
    unit_name = "commits" if commit_stride == 16 else "blocks"
    generated_label = (
        f"{commit_count} × {commit_stride} generated {unit_name}"
    )
    cards: list[str] = []
    for example in manifest["examples"]:
        takes = [
            _audio_cell("Source", example["source"]),
            _audio_cell("RAVE Direct", example["direct"]),
        ]
        takes.extend(
            _audio_cell(str(rollout["label"]), rollout)
            for rollout in example["rollouts"]
        )
        cards.append(
            '<section class="instrument">'
            '<header class="instrument__head">'
            f'<h2>{escape(str(example["category"]))}</h2>'
            f'<code>{escape(str(example["sample_id"]))}</code>'
            "</header>"
            '<div class="takes">'
            + "".join(takes)
            + "</div></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(page_title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #edf1f5;
      --ink: #17243a;
      --muted: #657286;
      --rule: #bac5d1;
      --rave: #286aa6;
      --flow: #c56b17;
      --panel: #f8fafc;
      --focus: #704f91;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Segoe UI", sans-serif;
    }}
    main {{ width: min(1480px, calc(100% - 32px)); margin: 0 auto 64px; }}
    .masthead {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(420px, 1.45fr);
      gap: 36px;
      padding: 48px 0 32px;
      border-bottom: 2px solid var(--ink);
    }}
    .eyebrow, code, .take__label {{
      font-family: "Cascadia Mono", Consolas, monospace;
      letter-spacing: .04em;
    }}
    .eyebrow {{ color: var(--flow); font-weight: 700; text-transform: uppercase; }}
    h1, h2 {{ font-family: Bahnschrift, "Arial Narrow", sans-serif; margin: 0; }}
    h1 {{ font-size: clamp(42px, 7vw, 94px); line-height: .86; letter-spacing: -.045em; }}
    .facts {{ align-self: end; color: var(--muted); font-size: 15px; line-height: 1.55; }}
    .facts strong {{ color: var(--ink); }}
    .timeline {{ display: grid; grid-template-columns: 1fr 5fr; gap: 4px; margin-top: 24px; }}
    .timeline span {{ padding: 10px 8px; font: 12px "Cascadia Mono", monospace; text-align: center; }}
    .timeline__seed {{ background: var(--rave); color: white; }}
    .timeline__flow {{ background: var(--flow); color: white; }}
    .legend {{ display: flex; gap: 20px; margin-top: 10px; font: 12px "Cascadia Mono", monospace; }}
    .instrument {{ border-bottom: 1px solid var(--rule); padding: 28px 0 30px; }}
    .instrument__head {{ display: flex; align-items: baseline; gap: 18px; margin-bottom: 18px; }}
    .instrument__head h2 {{ font-size: 30px; }}
    .instrument__head code {{ color: var(--muted); font-size: 11px; }}
    .takes {{ display: grid; grid-template-columns: repeat(3, minmax(250px, 1fr)); gap: 10px; }}
    .take {{ background: var(--panel); border-left: 4px solid var(--rule); padding: 12px 14px 14px; }}
    .take:nth-child(2) {{ border-left-color: var(--rave); }}
    .take:nth-child(n+3) {{ border-left-color: var(--flow); }}
    .take__label {{ min-height: 32px; font-size: 12px; font-weight: 650; }}
    audio {{ width: 100%; height: 36px; }}
    audio:focus-visible {{ outline: 3px solid var(--focus); outline-offset: 3px; }}
    .note {{ margin-top: 18px; color: var(--muted); max-width: 80ch; }}
    @media (max-width: 900px) {{
      .masthead {{ grid-template-columns: 1fr; }}
      .takes {{ grid-template-columns: 1fr; }}
      .timeline span {{ font-size: 9px; padding-inline: 2px; }}
    }}
  </style>
</head>
<body>
<main>
  <header class="masthead">
    <div>
      <div class="eyebrow">{escape(eyebrow)}</div>
      <h1>{escape(headline)}</h1>
    </div>
    <div class="facts">
      <strong>Checkpoint {int(checkpoint['update']):,}</strong><br>
      128D Serum Balanced RAVE · 44.1 kHz · hop 2048<br>
      {escape(description)}
      <div class="timeline">
        <span class="timeline__seed">32 real seed</span>
        <span class="timeline__flow">{generated_frames} generated</span>
      </div>
      <div class="legend"><span>32 real seed</span><span>{generated_label}</span></div>
    </div>
  </header>
  <p class="note">Players use source-RMS-matched files for fair loudness. The package also contains untouched float WAVs under <code>raw/</code>.</p>
  {''.join(cards)}
</main>
</body>
</html>
"""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _load_statistics(path: str | Path) -> FlowStatistics:
    with np.load(path, allow_pickle=False) as values:
        return FlowStatistics(
            mean=torch.from_numpy(values["mean"].copy()),
            latent_std=torch.from_numpy(values["latent_std"].copy()),
            delta_std=torch.from_numpy(values["delta_std"].copy()),
            latent_norm_p01=torch.as_tensor(
                values["latent_norm_p01"].copy()
            ),
            latent_norm_p99=torch.as_tensor(
                values["latent_norm_p99"].copy()
            ),
        )


def _packed_latent(root: Path, row: dict[str, Any]) -> np.ndarray:
    with np.load(root / str(row["shard"]), allow_pickle=False) as values:
        latent = values["latents"][
            int(row["shard_row"]), : int(row["length"])
        ].astype(np.float32)
    if latent.ndim != 2 or not np.isfinite(latent).all():
        raise ValueError(f"invalid packed latent for {row['sample_id']}")
    return latent


def _mono_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, actual_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"invalid source audio: {path}")
    if actual_rate != sample_rate:
        divisor = math.gcd(actual_rate, sample_rate)
        mono = resample_poly(
            mono,
            sample_rate // divisor,
            actual_rate // divisor,
        ).astype(np.float32)
        expected_samples = round(audio.shape[0] * sample_rate / actual_rate)
        mono = mono[:expected_samples]
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite resampled source audio: {path}")
    return np.ascontiguousarray(mono, dtype=np.float32)


def _audio_facts(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("rendered audio must be non-empty and finite")
    return {
        "samples": int(audio.size),
        "seconds": float(audio.size / sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
    }


def _write_audio_pair(
    output: Path,
    stem: str,
    source_reference: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    raw_path = output / "raw" / f"{stem}.wav"
    matched_path = output / "matched" / f"{stem}.wav"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(audio, dtype=np.float32).reshape(-1)
    matched = match_rms(source_reference, raw)
    sf.write(raw_path, raw, sample_rate, subtype="FLOAT")
    sf.write(matched_path, matched, sample_rate, subtype="PCM_16")
    return {
        "raw_wav": raw_path.relative_to(output).as_posix(),
        "matched_wav": matched_path.relative_to(output).as_posix(),
        "raw": _audio_facts(raw, sample_rate),
        "matched": _audio_facts(matched, sample_rate),
    }


def _decode_latent(
    codec: Any,
    latent: np.ndarray | Tensor,
    *,
    device: torch.device,
    random_seed: int,
) -> np.ndarray:
    tensor = torch.as_tensor(
        latent,
        device=device,
        dtype=torch.float32,
    )
    if tensor.ndim != 2:
        raise ValueError("decode latent must be [frames, channels]")
    decoded = decode_with_seed(
        codec,
        tensor.transpose(0, 1).unsqueeze(0).contiguous(),
        random_seed,
    )
    audio = decoded.detach().float().cpu().reshape(-1).numpy()
    if not np.isfinite(audio).all():
        raise ValueError("RAVE decoder produced non-finite audio")
    return audio


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "sample"


def _codec_latent_size(codec: Any) -> int:
    value = getattr(codec, "latent_size")
    if isinstance(value, Tensor):
        return int(value.flatten()[0].item())
    if hasattr(value, "__len__"):
        return int(value[0])
    return int(value)


def render_flow_audition(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    categories: Sequence[str] = (
        "Pad",
        "Lead",
        "Bass",
        "Pluck",
        "Keys",
        "Synth",
    ),
    explorations: Sequence[float] = (0.0, 0.5, 1.0),
    generation_seeds: Sequence[int] = (17, 71),
    candidate_count: int = 4,
    commit_stride_frames: int = 16,
    generated_frames: int = 320,
    selection_seed: int = 20260802,
    device_name: str = "cuda",
) -> dict[str, Any]:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audition output is not empty: {output}")
    if generated_frames <= 0:
        raise ValueError("generated_frames must be positive")
    if not explorations or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in explorations
    ):
        raise ValueError("explorations must be non-empty and in [0, 1]")
    if not generation_seeds or any(value < 0 for value in generation_seeds):
        raise ValueError("generation seeds must be non-empty and non-negative")
    if candidate_count not in {1, 2, 4}:
        raise ValueError("candidate_count must be 1, 2, or 4")
    if commit_stride_frames != 16:
        raise ValueError("commit_stride_frames must be 16")

    config = ZraveFlowConfig.load(config_path)
    if config.model.pitch_conditioning:
        raise ValueError("audition requires a pure flow config")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    update = validate_pure_checkpoint_payload(
        payload,
        latent_dim=config.model.latent_dim,
        context_frames=config.model.context_frames,
        future_frames=config.model.future_frames,
        config_sha256=_sha256_file(config_path),
        pack_index_sha256=_sha256_file(index_path),
        statistics_sha256=_sha256_file(statistics_path),
        exploration_enabled=config.exploration.enabled,
    )
    checkpoint_architecture = str(payload["architecture"])
    initialization = payload.get("initialization")
    if initialization is not None and not isinstance(initialization, dict):
        raise ValueError("checkpoint initialization lineage is malformed")
    checkpoint_hash = _sha256_file(checkpoint_path)
    statistics = _load_statistics(statistics_path)
    statistics.validate(config.model.latent_dim)
    device = torch.device(device_name)
    model = ZraveFlowTransformer(config.model, statistics).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    del payload

    codec_path = Path(config.rave.checkpoint)
    codec_hash = _sha256_file(codec_path)
    if codec_hash != config.rave.expected_sha256:
        raise ValueError("RAVE codec hash mismatch")
    codec = torch.jit.load(str(codec_path), map_location=device).eval()
    if _codec_latent_size(codec) != config.model.latent_dim:
        raise ValueError("RAVE codec latent dimension mismatch")

    sequence_rows = _read_jsonl(packed_root / "sequences.jsonl")
    selected = select_audition_rows(
        sequence_rows,
        categories=categories,
        split="test",
        seed=selection_seed,
        minimum_active_frames=config.model.context_frames,
    )
    manifest_rows = {
        str(row["sample_id"]): row
        for row in _read_jsonl(config.data.unified_manifest)
    }
    latents: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sample_id = str(row["sample_id"])
            source_row = manifest_rows.get(sample_id)
            if source_row is None:
                raise ValueError(f"manifest lacks {sample_id}")
            latent = _packed_latent(packed_root, row)
            if latent.shape[1] != config.model.latent_dim:
                raise ValueError(f"latent dimension mismatch for {sample_id}")
            source = _mono_audio(
                str(source_row["audio_path"]),
                config.rave.sample_rate,
            )
            direct = _decode_latent(
                codec,
                latent,
                device=device,
                random_seed=config.seed + index,
            )[: source.size]
            stem = f"{index:02d}-{_slug(str(row['category']))}"
            examples.append(
                {
                    "category": str(row["category"]),
                    "sample_id": sample_id,
                    "canonical_preset_id": str(
                        row["canonical_preset_id"]
                    ),
                    "midi_note": int(row["midi_note"]),
                    "velocity": int(row["velocity"]),
                    "source": _write_audio_pair(
                        output,
                        f"{stem}-source",
                        source,
                        source,
                        config.rave.sample_rate,
                    ),
                    "direct": _write_audio_pair(
                        output,
                        f"{stem}-rave-direct",
                        source,
                        direct,
                        config.rave.sample_rate,
                    ),
                    "rollouts": [],
                }
            )
            latents.append(latent)
            sources.append(source)

        history = torch.from_numpy(
            np.stack(
                [latent[: config.model.context_frames] for latent in latents]
            )
        ).to(device)
        for exploration_index, exploration in enumerate(explorations):
            for generation_seed in generation_seeds:
                rollout = rollout_exploration_flow(
                    model,
                    statistics,
                    history,
                    generated_frames,
                    generation_seed=generation_seed,
                    exploration=float(exploration),
                    candidate_count=candidate_count,
                    stride_frames=commit_stride_frames,
                    solver_steps=config.model.solver_steps,
                )
                complete = torch.cat((history, rollout.generated), dim=1)
                for index, example in enumerate(examples):
                    audio = _decode_latent(
                        codec,
                        complete[index],
                        device=device,
                        # Decoder randomness must stay fixed within an example;
                        # cross-seed diversity belongs to the predictor latent.
                        random_seed=config.seed + index,
                    )
                    exploration_name = str(float(exploration)).replace(
                        ".", "p"
                    )
                    stem = (
                        f"{index:02d}-{_slug(str(example['category']))}"
                        f"-flow-e{exploration_name}-s{generation_seed}"
                    )
                    rendered = _write_audio_pair(
                        output,
                        stem,
                        sources[index],
                        audio,
                        config.rave.sample_rate,
                    )
                    rendered.update(
                        {
                            "label": (
                                f"Flow · E{float(exploration):g} · "
                                f"Best of {candidate_count} · "
                                f"Seed {generation_seed}"
                            ),
                            "exploration": float(exploration),
                            "generation_seed": int(generation_seed),
                            "candidate_count": candidate_count,
                            "commit_stride_frames": commit_stride_frames,
                            "selected_candidate_indices": (
                                rollout.selected_candidate_indices[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ),
                            "rejection_metrics": {
                                "candidate_scores": (
                                    rollout.candidate_scores[index]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist()
                                ),
                                "motion": (
                                    rollout.candidate_motion[index]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist()
                                ),
                                "boundary_rms": (
                                    rollout.candidate_boundary_rms[index]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist()
                                ),
                                "norm_violation": (
                                    rollout.candidate_norm_violation[index]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .tolist()
                                ),
                            },
                        }
                    )
                    example["rollouts"].append(rendered)

    manifest: dict[str, Any] = {
        "schema": 1,
        "title": "Serum128 pure-flow audition",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "update": update,
            "sha256": checkpoint_hash,
            "architecture": checkpoint_architecture,
            "initialization": initialization,
        },
        "codec": {
            "path": str(codec_path.resolve()),
            "sha256": codec_hash,
            "latent_dim": config.model.latent_dim,
        },
        "sample_rate": config.rave.sample_rate,
        "latent_hop": config.rave.latent_hop,
        "context_frames": config.model.context_frames,
        "generated_frames": generated_frames,
        "commit_stride_frames": commit_stride_frames,
        "solver_steps": config.model.solver_steps,
        "explorations": [float(value) for value in explorations],
        "candidate_count": candidate_count,
        "generation_seeds": [int(value) for value in generation_seeds],
        "selection_seed": selection_seed,
        "examples": examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        render_index_html(manifest),
        encoding="utf-8",
    )
    return manifest


def render_midi_flow_audition(
    config_path: str | Path,
    checkpoint_path: str | Path,
    pitch_qualification_path: str | Path,
    output_root: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_initializer_sha256: str,
    expected_initializer_update: int = 85000,
    categories: Sequence[str] = (
        "Pad",
        "Lead",
        "Bass",
        "Pluck",
        "Keys",
        "Synth",
    ),
    generation_seeds: Sequence[int] = (17, 71),
    generated_frames: int = 320,
    temperature: float = 1.0,
    wander_delay_frames: int = 32,
    pitch_guidance: float | None = None,
    expected_note_vocabulary: Sequence[int] = (36, 62, 82),
    selection_seed: int = 20260812,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Render paired matched/note-swap MIDI sequence continuations."""

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audition output is not empty: {output}")
    _require_sha256(
        "expected_checkpoint_sha256",
        expected_checkpoint_sha256,
    )
    _require_sha256(
        "expected_initializer_sha256",
        expected_initializer_sha256,
    )
    if generated_frames <= 0:
        raise ValueError("generated_frames must be positive")
    if (
        len(generation_seeds) < 2
        or len(set(generation_seeds)) != len(generation_seeds)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in generation_seeds
        )
    ):
        raise ValueError(
            "generation seeds must contain at least two unique, "
            "non-negative integers"
        )
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be non-empty and unique")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if wander_delay_frames not in {16, 32, 48}:
        raise ValueError("wander_delay_frames must be 16, 32, or 48")
    if (
        len(expected_note_vocabulary) < 2
        or tuple(sorted(set(expected_note_vocabulary)))
        != tuple(expected_note_vocabulary)
    ):
        raise ValueError(
            "expected_note_vocabulary must be sorted, unique, and non-trivial"
        )

    config = ZraveFlowConfig.load(config_path)
    if config.model.latent_dim != 128:
        raise ValueError("MIDI audition requires a 128D config")
    if not config.model.pitch_conditioning:
        raise ValueError("MIDI audition requires pitch conditioning")
    if not config.model.midi_sequence_conditioning:
        raise ValueError("MIDI audition requires sequence conditioning")
    if config.exploration.enabled:
        raise ValueError("MIDI audition does not accept exploration models")
    if not config.segment_sampling.enabled:
        raise ValueError("MIDI audition requires segment sampling")
    resolved_pitch_guidance = (
        config.model.pitch_guidance
        if pitch_guidance is None
        else float(pitch_guidance)
    )
    if (
        not math.isfinite(resolved_pitch_guidance)
        or not 1.0 <= resolved_pitch_guidance <= 5.0
    ):
        raise ValueError("pitch_guidance must be finite and in [1, 5]")

    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    config_hash = _sha256_file(config_path)
    pack_hash = _sha256_file(index_path)
    statistics_hash = _sha256_file(statistics_path)
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError("MIDI flow checkpoint SHA-256 mismatch")
    pitch_artifacts = _validated_pitch_qualification(
        pitch_qualification_path,
        expected_pack_hash=pack_hash,
        latent_dim=config.model.latent_dim,
        note_min=config.model.note_min,
        note_max=config.model.note_max,
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    update = validate_midi_sequence_checkpoint_payload(
        payload,
        latent_dim=config.model.latent_dim,
        context_frames=config.model.context_frames,
        future_frames=config.model.future_frames,
        config_sha256=config_hash,
        pack_index_sha256=pack_hash,
        statistics_sha256=statistics_hash,
        pitch_checkpoint_sha256=str(
            pitch_artifacts["checkpoint_sha256"]
        ),
        pitch_qualification_sha256=str(
            pitch_artifacts["qualification_sha256"]
        ),
        maximum_updates=config.train.max_updates,
        segment_sampling=config.segment_sampling.enabled,
        expected_initializer_sha256=expected_initializer_sha256,
        expected_initializer_update=expected_initializer_update,
    )
    checkpoint_architecture = str(payload["architecture"])
    checkpoint_contract = dict(payload["contract"])
    initialization = dict(payload["initialization"])

    statistics = _load_statistics(statistics_path)
    statistics.validate(config.model.latent_dim)
    device = torch.device(device_name)
    model = ZraveFlowTransformer(config.model, statistics).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    del payload

    codec_path = Path(config.rave.checkpoint)
    codec_hash = _sha256_file(codec_path)
    if codec_hash != config.rave.expected_sha256:
        raise ValueError("RAVE codec hash mismatch")
    codec = torch.jit.load(str(codec_path), map_location=device).eval()
    if _codec_latent_size(codec) != config.model.latent_dim:
        raise ValueError("RAVE codec latent dimension mismatch")

    sequence_rows = _read_jsonl(packed_root / "sequences.jsonl")
    observed_note_vocabulary = tuple(
        sorted({int(row["midi_note"]) for row in sequence_rows})
    )
    if observed_note_vocabulary != tuple(expected_note_vocabulary):
        raise ValueError(
            "pack MIDI note vocabulary mismatch: expected "
            f"{tuple(expected_note_vocabulary)}, got "
            f"{observed_note_vocabulary}"
        )
    selected = select_audition_rows(
        sequence_rows,
        categories=categories,
        split="test",
        seed=selection_seed,
        minimum_active_frames=config.model.context_frames,
    )
    manifest_rows = {
        str(row["sample_id"]): row
        for row in _read_jsonl(config.data.unified_manifest)
    }
    latents: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sample_id = str(row["sample_id"])
            source_row = manifest_rows.get(sample_id)
            if source_row is None:
                raise ValueError(f"manifest lacks {sample_id}")
            latent = _packed_latent(packed_root, row)
            if latent.shape[1] != config.model.latent_dim:
                raise ValueError(f"latent dimension mismatch for {sample_id}")
            if latent.shape[0] < config.model.context_frames:
                raise ValueError(f"latent history is too short for {sample_id}")
            source = _mono_audio(
                str(source_row["audio_path"]),
                config.rave.sample_rate,
            )
            decoder_seed = config.seed + index
            direct = _decode_latent(
                codec,
                latent,
                device=device,
                random_seed=decoder_seed,
            )[: source.size]
            stem = f"{index:02d}-{_slug(str(row['category']))}"
            direct_render = _write_audio_pair(
                output,
                f"{stem}-rave-direct",
                source,
                direct,
                config.rave.sample_rate,
            )
            direct_render["decoder_seed"] = decoder_seed
            recorded_note = int(row["midi_note"])
            recorded_velocity = int(row["velocity"])
            examples.append(
                {
                    "category": str(row["category"]),
                    "split": str(row["split"]),
                    "sample_id": sample_id,
                    "canonical_preset_id": str(
                        row["canonical_preset_id"]
                    ),
                    "midi_note": recorded_note,
                    "velocity": recorded_velocity,
                    "history_requested_notes": [
                        recorded_note
                    ] * config.model.context_frames,
                    "history_requested_velocities": [
                        recorded_velocity
                    ] * config.model.context_frames,
                    "decoder_seed": decoder_seed,
                    "source": _write_audio_pair(
                        output,
                        f"{stem}-source",
                        source,
                        source,
                        config.rave.sample_rate,
                    ),
                    "direct": direct_render,
                    "rollouts": [],
                }
            )
            latents.append(latent)
            sources.append(source)

        history = torch.from_numpy(
            np.stack(
                [latent[: config.model.context_frames] for latent in latents]
            )
        ).to(device=device, dtype=torch.float32)
        recorded_notes = torch.tensor(
            [example["midi_note"] for example in examples],
            device=device,
            dtype=torch.long,
        )
        recorded_velocities = torch.tensor(
            [example["velocity"] for example in examples],
            device=device,
            dtype=torch.long,
        )
        history_notes = recorded_notes[:, None].expand(
            -1,
            config.model.context_frames,
        ).clone()
        history_velocities = recorded_velocities[:, None].expand(
            -1,
            config.model.context_frames,
        ).clone()
        controls = build_midi_audition_control_sequences(
            recorded_notes,
            recorded_velocities,
            generated_frames,
            note_min=config.model.note_min,
            note_max=config.model.note_max,
            available_notes=observed_note_vocabulary,
        )
        for condition_kind, (
            requested_notes,
            requested_velocities,
        ) in controls.items():
            for generation_seed in generation_seeds:
                generated = rollout_midi_sequence_flow(
                    model,
                    statistics,
                    history,
                    history_notes,
                    history_velocities,
                    requested_notes,
                    requested_velocities,
                    generated_frames,
                    generation_seed=generation_seed,
                    temperature=temperature,
                    wander_delay_frames=wander_delay_frames,
                    pitch_guidance=resolved_pitch_guidance,
                    solver_steps=config.model.solver_steps,
                )
                complete = torch.cat((history, generated), dim=1)
                for index, example in enumerate(examples):
                    decoder_seed = int(example["decoder_seed"])
                    audio = _decode_latent(
                        codec,
                        complete[index],
                        device=device,
                        # Paired predictor seeds and control variants share
                        # one decoder RNG per source example.
                        random_seed=decoder_seed,
                    )
                    condition_note = int(requested_notes[index, 0].item())
                    condition_velocity = int(
                        requested_velocities[index, 0].item()
                    )
                    stem = (
                        f"{index:02d}-{_slug(str(example['category']))}"
                        f"-midi-{condition_kind}-n{condition_note}"
                        f"-v{condition_velocity}-s{generation_seed}"
                    )
                    rendered = _write_audio_pair(
                        output,
                        stem,
                        sources[index],
                        audio,
                        config.rave.sample_rate,
                    )
                    rendered.update(
                        {
                            "label": (
                                f"MIDI {condition_kind.replace('_', ' ')} "
                                f"· N{condition_note} V{condition_velocity} "
                                f"· Seed {generation_seed}"
                            ),
                            "condition_kind": condition_kind,
                            "condition_midi_note": condition_note,
                            "condition_velocity": condition_velocity,
                            "requested_notes": (
                                requested_notes[index].detach().cpu().tolist()
                            ),
                            "requested_velocities": (
                                requested_velocities[index]
                                .detach()
                                .cpu()
                                .tolist()
                            ),
                            "generation_seed": int(generation_seed),
                            "decoder_seed": decoder_seed,
                            "temperature": float(temperature),
                            "wander_delay_frames": wander_delay_frames,
                            "pitch_guidance": resolved_pitch_guidance,
                            "commit_stride_frames": (
                                config.model.future_frames
                            ),
                        }
                    )
                    example["rollouts"].append(rendered)

    manifest: dict[str, Any] = {
        "schema": 1,
        "title": "Serum128 MIDI sequence-flow audition",
        "conditioning": "midi_sequence",
        "latent_dim": config.model.latent_dim,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "update": update,
            "sha256": checkpoint_hash,
            "architecture": checkpoint_architecture,
            "contract": checkpoint_contract,
            "initialization": initialization,
        },
        "pitch_probe": {
            "checkpoint_path": str(pitch_artifacts["checkpoint_path"]),
            "checkpoint_sha256": pitch_artifacts["checkpoint_sha256"],
            "qualification_path": str(
                pitch_artifacts["qualification_path"]
            ),
            "qualification_sha256": pitch_artifacts[
                "qualification_sha256"
            ],
            "qualification_update": pitch_artifacts["update"],
            "gates": pitch_artifacts["gates"],
            "metrics": pitch_artifacts["metrics"],
        },
        "codec": {
            "path": str(codec_path.resolve()),
            "sha256": codec_hash,
            "latent_dim": config.model.latent_dim,
            "decoder_seed_policy": "fixed_per_example",
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_hash,
        },
        "pack_index_sha256": pack_hash,
        "statistics_sha256": statistics_hash,
        "sample_rate": config.rave.sample_rate,
        "latent_hop": config.rave.latent_hop,
        "context_frames": config.model.context_frames,
        "history_frames": config.model.context_frames,
        "future_frames": config.model.future_frames,
        "generated_frames": generated_frames,
        "commit_stride_frames": config.model.future_frames,
        "solver_steps": config.model.solver_steps,
        "temperature": float(temperature),
        "wander_delay_frames": wander_delay_frames,
        "pitch_guidance": resolved_pitch_guidance,
        "generation_seeds": [int(value) for value in generation_seeds],
        "selection_seed": selection_seed,
        "selected_split": "test",
        "selected_categories": list(categories),
        "controls": {
            "matched": "recorded note and recorded velocity",
            "note_vocabulary": list(observed_note_vocabulary),
            "note_swap_policy": "next observed note in sorted cyclic order",
            "note_swap_mapping": {
                str(note): observed_note_vocabulary[
                    (index + 1) % len(observed_note_vocabulary)
                ]
                for index, note in enumerate(observed_note_vocabulary)
            },
        },
        "examples": examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        render_index_html(manifest),
        encoding="utf-8",
    )
    return manifest
