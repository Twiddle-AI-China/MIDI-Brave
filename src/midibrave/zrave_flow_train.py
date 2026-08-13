from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .zrave_flow_config import (
    FLOW_MODEL_PROFILES,
    FlowExplorationConfig,
    ZraveFlowConfig,
    data_selection_sha256,
)
from .zrave_flow_evaluate import (
    GateEarlyStopState,
    evaluate_flow_checkpoint,
)
from .zrave_flow_exploration import (
    exploration_controls,
    trailing_history_mask,
)
from .zrave_flow_loss import (
    FlowLossReport,
    PitchWeightController,
    distributed_gradient_l2_norm,
    make_flow_training_pair,
    zrave_flow_loss,
    zrave_pure_flow_loss,
)
from .zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    retention_curve,
    sample_flow_block,
    sample_pure_flow_block,
)
from .zrave_flow_sampler import FlowBatch, GpuFlowSampler
from .zrave_pitch_probe import LatentPitchProbe
from .zrave_pitch_train import load_qualified_pitch_probe


class StatefulSampler(Protocol):
    def state_dict(self) -> dict[str, Tensor]: ...

    def load_state_dict(self, state: dict[str, Tensor]) -> None: ...


def maximum_valid_future(update: int, short_updates: int) -> int:
    if update < 0:
        raise ValueError("update must be non-negative")
    if short_updates <= 0:
        raise ValueError("short_updates must be positive")
    return 32 if update < short_updates else 64


def use_exposure_batch(
    update: int,
    maximum_updates: int,
    start_fraction: float,
    probability: float,
    generator: torch.Generator,
    *,
    start_update: int | None = None,
) -> bool:
    if update < 0 or maximum_updates <= 0:
        raise ValueError("training updates are invalid")
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be in [0, 1]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    threshold = (
        math.ceil(maximum_updates * start_fraction)
        if start_update is None
        else start_update
    )
    if not 0 <= threshold < maximum_updates:
        raise ValueError("exposure start is outside the training schedule")
    if update < threshold:
        return False
    draw = torch.rand(
        (),
        generator=generator,
        device=generator.device,
    )
    return float(draw.item()) < probability


def resolve_phase3_start_update(
    *,
    maximum_updates: int,
    default_fraction: float,
    resumed_update: int,
    requested: int | None,
    restored: int | None,
) -> int:
    if maximum_updates <= 0 or resumed_update < 0:
        raise ValueError("phase3 training schedule is invalid")
    default = math.ceil(maximum_updates * default_fraction)
    if restored is not None:
        if requested is not None and requested != restored:
            raise ValueError("phase3 start conflicts with checkpoint")
        selected = restored
    elif requested is not None:
        if requested != resumed_update:
            raise ValueError("legacy checkpoint phase3 start must match resume update")
        selected = requested
    else:
        selected = default
    if not 0 <= selected < maximum_updates:
        raise ValueError("phase3 start is outside the training schedule")
    return selected


def roll_exposure_history(
    history: Tensor,
    generated_prefix: Tensor,
) -> Tensor:
    if history.ndim != 3 or generated_prefix.ndim != 3:
        raise ValueError("history and generated_prefix must be rank three")
    if (
        history.shape[0] != generated_prefix.shape[0]
        or history.shape[2] != generated_prefix.shape[2]
    ):
        raise ValueError("exposure tensors have incompatible shapes")
    if generated_prefix.shape[1] <= 0:
        raise ValueError("generated_prefix must not be empty")
    context_frames = history.shape[1]
    return torch.cat(
        (history, generated_prefix.detach()),
        dim=1,
    )[:, -context_frames:].detach()


def allowed_exploration_exposure_depth(
    update: int,
    config: FlowExplorationConfig,
) -> int:
    if update < 0:
        raise ValueError("update must be non-negative")
    if not config.enabled or update < config.exposure_start_update:
        return 0
    elapsed = update - config.exposure_start_update
    depth = 1 + (elapsed * config.exposure_max_depth // config.exposure_ramp_updates)
    return min(config.exposure_max_depth, depth)


def exploration_exposure_depth(
    update: int,
    config: FlowExplorationConfig,
    generator: torch.Generator,
) -> int:
    allowed = allowed_exploration_exposure_depth(update, config)
    if allowed == 0:
        return 0
    elapsed = update - config.exposure_start_update
    ramp = min(
        1.0,
        max(0.0, elapsed / config.exposure_ramp_updates),
    )
    probability = config.exposure_probability * ramp
    draw = torch.rand(
        (),
        generator=generator,
        device=generator.device,
    )
    if float(draw.item()) >= probability:
        return 0
    return int(
        torch.randint(
            1,
            allowed + 1,
            (),
            generator=generator,
            device=generator.device,
        ).item()
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _ensure_final_checkpoint(
    checkpoint_root: Path,
    maximum_update: int,
) -> Path:
    """Idempotently repair the final alias after an allocation crash."""

    if maximum_update <= 0:
        raise ValueError("maximum checkpoint update must be positive")
    maximum_checkpoint = checkpoint_root / f"step-{maximum_update:06d}.pt"
    if not maximum_checkpoint.is_file():
        raise RuntimeError(
            "maximum-update checkpoint is missing at training exit: "
            f"{maximum_checkpoint}"
        )
    final_checkpoint = checkpoint_root / "final.pt"
    if not final_checkpoint.is_file() or _sha256_file(final_checkpoint) != _sha256_file(
        maximum_checkpoint
    ):
        _atomic_copy(maximum_checkpoint, final_checkpoint)
    return final_checkpoint


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def _unwrapped(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


_BASE_CONTRACT_HASHES = {
    "config_sha256",
    "pack_index_sha256",
    "statistics_sha256",
}
_PITCH_CONTRACT_HASHES = {
    "pitch_checkpoint_sha256",
    "pitch_qualification_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_PROFILE_FIELDS = (
    "d_model",
    "context_layers",
    "future_layers",
    "heads",
    "feedforward_dim",
)
_MODEL_PROFILE_CONTRACT_FIELDS = {"model_profile", *_MODEL_PROFILE_FIELDS}
_DATA_SELECTION_CONTRACT_FIELD = "data_selection_sha256"


def _resolved_contract_model_profile(
    contract: dict[str, object],
) -> tuple[str, tuple[int, int, int, int, int]]:
    """Resolve legacy contracts as standard and reject partial profiles."""

    present = _MODEL_PROFILE_CONTRACT_FIELDS & set(contract)
    if not present:
        return "standard", FLOW_MODEL_PROFILES["standard"]
    missing = _MODEL_PROFILE_CONTRACT_FIELDS - set(contract)
    if missing:
        raise ValueError(
            "flow checkpoint model profile fields missing: "
            + ", ".join(sorted(missing))
        )
    profile = contract["model_profile"]
    if not isinstance(profile, str) or profile not in FLOW_MODEL_PROFILES:
        raise ValueError("model_profile must be standard, small, or tiny")
    raw_dimensions = tuple(contract[name] for name in _MODEL_PROFILE_FIELDS)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in raw_dimensions
    ):
        raise ValueError("model profile dimensions must be integers")
    dimensions = tuple(int(value) for value in raw_dimensions)
    expected = FLOW_MODEL_PROFILES[profile]
    if dimensions != expected:
        raise ValueError(
            f"flow checkpoint model profile {profile} dimensions "
            f"must be {expected}, got {dimensions}"
        )
    return profile, dimensions


def _validate_checkpoint_contract(contract: dict[str, object]) -> None:
    _resolved_contract_model_profile(contract)
    pitch_conditioning = contract.get("pitch_conditioning", True)
    if not isinstance(pitch_conditioning, bool):
        raise ValueError("pitch_conditioning must be boolean")
    exploration_enabled = contract.get("exploration_enabled", False)
    if not isinstance(exploration_enabled, bool):
        raise ValueError("exploration_enabled must be boolean")
    midi_sequence_conditioning = contract.get("midi_sequence_conditioning", False)
    if not isinstance(midi_sequence_conditioning, bool):
        raise ValueError("midi_sequence_conditioning must be boolean")
    segment_sampling = contract.get("segment_sampling", False)
    if not isinstance(segment_sampling, bool):
        raise ValueError("segment_sampling must be boolean")
    if midi_sequence_conditioning and not pitch_conditioning:
        raise ValueError("MIDI sequence conditioning requires pitch conditioning")
    if pitch_conditioning and exploration_enabled:
        raise ValueError("MIDI conditioning cannot enable exploration v2 yet")
    required = _BASE_CONTRACT_HASHES | {"world_size", "batch_per_gpu"}
    if pitch_conditioning:
        required |= _PITCH_CONTRACT_HASHES
    missing = required - set(contract)
    if missing:
        raise ValueError(
            "flow checkpoint contract missing: " + ", ".join(sorted(missing))
        )
    if not pitch_conditioning:
        unexpected = _PITCH_CONTRACT_HASHES & set(contract)
        if unexpected:
            raise ValueError("pure flow checkpoint contract contains pitch hashes")
    for name in _BASE_CONTRACT_HASHES | (
        _PITCH_CONTRACT_HASHES if pitch_conditioning else set()
    ):
        if not _SHA256.fullmatch(str(contract[name])):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    selection_hash = contract.get(_DATA_SELECTION_CONTRACT_FIELD)
    if selection_hash is not None and not _SHA256.fullmatch(str(selection_hash)):
        raise ValueError("data_selection_sha256 must be a lowercase SHA-256")
    if int(contract["world_size"]) <= 0:
        raise ValueError("world_size must be positive")
    if int(contract["batch_per_gpu"]) <= 0:
        raise ValueError("batch_per_gpu must be positive")


def build_flow_checkpoint_contract(
    *,
    config: ZraveFlowConfig,
    world_size: int,
    batch_per_gpu: int,
    maximum_updates: int,
    config_sha256: str,
    pack_index_sha256: str,
    statistics_sha256: str,
    pitch_checkpoint_sha256: str | None = None,
    pitch_qualification_sha256: str | None = None,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "config_sha256": config_sha256,
        "pack_index_sha256": pack_index_sha256,
        "statistics_sha256": statistics_sha256,
        "world_size": world_size,
        "batch_per_gpu": batch_per_gpu,
        "maximum_updates": maximum_updates,
        "latent_dim": config.model.latent_dim,
        "context_frames": config.model.context_frames,
        "future_frames": config.model.future_frames,
        "model_profile": config.model.profile,
        "d_model": config.model.d_model,
        "context_layers": config.model.context_layers,
        "future_layers": config.model.future_layers,
        "heads": config.model.heads,
        "feedforward_dim": config.model.feedforward_dim,
        "pitch_conditioning": config.model.pitch_conditioning,
        "midi_sequence_conditioning": (config.model.midi_sequence_conditioning),
        "exploration_enabled": config.exploration.enabled,
        "segment_sampling": config.segment_sampling.enabled,
    }
    selection_sha256 = data_selection_sha256(config)
    if selection_sha256 is not None:
        contract[_DATA_SELECTION_CONTRACT_FIELD] = selection_sha256
    pitch_hashes = (
        pitch_checkpoint_sha256,
        pitch_qualification_sha256,
    )
    if config.model.pitch_conditioning:
        if any(value is None for value in pitch_hashes):
            raise ValueError("pitch-conditioned flow requires both pitch hashes")
        contract["pitch_checkpoint_sha256"] = pitch_checkpoint_sha256
        contract["pitch_qualification_sha256"] = pitch_qualification_sha256
    elif any(value is not None for value in pitch_hashes):
        raise ValueError("pure flow contract must not contain pitch hashes")
    _validate_checkpoint_contract(contract)
    return contract


def _checkpoint_architecture(contract: dict[str, object]) -> str:
    if contract.get("midi_sequence_conditioning", False):
        return "zrave_midi_sequence_flow_transformer_v2"
    if contract.get("pitch_conditioning", True):
        return "zrave_conditional_flow_transformer_v1"
    if contract.get("exploration_enabled", False):
        return "zrave_pure_flow_transformer_v2"
    return "zrave_pure_flow_transformer_v1"


def _validated_initialization(
    value: object,
) -> dict[str, int | str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("checkpoint initialization must be a mapping")
    update = value.get("source_update")
    architecture = value.get("source_architecture")
    digest = value.get("checkpoint_sha256")
    if not isinstance(update, int) or isinstance(update, bool) or update < 0:
        raise ValueError("checkpoint initialization update is invalid")
    if architecture not in {
        "zrave_pure_flow_transformer_v1",
        "zrave_pure_flow_transformer_v2",
    }:
        raise ValueError("checkpoint initialization architecture is invalid")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("checkpoint initialization hash is invalid")
    return {
        "source_update": update,
        "source_architecture": architecture,
        "checkpoint_sha256": digest,
    }


def _require_expected_initialization(
    initialization: dict[str, int | str] | None,
    *,
    expected_initializer_sha256: str | None,
    expected_initializer_update: int | None,
    context: str,
) -> dict[str, int | str] | None:
    expected_values = (
        expected_initializer_sha256,
        expected_initializer_update,
    )
    if all(value is None for value in expected_values):
        return initialization
    if any(value is None for value in expected_values):
        raise ValueError(
            "expected initializer SHA-256 and update must be provided together"
        )
    assert expected_initializer_sha256 is not None
    assert expected_initializer_update is not None
    if not isinstance(expected_initializer_sha256, str) or not _SHA256.fullmatch(
        expected_initializer_sha256
    ):
        raise ValueError("expected initializer SHA-256 must be a lowercase SHA-256")
    if (
        not isinstance(expected_initializer_update, int)
        or isinstance(expected_initializer_update, bool)
        or expected_initializer_update <= 0
    ):
        raise ValueError("expected initializer update must be positive")
    if initialization is None:
        raise ValueError(f"{context} lacks initialization lineage")
    if initialization["checkpoint_sha256"] != expected_initializer_sha256:
        raise ValueError(f"{context} initializer SHA-256 mismatch")
    if initialization["source_update"] != expected_initializer_update:
        raise ValueError(f"{context} initializer update mismatch")
    return initialization


def save_flow_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: StatefulSampler,
    pitch_weight_controller: PitchWeightController | None,
    update: int,
    contract: dict[str, object],
    phase3_start_update: int | None = None,
    latest_gate_report_sha256: str | None = None,
    consecutive_gate_passes: int = 0,
    checkpoint_process_group: dist.ProcessGroup | None = None,
    initialization: dict[str, int | str] | None = None,
) -> None:
    _validate_checkpoint_contract(contract)
    if update < 0 or consecutive_gate_passes < 0:
        raise ValueError("checkpoint counters must be non-negative")
    if phase3_start_update is not None and phase3_start_update < 0:
        raise ValueError("phase3 start update must be non-negative")
    if latest_gate_report_sha256 is not None and not _SHA256.fullmatch(
        latest_gate_report_sha256
    ):
        raise ValueError("latest gate report hash is invalid")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if int(contract["world_size"]) != world_size:
        raise ValueError("checkpoint contract world size mismatch")
    local_sampler = sampler.state_dict()
    local_rng = _capture_rng_state()
    if dist.is_initialized():
        if checkpoint_process_group is None:
            raise ValueError("distributed checkpoint requires a CPU process group")
        sampler_states: list[object] | None = [None] * world_size if rank == 0 else None
        rng_states: list[object] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(
            local_sampler,
            sampler_states,
            dst=0,
            group=checkpoint_process_group,
        )
        dist.gather_object(
            local_rng,
            rng_states,
            dst=0,
            group=checkpoint_process_group,
        )
        if rank != 0:
            return
        assert sampler_states is not None and rng_states is not None
    else:
        sampler_states = [local_sampler]
        rng_states = [local_rng]
    payload: dict[str, object] = {
        "format": 1,
        "architecture": _checkpoint_architecture(contract),
        "model": _unwrapped(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler_by_rank": sampler_states,
        "rng_by_rank": rng_states,
        "update": int(update),
        "contract": dict(contract),
        "latest_gate_report_sha256": latest_gate_report_sha256,
        "consecutive_gate_passes": int(consecutive_gate_passes),
    }
    resolved_initialization = _validated_initialization(initialization)
    if resolved_initialization is not None:
        payload["initialization"] = resolved_initialization
    if phase3_start_update is not None:
        payload["phase3_start_update"] = int(phase3_start_update)
    if contract.get("pitch_conditioning", True):
        if pitch_weight_controller is None:
            raise ValueError("conditional checkpoint requires pitch controller")
        payload["pitch_weight_controller"] = pitch_weight_controller.state_dict()
    _atomic_torch_save(Path(path), payload)


def load_flow_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: StatefulSampler,
    pitch_weight_controller: PitchWeightController | None,
    expected_contract: dict[str, object],
    expected_initializer_sha256: str | None = None,
    expected_initializer_update: int | None = None,
) -> dict[str, object]:
    _validate_checkpoint_contract(expected_contract)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("flow checkpoint must be a mapping")
    if payload.get("format") != 1:
        raise ValueError("unsupported flow checkpoint format")
    if payload.get("architecture") != _checkpoint_architecture(expected_contract):
        raise ValueError("flow checkpoint architecture mismatch")
    actual_contract = payload.get("contract")
    if not isinstance(actual_contract, dict):
        raise ValueError("flow checkpoint has no contract")
    expected_profile = _resolved_contract_model_profile(expected_contract)
    actual_profile = _resolved_contract_model_profile(actual_contract)
    if actual_profile != expected_profile:
        raise ValueError(
            "flow checkpoint model profile mismatch: "
            f"expected {expected_profile}, got {actual_profile}"
        )
    if actual_contract.get(_DATA_SELECTION_CONTRACT_FIELD) != (
        expected_contract.get(_DATA_SELECTION_CONTRACT_FIELD)
    ):
        raise ValueError("flow checkpoint data_selection_sha256 mismatch")
    for name, expected in expected_contract.items():
        if name in _MODEL_PROFILE_CONTRACT_FIELDS:
            continue
        if (
            name
            in {
                "exploration_enabled",
                "midi_sequence_conditioning",
                "segment_sampling",
            }
            and expected is False
            and name not in actual_contract
        ):
            continue
        if actual_contract.get(name) != expected:
            raise ValueError(f"flow checkpoint {name} mismatch")
    initialization = _require_expected_initialization(
        _validated_initialization(payload.get("initialization")),
        expected_initializer_sha256=expected_initializer_sha256,
        expected_initializer_update=expected_initializer_update,
        context="flow checkpoint",
    )
    _unwrapped(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler_state = payload.get("scaler")
        if not isinstance(scaler_state, dict):
            raise ValueError("flow checkpoint is missing scaler state")
        scaler.load_state_dict(scaler_state)
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler_states = payload.get("sampler_by_rank")
    rng_states = payload.get("rng_by_rank")
    if (
        not isinstance(sampler_states, list)
        or rank >= len(sampler_states)
        or not isinstance(rng_states, list)
        or rank >= len(rng_states)
    ):
        raise ValueError("flow checkpoint lacks rank resume state")
    sampler.load_state_dict(sampler_states[rank])
    _restore_rng_state(rng_states[rank])
    if expected_contract.get("pitch_conditioning", True):
        if pitch_weight_controller is None:
            raise ValueError("conditional checkpoint requires pitch controller")
        controller_state = payload.get("pitch_weight_controller")
        if not isinstance(controller_state, dict):
            raise ValueError("flow checkpoint lacks pitch controller state")
        pitch_weight_controller.load_state_dict(controller_state)
    gate_hash = payload.get("latest_gate_report_sha256")
    if gate_hash is not None and (
        not isinstance(gate_hash, str) or not _SHA256.fullmatch(gate_hash)
    ):
        raise ValueError("flow checkpoint gate report hash is invalid")
    phase3_start_update = payload.get("phase3_start_update")
    if phase3_start_update is not None and (
        not isinstance(phase3_start_update, int)
        or isinstance(phase3_start_update, bool)
        or phase3_start_update < 0
    ):
        raise ValueError("flow checkpoint phase3 start is invalid")
    return {
        "update": int(payload["update"]),
        "phase3_start_update": phase3_start_update,
        "latest_gate_report_sha256": gate_hash,
        "consecutive_gate_passes": int(payload.get("consecutive_gate_passes", 0)),
        "initialization": initialization,
    }


def load_flow_initial_weights(
    path: str | Path,
    *,
    model: nn.Module,
    expected_contract: dict[str, object],
    expected_initializer_sha256: str | None = None,
    expected_initializer_update: int | None = None,
) -> dict[str, int | str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("initializer must be a format-1 flow checkpoint")
    architecture = payload.get("architecture")
    if architecture not in {
        "zrave_pure_flow_transformer_v1",
        "zrave_pure_flow_transformer_v2",
    }:
        raise ValueError("initializer must be a pure flow checkpoint")
    actual_contract = payload.get("contract")
    if not isinstance(actual_contract, dict):
        raise ValueError("initializer has no checkpoint contract")
    source_profile = _resolved_contract_model_profile(actual_contract)
    target_profile = _resolved_contract_model_profile(expected_contract)
    if source_profile != target_profile:
        raise ValueError(
            "initializer model profile mismatch: "
            f"source {source_profile}, target {target_profile}"
        )
    expected_data_selection = expected_contract.get(_DATA_SELECTION_CONTRACT_FIELD)
    if expected_data_selection is not None and (
        actual_contract.get(_DATA_SELECTION_CONTRACT_FIELD) != expected_data_selection
    ):
        raise ValueError("initializer data_selection_sha256 mismatch")
    required = (
        "pack_index_sha256",
        "statistics_sha256",
        "latent_dim",
        "context_frames",
        "future_frames",
        "pitch_conditioning",
    )
    for name in required:
        if name == "pitch_conditioning":
            continue
        if actual_contract.get(name) != expected_contract.get(name):
            raise ValueError(f"initializer {name} mismatch")
    if actual_contract.get("pitch_conditioning") is not False:
        raise ValueError("initializer must disable pitch conditioning")
    if (
        not expected_contract.get("midi_sequence_conditioning", False)
        and expected_contract.get("pitch_conditioning") is not False
    ):
        raise ValueError("only MIDI sequence models may initialize from pure flow")
    update = payload.get("update")
    if not isinstance(update, int) or isinstance(update, bool) or update < 0:
        raise ValueError("initializer update is invalid")
    initialization = {
        "source_update": update,
        "source_architecture": str(architecture),
        "checkpoint_sha256": _sha256_file(path),
    }
    _require_expected_initialization(
        initialization,
        expected_initializer_sha256=expected_initializer_sha256,
        expected_initializer_update=expected_initializer_update,
        context="initializer checkpoint",
    )
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("initializer has no model state")
    target = _unwrapped(model)
    target_state = target.state_dict()
    mismatched_shapes = sorted(
        name
        for name, value in state.items()
        if name in target_state
        and (not isinstance(value, Tensor) or value.shape != target_state[name].shape)
    )
    if mismatched_shapes:
        raise ValueError(
            "initializer shared parameter shape mismatch: "
            + ", ".join(mismatched_shapes)
        )
    if expected_contract.get(
        "midi_sequence_conditioning", False
    ) or expected_contract.get("segment_sampling", False):
        shared = {
            name: value
            for name, value in state.items()
            if name in target_state and target_state[name].shape == value.shape
        }
        missing, unexpected = target.load_state_dict(shared, strict=False)
        if unexpected:
            raise ValueError(
                "MIDI initializer has unexpected parameters: " + ", ".join(unexpected)
            )
        allowed_prefixes = ("null_memory", "midi_")
        invalid_missing = [
            name for name in missing if not name.startswith(allowed_prefixes)
        ]
        if invalid_missing:
            raise ValueError(
                "MIDI initializer is missing shared parameters: "
                + ", ".join(invalid_missing)
            )
    else:
        target.load_state_dict(state, strict=True)
    return initialization


def _load_statistics(root: Path) -> FlowStatistics:
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
        return FlowStatistics(
            mean=torch.from_numpy(values["mean"].copy()),
            latent_std=torch.from_numpy(values["latent_std"].copy()),
            delta_std=torch.from_numpy(values["delta_std"].copy()),
            latent_norm_p01=torch.as_tensor(values["latent_norm_p01"].copy()),
            latent_norm_p99=torch.as_tensor(values["latent_norm_p99"].copy()),
        )


def _shared_validation_sampler(
    train_sampler: GpuFlowSampler,
    *,
    seed: int,
) -> GpuFlowSampler:
    """Share immutable resident tensors while isolating validation state."""

    return GpuFlowSampler(
        latents=train_sampler.latents,
        lengths=train_sampler.lengths,
        active_frames=train_sampler.active_frames,
        notes=train_sampler.notes,
        velocities=train_sampler.velocities,
        split_codes=train_sampler.split_codes,
        source_codes=train_sampler.source_codes,
        category_codes=train_sampler.category_codes,
        maximum_future_frames=train_sampler.maximum_future_frames,
        pitch_pairs=train_sampler.pitch_pairs,
        record_eligible=train_sampler.record_eligible,
        source_weights=train_sampler.source_weights,
        context_frames=train_sampler.context_frames,
        future_frames=train_sampler.future_frames,
        wander_delays=train_sampler.wander_delays,
        pitch_transition_fraction=train_sampler.pitch_transition_fraction,
        seed=seed,
        device=train_sampler.device,
        allowed_split_code=1,
        segment_sampling_enabled=train_sampler.segment_sampling_enabled,
        segment_divisions=train_sampler.segment_divisions,
        segment_include_first=train_sampler.segment_include_first,
    )


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Z-RAVE flow training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _new_checkpoint_process_group(
    world_size: int,
) -> dist.ProcessGroup | None:
    return dist.new_group(backend="gloo") if world_size > 1 else None


def _seed_everything(seed: int, rank: int) -> None:
    selected = seed + rank
    random.seed(selected)
    np.random.seed(selected)
    torch.manual_seed(selected)
    torch.cuda.manual_seed(selected)


def _default_generator(device: torch.device) -> torch.Generator:
    if device.type == "cuda":
        assert device.index is not None
        return torch.cuda.default_generators[device.index]
    return torch.default_generator


def _learning_rate(
    config: ZraveFlowConfig,
    update: int,
    maximum_updates: int,
) -> float:
    peak = config.optimizer.learning_rate
    floor = config.optimizer.minimum_learning_rate
    warmup = config.train.warmup_updates
    if update <= warmup:
        return peak * update / warmup
    progress = min(
        1.0,
        (update - warmup) / max(1, maximum_updates - warmup),
    )
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    value: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _future_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    flow = _unwrapped(model)
    prefixes = (
        "future_projection.",
        "future_position",
        "time_embedding.",
        "midi_embedding.",
        "midi_sequence_conditioner.",
        "future_layers.",
        "output_norm.",
        "velocity_projection.",
    )
    selected = tuple(
        parameter
        for name, parameter in flow.named_parameters()
        if name.startswith(prefixes) and parameter.requires_grad
    )
    if not selected:
        raise ValueError("flow model has no future-backbone parameters")
    return selected


def _effective_pitch_weight(
    controller: PitchWeightController,
    update: int,
) -> float:
    if update < controller.warmup_updates:
        return min(
            controller.value,
            controller.initial * update / controller.warmup_updates,
        )
    return controller.value


def _replace_pitch_weight(
    report: FlowLossReport,
    pitch_weight: float,
) -> FlowLossReport:
    total = (
        report.components["flow"]
        + pitch_weight * report.components["pitch"]
        + 0.10 * report.components["boundary"]
        + 0.02 * report.components["statistics"]
    )
    return FlowLossReport(total=total, components=report.components)


def _prepare_exposure_batch(
    model: ZraveFlowTransformer,
    batch: FlowBatch,
    *,
    generation_seed: int,
    block_index: int,
) -> FlowBatch:
    if model.pitch_conditioning:
        generated = sample_flow_block(
            model,
            model.statistics(),
            batch.history,
            batch.midi_note,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=1.0,
            wander_delay_frames=32,
            pitch_guidance=3.0,
            solver_steps=4,
        )[:, :32]
    else:
        generated = sample_pure_flow_block(
            model,
            model.statistics(),
            batch.history,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=1.0,
            wander_delay_frames=32,
            solver_steps=4,
        )[:, :32]
    rolled_history = roll_exposure_history(batch.history, generated)
    shifted_future = torch.zeros_like(batch.future)
    shifted_future[:, :32] = batch.future[:, 32:64]
    shifted_mask = torch.zeros_like(batch.future_mask)
    shifted_mask[:, :32] = batch.future_mask[:, 32:64]
    if not torch.all(shifted_mask[:, :32]):
        raise ValueError("exposure batch lacks a complete tail target")
    return FlowBatch(
        history=rolled_history,
        future=shifted_future,
        future_mask=shifted_mask,
        midi_note=batch.midi_note,
        source_code=batch.source_code,
        category_code=batch.category_code,
        wander_delay_frames=batch.wander_delay_frames,
        history_midi_note=batch.midi_note,
        pitch_transition_mask=batch.pitch_transition_mask,
        velocity=batch.velocity,
        history_velocity=batch.velocity,
    )


def _prepare_exploration_exposure_batch(
    model: ZraveFlowTransformer,
    batch: FlowBatch,
    *,
    generation_seed: int,
    block_index: int,
    depth: int,
    stride_frames: int,
    exploration: float,
    schedule_offset_frames: int,
) -> FlowBatch:
    if model.pitch_conditioning:
        raise ValueError("exploration exposure requires pure flow")
    if not 1 <= depth <= 3:
        raise ValueError("exploration exposure depth must be in [1, 3]")
    if stride_frames != 16:
        raise ValueError("exploration exposure stride must be 16")
    if schedule_offset_frames < 0:
        raise ValueError("schedule offset must be non-negative")
    controls = exploration_controls(
        torch.tensor([exploration], device=batch.history.device)
    )
    temperature = float(controls.temperature.item())
    wander_delay = float(controls.wander_delay_frames.item())
    visible_history = int(controls.visible_history_frames.item())
    rolled_history = batch.history
    for step in range(depth):
        generated = sample_pure_flow_block(
            model,
            model.statistics(),
            rolled_history,
            generation_seed=generation_seed,
            block_index=block_index + step,
            temperature=temperature,
            wander_delay_frames=wander_delay,
            solver_steps=4,
            schedule_offset_frames=(schedule_offset_frames + step * stride_frames),
            visible_history_frames=visible_history,
        )[:, :stride_frames]
        rolled_history = roll_exposure_history(
            rolled_history,
            generated,
        )
    shift = depth * stride_frames
    valid = batch.future.shape[1] - shift
    shifted_future = torch.zeros_like(batch.future)
    shifted_future[:, :valid] = batch.future[:, shift:]
    shifted_mask = torch.zeros_like(batch.future_mask)
    shifted_mask[:, :valid] = batch.future_mask[:, shift:]
    if not torch.all(shifted_mask[:, :valid]):
        raise ValueError("exploration exposure batch lacks a complete shifted target")
    return FlowBatch(
        history=rolled_history,
        future=shifted_future,
        future_mask=shifted_mask,
        midi_note=batch.midi_note,
        source_code=batch.source_code,
        category_code=batch.category_code,
        wander_delay_frames=batch.wander_delay_frames,
        history_midi_note=batch.history_midi_note,
        pitch_transition_mask=batch.pitch_transition_mask,
        velocity=batch.velocity,
        history_velocity=batch.history_velocity,
    )


def _pitch_diagnostics(
    pair: Any,
    predicted_velocity: Tensor,
    future_mask: Tensor,
    midi_note: Tensor,
    transition_mask: Tensor,
    pitch_probe: LatentPitchProbe,
    statistics: FlowStatistics,
) -> dict[str, float]:
    with torch.no_grad():
        normalized = (
            pair.noisy_future
            + (1.0 - pair.flow_time[:, None, None])
            * predicted_velocity.detach().float()
        )
        estimate = normalized * statistics.latent_std.to(
            normalized.device
        ) + statistics.mean.to(normalized.device)
        windows: list[Tensor] = []
        diagnostic_notes: list[Tensor] = []
        note_sequence = (
            midi_note[:, None].expand(-1, estimate.shape[1])
            if midi_note.ndim == 1
            else midi_note
        )
        if note_sequence.shape != estimate.shape[:2]:
            raise ValueError("pitch diagnostic MIDI sequence shape mismatch")
        for sample in range(estimate.shape[0]):
            valid_frames = int(future_mask[sample].sum().item())
            if valid_frames <= 0:
                raise ValueError("pitch diagnostic target is empty")
            sample_notes = note_sequence[sample, :valid_frames]
            changes = torch.nonzero(
                sample_notes[1:] != sample_notes[:-1],
                as_tuple=False,
            ).flatten()
            boundaries = [
                0,
                *(int(value.item()) + 1 for value in changes),
                valid_frames,
            ]
            runs = list(zip(boundaries[:-1], boundaries[1:], strict=True))
            start, end = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
            valid = estimate[sample, start : min(end, start + 16)]
            if valid.shape[0] < 16:
                valid = torch.cat(
                    (valid, valid[-1:].expand(16 - valid.shape[0], -1)),
                    dim=0,
                )
            windows.append(valid)
            diagnostic_notes.append(sample_notes[start])
        output = pitch_probe(torch.stack(windows))
        target_notes = torch.stack(diagnostic_notes)
        predicted = output.logits.argmax(dim=-1) + pitch_probe.note_min
        cents = (output.expected_midi.float() - target_notes.float()).abs() * 100.0
        values = torch.zeros(6, device=estimate.device, dtype=torch.float64)
        for offset, mask in (
            (0, ~transition_mask),
            (3, transition_mask),
        ):
            values[offset] = mask.sum()
            values[offset + 1] = ((predicted == target_notes) & mask).sum()
            values[offset + 2] = cents.masked_select(mask).double().sum()
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        matched_count = max(1.0, float(values[0]))
        transition_count = max(1.0, float(values[3]))
        return {
            "pitch_matched_accuracy": float(values[1]) / matched_count,
            "pitch_matched_mean_absolute_cents": (float(values[2]) / matched_count),
            "pitch_transition_accuracy": (float(values[4]) / transition_count),
            "pitch_transition_mean_absolute_cents": (
                float(values[5]) / transition_count
            ),
        }


_NEAR_STATIC_NORMALIZED_DELTA_RMS = 0.10
_NEAR_ZERO_NORMALIZED_COORDINATE = 0.10
_SEGMENT_DIVISION_BINS = (2, 4, 8)
_SEGMENT_INDEX_BINS = tuple(range(8))
_VELOCITY_BIN_WIDTH = 16


def _distributed_sum(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _distributed_values(value: Tensor, *, name: str) -> Tensor:
    """Gather a small diagnostic vector exactly across all DDP ranks."""

    flattened = value.detach().reshape(-1).float()
    if not torch.isfinite(flattened).all():
        raise FloatingPointError(f"non-finite {name} diagnostic")
    if not dist.is_initialized():
        return flattened
    world_size = dist.get_world_size()
    local_count = torch.tensor(
        [flattened.numel()],
        device=flattened.device,
        dtype=torch.long,
    )
    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    sizes = [int(count.item()) for count in counts]
    maximum = max(sizes)
    padded = torch.zeros(
        maximum,
        device=flattened.device,
        dtype=flattened.dtype,
    )
    padded[: flattened.numel()] = flattened
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes, strict=True)])


def _moments(value: Tensor) -> Tensor:
    flattened = value.detach().reshape(-1).float()
    if not torch.isfinite(flattened).all():
        raise FloatingPointError("diagnostic moment input is non-finite")
    return torch.stack(
        (
            torch.tensor(
                float(flattened.numel()),
                device=flattened.device,
                dtype=torch.float64,
            ),
            flattened.sum(dtype=torch.float64),
            flattened.square().sum(dtype=torch.float64),
        )
    )


def _mean_std(summary: Tensor) -> tuple[float, float]:
    count, total, total_square = (float(value) for value in summary)
    if count == 0.0:
        return 0.0, 0.0
    if count < 0.0:
        raise ValueError("diagnostic summary count must be non-negative")
    mean = total / count
    variance = max(0.0, total_square / count - mean * mean)
    return mean, math.sqrt(variance)


def _quantiles(
    value: Tensor,
    probabilities: tuple[float, ...],
    *,
    name: str,
) -> tuple[float, ...]:
    global_count = _distributed_sum(
        torch.tensor(
            [value.numel()],
            device=value.device,
            dtype=torch.long,
        )
    )
    if int(global_count.item()) == 0:
        return tuple(0.0 for _ in probabilities)
    gathered = _distributed_values(value, name=name)
    requested = torch.tensor(
        probabilities,
        device=gathered.device,
        dtype=gathered.dtype,
    )
    result = torch.quantile(gathered, requested)
    if not torch.isfinite(result).all():
        raise FloatingPointError(f"non-finite {name} quantile")
    return tuple(float(item) for item in result)


def _integer_tensor(value: Tensor, *, name: str) -> Tensor:
    if value.is_floating_point() and (
        not torch.isfinite(value).all() or not torch.equal(value, value.round())
    ):
        raise ValueError(f"{name} must contain finite integers")
    return value.to(dtype=torch.long)


def _categorical_histogram(value: Tensor, *, name: str) -> Tensor:
    codes = _integer_tensor(value.detach(), name=name).reshape(-1)
    if codes.numel() == 0 or torch.any(codes < 0):
        raise ValueError(f"{name} must contain non-negative codes")
    maximum = codes.max().to(dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    histogram = torch.bincount(
        codes,
        minlength=int(maximum.item()) + 1,
    ).to(dtype=torch.float64)
    return _distributed_sum(histogram)


@torch.no_grad()
def _flow_batch_diagnostics(
    *,
    batch: FlowBatch,
    predicted_velocity: Tensor,
    estimated_future: Tensor,
    statistics: FlowStatistics,
    pitch_conditioning: bool,
    midi_sequence_conditioning: bool,
    note_min: int,
    note_max: int,
    pitch_present: Tensor | None,
) -> dict[str, float]:
    """Return low-cost, mask-aware, globally reduced update diagnostics."""

    future = batch.future.detach().float()
    prediction = predicted_velocity.detach().float()
    estimate = estimated_future.detach().float()
    mask = batch.future_mask.detach().to(device=future.device, dtype=torch.bool)
    if (
        future.ndim != 3
        or prediction.shape != future.shape
        or estimate.shape != future.shape
    ):
        raise ValueError("flow diagnostics require matching rank-three tensors")
    if mask.shape != future.shape[:2]:
        raise ValueError("flow diagnostic future mask shape mismatch")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("flow diagnostics require a valid future per sample")
    expanded_mask = mask.unsqueeze(-1).expand_as(future)
    target_values = future.masked_select(expanded_mask)
    predicted_values = prediction.masked_select(expanded_mask)
    target_norms = torch.linalg.vector_norm(future, dim=-1).masked_select(mask)
    predicted_norms = torch.linalg.vector_norm(prediction, dim=-1).masked_select(mask)
    delta_mask = mask[:, 1:] & mask[:, :-1]
    delta_scale = statistics.delta_std.to(
        device=future.device,
        dtype=future.dtype,
    )
    if (
        delta_scale.shape != (future.shape[-1],)
        or not torch.isfinite(delta_scale).all()
        or not torch.all(delta_scale > 0)
    ):
        raise ValueError("flow diagnostic delta_std is invalid")
    normalized_delta = (future[:, 1:] - future[:, :-1]) / delta_scale.view(1, 1, -1)
    motion = normalized_delta.square().mean(dim=-1).sqrt().masked_select(delta_mask)
    valid_lengths = mask.sum(dim=1).float()
    latent_mean = statistics.mean.to(
        device=future.device,
        dtype=future.dtype,
    )
    latent_std = statistics.latent_std.to(
        device=future.device,
        dtype=future.dtype,
    )
    if (
        latent_mean.shape != (future.shape[-1],)
        or latent_std.shape != (future.shape[-1],)
        or not torch.isfinite(latent_mean).all()
        or not torch.isfinite(latent_std).all()
        or not torch.all(latent_std > 0)
    ):
        raise ValueError("flow diagnostic latent statistics are invalid")
    normalized_coordinates = (future - latent_mean) / latent_std
    valid_normalized = normalized_coordinates.masked_select(expanded_mask)
    near_zero_coordinates = valid_normalized.abs() <= _NEAR_ZERO_NORMALIZED_COORDINATE
    weights = mask.to(dtype=future.dtype).unsqueeze(-1)
    frame_count = weights.sum()
    channel_sum = (future * weights).sum(dim=(0, 1))
    channel_sum_square = (future.square() * weights).sum(dim=(0, 1))
    channel_moments = _distributed_sum(
        torch.cat(
            (
                frame_count.reshape(1).to(dtype=torch.float64),
                channel_sum.to(dtype=torch.float64),
                channel_sum_square.to(dtype=torch.float64),
            )
        )
    )
    channel_count = float(channel_moments[0])
    channel_mean = channel_moments[1 : 1 + future.shape[-1]] / channel_count
    channel_variance = (
        channel_moments[1 + future.shape[-1] :] / channel_count - channel_mean.square()
    ).clamp_min(0.0)
    observed_reference_std_ratio = channel_variance.sqrt() / latent_std.to(
        dtype=torch.float64
    )
    ratio_p10, ratio_p50, ratio_p90 = tuple(
        float(value)
        for value in torch.quantile(
            observed_reference_std_ratio,
            torch.tensor(
                [0.10, 0.50, 0.90],
                device=future.device,
                dtype=torch.float64,
            ),
        )
    )

    estimated_values = estimate.masked_select(expanded_mask)
    estimated_norms = torch.linalg.vector_norm(estimate, dim=-1).masked_select(mask)
    estimated_normalized_delta = (
        estimate[:, 1:] - estimate[:, :-1]
    ) / delta_scale.view(1, 1, -1)
    estimated_motion = (
        estimated_normalized_delta.square()
        .mean(dim=-1)
        .sqrt()
        .masked_select(delta_mask)
    )
    estimated_normalized_coordinates = (estimate - latent_mean) / latent_std
    estimated_near_zero = (
        estimated_normalized_coordinates.masked_select(expanded_mask).abs()
        <= _NEAR_ZERO_NORMALIZED_COORDINATE
    )
    estimated_channel_sum = (estimate * weights).sum(dim=(0, 1))
    estimated_channel_sum_square = (estimate.square() * weights).sum(dim=(0, 1))
    estimated_channel_moments = _distributed_sum(
        torch.cat(
            (
                frame_count.reshape(1).to(dtype=torch.float64),
                estimated_channel_sum.to(dtype=torch.float64),
                estimated_channel_sum_square.to(dtype=torch.float64),
            )
        )
    )
    estimated_channel_count = float(estimated_channel_moments[0])
    estimated_channel_mean = (
        estimated_channel_moments[1 : 1 + future.shape[-1]] / estimated_channel_count
    )
    estimated_channel_variance = (
        estimated_channel_moments[1 + future.shape[-1] :] / estimated_channel_count
        - estimated_channel_mean.square()
    ).clamp_min(0.0)
    estimated_std_ratio = estimated_channel_variance.sqrt() / latent_std.to(
        dtype=torch.float64
    )
    estimate_ratio_p10, estimate_ratio_p50, estimate_ratio_p90 = tuple(
        float(value)
        for value in torch.quantile(
            estimated_std_ratio,
            torch.tensor(
                [0.10, 0.50, 0.90],
                device=future.device,
                dtype=torch.float64,
            ),
        )
    )
    estimated_summary = _distributed_sum(
        torch.cat(
            (
                _moments(estimated_values),
                _moments(estimated_norms),
                _moments(estimated_motion),
                torch.as_tensor(
                    [
                        float(
                            (estimated_motion < _NEAR_STATIC_NORMALIZED_DELTA_RMS)
                            .sum()
                            .item()
                        ),
                        float(estimated_near_zero.sum()),
                        float(estimated_near_zero.numel()),
                    ],
                    device=future.device,
                    dtype=torch.float64,
                ),
            )
        )
    )
    estimated_mean, estimated_std = _mean_std(estimated_summary[0:3])
    estimated_norm_mean, estimated_norm_std = _mean_std(estimated_summary[3:6])
    estimated_motion_mean, _estimated_motion_std = _mean_std(estimated_summary[6:9])
    estimated_motion_count = float(estimated_summary[6])
    estimated_near_static_fraction = (
        float(estimated_summary[9]) / estimated_motion_count
        if estimated_motion_count
        else 0.0
    )
    estimated_near_zero_fraction = float(estimated_summary[10]) / float(
        estimated_summary[11]
    )
    estimate_norm_p01, estimate_norm_p50, estimate_norm_p99 = _quantiles(
        estimated_norms,
        (0.01, 0.50, 0.99),
        name="estimated clean latent frame norm",
    )
    estimate_motion_p50, estimate_motion_p95 = _quantiles(
        estimated_motion,
        (0.50, 0.95),
        name="normalized estimated clean frame delta",
    )

    local_summary = torch.cat(
        (
            _moments(target_values),
            _moments(target_norms),
            _moments(motion),
            torch.as_tensor(
                [float((motion < _NEAR_STATIC_NORMALIZED_DELTA_RMS).sum().item())],
                device=future.device,
                dtype=torch.float64,
            ),
            _moments(predicted_values),
            _moments(predicted_norms),
            torch.as_tensor(
                [
                    float(mask.sum()),
                    float(mask.numel()),
                    float(near_zero_coordinates.sum()),
                    float(near_zero_coordinates.numel()),
                ],
                device=future.device,
                dtype=torch.float64,
            ),
        )
    )
    summary = _distributed_sum(local_summary)
    target_mean, target_std = _mean_std(summary[0:3])
    target_norm_mean, target_norm_std = _mean_std(summary[3:6])
    motion_mean, _motion_std = _mean_std(summary[6:9])
    motion_count = float(summary[6])
    near_static_fraction = float(summary[9]) / motion_count if motion_count else 0.0
    predicted_mean, predicted_std = _mean_std(summary[10:13])
    predicted_norm_mean, predicted_norm_std = _mean_std(summary[13:16])
    valid_frames = float(summary[16])
    available_frames = float(summary[17])
    near_zero_fraction = float(summary[18]) / float(summary[19])
    target_norm_p01, target_norm_p50, target_norm_p99 = _quantiles(
        target_norms,
        (0.01, 0.50, 0.99),
        name="target latent frame norm",
    )
    motion_p50, motion_p95 = _quantiles(
        motion,
        (0.50, 0.95),
        name="normalized target frame delta",
    )
    length_mean, _length_std = _mean_std(_distributed_sum(_moments(valid_lengths)))
    length_p50, length_p95 = _quantiles(
        valid_lengths,
        (0.50, 0.95),
        name="valid future length",
    )
    metrics = {
        "batch/future_valid_fraction": valid_frames / available_frames,
        "batch/effective_future_frames_mean": length_mean,
        "batch/effective_future_frames_p50": length_p50,
        "batch/effective_future_frames_p95": length_p95,
        "target/latent_mean": target_mean,
        "target/latent_std": target_std,
        "target/latent_frame_norm_mean": target_norm_mean,
        "target/latent_frame_norm_std": target_norm_std,
        "target/latent_frame_norm_p01": target_norm_p01,
        "target/latent_frame_norm_p50": target_norm_p50,
        "target/latent_frame_norm_p99": target_norm_p99,
        "target/motion_normalized_delta_rms_mean": motion_mean,
        "target/motion_normalized_delta_rms_p50": motion_p50,
        "target/motion_normalized_delta_rms_p95": motion_p95,
        "target/near_static_fraction": near_static_fraction,
        "target/near_static_normalized_delta_rms_threshold": (
            _NEAR_STATIC_NORMALIZED_DELTA_RMS
        ),
        "target/normalized_coordinate_near_zero_fraction": near_zero_fraction,
        "target/normalized_coordinate_near_zero_threshold": (
            _NEAR_ZERO_NORMALIZED_COORDINATE
        ),
        "target/channel_observed_reference_std_ratio_p10": ratio_p10,
        "target/channel_observed_reference_std_ratio_p50": ratio_p50,
        "target/channel_observed_reference_std_ratio_p90": ratio_p90,
        "model/predicted_velocity_mean": predicted_mean,
        "model/predicted_velocity_std": predicted_std,
        "model/predicted_velocity_frame_norm_mean": predicted_norm_mean,
        "model/predicted_velocity_frame_norm_std": predicted_norm_std,
        "estimate/latent_mean": estimated_mean,
        "estimate/latent_std": estimated_std,
        "estimate/latent_frame_norm_mean": estimated_norm_mean,
        "estimate/latent_frame_norm_std": estimated_norm_std,
        "estimate/latent_frame_norm_p01": estimate_norm_p01,
        "estimate/latent_frame_norm_p50": estimate_norm_p50,
        "estimate/latent_frame_norm_p99": estimate_norm_p99,
        "estimate/motion_normalized_delta_rms_mean": estimated_motion_mean,
        "estimate/motion_normalized_delta_rms_p50": estimate_motion_p50,
        "estimate/motion_normalized_delta_rms_p95": estimate_motion_p95,
        "estimate/near_static_fraction": estimated_near_static_fraction,
        "estimate/near_static_normalized_delta_rms_threshold": (
            _NEAR_STATIC_NORMALIZED_DELTA_RMS
        ),
        "estimate/normalized_coordinate_near_zero_fraction": (
            estimated_near_zero_fraction
        ),
        "estimate/normalized_coordinate_near_zero_threshold": (
            _NEAR_ZERO_NORMALIZED_COORDINATE
        ),
        "estimate/channel_observed_reference_std_ratio_p10": (estimate_ratio_p10),
        "estimate/channel_observed_reference_std_ratio_p50": (estimate_ratio_p50),
        "estimate/channel_observed_reference_std_ratio_p90": (estimate_ratio_p90),
    }

    expected_code_shape = (future.shape[0],)
    for name, values in (
        ("source", batch.source_code),
        ("category", batch.category_code),
    ):
        if values.shape != expected_code_shape:
            raise ValueError(f"{name} coverage metadata shape mismatch")
        histogram = _categorical_histogram(
            values,
            name=f"{name} coverage code",
        )
        count = float(histogram.sum())
        for code, observed in enumerate(histogram):
            metrics[f"coverage/{name}_code_{code}_fraction"] = float(observed) / count

    division = batch.segment_division
    segment_index = batch.segment_index
    if (division is None) != (segment_index is None):
        raise ValueError("segment diagnostics require division and index together")
    if division is not None and segment_index is not None:
        divisions = _integer_tensor(division.detach(), name="segment division")
        indices = _integer_tensor(segment_index.detach(), name="segment index")
        expected_shape = (future.shape[0],)
        if divisions.shape != expected_shape or indices.shape != expected_shape:
            raise ValueError("segment diagnostic metadata shape mismatch")
        valid_division = torch.zeros_like(divisions, dtype=torch.bool)
        for value in _SEGMENT_DIVISION_BINS:
            valid_division |= divisions == value
        if not torch.all(valid_division) or not torch.all(
            (indices >= 0) & (indices < divisions)
        ):
            raise ValueError("segment diagnostic metadata is out of range")
        histogram = torch.stack(
            [
                *[(divisions == value).sum() for value in _SEGMENT_DIVISION_BINS],
                *[(indices == value).sum() for value in _SEGMENT_INDEX_BINS],
                *[
                    ((divisions == value) & (indices == index)).sum()
                    for value in _SEGMENT_DIVISION_BINS
                    for index in range(value)
                ],
            ]
        ).to(dtype=torch.float64)
        histogram = _distributed_sum(histogram)
        segment_count = float(histogram[: len(_SEGMENT_DIVISION_BINS)].sum())
        for offset, value in enumerate(_SEGMENT_DIVISION_BINS):
            metrics[f"segment/division_{value}_fraction"] = (
                float(histogram[offset]) / segment_count
            )
        index_offset = len(_SEGMENT_DIVISION_BINS)
        for offset, value in enumerate(_SEGMENT_INDEX_BINS):
            metrics[f"segment/index_{value}_fraction"] = (
                float(histogram[index_offset + offset]) / segment_count
            )
        joint_offset = index_offset + len(_SEGMENT_INDEX_BINS)
        offset = 0
        for value in _SEGMENT_DIVISION_BINS:
            for index in range(value):
                metrics[f"segment/division_{value}_index_{index}_fraction"] = (
                    float(histogram[joint_offset + offset]) / segment_count
                )
                offset += 1
        metrics["segment/effective_future_frames_mean"] = length_mean
        metrics["segment/effective_future_frames_p50"] = length_p50
        metrics["segment/effective_future_frames_p95"] = length_p95

    if pitch_conditioning:
        transition = batch.pitch_transition_mask.detach().to(
            device=future.device,
            dtype=torch.bool,
        )
        if transition.shape != (future.shape[0],):
            raise ValueError("MIDI transition diagnostic shape mismatch")
        if pitch_present is None:
            raise ValueError("MIDI diagnostics require condition presence")
        present = pitch_present.detach().to(device=future.device, dtype=torch.bool)
        if present.shape != (future.shape[0],):
            raise ValueError("MIDI condition presence shape mismatch")
        midi_summary = _distributed_sum(
            torch.as_tensor(
                [
                    float(transition.sum()),
                    float(transition.numel()),
                    float(present.sum()),
                    float(present.numel()),
                ],
                device=future.device,
                dtype=torch.float64,
            )
        )
        metrics["midi/transition_batch_fraction"] = float(
            midi_summary[0] / midi_summary[1]
        )
        metrics["midi/condition_present_fraction"] = float(
            midi_summary[2] / midi_summary[3]
        )

        note_sequence = (
            batch.future_midi_sequence
            if midi_sequence_conditioning and batch.future_midi_sequence is not None
            else batch.midi_note[:, None].expand_as(mask)
        )
        if batch.velocity is None:
            raise ValueError("MIDI diagnostics require velocity metadata")
        velocity_sequence = (
            batch.future_velocity_sequence
            if midi_sequence_conditioning and batch.future_velocity_sequence is not None
            else batch.velocity[:, None].expand_as(mask)
        )
        if note_sequence.shape != mask.shape or velocity_sequence.shape != mask.shape:
            raise ValueError("MIDI sequence diagnostic shape mismatch")
        notes = note_sequence.masked_select(mask)
        velocities = velocity_sequence.masked_select(mask)
        if midi_sequence_conditioning:
            note_pairs = mask[:, 1:] & mask[:, :-1]
            note_changes = (note_sequence[:, 1:] != note_sequence[:, :-1]) & note_pairs
            change_summary = _distributed_sum(
                torch.as_tensor(
                    [float(note_changes.sum()), float(note_pairs.sum())],
                    device=future.device,
                    dtype=torch.float64,
                )
            )
            metrics["midi/note_change_frame_fraction"] = float(
                change_summary[0] / change_summary[1]
                if float(change_summary[1])
                else 0.0
            )
            event_frames = (
                batch.midi_event_frame.detach()
                if batch.midi_event_frame is not None
                else torch.full(
                    (future.shape[0],),
                    -1,
                    device=future.device,
                    dtype=torch.long,
                )
            )
            events = _integer_tensor(event_frames, name="MIDI event frame")
            if events.shape != (future.shape[0],) or torch.any(events < -1):
                raise ValueError("MIDI event frame metadata is invalid")
            event_present = events >= 0
            if torch.any(events[event_present] >= valid_lengths[event_present]):
                raise ValueError("MIDI event frame is outside valid future")
            event_summary = _distributed_sum(
                torch.as_tensor(
                    [float(event_present.sum()), float(event_present.numel())],
                    device=future.device,
                    dtype=torch.float64,
                )
            )
            event_p50, event_p95 = _quantiles(
                events.masked_select(event_present).float(),
                (0.50, 0.95),
                name="MIDI event frame",
            )
            metrics["midi/event_present_fraction"] = float(
                event_summary[0] / event_summary[1]
            )
            metrics["midi/event_frame_p50"] = event_p50
            metrics["midi/event_frame_p95"] = event_p95

        notes = _integer_tensor(notes, name="MIDI note diagnostic")
        valid_notes = (notes == -1) | ((notes >= note_min) & (notes <= note_max))
        if not torch.all(valid_notes):
            raise ValueError("MIDI note diagnostic is out of range")
        note_indices = torch.where(notes == -1, 0, notes - note_min + 1)
        note_histogram = torch.bincount(
            note_indices,
            minlength=note_max - note_min + 2,
        ).to(dtype=torch.float64)
        note_histogram = _distributed_sum(note_histogram)
        note_count = float(note_histogram.sum())
        metrics["midi/note_null_fraction"] = float(note_histogram[0]) / note_count
        for note in range(note_min, note_max + 1):
            metrics[f"midi/note_{note:03d}_fraction"] = (
                float(note_histogram[note - note_min + 1]) / note_count
            )
        sounding_notes = notes[notes != -1].float()
        metrics["midi/note_sounding_fraction"] = float(
            note_histogram[1:].sum() / note_histogram.sum()
        )
        note_mean, note_std = _mean_std(_distributed_sum(_moments(sounding_notes)))
        metrics["midi/note_mean"] = note_mean
        metrics["midi/note_std"] = note_std

        velocities = _integer_tensor(
            velocities,
            name="MIDI velocity diagnostic",
        )
        if torch.any(velocities < 0) or torch.any(velocities > 127):
            raise ValueError("MIDI velocity diagnostic is out of range")
        velocity_histogram = torch.bincount(
            velocities // _VELOCITY_BIN_WIDTH,
            minlength=128 // _VELOCITY_BIN_WIDTH,
        ).to(dtype=torch.float64)
        velocity_histogram = _distributed_sum(velocity_histogram)
        velocity_count = float(velocity_histogram.sum())
        for lower in range(0, 128, _VELOCITY_BIN_WIDTH):
            upper = lower + _VELOCITY_BIN_WIDTH - 1
            metrics[f"midi/velocity_{lower:03d}_{upper:03d}_fraction"] = (
                float(velocity_histogram[lower // _VELOCITY_BIN_WIDTH]) / velocity_count
            )
        velocity_mean, velocity_std = _mean_std(
            _distributed_sum(_moments(velocities.float()))
        )
        metrics["midi/velocity_mean"] = velocity_mean
        metrics["midi/velocity_std"] = velocity_std

    if any(not math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("flow update diagnostics are non-finite")
    return metrics


@dataclass(frozen=True)
class FlowUpdateResult:
    loss: float
    components: dict[str, float]
    pitch_weight: float
    gradient_norm: float
    amp_scale: float
    global_valid_frames: int
    duration_seconds: float
    pitch_metrics: dict[str, float]
    exposure: bool
    exposure_depth: int
    exploration: float
    visible_history_frames: int
    schedule_offset_frames: int
    applied: bool
    learning_rate: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


def _all_rank_finite(value: Tensor) -> bool:
    flag = torch.tensor(
        int(not torch.isfinite(value)),
        device=value.device,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item()) == 0


def _apply_amp_optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_norm: Tensor,
) -> tuple[bool, float]:
    if _all_rank_finite(gradient_norm):
        scaler.step(optimizer)
        scaler.update()
        return True, float(scaler.get_scale())
    new_scale = max(
        float(scaler.get_scale()) * float(scaler.get_backoff_factor()),
        1.0,
    )
    optimizer.zero_grad(set_to_none=True)
    scaler.update(new_scale=new_scale)
    return False, new_scale


def _next_nonfinite_gradient_streak(
    current: int,
    *,
    applied: bool,
    maximum: int,
) -> int:
    if current < 0 or maximum <= 0:
        raise ValueError("non-finite gradient streak is invalid")
    if applied:
        return 0
    updated = current + 1
    if updated >= maximum:
        raise FloatingPointError(f"{maximum} consecutive non-finite flow gradients")
    return updated


def _run_flow_update(
    *,
    training_model: nn.Module,
    sampler: GpuFlowSampler,
    pitch_probe: LatentPitchProbe | None,
    statistics: FlowStatistics,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    controller: PitchWeightController | None,
    config: ZraveFlowConfig,
    update: int,
    maximum_updates: int,
    batch_per_gpu: int,
    device: torch.device,
    rank: int,
    force_exposure: bool | None = None,
    exposure_start_update: int | None = None,
) -> FlowUpdateResult:
    generator = _default_generator(device)
    exploration = 0.0
    visible_history_frames = config.model.context_frames
    schedule_offset_frames = 0
    exposure_depth = 0
    if config.exploration.enabled:
        visibility_index = int(
            torch.randint(
                len(config.exploration.visible_history_frames),
                (),
                generator=generator,
                device=generator.device,
            ).item()
        )
        visible_history_frames = config.exploration.visible_history_frames[
            visibility_index
        ]
        exploration = (config.model.context_frames - visible_history_frames) / (
            config.model.context_frames - 8
        )
        offset_index = int(
            torch.randint(
                len(config.exploration.schedule_offsets),
                (),
                generator=generator,
                device=generator.device,
            ).item()
        )
        schedule_offset_frames = config.exploration.schedule_offsets[offset_index]
        exposure_depth = (
            exploration_exposure_depth(
                update,
                config.exploration,
                generator,
            )
            if force_exposure is None
            else (config.exploration.exposure_max_depth if force_exposure else 0)
        )
        exposure = exposure_depth > 0
    else:
        exposure = (
            use_exposure_batch(
                update,
                maximum_updates,
                config.train.exposure_start_fraction,
                config.train.exposure_probability,
                generator,
                start_update=exposure_start_update,
            )
            if force_exposure is None
            else force_exposure
        )
        exposure_depth = int(exposure)
    if config.segment_sampling.enabled or config.model.midi_sequence_conditioning:
        # Natural 2/4/8 targets are shorter than 64 frames, so the legacy
        # generated-history exposure contract cannot shift them safely.
        # Sequence MIDI also needs a condition-aware rollout before exposure
        # can be enabled without silently dropping the requested control.
        exposure = False
        exposure_depth = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    maximum_future = maximum_valid_future(
        update,
        config.train.short_future_updates,
    )
    batch = sampler.sample(
        batch_per_gpu,
        maximum_valid_future=(64 if exposure else maximum_future),
        require_full_future=exposure,
    )
    if exposure:
        if config.exploration.enabled:
            batch = _prepare_exploration_exposure_batch(
                _unwrapped(training_model),
                batch,
                generation_seed=config.seed + update,
                block_index=rank * 4,
                depth=exposure_depth,
                stride_frames=(config.exploration.rollout_stride_frames),
                exploration=exploration,
                schedule_offset_frames=schedule_offset_frames,
            )
        else:
            batch = _prepare_exposure_batch(
                _unwrapped(training_model),
                batch,
                generation_seed=config.seed + update,
                block_index=rank,
            )
    if config.exploration.enabled:
        controls = exploration_controls(
            torch.full(
                (batch_per_gpu,),
                exploration,
                device=device,
            )
        )
        temperatures = controls.temperature
        wander_delays = controls.wander_delay_frames
        history_mask = trailing_history_mask(
            torch.full(
                (batch_per_gpu,),
                visible_history_frames,
                device=device,
                dtype=torch.long,
            ),
            config.model.context_frames,
        )
    else:
        temperatures = 0.7 + 0.6 * torch.rand(
            batch_per_gpu,
            device=device,
        )
        wander_delays = batch.wander_delay_frames
        history_mask = batch.history_mask
    schedule_offsets: Tensor | int = (
        batch.absolute_start
        if batch.absolute_start is not None
        else schedule_offset_frames
    )
    pair = make_flow_training_pair(
        batch.future,
        batch.future_mask,
        temperatures,
        wander_delays,
        statistics,
        generator,
        schedule_offset_frames=schedule_offsets,
    )
    next_update = update + 1
    learning_rate = _learning_rate(
        config,
        next_update,
        maximum_updates,
    )
    _set_learning_rate(optimizer, learning_rate)
    optimizer.zero_grad(set_to_none=True)
    retention = retention_curve(
        wander_delays,
        64,
        offset_frames=schedule_offsets,
    )
    pitch_present: Tensor | None = None
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        if config.model.pitch_conditioning:
            if pitch_probe is None or controller is None:
                raise ValueError("conditional flow update requires pitch runtime")
            dropout = config.model.condition_dropout
            pitch_present = torch.rand(batch_per_gpu, device=device) >= dropout
            context_present = torch.rand(batch_per_gpu, device=device) >= dropout
            if config.model.midi_sequence_conditioning:
                if batch.velocity is None or batch.history_velocity is None:
                    raise ValueError("MIDI sequence batch is missing velocity metadata")
                history_notes = (
                    batch.history_midi_sequence
                    if batch.history_midi_sequence is not None
                    else batch.history_midi_note[:, None].expand(
                        -1, config.model.context_frames
                    )
                )
                history_velocities = (
                    batch.history_velocity_sequence
                    if batch.history_velocity_sequence is not None
                    else batch.history_velocity[:, None].expand(
                        -1, config.model.context_frames
                    )
                )
                future_notes = (
                    batch.future_midi_sequence
                    if batch.future_midi_sequence is not None
                    else batch.midi_note[:, None].expand(-1, config.model.future_frames)
                )
                future_velocities = (
                    batch.future_velocity_sequence
                    if batch.future_velocity_sequence is not None
                    else batch.velocity[:, None].expand(-1, config.model.future_frames)
                )
                predicted_velocity = training_model(
                    pair.noisy_future,
                    pair.flow_time,
                    batch.history,
                    retention=retention,
                    future_mask=batch.future_mask,
                    context_present=context_present,
                    midi_present=pitch_present,
                    history_mask=history_mask,
                    history_midi_note=history_notes,
                    history_velocity=history_velocities,
                    future_midi_note=future_notes,
                    future_velocity=future_velocities,
                )
            else:
                conditioned_note = torch.where(
                    pitch_present,
                    batch.midi_note,
                    torch.full_like(batch.midi_note, -1),
                )
                predicted_velocity = training_model(
                    pair.noisy_future,
                    pair.flow_time,
                    batch.history,
                    conditioned_note,
                    retention,
                    future_mask=batch.future_mask,
                    context_present=context_present,
                    history_mask=history_mask,
                )
            pitch_weight = _effective_pitch_weight(
                controller,
                next_update,
            )
            report = zrave_flow_loss(
                predicted_velocity,
                pair,
                batch.history,
                batch.future_mask,
                (
                    batch.future_midi_sequence
                    if batch.future_midi_sequence is not None
                    else batch.midi_note
                ),
                pitch_probe,
                statistics,
                pitch_weight,
                history_mask=history_mask,
            )
        else:
            predicted_velocity = training_model(
                pair.noisy_future,
                pair.flow_time,
                batch.history,
                retention=retention,
                future_mask=batch.future_mask,
                history_mask=history_mask,
            )
            pitch_weight = 0.0
            report = zrave_pure_flow_loss(
                predicted_velocity,
                pair,
                batch.history,
                batch.future_mask,
                statistics,
                temporal_weight=(
                    config.exploration.temporal_loss_weight
                    if config.exploration.enabled
                    else 0.0
                ),
                history_mask=history_mask,
            )
    if (
        config.model.pitch_conditioning
        and next_update % config.loss.pitch_gradient_measure_every == 0
    ):
        assert controller is not None
        parameters = _future_parameters(training_model)
        flow_norm = distributed_gradient_l2_norm(
            report.components["flow"],
            parameters,
        )
        pitch_norm = distributed_gradient_l2_norm(
            report.components["pitch"],
            parameters,
        )
        controller.distributed_update(
            flow_norm,
            pitch_norm,
            next_update,
        )
        pitch_weight = _effective_pitch_weight(
            controller,
            next_update,
        )
        report = _replace_pitch_weight(report, pitch_weight)
    if not _all_rank_finite(report.total):
        raise FloatingPointError("non-finite flow loss")
    scaler.scale(report.total).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        training_model.parameters(),
        config.train.gradient_clip,
    )
    applied, amp_scale = _apply_amp_optimizer_step(
        optimizer=optimizer,
        scaler=scaler,
        gradient_norm=gradient_norm,
    )
    diagnostics_due = applied and (
        next_update == 1 or next_update % config.train.log_every == 0
    )
    if diagnostics_due:
        normalized_estimate = (
            pair.noisy_future
            + (1.0 - pair.flow_time[:, None, None])
            * predicted_velocity.detach().float()
        )
        estimated_future = normalized_estimate * statistics.latent_std.to(
            normalized_estimate.device
        ) + statistics.mean.to(normalized_estimate.device)
        pitch_metrics = (
            _pitch_diagnostics(
                pair,
                predicted_velocity,
                batch.future_mask,
                (
                    batch.future_midi_sequence
                    if batch.future_midi_sequence is not None
                    else batch.midi_note
                ),
                batch.pitch_transition_mask,
                pitch_probe,
                statistics,
            )
            if pitch_probe is not None
            else {}
        )
        diagnostics = _flow_batch_diagnostics(
            batch=batch,
            predicted_velocity=predicted_velocity,
            estimated_future=estimated_future,
            statistics=statistics,
            pitch_conditioning=config.model.pitch_conditioning,
            midi_sequence_conditioning=(config.model.midi_sequence_conditioning),
            note_min=config.model.note_min,
            note_max=config.model.note_max,
            pitch_present=pitch_present,
        )
        component_names = tuple(sorted(report.components))
        reported_values = torch.stack(
            (
                report.total.detach(),
                *(report.components[name].detach() for name in component_names),
            )
        ).to(dtype=torch.float64)
        reported_values = _distributed_sum(reported_values)
        if dist.is_initialized():
            reported_values /= dist.get_world_size()
        if not torch.isfinite(reported_values).all():
            raise FloatingPointError("non-finite reduced flow loss report")
        reported_loss = float(reported_values[0])
        reported_components = {
            name: float(reported_values[index + 1])
            for index, name in enumerate(component_names)
        }
    else:
        pitch_metrics = {}
        diagnostics = {}
        reported_loss = float(report.total.detach())
        reported_components = {
            name: float(value.detach()) for name, value in report.components.items()
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    timing = torch.tensor(
        [duration, float(batch.future_mask.sum())],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        duration_value = timing[:1].clone()
        frame_value = timing[1:].clone()
        dist.all_reduce(duration_value, op=dist.ReduceOp.MAX)
        dist.all_reduce(frame_value, op=dist.ReduceOp.SUM)
        duration = float(duration_value.item())
        global_frames = int(frame_value.item())
    else:
        global_frames = int(timing[1].item())
    return FlowUpdateResult(
        loss=reported_loss,
        components=reported_components,
        pitch_weight=pitch_weight,
        gradient_norm=float(gradient_norm),
        amp_scale=amp_scale,
        global_valid_frames=global_frames,
        duration_seconds=duration,
        pitch_metrics=pitch_metrics,
        exposure=exposure,
        exposure_depth=exposure_depth,
        exploration=exploration,
        visible_history_frames=visible_history_frames,
        schedule_offset_frames=(
            int(batch.absolute_start.float().mean().item())
            if batch.absolute_start is not None
            else schedule_offset_frames
        ),
        applied=applied,
        learning_rate=learning_rate,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def _validation_snapshot(
    *,
    model: nn.Module,
    sampler: GpuFlowSampler,
    pitch_probe: LatentPitchProbe | None,
    statistics: FlowStatistics,
    config: ZraveFlowConfig,
    batch_per_gpu: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    sampler_state = sampler.state_dict()
    rng_state = _capture_rng_state()
    was_training = model.training
    try:
        model.eval()
        validation_generator = torch.Generator(device=device)
        validation_generator.manual_seed(config.seed + 8_001_003)
        batch = sampler.sample(
            batch_per_gpu,
            maximum_valid_future=64,
        )
        schedule_offsets: Tensor | int = (
            batch.absolute_start if batch.absolute_start is not None else 0
        )
        pair = make_flow_training_pair(
            batch.future,
            batch.future_mask,
            torch.ones(batch_per_gpu, device=device),
            batch.wander_delay_frames,
            statistics,
            validation_generator,
            schedule_offset_frames=schedule_offsets,
        )
        retention = retention_curve(
            batch.wander_delay_frames,
            64,
            offset_frames=schedule_offsets,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            if config.model.pitch_conditioning:
                if pitch_probe is None:
                    raise ValueError("conditional validation requires a pitch probe")
                if config.model.midi_sequence_conditioning:
                    if batch.velocity is None or batch.history_velocity is None:
                        raise ValueError(
                            "MIDI validation batch lacks velocity metadata"
                        )
                    history_notes = (
                        batch.history_midi_sequence
                        if batch.history_midi_sequence is not None
                        else batch.history_midi_note[:, None].expand(
                            -1, config.model.context_frames
                        )
                    )
                    history_velocities = (
                        batch.history_velocity_sequence
                        if batch.history_velocity_sequence is not None
                        else batch.history_velocity[:, None].expand(
                            -1, config.model.context_frames
                        )
                    )
                    future_notes = (
                        batch.future_midi_sequence
                        if batch.future_midi_sequence is not None
                        else batch.midi_note[:, None].expand(
                            -1, config.model.future_frames
                        )
                    )
                    future_velocities = (
                        batch.future_velocity_sequence
                        if batch.future_velocity_sequence is not None
                        else batch.velocity[:, None].expand(
                            -1, config.model.future_frames
                        )
                    )
                    velocity = model(
                        pair.noisy_future,
                        pair.flow_time,
                        batch.history,
                        retention=retention,
                        future_mask=batch.future_mask,
                        history_mask=batch.history_mask,
                        history_midi_note=history_notes,
                        history_velocity=history_velocities,
                        future_midi_note=future_notes,
                        future_velocity=future_velocities,
                    )
                else:
                    velocity = model(
                        pair.noisy_future,
                        pair.flow_time,
                        batch.history,
                        batch.midi_note,
                        retention,
                        future_mask=batch.future_mask,
                        history_mask=batch.history_mask,
                    )
                report = zrave_flow_loss(
                    velocity,
                    pair,
                    batch.history,
                    batch.future_mask,
                    (
                        batch.future_midi_sequence
                        if batch.future_midi_sequence is not None
                        else batch.midi_note
                    ),
                    pitch_probe,
                    statistics,
                    pitch_weight=0.30,
                    history_mask=batch.history_mask,
                )
            else:
                velocity = model(
                    pair.noisy_future,
                    pair.flow_time,
                    batch.history,
                    retention=retention,
                    future_mask=batch.future_mask,
                    history_mask=batch.history_mask,
                )
                report = zrave_pure_flow_loss(
                    velocity,
                    pair,
                    batch.history,
                    batch.future_mask,
                    statistics,
                    temporal_weight=(
                        config.exploration.temporal_loss_weight
                        if config.exploration.enabled
                        else 0.0
                    ),
                    history_mask=batch.history_mask,
                )
        component_names = tuple(sorted(report.components))
        values = torch.stack(
            (
                report.total.detach(),
                *(report.components[name].detach() for name in component_names),
            )
        ).to(device=device, dtype=torch.float64)
        values = _distributed_sum(values)
        if dist.is_initialized():
            values /= dist.get_world_size()
        normalized = (
            pair.noisy_future + (1.0 - pair.flow_time[:, None, None]) * velocity.float()
        )
        valid = batch.future_mask.unsqueeze(-1).expand_as(normalized)
        diversity_summary = _distributed_sum(_moments(normalized.masked_select(valid)))
        _diversity_mean, diversity_std = _mean_std(diversity_summary)
        if not torch.isfinite(values).all() or not math.isfinite(diversity_std):
            raise FloatingPointError("non-finite validation snapshot")
        validation = {
            "total": float(values[0]),
            **{
                name: float(values[index + 1])
                for index, name in enumerate(component_names)
            },
        }
        return validation, {"normalized_clean_std": diversity_std}
    finally:
        sampler.load_state_dict(sampler_state)
        _restore_rng_state(rng_state)
        model.train(was_training)


def _qualification_artifacts(
    path: str | Path,
) -> tuple[Path, Path, dict[str, object]]:
    candidate = Path(path)
    if candidate.is_dir():
        qualification_path = candidate / "qualification.json"
    elif candidate.suffix == ".json":
        qualification_path = candidate
    else:
        qualification_path = candidate.parent.parent / "qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    checkpoint_value = qualification.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("pitch qualification lacks checkpoint")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = qualification_path.parent / checkpoint
    return qualification_path, checkpoint, qualification


def _git_commit(root: Path) -> str:
    resolved_root = root.resolve()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={resolved_root}",
            "rev-parse",
            "HEAD",
        ],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Git HEAD is not an exact 40-character commit")
    return commit


def summarize_flow_benchmark(
    *,
    batch_per_gpu: int,
    world_size: int,
    durations_seconds: Iterable[float],
    global_valid_frames: Iterable[int],
    exposure_safety_updates: int,
    peak_memory_mib: float,
    total_memory_mib: float,
    nonfinite_updates: int,
    config_sha256: str,
    pack_index_sha256: str,
    pitch_checkpoint_sha256: str | None,
    git_commit: str,
    require_exposure_safety: bool = True,
) -> dict[str, object]:
    for name, digest in (
        ("config_sha256", config_sha256),
        ("pack_index_sha256", pack_index_sha256),
    ):
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if pitch_checkpoint_sha256 is not None and not _SHA256.fullmatch(
        pitch_checkpoint_sha256
    ):
        raise ValueError("pitch_checkpoint_sha256 must be a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("git_commit must be an exact Git commit ID")
    durations = np.asarray(list(durations_seconds), dtype=np.float64)
    frames = np.asarray(list(global_valid_frames), dtype=np.float64)
    if durations.shape != frames.shape:
        raise ValueError("benchmark duration/frame counts differ")
    global_batch = batch_per_gpu * world_size
    valid = (
        durations.size > 0
        and np.isfinite(durations).all()
        and (durations > 0).all()
        and np.isfinite(frames).all()
        and (frames > 0).all()
        and (
            exposure_safety_updates == 5
            if require_exposure_safety
            else exposure_safety_updates == 0
        )
        and nonfinite_updates == 0
        and math.isfinite(peak_memory_mib)
        and 0.0 < peak_memory_mib < total_memory_mib
    )
    windows_per_second = (
        global_batch / durations if durations.size else np.asarray([], dtype=np.float64)
    )
    frames_per_second = (
        frames / durations if durations.size else np.asarray([], dtype=np.float64)
    )
    report: dict[str, object] = {
        "status": "ok" if valid else "invalid",
        "batch_per_gpu": int(batch_per_gpu),
        "world_size": int(world_size),
        "global_batch": int(global_batch),
        "measured_updates": int(durations.size),
        "exposure_safety_updates": int(exposure_safety_updates),
        "median_windows_per_second": (
            float(np.median(windows_per_second)) if windows_per_second.size else 0.0
        ),
        "median_valid_latent_frames_per_second": float(
            np.median(frames_per_second) if frames_per_second.size else 0.0
        ),
        "p10_windows_per_second": (
            float(np.percentile(windows_per_second, 10))
            if windows_per_second.size
            else 0.0
        ),
        "peak_memory_mib": float(peak_memory_mib),
        "total_memory_mib": float(total_memory_mib),
        "nonfinite_updates": int(nonfinite_updates),
        "exposure_safety_required": bool(require_exposure_safety),
        "config_sha256": config_sha256,
        "pack_index_sha256": pack_index_sha256,
        "git_commit": git_commit,
    }
    if pitch_checkpoint_sha256 is not None:
        report["pitch_checkpoint_sha256"] = pitch_checkpoint_sha256
    return report


def _build_runtime(
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = ZraveFlowConfig.load(args.config)
    rank, local_rank, world_size, device = _setup_distributed()
    checkpoint_process_group = _new_checkpoint_process_group(world_size)
    _seed_everything(config.seed, rank)
    batch_per_gpu = int(args.batch_per_gpu or config.train.batch_per_gpu)
    maximum_updates = int(args.max_updates or config.train.max_updates)
    if batch_per_gpu <= 0 or maximum_updates <= 0:
        raise ValueError("training batch and update counts must be positive")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    pack_hash = _sha256_file(index_path)
    statistics = _load_statistics(packed_root)
    pitch_probe: LatentPitchProbe | None = None
    pitch_checkpoint_hash: str | None = None
    pitch_qualification_hash: str | None = None
    if config.model.pitch_conditioning:
        if args.pitch_probe is None:
            raise ValueError("conditional flow config requires --pitch-probe")
        qualification_path, pitch_checkpoint, _qualification = _qualification_artifacts(
            args.pitch_probe
        )
        pitch_probe = load_qualified_pitch_probe(
            args.pitch_probe,
            pack_hash,
        ).to(device)
        pitch_probe.requires_grad_(False)
        pitch_probe.eval()
        pitch_checkpoint_hash = _sha256_file(pitch_checkpoint)
        pitch_qualification_hash = _sha256_file(qualification_path)
    elif args.pitch_probe is not None:
        raise ValueError("pure flow config does not accept --pitch-probe")
    train_sampler = GpuFlowSampler.from_pack(
        config,
        split="train",
        device=device,
        seed=config.seed + 3000 + rank,
    )
    validation_sampler = _shared_validation_sampler(
        train_sampler,
        seed=config.seed + 4000 + rank,
    )
    model = ZraveFlowTransformer(
        config.model,
        statistics,
    ).to(device)
    training_model: nn.Module
    if world_size > 1:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    else:
        training_model = model
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    if config.source_path is None:
        raise ValueError("training requires a file-backed config")
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=world_size,
        batch_per_gpu=batch_per_gpu,
        maximum_updates=maximum_updates,
        config_sha256=_sha256_file(config.source_path),
        pack_index_sha256=pack_hash,
        statistics_sha256=_sha256_file(statistics_path),
        pitch_checkpoint_sha256=pitch_checkpoint_hash,
        pitch_qualification_sha256=pitch_qualification_hash,
    )
    controller = (
        PitchWeightController(
            initial=config.loss.pitch_initial,
            minimum=config.loss.pitch_minimum,
            maximum=config.loss.pitch_maximum,
            target_minimum=config.loss.pitch_gradient_minimum_ratio,
            target_maximum=config.loss.pitch_gradient_maximum_ratio,
            warmup_updates=config.loss.pitch_warmup_updates,
        )
        if config.model.pitch_conditioning
        else None
    )
    return {
        "config": config,
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "checkpoint_process_group": checkpoint_process_group,
        "batch_per_gpu": batch_per_gpu,
        "maximum_updates": maximum_updates,
        "statistics": statistics,
        "pitch_probe": pitch_probe,
        "train_sampler": train_sampler,
        "validation_sampler": validation_sampler,
        "training_model": training_model,
        "optimizer": optimizer,
        "scaler": scaler,
        "contract": contract,
        "controller": controller,
    }


def _update_scalars(result: FlowUpdateResult) -> dict[str, float]:
    values: dict[str, float] = {
        "train/loss": result.loss,
        **{f"train/{name}": value for name, value in result.components.items()},
        **{f"probe/{name}": value for name, value in result.pitch_metrics.items()},
        **{f"diagnostics/{name}": value for name, value in result.diagnostics.items()},
        "train/learning_rate": result.learning_rate,
        "train/valid_latent_frames_per_second": (
            result.global_valid_frames / result.duration_seconds
        ),
        "health/gradient_norm": result.gradient_norm,
        "health/amp_scale": result.amp_scale,
        "health/exposure_batch": float(result.exposure),
        "health/exposure_depth": float(result.exposure_depth),
        "train/exploration": result.exploration,
        "train/visible_history_frames": float(result.visible_history_frames),
        "train/schedule_offset_frames": float(result.schedule_offset_frames),
    }
    if result.pitch_metrics or result.pitch_weight:
        values["train/pitch_weight"] = result.pitch_weight
    finite: dict[str, float] = {}
    for name, value in values.items():
        parsed = float(value)
        if not math.isfinite(parsed):
            raise FloatingPointError(f"non-finite TensorBoard scalar: {name}")
        finite[name] = parsed
    return finite


def _last_metrics_update(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        data = b""
        while position > 0:
            size = min(8192, position)
            position -= size
            handle.seek(position)
            data = handle.read(size) + data
            lines = data.splitlines()
            if len(lines) > 1 or position == 0:
                for raw in reversed(lines):
                    if raw.strip():
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid metrics JSONL tail: {path}"
                            ) from error
                        if not isinstance(payload, dict):
                            raise ValueError("metrics JSONL tail is not an object")
                        update = payload.get("update")
                        if (
                            not isinstance(update, int)
                            or isinstance(update, bool)
                            or update < 0
                        ):
                            raise ValueError("metrics JSONL tail update is invalid")
                        return update
    return None


def _append_update_metrics(
    path: Path,
    result: FlowUpdateResult,
    update: int,
    *,
    previous_update: int | None,
    wall_time: float | None = None,
) -> int:
    if update <= 0:
        raise ValueError("metrics update must be positive")
    if previous_update is not None and update <= previous_update:
        raise ValueError(
            "metrics update must be strictly newer than JSONL tail: "
            f"{update} <= {previous_update}"
        )
    observed_wall_time = time.time() if wall_time is None else float(wall_time)
    if not math.isfinite(observed_wall_time) or observed_wall_time < 0.0:
        raise ValueError("metrics wall time must be finite and non-negative")
    payload = {
        "schema": 1,
        "update": update,
        "timestamp_utc": datetime.fromtimestamp(
            observed_wall_time,
            tz=UTC,
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "wall_time_unix_seconds": observed_wall_time,
        "scalars": _update_scalars(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return update


def _append_metrics_resume_boundary(
    path: Path,
    *,
    checkpoint_update: int,
    previous_tail_update: int,
    wall_time: float | None = None,
) -> int:
    """Record an intentional telemetry rollback without deleting evidence.

    Training state is checkpointed less often than metrics are logged. A hard
    allocation failure can therefore leave valid telemetry ahead of the newest
    recoverable checkpoint. Keep those orphaned observations append-only, mark
    the rollback explicitly, and let the resumed trajectory reuse update IDs.
    """

    if checkpoint_update < 0 or previous_tail_update <= checkpoint_update:
        raise ValueError("metrics resume boundary updates are invalid")
    observed_wall_time = time.time() if wall_time is None else float(wall_time)
    if not math.isfinite(observed_wall_time) or observed_wall_time < 0.0:
        raise ValueError("metrics wall time must be finite and non-negative")
    payload = {
        "schema": 1,
        "event": "resume_rollback_to_checkpoint",
        "update": checkpoint_update,
        "orphaned_tail_update": previous_tail_update,
        "timestamp_utc": datetime.fromtimestamp(
            observed_wall_time,
            tz=UTC,
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "wall_time_unix_seconds": observed_wall_time,
        "scalars": {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return checkpoint_update


def _append_metrics_event(
    path: Path,
    *,
    event: str,
    update: int,
    scalars: dict[str, float | int],
    wall_time: float | None = None,
) -> None:
    if not event or update < 0:
        raise ValueError("metrics event name/update is invalid")
    observed_wall_time = time.time() if wall_time is None else float(wall_time)
    if not math.isfinite(observed_wall_time) or observed_wall_time < 0.0:
        raise ValueError("metrics wall time must be finite and non-negative")
    finite: dict[str, float] = {}
    for name, value in scalars.items():
        parsed = float(value)
        if not name or not math.isfinite(parsed):
            raise FloatingPointError(f"non-finite metrics event scalar: {name}")
        finite[name] = parsed
    payload = {
        "schema": 1,
        "event": event,
        "update": update,
        "timestamp_utc": datetime.fromtimestamp(
            observed_wall_time,
            tz=UTC,
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "wall_time_unix_seconds": observed_wall_time,
        "scalars": finite,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _log_update(
    writer: SummaryWriter,
    result: FlowUpdateResult,
    update: int,
) -> None:
    for name, value in _update_scalars(result).items():
        writer.add_scalar(name, value, update)


def _benchmark(args: argparse.Namespace) -> None:
    runtime = _build_runtime(args)
    config: ZraveFlowConfig = runtime["config"]
    rank: int = runtime["rank"]
    device: torch.device = runtime["device"]
    if args.initialize_from:
        load_flow_initial_weights(
            args.initialize_from,
            model=runtime["training_model"],
            expected_contract=runtime["contract"],
        )
    warmup = int(args.benchmark_warmup)
    measured = int(args.benchmark_updates)
    exposure_updates = int(args.benchmark_exposure_updates)
    exposure_supported = not (
        config.segment_sampling.enabled or config.model.midi_sequence_conditioning
    )
    if not exposure_supported:
        exposure_updates = 0
    if warmup < 0 or measured <= 0 or exposure_updates < 0:
        raise ValueError("invalid benchmark update counts")
    update = 0
    nonfinite = 0
    for _ in range(warmup):
        result = _run_flow_update(
            training_model=runtime["training_model"],
            sampler=runtime["train_sampler"],
            pitch_probe=runtime["pitch_probe"],
            statistics=runtime["statistics"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            controller=runtime["controller"],
            config=config,
            update=update,
            maximum_updates=runtime["maximum_updates"],
            batch_per_gpu=runtime["batch_per_gpu"],
            device=device,
            rank=rank,
            force_exposure=False,
        )
        if result.applied:
            update += 1
        else:
            nonfinite += 1
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    frames: list[int] = []
    for _ in range(measured):
        result = _run_flow_update(
            training_model=runtime["training_model"],
            sampler=runtime["train_sampler"],
            pitch_probe=runtime["pitch_probe"],
            statistics=runtime["statistics"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            controller=runtime["controller"],
            config=config,
            update=update,
            maximum_updates=runtime["maximum_updates"],
            batch_per_gpu=runtime["batch_per_gpu"],
            device=device,
            rank=rank,
            force_exposure=False,
        )
        if result.applied:
            durations.append(result.duration_seconds)
            frames.append(result.global_valid_frames)
            update += 1
        else:
            nonfinite += 1
    completed_exposure = 0
    for _ in range(exposure_updates):
        result = _run_flow_update(
            training_model=runtime["training_model"],
            sampler=runtime["train_sampler"],
            pitch_probe=runtime["pitch_probe"],
            statistics=runtime["statistics"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            controller=runtime["controller"],
            config=config,
            update=update,
            maximum_updates=runtime["maximum_updates"],
            batch_per_gpu=runtime["batch_per_gpu"],
            device=device,
            rank=rank,
            force_exposure=True,
        )
        if result.applied:
            completed_exposure += 1
            update += 1
        else:
            nonfinite += 1
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    memory = torch.tensor(
        [peak, total],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        peak_value = memory[:1].clone()
        total_value = memory[1:].clone()
        dist.all_reduce(peak_value, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_value, op=dist.ReduceOp.MIN)
        peak = float(peak_value.item())
        total = float(total_value.item())
    commit_root = Path(__file__).resolve().parents[3]
    report = summarize_flow_benchmark(
        batch_per_gpu=runtime["batch_per_gpu"],
        world_size=runtime["world_size"],
        durations_seconds=durations,
        global_valid_frames=frames,
        exposure_safety_updates=completed_exposure,
        peak_memory_mib=peak,
        total_memory_mib=total,
        nonfinite_updates=nonfinite,
        config_sha256=str(runtime["contract"]["config_sha256"]),
        pack_index_sha256=str(runtime["contract"]["pack_index_sha256"]),
        pitch_checkpoint_sha256=(
            str(runtime["contract"]["pitch_checkpoint_sha256"])
            if "pitch_checkpoint_sha256" in runtime["contract"]
            else None
        ),
        git_commit=_git_commit(commit_root),
        require_exposure_safety=exposure_supported,
    )
    validation, diversity = _validation_snapshot(
        model=runtime["training_model"],
        sampler=runtime["validation_sampler"],
        pitch_probe=runtime["pitch_probe"],
        statistics=runtime["statistics"],
        config=config,
        batch_per_gpu=runtime["batch_per_gpu"],
        device=device,
    )
    report["validation_smoke"] = {
        "split": "validation",
        "losses": validation,
        "diversity": diversity,
        "finite": True,
    }
    if rank == 0:
        _atomic_json(Path(args.benchmark_report), report)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _load_flow_training_start(
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> dict[str, object]:
    state: dict[str, object] = {
        "update": 0,
        "restored_phase3_start": None,
        "latest_gate_hash": None,
        "consecutive_gate_passes": 0,
        "initialization": None,
    }
    expected_initializer_sha256 = getattr(
        args,
        "expected_initializer_sha256",
        None,
    )
    expected_initializer_update = getattr(
        args,
        "expected_initializer_update",
        None,
    )
    derived = (
        runtime["contract"].get("exploration_enabled", False)
        or runtime["contract"].get("segment_sampling", False)
        or runtime["contract"].get("midi_sequence_conditioning", False)
    )
    requires_initializer_lineage = derived and (
        runtime["contract"].get("model_profile", "standard") == "standard"
        or runtime["contract"].get("pitch_conditioning", True)
    )
    if (
        requires_initializer_lineage
        and (getattr(args, "resume", None) or getattr(args, "initialize_from", None))
        and expected_initializer_sha256 is None
        and expected_initializer_update is None
    ):
        raise ValueError(
            "derived flow resume/initialization requires expected initializer "
            "SHA-256 and update"
        )
    if getattr(args, "resume", None):
        restored = load_flow_checkpoint(
            args.resume,
            model=runtime["training_model"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            sampler=runtime["train_sampler"],
            pitch_weight_controller=runtime["controller"],
            expected_contract=runtime["contract"],
            expected_initializer_sha256=expected_initializer_sha256,
            expected_initializer_update=expected_initializer_update,
        )
        state.update(
            update=int(restored["update"]),
            restored_phase3_start=(
                int(restored["phase3_start_update"])
                if restored["phase3_start_update"] is not None
                else None
            ),
            latest_gate_hash=restored["latest_gate_report_sha256"],
            consecutive_gate_passes=int(restored["consecutive_gate_passes"]),
            initialization=restored["initialization"],
        )
    elif getattr(args, "initialize_from", None):
        if not derived:
            raise ValueError("--initialize-from requires a derived flow config")
        state["initialization"] = load_flow_initial_weights(
            args.initialize_from,
            model=runtime["training_model"],
            expected_contract=runtime["contract"],
            expected_initializer_sha256=expected_initializer_sha256,
            expected_initializer_update=expected_initializer_update,
        )
    return state


def _train(args: argparse.Namespace) -> None:
    runtime = _build_runtime(args)
    config: ZraveFlowConfig = runtime["config"]
    rank: int = runtime["rank"]
    start = _load_flow_training_start(args, runtime)
    update = int(start["update"])
    latest_gate_hash = start["latest_gate_hash"]
    consecutive_gate_passes = int(start["consecutive_gate_passes"])
    restored_phase3_start = start["restored_phase3_start"]
    initialization = start["initialization"]
    phase3_start_update = resolve_phase3_start_update(
        maximum_updates=runtime["maximum_updates"],
        default_fraction=(
            config.exploration.exposure_start_update / runtime["maximum_updates"]
            if config.exploration.enabled
            else config.train.exposure_start_fraction
        ),
        resumed_update=update,
        requested=args.phase3_start_update,
        restored=(
            int(restored_phase3_start) if restored_phase3_start is not None else None
        ),
    )
    gate_state = GateEarlyStopState(
        required_consecutive_passes=config.train.early_stop_gate_passes,
        consecutive_passes=consecutive_gate_passes,
    )
    output_root = Path(config.train.output_root)
    checkpoint_root = output_root / "checkpoints"
    if rank == 0 and initialization is not None:
        _atomic_json(
            output_root / "initialization.json",
            initialization,
        )
    metrics_path = output_root / "metrics.jsonl"
    last_metrics_update = _last_metrics_update(metrics_path)
    tensorboard_purge_step: int | None = None
    if last_metrics_update is not None and last_metrics_update > update:
        tensorboard_purge_step = update + 1
        last_metrics_update = _append_metrics_resume_boundary(
            metrics_path,
            checkpoint_update=update,
            previous_tail_update=last_metrics_update,
        )
    writer = (
        SummaryWriter(
            str(output_root / "tensorboard"),
            purge_step=tensorboard_purge_step,
        )
        if rank == 0
        else None
    )
    stopped_early = False
    nonfinite_gradient_skips = 0
    nonfinite_gradient_streak = 0
    while update < runtime["maximum_updates"]:
        result = _run_flow_update(
            training_model=runtime["training_model"],
            sampler=runtime["train_sampler"],
            pitch_probe=runtime["pitch_probe"],
            statistics=runtime["statistics"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            controller=runtime["controller"],
            config=config,
            update=update,
            maximum_updates=runtime["maximum_updates"],
            batch_per_gpu=runtime["batch_per_gpu"],
            device=runtime["device"],
            rank=rank,
            exposure_start_update=phase3_start_update,
        )
        nonfinite_gradient_streak = _next_nonfinite_gradient_streak(
            nonfinite_gradient_streak,
            applied=result.applied,
            maximum=8,
        )
        if not result.applied:
            nonfinite_gradient_skips += 1
            if writer is not None:
                writer.add_scalar(
                    "health/nonfinite_gradient_skips",
                    nonfinite_gradient_skips,
                    update,
                )
                writer.add_scalar(
                    "health/amp_scale",
                    result.amp_scale,
                    update,
                )
                writer.flush()
                _append_metrics_event(
                    metrics_path,
                    event="nonfinite_gradient_skip",
                    update=update,
                    scalars={
                        "health/nonfinite_gradient_skips": (nonfinite_gradient_skips),
                        "health/amp_scale": result.amp_scale,
                    },
                )
            continue
        update += 1
        if writer is not None and (update == 1 or update % config.train.log_every == 0):
            last_metrics_update = _append_update_metrics(
                metrics_path,
                result,
                update,
                previous_update=last_metrics_update,
            )
            _log_update(writer, result, update)
        if update % config.train.validation_every == 0:
            if runtime["validation_sampler"] is None:
                raise RuntimeError("validation runtime is incomplete")
            if config.model.pitch_conditioning and (
                runtime["pitch_probe"] is None or runtime["controller"] is None
            ):
                raise RuntimeError("conditional validation runtime is incomplete")
            validation, diversity = _validation_snapshot(
                model=runtime["training_model"],
                sampler=runtime["validation_sampler"],
                pitch_probe=runtime["pitch_probe"],
                statistics=runtime["statistics"],
                config=config,
                batch_per_gpu=runtime["batch_per_gpu"],
                device=runtime["device"],
            )
            if writer is not None:
                for name, value in validation.items():
                    writer.add_scalar(
                        f"validation/{name}",
                        value,
                        update,
                    )
                for name, value in diversity.items():
                    writer.add_scalar(
                        f"diversity/{name}",
                        value,
                        update,
                    )
                if runtime["controller"] is not None:
                    writer.add_scalar(
                        "health/pitch_controller_ema_flow_norm",
                        runtime["controller"].ema_flow_norm,
                        update,
                    )
                    writer.add_scalar(
                        "health/pitch_controller_ema_pitch_norm",
                        runtime["controller"].ema_pitch_norm,
                        update,
                    )
                writer.flush()
                validation_scalars = {
                    **{
                        f"validation/{name}": value
                        for name, value in validation.items()
                    },
                    **{f"diversity/{name}": value for name, value in diversity.items()},
                }
                if runtime["controller"] is not None:
                    validation_scalars.update(
                        {
                            "health/pitch_controller_ema_flow_norm": (
                                runtime["controller"].ema_flow_norm
                            ),
                            "health/pitch_controller_ema_pitch_norm": (
                                runtime["controller"].ema_pitch_norm
                            ),
                        }
                    )
                _append_metrics_event(
                    metrics_path,
                    event="validation",
                    update=update,
                    scalars=validation_scalars,
                )
        final_due = update == runtime["maximum_updates"]
        checkpoint_due = update % config.train.checkpoint_every == 0 or final_due
        if checkpoint_due:
            checkpoint = checkpoint_root / f"step-{update:06d}.pt"
            save_flow_checkpoint(
                checkpoint,
                model=runtime["training_model"],
                optimizer=runtime["optimizer"],
                scaler=runtime["scaler"],
                sampler=runtime["train_sampler"],
                pitch_weight_controller=runtime["controller"],
                update=update,
                contract=runtime["contract"],
                phase3_start_update=phase3_start_update,
                latest_gate_report_sha256=latest_gate_hash,
                consecutive_gate_passes=gate_state.consecutive_passes,
                checkpoint_process_group=runtime["checkpoint_process_group"],
                initialization=initialization,
            )
            if dist.is_initialized():
                dist.barrier()
            if (
                not config.model.pitch_conditioning
                or config.model.midi_sequence_conditioning
            ):
                if rank == 0 and final_due:
                    _atomic_copy(
                        checkpoint,
                        checkpoint_root / "final.pt",
                    )
                continue
            evaluation_message: list[object] = [None]
            if rank == 0:
                try:
                    evaluation_root = (
                        output_root / "evaluations" / f"update-{update:08d}"
                    )
                    evaluation = evaluate_flow_checkpoint(
                        config,
                        checkpoint,
                        split="validation",
                        output_root=evaluation_root,
                        device=runtime["device"],
                        model=_unwrapped(runtime["training_model"]),
                    )
                    stopped_early = gate_state.update(evaluation["gate"])
                    latest_gate_hash = _sha256_file(evaluation_root / "evaluation.json")
                    evaluation_message[0] = {
                        "latest_gate_report_sha256": latest_gate_hash,
                        "consecutive_gate_passes": (gate_state.consecutive_passes),
                        "stopped_early": stopped_early,
                        "gate": evaluation["gate"],
                    }
                except Exception as error:
                    evaluation_message[0] = {
                        "error": (f"{type(error).__name__}: {error}")
                    }
            if dist.is_initialized():
                dist.broadcast_object_list(
                    evaluation_message,
                    src=0,
                )
            message = evaluation_message[0]
            if not isinstance(message, dict):
                raise RuntimeError("checkpoint evaluation returned no state")
            if "error" in message:
                raise RuntimeError(
                    "checkpoint evaluation failed: " + str(message["error"])
                )
            latest_gate_hash = str(message["latest_gate_report_sha256"])
            gate_state.consecutive_passes = int(message["consecutive_gate_passes"])
            stopped_early = bool(message["stopped_early"])
            if writer is not None:
                writer.add_scalar(
                    "validation/hard_gate_passed",
                    float(bool(message["gate"]["passed"])),
                    update,
                )
                writer.add_scalar(
                    "health/consecutive_gate_passes",
                    gate_state.consecutive_passes,
                    update,
                )
                writer.flush()
            save_flow_checkpoint(
                checkpoint,
                model=runtime["training_model"],
                optimizer=runtime["optimizer"],
                scaler=runtime["scaler"],
                sampler=runtime["train_sampler"],
                pitch_weight_controller=runtime["controller"],
                update=update,
                contract=runtime["contract"],
                phase3_start_update=phase3_start_update,
                latest_gate_report_sha256=latest_gate_hash,
                consecutive_gate_passes=gate_state.consecutive_passes,
                checkpoint_process_group=runtime["checkpoint_process_group"],
                initialization=initialization,
            )
            if dist.is_initialized():
                dist.barrier()
            if rank == 0 and (final_due or stopped_early):
                _atomic_copy(
                    checkpoint,
                    checkpoint_root / "final.pt",
                )
            if stopped_early:
                break
    if rank == 0 and update >= runtime["maximum_updates"]:
        _ensure_final_checkpoint(
            checkpoint_root,
            runtime["maximum_updates"],
        )
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or benchmark the stochastic Z-RAVE flow model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pitch-probe")
    parser.add_argument("--batch-per-gpu", type=int)
    parser.add_argument("--max-updates", type=int)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume")
    checkpoint.add_argument("--initialize-from")
    parser.add_argument("--expected-initializer-sha256")
    parser.add_argument("--expected-initializer-update", type=int)
    parser.add_argument("--phase3-start-update", type=int)
    parser.add_argument("--benchmark-report")
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-updates", type=int, default=100)
    parser.add_argument(
        "--benchmark-exposure-updates",
        type=int,
        default=5,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.benchmark_report:
        _benchmark(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
