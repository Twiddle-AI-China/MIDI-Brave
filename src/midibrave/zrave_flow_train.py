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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_evaluate import (
    GateEarlyStopState,
    evaluate_flow_checkpoint,
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
) -> bool:
    if update < 0 or maximum_updates <= 0:
        raise ValueError("training updates are invalid")
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be in [0, 1]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    threshold = math.ceil(maximum_updates * start_fraction)
    if update < threshold:
        return False
    draw = torch.rand(
        (),
        generator=generator,
        device=generator.device,
    )
    return float(draw.item()) < probability


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
    return (
        model.module
        if isinstance(model, DistributedDataParallel)
        else model
    )


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


def _validate_checkpoint_contract(contract: dict[str, object]) -> None:
    pitch_conditioning = contract.get("pitch_conditioning", True)
    if not isinstance(pitch_conditioning, bool):
        raise ValueError("pitch_conditioning must be boolean")
    required = _BASE_CONTRACT_HASHES | {"world_size", "batch_per_gpu"}
    if pitch_conditioning:
        required |= _PITCH_CONTRACT_HASHES
    missing = required - set(contract)
    if missing:
        raise ValueError(
            "flow checkpoint contract missing: "
            + ", ".join(sorted(missing))
        )
    if not pitch_conditioning:
        unexpected = _PITCH_CONTRACT_HASHES & set(contract)
        if unexpected:
            raise ValueError(
                "pure flow checkpoint contract contains pitch hashes"
            )
    for name in _BASE_CONTRACT_HASHES | (
        _PITCH_CONTRACT_HASHES if pitch_conditioning else set()
    ):
        if not _SHA256.fullmatch(str(contract[name])):
            raise ValueError(f"{name} must be a lowercase SHA-256")
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
        "pitch_conditioning": config.model.pitch_conditioning,
    }
    pitch_hashes = (
        pitch_checkpoint_sha256,
        pitch_qualification_sha256,
    )
    if config.model.pitch_conditioning:
        if any(value is None for value in pitch_hashes):
            raise ValueError(
                "pitch-conditioned flow requires both pitch hashes"
            )
        contract["pitch_checkpoint_sha256"] = pitch_checkpoint_sha256
        contract["pitch_qualification_sha256"] = (
            pitch_qualification_sha256
        )
    elif any(value is not None for value in pitch_hashes):
        raise ValueError("pure flow contract must not contain pitch hashes")
    _validate_checkpoint_contract(contract)
    return contract


def _checkpoint_architecture(contract: dict[str, object]) -> str:
    return (
        "zrave_conditional_flow_transformer_v1"
        if contract.get("pitch_conditioning", True)
        else "zrave_pure_flow_transformer_v1"
    )


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
    latest_gate_report_sha256: str | None = None,
    consecutive_gate_passes: int = 0,
) -> None:
    _validate_checkpoint_contract(contract)
    if update < 0 or consecutive_gate_passes < 0:
        raise ValueError("checkpoint counters must be non-negative")
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
        sampler_states: list[object] | None = (
            [None] * world_size if rank == 0 else None
        )
        rng_states: list[object] | None = (
            [None] * world_size if rank == 0 else None
        )
        dist.gather_object(local_sampler, sampler_states, dst=0)
        dist.gather_object(local_rng, rng_states, dst=0)
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
    if contract.get("pitch_conditioning", True):
        if pitch_weight_controller is None:
            raise ValueError(
                "conditional checkpoint requires pitch controller"
            )
        payload["pitch_weight_controller"] = (
            pitch_weight_controller.state_dict()
        )
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
) -> dict[str, int | str | None]:
    _validate_checkpoint_contract(expected_contract)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("flow checkpoint must be a mapping")
    if payload.get("format") != 1:
        raise ValueError("unsupported flow checkpoint format")
    if payload.get("architecture") != _checkpoint_architecture(
        expected_contract
    ):
        raise ValueError("flow checkpoint architecture mismatch")
    actual_contract = payload.get("contract")
    if not isinstance(actual_contract, dict):
        raise ValueError("flow checkpoint has no contract")
    for name, expected in expected_contract.items():
        if actual_contract.get(name) != expected:
            raise ValueError(f"flow checkpoint {name} mismatch")
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
            raise ValueError(
                "conditional checkpoint requires pitch controller"
            )
        controller_state = payload.get("pitch_weight_controller")
        if not isinstance(controller_state, dict):
            raise ValueError("flow checkpoint lacks pitch controller state")
        pitch_weight_controller.load_state_dict(controller_state)
    gate_hash = payload.get("latest_gate_report_sha256")
    if gate_hash is not None and (
        not isinstance(gate_hash, str)
        or not _SHA256.fullmatch(gate_hash)
    ):
        raise ValueError("flow checkpoint gate report hash is invalid")
    return {
        "update": int(payload["update"]),
        "latest_gate_report_sha256": gate_hash,
        "consecutive_gate_passes": int(
            payload.get("consecutive_gate_passes", 0)
        ),
    }


def _load_statistics(root: Path) -> FlowStatistics:
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
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
    return floor + 0.5 * (peak - floor) * (
        1.0 + math.cos(math.pi * progress)
    )


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
    )


def _pitch_diagnostics(
    pair: Any,
    predicted_velocity: Tensor,
    midi_note: Tensor,
    transition_mask: Tensor,
    pitch_probe: LatentPitchProbe,
    statistics: FlowStatistics,
) -> dict[str, float]:
    with torch.no_grad():
        normalized = pair.noisy_future + (
            1.0 - pair.flow_time[:, None, None]
        ) * predicted_velocity.detach().float()
        estimate = (
            normalized
            * statistics.latent_std.to(normalized.device)
            + statistics.mean.to(normalized.device)
        )
        output = pitch_probe(estimate[:, :16])
        predicted = output.logits.argmax(dim=-1) + pitch_probe.note_min
        cents = (
            output.expected_midi.float() - midi_note.float()
        ).abs() * 100.0
        values = torch.zeros(6, device=estimate.device, dtype=torch.float64)
        for offset, mask in (
            (0, ~transition_mask),
            (3, transition_mask),
        ):
            values[offset] = mask.sum()
            values[offset + 1] = (
                (predicted == midi_note) & mask
            ).sum()
            values[offset + 2] = cents.masked_select(mask).double().sum()
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        matched_count = max(1.0, float(values[0]))
        transition_count = max(1.0, float(values[3]))
        return {
            "pitch_matched_accuracy": float(values[1]) / matched_count,
            "pitch_matched_mean_absolute_cents": (
                float(values[2]) / matched_count
            ),
            "pitch_transition_accuracy": (
                float(values[4]) / transition_count
            ),
            "pitch_transition_mean_absolute_cents": (
                float(values[5]) / transition_count
            ),
        }


@dataclass(frozen=True)
class FlowUpdateResult:
    loss: float
    components: dict[str, float]
    pitch_weight: float
    gradient_norm: float
    global_valid_frames: int
    duration_seconds: float
    pitch_metrics: dict[str, float]
    exposure: bool


def _all_rank_finite(value: Tensor) -> bool:
    flag = torch.tensor(
        int(not torch.isfinite(value)),
        device=value.device,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item()) == 0


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
) -> FlowUpdateResult:
    generator = _default_generator(device)
    exposure = (
        use_exposure_batch(
            update,
            maximum_updates,
            config.train.exposure_start_fraction,
            config.train.exposure_probability,
            generator,
        )
        if force_exposure is None
        else force_exposure
    )
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
        batch = _prepare_exposure_batch(
            _unwrapped(training_model),
            batch,
            generation_seed=config.seed + update,
            block_index=rank,
        )
    temperatures = 0.7 + 0.6 * torch.rand(
        batch_per_gpu,
        device=device,
    )
    pair = make_flow_training_pair(
        batch.future,
        batch.future_mask,
        temperatures,
        batch.wander_delay_frames,
        statistics,
        generator,
    )
    next_update = update + 1
    learning_rate = _learning_rate(
        config,
        next_update,
        maximum_updates,
    )
    _set_learning_rate(optimizer, learning_rate)
    optimizer.zero_grad(set_to_none=True)
    retention = retention_curve(batch.wander_delay_frames, 64)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        if config.model.pitch_conditioning:
            if pitch_probe is None or controller is None:
                raise ValueError(
                    "conditional flow update requires pitch runtime"
                )
            dropout = config.model.condition_dropout
            pitch_present = (
                torch.rand(batch_per_gpu, device=device) >= dropout
            )
            context_present = (
                torch.rand(batch_per_gpu, device=device) >= dropout
            )
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
                batch.midi_note,
                pitch_probe,
                statistics,
                pitch_weight,
            )
        else:
            predicted_velocity = training_model(
                pair.noisy_future,
                pair.flow_time,
                batch.history,
                retention=retention,
                future_mask=batch.future_mask,
            )
            pitch_weight = 0.0
            report = zrave_pure_flow_loss(
                predicted_velocity,
                pair,
                batch.history,
                batch.future_mask,
                statistics,
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
    if not _all_rank_finite(gradient_norm):
        raise FloatingPointError("non-finite flow gradient")
    scaler.step(optimizer)
    scaler.update()
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
    pitch_metrics = (
        _pitch_diagnostics(
            pair,
            predicted_velocity,
            batch.midi_note,
            batch.pitch_transition_mask,
            pitch_probe,
            statistics,
        )
        if pitch_probe is not None
        else {}
    )
    return FlowUpdateResult(
        loss=float(report.total.detach()),
        components={
            name: float(value.detach())
            for name, value in report.components.items()
        },
        pitch_weight=pitch_weight,
        gradient_norm=float(gradient_norm),
        global_valid_frames=global_frames,
        duration_seconds=duration,
        pitch_metrics=pitch_metrics,
        exposure=exposure,
    )


@torch.no_grad()
def _validation_snapshot(
    *,
    model: nn.Module,
    sampler: GpuFlowSampler,
    pitch_probe: LatentPitchProbe,
    statistics: FlowStatistics,
    config: ZraveFlowConfig,
    batch_per_gpu: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    sampler_state = sampler.state_dict()
    rng_state = _capture_rng_state()
    was_training = model.training
    model.eval()
    batch = sampler.sample(
        batch_per_gpu,
        maximum_valid_future=64,
    )
    pair = make_flow_training_pair(
        batch.future,
        batch.future_mask,
        torch.ones(batch_per_gpu, device=device),
        batch.wander_delay_frames,
        statistics,
        _default_generator(device),
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        velocity = model(
            pair.noisy_future,
            pair.flow_time,
            batch.history,
            batch.midi_note,
            retention_curve(batch.wander_delay_frames, 64),
            future_mask=batch.future_mask,
        )
        report = zrave_flow_loss(
            velocity,
            pair,
            batch.history,
            batch.future_mask,
            batch.midi_note,
            pitch_probe,
            statistics,
            pitch_weight=0.30,
        )
    values = torch.tensor(
        [
            float(report.total),
            *(
                float(report.components[name])
                for name in ("flow", "pitch", "boundary", "statistics")
            ),
        ],
        device=device,
        dtype=torch.float64,
    )
    normalized = pair.noisy_future + (
        1.0 - pair.flow_time[:, None, None]
    ) * velocity.float()
    valid = batch.future_mask.unsqueeze(-1).expand_as(normalized)
    diversity = normalized.masked_select(valid).std(unbiased=False)
    diversity_values = torch.tensor(
        [float(diversity)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(diversity_values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
        diversity_values /= dist.get_world_size()
    sampler.load_state_dict(sampler_state)
    _restore_rng_state(rng_state)
    if was_training:
        model.train()
    validation = dict(
        zip(
            ("total", "flow", "pitch", "boundary", "statistics"),
            values.cpu().tolist(),
            strict=True,
        )
    )
    return validation, {
        "normalized_clean_std": float(diversity_values.item())
    }


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
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    checkpoint_value = qualification.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("pitch qualification lacks checkpoint")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = qualification_path.parent / checkpoint
    return qualification_path, checkpoint, qualification


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
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
) -> dict[str, object]:
    for name, digest in (
        ("config_sha256", config_sha256),
        ("pack_index_sha256", pack_index_sha256),
    ):
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if (
        pitch_checkpoint_sha256 is not None
        and not _SHA256.fullmatch(pitch_checkpoint_sha256)
    ):
        raise ValueError(
            "pitch_checkpoint_sha256 must be a lowercase SHA-256"
        )
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
        and exposure_safety_updates == 5
        and nonfinite_updates == 0
        and math.isfinite(peak_memory_mib)
        and 0.0 < peak_memory_mib < total_memory_mib
    )
    windows_per_second = (
        global_batch / durations
        if durations.size
        else np.asarray([], dtype=np.float64)
    )
    frames_per_second = (
        frames / durations
        if durations.size
        else np.asarray([], dtype=np.float64)
    )
    report: dict[str, object] = {
        "status": "ok" if valid else "invalid",
        "batch_per_gpu": int(batch_per_gpu),
        "world_size": int(world_size),
        "global_batch": int(global_batch),
        "measured_updates": int(durations.size),
        "exposure_safety_updates": int(exposure_safety_updates),
        "median_windows_per_second": (
            float(np.median(windows_per_second))
            if windows_per_second.size
            else 0.0
        ),
        "median_valid_latent_frames_per_second": float(
            np.median(frames_per_second)
            if frames_per_second.size
            else 0.0
        ),
        "p10_windows_per_second": (
            float(np.percentile(windows_per_second, 10))
            if windows_per_second.size
            else 0.0
        ),
        "peak_memory_mib": float(peak_memory_mib),
        "total_memory_mib": float(total_memory_mib),
        "nonfinite_updates": int(nonfinite_updates),
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
            raise ValueError(
                "conditional flow config requires --pitch-probe"
            )
        qualification_path, pitch_checkpoint, _qualification = (
            _qualification_artifacts(args.pitch_probe)
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
    validation_sampler = (
        GpuFlowSampler.from_pack(
            config,
            split="validation",
            device=device,
            seed=config.seed + 4000 + rank,
        )
        if config.model.pitch_conditioning
        else None
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


def _log_update(
    writer: SummaryWriter,
    result: FlowUpdateResult,
    update: int,
) -> None:
    writer.add_scalar("train/loss", result.loss, update)
    for name, value in result.components.items():
        writer.add_scalar(f"train/{name}", value, update)
    for name, value in result.pitch_metrics.items():
        writer.add_scalar(f"train/{name}", value, update)
    writer.add_scalar(
        "train/valid_latent_frames_per_second",
        result.global_valid_frames / result.duration_seconds,
        update,
    )
    if result.pitch_metrics or result.pitch_weight:
        writer.add_scalar(
            "train/pitch_weight",
            result.pitch_weight,
            update,
        )
    writer.add_scalar(
        "health/gradient_norm",
        result.gradient_norm,
        update,
    )
    writer.add_scalar(
        "health/exposure_batch",
        float(result.exposure),
        update,
    )


def _benchmark(args: argparse.Namespace) -> None:
    runtime = _build_runtime(args)
    config: ZraveFlowConfig = runtime["config"]
    rank: int = runtime["rank"]
    device: torch.device = runtime["device"]
    warmup = int(args.benchmark_warmup)
    measured = int(args.benchmark_updates)
    exposure_updates = int(args.benchmark_exposure_updates)
    if warmup < 0 or measured <= 0 or exposure_updates < 0:
        raise ValueError("invalid benchmark update counts")
    update = 0
    for _ in range(warmup):
        _run_flow_update(
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
        update += 1
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    frames: list[int] = []
    nonfinite = 0
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
        durations.append(result.duration_seconds)
        frames.append(result.global_valid_frames)
        update += 1
    completed_exposure = 0
    for _ in range(exposure_updates):
        _run_flow_update(
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
        completed_exposure += 1
        update += 1
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
        pack_index_sha256=str(
            runtime["contract"]["pack_index_sha256"]
        ),
        pitch_checkpoint_sha256=(
            str(runtime["contract"]["pitch_checkpoint_sha256"])
            if "pitch_checkpoint_sha256" in runtime["contract"]
            else None
        ),
        git_commit=_git_commit(commit_root),
    )
    if rank == 0:
        _atomic_json(Path(args.benchmark_report), report)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _train(args: argparse.Namespace) -> None:
    runtime = _build_runtime(args)
    config: ZraveFlowConfig = runtime["config"]
    rank: int = runtime["rank"]
    update = 0
    latest_gate_hash: str | None = None
    consecutive_gate_passes = 0
    if args.resume:
        restored = load_flow_checkpoint(
            args.resume,
            model=runtime["training_model"],
            optimizer=runtime["optimizer"],
            scaler=runtime["scaler"],
            sampler=runtime["train_sampler"],
            pitch_weight_controller=runtime["controller"],
            expected_contract=runtime["contract"],
        )
        update = int(restored["update"])
        latest_gate_hash = restored["latest_gate_report_sha256"]
        consecutive_gate_passes = int(
            restored["consecutive_gate_passes"]
        )
    gate_state = GateEarlyStopState(
        required_consecutive_passes=config.train.early_stop_gate_passes,
        consecutive_passes=consecutive_gate_passes,
    )
    output_root = Path(config.train.output_root)
    checkpoint_root = output_root / "checkpoints"
    writer = (
        SummaryWriter(str(output_root / "tensorboard"))
        if rank == 0
        else None
    )
    stopped_early = False
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
        )
        update += 1
        if writer is not None and (
            update == 1 or update % config.train.log_every == 0
        ):
            _log_update(writer, result, update)
        if (
            config.model.pitch_conditioning
            and update % config.train.validation_every == 0
        ):
            if (
                runtime["validation_sampler"] is None
                or runtime["pitch_probe"] is None
                or runtime["controller"] is None
            ):
                raise RuntimeError(
                    "conditional validation runtime is incomplete"
                )
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
        final_due = update == runtime["maximum_updates"]
        checkpoint_due = (
            update % config.train.checkpoint_every == 0 or final_due
        )
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
                latest_gate_report_sha256=latest_gate_hash,
                consecutive_gate_passes=gate_state.consecutive_passes,
            )
            if dist.is_initialized():
                dist.barrier()
            if not config.model.pitch_conditioning:
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
                        output_root
                        / "evaluations"
                        / f"update-{update:08d}"
                    )
                    evaluation = evaluate_flow_checkpoint(
                        config,
                        checkpoint,
                        split="validation",
                        output_root=evaluation_root,
                        device=runtime["device"],
                        model=_unwrapped(runtime["training_model"]),
                    )
                    stopped_early = gate_state.update(
                        evaluation["gate"]
                    )
                    latest_gate_hash = _sha256_file(
                        evaluation_root / "evaluation.json"
                    )
                    evaluation_message[0] = {
                        "latest_gate_report_sha256": latest_gate_hash,
                        "consecutive_gate_passes": (
                            gate_state.consecutive_passes
                        ),
                        "stopped_early": stopped_early,
                        "gate": evaluation["gate"],
                    }
                except Exception as error:
                    evaluation_message[0] = {
                        "error": (
                            f"{type(error).__name__}: {error}"
                        )
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
                    "checkpoint evaluation failed: "
                    + str(message["error"])
                )
            latest_gate_hash = str(
                message["latest_gate_report_sha256"]
            )
            gate_state.consecutive_passes = int(
                message["consecutive_gate_passes"]
            )
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
                latest_gate_report_sha256=latest_gate_hash,
                consecutive_gate_passes=gate_state.consecutive_passes,
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
    parser.add_argument("--resume")
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
