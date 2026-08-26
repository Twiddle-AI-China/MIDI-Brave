from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import shutil
import socket
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from .atlas_flow_atlas import TimbreAtlas, explained_variance_ratio
from .atlas_flow_config import AtlasFlowConfig, load_atlas_flow_config
from .atlas_flow_data import (
    AtlasAudioDataset, CachedFlowDataset, as_flow_batch, cache_feature,
    feature_path, manifest_audit, trajectory_path,
)
from .atlas_flow_loss import atlas_flow_matching_loss
from .atlas_flow_model import AtlasFlowSystem
from .atlas_flow_objective import AtlasStage1Objective
from .data import SampleRecord, load_manifest


_STOP_REQUESTED = False
_LAST_NONFINITE_GRADIENT: dict[str, object] | None = None


class NumericRecoveryRequired(RuntimeError):
    """Signal that the launcher must roll back and select a safer profile."""


@dataclass(frozen=True)
class RecoveryProfile:
    name: str
    use_amp: bool
    flow_lr_multiplier: float
    joint_lr_multiplier: float
    grad_clip: float
    amp_initial_scale: float
    amp_max_scale: float
    amp_growth_interval: int

    def learning_rate_multiplier(self, stage: str) -> float:
        if stage == "flow":
            return self.flow_lr_multiplier
        if stage == "joint":
            return self.joint_lr_multiplier
        return 1.0


RECOVERY_PROFILES: dict[str, RecoveryProfile] = {
    "safe_amp": RecoveryProfile(
        "safe_amp", True, 0.5, 1.0, 1.0, 128.0, 128.0, 1_000_000,
    ),
    "fp32_rescue": RecoveryProfile(
        "fp32_rescue", False, 0.25, 0.5, 1.0, 1.0, 1.0, 1_000_000,
    ),
    "fp32_low_lr": RecoveryProfile(
        "fp32_low_lr", False, 0.125, 0.25, 0.5, 1.0, 1.0, 1_000_000,
    ),
}


def _recovery_profile(name: str | None) -> RecoveryProfile:
    selected = str(name or "safe_amp")
    try:
        return RECOVERY_PROFILES[selected]
    except KeyError as exception:
        raise ValueError(f"unknown recovery profile: {selected}") from exception


def _request_graceful_stop(_signum: int, _frame: object) -> None:
    """Request a checkpoint at the next completed atomic optimizer update."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local_rank


def _destroy_distributed() -> None:
    """Let every rank finish host-side reporting before tearing NCCL down."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _capture_rank_runtime_state(micro_batches_consumed: int) -> dict[str, object]:
    return {
        "micro_batches_consumed": int(micro_batches_consumed),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def _restore_rank_runtime_state(state: dict[str, object], device: torch.device) -> int:
    random.setstate(state["python_rng"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy_rng"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_rng"])  # type: ignore[arg-type]
    torch.cuda.set_rng_state(state["cuda_rng"], device=device)  # type: ignore[arg-type]
    return int(state["micro_batches_consumed"])


def _gather_rank_runtime_states(
    state: dict[str, object], world: int,
) -> list[dict[str, object]]:
    if not dist.is_initialized():
        return [state]
    gathered: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(gathered, state)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather every rank runtime state")
    return [item for item in gathered if item is not None]


def _resume_position(total_batches: int, batches_per_epoch: int) -> tuple[int, int]:
    if total_batches < 0 or batches_per_epoch < 1:
        raise ValueError("resume batch counts must be non-negative with a positive epoch")
    return divmod(total_batches, batches_per_epoch)


def _infinite(loader: DataLoader[object], start_batches: int = 0) -> Iterator[object]:
    epoch, offset = _resume_position(start_batches, len(loader))
    while True:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        yield from iterator
        epoch += 1
        offset = 0


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _gradient_buckets(
    gradients: Iterable[Tensor], bucket_cap_bytes: int,
) -> list[list[Tensor]]:
    """Group dense gradients without mixing devices or dtypes."""
    if bucket_cap_bytes < 1:
        raise ValueError("bucket_cap_bytes must be positive")
    buckets: list[list[Tensor]] = []
    current: list[Tensor] = []
    current_bytes = 0
    current_key: tuple[torch.device, torch.dtype] | None = None
    for gradient in gradients:
        if gradient.is_sparse:
            if current:
                buckets.append(current)
                current = []
                current_bytes = 0
                current_key = None
            buckets.append([gradient])
            continue
        key = (gradient.device, gradient.dtype)
        size = gradient.numel() * gradient.element_size()
        if current and (key != current_key or current_bytes + size > bucket_cap_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(gradient)
        current_bytes += size
        current_key = key
    if current:
        buckets.append(current)
    return buckets


def _sync_gradients(module: nn.Module, world: int) -> None:
    if world == 1:
        return
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    mode = os.environ.get("MIDIBRAVE_GRAD_SYNC", "per_parameter").strip().lower()
    if mode == "per_parameter":
        for gradient in gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world)
        return
    if mode != "bucketed":
        raise ValueError(f"unsupported MIDIBRAVE_GRAD_SYNC mode: {mode}")
    bucket_cap_mb = float(os.environ.get("MIDIBRAVE_GRAD_BUCKET_MB", "25"))
    buckets = _gradient_buckets(gradients, max(1, int(bucket_cap_mb * 2**20)))
    for bucket in buckets:
        if bucket[0].is_sparse:
            dist.all_reduce(bucket[0], op=dist.ReduceOp.SUM)
            bucket[0].div_(world)
            continue
        flattened = torch.cat([gradient.reshape(-1) for gradient in bucket])
        dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
        flattened.div_(world)
        offset = 0
        for gradient in bucket:
            elements = gradient.numel()
            gradient.copy_(flattened.narrow(0, offset, elements).view_as(gradient))
            offset += elements


def _all_finite(module: nn.Module, device: torch.device) -> bool:
    # Keep every test on-device and synchronize with the host only once. The
    # previous bool(tensor) expression introduced one CUDA synchronization per
    # parameter on every optimizer update.
    flag = torch.ones((), dtype=torch.int32, device=device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            flag = torch.minimum(
                flag, torch.isfinite(parameter.grad).all().to(dtype=torch.int32),
            )
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _nonfinite_gradient_report(module: nn.Module) -> dict[str, object]:
    bad: list[dict[str, object]] = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        mask = torch.isfinite(gradient)
        if bool(mask.all().item()):
            continue
        finite_values = gradient[mask].float()
        bad.append({
            "name": name,
            "nonfinite_elements": int((~mask).sum().item()),
            "finite_maximum_absolute": (
                float(finite_values.abs().max().item()) if finite_values.numel() else None
            ),
        })
        if len(bad) >= 16:
            break
    return {"parameters": bad}


def _module_gradient_norms(system: AtlasFlowSystem) -> dict[str, float]:
    groups: dict[str, nn.Module] = {
        "encoder": system.instrument.encoder,
        "adapter": system.instrument.adapter,
        "decoder": system.instrument.decoder,
        "pitch_conditioner": system.instrument.pitch_conditioner,
        "output_gain": system.instrument.output_gain,
        "pitch_adversary": system.instrument.pitch_adversary,
        "flow": system.flow,
    }
    names: list[str] = []
    values: list[Tensor] = []
    zero = torch.zeros((), dtype=torch.float32, device=next(system.parameters()).device)
    for name, module in groups.items():
        squares = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
        names.append(name)
        values.append(torch.stack(squares).sum().sqrt() if squares else zero)
    numbers = torch.stack(values).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, numbers)}


def _write_tensorboard_record(writer: SummaryWriter, record: dict[str, object]) -> None:
    """Write one JSONL-compatible metric record to TensorBoard."""
    step = int(record["update"])
    writer.add_scalar("loss/total", float(record["loss"]), step)
    for name, value in dict(record.get("components", {})).items():
        writer.add_scalar(f"loss/components/{name}", float(value), step)
    for name, value in dict(record.get("module_gradient_norms", {})).items():
        writer.add_scalar(f"gradient/modules/{name}", float(value), step)
    scalar_tags = {
        "gradient/global_norm": "gradient_norm",
        "optimizer/learning_rate": "learning_rate",
        "optimizer/amp_scale": "amp_scale",
        "stability/nonfinite_skips": "nonfinite_skips",
        "stability/consecutive_nonfinite_skips": "consecutive_nonfinite_skips",
        "runtime/updates_per_second": "updates_per_second",
        "runtime/peak_memory_mib": "peak_memory_mib",
        "runtime/peak_memory_reserved_mib": "peak_memory_reserved_mib",
        "data/micro_batches_consumed": "micro_batches_consumed",
        "recovery/profile_tier": "recovery_profile_tier",
        "recovery/healthy_steps": "healthy_steps_since_event",
    }
    for tag, key in scalar_tags.items():
        value = record.get(key)
        if value is not None:
            writer.add_scalar(tag, float(value), step)
    writer.flush()


def _state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _cgroup_memory_limit_bytes() -> int | None:
    for candidate in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value and value != "max":
            limit = int(value)
            # cgroup v1 reports a near-uint64 maximum when no limit is set.
            if limit < 2**60:
                return limit
    return None


def _system_available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _save_checkpoint(
    path: Path,
    system: AtlasFlowSystem,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    stage: str,
    update: int,
    config: AtlasFlowConfig,
    nonfinite_skips: int,
    consecutive_nonfinite_skips: int,
    batch_contract: dict[str, object] | None = None,
    run_identity: dict[str, object] | None = None,
    rank_runtime_states: list[dict[str, object]] | None = None,
    recovery_profile: str | None = None,
    stability_state: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    state_hash = _state_sha256(system)
    torch.save({
        "schema": "midibrave.atlas-flow.checkpoint.v1",
        "stage": stage,
        "update": update,
        "system": system.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": {
            "data": asdict(config.data), "model": asdict(config.model),
            "loss": asdict(config.loss), "train": asdict(config.train),
        },
        "parameter_sha256": state_hash,
        "nonfinite_skips": nonfinite_skips,
        "consecutive_nonfinite_skips": consecutive_nonfinite_skips,
        # Optional additions keep every checkpoint written before Spark
        # migration readable without a schema conversion.
        "batch_contract": batch_contract,
        "run_identity": run_identity,
        "rank_runtime_states": rank_runtime_states,
        "recovery_profile": recovery_profile,
        "stability_state": stability_state,
    }, temporary)
    temporary.replace(path)
    latest = path.parent / "latest.pt"
    latest_tmp = path.parent / "latest.pt.tmp"
    shutil.copyfile(path, latest_tmp)
    latest_tmp.replace(latest)


def _prune_recovery_checkpoints(directory: Path, keep: int = 3) -> None:
    candidates = sorted(directory.glob("step-*.pt"))
    for stale in candidates[:-keep]:
        stale.unlink()


def _load_weights(path: str | Path, system: AtlasFlowSystem) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "midibrave.atlas-flow.checkpoint.v1":
        raise ValueError("checkpoint schema mismatch")
    system.load_state_dict(checkpoint["system"], strict=True)
    return checkpoint


def _learning_rate(stage: str, config: AtlasFlowConfig) -> float:
    return {
        "stage1": config.train.stage1_lr,
        "flow": config.train.flow_lr,
        "joint": config.train.joint_lr,
    }[stage]


def _maximum_updates(stage: str, config: AtlasFlowConfig) -> int:
    return {
        "stage1": config.train.stage1_updates,
        "flow": config.train.flow_updates,
        "joint": config.train.joint_updates,
    }[stage]


def _scheduled_learning_rate(
    stage: str,
    update: int,
    maximum: int,
    config: AtlasFlowConfig,
    profile: RecoveryProfile | None = None,
) -> float:
    multiplier = profile.learning_rate_multiplier(stage) if profile is not None else 1.0
    peak = _learning_rate(stage, config) * multiplier
    minimum = config.train.minimum_lr * multiplier
    if update < config.train.warmup_updates:
        return peak * (update + 1) / config.train.warmup_updates
    progress = (update - config.train.warmup_updates) / max(maximum - config.train.warmup_updates, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum + (peak - minimum) * cosine


def _make_scaler(profile: RecoveryProfile) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        "cuda",
        enabled=profile.use_amp,
        init_scale=profile.amp_initial_scale,
        growth_interval=profile.amp_growth_interval,
    )


def _restore_scaler(
    scaler: torch.amp.GradScaler,
    checkpoint_state: dict[str, object] | None,
    profile: RecoveryProfile,
) -> None:
    if not profile.use_amp or not checkpoint_state:
        return
    state = dict(checkpoint_state)
    state["scale"] = min(float(state.get("scale", profile.amp_initial_scale)), profile.amp_max_scale)
    state["growth_interval"] = profile.amp_growth_interval
    state["_growth_tracker"] = 0
    scaler.load_state_dict(state)


def _numeric_recovery_reason(
    *,
    finite: bool,
    consecutive_nonfinite: int,
    recent_nonfinite: Iterable[bool],
    gradient_norm: float,
    recent_gradient_norms: Iterable[float],
    warmup_complete: bool = True,
    repeated_large_gradient_threshold: float = 100.0,
) -> str | None:
    if not finite and consecutive_nonfinite >= 2:
        return "two_consecutive_nonfinite_gradients"
    if sum(not value for value in recent_nonfinite) >= 4:
        return "four_nonfinite_gradients_in_200_attempts"
    if math.isfinite(gradient_norm) and gradient_norm > 1_000.0:
        return "gradient_norm_above_1000"
    # Raw (pre-clip) gradient spikes are expected while the joint optimizer is
    # linearly warming up.  Escalating precision before a complete post-warmup
    # observation window exists can consume the V100 memory headroom without
    # fixing an actual numerical problem.  The non-finite and >1000 hard guards
    # remain active throughout warmup.
    if (
        warmup_complete
        and sum(
            value > repeated_large_gradient_threshold
            for value in recent_gradient_norms
            if math.isfinite(value)
        ) >= 4
    ):
        threshold = f"{repeated_large_gradient_threshold:g}"
        return f"four_gradient_norms_above_{threshold}_in_50_updates"
    return None


def _is_out_of_memory_error(exception: BaseException) -> bool:
    """Recognize native and NCCL-wrapped CUDA OOM failures."""
    message = str(exception).lower()
    return isinstance(exception, torch.OutOfMemoryError) or any(
        marker in message for marker in ("out of memory", "cuda failure 2")
    )


def _update_nonfinite_counters(
    *, finite: bool, total: int, consecutive: int, maximum_consecutive: int = 8,
) -> tuple[int, int, bool]:
    """Track recoverable AMP skips without treating sparse events as divergence."""
    if finite:
        return total, 0, False
    total += 1
    consecutive += 1
    return total, consecutive, consecutive >= maximum_consecutive


def _batch_contract(stage: str, micro_batch: int, grad_accum: int, world: int) -> dict[str, object]:
    if micro_batch < 1 or grad_accum < 1 or world < 1:
        raise ValueError("micro_batch, grad_accum, and world_size must be positive")
    audio_micro: int | None = None
    flow_micro: int | None = None
    if stage == "stage1":
        audio_micro = micro_batch
    elif stage == "flow":
        flow_micro = micro_batch
    elif stage == "joint":
        if micro_batch % 4:
            raise ValueError("joint batch_per_gpu must be divisible by 4")
        audio_micro = micro_batch // 2
        flow_micro = micro_batch // 4
    else:
        raise ValueError(f"unknown training stage: {stage}")
    return {
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "world_size": world,
        "audio_micro_batch": audio_micro,
        "flow_micro_batch": flow_micro,
        "audio_global_batch": None if audio_micro is None else audio_micro * grad_accum * world,
        "flow_global_batch": None if flow_micro is None else flow_micro * grad_accum * world,
    }


def _active_parameters(stage: str, system: AtlasFlowSystem) -> list[nn.Parameter]:
    # requires_grad remains True for every trainable module throughout the
    # experiment. Stage-local optimizers are compute routing, not freezing.
    if stage == "stage1":
        return list(system.instrument.parameters())
    if stage == "flow":
        return list(system.flow.parameters())
    return list(system.parameters())


def _audio_loader(config: AtlasFlowConfig, rank: int, world: int, batch_size: int) -> DataLoader[object]:
    dataset = AtlasAudioDataset(config.data, "train")
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=config.train.seed) if world > 1 else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=config.data.prefetch_factor, drop_last=True,
        generator=torch.Generator().manual_seed(config.train.seed + rank + 101),
    )


def _flow_loader(config: AtlasFlowConfig, rank: int, world: int, batch_size: int) -> DataLoader[object]:
    dataset = CachedFlowDataset(config.data, config.model, "train")
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=config.train.seed + 1) if world > 1 else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=config.data.num_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=config.data.prefetch_factor, drop_last=True,
        generator=torch.Generator().manual_seed(config.train.seed + rank + 202),
    )


def _stage_iterators(
    config: AtlasFlowConfig,
    stage: str,
    rank: int,
    world: int,
    micro_batch: int,
    start_batches: int = 0,
) -> tuple[Iterator[object] | None, Iterator[object] | None]:
    contract = _batch_contract(stage, micro_batch, 1, world)
    audio_iterator = None
    flow_iterator = None
    audio_micro = contract["audio_micro_batch"]
    flow_micro = contract["flow_micro_batch"]
    if isinstance(audio_micro, int):
        audio_iterator = _infinite(
            _audio_loader(config, rank, world, audio_micro), start_batches,
        )
    if isinstance(flow_micro, int):
        flow_iterator = _infinite(
            _flow_loader(config, rank, world, flow_micro), start_batches,
        )
    return audio_iterator, flow_iterator


def _stage_forward(
    *,
    stage: str,
    system: AtlasFlowSystem,
    config: AtlasFlowConfig,
    stage1_objective: AtlasStage1Objective,
    audio_iterator: Iterator[object] | None,
    flow_iterator: Iterator[object] | None,
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    metrics: dict[str, Tensor] = {}
    loss = torch.zeros((), device=device)
    if audio_iterator is not None:
        audio_batch = _to_device(next(audio_iterator), device)
        audio_result = stage1_objective(system.instrument, audio_batch)  # type: ignore[arg-type]
        loss = loss + audio_result.total
        metrics.update({f"audio/{key}": value for key, value in audio_result.components.items()})
        if stage == "joint":
            cached = []
            for sample_id in audio_batch["sample_id_a"]:  # type: ignore[index]
                record = SampleRecord(
                    sample_id=str(sample_id), audio_path="", source_id="", preset_id="",
                    articulation_id="", midi_note=0, midi_note_sent=None,
                    transpose_semitones=None, velocity=127, sample_rate=44_100,
                    num_samples=220_500, duration_seconds=5.0, a4_tuning_hz=440.0,
                    render_or_recording="render", render_gain_db=0.0,
                )
                cached.append(torch.from_numpy(
                    np.load(trajectory_path(record, config.data)).astype(np.float32)
                ))
            cached_value = torch.stack(cached).to(device)
            frames = min(cached_value.shape[-1], audio_result.trajectory_a.shape[-1])
            alignment = torch.nn.functional.smooth_l1_loss(
                audio_result.trajectory_a[..., :frames].float(), cached_value[..., :frames]
            )
            loss = loss + 0.10 * alignment
            metrics["joint/cache_alignment"] = alignment
    if flow_iterator is not None:
        flow_values = _to_device(next(flow_iterator), device)
        flow_result = atlas_flow_matching_loss(
            system.flow, as_flow_batch(flow_values), config.loss  # type: ignore[arg-type]
        )
        loss = loss + flow_result.total
        metrics.update({f"flow/{key}": value for key, value in flow_result.components.items()})
    return loss, metrics


def _atomic_update(
    *,
    stage: str,
    system: AtlasFlowSystem,
    config: AtlasFlowConfig,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: list[nn.Parameter],
    stage1_objective: AtlasStage1Objective,
    audio_iterator: Iterator[object] | None,
    flow_iterator: Iterator[object] | None,
    device: torch.device,
    world: int,
    grad_accum: int,
    profile: RecoveryProfile,
    collect_metrics: bool = True,
) -> tuple[bool, float, dict[str, float], float, dict[str, float]]:
    """Run one optimizer update composed of exactly ``grad_accum`` micros."""
    global _LAST_NONFINITE_GRADIENT
    _LAST_NONFINITE_GRADIENT = None
    optimizer.zero_grad(set_to_none=True)
    loss_sum: Tensor | None = None
    component_sums: dict[str, Tensor] = {}
    for _ in range(grad_accum):
        with torch.autocast("cuda", dtype=torch.float16, enabled=profile.use_amp):
            loss, metrics = _stage_forward(
                stage=stage, system=system, config=config,
                stage1_objective=stage1_objective,
                audio_iterator=audio_iterator, flow_iterator=flow_iterator,
                device=device,
            )
        if collect_metrics:
            detached_loss = loss.detach().float()
            loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
            for key, value in metrics.items():
                detached = value.detach().float()
                component_sums[key] = component_sums.get(key, torch.zeros_like(detached)) + detached
        scaler.scale(loss / grad_accum).backward()
    scaler.unscale_(optimizer)
    _sync_gradients(system, world)
    finite = _all_finite(system, device)
    mean_loss = float((loss_sum / grad_accum).item()) if loss_sum is not None else float("nan")
    mean_components = {
        key: float((value / grad_accum).item()) for key, value in component_sums.items()
    }
    if not finite:
        _LAST_NONFINITE_GRADIENT = _nonfinite_gradient_report(system)
        module_norms = _module_gradient_norms(system) if collect_metrics else {}
        return False, mean_loss, mean_components, float("nan"), module_norms
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, profile.grad_clip)
    module_norms = _module_gradient_norms(system) if collect_metrics else {}
    scaler.step(optimizer)
    scaler.update()
    gradient_norm_value = float(gradient_norm.item())
    return True, mean_loss, mean_components, gradient_norm_value, module_norms


def train_stage(args: argparse.Namespace) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    graceful_signal = getattr(signal, "SIGUSR1", None)
    if graceful_signal is not None:
        signal.signal(graceful_signal, _request_graceful_stop)
    config = load_atlas_flow_config(args.config)
    stage = str(args.stage)
    profile = _recovery_profile(args.recovery_profile)
    rank, world, local_rank = _distributed()
    _seed(config.train.seed, rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    maximum = int(args.max_updates or _maximum_updates(stage, config))
    micro_batch = int(args.batch_per_gpu or (
        config.train.flow_batch_per_gpu if stage == "flow" else config.train.batch_per_gpu
    ))
    grad_accum = int(args.grad_accum)
    batch_contract = _batch_contract(stage, micro_batch, grad_accum, world)
    system = AtlasFlowSystem(config.model, config.data).to(device)
    initialization: dict[str, object] | None = None
    if args.initialize:
        initialization = _load_weights(args.initialize, system)
    parameters = _active_parameters(stage, system)
    optimizer = torch.optim.AdamW(
        parameters, lr=_learning_rate(stage, config), betas=(0.9, 0.95),
        weight_decay=0.01, fused=True,
    )
    scaler = _make_scaler(profile)
    update = 0
    nonfinite_skips = 0
    consecutive_nonfinite_skips = 0
    micro_batches_consumed = 0
    restored: dict[str, object] | None = None
    if args.resume:
        restored = _load_weights(args.resume, system)
        if restored["stage"] != stage:
            raise ValueError("resume stage mismatch")
        optimizer.load_state_dict(restored["optimizer"])
        _restore_scaler(scaler, restored.get("scaler"), profile)  # type: ignore[arg-type]
        update = int(restored["update"])
        nonfinite_skips = int(restored.get("nonfinite_skips", 0))
        # v1 checkpoints written before this field tracked only lifetime skips.
        # Sparse historical skips must not be interpreted as one active streak.
        consecutive_nonfinite_skips = int(restored.get("consecutive_nonfinite_skips", 0))
        rank_states = restored.get("rank_runtime_states")
        if isinstance(rank_states, list) and len(rank_states) == world:
            micro_batches_consumed = _restore_rank_runtime_state(
                rank_states[rank], device,
            )
        else:
            # Legacy checkpoints did not persist the iterator cursor, but every
            # recoverable non-finite attempt still consumed one atomic batch.
            micro_batches_consumed = (update + nonfinite_skips) * grad_accum
    initial_update = update
    session_id = str(args.session_id or os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}")
    run_identity: dict[str, object] = {
        "host": socket.gethostname(),
        "session_id": session_id,
        "resume_update": initial_update,
        "recovery_profile": profile.name,
        "precision_mode": "amp_fp16" if profile.use_amp else "fp32",
    }
    output = Path(config.train.output_root) / stage
    tensorboard: SummaryWriter | None = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        parameter_report = {
                "all_requires_grad": all(parameter.requires_grad for parameter in system.parameters()),
                "modules": system.parameter_report(),
                "optimizer_parameters": sum(parameter.numel() for parameter in parameters),
                "initialization": str(args.initialize) if args.initialize else "from_scratch",
                "resume": str(args.resume) if args.resume else None,
                "batch_contract": batch_contract,
                "gradient_sync": os.environ.get("MIDIBRAVE_GRAD_SYNC", "per_parameter"),
                "gradient_bucket_mb": float(os.environ.get("MIDIBRAVE_GRAD_BUCKET_MB", "25")),
                "tensorboard_log_dir": str(output / "tensorboard"),
                "learning_rate_multiplier": profile.learning_rate_multiplier(stage),
                "gradient_clip": profile.grad_clip,
                "amp_max_scale": profile.amp_max_scale,
                **run_identity,
        }
        (output / "parameter-report.json").write_text(
            json.dumps(parameter_report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        tensorboard = SummaryWriter(
            log_dir=str(output / "tensorboard"),
            purge_step=initial_update if initial_update > 0 else None,
            max_queue=10,
            flush_secs=20,
        )
        tensorboard.add_text(
            "run/identity", json.dumps(parameter_report, indent=2, sort_keys=True), initial_update,
        )
    audio_iterator, flow_iterator = _stage_iterators(
        config, stage, rank, world, micro_batch, micro_batches_consumed,
    )
    stage1_objective = AtlasStage1Objective(config.loss)
    started = time.perf_counter()
    stopped_gracefully = False
    recent_nonfinite: deque[bool] = deque(maxlen=200)
    recent_gradient_norms: deque[float] = deque(maxlen=50)
    healthy_steps_since_event = 0
    if restored is not None and isinstance(restored.get("stability_state"), dict):
        stability_state = restored["stability_state"]
        recent_nonfinite.extend(bool(value) for value in stability_state.get("recent_nonfinite", []))
        recent_gradient_norms.extend(
            float(value) for value in stability_state.get("recent_gradient_norms", [])
        )
        healthy_steps_since_event = int(stability_state.get("healthy_steps_since_event", 0))
    while update < maximum:
        learning_rate = _scheduled_learning_rate(stage, update, maximum, config, profile)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        collect_metrics = rank == 0 and (
            update + 1 == 1 or (update + 1) % config.train.log_every == 0
        )
        finite, mean_loss, mean_components, gradient_norm, module_norms = _atomic_update(
            stage=stage, system=system, config=config, optimizer=optimizer,
            scaler=scaler, parameters=parameters, stage1_objective=stage1_objective,
            audio_iterator=audio_iterator, flow_iterator=flow_iterator,
            device=device, world=world, grad_accum=grad_accum,
            profile=profile,
            collect_metrics=collect_metrics,
        )
        micro_batches_consumed += grad_accum
        recent_nonfinite.append(finite)
        nonfinite_skips, consecutive_nonfinite_skips, _ = _update_nonfinite_counters(
            finite=finite,
            total=nonfinite_skips,
            consecutive=consecutive_nonfinite_skips,
            maximum_consecutive=2,
        )
        if finite:
            update += 1
            recent_gradient_norms.append(gradient_norm)
            healthy_steps_since_event = (
                healthy_steps_since_event + 1 if gradient_norm <= 100.0 else 0
            )
        else:
            healthy_steps_since_event = 0
            optimizer.zero_grad(set_to_none=True)
            if profile.use_amp:
                scaler.update(max(scaler.get_scale() / 2.0, 1.0))
        recovery_reason = _numeric_recovery_reason(
            finite=finite,
            consecutive_nonfinite=consecutive_nonfinite_skips,
            recent_nonfinite=recent_nonfinite,
            gradient_norm=gradient_norm,
            recent_gradient_norms=recent_gradient_norms,
            warmup_complete=update >= (
                config.train.warmup_updates + (recent_gradient_norms.maxlen or 0)
            ),
            # Joint combines several audio objectives whose healthy pre-clip
            # norms are materially larger than the flow-only objective.
            repeated_large_gradient_threshold=500.0 if stage == "joint" else 100.0,
        )
        if recovery_reason is not None:
            event: dict[str, object] = {
                "schema": "midibrave.atlas-flow.numeric-recovery.v1",
                "stage": stage,
                "update": update,
                "reason": recovery_reason,
                "recovery_profile": profile.name,
                "precision_mode": "amp_fp16" if profile.use_amp else "fp32",
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm,
                "amp_scale": scaler.get_scale(),
                "nonfinite_skips": nonfinite_skips,
                "consecutive_nonfinite_skips": consecutive_nonfinite_skips,
                "micro_batches_consumed": micro_batches_consumed,
                "gradient_diagnostic": _LAST_NONFINITE_GRADIENT,
                **run_identity,
            }
            if rank == 0:
                _atomic_json(output / "numeric-recovery-required.json", event)
                _atomic_jsonl(output / "recovery-events.jsonl", event)
                if args.recovery_state:
                    _atomic_json(Path(args.recovery_state), {**event, "status": "recovery_required"})
                if tensorboard is not None:
                    tensorboard.add_scalar("recovery/event", 1.0, update)
                    tensorboard.add_scalar(
                        "recovery/profile_tier", list(RECOVERY_PROFILES).index(profile.name), update,
                    )
                    tensorboard.flush()
                    tensorboard.close()
            _destroy_distributed()
            raise NumericRecoveryRequired(recovery_reason)
        if not finite:
            continue
        if rank == 0 and (update == 1 or update % config.train.log_every == 0):
            elapsed = time.perf_counter() - started
            metric_record: dict[str, object] = {
                "stage": stage, "update": update, "loss": mean_loss,
                "components": mean_components,
                "gradient_norm": gradient_norm,
                "module_gradient_norms": module_norms,
                "amp_scale": scaler.get_scale(), "nonfinite_skips": nonfinite_skips,
                "consecutive_nonfinite_skips": consecutive_nonfinite_skips,
                "learning_rate": learning_rate,
                "updates_per_second": (update - initial_update) / max(elapsed, 1.0e-6),
                "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                "batch_contract": batch_contract,
                "micro_batches_consumed": micro_batches_consumed,
                "recovery_profile_tier": list(RECOVERY_PROFILES).index(profile.name),
                "healthy_steps_since_event": healthy_steps_since_event,
                **run_identity,
            }
            _atomic_jsonl(output / "metrics.jsonl", metric_record)
            if tensorboard is not None:
                _write_tensorboard_record(tensorboard, metric_record)
        stop_requested = _STOP_REQUESTED
        if dist.is_initialized():
            stop_flag = torch.tensor(int(stop_requested), device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            stop_requested = bool(stop_flag.item())
        due = update % config.train.checkpoint_every == 0 or update == maximum or stop_requested
        recovery_due = update % 500 == 0 and healthy_steps_since_event >= 500
        if due or recovery_due:
            if dist.is_initialized():
                dist.barrier()
            rank_runtime_states = _gather_rank_runtime_states(
                _capture_rank_runtime_state(micro_batches_consumed), world,
            )
            if rank == 0:
                stability_state = {
                    "healthy_steps_since_event": healthy_steps_since_event,
                    "recent_nonfinite": list(recent_nonfinite),
                    "recent_gradient_norms": list(recent_gradient_norms),
                }
                checkpoint_arguments = {
                    "stage": stage,
                    "update": update,
                    "config": config,
                    "nonfinite_skips": nonfinite_skips,
                    "consecutive_nonfinite_skips": consecutive_nonfinite_skips,
                    "batch_contract": batch_contract,
                    "run_identity": run_identity,
                    "rank_runtime_states": rank_runtime_states,
                    "recovery_profile": profile.name,
                    "stability_state": stability_state,
                }
                if recovery_due:
                    recovery_directory = output / "checkpoints" / "recovery"
                    _save_checkpoint(
                        recovery_directory / f"step-{update:09d}.pt",
                        system, optimizer, scaler, **checkpoint_arguments,
                    )
                    _prune_recovery_checkpoints(recovery_directory, keep=3)
                if due:
                    _save_checkpoint(
                        output / "checkpoints" / f"step-{update:09d}.pt",
                        system, optimizer, scaler, **checkpoint_arguments,
                    )
                if args.recovery_state:
                    _atomic_json(Path(args.recovery_state), {
                        "schema": "midibrave.atlas-flow.recovery-state.v1",
                        "status": "running",
                        "stage": stage,
                        "update": update,
                        "checkpoint": str(
                            output / "checkpoints" / "recovery" / "latest.pt"
                            if recovery_due else output / "checkpoints" / "latest.pt"
                        ),
                        "recovery_profile": profile.name,
                        "micro_batches_consumed": micro_batches_consumed,
                        **run_identity,
                    })
            if dist.is_initialized():
                dist.barrier()
        if stop_requested:
            stopped_gracefully = True
            if rank == 0:
                (output / "graceful-stop.json").write_text(
                    json.dumps({
                        "stage": stage, "update": update,
                        "checkpoint": str(output / "checkpoints" / f"step-{update:09d}.pt"),
                        "batch_contract": batch_contract, **run_identity,
                    }, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            break
    if rank == 0 and args.recovery_state:
        _atomic_json(Path(args.recovery_state), {
            "schema": "midibrave.atlas-flow.recovery-state.v1",
            "status": "paused" if stopped_gracefully else "complete",
            "stage": stage,
            "update": update,
            "recovery_profile": profile.name,
            "micro_batches_consumed": micro_batches_consumed,
            **run_identity,
        })
    if tensorboard is not None:
        tensorboard.close()
    _destroy_distributed()
    if stopped_gracefully:
        print(json.dumps({"graceful_stop": True, "stage": stage, "update": update}), flush=True)


def _cache_one(arguments: tuple[SampleRecord, object, bool]) -> str:
    record, data, overwrite = arguments
    return str(cache_feature(record, data, overwrite=overwrite))  # type: ignore[arg-type]


def preprocess(args: argparse.Namespace) -> None:
    config = load_atlas_flow_config(args.config)
    report = manifest_audit(config.data, check_audio=True)
    records = load_manifest(config.data.manifest)
    tasks = [(record, config.data, bool(args.overwrite)) for record in records]
    with ProcessPoolExecutor(max_workers=int(args.workers or config.data.num_workers)) as executor:
        for index, _ in enumerate(executor.map(_cache_one, tasks, chunksize=4), 1):
            if index % 100 == 0:
                print(json.dumps({"cached": index, "total": len(tasks)}), flush=True)
    destination = Path(config.data.feature_cache) / "preprocess-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({**report, "cached_features": len(tasks)}, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def cache_trajectories(args: argparse.Namespace) -> None:
    config = load_atlas_flow_config(args.config)
    rank, world, local_rank = _distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    system = AtlasFlowSystem(config.model, config.data).to(device)
    _load_weights(args.checkpoint, system)
    system.instrument.eval()
    records = load_manifest(config.data.manifest)
    selected = records[rank::world]
    batch_size = int(args.batch_size)
    for offset in range(0, len(selected), batch_size):
        chunk = selected[offset:offset + batch_size]
        features = torch.stack([
            torch.from_numpy(np.load(feature_path(record, config.data)).astype(np.float32))
            for record in chunk
        ]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            trajectories = system.instrument.encode(features).float().cpu().numpy()
        for record, trajectory in zip(chunk, trajectories):
            destination = trajectory_path(record, config.data)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, trajectory.astype(np.float16), allow_pickle=False)
            temporary.replace(destination)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        grouped: dict[str, list[np.ndarray]] = {}
        splits: dict[str, str] = {}
        for record in records:
            trajectory = np.load(trajectory_path(record, config.data)).astype(np.float32).T
            grouped.setdefault(record.preset_id, []).append(trajectory)
            splits[record.preset_id] = str(record.split)
        arrays = {key: np.stack(value) for key, value in grouped.items()}
        train_presets = [key for key, split in splits.items() if split == "train"]
        atlas = TimbreAtlas.fit(
            arrays, dimensions=config.model.atlas_dim, neighbors=config.model.atlas_neighbors,
            fit_presets=train_presets,
        )
        atlas.save(config.data.atlas_path)
        report = {
            "schema": "midibrave.atlas-flow.cache.v1",
            "trajectories": len(records), "presets": len(arrays),
            "pca_fit_presets": len(train_presets),
            "explained_variance_8d": explained_variance_ratio(
                atlas.anchors[[item in set(train_presets) for item in atlas.preset_ids]], atlas.components
            ),
        }
        Path(config.data.atlas_path).with_suffix(".report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _destroy_distributed()


def qualify(args: argparse.Namespace) -> None:
    """Exercise the exact real-data training path without formal run writes."""
    config = load_atlas_flow_config(args.config)
    stage = str(args.stage)
    profile = _recovery_profile(args.recovery_profile)
    rank, world, local_rank = _distributed()
    _seed(config.train.seed, rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.empty_cache()
    micro_batch = int(args.batch_per_gpu)
    grad_accum = int(args.grad_accum)
    contract = _batch_contract(stage, micro_batch, grad_accum, world)
    system = AtlasFlowSystem(config.model, config.data).to(device)
    restored = _load_weights(args.checkpoint, system)
    parameters = _active_parameters(stage, system)
    optimizer = torch.optim.AdamW(
        parameters, lr=_learning_rate(stage, config), betas=(0.9, 0.95),
        weight_decay=0.01, fused=True,
    )
    scaler = _make_scaler(profile)
    checkpoint_mode = "initialize"
    restored_update = 0
    if restored.get("stage") == stage:
        optimizer.load_state_dict(restored["optimizer"])
        _restore_scaler(scaler, restored.get("scaler"), profile)  # type: ignore[arg-type]
        restored_update = int(restored.get("update", 0))
        checkpoint_mode = "resume"
    start_batches = (
        restored_update + int(restored.get("nonfinite_skips", 0))
    ) * grad_accum
    rank_states = restored.get("rank_runtime_states")
    if isinstance(rank_states, list) and len(rank_states) == world:
        start_batches = _restore_rank_runtime_state(rank_states[rank], device)
    audio_iterator, flow_iterator = _stage_iterators(
        config, stage, rank, world, micro_batch, start_batches,
    )
    objective = AtlasStage1Objective(config.loss)
    passed = True
    error: str | None = None
    losses: list[float] = []
    gradient_norms: list[float] = []
    recent_gradient_norms: deque[float] = deque(maxlen=50)
    qualification_writer = (
        SummaryWriter(log_dir=str(args.tensorboard_log_dir), max_queue=10, flush_secs=20)
        if rank == 0 and args.tensorboard_log_dir else None
    )
    last_components: dict[str, float] = {}
    last_module_norms: dict[str, float] = {}
    last_gradient_norm = float("nan")
    warmup_requested = int(args.warmup_updates)
    measured_requested = int(args.updates)
    if warmup_requested < 0 or measured_requested < 1:
        raise ValueError("warmup_updates must be non-negative and updates must be positive")
    warmup_completed = 0
    amp_scale_start = float(scaler.get_scale())
    started: float | None = None
    cuda_free_before: int | None = None
    cuda_total = int(torch.cuda.get_device_properties(device).total_memory)
    if warmup_requested == 0:
        torch.cuda.reset_peak_memory_stats(device)
        cuda_free_before, cuda_total_runtime = torch.cuda.mem_get_info(device)
        cuda_total = int(cuda_total_runtime)
        started = time.perf_counter()
    try:
        for offset in range(warmup_requested + measured_requested):
            effective_update = restored_update + offset
            learning_rate = _scheduled_learning_rate(
                stage, effective_update, _maximum_updates(stage, config), config, profile,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            finite, loss, components, gradient_norm, module_norms = _atomic_update(
                stage=stage, system=system, config=config, optimizer=optimizer,
                scaler=scaler, parameters=parameters, stage1_objective=objective,
                audio_iterator=audio_iterator, flow_iterator=flow_iterator,
                device=device, world=world, grad_accum=grad_accum,
                profile=profile,
            )
            if not finite or not math.isfinite(loss) or not math.isfinite(gradient_norm):
                raise RuntimeError("non-finite loss or gradient during real-data qualification")
            last_components = components
            last_module_norms = module_norms
            last_gradient_norm = gradient_norm
            recent_gradient_norms.append(gradient_norm)
            if qualification_writer is not None:
                tensorboard_step = restored_update + offset + 1
                qualification_writer.add_scalar("loss/total", loss, tensorboard_step)
                qualification_writer.add_scalar(
                    "gradient/global_norm", gradient_norm, tensorboard_step,
                )
                qualification_writer.add_scalar(
                    "optimizer/learning_rate", learning_rate, tensorboard_step,
                )
                qualification_writer.add_scalar(
                    "optimizer/amp_scale", scaler.get_scale(), tensorboard_step,
                )
                for key, value in components.items():
                    qualification_writer.add_scalar(
                        f"loss/components/{key}", value, tensorboard_step,
                    )
                if offset % 20 == 0:
                    qualification_writer.flush()
            recovery_reason = _numeric_recovery_reason(
                finite=True,
                consecutive_nonfinite=0,
                recent_nonfinite=[True],
                gradient_norm=gradient_norm,
                recent_gradient_norms=recent_gradient_norms,
            )
            if recovery_reason is not None:
                raise RuntimeError(f"numeric recovery threshold: {recovery_reason}")
            torch.cuda.synchronize(device)
            if offset < warmup_requested:
                warmup_completed += 1
                if warmup_completed == warmup_requested:
                    torch.cuda.reset_peak_memory_stats(device)
                    cuda_free_before, cuda_total_runtime = torch.cuda.mem_get_info(device)
                    cuda_total = int(cuda_total_runtime)
                    started = time.perf_counter()
                continue
            if started is None:
                torch.cuda.reset_peak_memory_stats(device)
                cuda_free_before, cuda_total_runtime = torch.cuda.mem_get_info(device)
                cuda_total = int(cuda_total_runtime)
                started = time.perf_counter()
            losses.append(loss)
            gradient_norms.append(gradient_norm)
    except (torch.OutOfMemoryError, RuntimeError) as exception:
        passed = False
        error = str(exception).splitlines()[0]
    elapsed = time.perf_counter() - started if started is not None else 0.0
    seconds_per_update = elapsed / len(losses) if losses else None
    work_items = sum(
        int(value) for value in (
            contract["audio_global_batch"], contract["flow_global_batch"],
        ) if isinstance(value, int)
    )
    examples_per_second = (
        work_items / seconds_per_update if seconds_per_update is not None and seconds_per_update > 0 else None
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    cgroup_limit = _cgroup_memory_limit_bytes()
    effective_limit = min(cuda_total, cgroup_limit) if cgroup_limit is not None else cuda_total
    memory_headroom = max(effective_limit - peak_reserved, 0)
    system_available = _system_available_memory_bytes()
    window = min(500, len(losses))
    first_median = float(np.median(losses[:window])) if window else None
    last_median = float(np.median(losses[-window:])) if window else None
    gradient_p99 = float(np.percentile(gradient_norms, 99)) if gradient_norms else None
    gradient_maximum = max(gradient_norms) if gradient_norms else None
    stability_errors: list[str] = []
    if args.stability_gate and losses:
        if first_median is not None and last_median is not None and last_median > first_median * 1.25:
            stability_errors.append("final loss median exceeds 1.25x initial median")
        if gradient_p99 is not None and gradient_p99 > 100.0:
            stability_errors.append("gradient p99 exceeds 100")
        if gradient_maximum is not None and gradient_maximum > 1_000.0:
            stability_errors.append("gradient maximum exceeds 1000")
    if stability_errors:
        passed = False
        error = "; ".join(stability_errors)
    saved_checkpoint: str | None = None
    if passed and len(losses) == measured_requested and args.save_checkpoint:
        qualification_update = restored_update + warmup_completed + len(losses)
        qualification_cursor = start_batches + (
            warmup_completed + len(losses)
        ) * grad_accum
        if dist.is_initialized():
            dist.barrier()
        rank_runtime_states = _gather_rank_runtime_states(
            _capture_rank_runtime_state(qualification_cursor), world,
        )
        if rank == 0:
            checkpoint_path = Path(args.save_checkpoint)
            _save_checkpoint(
                checkpoint_path,
                system,
                optimizer,
                scaler,
                stage=stage,
                update=qualification_update,
                config=config,
                nonfinite_skips=int(restored.get("nonfinite_skips", 0)),
                consecutive_nonfinite_skips=0,
                batch_contract=contract,
                run_identity={
                    "host": socket.gethostname(),
                    "session_id": f"qualification-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
                    "resume_update": restored_update,
                    "recovery_profile": profile.name,
                    "precision_mode": "amp_fp16" if profile.use_amp else "fp32",
                },
                rank_runtime_states=rank_runtime_states,
                recovery_profile=profile.name,
                stability_state={
                    "healthy_steps_since_event": warmup_completed + len(losses),
                    "recent_nonfinite": [True] * min(warmup_completed + len(losses), 200),
                    "recent_gradient_norms": list(recent_gradient_norms),
                },
            )
            saved_checkpoint = str(checkpoint_path)
        if dist.is_initialized():
            dist.barrier()
    report: dict[str, object] = {
        "schema": "midibrave.atlas-flow.qualification.v2",
        "passed": passed,
        "stage": stage,
        "checkpoint": str(args.checkpoint),
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_stage": restored.get("stage"),
        "resume_update": restored_update,
        "warmup_updates_requested": warmup_requested,
        "warmup_updates_completed": warmup_completed,
        "updates_requested": measured_requested,
        "updates_completed": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_first_500_median": first_median,
        "loss_last_500_median": last_median,
        "gradient_norm_p99": gradient_p99,
        "gradient_norm_maximum": gradient_maximum,
        "stability_errors": stability_errors,
        "recovery_profile": profile.name,
        "precision_mode": "amp_fp16" if profile.use_amp else "fp32",
        "saved_checkpoint": saved_checkpoint,
        "last_components": last_components,
        "gradient_norm": last_gradient_norm,
        "module_gradient_norms": last_module_norms,
        "seconds_per_atomic_update": seconds_per_update,
        "effective_examples_per_second": examples_per_second,
        "effective_work_items_per_update": work_items,
        "peak_memory_mib": peak_allocated / 2**20,
        "peak_memory_allocated_mib": peak_allocated / 2**20,
        "peak_memory_reserved_mib": peak_reserved / 2**20,
        "cuda_total_memory_mib": cuda_total / 2**20,
        "cuda_free_before_measurement_mib": None if cuda_free_before is None else cuda_free_before / 2**20,
        "cgroup_memory_limit_mib": None if cgroup_limit is None else cgroup_limit / 2**20,
        "effective_memory_limit_mib": effective_limit / 2**20,
        "memory_headroom_mib": memory_headroom / 2**20,
        "system_available_memory_mib": None if system_available is None else system_available / 2**20,
        "amp_scale_start": amp_scale_start,
        "amp_scale_end": float(scaler.get_scale()),
        "batch_contract": contract,
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "error": error,
    }
    if rank == 0:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True), flush=True)
    if qualification_writer is not None:
        qualification_writer.close()
    _destroy_distributed()
    if not passed or len(losses) != measured_requested:
        raise SystemExit(72)


def benchmark(args: argparse.Namespace) -> None:
    config = load_atlas_flow_config(args.config)
    rank, world, local_rank = _distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    candidates = [int(value) for value in args.batches.split(",")]
    reports: list[dict[str, object]] = []
    for batch_size in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        system = AtlasFlowSystem(config.model, config.data).to(device)
        model = system.flow if args.kind == "flow" else system.instrument
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, fused=True)
        shape_h = (batch_size, config.model.context_frames, config.model.trajectory_dim)
        shape_f = (batch_size, config.model.future_frames, config.model.trajectory_dim)
        success, error = True, None
        started = time.perf_counter()
        try:
            if args.kind == "flow":
                for _ in range(int(args.updates)):
                    values = {
                        "history": torch.randn(shape_h, device=device, dtype=torch.float16),
                        "history_anchor": torch.randn(shape_h, device=device, dtype=torch.float16),
                        "target": torch.randn(shape_f, device=device, dtype=torch.float16),
                        "anchor_path": torch.randn(shape_f, device=device, dtype=torch.float16),
                        "atlas_path": torch.randn(batch_size, config.model.future_frames, 8, device=device, dtype=torch.float16),
                        "lifecycle": torch.randn(batch_size, config.model.future_frames, 4, device=device, dtype=torch.float16),
                        "history_mask": torch.ones(batch_size, config.model.context_frames, device=device, dtype=torch.bool),
                    }
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        result = atlas_flow_matching_loss(model, as_flow_batch(values), config.loss)
                    result.total.backward()
                    optimizer.step()
                    torch.cuda.synchronize(device)
            else:
                loader = _infinite(_audio_loader(config, rank, world, batch_size))
                objective = AtlasStage1Objective(config.loss)
                for _ in range(int(args.updates)):
                    values = _to_device(next(loader), device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        result = objective(model, values)  # type: ignore[arg-type]
                    result.total.backward()
                    optimizer.step()
                    torch.cuda.synchronize(device)
        except torch.OutOfMemoryError as exception:
            success, error = False, str(exception).splitlines()[0]
        elapsed = time.perf_counter() - started
        report = {
            "kind": args.kind, "batch_per_gpu": batch_size, "world_size": world, "passed": success,
            "seconds_per_update": elapsed / max(int(args.updates), 1),
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "error": error,
        }
        if rank == 0:
            reports.append(report)
        if args.kind == "audio":
            del loader, objective
        del model, system, optimizer
    if rank == 0:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"candidates": reports}, indent=2) + "\n", encoding="utf-8")
    _destroy_distributed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MIDI-decoupled Atlas Trajectory Flow model.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("preprocess")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--workers", type=int)
    prepare.add_argument("--overwrite", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--stage", required=True, choices=("stage1", "flow", "joint"))
    train.add_argument("--initialize")
    train.add_argument("--resume")
    train.add_argument("--max-updates", type=int)
    train.add_argument("--batch-per-gpu", type=int)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--session-id")
    train.add_argument(
        "--recovery-profile", choices=tuple(RECOVERY_PROFILES), default="safe_amp",
    )
    train.add_argument("--recovery-state")
    cache = sub.add_parser("cache-trajectories")
    cache.add_argument("--config", required=True)
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--batch-size", type=int, default=32)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", required=True)
    bench.add_argument("--kind", choices=("flow", "audio"), default="flow")
    bench.add_argument("--batches", default="96,88,80,72")
    bench.add_argument("--updates", type=int, default=5)
    bench.add_argument("--report", required=True)
    qualification = sub.add_parser("qualify")
    qualification.add_argument("--config", required=True)
    qualification.add_argument("--stage", required=True, choices=("stage1", "flow", "joint"))
    qualification.add_argument("--checkpoint", required=True)
    qualification.add_argument("--batch-per-gpu", required=True, type=int)
    qualification.add_argument("--grad-accum", required=True, type=int)
    qualification.add_argument("--warmup-updates", type=int, default=0)
    qualification.add_argument("--updates", type=int, default=5)
    qualification.add_argument("--report", required=True)
    qualification.add_argument(
        "--recovery-profile", choices=tuple(RECOVERY_PROFILES), default="safe_amp",
    )
    qualification.add_argument("--stability-gate", action="store_true")
    qualification.add_argument("--tensorboard-log-dir")
    qualification.add_argument("--save-checkpoint")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preprocess":
        preprocess(args)
    elif args.command == "cache-trajectories":
        cache_trajectories(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "qualify":
        qualify(args)
    else:
        try:
            train_stage(args)
        except NumericRecoveryRequired as exception:
            print(json.dumps({
                "stage": str(args.stage), "numeric_recovery_required": True,
                "recovery_profile": str(args.recovery_profile),
                "error": str(exception),
            }), flush=True)
            raise SystemExit(75) from exception
        except RuntimeError as exception:
            if not _is_out_of_memory_error(exception):
                raise
            if args.recovery_state and int(os.environ.get("RANK", "0")) == 0:
                _atomic_json(Path(args.recovery_state), {
                    "schema": "midibrave.atlas-flow.recovery-state.v1",
                    "status": "oom",
                    "stage": str(args.stage),
                    "recovery_profile": str(args.recovery_profile),
                    "error": str(exception).splitlines()[0],
                })
            print(json.dumps({
                "stage": str(args.stage), "oom": True,
                "error": str(exception).splitlines()[0],
            }), flush=True)
            raise SystemExit(72) from exception


if __name__ == "__main__":
    main()
