from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel

from .zrave_config import ZraveConfig
from .zrave_model import (
    ZraveStatistics,
    ZraveTransformer,
    zrave_prediction_loss,
)


class GpuWindowSampler:
    def __init__(
        self,
        latents: Tensor,
        lengths: Tensor,
        splits: Tensor,
        *,
        context_frames: int,
        horizon_frames: int,
        split_code: int,
        seed: int,
    ) -> None:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [sequence, frame, channel]")
        if lengths.shape != (latents.shape[0],):
            raise ValueError("lengths must have one entry per latent sequence")
        if splits.shape != lengths.shape:
            raise ValueError("splits must have one entry per latent sequence")
        if lengths.device != latents.device:
            lengths = lengths.to(latents.device)
        if splits.device != latents.device:
            splits = splits.to(latents.device)
        self.latents = latents
        self.lengths = lengths.long()
        self.splits = splits.long()
        self.context_frames = int(context_frames)
        self.horizon_frames = int(horizon_frames)
        self.total_frames = self.context_frames + self.horizon_frames
        self.eligible_indices = torch.nonzero(
            (self.splits == split_code)
            & (self.lengths >= self.total_frames),
            as_tuple=False,
        ).flatten()
        if not self.eligible_indices.numel():
            raise ValueError(
                f"split {split_code} has no sequence covering "
                f"{self.total_frames} frames"
            )
        self.generator = torch.Generator(device=latents.device)
        self.generator.manual_seed(int(seed))
        self.last_sequence_indices = torch.empty(
            0,
            dtype=torch.long,
            device=latents.device,
        )
        self.last_start_indices = torch.empty(
            0,
            dtype=torch.long,
            device=latents.device,
        )

    @classmethod
    def from_pack(
        cls,
        packed_root: str | Path,
        *,
        device: torch.device | str,
        context_frames: int,
        horizon_frames: int,
        split_code: int,
        seed: int,
    ) -> "GpuWindowSampler":
        root = Path(packed_root)
        latents = torch.from_numpy(
            np.load(root / "latents.npy", allow_pickle=False)
        ).to(device=device)
        lengths = torch.from_numpy(
            np.load(root / "lengths.npy", allow_pickle=False)
        ).to(device=device)
        splits = torch.from_numpy(
            np.load(root / "splits.npy", allow_pickle=False)
        ).to(device=device)
        return cls(
            latents,
            lengths,
            splits,
            context_frames=context_frames,
            horizon_frames=horizon_frames,
            split_code=split_code,
            seed=seed,
        )

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        choices = torch.randint(
            self.eligible_indices.numel(),
            (batch_size,),
            generator=self.generator,
            device=self.latents.device,
        )
        sequence_indices = self.eligible_indices[choices]
        maximum_start = (
            self.lengths[sequence_indices] - self.total_frames
        )
        start_indices = torch.floor(
            torch.rand(
                batch_size,
                generator=self.generator,
                device=self.latents.device,
            )
            * (maximum_start + 1).float()
        ).long()
        frame_indices = start_indices[:, None] + torch.arange(
            self.total_frames,
            device=self.latents.device,
        )[None]
        windows = self.latents[sequence_indices[:, None], frame_indices]
        self.last_sequence_indices = sequence_indices
        self.last_start_indices = start_indices
        return (
            windows[:, : self.context_frames],
            windows[:, self.context_frames :],
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, Tensor):
            raise ValueError("sampler state is missing generator_state")
        self.generator.set_state(generator_state)


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


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler_states: list[dict[str, Tensor]],
    rng_states: list[dict[str, object]],
    update: int,
    contract: dict[str, object],
    world_size: int,
    batch_per_gpu: int,
    best_validation_metric: float,
    validations_without_improvement: int,
) -> dict[str, object]:
    unwrapped = (
        model.module
        if isinstance(model, DistributedDataParallel)
        else model
    )
    return {
        "format": 1,
        "architecture": "zrave_transformer_v1",
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler_by_rank": sampler_states,
        "rng_by_rank": rng_states,
        "update": int(update),
        "contract": dict(contract),
        "world_size": int(world_size),
        "batch_per_gpu": int(batch_per_gpu),
        "best_validation_metric": float(best_validation_metric),
        "validations_without_improvement": int(
            validations_without_improvement
        ),
    }


def save_zrave_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: GpuWindowSampler,
    update: int,
    contract: dict[str, object],
    world_size: int,
    batch_per_gpu: int,
    best_validation_metric: float,
    validations_without_improvement: int,
) -> None:
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        sampler_states=[sampler.state_dict()],
        rng_states=[_capture_rng_state()],
        update=update,
        contract=contract,
        world_size=world_size,
        batch_per_gpu=batch_per_gpu,
        best_validation_metric=best_validation_metric,
        validations_without_improvement=validations_without_improvement,
    )
    _atomic_torch_save(Path(path), payload)


def load_zrave_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: GpuWindowSampler,
    expected_contract: dict[str, object],
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Z-RAVE checkpoint must be a mapping")
    if payload.get("format") != 1:
        raise ValueError("unsupported Z-RAVE checkpoint format")
    if payload.get("architecture") != "zrave_transformer_v1":
        raise ValueError("Z-RAVE checkpoint architecture mismatch")
    actual_contract = payload.get("contract")
    if not isinstance(actual_contract, dict):
        raise ValueError("Z-RAVE checkpoint has no immutable contract")
    for name, expected in expected_contract.items():
        if actual_contract.get(name) != expected:
            raise ValueError(f"Z-RAVE checkpoint {name} mismatch")
    unwrapped = (
        model.module
        if isinstance(model, DistributedDataParallel)
        else model
    )
    unwrapped.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        if payload.get("scaler") is None:
            raise ValueError("Z-RAVE checkpoint is missing scaler state")
        scaler.load_state_dict(payload["scaler"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler_states = payload.get("sampler_by_rank")
    rng_states = payload.get("rng_by_rank")
    if (
        not isinstance(sampler_states, list)
        or rank >= len(sampler_states)
        or not isinstance(rng_states, list)
        or rank >= len(rng_states)
    ):
        raise ValueError("Z-RAVE checkpoint lacks rank-specific resume state")
    sampler.load_state_dict(sampler_states[rank])
    _restore_rng_state(rng_states[rank])
    return {
        "update": int(payload["update"]),
        "world_size": int(payload["world_size"]),
        "batch_per_gpu": int(payload["batch_per_gpu"]),
        "best_validation_metric": float(
            payload["best_validation_metric"]
        ),
        "validations_without_improvement": int(
            payload["validations_without_improvement"]
        ),
    }


def summarize_benchmark(
    *,
    batch_per_gpu: int,
    world_size: int,
    durations_seconds: Iterable[float],
    peak_memory_mib: float,
    total_memory_mib: float,
    nonfinite_updates: int,
) -> dict[str, object]:
    durations = np.asarray(list(durations_seconds), dtype=np.float64)
    global_batch = int(batch_per_gpu) * int(world_size)
    valid_durations = durations[
        np.isfinite(durations) & (durations > 0.0)
    ]
    throughputs = (
        global_batch / valid_durations
        if valid_durations.size
        else np.asarray([], dtype=np.float64)
    )
    valid = (
        valid_durations.size == durations.size
        and durations.size > 0
        and nonfinite_updates == 0
        and math.isfinite(peak_memory_mib)
        and 0.0 < peak_memory_mib < total_memory_mib
    )
    return {
        "batch_per_gpu": int(batch_per_gpu),
        "world_size": int(world_size),
        "global_batch": global_batch,
        "measured_updates": int(durations.size),
        "median_windows_per_second": (
            float(np.median(throughputs)) if throughputs.size else 0.0
        ),
        "p10_windows_per_second": (
            float(np.percentile(throughputs, 10))
            if throughputs.size
            else 0.0
        ),
        "peak_memory_mib": float(peak_memory_mib),
        "total_memory_mib": float(total_memory_mib),
        "nonfinite_updates": int(nonfinite_updates),
        "durations_seconds": durations.tolist(),
        "status": "ok" if valid else "invalid",
    }


def _load_statistics(root: Path) -> ZraveStatistics:
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
        return ZraveStatistics(
            mean=torch.from_numpy(values["mean"].copy()),
            latent_std=torch.from_numpy(values["latent_std"].copy()),
            delta_std=torch.from_numpy(values["delta_std"].copy()),
            acceleration_std=torch.from_numpy(
                values["acceleration_std"].copy()
            ),
        )


def _seed_everything(seed: int, rank: int) -> None:
    chosen = seed + rank
    random.seed(chosen)
    np.random.seed(chosen)
    torch.manual_seed(chosen)
    torch.cuda.manual_seed(chosen)


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Z-RAVE training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _all_reduce_max(value: Tensor) -> Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def _all_reduce_sum(value: Tensor) -> Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _learning_rate(config: ZraveConfig, update: int, maximum: int) -> float:
    peak = config.optimizer.learning_rate
    floor = config.optimizer.minimum_learning_rate
    if update < config.train.warmup_updates:
        return peak * (update + 1) / config.train.warmup_updates
    progress = min(
        1.0,
        (update - config.train.warmup_updates)
        / max(1, maximum - config.train.warmup_updates),
    )
    return floor + 0.5 * (peak - floor) * (
        1.0 + math.cos(math.pi * progress)
    )


def _rollout(
    model: ZraveTransformer,
    history: Tensor,
    frames: int,
) -> Tensor:
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    while remaining > 0:
        prediction = model(current).latent
        take = min(remaining, prediction.shape[1])
        chunk = prediction[:, :take]
        chunks.append(chunk)
        current = torch.cat([current, chunk], dim=1)[
            :, -model.config.context_frames :
        ]
        remaining -= take
    return torch.cat(chunks, dim=1)


@torch.inference_mode()
def _validation_metric(
    model: ZraveTransformer,
    sampler: GpuWindowSampler,
    *,
    batch_size: int,
    batches: int,
    latent_std: Tensor,
) -> float:
    model.eval()
    total = torch.zeros(2, device=sampler.latents.device, dtype=torch.float64)
    scale = latent_std.to(
        device=sampler.latents.device,
        dtype=torch.float32,
    )
    for _ in range(batches):
        history, target = sampler.sample(batch_size)
        prediction = _rollout(model, history, target.shape[1])
        error = torch.nn.functional.smooth_l1_loss(
            prediction.float() / scale,
            target.float() / scale,
            reduction="sum",
        )
        total[0] += error.double()
        total[1] += target.numel()
    _all_reduce_sum(total)
    model.train()
    return float((total[0] / total[1]).item())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gather_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sampler: GpuWindowSampler,
    update: int,
    contract: dict[str, object],
    world_size: int,
    batch_per_gpu: int,
    best_validation_metric: float,
    validations_without_improvement: int,
) -> dict[str, object] | None:
    local_sampler = sampler.state_dict()
    local_rng = _capture_rng_state()
    if dist.is_initialized():
        sampler_states: list[object] | None = (
            [None] * world_size if dist.get_rank() == 0 else None
        )
        rng_states: list[object] | None = (
            [None] * world_size if dist.get_rank() == 0 else None
        )
        dist.gather_object(local_sampler, sampler_states, dst=0)
        dist.gather_object(local_rng, rng_states, dst=0)
        if dist.get_rank() != 0:
            return None
        assert sampler_states is not None and rng_states is not None
        typed_sampler_states = list(sampler_states)
        typed_rng_states = list(rng_states)
    else:
        typed_sampler_states = [local_sampler]
        typed_rng_states = [local_rng]
    return _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        sampler_states=typed_sampler_states,
        rng_states=typed_rng_states,
        update=update,
        contract=contract,
        world_size=world_size,
        batch_per_gpu=batch_per_gpu,
        best_validation_metric=best_validation_metric,
        validations_without_improvement=validations_without_improvement,
    )


def _train(args: argparse.Namespace) -> None:
    config = ZraveConfig.load(args.config)
    rank, local_rank, world_size, device = _setup_distributed()
    _seed_everything(config.seed, rank)
    batch_per_gpu = int(args.batch_per_gpu or config.train.batch_per_gpu)
    maximum_updates = int(args.max_updates or config.train.max_updates)
    benchmark = args.benchmark_updates is not None
    packed_root = Path(config.data.packed_root)
    statistics = _load_statistics(packed_root)
    sampler = GpuWindowSampler.from_pack(
        packed_root,
        device=device,
        context_frames=config.model.context_frames,
        horizon_frames=config.model.horizon_frames,
        split_code=0,
        seed=config.seed + 1000 + rank,
    )
    model = ZraveTransformer(config.model, statistics).to(device)
    if world_size > 1:
        training_model: nn.Module = DistributedDataParallel(
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
    scaler = torch.amp.GradScaler("cuda")
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    if config.source_path is None:
        raise ValueError("training requires a file-backed configuration")
    contract: dict[str, object] = {
        "config_sha256": _sha256_file(config.source_path),
        "packed_index_sha256": _sha256_file(index_path),
        "statistics_sha256": _sha256_file(statistics_path),
        "world_size": world_size,
        "batch_per_gpu": batch_per_gpu,
        "latent_dim": config.model.latent_dim,
        "context_frames": config.model.context_frames,
        "horizon_frames": config.model.horizon_frames,
    }
    update = 0
    best_validation_metric = math.inf
    validations_without_improvement = 0
    if args.resume:
        restored = load_zrave_checkpoint(
            args.resume,
            model=training_model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=sampler,
            expected_contract=contract,
        )
        if restored["world_size"] != world_size:
            raise ValueError("resume world size mismatch")
        if restored["batch_per_gpu"] != batch_per_gpu:
            raise ValueError("resume batch-per-GPU mismatch")
        update = int(restored["update"])
        best_validation_metric = float(
            restored["best_validation_metric"]
        )
        validations_without_improvement = int(
            restored["validations_without_improvement"]
        )

    writer = None
    run_root = Path(config.train.output_root)
    if rank == 0 and not benchmark:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_root / "tensorboard")
    durations: list[float] = []
    nonfinite_updates = 0
    benchmark_warmup = int(args.benchmark_warmup or 0)
    benchmark_updates = int(args.benchmark_updates or 0)
    target_update = (
        benchmark_warmup + benchmark_updates
        if benchmark
        else maximum_updates
    )
    if benchmark:
        update = 0
        torch.cuda.reset_peak_memory_stats(device)

    stopped_early = False
    while update < target_update:
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(config, update, maximum_updates)
        history, target = sampler.sample(batch_per_gpu)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = training_model(history)
            loss = zrave_prediction_loss(
                prediction,
                history,
                target,
                statistics,
                config.loss,
            )
        finite = torch.tensor(
            int(torch.isfinite(loss.total).item()),
            device=device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item():
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(),
                config.train.gradient_clip,
            )
            gradient_finite = torch.tensor(
                int(torch.isfinite(gradient_norm).item()),
                device=device,
                dtype=torch.int32,
            )
            if dist.is_initialized():
                dist.all_reduce(gradient_finite, op=dist.ReduceOp.MIN)
            if gradient_finite.item():
                scaler.step(optimizer)
            else:
                nonfinite_updates += 1
            scaler.update()
        else:
            nonfinite_updates += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        elapsed_tensor = torch.tensor(
            elapsed,
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_max(elapsed_tensor)
        update += 1
        if benchmark and update > benchmark_warmup:
            durations.append(float(elapsed_tensor.item()))

        if writer is not None and update % config.train.log_every == 0:
            writer.add_scalar("train/loss_total", loss.total.item(), update)
            for name, value in loss.components.items():
                writer.add_scalar(f"train/loss_{name}", value.item(), update)
            writer.add_scalar(
                "train/learning_rate",
                optimizer.param_groups[0]["lr"],
                update,
            )
            writer.add_scalar(
                "train/global_windows_per_second",
                batch_per_gpu * world_size / elapsed_tensor.item(),
                update,
            )
            writer.add_scalar(
                "train/global_latent_frames_per_second",
                batch_per_gpu
                * world_size
                * config.model.horizon_frames
                / elapsed_tensor.item(),
                update,
            )
            writer.add_scalar(
                "health/nonfinite_updates",
                nonfinite_updates,
                update,
            )
            writer.add_scalar(
                "health/peak_gpu_memory_mib",
                torch.cuda.max_memory_allocated(device) / 2**20,
                update,
            )

        validation_due = (
            not benchmark
            and update % config.train.validation_every == 0
        )
        final_due = not benchmark and update == target_update
        improved = False
        if validation_due:
            validation_sampler = GpuWindowSampler(
                sampler.latents,
                sampler.lengths,
                sampler.splits,
                context_frames=config.model.context_frames,
                horizon_frames=128,
                split_code=1,
                seed=config.seed + 2000 + rank,
            )
            validation_metric = _validation_metric(
                model,
                validation_sampler,
                batch_size=batch_per_gpu,
                batches=config.train.validation_batches,
                latent_std=statistics.latent_std,
            )
            improved = validation_metric < best_validation_metric
            if improved:
                best_validation_metric = validation_metric
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if writer is not None:
                writer.add_scalar(
                    "validation/rollout_128_smooth_l1",
                    validation_metric,
                    update,
                )
                writer.add_scalar(
                    "validation/best_rollout_128_smooth_l1",
                    best_validation_metric,
                    update,
                )
            stopped_early = (
                validations_without_improvement
                >= config.train.early_stop_validations
            )

        checkpoint_due = (
            not benchmark
            and (
                update % config.train.checkpoint_every == 0
                or final_due
                or stopped_early
            )
        )
        if checkpoint_due:
            payload = _gather_checkpoint_payload(
                model=training_model,
                optimizer=optimizer,
                scaler=scaler,
                sampler=sampler,
                update=update,
                contract=contract,
                world_size=world_size,
                batch_per_gpu=batch_per_gpu,
                best_validation_metric=best_validation_metric,
                validations_without_improvement=(
                    validations_without_improvement
                ),
            )
            if rank == 0:
                assert payload is not None
                checkpoint_path = (
                    run_root / "checkpoints" / f"update-{update:08d}.pt"
                )
                _atomic_torch_save(checkpoint_path, payload)
                if improved:
                    best_path = run_root / "checkpoints" / "best.pt"
                    temporary_best = best_path.with_name(best_path.name + ".tmp")
                    temporary_best.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkpoint_path, temporary_best)
                    temporary_best.replace(best_path)
        if stopped_early:
            break

    if benchmark:
        peak = torch.tensor(
            torch.cuda.max_memory_allocated(device) / 2**20,
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_max(peak)
        total_memory = torch.tensor(
            torch.cuda.get_device_properties(device).total_memory / 2**20,
            device=device,
            dtype=torch.float64,
        )
        _all_reduce_max(total_memory)
        nonfinite = torch.tensor(
            nonfinite_updates,
            device=device,
            dtype=torch.int64,
        )
        _all_reduce_sum(nonfinite)
        if rank == 0:
            report = summarize_benchmark(
                batch_per_gpu=batch_per_gpu,
                world_size=world_size,
                durations_seconds=durations,
                peak_memory_mib=float(peak.item()),
                total_memory_mib=float(total_memory.item()),
                nonfinite_updates=int(nonfinite.item()),
            )
            if args.benchmark_output:
                _write_json(Path(args.benchmark_output), report)
            print(json.dumps(report, sort_keys=True), flush=True)
    elif rank == 0:
        _write_json(
            run_root / "training-summary.json",
            {
                "update": update,
                "best_validation_metric": best_validation_metric,
                "validations_without_improvement": (
                    validations_without_improvement
                ),
                "stopped_early": stopped_early,
                "world_size": world_size,
                "batch_per_gpu": batch_per_gpu,
                "nonfinite_updates": nonfinite_updates,
                "contract": contract,
            },
        )
    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the pure Z-RAVE sequence Transformer."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-per-gpu", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    parser.add_argument("--benchmark-updates", type=int)
    parser.add_argument("--benchmark-output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.benchmark_updates is not None:
        if args.benchmark_updates <= 0:
            raise ValueError("--benchmark-updates must be positive")
        if args.benchmark_warmup < 0:
            raise ValueError("--benchmark-warmup must be non-negative")
        if not args.benchmark_output:
            raise ValueError("benchmark mode requires --benchmark-output")
    _train(args)


if __name__ == "__main__":
    main()
