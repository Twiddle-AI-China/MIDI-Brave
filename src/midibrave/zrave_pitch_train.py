from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_sampler import GpuFlowSampler
from .zrave_pitch_probe import (
    LatentPitchProbe,
    freeze_pitch_probe,
    pitch_probe_loss,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _largest_remainder(total: int, buckets: int) -> list[int]:
    if total <= 0 or buckets <= 0:
        raise ValueError("allocation values must be positive")
    base, remainder = divmod(total, buckets)
    return [
        base + int(index < remainder)
        for index in range(buckets)
    ]


class PitchWindowSampler:
    def __init__(
        self,
        *,
        latents: Tensor,
        active_frames: Tensor,
        notes: Tensor,
        source_codes: Tensor,
        split_codes: Tensor,
        split_code: int,
        seed: int,
        device: str | torch.device,
        record_eligible: Tensor | None = None,
    ) -> None:
        if latents.ndim != 3 or latents.shape[-1] <= 0:
            raise ValueError(
                "latents must have shape [records, frames, latent_dim]"
            )
        records = latents.shape[0]
        metadata = (active_frames, notes, source_codes, split_codes)
        if any(value.shape != (records,) for value in metadata):
            raise ValueError("pitch sampler metadata length mismatch")
        if (
            record_eligible is not None
            and record_eligible.shape != (records,)
        ):
            raise ValueError("pitch sampler eligibility length mismatch")
        self.device = torch.device(device)
        self.latents = latents.to(self.device)
        self.active_frames = active_frames.cpu().long()
        self.notes = notes.cpu().long()
        self.source_codes = source_codes.cpu().long()
        self.split_codes = split_codes.cpu().long()
        self.split_code = int(split_code)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        if torch.any(self.active_frames > latents.shape[1]):
            raise ValueError("active_frames exceeds latent storage")
        eligible_mask = (
            (self.split_codes == self.split_code)
            & (self.active_frames >= 16)
            & (self.notes >= 21)
            & (self.notes <= 109)
        )
        if record_eligible is not None:
            eligible_mask &= record_eligible.cpu().bool()
        self.eligible = torch.nonzero(
            eligible_mask,
            as_tuple=False,
        ).flatten()
        if not self.eligible.numel():
            raise ValueError("split has no eligible pitch windows")
        self.note_values = torch.unique(
            self.notes[self.eligible],
            sorted=True,
        )
        self.note_pools = tuple(
            self.eligible[self.notes[self.eligible] == note]
            for note in self.note_values
        )
        self.last_record_indices = torch.empty(0, dtype=torch.long)
        self.last_start_indices = torch.empty(0, dtype=torch.long)
        self.last_source_codes = torch.empty(0, dtype=torch.long)

    @classmethod
    def from_pack(
        cls,
        config: ZraveFlowConfig,
        *,
        split: str,
        device: str | torch.device,
        seed: int,
    ) -> "PitchWindowSampler":
        split_codes = {"train": 0, "validation": 1, "test": 2}
        if split not in split_codes:
            raise ValueError(f"invalid split: {split}")
        packed = GpuFlowSampler.from_pack(
            config,
            split=split,
            device=device,
            seed=seed,
        )
        return cls(
            latents=packed.latents,
            active_frames=packed.active_frames,
            notes=packed.notes,
            source_codes=packed.source_codes,
            split_codes=packed.split_codes,
            split_code=split_codes[split],
            seed=seed,
            device=device,
            record_eligible=packed.record_eligible,
        )

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        counts = _largest_remainder(
            batch_size,
            len(self.note_pools),
        )
        note_order = torch.randperm(
            len(self.note_pools),
            generator=self.generator,
        )
        counts = [
            counts[int(note_order[index])]
            for index in range(len(counts))
        ]
        selected_parts: list[Tensor] = []
        for pool, count in zip(self.note_pools, counts, strict=True):
            if count:
                offsets = torch.randint(
                    pool.numel(),
                    (count,),
                    generator=self.generator,
                )
                selected_parts.append(pool[offsets])
        selected = torch.cat(selected_parts)
        permutation = torch.randperm(
            batch_size,
            generator=self.generator,
        )
        selected = selected[permutation]
        maximum_start = self.active_frames[selected] - 16
        starts = torch.floor(
            torch.rand(
                batch_size,
                generator=self.generator,
            )
            * (maximum_start + 1).float()
        ).long()
        windows = torch.empty(
            batch_size,
            16,
            self.latents.shape[-1],
            device=self.device,
            dtype=torch.float32,
        )
        for index in range(batch_size):
            record = int(selected[index])
            start = int(starts[index])
            windows[index] = self.latents[
                record,
                start : start + 16,
            ].float()
        self.last_record_indices = selected.clone()
        self.last_start_indices = starts.clone()
        self.last_source_codes = self.source_codes[selected].clone()
        return windows, self.notes[selected].to(self.device)

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if set(state) != {"generator_state"}:
            raise ValueError("invalid pitch sampler state")
        self.generator.set_state(state["generator_state"].cpu())


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


def save_pitch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: PitchWindowSampler,
    update: int,
    contract: dict[str, object],
) -> None:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
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
        assert sampler_states is not None
        assert rng_states is not None
    else:
        sampler_states = [local_sampler]
        rng_states = [local_rng]
    probe = _unwrapped(model)
    if not isinstance(probe, LatentPitchProbe):
        raise TypeError("pitch checkpoint model must be LatentPitchProbe")
    payload: dict[str, object] = {
        "format": 1,
        "architecture": "zrave_latent_pitch_probe_v1",
        "architecture_config": {
            "latent_dim": probe.latent_dim,
            "note_min": probe.note_min,
            "note_max": probe.note_max,
        },
        "model": probe.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler_by_rank": sampler_states,
        "rng_by_rank": rng_states,
        "update": int(update),
        "contract": dict(contract),
    }
    _atomic_torch_save(Path(path), payload)


def load_pitch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    sampler: PitchWindowSampler,
    expected_contract: dict[str, object],
) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("pitch checkpoint must be a mapping")
    if payload.get("format") != 1:
        raise ValueError("unsupported pitch checkpoint format")
    if payload.get("architecture") != "zrave_latent_pitch_probe_v1":
        raise ValueError("pitch checkpoint architecture mismatch")
    actual_contract = payload.get("contract")
    if not isinstance(actual_contract, dict):
        raise ValueError("pitch checkpoint has no contract")
    for name, expected in expected_contract.items():
        if actual_contract.get(name) != expected:
            raise ValueError(f"pitch checkpoint {name} mismatch")
    _unwrapped(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler_state = payload.get("scaler")
        if not isinstance(scaler_state, dict):
            raise ValueError("pitch checkpoint is missing scaler state")
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
        raise ValueError("pitch checkpoint lacks rank resume state")
    sampler.load_state_dict(sampler_states[rank])
    _restore_rng_state(rng_states[rank])
    return {"update": int(payload["update"])}


def pitch_probe_gate(
    metrics: dict[str, float | bool],
) -> dict[str, object]:
    required = {
        "accuracy",
        "median_absolute_cents",
        "p90_absolute_cents",
        "finite",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(
            "pitch metrics missing: " + ", ".join(sorted(missing))
        )
    accuracy = float(metrics["accuracy"])
    median = float(metrics["median_absolute_cents"])
    p90 = float(metrics["p90_absolute_cents"])
    finite = bool(metrics["finite"]) and all(
        math.isfinite(value) for value in (accuracy, median, p90)
    )
    gates = {
        "finite": finite,
        "accuracy_at_least_0_98": finite and accuracy >= 0.98,
        "median_absolute_cents_at_most_25": finite and median <= 25.0,
        "p90_absolute_cents_at_most_50": finite and p90 <= 50.0,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": dict(metrics),
    }


def _resolve_qualification_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "qualification.json"
    if candidate.suffix == ".json":
        return candidate
    return candidate.parent.parent / "qualification.json"


def load_qualified_pitch_probe(
    path: str | Path,
    expected_pack_hash: str,
) -> LatentPitchProbe:
    qualification_path = _resolve_qualification_path(path)
    if not qualification_path.is_file():
        raise ValueError("pitch probe qualification is missing")
    qualification = json.loads(
        qualification_path.read_text(encoding="utf-8")
    )
    if not qualification.get("passed"):
        raise ValueError("pitch probe did not pass qualification")
    if qualification.get("pack_index_sha256") != expected_pack_hash:
        raise ValueError("pitch probe pack hash mismatch")
    checkpoint_value = qualification.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("qualification has no checkpoint")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = qualification_path.parent / checkpoint
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("format") != 1
        or payload.get("architecture")
        != "zrave_latent_pitch_probe_v1"
    ):
        raise ValueError("qualified pitch checkpoint is invalid")
    contract = payload.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("pack_index_sha256") != expected_pack_hash
    ):
        raise ValueError("qualified checkpoint pack hash mismatch")
    architecture = payload.get("architecture_config")
    if not isinstance(architecture, dict):
        raise ValueError("qualified checkpoint lacks architecture")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("qualified checkpoint lacks model weights")
    if any(
        not isinstance(value, Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("qualified checkpoint has non-finite weights")
    probe = LatentPitchProbe(
        latent_dim=int(architecture["latent_dim"]),
        note_min=int(architecture["note_min"]),
        note_max=int(architecture["note_max"]),
    )
    probe.load_state_dict(state)
    return freeze_pitch_probe(probe)


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("pitch probe training requires CUDA")
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


def _learning_rate(update: int, maximum_updates: int) -> float:
    peak = 3.0e-4
    warmup = 500
    if update <= warmup:
        return peak * update / warmup
    progress = min(
        1.0,
        (update - warmup) / max(1, maximum_updates - warmup),
    )
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def _validate(
    model: nn.Module,
    sampler: PitchWindowSampler,
    *,
    batch_size: int,
    batches: int,
) -> dict[str, float | bool]:
    sampler_state = sampler.state_dict()
    model.eval()
    correct = 0
    count = 0
    errors: list[float] = []
    finite = True
    probe = _unwrapped(model)
    assert isinstance(probe, LatentPitchProbe)
    for _ in range(batches):
        windows, notes = sampler.sample(batch_size)
        output = model(windows)
        predicted = output.logits.argmax(dim=-1) + probe.note_min
        correct += int((predicted == notes).sum().item())
        count += notes.numel()
        cents = (
            output.expected_midi.float() - notes.float()
        ).abs() * 100.0
        finite &= bool(
            torch.isfinite(output.logits).all()
            and torch.isfinite(cents).all()
        )
        errors.extend(cents.cpu().tolist())
    payload = {
        "correct": correct,
        "count": count,
        "errors": errors,
        "finite": finite,
    }
    if dist.is_initialized():
        gathered: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        rows = gathered
    else:
        rows = [payload]
    total_correct = sum(int(row["correct"]) for row in rows)
    total_count = sum(int(row["count"]) for row in rows)
    all_errors = np.asarray(
        [
            error
            for row in rows
            for error in row["errors"]
        ],
        dtype=np.float64,
    )
    metrics: dict[str, float | bool] = {
        "accuracy": total_correct / max(1, total_count),
        "median_absolute_cents": float(np.median(all_errors)),
        "p90_absolute_cents": float(
            np.percentile(all_errors, 90)
        ),
        "finite": all(bool(row["finite"]) for row in rows),
    }
    sampler.load_state_dict(sampler_state)
    model.train()
    return metrics


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    value: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _finite_across_ranks(value: Tensor) -> bool:
    flag = torch.tensor(
        int(not torch.isfinite(value)),
        device=value.device,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item()) == 0


def _train(args: argparse.Namespace) -> None:
    config = ZraveFlowConfig.load(args.config)
    rank, local_rank, world_size, device = _setup_distributed()
    _seed_everything(config.seed, rank)
    batch_per_gpu = int(args.batch_per_gpu or 512)
    maximum_updates = int(args.max_updates or 10000)
    if batch_per_gpu <= 0:
        raise ValueError("batch-per-gpu must be positive")
    if not 1 <= maximum_updates <= 10000:
        raise ValueError("max-updates must be in [1, 10000]")
    if config.source_path is None:
        raise ValueError("training requires a file-backed config")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    pack_hash = _sha256_file(index_path)
    output_root = Path(config.train.output_root).parent / "pitch-probe"
    checkpoint_root = output_root / "checkpoints"
    train_sampler = PitchWindowSampler.from_pack(
        config,
        split="train",
        device=device,
        seed=config.seed + 1000 + rank,
    )
    validation_sampler = PitchWindowSampler.from_pack(
        config,
        split="validation",
        device=device,
        seed=config.seed + 2000 + rank,
    )
    model = LatentPitchProbe(
        latent_dim=config.model.latent_dim,
        note_min=config.model.note_min,
        note_max=config.model.note_max,
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
        lr=3.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    contract: dict[str, object] = {
        "pack_index_sha256": pack_hash,
        "config_sha256": _sha256_file(config.source_path),
        "world_size": world_size,
        "batch_per_gpu": batch_per_gpu,
        "maximum_updates": maximum_updates,
        "latent_dim": config.model.latent_dim,
        "note_min": config.model.note_min,
        "note_max": config.model.note_max,
        "optimizer": "adamw",
        "learning_rate": 3.0e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "warmup_updates": 500,
        "precision": "amp_fp16",
        "gradient_clip": 1.0,
    }
    update = 0
    if args.resume:
        restored = load_pitch_checkpoint(
            args.resume,
            model=training_model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=train_sampler,
            expected_contract=contract,
        )
        update = restored["update"]
    writer = (
        SummaryWriter(str(output_root / "tensorboard"))
        if rank == 0
        else None
    )
    qualified = False
    last_gate: dict[str, object] | None = None
    while update < maximum_updates:
        windows, notes = train_sampler.sample(batch_per_gpu)
        next_update = update + 1
        learning_rate = _learning_rate(next_update, maximum_updates)
        _set_optimizer_lr(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            report = pitch_probe_loss(training_model(windows), notes)
        if not _finite_across_ranks(report.total):
            raise FloatingPointError("non-finite pitch probe loss")
        scaler.scale(report.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            training_model.parameters(),
            1.0,
        )
        if not _finite_across_ranks(gradient_norm):
            raise FloatingPointError("non-finite pitch probe gradient")
        scaler.step(optimizer)
        scaler.update()
        update = next_update
        if writer is not None and (
            update == 1 or update % 20 == 0
        ):
            writer.add_scalar(
                "train/loss",
                float(report.total.detach()),
                update,
            )
            writer.add_scalar(
                "train/cross_entropy",
                float(report.components["cross_entropy"].detach()),
                update,
            )
            writer.add_scalar(
                "train/midi_smooth_l1",
                float(report.components["midi_smooth_l1"].detach()),
                update,
            )
            writer.add_scalar("train/learning_rate", learning_rate, update)
            writer.add_scalar(
                "train/gradient_norm",
                float(gradient_norm),
                update,
            )
        if update % 1000 != 0 and update != maximum_updates:
            continue
        metrics = _validate(
            training_model,
            validation_sampler,
            batch_size=batch_per_gpu,
            batches=16,
        )
        gate = pitch_probe_gate(metrics)
        last_gate = gate
        checkpoint_path = checkpoint_root / f"step-{update:06d}.pt"
        save_pitch_checkpoint(
            checkpoint_path,
            model=training_model,
            optimizer=optimizer,
            scaler=scaler,
            sampler=train_sampler,
            update=update,
            contract=contract,
        )
        if dist.is_initialized():
            dist.barrier()
        qualified = bool(gate["passed"])
        if rank == 0:
            assert writer is not None
            for name, value in metrics.items():
                if name != "finite":
                    writer.add_scalar(
                        f"validation/{name}",
                        float(value),
                        update,
                    )
            writer.flush()
            if qualified:
                best = checkpoint_root / "best-qualified.pt"
                temporary = best.with_name(best.name + ".tmp")
                shutil.copy2(checkpoint_path, temporary)
                temporary.replace(best)
                _atomic_json(
                    output_root / "qualification.json",
                    {
                        **gate,
                        "update": update,
                        "pack_index_sha256": pack_hash,
                        "config_sha256": contract["config_sha256"],
                        "checkpoint": "checkpoints/best-qualified.pt",
                    },
                )
        if qualified:
            break
    if rank == 0 and writer is not None:
        if not qualified:
            _atomic_json(
                output_root / "qualification.json",
                {
                    **(
                        last_gate
                        or pitch_probe_gate(
                            {
                                "accuracy": 0.0,
                                "median_absolute_cents": 1.0e30,
                                "p90_absolute_cents": 1.0e30,
                                "finite": False,
                            }
                        )
                    ),
                    "update": update,
                    "pack_index_sha256": pack_hash,
                    "config_sha256": contract["config_sha256"],
                    "checkpoint": None,
                },
            )
        writer.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if not qualified:
        raise RuntimeError(
            "pitch probe failed qualification within max-updates"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and qualify the Z-RAVE latent pitch probe."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-per-gpu", type=int, default=512)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--resume")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    _train(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
