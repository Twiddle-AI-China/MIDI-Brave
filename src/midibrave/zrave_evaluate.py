from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from .zrave_config import ZraveConfig
from .zrave_model import ZraveStatistics, ZraveTransformer


def persistence_baseline(history: Tensor, frames: int) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if history.ndim != 3 or history.shape[1] < 1:
        raise ValueError("history must have shape [batch, frames, channels]")
    return history[:, -1:].expand(-1, frames, -1).clone()


def linear_baseline(history: Tensor, frames: int) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if history.ndim != 3 or history.shape[1] < 2:
        raise ValueError("linear baseline requires two history frames")
    steps = torch.arange(
        1,
        frames + 1,
        device=history.device,
        dtype=history.dtype,
    )[None, :, None]
    delta = (history[:, -1] - history[:, -2])[:, None]
    return history[:, -1:] + steps * delta


def rollout_prediction(
    model: ZraveTransformer,
    history: Tensor,
    frames: int,
) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    while remaining:
        predicted = model(current).latent
        take = min(remaining, predicted.shape[1])
        chunk = predicted[:, :take]
        chunks.append(chunk)
        current = torch.cat([current, chunk], dim=1)[
            :, -model.config.context_frames :
        ]
        remaining -= take
    return torch.cat(chunks, dim=1)


def acceptance_gate(report: dict[str, Any]) -> dict[str, Any]:
    rollout = report["rollout"]
    contract = report["evaluation_contract"]
    gate_horizons = tuple(int(value) for value in contract["gate_horizons"])
    variance_horizon = int(contract["variance_horizon"])
    checks: dict[str, bool] = {"finite": bool(report.get("finite"))}
    for horizon_value in gate_horizons:
        horizon = str(horizon_value)
        metrics = rollout[horizon]
        model_error = float(metrics["model"]["normalized_smooth_l1"])
        checks[f"beats_persistence_{horizon}"] = model_error < float(
            metrics["persistence"]["normalized_smooth_l1"]
        )
        checks[f"beats_linear_{horizon}"] = model_error < float(
            metrics["linear"]["normalized_smooth_l1"]
        )
    variance_ratio = float(report["prediction_variance_ratio"])
    checks[
        f"variance_ratio_{variance_horizon}"
    ] = 0.5 <= variance_ratio <= 2.0
    return {"passed": all(checks.values()), "checks": checks}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


class _ErrorAccumulator:
    def __init__(self) -> None:
        self.smooth_l1 = 0.0
        self.square = 0.0
        self.count = 0

    def update(self, prediction: Tensor, target: Tensor, scale: Tensor) -> None:
        normalized_prediction = prediction.float() / scale
        normalized_target = target.float() / scale
        self.smooth_l1 += float(
            functional.smooth_l1_loss(
                normalized_prediction,
                normalized_target,
                reduction="sum",
            ).item()
        )
        self.square += float(
            torch.square(normalized_prediction - normalized_target).sum().item()
        )
        self.count += target.numel()

    def result(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("cannot finalize empty evaluation accumulator")
        return {
            "normalized_smooth_l1": self.smooth_l1 / self.count,
            "normalized_rmse": math.sqrt(self.square / self.count),
        }


def _fixed_windows(
    index: dict[str, Any],
    *,
    split: str,
    context_frames: int,
    future_frames: int,
    seed: int,
) -> tuple[list[int], list[int], str]:
    sequence_indices: list[int] = []
    starts: list[int] = []
    audit: list[dict[str, object]] = []
    presets: set[str] = set()
    total = context_frames + future_frames
    for sequence in index["sequences"]:
        if sequence["split"] != split:
            continue
        length = int(sequence["length"])
        if length < total:
            raise ValueError(
                f"evaluation sequence is shorter than {total}: "
                f"{sequence['sample_id']}"
            )
        payload = (
            f"{seed}:zrave-evaluation:{split}:{sequence['sample_id']}"
        ).encode("utf-8")
        start = int.from_bytes(
            hashlib.sha256(payload).digest()[:8],
            "big",
        ) % (length - total + 1)
        sequence_indices.append(int(sequence["index"]))
        starts.append(start)
        presets.add(str(sequence["preset_id"]))
        audit.append(
            {
                "sample_id": sequence["sample_id"],
                "preset_id": sequence["preset_id"],
                "start": start,
            }
        )
    if not sequence_indices:
        raise ValueError(f"packed dataset has no {split} evaluation sequence")
    audit_hash = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if not presets:
        raise ValueError(f"packed dataset has no {split} preset")
    return sequence_indices, starts, audit_hash


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _evaluation_contract(
    config: ZraveConfig,
    index: dict[str, Any],
    split: str,
) -> dict[str, object]:
    lengths = [
        int(sequence["length"])
        for sequence in index["sequences"]
        if sequence["split"] == split
    ]
    if not lengths:
        raise ValueError(f"packed dataset has no {split} sequence")
    available_future = min(lengths) - config.model.context_frames
    rollout_horizons = tuple(
        horizon
        for horizon in (8, 16, 32, 64, 128)
        if horizon <= available_future
    )
    if len(rollout_horizons) < 2:
        raise ValueError(
            f"evaluation needs at least 16 future frames, "
            f"found {available_future}"
        )
    teacher_horizons = tuple(
        horizon
        for horizon in (1, 2, 4, 8, 16)
        if horizon <= config.model.horizon_frames
    )
    if not teacher_horizons:
        raise ValueError("model horizon is too short for evaluation")
    maximum = rollout_horizons[-1]
    preferred_short = 32 if maximum >= 128 else 16
    short = max(
        horizon
        for horizon in rollout_horizons
        if horizon <= preferred_short
    )
    return {
        "available_future_frames": available_future,
        "teacher_horizons": list(teacher_horizons),
        "rollout_horizons": list(rollout_horizons),
        "gate_horizons": [short, maximum],
        "variance_horizon": maximum,
    }


@torch.inference_mode()
def evaluate_checkpoint(
    config: ZraveConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    statistics = _load_statistics(packed_root)
    model = ZraveTransformer(config.model, statistics).to(device)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("evaluation requires a format-1 Z-RAVE checkpoint")
    if payload.get("architecture") != "zrave_transformer_v1":
        raise ValueError("evaluation checkpoint architecture mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("evaluation checkpoint has no immutable contract")
    expected_contract = {
        "packed_index_sha256": _sha256_file(index_path),
        "statistics_sha256": _sha256_file(statistics_path),
        "latent_dim": config.model.latent_dim,
        "context_frames": config.model.context_frames,
        "horizon_frames": config.model.horizon_frames,
    }
    for name, expected in expected_contract.items():
        if contract.get(name) != expected:
            raise ValueError(f"evaluation checkpoint {name} mismatch")
    model.load_state_dict(payload["model"])
    model.eval()

    evaluation_contract = _evaluation_contract(config, index, split)
    maximum_horizon = int(evaluation_contract["variance_horizon"])
    teacher_horizons = tuple(
        int(value) for value in evaluation_contract["teacher_horizons"]
    )
    rollout_horizons = tuple(
        int(value) for value in evaluation_contract["rollout_horizons"]
    )
    latents = torch.from_numpy(
        np.load(packed_root / "latents.npy", allow_pickle=False)
    ).to(device=device)
    sequence_indices, starts, window_hash = _fixed_windows(
        index,
        split=split,
        context_frames=config.model.context_frames,
        future_frames=maximum_horizon,
        seed=config.seed,
    )
    latent_scale = statistics.latent_std.to(
        device=device,
        dtype=torch.float32,
    )
    delta_scale = statistics.delta_std.to(
        device=device,
        dtype=torch.float32,
    )
    teacher_latent = {
        horizon: _ErrorAccumulator() for horizon in teacher_horizons
    }
    teacher_delta = {
        horizon: _ErrorAccumulator() for horizon in teacher_horizons
    }
    rollout_accumulators = {
        horizon: {
            name: _ErrorAccumulator()
            for name in ("model", "persistence", "linear")
        }
        for horizon in rollout_horizons
    }
    prediction_sum = 0.0
    prediction_square_sum = 0.0
    reference_sum = 0.0
    reference_square_sum = 0.0
    variance_count = 0
    total_frames = config.model.context_frames + maximum_horizon
    offsets = torch.arange(total_frames, device=device)

    for offset in range(0, len(sequence_indices), batch_size):
        chosen_sequences = torch.tensor(
            sequence_indices[offset : offset + batch_size],
            device=device,
            dtype=torch.long,
        )
        chosen_starts = torch.tensor(
            starts[offset : offset + batch_size],
            device=device,
            dtype=torch.long,
        )
        frame_indices = chosen_starts[:, None] + offsets[None]
        windows = latents[chosen_sequences[:, None], frame_indices]
        history = windows[:, : config.model.context_frames]
        target = windows[:, config.model.context_frames :]
        direct = model(history)
        target_direct = target[:, : config.model.horizon_frames]
        target_delta = torch.diff(
            torch.cat([history[:, -1:], target_direct], dim=1).float(),
            dim=1,
        )
        for horizon in teacher_horizons:
            teacher_latent[horizon].update(
                direct.latent[:, :horizon],
                target_direct[:, :horizon],
                latent_scale,
            )
            teacher_delta[horizon].update(
                direct.delta[:, :horizon],
                target_delta[:, :horizon],
                delta_scale,
            )

        predicted = rollout_prediction(model, history, maximum_horizon)
        baselines = {
            "model": predicted,
            "persistence": persistence_baseline(
                history,
                maximum_horizon,
            ),
            "linear": linear_baseline(history, maximum_horizon),
        }
        for horizon in rollout_horizons:
            for name, value in baselines.items():
                rollout_accumulators[horizon][name].update(
                    value[:, :horizon],
                    target[:, :horizon],
                    latent_scale,
                )
        prediction_full = predicted.float()
        reference_full = target.float()
        prediction_sum += float(prediction_full.sum().item())
        prediction_square_sum += float(
            torch.square(prediction_full).sum().item()
        )
        reference_sum += float(reference_full.sum().item())
        reference_square_sum += float(
            torch.square(reference_full).sum().item()
        )
        variance_count += reference_full.numel()

    prediction_variance = max(
        0.0,
        prediction_square_sum / variance_count
        - (prediction_sum / variance_count) ** 2,
    )
    reference_variance = max(
        0.0,
        reference_square_sum / variance_count
        - (reference_sum / variance_count) ** 2,
    )
    variance_ratio = prediction_variance / max(reference_variance, 1.0e-12)
    teacher_report: dict[str, dict[str, float]] = {}
    for horizon in teacher_horizons:
        teacher_report[str(horizon)] = {
            **teacher_latent[horizon].result(),
            "normalized_delta_smooth_l1": (
                teacher_delta[horizon].result()["normalized_smooth_l1"]
            ),
            "normalized_delta_rmse": (
                teacher_delta[horizon].result()["normalized_rmse"]
            ),
        }
    rollout_report = {
        str(horizon): {
            name: accumulator.result()
            for name, accumulator in rollout_accumulators[horizon].items()
        }
        for horizon in rollout_horizons
    }
    report: dict[str, Any] = {
        "schema": 1,
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_update": int(payload["update"]),
        "packed_index_sha256": _sha256_file(index_path),
        "statistics_sha256": _sha256_file(statistics_path),
        "evaluation_window_sha256": window_hash,
        "windows": len(sequence_indices),
        "evaluation_contract": evaluation_contract,
        "teacher_forced": teacher_report,
        "rollout": rollout_report,
        "prediction_variance": prediction_variance,
        "reference_variance": reference_variance,
        "prediction_variance_ratio": variance_ratio,
    }
    report["finite"] = _all_finite(report)
    report["acceptance"] = acceptance_gate(report)
    return report


def _atomic_json(path: Path, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Z-RAVE Transformer against fixed baselines."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = evaluate_checkpoint(
        ZraveConfig.load(args.config),
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if (
        args.split == "test"
        and not report["acceptance"]["passed"]
        and not args.allow_failed_gate
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
