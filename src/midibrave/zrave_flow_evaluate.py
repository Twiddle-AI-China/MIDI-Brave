from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn

from .zrave_codec import decode_with_seed
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    sample_flow_block,
)


_CATEGORIES = (
    "Pad",
    "Lead",
    "Bass",
    "Pluck",
    "Keys",
    "Arp",
    "Chord",
    "Synth",
)
_SEEDS = (17, 29, 43, 71)
_CONTROLS = (
    (0.7, 32),
    (1.0, 16),
    (1.0, 32),
    (1.0, 48),
    (1.3, 32),
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def rollout_flow(
    model: object,
    statistics: FlowStatistics,
    history: Tensor,
    midi_note: Tensor,
    frames: int,
    *,
    generation_seed: int,
    temperature: float,
    wander_delay_frames: int,
    pitch_guidance: float = 3.0,
    solver_steps: int = 8,
    sample_block: Callable[..., Tensor] = sample_flow_block,
) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if history.ndim != 3 or history.shape[1:] != (32, 16):
        raise ValueError("history must have shape [batch, 32, 16]")
    if midi_note.shape != (history.shape[0],):
        raise ValueError("midi_note must have shape [batch]")
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    block_index = 0
    while remaining:
        generated = sample_block(
            model,
            statistics,
            current,
            midi_note,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            pitch_guidance=pitch_guidance,
            solver_steps=solver_steps,
        )
        if generated.shape != (history.shape[0], 64, 16):
            raise ValueError(
                "flow block sampler must return [batch, 64, 16]"
            )
        take = min(remaining, 64)
        chunk = generated[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[:, -32:].detach()
        remaining -= take
        block_index += 1
    return torch.cat(chunks, dim=1)


def flow_acceptance_gate(
    report: dict[str, object],
) -> dict[str, object]:
    numeric_thresholds: tuple[
        tuple[str, Callable[[float], bool]], ...
    ] = (
        ("f0_median_cents", lambda value: value <= 50.0),
        ("f0_p90_cents", lambda value: value <= 100.0),
        ("voiced_following", lambda value: value >= 0.90),
        ("swapped_f0_median_cents", lambda value: value <= 50.0),
        ("swapped_f0_p90_cents", lambda value: value <= 100.0),
        ("swapped_voiced_following", lambda value: value >= 0.90),
        ("nonfinite_renders", lambda value: value == 0.0),
        (
            "boundary_jump_over_real_p95",
            lambda value: value <= 1.0,
        ),
        ("silence_fraction", lambda value: value <= 0.01),
        (
            "maximum_short_cycle_autocorrelation",
            lambda value: value <= 0.95,
        ),
        ("tail_delta_ratio", lambda value: value >= 0.25),
        (
            "tail_seed_diversity_ratio",
            lambda value: value >= 0.50,
        ),
        (
            "first_second_seed_diversity_ratio",
            lambda value: value < 1.0,
        ),
        ("nearest_training_nrmse", lambda value: value > 0.001),
    )
    gates: dict[str, bool] = {}
    for name, predicate in numeric_thresholds:
        try:
            value = float(report[name])
        except (KeyError, TypeError, ValueError):
            gates[name] = False
            continue
        gates[name] = math.isfinite(value) and predicate(value)
    gates["exact_reproduction"] = (
        report.get("exact_reproduction") is True
    )
    return {"passed": all(gates.values()), "gates": gates}


@dataclass
class GateEarlyStopState:
    required_consecutive_passes: int
    consecutive_passes: int = 0

    def __post_init__(self) -> None:
        if self.required_consecutive_passes <= 0:
            raise ValueError(
                "required_consecutive_passes must be positive"
            )
        if self.consecutive_passes < 0:
            raise ValueError("consecutive_passes must be non-negative")

    def update(self, gate: dict[str, object]) -> bool:
        if gate.get("passed") is True:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        return self.consecutive_passes >= self.required_consecutive_passes

    def state_dict(self) -> dict[str, int]:
        return {
            "required_consecutive_passes": (
                self.required_consecutive_passes
            ),
            "consecutive_passes": self.consecutive_passes,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if set(state) != {
            "required_consecutive_passes",
            "consecutive_passes",
        }:
            raise ValueError("invalid gate early-stop state")
        if (
            int(state["required_consecutive_passes"])
            != self.required_consecutive_passes
        ):
            raise ValueError("gate early-stop contract mismatch")
        passes = int(state["consecutive_passes"])
        if passes < 0:
            raise ValueError("consecutive gate passes cannot be negative")
        self.consecutive_passes = passes


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _state_dict_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not an object"
                )
            rows.append(value)
    return rows


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


def _sequence_latent(
    packed_root: Path,
    row: dict[str, Any],
) -> np.ndarray:
    path = packed_root / str(row["shard"])
    with np.load(path, allow_pickle=False) as values:
        latent = values["latents"][
            int(row["shard_row"]),
            : int(row["length"]),
        ].astype(np.float32)
    return latent


def _fixed_cases(
    rows: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in _CATEGORIES:
        candidates = [
            row
            for row in rows
            if str(row["split"]) == split
            and str(row["category"]).casefold() == category.casefold()
            and int(row["active_frames"]) >= 33
        ]
        if not candidates:
            raise ValueError(
                f"{split} split has no active {category} evaluation row"
            )
        candidates.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:{split}:{category}:"
                    f"{row['canonical_preset_id']}:{row['sample_id']}"
                ).encode("utf-8")
            ).digest()
        )
        selected.append(candidates[0])
    return selected


@dataclass(frozen=True)
class _ReferenceBank:
    normalized: Tensor
    real_delta_rms: float
    real_boundary_p95: float
    same_note_different_preset_distance: float
    audit_sha256: str


def _build_reference_bank(
    packed_root: Path,
    rows: list[dict[str, Any]],
    statistics: FlowStatistics,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> _ReferenceBank:
    eligible = [
        row
        for row in rows
        if str(row["split"]) == "train"
        and int(row["active_frames"]) >= 64
    ]
    if not eligible:
        raise ValueError("training split has no 64-frame reference rows")
    eligible.sort(
        key=lambda row: (
            str(row["sample_id"]),
            int(row["packed_index"]),
        )
    )
    specifications: list[dict[str, Any]] = []
    for index in range(count):
        digest = hashlib.sha256(
            f"{seed}:training-reference:{index}".encode("utf-8")
        ).digest()
        row = eligible[
            int.from_bytes(digest[:8], "big") % len(eligible)
        ]
        maximum_start = int(row["active_frames"]) - 64
        start = (
            int.from_bytes(digest[8:16], "big")
            % (maximum_start + 1)
        )
        specifications.append(
            {
                "index": index,
                "shard": str(row["shard"]),
                "shard_row": int(row["shard_row"]),
                "start": start,
                "midi_note": int(row["midi_note"]),
                "preset": str(row["canonical_preset_id"]),
                "sample_id": str(row["sample_id"]),
            }
        )
    raw = np.empty((count, 64, 16), dtype=np.float32)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for specification in specifications:
        grouped[str(specification["shard"])].append(specification)
    for shard, shard_specs in grouped.items():
        with np.load(
            packed_root / shard,
            allow_pickle=False,
        ) as values:
            latents = values["latents"]
            for specification in shard_specs:
                start = int(specification["start"])
                raw[int(specification["index"])] = latents[
                    int(specification["shard_row"]),
                    start : start + 64,
                ].astype(np.float32)
    deltas = np.diff(raw, axis=1)
    delta_norm = np.linalg.norm(deltas, axis=-1)
    real_delta_rms = float(np.sqrt(np.mean(np.square(deltas))))
    real_boundary_p95 = float(np.percentile(delta_norm, 95))
    mean = statistics.mean.cpu().numpy().reshape(1, 1, 16)
    std = statistics.latent_std.cpu().numpy().reshape(1, 1, 16)
    normalized = (raw - mean) / std

    by_note: dict[int, list[int]] = defaultdict(list)
    for specification in specifications:
        by_note[int(specification["midi_note"])].append(
            int(specification["index"])
        )
    distances: list[float] = []
    for indices in by_note.values():
        for left_position, left in enumerate(indices):
            left_preset = specifications[left]["preset"]
            right = next(
                (
                    candidate
                    for candidate in indices[left_position + 1 :]
                    if specifications[candidate]["preset"] != left_preset
                ),
                None,
            )
            if right is not None:
                distances.append(
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    normalized[left]
                                    - normalized[right]
                                )
                            )
                        )
                    )
                )
            if len(distances) >= 512:
                break
        if len(distances) >= 512:
            break
    if not distances or min(distances) <= 0.0:
        raise ValueError(
            "reference bank lacks same-note different-preset pairs"
        )
    return _ReferenceBank(
        normalized=torch.from_numpy(normalized).to(device),
        real_delta_rms=real_delta_rms,
        real_boundary_p95=real_boundary_p95,
        same_note_different_preset_distance=float(
            np.median(np.asarray(distances, dtype=np.float64))
        ),
        audit_sha256=_sha256_json(specifications),
    )


def _nearest_training_nrmse(
    generated: Tensor,
    reference: Tensor,
) -> float:
    if generated.ndim != 3 or generated.shape[1:] != (64, 16):
        raise ValueError("nearest-neighbor input must be [rows, 64, 16]")
    left = generated.float().reshape(generated.shape[0], -1)
    right = reference.float().reshape(reference.shape[0], -1)
    minimum = torch.full(
        (left.shape[0],),
        math.inf,
        device=left.device,
    )
    dimension_scale = math.sqrt(left.shape[1])
    for start in range(0, right.shape[0], 512):
        distance = torch.cdist(
            left,
            right[start : start + 512],
        ) / dimension_scale
        minimum = torch.minimum(minimum, distance.min(dim=1).values)
    return float(minimum.min().item())


def _normalized_autocorrelation(latent: Tensor, lag: int) -> float:
    centered = latent.float() - latent.float().mean(dim=0, keepdim=True)
    left = centered[:-lag].reshape(-1)
    right = centered[lag:].reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1.0e-12:
        return 1.0
    return float(torch.dot(left, right).abs() / denominator)


def _silence_fraction(audio: Tensor, sample_rate: int) -> float:
    waveform = audio.float().reshape(-1)
    window = max(1, round(0.050 * sample_rate))
    padding = (-waveform.numel()) % window
    if padding:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    rms = waveform.reshape(-1, window).square().mean(dim=1).sqrt()
    return float((rms < 1.0e-3).float().mean().item())


@torch.no_grad()
def _audio_pitch_metrics(
    audio: Tensor,
    midi_note: int,
    sample_rate: int,
) -> tuple[list[float], int, int]:
    import torchcrepe

    pitch, periodicity = torchcrepe.predict(
        audio.float().reshape(1, -1),
        sample_rate,
        512,
        20.0,
        4000.0,
        "tiny",
        batch_size=1024,
        device=audio.device,
        return_periodicity=True,
    )
    target_hz = 440.0 * 2.0 ** ((midi_note - 69.0) / 12.0)
    cents = 1200.0 * torch.log2(
        (pitch.float() + 1.0e-7) / target_hz
    )
    finite = torch.isfinite(cents) & torch.isfinite(periodicity)
    voiced = finite & (periodicity >= 0.5)
    absolute = cents.abs()
    errors = absolute.masked_select(voiced).cpu().tolist()
    followed = int((voiced & (absolute <= 100.0)).sum().item())
    return errors, followed, int(finite.sum().item())


def _pairwise_distance(values: list[Tensor]) -> float:
    distances = [
        float((left.float() - right.float()).square().mean().sqrt())
        for left, right in itertools.combinations(values, 2)
    ]
    if not distances:
        raise ValueError("seed diversity needs at least two values")
    return float(np.mean(np.asarray(distances, dtype=np.float64)))


def _load_codec(
    config: ZraveFlowConfig,
    device: torch.device,
    injected: nn.Module | None,
) -> tuple[nn.Module, str]:
    if injected is None:
        path = Path(config.rave.checkpoint)
        digest = _sha256_file(path)
        if digest != config.rave.expected_sha256:
            raise ValueError("RAVE codec hash mismatch")
        codec = torch.jit.load(str(path), map_location=device)
    else:
        digest = config.rave.expected_sha256
        codec = injected
    codec = codec.to(device).eval()
    latent_size = getattr(codec, "latent_size")
    if isinstance(latent_size, Tensor):
        latent_size = int(latent_size.flatten()[0])
    elif hasattr(latent_size, "__len__"):
        latent_size = int(latent_size[0])
    else:
        latent_size = int(latent_size)
    if latent_size != 16:
        raise ValueError("RAVE codec latent size is not 16")
    return codec, digest


def _decode(codec: nn.Module, latent: Tensor, seed: int) -> Tensor:
    layout = latent.transpose(1, 2).contiguous()
    return decode_with_seed(codec, layout, seed).float()


def _safe_name(value: float) -> str:
    return str(value).replace(".", "p")


def _model_from_checkpoint(
    config: ZraveFlowConfig,
    statistics: FlowStatistics,
    checkpoint_path: Path,
    index_hash: str,
    statistics_hash: str,
    device: torch.device,
    injected: ZraveFlowTransformer | None,
) -> tuple[ZraveFlowTransformer, dict[str, object]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("format") != 1
        or payload.get("architecture")
        != "zrave_conditional_flow_transformer_v1"
    ):
        raise ValueError("evaluation checkpoint architecture mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("evaluation checkpoint lacks contract")
    expected = {
        "pack_index_sha256": index_hash,
        "statistics_sha256": statistics_hash,
        "latent_dim": config.model.latent_dim,
        "context_frames": config.model.context_frames,
        "future_frames": config.model.future_frames,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"evaluation checkpoint {name} mismatch")
    model = (
        ZraveFlowTransformer(config.model, statistics).to(device)
        if injected is None
        else injected
    )
    model.load_state_dict(payload["model"])
    return model, payload


def _rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state: dict[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


@torch.inference_mode()
def _evaluate_flow_checkpoint_impl(
    config: ZraveFlowConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
    output_root: str | Path,
    device: str | torch.device,
    model: ZraveFlowTransformer | None = None,
    codec: nn.Module | None = None,
    reference_rows: int = 10000,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if reference_rows != 10000:
        raise ValueError("reference_rows must be exactly 10000")
    selected_device = torch.device(device)
    rng = _rng_state(selected_device)
    deterministic_before = (
        torch.are_deterministic_algorithms_enabled()
    )
    torch.use_deterministic_algorithms(True)
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    checkpoint = Path(checkpoint_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    index_hash = _sha256_file(index_path)
    statistics_hash = _sha256_file(statistics_path)
    checkpoint_hash = _sha256_file(checkpoint)
    config_hash = (
        _sha256_file(config.source_path)
        if config.source_path is not None
        else _sha256_json(config.as_dict())
    )
    statistics = _load_statistics(packed_root)
    flow_model, payload = _model_from_checkpoint(
        config,
        statistics,
        checkpoint,
        index_hash,
        statistics_hash,
        selected_device,
        model,
    )
    was_training = flow_model.training
    flow_model.eval()
    model_hash = _state_dict_sha256(flow_model.state_dict())
    rave, codec_hash = _load_codec(config, selected_device, codec)
    rows = _read_jsonl(packed_root / "sequences.jsonl")
    cases = _fixed_cases(rows, split=split, seed=config.seed)
    reference = _build_reference_bank(
        packed_root,
        rows,
        statistics,
        count=reference_rows,
        seed=config.seed,
        device=selected_device,
    )

    rendered_rows: list[dict[str, object]] = []
    generated_short_normalized: list[Tensor] = []
    boundary_jumps: list[float] = []
    tail_delta_rms: list[float] = []
    silence_fractions: list[float] = []
    autocorrelations: list[float] = []
    latent_norm_violations = 0
    latent_norm_count = 0
    nonfinite = 0
    exact = True
    matched_errors: list[float] = []
    swapped_errors: list[float] = []
    matched_followed = 0
    matched_frames = 0
    swapped_followed = 0
    swapped_frames = 0
    diversity_groups: dict[
        tuple[str, int, float, int],
        list[Tensor],
    ] = defaultdict(list)
    first_groups: dict[
        tuple[str, int, float, int],
        list[Tensor],
    ] = defaultdict(list)
    wav_root = output / "wavs"
    wav_root.mkdir(parents=True, exist_ok=True)
    mean = statistics.mean.to(selected_device)
    std = statistics.latent_std.to(selected_device)
    lower = float(statistics.latent_norm_p01)
    upper = float(statistics.latent_norm_p99)
    first_second_frames = max(
        1,
        round(config.rave.sample_rate / config.rave.latent_hop),
    )

    for case_index, case in enumerate(cases):
        sequence = torch.from_numpy(
            _sequence_latent(packed_root, case)
        ).to(selected_device)
        maximum_start = int(case["active_frames"]) - 33
        digest = hashlib.sha256(
            (
                f"{config.seed}:{split}:history:{case['sample_id']}"
            ).encode("utf-8")
        ).digest()
        start = int.from_bytes(digest[:8], "big") % (
            maximum_start + 1
        )
        history = sequence[start : start + 32].unsqueeze(0)
        recorded_note = int(case["midi_note"])
        swapped_note = (
            recorded_note + 12
            if recorded_note + 12 <= config.model.note_max
            else recorded_note - 12
        )
        for note_kind, note in (
            ("matched", recorded_note),
            ("swapped", swapped_note),
        ):
            note_tensor = torch.tensor([note], device=selected_device)
            for temperature, delay in _CONTROLS:
                for generation_seed in _SEEDS:
                    short = rollout_flow(
                        flow_model,
                        statistics,
                        history,
                        note_tensor,
                        64,
                        generation_seed=generation_seed,
                        temperature=temperature,
                        wander_delay_frames=delay,
                    )
                    long = rollout_flow(
                        flow_model,
                        statistics,
                        history,
                        note_tensor,
                        320,
                        generation_seed=generation_seed,
                        temperature=temperature,
                        wander_delay_frames=delay,
                    )
                    exact &= torch.equal(short, long[:, :64])
                    finite_latent = bool(
                        torch.isfinite(short).all()
                        and torch.isfinite(long).all()
                    )
                    if not finite_latent:
                        nonfinite += 1
                    safe_short = torch.nan_to_num(short)
                    safe_long = torch.nan_to_num(long)
                    normalized_short = (safe_short - mean) / std
                    generated_short_normalized.append(
                        normalized_short.squeeze(0)
                    )
                    boundary_jumps.append(
                        float(
                            torch.linalg.vector_norm(
                                safe_long[:, 0] - history[:, -1],
                                dim=-1,
                            ).max()
                        )
                    )
                    tail_delta_rms.append(
                        float(
                            (
                                safe_long[:, -64:, :][:, 1:]
                                - safe_long[:, -64:, :][:, :-1]
                            )
                            .square()
                            .mean()
                            .sqrt()
                        )
                    )
                    norms = torch.linalg.vector_norm(
                        safe_long,
                        dim=-1,
                    )
                    latent_norm_violations += int(
                        ((norms < lower) | (norms > upper)).sum()
                    )
                    latent_norm_count += norms.numel()
                    autocorrelations.append(
                        max(
                            _normalized_autocorrelation(
                                safe_long.squeeze(0),
                                lag,
                            )
                            for lag in range(1, 9)
                        )
                    )
                    group = (
                        str(case["sample_id"]),
                        note,
                        temperature,
                        delay,
                    )
                    diversity_groups[group].append(
                        ((safe_long[:, -64:] - mean) / std).squeeze(0)
                    )
                    first_groups[group].append(
                        ((safe_long[:, :first_second_frames] - mean) / std)
                        .squeeze(0)
                    )
                    short_audio = _decode(
                        rave,
                        safe_short,
                        config.seed,
                    )
                    long_audio = _decode(
                        rave,
                        safe_long,
                        config.seed,
                    )
                    finite_audio = bool(
                        torch.isfinite(short_audio).all()
                        and torch.isfinite(long_audio).all()
                    )
                    if not finite_audio:
                        nonfinite += 1
                    short_audio = torch.nan_to_num(short_audio)
                    long_audio = torch.nan_to_num(long_audio)
                    silence = _silence_fraction(
                        long_audio,
                        config.rave.sample_rate,
                    )
                    silence_fractions.append(silence)
                    errors, followed, frames = _audio_pitch_metrics(
                        short_audio,
                        note,
                        config.rave.sample_rate,
                    )
                    if note_kind == "matched":
                        matched_errors.extend(errors)
                        matched_followed += followed
                        matched_frames += frames
                    else:
                        swapped_errors.extend(errors)
                        swapped_followed += followed
                        swapped_frames += frames
                    prefix = (
                        f"{case_index:02d}-{case['category']}-"
                        f"{note_kind}-n{note}-s{generation_seed}-"
                        f"t{_safe_name(temperature)}-d{delay}"
                    )
                    for length, audio, latent in (
                        (64, short_audio, short),
                        (320, long_audio, long),
                    ):
                        wav_path = wav_root / f"{prefix}-f{length}.wav"
                        sf.write(
                            wav_path,
                            audio.squeeze().cpu().numpy().astype(
                                np.float32
                            ),
                            config.rave.sample_rate,
                            subtype="FLOAT",
                        )
                        rendered_rows.append(
                            {
                                "split": split,
                                "sample_id": case["sample_id"],
                                "preset_id": case[
                                    "canonical_preset_id"
                                ],
                                "category": case["category"],
                                "history_start": start,
                                "recorded_midi_note": recorded_note,
                                "condition_midi_note": note,
                                "condition_kind": note_kind,
                                "generation_seed": generation_seed,
                                "temperature": temperature,
                                "wander_delay_frames": delay,
                                "frames": length,
                                "finite_latent": finite_latent,
                                "finite_audio": finite_audio,
                                "silence_fraction": (
                                    silence if length == 320 else None
                                ),
                                "wav": str(
                                    wav_path.relative_to(output)
                                ),
                                "codec_sha256": codec_hash,
                                "model_sha256": model_hash,
                                "evaluated_checkpoint_sha256": (
                                    checkpoint_hash
                                ),
                                "config_sha256": config_hash,
                                "pack_index_sha256": index_hash,
                                "latent_sha256": hashlib.sha256(
                                    latent.detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .tobytes()
                                ).hexdigest(),
                            }
                        )

    normalized_generated = torch.stack(
        generated_short_normalized,
        dim=0,
    )
    nearest = _nearest_training_nrmse(
        normalized_generated,
        reference.normalized,
    )
    tail_seed_distance = float(
        np.median(
            np.asarray(
                [
                    _pairwise_distance(values)
                    for values in diversity_groups.values()
                ],
                dtype=np.float64,
            )
        )
    )
    first_seed_distance = float(
        np.median(
            np.asarray(
                [
                    _pairwise_distance(values)
                    for values in first_groups.values()
                ],
                dtype=np.float64,
            )
        )
    )

    def _pitch_summary(
        errors: list[float],
        followed: int,
        frames: int,
    ) -> tuple[float, float, float]:
        if not errors or frames <= 0:
            return 1.0e30, 1.0e30, 0.0
        values = np.asarray(errors, dtype=np.float64)
        return (
            float(np.median(values)),
            float(np.percentile(values, 90)),
            followed / frames,
        )

    matched_median, matched_p90, matched_following = _pitch_summary(
        matched_errors,
        matched_followed,
        matched_frames,
    )
    swapped_median, swapped_p90, swapped_following = _pitch_summary(
        swapped_errors,
        swapped_followed,
        swapped_frames,
    )
    report: dict[str, object] = {
        "f0_median_cents": matched_median,
        "f0_p90_cents": matched_p90,
        "voiced_following": matched_following,
        "swapped_f0_median_cents": swapped_median,
        "swapped_f0_p90_cents": swapped_p90,
        "swapped_voiced_following": swapped_following,
        "exact_reproduction": exact,
        "nonfinite_renders": nonfinite,
        "boundary_jump_over_real_p95": (
            max(boundary_jumps) / reference.real_boundary_p95
        ),
        "silence_fraction": max(silence_fractions),
        "maximum_short_cycle_autocorrelation": max(autocorrelations),
        "tail_delta_ratio": (
            float(np.median(tail_delta_rms))
            / reference.real_delta_rms
        ),
        "tail_seed_diversity_ratio": (
            tail_seed_distance
            / reference.same_note_different_preset_distance
        ),
        "first_second_seed_diversity_ratio": (
            first_seed_distance
            / reference.same_note_different_preset_distance
        ),
        "nearest_training_nrmse": nearest,
        "latent_norm_violation_fraction": (
            latent_norm_violations / max(1, latent_norm_count)
        ),
    }
    gate = flow_acceptance_gate(report)
    evaluation: dict[str, object] = {
        **report,
        "gate": gate,
        "passed": gate["passed"],
        "split": split,
        "render_rows": len(rendered_rows),
        "fixed_case_sample_ids": [
            case["sample_id"] for case in cases
        ],
        "reference_rows": reference_rows,
        "reference_audit_sha256": reference.audit_sha256,
        "real_boundary_p95": reference.real_boundary_p95,
        "real_delta_rms": reference.real_delta_rms,
        "same_note_different_preset_distance": (
            reference.same_note_different_preset_distance
        ),
        "codec_sha256": codec_hash,
        "model_sha256": model_hash,
        "evaluated_checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "pack_index_sha256": index_hash,
        "statistics_sha256": statistics_hash,
        "checkpoint_update": int(payload["update"]),
    }
    _atomic_jsonl(output / "rows.jsonl", rendered_rows)
    _atomic_json(output / "evaluation.json", evaluation)
    if was_training:
        flow_model.train()
    _restore_rng(rng, selected_device)
    torch.use_deterministic_algorithms(deterministic_before)
    return evaluation


def evaluate_flow_checkpoint(
    config: ZraveFlowConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
    output_root: str | Path,
    device: str | torch.device,
    model: ZraveFlowTransformer | None = None,
    codec: nn.Module | None = None,
    reference_rows: int = 10000,
) -> dict[str, object]:
    selected_device = torch.device(device)
    state = _rng_state(selected_device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    model_was_training = model.training if model is not None else False
    try:
        return _evaluate_flow_checkpoint_impl(
            config,
            checkpoint_path,
            split=split,
            output_root=output_root,
            device=selected_device,
            model=model,
            codec=codec,
            reference_rows=reference_rows,
        )
    finally:
        _restore_rng(state, selected_device)
        torch.use_deterministic_algorithms(deterministic)
        if model is not None and model_was_training:
            model.train()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate stochastic Z-RAVE flow checkpoints."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("validation", "test"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    torch.use_deterministic_algorithms(True)
    config = ZraveFlowConfig.load(args.config)
    evaluate_flow_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
        output_root=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
