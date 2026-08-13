from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


_ENVELOPE_FRAME_SECONDS = 0.020
_ENVELOPE_HOP_SECONDS = 0.005
_ENVELOPE_SILENCE_FLOOR_DBFS = -80.0
_ENVELOPE_ONSET_PEAK_RATIO = 0.05
_ENVELOPE_ATTACK_PEAK_RATIO = 0.90
_ENVELOPE_ATTACK_HOLD_SECONDS = 0.015
_ENVELOPE_SUSTAIN_START_FRACTION = 0.35
_ENVELOPE_SUSTAIN_END_FRACTION = 0.65
_ENVELOPE_TAIL_FRACTION = 0.25
_ENVELOPE_MAXIMUM_TAIL_SECONDS = 1.0
_ENVELOPE_LAST_RMS_SECONDS = 0.100
_ENVELOPE_DBFS_FLOOR = -120.0
_ENVELOPE_SUMMARY_METRICS = (
    "envelope_peak_rms_dbfs_proxy",
    "envelope_attack_ms_proxy",
    "envelope_decay_slope_db_per_second_proxy",
    "envelope_peak_to_sustain_drop_db_proxy",
    "envelope_sustain_rms_dbfs_proxy",
    "envelope_sustain_variation_db_proxy",
    "envelope_tail_release_slope_db_per_second_proxy",
    "envelope_tail_rms_dbfs_proxy",
)


def _envelope_algorithm_proxy() -> dict[str, object]:
    return {
        "schema": 1,
        "report_only": True,
        "claim": (
            "Relative full-WAV amplitude-envelope proxies; these are not "
            "true ADSR parameters because the audition renders do not carry "
            "an authoritative note-off time."
        ),
        "analysis_scope": "full_mono_wav",
        "envelope": "sliding_rms",
        "frame_seconds": _ENVELOPE_FRAME_SECONDS,
        "hop_seconds": _ENVELOPE_HOP_SECONDS,
        "dbfs_floor": _ENVELOPE_DBFS_FLOOR,
        "silence_floor_dbfs": _ENVELOPE_SILENCE_FLOOR_DBFS,
        "attack": {
            "onset_peak_ratio": _ENVELOPE_ONSET_PEAK_RATIO,
            "crossing_peak_ratio": _ENVELOPE_ATTACK_PEAK_RATIO,
            "minimum_hold_seconds": _ENVELOPE_ATTACK_HOLD_SECONDS,
            "definition": (
                "elapsed time from the first active RMS frame to the first "
                "sustained crossing of the peak-relative threshold"
            ),
        },
        "sustain": {
            "start_fraction": _ENVELOPE_SUSTAIN_START_FRACTION,
            "end_fraction": _ENVELOPE_SUSTAIN_END_FRACTION,
            "rms_definition": "root mean square of frame RMS values",
            "variation_definition": "p90 minus p10 frame RMS dBFS",
        },
        "decay": {
            "definition": (
                "signed dB-per-second line from peak RMS to the center of "
                "the sustain proxy window; null when the peak is not earlier"
            )
        },
        "tail": {
            "fraction": _ENVELOPE_TAIL_FRACTION,
            "maximum_seconds": _ENVELOPE_MAXIMUM_TAIL_SECONDS,
            "slope_definition": "least-squares frame RMS dBFS slope",
            "last_rms_seconds": _ENVELOPE_LAST_RMS_SECONDS,
        },
        "aggregation_unit": "triplet_take",
    }


@dataclass(frozen=True)
class GateThresholds:
    """Hard limits for decoded 128D Z-RAVE long-rollout auditions."""

    silence_dbfs: float = -60.0
    maximum_silence_ratio: float = 0.01
    maximum_absolute_tail_rms_drift_db: float = 12.0
    maximum_static_tone_fraction: float = 0.98
    maximum_short_cycle_correlation: float = 0.995
    maximum_block_boundary_jump_ratio: float = 4.0
    minimum_cross_seed_waveform_nrmse: float = 0.01
    minimum_cross_seed_groups: int = 1
    tail_window_seconds: float = 1.0
    cycle_tail_seconds: float = 8.0
    maximum_cycle_lag_frames: int = 8

    def __post_init__(self) -> None:
        finite_values = (
            self.silence_dbfs,
            self.maximum_silence_ratio,
            self.maximum_absolute_tail_rms_drift_db,
            self.maximum_static_tone_fraction,
            self.maximum_short_cycle_correlation,
            self.maximum_block_boundary_jump_ratio,
            self.minimum_cross_seed_waveform_nrmse,
            self.tail_window_seconds,
            self.cycle_tail_seconds,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("gate thresholds must be finite")
        if not 0.0 <= self.maximum_silence_ratio <= 1.0:
            raise ValueError("maximum_silence_ratio must be in [0, 1]")
        if not 0.0 <= self.maximum_static_tone_fraction <= 1.0:
            raise ValueError("maximum_static_tone_fraction must be in [0, 1]")
        if not 0.0 <= self.maximum_short_cycle_correlation <= 1.0:
            raise ValueError("maximum_short_cycle_correlation must be in [0, 1]")
        if (
            min(
                self.maximum_absolute_tail_rms_drift_db,
                self.maximum_block_boundary_jump_ratio,
                self.minimum_cross_seed_waveform_nrmse,
                self.tail_window_seconds,
                self.cycle_tail_seconds,
            )
            < 0.0
        ):
            raise ValueError("gate magnitude thresholds must be non-negative")
        if self.minimum_cross_seed_groups < 0:
            raise ValueError("minimum_cross_seed_groups must be non-negative")
        if self.maximum_cycle_lag_frames <= 0:
            raise ValueError("maximum_cycle_lag_frames must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GateThresholds:
        unknown = sorted(set(value) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("unknown gate threshold fields: " + ", ".join(unknown))
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True)
class AuditionTriplet:
    identifier: str
    group_id: str
    generation_seed: str | int | None
    source: Path
    direct: Path
    generated: Path
    sample_rate: int | None
    seed_boundary_seconds: float
    latent_hop: int | None
    block_stride_frames: int | None
    category: str | None = None


@dataclass(frozen=True)
class _LoadedAudio:
    path: Path
    audio: np.ndarray
    sample_rate: int
    channels: int
    finite: bool
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _audio_path(value: object, role: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        for key in ("raw_wav", "wav", "file", "path", "matched_wav"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
    raise ValueError(f"{role} WAV path is missing")


def _resolve_path(root: Path, value: object, role: str) -> Path:
    candidate = Path(_audio_path(value, role)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _seed_boundary_seconds(
    manifest: Mapping[str, object],
    row: Mapping[str, object] | None = None,
) -> float:
    sources = (row or {}, manifest)
    for source in sources:
        seconds = _nonnegative_float(source.get("seed_boundary_seconds"))
        if seconds is not None:
            return seconds
    sample_rate = _positive_int(manifest.get("sample_rate"))
    latent_hop = _positive_int(manifest.get("latent_hop"))
    frames = _positive_int(
        (row or {}).get("seed_frames")
        or manifest.get("history_frames")
        or manifest.get("context_frames")
        or manifest.get("seed_frames")
    )
    if sample_rate and latent_hop and frames:
        return frames * latent_hop / sample_rate
    return 0.0


def _block_stride_frames(
    manifest: Mapping[str, object],
    row: Mapping[str, object] | None = None,
) -> int | None:
    source = row or {}
    return _positive_int(
        source.get("commit_stride_frames")
        or source.get("block_stride_frames")
        or manifest.get("commit_stride_frames")
        or manifest.get("block_stride_frames")
        or manifest.get("future_frames")
        or manifest.get("short_future_frames")
        or manifest.get("horizon_frames")
    )


def _generic_group_id(row: Mapping[str, object], identifier: str) -> str:
    explicit = row.get("group_id")
    if explicit not in (None, ""):
        return str(explicit)
    parts = [
        row.get("sample_id"),
        row.get("case_id"),
        row.get("category"),
        row.get("condition_kind"),
        row.get("condition_midi_note"),
        row.get("exploration"),
        row.get("temperature"),
        row.get("wander_delay_frames"),
    ]
    compact = [str(value) for value in parts if value not in (None, "")]
    if compact:
        return "|".join(compact)
    stripped = re.sub(
        r"(?i)(?:^|[-_:])(?:seed|s)[-_]?\d+(?=$|[-_:])",
        "",
        identifier,
    ).strip("-_:")
    return stripped or identifier


def _generic_triplets(
    manifest: Mapping[str, object],
    root: Path,
    rows: Sequence[object],
) -> list[AuditionTriplet]:
    sample_rate = _positive_int(manifest.get("sample_rate"))
    latent_hop = _positive_int(manifest.get("latent_hop"))
    triplets: list[AuditionTriplet] = []
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise TypeError(f"triplet row {index} must be a mapping")
        audio = value.get("audio")
        nested = audio if isinstance(audio, Mapping) else {}
        identifier = str(
            value.get("id") or value.get("identifier") or f"row-{index:04d}"
        )
        source = value.get("source", nested.get("source"))
        direct = value.get(
            "direct",
            nested.get("direct", nested.get("rave_direct")),
        )
        generated = value.get(
            "generated",
            nested.get("generated", nested.get("predicted")),
        )
        triplets.append(
            AuditionTriplet(
                identifier=identifier,
                group_id=_generic_group_id(value, identifier),
                generation_seed=value.get("generation_seed", value.get("seed")),
                source=_resolve_path(root, source, "source"),
                direct=_resolve_path(root, direct, "direct"),
                generated=_resolve_path(root, generated, "generated"),
                sample_rate=(_positive_int(value.get("sample_rate")) or sample_rate),
                seed_boundary_seconds=_seed_boundary_seconds(manifest, value),
                latent_hop=(_positive_int(value.get("latent_hop")) or latent_hop),
                block_stride_frames=_block_stride_frames(manifest, value),
                category=(
                    str(value["category"])
                    if value.get("category") is not None
                    else None
                ),
            )
        )
    return triplets


def _flow_audition_triplets(
    manifest: Mapping[str, object],
    root: Path,
) -> list[AuditionTriplet]:
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        raise TypeError("flow audition examples must be a list")
    sample_rate = _positive_int(manifest.get("sample_rate"))
    latent_hop = _positive_int(manifest.get("latent_hop"))
    triplets: list[AuditionTriplet] = []
    for example_index, example_value in enumerate(examples):
        if not isinstance(example_value, Mapping):
            raise TypeError("flow audition example must be a mapping")
        rollouts = example_value.get("rollouts")
        if not isinstance(rollouts, list) or not rollouts:
            raise ValueError("flow audition example has no rollouts")
        sample_id = str(
            example_value.get("sample_id") or f"example-{example_index:04d}"
        )
        for rollout_index, rollout_value in enumerate(rollouts):
            if not isinstance(rollout_value, Mapping):
                raise TypeError("flow rollout must be a mapping")
            seed = rollout_value.get("generation_seed")
            controls = (
                rollout_value.get("exploration"),
                rollout_value.get("temperature"),
                rollout_value.get("wander_delay_frames"),
                rollout_value.get("condition_kind"),
                rollout_value.get("condition_midi_note"),
            )
            control_key = "|".join(
                "" if value is None else str(value) for value in controls
            )
            triplets.append(
                AuditionTriplet(
                    identifier=(f"{sample_id}:rollout-{rollout_index}:seed-{seed}"),
                    group_id=f"{sample_id}|{control_key}",
                    generation_seed=seed,
                    source=_resolve_path(root, example_value.get("source"), "source"),
                    direct=_resolve_path(root, example_value.get("direct"), "direct"),
                    generated=_resolve_path(root, rollout_value, "generated"),
                    sample_rate=sample_rate,
                    seed_boundary_seconds=_seed_boundary_seconds(manifest),
                    latent_hop=latent_hop,
                    block_stride_frames=_block_stride_frames(manifest, rollout_value),
                    category=(
                        str(example_value["category"])
                        if example_value.get("category") is not None
                        else None
                    ),
                )
            )
    return triplets


def _pure_audition_triplets(
    manifest: Mapping[str, object],
    root: Path,
) -> list[AuditionTriplet]:
    cases = manifest.get("cases")
    files = manifest.get("files")
    if not isinstance(cases, list) or not isinstance(files, list):
        raise TypeError("pure audition manifest needs cases and files")
    case_map = {
        int(case["case_index"]): case
        for case in cases
        if isinstance(case, Mapping) and "case_index" in case
    }
    role_map: dict[int, dict[str, Mapping[str, object]]] = {}
    for value in files:
        if not isinstance(value, Mapping):
            raise TypeError("pure audition file row must be a mapping")
        case_index = int(value.get("case_index", -1))
        role_map.setdefault(case_index, {})[str(value.get("role"))] = value
    sample_rate = _positive_int(manifest.get("sample_rate"))
    latent_hop = _positive_int(manifest.get("latent_hop"))
    triplets: list[AuditionTriplet] = []
    for case_index, case in sorted(case_map.items()):
        roles = role_map.get(case_index, {})
        if "source" not in roles or "rave_direct" not in roles:
            raise ValueError(f"pure audition case {case_index} lacks triplet base")
        for role, generated in sorted(roles.items()):
            match = re.fullmatch(r"seed(\d+)[_-]long", role)
            if match is None:
                continue
            seed = int(match.group(1))
            sample_id = str(case.get("sample_id") or f"case-{case_index}")
            triplets.append(
                AuditionTriplet(
                    identifier=f"{sample_id}:{role}",
                    group_id=sample_id,
                    generation_seed=seed,
                    source=_resolve_path(root, roles["source"], "source"),
                    direct=_resolve_path(root, roles["rave_direct"], "direct"),
                    generated=_resolve_path(root, generated, "generated"),
                    sample_rate=sample_rate,
                    seed_boundary_seconds=_seed_boundary_seconds(manifest),
                    latent_hop=latent_hop,
                    block_stride_frames=_block_stride_frames(manifest),
                    category=(
                        str(case["category"])
                        if case.get("category") is not None
                        else None
                    ),
                )
            )
    if not triplets:
        raise ValueError("pure audition manifest has no long seed renders")
    return triplets


def load_audition_triplets(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], list[AuditionTriplet]]:
    """Load generic or repository-native audition manifests."""

    source = Path(manifest_path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("audition manifest must be a mapping")
    if isinstance(payload.get("triplets"), list):
        triplets = _generic_triplets(payload, source.parent, payload["triplets"])
    elif isinstance(payload.get("examples"), list):
        triplets = _flow_audition_triplets(payload, source.parent)
    elif isinstance(payload.get("cases"), list) and isinstance(
        payload.get("files"), list
    ):
        triplets = _pure_audition_triplets(payload, source.parent)
    elif isinstance(payload.get("rows"), list):
        triplets = _generic_triplets(payload, source.parent, payload["rows"])
    else:
        raise TypeError(
            "manifest must contain triplets, examples, cases/files, or rows"
        )
    if not triplets:
        raise ValueError("audition manifest contains no WAV triplets")
    identifiers = [triplet.identifier for triplet in triplets]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("audition triplet identifiers must be unique")
    return payload, triplets


def _load_audio(path: Path) -> _LoadedAudio:
    if not path.is_file():
        raise ValueError(f"audio file does not exist: {path}")
    value, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"audio file is empty: {path}")
    finite = bool(np.isfinite(value).all())
    mono = np.mean(value, axis=1, dtype=np.float64).astype(np.float32)
    return _LoadedAudio(
        path=path,
        audio=np.ascontiguousarray(mono),
        sample_rate=int(sample_rate),
        channels=int(value.shape[1]),
        finite=finite,
        sha256=_sha256_file(path),
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    values = [
        _rms(audio[start : start + frame_samples])
        for start in range(0, audio.size, frame_samples)
        if audio[start : start + frame_samples].size
    ]
    return np.asarray(values, dtype=np.float64)


def _rms_dbfs(value: float) -> float:
    floor = 10.0 ** (_ENVELOPE_DBFS_FLOOR / 20.0)
    return max(
        _ENVELOPE_DBFS_FLOOR,
        20.0 * math.log10(max(float(value), floor)),
    )


def _sliding_envelope_rms(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("envelope proxy audio must be a non-empty mono array")
    if not np.isfinite(samples).all():
        raise ValueError("envelope proxy audio must be finite")
    if sample_rate <= 0:
        raise ValueError("envelope proxy sample_rate must be positive")
    requested_frame = max(1, round(_ENVELOPE_FRAME_SECONDS * sample_rate))
    frame_samples = min(samples.size, requested_frame)
    hop_samples = max(1, round(_ENVELOPE_HOP_SECONDS * sample_rate))
    last_start = samples.size - frame_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    rms = np.asarray(
        [_rms(samples[start : start + frame_samples]) for start in starts],
        dtype=np.float64,
    )
    centers = (np.asarray(starts, dtype=np.float64) + frame_samples / 2.0) / sample_rate
    return rms, centers, frame_samples, hop_samples


def envelope_proxy_metrics(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float | int | bool | None]:
    """Compute report-only amplitude-envelope proxies for one mono WAV take."""

    samples = np.asarray(audio, dtype=np.float64)
    rms, centers, frame_samples, hop_samples = _sliding_envelope_rms(
        samples,
        sample_rate,
    )
    duration = samples.size / sample_rate
    rms_db = np.asarray([_rms_dbfs(value) for value in rms], dtype=np.float64)
    peak_index = int(np.argmax(rms))
    peak_rms = float(rms[peak_index])
    peak_dbfs = _rms_dbfs(peak_rms)
    silence_rms = 10.0 ** (_ENVELOPE_SILENCE_FLOOR_DBFS / 20.0)
    silent = peak_rms < silence_rms

    sustain_start = duration * _ENVELOPE_SUSTAIN_START_FRACTION
    sustain_end = duration * _ENVELOPE_SUSTAIN_END_FRACTION
    sustain_indices = np.flatnonzero(
        (centers >= sustain_start) & (centers <= sustain_end)
    )
    if not sustain_indices.size:
        sustain_indices = np.asarray(
            [int(np.argmin(np.abs(centers - duration * 0.5)))],
            dtype=np.int64,
        )
    sustain_values = rms[sustain_indices]
    sustain_db_values = rms_db[sustain_indices]
    sustain_rms = float(np.sqrt(np.mean(np.square(sustain_values))))
    sustain_dbfs = _rms_dbfs(sustain_rms)
    sustain_variation = float(
        np.percentile(sustain_db_values, 90) - np.percentile(sustain_db_values, 10)
    )
    sustain_center = float(np.mean(centers[sustain_indices]))

    attack_ms: float | None = None
    attack_crossing_found = False
    attack_hold_frames = min(
        max(
            1,
            math.ceil(
                _ENVELOPE_ATTACK_HOLD_SECONDS
                / max(hop_samples / sample_rate, 1.0 / sample_rate)
            ),
        ),
        rms.size,
    )
    if not silent:
        onset_threshold = max(
            silence_rms,
            peak_rms * _ENVELOPE_ONSET_PEAK_RATIO,
        )
        onset_candidates = np.flatnonzero(rms >= onset_threshold)
        onset_index = int(onset_candidates[0])
        attack_threshold = peak_rms * _ENVELOPE_ATTACK_PEAK_RATIO
        final_crossing_start = rms.size - attack_hold_frames
        for index in range(onset_index, final_crossing_start + 1):
            if bool(
                np.all(rms[index : index + attack_hold_frames] >= attack_threshold)
            ):
                attack_ms = max(
                    0.0,
                    float(centers[index] - centers[onset_index]) * 1000.0,
                )
                attack_crossing_found = True
                break

    peak_to_sustain_drop = max(0.0, peak_dbfs - sustain_dbfs)
    peak_time = float(centers[peak_index])
    decay_seconds = sustain_center - peak_time
    decay_slope = (
        (sustain_dbfs - peak_dbfs) / decay_seconds
        if not silent
        and decay_seconds > max(1.0 / sample_rate, 0.5 * hop_samples / sample_rate)
        else None
    )

    tail_seconds = min(
        duration,
        _ENVELOPE_MAXIMUM_TAIL_SECONDS,
        max(duration * _ENVELOPE_TAIL_FRACTION, frame_samples / sample_rate),
    )
    tail_start = max(0.0, duration - tail_seconds)
    tail_indices = np.flatnonzero(centers >= tail_start)
    if not tail_indices.size:
        tail_indices = np.asarray([rms.size - 1], dtype=np.int64)
    tail_times = centers[tail_indices]
    tail_db = rms_db[tail_indices]
    if silent or tail_indices.size < 2 or float(np.ptp(tail_times)) <= 0.0:
        tail_slope = 0.0
    else:
        centered_time = tail_times - float(tail_times[0])
        tail_slope = float(np.polyfit(centered_time, tail_db, 1)[0])
    last_window_samples = min(
        samples.size,
        max(1, round(_ENVELOPE_LAST_RMS_SECONDS * sample_rate)),
    )
    tail_rms_dbfs = _rms_dbfs(_rms(samples[-last_window_samples:]))

    payload: dict[str, float | int | bool | None] = {
        "envelope_duration_seconds_proxy": duration,
        "envelope_analysis_samples_proxy": int(samples.size),
        "envelope_frame_count_proxy": int(rms.size),
        "envelope_frame_ms_proxy": frame_samples * 1000.0 / sample_rate,
        "envelope_hop_ms_proxy": hop_samples * 1000.0 / sample_rate,
        "envelope_silent_proxy": silent,
        "envelope_peak_rms_dbfs_proxy": peak_dbfs,
        "envelope_attack_crossing_found_proxy": attack_crossing_found,
        "envelope_attack_ms_proxy": attack_ms,
        "envelope_attack_hold_ms_proxy": (
            attack_hold_frames * hop_samples * 1000.0 / sample_rate
        ),
        "envelope_decay_slope_db_per_second_proxy": decay_slope,
        "envelope_peak_to_sustain_drop_db_proxy": peak_to_sustain_drop,
        "envelope_sustain_rms_dbfs_proxy": sustain_dbfs,
        "envelope_sustain_variation_db_proxy": sustain_variation,
        "envelope_sustain_window_start_seconds_proxy": sustain_start,
        "envelope_sustain_window_end_seconds_proxy": sustain_end,
        "envelope_tail_release_slope_db_per_second_proxy": tail_slope,
        "envelope_tail_rms_dbfs_proxy": tail_rms_dbfs,
        "envelope_tail_slope_window_start_seconds_proxy": tail_start,
        "envelope_tail_slope_window_end_seconds_proxy": duration,
        "envelope_tail_rms_window_seconds_proxy": (last_window_samples / sample_rate),
    }
    for name, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite envelope proxy metric: {name}")
    return payload


def _silence_ratio(
    audio: np.ndarray,
    sample_rate: int,
    silence_dbfs: float,
) -> float:
    frame_samples = max(1, round(0.050 * sample_rate))
    rms = _frame_rms(audio, frame_samples)
    threshold = 10.0 ** (silence_dbfs / 20.0)
    return float(np.mean(rms < threshold))


def _tail_rms_metrics(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> dict[str, float]:
    requested = max(1, round(window_seconds * sample_rate))
    window = min(requested, max(1, audio.size // 2))
    head_rms = _rms(audio[:window])
    tail_rms = _rms(audio[-window:])
    floor = 1.0e-12
    drift = 20.0 * math.log10(max(tail_rms, floor) / max(head_rms, floor))
    return {
        "head_rms": head_rms,
        "tail_rms": tail_rms,
        "tail_rms_drift_db": drift,
        "absolute_tail_rms_drift_db": abs(drift),
    }


def _temporal_features(
    audio: np.ndarray,
    frame_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_samples = max(16, frame_samples)
    frames = audio.size // frame_samples
    if frames < 2:
        return (
            np.empty((0, 25), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 24), dtype=np.float64),
        )
    matrix = audio[: frames * frame_samples].reshape(frames, frame_samples)
    window = np.hanning(frame_samples).astype(np.float64)
    windowed = matrix.astype(np.float64) * window[None, :]
    spectrum = np.square(np.abs(np.fft.rfft(windowed, axis=1)))
    edges = np.linspace(0, spectrum.shape[1], 25, dtype=np.int64)
    bands = np.stack(
        [
            spectrum[:, left : max(left + 1, right)].sum(axis=1)
            for left, right in itertools.pairwise(edges)
        ],
        axis=1,
    )
    total = np.maximum(bands.sum(axis=1, keepdims=True), 1.0e-24)
    normalized = bands / total
    rms_db = 20.0 * np.log10(
        np.maximum(
            np.sqrt(np.mean(np.square(matrix, dtype=np.float64), axis=1)),
            1.0e-12,
        )
    )
    features = np.concatenate(
        (rms_db[:, None] / 20.0, np.sqrt(normalized)),
        axis=1,
    )
    return features, rms_db, normalized


def _static_and_cycle_metrics(
    audio: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    thresholds: GateThresholds,
) -> dict[str, float | int | None]:
    maximum_samples = max(1, round(thresholds.cycle_tail_seconds * sample_rate))
    tail = audio[-maximum_samples:]
    features, rms_db, spectrum = _temporal_features(tail, frame_samples)
    if features.shape[0] < 2:
        return {
            "analysis_frames": int(features.shape[0]),
            "static_tone_fraction": None,
            "maximum_short_cycle_correlation": None,
            "maximum_short_cycle_lag_frames": None,
        }
    roots = np.sqrt(np.maximum(spectrum, 0.0))
    spectral_similarity = np.sum(roots[:-1] * roots[1:], axis=1)
    rms_change = np.abs(np.diff(rms_db))
    static = (spectral_similarity >= 0.9995) & (rms_change <= 0.10)

    # Correlate temporal innovations, not the raw feature trajectory. Raw
    # autocorrelation labels every smoothly evolving sustained note as a
    # short cycle, while differencing preserves exact 1--8 frame repeats.
    innovations = np.diff(features, axis=0)
    centered = innovations - np.mean(
        innovations,
        axis=0,
        keepdims=True,
    )
    correlations: list[tuple[float, int]] = []
    maximum_lag = min(
        thresholds.maximum_cycle_lag_frames,
        innovations.shape[0] - 1,
    )
    if maximum_lag <= 0:
        return {
            "analysis_frames": int(features.shape[0]),
            "static_tone_fraction": float(np.mean(static)),
            "maximum_short_cycle_correlation": None,
            "maximum_short_cycle_lag_frames": None,
        }
    for lag in range(1, maximum_lag + 1):
        left = centered[:-lag].reshape(-1)
        right = centered[lag:].reshape(-1)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = (
            abs(float(np.dot(left, right))) / denominator
            if denominator > 1.0e-12
            else 1.0
        )
        correlations.append((min(1.0, correlation), lag))
    correlation, lag = max(correlations)
    return {
        "analysis_frames": int(features.shape[0]),
        "static_tone_fraction": float(np.mean(static)),
        "maximum_short_cycle_correlation": correlation,
        "maximum_short_cycle_lag_frames": lag,
    }


def _boundary_metrics(
    audio: np.ndarray,
    sample_rate: int,
    generated_start: int,
    block_samples: int | None,
) -> dict[str, float | int | None]:
    boundaries: list[int] = []
    if generated_start > 0 and generated_start < audio.size:
        boundaries.append(generated_start)
    if block_samples is not None and block_samples > 0:
        first = (
            generated_start + block_samples if generated_start > 0 else block_samples
        )
        boundaries.extend(range(first, audio.size, block_samples))
    boundaries = sorted(set(boundaries))
    if not boundaries:
        return {
            "boundary_count": 0,
            "block_boundary_jump_ratio": None,
            "block_boundary_rms_step_db_p95": None,
        }
    generated = audio[generated_start:]
    adjacent = np.abs(np.diff(generated.astype(np.float64)))
    reference = float(np.percentile(adjacent, 95)) if adjacent.size else 0.0
    jumps = np.asarray(
        [abs(float(audio[index]) - float(audio[index - 1])) for index in boundaries],
        dtype=np.float64,
    )
    if reference <= 1.0e-12:
        ratio = 0.0 if float(jumps.max(initial=0.0)) <= 1.0e-12 else 1.0e12
    else:
        ratio = float(np.percentile(jumps, 95) / reference)
    local = max(1, round(0.010 * sample_rate))
    steps: list[float] = []
    for index in boundaries:
        before = audio[max(0, index - local) : index]
        after = audio[index : min(audio.size, index + local)]
        if before.size and after.size:
            steps.append(
                abs(
                    20.0
                    * math.log10(max(_rms(after), 1.0e-12) / max(_rms(before), 1.0e-12))
                )
            )
    return {
        "boundary_count": len(boundaries),
        "block_boundary_jump_ratio": ratio,
        "block_boundary_rms_step_db_p95": (
            float(np.percentile(np.asarray(steps), 95)) if steps else None
        ),
    }


def _normalized_rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    length = min(left.size, right.size)
    if length <= 0:
        raise ValueError("audio distance needs overlapping samples")
    left64 = left[:length].astype(np.float64)
    right64 = right[:length].astype(np.float64)
    scale = math.sqrt((_rms(left64) ** 2 + _rms(right64) ** 2) / 2.0)
    if scale <= 1.0e-12:
        return 0.0
    return _rms(left64 - right64) / scale


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def _metric_values(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _envelope_role_metric_values(
    rows: Sequence[Mapping[str, object]],
    role: str,
    name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        takes = row.get("envelope_takes_proxy")
        if not isinstance(takes, Mapping):
            continue
        metrics = takes.get(role)
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
            if math.isfinite(parsed):
                values.append(parsed)
    return values


def _envelope_absolute_difference_values(
    rows: Sequence[Mapping[str, object]],
    baseline_role: str,
    name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        takes = row.get("envelope_takes_proxy")
        if not isinstance(takes, Mapping):
            continue
        generated = takes.get("generated")
        baseline = takes.get(baseline_role)
        if not isinstance(generated, Mapping) or not isinstance(
            baseline,
            Mapping,
        ):
            continue
        generated_value = generated.get(name)
        baseline_value = baseline.get(name)
        if (
            isinstance(generated_value, (int, float))
            and not isinstance(generated_value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
        ):
            difference = abs(float(generated_value) - float(baseline_value))
            if math.isfinite(difference):
                values.append(difference)
    return values


def _envelope_summary_proxy(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "report_only": True,
        "distribution_unit": "triplet_take",
        "take_distributions": {
            role: {
                name: _summary(_envelope_role_metric_values(rows, role, name))
                for name in _ENVELOPE_SUMMARY_METRICS
            }
            for role in ("source", "direct", "generated")
        },
        "generated_absolute_difference_distributions": {
            f"generated_vs_{role}": {
                name: _summary(_envelope_absolute_difference_values(rows, role, name))
                for name in _ENVELOPE_SUMMARY_METRICS
            }
            for role in ("source", "direct")
        },
    }


def _check(
    value: float | None,
    threshold: float,
    operator: str,
    predicate: bool,
) -> dict[str, object]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(predicate),
    }


def acceptance_gate(
    report: Mapping[str, object],
    thresholds: GateThresholds | None = None,
) -> dict[str, object]:
    """Apply independent hard checks; no scalar can hide a failed mode."""

    limits = thresholds or GateThresholds()
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("gate report has no summary")

    def maximum(name: str) -> float | None:
        value = summary.get(name)
        if not isinstance(value, Mapping):
            return None
        result = value.get("maximum")
        parsed = float(result) if isinstance(result, (int, float)) else None
        return parsed if parsed is not None and math.isfinite(parsed) else None

    load_errors = int(summary.get("load_errors") or 0)
    nonfinite = int(summary.get("nonfinite_files") or 0)
    analyzed = int(summary.get("analyzed_rollouts") or 0)
    triplets = int(summary.get("triplets") or 0)
    silence = maximum("silence_ratio")
    drift = maximum("absolute_tail_rms_drift_db")
    static = maximum("static_tone_fraction")
    cycle = maximum("maximum_short_cycle_correlation")
    boundary = maximum("block_boundary_jump_ratio")
    diversity = summary.get("cross_seed_diversity")
    if isinstance(diversity, Mapping):
        diversity_groups = int(diversity.get("groups") or 0)
        minimum_group = diversity.get("minimum_group_median_waveform_nrmse")
        minimum_group_value = (
            float(minimum_group) if isinstance(minimum_group, (int, float)) else None
        )
        if minimum_group_value is not None and not math.isfinite(minimum_group_value):
            minimum_group_value = None
    else:
        diversity_groups = 0
        minimum_group_value = None
    complete_metrics = analyzed == triplets and triplets > 0
    checks = {
        "wav_load": _check(load_errors, 0, "==", load_errors == 0),
        "finite": _check(nonfinite, 0, "==", nonfinite == 0),
        "complete_metrics": _check(analyzed, triplets, "==", complete_metrics),
        "silence_ratio": _check(
            silence,
            limits.maximum_silence_ratio,
            "<=",
            silence is not None and silence <= limits.maximum_silence_ratio,
        ),
        "absolute_tail_rms_drift_db": _check(
            drift,
            limits.maximum_absolute_tail_rms_drift_db,
            "<=",
            drift is not None and drift <= limits.maximum_absolute_tail_rms_drift_db,
        ),
        "static_tone_fraction": _check(
            static,
            limits.maximum_static_tone_fraction,
            "<=",
            static is not None and static <= limits.maximum_static_tone_fraction,
        ),
        "short_cycle_correlation": _check(
            cycle,
            limits.maximum_short_cycle_correlation,
            "<=",
            cycle is not None and cycle <= limits.maximum_short_cycle_correlation,
        ),
        "block_boundary_jump_ratio": _check(
            boundary,
            limits.maximum_block_boundary_jump_ratio,
            "<=",
            boundary is not None
            and boundary <= limits.maximum_block_boundary_jump_ratio,
        ),
        "cross_seed_groups": _check(
            diversity_groups,
            limits.minimum_cross_seed_groups,
            ">=",
            diversity_groups >= limits.minimum_cross_seed_groups,
        ),
        "cross_seed_waveform_nrmse": _check(
            minimum_group_value,
            limits.minimum_cross_seed_waveform_nrmse,
            ">=",
            minimum_group_value is not None
            and minimum_group_value >= limits.minimum_cross_seed_waveform_nrmse,
        ),
    }
    if limits.minimum_cross_seed_groups == 0 and diversity_groups == 0:
        checks["cross_seed_waveform_nrmse"] = _check(
            None,
            limits.minimum_cross_seed_waveform_nrmse,
            "not-required",
            True,
        )
    failed = [name for name, value in checks.items() if not value["passed"]]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def evaluate_wav_triplets(
    triplets: Sequence[AuditionTriplet],
    *,
    thresholds: GateThresholds | None = None,
    manifest: str | Path | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Evaluate already-normalized source/direct/generated WAV triplets."""

    limits = thresholds or GateThresholds()
    if not triplets:
        raise ValueError("at least one audition triplet is required")
    cache: dict[Path, _LoadedAudio | Exception] = {}

    def cached(path: Path) -> _LoadedAudio:
        if path not in cache:
            try:
                cache[path] = _load_audio(path)
            except (OSError, RuntimeError, ValueError) as error:
                cache[path] = error
        result = cache[path]
        if isinstance(result, Exception):
            raise result
        return result

    rows: list[dict[str, object]] = []
    generated_audio: dict[str, np.ndarray] = {}
    for triplet in triplets:
        row: dict[str, object] = {
            "id": triplet.identifier,
            "group_id": triplet.group_id,
            "generation_seed": triplet.generation_seed,
            "category": triplet.category,
            "source": str(triplet.source),
            "direct": str(triplet.direct),
            "generated": str(triplet.generated),
            "metrics": None,
            "envelope_takes_proxy": {
                "source": None,
                "direct": None,
                "generated": None,
            },
            "errors": [],
        }
        loaded: dict[str, _LoadedAudio] = {}
        for role, path in (
            ("source", triplet.source),
            ("direct", triplet.direct),
            ("generated", triplet.generated),
        ):
            try:
                audio = cached(path)
                loaded[role] = audio
                if triplet.sample_rate and audio.sample_rate != triplet.sample_rate:
                    row["errors"].append(
                        f"{role} sample rate {audio.sample_rate} != "
                        f"{triplet.sample_rate}"
                    )
            except (OSError, RuntimeError, ValueError) as error:
                row["errors"].append(f"{role}: {error}")
        if len(loaded) == 3:
            rates = {audio.sample_rate for audio in loaded.values()}
            if len(rates) != 1:
                row["errors"].append("triplet sample rates do not match")
            row["finite"] = {role: audio.finite for role, audio in loaded.items()}
            row["all_finite"] = all(audio.finite for audio in loaded.values())
        else:
            row["finite"] = None
            row["all_finite"] = False

        envelope_takes = row["envelope_takes_proxy"]
        assert isinstance(envelope_takes, dict)
        for role in ("source", "direct", "generated"):
            audio = loaded.get(role)
            if audio is not None and audio.finite:
                envelope_takes[role] = envelope_proxy_metrics(
                    audio.audio,
                    audio.sample_rate,
                )

        generated = loaded.get("generated")
        errors = row["errors"]
        assert isinstance(errors, list)
        if generated is not None and bool(row["all_finite"]) and not errors:
            generated_start = round(
                triplet.seed_boundary_seconds * generated.sample_rate
            )
            if not 0 <= generated_start < generated.audio.size:
                errors.append("seed boundary is outside generated WAV")
            else:
                predicted = generated.audio[generated_start:]
                frame_samples = (
                    triplet.latent_hop
                    if triplet.latent_hop is not None
                    else max(16, round(0.050 * generated.sample_rate))
                )
                block_samples = (
                    triplet.latent_hop * triplet.block_stride_frames
                    if triplet.latent_hop is not None
                    and triplet.block_stride_frames is not None
                    else None
                )
                metrics: dict[str, object] = {
                    "sample_rate": generated.sample_rate,
                    "samples": int(generated.audio.size),
                    "generated_samples": int(predicted.size),
                    "generated_seconds": predicted.size / generated.sample_rate,
                    "peak": float(np.max(np.abs(predicted))),
                    "rms": _rms(predicted),
                    "silence_ratio": _silence_ratio(
                        predicted,
                        generated.sample_rate,
                        limits.silence_dbfs,
                    ),
                    **_tail_rms_metrics(
                        predicted,
                        generated.sample_rate,
                        limits.tail_window_seconds,
                    ),
                    **_static_and_cycle_metrics(
                        predicted,
                        generated.sample_rate,
                        frame_samples,
                        limits,
                    ),
                    **_boundary_metrics(
                        generated.audio,
                        generated.sample_rate,
                        generated_start,
                        block_samples,
                    ),
                }
                direct = loaded["direct"].audio
                source = loaded["source"].audio
                metrics["source_direct_nrmse"] = _normalized_rms_distance(
                    source, direct
                )
                if generated_start > 0:
                    seed_samples = min(
                        generated_start,
                        direct.size,
                        generated.audio.size,
                    )
                    metrics["direct_generated_seed_nrmse"] = (
                        _normalized_rms_distance(
                            direct[:seed_samples],
                            generated.audio[:seed_samples],
                        )
                        if seed_samples
                        else None
                    )
                else:
                    metrics["direct_generated_seed_nrmse"] = None
                row["metrics"] = metrics
                generated_audio[triplet.identifier] = predicted
        rows.append(row)

    groups: dict[str, list[tuple[AuditionTriplet, np.ndarray]]] = {}
    for triplet in triplets:
        audio = generated_audio.get(triplet.identifier)
        if audio is not None and triplet.generation_seed is not None:
            groups.setdefault(triplet.group_id, []).append((triplet, audio))
    group_rows: list[dict[str, object]] = []
    all_distances: list[float] = []
    for group_id, values in sorted(groups.items()):
        pair_rows: list[dict[str, object]] = []
        distances: list[float] = []
        for (left_row, left), (right_row, right) in itertools.combinations(values, 2):
            if str(left_row.generation_seed) == str(right_row.generation_seed):
                continue
            distance = _normalized_rms_distance(left, right)
            distances.append(distance)
            all_distances.append(distance)
            pair_rows.append(
                {
                    "left_seed": left_row.generation_seed,
                    "right_seed": right_row.generation_seed,
                    "waveform_nrmse": distance,
                }
            )
        if distances:
            group_rows.append(
                {
                    "group_id": group_id,
                    "seeds": sorted(
                        {str(value[0].generation_seed) for value in values}
                    ),
                    "pairs": pair_rows,
                    "waveform_nrmse": _summary(distances),
                }
            )

    file_rows: list[dict[str, object]] = []
    load_errors = 0
    nonfinite_files = 0
    for path, result in sorted(cache.items(), key=lambda item: str(item[0])):
        if isinstance(result, Exception):
            load_errors += 1
            file_rows.append({"path": str(path), "error": str(result)})
        else:
            nonfinite_files += int(not result.finite)
            file_rows.append(
                {
                    "path": str(path),
                    "sample_rate": result.sample_rate,
                    "samples": int(result.audio.size),
                    "channels": result.channels,
                    "finite": result.finite,
                    "sha256": result.sha256,
                }
            )
    analyzed = sum(isinstance(row.get("metrics"), Mapping) for row in rows)
    group_medians = [
        float(group["waveform_nrmse"]["median"])
        for group in group_rows
        if isinstance(group.get("waveform_nrmse"), Mapping)
        and group["waveform_nrmse"].get("median") is not None
    ]
    summary: dict[str, object] = {
        "triplets": len(triplets),
        "analyzed_rollouts": analyzed,
        "unique_wav_files": len(cache),
        "load_errors": load_errors,
        "nonfinite_files": nonfinite_files,
        "silence_ratio": _summary(_metric_values(rows, "silence_ratio")),
        "absolute_tail_rms_drift_db": _summary(
            _metric_values(rows, "absolute_tail_rms_drift_db")
        ),
        "static_tone_fraction": _summary(_metric_values(rows, "static_tone_fraction")),
        "maximum_short_cycle_correlation": _summary(
            _metric_values(rows, "maximum_short_cycle_correlation")
        ),
        "block_boundary_jump_ratio": _summary(
            _metric_values(rows, "block_boundary_jump_ratio")
        ),
        "envelope_summary_proxy": _envelope_summary_proxy(rows),
        "cross_seed_diversity": {
            "groups": len(group_rows),
            "pairs": len(all_distances),
            "waveform_nrmse": _summary(all_distances),
            "minimum_group_median_waveform_nrmse": (
                min(group_medians) if group_medians else None
            ),
            "group_summary": group_rows,
        },
    }
    report: dict[str, object] = {
        "schema": 1,
        "kind": "zrave-128d-long-rollout-acceptance-gate",
        "manifest": str(Path(manifest).resolve()) if manifest else None,
        "manifest_sha256": manifest_sha256,
        "thresholds": asdict(limits),
        "envelope_algorithm_proxy": _envelope_algorithm_proxy(),
        "files": file_rows,
        "rows": rows,
        "summary": summary,
    }
    gate = acceptance_gate(report, limits)
    report["acceptance"] = gate
    report["passed"] = gate["passed"]
    return report


def evaluate_manifest(
    manifest_path: str | Path,
    *,
    thresholds: GateThresholds | None = None,
) -> dict[str, object]:
    source = Path(manifest_path).expanduser().resolve()
    manifest, triplets = load_audition_triplets(source)
    latent_dim = _positive_int(manifest.get("latent_dim"))
    codec = manifest.get("codec")
    if latent_dim is None and isinstance(codec, Mapping):
        latent_dim = _positive_int(codec.get("latent_dim"))
    if latent_dim is not None and latent_dim != 128:
        raise ValueError(f"flow gate requires a 128D audition, got {latent_dim}D")
    report = evaluate_wav_triplets(
        triplets,
        thresholds=thresholds,
        manifest=source,
        manifest_sha256=_sha256_file(source),
    )
    report["manifest_format"] = (
        "flow-examples"
        if "examples" in manifest
        else "pure-cases-files"
        if "cases" in manifest and "files" in manifest
        else "generic-triplets"
    )
    report["latent_dim"] = latent_dim
    return report


evaluate_flow_gate = evaluate_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decoded 128D Z-RAVE long rollouts and apply hard "
            "acceptance gates."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--thresholds-json")
    parser.add_argument("--maximum-silence-ratio", type=float)
    parser.add_argument("--maximum-absolute-tail-rms-drift-db", type=float)
    parser.add_argument("--maximum-static-tone-fraction", type=float)
    parser.add_argument("--maximum-short-cycle-correlation", type=float)
    parser.add_argument("--maximum-block-boundary-jump-ratio", type=float)
    parser.add_argument("--minimum-cross-seed-waveform-nrmse", type=float)
    parser.add_argument("--minimum-cross-seed-groups", type=int)
    parser.add_argument(
        "--fail-on-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _thresholds_from_args(args: argparse.Namespace) -> GateThresholds:
    thresholds = GateThresholds()
    if args.thresholds_json:
        payload = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("thresholds JSON must be a mapping")
        thresholds = GateThresholds.from_mapping(payload)
    overrides = {
        name: getattr(args, name)
        for name in (
            "maximum_silence_ratio",
            "maximum_absolute_tail_rms_drift_db",
            "maximum_static_tone_fraction",
            "maximum_short_cycle_correlation",
            "maximum_block_boundary_jump_ratio",
            "minimum_cross_seed_waveform_nrmse",
            "minimum_cross_seed_groups",
        )
        if getattr(args, name) is not None
    }
    return replace(thresholds, **overrides)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    thresholds = _thresholds_from_args(args)
    report = evaluate_manifest(args.manifest, thresholds=thresholds)
    if args.output:
        _atomic_json(Path(args.output), report)
    print(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if args.fail_on_reject and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
