from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_TORCHCREPE_VERSION = "0.0.24"
_OFFICIAL_MODEL_SHA256 = {
    "tiny": "d4993eea36ed1a0ad9ac549c740dae5265b049ce72004f00c2f59e01c0be8432",
    "full": "133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986",
}
_HARD_CONDITION_KINDS = ("matched", "note_swap")
_PITCH_PANEL_CONDITION_KINDS = (*_HARD_CONDITION_KINDS, "note_step")
_VELOCITY_PANEL_CONDITION_KIND = "velocity_step"


@dataclass(frozen=True)
class DecodedAudioCrepeConfig:
    crepe_hop_length: int = 512
    fmin_hz: float = 50.0
    fmax_hz: float = 2000.0
    model_capacity: str = "tiny"
    periodicity_threshold: float = 0.5
    settling_consecutive_frames: int = 3
    maximum_p90_absolute_cents: float = 100.0
    minimum_within_100_cents: float = 0.90

    def __post_init__(self) -> None:
        if self.crepe_hop_length <= 0:
            raise ValueError("CREPE hop length must be positive")
        if (
            not math.isfinite(self.fmin_hz)
            or not math.isfinite(self.fmax_hz)
            or self.fmin_hz <= 0.0
            or self.fmax_hz <= self.fmin_hz
            or self.fmax_hz > 2006.0
        ):
            raise ValueError("CREPE frequency range must be within (0, 2006] Hz")
        if self.model_capacity not in _OFFICIAL_MODEL_SHA256:
            raise ValueError("CREPE model capacity must be tiny or full")
        if (
            not math.isfinite(self.periodicity_threshold)
            or not 0.0 <= self.periodicity_threshold <= 1.0
        ):
            raise ValueError("periodicity threshold must be in [0, 1]")
        if self.settling_consecutive_frames <= 0:
            raise ValueError("settling consecutive frames must be positive")
        if (
            not math.isfinite(self.maximum_p90_absolute_cents)
            or self.maximum_p90_absolute_cents < 0.0
        ):
            raise ValueError("maximum p90 cents must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_within_100_cents)
            or not 0.0 <= self.minimum_within_100_cents <= 1.0
        ):
            raise ValueError("minimum within-100 ratio must be in [0, 1]")


@dataclass(frozen=True)
class PitchTrack:
    frequency_hz: np.ndarray
    periodicity: np.ndarray
    frame_offsets_samples: np.ndarray | None = None


@dataclass(frozen=True)
class PitchTrackerContract:
    implementation: str
    package_version: str
    package_code_sha256: str
    model_capacity: str
    model_path: str
    model_sha256: str
    decoder: str = "torchcrepe.decode.viterbi"
    dither_seed_policy: str = "sha256(float32_audio+canonical_tracker_inference_config)"
    torch_version: str = "injected"
    torchaudio_version: str = "injected"
    numpy_version: str = np.__version__
    device: str = "injected"


PitchTracker = Callable[[np.ndarray, int, DecodedAudioCrepeConfig], PitchTrack]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_code_sha256(package_root: Path) -> str:
    paths = sorted(package_root.glob("*.py"), key=lambda item: item.name)
    if not paths:
        raise RuntimeError("decoded_audio_crepe blocked: torchcrepe code is missing")
    digest = hashlib.sha256()
    for path in paths:
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _official_torchcrepe_tracker(
    config: DecodedAudioCrepeConfig,
    *,
    device: str,
    expected_model_sha256: str | None,
) -> tuple[PitchTracker, PitchTrackerContract]:
    try:
        import torch
        import torchcrepe
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "decoded_audio_crepe blocked: torchcrepe 0.0.24 and its offline "
            "runtime dependencies are unavailable"
        ) from error

    try:
        version = importlib.metadata.version("torchcrepe")
        torchaudio_version = importlib.metadata.version("torchaudio")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "decoded_audio_crepe blocked: torchcrepe/torchaudio distribution "
            "metadata is unavailable"
        ) from error
    if version != _TORCHCREPE_VERSION:
        raise RuntimeError(
            "decoded_audio_crepe blocked: expected torchcrepe "
            f"{_TORCHCREPE_VERSION}, got {version}"
        )

    package_root = Path(torchcrepe.__file__).resolve().parent
    model_path = package_root / "assets" / f"{config.model_capacity}.pth"
    if not model_path.is_file():
        raise RuntimeError(
            "decoded_audio_crepe blocked: official offline model asset is "
            f"missing: {model_path}"
        )
    model_sha256 = _sha256_file(model_path)
    authority_sha256 = _OFFICIAL_MODEL_SHA256[config.model_capacity]
    if expected_model_sha256 is not None and not _is_sha256(expected_model_sha256):
        raise ValueError("expected CREPE model hash must be a lowercase SHA-256")
    if expected_model_sha256 is not None and expected_model_sha256 != authority_sha256:
        raise RuntimeError(
            "decoded_audio_crepe blocked: requested model SHA-256 does not "
            "match the torchcrepe 0.0.24 authority"
        )
    if model_sha256 != authority_sha256:
        raise RuntimeError(
            "decoded_audio_crepe blocked: official model asset SHA-256 "
            f"mismatch for {model_path}"
        )

    def track(
        audio: np.ndarray,
        sample_rate: int,
        tracker_config: DecodedAudioCrepeConfig,
    ) -> PitchTrack:
        canonical_audio = np.ascontiguousarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(canonical_audio).reshape(1, -1)
        waveform = waveform.to(device=device)
        seed_digest = hashlib.sha256()
        seed_digest.update(canonical_audio.tobytes())
        seed_digest.update(
            json.dumps(
                {
                    "crepe_hop_length": tracker_config.crepe_hop_length,
                    "fmin_hz": tracker_config.fmin_hz,
                    "fmax_hz": tracker_config.fmax_hz,
                    "model_capacity": tracker_config.model_capacity,
                    "decoder": "torchcrepe.decode.viterbi",
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        dither_seed = int.from_bytes(seed_digest.digest()[:4], "big")
        numpy_random_state = np.random.get_state()
        try:
            # torchcrepe's public viterbi decoder dithers pitch-bin centers
            # through scipy.stats/NumPy. Seed it per audio/config so reports do
            # not depend on rollout order, then restore the caller's RNG.
            np.random.seed(dither_seed)
            with torch.inference_mode():
                pitch, periodicity = torchcrepe.predict(
                    waveform,
                    sample_rate,
                    tracker_config.crepe_hop_length,
                    tracker_config.fmin_hz,
                    tracker_config.fmax_hz,
                    tracker_config.model_capacity,
                    batch_size=1024,
                    device=device,
                    return_periodicity=True,
                    pad=True,
                )
        finally:
            np.random.set_state(numpy_random_state)
        return PitchTrack(
            frequency_hz=pitch.detach().float().cpu().numpy().reshape(-1),
            periodicity=(periodicity.detach().float().cpu().numpy().reshape(-1)),
            frame_offsets_samples=_official_frame_offsets_samples(
                pitch.numel(),
                sample_rate=sample_rate,
                requested_hop_length=tracker_config.crepe_hop_length,
            ),
        )

    return track, PitchTrackerContract(
        implementation="torchcrepe.predict/decoded_audio_crepe",
        package_version=version,
        package_code_sha256=_package_code_sha256(package_root),
        model_capacity=config.model_capacity,
        model_path=str(model_path),
        model_sha256=model_sha256,
        torch_version=str(torch.__version__),
        torchaudio_version=torchaudio_version,
        numpy_version=np.__version__,
        device=device,
    )


def _number(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"manifest {name} must be a positive integer")
    return value


def _safe_manifest_audio_path(manifest_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("rollout raw_wav must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in {".", ".."} for part in candidate.parts):
        raise ValueError("rollout raw_wav must be a safe relative path")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    if resolved.parent != root and root not in resolved.parents:
        raise ValueError("rollout raw_wav resolves outside the audition directory")
    if not resolved.is_file():
        raise ValueError(f"rollout raw_wav is missing: {resolved}")
    return resolved


def _load_generated_audio(
    path: Path,
    *,
    sample_rate: int,
    history_frames: int,
    latent_hop: int,
    generated_frames: int,
) -> np.ndarray:
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(
            f"rollout sample rate mismatch for {path}: {actual_rate} != {sample_rate}"
        )
    mono = audio.mean(axis=1, dtype=np.float32)
    start = history_frames * latent_hop
    end = start + generated_frames * latent_hop
    if mono.size < end:
        raise ValueError(
            f"rollout is too short for declared generated interval: {path}"
        )
    generated = np.ascontiguousarray(mono[start:end], dtype=np.float32)
    if not generated.size or not np.isfinite(generated).all():
        raise ValueError(f"rollout generated audio is empty or non-finite: {path}")
    return generated


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _distribution(
    values: Sequence[float | int | None],
    *,
    worst: str,
) -> dict[str, float | int | str | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if finite.size and not np.isfinite(finite).all():
        raise ValueError("metric distribution contains non-finite values")
    if not finite.size:
        return {
            "count": 0,
            "p10": None,
            "median": None,
            "p90": None,
            "worst": None,
            "worst_direction": worst,
        }
    return {
        "count": int(finite.size),
        "p10": _percentile(finite, 10),
        "median": float(np.median(finite)),
        "p90": _percentile(finite, 90),
        "worst": float(finite.min() if worst == "minimum" else finite.max()),
        "worst_direction": worst,
    }


def _midi_to_hz(notes: np.ndarray) -> np.ndarray:
    return 440.0 * np.power(2.0, (notes.astype(np.float64) - 69.0) / 12.0)


def _official_frame_offsets_samples(
    frame_count: int,
    *,
    sample_rate: int,
    requested_hop_length: int,
) -> np.ndarray:
    """Reproduce torchcrepe's resampled frame-center timestamps.

    torchcrepe first converts the requested source-rate hop with integer
    truncation to its 16 kHz internal rate. Mapping with ``index * requested``
    would therefore drift by more than one latent frame over a long rollout.
    """

    if frame_count <= 0 or sample_rate <= 0 or requested_hop_length <= 0:
        raise ValueError("CREPE frame-offset inputs must be positive")
    internal_rate = 16000
    internal_hop = (
        requested_hop_length
        if sample_rate == internal_rate
        else int(requested_hop_length * internal_rate / sample_rate)
    )
    if internal_hop <= 0:
        raise ValueError("CREPE hop becomes zero after internal resampling")
    return (
        np.arange(frame_count, dtype=np.float64)
        * internal_hop
        * sample_rate
        / internal_rate
    )


def _transition_settling(
    *,
    requested_notes: np.ndarray,
    previous_requested_note: int | None,
    offsets: np.ndarray,
    frequency_hz: np.ndarray,
    periodicity: np.ndarray,
    finite: np.ndarray,
    sample_rate: int,
    latent_hop: int,
    periodicity_threshold: float,
    consecutive_frames: int,
) -> list[dict[str, float | int | bool | str | None]]:
    internal_changes = np.flatnonzero(requested_notes[1:] != requested_notes[:-1]) + 1
    changes = internal_changes.tolist()
    if (
        previous_requested_note is not None
        and int(requested_notes[0]) != previous_requested_note
    ):
        changes.insert(0, 0)
    events: list[dict[str, float | int | bool | str | None]] = []
    for event_position, latent_index in enumerate(changes):
        start_sample = int(latent_index * latent_hop)
        end_sample = (
            int(changes[event_position + 1] * latent_hop)
            if event_position + 1 < len(changes)
            else int(requested_notes.size * latent_hop)
        )
        selected = np.flatnonzero((offsets >= start_sample) & (offsets < end_sample))
        target_note = int(requested_notes[latent_index])
        target_hz = float(_midi_to_hz(np.asarray([target_note]))[0])
        settled_offset: float | None = None
        run = 0
        run_start = 0.0
        for index in selected:
            follows = False
            if finite[index]:
                cents = 1200.0 * math.log2(float(frequency_hz[index]) / target_hz)
                follows = (
                    periodicity[index] >= periodicity_threshold and abs(cents) <= 100.0
                )
            if follows:
                if run == 0:
                    run_start = float(offsets[index])
                run += 1
                if run >= consecutive_frames:
                    settled_offset = run_start
                    break
            else:
                run = 0
        events.append(
            {
                "event_kind": (
                    "history_to_generated_boundary"
                    if latent_index == 0
                    else "within_generated_request_change"
                ),
                "latent_frame": int(latent_index),
                "from_midi_note": (
                    previous_requested_note
                    if latent_index == 0
                    else int(requested_notes[latent_index - 1])
                ),
                "to_midi_note": target_note,
                "settled": settled_offset is not None,
                "settling_ms": (
                    None
                    if settled_offset is None
                    else (settled_offset - start_sample) * 1000.0 / sample_rate
                ),
            }
        )
    return events


def _rollout_metrics(
    audio: np.ndarray,
    requested_notes: Sequence[object],
    *,
    previous_requested_note: int | None,
    sample_rate: int,
    latent_hop: int,
    config: DecodedAudioCrepeConfig,
    tracker: PitchTracker,
) -> tuple[dict[str, Any], np.ndarray]:
    if not requested_notes:
        raise ValueError("rollout requested_notes must be non-empty")
    if any(
        isinstance(note, bool) or not isinstance(note, int) or not 0 <= note <= 127
        for note in requested_notes
    ):
        raise ValueError("rollout requested_notes must contain MIDI integers")
    if previous_requested_note is not None and (
        isinstance(previous_requested_note, bool)
        or not isinstance(previous_requested_note, int)
        or not 0 <= previous_requested_note <= 127
    ):
        raise ValueError("previous requested note must be a MIDI integer")
    notes = np.asarray(requested_notes, dtype=np.int64)
    pitch_track = tracker(audio, sample_rate, config)
    frequency = np.asarray(pitch_track.frequency_hz, dtype=np.float64).reshape(-1)
    periodicity = np.asarray(pitch_track.periodicity, dtype=np.float64).reshape(-1)
    if frequency.size != periodicity.size or not frequency.size:
        raise ValueError("pitch tracker returned empty or misaligned arrays")

    if pitch_track.frame_offsets_samples is None:
        offsets = _official_frame_offsets_samples(
            frequency.size,
            sample_rate=sample_rate,
            requested_hop_length=config.crepe_hop_length,
        )
    else:
        offsets = np.asarray(
            pitch_track.frame_offsets_samples,
            dtype=np.float64,
        ).reshape(-1)
        if offsets.size != frequency.size:
            raise ValueError("pitch tracker frame offsets are misaligned")
        if (
            not np.isfinite(offsets).all()
            or offsets[0] != 0.0
            or np.any(np.diff(offsets) <= 0.0)
        ):
            raise ValueError("pitch tracker frame offsets are invalid")
    in_range = offsets < audio.size
    frequency = frequency[in_range]
    periodicity = periodicity[in_range]
    offsets = offsets[in_range]
    latent_indices = np.minimum(
        np.floor(offsets / latent_hop).astype(np.int64),
        notes.size - 1,
    )
    targets = notes[latent_indices]
    finite = np.isfinite(frequency) & (frequency > 0.0) & np.isfinite(periodicity)
    voiced = finite & (periodicity >= config.periodicity_threshold)
    target_hz = _midi_to_hz(targets)
    signed_cents = np.full(frequency.shape, np.nan, dtype=np.float64)
    signed_cents[finite] = 1200.0 * np.log2(frequency[finite] / target_hz[finite])
    voiced_cents = signed_cents[voiced]
    absolute = np.abs(voiced_cents)
    predicted_midi = np.rint(69.0 + 12.0 * np.log2(frequency[voiced] / 440.0)).astype(
        np.int64
    )
    voiced_targets = targets[voiced]
    octave_shift = np.rint(voiced_cents / 1200.0).astype(np.int64)
    octave_error = (np.abs(octave_shift) >= 1) & (
        np.abs(voiced_cents - octave_shift * 1200.0) <= 100.0
    )
    analysis_count = int(finite.sum())
    voiced_count = int(voiced.sum())
    metrics: dict[str, Any] = {
        "tracker_frame_count": int(in_range.size),
        "in_generated_interval_frame_count": int(in_range.sum()),
        "finite_analysis_frame_count": analysis_count,
        "voiced_frame_count": voiced_count,
        "voiced_coverage": (voiced_count / analysis_count if analysis_count else None),
        "absolute_cents_median": (
            float(np.median(absolute)) if absolute.size else None
        ),
        "absolute_cents_p90": (_percentile(absolute, 90) if absolute.size else None),
        "within_50_cents": (
            float(np.mean(absolute <= 50.0)) if absolute.size else None
        ),
        "within_100_cents": (
            float(np.mean(absolute <= 100.0)) if absolute.size else None
        ),
        "exact_midi_note_accuracy": (
            float(np.mean(predicted_midi == voiced_targets)) if absolute.size else None
        ),
        "octave_error_fraction": (
            float(np.mean(octave_error)) if absolute.size else None
        ),
    }
    events = _transition_settling(
        requested_notes=notes,
        previous_requested_note=previous_requested_note,
        offsets=offsets,
        frequency_hz=frequency,
        periodicity=periodicity,
        finite=finite,
        sample_rate=sample_rate,
        latent_hop=latent_hop,
        periodicity_threshold=config.periodicity_threshold,
        consecutive_frames=config.settling_consecutive_frames,
    )
    metrics["transition"] = {
        "event_count": len(events),
        "settled_event_count": sum(bool(event["settled"]) for event in events),
        "settling_ms": _distribution(
            [event["settling_ms"] for event in events],
            worst="maximum",
        ),
        "events": events,
        "status": (
            "not_applicable_no_requested_note_change" if not events else "measured"
        ),
    }
    return metrics, voiced_cents


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    voiced_cents: Sequence[np.ndarray],
) -> dict[str, Any]:
    all_cents = (
        np.concatenate([value for value in voiced_cents if value.size])
        if any(value.size for value in voiced_cents)
        else np.asarray([], dtype=np.float64)
    )
    absolute = np.abs(all_cents)
    analysis_count = sum(int(row["finite_analysis_frame_count"]) for row in rows)
    voiced_count = sum(int(row["voiced_frame_count"]) for row in rows)
    octave_shift = np.rint(all_cents / 1200.0).astype(np.int64)
    octave_error = (np.abs(octave_shift) >= 1) & (
        np.abs(all_cents - octave_shift * 1200.0) <= 100.0
    )
    metrics = (
        "finite_analysis_frame_count",
        "voiced_frame_count",
        "voiced_coverage",
        "absolute_cents_median",
        "absolute_cents_p90",
        "within_50_cents",
        "within_100_cents",
        "exact_midi_note_accuracy",
        "octave_error_fraction",
    )
    minimum_worst = {
        "finite_analysis_frame_count",
        "voiced_frame_count",
        "voiced_coverage",
        "within_50_cents",
        "within_100_cents",
        "exact_midi_note_accuracy",
    }
    events = [event for row in rows for event in row["transition"]["events"]]
    return {
        "rollout_count": len(rows),
        "aggregate": {
            "finite_analysis_frame_count": analysis_count,
            "voiced_frame_count": voiced_count,
            "voiced_coverage": (
                voiced_count / analysis_count if analysis_count else None
            ),
            "absolute_cents_median": (
                float(np.median(absolute)) if absolute.size else None
            ),
            "absolute_cents_p90": (
                _percentile(absolute, 90) if absolute.size else None
            ),
            "within_50_cents": (
                float(np.mean(absolute <= 50.0)) if absolute.size else None
            ),
            "within_100_cents": (
                float(np.mean(absolute <= 100.0)) if absolute.size else None
            ),
            "exact_midi_note_accuracy": (
                float(np.mean(np.rint(all_cents / 100.0) == 0.0))
                if absolute.size
                else None
            ),
            "octave_error_fraction": (
                float(np.mean(octave_error)) if absolute.size else None
            ),
        },
        "rollout_distributions": {
            name: _distribution(
                [row[name] for row in rows],
                worst="minimum" if name in minimum_worst else "maximum",
            )
            for name in metrics
        },
        "transition": {
            "event_count": len(events),
            "settled_event_count": sum(bool(event["settled"]) for event in events),
            "settling_ms": _distribution(
                [event["settling_ms"] for event in events],
                worst="maximum",
            ),
            "status": (
                "not_applicable_no_requested_note_change" if not events else "measured"
            ),
        },
    }


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1.0e-12))


def _velocity_step_metrics(
    audio: np.ndarray,
    requested_velocities: Sequence[object],
    *,
    latent_hop: int,
) -> dict[str, Any]:
    """Measure velocity-request direction through a loudness-response proxy.

    Steady regions are the trailing half of each pre/post request segment.  No
    claim is made that decoded RMS is an accurate reconstruction of MIDI
    velocity; this panel remains report-only.
    """

    if not requested_velocities:
        raise ValueError("velocity_step requested_velocities must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 127
        for value in requested_velocities
    ):
        raise ValueError(
            "velocity_step requested_velocities must contain MIDI integers"
        )
    velocities = np.asarray(requested_velocities, dtype=np.int64)
    expected_samples = velocities.size * latent_hop
    if audio.size != expected_samples:
        raise ValueError("velocity_step audio/control lengths are misaligned")
    changes = (np.flatnonzero(velocities[1:] != velocities[:-1]) + 1).tolist()
    events: list[dict[str, Any]] = []
    boundaries = [0, *changes, int(velocities.size)]
    for event_index, change in enumerate(changes):
        previous_start = boundaries[event_index]
        following_end = boundaries[event_index + 2]
        pre_start = previous_start + (change - previous_start) // 2
        post_start = change + (following_end - change) // 2
        if pre_start >= change or post_start >= following_end:
            raise ValueError("velocity_step has no non-empty steady regions")
        pre_audio = audio[pre_start * latent_hop : change * latent_hop]
        post_audio = audio[post_start * latent_hop : following_end * latent_hop]
        pre_rms_db = _rms_db(pre_audio)
        post_rms_db = _rms_db(post_audio)
        delta_db = post_rms_db - pre_rms_db
        from_velocity = int(velocities[change - 1])
        to_velocity = int(velocities[change])
        requested_direction = "increase" if to_velocity > from_velocity else "decrease"
        direction_consistent = (
            delta_db > 0.0 if requested_direction == "increase" else delta_db < 0.0
        )
        events.append(
            {
                "event_kind": "within_generated_velocity_request_change",
                "latent_frame": int(change),
                "from_velocity": from_velocity,
                "to_velocity": to_velocity,
                "requested_direction": requested_direction,
                "pre_steady_latent_frames": [int(pre_start), int(change)],
                "post_steady_latent_frames": [int(post_start), int(following_end)],
                "pre_steady_rms_db": pre_rms_db,
                "post_steady_rms_db": post_rms_db,
                "rms_delta_db": delta_db,
                "requested_direction_consistent": direction_consistent,
            }
        )
    return {
        "metric_kind": "decoded_audio_velocity_loudness_proxy",
        "status": "measured" if events else "invalid_no_velocity_transition",
        "report_only": True,
        "semantic_warning": (
            "Generated-audio steady-state RMS direction is a velocity-response "
            "proxy only; it is not MIDI velocity accuracy."
        ),
        "transition_at_generated_midpoint": changes == [velocities.size // 2],
        "event_count": len(events),
        "events": events,
    }


def _velocity_proxy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events = [event for row in rows for event in row.get("events", [])]
    consistent = [bool(event["requested_direction_consistent"]) for event in events]
    return {
        "metric_kind": "decoded_audio_velocity_loudness_proxy",
        "status": "measured" if events else "not_present",
        "report_only": True,
        "semantic_warning": (
            "Generated-audio steady-state RMS direction is a velocity-response "
            "proxy only; it is not MIDI velocity accuracy."
        ),
        "rollout_count": len(rows),
        "event_count": len(events),
        "requested_direction_consistent_count": sum(consistent),
        "requested_direction_consistency_fraction": (
            sum(consistent) / len(consistent) if consistent else None
        ),
        "rms_delta_db": _distribution(
            [event["rms_delta_db"] for event in events],
            worst="minimum",
        ),
        "events": [dict(event) for event in events],
    }


def _hard_checks(
    conditions: Mapping[str, Mapping[str, Any]],
    config: DecodedAudioCrepeConfig,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    checks: dict[str, dict[str, Any]] = {}
    for kind in _HARD_CONDITION_KINDS:
        condition = conditions.get(kind)
        aggregate = (
            condition.get("aggregate") if isinstance(condition, Mapping) else None
        )
        rollout_count = (
            condition.get("rollout_count", 0) if isinstance(condition, Mapping) else 0
        )
        analysis_count = (
            aggregate.get("finite_analysis_frame_count", 0)
            if isinstance(aggregate, Mapping)
            else 0
        )
        voiced_count = (
            aggregate.get("voiced_frame_count", 0)
            if isinstance(aggregate, Mapping)
            else 0
        )
        p90 = (
            aggregate.get("absolute_cents_p90")
            if isinstance(aggregate, Mapping)
            else None
        )
        within = (
            aggregate.get("within_100_cents")
            if isinstance(aggregate, Mapping)
            else None
        )
        values = {
            f"{kind}_rollout_coverage": (rollout_count, ">", 0, rollout_count > 0),
            f"{kind}_finite_analysis_coverage": (
                analysis_count,
                ">",
                0,
                analysis_count > 0,
            ),
            f"{kind}_voiced_measurement_coverage": (
                voiced_count,
                ">",
                0,
                voiced_count > 0,
            ),
            f"{kind}_p90_absolute_cents": (
                p90,
                "<=",
                config.maximum_p90_absolute_cents,
                isinstance(p90, (int, float))
                and not isinstance(p90, bool)
                and math.isfinite(float(p90))
                and float(p90) <= config.maximum_p90_absolute_cents,
            ),
            f"{kind}_within_100_cents": (
                within,
                ">=",
                config.minimum_within_100_cents,
                isinstance(within, (int, float))
                and not isinstance(within, bool)
                and math.isfinite(float(within))
                and float(within) >= config.minimum_within_100_cents,
            ),
        }
        for name, (value, operator, threshold, passed) in values.items():
            checks[name] = {
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return checks, failed


def evaluate_decoded_audio_midi_manifest(
    manifest_path: str | Path,
    *,
    config: DecodedAudioCrepeConfig | None = None,
    device: str = "cpu",
    expected_model_sha256: str | None = None,
    pitch_tracker: PitchTracker | None = None,
    tracker_contract: PitchTrackerContract | None = None,
) -> dict[str, Any]:
    resolved_config = config or DecodedAudioCrepeConfig()
    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("MIDI audition manifest must be an object")
    if manifest.get("conditioning") != "midi_sequence":
        raise ValueError("decoded_audio_crepe requires a MIDI sequence audition")
    sample_rate = _number(manifest.get("sample_rate"), "sample_rate")
    latent_hop = _number(manifest.get("latent_hop"), "latent_hop")
    history_frames = _number(manifest.get("history_frames"), "history_frames")
    generated_frames = _number(manifest.get("generated_frames"), "generated_frames")
    if pitch_tracker is None:
        pitch_tracker, resolved_tracker_contract = _official_torchcrepe_tracker(
            resolved_config,
            device=device,
            expected_model_sha256=expected_model_sha256,
        )
    else:
        if tracker_contract is None:
            raise ValueError("an injected pitch tracker requires a tracker contract")
        resolved_tracker_contract = tracker_contract
        if resolved_tracker_contract.model_capacity != resolved_config.model_capacity:
            raise ValueError("injected tracker model capacity does not match config")
        if not all(
            _is_sha256(value)
            for value in (
                resolved_tracker_contract.package_code_sha256,
                resolved_tracker_contract.model_sha256,
            )
        ):
            raise ValueError("injected tracker contract hashes are invalid")

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("MIDI audition manifest has no examples")
    rows_by_kind: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in _PITCH_PANEL_CONDITION_KINDS
    }
    cents_by_kind: dict[str, list[np.ndarray]] = {
        kind: [] for kind in _PITCH_PANEL_CONDITION_KINDS
    }
    velocity_proxy_rows: list[dict[str, Any]] = []
    evaluated_rollouts: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("MIDI audition example must be an object")
        rollouts = example.get("rollouts")
        if not isinstance(rollouts, list):
            raise ValueError("MIDI audition example rollouts must be a list")
        history_requested_notes = example.get("history_requested_notes")
        if history_requested_notes is None:
            previous_requested_note = None
        elif (
            not isinstance(history_requested_notes, list)
            or len(history_requested_notes) != history_frames
            or any(
                isinstance(note, bool)
                or not isinstance(note, int)
                or not 0 <= note <= 127
                for note in history_requested_notes
            )
        ):
            raise ValueError(
                "example history_requested_notes must match history_frames"
            )
        else:
            previous_requested_note = history_requested_notes[-1]
        for rollout_index, rollout in enumerate(rollouts):
            if not isinstance(rollout, Mapping):
                raise ValueError("MIDI audition rollout must be an object")
            kind = rollout.get("condition_kind")
            if (
                kind not in _PITCH_PANEL_CONDITION_KINDS
                and kind != _VELOCITY_PANEL_CONDITION_KIND
            ):
                continue
            audio_path = _safe_manifest_audio_path(source, rollout.get("raw_wav"))
            audio_sha256 = _sha256_file(audio_path)
            audio = _load_generated_audio(
                audio_path,
                sample_rate=sample_rate,
                history_frames=history_frames,
                latent_hop=latent_hop,
                generated_frames=generated_frames,
            )
            if _sha256_file(audio_path) != audio_sha256:
                raise ValueError(f"rollout raw_wav changed while reading: {audio_path}")
            if kind == _VELOCITY_PANEL_CONDITION_KIND:
                requested_velocities = rollout.get("requested_velocities")
                if (
                    not isinstance(requested_velocities, list)
                    or len(requested_velocities) != generated_frames
                ):
                    raise ValueError(
                        "velocity_step requested_velocities must match manifest "
                        "generated_frames"
                    )
                velocity_metrics = _velocity_step_metrics(
                    audio,
                    requested_velocities,
                    latent_hop=latent_hop,
                )
                velocity_proxy_rows.append(velocity_metrics)
                evaluated_rollouts.append(
                    {
                        "example_index": example_index,
                        "rollout_index": rollout_index,
                        "sample_id": example.get("sample_id"),
                        "condition_kind": kind,
                        "generation_seed": rollout.get("generation_seed"),
                        "raw_wav": str(audio_path.relative_to(source.parent.resolve())),
                        "raw_wav_sha256": audio_sha256,
                        "generated_sample_start": history_frames * latent_hop,
                        "generated_sample_count": generated_frames * latent_hop,
                        "metrics": velocity_metrics,
                    }
                )
                continue
            requested_notes = rollout.get("requested_notes")
            if (
                not isinstance(requested_notes, list)
                or len(requested_notes) != generated_frames
            ):
                raise ValueError(
                    "rollout requested_notes must match manifest generated_frames"
                )
            metrics, cents = _rollout_metrics(
                audio,
                requested_notes,
                previous_requested_note=previous_requested_note,
                sample_rate=sample_rate,
                latent_hop=latent_hop,
                config=resolved_config,
                tracker=pitch_tracker,
            )
            rows_by_kind[str(kind)].append(metrics)
            cents_by_kind[str(kind)].append(cents)
            evaluated_rollouts.append(
                {
                    "example_index": example_index,
                    "rollout_index": rollout_index,
                    "sample_id": example.get("sample_id"),
                    "condition_kind": kind,
                    "generation_seed": rollout.get("generation_seed"),
                    "raw_wav": str(audio_path.relative_to(source.parent.resolve())),
                    "raw_wav_sha256": audio_sha256,
                    "generated_sample_start": history_frames * latent_hop,
                    "generated_sample_count": generated_frames * latent_hop,
                    "metrics": metrics,
                }
            )

    missing_conditions = [
        kind for kind in _HARD_CONDITION_KINDS if not rows_by_kind[kind]
    ]
    if missing_conditions:
        raise ValueError(
            "MIDI audition manifest lacks required decoded_audio_crepe "
            f"conditions: {missing_conditions}"
        )

    conditions = {
        kind: _condition_summary(rows_by_kind[kind], cents_by_kind[kind])
        for kind in _PITCH_PANEL_CONDITION_KINDS
        if rows_by_kind[kind]
    }
    velocity_proxy = _velocity_proxy_summary(velocity_proxy_rows)
    checks, failed = _hard_checks(conditions, resolved_config)
    checkpoint = manifest.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    initialization = checkpoint.get("initialization")
    initialization = initialization if isinstance(initialization, Mapping) else {}
    pitch_probe = manifest.get("pitch_probe")
    pitch_probe = pitch_probe if isinstance(pitch_probe, Mapping) else {}
    manifest_sha256 = _sha256_file(source)
    manifest_lineage = {
        "audition_manifest_path": str(source),
        "audition_manifest_sha256": manifest_sha256,
        "audition_config_sha256": (
            manifest.get("config", {}).get("sha256")
            if isinstance(manifest.get("config"), Mapping)
            else None
        ),
        "generator_checkpoint_sha256": checkpoint.get("sha256"),
        "initializer_checkpoint_sha256": initialization.get("checkpoint_sha256"),
        "pitch_probe_checkpoint_sha256": pitch_probe.get("checkpoint_sha256"),
        "pitch_probe_qualification_sha256": pitch_probe.get("qualification_sha256"),
        "codec_sha256": (
            manifest.get("codec", {}).get("sha256")
            if isinstance(manifest.get("codec"), Mapping)
            else None
        ),
        "pack_index_sha256": manifest.get("pack_index_sha256"),
        "statistics_sha256": manifest.get("statistics_sha256"),
    }
    required_lineage = (
        "audition_manifest_sha256",
        "audition_config_sha256",
        "generator_checkpoint_sha256",
        "initializer_checkpoint_sha256",
        "pitch_probe_checkpoint_sha256",
        "pitch_probe_qualification_sha256",
        "codec_sha256",
        "pack_index_sha256",
        "statistics_sha256",
    )
    invalid_lineage = [
        name for name in required_lineage if not _is_sha256(manifest_lineage[name])
    ]
    if invalid_lineage:
        raise ValueError(
            "MIDI audition manifest has invalid or missing hash lineage: "
            f"{invalid_lineage}"
        )
    checks["complete_hash_lineage"] = {
        "value": not invalid_lineage,
        "operator": "==",
        "threshold": True,
        "passed": not invalid_lineage,
        "invalid_or_missing": invalid_lineage,
    }

    evaluation_config = asdict(resolved_config)
    return {
        "schema": 1,
        "kind": "zrave-midi-decoded-audio-crepe-adherence-gate",
        "metric_kind": "decoded_audio_crepe",
        "manifest": str(source),
        "manifest_sha256": manifest_sha256,
        "passed": not failed,
        "failed_checks": failed,
        "hard_checks": checks,
        "report_only_metrics": {
            "voiced_coverage": (
                "reported independently; no hard threshold until calibrated "
                "against held-out source and direct-RAVE ceilings"
            ),
            "note_step_transition_settling_ms": {
                "status": ("measured" if "note_step" in conditions else "not_present"),
                "hard_gate": False,
                "reason": (
                    "reported independently until a held-out transition "
                    "baseline is calibrated"
                ),
                "summary": (
                    conditions["note_step"]["transition"]
                    if "note_step" in conditions
                    else None
                ),
            },
            "velocity_step_rms_response": velocity_proxy,
        },
        "lineage": manifest_lineage,
        "tool": {
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "tracker": asdict(resolved_tracker_contract),
        "evaluation_config": evaluation_config,
        "evaluation_config_sha256": _canonical_sha256(evaluation_config),
        "analysis_scope": {
            "audio_region": "generated_only",
            "generated_start_sample": history_frames * latent_hop,
            "generated_sample_count": generated_frames * latent_hop,
            "f0_frame_to_midi_policy": (
                "floor(crepe_frame_center_sample / latent_hop), clamped to "
                "the final requested latent frame"
            ),
            "voiced_coverage_denominator": "all finite in-range CREPE frames",
            "pitch_error_denominator": (
                "finite frames at or above periodicity threshold"
            ),
            "octave_error_definition": (
                "nonzero nearest 1200-cent shift with <=100-cent residual"
            ),
            "velocity_proxy_steady_region_policy": (
                "trailing half of each pre/post velocity-request segment; "
                "generated audio only"
            ),
        },
        "conditions": conditions,
        "velocity_loudness_proxy": velocity_proxy,
        "rollouts": evaluated_rollouts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decoded_audio_crepe MIDI adherence on only the generated "
            "region of a 128D MIDI audition manifest. This is the decoded-audio "
            "gate, not the latent pitch-probe proxy. note_step settling and the "
            "velocity_step RMS-direction proxy are report-only."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crepe-hop-length", type=int, default=512)
    parser.add_argument("--fmin-hz", type=float, default=50.0)
    parser.add_argument("--fmax-hz", type=float, default=2000.0)
    parser.add_argument("--model-capacity", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--periodicity-threshold", type=float, default=0.5)
    parser.add_argument("--settling-consecutive-frames", type=int, default=3)
    parser.add_argument("--maximum-p90-absolute-cents", type=float, default=100.0)
    parser.add_argument("--minimum-within-100-cents", type=float, default=0.90)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = evaluate_decoded_audio_midi_manifest(
        args.manifest,
        config=DecodedAudioCrepeConfig(
            crepe_hop_length=args.crepe_hop_length,
            fmin_hz=args.fmin_hz,
            fmax_hz=args.fmax_hz,
            model_capacity=args.model_capacity,
            periodicity_threshold=args.periodicity_threshold,
            settling_consecutive_frames=args.settling_consecutive_frames,
            maximum_p90_absolute_cents=args.maximum_p90_absolute_cents,
            minimum_within_100_cents=args.minimum_within_100_cents,
        ),
        device=args.device,
        expected_model_sha256=args.expected_model_sha256,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_reject and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
