from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FEATURE_VERSION = "zrave-timbre-proxy-v1"
DISCLAIMER = (
    "Relative engineering proxies derived from latent trajectories or audio; "
    "they are not perceptual ground-truth timbre labels."
)
_SPLITS = ("train", "validation", "test")
_REQUIRED_SEQUENCE_FIELDS = {
    "manifest_index",
    "sample_id",
    "source_name",
    "source_code",
    "category",
    "category_code",
    "canonical_preset_id",
    "split",
    "split_code",
    "midi_note",
    "velocity",
    "maximum_future_frames",
    "length",
    "active_frames",
    "packed_index",
    "shard",
    "shard_row",
}
_LATENT_AXES = {
    "temporal_rate": (
        "latent_temporal_rate_proxy_score",
        "latent_slow_proxy",
        "latent_fast_proxy",
    ),
    "onset": (
        "latent_onset_proxy_score",
        "latent_soft_onset_proxy",
        "latent_hard_onset_proxy",
    ),
    "motion": (
        "latent_motion_proxy_score",
        "latent_static_proxy",
        "latent_moving_proxy",
    ),
    "irregularity": (
        "latent_irregularity_proxy_score",
        "latent_smooth_proxy",
        "latent_irregular_proxy",
    ),
}
_AUDIO_AXES = {
    "brightness": (
        "audio_brightness_proxy_score",
        "audio_dark_proxy",
        "audio_bright_proxy",
    ),
    "attack": (
        "audio_attack_proxy_score",
        "audio_soft_attack_proxy",
        "audio_hard_attack_proxy",
    ),
    "motion": (
        "audio_motion_proxy_score",
        "audio_static_proxy",
        "audio_moving_proxy",
    ),
    "noisiness": (
        "audio_noisiness_proxy_score",
        "audio_clean_proxy",
        "audio_noisy_proxy",
    ),
}
_AXES_BY_SOURCE = {"latent": _LATENT_AXES, "audio": _AUDIO_AXES}
_SHARD_METADATA_FIELDS = {
    "lengths": "length",
    "active_frames": "active_frames",
    "notes": "midi_note",
    "velocities": "velocity",
    "split_codes": "split_code",
    "source_codes": "source_code",
    "category_codes": "category_code",
    "maximum_future_frames": "maximum_future_frames",
    "manifest_indices": "manifest_index",
}
_AXIS_DESCRIPTIONS = {
    "latent": {
        "temporal_rate": "temporal frequency centroid of centered latent trajectories",
        "onset": "early latent-norm rise relative to median latent norm",
        "motion": "median frame delta norm relative to median latent norm",
        "irregularity": "temporal spectral flatness of centered latent trajectories",
    },
    "audio": {
        "brightness": "median normalized spectral centroid over active audio frames",
        "attack": "early RMS rise relative to median active-frame RMS",
        "motion": "median active-frame RMS and spectral-centroid change",
        "noisiness": "median spectral flatness over active audio frames",
    },
}


@dataclass(frozen=True)
class TaxonomyLimits:
    minimum_bucket_presets: int = 20
    minimum_bucket_records: int = 100
    minimum_train_presets: int = 10
    minimum_validation_presets: int = 1
    minimum_test_presets: int = 1

    def validate(self) -> None:
        values = {
            "minimum_bucket_presets": self.minimum_bucket_presets,
            "minimum_bucket_records": self.minimum_bucket_records,
            "minimum_train_presets": self.minimum_train_presets,
            "minimum_validation_presets": self.minimum_validation_presets,
            "minimum_test_presets": self.minimum_test_presets,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("taxonomy minimum counts must be non-negative integers")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_named_file_hashes(
    paths: Mapping[str, Path],
    *,
    workers: int = 1,
) -> dict[str, object]:
    """Bind every selected sample ID to its audio content without a huge report."""

    if workers <= 0:
        raise ValueError("taxonomy workers must be positive")
    unique_paths = sorted(set(paths.values()), key=str)
    if workers == 1:
        digests = [_sha256_file(path) for path in unique_paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            digests = list(executor.map(_sha256_file, unique_paths))
    content_hashes = dict(zip(unique_paths, digests, strict=True))
    aggregate = hashlib.sha256()
    for sample_id in sorted(paths):
        path = paths[sample_id]
        if path not in content_hashes:
            content_hashes[path] = _sha256_file(path)
        digest = content_hashes[path]
        aggregate.update(sample_id.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "count": len(paths),
        "unique_files": len(content_hashes),
        "sample_id_content_sha256": aggregate.hexdigest(),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_ids(path: Path, values: Iterable[str]) -> None:
    identifiers = list(values)
    path.write_text(
        "".join(f"{value}\n" for value in identifiers),
        encoding="utf-8",
    )


def _safe_pack_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe pack-relative path: {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"pack path escapes root: {value!r}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _finite_score(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {name}")
    return parsed


def _latent_proxy_features(latent: np.ndarray) -> dict[str, float]:
    """Return relative trajectory proxies without claiming audio semantics."""

    value = np.asarray(latent, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 4 or value.shape[1] < 1:
        raise ValueError("latent proxy input must be [frames>=4, channels>=1]")
    if not np.isfinite(value).all():
        raise ValueError("latent proxy input contains non-finite values")
    epsilon = 1.0e-12
    centered = value - value.mean(axis=0, keepdims=True)
    temporal_power = np.square(np.abs(np.fft.rfft(centered, axis=0))).sum(axis=1)
    temporal_power = temporal_power[1:]
    frequencies = np.linspace(0.0, 1.0, temporal_power.size + 1)[1:]
    total_temporal_power = float(temporal_power.sum())
    if total_temporal_power <= epsilon:
        temporal_rate = 0.0
        irregularity = 0.0
    else:
        temporal_rate = float(
            np.sum(frequencies * temporal_power) / total_temporal_power
        )
        irregularity = float(
            np.exp(np.mean(np.log(temporal_power + epsilon)))
            / float(np.mean(temporal_power + epsilon))
        )

    envelope = np.linalg.norm(value, axis=1)
    early_frames = max(2, min(8, value.shape[0] // 4))
    envelope_scale = max(float(np.median(np.abs(envelope))), epsilon)
    attack = float(
        max(0.0, float(np.max(envelope[:early_frames]) - envelope[0])) / envelope_scale
    )
    delta = np.diff(value, axis=0)
    motion = float(np.median(np.linalg.norm(delta, axis=1)) / envelope_scale)
    return {
        "latent_temporal_rate_proxy_score": _finite_score(
            temporal_rate, "latent temporal-rate proxy"
        ),
        "latent_onset_proxy_score": _finite_score(attack, "latent onset proxy"),
        "latent_motion_proxy_score": _finite_score(motion, "latent motion proxy"),
        "latent_irregularity_proxy_score": _finite_score(
            irregularity, "latent irregularity proxy"
        ),
    }


def _audio_frames(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    if audio.size < frame_samples:
        audio = np.pad(audio, (0, frame_samples - audio.size))
    starts = range(0, max(1, audio.size - frame_samples + 1), frame_samples // 2)
    frames = [audio[start : start + frame_samples] for start in starts]
    if not frames:
        frames = [audio[:frame_samples]]
    if frames[-1].size < frame_samples:
        frames[-1] = np.pad(frames[-1], (0, frame_samples - frames[-1].size))
    return np.stack(frames).astype(np.float64, copy=False)


def _audio_proxy_features(path: Path) -> dict[str, float]:
    channels, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if channels.shape[0] == 0 or channels.shape[1] == 0 or sample_rate <= 0:
        raise ValueError(f"invalid audio for taxonomy: {path}")
    if not np.isfinite(channels).all():
        raise ValueError(f"non-finite audio for taxonomy: {path}")
    audio = channels.mean(axis=1, dtype=np.float64)
    frame_samples = max(256, round(0.046 * sample_rate))
    frames = _audio_frames(audio, frame_samples)
    window = np.hanning(frame_samples)
    spectrum = np.square(np.abs(np.fft.rfft(frames * window, axis=1)))
    epsilon = 1.0e-12
    frequencies = np.linspace(0.0, 1.0, spectrum.shape[1])
    energy = spectrum.sum(axis=1)
    centroid = (spectrum * frequencies[None, :]).sum(axis=1) / np.maximum(
        energy,
        epsilon,
    )
    flatness = np.exp(np.mean(np.log(spectrum + epsilon), axis=1)) / np.maximum(
        np.mean(spectrum + epsilon, axis=1),
        epsilon,
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak_rms = float(np.max(rms))
    if peak_rms <= 1.0e-7:
        raise ValueError(f"audio is silent or too quiet for taxonomy: {path}")
    active_mask = rms >= max(1.0e-7, peak_rms * 1.0e-4)
    if int(active_mask.sum()) < 2:
        raise ValueError(f"audio has fewer than two active analysis frames: {path}")
    active_rms = rms[active_mask]
    active_centroid = centroid[active_mask]
    active_flatness = flatness[active_mask]
    early_frames = max(2, min(10, rms.size // 4))
    rms_scale = max(float(np.median(active_rms)), epsilon)
    attack = max(0.0, float(np.max(rms[:early_frames]) - rms[0])) / rms_scale
    rms_motion = float(np.median(np.abs(np.diff(active_rms)))) / rms_scale
    centroid_motion = (
        float(np.median(np.abs(np.diff(active_centroid))))
        if active_centroid.size > 1
        else 0.0
    )
    return {
        "audio_brightness_proxy_score": _finite_score(
            float(np.median(active_centroid)), "audio brightness proxy"
        ),
        "audio_attack_proxy_score": _finite_score(attack, "audio attack proxy"),
        "audio_motion_proxy_score": _finite_score(
            rms_motion + centroid_motion, "audio motion proxy"
        ),
        "audio_noisiness_proxy_score": _finite_score(
            float(np.median(active_flatness)), "audio noisiness proxy"
        ),
    }


def _validate_sequences(rows: Sequence[Mapping[str, object]]) -> None:
    sample_ids: set[str] = set()
    packed_indices: set[int] = set()
    shard_locations: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        missing = _REQUIRED_SEQUENCE_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"sequence row {index} is missing: {', '.join(sorted(missing))}"
            )
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        for name in ("source_name", "category", "canonical_preset_id", "shard"):
            value = row[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid sequence {name}: {value!r}")
        split = str(row["split"])
        if split not in _SPLITS:
            raise ValueError(f"invalid sequence split: {split!r}")
        for name in (
            "manifest_index",
            "source_code",
            "category_code",
            "split_code",
            "midi_note",
            "velocity",
            "maximum_future_frames",
            "length",
            "active_frames",
            "packed_index",
            "shard_row",
        ):
            value = row[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if name in {"length", "active_frames"} else 0)
            ):
                raise ValueError(f"invalid sequence {name}: {value!r}")
        if int(row["active_frames"]) > int(row["length"]):
            raise ValueError("sequence active_frames exceeds length")
        packed_index = int(row["packed_index"])
        if packed_index != index or packed_index in packed_indices:
            raise ValueError(
                "sequence packed_index must be unique and equal JSONL row order: "
                f"row {index}, value {packed_index}"
            )
        packed_indices.add(packed_index)
        location = (str(row["shard"]), int(row["shard_row"]))
        if location in shard_locations:
            raise ValueError(f"duplicate sequence shard location: {location}")
        shard_locations.add(location)


def _manifest_audio_paths(path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in _read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        audio_value = row.get("audio_path")
        if not sample_id or not isinstance(audio_value, str):
            raise ValueError("audio manifest rows require sample_id and audio_path")
        if sample_id in paths:
            raise ValueError(f"duplicate manifest sample_id: {sample_id}")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        paths[sample_id] = audio_path
    return paths


def _validate_preset_consistency(rows: Sequence[Mapping[str, object]]) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    categories: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        preset_id = str(row["canonical_preset_id"])
        splits[preset_id].add(str(row["split"]))
        categories[preset_id].add(str(row["category"]))
    for preset_id in sorted(splits):
        if len(splits[preset_id]) != 1:
            raise ValueError(
                f"preset appears in multiple splits: {preset_id}: "
                f"{sorted(splits[preset_id])}"
            )
        if len(categories[preset_id]) != 1:
            raise ValueError(f"preset appears in multiple categories: {preset_id}")


def _source_contract(
    pack_root: Path,
    rows: Sequence[Mapping[str, object]],
    manifest_path: Path | None,
) -> tuple[dict[str, object], dict[str, Path]]:
    index_path = pack_root / "index.json"
    sequences_path = pack_root / "sequences.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError("pack index must be an object")
    if int(index.get("records", -1)) != len(rows):
        raise ValueError("pack index record count does not match sequences")
    expected_shards = index.get("shard_sha256")
    if not isinstance(expected_shards, dict) or not expected_shards:
        raise ValueError("pack index has no shard hashes")
    referenced = {str(row["shard"]) for row in rows}
    if referenced != set(expected_shards):
        raise ValueError("sequence shard set does not match pack index")
    shard_paths: dict[str, Path] = {}
    shard_hashes: dict[str, str] = {}
    for name in sorted(referenced):
        path = _safe_pack_path(pack_root, name)
        digest = _sha256_file(path)
        if digest != expected_shards[name]:
            raise ValueError(f"pack shard hash mismatch: {name}")
        shard_paths[name] = path
        shard_hashes[name] = digest
    contract: dict[str, object] = {
        "pack_root": str(pack_root),
        "index": {"path": str(index_path), "sha256": _sha256_file(index_path)},
        "sequences": {
            "path": str(sequences_path),
            "sha256": _sha256_file(sequences_path),
            "records": len(rows),
        },
        "shards": shard_hashes,
    }
    if manifest_path is not None:
        contract["manifest"] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
    return contract, shard_paths


def _record_features_from_latents(
    rows: Sequence[Mapping[str, object]],
    shard_paths: Mapping[str, Path],
) -> dict[str, dict[str, float]]:
    by_shard: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["shard"])].append(row)
    features: dict[str, dict[str, float]] = {}
    for name in sorted(by_shard):
        with np.load(shard_paths[name], allow_pickle=False) as shard:
            latents = _validate_shard_members(name, shard, by_shard[name])
            for row in by_shard[name]:
                shard_row = int(row["shard_row"])
                active = int(row["active_frames"])
                if active < 4:
                    raise ValueError(
                        f"sequence needs four active frames: {row['sample_id']}"
                    )
                features[str(row["sample_id"])] = _latent_proxy_features(
                    latents[shard_row, :active]
                )
    return features


def _validate_shard_members(
    name: str,
    shard: Any,
    rows: Sequence[Mapping[str, object]],
) -> np.ndarray:
    required_arrays = {"latents", *_SHARD_METADATA_FIELDS}
    missing_arrays = required_arrays - set(shard.files)
    if missing_arrays:
        raise ValueError(
            f"pack shard {name} lacks arrays: " + ", ".join(sorted(missing_arrays))
        )
    latents = shard["latents"]
    if latents.ndim != 3:
        raise ValueError(f"pack shard latents are not rank three: {name}")
    for array_name in _SHARD_METADATA_FIELDS:
        array = shard[array_name]
        if array.ndim != 1 or array.shape[0] != latents.shape[0]:
            raise ValueError(f"pack shard metadata shape mismatch: {name}:{array_name}")
    for row in rows:
        shard_row = int(row["shard_row"])
        length = int(row["length"])
        if shard_row >= latents.shape[0] or length > latents.shape[1]:
            raise ValueError(f"sequence location is outside shard: {name}")
        for array_name, row_name in _SHARD_METADATA_FIELDS.items():
            packed_value = int(shard[array_name][shard_row])
            sequence_value = int(row[row_name])
            if packed_value != sequence_value:
                raise ValueError(
                    "sequence metadata does not match shard: "
                    f"{row['sample_id']}:{row_name} "
                    f"{sequence_value} != {packed_value}"
                )
    return latents


def _validate_shard_metadata(
    rows: Sequence[Mapping[str, object]],
    shard_paths: Mapping[str, Path],
) -> None:
    by_shard: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["shard"])].append(row)
    for name in sorted(by_shard):
        with np.load(shard_paths[name], allow_pickle=False) as shard:
            _validate_shard_members(name, shard, by_shard[name])


def _record_features_from_audio(
    rows: Sequence[Mapping[str, object]],
    paths: Mapping[str, Path],
    *,
    workers: int = 1,
) -> dict[str, dict[str, float]]:
    missing = sorted({str(row["sample_id"]) for row in rows} - set(paths))
    if missing:
        raise ValueError(
            "audio manifest lacks packed samples: " + ", ".join(missing[:10])
        )
    if workers <= 0:
        raise ValueError("taxonomy workers must be positive")
    sample_ids = [str(row["sample_id"]) for row in rows]
    audio_paths = [paths[sample_id] for sample_id in sample_ids]
    if workers == 1:
        values = [_audio_proxy_features(path) for path in audio_paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(_audio_proxy_features, audio_paths))
    return dict(zip(sample_ids, values, strict=True))


def _aggregate_presets(
    rows: Sequence[Mapping[str, object]],
    record_features: Mapping[str, Mapping[str, float]],
    feature_source: str,
) -> list[dict[str, object]]:
    score_names = tuple(
        score_name
        for score_name, _low, _high in _AXES_BY_SOURCE[feature_source].values()
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["canonical_preset_id"])].append(row)
    presets: list[dict[str, object]] = []
    for preset_id in sorted(grouped):
        members = grouped[preset_id]
        splits = {str(row["split"]) for row in members}
        if len(splits) != 1:
            raise ValueError(
                f"preset appears in multiple splits: {preset_id}: {sorted(splits)}"
            )
        categories = {str(row["category"]) for row in members}
        if len(categories) != 1:
            raise ValueError(f"preset appears in multiple categories: {preset_id}")
        sources = sorted({str(row["source_name"]) for row in members})
        scores = {
            name: float(
                np.median(
                    [record_features[str(row["sample_id"])][name] for row in members]
                )
            )
            for name in score_names
        }
        presets.append(
            {
                "feature_version": FEATURE_VERSION,
                "feature_source": feature_source,
                "canonical_preset_id": preset_id,
                "split": next(iter(splits)),
                "category": next(iter(categories)),
                "source_names": sources,
                "record_count": len(members),
                "sample_ids": sorted(str(row["sample_id"]) for row in members),
                "proxy_scores": scores,
            }
        )
    return presets


def _bucket_counts(presets: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_split: dict[str, dict[str, int]] = {}
    for split in _SPLITS:
        members = [row for row in presets if row["split"] == split]
        by_split[split] = {
            "presets": len(members),
            "records": sum(int(row["record_count"]) for row in members),
        }
    by_category: dict[str, dict[str, int]] = {}
    categories = sorted({str(row["category"]) for row in presets})
    for category in categories:
        members = [row for row in presets if row["category"] == category]
        by_category[category] = {
            "presets": len(members),
            "records": sum(int(row["record_count"]) for row in members),
        }
    return {
        "presets": len(presets),
        "records": sum(int(row["record_count"]) for row in presets),
        "by_split": by_split,
        "by_category": by_category,
    }


def _validate_bucket_sizes(
    bucket_name: str,
    counts: Mapping[str, object],
    limits: TaxonomyLimits,
) -> None:
    failures: list[str] = []
    if int(counts["presets"]) < limits.minimum_bucket_presets:
        failures.append(
            f"presets {counts['presets']} < {limits.minimum_bucket_presets}"
        )
    if int(counts["records"]) < limits.minimum_bucket_records:
        failures.append(
            f"records {counts['records']} < {limits.minimum_bucket_records}"
        )
    by_split = counts["by_split"]
    assert isinstance(by_split, Mapping)
    split_minima = {
        "train": limits.minimum_train_presets,
        "validation": limits.minimum_validation_presets,
        "test": limits.minimum_test_presets,
    }
    for split, minimum in split_minima.items():
        split_counts = by_split[split]
        assert isinstance(split_counts, Mapping)
        if int(split_counts["presets"]) < minimum:
            failures.append(f"{split} presets {split_counts['presets']} < {minimum}")
    if failures:
        detail = "; ".join(failures)
        raise ValueError(f"taxonomy bucket {bucket_name} is too small: {detail}")


def _taxonomy(
    presets: list[dict[str, object]],
    feature_source: str,
    quantile: float,
    limits: TaxonomyLimits,
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]]]:
    if not 0.0 < quantile < 1.0 or not math.isfinite(quantile):
        raise ValueError("threshold quantile must be finite and in (0, 1)")
    train = [row for row in presets if row["split"] == "train"]
    if len(train) < 2:
        raise ValueError("taxonomy needs at least two train presets")
    thresholds: dict[str, float] = {}
    buckets: dict[str, list[dict[str, object]]] = {}
    axes = _AXES_BY_SOURCE[feature_source]
    for axis, (score_name, low_name, high_name) in axes.items():
        values = np.asarray(
            [float(row["proxy_scores"][score_name]) for row in train],
            dtype=np.float64,
        )
        threshold = float(np.quantile(values, quantile, method="linear"))
        thresholds[axis] = _finite_score(threshold, f"{axis} threshold")
        low: list[dict[str, object]] = []
        high: list[dict[str, object]] = []
        for row in presets:
            score = float(row["proxy_scores"][score_name])
            bucket = low_name if score < threshold else high_name
            row.setdefault("proxy_buckets", {})
            row["proxy_buckets"][axis] = bucket
            (low if score < threshold else high).append(row)
        for name, members in ((low_name, low), (high_name, high)):
            counts = _bucket_counts(members)
            _validate_bucket_sizes(name, counts, limits)
            buckets[name] = members
    return thresholds, buckets


def build_timbre_taxonomy(
    pack_root: str | Path,
    output_root: str | Path,
    *,
    feature_source: str = "latent",
    manifest_path: str | Path | None = None,
    threshold_quantile: float = 0.5,
    limits: TaxonomyLimits | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Build versioned, preset-level proxy taxonomy sidecars and allowlists."""

    if feature_source not in {"latent", "audio"}:
        raise ValueError("feature_source must be latent or audio")
    resolved_limits = limits or TaxonomyLimits()
    resolved_limits.validate()
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 64
    ):
        raise ValueError("taxonomy workers must be an integer in [1, 64]")
    pack = Path(pack_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else None
    )
    if feature_source == "audio" and manifest is None:
        raise ValueError("audio feature_source requires manifest_path")
    if feature_source == "latent" and manifest is not None:
        raise ValueError("manifest_path is only valid for audio feature_source")
    if not pack.is_dir():
        raise FileNotFoundError(pack)
    if output.exists():
        raise FileExistsError(f"taxonomy output already exists: {output}")
    sequences_path = pack / "sequences.jsonl"
    rows = _read_jsonl(sequences_path)
    _validate_sequences(rows)
    _validate_preset_consistency(rows)
    source_contract, shard_paths = _source_contract(pack, rows, manifest)
    if feature_source == "audio":
        _validate_shard_metadata(rows, shard_paths)
        assert manifest is not None
        manifest_paths = _manifest_audio_paths(manifest)
        sample_ids = {str(row["sample_id"]) for row in rows}
        missing = sorted(sample_ids - set(manifest_paths))
        if missing:
            raise ValueError(
                "audio manifest lacks packed samples: " + ", ".join(missing[:10])
            )
        selected_paths = {
            sample_id: manifest_paths[sample_id] for sample_id in sample_ids
        }
        source_contract["audio_files"] = _aggregate_named_file_hashes(
            selected_paths,
            workers=workers,
        )
        record_features = _record_features_from_audio(
            rows,
            selected_paths,
            workers=workers,
        )
    else:
        record_features = _record_features_from_latents(rows, shard_paths)
    presets = _aggregate_presets(rows, record_features, feature_source)
    thresholds, buckets = _taxonomy(
        presets,
        feature_source,
        threshold_quantile,
        resolved_limits,
    )
    overall_counts = _bucket_counts(presets)

    temporary = output.with_name(f".{output.name}.tmp-{secrets.token_hex(8)}")
    if temporary.exists():
        raise FileExistsError(f"taxonomy temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "buckets").mkdir()
    _write_jsonl(temporary / "presets.jsonl", presets)

    bucket_reports: dict[str, object] = {}
    for bucket_name in sorted(buckets):
        members = sorted(
            buckets[bucket_name], key=lambda row: str(row["canonical_preset_id"])
        )
        counts = _bucket_counts(members)
        axis = next(
            name
            for name, (_score, low, high) in _AXES_BY_SOURCE[feature_source].items()
            if bucket_name in {low, high}
        )
        _score_name, low_name, _high_name = _AXES_BY_SOURCE[feature_source][axis]
        comparison = "<" if bucket_name == low_name else ">="
        ids_by_split = {
            split: [
                str(row["canonical_preset_id"])
                for row in members
                if row["split"] == split
            ]
            for split in _SPLITS
        }
        payload = {
            "schema": 1,
            "feature_version": FEATURE_VERSION,
            "proxy_disclaimer": DISCLAIMER,
            "feature_source": feature_source,
            "bucket": bucket_name,
            "axis": axis,
            "score_name": _score_name,
            "proxy_definition": _AXIS_DESCRIPTIONS[feature_source][axis],
            "threshold": thresholds[axis],
            "comparison": comparison,
            "counts": counts,
            "preset_ids": [str(row["canonical_preset_id"]) for row in members],
            "preset_ids_by_split": ids_by_split,
            "source_hashes": source_contract,
        }
        json_relative = f"buckets/{bucket_name}.json"
        jsonl_relative = f"buckets/{bucket_name}.jsonl"
        ids_relative = f"buckets/{bucket_name}.ids.txt"
        _write_json(temporary / json_relative, payload)
        _write_jsonl(
            temporary / jsonl_relative,
            (
                {
                    "schema": 1,
                    "kind": "zrave-preset-timbre-proxy-allowlist-row",
                    "feature_version": FEATURE_VERSION,
                    "feature_source": feature_source,
                    "bucket": bucket_name,
                    "axis": axis,
                    "score_name": _score_name,
                    "threshold": thresholds[axis],
                    "comparison": comparison,
                    "canonical_preset_id": row["canonical_preset_id"],
                    "split": row["split"],
                    "category": row["category"],
                    "record_count": row["record_count"],
                    "proxy_score": row["proxy_scores"][_score_name],
                }
                for row in members
            ),
        )
        _write_ids(temporary / ids_relative, payload["preset_ids"])
        split_paths: dict[str, str] = {}
        for split in _SPLITS:
            relative = f"buckets/{bucket_name}.{split}.ids.txt"
            _write_ids(temporary / relative, ids_by_split[split])
            split_paths[split] = relative
        bucket_reports[bucket_name] = {
            "axis": axis,
            "score_name": _score_name,
            "proxy_definition": _AXIS_DESCRIPTIONS[feature_source][axis],
            "threshold": thresholds[axis],
            "comparison": comparison,
            "counts": counts,
            "json": json_relative,
            "jsonl": jsonl_relative,
            "ids": ids_relative,
            "ids_by_split": split_paths,
        }

    report: dict[str, object] = {
        "schema": 1,
        "kind": "zrave-preset-timbre-proxy-taxonomy",
        "feature_version": FEATURE_VERSION,
        "proxy_disclaimer": DISCLAIMER,
        "feature_source": feature_source,
        "tool": {
            "module_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "axes": {
            axis: {
                "score_name": score_name,
                "low_bucket": low_name,
                "high_bucket": high_name,
                "proxy_definition": _AXIS_DESCRIPTIONS[feature_source][axis],
            }
            for axis, (score_name, low_name, high_name) in _AXES_BY_SOURCE[
                feature_source
            ].items()
        },
        "aggregation": "median_across_all_retained_renders_per_preset",
        "threshold_fit": {
            "split": "train",
            "quantile": threshold_quantile,
            "thresholds": thresholds,
        },
        "limits": resolved_limits.__dict__,
        "source_hashes": source_contract,
        "counts": overall_counts,
        "presets_jsonl": "presets.jsonl",
        "buckets": bucket_reports,
    }
    _write_json(temporary / "taxonomy.report.json", report)
    temporary.replace(output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build versioned preset-level Z-RAVE timbre proxy sidecars. "
            "The buckets are relative engineering proxies, not perceptual labels."
        )
    )
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--feature-source",
        choices=("latent", "audio"),
        default="latent",
    )
    parser.add_argument(
        "--manifest",
        help="Required for --feature-source audio; maps sample_id to audio_path.",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.5)
    parser.add_argument("--minimum-bucket-presets", type=int, default=20)
    parser.add_argument("--minimum-bucket-records", type=int, default=100)
    parser.add_argument("--minimum-train-presets", type=int, default=10)
    parser.add_argument("--minimum-validation-presets", type=int, default=1)
    parser.add_argument("--minimum-test-presets", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel audio read/hash workers; output remains deterministic.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_timbre_taxonomy(
        args.pack_root,
        args.output_root,
        feature_source=args.feature_source,
        manifest_path=args.manifest,
        threshold_quantile=args.threshold_quantile,
        workers=args.workers,
        limits=TaxonomyLimits(
            minimum_bucket_presets=args.minimum_bucket_presets,
            minimum_bucket_records=args.minimum_bucket_records,
            minimum_train_presets=args.minimum_train_presets,
            minimum_validation_presets=args.minimum_validation_presets,
            minimum_test_presets=args.minimum_test_presets,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
