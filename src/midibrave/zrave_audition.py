from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np


def _rows_by_sample_id(
    rows: Iterable[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("audition manifest row is missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate audition sample_id: {sample_id}")
        result[sample_id] = row
    return result


def select_audition_rows(
    index: dict[str, object],
    manifest_rows: Iterable[dict[str, object]],
    categories: Iterable[str],
) -> list[dict[str, object]]:
    sequences = index.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("packed index must contain a sequences list")
    manifest_by_id = _rows_by_sample_id(manifest_rows)
    category_order = tuple(categories)
    if not category_order:
        raise ValueError("audition categories must not be empty")

    groups: list[dict[tuple[int, int], list[dict[str, object]]]] = []
    labels: list[tuple[str, str]] = []
    for category in category_order:
        for split in ("validation", "test"):
            matching = [
                sequence
                for sequence in sequences
                if isinstance(sequence, dict)
                and sequence.get("category") == category
                and sequence.get("split") == split
            ]
            if not matching:
                raise ValueError(
                    f"missing held-out audition data for {category}/{split}"
                )
            preset_id = min(str(row["preset_id"]) for row in matching)
            conditions: dict[
                tuple[int, int], list[dict[str, object]]
            ] = {}
            for sequence in matching:
                if str(sequence["preset_id"]) != preset_id:
                    continue
                sample_id = str(sequence["sample_id"])
                manifest_row = manifest_by_id.get(sample_id)
                if manifest_row is None:
                    raise ValueError(
                        f"packed sample is absent from manifest: {sample_id}"
                    )
                condition = (
                    int(manifest_row["midi_note"]),
                    int(manifest_row["velocity"]),
                )
                joined = dict(manifest_row)
                joined.update(
                    {
                        "category": category,
                        "split": split,
                        "preset_id": preset_id,
                        "sequence_index": int(sequence["index"]),
                        "length": int(sequence["length"]),
                    }
                )
                conditions.setdefault(condition, []).append(joined)
            if not conditions:
                raise ValueError(
                    f"preset has no audition conditions: {category}/{split}"
                )
            groups.append(conditions)
            labels.append((category, split))

    common = set(groups[0])
    for group in groups[1:]:
        common.intersection_update(group)
    if not common:
        raise ValueError("held-out presets have no shared MIDI condition")
    chosen_condition = min(
        common,
        key=lambda value: (
            -value[1],
            abs(value[0] - 60),
            value[0],
        ),
    )

    selected: list[dict[str, object]] = []
    for (category, split), group in zip(labels, groups, strict=True):
        candidates = sorted(
            group[chosen_condition],
            key=lambda row: str(row["sample_id"]),
        )
        row = dict(candidates[0])
        row["category"] = category
        row["split"] = split
        selected.append(row)
    return selected


def deterministic_start(
    sample_id: str,
    length: int,
    required_frames: int,
    seed: int,
) -> int:
    if required_frames <= 0:
        raise ValueError("required frames must be positive")
    if length < required_frames:
        raise ValueError(
            f"latent sequence is shorter than {required_frames}: "
            f"{sample_id} has {length}"
        )
    payload = f"{seed}:zrave-audition:{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (length - required_frames + 1)


def _mono_float32(value: np.ndarray, name: str) -> np.ndarray:
    clip = np.asarray(value, dtype=np.float32)
    if clip.ndim != 1 or clip.size == 0:
        raise ValueError(f"{name} audio must be a non-empty mono array")
    if not np.isfinite(clip).all():
        raise ValueError(f"{name} audio contains non-finite values")
    return np.ascontiguousarray(clip)


def shared_peak_gain(
    *clips: np.ndarray,
    target_peak: float = 0.95,
) -> float:
    if not clips:
        raise ValueError("at least one audio clip is required")
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target peak must be in (0, 1]")
    peak = max(float(np.max(np.abs(clip))) for clip in clips)
    if peak < 1.0e-8:
        raise ValueError("audition audio is silent")
    return target_peak / peak


def prepare_triplet(
    source: np.ndarray,
    direct: np.ndarray,
    predicted: np.ndarray,
    target_peak: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    clips = (
        _mono_float32(source, "source"),
        _mono_float32(direct, "direct"),
        _mono_float32(predicted, "predicted"),
    )
    if len({clip.shape[0] for clip in clips}) != 1:
        raise ValueError("audition clips must have identical sample counts")
    gain = shared_peak_gain(*clips, target_peak=target_peak)
    rendered = tuple(np.ascontiguousarray(clip * gain) for clip in clips)
    if any(float(np.max(np.abs(clip))) > 1.0 for clip in rendered):
        raise ValueError("normalized audition audio exceeds [-1, 1]")
    return rendered[0], rendered[1], rendered[2], gain
