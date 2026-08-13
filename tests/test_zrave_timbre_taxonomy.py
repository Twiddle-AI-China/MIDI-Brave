from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from midibrave.zrave_timbre_taxonomy import (
    FEATURE_VERSION,
    TaxonomyLimits,
    build_timbre_taxonomy,
)

_PERMISSIVE = TaxonomyLimits(
    minimum_bucket_presets=0,
    minimum_bucket_records=0,
    minimum_train_presets=0,
    minimum_validation_presets=0,
    minimum_test_presets=0,
)
_LATENT_BUCKETS = {
    "latent_slow_proxy",
    "latent_fast_proxy",
    "latent_soft_onset_proxy",
    "latent_hard_onset_proxy",
    "latent_static_proxy",
    "latent_moving_proxy",
    "latent_smooth_proxy",
    "latent_irregular_proxy",
}
_AUDIO_BUCKETS = {
    "audio_dark_proxy",
    "audio_bright_proxy",
    "audio_soft_attack_proxy",
    "audio_hard_attack_proxy",
    "audio_static_proxy",
    "audio_moving_proxy",
    "audio_clean_proxy",
    "audio_noisy_proxy",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latent(code: float, *, identical: bool = False) -> np.ndarray:
    if identical:
        code = 2.0
    frames = 40
    time = np.arange(frames, dtype=np.float64)
    frequency = 1.0 + code
    phase = 2.0 * np.pi * frequency * time / frames
    rise = 1.0 - np.exp(-time / (1.5 + 0.35 * code))
    rng = np.random.default_rng(round(code * 1000) + 17)
    irregular = rng.normal(size=frames) * (0.005 + 0.008 * code)
    value = np.stack(
        (
            (0.08 + rise) * np.sin(phase),
            (0.08 + rise) * np.cos(phase),
            rise * (time / frames) * (0.2 + 0.03 * code),
            irregular,
        ),
        axis=1,
    )
    return value.astype(np.float16)


def _write_pack(
    root: Path,
    *,
    heldout_offset: float = 0.0,
    identical_train: bool = False,
    renders: int = 2,
) -> list[dict[str, object]]:
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    latents: list[np.ndarray] = []
    split_codes: list[int] = []
    preset_index = 0
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        for _ in range(count):
            preset_id = f"serum:{preset_index:06d}"
            base_code = float(preset_index)
            if split != "train":
                base_code += heldout_offset
            for render in range(renders):
                row_index = len(rows)
                code = base_code + 0.025 * render
                latents.append(
                    _latent(
                        code,
                        identical=identical_train and split == "train",
                    )
                )
                rows.append(
                    {
                        "manifest_index": row_index,
                        "sample_id": f"sample-{preset_index}-{render}",
                        "source_name": "serum_full",
                        "source_code": 0,
                        "category": "Pad",
                        "category_code": 0,
                        "canonical_preset_id": preset_id,
                        "split": split,
                        "split_code": {
                            "train": 0,
                            "validation": 1,
                            "test": 2,
                        }[split],
                        "midi_note": 36 + render,
                        "velocity": 100,
                        "maximum_future_frames": 32,
                        "length": 40,
                        "active_frames": 40,
                        "packed_index": row_index,
                        "shard": "shard-000000.npz",
                        "shard_row": row_index,
                    }
                )
                split_codes.append({"train": 0, "validation": 1, "test": 2}[split])
            preset_index += 1
    shard = root / "shard-000000.npz"
    np.savez(
        shard,
        latents=np.stack(latents),
        lengths=np.full(len(rows), 40, dtype=np.int16),
        active_frames=np.full(len(rows), 40, dtype=np.int16),
        notes=np.asarray([row["midi_note"] for row in rows], dtype=np.int16),
        velocities=np.full(len(rows), 100, dtype=np.int16),
        split_codes=np.asarray(split_codes, dtype=np.uint8),
        source_codes=np.zeros(len(rows), dtype=np.uint8),
        category_codes=np.zeros(len(rows), dtype=np.uint16),
        maximum_future_frames=np.full(len(rows), 32, dtype=np.uint8),
        manifest_indices=np.arange(len(rows), dtype=np.int64),
    )
    (root / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "records": len(rows),
                "shard_sha256": {shard.name: _sha256(shard)},
            }
        ),
        encoding="utf-8",
    )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_audio_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    audio_root = root / "audio"
    audio_root.mkdir()
    manifest = root / "audio-manifest.jsonl"
    manifest_rows = []
    sample_rate = 8000
    time = np.arange(2400, dtype=np.float64) / sample_rate
    for index, row in enumerate(rows):
        frequency = 100.0 + 17.0 * index
        envelope = np.minimum(1.0, time * (8.0 + index))
        audio = 0.25 * envelope * np.sin(2.0 * np.pi * frequency * time)
        audio_path = audio_root / f"{row['sample_id']}.wav"
        sf.write(audio_path, audio.astype(np.float32), sample_rate, subtype="FLOAT")
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "audio_path": str(audio_path.relative_to(root)),
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return manifest


def test_latent_taxonomy_writes_versioned_preset_allowlists(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    rows = _write_pack(pack)
    output = tmp_path / "taxonomy-a"
    report = build_timbre_taxonomy(pack, output, limits=_PERMISSIVE)

    assert report["feature_version"] == FEATURE_VERSION
    assert set(report["buckets"]) == _LATENT_BUCKETS
    assert not (set(report["buckets"]) & _AUDIO_BUCKETS)
    latent_report_text = (output / "taxonomy.report.json").read_text()
    assert "brightness_proxy" not in latent_report_text
    assert "darkness_proxy" not in latent_report_text
    assert "noisiness_proxy" not in latent_report_text
    assert report["threshold_fit"]["split"] == "train"
    assert report["counts"]["presets"] == 16
    assert report["counts"]["records"] == len(rows)

    preset_rows = _read_jsonl(output / "presets.jsonl")
    assert len(preset_rows) == 16
    assert {row["record_count"] for row in preset_rows} == {2}
    all_ids = {row["canonical_preset_id"] for row in preset_rows}
    axes = {
        "temporal_rate": ("latent_slow_proxy", "latent_fast_proxy"),
        "onset": ("latent_soft_onset_proxy", "latent_hard_onset_proxy"),
        "motion": ("latent_static_proxy", "latent_moving_proxy"),
        "irregularity": ("latent_smooth_proxy", "latent_irregular_proxy"),
    }
    for axis, pair in axes.items():
        low = {
            row["canonical_preset_id"]
            for row in _read_jsonl(output / "buckets" / f"{pair[0]}.jsonl")
        }
        high = {
            row["canonical_preset_id"]
            for row in _read_jsonl(output / "buckets" / f"{pair[1]}.jsonl")
        }
        assert low.isdisjoint(high)
        assert low | high == all_ids
        for bucket in pair:
            bucket_report = report["buckets"][bucket]
            assert bucket_report["axis"] == axis
            assert (output / bucket_report["ids"]).is_file()
            assert (output / bucket_report["json"]).is_file()
            assert (output / bucket_report["jsonl"]).is_file()

    second_output = tmp_path / "taxonomy-b"
    second_report = build_timbre_taxonomy(
        pack,
        second_output,
        limits=_PERMISSIVE,
    )
    assert second_report["source_hashes"] == report["source_hashes"]
    assert (second_output / "taxonomy.report.json").read_bytes() == (
        output / "taxonomy.report.json"
    ).read_bytes()


def test_audio_mode_uses_distinct_audio_proxy_names_and_hashes(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    rows = _write_pack(pack, renders=1)
    manifest = _write_audio_manifest(tmp_path, rows)
    report = build_timbre_taxonomy(
        pack,
        tmp_path / "audio-taxonomy",
        feature_source="audio",
        manifest_path=manifest,
        limits=_PERMISSIVE,
        workers=3,
    )

    assert set(report["buckets"]) == _AUDIO_BUCKETS
    assert not (set(report["buckets"]) & _LATENT_BUCKETS)
    audio_hashes = report["source_hashes"]["audio_files"]
    assert audio_hashes["count"] == len(rows)
    assert len(audio_hashes["sample_id_content_sha256"]) == 64


def test_audio_taxonomy_parallel_workers_are_deterministic(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    rows = _write_pack(pack, renders=1)
    manifest = _write_audio_manifest(tmp_path, rows)

    serial = build_timbre_taxonomy(
        pack,
        tmp_path / "serial",
        feature_source="audio",
        manifest_path=manifest,
        limits=_PERMISSIVE,
        workers=1,
    )
    parallel = build_timbre_taxonomy(
        pack,
        tmp_path / "parallel",
        feature_source="audio",
        manifest_path=manifest,
        limits=_PERMISSIVE,
        workers=4,
    )

    assert parallel == serial
    assert (tmp_path / "parallel/taxonomy.report.json").read_bytes() == (
        tmp_path / "serial/taxonomy.report.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="workers"):
        build_timbre_taxonomy(pack, tmp_path / "bad", workers=0)


def test_thresholds_do_not_depend_on_validation_or_test(tmp_path: Path) -> None:
    first_pack = tmp_path / "first-pack"
    second_pack = tmp_path / "second-pack"
    _write_pack(first_pack, heldout_offset=0.0, renders=1)
    _write_pack(second_pack, heldout_offset=101.0, renders=1)

    first = build_timbre_taxonomy(
        first_pack,
        tmp_path / "first-out",
        limits=_PERMISSIVE,
    )
    second = build_timbre_taxonomy(
        second_pack,
        tmp_path / "second-out",
        limits=_PERMISSIVE,
    )
    assert first["threshold_fit"]["thresholds"] == second["threshold_fit"]["thresholds"]
    assert first["source_hashes"]["shards"] != second["source_hashes"]["shards"]


def test_rejects_canonical_preset_split_leakage(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_pack(pack)
    sequences = _read_jsonl(pack / "sequences.jsonl")
    sequences[1]["split"] = "validation"
    (pack / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sequences),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preset appears in multiple splits"):
        build_timbre_taxonomy(
            pack,
            tmp_path / "taxonomy",
            limits=_PERMISSIVE,
        )


def test_rejects_bucket_below_configured_train_floor(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_pack(pack, identical_train=True, renders=1)
    limits = TaxonomyLimits(
        minimum_bucket_presets=0,
        minimum_bucket_records=0,
        minimum_train_presets=1,
        minimum_validation_presets=0,
        minimum_test_presets=0,
    )

    with pytest.raises(ValueError, match=r"too small: train presets 0 < 1"):
        build_timbre_taxonomy(pack, tmp_path / "taxonomy", limits=limits)


def test_rejects_sequences_that_do_not_match_hashed_shard_metadata(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    _write_pack(pack, renders=1)
    sequences = _read_jsonl(pack / "sequences.jsonl")
    sequences[0]["midi_note"] = 99
    (pack / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sequences),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata does not match shard"):
        build_timbre_taxonomy(
            pack,
            tmp_path / "taxonomy",
            limits=_PERMISSIVE,
        )
