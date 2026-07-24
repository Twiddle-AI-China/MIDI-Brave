from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from midibrave.zrave_flow_config import ZraveFlowConfig
from midibrave.zrave_flow_manifest import (
    build_flow_manifest,
    preset_split,
    read_flow_manifest,
)


ROOT = Path(__file__).parents[1]
FORMAL_CONFIG = ROOT / "configs" / "zrave" / "octopus_flow.yaml"
FIELDS = {
    "sample_id",
    "source_name",
    "audio_path",
    "canonical_preset_id",
    "split",
    "midi_note",
    "velocity",
    "articulation_id",
    "category",
    "duration_seconds",
    "maximum_future_frames",
}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _touch_audio(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE data_management (
            data_item_id TEXT PRIMARY KEY,
            dataset_id TEXT,
            source_id TEXT,
            audio_path TEXT,
            source_meta_json TEXT,
            duration_sec REAL,
            status TEXT,
            is_duplicate INTEGER,
            is_silent INTEGER
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE auxiliary_info (
            data_item_id TEXT PRIMARY KEY,
            pitch_midi INTEGER,
            velocity INTEGER
        )
        """
    )
    connection.executemany(
        "INSERT INTO data_management VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "registry-17",
                "serum-octopus-v1",
                "xfer/serum:17",
                "registry/17.wav",
                json.dumps({"category": "Pads"}),
                5.0,
                "active",
                0,
                0,
            ),
            (
                "registry-18",
                "serum-octopus-v1",
                "xfer/serum:18",
                "registry/18.wav",
                json.dumps({"category": "Leads"}),
                5.0,
                "active",
                0,
                0,
            ),
            (
                "inactive",
                "serum-octopus-v1",
                "xfer/serum:20",
                "registry/inactive.wav",
                json.dumps({"category": "Pad"}),
                5.0,
                "inactive",
                0,
                0,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO auxiliary_info VALUES (?, ?, ?)",
        [
            ("registry-17", 60, 100),
            ("registry-18", 72, 100),
            ("inactive", 60, 100),
        ],
    )
    connection.commit()
    connection.close()


def _fixture_config(tmp_path: Path) -> ZraveFlowConfig:
    config = ZraveFlowConfig.load(FORMAL_CONFIG)
    roots = [tmp_path / f"source-{index}" for index in range(4)]
    registry = tmp_path / "registry" / "samples.db"
    dense_dexed = tmp_path / "dexed.jsonl"
    dense_serum = tmp_path / "serum.jsonl"
    broad = tmp_path / "broad.jsonl"
    serum_presets = tmp_path / "serum-presets.jsonl"
    _write_registry(registry)
    for relative in ("registry/17.wav", "registry/18.wav"):
        _touch_audio(roots[0], relative)
    _write_jsonl(
        dense_dexed,
        [
            {
                "sample_id": "dexed_p0028_n060_v100",
                "audio_path": "audio/dexed.wav",
                "preset_id": "dexed_p0028",
                "midi_note": 60,
                "velocity": 100,
                "articulation_id": "steady",
                "duration_seconds": 5.2,
            }
        ],
    )
    _touch_audio(roots[1], "audio/dexed.wav")
    _write_jsonl(
        dense_serum,
        [
            {
                "sample_id": "serum_s000017_n048_v100",
                "audio_path": "audio/serum17.wav",
                "preset_id": "serum_s000017",
                "midi_note": 48,
                "velocity": 100,
                "articulation_id": "steady",
                "duration_seconds": 5.0,
            },
            {
                "sample_id": "serum_s000019_n060_v100",
                "audio_path": "audio/serum19.wav",
                "preset_id": "serum_s000019",
                "midi_note": 60,
                "velocity": 100,
                "articulation_id": "steady",
                "duration_seconds": 5.0,
            },
        ],
    )
    _write_jsonl(
        serum_presets,
        [
            {"preset_id": "serum_s000017", "category": "Pads"},
            {"preset_id": "serum_s000019", "category": "Lead"},
        ],
    )
    _touch_audio(roots[2], "audio/serum17.wav")
    _touch_audio(roots[2], "audio/serum19.wav")
    _write_jsonl(
        broad,
        [
            {
                "preset_index": 28,
                "canonical_synth_id": "dexed/dexed",
                "midi_note": 60,
                "velocity": 54,
                "wav_path": "dexed/000028/note060_vel054.wav",
                "duration_sec": 5.0,
            },
            {
                "preset_index": 9,
                "canonical_synth_id": "surge-xt/surge-xt",
                "midi_note": 72,
                "velocity": 80,
                "wav_path": "surge-xt/000009/note072_vel080.wav",
                "duration_sec": 5.0,
            },
        ],
    )
    _touch_audio(roots[3], "dexed/000028/note060_vel054.wav")
    _touch_audio(roots[3], "surge-xt/000009/note072_vel080.wav")

    manifests = (registry, dense_dexed, dense_serum, broad)
    sources = tuple(
        replace(
            source,
            manifest=str(manifests[index]),
            audio_root=str(roots[index]),
            preset_metadata=(
                str(serum_presets) if source.name == "serum_dense" else None
            ),
        )
        for index, source in enumerate(config.data.sources)
    )
    data = replace(
        config.data,
        unified_manifest=str(tmp_path / "all.jsonl"),
        manifest_report=str(tmp_path / "report.json"),
        cache_root=str(tmp_path / "cache"),
        packed_root=str(tmp_path / "pack"),
        sources=sources,
    )
    return replace(config, data=data)


def test_manifest_merges_aliases_and_keeps_presets_in_one_split(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)

    report = build_flow_manifest(config)
    rows = read_flow_manifest(config.data.unified_manifest)

    assert report.rows == len(rows) == 7
    assert all(set(row) == FIELDS for row in rows)
    assert {"Pad", "Lead", "Dexed", "Surge"} == {
        row["category"] for row in rows
    }
    splits_by_preset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_preset[str(row["canonical_preset_id"])].add(
            str(row["split"])
        )
    assert all(len(splits) == 1 for splits in splits_by_preset.values())
    assert {
        row["split"]
        for row in rows
        if row["canonical_preset_id"] == "serum:000017"
    } == {preset_split("serum:000017", config.seed)}
    assert {
        row["canonical_preset_id"]
        for row in rows
        if row["source_name"] == "dexed_surge_broad"
    } == {"dexed:0028", "surge:000009"}
    assert all(str(row["sample_id"]).startswith(f"{row['source_name']}__")
               for row in rows)


def test_manifest_opens_registry_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path)
    registry = Path(config.data.sources[0].manifest)
    original_connect = sqlite3.connect
    calls: list[tuple[str, bool]] = []

    def recording_connect(
        database: str,
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        calls.append((str(database), bool(kwargs.get("uri"))))
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    build_flow_manifest(config)

    assert calls == [(f"file:{registry}?mode=ro", True)]
    assert not (registry.parent / f"{registry.name}-wal").exists()


def test_manifest_rejects_audio_path_escape(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    source = config.data.sources[1]
    _write_jsonl(
        Path(source.manifest),
        [
            {
                "sample_id": "escape",
                "audio_path": "../outside.wav",
                "preset_id": "dexed_p0001",
                "midi_note": 60,
                "velocity": 100,
                "duration_seconds": 5.0,
            }
        ],
    )
    (Path(source.audio_root).parent / "outside.wav").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="escapes audio_root"):
        build_flow_manifest(config)
