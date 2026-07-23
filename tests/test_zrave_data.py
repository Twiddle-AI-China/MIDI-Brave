from __future__ import annotations

import json
from pathlib import Path

from midibrave.zrave_config import (
    ZraveConfig,
    ZraveDataConfig,
    ZraveLossConfig,
    ZraveModelConfig,
    ZraveOptimizerConfig,
    ZraveRaveConfig,
    ZraveTrainConfig,
)
from midibrave.zrave_data import read_jsonl, select_balanced_manifest


CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _fixture_config(tmp_path: Path) -> ZraveConfig:
    preset_metadata = tmp_path / "presets.jsonl"
    eligible_manifest = tmp_path / "eligible.jsonl"
    presets: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for category in CATEGORIES:
        for preset_index in range(12):
            preset_id = f"{category.lower()}-{preset_index:02d}"
            presets.append(
                {
                    "preset_id": preset_id,
                    "preset_name": preset_id,
                    "category": category,
                    "bank": f"bank-{preset_index % 6}",
                }
            )
            for note in range(36, 72):
                for velocity in (64, 127):
                    sample_id = f"{preset_id}-n{note}-v{velocity}"
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "preset_id": preset_id,
                            "midi_note": note,
                            "velocity": velocity,
                            "audio_path": f"work/{sample_id}.wav",
                            "duration_seconds": 5.0,
                            "split": "source",
                        }
                    )
    # Eligible but incomplete: it must never displace a complete preset.
    presets.append(
        {
            "preset_id": "pad-incomplete",
            "category": "Pad",
            "bank": "bank-extra",
        }
    )
    samples.append(
        {
            "sample_id": "pad-incomplete-only",
            "preset_id": "pad-incomplete",
            "midi_note": 60,
            "velocity": 127,
            "audio_path": "work/incomplete.wav",
            "duration_seconds": 5.0,
        }
    )
    _write_jsonl(preset_metadata, presets)
    _write_jsonl(eligible_manifest, samples)
    return ZraveConfig(
        seed=20260723,
        data=ZraveDataConfig(
            preset_metadata=str(preset_metadata),
            eligible_manifest=str(eligible_manifest),
            audio_root=str(tmp_path),
            selected_manifest=str(tmp_path / "selected.jsonl"),
            metadata_output=str(tmp_path / "selected.meta.json"),
            cache_root=str(tmp_path / "cache"),
            packed_root=str(tmp_path / "pack"),
        ),
        rave=ZraveRaveConfig(
            source_config=str(tmp_path / "source.yaml"),
            checkpoint=str(tmp_path / "checkpoint.pt"),
        ),
        model=ZraveModelConfig(),
        loss=ZraveLossConfig(),
        optimizer=ZraveOptimizerConfig(),
        train=ZraveTrainConfig(output_root=str(tmp_path / "run")),
    )


def test_balanced_selection_is_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)

    first = select_balanced_manifest(config)
    first_bytes = Path(config.data.selected_manifest).read_bytes()
    second = select_balanced_manifest(config)

    assert first == second
    assert Path(config.data.selected_manifest).read_bytes() == first_bytes
    assert first["presets"] == 50
    assert first["samples"] == 3600
    assert first["categories"] == {
        "Bass": 10,
        "Keys": 10,
        "Lead": 10,
        "Pad": 10,
        "Pluck": 10,
    }
    assert first["splits"] == {"train": 40, "validation": 5, "test": 5}

    rows = read_jsonl(Path(config.data.selected_manifest))
    preset_splits = {
        (str(row["preset_id"]), str(row["split"]))
        for row in rows
    }
    assert len(preset_splits) == 50
    assert all(
        sum(row["preset_id"] == preset_id for row in rows) == 72
        for preset_id, _ in preset_splits
    )
    assert all(row["split"] != "source" for row in rows)
    assert not any(row["preset_id"] == "pad-incomplete" for row in rows)


def test_balanced_selection_prefers_distinct_banks(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)

    select_balanced_manifest(config)
    metadata = json.loads(
        Path(config.data.metadata_output).read_text(encoding="utf-8")
    )

    for category in CATEGORIES:
        selected = metadata["selected_presets"][category]
        assert len({item["bank"] for item in selected[:6]}) == 6
