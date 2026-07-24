from __future__ import annotations

import numpy as np
import pytest

from midibrave.zrave_audition import (
    deterministic_start,
    prepare_triplet,
    select_audition_rows,
)


CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")


def _audition_fixtures() -> tuple[dict[str, object], list[dict[str, object]]]:
    sequences: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    sequence_index = 0
    conditions = ((48, 64), (60, 127), (72, 127))
    for category in CATEGORIES:
        for split in ("validation", "test"):
            preset_id = f"{category.casefold()}-{split}"
            for note, velocity in conditions:
                sample_id = f"{preset_id}-n{note}-v{velocity}"
                sequences.append(
                    {
                        "index": sequence_index,
                        "sample_id": sample_id,
                        "preset_id": preset_id,
                        "category": category,
                        "split": split,
                        "length": 1658,
                    }
                )
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "preset_id": preset_id,
                        "audio_path": f"audio/{sample_id}.wav",
                        "midi_note": note,
                        "velocity": velocity,
                        "split": split,
                        "zrave_category": category,
                    }
                )
                sequence_index += 1
    return {"schema": 1, "sequences": sequences}, manifest


def test_selection_returns_validation_and_test_for_every_class() -> None:
    index, manifest = _audition_fixtures()

    selected = select_audition_rows(index, manifest, CATEGORIES)

    assert [(row["category"], row["split"]) for row in selected] == [
        (category, split)
        for category in CATEGORIES
        for split in ("validation", "test")
    ]
    assert {
        (row["midi_note"], row["velocity"]) for row in selected
    } == {(60, 127)}


def test_deterministic_start_stays_inside_sequence() -> None:
    first = deterministic_start("sample-a", 1658, 990, 20260724)

    assert first == deterministic_start("sample-a", 1658, 990, 20260724)
    assert 0 <= first <= 1658 - 990
    with pytest.raises(ValueError, match="shorter"):
        deterministic_start("sample-a", 989, 990, 20260724)


def test_triplet_uses_one_gain_and_preserves_relative_levels() -> None:
    source = np.full(16, 0.25, np.float32)
    direct = np.full(16, 0.50, np.float32)
    predicted = np.full(16, -1.00, np.float32)

    rendered = prepare_triplet(source, direct, predicted)

    assert rendered[3] == pytest.approx(0.95)
    assert np.max(np.abs(rendered[0])) == pytest.approx(0.2375)
    assert np.max(np.abs(rendered[1])) == pytest.approx(0.475)
    assert np.max(np.abs(rendered[2])) == pytest.approx(0.95)


def test_triplet_rejects_misaligned_or_invalid_audio() -> None:
    valid = np.ones(16, np.float32)

    with pytest.raises(ValueError, match="identical"):
        prepare_triplet(valid, valid[:-1], valid)
    with pytest.raises(ValueError, match="non-finite"):
        prepare_triplet(valid, np.full(16, np.nan, np.float32), valid)
    with pytest.raises(ValueError, match="silent"):
        prepare_triplet(np.zeros(16), np.zeros(16), np.zeros(16))
