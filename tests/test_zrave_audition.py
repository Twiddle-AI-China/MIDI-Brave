from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from midibrave.zrave_audition import (
    crop_decoded_future,
    crop_source_future,
    deterministic_start,
    duration_to_frames,
    prepare_triplet,
    rollout_latents,
    select_audition_rows,
    select_random_seed_rows,
    validate_audition_manifest,
)


CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")
RENDER_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "render_zrave_audition.py"
)
SBATCH_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "cloud"
    / "octopus_zrave_audition.sbatch"
)


class _CountingModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(context_frames=128)
        self.calls = 0

    def __call__(self, history: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        value = torch.full(
            (history.shape[0], 16, history.shape[2]),
            float(self.calls),
            device=history.device,
        )
        return SimpleNamespace(latent=value)


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


def test_rollout_consumes_all_sixteen_frame_chunks() -> None:
    history = torch.zeros(1, 128, 16)
    model = _CountingModel()

    predicted = rollout_latents(model, history, 35)

    assert predicted.shape == (1, 35, 16)
    assert model.calls == 3
    torch.testing.assert_close(predicted[:, :16], torch.ones(1, 16, 16))
    torch.testing.assert_close(
        predicted[:, 16:32],
        torch.full((1, 16, 16), 2.0),
    )
    torch.testing.assert_close(
        predicted[:, 32:],
        torch.full((1, 3, 16), 3.0),
    )


def test_future_crop_removes_exact_context() -> None:
    decoded = np.arange((128 + 32) * 128, dtype=np.float32)

    cropped = crop_decoded_future(
        decoded,
        context_frames=128,
        future_frames=32,
        latent_hop=128,
    )

    np.testing.assert_array_equal(
        cropped,
        decoded[128 * 128 : 160 * 128],
    )


def test_source_crop_includes_pack_warmup_and_window_start() -> None:
    audio = np.arange(400 * 128, dtype=np.float32)

    cropped = crop_source_future(
        audio,
        warmup_frames=64,
        window_start=5,
        context_frames=128,
        future_frames=32,
        latent_hop=128,
    )

    expected_start = (64 + 5 + 128) * 128
    np.testing.assert_array_equal(
        cropped,
        audio[expected_start : expected_start + 32 * 128],
    )


def test_duration_rounds_up_to_complete_latent_frames() -> None:
    assert duration_to_frames(2.5, 44100, 128) == 862


def test_random_seed_rows_are_distinct_and_reproducible() -> None:
    index, manifest = _audition_fixtures()

    first = select_random_seed_rows(
        index,
        manifest,
        count=2,
        required_frames=990,
        seed=20260724,
    )
    second = select_random_seed_rows(
        index,
        manifest,
        count=2,
        required_frames=990,
        seed=20260724,
    )

    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in second
    ]
    assert len({row["sample_id"] for row in first}) == 2
    assert {row["split"] for row in first} <= {"validation", "test"}


def test_manifest_contract_requires_all_five_classes_and_two_seeds() -> None:
    comparisons = [
        {
            "category": category,
            "split": split,
            "sample_count": 110336,
            "audio": {
                "original": "audio/original.wav",
                "direct": "audio/direct.wav",
                "predicted": "audio/predicted.wav",
            },
        }
        for category in CATEGORIES
        for split in ("validation", "test")
    ]
    manifest = {
        "schema": 1,
        "sample_rate": 44100,
        "context_frames": 128,
        "comparisons": comparisons,
        "random_continuations": [
            {
                "file": f"audio/random-{index}.wav",
                "seed_frames": 128,
                "transition_seconds": 128 * 128 / 44100,
            }
            for index in range(2)
        ],
    }

    validate_audition_manifest(manifest, CATEGORIES)

    manifest["random_continuations"].pop()
    with pytest.raises(ValueError, match="two random"):
        validate_audition_manifest(manifest, CATEGORIES)


def test_render_script_exposes_reproducible_cli_contract() -> None:
    script = RENDER_SCRIPT.read_text(encoding="utf-8")

    for option in (
        "--config",
        "--checkpoint",
        "--output",
        "--duration",
        "--random-seed",
        "--device",
    ):
        assert option in script
    assert "render_audition" in script


def test_octopus_audition_uses_exactly_one_slurm_gpu() -> None:
    script = SBATCH_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --gres=gpu:2" not in script
    assert "scripts/render_zrave_audition.py" in script
    assert "checkpoints/best.pt" in script
    assert "best-8000-v1" in script
    assert "zrave-sequence-comparison/index.html" in script
    assert "wav_count" in script
    assert "-ne 32" in script
