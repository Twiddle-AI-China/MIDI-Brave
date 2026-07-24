from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from midibrave.latent_cache import save_latent_cache
from midibrave.zrave_config import (
    ZraveConfig,
    ZraveDataConfig,
    ZraveLossConfig,
    ZraveModelConfig,
    ZraveOptimizerConfig,
    ZraveRaveConfig,
    ZraveTrainConfig,
)
from midibrave.zrave_data import (
    cache_selected_latents,
    latent_cache_path,
    pack_cached_latents,
    read_jsonl,
    select_balanced_manifest,
)


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
            checkpoint=str(tmp_path / "checkpoint.pt"),
            sample_rate=44100,
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


def _write_synthetic_caches(
    config: ZraveConfig,
    *,
    wrong_sample_id: bool = False,
) -> tuple[list[dict[str, object]], str]:
    checkpoint = Path(config.rave.checkpoint)
    checkpoint.write_bytes(b"frozen-rave-checkpoint")
    import hashlib

    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation", "test")):
        sample_id = f"sample-{index}"
        rows.append(
            {
                "sample_id": sample_id,
                "preset_id": f"preset-{index}",
                "midi_note": 60,
                "velocity": 100,
                "audio_path": f"{sample_id}.wav",
                "duration_seconds": 5.0,
                "split": split,
                "zrave_category": CATEGORIES[index],
                "zrave_bank": f"bank-{index}",
            }
        )
        frames = 220 + index
        latent = np.arange(16 * frames, dtype=np.float32).reshape(16, frames)
        save_latent_cache(
            latent_cache_path(config, sample_id),
            latent,
            "wrong-id" if wrong_sample_id and index == 1 else sample_id,
            128,
            checkpoint_hash,
        )
    _write_jsonl(Path(config.data.selected_manifest), rows)
    return rows, checkpoint_hash


def test_pack_cached_latents_trims_warmup_and_uses_train_statistics(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    rows, checkpoint_hash = _write_synthetic_caches(config)

    metadata = pack_cached_latents(config)

    pack_root = Path(config.data.packed_root)
    latents = np.load(pack_root / "latents.npy", mmap_mode="r")
    lengths = np.load(pack_root / "lengths.npy")
    splits = np.load(pack_root / "splits.npy")
    statistics = np.load(pack_root / "statistics.npz")
    assert latents.shape == (len(rows), 158, 16)
    assert latents.dtype == np.float16
    np.testing.assert_array_equal(lengths, np.asarray([156, 157, 158]))
    np.testing.assert_array_equal(splits, np.asarray([0, 1, 2]))
    assert np.all(lengths > 128 + 16)
    assert metadata.rave_checkpoint_sha256 == checkpoint_hash
    expected_train = np.arange(16 * 220, dtype=np.float32).reshape(16, 220)
    expected_train = expected_train[:, 64:].T
    np.testing.assert_allclose(
        statistics["mean"],
        expected_train.mean(axis=0),
        rtol=1e-5,
    )


def test_pack_rejects_cache_contract_mismatch(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _write_synthetic_caches(config, wrong_sample_id=True)

    with pytest.raises(ValueError, match="sample_id mismatch"):
        pack_cached_latents(config)


def test_cache_uses_one_standalone_torchscript_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path)
    checkpoint = Path(config.rave.checkpoint)
    checkpoint.write_bytes(b"standalone-rave")
    selected = Path(config.data.selected_manifest)
    _write_jsonl(
        selected,
        [
            {
                "sample_id": "pad-60-127",
                "preset_id": "pad",
                "audio_path": "pad.wav",
                "split": "train",
                "zrave_category": "Pad",
            }
        ],
    )
    (tmp_path / "pad.wav").write_bytes(b"fixture")

    class FakeStandaloneRave:
        sr = torch.tensor([44100])
        latent_size = 16

        def to(self, device: str) -> "FakeStandaloneRave":
            assert device == "cpu"
            return self

        def eval(self) -> "FakeStandaloneRave":
            return self

        def encode(self, audio: torch.Tensor) -> torch.Tensor:
            frames = audio.shape[-1] // config.data.latent_hop
            return torch.ones(1, 16, frames)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                latent.shape[0],
                1,
                latent.shape[-1] * config.data.latent_hop,
            )

    load_calls: list[tuple[str, object]] = []

    def fake_load(path: str, map_location: object) -> FakeStandaloneRave:
        load_calls.append((path, map_location))
        return FakeStandaloneRave()

    monkeypatch.setattr(torch.jit, "load", fake_load)
    monkeypatch.setattr(
        "midibrave.data.load_audio",
        lambda path, sample_rate: np.zeros(
            config.data.latent_hop * 4,
            dtype=np.float32,
        ),
    )

    report = cache_selected_latents(config, 0, 1, "cpu")

    assert report["cached"] == 1
    assert report["codec"] == "standalone_torchscript_rave"
    assert len(load_calls) == 1
    cache = np.load(latent_cache_path(config, "pad-60-127"))
    assert cache["latent"].shape == (16, 4)
    assert cache["checkpoint_hash"].item() == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()


def test_zrave_data_has_no_conditional_model_dependency() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "midibrave"
        / "zrave_data.py"
    ).read_text(encoding="utf-8").casefold()

    assert "predictivemidibrave" not in source
    assert "raveencoder" not in source
    assert "sourceconfig" not in source
    assert "clap" not in source
    assert "midi_control" not in source
