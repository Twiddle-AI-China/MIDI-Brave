from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from midibrave.cli import build_seed_bank_from_cache, create_fixture
from midibrave.config import Config
from midibrave.data import load_manifest
from midibrave.latent_cache import save_latent_cache
from midibrave.seed_bank import SeedBank


def make_bank() -> SeedBank:
    clap = np.eye(3, dtype=np.float32)
    latents = np.arange(3 * 4 * 16, dtype=np.float32).reshape(3, 4, 16)
    return SeedBank(
        latents, clap, np.array([48, 60, 72]), np.array([50, 80, 127]),
        ["a", "b", "c"], {
            "schema": 1, "latent_dim": 4, "history_frames": 16,
            "samples_per_latent": 128, "checkpoint_hash": "checkpoint-a",
        })


def test_seed_bank_selects_reproducible_clap_neighbor():
    bank = make_bank()
    control = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    first = bank.select(control, top_k=2, random_seed=17)
    second = bank.select(control, top_k=2, random_seed=17)
    assert first.index == second.index
    assert first.index in {0, 1}
    np.testing.assert_array_equal(first.latent, second.latent)


def test_seed_bank_round_trips_without_pickle(tmp_path):
    bank = make_bank()
    bank.save(tmp_path / "bank")
    loaded = SeedBank.load(tmp_path / "bank", {
        "latent_dim": 4, "history_frames": 16, "samples_per_latent": 128,
        "checkpoint_hash": "checkpoint-a",
    })
    np.testing.assert_array_equal(loaded.latents, bank.latents)
    assert loaded.sample_ids == bank.sample_ids


def test_seed_bank_rejects_empty_and_incompatible_contracts(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        SeedBank(np.empty((0, 4, 16), np.float32), np.empty((0, 3), np.float32),
                 np.empty(0), np.empty(0), [], {})
    bank = make_bank()
    bank.save(tmp_path / "bank")
    with pytest.raises(ValueError, match="checkpoint_hash"):
        SeedBank.load(tmp_path / "bank", {"checkpoint_hash": "checkpoint-b"})


def test_cli_builder_materializes_seed_bank_from_cache(tmp_path):
    manifest = create_fixture(tmp_path / "fixture", samples=4096)
    config = Config.load("configs/v3/smoke.yaml")
    assert config.predictive is not None
    data = replace(config.data, manifest=str(manifest),
                   cache_root=str(manifest.parent / "cache"),
                   dataset_root=str(manifest.parent))
    config = replace(config, data=data)
    rave_root = manifest.parent / "cache" / "rave"
    for record in load_manifest(manifest):
        latent = (
            np.arange(32, dtype=np.float32)[None, :]
            + 100.0 * record.midi_note
        )
        save_latent_cache(
            rave_root / f"{record.cache_id}.npz",
            np.repeat(latent, 16, axis=0),
            record.sample_id, 128, "checkpoint-a")
    output = tmp_path / "seeds" / "bank"
    result = build_seed_bank_from_cache(config, "checkpoint-a", output)
    assert result["seeds"] == len(load_manifest(manifest))
    loaded = SeedBank.load(output, {"checkpoint_hash": "checkpoint-a"})
    assert loaded.latents.shape[1:] == (16, 16)
    assert loaded.metadata["seed_offset_frames"] == 8
    assert loaded.metadata["seed_offset_samples"] == 8 * 128
    expected = (
        np.arange(8, 24, dtype=np.float32)[None, :]
        + 100.0 * loaded.notes[:, None, None]
    )
    np.testing.assert_array_equal(
        loaded.latents, np.broadcast_to(expected, loaded.latents.shape))
