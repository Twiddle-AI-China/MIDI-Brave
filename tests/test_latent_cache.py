from __future__ import annotations

import numpy as np
import pytest
import torch

from dataclasses import replace

from midibrave.cli import create_fixture
from midibrave.config import Config
from midibrave.data import PairDataset, load_manifest
from midibrave.latent_cache import (LatentStatisticsAccumulator,
                                    cache_rave_latents_from_checkpoint,
                                    load_latent_window, save_latent_cache)
from midibrave.predictive_model import PredictiveMidiBrave


def test_latent_cache_slices_by_sample_offset(tmp_path):
    latent = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    path = tmp_path / "sample.npz"
    save_latent_cache(path, latent, "sample-a", 128, "checkpoint-a")
    window = load_latent_window(path, "sample-a", 512, 8, 16, 128, "checkpoint-a")
    np.testing.assert_array_equal(window, latent[:, 4:12])


@pytest.mark.parametrize("field,value", [
    ("sample_id", "sample-b"),
    ("latent_dim", 8),
    ("hop", 64),
    ("checkpoint_hash", "checkpoint-b"),
])
def test_latent_cache_rejects_contract_mismatch(tmp_path, field, value):
    path = tmp_path / "sample.npz"
    save_latent_cache(path, np.ones((16, 32), np.float32),
                      "sample-a", 128, "checkpoint-a")
    expected = {
        "sample_id": "sample-a", "offset_samples": 0, "frames": 8,
        "latent_dim": 16, "hop": 128, "checkpoint_hash": "checkpoint-a",
    }
    expected[field] = value
    with pytest.raises(ValueError, match=field):
        load_latent_window(path, **expected)


def test_latent_statistics_match_direct_calculation():
    first = np.arange(24, dtype=np.float32).reshape(3, 8)
    second = np.arange(24, 48, dtype=np.float32).reshape(3, 8)
    accumulator = LatentStatisticsAccumulator(3)
    accumulator.update(first)
    accumulator.update(second)
    result = accumulator.finalize()
    combined = np.concatenate((first, second), axis=1)
    np.testing.assert_allclose(result.latent_std, combined.std(axis=1), rtol=1e-6)
    deltas = np.concatenate((np.diff(first), np.diff(second)), axis=1)
    np.testing.assert_allclose(result.delta_std, deltas.std(axis=1), rtol=1e-6)
    accelerations = np.concatenate((np.diff(first, n=2), np.diff(second, n=2)), axis=1)
    np.testing.assert_allclose(result.acceleration_std,
                               accelerations.std(axis=1), rtol=1e-6)


def test_pair_dataset_loads_aligned_rave_teacher_windows(tmp_path):
    manifest = create_fixture(tmp_path / "fixture", samples=4096)
    config = Config.load("configs/v3/smoke.yaml")
    assert config.predictive is not None
    data = replace(
        config.data, manifest=str(manifest), cache_root=str(manifest.parent / "cache"),
        dataset_root=str(manifest.parent), repeats=1,
    )
    rave_root = manifest.parent / "cache" / "rave"
    for record in load_manifest(manifest):
        latent = np.full((16, 32), record.midi_note, dtype=np.float32)
        save_latent_cache(rave_root / f"{record.cache_id}.npz", latent,
                          record.sample_id, 128, "checkpoint-a")
    predictive = replace(
        config.predictive, require_rave_cache=True)
    sample = PairDataset(data, config.seed, predictive,
                         rave_checkpoint_hash="checkpoint-a")[0]
    assert sample["rave_a"].shape == (16, 32)
    assert sample["rave_b"].shape == (16, 32)


def test_cache_command_encodes_complete_fixture_renders(tmp_path):
    manifest = create_fixture(tmp_path / "fixture", samples=2048)
    config = Config.load("configs/v3/smoke.yaml")
    assert config.predictive is not None
    data = replace(config.data, manifest=str(manifest),
                   cache_root=str(manifest.parent / "cache"),
                   dataset_root=str(manifest.parent))
    config = replace(config, data=data)
    model = PredictiveMidiBrave(
        config.model, config.predictive,
        config.data.window_samples, config.data.sample_rate)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    result = cache_rave_latents_from_checkpoint(config, checkpoint, "cpu")
    assert result["cached"] == len(load_manifest(manifest))
    assert (manifest.parent / "cache" / "rave-statistics.npz").is_file()
