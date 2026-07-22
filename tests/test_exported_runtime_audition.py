from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.render_exported_runtime_audition import (
    _render_audio_matrix,
    build_clip_manifest,
    resolve_seed_selections,
    select_diverse_timbres,
    select_notes,
    validate_runtime_metadata,
)
from midibrave.seed_bank import SeedBank


class _FiniteRuntime:
    def initial_state(self, clap, random_seed: int, top_k: int = 8):
        del top_k
        return torch.full((1, 2, 3), float(random_seed), device=clap.device)

    def step(self, history, clap, note, velocity):
        del clap, velocity
        block = note.float().view(-1, 1, 1).expand(-1, 1, 2)
        return block, history + 1.0


class _NonFiniteRuntime(_FiniteRuntime):
    def step(self, history, clap, note, velocity):
        block, history = super().step(history, clap, note, velocity)
        block[0, 0, 0] = float("nan")
        return block, history


class _TracingRuntime(_FiniteRuntime):
    def __init__(self):
        self.initial_claps = []
        self.step_claps = []

    def initial_state(self, clap, random_seed: int, top_k: int = 8):
        self.initial_claps.append(clap.detach().cpu().numpy().copy())
        return super().initial_state(clap, random_seed, top_k)

    def step(self, history, clap, note, velocity):
        self.step_claps.append(clap.detach().cpu().numpy().copy())
        return super().step(history, clap, note, velocity)


def test_select_notes_uses_only_available_training_notes():
    assert select_notes([36, 37, 48, 59, 60, 71], 4) == [36, 48, 59, 71]


def test_select_diverse_timbres_returns_unique_farthest_candidates():
    candidates = [
        ("pad-a", "a", np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        ("pad-b", "b", np.asarray([0.9, 0.1, 0.0], dtype=np.float32)),
        ("pad-c", "c", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        ("pad-d", "d", np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
    ]

    selected = select_diverse_timbres(candidates, 3, random_seed=7)

    assert len(selected) == 3
    assert len({item[0] for item in selected}) == 3
    matrix = np.stack([item[2] for item in selected])
    similarity = matrix @ matrix.T
    assert float(np.max(similarity - np.eye(3, dtype=np.float32))) <= 0.994


def test_runtime_metadata_must_prove_encoder_free(tmp_path):
    metadata_path = tmp_path / "runtime.pt.json"
    metadata_path.write_text(json.dumps({"encoder_free": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="encoder_free=true"):
        validate_runtime_metadata(metadata_path)


def test_runtime_metadata_requires_reproducibility_contract(tmp_path):
    metadata_path = tmp_path / "runtime.pt.json"
    metadata_path.write_text(json.dumps({
        "encoder_free": True,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "checkpoint_sha256": "checkpoint",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="seed_bank_npz_sha256"):
        validate_runtime_metadata(metadata_path)


def test_seed_selections_are_distinct_and_auditable():
    bank = SeedBank(
        latents=np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3),
        clap=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        notes=np.asarray([48, 60, 72]), velocities=np.asarray([127, 127, 127]),
        sample_ids=["seed-a", "seed-b", "seed-c"],
        metadata={"latent_dim": 2, "history_frames": 3},
    )

    selected = resolve_seed_selections(
        bank, np.asarray([1.0, 0.0], dtype=np.float32),
        seed_count=3, random_seed=100, top_k=3)

    assert [item["id"] for item in selected] == [0, 1, 2]
    assert len({item["bank_index"] for item in selected}) == 3
    assert {item["sample_id"] for item in selected} == {"seed-a", "seed-b", "seed-c"}
    assert all(item["distance"] >= 0.0 for item in selected)


def test_render_audio_matrix_uses_seed_note_and_exact_duration():
    combinations = [
        {"seed": 0, "timbre": 0, "note": 36},
        {"seed": 1, "timbre": 0, "note": 60},
    ]
    claps = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    audio = _render_audio_matrix(
        _FiniteRuntime(), claps, np.asarray([0.5, 0.5], dtype=np.float32),
        combinations, duration_samples=5,
        warmup_samples=2, block_samples=2, velocity=127.0,
        random_seed=100, device=torch.device("cpu"),
    )

    assert audio.shape == (2, 5)
    assert np.all(audio[0] == 36.0)
    assert np.all(audio[1] == 60.0)


def test_render_audio_matrix_rejects_nonfinite_audio():
    combinations = [{"seed": 0, "timb": 0, "timbre": 0, "note": 36}]
    claps = np.asarray([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _render_audio_matrix(
            _NonFiniteRuntime(), claps,
            np.asarray([0.5, 0.5], dtype=np.float32), combinations,
            duration_samples=2,
            warmup_samples=0, block_samples=2, velocity=127.0,
            random_seed=0, device=torch.device("cpu"),
        )


def test_render_audio_matrix_keeps_seed_anchor_fixed_when_timbre_changes():
    runtime = _TracingRuntime()
    combinations = [
        {"seed": 0, "timbre": 0, "note": 36},
        {"seed": 0, "timbre": 1, "note": 36},
    ]
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    anchor = np.asarray([0.5, 0.5], dtype=np.float32)

    _render_audio_matrix(
        runtime, targets, anchor, combinations, duration_samples=2,
        warmup_samples=0, block_samples=2, velocity=127.0,
        random_seed=9, device=torch.device("cpu"),
    )

    assert len(runtime.initial_claps) == 2
    assert all(np.array_equal(value[0], anchor) for value in runtime.initial_claps)
    assert np.array_equal(runtime.step_claps[0][0], targets[0])
    assert np.array_equal(runtime.step_claps[1][0], targets[1])


def test_build_clip_manifest_is_complete_cartesian_product():
    timbres = [
        ("pad-a", "sample-a", np.ones(3, dtype=np.float32)),
        ("pad-b", "sample-b", np.ones(3, dtype=np.float32)),
    ]

    combinations, clips = build_clip_manifest(3, timbres, [36, 60])

    assert len(combinations) == 12
    assert len(clips) == 12
    assert len({(row["seed"], row["timbre"], row["note"]) for row in clips}) == 12
    assert clips[-1]["file"] == "runtime-examples/s2-t1-n060.wav"
