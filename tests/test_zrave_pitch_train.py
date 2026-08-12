from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from midibrave.zrave_pitch_probe import (
    LatentPitchProbe,
    pitch_probe_loss,
)
from midibrave.zrave_pitch_train import (
    PitchWindowSampler,
    load_qualified_pitch_probe,
    load_pitch_checkpoint,
    pitch_probe_gate,
    save_pitch_checkpoint,
)


def _sampler(seed: int) -> PitchWindowSampler:
    generator = torch.Generator().manual_seed(88)
    latents = torch.randn(8, 24, 16, generator=generator)
    return PitchWindowSampler(
        latents=latents,
        active_frames=torch.full((8,), 24),
        notes=torch.tensor([48, 48, 60, 60, 72, 72, 84, 84]),
        source_codes=torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]),
        split_codes=torch.zeros(8, dtype=torch.long),
        split_code=0,
        seed=seed,
        device="cpu",
    )


def _run_updates(
    root: Path,
    *,
    updates: int,
    save: bool = False,
    resume: bool = False,
) -> dict[str, float]:
    torch.manual_seed(17)
    model = LatentPitchProbe()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    sampler = _sampler(seed=123)
    checkpoint = root / "checkpoint.pt"
    start = 0
    contract = {
        "pack_index_sha256": "pack-hash",
        "config_sha256": "config-hash",
        "world_size": 1,
        "batch_per_gpu": 4,
    }
    if resume:
        restored = load_pitch_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            expected_contract=contract,
        )
        start = restored["update"]

    model.train()
    for _update in range(start, start + updates):
        windows, notes = sampler.sample(4)
        optimizer.zero_grad(set_to_none=True)
        loss = pitch_probe_loss(model(windows), notes).total
        loss.backward()
        optimizer.step()

    if save:
        save_pitch_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            update=start + updates,
            contract=contract,
        )
    next_windows, next_notes = sampler.sample(4)
    with torch.no_grad():
        next_loss = pitch_probe_loss(
            model(next_windows),
            next_notes,
        ).total
    return {"next_loss": float(next_loss)}


def test_probe_qualification_requires_98_percent_accuracy() -> None:
    passed = pitch_probe_gate(
        {
            "accuracy": 0.981,
            "median_absolute_cents": 10.0,
            "p90_absolute_cents": 35.0,
            "finite": True,
        }
    )
    failed = pitch_probe_gate(
        {
            "accuracy": 0.979,
            "median_absolute_cents": 10.0,
            "p90_absolute_cents": 35.0,
            "finite": True,
        }
    )

    assert passed["passed"]
    assert not failed["passed"]


def test_probe_checkpoint_reproduces_next_batch(tmp_path: Path) -> None:
    first = _run_updates(tmp_path / "continuous", updates=3)
    _run_updates(tmp_path / "resumed", updates=2, save=True)
    resumed = _run_updates(
        tmp_path / "resumed",
        updates=1,
        resume=True,
    )

    assert first["next_loss"] == resumed["next_loss"]


def test_pitch_sampler_balances_notes_and_uses_dense_sources() -> None:
    sampler = _sampler(seed=9)

    _windows, notes = sampler.sample(12)

    assert torch.bincount(notes - 48, minlength=37)[[0, 12, 24, 36]].tolist() == [
        3,
        3,
        3,
        3,
    ]
    assert set(sampler.last_source_codes.tolist()) == {1, 2}


def test_pitch_sampler_uses_every_source_with_valid_midi() -> None:
    sampler = PitchWindowSampler(
        latents=torch.randn(3, 24, 16),
        active_frames=torch.full((3,), 24),
        notes=torch.tensor([48, 60, 72]),
        source_codes=torch.tensor([0, 1, 2]),
        split_codes=torch.zeros(3, dtype=torch.long),
        split_code=0,
        seed=21,
        device="cpu",
    )

    _windows, notes = sampler.sample(3)

    assert sorted(notes.tolist()) == [48, 60, 72]
    assert set(sampler.last_source_codes.tolist()) == {0, 1, 2}


def test_pitch_sampler_supports_serum128_latents() -> None:
    sampler = PitchWindowSampler(
        latents=torch.randn(3, 24, 128),
        active_frames=torch.full((3,), 24),
        notes=torch.tensor([36, 62, 82]),
        source_codes=torch.zeros(3, dtype=torch.long),
        split_codes=torch.zeros(3, dtype=torch.long),
        split_code=0,
        seed=22,
        device="cpu",
    )

    windows, notes = sampler.sample(3)

    assert windows.shape == (3, 16, 128)
    assert sorted(notes.tolist()) == [36, 62, 82]


def test_pitch_sampler_honors_prepacked_category_filter() -> None:
    sampler = PitchWindowSampler(
        latents=torch.randn(4, 24, 128),
        active_frames=torch.full((4,), 24),
        notes=torch.tensor([36, 36, 62, 82]),
        source_codes=torch.zeros(4, dtype=torch.long),
        split_codes=torch.zeros(4, dtype=torch.long),
        split_code=0,
        seed=23,
        device="cpu",
        record_eligible=torch.tensor([False, True, True, True]),
    )

    _windows, notes = sampler.sample(6)

    assert sorted(set(notes.tolist())) == [36, 62, 82]
    assert 0 not in sampler.last_record_indices.tolist()


def test_qualified_loader_checks_pack_hash_and_freezes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pitch-probe"
    checkpoint = root / "checkpoints" / "best-qualified.pt"
    model = LatentPitchProbe()
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _sampler(seed=4)
    save_pitch_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        update=1000,
        contract={"pack_index_sha256": "pack-hash"},
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "qualification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "pack_index_sha256": "pack-hash",
                "checkpoint": "checkpoints/best-qualified.pt",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_qualified_pitch_probe(root, "pack-hash")

    assert not loaded.training
    assert all(
        not parameter.requires_grad for parameter in loaded.parameters()
    )
    with pytest.raises(ValueError, match="pack hash"):
        load_qualified_pitch_probe(root, "different-pack")
