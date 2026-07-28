from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from midibrave.zrave_flow_model import FlowStatistics
from midibrave.zrave_pure_flow_audition import (
    _decode_latent,
    audio_diagnostics,
    audition_file_names,
    rollout_pure_flow,
    select_audition_cases,
    validate_pure_checkpoint,
)


def _statistics() -> FlowStatistics:
    return FlowStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(10.0),
    )


class _RecordingSampler:
    def __init__(self) -> None:
        self.block_indices: list[int] = []
        self.histories: list[torch.Tensor] = []

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        history = args[2]
        assert isinstance(history, torch.Tensor)
        block_index = int(kwargs["block_index"])
        self.block_indices.append(block_index)
        self.histories.append(history.clone())
        return torch.full(
            (history.shape[0], 64, 16),
            float(block_index + 1),
        )


def test_pure_rollout_uses_block_indices_and_generated_history() -> None:
    sampler = _RecordingSampler()

    output = rollout_pure_flow(
        object(),
        _statistics(),
        torch.zeros(1, 32, 16),
        130,
        generation_seed=17,
        temperature=1.0,
        wander_delay_frames=32,
        solver_steps=8,
        sample_block=sampler,
    )

    assert output.shape == (1, 130, 16)
    assert sampler.block_indices == [0, 1, 2]
    assert torch.equal(sampler.histories[0], torch.zeros(1, 32, 16))
    assert torch.equal(sampler.histories[1], torch.ones(1, 32, 16))
    assert torch.equal(
        sampler.histories[2],
        torch.full((1, 32, 16), 2.0),
    )
    assert torch.equal(output[:, :64], torch.ones(1, 64, 16))
    assert torch.equal(output[:, 64:128], torch.full((1, 64, 16), 2.0))
    assert torch.equal(output[:, 128:], torch.full((1, 2, 16), 3.0))


def test_select_cases_is_deterministic_and_rejects_short_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sequences.jsonl"
    rows = [
        {
            "split": "test",
            "source_name": "serum_full",
            "category": "Pad",
            "active_frames": 107,
            "maximum_future_frames": 64,
            "canonical_preset_id": "serum:pad",
            "sample_id": "pad-good",
        },
        {
            "split": "test",
            "source_name": "serum_full",
            "category": "Pad",
            "active_frames": 95,
            "maximum_future_frames": 64,
            "canonical_preset_id": "serum:pad-short",
            "sample_id": "pad-short",
        },
        {
            "split": "test",
            "source_name": "serum_full",
            "category": "Bass",
            "active_frames": 100,
            "maximum_future_frames": 64,
            "canonical_preset_id": "serum:bass",
            "sample_id": "bass-good",
        },
        {
            "split": "validation",
            "source_name": "serum_full",
            "category": "Bass",
            "active_frames": 107,
            "maximum_future_frames": 64,
            "canonical_preset_id": "serum:bass-validation",
            "sample_id": "bass-validation",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    first = select_audition_cases(
        path,
        categories=("Pad", "Bass"),
        seed=20260728,
    )
    replay = select_audition_cases(
        path,
        categories=("Pad", "Bass"),
        seed=20260728,
    )

    assert first == replay
    assert [row["category"] for row in first] == ["Pad", "Bass"]
    assert [row["sample_id"] for row in first] == [
        "pad-good",
        "bass-good",
    ]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            latent_dim=16,
            context_frames=32,
            future_frames=64,
            pitch_conditioning=False,
        )
    )


def _payload() -> dict[str, object]:
    return {
        "format": 1,
        "architecture": "zrave_pure_flow_transformer_v1",
        "update": 45000,
        "contract": {
            "pack_index_sha256": "a" * 64,
            "statistics_sha256": "b" * 64,
            "latent_dim": 16,
            "context_frames": 32,
            "future_frames": 64,
            "pitch_conditioning": False,
        },
    }


def test_validate_pure_checkpoint_accepts_exact_contract() -> None:
    validate_pure_checkpoint(
        _payload(),
        _config(),
        index_hash="a" * 64,
        statistics_hash="b" * 64,
        expected_update=45000,
    )


def test_validate_pure_checkpoint_rejects_pitch_conditioning() -> None:
    payload = _payload()
    payload["architecture"] = "zrave_conditional_flow_transformer_v1"
    contract = payload["contract"]
    assert isinstance(contract, dict)
    contract["pitch_conditioning"] = True

    with pytest.raises(ValueError, match="pure flow"):
        validate_pure_checkpoint(
            payload,
            _config(),
            index_hash="a" * 64,
            statistics_hash="b" * 64,
            expected_update=45000,
        )


def test_audition_file_names_define_six_roles_per_category() -> None:
    names = audition_file_names(2, "Lead")

    assert names == {
        "source": "wavs/02-lead-source.wav",
        "rave_direct": "wavs/02-lead-rave-direct.wav",
        "seed17_short": "wavs/02-lead-seed17-short.wav",
        "seed29_short": "wavs/02-lead-seed29-short.wav",
        "seed17_long": "wavs/02-lead-seed17-long.wav",
        "seed29_long": "wavs/02-lead-seed29-long.wav",
    }


def test_audio_diagnostics_reports_finite_mono_audio() -> None:
    audio = torch.tensor([0.0, -0.5, 0.5, 0.0])

    report = audio_diagnostics(audio, sample_rate=4)

    assert report == {
        "samples": 4,
        "seconds": 1.0,
        "peak": 0.5,
        "rms": pytest.approx(2.0**-1.5),
    }


def test_audio_diagnostics_rejects_nonfinite_audio() -> None:
    with pytest.raises(ValueError, match="non-finite audio"):
        audio_diagnostics(
            torch.tensor([0.0, float("nan")]),
            sample_rate=44100,
        )


class _ColdStartCodec(torch.nn.Module):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        audio = torch.ones(
            latent.shape[0],
            1,
            latent.shape[2] * 4,
            device=latent.device,
        )
        audio[:, :, :4] = 99.0
        return audio


def test_decode_latent_crops_repeated_frame_decoder_preroll() -> None:
    decoded = _decode_latent(
        _ColdStartCodec(),
        torch.arange(48, dtype=torch.float32).reshape(3, 16),
        random_seed=17,
        device=torch.device("cpu"),
        latent_hop=4,
        decoder_preroll_frames=32,
    )

    assert decoded.shape == (12,)
    assert torch.equal(decoded, torch.ones(12))
