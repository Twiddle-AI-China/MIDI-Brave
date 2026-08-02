from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from midibrave.zrave_flow_audition import (
    _mono_audio,
    render_index_html,
    match_rms,
    rollout_pure_flow,
    select_audition_rows,
    validate_pure_checkpoint_payload,
)


SBATCH = (
    Path(__file__).parents[1]
    / "scripts"
    / "cloud"
    / "octopus_serum128_flow_audition.sbatch"
)


class _FlowShape:
    latent_dim = 128
    context_frames = 32
    future_frames = 64


def test_rollout_pure_flow_updates_context_and_trims_final_block() -> None:
    calls: list[tuple[int, torch.Tensor]] = []

    def sample_block(
        model: object,
        statistics: object,
        history: torch.Tensor,
        *,
        generation_seed: int,
        block_index: int,
        temperature: float,
        wander_delay_frames: int,
        solver_steps: int,
    ) -> torch.Tensor:
        del statistics, generation_seed, temperature
        del wander_delay_frames, solver_steps
        calls.append((block_index, history.clone()))
        return torch.full(
            (
                history.shape[0],
                model.future_frames,
                model.latent_dim,
            ),
            float(block_index + 1),
        )

    history = torch.zeros(1, 32, 128)
    generated = rollout_pure_flow(
        _FlowShape(),
        object(),
        history,
        frames=130,
        generation_seed=17,
        temperature=1.0,
        wander_delay_frames=32,
        solver_steps=8,
        sample_block=sample_block,
    )

    assert generated.shape == (1, 130, 128)
    torch.testing.assert_close(generated[:, :64], torch.ones(1, 64, 128))
    torch.testing.assert_close(
        generated[:, 64:128],
        torch.full((1, 64, 128), 2.0),
    )
    torch.testing.assert_close(
        generated[:, 128:],
        torch.full((1, 2, 128), 3.0),
    )
    assert [block_index for block_index, _ in calls] == [0, 1, 2]
    torch.testing.assert_close(calls[0][1], history)
    torch.testing.assert_close(calls[1][1], torch.ones(1, 32, 128))
    torch.testing.assert_close(calls[2][1], torch.full((1, 32, 128), 2.0))


def test_select_audition_rows_is_deterministic_and_category_ordered() -> None:
    rows = [
        {
            "sample_id": "pad-b",
            "category": "Pad",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "lead-a",
            "category": "Lead",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "pad-a",
            "category": "Pad",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "lead-too-short",
            "category": "Lead",
            "split": "test",
            "active_frames": 12,
        },
    ]

    first = select_audition_rows(
        rows,
        categories=("Pad", "Lead"),
        split="test",
        seed=29,
        minimum_active_frames=64,
    )
    second = select_audition_rows(
        list(reversed(rows)),
        categories=("Pad", "Lead"),
        split="test",
        seed=29,
        minimum_active_frames=64,
    )

    assert first == second
    assert [row["category"] for row in first] == ["Pad", "Lead"]
    assert first[1]["sample_id"] == "lead-a"


def test_select_audition_rows_rejects_missing_category() -> None:
    with pytest.raises(ValueError, match="Bass"):
        select_audition_rows(
            [],
            categories=("Bass",),
            split="test",
            seed=1,
            minimum_active_frames=32,
        )


def test_match_rms_is_finite_peak_limited_and_handles_silence() -> None:
    reference = np.full(1024, 0.25, dtype=np.float32)
    candidate = np.full(2048, 0.05, dtype=np.float32)

    matched = match_rms(reference, candidate)

    assert matched.dtype == np.float32
    assert np.isfinite(matched).all()
    assert np.max(np.abs(matched)) <= 0.980001
    assert np.sqrt(np.mean(np.square(matched))) == pytest.approx(0.25)
    assert np.array_equal(
        match_rms(reference, np.zeros(32, dtype=np.float32)),
        np.zeros(32, dtype=np.float32),
    )


def test_mono_audio_resamples_like_training_pipeline(tmp_path: Path) -> None:
    source_rate = 48000
    target_rate = 44100
    time = np.arange(source_rate // 10, dtype=np.float32) / source_rate
    audio = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    path = tmp_path / "source-48k.wav"
    sf.write(path, audio, source_rate, subtype="FLOAT")

    loaded = _mono_audio(path, target_rate)

    assert loaded.shape == (target_rate // 10,)
    assert loaded.dtype == np.float32
    assert np.isfinite(loaded).all()


def _checkpoint_payload() -> dict[str, object]:
    return {
        "format": 1,
        "architecture": "zrave_pure_flow_transformer_v1",
        "update": 85000,
        "model": {"weight": torch.ones(2)},
        "contract": {
            "latent_dim": 128,
            "context_frames": 32,
            "future_frames": 64,
            "pitch_conditioning": False,
            "config_sha256": "a" * 64,
            "pack_index_sha256": "b" * 64,
            "statistics_sha256": "c" * 64,
        },
    }


def test_validate_pure_checkpoint_payload_accepts_exact_contract() -> None:
    update = validate_pure_checkpoint_payload(
        _checkpoint_payload(),
        latent_dim=128,
        context_frames=32,
        future_frames=64,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )

    assert update == 85000


def test_validate_pure_checkpoint_payload_rejects_wrong_architecture() -> None:
    payload = _checkpoint_payload()
    payload["architecture"] = "zrave_conditional_flow_transformer_v1"

    with pytest.raises(ValueError, match="architecture"):
        validate_pure_checkpoint_payload(
            payload,
            latent_dim=128,
            context_frames=32,
            future_frames=64,
            config_sha256="a" * 64,
            pack_index_sha256="b" * 64,
            statistics_sha256="c" * 64,
        )


def test_render_index_html_labels_audio_facts_and_timeline() -> None:
    manifest = {
        "checkpoint": {"update": 85000, "sha256": "f" * 64},
        "context_frames": 32,
        "generated_frames": 320,
        "sample_rate": 44100,
        "latent_hop": 2048,
        "examples": [
            {
                "category": "Pad",
                "sample_id": "pad-a",
                "source": {"matched_wav": "matched/pad-source.wav"},
                "direct": {"matched_wav": "matched/pad-direct.wav"},
                "rollouts": [
                    {
                        "label": "Flow · T0.7 · Seed 17",
                        "matched_wav": "matched/pad-flow.wav",
                    }
                ],
            }
        ],
    }

    html = render_index_html(manifest)

    assert "Source" in html
    assert "RAVE Direct" in html
    assert "Flow · T0.7 · Seed 17" in html
    assert "32 real seed" in html
    assert "5 × 64 generated blocks" in html
    assert "matched/pad-flow.wav" in html


def test_octopus_audition_uses_one_gpu_and_read_only_inputs() -> None:
    script = SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "SERUM128_FLOW_AUDITION_CHECKPOINT" in script
    assert "octopus_serum128_pure_flow.yaml" in script
    assert "render_zrave_flow_audition.py" in script
    assert "/data/midibrave-zrave-flow-serum128:ro" in script
    assert "/data/datasets/Timbre_A/serum-octopus-v1:/source:ro" in script
    assert "--generated-frames 320" in script
