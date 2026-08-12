from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import midibrave.zrave_flow_audition as flow_audition
from midibrave.zrave_flow_audition import (
    _mono_audio,
    build_midi_audition_control_sequences,
    match_rms,
    render_index_html,
    rollout_exploration_flow,
    rollout_midi_sequence_flow,
    rollout_pure_flow,
    select_audition_rows,
    validate_midi_sequence_checkpoint_payload,
    validate_pure_checkpoint_payload,
)
from midibrave.zrave_flow_model import FlowStatistics


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


class _MidiFlowShape(_FlowShape):
    midi_sequence_conditioning = True
    note_min = 21
    note_max = 109


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


def test_midi_rollout_chunks_future_and_advances_all_histories() -> None:
    calls: list[dict[str, object]] = []

    def sample_block(
        model: object,
        statistics: object,
        history: torch.Tensor,
        history_midi_note: torch.Tensor,
        history_velocity: torch.Tensor,
        future_midi_note: torch.Tensor,
        future_velocity: torch.Tensor,
        *,
        generation_seed: int,
        block_index: int,
        temperature: float,
        wander_delay_frames: int,
        pitch_guidance: float,
        solver_steps: int,
        history_mask: torch.Tensor,
        schedule_offset_frames: int,
    ) -> torch.Tensor:
        del statistics, generation_seed, temperature
        del wander_delay_frames, pitch_guidance, solver_steps
        calls.append(
            {
                "block_index": block_index,
                "offset": schedule_offset_frames,
                "history": history.clone(),
                "history_note": history_midi_note.clone(),
                "history_velocity": history_velocity.clone(),
                "future_note": future_midi_note.clone(),
                "future_velocity": future_velocity.clone(),
                "history_mask": history_mask.clone(),
            }
        )
        return torch.full(
            (
                history.shape[0],
                model.future_frames,
                model.latent_dim,
            ),
            float(block_index + 1),
        )

    future_notes = torch.cat(
        (
            torch.full((1, 64), 60),
            torch.full((1, 64), 62),
            torch.full((1, 2), 64),
        ),
        dim=1,
    )
    future_velocities = torch.cat(
        (
            torch.full((1, 64), 70),
            torch.full((1, 64), 80),
            torch.full((1, 2), 90),
        ),
        dim=1,
    )
    generated = rollout_midi_sequence_flow(
        _MidiFlowShape(),
        object(),
        torch.zeros(1, 32, 128),
        torch.full((1, 32), 55),
        torch.full((1, 32), 40),
        future_notes,
        future_velocities,
        frames=130,
        generation_seed=17,
        temperature=1.0,
        wander_delay_frames=32,
        pitch_guidance=3.0,
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
    assert [call["block_index"] for call in calls] == [0, 1, 2]
    assert [call["offset"] for call in calls] == [0, 64, 128]
    torch.testing.assert_close(
        calls[1]["history_note"],
        torch.full((1, 32), 60),
    )
    torch.testing.assert_close(
        calls[1]["history_velocity"],
        torch.full((1, 32), 70),
    )
    torch.testing.assert_close(
        calls[2]["history_note"],
        torch.full((1, 32), 62),
    )
    torch.testing.assert_close(
        calls[2]["history_velocity"],
        torch.full((1, 32), 80),
    )
    torch.testing.assert_close(
        calls[2]["future_note"],
        torch.full((1, 64), 64),
    )
    torch.testing.assert_close(
        calls[2]["future_velocity"],
        torch.full((1, 64), 90),
    )
    assert all(torch.all(call["history_mask"]) for call in calls)


def test_midi_rollout_rejects_incomplete_future_control() -> None:
    with pytest.raises(ValueError, match="future_midi_note"):
        rollout_midi_sequence_flow(
            _MidiFlowShape(),
            object(),
            torch.zeros(1, 32, 128),
            torch.full((1, 32), 55),
            torch.full((1, 32), 80),
            torch.full((1, 63), 60),
            torch.full((1, 64), 80),
            frames=64,
            generation_seed=17,
            temperature=1.0,
            wander_delay_frames=32,
            pitch_guidance=3.0,
            solver_steps=8,
        )


def test_midi_audition_controls_use_observed_note_cycle() -> None:
    controls = build_midi_audition_control_sequences(
        torch.tensor([36, 62, 82]),
        torch.tensor([54, 108, 54]),
        frames=4,
        note_min=21,
        note_max=109,
        available_notes=(36, 62, 82),
    )

    matched_notes, matched_velocities = controls["matched"]
    swapped_notes, swapped_velocities = controls["note_swap"]
    torch.testing.assert_close(
        matched_notes,
        torch.tensor(
            [
                [36, 36, 36, 36],
                [62, 62, 62, 62],
                [82, 82, 82, 82],
            ]
        ),
    )
    torch.testing.assert_close(
        swapped_notes,
        torch.tensor(
            [
                [62, 62, 62, 62],
                [82, 82, 82, 82],
                [36, 36, 36, 36],
            ]
        ),
    )
    torch.testing.assert_close(matched_velocities, swapped_velocities)


def test_exploration_rollout_commits_sixteen_without_resetting_offset() -> None:
    offsets: list[int] = []

    def sample_candidates(
        model: object,
        statistics: object,
        history: torch.Tensor,
        *,
        generation_seed: int,
        commit_index: int,
        candidate_count: int,
        temperature: float,
        wander_delay_frames: float,
        solver_steps: int,
        schedule_offset_frames: int,
        visible_history_frames: int,
    ) -> torch.Tensor:
        del statistics, generation_seed, temperature
        del wander_delay_frames, solver_steps, visible_history_frames
        offsets.append(schedule_offset_frames)
        values = torch.arange(
            1,
            model.future_frames + 1,
            dtype=history.dtype,
            device=history.device,
        ).view(1, 1, -1, 1)
        values = values.expand(
            history.shape[0],
            candidate_count,
            -1,
            model.latent_dim,
        )
        return values + float(commit_index)

    statistics = FlowStatistics(
        mean=torch.zeros(128),
        latent_std=torch.ones(128),
        delta_std=torch.ones(128),
        latent_norm_p01=torch.tensor(0.0),
        latent_norm_p99=torch.tensor(10000.0),
    )
    result = rollout_exploration_flow(
        _FlowShape(),
        statistics,
        torch.zeros(1, 32, 128),
        frames=48,
        generation_seed=9,
        exploration=1.0,
        candidate_count=2,
        stride_frames=16,
        solver_steps=8,
        sample_candidates=sample_candidates,
    )

    assert result.generated.shape == (1, 48, 128)
    assert result.selected_candidate_indices.shape == (1, 3)
    assert offsets == [0, 16, 32]


def test_exploration_rollout_replays_identically() -> None:
    calls: list[tuple[int, int]] = []

    def sample_candidates(
        model: object,
        statistics: object,
        history: torch.Tensor,
        *,
        generation_seed: int,
        commit_index: int,
        candidate_count: int,
        **controls: object,
    ) -> torch.Tensor:
        del statistics, controls
        calls.append((generation_seed, commit_index))
        generator = torch.Generator(device=history.device)
        generator.manual_seed(generation_seed + commit_index)
        return torch.randn(
            history.shape[0],
            candidate_count,
            model.future_frames,
            model.latent_dim,
            generator=generator,
            device=history.device,
        )

    statistics = FlowStatistics(
        mean=torch.zeros(128),
        latent_std=torch.ones(128),
        delta_std=torch.ones(128),
        latent_norm_p01=torch.tensor(0.0),
        latent_norm_p99=torch.tensor(10000.0),
    )
    arguments = (
        _FlowShape(),
        statistics,
        torch.zeros(1, 32, 128),
        32,
    )
    first = rollout_exploration_flow(
        *arguments,
        generation_seed=41,
        exploration=0.75,
        candidate_count=2,
        sample_candidates=sample_candidates,
    )
    second = rollout_exploration_flow(
        *arguments,
        generation_seed=41,
        exploration=0.75,
        candidate_count=2,
        sample_candidates=sample_candidates,
    )

    torch.testing.assert_close(first.generated, second.generated)
    torch.testing.assert_close(
        first.selected_candidate_indices,
        second.selected_candidate_indices,
    )


def test_renderer_source_keeps_decoder_seed_independent_of_flow_seed() -> None:
    source = Path(flow_audition.__file__).read_text(encoding="utf-8")

    assert "generation_seed * 100" not in source
    assert "random_seed=config.seed + index" in source


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


def _midi_checkpoint_payload() -> dict[str, object]:
    return {
        "format": 1,
        "architecture": "zrave_midi_sequence_flow_transformer_v2",
        "update": 1000,
        "model": {"weight": torch.ones(2)},
        "contract": {
            "latent_dim": 128,
            "context_frames": 32,
            "future_frames": 64,
            "pitch_conditioning": True,
            "midi_sequence_conditioning": True,
            "exploration_enabled": False,
            "segment_sampling": True,
            "maximum_updates": 20000,
            "world_size": 1,
            "batch_per_gpu": 16,
            "config_sha256": "a" * 64,
            "pack_index_sha256": "b" * 64,
            "statistics_sha256": "c" * 64,
            "pitch_checkpoint_sha256": "d" * 64,
            "pitch_qualification_sha256": "e" * 64,
        },
        "initialization": {
            "source_update": 85000,
            "source_architecture": "zrave_pure_flow_transformer_v1",
            "checkpoint_sha256": "f" * 64,
        },
    }


def _validate_midi_payload(payload: object) -> int:
    return validate_midi_sequence_checkpoint_payload(
        payload,
        latent_dim=128,
        context_frames=32,
        future_frames=64,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
        pitch_checkpoint_sha256="d" * 64,
        pitch_qualification_sha256="e" * 64,
        maximum_updates=20000,
        segment_sampling=True,
        expected_initializer_sha256="f" * 64,
        expected_initializer_update=85000,
    )


def test_validate_midi_checkpoint_payload_accepts_exact_contract() -> None:
    assert _validate_midi_payload(_midi_checkpoint_payload()) == 1000


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "architecture"),
        ("pitch_hash", "pitch_checkpoint_sha256"),
        ("initializer_hash", "initializer SHA-256"),
        ("initializer_update", "initializer update"),
    ],
)
def test_validate_midi_checkpoint_payload_rejects_contract_drift(
    mutation: str,
    message: str,
) -> None:
    payload = _midi_checkpoint_payload()
    if mutation == "architecture":
        payload["architecture"] = "zrave_conditional_flow_transformer_v1"
    elif mutation == "pitch_hash":
        payload["contract"]["pitch_checkpoint_sha256"] = "0" * 64
    elif mutation == "initializer_hash":
        payload["initialization"]["checkpoint_sha256"] = "0" * 64
    else:
        payload["initialization"]["source_update"] = 84000

    with pytest.raises(ValueError, match=message):
        _validate_midi_payload(payload)


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


def test_validate_pure_checkpoint_payload_accepts_exploration_v2() -> None:
    payload = _checkpoint_payload()
    payload["architecture"] = "zrave_pure_flow_transformer_v2"
    payload["contract"]["exploration_enabled"] = True

    update = validate_pure_checkpoint_payload(
        payload,
        latent_dim=128,
        context_frames=32,
        future_frames=64,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
        exploration_enabled=True,
    )

    assert update == 85000


def test_validate_pure_checkpoint_payload_rejects_v2_contract_mismatch() -> None:
    payload = _checkpoint_payload()
    payload["architecture"] = "zrave_pure_flow_transformer_v2"
    payload["contract"]["exploration_enabled"] = False

    with pytest.raises(ValueError, match="exploration_enabled"):
        validate_pure_checkpoint_payload(
            payload,
            latent_dim=128,
            context_frames=32,
            future_frames=64,
            config_sha256="a" * 64,
            pack_index_sha256="b" * 64,
            statistics_sha256="c" * 64,
            exploration_enabled=True,
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


def test_render_index_html_describes_sixteen_frame_commits() -> None:
    manifest = {
        "checkpoint": {"update": 3000, "sha256": "f" * 64},
        "context_frames": 32,
        "generated_frames": 320,
        "commit_stride_frames": 16,
        "sample_rate": 44100,
        "latent_hop": 2048,
        "examples": [],
    }

    html = render_index_html(manifest)

    assert "20 × 16 generated commits" in html


def test_audition_cli_exposes_exploration_controls() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "render_zrave_flow_audition.py"
    ).read_text(encoding="utf-8")

    assert '"--explorations"' in script
    assert '"--candidate-count"' in script
    assert "explorations=args.explorations" in script
    assert "candidate_count=args.candidate_count" in script


def test_midi_audition_cli_requires_artifact_hash_contract() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "render_zrave_midi_flow_audition.py"
    ).read_text(encoding="utf-8")

    assert '"--expected-checkpoint-sha256"' in script
    assert '"--expected-initializer-sha256"' in script
    assert '"--pitch-qualification"' in script
    assert '"--note-vocabulary"' in script
    assert 'default=[36, 62, 82]' in script
    assert "swap_semitones" not in script
    assert "render_midi_flow_audition(" in script
    assert "expected_checkpoint_sha256=args.expected_checkpoint_sha256" in script
    assert "expected_initializer_sha256=args.expected_initializer_sha256" in script


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
