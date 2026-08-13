from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import midibrave.zrave_flow_audition as flow_audition
from midibrave.zrave_flow_audition import (
    _mono_audio,
    build_midi_audition_control_sequences,
    latent_pitch_probe_adherence,
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
from midibrave.zrave_pitch_probe import LatentPitchProbe

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
    step_notes, step_velocities = controls["note_step"]
    velocity_step_notes, velocity_step_velocities = controls["velocity_step"]
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
    torch.testing.assert_close(
        step_notes,
        torch.tensor(
            [
                [36, 36, 62, 62],
                [62, 62, 82, 82],
                [82, 82, 36, 36],
            ]
        ),
    )
    torch.testing.assert_close(step_velocities, matched_velocities)
    torch.testing.assert_close(velocity_step_notes, matched_notes)
    torch.testing.assert_close(
        velocity_step_velocities,
        torch.tensor(
            [
                [54, 54, 108, 108],
                [108, 108, 54, 54],
                [54, 54, 108, 108],
            ]
        ),
    )


def test_midi_audition_controls_require_observed_velocity_pair() -> None:
    with pytest.raises(ValueError, match="velocity step"):
        build_midi_audition_control_sequences(
            torch.tensor([36, 62]),
            torch.tensor([54, 54]),
            frames=32,
            note_min=21,
            note_max=109,
            available_notes=(36, 62),
        )


def test_latent_pitch_probe_adherence_uses_lower_window_midpoint_note() -> None:
    probe = LatentPitchProbe(latent_dim=4, note_min=36, note_max=82)
    with torch.no_grad():
        probe.output.weight.zero_()
        probe.output.bias.fill_(-20.0)
        probe.output.bias[62 - probe.note_min] = 20.0
    requested = torch.cat((torch.full((1, 8), 36), torch.full((1, 24), 62)), dim=1)

    rows = latent_pitch_probe_adherence(
        torch.zeros(1, 32, 4),
        requested,
        probe,
    )

    assert len(rows) == 2
    assert [row["midpoint_frame"] for row in rows] == [7, 23]
    assert [row["requested_midi_note"] for row in rows] == [36, 62]
    assert [row["exact_class"] for row in rows] == [False, True]
    assert [row["within_50_cents"] for row in rows] == [False, True]


def test_latent_pitch_probe_adherence_rejects_partial_window() -> None:
    probe = LatentPitchProbe(latent_dim=4, note_min=36, note_max=82)
    with pytest.raises(ValueError, match="complete window"):
        latent_pitch_probe_adherence(
            torch.zeros(1, 15, 4),
            torch.full((1, 15), 62),
            probe,
        )


def test_latent_pitch_probe_summary_reports_all_required_proxy_metrics() -> None:
    summary = flow_audition._pitch_adherence_summary(
        [
            {
                "absolute_cents": 20.0,
                "exact_class": True,
                "within_50_cents": True,
                "within_100_cents": True,
            },
            {
                "absolute_cents": 120.0,
                "exact_class": False,
                "within_50_cents": False,
                "within_100_cents": False,
            },
        ]
    )

    assert summary == {
        "window_count": 2,
        "exact_class_accuracy": 0.5,
        "absolute_cents_median": 70.0,
        "absolute_cents_p90": pytest.approx(110.0),
        "within_50_cents": 0.5,
        "within_100_cents": 0.5,
        "voiced_coverage": None,
    }


def test_latent_note_step_reports_probe_resolution_settling() -> None:
    requested = [36] * 16 + [62] * 32
    rows = [
        _row
        for _row in (
            {
                "midpoint_frame": 7,
                "requested_midi_note": 36,
                "within_100_cents": True,
            },
            {
                "midpoint_frame": 23,
                "requested_midi_note": 62,
                "within_100_cents": False,
            },
            {
                "midpoint_frame": 39,
                "requested_midi_note": 62,
                "within_100_cents": True,
            },
        )
    ]

    report = flow_audition._latent_pitch_transition_settling(
        rows,
        requested,
        latent_hop=2048,
        sample_rate=44100,
    )

    assert report["metric_kind"] == "latent_pitch_probe_transition_proxy"
    assert report["event_count"] == 1
    assert report["settled_event_count"] == 1
    assert report["events"][0]["settling_frames"] == 23
    assert report["settling_frames"]["median"] == 23.0
    assert report["status"] == "measured"
    assert "report-only" in report["semantic_warning"]


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


def _allowlist_config(path: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            sources=(
                SimpleNamespace(
                    name="serum_balanced",
                    preset_allowlist=(None if path is None else str(path)),
                ),
            ),
        ),
    )


def test_audition_preset_allowlist_filter_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    allowlist = tmp_path / "pad-lead.json"
    allowlist.write_text(
        json.dumps({"preset_ids": ["serum:pad", "serum:lead"]}),
        encoding="utf-8",
    )
    rows = [
        {
            "sample_id": "excluded-category-match",
            "source_name": "serum_balanced",
            "canonical_preset_id": "serum:other",
            "category": "Pad",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "excluded-source-match",
            "source_name": "other_source",
            "canonical_preset_id": "serum:pad",
            "category": "Pad",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "allowed-pad",
            "source_name": "serum_balanced",
            "canonical_preset_id": "serum:pad",
            "category": "Pad",
            "split": "test",
            "active_frames": 100,
        },
        {
            "sample_id": "allowed-lead",
            "source_name": "serum_balanced",
            "canonical_preset_id": "serum:lead",
            "category": "Lead",
            "split": "test",
            "active_frames": 100,
        },
    ]
    config = _allowlist_config(allowlist)

    filtered, digest, allowed = flow_audition._filter_audition_preset_allowlist(
        rows, config
    )
    reversed_filtered, reversed_digest, reversed_allowed = (
        flow_audition._filter_audition_preset_allowlist(
            list(reversed(rows)),
            config,
        )
    )
    first = select_audition_rows(
        filtered,
        categories=("Pad", "Lead"),
        split="test",
        seed=31,
        minimum_active_frames=32,
    )
    second = select_audition_rows(
        reversed_filtered,
        categories=("Pad", "Lead"),
        split="test",
        seed=31,
        minimum_active_frames=32,
    )

    assert first == second
    assert [row["sample_id"] for row in first] == [
        "allowed-pad",
        "allowed-lead",
    ]
    assert {row["canonical_preset_id"] for row in first} <= allowed
    assert allowed == reversed_allowed == frozenset({"serum:pad", "serum:lead"})
    assert (
        digest == reversed_digest == hashlib.sha256(allowlist.read_bytes()).hexdigest()
    )


def test_audition_preset_allowlist_rejects_unknown_pack_id(
    tmp_path: Path,
) -> None:
    allowlist = tmp_path / "missing.ids.txt"
    allowlist.write_text("serum:missing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from pack"):
        flow_audition._filter_audition_preset_allowlist(
            [
                {
                    "source_name": "serum_balanced",
                    "canonical_preset_id": "serum:present",
                }
            ],
            _allowlist_config(allowlist),
        )


def test_audition_without_preset_allowlist_preserves_rows_and_metadata() -> None:
    rows = [{"sample_id": "unchanged"}]

    filtered, digest, allowed = flow_audition._filter_audition_preset_allowlist(
        rows,
        _allowlist_config(None),
    )

    assert filtered is rows
    assert digest is None
    assert allowed is None


def test_render_flow_audition_records_allowlist_hash_and_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "checkpoint.pt"
    codec_path = tmp_path / "codec.ts"
    manifest_path = tmp_path / "unified.jsonl"
    allowlist = tmp_path / "pad.json"
    output = tmp_path / "audition"
    config_path.write_text("config\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    codec_path.write_bytes(b"codec")
    (pack / "index.json").write_text("{}\n", encoding="utf-8")
    (pack / "statistics.npz").write_bytes(b"statistics")
    allowlist.write_text(
        json.dumps({"preset_ids": ["serum:allowed"]}) + "\n",
        encoding="utf-8",
    )
    sequence_rows = [
        {
            "sample_id": "excluded",
            "source_name": "serum_balanced",
            "canonical_preset_id": "serum:excluded",
            "category": "Pad",
            "split": "test",
            "active_frames": 32,
            "midi_note": 60,
            "velocity": 100,
        },
        {
            "sample_id": "allowed",
            "source_name": "serum_balanced",
            "canonical_preset_id": "serum:allowed",
            "category": "Pad",
            "split": "test",
            "active_frames": 32,
            "midi_note": 60,
            "velocity": 100,
        },
    ]
    (pack / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sequence_rows),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({"sample_id": "allowed", "audio_path": "unused.wav"}) + "\n",
        encoding="utf-8",
    )
    codec_hash = hashlib.sha256(codec_path.read_bytes()).hexdigest()
    config = SimpleNamespace(
        seed=7,
        data=SimpleNamespace(
            packed_root=str(pack),
            unified_manifest=str(manifest_path),
            sources=(
                SimpleNamespace(
                    name="serum_balanced",
                    allowed_categories=("Pad",),
                    preset_allowlist=str(allowlist),
                ),
            ),
        ),
        model=SimpleNamespace(
            pitch_conditioning=False,
            profile="standard",
            latent_dim=2,
            context_frames=2,
            future_frames=4,
            solver_steps=8,
        ),
        exploration=SimpleNamespace(enabled=False),
        rave=SimpleNamespace(
            checkpoint=str(codec_path),
            expected_sha256=codec_hash,
            sample_rate=44100,
            latent_hop=2048,
        ),
    )

    class _Statistics:
        def validate(self, latent_dim: int) -> None:
            assert latent_dim == 2

    class _Model:
        def __init__(self, *_args: object) -> None:
            pass

        def to(self, _device: torch.device) -> _Model:
            return self

        def load_state_dict(
            self,
            _state: object,
            *,
            strict: bool,
        ) -> None:
            assert strict

        def eval(self) -> _Model:
            return self

        def requires_grad_(self, _enabled: bool) -> _Model:
            return self

    class _Codec:
        def eval(self) -> _Codec:
            return self

    def fake_rollout(
        _model: object,
        _statistics: object,
        history: torch.Tensor,
        frames: int,
        **_kwargs: object,
    ) -> flow_audition.ExplorationRolloutResult:
        batch = history.shape[0]
        metric = torch.zeros(batch, 1, 1)
        return flow_audition.ExplorationRolloutResult(
            generated=torch.zeros(batch, frames, 2),
            selected_candidate_indices=torch.zeros(
                batch,
                1,
                dtype=torch.long,
            ),
            candidate_scores=metric,
            candidate_motion=metric,
            candidate_boundary_rms=metric,
            candidate_norm_violation=metric,
        )

    observed_checkpoint_contract: dict[str, object] = {}

    def fake_validate_checkpoint(
        *_args: object,
        **kwargs: object,
    ) -> int:
        observed_checkpoint_contract.update(kwargs)
        return 1000

    monkeypatch.setattr(
        flow_audition,
        "ZraveFlowConfig",
        SimpleNamespace(load=lambda _path: config),
    )
    monkeypatch.setattr(
        flow_audition.torch,
        "load",
        lambda *_args, **_kwargs: {
            "architecture": "zrave_pure_flow_transformer_v1",
            "model": {},
        },
    )
    monkeypatch.setattr(
        flow_audition,
        "validate_pure_checkpoint_payload",
        fake_validate_checkpoint,
    )
    monkeypatch.setattr(
        flow_audition,
        "_load_statistics",
        lambda _path: _Statistics(),
    )
    monkeypatch.setattr(flow_audition, "ZraveFlowTransformer", _Model)
    monkeypatch.setattr(
        flow_audition.torch.jit,
        "load",
        lambda *_args, **_kwargs: _Codec(),
    )
    monkeypatch.setattr(
        flow_audition,
        "_codec_latent_size",
        lambda _codec: 2,
    )
    monkeypatch.setattr(
        flow_audition,
        "_packed_latent",
        lambda *_args, **_kwargs: np.zeros((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        flow_audition,
        "_mono_audio",
        lambda *_args, **_kwargs: np.ones(4, dtype=np.float32),
    )
    monkeypatch.setattr(
        flow_audition,
        "_decode_latent",
        lambda *_args, **_kwargs: np.zeros(4, dtype=np.float32),
    )
    monkeypatch.setattr(
        flow_audition,
        "_write_audio_pair",
        lambda _output, stem, *_args, **_kwargs: {"matched_wav": f"matched/{stem}.wav"},
    )
    monkeypatch.setattr(
        flow_audition,
        "rollout_exploration_flow",
        fake_rollout,
    )

    rendered = flow_audition.render_flow_audition(
        config_path,
        checkpoint_path,
        output,
        categories=("Pad",),
        explorations=(0.0,),
        generation_seeds=(17,),
        candidate_count=1,
        generated_frames=1,
        device_name="cpu",
    )

    expected_hash = hashlib.sha256(allowlist.read_bytes()).hexdigest()
    assert observed_checkpoint_contract[
        "expected_data_selection_sha256"
    ] == flow_audition.data_selection_sha256(config)
    assert rendered["preset_allowlist_sha256"] == expected_hash
    assert [example["canonical_preset_id"] for example in rendered["examples"]] == [
        "serum:allowed"
    ]
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["preset_allowlist_sha256"] == expected_hash


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


def _validate_midi_payload(
    payload: object,
    *,
    expected_data_selection_sha256: str | None = None,
) -> int:
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
        expected_data_selection_sha256=(expected_data_selection_sha256),
    )


def test_validate_midi_checkpoint_payload_accepts_exact_contract() -> None:
    assert _validate_midi_payload(_midi_checkpoint_payload()) == 1000


def test_validate_midi_checkpoint_payload_binds_data_selection() -> None:
    payload = _midi_checkpoint_payload()
    payload["contract"]["data_selection_sha256"] = "9" * 64

    assert (
        _validate_midi_payload(
            payload,
            expected_data_selection_sha256="9" * 64,
        )
        == 1000
    )

    payload["contract"]["data_selection_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="data_selection_sha256"):
        _validate_midi_payload(
            payload,
            expected_data_selection_sha256="9" * 64,
        )


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


def test_validate_pure_checkpoint_payload_binds_data_selection() -> None:
    payload = _checkpoint_payload()
    payload["contract"]["data_selection_sha256"] = "9" * 64

    update = validate_pure_checkpoint_payload(
        payload,
        latent_dim=128,
        context_frames=32,
        future_frames=64,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
        expected_data_selection_sha256="9" * 64,
    )
    assert update == 85000

    payload["contract"].pop("data_selection_sha256")
    with pytest.raises(ValueError, match="data_selection_sha256"):
        validate_pure_checkpoint_payload(
            payload,
            latent_dim=128,
            context_frames=32,
            future_frames=64,
            config_sha256="a" * 64,
            pack_index_sha256="b" * 64,
            statistics_sha256="c" * 64,
            expected_data_selection_sha256="9" * 64,
        )


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
        Path(__file__).parents[1] / "scripts" / "render_zrave_flow_audition.py"
    ).read_text(encoding="utf-8")

    assert '"--explorations"' in script
    assert '"--candidate-count"' in script
    assert "explorations=args.explorations" in script
    assert "candidate_count=args.candidate_count" in script


def test_midi_audition_cli_requires_artifact_hash_contract() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "render_zrave_midi_flow_audition.py"
    ).read_text(encoding="utf-8")

    assert '"--expected-checkpoint-sha256"' in script
    assert '"--expected-initializer-sha256"' in script
    assert '"--pitch-qualification"' in script
    assert '"--note-vocabulary"' in script
    assert "default=[36, 62, 82]" in script
    assert "swap_semitones" not in script
    assert "render_midi_flow_audition(" in script
    assert "expected_checkpoint_sha256=args.expected_checkpoint_sha256" in script
    assert "expected_initializer_sha256=args.expected_initializer_sha256" in script


def test_lvzihao_midi_audition_requires_both_independent_gates() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "lvzihao" / "midi_audition.sh"
    ).read_text(encoding="utf-8")

    generic = "python -m midibrave.zrave_flow_gate"
    midi = "python -m midibrave.zrave_midi_adherence_gate"
    assert generic in script
    assert midi in script
    assert script.index(generic) < script.index(midi) < script.index('mv -- "$partial"')
    assert '"$partial_container/midi-gate.json"' in script
    assert script.count("--fail-on-reject") >= 2


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
