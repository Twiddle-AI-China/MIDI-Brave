from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from midibrave import zrave_decoded_audio_midi_gate as decoded_gate
from midibrave.zrave_decoded_audio_midi_gate import (
    DecodedAudioCrepeConfig,
    PitchTrack,
    PitchTrackerContract,
    evaluate_decoded_audio_midi_manifest,
)


def _hash(character: str) -> str:
    return character * 64


def _hz(note: int, cents: float = 0.0) -> float:
    return 440.0 * 2.0 ** ((note - 69.0 + cents / 100.0) / 12.0)


def _contract() -> PitchTrackerContract:
    return PitchTrackerContract(
        implementation="injected-test-decoded_audio_crepe",
        package_version="0.0.24-test",
        package_code_sha256=_hash("a"),
        model_capacity="tiny",
        model_path="/offline/test/tiny.pth",
        model_sha256=_hash("b"),
    )


class _Tracks:
    def __init__(self, tracks: Sequence[PitchTrack]) -> None:
        self._tracks = iter(tracks)

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: DecodedAudioCrepeConfig,
    ) -> PitchTrack:
        assert audio.shape == (800,)
        assert sample_rate == 1000
        assert config.crepe_hop_length == 100
        return next(self._tracks)


def _track(
    frequencies: Sequence[float],
    periodicity: float | Sequence[float] = 1.0,
) -> PitchTrack:
    frequency = np.asarray(frequencies, dtype=np.float32)
    if isinstance(periodicity, (float, int)):
        periodicities = np.full(frequency.shape, periodicity, dtype=np.float32)
    else:
        periodicities = np.asarray(periodicity, dtype=np.float32)
    return PitchTrack(frequency_hz=frequency, periodicity=periodicities)


def _write_manifest(
    tmp_path: Path,
    *,
    matched_notes: Sequence[int] = (60,) * 8,
    swapped_notes: Sequence[int] = (72,) * 8,
    include_step_panels: bool = False,
) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    names = ["matched", "note-swap"]
    if include_step_panels:
        names.extend(("note-step", "velocity-step"))
    for name in names:
        # Two history frames followed by eight generated frames.
        audio = np.zeros(1000, dtype=np.float32)
        if name == "velocity-step":
            audio[200:600] = 0.1
            audio[600:1000] = 0.4
        sf.write(raw / f"{name}.wav", audio, 1000)
    rollouts = [
        {
            "condition_kind": "matched",
            "generation_seed": 17,
            "requested_notes": list(matched_notes),
            "raw_wav": "raw/matched.wav",
        },
        {
            "condition_kind": "note_swap",
            "generation_seed": 17,
            "requested_notes": list(swapped_notes),
            "raw_wav": "raw/note-swap.wav",
        },
    ]
    if include_step_panels:
        rollouts.extend(
            (
                {
                    "condition_kind": "note_step",
                    "generation_seed": 17,
                    "requested_notes": [60] * 4 + [72] * 4,
                    "requested_velocities": [54] * 8,
                    "raw_wav": "raw/note-step.wav",
                },
                {
                    "condition_kind": "velocity_step",
                    "generation_seed": 17,
                    "requested_notes": [60] * 8,
                    "requested_velocities": [54] * 4 + [108] * 4,
                    "raw_wav": "raw/velocity-step.wav",
                },
            )
        )
    manifest = {
        "schema": 1,
        "conditioning": "midi_sequence",
        "sample_rate": 1000,
        "latent_hop": 100,
        "history_frames": 2,
        "generated_frames": 8,
        "config": {"path": "/config.yaml", "sha256": _hash("c")},
        "checkpoint": {
            "path": "/checkpoint.pt",
            "sha256": _hash("d"),
            "initialization": {"checkpoint_sha256": _hash("2")},
        },
        "pitch_probe": {
            "checkpoint_sha256": _hash("3"),
            "qualification_sha256": _hash("4"),
        },
        "codec": {"path": "/codec.ts", "sha256": _hash("e")},
        "pack_index_sha256": _hash("f"),
        "statistics_sha256": _hash("1"),
        "examples": [
            {
                "sample_id": "serum:test",
                "history_requested_notes": [60, 60],
                "rollouts": rollouts,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _config(**overrides: object) -> DecodedAudioCrepeConfig:
    values: dict[str, object] = {
        "crepe_hop_length": 100,
        "fmin_hz": 50.0,
        "fmax_hz": 2000.0,
        "model_capacity": "tiny",
        "periodicity_threshold": 0.5,
        "settling_consecutive_frames": 2,
        "maximum_p90_absolute_cents": 100.0,
        "minimum_within_100_cents": 0.90,
    }
    values.update(overrides)
    return DecodedAudioCrepeConfig(**values)


def _evaluate(
    manifest: Path,
    tracks: Sequence[PitchTrack],
    *,
    config: DecodedAudioCrepeConfig | None = None,
) -> dict[str, object]:
    return evaluate_decoded_audio_midi_manifest(
        manifest,
        config=config or _config(),
        pitch_tracker=_Tracks(tracks),
        tracker_contract=_contract(),
    )


def test_decoded_audio_crepe_passes_voiced_matched_and_note_swap(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    # Official padded prediction has an endpoint frame; it is deliberately
    # removed because its center lies outside the generated interval.
    matched = _track([_hz(60)] * 9)
    swapped = _track([_hz(72)] * 9)

    report = _evaluate(manifest, [matched, swapped])

    assert report["metric_kind"] == "decoded_audio_crepe"
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["conditions"]["matched"]["rollout_count"] == 1
    matched_aggregate = report["conditions"]["matched"]["aggregate"]
    assert matched_aggregate["finite_analysis_frame_count"] == 8
    assert matched_aggregate["voiced_frame_count"] == 8
    assert matched_aggregate["voiced_coverage"] == pytest.approx(1.0)
    assert matched_aggregate["within_100_cents"] == pytest.approx(1.0)
    assert report["analysis_scope"]["audio_region"] == "generated_only"
    boundary = report["rollouts"][1]["metrics"]["transition"]["events"][0]
    assert boundary["event_kind"] == "history_to_generated_boundary"
    assert boundary["from_midi_note"] == 60
    assert boundary["to_midi_note"] == 72
    assert boundary["settling_ms"] == pytest.approx(0.0)


def test_frame_mapping_reproduces_torchcrepe_resample_hop_without_drift() -> None:
    offsets = decoded_gate._official_frame_offsets_samples(
        1281,
        sample_rate=44100,
        requested_hop_length=512,
    )

    # torchcrepe truncates 512 * 16000 / 44100 to an internal 185-sample hop.
    assert offsets[1] == pytest.approx(185 * 44100 / 16000)
    assert offsets[-1] == pytest.approx(652680.0)
    assert offsets[-1] != 1280 * 512


def test_silence_is_reported_and_rejected_without_hiding_finite_frames(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    silent_matched = _track([_hz(60)] * 9, periodicity=0.0)
    silent_swapped = _track([_hz(72)] * 9, periodicity=0.0)

    report = _evaluate(manifest, [silent_matched, silent_swapped])

    assert report["passed"] is False
    aggregate = report["conditions"]["matched"]["aggregate"]
    assert aggregate["finite_analysis_frame_count"] == 8
    assert aggregate["voiced_frame_count"] == 0
    assert aggregate["voiced_coverage"] == 0.0
    assert "matched_voiced_measurement_coverage" in report["failed_checks"]


def test_wrong_octave_is_counted_and_fails_pitch_thresholds(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    wrong_matched = _track([_hz(72)] * 9)
    wrong_swapped = _track([_hz(84)] * 9)

    report = _evaluate(manifest, [wrong_matched, wrong_swapped])

    assert report["passed"] is False
    aggregate = report["conditions"]["matched"]["aggregate"]
    assert aggregate["absolute_cents_p90"] == pytest.approx(1200.0, abs=0.01)
    assert aggregate["within_100_cents"] == 0.0
    assert aggregate["octave_error_fraction"] == 1.0
    assert "matched_p90_absolute_cents" in report["failed_checks"]


def test_note_change_reports_consecutive_frame_settling(tmp_path: Path) -> None:
    notes = (60, 60, 60, 60, 72, 72, 72, 72)
    manifest = _write_manifest(
        tmp_path,
        matched_notes=notes,
        swapped_notes=notes,
    )
    # At 400 ms the request changes. Two stale frames precede two consecutive
    # voiced in-tune frames, so the settling start is 600 ms => 200 ms.
    frequencies = [_hz(60)] * 6 + [_hz(72)] * 3

    report = _evaluate(
        manifest,
        [_track(frequencies), _track(frequencies)],
    )

    transition = report["conditions"]["matched"]["transition"]
    assert transition["event_count"] == 1
    assert transition["settled_event_count"] == 1
    assert transition["settling_ms"]["median"] == pytest.approx(200.0)
    rollout_event = report["rollouts"][0]["metrics"]["transition"]["events"][0]
    assert rollout_event["from_midi_note"] == 60
    assert rollout_event["to_midi_note"] == 72
    assert rollout_event["settling_ms"] == pytest.approx(200.0)


def test_note_and_velocity_step_are_report_only_monitoring_panels(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path, include_step_panels=True)
    note_step_frequencies = [_hz(60)] * 6 + [_hz(72)] * 3

    report = _evaluate(
        manifest,
        [
            _track([_hz(60)] * 9),
            _track([_hz(72)] * 9),
            _track(note_step_frequencies),
        ],
    )

    assert report["passed"] is True
    note_step = report["conditions"]["note_step"]
    assert note_step["transition"]["event_count"] == 1
    assert note_step["transition"]["settled_event_count"] == 1
    assert note_step["transition"]["settling_ms"]["median"] == pytest.approx(200.0)
    note_panel = report["report_only_metrics"]["note_step_transition_settling_ms"]
    assert note_panel["status"] == "measured"
    assert note_panel["hard_gate"] is False

    velocity = report["velocity_loudness_proxy"]
    assert velocity["status"] == "measured"
    assert velocity["report_only"] is True
    assert velocity["requested_direction_consistency_fraction"] == 1.0
    assert velocity["events"][0]["from_velocity"] == 54
    assert velocity["events"][0]["to_velocity"] == 108
    assert velocity["events"][0]["rms_delta_db"] == pytest.approx(
        20.0 * np.log10(4.0), abs=0.01
    )
    assert "not MIDI velocity accuracy" in velocity["semantic_warning"]


def test_constant_control_marks_transition_metric_not_applicable(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    report = _evaluate(
        manifest,
        [_track([_hz(60)] * 9), _track([_hz(72)] * 9)],
    )

    transition = report["conditions"]["matched"]["transition"]
    assert transition["event_count"] == 0
    assert transition["settling_ms"]["count"] == 0
    assert transition["settling_ms"]["median"] is None
    assert transition["status"] == "not_applicable_no_requested_note_change"


def test_report_records_manifest_audio_tool_model_and_config_hashes(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    report = _evaluate(
        manifest,
        [_track([_hz(60)] * 9), _track([_hz(72)] * 9)],
    )

    assert (
        report["lineage"]["audition_manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert report["manifest_sha256"] == report["lineage"]["audition_manifest_sha256"]
    assert report["lineage"]["audition_config_sha256"] == _hash("c")
    assert report["lineage"]["generator_checkpoint_sha256"] == _hash("d")
    assert report["lineage"]["initializer_checkpoint_sha256"] == _hash("2")
    assert report["lineage"]["pitch_probe_checkpoint_sha256"] == _hash("3")
    assert len(report["tool"]["module_sha256"]) == 64
    assert report["tracker"]["model_sha256"] == _hash("b")
    assert len(report["evaluation_config_sha256"]) == 64
    raw_path = tmp_path / report["rollouts"][0]["raw_wav"]
    assert (
        report["rollouts"][0]["raw_wav_sha256"]
        == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    )


def test_custom_p90_threshold_is_a_hard_check_but_voicing_is_report_only(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    matched = _track([_hz(60, 80.0)] * 9)
    swapped = _track([_hz(72, 80.0)] * 9)

    accepted = _evaluate(manifest, [matched, swapped])
    rejected = _evaluate(
        manifest,
        [_track([_hz(60, 80.0)] * 9), _track([_hz(72, 80.0)] * 9)],
        config=_config(maximum_p90_absolute_cents=50.0),
    )

    assert accepted["passed"] is True
    assert rejected["passed"] is False
    assert "matched_p90_absolute_cents" in rejected["failed_checks"]
    assert "voiced_coverage" in rejected["report_only_metrics"]


@pytest.mark.parametrize("raw_wav", ["../outside.wav", "/tmp/outside.wav"])
def test_manifest_rejects_unsafe_audio_paths(tmp_path: Path, raw_wav: str) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["examples"][0]["rollouts"][0]["raw_wav"] = raw_wav
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="safe relative path"):
        _evaluate(
            manifest,
            [_track([_hz(60)] * 9), _track([_hz(72)] * 9)],
        )


def test_missing_authoritative_manifest_hash_is_a_hard_integrity_error(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["codec"]["sha256"] = None
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="codec_sha256"):
        _evaluate(
            manifest,
            [_track([_hz(60)] * 9), _track([_hz(72)] * 9)],
        )


def test_missing_condition_is_a_hard_panel_integrity_error(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["examples"][0]["rollouts"] = payload["examples"][0]["rollouts"][:1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="note_swap"):
        _evaluate(manifest, [_track([_hz(60)] * 9)])


def test_cli_fail_on_reject_still_materializes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "decoded-audio-midi-gate.json"
    monkeypatch.setattr(
        decoded_gate,
        "evaluate_decoded_audio_midi_manifest",
        lambda *_args, **_kwargs: {
            "passed": False,
            "metric_kind": "decoded_audio_crepe",
        },
    )

    with pytest.raises(SystemExit, match="1"):
        decoded_gate.main(
            [
                "--manifest",
                str(tmp_path / "unused.json"),
                "--output",
                str(output),
                "--fail-on-reject",
            ]
        )
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
