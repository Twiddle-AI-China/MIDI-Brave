from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from midibrave.zrave_flow_gate import (
    GateThresholds,
    envelope_proxy_metrics,
    evaluate_manifest,
    load_audition_triplets,
    main,
)


def _enveloped_tone(
    sample_rate: int,
    seconds: float,
    envelope: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    amplitude = envelope(time)
    return (0.5 * np.asarray(amplitude) * np.sin(2.0 * np.pi * 220.0 * time)).astype(
        np.float32
    )


def _moving_tone(
    sample_rate: int,
    seconds: float,
    *,
    seed: int,
) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    frequency = 180.0 + seed * 7.0 + 28.0 * np.sin(2.0 * np.pi * 0.31 * time)
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    amplitude = 0.22 * (1.0 + 0.30 * np.sin(2.0 * np.pi * (0.43 + seed * 0.01) * time))
    return (amplitude * np.sin(phase)).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sample_rate: int) -> str:
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path.name


def _manifest(
    tmp_path: Path,
    generated: list[np.ndarray],
    *,
    latent_hop: int = 200,
    block_stride_frames: int = 8,
) -> Path:
    sample_rate = 8000
    source = _moving_tone(sample_rate, generated[0].size / sample_rate, seed=0)
    source_path = _write(tmp_path / "source.wav", source, sample_rate)
    direct_path = _write(tmp_path / "direct.wav", source * 0.98, sample_rate)
    rows = []
    for index, audio in enumerate(generated):
        path = _write(tmp_path / f"generated-{index}.wav", audio, sample_rate)
        rows.append(
            {
                "id": f"pad-seed-{index}",
                "sample_id": "pad-a",
                "category": "Pad",
                "generation_seed": index,
                "source": source_path,
                "direct": direct_path,
                "generated": path,
            }
        )
    payload = {
        "schema": 1,
        "latent_dim": 128,
        "sample_rate": sample_rate,
        "latent_hop": latent_hop,
        "seed_boundary_seconds": 0.2,
        "block_stride_frames": block_stride_frames,
        "triplets": rows,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _permissive_thresholds(**changes: object) -> GateThresholds:
    values: dict[str, object] = {
        "maximum_silence_ratio": 1.0,
        "maximum_absolute_tail_rms_drift_db": 300.0,
        "maximum_static_tone_fraction": 1.0,
        "maximum_short_cycle_correlation": 1.0,
        "maximum_block_boundary_jump_ratio": 1.0e9,
        "minimum_cross_seed_waveform_nrmse": 0.0,
        "minimum_cross_seed_groups": 1,
        "tail_window_seconds": 0.5,
        "cycle_tail_seconds": 4.0,
    }
    values.update(changes)
    return GateThresholds.from_mapping(values)


def test_generic_triplets_produce_json_safe_passing_report(
    tmp_path: Path,
) -> None:
    first = _moving_tone(8000, 4.2, seed=1)
    second = _moving_tone(8000, 4.2, seed=2)
    manifest = _manifest(tmp_path, [first, second])

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(
            maximum_silence_ratio=0.01,
            maximum_absolute_tail_rms_drift_db=12.0,
            maximum_block_boundary_jump_ratio=4.0,
            minimum_cross_seed_waveform_nrmse=0.01,
        ),
    )

    assert report["passed"] is True
    assert report["latent_dim"] == 128
    assert report["manifest_format"] == "generic-triplets"
    summary = report["summary"]
    assert summary["triplets"] == 2
    assert summary["analyzed_rollouts"] == 2
    assert summary["cross_seed_diversity"]["groups"] == 1
    assert summary["cross_seed_diversity"]["pairs"] == 1
    json.dumps(report, allow_nan=False)


def test_silent_tail_fails_silence_and_rms_drift_checks(
    tmp_path: Path,
) -> None:
    audio = _moving_tone(8000, 4.2, seed=1)
    audio[-16000:] = 0.0
    other = audio.copy()
    other[:8000] *= -1.0
    manifest = _manifest(tmp_path, [audio, other])

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(
            maximum_silence_ratio=0.10,
            maximum_absolute_tail_rms_drift_db=20.0,
        ),
    )

    checks = report["acceptance"]["checks"]
    assert checks["silence_ratio"]["passed"] is False
    assert checks["absolute_tail_rms_drift_db"]["passed"] is False
    assert report["passed"] is False


def test_nonfinite_float_wav_is_counted_and_rejected(tmp_path: Path) -> None:
    first = _moving_tone(8000, 2.0, seed=1)
    second = first.copy()
    second[100] = np.nan
    manifest = _manifest(tmp_path, [first, second])

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(),
    )

    assert report["summary"]["nonfinite_files"] == 1
    assert report["rows"][1]["envelope_takes_proxy"]["generated"] is None
    assert report["acceptance"]["checks"]["finite"]["passed"] is False
    assert report["acceptance"]["checks"]["complete_metrics"]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_short_repeating_tail_trips_cycle_proxy(tmp_path: Path) -> None:
    generator = np.random.default_rng(7)
    frame = generator.normal(0.0, 0.15, size=200).astype(np.float32)
    repeated = np.tile(frame, 168)
    other_frame = generator.normal(0.0, 0.15, size=200).astype(np.float32)
    other = np.tile(other_frame, 168)
    manifest = _manifest(tmp_path, [repeated, other])

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(
            maximum_short_cycle_correlation=0.95,
        ),
    )

    cycle = report["summary"]["maximum_short_cycle_correlation"]["maximum"]
    assert cycle == pytest.approx(1.0)
    assert report["acceptance"]["checks"]["short_cycle_correlation"]["passed"] is False


def test_block_discontinuities_are_normalized_against_ordinary_motion(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    latent_hop = 200
    stride = 8
    block = latent_hop * stride
    first = _moving_tone(sample_rate, 4.2, seed=1)
    second = _moving_tone(sample_rate, 4.2, seed=2)
    seed_samples = round(0.2 * sample_rate)
    for audio in (first, second):
        for boundary_index, start in enumerate(
            range(seed_samples + block, audio.size, block), 1
        ):
            audio[start:] += 0.35 if boundary_index % 2 else -0.35
    manifest = _manifest(
        tmp_path,
        [first, second],
        latent_hop=latent_hop,
        block_stride_frames=stride,
    )

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(
            maximum_block_boundary_jump_ratio=4.0,
        ),
    )

    observed = report["summary"]["block_boundary_jump_ratio"]["maximum"]
    assert observed > 4.0
    assert (
        report["acceptance"]["checks"]["block_boundary_jump_ratio"]["passed"] is False
    )


def test_identical_seeds_fail_cross_seed_diversity(tmp_path: Path) -> None:
    audio = _moving_tone(8000, 4.2, seed=1)
    manifest = _manifest(tmp_path, [audio, audio.copy()])

    report = evaluate_manifest(
        manifest,
        thresholds=_permissive_thresholds(
            minimum_cross_seed_waveform_nrmse=0.01,
        ),
    )

    diversity = report["summary"]["cross_seed_diversity"]
    assert diversity["minimum_group_median_waveform_nrmse"] == 0.0
    assert (
        report["acceptance"]["checks"]["cross_seed_waveform_nrmse"]["passed"] is False
    )


def test_native_flow_manifest_adapter_uses_raw_triplets(tmp_path: Path) -> None:
    audio = _moving_tone(8000, 1.0, seed=1)
    source = _write(tmp_path / "source.wav", audio, 8000)
    direct = _write(tmp_path / "direct.wav", audio, 8000)
    generated_17 = _write(tmp_path / "g17.wav", audio, 8000)
    generated_29 = _write(tmp_path / "g29.wav", -audio, 8000)
    payload = {
        "schema": 1,
        "sample_rate": 8000,
        "latent_hop": 200,
        "context_frames": 4,
        "commit_stride_frames": 2,
        "codec": {"latent_dim": 128},
        "examples": [
            {
                "sample_id": "pad-a",
                "category": "Pad",
                "source": {"raw_wav": source},
                "direct": {"raw_wav": direct},
                "rollouts": [
                    {
                        "raw_wav": generated_17,
                        "generation_seed": 17,
                        "exploration": 0.5,
                    },
                    {
                        "raw_wav": generated_29,
                        "generation_seed": 29,
                        "exploration": 0.5,
                    },
                ],
            }
        ],
    }
    manifest = tmp_path / "flow-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    _, triplets = load_audition_triplets(manifest)

    assert len(triplets) == 2
    assert {triplet.generation_seed for triplet in triplets} == {17, 29}
    assert len({triplet.group_id for triplet in triplets}) == 1
    assert all(
        triplet.seed_boundary_seconds == pytest.approx(0.1) for triplet in triplets
    )


def test_cli_writes_report_and_can_return_rejected_result(tmp_path: Path) -> None:
    audio = _moving_tone(8000, 2.0, seed=1)
    manifest = _manifest(tmp_path, [audio, audio.copy()])
    output = tmp_path / "gate.json"

    main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--no-fail-on-reject",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "zrave-128d-long-rollout-acceptance-gate"
    assert report["passed"] is False


def test_envelope_proxy_is_sample_rate_invariant_and_finite() -> None:
    observed = []
    for sample_rate in (8000, 16000):
        audio = _enveloped_tone(
            sample_rate,
            1.0,
            lambda time: np.minimum(time / 0.1, 1.0),
        )
        metrics = envelope_proxy_metrics(audio, sample_rate)
        observed.append(metrics)
        for value in metrics.values():
            if isinstance(value, float):
                assert math.isfinite(value)
        json.dumps(metrics, allow_nan=False)

    first, second = observed
    assert first["envelope_attack_ms_proxy"] == pytest.approx(
        second["envelope_attack_ms_proxy"],
        abs=5.1,
    )
    assert first["envelope_sustain_rms_dbfs_proxy"] == pytest.approx(
        second["envelope_sustain_rms_dbfs_proxy"],
        abs=0.02,
    )
    assert first["envelope_tail_release_slope_db_per_second_proxy"] == (
        pytest.approx(
            second["envelope_tail_release_slope_db_per_second_proxy"],
            abs=0.02,
        )
    )


def test_envelope_proxy_handles_silence_short_audio_and_nonfinite() -> None:
    silent = envelope_proxy_metrics(np.zeros(3, dtype=np.float32), 8000)
    assert silent["envelope_silent_proxy"] is True
    assert silent["envelope_attack_crossing_found_proxy"] is False
    assert silent["envelope_attack_ms_proxy"] is None
    assert silent["envelope_sustain_rms_dbfs_proxy"] == -120.0
    assert silent["envelope_tail_release_slope_db_per_second_proxy"] == 0.0

    short = envelope_proxy_metrics(np.asarray([0.1], dtype=np.float32), 8000)
    assert short["envelope_analysis_samples_proxy"] == 1
    assert short["envelope_frame_count_proxy"] == 1
    assert short["envelope_attack_ms_proxy"] == 0.0
    assert short["envelope_tail_release_slope_db_per_second_proxy"] == 0.0
    json.dumps({"silent": silent, "short": short}, allow_nan=False)

    with pytest.raises(ValueError, match="must be finite"):
        envelope_proxy_metrics(np.asarray([0.0, np.nan]), 8000)


def test_envelope_proxy_distinguishes_transient_sustain_and_tail_release() -> None:
    sample_rate = 8000
    transient = envelope_proxy_metrics(
        _enveloped_tone(
            sample_rate,
            1.0,
            lambda time: np.minimum(time / 0.01, 1.0) * np.exp(-6.0 * time),
        ),
        sample_rate,
    )
    sustained = envelope_proxy_metrics(
        _enveloped_tone(
            sample_rate,
            1.0,
            lambda time: np.minimum(time / 0.01, 1.0),
        ),
        sample_rate,
    )
    released = envelope_proxy_metrics(
        _enveloped_tone(
            sample_rate,
            1.0,
            lambda time: np.minimum(time / 0.01, 1.0)
            * np.where(time < 0.7, 1.0, np.maximum(0.0, (1.0 - time) / 0.3)),
        ),
        sample_rate,
    )

    assert transient["envelope_peak_to_sustain_drop_db_proxy"] > 15.0
    assert sustained["envelope_peak_to_sustain_drop_db_proxy"] < 1.0
    assert transient["envelope_decay_slope_db_per_second_proxy"] < -20.0
    assert sustained["envelope_decay_slope_db_per_second_proxy"] > -1.0
    assert released["envelope_tail_release_slope_db_per_second_proxy"] < -20.0
    assert sustained["envelope_tail_release_slope_db_per_second_proxy"] > -1.0


def test_gate_report_persists_per_take_envelope_proxy_distributions(
    tmp_path: Path,
) -> None:
    first = _moving_tone(8000, 2.0, seed=1)
    second = _moving_tone(8000, 2.0, seed=2)
    manifest = _manifest(tmp_path, [first, second])

    report = evaluate_manifest(manifest, thresholds=_permissive_thresholds())

    algorithm = report["envelope_algorithm_proxy"]
    assert algorithm["report_only"] is True
    assert "not true ADSR" in algorithm["claim"]
    assert algorithm["frame_seconds"] == 0.020
    assert algorithm["hop_seconds"] == 0.005
    for row in report["rows"]:
        takes = row["envelope_takes_proxy"]
        assert set(takes) == {"source", "direct", "generated"}
        for metrics in takes.values():
            assert metrics["envelope_frame_ms_proxy"] == 20.0
            assert metrics["envelope_sustain_window_start_seconds_proxy"] > 0.0
            json.dumps(metrics, allow_nan=False)
    summary = report["summary"]["envelope_summary_proxy"]
    assert summary["report_only"] is True
    assert (
        summary["take_distributions"]["generated"]["envelope_sustain_rms_dbfs_proxy"][
            "count"
        ]
        == 2
    )
    assert (
        summary["generated_absolute_difference_distributions"]["generated_vs_source"][
            "envelope_attack_ms_proxy"
        ]["count"]
        == 2
    )
    assert not any(
        name.startswith("envelope_") for name in report["acceptance"]["checks"]
    )
    json.dumps(report, allow_nan=False)
