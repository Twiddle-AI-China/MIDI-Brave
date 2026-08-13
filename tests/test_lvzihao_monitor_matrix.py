from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.lvzihao.monitor_matrix import (
    build_matrix_snapshot,
    markdown_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_matrix_snapshot_collects_metrics_checkpoints_and_gates(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "matrix.tsv"
    _write(
        queue,
        "# comment\n"
        "pad-train\ttrain\tconfigs/pad.yaml\truns/pad\t10000\n"
        "pad-1k\taudition_report\tconfigs/pad.yaml\truns/pad\tstep-001000.pt\n",
    )
    work = tmp_path / "work"
    _write(
        work / "runs/pad/metrics.jsonl",
        json.dumps(
            {
                "schema": 1,
                "update": 20,
                "timestamp_utc": "2026-08-13T00:00:00Z",
                "wall_time_unix_seconds": 1.0,
                "scalars": {
                    "train/loss": 0.25,
                    "health/gradient_norm": 0.5,
                    "diagnostics/target/near_static_fraction": 0.125,
                    "diagnostics/estimate/near_static_fraction": 0.25,
                    "diagnostics/estimate/normalized_coordinate_near_zero_fraction": 0.5,
                },
            }
        )
        + "\n",
    )
    _write(work / "runs/pad/checkpoints/step-001000.pt", "checkpoint")
    _write(work / "runs/pad/checkpoints/final.pt", "checkpoint")
    _write(
        work / "auditions/pad-1k/gate.json",
        json.dumps(
            {
                "passed": False,
                "summary": {
                    "envelope_summary_proxy": {
                        "report_only": True,
                        "take_distributions": {},
                    }
                },
            }
        )
        + "\n",
    )
    _write(
        work / "lvzihao-state/queues/matrix/done/pad-train.txt",
        "done\n",
    )
    _write(
        work / "auditions/pad-1k/qualification.json",
        '{"schema":1,"qualification_status":"research_only","qualified":false}\n',
    )

    snapshot = build_matrix_snapshot(queue_path=queue, work_root=work)

    assert snapshot["planned_runs"] == 1
    run = snapshot["runs"][0]
    assert run["latest_metrics_update"] == 20
    assert run["scalars"]["diagnostics/target/near_static_fraction"] == 0.125
    assert run["headline_scalars"]["diagnostics/estimate/near_static_fraction"] == 0.25
    assert run["headline_scalars"]["train/loss"] == 0.25
    assert run["checkpoint_updates"] == [1000]
    assert run["final_checkpoint_present"] is True
    assert run["train_queue_done"] is True
    assert run["evaluations"][0]["generic_gate"] is False
    assert run["evaluations"][0]["envelope_proxy_summary"]["report_only"] is True
    assert run["evaluations"][0]["qualification"] == {
        "status": "research_only",
        "qualified": False,
        "midi_adherence_calibrated": None,
    }
    rendered = markdown_snapshot(snapshot)
    assert "runs/pad" in rendered
    assert "step-001000.pt:audio=FAIL" in rendered


def test_matrix_snapshot_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    queue = tmp_path / "matrix.tsv"
    _write(queue, "pad-train\ttrain\tc.yaml\truns/pad\t10000\n")
    _write(
        tmp_path / "work/runs/pad/metrics.jsonl",
        '{"update":1,"scalars":{"train/loss":NaN}}\n',
    )

    with pytest.raises(ValueError, match="non-finite"):
        build_matrix_snapshot(queue_path=queue, work_root=tmp_path / "work")


def test_matrix_snapshot_rejects_duplicate_train_run(tmp_path: Path) -> None:
    queue = tmp_path / "matrix.tsv"
    _write(
        queue,
        "pad-a\ttrain\ta.yaml\truns/pad\t1000\npad-b\ttrain\tb.yaml\truns/pad\t2000\n",
    )

    with pytest.raises(ValueError, match="multiple train rows"):
        build_matrix_snapshot(queue_path=queue, work_root=tmp_path / "work")


def test_monitor_shell_is_login_host_read_only_wrapper() -> None:
    script = (ROOT / "scripts/lvzihao/monitor_matrix.sh").read_text(encoding="utf-8")

    assert "monitor_matrix.py" in script
    assert "--json-output" in script
    assert "--markdown-output" in script
    assert "qgpu" not in script
    assert "docker" not in script


def test_matrix_snapshot_understands_initializer_bound_midi_rows(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "midi.tsv"
    initializer = "runs/tiny-pure/pad/checkpoints/step-010000.pt"
    _write(
        queue,
        (
            "pad-train\tmidi_train_from\tc.yaml\truns/midi/pad\t"
            f"10000|{initializer}\n"
            "pad-report\tmidi_audition_report_from\tc.yaml\truns/midi/pad\t"
            f"step-010000.pt|{initializer}\n"
        ),
    )
    work = tmp_path / "work"
    _write(work / "auditions/pad-report/gate.json", '{"passed": true}\n')
    _write(
        work / "auditions/pad-report/midi-gate.json",
        '{"passed":false,"metric_kind":"latent_pitch_probe_proxy",'
        '"metrics":{"by_condition_kind":{"matched":{"window_count":4}}}}\n',
    )
    _write(
        work / "auditions/pad-report/decoded-audio-midi-gate.json",
        '{"passed":true,"metric_kind":"decoded_audio_crepe",'
        '"conditions":{"matched":{"aggregate":{"voiced_coverage":0.9}}},'
        '"velocity_loudness_proxy":{'
        '"requested_direction_consistency_fraction":0.75},'
        '"report_only_metrics":{"note_step_transition_settling_ms":{}}}\n',
    )
    _write(
        work / "auditions/pad-report/qualification.json",
        '{"schema":1,'
        '"qualification_status":"provisional_gates_passed_uncalibrated",'
        '"qualified":false,"midi_adherence_calibrated":false}\n',
    )
    _write(
        work / "clap-reports/pad-report-clap.json",
        json.dumps(
            {
                "kind": "zrave-frozen-clap-audio-preservation-report",
                "report_only": True,
                "summary": {
                    "generated_vs_source_audio_cosine": {
                        "count": 2,
                        "median": 0.8,
                    }
                },
            }
        )
        + "\n",
    )

    snapshot = build_matrix_snapshot(queue_path=queue, work_root=work)

    run = snapshot["runs"][0]
    assert run["planned_max_updates"] == 10000
    assert run["evaluations"][0]["generic_gate"] is True
    assert run["evaluations"][0]["midi_probe_gate"] is False
    assert run["evaluations"][0]["decoded_audio_midi_gate"] is True
    assert (
        run["evaluations"][0]["midi_probe_summary"]["by_condition_kind"]["matched"][
            "window_count"
        ]
        == 4
    )
    assert (
        run["evaluations"][0]["decoded_audio_midi_summary"]["velocity_loudness_proxy"][
            "requested_direction_consistency_fraction"
        ]
        == 0.75
    )
    assert run["evaluations"][0]["qualification"]["qualified"] is False
    assert run["evaluations"][0]["clap_report_present"] is True
    assert (
        run["evaluations"][0]["clap_audio_audio_summary"][
            "generated_vs_source_audio_cosine"
        ]["median"]
        == 0.8
    )


def test_matrix_snapshot_exposes_append_only_resume_rollback(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "matrix.tsv"
    _write(queue, "pad-train\ttrain\tc.yaml\truns/pad\t10000\n")
    _write(
        tmp_path / "work/runs/pad/metrics.jsonl",
        '{"schema":1,"update":1020,"scalars":{"train/loss":0.3}}\n'
        '{"schema":1,"event":"resume_rollback_to_checkpoint",'
        '"update":1000,"orphaned_tail_update":1020,"scalars":{}}\n',
    )

    snapshot = build_matrix_snapshot(
        queue_path=queue,
        work_root=tmp_path / "work",
    )

    run = snapshot["runs"][0]
    assert run["latest_metrics_update"] == 1000
    assert run["telemetry_event"] == "resume_rollback_to_checkpoint"
    assert run["orphaned_tail_update"] == 1020


def test_matrix_snapshot_merges_train_validation_and_health_same_update(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "matrix.tsv"
    _write(queue, "pad-train\ttrain\tc.yaml\truns/pad\t10000\n")
    _write(
        tmp_path / "work/runs/pad/metrics.jsonl",
        '{"schema":1,"update":1000,"scalars":{"train/loss":0.3}}\n'
        '{"schema":1,"event":"validation","update":1000,'
        '"scalars":{"validation/total":0.4}}\n'
        '{"schema":1,"event":"nonfinite_gradient_skip","update":1000,'
        '"scalars":{"health/nonfinite_gradient_skips":1}}\n',
    )

    snapshot = build_matrix_snapshot(
        queue_path=queue,
        work_root=tmp_path / "work",
    )

    scalars = snapshot["runs"][0]["scalars"]
    assert scalars == {
        "train/loss": 0.3,
        "validation/total": 0.4,
        "health/nonfinite_gradient_skips": 1.0,
    }
