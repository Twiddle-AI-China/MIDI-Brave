from __future__ import annotations

import json
from pathlib import Path

import pytest

from midibrave.zrave_midi_adherence_gate import (
    evaluate_midi_adherence_manifest,
    main,
    midi_adherence_gate,
)


def _window(
    note: int,
    cents: float = 20.0,
    *,
    start: int = 0,
) -> dict[str, object]:
    return {
        "start_frame": start,
        "midpoint_frame": start + 7,
        "requested_midi_note": note,
        "predicted_midi_class": note,
        "expected_midi": note + cents / 100.0,
        "absolute_cents": cents,
        "exact_class": True,
        "within_50_cents": cents <= 50.0,
        "within_100_cents": cents <= 100.0,
    }


def _summary(cents: float = 20.0) -> dict[str, object]:
    return {
        "window_count": 2,
        "exact_class_accuracy": 1.0,
        "absolute_cents_median": cents,
        "absolute_cents_p90": cents,
        "within_50_cents": float(cents <= 50.0),
        "within_100_cents": float(cents <= 100.0),
        "voiced_coverage": None,
    }


def _manifest(cents: float = 20.0) -> dict[str, object]:
    rollouts = []
    for kind, note in (("matched", 36), ("note_swap", 62)):
        rollouts.append(
            {
                "condition_kind": kind,
                "generation_seed": 17,
                "requested_notes": [note] * 32,
                "pitch_adherence_proxy": {
                    "metric_kind": "latent_pitch_probe_proxy",
                    "window_frames": 16,
                    "windows": [
                        _window(note, cents, start=0),
                        _window(note, cents, start=16),
                    ],
                    "summary": _summary(cents),
                },
            }
        )
    return {
        "pitch_probe": {
            "checkpoint_sha256": "a" * 64,
            "qualification_sha256": "b" * 64,
        },
        "controls": {"note_swap_mapping": {"36": 62}},
        "pitch_adherence_proxy": {
            "metric_kind": "latent_pitch_probe_proxy",
            "probe_checkpoint_sha256": "a" * 64,
            "probe_qualification_sha256": "b" * 64,
            "by_condition_kind": {
                "matched": _summary(cents),
                "note_swap": _summary(cents),
            },
            "overall": _summary(cents),
            "voiced_coverage": None,
        },
        "examples": [{"rollouts": rollouts}],
    }


def test_gate_accepts_complete_finite_matched_and_note_swap_proxy() -> None:
    report = midi_adherence_gate(_manifest())

    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["metric_kind"] == "latent_pitch_probe_proxy"
    assert "not decoded-audio/CREPE" in report["semantic_warning"]


def test_note_step_is_summarized_report_only_without_expanding_hard_gate() -> None:
    manifest = _manifest()
    manifest["latent_hop"] = 2048
    manifest["sample_rate"] = 44100
    manifest["examples"][0]["rollouts"].append(
        {
            "condition_kind": "note_step",
            "generation_seed": 17,
            "requested_notes": [36] * 16 + [62] * 16,
            "pitch_adherence_proxy": {
                "metric_kind": "latent_pitch_probe_proxy",
                "window_frames": 16,
                "windows": [
                    _window(36, start=0),
                    _window(62, start=16),
                ],
                "summary": _summary(),
            },
        }
    )

    report = midi_adherence_gate(manifest)

    assert report["passed"] is True
    panel = report["report_only_metrics"]["note_step"]
    assert panel["status"] == "measured"
    assert panel["hard_gate"] is False
    assert panel["transition"]["event_count"] == 1
    assert panel["transition"]["settled_event_count"] == 1
    assert panel["transition"]["events"][0]["settling_frames"] == 7


@pytest.mark.parametrize(
    ("mutation", "failed"),
    [
        ("bad_p90", "aggregate_consistency"),
        ("bad_within", "aggregate_consistency"),
        ("missing_kind", "aggregate_consistency"),
        ("unchanged_swap", "note_swap_changes_requested_note"),
        ("nonfinite", "finite_windows"),
        ("actual_bad_p90", "matched_p90_absolute_cents"),
        ("actual_bad_within", "note_swap_within_100_cents"),
    ],
)
def test_gate_rejects_incomplete_or_failed_proxy(
    mutation: str,
    failed: str,
) -> None:
    manifest = _manifest()
    if mutation == "bad_p90":
        manifest["pitch_adherence_proxy"]["by_condition_kind"]["matched"][
            "absolute_cents_p90"
        ] = 101.0
    elif mutation == "bad_within":
        manifest["pitch_adherence_proxy"]["by_condition_kind"]["note_swap"][
            "within_100_cents"
        ] = 0.89
    elif mutation == "missing_kind":
        del manifest["pitch_adherence_proxy"]["by_condition_kind"]["note_swap"]
    elif mutation == "unchanged_swap":
        rollout = manifest["examples"][0]["rollouts"][1]
        rollout["requested_notes"] = [36] * 32
    elif mutation == "nonfinite":
        manifest["examples"][0]["rollouts"][0]["pitch_adherence_proxy"]["windows"][0][
            "absolute_cents"
        ] = float("nan")
    elif mutation == "actual_bad_p90":
        windows = manifest["examples"][0]["rollouts"][0]["pitch_adherence_proxy"][
            "windows"
        ]
        for window in windows:
            window["absolute_cents"] = 120.0
            window["within_50_cents"] = False
            window["within_100_cents"] = False
        manifest["pitch_adherence_proxy"]["by_condition_kind"]["matched"] = _summary(
            120.0
        )
    else:
        windows = manifest["examples"][0]["rollouts"][1]["pitch_adherence_proxy"][
            "windows"
        ]
        windows[0]["absolute_cents"] = 120.0
        windows[0]["within_50_cents"] = False
        windows[0]["within_100_cents"] = False
        declared = _summary(20.0)
        declared["absolute_cents_median"] = 70.0
        declared["absolute_cents_p90"] = 110.0
        declared["within_50_cents"] = 0.5
        declared["within_100_cents"] = 0.5
        manifest["pitch_adherence_proxy"]["by_condition_kind"]["note_swap"] = declared

    report = midi_adherence_gate(manifest)
    assert report["passed"] is False
    assert failed in report["failed_checks"]


def test_evaluate_and_cli_write_hashed_report_and_fail_on_reject(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    report = evaluate_midi_adherence_manifest(manifest_path)
    assert report["passed"] is True
    assert len(report["manifest_sha256"]) == 64

    rejected = _manifest(cents=120.0)
    manifest_path.write_text(json.dumps(rejected), encoding="utf-8")
    output = tmp_path / "gate.json"
    with pytest.raises(SystemExit, match="1"):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--fail-on-reject",
            ]
        )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["passed"] is False
