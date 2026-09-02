#!/usr/bin/env python3
"""Summarize one lvzihao experiment queue without touching training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


_CHECKPOINT = re.compile(r"^step-([0-9]{6})\.pt$")
_TRAIN_ACTIONS = {"train", "midi_train_from"}
_EVALUATION_ACTIONS = {
    "audition",
    "audition_report",
    "midi_audition",
    "midi_audition_report",
    "midi_audition_from",
    "midi_audition_report_from",
}
_DISPLAY_SCALARS = (
    "train/loss",
    "train/flow",
    "train/boundary",
    "train/statistics",
    "train/temporal",
    "train/pitch",
    "train/learning_rate",
    "health/gradient_norm",
    "health/amp_scale",
    "train/valid_latent_frames_per_second",
    "diagnostics/batch/future_valid_fraction",
    "diagnostics/target/latent_frame_norm_p50",
    "diagnostics/target/motion_normalized_delta_rms_p50",
    "diagnostics/target/near_static_fraction",
    "diagnostics/target/normalized_coordinate_near_zero_fraction",
    "diagnostics/target/channel_observed_reference_std_ratio_p50",
    "diagnostics/estimate/latent_frame_norm_p50",
    "diagnostics/estimate/motion_normalized_delta_rms_p50",
    "diagnostics/estimate/near_static_fraction",
    "diagnostics/estimate/normalized_coordinate_near_zero_fraction",
    "diagnostics/estimate/channel_observed_reference_std_ratio_p50",
    "diagnostics/midi/transition_batch_fraction",
    "diagnostics/midi/event_present_fraction",
    "probe/pitch_matched_accuracy",
    "probe/pitch_transition_accuracy",
    "probe/pitch_matched_mean_absolute_cents",
    "probe/pitch_transition_mean_absolute_cents",
)


@dataclass(frozen=True)
class QueueRow:
    experiment_id: str
    action: str
    config_relative: str
    run_relative: str
    spec: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_queue(path: Path) -> list[QueueRow]:
    rows: list[QueueRow] = []
    identifiers: set[str] = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw or raw.startswith("#"):
            continue
        values = raw.split("\t")
        if len(values) != 5 or any(not value for value in values):
            raise ValueError(
                f"queue line {line_number} must have five non-empty TSV fields"
            )
        row = QueueRow(*values)
        if row.experiment_id in identifiers:
            raise ValueError(f"duplicate queue experiment id: {row.experiment_id}")
        identifiers.add(row.experiment_id)
        rows.append(row)
    if not rows:
        raise ValueError("queue contains no experiment rows")
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    current: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"metrics row is not an object: {path}")
            update = value.get("update")
            scalars = value.get("scalars")
            if (
                isinstance(update, bool)
                or not isinstance(update, int)
                or update < 0
                or not isinstance(scalars, dict)
            ):
                raise ValueError(f"metrics row contract is invalid: {path}")
            parsed_scalars: dict[str, float] = {}
            for name, scalar in scalars.items():
                if (
                    not isinstance(name, str)
                    or isinstance(scalar, bool)
                    or not isinstance(scalar, (int, float))
                    or not math.isfinite(float(scalar))
                ):
                    raise ValueError(
                        f"non-finite/invalid metrics scalar in {path}: {name}"
                    )
                parsed_scalars[name] = float(scalar)
            rollback = value.get("event") == "resume_rollback_to_checkpoint"
            if current is None or rollback or update > int(current["update"]):
                current = dict(value)
                current["scalars"] = parsed_scalars
            elif update == int(current["update"]):
                current["scalars"].update(parsed_scalars)
                for name in (
                    "event",
                    "timestamp_utc",
                    "wall_time_unix_seconds",
                    "orphaned_tail_update",
                ):
                    if name in value:
                        current[name] = value[name]
    return current


def _gate_state(path: Path) -> bool | None:
    value = _read_json(path)
    if value is None:
        return None
    passed = value.get("passed")
    if not isinstance(passed, bool):
        raise ValueError(f"gate lacks boolean passed field: {path}")
    return passed


def _generic_gate_envelope_summary(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if value is None:
        return None
    summary = value.get("summary")
    if summary is None:
        return None
    if not isinstance(summary, dict):
        raise ValueError(f"generic gate lacks summary object: {path}")
    envelope = summary.get("envelope_summary_proxy")
    if envelope is None:
        return None
    if not isinstance(envelope, dict) or envelope.get("report_only") is not True:
        raise ValueError(f"generic gate envelope proxy is invalid: {path}")
    return dict(envelope)


def _clap_report_summary(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if value is None:
        return None
    if (
        value.get("kind") != "zrave-frozen-clap-audio-preservation-report"
        or value.get("report_only") is not True
        or not isinstance(value.get("summary"), dict)
    ):
        raise ValueError(f"CLAP report contract is invalid: {path}")
    return dict(value["summary"])


def _qualification_state(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if value is None:
        return None
    status = value.get("qualification_status")
    qualified = value.get("qualified")
    if (
        value.get("schema") != 1
        or not isinstance(status, str)
        or not status
        or not isinstance(qualified, bool)
    ):
        raise ValueError(f"qualification sidecar contract is invalid: {path}")
    return {
        "status": status,
        "qualified": qualified,
        "midi_adherence_calibrated": value.get("midi_adherence_calibrated"),
    }


def _midi_probe_summary(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if value is None:
        return None
    if value.get("metric_kind") != "latent_pitch_probe_proxy":
        raise ValueError(f"MIDI probe gate metric kind is invalid: {path}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"MIDI probe gate lacks metrics: {path}")
    return dict(metrics)


def _decoded_midi_summary(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if value is None:
        return None
    if value.get("metric_kind") != "decoded_audio_crepe":
        raise ValueError(f"decoded MIDI gate metric kind is invalid: {path}")
    conditions = value.get("conditions")
    velocity = value.get("velocity_loudness_proxy")
    report_only = value.get("report_only_metrics")
    if (
        not isinstance(conditions, dict)
        or not isinstance(velocity, dict)
        or not isinstance(report_only, dict)
    ):
        raise ValueError(f"decoded MIDI gate summary is invalid: {path}")
    return {
        "conditions": conditions,
        "velocity_loudness_proxy": velocity,
        "report_only_metrics": report_only,
    }


def _checkpoint_updates(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    updates: list[int] = []
    for path in root.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match and path.is_file():
            updates.append(int(match.group(1)))
    return sorted(set(updates))


def build_matrix_snapshot(
    *,
    queue_path: str | Path,
    work_root: str | Path,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    queue = Path(queue_path).expanduser().resolve()
    work = Path(work_root).expanduser().resolve()
    rows = _read_queue(queue)
    queue_name = queue.stem
    state = (
        Path(state_root).expanduser().resolve()
        if state_root is not None
        else work / "lvzihao-state"
    )
    queue_state = state / "queues" / queue_name
    blocked = (queue_state / "blocked.txt").is_file()
    queue_complete = (queue_state / "complete.txt").is_file()
    done_root = queue_state / "done"

    train_rows: dict[str, QueueRow] = {}
    evaluations: dict[str, list[QueueRow]] = {}
    for row in rows:
        if row.action in _TRAIN_ACTIONS:
            previous = train_rows.setdefault(row.run_relative, row)
            if previous is not row:
                raise ValueError(f"multiple train rows target run {row.run_relative}")
        if row.action in _EVALUATION_ACTIONS:
            evaluations.setdefault(row.run_relative, []).append(row)

    runs: list[dict[str, Any]] = []
    for run_relative, train_row in train_rows.items():
        run_root = work / run_relative
        metrics = _last_jsonl(run_root / "metrics.jsonl")
        checkpoints = _checkpoint_updates(run_root / "checkpoints")
        all_scalar_values: dict[str, float] = {}
        scalar_subset: dict[str, float] = {}
        if metrics is not None:
            all_scalars = metrics["scalars"]
            all_scalar_values = {
                str(name): float(value) for name, value in all_scalars.items()
            }
            scalar_subset = {
                name: float(all_scalars[name])
                for name in _DISPLAY_SCALARS
                if name in all_scalars
            }
        evaluation_rows: list[dict[str, Any]] = []
        for row in evaluations.get(run_relative, []):
            output = work / "auditions" / row.experiment_id
            clap_path = work / "clap-reports" / f"{row.experiment_id}-clap.json"
            evaluation_rows.append(
                {
                    "experiment_id": row.experiment_id,
                    "action": row.action,
                    "checkpoint": row.spec,
                    "queue_done": (done_root / f"{row.experiment_id}.txt").is_file(),
                    "generic_gate": _gate_state(output / "gate.json"),
                    "envelope_proxy_summary": _generic_gate_envelope_summary(
                        output / "gate.json"
                    ),
                    "midi_probe_gate": _gate_state(output / "midi-gate.json"),
                    "midi_probe_summary": _midi_probe_summary(
                        output / "midi-gate.json"
                    ),
                    "decoded_audio_midi_gate": _gate_state(
                        output / "decoded-audio-midi-gate.json"
                    ),
                    "decoded_audio_midi_summary": _decoded_midi_summary(
                        output / "decoded-audio-midi-gate.json"
                    ),
                    "qualification": _qualification_state(
                        output / "qualification.json"
                    ),
                    "clap_report_present": clap_path.is_file(),
                    "clap_audio_audio_summary": _clap_report_summary(clap_path),
                }
            )
        runs.append(
            {
                "run": run_relative,
                "config": train_row.config_relative,
                "planned_max_updates": int(train_row.spec.split("|", 1)[0]),
                "train_queue_done": (
                    done_root / f"{train_row.experiment_id}.txt"
                ).is_file(),
                "latest_metrics_update": (
                    int(metrics["update"]) if metrics is not None else None
                ),
                "latest_metrics_timestamp_utc": (
                    metrics.get("timestamp_utc") if metrics is not None else None
                ),
                "telemetry_event": (
                    metrics.get("event") if metrics is not None else None
                ),
                "orphaned_tail_update": (
                    metrics.get("orphaned_tail_update") if metrics is not None else None
                ),
                "headline_scalars": scalar_subset,
                "scalars": all_scalar_values,
                "checkpoint_updates": checkpoints,
                "final_checkpoint_present": (
                    run_root / "checkpoints" / "final.pt"
                ).is_file(),
                "evaluations": evaluation_rows,
            }
        )

    return {
        "schema": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "queue": str(queue),
        "queue_sha256": _sha256_file(queue),
        "queue_state": str(queue_state),
        "queue_blocked": blocked,
        "queue_complete": queue_complete,
        "planned_runs": len(train_rows),
        "runs": runs,
    }


def _short(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def markdown_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        f"queue: `{snapshot['queue']}`",
        (
            f"state: blocked={snapshot['queue_blocked']} "
            f"complete={snapshot['queue_complete']} "
            f"generated={snapshot['generated_at_utc']}"
        ),
        "",
        "| run | update | loss | grad | frames/s | target static | estimate static | estimate near-zero | checkpoints | gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in snapshot["runs"]:
        scalars = run["headline_scalars"]
        gates: list[str] = []
        for evaluation in run["evaluations"]:
            generic = evaluation["generic_gate"]
            midi = evaluation["midi_probe_gate"]
            decoded = evaluation["decoded_audio_midi_gate"]
            if generic is None and midi is None and decoded is None:
                continue
            value = f"{evaluation['checkpoint']}:audio={_short(generic)}"
            if midi is not None:
                value += f",midi={_short(midi)}"
            if decoded is not None:
                value += f",decoded={_short(decoded)}"
            if evaluation["clap_report_present"]:
                value += ",clap=REPORT"
            qualification = evaluation["qualification"]
            if qualification is not None:
                value += f",status={qualification['status']}"
            gates.append(value)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(run["run"]),
                    _short(run["latest_metrics_update"]),
                    _short(scalars.get("train/loss")),
                    _short(scalars.get("health/gradient_norm")),
                    _short(scalars.get("train/valid_latent_frames_per_second")),
                    _short(scalars.get("diagnostics/target/near_static_fraction")),
                    _short(scalars.get("diagnostics/estimate/near_static_fraction")),
                    _short(
                        scalars.get(
                            "diagnostics/estimate/normalized_coordinate_near_zero_fraction"
                        )
                    ),
                    ",".join(str(value) for value in run["checkpoint_updates"]) or "-",
                    "; ".join(gates) or "-",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only status snapshot for a single-GPU training matrix."
    )
    parser.add_argument("--queue", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument(
        "--state-root",
        help="platform-specific queue state root (defaults to lvzihao-state)",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    snapshot = build_matrix_snapshot(
        queue_path=args.queue,
        work_root=args.work_root,
        state_root=args.state_root,
    )
    markdown = markdown_snapshot(snapshot)
    if args.json_output:
        _atomic_text(
            Path(args.json_output),
            json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if args.markdown_output:
        _atomic_text(Path(args.markdown_output), markdown)
    print(markdown, end="")


if __name__ == "__main__":
    main()
