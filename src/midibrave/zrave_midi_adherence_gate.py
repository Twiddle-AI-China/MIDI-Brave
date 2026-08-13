from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MidiAdherenceThresholds:
    maximum_p90_absolute_cents: float = 100.0
    minimum_within_100_cents: float = 0.90
    required_condition_kinds: tuple[str, ...] = ("matched", "note_swap")

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_p90_absolute_cents)
            or self.maximum_p90_absolute_cents < 0.0
        ):
            raise ValueError("maximum p90 cents must be finite and non-negative")
        if (
            not math.isfinite(self.minimum_within_100_cents)
            or not 0.0 <= self.minimum_within_100_cents <= 1.0
        ):
            raise ValueError("minimum within-100 ratio must be in [0, 1]")
        if not self.required_condition_kinds or len(
            set(self.required_condition_kinds)
        ) != len(self.required_condition_kinds):
            raise ValueError("required condition kinds must be non-empty and unique")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    return (
        value if all(character in "0123456789abcdef" for character in value) else None
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _check(
    value: object,
    operator: str,
    threshold: object,
    passed: bool,
) -> dict[str, object]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _summary(rows: list[Mapping[str, object]]) -> dict[str, float | int | None]:
    errors = [float(row["absolute_cents"]) for row in rows]
    if not errors:
        return {
            "window_count": 0,
            "exact_class_accuracy": None,
            "absolute_cents_median": None,
            "absolute_cents_p90": None,
            "within_50_cents": None,
            "within_100_cents": None,
            "voiced_coverage": None,
        }
    ordered = sorted(errors)
    index = 0.9 * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    p90 = ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])
    return {
        "window_count": len(rows),
        "exact_class_accuracy": sum(bool(row["exact_class"]) for row in rows)
        / len(rows),
        "absolute_cents_median": (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
        ),
        "absolute_cents_p90": p90,
        "within_50_cents": sum(error <= 50.0 for error in errors) / len(errors),
        "within_100_cents": sum(error <= 100.0 for error in errors) / len(errors),
        "voiced_coverage": None,
    }


def _settling_distribution(
    values: Sequence[float | int | None],
) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    ordered = sorted(finite)
    index = 0.9 * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return {
        "count": len(ordered),
        "median": (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
        ),
        "p90": ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower]),
        "maximum": ordered[-1],
    }


def _report_only_note_step_panel(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Recompute note-step latent-probe coverage without affecting pass/fail."""

    examples = manifest.get("examples")
    rows: list[Mapping[str, object]] = []
    events: list[dict[str, object]] = []
    rollout_count = 0
    valid = True
    latent_hop = manifest.get("latent_hop")
    sample_rate = manifest.get("sample_rate")
    timing_valid = (
        isinstance(latent_hop, int)
        and not isinstance(latent_hop, bool)
        and latent_hop > 0
        and isinstance(sample_rate, int)
        and not isinstance(sample_rate, bool)
        and sample_rate > 0
    )
    if not isinstance(examples, list):
        examples = []
        valid = False
    for example in examples:
        rollouts = example.get("rollouts") if isinstance(example, Mapping) else None
        if not isinstance(rollouts, list):
            valid = False
            continue
        for rollout in rollouts:
            if (
                not isinstance(rollout, Mapping)
                or rollout.get("condition_kind") != "note_step"
            ):
                continue
            rollout_count += 1
            notes = rollout.get("requested_notes")
            proxy = rollout.get("pitch_adherence_proxy")
            windows = proxy.get("windows") if isinstance(proxy, Mapping) else None
            if (
                not isinstance(notes, list)
                or not notes
                or not isinstance(windows, list)
            ):
                valid = False
                continue
            changes = [
                index
                for index in range(1, len(notes))
                if notes[index] != notes[index - 1]
            ]
            if not changes:
                valid = False
            valid_windows: list[Mapping[str, object]] = []
            for window in windows:
                if not isinstance(window, Mapping):
                    valid = False
                    continue
                midpoint = window.get("midpoint_frame")
                cents = _finite_number(window.get("absolute_cents"))
                if (
                    not isinstance(midpoint, int)
                    or isinstance(midpoint, bool)
                    or not 0 <= midpoint < len(notes)
                    or window.get("requested_midi_note") != notes[midpoint]
                    or cents is None
                    or cents < 0.0
                ):
                    valid = False
                    continue
                valid_windows.append(window)
                rows.append(window)
            for change in changes:
                target = notes[change]
                eligible = [
                    window
                    for window in valid_windows
                    if int(window["midpoint_frame"]) >= change
                    and window.get("requested_midi_note") == target
                ]
                settled = next(
                    (
                        window
                        for window in eligible
                        if window.get("within_100_cents") is True
                    ),
                    None,
                )
                settling_frames = (
                    None if settled is None else int(settled["midpoint_frame"]) - change
                )
                events.append(
                    {
                        "latent_frame": change,
                        "from_midi_note": notes[change - 1],
                        "to_midi_note": target,
                        "settled": settled is not None,
                        "settling_frames": settling_frames,
                        "settling_ms": (
                            settling_frames
                            * int(latent_hop)
                            * 1000.0
                            / int(sample_rate)
                            if settling_frames is not None and timing_valid
                            else None
                        ),
                    }
                )
    status = (
        "not_present"
        if rollout_count == 0
        else ("measured" if valid and events else "invalid")
    )
    return {
        "status": status,
        "hard_gate": False,
        "metric_kind": "latent_pitch_probe_transition_proxy",
        "rollout_count": rollout_count,
        "pitch_summary": _summary(rows),
        "transition": {
            "event_count": len(events),
            "settled_event_count": sum(event["settled"] is True for event in events),
            "settling_frames": _settling_distribution(
                [event["settling_frames"] for event in events]
            ),
            "settling_ms": _settling_distribution(
                [event["settling_ms"] for event in events]
            ),
            "events": events,
        },
        "semantic_warning": (
            "note_step settling is a report-only 16-frame latent-probe proxy; "
            "it is not decoded-audio MIDI accuracy."
        ),
    }


def midi_adherence_gate(
    manifest: Mapping[str, object],
    thresholds: MidiAdherenceThresholds | None = None,
) -> dict[str, object]:
    limits = thresholds or MidiAdherenceThresholds()
    aggregate = manifest.get("pitch_adherence_proxy")
    pitch_probe = manifest.get("pitch_probe")
    examples = manifest.get("examples")
    controls = manifest.get("controls")
    checks: dict[str, dict[str, object]] = {}

    correct_kind = (
        isinstance(aggregate, Mapping)
        and aggregate.get("metric_kind") == "latent_pitch_probe_proxy"
    )
    checks["metric_kind"] = _check(
        aggregate.get("metric_kind") if isinstance(aggregate, Mapping) else None,
        "==",
        "latent_pitch_probe_proxy",
        correct_kind,
    )
    checkpoint_hash = (
        _sha256(pitch_probe.get("checkpoint_sha256"))
        if isinstance(pitch_probe, Mapping)
        else None
    )
    qualification_hash = (
        _sha256(pitch_probe.get("qualification_sha256"))
        if isinstance(pitch_probe, Mapping)
        else None
    )
    probe_hashes = bool(
        checkpoint_hash
        and qualification_hash
        and isinstance(aggregate, Mapping)
        and aggregate.get("probe_checkpoint_sha256") == checkpoint_hash
        and aggregate.get("probe_qualification_sha256") == qualification_hash
    )
    checks["probe_hashes"] = _check(probe_hashes, "==", True, probe_hashes)

    by_kind = aggregate.get("by_condition_kind") if correct_kind else None
    requested_by_kind: dict[str, list[int]] = {
        kind: [] for kind in limits.required_condition_kinds
    }
    windows_by_kind: dict[str, list[Mapping[str, object]]] = {
        kind: [] for kind in limits.required_condition_kinds
    }
    finite_windows = True
    swap_mapping_valid = True
    control_pairs = 0
    if isinstance(examples, list):
        for example in examples:
            if not isinstance(example, Mapping):
                finite_windows = False
                continue
            rollouts = example.get("rollouts")
            if not isinstance(rollouts, list):
                finite_windows = False
                continue
            paired: dict[tuple[str, int], list[int]] = {}
            for rollout in rollouts:
                if not isinstance(rollout, Mapping):
                    finite_windows = False
                    continue
                kind = rollout.get("condition_kind")
                if kind in requested_by_kind:
                    notes = rollout.get("requested_notes")
                    seed = rollout.get("generation_seed")
                    if (
                        not isinstance(notes, list)
                        or not notes
                        or len(notes) % 16
                        or isinstance(seed, bool)
                        or not isinstance(seed, int)
                    ):
                        finite_windows = False
                        continue
                    else:
                        for note in notes:
                            if isinstance(note, bool) or not isinstance(note, int):
                                finite_windows = False
                            else:
                                requested_by_kind[str(kind)].append(note)
                        paired[(str(kind), seed)] = notes
                    proxy = rollout.get("pitch_adherence_proxy")
                    windows = (
                        proxy.get("windows") if isinstance(proxy, Mapping) else None
                    )
                    if (
                        not isinstance(proxy, Mapping)
                        or proxy.get("metric_kind") != "latent_pitch_probe_proxy"
                        or proxy.get("window_frames") != 16
                        or not isinstance(windows, list)
                        or len(windows) != len(notes) // 16
                    ):
                        finite_windows = False
                        continue
                    for window_index, window in enumerate(windows):
                        if not isinstance(window, Mapping):
                            finite_windows = False
                            continue
                        start = 16 * window_index
                        midpoint = start + 7
                        numeric = (
                            window.get("absolute_cents"),
                            window.get("expected_midi"),
                        )
                        error = _finite_number(window.get("absolute_cents"))
                        predicted = window.get("predicted_midi_class")
                        requested = window.get("requested_midi_note")
                        valid = (
                            all(_finite_number(value) is not None for value in numeric)
                            and error is not None
                            and error >= 0.0
                            and isinstance(predicted, int)
                            and not isinstance(predicted, bool)
                            and isinstance(requested, int)
                            and not isinstance(requested, bool)
                            and requested == notes[midpoint]
                            and window.get("start_frame") == start
                            and window.get("midpoint_frame") == midpoint
                            and window.get("exact_class") is (predicted == requested)
                            and window.get("within_50_cents") is (error <= 50.0)
                            and window.get("within_100_cents") is (error <= 100.0)
                        )
                        if not valid:
                            finite_windows = False
                        else:
                            windows_by_kind[str(kind)].append(window)
            seeds = {
                seed for kind, seed in paired if kind in limits.required_condition_kinds
            }
            for seed in seeds:
                matched_notes = paired.get(("matched", seed))
                swapped_notes = paired.get(("note_swap", seed))
                if matched_notes is None or swapped_notes is None:
                    swap_mapping_valid = False
                    continue
                control_pairs += 1
                if len(matched_notes) != len(swapped_notes):
                    swap_mapping_valid = False
                    continue
                mapping = (
                    controls.get("note_swap_mapping")
                    if isinstance(controls, Mapping)
                    else None
                )
                if not isinstance(mapping, Mapping) or not mapping:
                    swap_mapping_valid = False
                    continue
                for matched_note, swapped_note in zip(
                    matched_notes, swapped_notes, strict=True
                ):
                    expected = mapping.get(str(matched_note))
                    if (
                        matched_note == swapped_note
                        or isinstance(expected, bool)
                        or not isinstance(expected, int)
                        or expected != swapped_note
                    ):
                        swap_mapping_valid = False
    else:
        finite_windows = False
    checks["finite_windows"] = _check(finite_windows, "==", True, finite_windows)

    recomputed = {
        kind: _summary(windows_by_kind[kind])
        for kind in limits.required_condition_kinds
    }
    aggregate_consistent = isinstance(by_kind, Mapping)
    for kind in limits.required_condition_kinds:
        declared = by_kind.get(kind) if isinstance(by_kind, Mapping) else None
        actual = recomputed[kind]
        if not isinstance(declared, Mapping):
            aggregate_consistent = False
            continue
        for name in (
            "window_count",
            "exact_class_accuracy",
            "absolute_cents_median",
            "absolute_cents_p90",
            "within_50_cents",
            "within_100_cents",
            "voiced_coverage",
        ):
            left = declared.get(name)
            right = actual[name]
            if isinstance(right, float):
                aggregate_consistent &= _finite_number(
                    left
                ) is not None and math.isclose(
                    float(left), right, rel_tol=1.0e-9, abs_tol=1.0e-9
                )
            else:
                aggregate_consistent &= left == right
    checks["aggregate_consistency"] = _check(
        aggregate_consistent, "==", True, aggregate_consistent
    )

    for kind in limits.required_condition_kinds:
        summary = recomputed[kind]
        count = int(summary["window_count"])
        p90 = _finite_number(summary["absolute_cents_p90"])
        within = _finite_number(summary["within_100_cents"])
        checks[f"{kind}_coverage"] = _check(count, ">", 0, count > 0)
        checks[f"{kind}_p90_absolute_cents"] = _check(
            p90,
            "<=",
            limits.maximum_p90_absolute_cents,
            p90 is not None and p90 <= limits.maximum_p90_absolute_cents,
        )
        checks[f"{kind}_within_100_cents"] = _check(
            within,
            ">=",
            limits.minimum_within_100_cents,
            within is not None and within >= limits.minimum_within_100_cents,
        )

    changed = control_pairs > 0 and swap_mapping_valid
    checks["note_swap_changes_requested_note"] = _check(
        changed,
        "==",
        True,
        changed,
    )

    failed = [name for name, check in checks.items() if not check["passed"]]
    note_step_panel = _report_only_note_step_panel(manifest)
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "thresholds": asdict(limits),
        "metric_kind": "latent_pitch_probe_proxy",
        "metrics": {
            "by_condition_kind": recomputed,
            "report_only_note_step": note_step_panel,
        },
        "report_only_metrics": {"note_step": note_step_panel},
        "semantic_warning": (
            "This gate measures a qualified frozen probe on generated latents; "
            "it is not decoded-audio/CREPE MIDI accuracy and voiced coverage is N/A."
        ),
        "decoded_audio_midi_status": (
            "separate_panel: CREPE note_step settling and the velocity_step "
            "RMS-response proxy are evaluated by decoded_audio_crepe"
        ),
    }


def evaluate_midi_adherence_manifest(
    manifest_path: str | Path,
    *,
    thresholds: MidiAdherenceThresholds | None = None,
) -> dict[str, object]:
    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("MIDI audition manifest must be an object")
    gate = midi_adherence_gate(manifest, thresholds)
    return {
        "schema": 1,
        "kind": "zrave-midi-latent-pitch-probe-adherence-gate",
        "manifest": str(source),
        "manifest_sha256": _sha256_file(source),
        **gate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate the G3 latent_pitch_probe_proxy. This is not decoded-audio "
            "MIDI accuracy; CREPE voiced coverage/transition settling are not run."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-p90-absolute-cents", type=float, default=100.0)
    parser.add_argument("--minimum-within-100-cents", type=float, default=0.90)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = evaluate_midi_adherence_manifest(
        args.manifest,
        thresholds=MidiAdherenceThresholds(
            maximum_p90_absolute_cents=args.maximum_p90_absolute_cents,
            minimum_within_100_cents=args.minimum_within_100_cents,
        ),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_reject and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
