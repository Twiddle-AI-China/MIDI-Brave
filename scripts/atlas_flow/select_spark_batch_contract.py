from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parse_spec(value: str) -> tuple[int, int]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid batch specification: {value}")
    micro, accum = (int(item) for item in parts)
    if micro < 1 or accum < 1:
        raise ValueError("batch values must be positive")
    return micro, accum


def select_contract(
    reports: list[dict[str, Any]],
    *,
    stage: str,
    baseline: tuple[int, int],
    minimum_headroom_mib: float = 24_576.0,
    minimum_system_available_mib: float = 20_480.0,
    minimum_relative_gain: float = 0.05,
    tie_fraction: float = 0.03,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for report in reports:
        contract = report.get("batch_contract", {})
        micro = int(contract.get("micro_batch", 0))
        accum = int(contract.get("grad_accum", 0))
        reasons: list[str] = []
        if report.get("stage") != stage:
            reasons.append("stage_mismatch")
        if not bool(report.get("passed")):
            reasons.append("qualification_failed")
        if int(report.get("updates_completed", 0)) != int(report.get("updates_requested", -1)):
            reasons.append("incomplete_measurement")
        throughput = report.get("effective_examples_per_second")
        if not isinstance(throughput, (int, float)) or not math.isfinite(float(throughput)) or float(throughput) <= 0:
            reasons.append("invalid_throughput")
        headroom = report.get("memory_headroom_mib")
        if not isinstance(headroom, (int, float)) or float(headroom) < minimum_headroom_mib:
            reasons.append("insufficient_memory_headroom")
        available = report.get("system_available_memory_mib")
        if not isinstance(available, (int, float)) or float(available) < minimum_system_available_mib:
            reasons.append("insufficient_system_memory")
        evaluated.append({
            "micro_batch": micro,
            "grad_accum": accum,
            "throughput": float(throughput) if isinstance(throughput, (int, float)) else None,
            "seconds_per_atomic_update": report.get("seconds_per_atomic_update"),
            "peak_memory_allocated_mib": report.get("peak_memory_allocated_mib"),
            "peak_memory_reserved_mib": report.get("peak_memory_reserved_mib"),
            "memory_headroom_mib": headroom,
            "system_available_memory_mib": available,
            "safe": not reasons,
            "rejection_reasons": reasons,
            "report": report,
        })

    safe = [candidate for candidate in evaluated if candidate["safe"]]
    if not safe:
        raise ValueError("no qualification candidate satisfies the memory and finite-value gates")
    baseline_candidate = next(
        (candidate for candidate in safe if (candidate["micro_batch"], candidate["grad_accum"]) == baseline),
        None,
    )
    fastest = max(safe, key=lambda candidate: float(candidate["throughput"]))
    reason = "maximum_throughput"
    if baseline_candidate is not None and float(fastest["throughput"]) < (
        float(baseline_candidate["throughput"]) * (1.0 + minimum_relative_gain)
    ):
        selected = baseline_candidate
        reason = "gain_below_minimum"
    else:
        threshold = float(fastest["throughput"]) * (1.0 - tie_fraction)
        tied = [candidate for candidate in safe if float(candidate["throughput"]) >= threshold]
        selected = max(tied, key=lambda candidate: int(candidate["micro_batch"]))
        if selected is not fastest:
            reason = "larger_micro_within_tie"

    remaining = sorted(
        (candidate for candidate in safe if candidate is not selected),
        key=lambda candidate: float(candidate["throughput"]),
        reverse=True,
    )
    ranked = [selected, *remaining]
    baseline_throughput = None if baseline_candidate is None else float(baseline_candidate["throughput"])
    relative_gain = (
        None if baseline_throughput is None
        else float(selected["throughput"]) / baseline_throughput - 1.0
    )
    selected_report = selected["report"]
    return {
        "schema": "midibrave.atlas-flow.batch-contract.v2",
        "stage": stage,
        "selection_reason": reason,
        "minimum_headroom_mib": minimum_headroom_mib,
        "minimum_system_available_mib": minimum_system_available_mib,
        "minimum_relative_gain": minimum_relative_gain,
        "tie_fraction": tie_fraction,
        "baseline": {"micro_batch": baseline[0], "grad_accum": baseline[1]},
        "selected": {
            "micro_batch": selected["micro_batch"],
            "grad_accum": selected["grad_accum"],
            "throughput": selected["throughput"],
            "relative_gain_vs_baseline": relative_gain,
            "batch_contract": selected_report["batch_contract"],
        },
        "ranked_safe_candidates": [
            {
                "micro_batch": candidate["micro_batch"],
                "grad_accum": candidate["grad_accum"],
                "throughput": candidate["throughput"],
            }
            for candidate in ranked
        ],
        "candidates": [
            {key: value for key, value in candidate.items() if key != "report"}
            for candidate in evaluated
        ],
        "hardware": {
            "host": selected_report.get("host"),
            "device": selected_report.get("device"),
            "compute_capability": selected_report.get("compute_capability"),
            "torch_version": selected_report.get("torch_version"),
            "cuda_version": selected_report.get("cuda_version"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a safe throughput-maximizing Spark batch contract.")
    parser.add_argument("--stage", required=True, choices=("stage1", "flow", "joint"))
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--minimum-headroom-mib", type=float, default=24_576.0)
    parser.add_argument("--minimum-system-available-mib", type=float, default=20_480.0)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.05)
    parser.add_argument("--tie-fraction", type=float, default=0.03)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-output", required=True)
    parser.add_argument("--ranked-output", required=True)
    args = parser.parse_args()

    report_paths = [Path(value) for value in args.report]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    result = select_contract(
        reports,
        stage=args.stage,
        baseline=_parse_spec(args.baseline),
        minimum_headroom_mib=args.minimum_headroom_mib,
        minimum_system_available_mib=args.minimum_system_available_mib,
        minimum_relative_gain=args.minimum_relative_gain,
        tie_fraction=args.tie_fraction,
    )
    result["checkpoint_sha256"] = args.checkpoint_sha256
    result["image"] = args.image
    result["report_paths"] = [str(path) for path in report_paths]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = result["selected"]
    Path(args.text_output).write_text(
        f"{selected['micro_batch']} {selected['grad_accum']}\n", encoding="utf-8",
    )
    Path(args.ranked_output).write_text(
        "".join(
            f"{candidate['micro_batch']} {candidate['grad_accum']}\n"
            for candidate in result["ranked_safe_candidates"]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
