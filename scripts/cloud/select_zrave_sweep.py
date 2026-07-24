#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BATCHES = (128, 256, 384, 512)
THROUGHPUT_METRICS = (
    "median_windows_per_second",
    "median_valid_latent_frames_per_second",
)


def _valid(
    candidate: dict[str, Any],
    metric: str = "median_windows_per_second",
) -> bool:
    if metric not in THROUGHPUT_METRICS:
        raise ValueError(f"unsupported sweep metric: {metric}")
    try:
        throughput = float(candidate[metric])
        peak = float(candidate["peak_memory_mib"])
        total = float(candidate["total_memory_mib"])
        return (
            candidate.get("status") == "ok"
            and int(candidate["world_size"]) == 8
            and int(candidate["measured_updates"]) == 100
            and int(candidate["nonfinite_updates"]) == 0
            and math.isfinite(throughput)
            and throughput > 0.0
            and math.isfinite(peak)
            and math.isfinite(total)
            and 0.0 < peak < total
            and (
                metric != "median_valid_latent_frames_per_second"
                or int(candidate["exposure_safety_updates"]) == 5
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_candidate(
    candidates: list[dict[str, Any]],
    metric: str = "median_windows_per_second",
) -> dict[str, Any]:
    if metric not in THROUGHPUT_METRICS:
        raise ValueError(f"unsupported sweep metric: {metric}")
    valid = [
        candidate
        for candidate in candidates
        if _valid(candidate, metric)
    ]
    if not valid:
        raise ValueError("no valid Z-RAVE sweep candidate")
    fastest = max(
        float(candidate[metric])
        for candidate in valid
    )
    tied = [
        candidate
        for candidate in valid
        if float(candidate[metric]) >= fastest * 0.98
    ]
    return min(tied, key=lambda candidate: int(candidate["batch_per_gpu"]))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_candidates(
    root: Path,
    batches: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("sweep batches must be positive")
    if len(set(batches)) != len(batches):
        raise ValueError("sweep batches must be unique")
    candidates: list[dict[str, Any]] = []
    observed_batches: list[int] = []
    for batch in batches:
        path = root / f"batch-{batch}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing sweep report: {path}")
        candidate = json.loads(path.read_text(encoding="utf-8"))
        candidate["report_path"] = str(path.resolve())
        candidates.append(candidate)
        observed_batches.append(int(candidate.get("batch_per_gpu", -1)))
    if tuple(observed_batches) != batches:
        raise ValueError(
            f"sweep batch contract mismatch: {observed_batches}"
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the highest-throughput valid Z-RAVE batch."
    )
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latest-output")
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=list(EXPECTED_BATCHES),
    )
    parser.add_argument(
        "--metric",
        choices=THROUGHPUT_METRICS,
        default="median_windows_per_second",
    )
    args = parser.parse_args()
    root = Path(args.sweep_root)
    batches = tuple(args.batches)
    candidates = load_candidates(root, batches)
    winner = select_candidate(candidates, metric=args.metric)
    rule = (
        "highest median global windows/s; within 2% choose smaller batch"
        if args.metric == "median_windows_per_second"
        else (
            "highest median valid latent frames/s; "
            "within 2% choose smaller batch"
        )
    )
    selection = {
        "schema": 1,
        "rule": rule,
        "expected_batches": list(batches),
        "candidates": candidates,
        "selected": winner,
        "batch_per_gpu": int(winner["batch_per_gpu"]),
        "median_windows_per_second": float(
            winner["median_windows_per_second"]
        ),
        "metric": args.metric,
        "metric_value": float(winner[args.metric]),
    }
    output = Path(args.output)
    _atomic_json(output, selection)
    if args.latest_output:
        _atomic_json(Path(args.latest_output), selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
