#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BATCHES = (128, 256, 384, 512)


def _valid(candidate: dict[str, Any]) -> bool:
    try:
        throughput = float(candidate["median_windows_per_second"])
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
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if _valid(candidate)]
    if not valid:
        raise ValueError("no valid Z-RAVE sweep candidate")
    fastest = max(
        float(candidate["median_windows_per_second"])
        for candidate in valid
    )
    tied = [
        candidate
        for candidate in valid
        if float(candidate["median_windows_per_second"]) >= fastest * 0.98
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the highest-throughput valid Z-RAVE batch."
    )
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latest-output")
    args = parser.parse_args()
    root = Path(args.sweep_root)
    candidates: list[dict[str, Any]] = []
    observed_batches: list[int] = []
    for batch in EXPECTED_BATCHES:
        path = root / f"batch-{batch}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing sweep report: {path}")
        candidate = json.loads(path.read_text(encoding="utf-8"))
        candidate["report_path"] = str(path.resolve())
        candidates.append(candidate)
        observed_batches.append(int(candidate.get("batch_per_gpu", -1)))
    if tuple(observed_batches) != EXPECTED_BATCHES:
        raise ValueError(
            f"sweep batch contract mismatch: {observed_batches}"
        )
    winner = select_candidate(candidates)
    selection = {
        "schema": 1,
        "rule": (
            "highest median global windows/s; within 2% choose smaller batch"
        ),
        "expected_batches": list(EXPECTED_BATCHES),
        "candidates": candidates,
        "selected": winner,
        "batch_per_gpu": int(winner["batch_per_gpu"]),
        "median_windows_per_second": float(
            winner["median_windows_per_second"]
        ),
    }
    output = Path(args.output)
    _atomic_json(output, selection)
    if args.latest_output:
        _atomic_json(Path(args.latest_output), selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
