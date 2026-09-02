#!/usr/bin/env python3
"""Select and verify a single-GPU Z-RAVE throughput sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


METRIC = "median_valid_latent_frames_per_second"
DEFAULT_PLATFORM = "lvzihao-rtx5080-single-gpu"
_SAFE_PLATFORM = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _platform(value: str) -> str:
    if not _SAFE_PLATFORM.fullmatch(value):
        raise ValueError(f"unsafe sweep platform identifier: {value!r}")
    return value


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _valid(
    report: dict[str, Any],
    *,
    measured_updates: int,
    exposure_updates: int,
) -> bool:
    try:
        throughput = float(report[METRIC])
        peak = float(report["peak_memory_mib"])
        total = float(report["total_memory_mib"])
        return (
            report.get("status") == "ok"
            and int(report["world_size"]) == 1
            and int(report["measured_updates"]) == measured_updates
            and int(report["exposure_safety_updates"])
            == exposure_updates
            and int(report["nonfinite_updates"]) == 0
            and math.isfinite(throughput)
            and throughput > 0.0
            and math.isfinite(peak)
            and math.isfinite(total)
            and 0.0 < peak < total
        )
    except (KeyError, TypeError, ValueError):
        return False


def select(args: argparse.Namespace) -> None:
    platform = _platform(args.platform)
    if (
        not args.batches
        or any(batch <= 0 for batch in args.batches)
        or len(set(args.batches)) != len(args.batches)
    ):
        raise ValueError("sweep batches must be positive and unique")
    root = Path(args.sweep_root)
    reports: list[dict[str, Any]] = []
    for batch in args.batches:
        path = root / f"batch-{batch}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report.get("batch_per_gpu", -1)) != batch:
            raise ValueError(f"batch report contract mismatch: {path}")
        report["report_path"] = str(path.resolve())
        reports.append(report)

    valid = [
        report
        for report in reports
        if _valid(
            report,
            measured_updates=args.measured_updates,
            exposure_updates=args.exposure_updates,
        )
    ]
    if not valid:
        raise ValueError("no valid single-GPU sweep candidate")
    contracts = {
        (
            str(report["config_sha256"]),
            str(report["pack_index_sha256"]),
            str(report["git_commit"]),
        )
        for report in valid
    }
    if len(contracts) != 1:
        raise ValueError("valid sweep reports do not share one data/code contract")

    fastest = max(float(report[METRIC]) for report in valid)
    tied = [
        report
        for report in valid
        if float(report[METRIC]) >= fastest * 0.98
    ]
    winner = min(tied, key=lambda report: int(report["batch_per_gpu"]))
    selection = {
        "schema": 1,
        "platform": platform,
        "rule": (
            "highest median valid latent frames/s; within 2% choose "
            "the smaller batch"
        ),
        "metric": METRIC,
        "expected_batches": args.batches,
        "candidates": reports,
        "batch_per_gpu": int(winner["batch_per_gpu"]),
        "selected": winner,
    }
    _atomic_json(Path(args.output), selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


def print_verified_batch(args: argparse.Namespace) -> None:
    platform = _platform(args.platform)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if (
        selection.get("schema") != 1
        or selection.get("platform") != platform
        or selection.get("metric") != METRIC
    ):
        raise ValueError(f"unsupported {platform} sweep selection")
    selected = selection["selected"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = {
        "config_sha256": _digest(Path(args.config)),
        "pack_index_sha256": _digest(Path(args.pack_index)),
        "git_commit": commit,
    }
    for name, value in expected.items():
        if selected.get(name) != value:
            raise ValueError(
                f"sweep {name} mismatch: {selected.get(name)} != {value}"
            )
    if int(selected.get("world_size", 0)) != 1:
        raise ValueError("sweep was not measured with one GPU")
    batch = int(selection["batch_per_gpu"])
    if batch <= 0 or batch != int(selected.get("batch_per_gpu", -1)):
        raise ValueError("selected batch is invalid")
    print(batch)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    choose = commands.add_parser("select")
    choose.add_argument("--sweep-root", required=True)
    choose.add_argument("--output", required=True)
    choose.add_argument("--batches", nargs="+", type=int, required=True)
    choose.add_argument("--measured-updates", type=int, required=True)
    choose.add_argument("--exposure-updates", type=int, required=True)
    choose.add_argument("--platform", default=DEFAULT_PLATFORM)
    choose.set_defaults(handler=select)

    verify = commands.add_parser("print-verified-batch")
    verify.add_argument("--selection", required=True)
    verify.add_argument("--config", required=True)
    verify.add_argument("--pack-index", required=True)
    verify.add_argument("--project-root", required=True)
    verify.add_argument("--platform", default=DEFAULT_PLATFORM)
    verify.set_defaults(handler=print_verified_batch)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
