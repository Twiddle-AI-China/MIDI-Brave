#!/usr/bin/env python3
"""Reject cross-platform or reordered Spark tiny-category queues."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "arp",
    "bass",
    "fx",
    "lead",
    "pad",
    "pluck",
    "synth",
    "chord_pilot",
    "keys_pilot",
    "lead_pluck",
    "keys_harmony",
)
TRAIN_ACTIONS = ("smoke", "sweep", "train")
REPORT_ACTION = "audition_report"


def _rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    parsed = []
    for number, row in enumerate(
        csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"),
        start=1,
    ):
        if not row or row[0].startswith("#"):
            continue
        if len(row) != 5:
            raise ValueError(f"queue line {number} must contain exactly five columns")
        parsed.append(tuple(row))
    if not parsed:
        raise ValueError("queue is empty")
    return parsed


def validate(queue_path: Path, *, code_root: Path = ROOT) -> dict[str, int]:
    queue = queue_path.resolve()
    spark_root = (code_root / "scripts/spark").resolve()
    if not queue.is_relative_to(spark_root):
        raise ValueError("Spark queue must stay below scripts/spark")
    if queue.name != "tiny_categories_v1.queue.tsv":
        raise ValueError("only the canonical Spark tiny category queue is accepted")
    rows = _rows(queue)
    identifiers = [row[0] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("queue experiment identifiers must be unique")
    actions = {row[1] for row in rows}
    if not actions <= {*TRAIN_ACTIONS, REPORT_ACTION}:
        raise ValueError(f"unsupported Spark category actions: {sorted(actions)}")
    try:
        first_report = next(i for i, row in enumerate(rows) if row[1] == REPORT_ACTION)
    except StopIteration as error:
        raise ValueError("Spark category queue is missing report rows") from error
    if any(row[1] == REPORT_ACTION for row in rows[:first_report]):
        raise AssertionError("unreachable report ordering failure")
    if any(row[1] != REPORT_ACTION for row in rows[first_report:]):
        raise ValueError("all model training must complete before any report")

    by_family: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for row in rows:
        identifier, action, config_relative, run_relative, spec = row
        if not identifier.startswith("spark-tinycat-v1-"):
            raise ValueError(f"non-Spark experiment identifier: {identifier}")
        run_prefix = "runs/spark-tiny-categories-v1/"
        config_prefix = "configs/zrave/generated/spark_tiny_categories_v1/"
        if not run_relative.startswith(run_prefix):
            raise ValueError(f"run path crosses platform boundary: {run_relative}")
        family = run_relative.removeprefix(run_prefix)
        if family not in FAMILIES or "/" in family:
            raise ValueError(f"unknown Spark category family: {family}")
        if config_relative != f"{config_prefix}{family}.yaml":
            raise ValueError(f"config/run family mismatch for {identifier}")
        if "|" in spec:
            raise ValueError("scratch matrix rows must not carry an initializer")
        by_family[family].append(row)

    if tuple(by_family) != FAMILIES:
        raise ValueError("Spark matrix family order or membership changed")
    for family in FAMILIES:
        family_rows = by_family[family]
        training = [row for row in family_rows if row[1] in TRAIN_ACTIONS]
        reports = [row for row in family_rows if row[1] == REPORT_ACTION]
        if [row[1] for row in training] != list(TRAIN_ACTIONS):
            raise ValueError(f"family {family} must run smoke, sweep, then train")
        if training[0][4] != "-":
            raise ValueError(f"family {family} smoke spec must be '-'")
        if training[1][4] != "16,32,64,96,128,160":
            raise ValueError(f"family {family} sweep candidates changed")
        if training[2][4] != "10000":
            raise ValueError(f"family {family} train target must be 10000")
        if [row[4] for row in reports] != [
            "step-001000.pt",
            "step-005000.pt",
            "step-010000.pt",
        ]:
            raise ValueError(f"family {family} report checkpoints changed")
        config = code_root / training[2][2]
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        model = raw["model"]
        train = raw["train"]
        if (
            model.get("profile") != "tiny"
            or model.get("latent_dim") != 128
            or model.get("pitch_conditioning") is not False
            or model.get("midi_sequence_conditioning") is not False
        ):
            raise ValueError(f"family {family} is not a pure RAVE128 tiny model")
        if train.get("max_updates") != 10000:
            raise ValueError(f"family {family} config update target changed")
        expected_output = (
            "/data/midibrave-zrave-flow-serum128/"
            f"runs/spark-tiny-categories-v1/{family}"
        )
        if train.get("output_root") != expected_output:
            raise ValueError(f"family {family} output_root crosses platform boundary")
    return {"families": len(FAMILIES), "training_rows": first_report, "reports": len(rows) - first_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(args.queue)
    print(
        "validated Spark tiny category matrix: "
        f"{summary['families']} models, {summary['training_rows']} pre-report rows, "
        f"{summary['reports']} reports"
    )


if __name__ == "__main__":
    main()
