from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from midibrave.atlas_flow_train import _write_tensorboard_record


def backfill(run_root: Path) -> dict[str, object]:
    report: dict[str, object] = {"schema": "midibrave.tensorboard.backfill.v1", "stages": {}}
    for stage_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        metrics_path = stage_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        by_update: dict[int, dict[str, object]] = {}
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    by_update[int(record["update"])] = record
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        writer = SummaryWriter(log_dir=str(stage_dir / "tensorboard"))
        for update in sorted(by_update):
            _write_tensorboard_record(writer, by_update[update])
        writer.close()
        report["stages"][stage_dir.name] = {
            "records": len(by_update),
            "first_update": min(by_update) if by_update else None,
            "last_update": max(by_update) if by_update else None,
            "log_dir": str(stage_dir / "tensorboard"),
        }
    destination = run_root / "tensorboard-backfill-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(backfill(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
