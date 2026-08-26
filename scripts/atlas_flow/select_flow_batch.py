from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", type=int, default=72)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    passed = [
        row for row in report["candidates"]
        if row["passed"] and float(row["peak_memory_mib"]) <= 14_500
    ]
    batch = max((int(row["batch_per_gpu"]) for row in passed), default=args.minimum)
    if batch < args.minimum:
        batch = args.minimum
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{batch}\n", encoding="utf-8")
    print(batch)


if __name__ == "__main__":
    main()
