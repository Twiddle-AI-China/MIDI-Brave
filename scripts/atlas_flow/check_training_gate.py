from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--minimum-update", type=int, required=True)
    parser.add_argument("--minimum-relative-decrease", type=float, default=0.0)
    parser.add_argument("--require-all-modules", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if not rows or max(int(row["update"]) for row in rows) < args.minimum_update:
        errors.append("required update was not reached")
    losses = [float(row["loss"]) for row in rows]
    if not losses or not all(math.isfinite(value) for value in losses):
        errors.append("loss contains NaN/Inf")
    width = min(10, max(1, len(losses) // 4))
    first = sum(losses[:width]) / width if losses else float("nan")
    last = sum(losses[-width:]) / width if losses else float("nan")
    relative_decrease = (first - last) / max(abs(first), 1.0e-8)
    if relative_decrease < args.minimum_relative_decrease:
        errors.append(
            f"loss decrease {relative_decrease:.4f} < {args.minimum_relative_decrease:.4f}"
        )
    # Lifetime AMP skips are expected during long FP16 runs and are diagnostic,
    # not a failure by themselves. Only an unresolved consecutive streak means
    # numerical divergence; the trainer also aborts immediately at 8.
    maximum_skips = max((int(row.get("nonfinite_skips", 0)) for row in rows), default=0)
    maximum_consecutive_skips = max(
        (int(row.get("consecutive_nonfinite_skips", 0)) for row in rows), default=0,
    )
    if maximum_consecutive_skips >= 8:
        errors.append(
            f"consecutive non-finite atomic skips observed: {maximum_consecutive_skips}"
        )
    if args.require_all_modules and rows:
        required = {"encoder", "adapter", "decoder", "pitch_conditioner", "output_gain", "pitch_adversary", "flow"}
        observed = {
            name
            for row in rows[-width:]
            for name, value in row.get("module_gradient_norms", {}).items()
            if float(value) > 0.0
        }
        missing = sorted(required - observed)
        if missing:
            errors.append("modules without non-zero gradients: " + ", ".join(missing))
    report = {
        "schema": "midibrave.atlas-flow.training-gate.v1",
        "passed": not errors,
        "minimum_update": args.minimum_update,
        "rows": len(rows),
        "first_loss": first,
        "last_loss": last,
        "relative_decrease": relative_decrease,
        "maximum_nonfinite_skips": maximum_skips,
        "maximum_consecutive_nonfinite_skips": maximum_consecutive_skips,
        "errors": errors,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
