from __future__ import annotations

import argparse

from midibrave.zrave_audition import render_audition


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render source/direct/predicted Z-RAVE audition comparisons"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_audition(
        args.config,
        args.checkpoint,
        args.output,
        duration_seconds=args.duration,
        random_seed=args.random_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
