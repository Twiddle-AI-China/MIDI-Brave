from __future__ import annotations

import argparse

from midibrave.zrave_audition import render_long_rollout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 32-seed to 320-predicted Z-RAVE continuations."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-frames", type=int, default=32)
    parser.add_argument("--predicted-frames", type=int, default=320)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_long_rollout(
        args.config,
        args.checkpoint,
        args.output,
        seed_frames=args.seed_frames,
        predicted_frames=args.predicted_frames,
        random_seed=args.random_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
