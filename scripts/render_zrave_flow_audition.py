from __future__ import annotations

import argparse

from midibrave.zrave_flow_audition import render_flow_audition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Serum128 exploration-flow comparison."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Pad", "Lead", "Bass", "Pluck", "Keys", "Synth"],
    )
    parser.add_argument(
        "--explorations",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
    )
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=[17, 71],
    )
    parser.add_argument("--generated-frames", type=int, default=320)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_flow_audition(
        args.config,
        args.checkpoint,
        args.output,
        categories=args.categories,
        explorations=args.explorations,
        generation_seeds=args.generation_seeds,
        candidate_count=args.candidate_count,
        generated_frames=args.generated_frames,
        selection_seed=args.selection_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
