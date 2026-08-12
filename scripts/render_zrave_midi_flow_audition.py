from __future__ import annotations

import argparse

from midibrave.zrave_flow_audition import render_midi_flow_audition


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render paired Serum128 MIDI matched/note-swap continuations."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-initializer-sha256", required=True)
    parser.add_argument(
        "--expected-initializer-update",
        type=int,
        default=85000,
    )
    parser.add_argument("--pitch-qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Pad", "Lead", "Bass", "Pluck", "Keys", "Synth"],
    )
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=[17, 71],
    )
    parser.add_argument("--generated-frames", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--wander-delay-frames",
        type=int,
        choices=(16, 32, 48),
        default=32,
    )
    parser.add_argument("--pitch-guidance", type=float)
    parser.add_argument(
        "--note-vocabulary",
        nargs="+",
        type=int,
        default=[36, 62, 82],
    )
    parser.add_argument("--selection-seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_midi_flow_audition(
        args.config,
        args.checkpoint,
        args.pitch_qualification,
        args.output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_initializer_sha256=args.expected_initializer_sha256,
        expected_initializer_update=args.expected_initializer_update,
        categories=args.categories,
        generation_seeds=args.generation_seeds,
        generated_frames=args.generated_frames,
        temperature=args.temperature,
        wander_delay_frames=args.wander_delay_frames,
        pitch_guidance=args.pitch_guidance,
        expected_note_vocabulary=args.note_vocabulary,
        selection_seed=args.selection_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
