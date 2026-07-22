from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from midibrave.realtime_engine import RealtimeEngine


MIDI_NOTES = (60, 64, 67, 72, 55, 48)


def run_smoke(engine: RealtimeEngine, seconds: float) -> dict[str, object]:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be a positive finite value")

    status = engine.status()
    sample_rate = int(status["sampleRate"])
    block_samples = int(status["blockSamples"])
    target_blocks = math.ceil(seconds * sample_rate / block_samples)
    session = engine.new_session()
    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=0)

    finite = True
    render_times: list[float] = []
    started = time.perf_counter()
    for block_index in range(target_blocks):
        phase = 2.0 * math.pi * block_index / target_blocks
        note = MIDI_NOTES[(block_index // 64) % len(MIDI_NOTES)]
        session.update(
            seq=block_index + 1,
            x=0.8 * math.cos(phase),
            y=0.8 * math.sin(phase),
            note=note,
            velocity=0.8,
        )
        audio, render_ms = session.render_block()
        render_ms = float(render_ms)
        finite = finite and bool(np.isfinite(audio).all())
        finite = finite and math.isfinite(render_ms)
        render_times.append(render_ms)
    wall_elapsed_seconds = time.perf_counter() - started

    render_p50_ms, render_p95_ms = np.percentile(
        np.asarray(render_times, dtype=np.float64),
        (50, 95),
    )
    audio_seconds = target_blocks * block_samples / sample_rate
    throughput_ratio = audio_seconds / wall_elapsed_seconds
    report = {
        "blocks": len(render_times),
        "finite": finite,
        "audio_seconds": float(audio_seconds),
        "wall_elapsed_seconds": float(wall_elapsed_seconds),
        "throughput_ratio": float(throughput_ratio),
        "render_mean_ms": float(np.mean(render_times)),
        "render_p50_ms": float(render_p50_ms),
        "render_p95_ms": float(render_p95_ms),
        "render_max_ms": float(np.max(render_times)),
        "block_deadline_ms": float(status["blockDeadlineMs"]),
        "realtime_pass": finite and throughput_ratio > 1.0,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MidiBrave CUDA realtime acceptance smoke test",
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = RealtimeEngine.load(args.runtime)
    report = run_smoke(engine, args.seconds)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["realtime_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
