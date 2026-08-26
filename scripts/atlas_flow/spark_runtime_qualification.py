from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from midibrave.atlas_flow_runtime import AtlasFlowLiveEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify the Atlas Flow live runtime on one allocated GPU",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return float(ordered[index])


def _exercise_voice(
    engine: AtlasFlowLiveEngine,
    *,
    note: int,
    pca_normalized: np.ndarray,
    seed: int,
) -> dict[str, object]:
    session = engine.new_session()
    started = time.perf_counter()
    session.start(
        note=note,
        velocity=1.0,
        seed=seed,
        temperature=0.0,
        morph_seconds=1.0,
        pca_normalized=pca_normalized,
    )
    plan_ms = (time.perf_counter() - started) * 1_000.0

    held: list[np.ndarray] = []
    block_ms: list[float] = []
    for _ in range(24):
        block, elapsed = session.render_block()
        held.append(block)
        block_ms.append(elapsed)

    held_peaks = [float(np.max(np.abs(block))) for block in held[2:]]
    session.note_off()
    release: list[np.ndarray] = []
    for _ in range(48):
        block, elapsed = session.render_block()
        release.append(block)
        block_ms.append(elapsed)
        if session.snapshot()["lifecycle"] == "idle":
            break
    final_block, elapsed = session.render_block()
    block_ms.append(elapsed)

    waveform = np.concatenate([*held, *release, final_block], axis=0)
    snapshot = session.snapshot()
    projected = np.asarray(snapshot["projectedPcaNormalized"], dtype=np.float32)
    return {
        "note": note,
        "plan_ms": plan_ms,
        "samples": int(waveform.shape[0]),
        "finite": bool(np.isfinite(waveform).all()),
        "peak": float(np.max(np.abs(waveform))),
        "minimum_held_block_peak": min(held_peaks),
        "silent_held_blocks": int(sum(peak <= 1.0e-8 for peak in held_peaks)),
        "release_reached_idle": snapshot["lifecycle"] == "idle",
        "final_block_peak": float(np.max(np.abs(final_block))),
        "projected_pca_finite": bool(
            projected.shape == (8,) and np.isfinite(projected).all()
        ),
        "block_ms_p95": _percentile(block_ms, 0.95),
        "block_ms_max": max(block_ms),
    }


def main() -> None:
    args = _parser().parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"qualification requires exactly one visible GPU, got {torch.cuda.device_count()}",
        )
    torch.manual_seed(20260826)
    engine = AtlasFlowLiveEngine.load(args.config, args.checkpoint, args.device)
    normalized = np.stack([
        engine.normalize(coordinate) for coordinate in engine.atlas.coordinates
    ])
    targets = (
        normalized[engine.center_index],
        normalized[0],
        normalized[-1],
    )
    voices = [
        _exercise_voice(
            engine,
            note=note,
            pca_normalized=target,
            seed=20260826 + index,
        )
        for index, (note, target) in enumerate(zip((36, 57, 71), targets))
    ]
    deadline_ms = float(engine.status()["blockDeadlineMs"])
    passed = all(
        voice["finite"]
        and voice["peak"] > 1.0e-6
        and voice["silent_held_blocks"] == 0
        and voice["release_reached_idle"]
        and voice["final_block_peak"] <= 1.0e-7
        and voice["projected_pca_finite"]
        and voice["block_ms_p95"] < deadline_ms
        for voice in voices
    )
    report = {
        "schema": "midibrave.atlas-flow.spark-runtime-qualification.v1",
        "passed": bool(passed),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "visible_gpus": torch.cuda.device_count(),
        "checkpoint": args.checkpoint.name,
        "block_deadline_ms": deadline_ms,
        "plan_ms_median": statistics.median(
            float(voice["plan_ms"]) for voice in voices
        ),
        "voices": voices,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
