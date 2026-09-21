"""Short V100 qualification for the Atlas Flow demo runtime.

The Kraken migration record left this open: code and weights were copied but no
forward pass was ever run on SM70. This checks the three things that could
plausibly differ from Spark's GB10 — that the checkpoint loads, that planning
fits the audio deadline, and that the offline render path produces finite,
non-silent audio — and prints one JSON object.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from midibrave.atlas_flow_demo_server import encode_vorbis, parse_take, render_take
from midibrave.atlas_flow_runtime import LIVE_BLOCK_SAMPLES, AtlasFlowLiveEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=48)
    args = parser.parse_args()

    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }

    started = time.perf_counter()
    engine = AtlasFlowLiveEngine.load(args.config, args.checkpoint, "cuda:0")
    report["loadSeconds"] = round(time.perf_counter() - started, 2)
    report["checkpoint"] = engine.checkpoint.name
    report["presets"] = len(engine.atlas.preset_ids)

    rate = engine.config.data.sample_rate
    deadline = LIVE_BLOCK_SAMPLES / rate * 1000.0
    report["blockDeadlineMs"] = round(deadline, 2)

    session = engine.new_session()
    plan_started = time.perf_counter()
    session.start(
        pca_normalized=engine.default_normalized.astype(float).tolist(),
        note=50, velocity=0.8, seed=20260822, temperature=0.0, morph_seconds=0.6,
    )
    report["firstPlanMs"] = round((time.perf_counter() - plan_started) * 1000.0, 1)

    blocks = []
    times = []
    for _ in range(args.blocks):
        began = time.perf_counter()
        pcm, _ = session.render_block()
        times.append((time.perf_counter() - began) * 1000.0)
        blocks.append(pcm)
    audio = np.concatenate(blocks, axis=0)[:, 0]
    report["blockP50Ms"] = round(float(np.percentile(times, 50)), 2)
    report["blockP95Ms"] = round(float(np.percentile(times, 95)), 2)
    report["realTimeFactor"] = round(deadline / max(float(np.percentile(times, 95)), 1e-6), 1)
    report["finite"] = bool(np.isfinite(audio).all())
    report["peakDbfs"] = round(20 * math.log10(max(float(np.abs(audio).max()), 1e-9)), 2)
    report["silentBlocks"] = int(sum(1 for block in blocks if float(np.abs(block).max()) < 1e-6))

    # Re-plan mid-note the way a roam does, then the offline render path.
    morph_started = time.perf_counter()
    session.update(
        seq=2, pca_normalized=engine.normalize(engine.atlas.coordinates[7]).astype(float).tolist(),
        note=57, velocity=0.8, temperature=0.0, morph_seconds=0.6,
    )
    report["morphPlanMs"] = round((time.perf_counter() - morph_started) * 1000.0, 1)

    take = parse_take({
        "steps": [
            {"pca": engine.default_normalized.astype(float).tolist(), "note": 50, "seconds": 2.0},
            {"pca": engine.normalize(engine.atlas.coordinates[7]).astype(float).tolist(),
             "note": 57, "seconds": 2.0},
        ],
        "seed": 20260822, "velocity": 0.8, "temperature": 0.0, "morphSeconds": 1.0,
    }, engine.atlas.components.shape[0])
    render_started = time.perf_counter()
    result = render_take(engine, take)
    encoded = encode_vorbis(result["audio"], result["sampleRate"])
    report["offlineRenderSeconds"] = round(time.perf_counter() - render_started, 2)
    report["offlineAudioSeconds"] = result["seconds"]
    report["offlineBytes"] = len(encoded)
    report["offlinePeakDbfs"] = result["peakDbfs"]
    report["offlineRmsDbfs"] = result["rmsDbfs"]
    report["passed"] = bool(
        report["finite"]
        and report["silentBlocks"] == 0
        and report["blockP95Ms"] < deadline
        and report["offlineBytes"] > 1000
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
