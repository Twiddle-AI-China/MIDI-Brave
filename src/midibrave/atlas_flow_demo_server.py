"""Bandwidth-light Atlas Flow web demo.

The live WebSocket instrument streams 44.1 kHz stereo float PCM, which needs
roughly 2.8 Mbit/s. Remote access to Spark goes through a ~1.3 Mbit/s tunnel,
so this server adds a render-on-demand path instead: the browser posts a short
control take (one or more atlas waypoints), the GPU renders it offline with the
evaluation-grade solver, and only compressed Vorbis audio crosses the link.

The live WebSocket runtime is still mounted under ``/live/`` for LAN clients.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Mapping, Sequence

from aiohttp import web
import numpy as np
import soundfile as sf

from . import atlas_flow_runtime as runtime_module
from .atlas_flow_runtime import LIVE_BLOCK_SAMPLES, AtlasFlowLiveEngine
from .atlas_flow_runtime_server import create_app as create_runtime_app
from .atlas_flow_stream import supported_profiles


RENDER_SCHEMA = "midibrave.atlas-flow.demo-take.v2"
OFFLINE_SOLVER_STEPS = 8
MAX_STEPS = 8
MAX_VOICES = 4
MAX_TOTAL_SECONDS = 24.0
MIN_STEP_SECONDS = 1.0
MAX_STEP_SECONDS = 10.0
RELEASE_SECONDS = 2.4

_ENGINE = web.AppKey("atlas_flow_demo_engine", AtlasFlowLiveEngine)
_EXECUTOR = web.AppKey("atlas_flow_demo_executor", ThreadPoolExecutor)
_LOCK = web.AppKey("atlas_flow_demo_lock", asyncio.Lock)
_CACHE = web.AppKey("atlas_flow_demo_cache", Path)
_EVALUATION = web.AppKey("atlas_flow_demo_evaluation", Path)
_WEB = web.AppKey("atlas_flow_demo_web", Path)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _clip(value: object, name: str, low: float, high: float) -> float:
    return float(np.clip(_finite(value, name), low, high))


def parse_take(payload: object, dimensions: int) -> dict[str, object]:
    """Validate a render request into a canonical, cache-keyable take."""
    if not isinstance(payload, Mapping):
        raise ValueError("request must be a JSON object")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("steps must be a list")
    if not 1 <= len(raw_steps) <= MAX_STEPS:
        raise ValueError(f"steps must hold 1 to {MAX_STEPS} waypoints")
    steps: list[dict[str, object]] = []
    total = 0.0
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            raise ValueError(f"steps[{index}] must be an object")
        pca = raw.get("pca")
        if not isinstance(pca, Sequence) or len(pca) != dimensions:
            raise ValueError(f"steps[{index}].pca needs {dimensions} values")
        coordinates = [
            _clip(item, f"steps[{index}].pca[{axis}]", -1.0, 1.0)
            for axis, item in enumerate(pca)
        ]
        # The model is monophonic, so a chord is several independent voices
        # rendered at the same timbre coordinate and summed.
        raw_notes = raw.get("notes")
        if raw_notes is None:
            raw_notes = [raw.get("note")]
        if not isinstance(raw_notes, Sequence) or isinstance(raw_notes, (str, bytes)):
            raise ValueError(f"steps[{index}].notes must be a list")
        if not 1 <= len(raw_notes) <= MAX_VOICES:
            raise ValueError(f"steps[{index}].notes takes 1 to {MAX_VOICES} notes")
        notes = []
        for voice, item in enumerate(raw_notes):
            value = _clip(item, f"steps[{index}].notes[{voice}]", 36, 71)
            if not float(value).is_integer():
                raise ValueError(f"steps[{index}].notes[{voice}] must be an integer")
            notes.append(int(value))
        notes = sorted(dict.fromkeys(notes))
        note = notes[0]
        seconds = _clip(
            raw.get("seconds", 4.0), f"steps[{index}].seconds",
            MIN_STEP_SECONDS, MAX_STEP_SECONDS,
        )
        total += seconds
        steps.append({
            "pca": [round(item, 6) for item in coordinates],
            "note": int(note),
            "notes": notes,
            "seconds": round(seconds, 3),
        })
    if total > MAX_TOTAL_SECONDS:
        raise ValueError(f"take may not exceed {MAX_TOTAL_SECONDS:g} seconds")
    seed = _finite(payload.get("seed", 20260822), "seed")
    if not seed.is_integer():
        raise ValueError("seed must be an integer")
    return {
        "steps": steps,
        "seed": int(seed) % 2_147_483_647,
        "velocity": round(_clip(payload.get("velocity", 0.8), "velocity", 0.0, 1.0), 4),
        "temperature": round(_clip(payload.get("temperature", 0.0), "temperature", 0.0, 1.0), 4),
        "morphSeconds": round(_clip(payload.get("morphSeconds", 2.0), "morphSeconds", 0.5, 5.0), 3),
        "solverSteps": OFFLINE_SOLVER_STEPS,
        "release": RELEASE_SECONDS,
    }


def take_identifier(take: Mapping[str, object]) -> str:
    canonical = json.dumps(take, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def render_take(engine: AtlasFlowLiveEngine, take: Mapping[str, object]) -> dict[str, object]:
    """Render one take offline. Call from a single worker thread only.

    ``LIVE_SOLVER_STEPS`` is a module constant read inside the planner, so the
    offline solver budget is applied by swapping it for the duration of the
    render. The caller serialises renders, so no other plan observes the swap.
    """
    rate = float(engine.config.data.sample_rate)
    plan_times: list[float] = []
    landings: list[dict[str, object]] = []
    voices = max(len(step["notes"]) for step in take["steps"])
    original = runtime_module.LIVE_SOLVER_STEPS
    runtime_module.LIVE_SOLVER_STEPS = int(take["solverSteps"])
    started = time.perf_counter()
    rendered: list[np.ndarray] = []
    try:
        # Each voice is a full independent pass at the same coordinates, so a
        # chord keeps every voice's own attack and morph instead of one voice
        # retriggered at different pitches.
        for voice in range(voices):
            session = engine.new_session()
            blocks: list[np.ndarray] = []
            for index, step in enumerate(take["steps"]):
                pitch = step["notes"][min(voice, len(step["notes"]) - 1)]
                plan_started = time.perf_counter()
                controls = {
                    "pca_normalized": list(step["pca"]),
                    "note": int(pitch),
                    "velocity": float(take["velocity"]),
                    "temperature": float(take["temperature"]),
                    "morph_seconds": float(take["morphSeconds"]),
                }
                if index == 0:
                    session.start(seed=int(take["seed"]) + voice, **controls)
                else:
                    session.update(seq=index + 1, **controls)
                if voice == 0:
                    plan_times.append((time.perf_counter() - plan_started) * 1000.0)
                for _ in range(max(1, math.ceil(float(step["seconds"]) * rate / LIVE_BLOCK_SAMPLES))):
                    blocks.append(session.render_block()[0])
                if voice == 0:
                    # Read the landing after the morph has run, while held.
                    reached = session.snapshot()
                    landings.append({
                        "notes": list(step["notes"]),
                        "component": reached.get("component"),
                        "planMode": reached.get("planMode"),
                        "projectedPca": reached.get("projectedPcaNormalized"),
                        "audiblePca": reached.get("audiblePcaNormalized"),
                    })
            session.note_off()
            for _ in range(max(1, math.ceil(float(take["release"]) * rate / LIVE_BLOCK_SAMPLES))):
                blocks.append(session.render_block()[0])
            rendered.append(np.concatenate(blocks, axis=0)[:, 0].astype(np.float32))
    finally:
        runtime_module.LIVE_SOLVER_STEPS = original
    length = min(item.shape[0] for item in rendered)
    # Summing n voices would raise the level by up to n; 1/sqrt(n) keeps the
    # loudness of a chord comparable to a single note.
    audio = sum(item[:length] for item in rendered) / math.sqrt(len(rendered))
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise RuntimeError("render produced non-finite audio")
    peak = float(np.abs(audio).max())
    return {
        "audio": audio,
        "sampleRate": int(rate),
        "seconds": round(audio.shape[0] / rate, 3),
        "peakDbfs": round(20.0 * math.log10(max(peak, 1.0e-9)), 2),
        "rmsDbfs": round(20.0 * math.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1.0e-9)), 2),
        "clipped": bool(peak > 1.0),
        "planMs": [round(value, 1) for value in plan_times],
        "renderMs": round((time.perf_counter() - started) * 1000.0, 1),
        "voices": voices,
        "landings": landings,
    }


def encode_vorbis(audio: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), rate, format="OGG", subtype="VORBIS")
    return buffer.getvalue()


def explained_variance(engine: AtlasFlowLiveEngine) -> list[float]:
    """Share of anchor variance each atlas axis carries, for honest axis labels."""
    atlas = engine.atlas
    centred = np.asarray(atlas.anchors, dtype=np.float64) - np.asarray(atlas.center, dtype=np.float64)
    total = float(np.var(centred, axis=0).sum())
    if total <= 0.0:
        return [0.0] * atlas.coordinates.shape[1]
    return [
        round(float(np.var(atlas.coordinates[:, axis]) / total), 5)
        for axis in range(atlas.coordinates.shape[1])
    ]


def _safe_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise web.HTTPBadRequest(text="invalid name")
    return name


def create_app(
    engine: AtlasFlowLiveEngine,
    *,
    cache_root: Path,
    evaluation_root: Path,
    web_root: Path,
) -> web.Application:
    app = web.Application(client_max_size=64 * 1024)
    app[_ENGINE] = engine
    app[_CACHE] = cache_root
    app[_EVALUATION] = evaluation_root
    app[_WEB] = web_root
    (cache_root / "takes").mkdir(parents=True, exist_ok=True)
    (cache_root / "audition").mkdir(parents=True, exist_ok=True)

    async def startup(application: web.Application) -> None:
        application[_EXECUTOR] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="atlas-flow-demo",
        )
        application[_LOCK] = asyncio.Lock()

    async def cleanup(application: web.Application) -> None:
        application[_EXECUTOR].shutdown(wait=True, cancel_futures=True)

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(request.app[_WEB] / "index.html")

    async def asset(request: web.Request) -> web.FileResponse:
        target = request.app[_WEB] / _safe_name(request.match_info["filename"])
        if not target.is_file():
            raise web.HTTPNotFound(text="asset not found")
        return web.FileResponse(target)

    async def status(request: web.Request) -> web.Response:
        payload = dict(request.app[_ENGINE].status())
        payload["offlineSolverSteps"] = OFFLINE_SOLVER_STEPS
        payload["maxSteps"] = MAX_STEPS
        payload["maxTotalSeconds"] = MAX_TOTAL_SECONDS
        payload["releaseSeconds"] = RELEASE_SECONDS
        payload["presetIds"] = list(request.app[_ENGINE].atlas.preset_ids)
        payload["streamProfiles"] = supported_profiles(LIVE_BLOCK_SAMPLES)
        payload["pcaExplained"] = explained_variance(request.app[_ENGINE])
        return web.json_response(payload)

    async def evaluation(request: web.Request) -> web.Response:
        path = request.app[_EVALUATION] / "evaluation.json"
        if not path.is_file():
            return web.json_response({"passed": False, "gates": {}, "rows": []}, status=202)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint"] = Path(str(payload.get("checkpoint", ""))).name
        for row in payload.get("rows", []):
            audio = row.get("audio")
            if isinstance(audio, Mapping):
                row["audio"] = {
                    mode: f"/api/audition/{Path(str(value)).stem}.ogg"
                    for mode, value in audio.items()
                }
        return web.json_response(payload)

    async def audition(request: web.Request) -> web.Response:
        name = _safe_name(request.match_info["name"])
        if not name.endswith(".ogg"):
            raise web.HTTPNotFound(text="unsupported audition format")
        target = request.app[_CACHE] / "audition" / name
        if not target.is_file():
            source = request.app[_EVALUATION] / "audio" / f"{name[:-4]}.wav"
            if not source.is_file():
                raise web.HTTPNotFound(text="audition clip not found")

            def transcode() -> None:
                data, rate = sf.read(source, dtype="float32", always_2d=True)
                payload = encode_vorbis(data[:, 0], int(rate))
                temporary = target.with_suffix(".ogg.tmp")
                temporary.write_bytes(payload)
                temporary.replace(target)

            await asyncio.get_running_loop().run_in_executor(
                request.app[_EXECUTOR], transcode,
            )
        return web.FileResponse(target, headers={"Cache-Control": "public, max-age=86400"})

    async def render(request: web.Request) -> web.Response:
        engine_ref = request.app[_ENGINE]
        try:
            take = parse_take(await request.json(), engine_ref.atlas.components.shape[0])
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)
        identifier = take_identifier(take)
        audio_path = request.app[_CACHE] / "takes" / f"{identifier}.ogg"
        meta_path = audio_path.with_suffix(".json")
        if audio_path.is_file() and meta_path.is_file():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if payload.get("schema") == RENDER_SCHEMA:
                payload["cached"] = True
                return web.json_response(payload)
        loop = asyncio.get_running_loop()
        async with request.app[_LOCK]:
            try:
                result = await loop.run_in_executor(
                    request.app[_EXECUTOR], render_take, engine_ref, take,
                )
                encoded = await loop.run_in_executor(
                    request.app[_EXECUTOR], encode_vorbis,
                    result["audio"], result["sampleRate"],
                )
            except (RuntimeError, TypeError, ValueError) as error:
                return web.json_response({"error": str(error)}, status=500)
        payload = {key: value for key, value in result.items() if key != "audio"}
        payload.update({
            "schema": RENDER_SCHEMA,
            "id": identifier,
            "url": f"/api/take/{identifier}.ogg",
            "bytes": len(encoded),
            "take": take,
            "cached": False,
        })
        temporary = audio_path.with_suffix(".ogg.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(audio_path)
        meta_path.write_text(json.dumps(payload), encoding="utf-8")
        return web.json_response(payload)

    async def take_audio(request: web.Request) -> web.FileResponse:
        target = request.app[_CACHE] / "takes" / _safe_name(request.match_info["name"])
        if target.suffix != ".ogg" or not target.is_file():
            raise web.HTTPNotFound(text="take not found")
        return web.FileResponse(target, headers={"Cache-Control": "public, max-age=86400"})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "checkpoint": request.app[_ENGINE].checkpoint.name})

    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/evaluation", evaluation)
    app.router.add_get("/api/audition/{name}", audition)
    app.router.add_get("/api/take/{name}", take_audio)
    app.router.add_post("/api/render", render)
    app.add_subapp("/live/", create_runtime_app(engine))
    app.router.add_get("/{filename}", asset)
    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Atlas Flow web demo")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18795)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    engine = AtlasFlowLiveEngine.load(args.config, args.checkpoint, args.device)
    web.run_app(
        create_app(
            engine,
            cache_root=args.cache_root,
            evaluation_root=args.evaluation_root,
            web_root=args.web_root,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
