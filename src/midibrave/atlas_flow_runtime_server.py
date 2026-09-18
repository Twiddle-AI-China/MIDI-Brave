from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from numbers import Real
from pathlib import Path
import time
from typing import Mapping

from aiohttp import WSCloseCode, WSMsgType, web
import numpy as np

from .atlas_flow_runtime import AtlasFlowLiveEngine
from .atlas_flow_runtime import DEFAULT_MORPH_SECONDS, LIVE_BLOCK_SAMPLES
from .atlas_flow_stream import StreamFormat, parse_stream, supported_profiles


# Minimum spacing between plans. A roam asks for a new trajectory on every
# pointer move, and each plan is a GPU burst that competes with block rendering
# for the same device and interpreter; on V100 a plan costs ~180 ms against
# ~70 ms on the retired GB10 host, so plans have to be spaced further apart
# there or the audio buffer starves. Override with ATLAS_FLOW_PLAN_INTERVAL.
PLAN_INTERVAL_SECONDS = float(os.environ.get("ATLAS_FLOW_PLAN_INTERVAL", "0.25"))


_ENGINE = web.AppKey("atlas_flow_engine", AtlasFlowLiveEngine)
_CLIENT_LOCK = web.AppKey("atlas_flow_client_lock", asyncio.Lock)
_ACTIVE = web.AppKey("atlas_flow_active_client", dict)


def _number(message: Mapping[str, object], name: str) -> float:
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(message: Mapping[str, object], name: str) -> int:
    value = _number(message, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _controls(message: Mapping[str, object]) -> dict[str, object]:
    raw_pca = message.get("pcaNormalized")
    pca: list[float] | None = None
    if raw_pca is not None:
        if not isinstance(raw_pca, (list, tuple)) or len(raw_pca) != 8:
            raise ValueError("pcaNormalized must contain exactly eight values")
        pca = []
        for index, raw_value in enumerate(raw_pca):
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"pcaNormalized[{index}] must be finite")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"pcaNormalized[{index}] must be finite")
            pca.append(float(np.clip(value, -1.0, 1.0)))
    x = message.get("x")
    y = message.get("y")
    if pca is None and (x is None or y is None):
        raise ValueError("control needs pcaNormalized or legacy x/y")
    morph = message.get("morphSeconds", DEFAULT_MORPH_SECONDS)
    if isinstance(morph, bool) or not isinstance(morph, Real) or not math.isfinite(float(morph)):
        raise ValueError("morphSeconds must be finite")
    return {
        "pca_normalized": pca,
        "x": None if x is None else float(np.clip(_number(message, "x"), -1.0, 1.0)),
        "y": None if y is None else float(np.clip(_number(message, "y"), -1.0, 1.0)),
        "note": int(np.clip(_integer(message, "note"), 36, 71)),
        "velocity": float(np.clip(_number(message, "velocity"), 0.0, 1.0)),
        "temperature": float(np.clip(_number(message, "temperature"), 0.0, 1.0)),
        "morph_seconds": float(np.clip(float(morph), 0.5, 5.0)),
    }


def parse_message(message: object) -> dict[str, object]:
    if not isinstance(message, Mapping):
        raise ValueError("message must be a JSON object")
    message_type = message.get("type")
    if message_type not in {
        "start", "control", "note_off", "reseed", "buffer", "stop",
    }:
        raise ValueError("unsupported message type")
    if message_type == "start":
        return {
            "type": "start", **_controls(message),
            "seed": _integer(message, "seed"),
            "stream": parse_stream(message.get("stream")),
        }
    if message_type == "control":
        return {
            "type": "control", "seq": _integer(message, "seq"),
            **_controls(message),
        }
    if message_type == "reseed":
        return {"type": "reseed", "seed": _integer(message, "seed")}
    if message_type == "buffer":
        return {
            "type": "buffer",
            "bufferedFrames": max(0, _integer(message, "bufferedFrames")),
            "underruns": max(0, _integer(message, "underruns")),
        }
    return {"type": str(message_type)}


@dataclass
class _ClientState:
    buffered_frames: int = 0
    underruns: int = 0
    running: bool = False
    stream: StreamFormat = field(default_factory=StreamFormat)
    wake_producer: asyncio.Event = field(default_factory=asyncio.Event)

    def pause(self) -> None:
        self.running = False
        self.wake_producer.clear()

    def resume(self) -> None:
        self.running = True
        self.wake_producer.set()


async def _send_error(ws: web.WebSocketResponse, error: object) -> None:
    if ws.closed:
        return
    try:
        await ws.send_json({
            "type": "error",
            "message": str(error).strip() or type(error).__name__,
        })
    except ConnectionError:
        pass


async def _receive(
    ws: web.WebSocketResponse,
    session,
    state: _ClientState,
    planner_wake: asyncio.Event,
) -> None:
    async for frame in ws:
        if frame.type == WSMsgType.TEXT:
            try:
                message = parse_message(json.loads(frame.data))
                kind = message["type"]
                if kind == "buffer":
                    state.buffered_frames = int(message["bufferedFrames"])
                    state.underruns = int(message["underruns"])
                    continue
                if kind == "stop":
                    session.stop()
                    continue
                if kind == "note_off":
                    session.note_off()
                    continue
                if kind == "reseed":
                    session.reseed(int(message["seed"]))
                    planner_wake.set()
                    continue
                if kind == "start":
                    state.stream = message["stream"]
                    await ws.send_json({
                        "type": "stream",
                        **state.stream.describe(LIVE_BLOCK_SAMPLES),
                    })
                    session.request_start(
                        pca_normalized=message["pca_normalized"],
                        x=message["x"], y=message["y"],
                        note=message["note"], velocity=message["velocity"],
                        seed=message["seed"],
                        temperature=message["temperature"],
                        morph_seconds=message["morph_seconds"],
                    )
                    state.resume()
                else:
                    session.request_control(
                        seq=message["seq"],
                        pca_normalized=message["pca_normalized"],
                        x=message["x"], y=message["y"],
                        note=message["note"], velocity=message["velocity"],
                        temperature=message["temperature"],
                        morph_seconds=message["morph_seconds"],
                    )
                planner_wake.set()
            except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
                await _send_error(ws, error)
        elif frame.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR}:
            break


async def _produce(
    ws: web.WebSocketResponse,
    session,
    state: _ClientState,
    sample_rate: float,
    block_samples: int,
) -> None:
    block_seconds = block_samples / sample_rate
    block_index = 0
    render_times: list[float] = []
    while not ws.closed:
        if not state.running:
            await state.wake_producer.wait()
            continue
        target_frames = state.stream.target_frames(block_samples)
        if state.buffered_frames > target_frames:
            await asyncio.sleep(0.005)
            continue
        started = time.perf_counter()
        try:
            pcm, render_ms = session.render_block()
            if not state.running:
                continue
            value = np.asarray(pcm, dtype="<f4")
            if value.shape != (block_samples, 2) or not np.isfinite(value).all():
                raise RuntimeError("runtime emitted invalid PCM")
            await ws.send_bytes(state.stream.encode(value))
            render_times.append(float(render_ms))
            if len(render_times) > 256:
                del render_times[:-256]
            block_index += 1
            # An idle voice is silence, and silence is not worth a thin link.
            # The next start message resumes production.
            if session.snapshot()["lifecycle"] == "idle":
                state.pause()
                continue
            # Telemetry drives the on-map voice marker, so keep it twice as
            # frequent as the original dashboard needed. It is a few hundred
            # bytes against an audio stream measured in kilobytes.
            if block_index % 4 == 0:
                p50, p95 = np.percentile(render_times, (50, 95))
                await ws.send_json({
                    "type": "telemetry",
                    "renderP50Ms": float(p50),
                    "renderP95Ms": float(p95),
                    "bufferedFrames": state.buffered_frames,
                    "underruns": state.underruns,
                    **session.snapshot(),
                })
            elapsed = time.perf_counter() - started
            # Filling at 0.35 of real time refills a drained buffer quickly but
            # overshoots the client's ceiling, and every overshoot the client
            # sheds is an audible discontinuity. 0.6 still recovers faster than
            # playback drains without running the queue away.
            pacing = 0.6 if state.buffered_frames < target_frames * 0.5 else 0.97
            await asyncio.sleep(max(0.0, block_seconds * pacing - elapsed))
        except asyncio.CancelledError:
            raise
        except (ConnectionError, RuntimeError, TypeError, ValueError) as error:
            state.pause()
            if not ws.closed:
                await _send_error(ws, error)


async def _plan(
    ws: web.WebSocketResponse,
    session,
    executor: ThreadPoolExecutor,
    planner_wake: asyncio.Event,
) -> None:
    loop = asyncio.get_running_loop()
    last_started = 0.0
    while not ws.closed:
        await planner_wake.wait()
        planner_wake.clear()
        while session.plan_pending() and not ws.closed:
            delay = PLAN_INTERVAL_SECONDS - (time.perf_counter() - last_started)
            if delay > 0.0:
                await asyncio.sleep(delay)
            last_started = time.perf_counter()
            try:
                await loop.run_in_executor(executor, session.plan_latest)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, TypeError, ValueError) as error:
                await _send_error(ws, error)


def create_app(engine: AtlasFlowLiveEngine) -> web.Application:
    app = web.Application(client_max_size=64 * 1024)
    app[_ENGINE] = engine
    app[_CLIENT_LOCK] = asyncio.Lock()
    app[_ACTIVE] = {"ws": None}

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            **engine.status(),
            "streamProfiles": supported_profiles(LIVE_BLOCK_SAMPLES),
        })

    async def runtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15.0, max_msg_size=64 * 1024)
        await ws.prepare(request)
        # One voice, newest client wins. A reloaded tab (or one whose socket is
        # still being reaped) would otherwise lock every later visitor out.
        holder = request.app[_ACTIVE]
        previous = holder.get("ws")
        if previous is not None and previous is not ws and not previous.closed:
            await previous.close(code=WSCloseCode.GOING_AWAY, message=b"superseded")
        lock = request.app[_CLIENT_LOCK]
        try:
            await asyncio.wait_for(lock.acquire(), timeout=10.0)
        except asyncio.TimeoutError:
            await ws.send_json({"type": "error", "message": "runtime is busy"})
            await ws.close()
            return ws
        holder["ws"] = ws
        try:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-flow")
            receiver = producer = planner = None
            try:
                session = request.app[_ENGINE].new_session()
                status = request.app[_ENGINE].status()
                await ws.send_json({"type": "ready", **status})
                state = _ClientState()
                planner_wake = asyncio.Event()
                receiver = asyncio.create_task(
                    _receive(ws, session, state, planner_wake), name="atlas-flow-receive",
                )
                producer = asyncio.create_task(
                    _produce(
                        ws, session, state,
                        float(status["sampleRate"]), int(status["blockSamples"]),
                    ),
                    name="atlas-flow-produce",
                )
                planner = asyncio.create_task(
                    _plan(ws, session, executor, planner_wake),
                    name="atlas-flow-plan",
                )
                await receiver
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                await _send_error(ws, error)
            finally:
                tasks = [item for item in (receiver, producer, planner) if item is not None]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            lock.release()
            if holder.get("ws") is ws:
                holder["ws"] = None
        return ws

    app.router.add_get("/api/health", health)
    app.router.add_get("/api/runtime-status", health)
    app.router.add_get("/runtime", runtime)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Atlas Flow CUDA runtime")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = AtlasFlowLiveEngine.load(args.config, args.checkpoint, args.device)
    web.run_app(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
