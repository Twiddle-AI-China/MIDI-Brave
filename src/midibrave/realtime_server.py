from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Mapping

from aiohttp import WSMsgType, web
import numpy as np


_ENGINE = web.AppKey("midibrave_realtime_engine", object)
_CLIENT_LOCK = web.AppKey("midibrave_realtime_client_lock", asyncio.Lock)


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
    return {
        "x": float(np.clip(_number(message, "x"), -1.0, 1.0)),
        "y": float(np.clip(_number(message, "y"), -1.0, 1.0)),
        "note": int(np.clip(_integer(message, "note"), 21, 109)),
        "velocity": float(
            np.clip(_number(message, "velocity"), 0.0, 1.0)
        ),
    }


def parse_message(message: object) -> dict[str, object]:
    """Validate and normalize one browser control frame."""
    if not isinstance(message, Mapping):
        raise ValueError("message must be a JSON object")
    message_type = message.get("type")
    if message_type not in {"start", "control", "reseed", "buffer", "stop"}:
        raise ValueError("unsupported message type")
    if message_type == "start":
        return {
            "type": "start",
            **_controls(message),
            "seed": _integer(message, "seed"),
        }
    if message_type == "control":
        return {
            "type": "control",
            "seq": _integer(message, "seq"),
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
    return {"type": "stop"}


@dataclass
class _ClientState:
    buffered_frames: int = 0
    underruns: int = 0
    running: bool = False
    wake_producer: asyncio.Event = field(default_factory=asyncio.Event)


async def _send_error(ws: web.WebSocketResponse, error: object) -> None:
    if ws.closed:
        return
    message = str(error).strip() or type(error).__name__
    try:
        await ws.send_json({"type": "error", "message": message})
    except ConnectionError:
        pass


async def _receive_messages(
        ws: web.WebSocketResponse,
        session,
        executor: ThreadPoolExecutor,
        state: _ClientState,
) -> None:
    loop = asyncio.get_running_loop()
    async for frame in ws:
        if frame.type == WSMsgType.TEXT:
            try:
                message = parse_message(json.loads(frame.data))
                message_type = message["type"]
                if message_type == "start":
                    state.running = False
                    state.wake_producer.clear()
                    await loop.run_in_executor(
                        executor,
                        partial(
                            session.start,
                            x=message["x"],
                            y=message["y"],
                            note=message["note"],
                            velocity=message["velocity"],
                            seed=message["seed"],
                        ),
                    )
                    state.running = True
                    state.wake_producer.set()
                elif message_type == "control":
                    session.update(
                        seq=message["seq"],
                        x=message["x"],
                        y=message["y"],
                        note=message["note"],
                        velocity=message["velocity"],
                    )
                elif message_type == "reseed":
                    was_running = state.running
                    state.running = False
                    state.wake_producer.clear()
                    await loop.run_in_executor(
                        executor, session.reseed, message["seed"]
                    )
                    state.running = was_running
                    if was_running:
                        state.wake_producer.set()
                elif message_type == "buffer":
                    state.buffered_frames = message["bufferedFrames"]
                    state.underruns = message["underruns"]
                else:
                    state.running = False
                    state.wake_producer.clear()
            except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
                state.running = False
                state.wake_producer.clear()
                await _send_error(ws, error)
        elif frame.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR}:
            break


async def _produce_audio(
        ws: web.WebSocketResponse,
        session,
        executor: ThreadPoolExecutor,
        state: _ClientState,
        sample_rate: float,
        block_samples: int,
) -> None:
    loop = asyncio.get_running_loop()
    block_seconds = block_samples / sample_rate
    render_times: list[float] = []
    block_index = 0
    while not ws.closed:
        if not state.running:
            await state.wake_producer.wait()
            continue
        if state.buffered_frames > 8192:
            await asyncio.sleep(0.005)
            continue
        started = time.perf_counter()
        try:
            pcm, render_ms = await loop.run_in_executor(
                executor, session.render_block
            )
            if not state.running:
                continue
            pcm_array = np.asarray(pcm)
            render_ms = float(render_ms)
            if not np.isfinite(pcm_array).all() or not math.isfinite(render_ms):
                raise RuntimeError("runtime emitted non-finite audio or timing")
            await ws.send_bytes(
                pcm_array.astype("<f4", copy=False).tobytes()
            )
            render_times.append(render_ms)
            if len(render_times) > 256:
                del render_times[:-256]
            block_index += 1
            if block_index % 8 == 0:
                control = session.snapshot()
                p50, p95 = np.percentile(render_times, (50, 95))
                await ws.send_json({
                    "type": "telemetry",
                    "renderP50Ms": float(p50),
                    "renderP95Ms": float(p95),
                    "bufferedFrames": state.buffered_frames,
                    "underruns": state.underruns,
                    "appliedSeq": control.seq,
                    "note": control.note,
                    "x": control.x,
                    "y": control.y,
                })
            elapsed = time.perf_counter() - started
            pacing = 0.35 if state.buffered_frames < 4096 else 0.97
            await asyncio.sleep(max(0.0, block_seconds * pacing - elapsed))
        except asyncio.CancelledError:
            raise
        except (ConnectionError, RuntimeError, TypeError, ValueError) as error:
            state.running = False
            state.wake_producer.clear()
            if ws.closed:
                return
            await _send_error(ws, error)


def _static_handler(path: Path):
    async def serve(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(path)

    return serve


def create_app(engine, web_root: str | Path | None) -> web.Application:
    """Create the single-client realtime instrument application."""
    app = web.Application()
    app[_ENGINE] = engine
    app[_CLIENT_LOCK] = asyncio.Lock()

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(engine.status())

    async def runtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=15.0, max_msg_size=64 * 1024)
        await ws.prepare(request)
        client_lock = request.app[_CLIENT_LOCK]
        if client_lock.locked():
            await ws.send_json({
                "type": "error",
                "message": "realtime runtime is busy",
            })
            await ws.close()
            return ws

        async with client_lock:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="midibrave-runtime"
            )
            receiver = None
            producer = None
            try:
                session = request.app[_ENGINE].new_session()
                runtime_status = request.app[_ENGINE].status()
                sample_rate = float(runtime_status["sampleRate"])
                block_samples = int(runtime_status["blockSamples"])
                if (
                    not math.isfinite(sample_rate)
                    or sample_rate <= 0
                    or block_samples <= 0
                ):
                    raise RuntimeError("runtime audio geometry is invalid")
                await ws.send_json({"type": "ready", **runtime_status})
                state = _ClientState()
                receiver = asyncio.create_task(
                    _receive_messages(ws, session, executor, state),
                    name="midibrave-runtime-receiver",
                )
                producer = asyncio.create_task(
                    _produce_audio(
                        ws,
                        session,
                        executor,
                        state,
                        sample_rate,
                        block_samples,
                    ),
                    name="midibrave-runtime-producer",
                )
                await receiver
            except (RuntimeError, TypeError, ValueError, KeyError) as error:
                await _send_error(ws, error)
            finally:
                tasks = [task for task in (receiver, producer) if task is not None]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                executor.shutdown(wait=True, cancel_futures=True)
        return ws

    app.router.add_get("/api/runtime-status", status)
    app.router.add_get("/runtime", runtime)
    if web_root is not None:
        root = Path(web_root)
        assets = {
            "/": "index.html",
            "/instrument.js": "instrument.js",
            "/instrument-core.js": "instrument-core.js",
            "/instrument.css": "instrument.css",
            "/pcm-player-worklet.js": "pcm-player-worklet.js",
        }
        for route, filename in assets.items():
            app.router.add_get(route, _static_handler(root / filename))
    return app
