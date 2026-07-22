import asyncio
import json
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from midibrave.realtime_server import create_app, parse_message


def test_control_protocol_clamps_values_and_requires_sequence():
    source = {
        "type": "control",
        "seq": 4,
        "x": 2,
        "y": -3,
        "note": 200,
        "velocity": -1,
        "ignored": "field",
    }

    message = parse_message(source)

    assert message == {
        "type": "control",
        "seq": 4,
        "x": 1.0,
        "y": -1.0,
        "note": 109,
        "velocity": 0.0,
    }
    assert message is not source
    with pytest.raises(ValueError, match="message type"):
        parse_message({"type": "unknown"})


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "control", "seq": True, "x": 0, "y": 0,
         "note": 60, "velocity": 0.8},
        {"type": "control", "seq": 1, "x": float("nan"), "y": 0,
         "note": 60, "velocity": 0.8},
        {"type": "start", "x": 0, "y": float("inf"), "note": 60,
         "velocity": 0.8, "seed": 0},
        {"type": "reseed", "seed": False},
        {"type": "buffer", "bufferedFrames": "12", "underruns": 0},
    ],
)
def test_protocol_rejects_boolean_non_numeric_and_non_finite_values(payload):
    with pytest.raises(ValueError, match="finite|number|integer"):
        parse_message(payload)


def test_status_route_uses_engine_contract_and_none_registers_no_static_route():
    class Engine:
        def status(self):
            return {"sampleRate": 44100, "blockSamples": 512, "cuda": "RTX"}

    async def scenario():
        client = TestClient(TestServer(create_app(Engine(), None)))
        await client.start_server()
        try:
            response = await client.get("/api/runtime-status")
            assert response.status == 200
            assert await response.json() == {
                "sampleRate": 44100,
                "blockSamples": 512,
                "cuda": "RTX",
            }
            static_response = await client.get("/")
            assert static_response.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_protocol_normalizes_each_supported_message_type():
    assert parse_message({
        "type": "start", "x": 2, "y": -2, "note": 12,
        "velocity": 2, "seed": 7,
    }) == {
        "type": "start", "x": 1.0, "y": -1.0, "note": 21,
        "velocity": 1.0, "seed": 7,
    }
    assert parse_message({"type": "reseed", "seed": -4}) == {
        "type": "reseed", "seed": -4,
    }
    assert parse_message({
        "type": "buffer", "bufferedFrames": -20, "underruns": -1,
    }) == {"type": "buffer", "bufferedFrames": 0, "underruns": 0}
    assert parse_message({"type": "stop", "ignored": True}) == {
        "type": "stop",
    }


def test_static_instrument_routes_serve_only_named_assets(tmp_path):
    files = {
        "/": ("index.html", "index"),
        "/instrument.js": ("instrument.js", "instrument"),
        "/instrument-core.js": ("instrument-core.js", "core"),
        "/instrument.css": ("instrument.css", "css"),
        "/pcm-player-worklet.js": ("pcm-player-worklet.js", "worklet"),
    }
    for filename, content in files.values():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    class Engine:
        def status(self):
            return {}

    async def scenario():
        client = TestClient(TestServer(create_app(Engine(), tmp_path)))
        await client.start_server()
        try:
            for route, (_, content) in files.items():
                response = await client.get(route)
                assert response.status == 200
                assert await response.text() == content
        finally:
            await client.close()

    asyncio.run(scenario())


class RecordingSession:
    def __init__(self, *, non_finite=False):
        self.non_finite = non_finite
        self.control = SimpleNamespace(
            seq=0, x=0.0, y=0.0, note=60, velocity=0.8,
        )
        self.runtime_threads = []
        self.update_sequences = []
        self.render_count = 0
        self.reseed_values = []
        self.lock = threading.Lock()

    def _record_runtime_thread(self):
        with self.lock:
            self.runtime_threads.append(threading.current_thread())

    def start(self, *, x, y, note, velocity, seed):
        self._record_runtime_thread()
        with self.lock:
            self.control = SimpleNamespace(
                seq=0, x=x, y=y, note=note, velocity=velocity,
            )

    def update(self, *, seq, x, y, note, velocity):
        with self.lock:
            self.update_sequences.append(seq)
            if seq <= self.control.seq:
                return False
            self.control = SimpleNamespace(
                seq=seq, x=x, y=y, note=note, velocity=velocity,
            )
            return True

    def reseed(self, seed):
        self._record_runtime_thread()
        with self.lock:
            self.reseed_values.append(seed)

    def render_block(self):
        self._record_runtime_thread()
        with self.lock:
            self.render_count += 1
            render_count = self.render_count
        pcm = np.full((512, 2), 0.125, dtype=np.float32)
        if self.non_finite:
            pcm[0, 0] = np.nan
        return pcm, float(render_count)

    def snapshot(self):
        with self.lock:
            return SimpleNamespace(**vars(self.control))


class RecordingEngine:
    def __init__(self, *, non_finite=False):
        self.non_finite = non_finite
        self.sessions = []

    def status(self):
        return {
            "engine": "test-runtime",
            "sampleRate": 44100,
            "blockSamples": 512,
        }

    def new_session(self):
        session = RecordingSession(non_finite=self.non_finite)
        self.sessions.append(session)
        return session


async def receive_until(ws, expected_type):
    deadline = asyncio.get_running_loop().time() + 2.0
    binary_messages = []
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        assert remaining > 0, f"timed out waiting for {expected_type}"
        message = await ws.receive(timeout=remaining)
        if message.type == WSMsgType.BINARY:
            binary_messages.append(message.data)
            if expected_type == "binary":
                return message.data, binary_messages
        elif message.type == WSMsgType.TEXT:
            payload = json.loads(message.data)
            if payload.get("type") == expected_type:
                return payload, binary_messages
        else:
            pytest.fail(
                f"websocket closed while waiting for {expected_type}: {message}"
            )


def test_websocket_streams_telemetry_and_serializes_runtime_calls():
    async def scenario():
        engine = RecordingEngine()
        client = TestClient(TestServer(create_app(engine, None)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/runtime")
            ready = await ws.receive_json()
            assert ready == {
                "type": "ready",
                "engine": "test-runtime",
                "sampleRate": 44100,
                "blockSamples": 512,
            }
            await ws.send_json({
                "type": "start", "x": 0.1, "y": -0.2, "note": 64,
                "velocity": 0.7, "seed": 3,
            })
            await ws.send_json({
                "type": "control", "seq": 4, "x": 0.8, "y": -0.5,
                "note": 72, "velocity": 0.6,
            })
            await ws.send_json({
                "type": "control", "seq": 2, "x": -0.8, "y": 0.5,
                "note": 48, "velocity": 0.2,
            })
            await ws.send_json({"type": "reseed", "seed": 9})

            telemetry, binary = await receive_until(ws, "telemetry")
            assert binary
            assert all(len(block) == 512 * 2 * 4 for block in binary)
            assert telemetry["renderP50Ms"] == pytest.approx(4.5)
            assert telemetry["renderP95Ms"] == pytest.approx(7.65)
            assert telemetry["bufferedFrames"] == 0
            assert telemetry["underruns"] == 0
            assert telemetry["appliedSeq"] == 4
            assert telemetry["note"] == 72
            assert telemetry["x"] == pytest.approx(0.8)
            assert telemetry["y"] == pytest.approx(-0.5)

            await ws.send_json({
                "type": "buffer", "bufferedFrames": 9000, "underruns": 2,
            })
            await asyncio.sleep(0.02)
            session = engine.sessions[0]
            assert session.update_sequences == [4, 2]
            assert session.reseed_values == [9]
            runtime_threads = set(session.runtime_threads)
            assert len(runtime_threads) == 1
            assert next(iter(runtime_threads)) is not threading.current_thread()

            await ws.close()
            deadline = time.monotonic() + 2.0
            while any(thread.is_alive() for thread in runtime_threads):
                assert time.monotonic() < deadline, "executor thread leaked"
                await asyncio.sleep(0.01)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_busy_websocket_does_not_create_a_session_or_interrupt_first_client():
    async def scenario():
        engine = RecordingEngine()
        client = TestClient(TestServer(create_app(engine, None)))
        await client.start_server()
        try:
            first = await client.ws_connect("/runtime")
            assert (await first.receive_json())["type"] == "ready"

            second = await client.ws_connect("/runtime")
            busy = await second.receive_json()
            assert busy["type"] == "error"
            assert "busy" in busy["message"]
            assert len(engine.sessions) == 1

            await first.send_json({
                "type": "start", "x": 0, "y": 0, "note": 60,
                "velocity": 0.8, "seed": 0,
            })
            pcm, _ = await receive_until(first, "binary")
            assert len(pcm) == 512 * 2 * 4
            assert len(engine.sessions) == 1
            await second.close()
            await first.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_non_finite_runtime_error_stops_audio_without_closing_websocket():
    async def scenario():
        engine = RecordingEngine(non_finite=True)
        client = TestClient(TestServer(create_app(engine, None)))
        await client.start_server()
        try:
            ws = await client.ws_connect("/runtime")
            assert (await ws.receive_json())["type"] == "ready"
            await ws.send_json({
                "type": "start", "x": 0, "y": 0, "note": 60,
                "velocity": 0.8, "seed": 0,
            })

            error, binary = await receive_until(ws, "error")
            assert not binary
            assert "finite" in error["message"]
            assert not ws.closed
            await ws.ping()
            await ws.close()
        finally:
            await client.close()

    asyncio.run(scenario())
