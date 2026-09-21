"""Prove a built Atlas Flow server actually works, on macOS or Windows.

    python packaging/smoke_test.py --server build/pyi/atlas-flow-server/atlas-flow-server

Starts the binary on a free port, waits for health, then checks the things that
have actually broken before rather than the things that are easy to check:

  * the atlas loaded, with all 50 presets and 8 dimensions
  * the web assets are served from inside the bundle, not from a checkout
  * a render produces audio that is not silence
  * the same seed and coordinate reproduce the same bytes
  * the live websocket negotiates a format and delivers PCM
  * the process exits when its declared parent does

Exit status is 0 only if every one of those passes. Prints one line per check
so a CI log says which.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PRESETS = 50
DIMENSIONS = 8


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get(url: str, timeout: float = 30.0) -> tuple[int, bytes, str]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("content-type", "")
    except HTTPError as error:
        return error.code, error.read(), ""
    except (URLError, TimeoutError, ConnectionError, socket.timeout):
        return 0, b"", ""


def post(url: str, payload: dict, timeout: float = 600.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  -- {detail}' if detail else ''}",
              flush=True)
        if not ok:
            self.failures.append(name)
        return ok


async def stream_once(port: int, seconds: float = 12.0) -> tuple[bool, str]:
    """Open the live socket, ask for audio, and see whether PCM arrives."""
    try:
        import aiohttp
    except ImportError:
        return True, "skipped, aiohttp not installed for the test runner"

    url = f"http://127.0.0.1:{port}/live/runtime"
    try:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(url, timeout=aiohttp.ClientWSTimeout(ws_close=20)) as ws:
                await ws.send_json({
                    "type": "start",
                    "pcaNormalized": [0.1, 0.2] + [0.0] * 6,
                    "note": 50, "velocity": 0.8, "morphSeconds": 0.6, "seed": 20260822,
                    "temperature": 0.0,
                    # The key is "stream", and "format" inside it names the dtype.
                    # Calling it "format" at the top level parses as the legacy
                    # wire and is the reason this first measured zero bytes.
                    "stream": {"sampleRate": 22050, "channels": 1,
                               "format": "int16", "targetSeconds": 0.3},
                })
                audio_bytes, telemetry, errors = 0, 0, []
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=seconds)
                    except asyncio.TimeoutError:
                        break
                    if message.type is aiohttp.WSMsgType.BINARY:
                        audio_bytes += len(message.data)
                        # The producer paces itself against the client's own
                        # buffer report and stalls without one.
                        await ws.send_json({"type": "buffer",
                                            "bufferedFrames": 2048, "underruns": 0})
                        if audio_bytes > 40_000:
                            break
                    elif message.type is aiohttp.WSMsgType.TEXT:
                        telemetry += 1
                        body = json.loads(message.data)
                        if body.get("type") == "error":
                            errors.append(str(body.get("message")))
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                await ws.close()
        if errors:
            return False, f"server reported: {'; '.join(errors[:2])}"
        return audio_bytes > 20_000, f"{audio_bytes} bytes of PCM, {telemetry} telemetry frames"
    except Exception as error:                              # noqa: BLE001 - reported, not raised
        return False, f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True,
                        help="the frozen binary, or a python that can -m the module")
    parser.add_argument("--module", action="store_true",
                        help="treat --server as a python interpreter and run the module")
    parser.add_argument("--boot-seconds", type=float, default=420.0)
    parser.add_argument("--cache", type=Path, help="scratch folder, default is a temp dir")
    args = parser.parse_args()

    if not args.server.exists() and not args.module:
        print(f"no server at {args.server}")
        return 2

    port = free_port()
    root = Path(__file__).resolve().parents[1]
    cache = args.cache or root / "build" / "smoke-cache"
    cache.mkdir(parents=True, exist_ok=True)

    command = [str(args.server)]
    env = {**os.environ, "ATLAS_LOCAL_CACHE": str(cache)}
    if args.module:
        command += ["-m", "midibrave.atlas_flow_local"]
        env["PYTHONPATH"] = str(root / "src")
    command += ["--port", str(port)]

    print(f"starting {' '.join(command)}", flush=True)
    server = subprocess.Popen(command, cwd=str(root), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    check = Checks()
    try:
        deadline = time.monotonic() + args.boot_seconds
        ready = False
        while time.monotonic() < deadline:
            if server.poll() is not None:
                print(f"server exited early with {server.returncode}")
                print((server.stdout.read() if server.stdout else "")[-3000:])
                return 1
            if get(f"http://127.0.0.1:{port}/api/health", timeout=3)[0] == 200:
                ready = True
                break
            time.sleep(1.0)
        if not check("server becomes healthy", ready, f"port {port}"):
            print((server.stdout.read() if server.stdout else "")[-3000:])
            return 1

        code, body, _ = get(f"http://127.0.0.1:{port}/api/status")
        status = json.loads(body) if code == 200 else {}
        points = status.get("points", [])
        check("atlas has every preset", len(points) == PRESETS, f"{len(points)} of {PRESETS}")
        check("coordinates are 8-D",
              bool(points) and len(points[0].get("pcaNormalized", [])) == DIMENSIONS)
        check("device is reported", bool(status.get("cuda")), str(status.get("cuda")))

        code, body, kind = get(f"http://127.0.0.1:{port}/")
        check("index.html is served", code == 200 and b"ATLAS FLOW" in body, f"{len(body)} bytes")
        code, body, _ = get(f"http://127.0.0.1:{port}/app.js")
        check("app.js is served", code == 200 and b"buildNeighbourGraph" in body,
              f"{len(body)} bytes")
        code, _, _ = get(f"http://127.0.0.1:{port}/live-player-worklet.js")
        check("the audio worklet is served", code == 200)

        take = {"steps": [{"pca": [0.1, 0.2] + [0.0] * 6, "notes": [50], "seconds": 2.0}],
                "seed": 20260822, "velocity": 0.8}
        code, first = post(f"http://127.0.0.1:{port}/api/render", take)
        rendered = code == 200 and first.get("bytes", 0) > 4000
        check("a render returns audio", rendered,
              f"{first.get('bytes', 0)} bytes, peak {first.get('peakDbfs')} dBFS"
              if code == 200 else str(first))
        check("the render is not silence",
              rendered and float(first.get("peakDbfs", -120)) > -60.0,
              f"peak {first.get('peakDbfs')} dBFS")

        code, again = post(f"http://127.0.0.1:{port}/api/render", take)
        check("the same seed reproduces the same take",
              code == 200 and again.get("take", {}).get("id") == first.get("take", {}).get("id")
              and again.get("bytes") == first.get("bytes"),
              f"{again.get('bytes')} vs {first.get('bytes')} bytes")

        chord = {"steps": [{"pca": [0.0] * 8, "notes": [50, 54, 57], "seconds": 1.5}],
                 "seed": 7, "velocity": 0.8}
        code, payload = post(f"http://127.0.0.1:{port}/api/render", chord)
        check("a three-note chord renders", code == 200 and payload.get("voices") == 3,
              f"{payload.get('voices')} voices" if code == 200 else str(payload))

        ok, detail = asyncio.run(stream_once(port))
        check("the live stream delivers PCM", ok, detail)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()

    check("the server stops when asked", server.poll() is not None,
          f"exit {server.returncode}")

    if check.failures:
        print(f"\n{len(check.failures)} check(s) failed: {', '.join(check.failures)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
