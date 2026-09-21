"""Local stand-in for the Atlas Flow demo server, for frontend work off-Spark.

Serves the real static frontend plus a synthetic atlas, a synthetic live PCM
stream (a moving formant so the analyser views have something to show), and
synthetic renders. Speaks the same wire protocol as atlas_flow_runtime_server.
"""
from __future__ import annotations
import asyncio, io, json, math, struct
from pathlib import Path
import numpy as np
from aiohttp import WSMsgType, web

WEB = Path(__file__).resolve().parent.parent / "atlas-flow-web-demo"
RATE = 44100
BLOCK = 4096
EXPLAINED = [0.4231, 0.19825, 0.11953, 0.05508, 0.03083, 0.02703, 0.02255, 0.01569]
PROFILES = [
    {"sampleRate": 44100.0, "channels": 2, "format": "float32", "kbitPerSecond": 2822.4},
    {"sampleRate": 44100.0, "channels": 1, "format": "int16", "kbitPerSecond": 705.6},
    {"sampleRate": 22050.0, "channels": 1, "format": "int16", "kbitPerSecond": 352.8},
    {"sampleRate": 14700.0, "channels": 1, "format": "int16", "kbitPerSecond": 235.2},
    {"sampleRate": 11025.0, "channels": 1, "format": "int16", "kbitPerSecond": 176.4},
]

rng = np.random.default_rng(7)
# six colonies plus eight loners, echoing the real component sizes
CENTRES = [(-0.55, 0.35), (0.25, 0.1), (0.7, -0.35), (-0.2, -0.5), (0.45, 0.6), (-0.75, -0.15)]
SIZES = [21, 11, 5, 2, 2, 1]
points = []
component = 0
for centre, size in zip(CENTRES, SIZES):
    for _ in range(size):
        pca = [float(np.clip(centre[0] + rng.normal(0, 0.16), -1, 1)),
               float(np.clip(centre[1] + rng.normal(0, 0.14), -1, 1))] + list(rng.normal(0, .3, 6))
        points.append({"presetId": f"serum_s{len(points):06d}", "x": pca[0], "y": pca[1],
                       "pcaNormalized": [float(v) for v in pca], "component": component})
    component += 1
while len(points) < 50:
    pca = list(rng.uniform(-0.95, 0.95, 2)) + list(rng.normal(0, .3, 6))
    points.append({"presetId": f"serum_s{len(points):06d}", "x": pca[0], "y": pca[1],
                   "pcaNormalized": [float(v) for v in pca], "component": component})
    component += 1
TESTS = [points[3]["presetId"], points[25]["presetId"], points[47]["presetId"]]


def wav(samples: np.ndarray, rate: int = RATE) -> bytes:
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(data))
    return header + data


def voice(seconds: float, centre: float, rate: int = RATE, drift: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    out = np.zeros_like(t)
    for harmonic in range(1, 14):
        formant = np.exp(-((harmonic * 110 - (centre + drift * t / max(seconds, 1e-9))) / 700) ** 2)
        out += formant / harmonic * np.sin(2 * np.pi * 110 * harmonic * t + harmonic)
    envelope = np.minimum(1, t / 0.15) * np.exp(-t / (seconds * 1.6))
    return 0.35 * out / (np.abs(out).max() + 1e-9) * envelope


async def status(request):
    return web.json_response({
        "ok": True, "cuda": "MOCK GPU", "checkpoint": "mock-step-000030000.pt",
        "sampleRate": RATE, "blockSamples": BLOCK, "defaultPcaNormalized": points[0]["pcaNormalized"],
        "pcaAxes": [{"index": i + 1, "label": f"PC{i + 1}", "minimum": -1, "maximum": 1} for i in range(8)],
        "points": points, "edges": [], "components": SIZES + [1] * 8,
        "streamProfiles": PROFILES, "pcaExplained": EXPLAINED, "presetIds": [p["presetId"] for p in points],
        "maxSteps": 4, "maxTotalSeconds": 24, "releaseSeconds": 2.4, "offlineSolverSteps": 8,
        "limitations": ["Mock server: synthetic audio, synthetic atlas."],
    })


async def evaluation(request):
    rows = [{"preset_id": preset, "note": note,
             "audio": {mode: f"/api/audition/{preset}-n{note:03d}-{mode}.wav"
                       for mode in ("source", "dynamic", "static", "flow")}}
            for preset in TESTS for note in (36, 43, 50, 57, 64, 71)]
    return web.json_response({
        "passed": False, "checkpoint": "mock.pt", "rows": rows,
        "gates": {"midi_following_95pct": False, "dynamic_median_5pct": True,
                  "dynamic_benefit_70pct": True, "silence_le_1pct": True,
                  "tail_drift_le_12db": False, "boundary_ratio_le_4": False,
                  "rms_median_le_6db": False, "rms_p90_le_12db": False, "deterministic": True},
    })


async def audition(request):
    name = request.match_info["name"]
    note = int(name.split("-n")[1][:3])
    centre = 400 + note * 18
    return web.Response(body=wav(voice(3.0, centre)), content_type="audio/wav")


async def render(request):
    body = await request.json()
    steps = body["steps"]
    audio = np.concatenate([voice(step["seconds"], 300 + 900 * (step["pca"][0] + 1) / 2, drift=400)
                            for step in steps] + [voice(2.4, 500) * 0.4])
    request.app["takes"][str(len(request.app["takes"]))] = wav(audio)
    ident = str(len(request.app["takes"]) - 1)
    return web.json_response({
        "id": ident, "url": f"/api/take/{ident}.wav", "seconds": len(audio) / RATE,
        "bytes": len(audio) * 2, "peakDbfs": -12.0, "rmsDbfs": -22.0, "clipped": False,
        "planMs": [80.0] * len(steps), "renderMs": 420.0, "cached": False, "take": body,
        "landings": [{"component": 0, "planMode": "graph_plan"} for _ in steps],
    })


async def take(request):
    ident = request.match_info["name"].split(".")[0]
    return web.Response(body=request.app["takes"][ident], content_type="audio/wav")


async def runtime(request):
    ws = web.WebSocketResponse(heartbeat=15.0, max_msg_size=64 * 1024)
    await ws.prepare(request)
    await ws.send_json({"type": "ready", "cuda": "MOCK GPU", "sampleRate": RATE,
                        "blockSamples": BLOCK, "checkpoint": "mock.pt"})
    state = {"run": False, "rate": RATE, "channels": 1, "format": "int16",
             "decimation": 1, "seq": 0, "pca": [0.0] * 8, "phase": 0.0, "buffered": 0}

    async def producer():
        while not ws.closed:
            if not state["run"] or state["buffered"] > state["rate"] * 1.4:
                await asyncio.sleep(0.02)
                continue
            frames = BLOCK // state["decimation"]
            t = (np.arange(frames) + state["phase"]) / state["rate"]
            state["phase"] += frames
            centre = 300 + 1500 * (state["pca"][0] + 1) / 2
            block = np.zeros(frames)
            for harmonic in range(1, 16):
                gain = math.exp(-((harmonic * 110 - centre) / 500) ** 2)
                block += gain / harmonic * np.sin(2 * math.pi * 110 * harmonic * t)
            block = 0.3 * block / (np.abs(block).max() + 1e-9)
            block *= 0.6 + 0.4 * (state["pca"][1] + 1) / 2     # level varies with PC2
            payload = (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()
            await ws.send_bytes(payload)
            state["buffered"] += frames
            await ws.send_json({"type": "telemetry", "seq": state["seq"], "lifecycle": "held_sustain",
                                "planMode": "graph_plan", "planMs": 71.0, "component": 0,
                                "audiblePcaNormalized": state["pca"],
                                "targetPcaNormalized": state["pca"], "planPending": False})
            await asyncio.sleep(frames / state["rate"] * 0.9)

    task = asyncio.create_task(producer())
    try:
        async for message in ws:
            if message.type is not WSMsgType.TEXT:
                break
            value = json.loads(message.data)
            if value["type"] == "start":
                wire = value.get("stream") or {}
                state.update(rate=int(wire.get("sampleRate", RATE)), channels=wire.get("channels", 2),
                             format=wire.get("format", "float32"), run=True,
                             pca=value.get("pcaNormalized", state["pca"]))
                state["decimation"] = round(RATE / state["rate"])
                await ws.send_json({"type": "stream", "sampleRate": state["rate"],
                                    "channels": state["channels"], "format": state["format"],
                                    "decimation": state["decimation"], "blockFrames": BLOCK / state["decimation"],
                                    "targetSeconds": wire.get("targetSeconds", 1.2),
                                    "kbitPerSecond": state["rate"] * state["channels"] * 2 * 8 / 1000})
            elif value["type"] == "control":
                state["pca"] = value.get("pcaNormalized", state["pca"])
                state["seq"] = value.get("seq", state["seq"])
            elif value["type"] == "buffer":
                state["buffered"] = value.get("bufferedFrames", 0)
            elif value["type"] in ("note_off", "stop"):
                state["run"] = value["type"] == "note_off"
    finally:
        task.cancel()
    return ws


async def asset(request):
    name = request.match_info.get("filename", "index.html")
    target = WEB / name
    if not target.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(target)


app = web.Application()
app["takes"] = {}
app.router.add_get("/", lambda request: asset(request))
app.router.add_get("/api/status", status)
app.router.add_get("/api/evaluation", evaluation)
app.router.add_get("/api/audition/{name}", audition)
app.router.add_get("/api/take/{name}", take)
app.router.add_post("/api/render", render)
app.router.add_get("/live/runtime", runtime)
app.router.add_get("/{filename}", asset)
web.run_app(app, host="127.0.0.1", port=18796, print=None)
