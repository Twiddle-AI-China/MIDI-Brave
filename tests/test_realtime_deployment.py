import time

import numpy as np
import pytest
import torch

import midibrave.realtime_engine as realtime_engine
from midibrave.realtime_engine import RealtimeEngine
from scripts.smoke_realtime_instrument import run_smoke


class WarmupSession:
    def __init__(self):
        self.starts = []
        self.render_calls = 0

    def start(self, **control):
        self.starts.append(control)

    def render_block(self):
        self.render_calls += 1
        return np.zeros((512, 2), dtype=np.float32), 1.0


class LoadedRuntime:
    def __init__(self):
        rng = np.random.default_rng(12)
        self.seed_clap = torch.from_numpy(
            rng.normal(size=(32, 512)).astype(np.float32),
        )

    def eval(self):
        return self


def test_engine_load_renders_sixteen_startup_warmup_blocks(monkeypatch):
    session = WarmupSession()
    contract = {
        "runtime_sha256": "test-runtime",
        "sample_rate": 44100,
        "history_frames": 16,
        "horizon_frames": 8,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "block_samples": 512,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(realtime_engine, "verify_runtime_contract", lambda *_: contract)
    monkeypatch.setattr(torch.jit, "load", lambda *_, **__: LoadedRuntime())
    monkeypatch.setattr(RealtimeEngine, "new_session", lambda _: session)

    RealtimeEngine.load("runtime.pt")

    assert session.starts == [
        {"x": 0.0, "y": 0.0, "note": 60, "velocity": 0.8, "seed": 0},
    ]
    assert session.render_calls == 16


class SmokeSession:
    def __init__(self, render_delay: float):
        self.render_delay = render_delay
        self.started = None
        self.updates = []

    def start(self, **control):
        self.started = control

    def update(self, **control):
        self.updates.append(control)
        return True

    def render_block(self):
        if self.render_delay:
            time.sleep(self.render_delay)
        return np.zeros((1, 2), dtype=np.float32), self.render_delay * 1000.0


class SmokeEngine:
    def __init__(self, render_delay: float):
        self.session = SmokeSession(render_delay)

    def new_session(self):
        return self.session

    def status(self):
        return {
            "sampleRate": 100,
            "blockSamples": 1,
            "blockDeadlineMs": 10.0,
        }


@pytest.mark.parametrize(
    ("render_delay", "expected_pass"),
    [(0.0, True), (0.02, False)],
)
def test_smoke_reports_sustained_throughput(render_delay, expected_pass):
    report = run_smoke(SmokeEngine(render_delay), 0.01)

    assert report["blocks"] == 1
    assert report["audio_seconds"] == pytest.approx(0.01)
    assert report["wall_elapsed_seconds"] > 0.0
    assert report["throughput_ratio"] == pytest.approx(
        report["audio_seconds"] / report["wall_elapsed_seconds"],
    )
    assert report["render_mean_ms"] == pytest.approx(render_delay * 1000.0)
    assert report["render_p50_ms"] == pytest.approx(render_delay * 1000.0)
    assert report["render_p95_ms"] == pytest.approx(render_delay * 1000.0)
    assert report["render_max_ms"] == pytest.approx(render_delay * 1000.0)
    assert report["finite"] is True
    assert report["realtime_pass"] is expected_pass
