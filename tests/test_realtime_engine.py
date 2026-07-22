import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest
import torch

from midibrave.realtime_engine import (
    RealtimeEngine,
    RuntimeSession,
    verify_runtime_contract,
)
from midibrave.realtime_plane import TimbrePlane


class FakeRuntime:
    def __init__(self):
        self.initial_calls = 0
        self.step_calls = 0
        self.initial_lock_states: list[bool] = []
        self.step_lock_states: list[bool] = []
        self.histories: list[torch.Tensor] = []
        self.trajectories: list[torch.Tensor] = []
        self.lock_probe = lambda: False

    def initial_state(self, clap, random_seed, top_k=8):
        self.initial_calls += 1
        self.initial_lock_states.append(self.lock_probe())
        return torch.zeros(1, 16, 16, device=clap.device)

    def step(self, history, clap, note, velocity):
        self.step_calls += 1
        self.step_lock_states.append(self.lock_probe())
        self.histories.append(history.detach().clone())
        self.trajectories.append(clap.detach().clone())
        return torch.full((1, 1, 512), 0.05, device=history.device), history + 1


class ConcurrentFakeRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.activity_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _begin_call(self):
        with self.activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _end_call(self):
        with self.activity_lock:
            self.active -= 1

    def initial_state(self, clap, random_seed, top_k=8):
        self.initial_calls += 1
        self._begin_call()
        try:
            time.sleep(0.05)
            return torch.zeros(1, 16, 16, device=clap.device)
        finally:
            self._end_call()

    def step(self, history, clap, note, velocity):
        self.step_calls += 1
        self._begin_call()
        try:
            time.sleep(0.05)
            return (
                torch.full((1, 1, 512), 0.05, device=history.device),
                history + 1,
            )
        finally:
            self._end_call()


class BlockingInitialRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.block_next = False
        self.initial_entered = threading.Event()
        self.initial_release = threading.Event()

    def block_next_initial_state(self):
        self.block_next = True
        self.initial_entered.clear()
        self.initial_release.clear()

    def initial_state(self, clap, random_seed, top_k=8):
        self.initial_calls += 1
        if self.block_next:
            self.block_next = False
            self.initial_entered.set()
            if not self.initial_release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release initial_state")
        return torch.zeros(1, 16, 16, device=clap.device)


class NonFiniteRuntime(FakeRuntime):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def initial_state(self, clap, random_seed, top_k=8):
        history = super().initial_state(clap, random_seed, top_k)
        if self.failure == "initial":
            history[0, 0, 0] = torch.nan
        return history

    def step(self, history, clap, note, velocity):
        audio, new_history = super().step(history, clap, note, velocity)
        if self.failure == "audio":
            audio[0, 0, 0] = torch.nan
        if self.failure == "history":
            new_history[0, 0, 0] = torch.inf
        return audio, new_history


class DeferredAudio:
    def __init__(self, value, entered, release):
        self.value = value
        self.entered = entered
        self.release = release

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to convert audio")
        return self.value.numpy()


class TransactionalFakeRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.first_step_entered = threading.Event()
        self.first_step_release = threading.Event()
        self.audio_conversion_entered = threading.Event()
        self.audio_conversion_release = threading.Event()
        self.second_step_entered = threading.Event()
        self.reseed_initial_entered = threading.Event()

    def initial_state(self, clap, random_seed, top_k=8):
        self.initial_calls += 1
        if self.initial_calls > 1:
            self.reseed_initial_entered.set()
        return torch.zeros(1, 16, 16, device=clap.device)

    def step(self, history, clap, note, velocity):
        self.step_calls += 1
        self.histories.append(history.detach().clone())
        audio = torch.full((1, 1, 512), 0.05, device=history.device)
        new_history = history + 1
        if self.step_calls == 1:
            self.first_step_entered.set()
            if not self.first_step_release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to release first step")
            audio = DeferredAudio(
                audio,
                self.audio_conversion_entered,
                self.audio_conversion_release,
            )
        else:
            self.second_step_entered.set()
        return audio, new_history


class MinimalScriptRuntime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seed_clap", torch.arange(3 * 512).reshape(3, 512))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.seed_clap[0].to(value)


def plane():
    rng = np.random.default_rng(7)
    return TimbrePlane.fit(rng.normal(size=(32, 512)).astype(np.float32))


def contract():
    return {
        "runtime_sha256": "test-runtime",
        "sample_rate": 44100,
        "history_frames": 16,
        "horizon_frames": 8,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "block_samples": 512,
    }


def test_contract_verifies_runtime_hash_and_geometry(tmp_path):
    runtime = tmp_path / "runtime.pt"
    with pytest.raises(ValueError, match="runtime is missing"):
        verify_runtime_contract(runtime)

    runtime.write_bytes(b"runtime")
    with pytest.raises(ValueError, match="sidecar is missing"):
        verify_runtime_contract(runtime)

    sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
    sidecar = runtime.with_suffix(".pt.json")
    contract_data = {
        "encoder_free": True,
        "runtime_sha256": sha,
        "sample_rate": 44100,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "history_frames": 16,
        "horizon_frames": 8,
    }
    sidecar.write_text(json.dumps(contract_data), encoding="utf-8")
    contract = verify_runtime_contract(runtime)
    assert contract["block_samples"] == 512

    with pytest.raises(ValueError, match="approved deployment runtime"):
        verify_runtime_contract(runtime, "0" * 64)

    invalid_geometry = dict(contract_data, stride_frames=5)
    sidecar.write_text(json.dumps(invalid_geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="stride_frames"):
        verify_runtime_contract(runtime)

    sidecar.write_text(json.dumps(contract_data), encoding="utf-8")
    runtime.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_runtime_contract(runtime)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sample_rate", 44100.0),
        ("sample_rate", 44100.5),
        ("stride_frames", 4.0),
        ("encoder_free", 1),
    ],
)
def test_contract_rejects_non_integer_geometry_metadata(tmp_path, key, value):
    runtime = tmp_path / "runtime.pt"
    runtime.write_bytes(b"runtime")
    metadata = {
        "encoder_free": True,
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "sample_rate": 44100,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "history_frames": 16,
        "horizon_frames": 8,
    }
    metadata[key] = value
    runtime.with_suffix(".pt.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime"):
        verify_runtime_contract(runtime)


def test_xy_and_note_updates_preserve_history_until_explicit_reseed():
    runtime = FakeRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), {
        "history_frames": 16,
        "horizon_frames": 8,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "block_samples": 512,
    })
    runtime.lock_probe = session.lock.locked

    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)
    first, _ = session.render_block()
    assert session.update(seq=1, x=0.8, y=-0.5, note=72, velocity=0.6)
    assert not session.update(seq=1, x=-0.8, y=0.5, note=48, velocity=0.2)
    second, _ = session.render_block()

    assert first.shape == second.shape == (512, 2)
    assert first.dtype == second.dtype == np.dtype("<f4")
    assert runtime.initial_calls == 1
    assert runtime.step_calls == 2
    assert [tuple(value.shape) for value in runtime.trajectories] == [
        (1, 512, 8),
        (1, 512, 8),
    ]
    torch.testing.assert_close(runtime.histories[0], torch.zeros(1, 16, 16))
    torch.testing.assert_close(runtime.histories[1], torch.ones(1, 16, 16))
    torch.testing.assert_close(
        session.current_clap,
        runtime.trajectories[1][0, :, 3],
    )
    assert runtime.initial_lock_states == [False]
    assert runtime.step_lock_states == [False, False]

    session.reseed(9)
    assert runtime.initial_calls == 2
    assert runtime.initial_lock_states == [False, False]


@pytest.mark.parametrize(
    "override",
    [
        {"seq": np.inf},
        {"x": np.nan},
        {"y": np.inf},
        {"note": -np.inf},
        {"velocity": np.nan},
    ],
)
def test_control_rejects_non_finite_values(override):
    runtime = FakeRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), contract())
    values = {
        "seq": 1,
        "x": 0.0,
        "y": 0.0,
        "note": 60,
        "velocity": 0.8,
    }
    values.update(override)

    with pytest.raises(ValueError, match="finite"):
        session.update(**values)


@pytest.mark.filterwarnings(
    "ignore:`torch.jit.*` is deprecated:DeprecationWarning",
)
def test_torchscript_loaded_runtime_exposes_registered_seed_clap(tmp_path):
    expected = MinimalScriptRuntime().seed_clap.detach().clone()
    scripted = torch.jit.script(MinimalScriptRuntime())
    runtime_path = tmp_path / "minimal-runtime.pt"
    torch.jit.save(scripted, str(runtime_path))

    loaded = torch.jit.load(str(runtime_path), map_location="cpu")

    torch.testing.assert_close(loaded.seed_clap, expected)


def test_session_rejects_non_finite_initial_history():
    session = RuntimeSession(
        NonFiniteRuntime("initial"),
        plane(),
        torch.device("cpu"),
        contract(),
    )

    with pytest.raises(RuntimeError, match="initial history is non-finite"):
        session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)


@pytest.mark.parametrize("failure", ["audio", "history"])
def test_session_rejects_non_finite_step_outputs(failure):
    session = RuntimeSession(
        NonFiniteRuntime(failure),
        plane(),
        torch.device("cpu"),
        contract(),
    )
    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)

    with pytest.raises(RuntimeError, match="non-finite audio or history"):
        session.render_block()


def test_engine_serializes_runtime_calls_across_sessions():
    runtime = ConcurrentFakeRuntime()
    engine = RealtimeEngine(runtime, plane(), torch.device("cpu"), contract())
    sessions = [engine.new_session(), engine.new_session()]

    start_barrier = threading.Barrier(2)

    def start(index):
        start_barrier.wait()
        sessions[index].start(
            x=0.0,
            y=0.0,
            note=60 + index,
            velocity=0.8,
            seed=index,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(start, range(2)))
    assert runtime.max_active == 1

    runtime.max_active = 0
    render_barrier = threading.Barrier(2)

    def render(index):
        render_barrier.wait()
        return sessions[index].render_block()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(render, range(2)))
    assert runtime.max_active == 1
    assert [audio.shape for audio, _ in results] == [(512, 2), (512, 2)]


def test_start_keeps_control_updated_while_initial_state_is_running():
    runtime = BlockingInitialRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), contract())
    runtime.block_next_initial_state()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            session.start,
            x=0.25,
            y=-0.5,
            note=70,
            velocity=0.7,
            seed=3,
        )
        assert runtime.initial_entered.wait(timeout=2.0)
        try:
            assert session.snapshot().note == 70
            assert session.update(
                seq=5,
                x=0.8,
                y=0.4,
                note=76,
                velocity=0.6,
            )
        finally:
            runtime.initial_release.set()
        future.result(timeout=2.0)

    control = session.snapshot()
    assert (control.seq, control.x, control.y) == (5, 0.8, 0.4)
    assert (control.note, control.velocity) == (76, 0.6)


def test_reseed_preserves_latest_control_and_sequence():
    runtime = BlockingInitialRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), contract())
    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)
    assert session.update(seq=7, x=0.3, y=-0.4, note=72, velocity=0.5)
    runtime.block_next_initial_state()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(session.reseed, 9)
        assert runtime.initial_entered.wait(timeout=2.0)
        try:
            assert session.update(
                seq=8,
                x=-0.6,
                y=0.7,
                note=74,
                velocity=0.9,
            )
        finally:
            runtime.initial_release.set()
        future.result(timeout=2.0)

    control = session.snapshot()
    assert (control.seq, control.x, control.y) == (8, -0.6, 0.7)
    assert (control.note, control.velocity) == (74, 0.9)
    assert not session.update(seq=7, x=0.0, y=0.0, note=60, velocity=0.8)
    assert not session.update(seq=8, x=0.0, y=0.0, note=60, velocity=0.8)
    assert session.update(seq=9, x=0.1, y=0.2, note=75, velocity=0.4)
    assert session.snapshot().seq == 9


def test_same_session_render_blocks_commit_history_transactionally():
    runtime = TransactionalFakeRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), contract())
    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)
    second_started = threading.Event()

    def second_render():
        second_started.set()
        return session.render_block()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(session.render_block)
        assert runtime.first_step_entered.wait(timeout=2.0)
        second = executor.submit(second_render)
        assert second_started.wait(timeout=2.0)
        try:
            runtime.first_step_release.set()
            assert runtime.audio_conversion_entered.wait(timeout=2.0)
            runtime.second_step_entered.wait(timeout=0.25)
        finally:
            runtime.first_step_release.set()
            runtime.audio_conversion_release.set()
        first.result(timeout=2.0)
        second.result(timeout=2.0)

    torch.testing.assert_close(runtime.histories[0], torch.zeros(1, 16, 16))
    torch.testing.assert_close(runtime.histories[1], torch.ones(1, 16, 16))
    torch.testing.assert_close(session.history, torch.full((1, 16, 16), 2.0))


def test_render_then_reseed_commits_reseed_history_last():
    runtime = TransactionalFakeRuntime()
    session = RuntimeSession(runtime, plane(), torch.device("cpu"), contract())
    session.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=3)

    with ThreadPoolExecutor(max_workers=2) as executor:
        render = executor.submit(session.render_block)
        assert runtime.first_step_entered.wait(timeout=2.0)
        reseed = executor.submit(session.reseed, 9)
        try:
            runtime.first_step_release.set()
            assert runtime.audio_conversion_entered.wait(timeout=2.0)
            reseed_entered_early = runtime.reseed_initial_entered.wait(timeout=0.25)
            if reseed_entered_early:
                reseed.result(timeout=2.0)
        finally:
            runtime.first_step_release.set()
            runtime.audio_conversion_release.set()
        render.result(timeout=2.0)
        reseed.result(timeout=2.0)

    torch.testing.assert_close(session.history, torch.zeros(1, 16, 16))
