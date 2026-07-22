import hashlib
import json

import numpy as np
import pytest
import torch

from midibrave.realtime_engine import RuntimeSession, verify_runtime_contract
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


def plane():
    rng = np.random.default_rng(7)
    return TimbrePlane.fit(rng.normal(size=(32, 512)).astype(np.float32))


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
