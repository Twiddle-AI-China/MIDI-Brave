from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .realtime_plane import TimbrePlane


EXPECTED_RUNTIME_SHA256 = (
    "c9b93dcec1a471ccbf97fab9074717f8cec1f2598dfa5d86e808d05a153f78ec"
)


@dataclass(frozen=True)
class Control:
    seq: int
    x: float
    y: float
    note: int
    velocity: float


def verify_runtime_contract(
        runtime_path: str | Path, expected_sha256: str | None = None,
) -> dict[str, object]:
    path = Path(runtime_path)
    if not path.is_file():
        raise ValueError(f"runtime is missing: {path}")
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not sidecar_path.is_file():
        raise ValueError(f"runtime sidecar is missing: {sidecar_path}")
    contract = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if contract.get("runtime_sha256") != actual:
        raise ValueError("runtime SHA-256 does not match its sidecar")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("runtime SHA-256 is not the approved deployment runtime")
    required = {
        "sample_rate": 44100,
        "stride_frames": 4,
        "samples_per_latent": 128,
        "history_frames": 16,
        "horizon_frames": 8,
    }
    if contract.get("encoder_free") is not True:
        raise ValueError("runtime must be encoder-free")
    for key, expected in required.items():
        if int(contract.get(key, -1)) != expected:
            raise ValueError(f"runtime {key} must equal {expected}")
    contract["block_samples"] = (
        int(contract["stride_frames"]) * int(contract["samples_per_latent"])
    )
    return contract


class RuntimeSession:
    def __init__(self, runtime, plane: TimbrePlane, device: torch.device,
                 contract: dict[str, object]):
        self.runtime = runtime
        self.plane = plane
        self.device = device
        self.contract = contract
        self.lock = threading.Lock()
        self.control = Control(0, 0.0, 0.0, 60, 0.8)
        self.history: torch.Tensor | None = None
        self.current_clap: torch.Tensor | None = None

    @staticmethod
    def _control(seq: int, x: float, y: float, note: int,
                 velocity: float) -> Control:
        return Control(
            int(seq),
            float(np.clip(x, -1.0, 1.0)),
            float(np.clip(y, -1.0, 1.0)),
            int(np.clip(note, 21, 109)),
            float(np.clip(velocity, 0.0, 1.0)),
        )

    def _anchor(self) -> torch.Tensor:
        value = self.plane.map_xy(0.0, 0.0)
        return torch.from_numpy(value).to(self.device)

    def start(self, *, x: float, y: float, note: int,
              velocity: float, seed: int) -> None:
        control = self._control(0, x, y, note, velocity)
        anchor = self._anchor()
        history = self.runtime.initial_state(anchor.unsqueeze(0), int(seed), 8)
        if not bool(torch.isfinite(history).all()):
            raise RuntimeError("runtime initial history is non-finite")
        with self.lock:
            self.control = control
            self.current_clap = anchor
            self.history = history

    def update(self, *, seq: int, x: float, y: float, note: int,
               velocity: float) -> bool:
        candidate = self._control(seq, x, y, note, velocity)
        with self.lock:
            if candidate.seq <= self.control.seq:
                return False
            self.control = candidate
            return True

    def reseed(self, seed: int) -> None:
        with self.lock:
            control = self.control
        self.start(
            x=control.x,
            y=control.y,
            note=control.note,
            velocity=control.velocity,
            seed=seed,
        )

    def snapshot(self) -> Control:
        with self.lock:
            return self.control

    def render_block(self) -> tuple[np.ndarray, float]:
        with self.lock:
            control = self.control
            history = self.history
            current = self.current_clap
        if history is None or current is None:
            raise RuntimeError("runtime session has not started")
        target = torch.from_numpy(self.plane.map_xy(control.x, control.y)).to(
            self.device)
        horizon = int(self.contract["horizon_frames"])
        stride = int(self.contract["stride_frames"])
        alpha = torch.linspace(
            1.0 / horizon,
            1.0,
            horizon,
            device=self.device,
        )
        trajectory = current[:, None] * (1.0 - alpha) + target[:, None] * alpha
        trajectory = F.normalize(trajectory, dim=0).unsqueeze(0)
        note = torch.tensor([control.note], device=self.device, dtype=torch.long)
        velocity = torch.tensor(
            [control.velocity],
            device=self.device,
            dtype=torch.float32,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            audio, new_history = self.runtime.step(
                history,
                trajectory,
                note,
                velocity,
            )
            mono = audio.float().cpu().numpy()
        render_ms = (time.perf_counter() - started) * 1000.0
        expected = (1, 1, int(self.contract["block_samples"]))
        if tuple(mono.shape) != expected:
            raise RuntimeError(f"runtime emitted {mono.shape}, expected {expected}")
        if not np.isfinite(mono).all() or not bool(
                torch.isfinite(new_history).all()):
            raise RuntimeError("runtime emitted non-finite audio or history")
        with self.lock:
            self.history = new_history
            self.current_clap = trajectory[0, :, stride - 1]
        stereo = np.repeat(mono[0, 0, :, None], 2, axis=1)
        return stereo.astype("<f4", copy=False), render_ms


class RealtimeEngine:
    def __init__(self, runtime, plane: TimbrePlane, device: torch.device,
                 contract: dict[str, object]):
        self.runtime = runtime
        self.plane = plane
        self.device = device
        self.contract = contract

    @classmethod
    def load(cls, runtime_path: str | Path,
             device: str = "cuda") -> "RealtimeEngine":
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the realtime instrument requires CUDA")
        contract = verify_runtime_contract(runtime_path, EXPECTED_RUNTIME_SHA256)
        runtime = torch.jit.load(str(runtime_path), map_location=target).eval()
        seed_clap = runtime.seed_clap.detach().float().cpu().numpy()
        engine = cls(runtime, TimbrePlane.fit(seed_clap), target, contract)
        warmup = engine.new_session()
        warmup.start(x=0.0, y=0.0, note=60, velocity=0.8, seed=0)
        warmup.render_block()
        return engine

    def new_session(self) -> RuntimeSession:
        return RuntimeSession(
            self.runtime,
            self.plane,
            self.device,
            self.contract,
        )

    def status(self) -> dict[str, object]:
        return {
            "engine": "midibrave-encoder-free",
            "cuda": torch.cuda.get_device_name(self.device),
            "runtimeSha256": self.contract["runtime_sha256"],
            "sampleRate": self.contract["sample_rate"],
            "blockSamples": self.contract["block_samples"],
            "blockDeadlineMs": (
                float(self.contract["block_samples"])
                / float(self.contract["sample_rate"]) * 1000.0
            ),
            "plane": self.plane.metadata(),
        }
