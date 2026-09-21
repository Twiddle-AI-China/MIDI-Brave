"""Negotiable wire format for the live Atlas Flow PCM stream.

The engine renders 44.1 kHz stereo float32 blocks, which is ~2.8 Mbit/s on the
wire. That is fine on the office LAN and impossible through the ~1.3 Mbit/s
natapp tunnel, so a client may ask for a cheaper stream: drop the duplicated
channel, decimate by an integer factor, and/or send 16-bit samples.

Decimation keeps its anti-alias filter state across blocks, so a stream stays
click-free no matter how the engine block size divides the factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Mapping

import numpy as np
from scipy.signal import firwin, lfilter


SOURCE_RATE = 44_100
DECIMATIONS = (1, 2, 3, 4)
FORMATS = ("float32", "int16")
MIN_TARGET_SECONDS = 0.1
MAX_TARGET_SECONDS = 5.0
DEFAULT_TARGET_BLOCKS = 2.0


def supported_profiles(block_samples: int) -> list[dict[str, object]]:
    """Wire profiles offered to clients, cheapest last."""
    profiles = []
    for decimation in DECIMATIONS:
        for channels, dtype in ((2, "float32"), (1, "int16")):
            if decimation > 1 and dtype == "float32":
                continue
            rate = SOURCE_RATE / decimation
            width = 4 if dtype == "float32" else 2
            profiles.append({
                "sampleRate": rate,
                "channels": channels,
                "format": dtype,
                "kbitPerSecond": round(rate * channels * width * 8 / 1000.0, 1),
                "defaultTargetSeconds": round(
                    DEFAULT_TARGET_BLOCKS * block_samples / SOURCE_RATE, 3,
                ) if decimation == 1 and dtype == "float32" else 1.5,
            })
    return profiles


@dataclass
class StreamFormat:
    """One negotiated wire format plus its stateful decimation filter."""

    decimation: int = 1
    channels: int = 2
    dtype: str = "float32"
    target_seconds: float = 0.0
    _taps: np.ndarray | None = field(default=None, init=False, repr=False)
    _state: np.ndarray | None = field(default=None, init=False, repr=False)
    _offset: int = field(default=0, init=False, repr=False)

    @property
    def sample_rate(self) -> float:
        return SOURCE_RATE / self.decimation

    @property
    def sample_bytes(self) -> int:
        return (4 if self.dtype == "float32" else 2) * self.channels

    def describe(self, block_samples: int) -> dict[str, object]:
        return {
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "format": self.dtype,
            "decimation": self.decimation,
            "targetSeconds": self.target_seconds,
            "blockFrames": block_samples / self.decimation,
            "kbitPerSecond": round(
                self.sample_rate * self.sample_bytes * 8 / 1000.0, 1,
            ),
        }

    def target_frames(self, block_samples: int) -> int:
        """Client-side buffer ceiling, counted in this format's own frames."""
        if self.target_seconds <= 0.0:
            return int(DEFAULT_TARGET_BLOCKS * block_samples / self.decimation)
        return max(1, int(self.target_seconds * self.sample_rate))

    def _decimate(self, block: np.ndarray) -> np.ndarray:
        if self._taps is None:
            # 0.9 of the new Nyquist keeps the transition band inaudible while
            # staying short enough that per-block filtering is cheap.
            self._taps = firwin(16 * self.decimation + 1, 0.9 / self.decimation)
            self._state = np.zeros((self._taps.size - 1, block.shape[1]), dtype=np.float64)
        filtered, self._state = lfilter(
            self._taps, (1.0,), block.astype(np.float64), axis=0, zi=self._state,
        )
        start = (-self._offset) % self.decimation
        self._offset = (self._offset + block.shape[0]) % self.decimation
        return filtered[start::self.decimation]

    def encode(self, block: np.ndarray) -> bytes:
        """Encode one [frames, 2] float32 engine block into wire bytes."""
        data = block if self.channels == 2 else block[:, :1]
        if self.decimation > 1:
            data = self._decimate(data)
        if self.dtype == "int16":
            payload = np.clip(data, -1.0, 1.0) * 32_767.0
            return np.ascontiguousarray(payload.astype("<i2")).tobytes()
        return np.ascontiguousarray(data.astype("<f4")).tobytes()


def parse_stream(value: object) -> StreamFormat:
    """Validate a client stream request; absent or empty means the legacy wire."""
    if value is None:
        return StreamFormat()
    if not isinstance(value, Mapping):
        raise ValueError("stream must be a JSON object")
    rate = value.get("sampleRate", SOURCE_RATE)
    if isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(float(rate)):
        raise ValueError("stream.sampleRate must be a finite number")
    decimation = SOURCE_RATE / float(rate)
    if not any(abs(decimation - candidate) < 1.0e-6 for candidate in DECIMATIONS):
        raise ValueError("stream.sampleRate must be 44100 divided by 1, 2, 3 or 4")
    dtype = value.get("format", "float32")
    if dtype not in FORMATS:
        raise ValueError("stream.format must be float32 or int16")
    channels = value.get("channels", 2)
    if channels not in (1, 2):
        raise ValueError("stream.channels must be 1 or 2")
    target = value.get("targetSeconds", 0.0)
    if isinstance(target, bool) or not isinstance(target, Real) or not math.isfinite(float(target)):
        raise ValueError("stream.targetSeconds must be a finite number")
    target = float(target)
    if target and not MIN_TARGET_SECONDS <= target <= MAX_TARGET_SECONDS:
        raise ValueError(
            f"stream.targetSeconds must be between {MIN_TARGET_SECONDS} and {MAX_TARGET_SECONDS}"
        )
    return StreamFormat(
        decimation=int(round(decimation)),
        channels=int(channels),
        dtype=str(dtype),
        target_seconds=target,
    )


__all__ = ["SOURCE_RATE", "StreamFormat", "parse_stream", "supported_profiles"]
