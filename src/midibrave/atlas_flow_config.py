from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


def _section(kind: type[T], value: object, name: str) -> T:
    if not isinstance(value, dict):
        raise ValueError(f"missing config section: {name}")
    allowed = {item.name for item in fields(kind)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(unknown))}")
    try:
        return kind(**value)
    except TypeError as error:
        raise ValueError(f"invalid {name} section: {error}") from error


@dataclass(frozen=True)
class AtlasDataConfig:
    manifest: str
    audio_root: str
    feature_cache: str
    trajectory_cache: str
    atlas_path: str
    sample_rate: int = 44_100
    render_samples: int = 220_500
    note_on_sample: int = 4_410
    note_off_sample: int = 114_660
    feature_hop: int = 512
    trajectory_hop: int = 2_048
    window_samples: int = 49_152
    num_workers: int = 24
    prefetch_factor: int = 4

    def __post_init__(self) -> None:
        for name in ("manifest", "audio_root", "feature_cache", "trajectory_cache", "atlas_path"):
            if not getattr(self, name):
                raise ValueError(f"data.{name} is required")
        exact = {
            "sample_rate": 44_100,
            "render_samples": 220_500,
            "note_on_sample": 4_410,
            "note_off_sample": 114_660,
            "feature_hop": 512,
            "trajectory_hop": 2_048,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"data.{name} must be {expected}")
        if self.window_samples <= 0 or self.window_samples % 128:
            raise ValueError("data.window_samples must be a positive multiple of 128")
        if self.num_workers <= 0 or self.prefetch_factor <= 0:
            raise ValueError("data loader settings must be positive")


@dataclass(frozen=True)
class AtlasModelConfig:
    feature_dim: int = 96
    trajectory_dim: int = 128
    atlas_dim: int = 8
    atlas_neighbors: int = 4
    context_frames: int = 32
    future_frames: int = 64
    d_model: int = 256
    context_layers: int = 2
    future_layers: int = 8
    heads: int = 8
    feedforward_dim: int = 1_024
    dropout: float = 0.05
    decoder_timbre_dim: int = 256
    midi_dim: int = 32
    decoder_capacity: int = 64
    pqmf_bands: int = 16
    pqmf_taps: int = 256
    solver_steps: int = 8

    def __post_init__(self) -> None:
        exact: dict[str, int | float] = {
            "feature_dim": 96,
            "trajectory_dim": 128,
            "atlas_dim": 8,
            "atlas_neighbors": 4,
            "context_frames": 32,
            "future_frames": 64,
            "d_model": 256,
            "context_layers": 2,
            "future_layers": 8,
            "heads": 8,
            "feedforward_dim": 1_024,
            "dropout": 0.05,
            "decoder_timbre_dim": 256,
            "midi_dim": 32,
            "pqmf_bands": 16,
            "solver_steps": 8,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"model.{name} must be {expected}")
        if self.d_model % self.heads:
            raise ValueError("model.d_model must be divisible by model.heads")


@dataclass(frozen=True)
class AtlasLossConfig:
    flow: float = 1.0
    boundary: float = 0.10
    statistics: float = 0.02
    temporal_motion: float = 0.05
    stft: float = 1.0
    multiband: float = 0.5
    envelope: float = 0.1
    rms: float = 0.25
    pitch_adversary: float = 0.05
    same_preset: float = 0.1

    def __post_init__(self) -> None:
        exact = {"flow": 1.0, "boundary": 0.10, "statistics": 0.02, "temporal_motion": 0.05}
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"loss.{name} must be {expected}")
        if any(getattr(self, item.name) < 0 for item in fields(self)):
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class AtlasTrainConfig:
    output_root: str
    batch_per_gpu: int = 4
    flow_batch_per_gpu: int = 72
    stage1_updates: int = 60_000
    flow_updates: int = 100_000
    joint_updates: int = 30_000
    stage1_lr: float = 1.0e-4
    flow_lr: float = 1.0e-4
    joint_lr: float = 2.0e-5
    minimum_lr: float = 1.0e-6
    warmup_updates: int = 1_000
    checkpoint_every: int = 5_000
    validation_every: int = 5_000
    log_every: int = 20
    grad_clip: float = 1.0
    precision: str = "amp_fp16"
    seed: int = 20260822

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("train.output_root is required")
        for name in (
            "batch_per_gpu", "flow_batch_per_gpu", "stage1_updates", "flow_updates",
            "joint_updates", "warmup_updates", "checkpoint_every", "validation_every",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be positive")
        if self.precision != "amp_fp16":
            raise ValueError("V100 training requires train.precision=amp_fp16")
        if self.seed < 0:
            raise ValueError("train.seed must be non-negative")


@dataclass(frozen=True)
class AtlasFlowConfig:
    data: AtlasDataConfig
    model: AtlasModelConfig
    loss: AtlasLossConfig
    train: AtlasTrainConfig


def load_atlas_flow_config(path: str | Path) -> AtlasFlowConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    unknown = set(raw) - {"data", "model", "loss", "train"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {', '.join(sorted(unknown))}")
    return AtlasFlowConfig(
        data=_section(AtlasDataConfig, raw.get("data"), "data"),
        model=_section(AtlasModelConfig, raw.get("model"), "model"),
        loss=_section(AtlasLossConfig, raw.get("loss"), "loss"),
        train=_section(AtlasTrainConfig, raw.get("train"), "train"),
    )


__all__ = [
    "AtlasDataConfig", "AtlasFlowConfig", "AtlasLossConfig", "AtlasModelConfig",
    "AtlasTrainConfig", "load_atlas_flow_config",
]
