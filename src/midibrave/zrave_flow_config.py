from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isclose
from pathlib import Path
from typing import Any, TypeVar

import yaml


_SECTION = TypeVar("_SECTION")
_SOURCE_NAMES = (
    "serum_full",
    "pianobook_pitch",
    "dexed_surge_broad",
)
_SOURCE_WEIGHTS = (0.65, 0.10, 0.25)
_SOURCE_FUTURES = (64, 64, 64)
_WANDER_DELAYS = (16, 32, 48)
_CODEC_SHA256 = (
    "3ec093e132ce75d7fee3b8b734c739ebf"
    "8711a57ee332a60bce4359e2e34073e"
)


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _mapping(raw: object, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"missing config section: {name}")
    return dict(raw)


def _strict_section(
    section_type: type[_SECTION],
    raw: object,
    name: str,
) -> _SECTION:
    values = _mapping(raw, name)
    allowed = {field.name for field in fields(section_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown {name} config keys: {', '.join(sorted(unknown))}"
        )
    try:
        return section_type(**values)
    except TypeError as error:
        raise ValueError(f"invalid {name} config: {error}") from error


@dataclass(frozen=True)
class FlowSourceConfig:
    name: str
    kind: str
    manifest: str
    audio_root: str
    weight: float
    maximum_future_frames: int
    registry_dataset_id: str | None = None
    preset_metadata: str | None = None
    allowed_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_categories",
            tuple(self.allowed_categories),
        )
        if not self.name:
            raise ValueError("source name must not be empty")
        if self.kind not in {"registry", "jsonl"}:
            raise ValueError("source kind must be registry or jsonl")
        if not self.manifest or not self.audio_root:
            raise ValueError("source manifest and audio_root are required")
        _positive(f"source {self.name} weight", self.weight)
        _positive(
            f"source {self.name} maximum_future_frames",
            self.maximum_future_frames,
        )
        if self.maximum_future_frames > 64:
            raise ValueError("source maximum_future_frames cannot exceed 64")
        if self.kind == "registry" and not self.registry_dataset_id:
            raise ValueError("registry source requires registry_dataset_id")


@dataclass(frozen=True)
class FlowDataConfig:
    unified_manifest: str
    manifest_report: str
    cache_root: str
    packed_root: str
    shard_records: int
    maximum_audio_seconds: float
    sources: tuple[FlowSourceConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        for name in (
            "unified_manifest",
            "manifest_report",
            "cache_root",
            "packed_root",
        ):
            if not getattr(self, name):
                raise ValueError(f"data.{name} must not be empty")
        _positive("data.shard_records", self.shard_records)
        _positive(
            "data.maximum_audio_seconds",
            self.maximum_audio_seconds,
        )
        if self.maximum_audio_seconds != 5.0:
            raise ValueError("data.maximum_audio_seconds must be 5.0")
        total = sum(source.weight for source in self.sources)
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("source weights must sum to 1")
        names = tuple(source.name for source in self.sources)
        if names != _SOURCE_NAMES:
            raise ValueError(
                "data source names must be "
                + ", ".join(_SOURCE_NAMES)
                + " in that order"
            )
        weights = tuple(source.weight for source in self.sources)
        if weights != _SOURCE_WEIGHTS:
            raise ValueError(
                "data source weights must be 0.65, 0.10, 0.25"
            )
        futures = tuple(
            source.maximum_future_frames for source in self.sources
        )
        if futures != _SOURCE_FUTURES:
            raise ValueError(
                "data source maximum futures must be 64, 64, 64"
            )


@dataclass(frozen=True)
class FlowRaveConfig:
    checkpoint: str
    expected_sha256: str
    sample_rate: int = 44100
    latent_hop: int = 2048

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("rave.checkpoint is required")
        digest = self.expected_sha256.casefold()
        object.__setattr__(self, "expected_sha256", digest)
        if digest != _CODEC_SHA256:
            raise ValueError("rave.expected_sha256 must match the codec")
        if self.sample_rate != 44100:
            raise ValueError("rave.sample_rate must be 44100")
        if self.latent_hop != 2048:
            raise ValueError("rave.latent_hop must be 2048")


@dataclass(frozen=True)
class FlowModelConfig:
    latent_dim: int = 16
    context_frames: int = 32
    future_frames: int = 64
    d_model: int = 384
    context_layers: int = 4
    future_layers: int = 8
    heads: int = 8
    feedforward_dim: int = 1536
    dropout: float = 0.0
    note_min: int = 21
    note_max: int = 109
    condition_dropout: float = 0.10
    pitch_guidance: float = 3.0
    solver_steps: int = 8
    wander_delay_frames: tuple[int, ...] = _WANDER_DELAYS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wander_delay_frames",
            tuple(self.wander_delay_frames),
        )
        exact = {
            "latent_dim": 16,
            "context_frames": 32,
            "future_frames": 64,
            "d_model": 384,
            "context_layers": 4,
            "future_layers": 8,
            "heads": 8,
            "feedforward_dim": 1536,
            "note_min": 21,
            "note_max": 109,
            "solver_steps": 8,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"model.{name} must be {expected}")
        if self.dropout != 0.0:
            raise ValueError("model.dropout must be 0.0")
        if self.condition_dropout != 0.10:
            raise ValueError("model.condition_dropout must be 0.10")
        if not 1.0 <= self.pitch_guidance <= 5.0:
            raise ValueError("model.pitch_guidance must be in [1, 5]")
        if self.wander_delay_frames != _WANDER_DELAYS:
            raise ValueError("model.wander_delay_frames must be 16, 32, 48")


@dataclass(frozen=True)
class FlowLossConfig:
    flow: float = 1.0
    pitch_initial: float = 0.30
    pitch_minimum: float = 0.10
    pitch_maximum: float = 1.00
    pitch_warmup_updates: int = 1000
    pitch_gradient_minimum_ratio: float = 0.20
    pitch_gradient_maximum_ratio: float = 0.35
    pitch_gradient_measure_every: int = 100
    boundary: float = 0.10
    statistics: float = 0.02
    boundary_frames: int = 8

    def __post_init__(self) -> None:
        exact = {
            "flow": 1.0,
            "pitch_initial": 0.30,
            "pitch_minimum": 0.10,
            "pitch_maximum": 1.00,
            "pitch_warmup_updates": 1000,
            "pitch_gradient_minimum_ratio": 0.20,
            "pitch_gradient_maximum_ratio": 0.35,
            "pitch_gradient_measure_every": 100,
            "boundary": 0.10,
            "statistics": 0.02,
            "boundary_frames": 8,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"loss.{name} must be {expected}")


@dataclass(frozen=True)
class FlowOptimizerConfig:
    learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    def __post_init__(self) -> None:
        _positive("optimizer.learning_rate", self.learning_rate)
        _positive(
            "optimizer.minimum_learning_rate",
            self.minimum_learning_rate,
        )
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                "optimizer.minimum_learning_rate cannot exceed learning_rate"
            )
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")


@dataclass(frozen=True)
class FlowTrainConfig:
    output_root: str
    batch_per_gpu: int = 128
    max_updates: int = 100000
    warmup_updates: int = 1000
    short_future_updates: int = 5000
    checkpoint_every: int = 5000
    validation_every: int = 5000
    early_stop_gate_passes: int = 3
    log_every: int = 20
    gradient_clip: float = 1.0
    precision: str = "amp_fp16"
    exposure_start_fraction: float = 0.80
    exposure_probability: float = 0.25
    exposure_prefix_frames: int = 32
    pitch_transition_fraction: float = 0.20
    seed: int = 20260725

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("train.output_root must not be empty")
        for name in (
            "batch_per_gpu",
            "max_updates",
            "warmup_updates",
            "short_future_updates",
            "checkpoint_every",
            "validation_every",
            "early_stop_gate_passes",
            "log_every",
            "exposure_prefix_frames",
        ):
            _positive(f"train.{name}", getattr(self, name))
        _positive("train.gradient_clip", self.gradient_clip)
        if self.checkpoint_every != 5000:
            raise ValueError("train.checkpoint_every must be 5000")
        if self.validation_every != 5000:
            raise ValueError("train.validation_every must be 5000")
        if self.short_future_updates != 5000:
            raise ValueError("train.short_future_updates must be 5000")
        if self.precision != "amp_fp16":
            raise ValueError("train.precision must be amp_fp16")
        exact_floats = {
            "exposure_start_fraction": 0.80,
            "exposure_probability": 0.25,
            "pitch_transition_fraction": 0.20,
        }
        for name, expected in exact_floats.items():
            if getattr(self, name) != expected:
                raise ValueError(f"train.{name} must be {expected}")
        if self.exposure_prefix_frames != 32:
            raise ValueError("train.exposure_prefix_frames must be 32")
        if self.seed < 0:
            raise ValueError("train.seed must be non-negative")


@dataclass(frozen=True)
class ZraveFlowConfig:
    seed: int
    data: FlowDataConfig
    rave: FlowRaveConfig
    model: FlowModelConfig
    loss: FlowLossConfig
    optimizer: FlowOptimizerConfig
    train: FlowTrainConfig
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.seed != self.train.seed:
            raise ValueError("seed and train.seed must match")

    @classmethod
    def load(cls, path: str | Path) -> "ZraveFlowConfig":
        source_path = Path(path).resolve()
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        root = _mapping(raw, "root")
        allowed = {
            "seed",
            "data",
            "rave",
            "model",
            "loss",
            "optimizer",
            "train",
        }
        unknown = set(root) - allowed
        if unknown:
            raise ValueError(
                "unknown top-level config keys: "
                + ", ".join(sorted(unknown))
            )

        data_raw = _mapping(root.get("data"), "data")
        data_allowed = {field.name for field in fields(FlowDataConfig)}
        data_unknown = set(data_raw) - data_allowed
        if data_unknown:
            raise ValueError(
                "unknown data config keys: "
                + ", ".join(sorted(data_unknown))
            )
        source_rows = data_raw.get("sources")
        if not isinstance(source_rows, list):
            raise ValueError("data.sources must be a list")
        sources = tuple(
            _strict_section(FlowSourceConfig, row, f"source[{index}]")
            for index, row in enumerate(source_rows)
        )
        data_values = dict(data_raw)
        data_values["sources"] = sources

        model_raw = _mapping(root.get("model"), "model")
        if "wander_delay_frames" in model_raw:
            model_raw["wander_delay_frames"] = tuple(
                model_raw["wander_delay_frames"]
            )
        try:
            seed = int(root["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid config key: seed") from error
        return cls(
            seed=seed,
            data=FlowDataConfig(**data_values),
            rave=_strict_section(FlowRaveConfig, root.get("rave"), "rave"),
            model=_strict_section(FlowModelConfig, model_raw, "model"),
            loss=_strict_section(FlowLossConfig, root.get("loss"), "loss"),
            optimizer=_strict_section(
                FlowOptimizerConfig,
                root.get("optimizer"),
                "optimizer",
            ),
            train=_strict_section(
                FlowTrainConfig,
                root.get("train"),
                "train",
            ),
            source_path=source_path,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload
