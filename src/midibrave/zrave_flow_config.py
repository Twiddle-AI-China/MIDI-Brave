from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from math import isclose
from pathlib import Path
from typing import Any, TypeVar

import yaml


_SECTION = TypeVar("_SECTION")
_WANDER_DELAYS = (16, 32, 48)


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
        if not names:
            raise ValueError("data.sources must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("data source names must be unique")


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
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("rave.expected_sha256 must be a SHA-256 digest")
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
    pitch_conditioning: bool = True
    midi_sequence_conditioning: bool = False
    note_min: int = 21
    note_max: int = 109
    condition_dropout: float = 0.10
    pitch_guidance: float = 3.0
    solver_steps: int = 8
    wander_delay_frames: tuple[int, ...] = _WANDER_DELAYS

    def __post_init__(self) -> None:
        if not isinstance(self.pitch_conditioning, bool):
            raise ValueError("model.pitch_conditioning must be boolean")
        if not isinstance(self.midi_sequence_conditioning, bool):
            raise ValueError(
                "model.midi_sequence_conditioning must be boolean"
            )
        if self.midi_sequence_conditioning and not self.pitch_conditioning:
            raise ValueError(
                "model.midi_sequence_conditioning requires "
                "pitch_conditioning"
            )
        object.__setattr__(
            self,
            "wander_delay_frames",
            tuple(self.wander_delay_frames),
        )
        exact = {
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
        _positive("model.latent_dim", self.latent_dim)
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
        }
        for name, expected in exact_floats.items():
            if getattr(self, name) != expected:
                raise ValueError(f"train.{name} must be {expected}")
        if self.exposure_prefix_frames != 32:
            raise ValueError("train.exposure_prefix_frames must be 32")
        if self.pitch_transition_fraction not in {0.0, 0.20}:
            raise ValueError(
                "train.pitch_transition_fraction must be 0.0 or 0.20"
            )
        if self.seed < 0:
            raise ValueError("train.seed must be non-negative")


@dataclass(frozen=True)
class FlowExplorationConfig:
    enabled: bool = False
    visible_history_frames: tuple[int, ...] = (8, 16, 32)
    temperature_minimum: float = 0.7
    temperature_maximum: float = 1.3
    wander_delay_minimum: int = 16
    wander_delay_maximum: int = 48
    schedule_offsets: tuple[int, ...] = (0, 16, 32, 64, 128)
    rollout_stride_frames: int = 16
    candidate_count: int = 4
    temporal_loss_weight: float = 0.05
    exposure_start_update: int = 1000
    exposure_ramp_updates: int = 4000
    exposure_probability: float = 0.50
    exposure_max_depth: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visible_history_frames",
            tuple(self.visible_history_frames),
        )
        object.__setattr__(
            self,
            "schedule_offsets",
            tuple(self.schedule_offsets),
        )
        if not isinstance(self.enabled, bool):
            raise ValueError("exploration.enabled must be boolean")
        if self.visible_history_frames != (8, 16, 32):
            raise ValueError(
                "exploration.visible_history_frames must be 8, 16, 32"
            )
        if self.schedule_offsets != (0, 16, 32, 64, 128):
            raise ValueError(
                "exploration.schedule_offsets must be 0, 16, 32, 64, 128"
            )
        if (
            self.temperature_minimum,
            self.temperature_maximum,
        ) != (0.7, 1.3):
            raise ValueError(
                "exploration temperature range must be 0.7 through 1.3"
            )
        if (
            self.wander_delay_minimum,
            self.wander_delay_maximum,
        ) != (16, 48):
            raise ValueError(
                "exploration wander delay range must be 16 through 48"
            )
        if self.rollout_stride_frames != 16:
            raise ValueError(
                "exploration.rollout_stride_frames must be 16"
            )
        if self.exposure_max_depth != 3:
            raise ValueError("exploration.exposure_max_depth must be 3")
        if self.candidate_count not in {1, 2, 4}:
            raise ValueError(
                "exploration.candidate_count must be 1, 2, or 4"
            )
        if self.temporal_loss_weight != 0.05:
            raise ValueError(
                "exploration.temporal_loss_weight must be 0.05"
            )
        if self.exposure_start_update != 1000:
            raise ValueError(
                "exploration.exposure_start_update must be 1000"
            )
        if self.exposure_ramp_updates != 4000:
            raise ValueError(
                "exploration.exposure_ramp_updates must be 4000"
            )
        if self.exposure_probability != 0.50:
            raise ValueError(
                "exploration.exposure_probability must be 0.50"
            )


@dataclass(frozen=True)
class FlowSegmentSamplingConfig:
    """Optional causal task that predicts balanced latent partitions.

    Segment zero has no preceding latent context.  It is intentionally not a
    valid target until the model has an explicit beginning-of-sequence
    contract; silently filling its history from the target would leak future
    information.
    """

    enabled: bool = False
    divisions: tuple[int, ...] = (2, 4, 8)
    include_first: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "divisions", tuple(self.divisions))
        if not isinstance(self.enabled, bool):
            raise ValueError("segment_sampling.enabled must be boolean")
        if not isinstance(self.include_first, bool):
            raise ValueError(
                "segment_sampling.include_first must be boolean"
            )
        if (
            not self.divisions
            or tuple(sorted(set(self.divisions))) != self.divisions
            or any(value not in {2, 4, 8} for value in self.divisions)
        ):
            raise ValueError(
                "segment_sampling.divisions must be an ordered, unique "
                "subset of 2, 4, 8"
            )
        if self.include_first:
            raise ValueError(
                "segment_sampling.include_first requires an explicit "
                "beginning-of-sequence history contract"
            )


@dataclass(frozen=True)
class ZraveFlowConfig:
    seed: int
    data: FlowDataConfig
    rave: FlowRaveConfig
    model: FlowModelConfig
    loss: FlowLossConfig
    optimizer: FlowOptimizerConfig
    train: FlowTrainConfig
    exploration: FlowExplorationConfig = field(
        default_factory=FlowExplorationConfig
    )
    segment_sampling: FlowSegmentSamplingConfig = field(
        default_factory=FlowSegmentSamplingConfig
    )
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.seed != self.train.seed:
            raise ValueError("seed and train.seed must match")
        expected_transition_fraction = (
            0.20 if self.model.pitch_conditioning else 0.0
        )
        if (
            self.train.pitch_transition_fraction
            != expected_transition_fraction
        ):
            raise ValueError(
                "train.pitch_transition_fraction must match "
                "model.pitch_conditioning"
            )
        if self.exploration.enabled and self.model.pitch_conditioning:
            raise ValueError(
                "exploration with MIDI conditioning is not yet supported"
            )
        if self.exploration.enabled and self.segment_sampling.enabled:
            raise ValueError(
                "segment sampling cannot use generated-history exposure"
            )

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
            "exploration",
            "segment_sampling",
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
        exploration_raw = dict(root.get("exploration") or {})
        for name in ("visible_history_frames", "schedule_offsets"):
            if name in exploration_raw:
                exploration_raw[name] = tuple(exploration_raw[name])
        segment_sampling_raw = dict(root.get("segment_sampling") or {})
        if "divisions" in segment_sampling_raw:
            segment_sampling_raw["divisions"] = tuple(
                segment_sampling_raw["divisions"]
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
            exploration=_strict_section(
                FlowExplorationConfig,
                exploration_raw,
                "exploration",
            ),
            segment_sampling=_strict_section(
                FlowSegmentSamplingConfig,
                segment_sampling_raw,
                "segment_sampling",
            ),
            source_path=source_path,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload
