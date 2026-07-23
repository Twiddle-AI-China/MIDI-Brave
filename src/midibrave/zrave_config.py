from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


_SECTION = TypeVar("_SECTION")
_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _strict_section(
    section_type: type[_SECTION],
    raw: object,
    name: str,
) -> _SECTION:
    if not isinstance(raw, dict):
        raise ValueError(f"missing config section: {name}")
    allowed = {field.name for field in fields(section_type)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"unknown {name} config keys: {', '.join(sorted(unknown))}"
        )
    try:
        return section_type(**raw)
    except TypeError as error:
        raise ValueError(f"invalid {name} config: {error}") from error


@dataclass(frozen=True)
class ZraveDataConfig:
    preset_metadata: str
    eligible_manifest: str
    audio_root: str
    selected_manifest: str
    metadata_output: str
    cache_root: str
    packed_root: str
    categories: tuple[str, ...] = _CATEGORIES
    presets_per_category: int = 10
    conditions_per_preset: int = 72
    warmup_frames: int = 64
    latent_hop: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "categories", tuple(self.categories))
        if self.categories != _CATEGORIES:
            raise ValueError(
                "data.categories must be Pad, Bass, Lead, Pluck, Keys in that order"
            )
        if self.presets_per_category != 10:
            raise ValueError("data.presets_per_category must be 10")
        if self.conditions_per_preset != 72:
            raise ValueError("data.conditions_per_preset must be 72")
        if self.warmup_frames != 64:
            raise ValueError("data.warmup_frames must be 64")
        if self.latent_hop != 128:
            raise ValueError("data.latent_hop must be 128")
        for name in (
            "preset_metadata",
            "eligible_manifest",
            "audio_root",
            "selected_manifest",
            "metadata_output",
            "cache_root",
            "packed_root",
        ):
            if not getattr(self, name):
                raise ValueError(f"data.{name} must not be empty")


@dataclass(frozen=True)
class ZraveRaveConfig:
    source_config: str
    checkpoint: str

    def __post_init__(self) -> None:
        if not self.source_config or not self.checkpoint:
            raise ValueError("rave source_config and checkpoint are required")


@dataclass(frozen=True)
class ZraveModelConfig:
    latent_dim: int = 16
    context_frames: int = 128
    horizon_frames: int = 16
    d_model: int = 384
    layers: int = 8
    heads: int = 8
    feedforward_dim: int = 1536
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.latent_dim != 16:
            raise ValueError("model.latent_dim must be 16")
        if self.context_frames != 128:
            raise ValueError("model.context_frames must be 128")
        if self.horizon_frames != 16:
            raise ValueError("model.horizon_frames must be 16")
        for name in ("d_model", "layers", "heads", "feedforward_dim"):
            _positive(f"model.{name}", getattr(self, name))
        if self.d_model % self.heads:
            raise ValueError("model.d_model must be divisible by model.heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class ZraveLossConfig:
    future: float = 1.0
    delta: float = 0.5
    acceleration: float = 0.05
    horizon_discount: float = 0.95
    statistic_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        for name in ("future", "delta", "acceleration", "statistic_floor"):
            _positive(f"loss.{name}", getattr(self, name))
        if not 0.0 < self.horizon_discount <= 1.0:
            raise ValueError("loss.horizon_discount must be in (0, 1]")


@dataclass(frozen=True)
class ZraveOptimizerConfig:
    learning_rate: float = 3.0e-4
    minimum_learning_rate: float = 3.0e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    def __post_init__(self) -> None:
        _positive("optimizer.learning_rate", self.learning_rate)
        _positive("optimizer.minimum_learning_rate", self.minimum_learning_rate)
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                "optimizer.minimum_learning_rate cannot exceed learning_rate"
            )
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")


@dataclass(frozen=True)
class ZraveTrainConfig:
    output_root: str
    batch_per_gpu: int = 256
    max_updates: int = 50000
    warmup_updates: int = 1000
    checkpoint_every: int = 2000
    validation_every: int = 2000
    validation_batches: int = 64
    early_stop_validations: int = 5
    log_every: int = 20
    gradient_clip: float = 1.0
    precision: str = "amp_fp16"
    seed: int = 20260723

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("train.output_root must not be empty")
        for name in (
            "batch_per_gpu",
            "max_updates",
            "warmup_updates",
            "checkpoint_every",
            "validation_every",
            "validation_batches",
            "early_stop_validations",
            "log_every",
        ):
            _positive(f"train.{name}", getattr(self, name))
        _positive("train.gradient_clip", self.gradient_clip)
        if self.precision != "amp_fp16":
            raise ValueError("train.precision must be amp_fp16")


@dataclass(frozen=True)
class ZraveConfig:
    seed: int
    data: ZraveDataConfig
    rave: ZraveRaveConfig
    model: ZraveModelConfig
    loss: ZraveLossConfig
    optimizer: ZraveOptimizerConfig
    train: ZraveTrainConfig
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def load(cls, path: str | Path) -> "ZraveConfig":
        source_path = Path(path).resolve()
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        allowed = {
            "seed",
            "data",
            "rave",
            "model",
            "loss",
            "optimizer",
            "train",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"unknown top-level config keys: {', '.join(sorted(unknown))}"
            )
        if "seed" not in raw:
            raise ValueError("missing config key: seed")
        return cls(
            seed=int(raw["seed"]),
            data=_strict_section(ZraveDataConfig, raw.get("data"), "data"),
            rave=_strict_section(ZraveRaveConfig, raw.get("rave"), "rave"),
            model=_strict_section(ZraveModelConfig, raw.get("model"), "model"),
            loss=_strict_section(ZraveLossConfig, raw.get("loss"), "loss"),
            optimizer=_strict_section(
                ZraveOptimizerConfig,
                raw.get("optimizer"),
                "optimizer",
            ),
            train=_strict_section(ZraveTrainConfig, raw.get("train"), "train"),
            source_path=source_path,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data": {
                field.name: getattr(self.data, field.name)
                for field in fields(self.data)
            },
            "rave": {
                field.name: getattr(self.rave, field.name)
                for field in fields(self.rave)
            },
            "model": {
                field.name: getattr(self.model, field.name)
                for field in fields(self.model)
            },
            "loss": {
                field.name: getattr(self.loss, field.name)
                for field in fields(self.loss)
            },
            "optimizer": {
                field.name: getattr(self.optimizer, field.name)
                for field in fields(self.optimizer)
            },
            "train": {
                field.name: getattr(self.train, field.name)
                for field in fields(self.train)
            },
        }
