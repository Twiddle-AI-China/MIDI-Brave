from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config


@dataclass(frozen=True)
class LatentStatisticsResult:
    latent_std: np.ndarray
    delta_std: np.ndarray
    acceleration_std: np.ndarray


class _Moments:
    def __init__(self, channels: int):
        self.count = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.square_total = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if not values.shape[-1]:
            return
        values = values.astype(np.float64, copy=False)
        self.count += values.shape[-1]
        self.total += values.sum(axis=1)
        self.square_total += np.square(values).sum(axis=1)

    def std(self) -> np.ndarray:
        if not self.count:
            raise ValueError("cannot finalize empty latent statistics")
        mean = self.total / self.count
        variance = np.maximum(0.0, self.square_total / self.count - np.square(mean))
        return np.sqrt(variance).astype(np.float32)


class LatentStatisticsAccumulator:
    def __init__(self, channels: int):
        if channels <= 0:
            raise ValueError("latent statistic channels must be positive")
        self.channels = channels
        self.latent = _Moments(channels)
        self.delta = _Moments(channels)
        self.acceleration = _Moments(channels)

    def update(self, latent: np.ndarray) -> None:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.ndim != 2 or latent.shape[0] != self.channels:
            raise ValueError("latent must have shape [channels, frames]")
        if not np.isfinite(latent).all():
            raise ValueError("latent statistics reject non-finite values")
        self.latent.update(latent)
        self.delta.update(np.diff(latent, axis=1))
        self.acceleration.update(np.diff(latent, n=2, axis=1))

    def finalize(self) -> LatentStatisticsResult:
        return LatentStatisticsResult(
            self.latent.std(), self.delta.std(), self.acceleration.std())


def save_latent_cache(path: str | Path, latent: np.ndarray, sample_id: str,
                      hop: int, checkpoint_hash: str) -> None:
    path = Path(path)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2 or not latent.shape[1] or not np.isfinite(latent).all():
        raise ValueError("latent cache requires finite [channels, frames] data")
    if hop <= 0 or not sample_id or not checkpoint_hash:
        raise ValueError("latent cache metadata is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output, latent=latent, sample_id=np.asarray(sample_id),
            hop=np.asarray(hop, dtype=np.int64),
            checkpoint_hash=np.asarray(checkpoint_hash))
    temporary.replace(path)


def load_latent_window(path: str | Path, sample_id: str, offset_samples: int,
                       frames: int, latent_dim: int, hop: int,
                       checkpoint_hash: str) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as cached:
        latent = cached["latent"].astype(np.float32, copy=False)
        actual = {
            "sample_id": str(cached["sample_id"].item()),
            "latent_dim": int(latent.shape[0]),
            "hop": int(cached["hop"].item()),
            "checkpoint_hash": str(cached["checkpoint_hash"].item()),
        }
    expected = {
        "sample_id": sample_id, "latent_dim": latent_dim,
        "hop": hop, "checkpoint_hash": checkpoint_hash,
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise ValueError(f"latent cache {field} mismatch: {actual[field]} != {value}")
    if offset_samples < 0 or offset_samples % hop:
        raise ValueError("offset_samples must be non-negative and hop-aligned")
    if frames <= 0:
        raise ValueError("frames must be positive")
    start = offset_samples // hop
    stop = start + frames
    if stop > latent.shape[-1]:
        raise ValueError("latent cache does not cover the requested window")
    return latent[:, start:stop].copy()


def cache_rave_latents_from_checkpoint(config: Config, checkpoint_path: str | Path,
                                       device: str) -> dict[str, object]:
    import torch

    from .data import configured_roots, load_audio, load_manifest, record_audio_path
    from .predictive_model import PredictiveMidiBrave

    if config.predictive is None:
        raise ValueError("RAVE latent caching requires a predictive config")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    model = PredictiveMidiBrave(
        config.model, config.predictive,
        config.data.window_samples, config.data.sample_rate).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("predictive checkpoint must contain a model state dictionary")
    model.load_state_dict(payload["model"])
    model.eval()
    manifest = Path(config.data.manifest).resolve()
    roots = configured_roots(config.data, manifest)
    records = load_manifest(manifest)
    output = Path(config.data.cache_root) / "rave"
    statistics = LatentStatisticsAccumulator(config.predictive.rave_latent_dim)
    cached = 0
    with torch.no_grad():
        for record in records:
            audio = load_audio(record_audio_path(record, roots), config.data.sample_rate)
            usable = len(audio) - len(audio) % config.predictive.samples_per_latent
            if usable < config.predictive.samples_per_latent:
                raise ValueError(f"audio is too short for RAVE cache: {record.sample_id}")
            tensor = torch.from_numpy(audio[:usable]).view(1, 1, -1).to(device)
            latent = model.encode_audio(tensor, sample=False).latent[0].cpu().numpy()
            save_latent_cache(output / f"{record.cache_id}.npz", latent,
                              record.sample_id, config.predictive.samples_per_latent,
                              checkpoint_hash)
            statistics.update(latent)
            cached += 1
    values = statistics.finalize()
    stats_path = Path(config.data.cache_root) / "rave-statistics.npz"
    temporary = stats_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination, latent_std=values.latent_std,
            delta_std=values.delta_std,
            acceleration_std=values.acceleration_std,
            checkpoint_hash=np.asarray(checkpoint_hash),
            samples_per_latent=np.asarray(config.predictive.samples_per_latent,
                                          dtype=np.int64))
    temporary.replace(stats_path)
    return {"cached": cached, "checkpoint_hash": checkpoint_hash,
            "statistics": str(stats_path.resolve())}
