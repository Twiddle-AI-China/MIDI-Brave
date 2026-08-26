from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .atlas_flow_atlas import TimbreAtlas
from .atlas_flow_config import AtlasDataConfig, AtlasModelConfig
from .atlas_flow_model import FlowBatch
from .data import SampleRecord, load_manifest


FEATURE_SCHEMA = "pitch-normalized-96-v1"
TRAJECTORY_SCHEMA = "atlas-trajectory-128-v1"


def _safe_id(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def feature_path(record: SampleRecord, config: AtlasDataConfig) -> Path:
    return Path(config.feature_cache) / FEATURE_SCHEMA / f"{_safe_id(record.sample_id)}.npy"


def trajectory_path(record: SampleRecord, config: AtlasDataConfig) -> Path:
    return Path(config.trajectory_cache) / TRAJECTORY_SCHEMA / f"{_safe_id(record.sample_id)}.npy"


def resolve_audio(record: SampleRecord, config: AtlasDataConfig) -> Path:
    path = Path(record.audio_path)
    return path if path.is_absolute() else Path(config.audio_root) / path


def _frames(waveform: Tensor, length: int, hop: int) -> Tensor:
    half = length // 2
    padded = F.pad(waveform[None, None], (half, half), mode="reflect")[0, 0]
    return padded.unfold(0, length, hop)


@torch.no_grad()
def pitch_normalized_features(
    waveform: Tensor,
    midi_note: int,
    *,
    sample_rate: int = 44_100,
    hop: int = 512,
    fft_size: int = 2_048,
) -> Tensor:
    """64 harmonic + 24 relative-log-frequency + 8 statistics."""
    waveform = waveform.float().flatten()
    if waveform.numel() < fft_size:
        raise ValueError("audio is too short for feature extraction")
    spectrum = torch.stft(
        waveform, fft_size, hop, fft_size, torch.hann_window(fft_size, device=waveform.device),
        center=True, pad_mode="reflect", return_complex=True,
    )
    magnitude = spectrum.abs().clamp_min(1.0e-7)
    power = magnitude.square()
    frequencies = torch.fft.rfftfreq(fft_size, 1.0 / sample_rate).to(waveform.device)
    f0 = 440.0 * 2.0 ** ((float(midi_note) - 69.0) / 12.0)
    hz_per_bin = sample_rate / fft_size
    harmonic_hz = f0 * torch.arange(1, 65, device=waveform.device, dtype=torch.float32)
    harmonic_bin = torch.round(harmonic_hz / hz_per_bin).long().clamp(0, magnitude.shape[0] - 1)
    harmonic = magnitude[harmonic_bin]
    harmonic = torch.where(
        (harmonic_hz <= sample_rate / 2.0)[:, None], harmonic, torch.full_like(harmonic, 1.0e-7)
    )
    harmonic = torch.log1p(32.0 * harmonic)
    harmonic = harmonic - harmonic.mean(0, keepdim=True)
    ratios = torch.pow(2.0, torch.linspace(-1.0, 6.0, 24, device=waveform.device))
    relative_hz = f0 * ratios
    relative_bin = torch.round(relative_hz / hz_per_bin).long().clamp(0, magnitude.shape[0] - 1)
    relative = magnitude[relative_bin]
    relative = torch.where(
        (relative_hz <= sample_rate / 2.0)[:, None], relative, torch.full_like(relative, 1.0e-7)
    )
    relative = torch.log1p(32.0 * relative)
    relative = relative - relative.mean(0, keepdim=True)
    total_power = power.sum(0).clamp_min(1.0e-9)
    centroid = (power * frequencies[:, None]).sum(0) / total_power
    centroid = torch.log2((centroid / max(f0, 1.0e-3)).clamp_min(1.0e-4)) / 8.0
    flatness = magnitude.log().mean(0).exp() / magnitude.mean(0).clamp_min(1.0e-7)
    cumulative = power.cumsum(0) / total_power
    rolloff_index = (cumulative >= 0.85).float().argmax(0)
    rolloff = torch.log2((frequencies[rolloff_index] / max(f0, 1.0e-3)).clamp_min(1.0e-4)) / 8.0
    harmonic_power = harmonic.square().sum(0) / total_power
    low = power[frequencies <= 2_000.0].sum(0)
    high_ratio = (total_power - low).clamp_min(0.0) / total_power
    frame_rms = _frames(waveform, fft_size, hop).square().mean(-1).sqrt()[:magnitude.shape[1]]
    log_rms = (20.0 * torch.log10(frame_rms.clamp_min(1.0e-7)) + 60.0) / 60.0
    flux = F.pad(
        (magnitude[:, 1:] - magnitude[:, :-1]).abs().mean(0)
        / magnitude[:, 1:].mean(0).clamp_min(1.0e-7), (1, 0),
    )
    periodic = magnitude[harmonic_bin[:8]].mean(0) / magnitude.mean(0).clamp_min(1.0e-7)
    statistics = torch.stack(
        (log_rms, centroid, flatness, rolloff, harmonic_power, high_ratio, flux, periodic)
    ).clamp(-4.0, 4.0)
    result = torch.cat((harmonic, relative, statistics), 0)
    if result.shape[0] != 96 or not torch.isfinite(result).all():
        raise RuntimeError("invalid 96D feature result")
    return result


def cache_feature(record: SampleRecord, config: AtlasDataConfig, overwrite: bool = False) -> Path:
    destination = feature_path(record, config)
    if destination.is_file() and not overwrite:
        return destination
    audio, rate = sf.read(resolve_audio(record, config), dtype="float32", always_2d=True)
    if rate != config.sample_rate or audio.shape[1] != 1 or audio.shape[0] != config.render_samples:
        raise ValueError(f"{record.sample_id}: audio violates the fixed render contract")
    value = pitch_normalized_features(
        torch.from_numpy(audio[:, 0]), record.midi_note,
        sample_rate=config.sample_rate, hop=config.feature_hop,
    ).cpu().numpy().astype(np.float16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(destination)
    return destination


class AtlasAudioDataset(Dataset[dict[str, Any]]):
    """Preset-balanced same-timbre/cross-note pairs for autoencoding."""

    def __init__(self, config: AtlasDataConfig, split: str, repeats: int = 64) -> None:
        self.config = config
        self.split = split
        grouped: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in load_manifest(config.manifest):
            if record.split == split:
                grouped[record.preset_id].append(record)
        if not grouped:
            raise ValueError(f"manifest has no {split} records")
        self.grouped = {key: tuple(sorted(value, key=lambda item: item.sample_id)) for key, value in grouped.items()}
        self.presets = tuple(sorted(self.grouped))
        self.preset_to_index = {preset: index for index, preset in enumerate(self.presets)}
        self.repeats = repeats if split == "train" else 1

    def __len__(self) -> int:
        return len(self.presets) * self.repeats

    def _rng(self, index: int) -> np.random.Generator:
        seed = hashlib.sha256(f"{self.split}:{index}".encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(seed, "big"))

    def _audio_window(self, record: SampleRecord, start: int) -> Tensor:
        with sf.SoundFile(resolve_audio(record, self.config)) as handle:
            handle.seek(start)
            value = handle.read(self.config.window_samples, dtype="float32", always_2d=True)
        return torch.from_numpy(value[:, 0]).view(1, -1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = self._rng(index)
        preset = self.presets[index % len(self.presets)]
        records = self.grouped[preset]
        first = records[int(rng.integers(len(records)))]
        choices = [item for item in records if item.midi_note != first.midi_note] or list(records)
        second = choices[int(rng.integers(len(choices)))]
        maximum = self.config.render_samples - self.config.window_samples
        start = int(rng.integers(0, maximum + 1))
        features_a = torch.from_numpy(np.load(feature_path(first, self.config)).astype(np.float32))
        features_b = torch.from_numpy(np.load(feature_path(second, self.config)).astype(np.float32))
        return {
            "sample_id_a": first.sample_id,
            "sample_id_b": second.sample_id,
            "preset_id": preset,
            "preset_index": torch.tensor(self.preset_to_index[preset]),
            "features_a": features_a,
            "features_b": features_b,
            "audio_a": self._audio_window(first, start),
            "audio_b": self._audio_window(second, start),
            "note_a": torch.tensor(first.midi_note),
            "note_b": torch.tensor(second.midi_note),
            "window_start": torch.tensor(start),
        }


def lifecycle_frames(start_frame: int, count: int, config: AtlasDataConfig) -> np.ndarray:
    samples = (start_frame + np.arange(count, dtype=np.float32)) * config.trajectory_hop
    age = np.clip((samples - config.note_on_sample) / max(config.note_off_sample - config.note_on_sample, 1), 0, 1)
    gate = ((samples >= config.note_on_sample) & (samples < config.note_off_sample)).astype(np.float32)
    release = np.clip((samples - config.note_off_sample) / max(config.render_samples - config.note_off_sample, 1), 0, 1)
    onset = np.exp(-8.0 * age) * (samples >= config.note_on_sample)
    return np.stack((age, gate, release, onset), axis=-1).astype(np.float32)


class CachedFlowDataset(Dataset[dict[str, Tensor]]):
    """40% full-plan, 40% continuation, 20% same-class switch tasks."""

    def __init__(
        self,
        data: AtlasDataConfig,
        model: AtlasModelConfig,
        split: str,
        samples_per_epoch: int = 100_000,
    ) -> None:
        self.data = data
        self.model = model
        self.records = tuple(record for record in load_manifest(data.manifest) if record.split == split)
        if not self.records:
            raise ValueError(f"no {split} records")
        self.atlas = TimbreAtlas.load(data.atlas_path)
        self.atlas_index = {preset: index for index, preset in enumerate(self.atlas.preset_ids)}
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> np.random.Generator:
        digest = hashlib.sha256(f"flow:{index}".encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _trajectory(self, record: SampleRecord) -> np.ndarray:
        value = np.load(trajectory_path(record, self.data), allow_pickle=False).astype(np.float32)
        if value.ndim != 2 or value.shape[0] != 128 or value.shape[1] < 96:
            raise ValueError(f"invalid trajectory cache for {record.sample_id}: {value.shape}")
        return value.T

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = self._rng(index)
        record = self.records[int(rng.integers(len(self.records)))]
        source = self._trajectory(record)
        atlas_index = self.atlas_index[record.preset_id]
        source_anchor = self.atlas.anchors[atlas_index]
        source_coord = self.atlas.coordinates[atlas_index]
        context, future = self.model.context_frames, self.model.future_frames
        task_draw = float(rng.random())
        if task_draw < 0.40:
            start = 0
            history = np.repeat(source_anchor[None], context, axis=0)
            history_anchor = history.copy()
            history_mask = np.zeros(context, dtype=np.bool_)
            history_mask[-1] = True
            target = source[:future]
            anchor_path = np.repeat(source_anchor[None], future, axis=0)
            atlas_path = np.repeat(source_coord[None], future, axis=0)
        elif task_draw < 0.80:
            start = int(rng.integers(context, source.shape[0] - future + 1))
            history = source[start - context:start]
            history_anchor = np.repeat(source_anchor[None], context, axis=0)
            history_mask = np.ones(context, dtype=np.bool_)
            target = source[start:start + future]
            anchor_path = np.repeat(source_anchor[None], future, axis=0)
            atlas_path = np.repeat(source_coord[None], future, axis=0)
        else:
            candidates = [item for item in self.records if item.preset_id != record.preset_id]
            target_record = candidates[int(rng.integers(len(candidates)))]
            target_full = self._trajectory(target_record)
            target_index = self.atlas_index[target_record.preset_id]
            target_anchor = self.atlas.anchors[target_index]
            target_coord = self.atlas.coordinates[target_index]
            start = int(rng.integers(context, min(source.shape[0], target_full.shape[0]) - future + 1))
            history = source[start - context:start]
            history_anchor = np.repeat(source_anchor[None], context, axis=0)
            history_mask = np.ones(context, dtype=np.bool_)
            target = target_full[start:start + future]
            bridge_frames = min(future, max(2, round(0.5 * self.data.sample_rate / self.data.trajectory_hop)))
            phase = np.clip(np.arange(future, dtype=np.float32) / (bridge_frames - 1), 0, 1)
            smooth = phase * phase * (3.0 - 2.0 * phase)
            anchor_path = source_anchor[None] * (1 - smooth[:, None]) + target_anchor[None] * smooth[:, None]
            atlas_path = source_coord[None] * (1 - smooth[:, None]) + target_coord[None] * smooth[:, None]
        return {
            "history": torch.from_numpy(history.copy()),
            "history_anchor": torch.from_numpy(history_anchor.copy()),
            "target": torch.from_numpy(target.copy()),
            "anchor_path": torch.from_numpy(anchor_path.astype(np.float32)),
            "atlas_path": torch.from_numpy(atlas_path.astype(np.float32)),
            "lifecycle": torch.from_numpy(lifecycle_frames(start, future, self.data)),
            "history_mask": torch.from_numpy(history_mask),
        }


def as_flow_batch(batch: dict[str, Tensor]) -> FlowBatch:
    return FlowBatch(**{name: batch[name] for name in FlowBatch.__dataclass_fields__})


def manifest_audit(config: AtlasDataConfig, check_audio: bool = False) -> dict[str, object]:
    records = load_manifest(config.manifest)
    preset_splits: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for record in records:
        preset_splits[record.preset_id].add(str(record.split))
        if (record.sample_rate, record.num_samples, record.velocity) != (44_100, 220_500, 127):
            errors.append(f"{record.sample_id}: render contract mismatch")
        if check_audio and not resolve_audio(record, config).is_file():
            errors.append(f"{record.sample_id}: missing audio")
    leakage = {preset: sorted(splits) for preset, splits in preset_splits.items() if len(splits) != 1}
    counts = {split: sum(next(iter(value)) == split for value in preset_splits.values()) for split in ("train", "validation", "test")}
    if len(records) != 2_700 or len(preset_splits) != 50 or counts != {"train": 45, "validation": 2, "test": 3}:
        errors.append(f"cardinality mismatch: rows={len(records)} presets={len(preset_splits)} splits={counts}")
    if leakage:
        errors.append(f"preset leakage: {leakage}")
    report = {"records": len(records), "presets": len(preset_splits), "split_presets": counts, "errors": errors, "passed": not errors}
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors[:5]))
    return report


__all__ = [
    "AtlasAudioDataset", "CachedFlowDataset", "FEATURE_SCHEMA", "TRAJECTORY_SCHEMA",
    "as_flow_batch", "cache_feature", "feature_path", "lifecycle_frames", "manifest_audit",
    "pitch_normalized_features", "resolve_audio", "trajectory_path",
]
