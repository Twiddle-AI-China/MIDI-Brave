from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from .data import load_audio
from .zrave_codec import decode_with_seed, encode_posterior_mean
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_manifest import build_flow_manifest, read_flow_manifest


_SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
_SPLIT_NAMES = {value: key for key, value in _SPLIT_CODES.items()}
_SEED_CATEGORIES = (
    "Pad",
    "Lead",
    "Bass",
    "Pluck",
    "Keys",
    "Arp",
    "Chord",
    "Synth",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _atomic_npz(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _codec_scalar(value: object) -> int:
    if isinstance(value, Tensor):
        return int(value.flatten()[0].item())
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return int(value[0])  # type: ignore[index]
    return int(value)


def _load_and_validate_codec(
    config: ZraveFlowConfig,
    device: str,
    injected: nn.Module | None,
) -> tuple[nn.Module, str]:
    if injected is None:
        checkpoint = Path(config.rave.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing RAVE checkpoint: {checkpoint}")
        codec_hash = _sha256_file(checkpoint)
        if codec_hash != config.rave.expected_sha256:
            raise ValueError(
                f"RAVE checkpoint SHA-256 mismatch: {codec_hash}"
            )
        codec = torch.jit.load(str(checkpoint), map_location=device)
    else:
        codec_hash = config.rave.expected_sha256
        codec = injected
    codec = codec.to(device).eval()
    if _codec_scalar(getattr(codec, "sr")) != config.rave.sample_rate:
        raise ValueError("standalone RAVE sample-rate mismatch")
    if _codec_scalar(getattr(codec, "latent_size")) != config.model.latent_dim:
        raise ValueError("standalone RAVE latent-size mismatch")
    probe_frames = 4
    probe_audio = torch.zeros(
        1,
        1,
        probe_frames * config.rave.latent_hop,
        device=device,
    )
    with torch.inference_mode():
        probe = encode_posterior_mean(codec, probe_audio)
        if tuple(probe.shape) != (
            1,
            config.model.latent_dim,
            probe_frames,
        ):
            raise ValueError(
                f"standalone RAVE latent layout mismatch: {tuple(probe.shape)}"
            )
        if injected is None:
            decoded = decode_with_seed(codec, probe, config.seed)
            if decoded.shape[-1] != probe_frames * config.rave.latent_hop:
                raise ValueError("standalone RAVE decode-hop mismatch")
            if not torch.isfinite(decoded).all():
                raise ValueError("standalone RAVE decoder is non-finite")
    if not torch.isfinite(probe).all():
        raise ValueError("standalone RAVE encoder is non-finite")
    return codec, codec_hash


def _active_latent_frames(
    audio: np.ndarray,
    *,
    sample_rate: int,
    hop: int,
    encoded_frames: int,
) -> int:
    threshold = 10.0 ** (-60.0 / 20.0)
    window = max(1, round(sample_rate * 0.100))
    active = 0
    for frame in range(1, encoded_frames + 1):
        end = min(len(audio), frame * hop)
        start = max(0, end - window)
        if end <= start:
            continue
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio[start:end].astype(np.float64, copy=False)
                    )
                )
            )
        )
        if math.isfinite(rms) and rms > threshold:
            active = frame
    return min(active, encoded_frames)


def _part_paths(
    cache_root: Path,
    rank: int,
    part_index: int,
) -> tuple[Path, Path]:
    root = cache_root / f"rank-{rank}"
    stem = f"part-{part_index:06d}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _write_part(
    config: ZraveFlowConfig,
    rank: int,
    part_index: int,
    records: list[dict[str, Any]],
    codec_hash: str,
) -> tuple[Path, Path]:
    npz_path, json_path = _part_paths(
        Path(config.data.cache_root),
        rank,
        part_index,
    )
    maximum_length = max(record["latent"].shape[0] for record in records)
    count = len(records)
    latents = np.zeros(
        (count, maximum_length, config.model.latent_dim),
        dtype=np.float16,
    )
    for index, record in enumerate(records):
        sequence = record["latent"]
        latents[index, : sequence.shape[0]] = sequence.astype(
            np.float16,
            copy=False,
        )
    _atomic_npz(
        npz_path,
        latents=latents,
        lengths=np.asarray(
            [record["length"] for record in records],
            dtype=np.int16,
        ),
        active_frames=np.asarray(
            [record["active_frames"] for record in records],
            dtype=np.int16,
        ),
        notes=np.asarray(
            [record["midi_note"] for record in records],
            dtype=np.int16,
        ),
        velocities=np.asarray(
            [record["velocity"] for record in records],
            dtype=np.int16,
        ),
        split_codes=np.asarray(
            [record["split_code"] for record in records],
            dtype=np.uint8,
        ),
        source_codes=np.asarray(
            [record["source_code"] for record in records],
            dtype=np.uint8,
        ),
        category_codes=np.asarray(
            [record["category_code"] for record in records],
            dtype=np.uint16,
        ),
        maximum_future_frames=np.asarray(
            [record["maximum_future_frames"] for record in records],
            dtype=np.uint8,
        ),
        manifest_indices=np.asarray(
            [record["manifest_index"] for record in records],
            dtype=np.int64,
        ),
    )
    metadata_records: list[dict[str, object]] = []
    for local_index, record in enumerate(records):
        metadata_records.append(
            {
                key: value
                for key, value in record.items()
                if key != "latent"
            }
            | {"local_index": local_index}
        )
    _atomic_json(
        json_path,
        {
            "schema": 1,
            "rank": rank,
            "part_index": part_index,
            "codec_sha256": codec_hash,
            "manifest_sha256": _sha256_file(
                config.data.unified_manifest
            ),
            "npz_sha256": _sha256_file(npz_path),
            "records": metadata_records,
        },
    )
    return npz_path, json_path


def encode_flow_shards(
    config: ZraveFlowConfig,
    rank: int,
    world_size: int,
    device: str,
    codec: nn.Module | None = None,
) -> dict[str, object]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    rows = read_flow_manifest(config.data.unified_manifest)
    source_vocab = [source.name for source in config.data.sources]
    source_codes = {name: index for index, name in enumerate(source_vocab)}
    category_vocab = sorted({str(row["category"]) for row in rows})
    category_codes = {
        name: index for index, name in enumerate(category_vocab)
    }
    model, codec_hash = _load_and_validate_codec(
        config,
        device,
        codec,
    )
    assigned = list(enumerate(rows))[rank::world_size]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, object]] = []
    part_count = 0
    encoded_count = 0

    def flush() -> None:
        nonlocal part_count, encoded_count
        if not accepted:
            return
        _write_part(
            config,
            rank,
            part_count,
            list(accepted),
            codec_hash,
        )
        encoded_count += len(accepted)
        part_count += 1
        accepted.clear()

    with torch.inference_mode():
        for manifest_index, row in assigned:
            try:
                audio = load_audio(
                    Path(str(row["audio_path"])),
                    config.rave.sample_rate,
                )
                maximum_samples = round(
                    config.rave.sample_rate
                    * config.data.maximum_audio_seconds
                )
                audio = audio[:maximum_samples]
                usable = len(audio) - len(audio) % config.rave.latent_hop
                if usable < config.rave.latent_hop:
                    raise ValueError("audio is shorter than one latent hop")
                tensor = (
                    torch.from_numpy(audio[:usable])
                    .view(1, 1, -1)
                    .to(device)
                )
                encoded = encode_posterior_mean(model, tensor)
                if (
                    encoded.ndim != 3
                    or encoded.shape[0] != 1
                    or encoded.shape[1] != config.model.latent_dim
                ):
                    raise ValueError(
                        f"invalid encoded shape: {tuple(encoded.shape)}"
                    )
                if not torch.isfinite(encoded).all():
                    raise ValueError("encoded latent is non-finite")
                latent = encoded[0].transpose(0, 1).float().cpu().numpy()
                active_frames = _active_latent_frames(
                    audio[:usable],
                    sample_rate=config.rave.sample_rate,
                    hop=config.rave.latent_hop,
                    encoded_frames=latent.shape[0],
                )
                if active_frames < config.model.context_frames + 1:
                    raise ValueError(
                        f"only {active_frames} active latent frames"
                    )
                split = str(row["split"])
                if split not in _SPLIT_CODES:
                    raise ValueError(f"invalid split: {split}")
                source_name = str(row["source_name"])
                category = str(row["category"])
                accepted.append(
                    {
                        "latent": latent,
                        "manifest_index": manifest_index,
                        "sample_id": str(row["sample_id"]),
                        "source_name": source_name,
                        "source_code": source_codes[source_name],
                        "category": category,
                        "category_code": category_codes[category],
                        "canonical_preset_id": str(
                            row["canonical_preset_id"]
                        ),
                        "split": split,
                        "split_code": _SPLIT_CODES[split],
                        "midi_note": int(row["midi_note"]),
                        "velocity": int(row["velocity"]),
                        "articulation_id": str(row["articulation_id"]),
                        "maximum_future_frames": int(
                            row["maximum_future_frames"]
                        ),
                        "length": int(latent.shape[0]),
                        "active_frames": active_frames,
                    }
                )
                if len(accepted) == config.data.shard_records:
                    flush()
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                rejected.append(
                    {
                        "manifest_index": manifest_index,
                        "sample_id": row.get("sample_id"),
                        "reason": str(error),
                    }
                )
    flush()
    report: dict[str, object] = {
        "schema": 1,
        "rank": rank,
        "world_size": world_size,
        "assigned": len(assigned),
        "records": encoded_count,
        "shards": part_count,
        "rejected": len(rejected),
        "codec_sha256": codec_hash,
        "manifest_sha256": _sha256_file(config.data.unified_manifest),
        "source_vocab": source_vocab,
        "category_vocab": category_vocab,
        "rejections": rejected,
    }
    _atomic_json(
        Path(config.data.cache_root) / f"rank-{rank}" / "report.json",
        report,
    )
    return report


@dataclass
class _Moments:
    channels: int
    count: int = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        cast = values.astype(np.float64, copy=False).reshape(
            -1,
            self.channels,
        )
        if self.total is None:
            self.total = np.zeros(self.channels, dtype=np.float64)
            self.total_square = np.zeros(self.channels, dtype=np.float64)
        assert self.total_square is not None
        self.count += cast.shape[0]
        self.total += cast.sum(axis=0)
        self.total_square += np.square(cast).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0 or self.total is None or self.total_square is None:
            raise ValueError("cannot finalize empty training statistics")
        mean = self.total / self.count
        variance = np.maximum(
            self.total_square / self.count - np.square(mean),
            1.0e-12,
        )
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _read_part_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("records"),
        list,
    ):
        raise ValueError(f"invalid part metadata: {path}")
    npz_path = path.with_suffix(".npz")
    if payload.get("npz_sha256") != _sha256_file(npz_path):
        raise ValueError(f"part hash mismatch: {npz_path}")
    records: list[dict[str, Any]] = []
    for value in payload["records"]:
        if not isinstance(value, dict):
            raise ValueError(f"invalid record metadata: {path}")
        records.append(
            dict(value)
            | {
                "part_npz": str(npz_path),
                "part_json": str(path),
            }
        )
    return records


def _load_descriptor_sequence(descriptor: dict[str, Any]) -> np.ndarray:
    with np.load(descriptor["part_npz"], allow_pickle=False) as part:
        length = int(descriptor["length"])
        return part["latents"][
            int(descriptor["local_index"]),
            :length,
        ].astype(np.float32)


def _build_pitch_pairs(
    sequences: list[dict[str, object]],
) -> np.ndarray:
    buckets: dict[
        tuple[str, str, str, int, str, int],
        list[tuple[str, int]],
    ] = defaultdict(list)
    for index, row in enumerate(sequences):
        source = str(row["source_name"])
        key = (
            source,
            str(row["split"]),
            str(row["canonical_preset_id"]),
            int(row["velocity"]),
            str(row["articulation_id"]),
            int(row["midi_note"]),
        )
        buckets[key].append((str(row["sample_id"]), index))
    result = np.full(len(sequences), -1, dtype=np.int64)
    for index, row in enumerate(sequences):
        source = str(row["source_name"])
        base = (
            source,
            str(row["split"]),
            str(row["canonical_preset_id"]),
            int(row["velocity"]),
            str(row["articulation_id"]),
        )
        candidates: list[tuple[str, int]] = []
        note = int(row["midi_note"])
        for partner_note in (note - 12, note + 12):
            candidates.extend(buckets.get(base + (partner_note,), []))
        if candidates:
            result[index] = min(candidates)[1]
    return result


def _packed_sequence(
    packed_root: Path,
    row: dict[str, object],
) -> np.ndarray:
    with np.load(
        packed_root / str(row["shard"]),
        allow_pickle=False,
    ) as shard:
        length = int(row["length"])
        return shard["latents"][
            int(row["shard_row"]),
            :length,
        ].astype(np.float32)


def _write_seed_bank(
    packed_root: Path,
    sequences: list[dict[str, object]],
) -> dict[str, object]:
    priority = {
        "serum_full": 0,
        "dexed_surge_broad": 1,
        "pianobook_pitch": 2,
    }
    candidates = [
        row
        for row in sequences
        if row["split"] in {"train", "validation"}
        and row["source_name"] in priority
        and (
            row["category"] in _SEED_CATEGORIES
            or row["source_name"] == "pianobook_pitch"
        )
        and int(row["active_frames"]) >= 32
    ]
    candidates.sort(
        key=lambda row: (
            int(row["midi_note"]),
            str(row["category"]),
            priority[str(row["source_name"])],
            0 if row["split"] == "validation" else 1,
            str(row["canonical_preset_id"]),
            str(row["sample_id"]),
        )
    )
    selected: list[dict[str, object]] = []
    used: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in candidates:
        key = (int(row["midi_note"]), str(row["category"]))
        preset = str(row["canonical_preset_id"])
        if preset in used[key] or len(used[key]) >= 4:
            continue
        used[key].add(preset)
        selected.append(row)
    histories = np.zeros(
        (len(selected), 32, 16),
        dtype=np.float16,
    )
    for index, row in enumerate(selected):
        histories[index] = _packed_sequence(packed_root, row)[:32]
    _atomic_npz(
        packed_root / "seed-bank.npz",
        history=histories,
        notes=np.asarray(
            [int(row["midi_note"]) for row in selected],
            dtype=np.int16,
        ),
        source_codes=np.asarray(
            [int(row["source_code"]) for row in selected],
            dtype=np.uint8,
        ),
        category_codes=np.asarray(
            [int(row["category_code"]) for row in selected],
            dtype=np.uint16,
        ),
        manifest_indices=np.asarray(
            [int(row["manifest_index"]) for row in selected],
            dtype=np.int64,
        ),
        packed_indices=np.asarray(
            [int(row["packed_index"]) for row in selected],
            dtype=np.int64,
        ),
    )
    present_notes = {int(row["midi_note"]) for row in selected}
    payload: dict[str, object] = {
        "schema": 1,
        "records": len(selected),
        "missing_notes": [
            note for note in range(21, 110) if note not in present_notes
        ],
        "entries": [
            {
                "index": index,
                "sample_id": row["sample_id"],
                "canonical_preset_id": row["canonical_preset_id"],
                "midi_note": row["midi_note"],
                "category": row["category"],
                "source_name": row["source_name"],
            }
            for index, row in enumerate(selected)
        ],
    }
    _atomic_json(packed_root / "seed-bank.json", payload)
    return payload


def finalize_flow_pack(config: ZraveFlowConfig) -> dict[str, object]:
    cache_root = Path(config.data.cache_root)
    metadata_paths = sorted(cache_root.glob("rank-*/part-*.json"))
    if not metadata_paths:
        raise ValueError(f"no encoded parts under {cache_root}")
    descriptors: list[dict[str, Any]] = []
    for path in metadata_paths:
        descriptors.extend(_read_part_metadata(path))
    descriptors.sort(key=lambda row: int(row["manifest_index"]))
    manifest_indices = [int(row["manifest_index"]) for row in descriptors]
    if len(manifest_indices) != len(set(manifest_indices)):
        raise ValueError("duplicate manifest indices across encoded parts")

    packed_root = Path(config.data.packed_root)
    packed_root.mkdir(parents=True, exist_ok=True)
    latent_moments = _Moments(config.model.latent_dim)
    delta_moments = _Moments(config.model.latent_dim)
    norm_samples: list[np.ndarray] = []
    sequences: list[dict[str, object]] = []
    shard_hashes: dict[str, str] = {}
    shard_sizes: list[int] = []

    for shard_index, start in enumerate(
        range(0, len(descriptors), config.data.shard_records)
    ):
        chunk = descriptors[start : start + config.data.shard_records]
        source_parts: dict[str, dict[str, np.ndarray]] = {}
        for part_path in sorted({str(row["part_npz"]) for row in chunk}):
            with np.load(part_path, allow_pickle=False) as part:
                source_parts[part_path] = {
                    key: part[key].copy() for key in part.files
                }
        maximum_length = max(int(row["length"]) for row in chunk)
        count = len(chunk)
        latents = np.zeros(
            (count, maximum_length, config.model.latent_dim),
            dtype=np.float16,
        )
        for row_index, descriptor in enumerate(chunk):
            part = source_parts[str(descriptor["part_npz"])]
            local_index = int(descriptor["local_index"])
            length = int(descriptor["length"])
            sequence = part["latents"][local_index, :length].astype(
                np.float32
            )
            latents[row_index, :length] = sequence.astype(np.float16)
            active = int(descriptor["active_frames"])
            if descriptor["split"] == "train":
                valid = sequence[:active]
                latent_moments.update(valid)
                delta_moments.update(np.diff(valid, axis=0))
                norms = np.linalg.norm(valid, axis=-1)
                if len(norms) > 8:
                    positions = np.linspace(
                        0,
                        len(norms) - 1,
                        num=8,
                        dtype=np.int64,
                    )
                    norms = norms[positions]
                norm_samples.append(norms.astype(np.float32))
            sequence_row = {
                key: descriptor[key]
                for key in (
                    "manifest_index",
                    "sample_id",
                    "source_name",
                    "source_code",
                    "category",
                    "category_code",
                    "canonical_preset_id",
                    "split",
                    "split_code",
                    "midi_note",
                    "velocity",
                    "articulation_id",
                    "maximum_future_frames",
                    "length",
                    "active_frames",
                )
            }
            sequence_row.update(
                {
                    "packed_index": start + row_index,
                    "shard": f"shard-{shard_index:06d}.npz",
                    "shard_row": row_index,
                }
            )
            sequences.append(sequence_row)
        output_path = packed_root / f"shard-{shard_index:06d}.npz"
        _atomic_npz(
            output_path,
            latents=latents,
            lengths=np.asarray(
                [int(row["length"]) for row in chunk],
                dtype=np.int16,
            ),
            active_frames=np.asarray(
                [int(row["active_frames"]) for row in chunk],
                dtype=np.int16,
            ),
            notes=np.asarray(
                [int(row["midi_note"]) for row in chunk],
                dtype=np.int16,
            ),
            velocities=np.asarray(
                [int(row["velocity"]) for row in chunk],
                dtype=np.int16,
            ),
            split_codes=np.asarray(
                [int(row["split_code"]) for row in chunk],
                dtype=np.uint8,
            ),
            source_codes=np.asarray(
                [int(row["source_code"]) for row in chunk],
                dtype=np.uint8,
            ),
            category_codes=np.asarray(
                [int(row["category_code"]) for row in chunk],
                dtype=np.uint16,
            ),
            maximum_future_frames=np.asarray(
                [int(row["maximum_future_frames"]) for row in chunk],
                dtype=np.uint8,
            ),
            manifest_indices=np.asarray(
                [int(row["manifest_index"]) for row in chunk],
                dtype=np.int64,
            ),
        )
        shard_hashes[output_path.name] = _sha256_file(output_path)
        shard_sizes.append(count)

    mean, latent_std = latent_moments.finish()
    _delta_mean, delta_std = delta_moments.finish()
    sampled_norms = np.concatenate(norm_samples)
    latent_norm_p01, latent_norm_p99 = np.percentile(
        sampled_norms,
        [1.0, 99.0],
    )
    statistics_path = packed_root / "statistics.npz"
    source_vocab = [source.name for source in config.data.sources]
    category_vocab = sorted({str(row["category"]) for row in sequences})
    _atomic_npz(
        statistics_path,
        mean=mean,
        latent_std=latent_std,
        delta_std=delta_std,
        latent_norm_p01=np.asarray(latent_norm_p01, dtype=np.float32),
        latent_norm_p99=np.asarray(latent_norm_p99, dtype=np.float32),
        source_vocab=np.asarray(source_vocab),
        category_vocab=np.asarray(category_vocab),
        rave_checkpoint_sha256=np.asarray(
            config.rave.expected_sha256
        ),
        manifest_sha256=np.asarray(
            _sha256_file(config.data.unified_manifest)
        ),
        norm_samples_per_sequence=np.asarray(8, dtype=np.int16),
    )
    _atomic_jsonl(packed_root / "sequences.jsonl", sequences)
    pitch_pairs = _build_pitch_pairs(sequences)
    pitch_pairs_path = packed_root / "pitch-pairs.npy"
    _atomic_npy(pitch_pairs_path, pitch_pairs)
    seed_report = _write_seed_bank(packed_root, sequences)

    index: dict[str, object] = {
        "schema": 1,
        "records": len(sequences),
        "shard_records": shard_sizes,
        "latent_dim": config.model.latent_dim,
        "latent_hop": config.rave.latent_hop,
        "sample_rate": config.rave.sample_rate,
        "rave_checkpoint_sha256": config.rave.expected_sha256,
        "manifest_sha256": _sha256_file(config.data.unified_manifest),
        "config_sha256": _sha256_json(config.as_dict()),
        "statistics_sha256": _sha256_file(statistics_path),
        "pitch_pairs_sha256": _sha256_file(pitch_pairs_path),
        "seed_bank_npz_sha256": _sha256_file(
            packed_root / "seed-bank.npz"
        ),
        "seed_bank_json_sha256": _sha256_file(
            packed_root / "seed-bank.json"
        ),
        "source_vocab": source_vocab,
        "category_vocab": category_vocab,
        "shard_sha256": shard_hashes,
        "seed_bank_missing_notes": seed_report["missing_notes"],
    }
    index_path = packed_root / "index.json"
    _atomic_json(index_path, index)
    return {
        "records": len(sequences),
        "shards": len(shard_sizes),
        "shard_records": shard_sizes,
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare stochastic Z-RAVE flow latent packs."
    )
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    encode = subparsers.add_parser("encode")
    encode.add_argument(
        "--rank",
        type=int,
        default=int(os.environ.get("RANK", "0")),
    )
    encode.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
    )
    encode.add_argument("--device", default="cuda")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ZraveFlowConfig.load(args.config)
    if args.command == "manifest":
        report: object = build_flow_manifest(config).__dict__
    elif args.command == "encode":
        report = encode_flow_shards(
            config,
            args.rank,
            args.world_size,
            args.device,
        )
    elif args.command == "finalize":
        report = finalize_flow_pack(config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
