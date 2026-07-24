from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .zrave_config import ZraveConfig


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL file: {source}")
    return rows


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_codec_hash(config: ZraveConfig) -> str:
    checkpoint = Path(config.rave.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"missing standalone RAVE checkpoint: {checkpoint}"
        )
    actual = _sha256_file(checkpoint)
    expected = config.rave.expected_sha256
    if expected is not None and actual != expected:
        raise ValueError(
            f"standalone RAVE checkpoint SHA-256 mismatch: "
            f"{actual} != {expected}"
        )
    return actual


def _stable_key(seed: int, *values: object) -> bytes:
    payload = ":".join([str(seed), "zrave-balanced50", *map(str, values)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _preset_id(row: dict[str, Any]) -> str:
    value = (
        row.get("preset_id")
        or row.get("timbre_id")
        or row.get("instrument_id")
    )
    if not value:
        raise ValueError("row is missing preset_id/timbre_id/instrument_id")
    return str(value)


def _category_lookup(
    preset_rows: Iterable[dict[str, Any]],
    categories: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    canonical = {category.casefold(): category for category in categories}
    result: dict[str, dict[str, str]] = {}
    for row in preset_rows:
        preset_id = _preset_id(row)
        category_raw = str(row.get("category") or "").strip()
        category = canonical.get(category_raw.casefold())
        if category is None:
            continue
        if preset_id in result:
            raise ValueError(f"duplicate preset metadata: {preset_id}")
        result[preset_id] = {
            "category": category,
            "bank": str(row.get("bank") or preset_id),
        }
    return result


def _diverse_order(
    candidates: list[str],
    banks: dict[str, str],
    seed: int,
    category: str,
) -> list[str]:
    by_bank: dict[str, list[str]] = defaultdict(list)
    for preset_id in candidates:
        by_bank[banks[preset_id]].append(preset_id)
    for bank, values in by_bank.items():
        values.sort(key=lambda value: _stable_key(seed, category, bank, value))
    bank_order = sorted(
        by_bank,
        key=lambda bank: _stable_key(seed, category, "bank", bank),
    )
    ordered: list[str] = []
    round_index = 0
    while len(ordered) < len(candidates):
        added = False
        for bank in bank_order:
            values = by_bank[bank]
            if round_index < len(values):
                ordered.append(values[round_index])
                added = True
        if not added:
            break
        round_index += 1
    return ordered


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _atomic_npz(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def latent_cache_path(config: ZraveConfig, sample_id: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return Path(config.data.cache_root) / f"{digest}.npz"


def _load_cache(
    path: Path,
    *,
    sample_id: str,
    latent_dim: int,
    hop: int,
    checkpoint_hash: str,
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as cached:
            latent = cached["latent"].astype(np.float32, copy=False)
            actual = {
                "sample_id": str(cached["sample_id"].item()),
                "hop": int(cached["hop"].item()),
                "checkpoint_hash": str(cached["checkpoint_hash"].item()),
            }
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"invalid latent cache {path}: {error}") from error
    expected = {
        "sample_id": sample_id,
        "hop": hop,
        "checkpoint_hash": checkpoint_hash,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise ValueError(
                f"latent cache {field} mismatch: "
                f"{actual[field]} != {expected_value}"
            )
    if latent.ndim != 2 or latent.shape[0] != latent_dim:
        raise ValueError(
            f"latent cache latent_dim mismatch: {latent.shape} "
            f"does not have {latent_dim} channels"
        )
    if not np.isfinite(latent).all():
        raise ValueError(f"latent cache contains non-finite values: {path}")
    return latent


class _ChannelMoments:
    def __init__(self, channels: int) -> None:
        self.count = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.square_total = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        if values64.ndim != 2 or values64.shape[1] != self.total.shape[0]:
            raise ValueError("statistics values must have shape [frames, channels]")
        self.count += values64.shape[0]
        self.total += values64.sum(axis=0)
        self.square_total += np.square(values64).sum(axis=0)

    def finalize(self, floor: float) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty training statistics")
        mean = self.total / self.count
        variance = np.maximum(
            0.0,
            self.square_total / self.count - np.square(mean),
        )
        standard_deviation = np.maximum(np.sqrt(variance), floor)
        return mean.astype(np.float32), standard_deviation.astype(np.float32)


@dataclass(frozen=True)
class PackedDatasetMetadata:
    records: int
    maximum_length: int
    latent_dim: int
    rave_checkpoint_sha256: str
    selected_manifest_sha256: str
    latents_sha256: str
    statistics_sha256: str
    index_path: str


def pack_cached_latents(config: ZraveConfig) -> PackedDatasetMetadata:
    rows = read_jsonl(config.data.selected_manifest)
    checkpoint_path = Path(config.rave.checkpoint)
    checkpoint_hash = _verified_codec_hash(config)
    split_codes = {"train": 0, "validation": 1, "test": 2}
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    splits: list[int] = []
    index_records: list[dict[str, object]] = []
    latent_moments = _ChannelMoments(config.model.latent_dim)
    delta_moments = _ChannelMoments(config.model.latent_dim)
    acceleration_moments = _ChannelMoments(config.model.latent_dim)

    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"selected manifest row {row_index} lacks sample_id")
        split = str(row.get("split") or "")
        if split not in split_codes:
            raise ValueError(
                f"selected manifest row {row_index} has invalid split {split!r}"
            )
        latent = _load_cache(
            latent_cache_path(config, sample_id),
            sample_id=sample_id,
            latent_dim=config.model.latent_dim,
            hop=config.data.latent_hop,
            checkpoint_hash=checkpoint_hash,
        )
        if latent.shape[1] <= config.data.warmup_frames:
            raise ValueError(f"latent sequence is shorter than warmup: {sample_id}")
        sequence = latent[:, config.data.warmup_frames :].T.copy()
        minimum = (
            config.model.context_frames + config.model.horizon_frames
        )
        if sequence.shape[0] <= minimum:
            raise ValueError(
                f"latent sequence is too short after warmup: {sample_id}: "
                f"{sequence.shape[0]} <= {minimum}"
            )
        sequences.append(sequence)
        lengths.append(sequence.shape[0])
        splits.append(split_codes[split])
        index_records.append(
            {
                "index": row_index,
                "sample_id": sample_id,
                "preset_id": _preset_id(row),
                "category": row.get("zrave_category"),
                "bank": row.get("zrave_bank"),
                "split": split,
                "length": sequence.shape[0],
                "cache_sha256": _sha256_file(
                    latent_cache_path(config, sample_id)
                ),
            }
        )
        if split == "train":
            latent_moments.update(sequence)
            delta_moments.update(np.diff(sequence, axis=0))
            acceleration_moments.update(np.diff(sequence, n=2, axis=0))

    maximum_length = max(lengths)
    dense = np.zeros(
        (len(sequences), maximum_length, config.model.latent_dim),
        dtype=np.float16,
    )
    for index, sequence in enumerate(sequences):
        dense[index, : sequence.shape[0]] = sequence.astype(
            np.float16,
            copy=False,
        )

    pack_root = Path(config.data.packed_root)
    pack_root.mkdir(parents=True, exist_ok=True)
    latents_path = pack_root / "latents.npy"
    lengths_path = pack_root / "lengths.npy"
    splits_path = pack_root / "splits.npy"
    statistics_path = pack_root / "statistics.npz"
    _atomic_npy(latents_path, dense)
    _atomic_npy(lengths_path, np.asarray(lengths, dtype=np.int32))
    _atomic_npy(splits_path, np.asarray(splits, dtype=np.uint8))

    mean, latent_std = latent_moments.finalize(
        config.loss.statistic_floor
    )
    _, delta_std = delta_moments.finalize(config.loss.statistic_floor)
    _, acceleration_std = acceleration_moments.finalize(
        config.loss.statistic_floor
    )
    _atomic_npz(
        statistics_path,
        mean=mean,
        latent_std=latent_std,
        delta_std=delta_std,
        acceleration_std=acceleration_std,
        rave_checkpoint_sha256=np.asarray(checkpoint_hash),
        selected_manifest_sha256=np.asarray(
            _sha256_file(config.data.selected_manifest)
        ),
        latent_hop=np.asarray(config.data.latent_hop, dtype=np.int64),
        latent_dim=np.asarray(config.model.latent_dim, dtype=np.int64),
        warmup_frames=np.asarray(config.data.warmup_frames, dtype=np.int64),
    )
    index_path = pack_root / "index.json"
    index_payload = {
        "schema": 1,
        "records": len(rows),
        "maximum_length": maximum_length,
        "latent_dim": config.model.latent_dim,
        "latent_hop": config.data.latent_hop,
        "warmup_frames": config.data.warmup_frames,
        "rave_checkpoint": str(checkpoint_path.resolve()),
        "rave_checkpoint_sha256": checkpoint_hash,
        "rave_codec": "standalone_torchscript_rave",
        "rave_sample_rate": config.rave.sample_rate,
        "selected_manifest": str(
            Path(config.data.selected_manifest).resolve()
        ),
        "selected_manifest_sha256": _sha256_file(
            config.data.selected_manifest
        ),
        "latents_sha256": _sha256_file(latents_path),
        "lengths_sha256": _sha256_file(lengths_path),
        "splits_sha256": _sha256_file(splits_path),
        "statistics_sha256": _sha256_file(statistics_path),
        "split_records": {
            name: splits.count(code)
            for name, code in split_codes.items()
        },
        "sequences": index_records,
    }
    _atomic_json(index_path, index_payload)
    return PackedDatasetMetadata(
        records=len(rows),
        maximum_length=maximum_length,
        latent_dim=config.model.latent_dim,
        rave_checkpoint_sha256=checkpoint_hash,
        selected_manifest_sha256=index_payload[
            "selected_manifest_sha256"
        ],
        latents_sha256=index_payload["latents_sha256"],
        statistics_sha256=index_payload["statistics_sha256"],
        index_path=str(index_path.resolve()),
    )


def _cache_is_valid(
    config: ZraveConfig,
    sample_id: str,
    checkpoint_hash: str,
) -> bool:
    path = latent_cache_path(config, sample_id)
    if not path.is_file():
        return False
    try:
        _load_cache(
            path,
            sample_id=sample_id,
            latent_dim=config.model.latent_dim,
            hop=config.data.latent_hop,
            checkpoint_hash=checkpoint_hash,
        )
    except ValueError:
        return False
    return True


def cache_selected_latents(
    config: ZraveConfig,
    rank: int,
    world_size: int,
    device: str,
) -> dict[str, object]:
    import torch

    from .data import load_audio
    from .latent_cache import save_latent_cache

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    checkpoint_path = Path(config.rave.checkpoint)
    checkpoint_hash = _verified_codec_hash(config)
    codec = torch.jit.load(
        str(checkpoint_path),
        map_location=device,
    ).to(device).eval()
    sample_rate_value = codec.sr
    if hasattr(sample_rate_value, "__len__"):
        sample_rate_value = sample_rate_value[0]
    sample_rate = int(sample_rate_value)
    latent_dim = int(codec.latent_size)
    if sample_rate != config.rave.sample_rate:
        raise ValueError(
            f"standalone RAVE sample rate mismatch: "
            f"{sample_rate} != {config.rave.sample_rate}"
        )
    if latent_dim != config.model.latent_dim:
        raise ValueError(
            f"standalone RAVE latent dimension mismatch: "
            f"{latent_dim} != {config.model.latent_dim}"
        )

    probe_frames = 4
    probe_audio = torch.zeros(
        1,
        1,
        probe_frames * config.data.latent_hop,
        device=device,
    )
    with torch.inference_mode():
        probe_latent = codec.encode(probe_audio)
        probe_decoded = codec.decode(probe_latent)
    expected_latent_shape = (
        1,
        config.model.latent_dim,
        probe_frames,
    )
    if tuple(probe_latent.shape) != expected_latent_shape:
        raise ValueError(
            "standalone RAVE latent hop/layout mismatch: "
            f"{tuple(probe_latent.shape)} != {expected_latent_shape}"
        )
    if (
        probe_decoded.ndim != 3
        or probe_decoded.shape[0] != 1
        or probe_decoded.shape[1] != 1
        or probe_decoded.shape[-1]
        != probe_frames * config.data.latent_hop
    ):
        raise ValueError(
            "standalone RAVE decoder output does not match the latent hop"
        )
    if not torch.isfinite(probe_latent).all() or not torch.isfinite(
        probe_decoded
    ).all():
        raise ValueError("standalone RAVE codec probe produced non-finite data")

    rows = read_jsonl(config.data.selected_manifest)
    selected = rows[rank::world_size]
    cached = 0
    existing = 0
    with torch.inference_mode():
        for position, row in enumerate(selected, 1):
            sample_id = str(row["sample_id"])
            if _cache_is_valid(config, sample_id, checkpoint_hash):
                existing += 1
                continue
            audio_path = (
                Path(config.data.audio_root) / str(row["audio_path"])
            ).resolve()
            audio = load_audio(audio_path, sample_rate)
            usable = len(audio) - len(audio) % config.data.latent_hop
            if usable < config.data.latent_hop:
                raise ValueError(f"audio is too short: {sample_id}")
            tensor = torch.from_numpy(audio[:usable]).view(1, 1, -1).to(
                device
            )
            latent_tensor = codec.encode(tensor)
            if (
                latent_tensor.ndim != 3
                or latent_tensor.shape[0] != 1
                or latent_tensor.shape[1] != config.model.latent_dim
            ):
                raise ValueError(
                    f"standalone RAVE produced invalid latent shape: "
                    f"{tuple(latent_tensor.shape)}"
                )
            latent = latent_tensor[0].float().cpu().numpy()
            save_latent_cache(
                latent_cache_path(config, sample_id),
                latent,
                sample_id,
                config.data.latent_hop,
                checkpoint_hash,
            )
            cached += 1
            if position % 25 == 0 or position == len(selected):
                print(
                    json.dumps(
                        {
                            "event": "zrave_cache_progress",
                            "rank": rank,
                            "done": position,
                            "total": len(selected),
                            "cached": cached,
                            "existing": existing,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return {
        "rank": rank,
        "assigned": len(selected),
        "cached": cached,
        "existing": existing,
        "checkpoint_hash": checkpoint_hash,
        "codec": "standalone_torchscript_rave",
        "sample_rate": sample_rate,
        "latent_hop": config.data.latent_hop,
    }


def select_balanced_manifest(config: ZraveConfig) -> dict[str, object]:
    data = config.data
    preset_rows = read_jsonl(data.preset_metadata)
    eligible_rows = read_jsonl(data.eligible_manifest)
    preset_info = _category_lookup(preset_rows, data.categories)

    rows_by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        preset_id = _preset_id(row)
        if preset_id in preset_info:
            rows_by_preset[preset_id].append(row)

    complete: dict[str, list[dict[str, Any]]] = {}
    for preset_id, rows in rows_by_preset.items():
        conditions: set[tuple[int, int]] = set()
        sample_ids: set[str] = set()
        valid = True
        for row in rows:
            try:
                condition = (int(row["midi_note"]), int(row["velocity"]))
                sample_id = str(row["sample_id"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            conditions.add(condition)
            sample_ids.add(sample_id)
        if (
            valid
            and len(rows) == data.conditions_per_preset
            and len(sample_ids) == data.conditions_per_preset
            and len(conditions) == data.conditions_per_preset
        ):
            complete[preset_id] = rows

    category_candidates: dict[str, list[str]] = {
        category: [] for category in data.categories
    }
    banks: dict[str, str] = {}
    for preset_id in complete:
        info = preset_info[preset_id]
        category_candidates[info["category"]].append(preset_id)
        banks[preset_id] = info["bank"]

    selected_by_category: dict[str, list[str]] = {}
    for category in data.categories:
        ordered = _diverse_order(
            category_candidates[category],
            banks,
            config.seed,
            category,
        )
        if len(ordered) < data.presets_per_category:
            raise ValueError(
                f"category {category} has {len(ordered)} complete presets; "
                f"need {data.presets_per_category}"
            )
        selected_by_category[category] = ordered[: data.presets_per_category]

    split_by_preset: dict[str, str] = {}
    selected_metadata: dict[str, list[dict[str, str]]] = {}
    for category in data.categories:
        selected_metadata[category] = []
        for index, preset_id in enumerate(selected_by_category[category]):
            split = (
                "train"
                if index < 8
                else "validation"
                if index == 8
                else "test"
            )
            split_by_preset[preset_id] = split
            selected_metadata[category].append(
                {
                    "preset_id": preset_id,
                    "bank": banks[preset_id],
                    "split": split,
                }
            )

    selected_rows: list[dict[str, Any]] = []
    for category in data.categories:
        for preset_id in selected_by_category[category]:
            rows = sorted(
                complete[preset_id],
                key=lambda row: (
                    int(row["midi_note"]),
                    int(row["velocity"]),
                    str(row["sample_id"]),
                ),
            )
            for source in rows:
                row = dict(source)
                row["preset_id"] = preset_id
                row["split"] = split_by_preset[preset_id]
                row["zrave_category"] = category
                row["zrave_bank"] = banks[preset_id]
                selected_rows.append(row)

    output_manifest = Path(data.selected_manifest)
    output_metadata = Path(data.metadata_output)
    _atomic_jsonl(output_manifest, selected_rows)

    split_samples = Counter(str(row["split"]) for row in selected_rows)
    split_presets = Counter(split_by_preset.values())
    category_counts = {
        category: len(selected_by_category[category])
        for category in sorted(data.categories)
    }
    duration_seconds = sum(
        float(row.get("duration_seconds") or 0.0)
        for row in selected_rows
    )
    report: dict[str, object] = {
        "presets": len(split_by_preset),
        "samples": len(selected_rows),
        "categories": category_counts,
        "splits": {
            split: split_presets[split]
            for split in ("train", "validation", "test")
        },
        "duration_hours": duration_seconds / 3600.0,
    }
    metadata = {
        "schema": 1,
        "profile": "serum-balanced50-zrave",
        "seed": config.seed,
        **report,
        "preset_metadata": str(Path(data.preset_metadata).resolve()),
        "preset_metadata_sha256": _sha256_file(data.preset_metadata),
        "eligible_manifest": str(Path(data.eligible_manifest).resolve()),
        "eligible_manifest_sha256": _sha256_file(data.eligible_manifest),
        "selected_manifest": str(output_manifest.resolve()),
        "selected_manifest_sha256": _sha256_file(output_manifest),
        "selected_presets": selected_metadata,
        "split_samples": {
            split: split_samples[split]
            for split in ("train", "validation", "test")
        },
        "completeness_rule": {
            "unique_sample_ids": data.conditions_per_preset,
            "unique_midi_note_velocity_pairs": data.conditions_per_preset,
            "rows": data.conditions_per_preset,
        },
        "warmup_frames": data.warmup_frames,
        "latent_hop": data.latent_hop,
    }
    _atomic_json(output_metadata, metadata)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the balanced Z-RAVE latent dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", required=True)
    cache = subparsers.add_parser("cache-and-pack")
    cache.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ZraveConfig.load(args.config)
    if args.command == "select":
        print(
            json.dumps(
                select_balanced_manifest(config),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "cache-and-pack":
        import torch
        from torch import distributed as distributed

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available():
            raise RuntimeError("RAVE latent caching requires CUDA")
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            distributed.init_process_group("nccl")
        cache_report = cache_selected_latents(
            config,
            rank,
            world_size,
            f"cuda:{local_rank}",
        )
        print(json.dumps(cache_report, sort_keys=True), flush=True)
        if world_size > 1:
            distributed.barrier()
        if rank == 0:
            print(
                json.dumps(
                    pack_cached_latents(config).__dict__,
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        if world_size > 1:
            distributed.barrier()
            distributed.destroy_process_group()
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
