from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from .data import _atomic_save_npy, _valid_clap_cache, sha256_file
from .losses import FrozenClapReconstructionObjective
from .zrave_flow_gate import load_audition_triplets


_EXPECTED_IMPLEMENTATION = "laion-clap-1.1.7/HTSAT-base/no-fusion"
_RUNTIME_FINGERPRINT_DEPENDENCIES = (
    "torch",
    "torchaudio",
    "torchvision",
    "laion-clap",
    "transformers",
    "torchlibrosa",
    "librosa",
    "ftfy",
)
_EXPECTED_CONFIG_KEYS = {
    "schema",
    "implementation",
    "package",
    "package_version",
    "transformers_version",
    "torchlibrosa_version",
    "librosa_version",
    "ftfy_version",
    "audio_model",
    "enable_fusion",
    "checkpoint_filename",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "clap_sample_rate",
    "clap_input_samples",
    "data_truncating",
    "data_filling",
    "crop_seed",
    "import_time_hf_assets",
    "embedding_dimension",
    "embedding_normalization",
    "input_scope",
    "comparison",
    "caption_evaluation",
    "batch_size",
}


@dataclass(frozen=True)
class ClapMonitorConfig:
    schema: int
    implementation: str
    package: str
    package_version: str
    transformers_version: str
    torchlibrosa_version: str
    librosa_version: str
    ftfy_version: str
    audio_model: str
    enable_fusion: bool
    checkpoint_filename: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    clap_sample_rate: int
    clap_input_samples: int
    data_truncating: str
    data_filling: str
    crop_seed: str
    import_time_hf_assets: list[str]
    embedding_dimension: int
    embedding_normalization: str
    input_scope: str
    comparison: str
    caption_evaluation: bool
    batch_size: int

    @classmethod
    def load(cls, path: str | Path) -> ClapMonitorConfig:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("CLAP monitor config must be a JSON object")
        unknown = set(payload) - _EXPECTED_CONFIG_KEYS
        missing = _EXPECTED_CONFIG_KEYS - set(payload)
        if unknown or missing:
            raise ValueError(
                "CLAP monitor config keys mismatch: "
                f"missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        expected: dict[str, object] = {
            "schema": 1,
            "implementation": _EXPECTED_IMPLEMENTATION,
            "package": "laion-clap",
            "package_version": "1.1.7",
            "transformers_version": "5.13.0",
            "torchlibrosa_version": "0.1.0",
            "librosa_version": "0.11.0",
            "ftfy_version": "6.3.1",
            "audio_model": "HTSAT-base",
            "enable_fusion": False,
            "checkpoint_filename": "music_audioset_epoch_15_esc_90.14.pt",
            "checkpoint_bytes": 2_352_471_003,
            "checkpoint_sha256": (
                "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
            ),
            "clap_sample_rate": FrozenClapReconstructionObjective.CLAP_SAMPLE_RATE,
            "clap_input_samples": 480_000,
            "data_truncating": "rand_trunc",
            "data_filling": "repeatpad",
            "crop_seed": (
                "first_63_bits_of_source_audio_sha256/"
                "python_numpy_torch_cuda_scoped_restore"
            ),
            "import_time_hf_assets": [
                "bert-base-uncased-tokenizer",
                "roberta-base-tokenizer-and-model",
                "facebook/bart-base-tokenizer",
            ],
            "embedding_dimension": 512,
            "embedding_normalization": "l2",
            "input_scope": "complete_mono_wav_passed_to_clap_preprocessor",
            "comparison": "audio_audio_cosine",
            "caption_evaluation": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"CLAP monitor {name} must be {value!r}")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("CLAP monitor batch_size must be positive")
        if self.batch_size != 1:
            raise ValueError(
                "CLAP monitor batch_size must be 1 so each audio hash "
                "deterministically seeds its rand_trunc preprocessing"
            )


class _EmbeddingObjective(Protocol):
    def _embedding(
        self,
        audio: Tensor,
        valid_samples: Tensor,
        role: str = "generated",
    ) -> Tensor: ...


@dataclass(frozen=True)
class _AudioRecord:
    path: Path
    sample_rate: int
    samples: int
    sha256: str


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_audio_record(path: Path) -> tuple[_AudioRecord, np.ndarray]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audio, sample_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    if audio.shape[0] == 0 or audio.shape[1] != 1:
        raise ValueError(f"CLAP monitor requires non-empty mono audio: {source}")
    waveform = np.ascontiguousarray(audio[:, 0])
    if not np.isfinite(waveform).all():
        raise ValueError(f"CLAP monitor audio is non-finite: {source}")
    return (
        _AudioRecord(
            path=source,
            sample_rate=int(sample_rate),
            samples=int(waveform.size),
            sha256=sha256_file(source),
        ),
        waveform,
    )


def _cache_paths(cache_root: Path, audio_sha256: str) -> tuple[Path, Path]:
    root = cache_root / "clap"
    return root / f"{audio_sha256}.npy", root / f"{audio_sha256}.json"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_fingerprint(
    dependencies: Mapping[str, str],
    device: torch.device,
) -> dict[str, object]:
    versions: dict[str, str] = {}
    for name in _RUNTIME_FINGERPRINT_DEPENDENCIES:
        version = dependencies.get(name)
        if not isinstance(version, str) or not version:
            raise ValueError(
                f"CLAP runtime fingerprint lacks dependency version: {name}"
            )
        versions[name] = version
    execution: dict[str, object] = {
        "device": str(device),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": None,
        "cuda_device_capability": None,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CLAP CUDA runtime fingerprint requires CUDA")
        execution["cuda_device_name"] = torch.cuda.get_device_name(device)
        execution["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    payload: dict[str, object] = {
        "schema": 1,
        "dependencies": versions,
        "monitor_code_sha256": sha256_file(Path(__file__).resolve()),
        "execution": execution,
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _cache_contract(
    *,
    audio: _AudioRecord,
    config: ClapMonitorConfig,
    config_sha256: str,
    checkpoint_sha256: str,
    embedding_sha256: str,
    runtime_fingerprint: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "kind": "midibrave-clap-audio-embedding-cache",
        "implementation": config.implementation,
        "dimension": config.embedding_dimension,
        "normalization": config.embedding_normalization,
        "source_audio_sha256": audio.sha256,
        "source_sample_rate": audio.sample_rate,
        "source_samples": audio.samples,
        "preprocessing_crop_seed": (int(audio.sha256[:16], 16) & ((1 << 63) - 1)),
        "preprocessing_rngs": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "preprocessing_rng_scope": "save_seed_restore",
        "clap_checkpoint_sha256": checkpoint_sha256,
        "clap_monitor_config_sha256": config_sha256,
        "runtime_fingerprint": dict(runtime_fingerprint),
        "embedding_sha256": embedding_sha256,
    }


def _load_cached_embedding(
    cache_root: Path,
    audio: _AudioRecord,
    config: ClapMonitorConfig,
    config_sha256: str,
    checkpoint_sha256: str,
    runtime_fingerprint: Mapping[str, object],
) -> np.ndarray | None:
    embedding_path, metadata_path = _cache_paths(cache_root, audio.sha256)
    if not _valid_clap_cache(embedding_path, config.embedding_dimension):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedding = np.load(embedding_path, allow_pickle=False).astype(
            np.float32,
            copy=False,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    expected = _cache_contract(
        audio=audio,
        config=config,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        embedding_sha256=sha256_file(embedding_path),
        runtime_fingerprint=runtime_fingerprint,
    )
    if metadata != expected:
        return None
    norm = float(np.linalg.norm(embedding.astype(np.float64)))
    if not math.isfinite(norm) or not 0.999 <= norm <= 1.001:
        return None
    return embedding


def _write_cached_embedding(
    cache_root: Path,
    audio: _AudioRecord,
    embedding: np.ndarray,
    config: ClapMonitorConfig,
    config_sha256: str,
    checkpoint_sha256: str,
    runtime_fingerprint: Mapping[str, object],
) -> tuple[Path, str]:
    value = np.asarray(embedding, dtype=np.float32)
    if value.shape != (config.embedding_dimension,) or not np.isfinite(value).all():
        raise ValueError("CLAP embedding must be finite and match configured dimension")
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not 0.999 <= norm <= 1.001:
        raise ValueError("CLAP embedding must be L2 normalized")
    embedding_path, metadata_path = _cache_paths(cache_root, audio.sha256)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(embedding_path, value)
    embedding_sha256 = sha256_file(embedding_path)
    metadata = _cache_contract(
        audio=audio,
        config=config,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        embedding_sha256=embedding_sha256,
        runtime_fingerprint=runtime_fingerprint,
    )
    _atomic_json(metadata_path, metadata)
    return embedding_path, embedding_sha256


def _objective_embeddings(
    objective: _EmbeddingObjective,
    waveforms: Sequence[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    maximum = max(waveform.size for waveform in waveforms)
    batch = torch.zeros(
        (len(waveforms), 1, maximum),
        dtype=torch.float32,
        device=device,
    )
    valid = torch.empty(len(waveforms), dtype=torch.long, device=device)
    for index, waveform in enumerate(waveforms):
        length = int(waveform.size)
        batch[index, 0, :length] = torch.from_numpy(waveform).to(device)
        valid[index] = length
    with torch.no_grad():
        embedding = objective._embedding(batch, valid, role="target")
    value = embedding.detach().float().cpu().numpy().astype(np.float32)
    if value.ndim != 2 or value.shape[0] != len(waveforms):
        raise ValueError("frozen CLAP objective returned an invalid embedding batch")
    if not np.isfinite(value).all():
        raise ValueError("frozen CLAP objective returned non-finite embeddings")
    return value


@contextmanager
def _scoped_embedding_rng(seed: int) -> Iterator[None]:
    """Seed CLAP preprocessing without mutating the caller's RNG streams."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed % (1 << 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 1.0e-12:
        raise ValueError("CLAP cosine requires non-zero embeddings")
    value = float(np.dot(left64, right64) / denominator)
    if not math.isfinite(value):
        raise ValueError("CLAP cosine is non-finite")
    return min(1.0, max(-1.0, value))


def _cosine_summary(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> dict[str, float | int | str | None]:
    values: list[tuple[float, str]] = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append((float(value), str(row.get("id"))))
    if not values:
        return {
            "count": 0,
            "p10": None,
            "median": None,
            "p90": None,
            "worst": None,
            "worst_id": None,
        }
    array = np.asarray([value for value, _identifier in values], dtype=np.float64)
    worst_value, worst_id = min(values)
    return {
        "count": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "worst": worst_value,
        "worst_id": worst_id,
    }


def _unique_codec_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    unique: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in sorted(rows, key=lambda value: str(value.get("id"))):
        audio = row.get("audio_sha256")
        if not isinstance(audio, Mapping):
            continue
        source = audio.get("source")
        direct = audio.get("direct")
        if isinstance(source, str) and isinstance(direct, str):
            unique.setdefault((source, direct), row)
    return list(unique.values())


def _runtime_dependencies() -> dict[str, str]:
    packages = (
        "laion-clap",
        "transformers",
        "torchlibrosa",
        "librosa",
        "ftfy",
        "torchaudio",
        "torchvision",
    )
    versions: dict[str, str] = {"torch": torch.__version__}
    missing: list[str] = []
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "CLAP monitor runtime dependencies are missing: " + ", ".join(missing)
        )
    if versions["laion-clap"] != "1.1.7":
        raise RuntimeError("CLAP monitor requires laion-clap 1.1.7")
    return versions


def evaluate_clap_audition(
    manifest_path: str | Path,
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    cache_root: str | Path,
    device: str | torch.device = "cuda",
    expected_checkpoint_sha256: str | None = None,
    objective: _EmbeddingObjective | None = None,
    runtime_package_version: str | None = None,
    runtime_dependencies: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Report frozen-CLAP audio/audio preservation without applying a gate."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    config = ClapMonitorConfig.load(config_file)
    config_sha256 = sha256_file(config_file)
    checkpoint_sha256 = sha256_file(checkpoint_file)
    if checkpoint_file.name != config.checkpoint_filename:
        raise ValueError("CLAP checkpoint filename does not match monitor contract")
    if checkpoint_file.stat().st_size != config.checkpoint_bytes:
        raise ValueError("CLAP checkpoint byte size does not match monitor contract")
    if checkpoint_sha256 != config.checkpoint_sha256:
        raise ValueError("CLAP checkpoint SHA-256 does not match monitor contract")
    if (
        expected_checkpoint_sha256 is not None
        and expected_checkpoint_sha256 != config.checkpoint_sha256
    ):
        raise ValueError("caller CLAP checkpoint SHA-256 mismatches config contract")
    if objective is None:
        dependency_versions = _runtime_dependencies()
        package_version = dependency_versions["laion-clap"]
        pinned_versions = {
            "laion-clap": config.package_version,
            "transformers": config.transformers_version,
            "torchlibrosa": config.torchlibrosa_version,
            "librosa": config.librosa_version,
            "ftfy": config.ftfy_version,
        }
        mismatches = {
            name: (dependency_versions.get(name), expected)
            for name, expected in pinned_versions.items()
            if dependency_versions.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(
                "CLAP monitor dependency version mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
    else:
        package_version = runtime_package_version or config.package_version
        dependency_versions = dict(
            runtime_dependencies
            or {"laion-clap": package_version, "torch": torch.__version__}
        )
    resolved_device = torch.device(device)
    runtime_fingerprint = _runtime_fingerprint(
        dependency_versions,
        resolved_device,
    )

    _manifest, triplets = load_audition_triplets(manifest_file)
    paths = sorted(
        {
            path
            for triplet in triplets
            for path in (triplet.source, triplet.direct, triplet.generated)
        },
        key=str,
    )
    records: dict[Path, _AudioRecord] = {}
    waveforms: dict[Path, np.ndarray] = {}
    for path in paths:
        record, waveform = _load_audio_record(path)
        records[path] = record
        waveforms[path] = waveform
    sample_rates = {record.sample_rate for record in records.values()}
    if len(sample_rates) != 1:
        raise ValueError("CLAP audition WAV files must share one sample rate")
    sample_rate = next(iter(sample_rates))

    embeddings: dict[str, np.ndarray] = {}
    embedding_hashes: dict[str, str] = {}
    cache_hits = 0
    missing_records: list[_AudioRecord] = []
    pending_hashes: set[str] = set()
    for path in paths:
        record = records[path]
        if record.sha256 in embeddings or record.sha256 in pending_hashes:
            continue
        cached = _load_cached_embedding(
            cache,
            record,
            config,
            config_sha256,
            checkpoint_sha256,
            runtime_fingerprint,
        )
        if cached is None:
            missing_records.append(record)
            pending_hashes.add(record.sha256)
        else:
            embeddings[record.sha256] = cached
            embedding_path, _metadata_path = _cache_paths(cache, record.sha256)
            embedding_hashes[record.sha256] = sha256_file(embedding_path)
            cache_hits += 1

    if missing_records and objective is None:
        objective = FrozenClapReconstructionObjective(
            str(checkpoint_file),
            sample_rate,
            resolved_device,
            maximum_gradient_norm=0.0,
        )
    cache_writes = 0
    assert objective is not None or not missing_records
    peak_cuda_allocated_bytes = 0
    if missing_records and resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    for start in range(0, len(missing_records), config.batch_size):
        batch_records = missing_records[start : start + config.batch_size]
        assert objective is not None
        crop_seed = int(batch_records[0].sha256[:16], 16) & ((1 << 63) - 1)
        with _scoped_embedding_rng(crop_seed):
            values = _objective_embeddings(
                objective,
                [waveforms[record.path] for record in batch_records],
                resolved_device,
            )
        if values.shape[1] != config.embedding_dimension:
            raise ValueError("frozen CLAP embedding dimension mismatch")
        for record, embedding in zip(batch_records, values, strict=True):
            embedding_path, embedding_sha256 = _write_cached_embedding(
                cache,
                record,
                embedding,
                config,
                config_sha256,
                checkpoint_sha256,
                runtime_fingerprint,
            )
            if not _valid_clap_cache(
                embedding_path,
                config.embedding_dimension,
            ):
                raise ValueError("written CLAP cache failed the shared contract")
            embeddings[record.sha256] = embedding
            embedding_hashes[record.sha256] = embedding_sha256
            cache_writes += 1
    if missing_records and resolved_device.type == "cuda":
        peak_cuda_allocated_bytes = int(
            torch.cuda.max_memory_allocated(resolved_device)
        )

    rows: list[dict[str, object]] = []
    for triplet in triplets:
        hashes = {
            role: records[path].sha256
            for role, path in (
                ("source", triplet.source),
                ("direct", triplet.direct),
                ("generated", triplet.generated),
            )
        }
        audio_paths = {
            role: str(path.relative_to(manifest_file.parent))
            for role, path in (
                ("source", triplet.source),
                ("direct", triplet.direct),
                ("generated", triplet.generated),
            )
        }
        source = embeddings[hashes["source"]]
        direct = embeddings[hashes["direct"]]
        generated = embeddings[hashes["generated"]]
        rows.append(
            {
                "id": triplet.identifier,
                "group_id": triplet.group_id,
                "generation_seed": triplet.generation_seed,
                "category": triplet.category,
                "audio_paths": audio_paths,
                "audio_sha256": hashes,
                "embedding_sha256": {
                    role: embedding_hashes[digest] for role, digest in hashes.items()
                },
                "generated_vs_source_audio_cosine": _cosine(
                    generated,
                    source,
                ),
                "generated_vs_direct_audio_cosine": _cosine(
                    generated,
                    direct,
                ),
                "direct_vs_source_codec_ceiling_audio_cosine": _cosine(
                    direct,
                    source,
                ),
            }
        )
    codec_rows = _unique_codec_rows(rows)
    metric_rows = {
        "generated_vs_source_audio_cosine": rows,
        "generated_vs_direct_audio_cosine": rows,
        "direct_vs_source_codec_ceiling_audio_cosine": codec_rows,
    }
    collection = hashlib.sha256()
    for audio_sha256 in sorted(embeddings):
        collection.update(audio_sha256.encode("ascii"))
        collection.update(b"\0")
        collection.update(embedding_hashes[audio_sha256].encode("ascii"))
        collection.update(b"\n")
    report: dict[str, object] = {
        "schema": 1,
        "kind": "zrave-frozen-clap-audio-preservation-report",
        "report_only": True,
        "hard_thresholds": None,
        "manifest": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "clap_contract": {
            **asdict(config),
            "runtime_package_version": package_version,
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(config_file),
            "config_sha256": config_sha256,
            "runtime_dependencies": dependency_versions,
            "runtime_fingerprint": runtime_fingerprint,
            "import_time_hf_assets": list(config.import_time_hf_assets),
            "text_embedding_used": False,
        },
        "cache": {
            "root": str(cache),
            "layout": "clap/<source_audio_sha256>.npy + .json",
            "shared_embedding_contract": "finite float32 shape (512,)",
            "unique_audio_files": len(embeddings),
            "hits": cache_hits,
            "writes": cache_writes,
            "used_embedding_collection_sha256": collection.hexdigest(),
            "runtime_fingerprint_sha256": runtime_fingerprint["sha256"],
        },
        "runtime": {
            "device": str(resolved_device),
            "embedding_batch_size": config.batch_size,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
            "peak_cuda_allocated_gib": (peak_cuda_allocated_bytes / 2**30),
        },
        "claims": {
            "audio_audio_similarity_only": True,
            "text_semantic_accuracy_evaluated": False,
            "timbre_identity_ground_truth": False,
            "note": (
                "No caption/text embedding is evaluated. Audio/audio CLAP "
                "cosine is a report-only preservation proxy, not proof of "
                "text-semantic accuracy or perceptual timbre identity."
            ),
        },
        "rows": rows,
        "summary": {
            name: _cosine_summary(source_rows, name)
            for name, source_rows in metric_rows.items()
        },
        "aggregation": {
            "generated_comparisons": "one row per audition rollout",
            "codec_ceiling": "one row per unique source/direct audio pair",
            "worst": "minimum cosine",
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report frozen-CLAP audio/audio preservation for a Z-RAVE "
            "audition manifest. No text-semantic claim or hard gate is made."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-checkpoint-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = evaluate_clap_audition(
        args.manifest,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        cache_root=args.cache_root,
        device=args.device,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
