from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


_STFT_SETTINGS = ((512, 128), (1024, 256), (2048, 512))


def _mono_finite(value: np.ndarray, name: str) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 1 or not audio.size:
        raise ValueError(f"{name} must be non-empty mono audio")
    if not np.isfinite(audio).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(audio)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _spectral_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> tuple[float, float]:
    cosines: list[float] = []
    log_distances: list[float] = []
    for frame_size, hop_size in _STFT_SETTINGS:
        _, _, reference_stft = signal.stft(
            reference,
            fs=sample_rate,
            window="hann",
            nperseg=frame_size,
            noverlap=frame_size - hop_size,
            nfft=frame_size,
            boundary=None,
            padded=False,
        )
        _, _, estimate_stft = signal.stft(
            estimate,
            fs=sample_rate,
            window="hann",
            nperseg=frame_size,
            noverlap=frame_size - hop_size,
            nfft=frame_size,
            boundary=None,
            padded=False,
        )
        reference_magnitude = np.abs(reference_stft).astype(np.float64)
        estimate_magnitude = np.abs(estimate_stft).astype(np.float64)
        denominator = math.sqrt(
            float(np.sum(np.square(reference_magnitude)))
        ) * math.sqrt(
            float(np.sum(np.square(estimate_magnitude)))
        )
        cosine = (
            float(np.sum(reference_magnitude * estimate_magnitude) / denominator)
            if denominator > 0.0
            else 0.0
        )
        magnitude_peak = max(
            float(reference_magnitude.max(initial=0.0)),
            float(estimate_magnitude.max(initial=0.0)),
            1.0e-12,
        )
        floor = magnitude_peak * 1.0e-5
        log_distance = float(
            np.mean(
                np.abs(
                    np.log(np.maximum(reference_magnitude, floor))
                    - np.log(np.maximum(estimate_magnitude, floor))
                )
            )
        )
        cosines.append(cosine)
        log_distances.append(log_distance)
    return float(np.mean(cosines)), float(np.mean(log_distances))


def _basic_audio_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    reference_rms = _rms(reference)
    estimate_rms = _rms(estimate)
    if reference_rms <= 1.0e-12:
        raise ValueError("reference audio is silent")
    spectral_cosine, log_spectral_l1 = _spectral_metrics(
        reference,
        estimate,
        sample_rate,
    )
    reference_std = float(np.std(reference))
    estimate_std = float(np.std(estimate))
    if reference_std > 1.0e-12 and estimate_std > 1.0e-12:
        centered_reference = reference.astype(np.float64) - float(
            np.mean(reference)
        )
        centered_estimate = estimate.astype(np.float64) - float(
            np.mean(estimate)
        )
        correlation_denominator = math.sqrt(
            float(np.sum(np.square(centered_reference)))
        ) * math.sqrt(
            float(np.sum(np.square(centered_estimate)))
        )
        waveform_correlation = float(
            np.sum(centered_reference * centered_estimate)
            / correlation_denominator
        )
    else:
        waveform_correlation = 0.0
    return {
        "spectral_cosine": spectral_cosine,
        "log_spectral_l1": log_spectral_l1,
        "rms_drift_db": float(
            20.0
            * math.log10(
                max(estimate_rms, 1.0e-12) / max(reference_rms, 1.0e-12)
            )
        ),
        "waveform_correlation": waveform_correlation,
    }


def audio_pair_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    *,
    latent_hop: int,
    chunk_frames: int,
) -> dict[str, object]:
    direct = _mono_finite(reference, "reference")
    predicted = _mono_finite(estimate, "estimate")
    if direct.shape != predicted.shape:
        raise ValueError("reference and estimate must have identical shapes")
    if min(sample_rate, latent_hop, chunk_frames) <= 0:
        raise ValueError("audio metric rates and frame counts must be positive")
    full = _basic_audio_metrics(direct, predicted, sample_rate)
    chunk_samples = latent_hop * chunk_frames
    chunks: list[dict[str, float | int]] = []
    for chunk_index, start in enumerate(
        range(0, direct.shape[0], chunk_samples)
    ):
        stop = min(start + chunk_samples, direct.shape[0])
        metrics = _basic_audio_metrics(
            direct[start:stop],
            predicted[start:stop],
            sample_rate,
        )
        chunks.append(
            {
                "index": chunk_index,
                "start_sample": start,
                "end_sample": stop,
                **metrics,
            }
        )
    return {
        "sample_count": int(direct.shape[0]),
        **full,
        "tail_rms_drift_db": float(chunks[-1]["rms_drift_db"]),
        "chunks": chunks,
    }


def _summary(
    values: list[float],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("metric summary requires finite values")
    count = max(1, math.ceil(array.size * 0.1))
    ordered = np.sort(array)
    worst = ordered[:count] if higher_is_better else ordered[-count:]
    return {
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "worst_decile_mean": float(worst.mean()),
        "minimum": float(ordered[0]),
        "maximum": float(ordered[-1]),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    metrics = [row["metrics"] for row in rows]
    report: dict[str, object] = {
        "spectral_cosine": _summary(
            [float(item["spectral_cosine"]) for item in metrics],
            higher_is_better=True,
        ),
        "log_spectral_l1": _summary(
            [float(item["log_spectral_l1"]) for item in metrics],
            higher_is_better=False,
        ),
        "waveform_correlation": _summary(
            [float(item["waveform_correlation"]) for item in metrics],
            higher_is_better=True,
        ),
        "absolute_rms_drift_db": _summary(
            [abs(float(item["rms_drift_db"])) for item in metrics],
            higher_is_better=False,
        ),
    }
    if all("tail_rms_drift_db" in item for item in metrics):
        report["absolute_tail_rms_drift_db"] = _summary(
            [abs(float(item["tail_rms_drift_db"])) for item in metrics],
            higher_is_better=False,
        )
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_asset(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("audition audio path must be a non-empty string")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError("audition audio path escapes its package")
    if not path.is_file():
        raise ValueError(f"audition audio does not exist: {relative}")
    return path


def _read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=False)
    if actual_rate != sample_rate:
        raise ValueError(
            f"audio sample rate mismatch: {actual_rate} != {sample_rate}"
        )
    return _mono_finite(audio, str(path))


def _all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def evaluate_audition_audio(
    manifest_path: str | Path,
) -> dict[str, object]:
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("audition manifest must be a mapping")
    sample_rate = int(manifest.get("sample_rate") or 0)
    latent_hop = int(manifest.get("latent_hop") or 0)
    future_frames = int(manifest.get("future_frames") or 0)
    if min(sample_rate, latent_hop, future_frames) <= 0:
        raise ValueError("audition manifest has invalid audio dimensions")
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("audition manifest has no comparisons")
    expected_samples = future_frames * latent_hop
    root = source.parent
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("audition comparison must be a mapping")
        identifier = str(comparison.get("id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError("audition comparison IDs must be unique")
        identifiers.add(identifier)
        audio = comparison.get("audio")
        if not isinstance(audio, dict):
            raise ValueError("audition comparison has no audio mapping")
        direct = _read_audio(
            _resolve_asset(root, audio.get("direct")),
            sample_rate,
        )
        predicted = _read_audio(
            _resolve_asset(root, audio.get("predicted")),
            sample_rate,
        )
        if direct.shape[0] != expected_samples:
            raise ValueError(
                f"audition comparison {identifier} has unexpected length"
            )
        metrics = audio_pair_metrics(
            direct,
            predicted,
            sample_rate,
            latent_hop=latent_hop,
            chunk_frames=8,
        )
        rows.append(
            {
                "id": identifier,
                "category": comparison.get("category"),
                "split": comparison.get("split"),
                "metrics": metrics,
            }
        )
    categories = sorted(
        {
            str(row["category"])
            for row in rows
            if row.get("category") is not None
        }
    )
    category_summary = {
        category: _summarize_rows(
            [row for row in rows if str(row.get("category")) == category]
        )
        for category in categories
    }
    chunk_count = len(rows[0]["metrics"]["chunks"])
    if any(len(row["metrics"]["chunks"]) != chunk_count for row in rows):
        raise ValueError("audition comparisons have inconsistent chunks")
    chunk_summary = []
    for index in range(chunk_count):
        chunk_rows = [
            {
                "metrics": row["metrics"]["chunks"][index],
            }
            for row in rows
        ]
        chunk_summary.append(
            {
                "index": index,
                "summary": _summarize_rows(chunk_rows),
            }
        )
    transformer = manifest.get("transformer")
    checkpoint_update = (
        int(transformer["update"])
        if isinstance(transformer, dict) and "update" in transformer
        else None
    )
    report: dict[str, object] = {
        "schema": 1,
        "manifest": str(source.resolve()),
        "manifest_sha256": _sha256_file(source),
        "checkpoint_update": checkpoint_update,
        "sample_rate": sample_rate,
        "latent_hop": latent_hop,
        "future_frames": future_frames,
        "pairs": len(rows),
        "rows": rows,
        "summary": _summarize_rows(rows),
        "category_summary": category_summary,
        "chunk_summary": chunk_summary,
    }
    report["finite"] = _all_finite(report)
    return report


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure direct-versus-predicted Z-RAVE audition audio."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = evaluate_audition_audio(args.manifest)
    _atomic_json(Path(args.output), report)
    print(json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
