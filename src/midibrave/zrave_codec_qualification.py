from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from .zrave_audio_quality import audio_pair_metrics
from .zrave_audition import decode_prerolled_latent
from .zrave_codec import encode_posterior_mean
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_data import _load_rave_audio
from .zrave_pure_flow_audition import (
    _load_codec,
    _load_manifest_rows,
    _load_sequence_latent,
    _sha256_file,
)


_DEFAULT_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")
_DEFAULT_PREROLL_FRAMES = 32
_DEFAULT_AUDIBLE_END_FRAME = 107
_DECODER_SEED = 20260728


def aligned_codec_window(
    sequence: Tensor,
    source_audio: np.ndarray,
    *,
    latent_hop: int,
    preroll_frames: int = 32,
    audible_end_frame: int = 107,
) -> tuple[Tensor, Tensor, np.ndarray]:
    if latent_hop <= 0 or preroll_frames <= 0:
        raise ValueError("latent hop and pre-roll frames must be positive")
    if audible_end_frame <= preroll_frames:
        raise ValueError("audible end frame must follow pre-roll")
    if sequence.ndim != 2 or sequence.shape[1] != 16:
        raise ValueError("packed sequence must have shape [frames, 16]")
    if sequence.shape[0] < audible_end_frame:
        raise ValueError(
            f"packed sequence must contain at least {audible_end_frame} frames"
        )
    if not torch.isfinite(sequence[:audible_end_frame]).all():
        raise ValueError("packed sequence is non-finite")
    source = np.asarray(source_audio, dtype=np.float32)
    required_samples = audible_end_frame * latent_hop
    if source.ndim != 1 or source.shape[0] < required_samples:
        raise ValueError(
            f"source audio must contain at least {required_samples} samples"
        )
    if not np.isfinite(source[:required_samples]).all():
        raise ValueError("source audio is non-finite")
    return (
        sequence[:preroll_frames].contiguous(),
        sequence[preroll_frames:audible_end_frame].contiguous(),
        np.ascontiguousarray(
            source[preroll_frames * latent_hop : required_samples]
        ),
    )


def validate_latent_pair(
    packed: Tensor,
    fresh: Tensor,
    *,
    minimum_cosine: float = 0.99999,
    maximum_rmse: float = 0.005,
) -> dict[str, float]:
    if not -1.0 <= minimum_cosine <= 1.0 or maximum_rmse <= 0.0:
        raise ValueError("latent pairing thresholds are invalid")
    if packed.shape != fresh.shape or packed.ndim != 2 or not packed.numel():
        raise ValueError("latent pairing tensors must have the same 2D shape")
    packed_value = packed.detach().double().reshape(-1).cpu()
    fresh_value = fresh.detach().double().reshape(-1).cpu()
    if not torch.isfinite(packed_value).all() or not torch.isfinite(
        fresh_value
    ).all():
        raise ValueError("latent pairing tensors are non-finite")
    delta = packed_value - fresh_value
    rmse = float(delta.square().mean().sqrt())
    maximum_absolute_error = float(delta.abs().max())
    denominator = float(packed_value.norm() * fresh_value.norm())
    if denominator == 0.0:
        cosine = 1.0 if rmse == 0.0 else 0.0
    else:
        cosine = float(torch.dot(packed_value, fresh_value) / denominator)
    if (
        not math.isfinite(cosine)
        or cosine < minimum_cosine
        or not math.isfinite(rmse)
        or rmse > maximum_rmse
    ):
        raise ValueError(
            "source-to-packed latent pairing failed: "
            f"cosine={cosine:.8f}, rmse={rmse:.8f}"
        )
    return {
        "cosine": cosine,
        "rmse": rmse,
        "maximum_absolute_error": maximum_absolute_error,
    }


def shared_listening_gain(
    source: np.ndarray,
    direct: np.ndarray,
    *,
    ceiling: float = 0.95,
) -> float:
    if not 0.0 < ceiling <= 1.0:
        raise ValueError("listening ceiling must be in (0, 1]")
    values = [
        np.asarray(source, dtype=np.float32),
        np.asarray(direct, dtype=np.float32),
    ]
    if any(value.ndim != 1 or not value.size for value in values):
        raise ValueError("listening audio must be non-empty and mono")
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("listening audio is non-finite")
    peak = max(float(np.max(np.abs(value))) for value in values)
    return min(1.0, ceiling / peak) if peak > 0.0 else 1.0


def qualification_file_names(
    case_index: int,
    category: str,
) -> dict[str, str]:
    if case_index < 0:
        raise ValueError("case index must be non-negative")
    slug = category.casefold().replace(" ", "-")
    if not slug or not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("category cannot form a safe file name")
    prefix = f"{case_index:02d}-{slug}"
    return {
        "source": f"wavs/{prefix}-source.wav",
        "rave_direct": f"wavs/{prefix}-rave-direct.wav",
    }


def select_qualification_cases(
    sequence_manifest: str | Path,
    *,
    categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
    seed: int = 20260728,
    audible_end_frame: int = _DEFAULT_AUDIBLE_END_FRAME,
) -> list[dict[str, Any]]:
    if (
        not categories
        or len(set(categories)) != len(categories)
        or audible_end_frame <= 0
    ):
        raise ValueError("qualification selection contract is invalid")
    selected: dict[str, tuple[bytes, dict[str, Any]]] = {}
    path = Path(sequence_manifest)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            category = str(row.get("category", ""))
            if (
                category not in categories
                or str(row.get("split")) != "test"
                or str(row.get("source_name")) != "serum_full"
                or int(row.get("length", 0)) < audible_end_frame
            ):
                continue
            identity = (
                f"{seed}:{category}:"
                f"{row.get('canonical_preset_id')}:"
                f"{row.get('sample_id')}"
            )
            digest = hashlib.sha256(identity.encode("utf-8")).digest()
            current = selected.get(category)
            if current is None or digest < current[0]:
                selected[category] = (digest, row)
    missing = [category for category in categories if category not in selected]
    if missing:
        raise ValueError(
            "test split lacks codec qualification rows: "
            + ", ".join(missing)
        )
    return [selected[category][1] for category in categories]


def _audio_diagnostics(audio: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("audio diagnostics require finite mono audio")
    return {
        "samples": int(value.shape[0]),
        "peak": float(np.max(np.abs(value))),
        "rms": float(
            np.sqrt(np.mean(np.square(value, dtype=np.float64)))
        ),
    }


def _write_pcm16(
    output: Path,
    relative_path: str,
    audio: np.ndarray,
    *,
    sample_rate: int,
    gain: float,
) -> dict[str, object]:
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    listening = np.clip(
        np.asarray(audio, dtype=np.float32) * gain,
        -1.0,
        1.0,
    )
    sf.write(path, listening, sample_rate, subtype="PCM_16")
    return {
        "wav": relative_path,
        "sha256": _sha256_file(path),
        "subtype": "PCM_16",
        "listening": _audio_diagnostics(listening),
        "unscaled": _audio_diagnostics(audio),
    }


def write_qualification_case(
    output: str | Path,
    *,
    case_index: int,
    category: str,
    source: np.ndarray,
    direct: np.ndarray,
    packed: Tensor,
    fresh: Tensor,
    sample_rate: int,
    latent_hop: int,
    preroll_frames: int = _DEFAULT_PREROLL_FRAMES,
    audible_end_frame: int = _DEFAULT_AUDIBLE_END_FRAME,
) -> dict[str, object]:
    if min(sample_rate, latent_hop, preroll_frames) <= 0:
        raise ValueError("qualification audio dimensions must be positive")
    audible_frames = audible_end_frame - preroll_frames
    if audible_frames <= 0:
        raise ValueError("qualification audible window is empty")
    expected_samples = audible_frames * latent_hop
    reference = np.asarray(source, dtype=np.float32)
    reconstruction = np.asarray(direct, dtype=np.float32)
    if (
        reference.ndim != 1
        or reconstruction.ndim != 1
        or reference.shape != (expected_samples,)
        or reconstruction.shape != (expected_samples,)
        or not np.isfinite(reference).all()
        or not np.isfinite(reconstruction).all()
    ):
        raise ValueError(
            "qualification source and direct audio must match the "
            "aligned audible window"
        )
    pairing = validate_latent_pair(packed, fresh)
    head_samples = min(4 * latent_hop, expected_samples)
    head_peak = float(np.max(np.abs(reconstruction[:head_samples])))
    remaining = reconstruction[head_samples:]
    remaining_peak = (
        float(np.max(np.abs(remaining))) if remaining.size else 0.0
    )
    cold_start_ratio = (
        head_peak / remaining_peak
        if remaining_peak > 0.0
        else math.inf
    )
    if not math.isfinite(cold_start_ratio) or cold_start_ratio > 1.25:
        raise ValueError(
            "RAVE direct reconstruction retains a dominant cold-start "
            f"transient: ratio={cold_start_ratio:.6f}"
        )
    metrics = audio_pair_metrics(
        reference,
        reconstruction,
        sample_rate,
        latent_hop=latent_hop,
        chunk_frames=8,
    )
    gain = shared_listening_gain(reference, reconstruction)
    names = qualification_file_names(case_index, category)
    root = Path(output)
    audio = {
        role: _write_pcm16(
            root,
            names[role],
            value,
            sample_rate=sample_rate,
            gain=gain,
        )
        for role, value in (
            ("source", reference),
            ("rave_direct", reconstruction),
        )
    }
    return {
        "case_index": case_index,
        "category": category,
        "decoder_preroll_mode": "preceding_real_latent",
        "decoder_preroll_frames": preroll_frames,
        "source_start_frame": preroll_frames,
        "source_start_sample": preroll_frames * latent_hop,
        "audible_end_frame": audible_end_frame,
        "audible_frames": audible_frames,
        "audible_samples": expected_samples,
        "cold_start_head_frames": 4,
        "cold_start_peak_ratio": cold_start_ratio,
        "pairing": pairing,
        "audio_metrics": metrics,
        "listening_gain": gain,
        "audio": audio,
    }


def _codec_scalar(value: object) -> int:
    if isinstance(value, Tensor):
        return int(value.flatten()[0].item())
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return int(value[0])  # type: ignore[index]
    return int(value)


def _validate_pack_binding(
    index: object,
    config: ZraveFlowConfig,
    codec_hash: str,
) -> None:
    if not isinstance(index, dict):
        raise ValueError("packed index must be a mapping")
    expected = {
        "rave_checkpoint_sha256": codec_hash,
        "sample_rate": config.rave.sample_rate,
        "latent_hop": config.rave.latent_hop,
        "latent_dim": config.model.latent_dim,
    }
    for name, value in expected.items():
        if index.get(name) != value:
            raise ValueError(f"packed index {name} mismatch")


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def render_codec_qualification(
    config_path: str | Path,
    output_root: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, object]:
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("codec qualification output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path)
    config = ZraveFlowConfig.load(config_file)
    selected_device = torch.device(device)
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    codec, codec_hash = _load_codec(config, selected_device)
    if _codec_scalar(getattr(codec, "sr")) != config.rave.sample_rate:
        raise ValueError("RAVE codec sample rate mismatch")
    _validate_pack_binding(index, config, codec_hash)
    cases = select_qualification_cases(
        packed_root / "sequences.jsonl",
    )
    manifest_indices = {int(row["manifest_index"]) for row in cases}
    source_rows = _load_manifest_rows(
        Path(config.data.unified_manifest),
        manifest_indices,
    )
    case_reports: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        sequence = _load_sequence_latent(packed_root, case)
        source_row = source_rows[int(case["manifest_index"])]
        source_audio = _load_rave_audio(
            Path(str(source_row["audio_path"])),
            sample_rate=config.rave.sample_rate,
            maximum_seconds=config.data.maximum_audio_seconds,
        )
        preroll, audible, source = aligned_codec_window(
            sequence,
            source_audio,
            latent_hop=config.rave.latent_hop,
        )
        encoded_samples = (
            _DEFAULT_AUDIBLE_END_FRAME * config.rave.latent_hop
        )
        encoded = encode_posterior_mean(
            codec,
            torch.from_numpy(source_audio[:encoded_samples])
            .view(1, 1, -1)
            .to(selected_device),
        )
        fresh = (
            encoded[0]
            .transpose(0, 1)
            .float()
            .cpu()
            .contiguous()
        )
        if fresh.shape != sequence[:_DEFAULT_AUDIBLE_END_FRAME].shape:
            raise ValueError("fresh RAVE latent shape does not match packed data")
        direct = decode_prerolled_latent(
            codec,
            preroll.to(selected_device)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous(),
            audible.to(selected_device)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous(),
            _DECODER_SEED,
            latent_hop=config.rave.latent_hop,
        )
        report = write_qualification_case(
            output,
            case_index=case_index,
            category=str(case["category"]),
            source=source,
            direct=direct,
            packed=sequence[:_DEFAULT_AUDIBLE_END_FRAME],
            fresh=fresh,
            sample_rate=config.rave.sample_rate,
            latent_hop=config.rave.latent_hop,
        )
        report.update(
            {
                "sample_id": case["sample_id"],
                "canonical_preset_id": case["canonical_preset_id"],
                "manifest_index": int(case["manifest_index"]),
                "source_audio_path": source_row["audio_path"],
                "latent_shard": case["shard"],
                "latent_shard_row": int(case["shard_row"]),
            }
        )
        case_reports.append(report)
    if len(case_reports) != len(_DEFAULT_CATEGORIES):
        raise ValueError("codec qualification did not render five cases")
    manifest: dict[str, object] = {
        "format": 1,
        "architecture": "zrave_standalone_codec_qualification_v1",
        "codec_only": True,
        "transformer_checkpoint": None,
        "config": str(config_file),
        "config_sha256": _sha256_file(config_file),
        "pack_index_sha256": _sha256_file(index_path),
        "codec_sha256": codec_hash,
        "sample_rate": config.rave.sample_rate,
        "latent_dim": config.model.latent_dim,
        "latent_hop": config.rave.latent_hop,
        "decoder_seed": _DECODER_SEED,
        "decoder_preroll_mode": "preceding_real_latent",
        "decoder_preroll_frames": _DEFAULT_PREROLL_FRAMES,
        "source_start_frame": _DEFAULT_PREROLL_FRAMES,
        "source_start_sample": (
            _DEFAULT_PREROLL_FRAMES * config.rave.latent_hop
        ),
        "audible_end_frame": _DEFAULT_AUDIBLE_END_FRAME,
        "audible_frames": (
            _DEFAULT_AUDIBLE_END_FRAME - _DEFAULT_PREROLL_FRAMES
        ),
        "cases": case_reports,
    }
    _atomic_json(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        "Standalone RAVE codec qualification.\n"
        "No Transformer checkpoint, MIDI, CLAP, or generated latent is used.\n"
        "Each category compares aligned source audio with direct reconstruction "
        "from the exact packed latent sequence.\n"
        "Real latent frames 0:32 warm the decoder and are cropped; audible "
        "audio corresponds exactly to frames 32:107 (153,600 samples).\n"
        "WAV files are mono 44.1 kHz PCM-16. Source and direct use one shared "
        "attenuation gain per category, with no per-file normalization.\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an aligned standalone RAVE codec qualification."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = render_codec_qualification(
        args.config,
        args.output,
        device=args.device,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
