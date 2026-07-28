from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


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
