from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def loss_gradient_norm(loss: Tensor, latent: Tensor) -> Tensor:
    gradient, = torch.autograd.grad(loss, latent, retain_graph=True)
    if not torch.isfinite(gradient).all():
        raise ValueError("loss calibration produced a non-finite gradient")
    return gradient.square().mean().sqrt()


def calibrate_loss_weights(norm_samples: Mapping[str, Sequence[float]],
                           target_shares: Mapping[str, float],
                           initial_weights: Mapping[str, float],
                           anchor: str,
                           clip_ratio: tuple[float, float] = (0.25, 4.0),
                           epsilon: float = 1e-12) -> dict[str, float]:
    keys = set(norm_samples)
    if keys != set(target_shares) or keys != set(initial_weights):
        raise ValueError("calibration norms, shares, and initial weights must share keys")
    if anchor not in keys:
        raise ValueError("calibration anchor is missing")
    if not 0 < clip_ratio[0] <= clip_ratio[1]:
        raise ValueError("invalid calibration clip ratio")
    raw = {}
    for name in sorted(keys):
        samples = np.asarray(norm_samples[name], dtype=np.float64)
        if not samples.size or not np.isfinite(samples).all() or np.any(samples < 0):
            raise ValueError(f"invalid gradient norms for {name}")
        share = float(target_shares[name])
        initial = float(initial_weights[name])
        if share <= 0 or initial <= 0:
            raise ValueError("calibration shares and initial weights must be positive")
        raw[name] = share / max(float(np.median(samples)), epsilon)
    anchor_scale = float(initial_weights[anchor]) / raw[anchor]
    calibrated = {}
    for name in sorted(keys):
        initial = float(initial_weights[name])
        value = raw[name] * anchor_scale
        calibrated[name] = float(np.clip(
            value, initial * clip_ratio[0], initial * clip_ratio[1]))
    calibrated[anchor] = float(initial_weights[anchor])
    return calibrated


def cap_auxiliary_gradient(auxiliary: Tensor, reference: Tensor,
                           maximum_fraction: float, epsilon: float = 1e-12) -> Tensor:
    if auxiliary.shape != reference.shape:
        raise ValueError("auxiliary and reference gradients must share shape")
    if not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError("maximum gradient fraction must be between zero and one")
    if not torch.isfinite(auxiliary).all() or not torch.isfinite(reference).all():
        raise ValueError("gradient cap rejects non-finite values")
    auxiliary_norm = auxiliary.norm()
    limit = reference.norm() * maximum_fraction
    if auxiliary_norm <= limit:
        return auxiliary
    return auxiliary * (limit / auxiliary_norm.clamp_min(epsilon))
