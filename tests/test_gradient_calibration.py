from __future__ import annotations

import torch

from midibrave.calibration import (calibrate_loss_weights,
                                   cap_auxiliary_gradient, loss_gradient_norm)


def test_calibration_uses_inverse_median_norm_and_anchor():
    norms = {
        "future": [2.0, 2.0, 2.0],
        "rollout": [1.0, 1.0, 1.0],
        "audio": [4.0, 4.0, 4.0],
    }
    weights = calibrate_loss_weights(
        norms, {"future": 0.5, "rollout": 0.25, "audio": 0.25},
        {"future": 1.0, "rollout": 0.5, "audio": 0.25},
        anchor="future", clip_ratio=(0.25, 4.0))
    assert weights["future"] == 1.0
    assert weights["rollout"] == 1.0
    assert weights["audio"] == 0.25


def test_calibration_clips_relative_to_initial_weights():
    weights = calibrate_loss_weights(
        {"future": [1.0], "tiny": [0.00001]},
        {"future": 0.5, "tiny": 0.5},
        {"future": 1.0, "tiny": 0.5}, anchor="future",
        clip_ratio=(0.25, 4.0))
    assert weights["tiny"] == 2.0


def test_auxiliary_gradient_is_capped_after_correction():
    reference = torch.tensor([3.0, 4.0])
    auxiliary = reference * 2.0
    capped = cap_auxiliary_gradient(auxiliary, reference, 0.15)
    assert torch.allclose(capped.norm(), reference.norm() * 0.15)
    small = reference * 0.1
    assert torch.equal(cap_auxiliary_gradient(small, reference, 0.15), small)


def test_loss_gradient_norm_is_normalized_by_element_count():
    latent = torch.zeros(2, 3, 4, requires_grad=True)
    loss = latent.sum()
    assert torch.allclose(loss_gradient_norm(loss, latent), torch.tensor(1.0))
