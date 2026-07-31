from __future__ import annotations

import numpy as np
import torch

from midibrave.rave_full_latent_export import full_latent_fidelity_curve


def test_full_latent_curve_forces_official_export_to_keep_128_dims() -> None:
    curve = full_latent_fidelity_curve(128, dtype=torch.float32)

    assert curve.shape == (128,)
    assert torch.isfinite(curve).all()
    assert torch.all(curve[1:] >= curve[:-1])
    first_above = max(int(np.argmax(curve.numpy() > 0.999)), 1)
    exported_size = 2 ** int(np.ceil(np.log2(first_above)))
    assert exported_size == 128
