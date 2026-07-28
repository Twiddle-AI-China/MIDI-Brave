from __future__ import annotations

import numpy as np
import pytest
import torch

from midibrave.zrave_codec_qualification import (
    aligned_codec_window,
    shared_listening_gain,
    validate_latent_pair,
)


def test_aligned_codec_window_uses_real_preroll_and_matching_source() -> None:
    sequence = torch.arange(107 * 16).reshape(107, 16).float()
    source = np.arange(107 * 4, dtype=np.float32)

    preroll, audible, reference = aligned_codec_window(
        sequence,
        source,
        latent_hop=4,
    )

    assert torch.equal(preroll, sequence[:32])
    assert torch.equal(audible, sequence[32:107])
    assert np.array_equal(reference, source[32 * 4 : 107 * 4])


def test_aligned_codec_window_rejects_insufficient_real_context() -> None:
    with pytest.raises(ValueError, match="107"):
        aligned_codec_window(
            torch.zeros(106, 16),
            np.zeros(107 * 4, dtype=np.float32),
            latent_hop=4,
        )

    with pytest.raises(ValueError, match="source"):
        aligned_codec_window(
            torch.zeros(107, 16),
            np.zeros(107 * 4 - 1, dtype=np.float32),
            latent_hop=4,
        )


def test_validate_latent_pair_accepts_float16_storage_error() -> None:
    fresh = torch.linspace(-2.0, 2.0, 107 * 16).reshape(107, 16)
    packed = fresh.half().float()

    report = validate_latent_pair(packed, fresh)

    assert report["cosine"] >= 0.99999
    assert report["rmse"] < 0.005
    assert report["maximum_absolute_error"] < 0.005


def test_validate_latent_pair_rejects_wrong_source_binding() -> None:
    packed = torch.ones(107, 16)

    with pytest.raises(ValueError, match="pairing"):
        validate_latent_pair(packed, -packed)


def test_shared_listening_gain_is_one_shared_attenuation() -> None:
    source = np.array([-0.5, 0.5], dtype=np.float32)
    direct = np.array([-2.0, 1.0], dtype=np.float32)

    assert shared_listening_gain(source, direct) == pytest.approx(0.475)
    assert shared_listening_gain(source, source) == 1.0
