from __future__ import annotations

import torch

from midibrave.rave_encoder import RaveEncoder


def test_rave_encoder_has_128_sample_hop_and_finite_kl():
    encoder = RaveEncoder(16, 16, [2, 2, 2, 1], capacity=8)
    result = encoder(torch.randn(2, 1, 4096), sample=False)
    assert result.mean.shape == (2, 16, 32)
    assert result.logvar.shape == result.mean.shape
    assert torch.equal(result.latent, result.mean)
    assert result.kl.ndim == 0
    assert torch.isfinite(result.kl)


def test_rave_encoder_samples_only_when_requested():
    encoder = RaveEncoder(16, 8, [2, 2, 2, 1], capacity=8)
    audio = torch.randn(1, 1, 2048)
    deterministic = encoder(audio, sample=False)
    torch.manual_seed(3)
    sampled = encoder(audio, sample=True)
    assert not torch.equal(sampled.latent, deterministic.latent)
    assert torch.equal(deterministic.latent, deterministic.mean)


def test_rave_encoder_is_causal():
    torch.manual_seed(5)
    encoder = RaveEncoder(16, 8, [2, 2, 2, 1], capacity=8).eval()
    first = torch.randn(1, 1, 4096)
    second = first.clone()
    second[..., 2048:] = torch.randn_like(second[..., 2048:])
    a = encoder(first, sample=False).mean
    b = encoder(second, sample=False).mean
    assert torch.allclose(a[..., :16], b[..., :16], atol=1e-6)


def test_rave_encoder_backpropagates_finite_gradients():
    encoder = RaveEncoder(16, 8, [2, 2, 2, 1], capacity=8)
    result = encoder(torch.randn(2, 1, 2048), sample=True)
    (result.latent.square().mean() + result.kl).backward()
    gradients = [parameter.grad for parameter in encoder.parameters()
                 if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients
               if gradient is not None)


def test_rave_encoder_rejects_invalid_input_and_hop():
    try:
        RaveEncoder(16, 8, [2, 2, 3, 1], capacity=8)
    except ValueError as error:
        assert "128" in str(error)
    else:
        raise AssertionError("non-128-sample hop was accepted")
    encoder = RaveEncoder(16, 8, [2, 2, 2, 1], capacity=8)
    try:
        encoder(torch.randn(1, 2, 2048))
    except ValueError as error:
        assert "mono" in str(error)
    else:
        raise AssertionError("stereo input was accepted")
