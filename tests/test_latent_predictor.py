from __future__ import annotations

import torch

from midibrave.latent_predictor import MultiHorizonPredictor, rollout_blocks


def test_predictor_outputs_direct_cumulative_future():
    model = MultiHorizonPredictor(16, 256, 32, 64, 16, 8)
    history = torch.randn(3, 16, 16)
    clap = torch.randn(3, 256, 8)
    midi = torch.randn(3, 32, 8)
    result = model(history, clap, midi)
    assert result.delta.shape == (3, 16, 8)
    assert result.latent.shape == (3, 16, 8)
    expected = history[..., -1:] + result.delta.cumsum(-1)
    assert torch.allclose(result.latent, expected)


def test_predictor_future_control_path_is_causal():
    torch.manual_seed(7)
    model = MultiHorizonPredictor(4, 6, 3, 16, 16, 8).eval()
    history = torch.randn(1, 4, 16)
    clap_a = torch.randn(1, 6, 8)
    clap_b = clap_a.clone()
    clap_b[..., 4:] = torch.randn_like(clap_b[..., 4:])
    midi = torch.randn(1, 3, 8)
    first = model(history, clap_a, midi).delta
    second = model(history, clap_b, midi).delta
    assert torch.allclose(first[..., :4], second[..., :4], atol=1e-6)


def test_predictor_rejects_wrong_history():
    model = MultiHorizonPredictor(16, 256, 32, 64, 16, 8)
    history = torch.randn(1, 16, 15)
    clap = torch.randn(1, 256, 8)
    midi = torch.randn(1, 32, 8)
    try:
        model(history, clap, midi)
    except ValueError as error:
        assert "history" in str(error)
    else:
        raise AssertionError("wrong history length was accepted")


def test_rollout_consumes_four_frames_per_block():
    model = MultiHorizonPredictor(4, 6, 3, 16, 16, 8)
    history = torch.randn(2, 4, 16)
    clap = torch.randn(2, 6, 12)
    midi = torch.randn(2, 3, 12)
    result = rollout_blocks(model, history, clap, midi, stride_frames=4)
    assert result.latent.shape == (2, 4, 12)
    assert result.history.shape == (2, 4, 16)
    assert torch.equal(result.history[..., -4:], result.latent[..., -4:])
