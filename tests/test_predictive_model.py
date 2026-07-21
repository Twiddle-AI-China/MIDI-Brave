from __future__ import annotations

import torch

from midibrave.config import Config
from midibrave.predictive_model import PredictiveMidiBrave


def make_model() -> PredictiveMidiBrave:
    config = Config.load("configs/v3/smoke.yaml")
    assert config.predictive is not None
    return PredictiveMidiBrave(
        config.model, config.predictive,
        config.data.window_samples, config.data.sample_rate,
    )


def test_predictive_model_concatenates_rave_clap_and_midi():
    model = make_model()
    first_fusion = model.decoder.fusion.net[0]
    assert first_fusion.in_channels == 16 + 256 + 32
    audio = torch.randn(2, 1, 4096)
    clap = torch.randn(2, 512)
    note = torch.tensor([48, 60])
    velocity = torch.tensor([50.0, 127.0])
    result = model.forward_reconstruction(audio, clap, note, velocity,
                                          sample_encoder=False)
    assert result.posterior.latent.shape == (2, 16, 32)
    assert result.clap.shape == (2, 256, 32)
    assert result.midi.shape == (2, 32, 32)
    assert result.excitation.shape[0] == 2
    assert result.audio.shape == (2, 1, 4096)


def test_predictive_model_predicts_eight_future_frames():
    model = make_model()
    history = torch.randn(2, 16, 16)
    clap = torch.randn(2, 512)
    note = torch.tensor([48, 60])
    velocity = torch.tensor([80.0, 100.0])
    prediction = model.predict_future(history, clap, note, velocity)
    assert prediction.latent.shape == (2, 16, 8)
    assert prediction.delta.shape == prediction.latent.shape


def test_rave_pitch_adversary_is_training_only():
    model = make_model()
    latent = torch.randn(2, 16, 32)
    logits = model.rave_pitch_logits(latent, reversal_scale=0.5)
    assert logits.shape == (2, 128)


def test_predictive_decoder_rejects_wrong_rave_channels():
    model = make_model()
    try:
        model.decode_latents(
            torch.randn(1, 8, 32), torch.randn(1, 512),
            torch.tensor([60]), torch.tensor([100.0]),
        )
    except ValueError as error:
        assert "RAVE" in str(error)
    else:
        raise AssertionError("wrong RAVE latent dimension was accepted")
