from types import SimpleNamespace

import torch
from torch import nn

from midibrave.latent_predictor import LatentPrediction
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.trainer import (PredictiveStage, configure_predictive_stage,
                               predictor_counterfactual_rollout)


class _ConstantPredictor(nn.Module):
    history_frames = 2
    horizon_frames = 4

    def forward(self, history, clap_future, midi_future):
        latent = history[..., -1:].expand(-1, -1, self.horizon_frames)
        return LatentPrediction(torch.zeros_like(latent), latent)


class _CounterfactualModel:
    def __init__(self):
        self.predictor = _ConstantPredictor()
        self.transition = None
        self.legacy_decode_called = False

    @staticmethod
    def project_clap(clap, frames):
        return clap[..., :1].unsqueeze(-1).expand(-1, 1, frames)

    @staticmethod
    def midi_control(note, velocity, frames):
        return note.float()[:, None, None].expand(-1, 1, frames)

    def decode_latents(self, z_rave, clap, note, velocity, excitation_seed=None):
        self.legacy_decode_called = True
        return z_rave.mean(dim=1, keepdim=True).repeat_interleave(8, dim=-1)

    def decode_latent_transition(
            self, z_rave, source_clap, target_clap, source_note, target_note,
            source_velocity, target_velocity, history_frames,
            excitation_seed=None):
        self.transition = {
            "source_clap": source_clap.detach().clone(),
            "target_clap": target_clap.detach().clone(),
            "source_note": source_note.detach().clone(),
            "target_note": target_note.detach().clone(),
            "history_frames": history_frames,
        }
        return z_rave.mean(dim=1, keepdim=True).repeat_interleave(8, dim=-1)


def test_counterfactual_uses_aligned_direct_controls_and_returns_only_future():
    config = SimpleNamespace(predictive=SimpleNamespace(
        rave_latent_dim=2,
        history_frames=2,
        horizon_frames=4,
        stride_frames=2,
        predictor_control_rollout_frames=4,
        predictor_timbre_interpolation_steps=4,
        samples_per_latent=8,
    ))
    model = _CounterfactualModel()
    batch = {
        "rave_a": torch.randn(4, 2, 6),
        "clap_a": torch.tensor([
            [1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 1.0],
        ]),
        "note_a": torch.tensor([48, 60, 48, 60]),
        "note_b": torch.tensor([55, 67, 55, 67]),
        "velocity_a": torch.tensor([80.0, 90.0, 80.0, 90.0]),
        "preset_id": ["a", "a", "b", "b"],
    }

    result = predictor_counterfactual_rollout(model, batch, config)

    assert not model.legacy_decode_called
    assert model.transition is not None
    assert model.transition["history_frames"] == 2
    assert torch.equal(model.transition["source_clap"], batch["clap_a"])
    assert torch.equal(model.transition["source_note"], batch["note_a"])
    assert torch.equal(model.transition["target_note"], result.target_note)
    expected_timbre = torch.nn.functional.normalize(
        batch["clap_a"].index_select(0, result.target_index)[2:], dim=-1)
    assert torch.allclose(result.target_clap[2:], expected_timbre)
    assert torch.equal(result.timbre_alpha[2:], torch.ones(2))
    assert torch.equal(model.transition["target_clap"], result.target_clap)
    assert result.audio.shape[-1] == 4 * config.predictive.samples_per_latent


class _TransitionHarness:
    samples_per_latent = 2
    predictive_config = SimpleNamespace(rave_latent_dim=1)

    def __init__(self):
        self.decoder_call = None

    @staticmethod
    def project_clap(clap, frames):
        return clap[:, :1, None].expand(-1, 1, frames)

    @staticmethod
    def midi_control(note, velocity, frames):
        return note.float()[:, None, None].expand(-1, 1, frames)

    @staticmethod
    def _excitation_bands(note, samples, seed):
        return note.float()[:, None, None].expand(-1, 1, samples // 2)

    def decoder(self, z_clap, z_midi, excitation, output_samples, z_rave=None):
        self.decoder_call = (z_clap, z_midi, excitation, output_samples, z_rave)
        return z_rave.new_zeros(z_rave.shape[0], 1, output_samples)


def test_decode_latent_transition_builds_synchronized_control_trajectories():
    model = _TransitionHarness()
    z_rave = torch.zeros(1, 1, 5)

    audio = PredictiveMidiBrave.decode_latent_transition(
        model, z_rave,
        source_clap=torch.tensor([[1.0, 0.0]]),
        target_clap=torch.tensor([[2.0, 0.0]]),
        source_note=torch.tensor([48]), target_note=torch.tensor([60]),
        source_velocity=torch.tensor([80.0]), target_velocity=torch.tensor([90.0]),
        history_frames=2)

    assert audio.shape == (1, 1, 10)
    assert model.decoder_call is not None
    z_clap, z_midi, excitation, samples, decoded_rave = model.decoder_call
    assert torch.equal(z_clap, torch.tensor([[[1.0, 1.0, 2.0, 2.0, 2.0]]]))
    assert torch.equal(z_midi, torch.tensor([[[48.0, 48.0, 60.0, 60.0, 60.0]]]))
    assert torch.equal(excitation, torch.tensor([[[48.0, 48.0, 60.0, 60.0, 60.0]]]))
    assert samples == 10
    assert decoded_rave is z_rave


class _StageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.rave_pitch_adversary = nn.Linear(1, 1)
        self.clap_projection = nn.Linear(1, 1)
        self.midi = nn.Linear(1, 1)
        self.predictor = nn.Linear(1, 1)
        self.decoder = nn.Linear(1, 1)


def test_rollout_stage_trains_only_predictor():
    model = _StageModel()

    trainable = configure_predictive_stage(model, PredictiveStage.ROLLOUT)

    assert trainable
    assert {name.split(".", 1)[0] for name in trainable} == {"predictor"}
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    for module in (model.encoder, model.rave_pitch_adversary,
                   model.clap_projection, model.midi, model.decoder):
        assert all(not parameter.requires_grad for parameter in module.parameters())
