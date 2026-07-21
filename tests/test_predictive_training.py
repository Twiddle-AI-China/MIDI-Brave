from dataclasses import replace
from pathlib import Path

import pytest
import torch

from midibrave.predictive_losses import LatentStatistics

from midibrave.config import Config
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.trainer import (PredictiveStage, configure_predictive_stage,
                               _predictive_scaler_step,
                               counterfactual_control_assignment,
                               midi_swap_example_weights,
                               predictive_checkpoint_contract,
                               predictor_counterfactual_rollout,
                               predictive_loss_schedule,
                               predictive_latent_objective,
                               predictive_stage_objective,
                               load_predictive_checkpoint,
                               load_predictive_warm_start,
                               load_predictive_calibration,
                               save_predictive_calibration,
                               save_predictive_checkpoint,
                               validate_predictive_resume)


ROOT = Path(__file__).resolve().parents[1]


class _RecordingScaler:
    def __init__(self):
        self.step_calls = 0
        self.updates = []

    def step(self, optimizer):
        self.step_calls += 1

    def update(self, new_scale=None):
        self.updates.append(new_scale)


class _RecordingSwapPitch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.notes = None

    def forward(self, audio, note, confidence, valid):
        self.notes = note.detach().clone()
        return audio.float().square().mean()

    def source_rejection(self, audio, target_note, source_note,
                         confidence, valid, weights, margin=0.5):
        self.source_notes = source_note.detach().clone()
        self.target_notes = target_note.detach().clone()
        self.weights = weights.detach().clone()
        return audio.float().abs().mean()


def test_predictive_nonfinite_global_gradient_never_steps_optimizer():
    scaler = _RecordingScaler()

    applied = _predictive_scaler_step(
        scaler, object(), globally_finite=False, previous_scale=512.0)

    assert not applied
    assert scaler.step_calls == 0
    assert scaler.updates == [256.0]


def test_predictive_finite_global_gradient_steps_and_updates_scaler():
    scaler = _RecordingScaler()

    applied = _predictive_scaler_step(
        scaler, object(), globally_finite=True, previous_scale=512.0)

    assert applied
    assert scaler.step_calls == 1
    assert scaler.updates == [None]


def _config() -> Config:
    return Config.load(ROOT / "configs/v3/smoke.yaml")


def _model(config: Config) -> PredictiveMidiBrave:
    assert config.predictive is not None
    return PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)


def test_counterfactual_assignment_splits_batch_and_avoids_same_preset():
    presets = ["a", "a", "b", "b", "c", "c"]

    midi, timbre, target = counterfactual_control_assignment(
        presets, torch.device("cpu"))

    assert torch.equal(midi, torch.tensor([True, True, True, False, False, False]))
    assert not (midi & timbre).any()
    assert torch.equal(midi | timbre, torch.ones(6, dtype=torch.bool))
    for index in timbre.nonzero().flatten().tolist():
        assert presets[index] != presets[int(target[index])]


def test_counterfactual_assignment_falls_back_to_midi_without_cross_preset():
    midi, timbre, target = counterfactual_control_assignment(
        ["a", "a", "a", "a"], torch.device("cpu"))

    assert midi.all()
    assert not timbre.any()
    assert torch.equal(target, torch.arange(4))


def test_midi_swap_weights_boost_low_and_large_intervals_then_normalize():
    value = midi_swap_example_weights(
        torch.tensor([60, 60, 60]), torch.tensor([55, 40, 72]),
        torch.tensor([True, True, True]))

    assert value.mean().item() == pytest.approx(1.0)
    assert value[1] > value[0]
    assert value[2] > value[0]


def test_midi_swap_weights_keep_inactive_examples_zero():
    value = midi_swap_example_weights(
        torch.tensor([60, 60, 60]), torch.tensor([55, 40, 72]),
        torch.tensor([True, False, True]))

    assert value[1].item() == 0.0
    assert value.sum().item() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("stage", "gate", "included", "excluded"),
    [
        (PredictiveStage.RAVE, False,
         {"encoder", "clap_projection", "midi", "decoder"}, {"predictor"}),
        (PredictiveStage.PREDICTOR, False, {"predictor"},
         {"encoder", "clap_projection", "midi", "decoder"}),
        (PredictiveStage.ROLLOUT, False,
         {"predictor", "clap_projection", "midi", "decoder"}, {"encoder"}),
        (PredictiveStage.GAN, False, {"decoder"},
         {"encoder", "predictor", "clap_projection", "midi"}),
        (PredictiveStage.GAN, True, {"decoder", "predictor"},
         {"encoder", "clap_projection", "midi"}),
    ],
)
def test_stage_parameter_ownership(stage, gate, included, excluded):
    model = _model(_config())
    trainable = configure_predictive_stage(model, stage, gate)
    assert trainable
    for prefix in included:
        assert any(name == prefix or name.startswith(prefix + ".") for name in trainable)
    for prefix in excluded:
        assert not any(name == prefix or name.startswith(prefix + ".") for name in trainable)
    assert trainable == {name for name, value in model.named_parameters()
                         if value.requires_grad}


def test_predictive_loss_schedule_reaches_configured_weights():
    config = _config()
    assert config.predictive is not None and config.latent_loss is not None
    start = predictive_loss_schedule(config, PredictiveStage.RAVE, 0)
    mature = predictive_loss_schedule(config, PredictiveStage.RAVE, 4)
    assert start.rave_kl == 0.0
    assert mature.rave_kl == pytest.approx(config.latent_loss.rave_kl)
    assert start.rave_pitch_adversary == 0.0
    assert predictive_loss_schedule(
        config, PredictiveStage.RAVE, 2).rave_pitch_adversary == pytest.approx(
            config.latent_loss.rave_pitch_adversary)

    rollout_start = predictive_loss_schedule(config, PredictiveStage.ROLLOUT, 0)
    rollout_end = predictive_loss_schedule(config, PredictiveStage.ROLLOUT, 4)
    assert rollout_start.rollout == 0.0
    assert rollout_end.rollout == pytest.approx(config.latent_loss.rollout)
    assert rollout_end.teacher_forcing == pytest.approx(
        config.predictive.teacher_forcing_floor)

    gan = predictive_loss_schedule(config, PredictiveStage.GAN, 100)
    assert gan.gan_adversarial == pytest.approx(config.latent_loss.gan_adversarial)
    assert gan.gan_feature_matching == pytest.approx(
        config.latent_loss.gan_feature_matching)


def test_rave_objective_supervises_counterfactual_midi_note_swap():
    config = _config()
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.RAVE)
    pitch = _RecordingSwapPitch()
    frames = config.data.window_samples // config.data.pitch_hop_length
    batch = {
        "audio_a": torch.randn(2, 1, config.data.window_samples),
        "clap_a": torch.randn(2, config.model.clap_dim),
        "note_a": torch.tensor([48, 60]),
        "note_b": torch.tensor([55, 67]),
        "velocity_a": torch.tensor([80.0, 100.0]),
        "pitch_confidence_a": torch.ones(2, frames),
        "pitch_valid_mask_a": torch.ones(2, frames, dtype=torch.bool),
        "valid_samples_a": torch.full((2,), config.data.window_samples),
        "excitation_seed_a": torch.tensor([1, 2]),
    }

    objective = predictive_stage_objective(
        model, batch, config, PredictiveStage.RAVE, 4, swap_pitch=pitch)

    assert torch.equal(pitch.notes, batch["note_b"])
    assert objective.components["midi_swap_pitch"].item() > 0
    assert objective.diagnostic_tensors["midi_swap_audio"].shape == batch["audio_a"].shape
    objective.total.backward()
    assert any(parameter.grad is not None for parameter in model.midi.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_rave_objective_uses_one_mixed_counterfactual_decoder_forward():
    config = _config()
    assert config.latent_loss is not None
    config = replace(config, latent_loss=replace(
        config.latent_loss, rave_swap_source_rejection=0.5,
        clap_counterfactual=1.0))
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.RAVE)
    pitch = _RecordingSwapPitch()
    calls = []
    original_decode = model.decode_latents

    def recording_decode(z_rave, clap, note, velocity, excitation_seed=None):
        calls.append((clap.detach().clone(), note.detach().clone()))
        return original_decode(z_rave, clap, note, velocity, excitation_seed)

    model.decode_latents = recording_decode
    frames = config.data.window_samples // config.data.pitch_hop_length
    batch_size = 4
    batch = {
        "audio_a": torch.randn(batch_size, 1, config.data.window_samples),
        "clap_a": torch.randn(batch_size, config.model.clap_dim),
        "note_a": torch.tensor([48, 60, 52, 64]),
        "note_b": torch.tensor([36, 67, 40, 71]),
        "velocity_a": torch.tensor([80.0, 100.0, 90.0, 70.0]),
        "pitch_confidence_a": torch.ones(batch_size, frames),
        "pitch_valid_mask_a": torch.ones(batch_size, frames, dtype=torch.bool),
        "valid_samples_a": torch.full((batch_size,), config.data.window_samples),
        "excitation_seed_a": torch.arange(batch_size),
        "preset_id": ["a", "a", "b", "b"],
    }

    objective = predictive_stage_objective(
        model, batch, config, PredictiveStage.RAVE, 4, swap_pitch=pitch)

    assert len(calls) == 1
    counterfactual_clap, counterfactual_note = calls[0]
    assert torch.equal(counterfactual_note[:2], batch["note_b"][:2])
    assert torch.equal(counterfactual_note[2:], batch["note_a"][2:])
    assert torch.equal(counterfactual_clap[:2], batch["clap_a"][:2])
    assert not torch.equal(counterfactual_clap[2:], batch["clap_a"][2:])
    assert objective.components["midi_swap_source_rejection"].item() > 0
    diagnostics = objective.diagnostic_tensors
    assert diagnostics is not None
    assert diagnostics["counterfactual_timbre_mask"].sum().item() == 2
    assert diagnostics["counterfactual_audio"].shape == batch["audio_a"].shape


def test_format5_contract_rejects_stage_and_artifact_mismatch():
    config = _config()
    expected = predictive_checkpoint_contract(
        config, PredictiveStage.PREDICTOR, "stats-a", "calibration-a", False)
    validate_predictive_resume({"format": 5, "predictive_contract": expected}, expected)

    for key, value in (("stage", "rollout"),
                       ("latent_statistics_hash", "stats-b"),
                       ("calibration_hash", "calibration-b"),
                       ("history_frames", 99)):
        incompatible = dict(expected)
        incompatible[key] = value
        with pytest.raises(ValueError, match=key.replace("_", " ")):
            validate_predictive_resume(
                {"format": 5, "predictive_contract": incompatible}, expected)


def test_predictive_resume_rejects_legacy_format():
    expected = predictive_checkpoint_contract(
        _config(), PredictiveStage.RAVE, None, None, False)
    with pytest.raises(ValueError, match="format 5"):
        validate_predictive_resume({"format": 4}, expected)


def test_rave_checkpoint_can_warm_start_a_new_rave_finetune(tmp_path: Path):
    config = _config()
    source = _model(config)
    configure_predictive_stage(source, PredictiveStage.RAVE)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    contract = predictive_checkpoint_contract(
        config, PredictiveStage.RAVE, None, None, False)
    checkpoint = tmp_path / "rave.pt"
    save_predictive_checkpoint(
        checkpoint, source, optimizer, scaler, contract,
        stage_update=28_000, epoch=4, microbatch_offset=3,
        scheduler_state={"stage_update": 28_000}, sampler_state={"epoch": 4})

    target = _model(config)
    load_predictive_warm_start(checkpoint, target, config, PredictiveStage.RAVE)

    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)


def test_format5_checkpoint_restores_exact_training_state(tmp_path: Path):
    config = _config()
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.PREDICTOR)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    contract = predictive_checkpoint_contract(
        config, PredictiveStage.PREDICTOR, "stats", "calibration", False)
    original = next(model.predictor.parameters()).detach().clone()
    destination = tmp_path / "predictive.pt"
    save_predictive_checkpoint(
        destination, model, optimizer, scaler, contract,
        stage_update=7, epoch=2, microbatch_offset=3,
        scheduler_state={"last_update": 7}, sampler_state={"epoch": 2})
    with torch.no_grad():
        next(model.predictor.parameters()).zero_()
    restored = load_predictive_checkpoint(
        destination, model, optimizer, scaler, contract)
    assert torch.equal(next(model.predictor.parameters()), original)
    assert restored["stage_update"] == 7
    assert restored["epoch"] == 2
    assert restored["microbatch_offset"] == 3
    assert restored["scheduler_state"] == {"last_update": 7}
    assert restored["sampler_state"] == {"epoch": 2}


def test_predictive_elastic_resume_preserves_training_state_and_resets_rank_state(
        tmp_path: Path):
    config = _config()
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.PREDICTOR)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    contract = predictive_checkpoint_contract(
        config, PredictiveStage.PREDICTOR, "stats", "calibration", False)
    destination = tmp_path / "predictive.pt"
    save_predictive_checkpoint(
        destination, model, optimizer, scaler, contract,
        stage_update=7, epoch=2, microbatch_offset=3,
        scheduler_state={"last_update": 7},
        sampler_state={"training_update": 7})
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    payload["world_size"] = 4
    payload["rng_by_rank"] = payload["rng_by_rank"] * 4
    torch.save(payload, destination)

    with pytest.raises(ValueError, match="same DDP world size"):
        load_predictive_checkpoint(
            destination, model, optimizer, scaler, contract)

    restored = load_predictive_checkpoint(
        destination, model, optimizer, scaler, contract,
        allow_world_size_change=True)
    assert restored["stage_update"] == 7
    assert restored["epoch"] == 3
    assert restored["microbatch_offset"] == 0
    assert restored["sampler_state"] == {"training_update": 7}
    assert restored["rng"] is None
    assert restored["world_size_changed"] is True


def test_predictor_objective_uses_same_recording_future_and_backpropagates():
    config = _config()
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.PREDICTOR)
    latent = torch.randn(2, 16, 32)
    batch = {
        "rave_a": latent,
        "clap_a": torch.randn(2, 512),
        "note_a": torch.tensor([48, 60]),
        "velocity_a": torch.tensor([70.0, 110.0]),
    }
    statistics = LatentStatistics(
        torch.ones(16), torch.ones(16), torch.ones(16))
    objective = predictive_latent_objective(model, batch, config, statistics)
    assert set(objective.components) == {"future", "delta", "acceleration", "overlap"}
    assert objective.prediction.shape == (2, 16, 8)
    objective.total.backward()
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_predictor_counterfactual_rollout_mixes_controls_without_cross_latent_target():
    config = _config()
    assert config.predictive is not None
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.PREDICTOR)
    calls = []

    def recording_decode(z_rave, clap, note, velocity, excitation_seed=None):
        calls.append({
            "z_rave": z_rave.detach().clone(),
            "clap": clap.detach().clone(),
            "note": note.detach().clone(),
        })
        return z_rave.mean(dim=1, keepdim=True).repeat_interleave(
            config.predictive.samples_per_latent, dim=-1)

    model.decode_latents = recording_decode
    batch = {
        "rave_a": torch.randn(4, 16, 24),
        "rave_b": torch.full((4, 16, 24), float("nan")),
        "clap_a": torch.randn(4, 512),
        "note_a": torch.tensor([48, 60, 52, 64]),
        "note_b": torch.tensor([36, 67, 40, 71]),
        "velocity_a": torch.tensor([80.0, 100.0, 90.0, 70.0]),
        "excitation_seed_a": torch.arange(4),
        "preset_id": ["a", "a", "b", "b"],
    }

    result = predictor_counterfactual_rollout(model, batch, config)

    assert len(calls) == 1
    assert torch.equal(result.midi_mask, torch.tensor([True, True, False, False]))
    assert torch.equal(result.timbre_mask, ~result.midi_mask)
    assert torch.equal(result.target_note[:2], batch["note_b"][:2])
    assert torch.equal(result.target_note[2:], batch["note_a"][2:])
    assert torch.equal(result.target_clap[:2], batch["clap_a"][:2])
    assert not torch.equal(result.target_clap[2:], batch["clap_a"][2:])
    assert result.rollout.shape == (
        4, config.predictive.rave_latent_dim,
        config.predictive.predictor_control_rollout_frames)
    assert result.latent_sequence.shape[-1] == (
        config.predictive.history_frames
        + config.predictive.predictor_control_rollout_frames)
    assert torch.isfinite(result.rollout).all()
    assert torch.equal(calls[0]["note"], result.target_note)
    assert torch.equal(calls[0]["clap"], result.target_clap)


def test_rollout_objective_decodes_predicted_future():
    config = _config()
    model = _model(config)
    configure_predictive_stage(model, PredictiveStage.ROLLOUT)
    batch = {
        "audio_a": torch.randn(1, 1, 4096),
        "rave_a": torch.randn(1, 16, 32),
        "clap_a": torch.randn(1, 512),
        "note_a": torch.tensor([60]),
        "velocity_a": torch.tensor([100.0]),
        "excitation_seed_a": torch.tensor([123]),
    }
    statistics = LatentStatistics(
        torch.ones(16), torch.ones(16), torch.ones(16))
    objective = predictive_stage_objective(
        model, batch, config, PredictiveStage.ROLLOUT, 4, statistics)
    assert objective.generated_audio is not None
    assert objective.generated_audio.shape == (1, 1, 2048)
    assert torch.isfinite(objective.total)
    assert objective.components["rollout"].item() >= 0


def test_calibration_artifact_is_bound_to_statistics(tmp_path: Path):
    path = tmp_path / "calibration.json"
    weights = {"future": 1.0, "delta": 0.4,
               "acceleration": 0.02, "overlap": 0.15}
    digest = save_predictive_calibration(path, weights, "stats-a", 128)
    loaded, loaded_digest = load_predictive_calibration(path, "stats-a")
    assert loaded == weights
    assert loaded_digest == digest
    with pytest.raises(ValueError, match="statistics hash"):
        load_predictive_calibration(path, "stats-b")
