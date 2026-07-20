from pathlib import Path

import pytest
import torch

from midibrave.predictive_losses import LatentStatistics

from midibrave.config import Config
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.trainer import (PredictiveStage, configure_predictive_stage,
                               predictive_checkpoint_contract,
                               predictive_loss_schedule,
                               predictive_latent_objective,
                               predictive_stage_objective,
                               load_predictive_checkpoint,
                               load_predictive_calibration,
                               save_predictive_calibration,
                               save_predictive_checkpoint,
                               validate_predictive_resume)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> Config:
    return Config.load(ROOT / "configs/v3/smoke.yaml")


def _model(config: Config) -> PredictiveMidiBrave:
    assert config.predictive is not None
    return PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)


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
