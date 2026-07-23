from __future__ import annotations

from pathlib import Path

import pytest

from midibrave.config import Config


def test_v2_config_remains_legacy():
    config = Config.load("configs/v2/base.yaml")
    assert config.predictive is None
    assert config.latent_loss is None
    assert not config.is_predictive


def test_v3_predictive_contract_is_exact():
    config = Config.load("configs/v3/base.yaml")
    assert config.is_predictive
    assert config.predictive is not None
    assert config.latent_loss is not None
    assert config.predictive.architecture == "predictive_rave_v1"
    assert config.predictive.rave_latent_dim == 16
    assert config.predictive.clap_control_dim == 256
    assert config.predictive.history_frames == 16
    assert config.predictive.horizon_frames == 8
    assert config.predictive.stride_frames == 4
    assert config.predictive.samples_per_latent == 128
    assert config.predictive.predictor_control_start_updates == 1000
    assert config.predictive.predictor_control_every_updates == 4
    assert config.predictive.predictor_control_warmup_updates == 1000
    assert config.predictive.predictor_control_rollout_frames == 128
    assert config.predictive.predictor_control_gradient_fraction_max == 0.25
    assert config.predictive.predictor_timbre_interpolation_steps == 4
    assert config.latent_loss.future == 1.0
    assert config.latent_loss.delta == 0.5
    assert config.latent_loss.acceleration == 0.05
    assert config.latent_loss.predictor_rollout_stability == 0.0
    assert config.latent_loss.predictor_rollout_style == 0.0
    assert config.latent_loss.predictor_midi_style == 0.0
    assert config.latent_loss.predictor_timbre_style == 0.0
    assert config.latent_loss.predictor_seed_washout == 0.0


def test_phase2c_enables_control_recovery_without_small_batches():
    config = Config.load("configs/v3/octopus_pad50_phase2c.yaml")
    assert config.predictive is not None
    assert config.latent_loss is not None
    assert config.train.batch_per_gpu == 64
    assert config.predictive.predictor_control_gradient_fraction_max == 0.20
    assert config.latent_loss.predictor_rollout_style == 1.0
    assert config.latent_loss.predictor_rollout_stability == 0.5
    assert config.train.checkpoint_every == 500


def test_phase3_preserves_controls_and_has_a_dedicated_sweep_config():
    formal = Config.load("configs/v3/octopus_pad50_phase3.yaml")
    sweep = Config.load("configs/v3/octopus_pad50_phase3_sweep.yaml")

    assert formal.predictive is not None
    assert formal.latent_loss is not None
    assert formal.train.run_name == "pad50_phase3_control_preserving_rollout"
    assert formal.train.checkpoint_every == 500
    assert formal.latent_loss.clap_counterfactual == 1.0
    assert formal.latent_loss.predictor_midi_style == 1.0
    assert formal.predictive.predictor_control_start_updates == 0
    assert formal.predictive.predictor_control_every_updates == 4
    assert sweep.train.run_name == "pad50_phase3_batch_sweep"
    assert sweep.train.rollout_steps == 6
    assert sweep.train.log_every == 1


def test_control_rescue_uses_pitch_matched_window_clap_targets():
    config = Config.load(
        "configs/v3/octopus_scratch_pad_control_rescue.yaml")

    assert config.predictive is not None
    assert config.train.run_name == (
        "pad50_scratch_control_rescue_window_clap_v1")
    assert config.train.batch_per_gpu == 32
    assert config.train.rollout_steps == 1000
    assert config.train.checkpoint_every == 500
    assert config.predictive.predictor_timbre_exact_note
    assert config.predictive.predictor_window_clap_targets


def test_v3_rejects_stride_beyond_horizon(tmp_path: Path):
    text = Path("configs/v3/smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("stride_frames: 4", "stride_frames: 9"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="stride_frames"):
        Config.load(path)


def test_cached_rave_teacher_requires_fully_valid_audio_windows(tmp_path: Path):
    text = Path("configs/v3/smoke.yaml").read_text(encoding="utf-8")
    text = text.replace("minimum_valid_samples: 4096", "minimum_valid_samples: 2048")
    text = text.replace("require_rave_cache: false", "require_rave_cache: true")
    path = tmp_path / "partial-cached-window.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="require_rave_cache.*fully valid"):
        Config.load(path)


@pytest.mark.parametrize("old,new,match", [
    ("predictor_control_every_updates: 1", "predictor_control_every_updates: 0",
     "predictor_control_every_updates"),
    ("predictor_control_warmup_updates: 1", "predictor_control_warmup_updates: 0",
     "predictor_control_warmup_updates"),
    ("predictor_control_rollout_frames: 32", "predictor_control_rollout_frames: 6",
     "predictor_control_rollout_frames"),
    ("predictor_control_gradient_fraction_max: 0.25",
     "predictor_control_gradient_fraction_max: 1.1",
     "predictor_control_gradient_fraction_max"),
    ("predictor_timbre_interpolation_steps: 4",
     "predictor_timbre_interpolation_steps: 0",
     "predictor_timbre_interpolation_steps"),
])
def test_v3_rejects_invalid_predictor_control_schedule(
        tmp_path: Path, old: str, new: str, match: str):
    text = Path("configs/v3/smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad-control.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        Config.load(path)


@pytest.mark.parametrize("field", [
    "predictor_rollout_stability",
    "predictor_rollout_style",
    "predictor_midi_style",
    "predictor_timbre_style",
    "predictor_seed_washout",
])
def test_v3_rejects_negative_predictor_stability_weights(
        tmp_path: Path, field: str):
    text = Path("configs/v3/smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "negative-stability.yaml"
    path.write_text(text.replace(f"{field}: 0.0", f"{field}: -0.1"),
                    encoding="utf-8")

    with pytest.raises(ValueError, match="predictor stability"):
        Config.load(path)
