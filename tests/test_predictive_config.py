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
    assert config.latent_loss.future == 1.0
    assert config.latent_loss.delta == 0.5
    assert config.latent_loss.acceleration == 0.05


def test_v3_rejects_stride_beyond_horizon(tmp_path: Path):
    text = Path("configs/v3/smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("stride_frames: 4", "stride_frames: 9"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="stride_frames"):
        Config.load(path)
