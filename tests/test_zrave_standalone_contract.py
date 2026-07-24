from __future__ import annotations

from pathlib import Path

from midibrave.zrave_config import ZraveConfig


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "zrave" / "octopus_standalone50.yaml"
PREPARE = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_standalone_prepare.sbatch"
)
SWEEP = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_standalone_sweep.sbatch"
)
TRAIN = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_standalone_train.sbatch"
)
EVALUATE = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_standalone_evaluate.sbatch"
)


def test_standalone_config_has_the_measured_codec_contract() -> None:
    config = ZraveConfig.load(CONFIG)

    assert config.rave.sample_rate == 44100
    assert config.rave.expected_sha256 == (
        "3ec093e132ce75d7fee3b8b734c739ebf8711a57ee332a60bce4359e2e34073e"
    )
    assert config.data.cache_root.endswith("serum-balanced50-mean-v2")
    assert config.data.packed_root.endswith("serum-balanced50-mean-v2")
    assert config.train.output_root.endswith("zrave_transformer_mean50_v2")
    assert config.data.latent_hop == 2048
    assert config.data.warmup_frames == 4
    assert config.model.latent_dim == 16
    assert config.model.context_frames == 32
    assert config.model.horizon_frames == 8
    assert config.train.max_updates == 3000
    assert config.train.warmup_updates == 200
    assert config.train.checkpoint_every == 500
    assert config.train.validation_every == 250
    assert config.train.validation_batches == 4
    assert config.train.validation_rollout_frames == 64


def test_standalone_cloud_jobs_ban_conditional_dependencies() -> None:
    scripts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PREPARE, SWEEP, TRAIN, EVALUATE)
    ).casefold()

    assert "predictive" not in scripts
    assert "clap" not in scripts
    assert "midi_control" not in scripts
    assert "/data:/data" not in scripts
    assert "octopus_standalone50.yaml" in scripts
    assert scripts.count("#sbatch --gres=gpu:8") == 3
    assert "#sbatch --gres=gpu:1" in scripts
    assert "--benchmark-updates 100" in scripts
    assert "serum-balanced50-mean-v2" in scripts
    assert "zrave_transformer_mean50_v2" in scripts
    assert "posterior_mean_temp0_reset_v1" in scripts
    for batch in (1024, 1536, 2048, 2560):
        assert str(batch) in scripts
    assert "--batches 1024 1536 2048 2560" in scripts


def test_training_validation_horizon_is_config_driven() -> None:
    source = (
        ROOT / "src" / "midibrave" / "zrave_train.py"
    ).read_text(encoding="utf-8")

    assert "horizon_frames=128" not in source
    assert "config.train.validation_rollout_frames" in source
