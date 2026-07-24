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
ROLLOUT_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_standalone50_rollout.yaml"
)
ROLLOUT_SWEEP = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_rollout_sweep.sbatch"
)
ROLLOUT_TRAIN = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_rollout_train.sbatch"
)
ROLLOUT_EVALUATE = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_rollout_evaluate.sbatch"
)
ROLLOUT_AUDITION = (
    ROOT / "scripts" / "cloud" / "octopus_zrave_rollout_audition.sbatch"
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
    assert "zrave_resume" in scripts
    for batch in (1024, 1536, 2048, 2560):
        assert str(batch) in scripts
    assert "--batches 1024 1536 2048 2560" in scripts


def test_training_validation_horizon_is_config_driven() -> None:
    source = (
        ROOT / "src" / "midibrave" / "zrave_train.py"
    ).read_text(encoding="utf-8")

    assert "horizon_frames=128" not in source
    assert "config.train.validation_rollout_frames" in source


def test_rollout_config_keeps_inference_contract_and_samples_64_frames() -> None:
    config = ZraveConfig.load(ROLLOUT_CONFIG)

    assert config.model.context_frames == 32
    assert config.model.horizon_frames == 8
    assert config.train.rollout_max_depth == 7
    assert config.train.rollout_curriculum_updates == 500
    assert config.train.rollout_teacher_probability == 0.25
    assert config.model.horizon_frames * (
        config.train.rollout_max_depth + 1
    ) == 64
    assert config.train.output_root.endswith(
        "zrave_transformer_rollout_v3"
    )


def test_rollout_jobs_use_octopus_slurm_contract() -> None:
    sweep = ROLLOUT_SWEEP.read_text(encoding="utf-8")
    train = ROLLOUT_TRAIN.read_text(encoding="utf-8")

    for script in (sweep, train):
        lowered = script.casefold()
        assert "#SBATCH --partition=gpu8" in script
        assert "#SBATCH --gres=gpu:8" in script
        assert "srun --kill-on-bad-exit=1" in script
        assert "torchrun --standalone --nproc_per_node=8" in script
        assert "octopus_standalone50_rollout.yaml" in script
        assert ":ro" in script
        assert "zhongwei" not in lowered
        assert "clap" not in lowered
        assert "midi_control" not in lowered
    assert "for batch in 512 1024 1536 2048" in sweep
    assert "--benchmark-updates 100" in sweep
    assert "--warm-start" in train
    assert "zrave_transformer_mean50_v2/checkpoints/best.pt" in train


def test_rollout_evaluation_jobs_emit_latent_and_audio_reports() -> None:
    evaluate = ROLLOUT_EVALUATE.read_text(encoding="utf-8")
    audition = ROLLOUT_AUDITION.read_text(encoding="utf-8")

    for script in (evaluate, audition):
        assert "#SBATCH --partition=gpu1" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "srun --kill-on-bad-exit=1" in script
        assert "octopus_standalone50_rollout.yaml" in script
        assert "zrave_transformer_rollout_v3" in script
    assert "for split in validation test" in evaluate
    assert "-m midibrave.zrave_evaluate" in evaluate
    assert "render_zrave_audition.py" in audition
    assert "-m midibrave.zrave_audio_quality" in audition
    assert "audio-quality.json" in audition
