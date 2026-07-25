from __future__ import annotations

from pathlib import Path

from midibrave.zrave_flow_config import ZraveFlowConfig


ROOT = Path(__file__).parents[1]
CLOUD = ROOT / "scripts" / "cloud"
JOBS = (
    CLOUD / "octopus_zrave_flow_prepare.sbatch",
    CLOUD / "octopus_zrave_pitch_train.sbatch",
    CLOUD / "octopus_zrave_flow_sweep.sbatch",
    CLOUD / "octopus_zrave_flow_train.sbatch",
    CLOUD / "octopus_zrave_flow_evaluate.sbatch",
)
PURE_SWEEP = CLOUD / "octopus_zrave_pure_flow_sweep.sbatch"
PURE_TRAIN = CLOUD / "octopus_zrave_pure_flow_train.sbatch"
PURE_JOBS = (PURE_SWEEP, PURE_TRAIN)


def test_flow_jobs_use_only_octopus_slurm_gpu_contract() -> None:
    scripts = {
        path.name: path.read_text(encoding="utf-8") for path in JOBS
    }
    for script in scripts.values():
        lowered = script.casefold()
        assert "srun --kill-on-bad-exit=1" in script
        assert "docker run --rm" in script
        assert "zhongwei" not in lowered
        assert "clap" not in lowered
        assert (
            "/srv/data-branches/nvme/datasets:/data/datasets:ro"
            in script
        )
        assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert (
        "#SBATCH --gres=gpu:8"
        in scripts["octopus_zrave_flow_prepare.sbatch"]
    )
    assert (
        "#SBATCH --gres=gpu:8"
        in scripts["octopus_zrave_pitch_train.sbatch"]
    )
    assert (
        "#SBATCH --gres=gpu:8"
        in scripts["octopus_zrave_flow_sweep.sbatch"]
    )
    assert (
        "#SBATCH --gres=gpu:8"
        in scripts["octopus_zrave_flow_train.sbatch"]
    )
    assert (
        "#SBATCH --gres=gpu:1"
        in scripts["octopus_zrave_flow_evaluate.sbatch"]
    )
    assert (
        "--split test"
        in scripts["octopus_zrave_flow_evaluate.sbatch"]
    )


def test_sweep_has_four_candidates_and_100_measured_updates() -> None:
    script = (
        CLOUD / "octopus_zrave_flow_sweep.sbatch"
    ).read_text(encoding="utf-8")

    assert "for batch in 128 256 384 512" in script
    assert "--benchmark-warmup 20" in script
    assert "--benchmark-updates 100" in script
    assert "--benchmark-exposure-updates 5" in script
    assert "--batches 128 256 384 512" in script
    assert (
        "--metric median_valid_latent_frames_per_second" in script
    )


def test_pure_jobs_have_no_pitch_dependency() -> None:
    scripts = [
        path.read_text(encoding="utf-8") for path in PURE_JOBS
    ]

    for script in scripts:
        lowered = script.casefold()
        assert "#SBATCH --gres=gpu:8" in script
        assert "srun --kill-on-bad-exit=1" in script
        assert "docker run --rm" in script
        assert (
            "/srv/data-branches/nvme/datasets:/data/datasets:ro"
            in script
        )
        assert "--pitch-probe" not in script
        assert "pitch-probe" not in lowered
        assert "qualified.complete" not in lowered
        assert "preparation.complete" in script
        assert "packs/all-synth/index.json" in script


def test_pure_sweep_measures_four_candidates_for_100_updates() -> None:
    script = PURE_SWEEP.read_text(encoding="utf-8")

    assert "for batch in 128 256 384 512" in script
    assert "--benchmark-warmup 20" in script
    assert "--benchmark-updates 100" in script
    assert "--benchmark-exposure-updates 5" in script
    assert "--batches 128 256 384 512" in script
    assert (
        "--metric median_valid_latent_frames_per_second" in script
    )
    assert "sweeps/pure-flow-poc" in script


def test_pure_train_runs_100k_updates_from_selected_batch() -> None:
    script = PURE_TRAIN.read_text(encoding="utf-8")

    assert "--max-updates 100000" in script
    assert "--nproc_per_node=8" in script
    assert "pure-flow-poc-v1" in script
    assert "latest-selection.json" in script
    assert "statistics.npz" in script


def test_formal_config_uses_only_approved_sources_and_output_root() -> None:
    config = ZraveFlowConfig.load(
        ROOT / "configs" / "zrave" / "octopus_flow.yaml"
    )

    assert [source.name for source in config.data.sources] == [
        "serum_full",
        "pianobook_pitch",
        "dexed_surge_broad",
    ]
    assert config.data.sources[0].manifest.endswith(
        "/TimbreDatasets/registry/timbreclap_dataset.db"
    )
    assert (
        config.data.sources[0].registry_dataset_id
        == "serum-octopus-v1"
    )
    assert config.data.sources[1].registry_dataset_id == "pianobook"
    assert config.data.sources[1].audio_root.endswith(
        "/Timbre_A/derived/pianobook"
    )
    assert all(
        source.maximum_future_frames == 64
        for source in config.data.sources
    )
    assert (
        config.train.output_root
        == "/data/midibrave-zrave-flow/runs/flow-v1"
    )
