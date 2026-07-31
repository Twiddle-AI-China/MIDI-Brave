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
PURE_AUDITION = CLOUD / "octopus_zrave_pure_flow_audition.sbatch"
PURE_JOBS = (PURE_SWEEP, PURE_TRAIN)
SERUM128_CONFIG = ROOT / "configs" / "zrave" / "octopus_serum128_pure_flow.yaml"
SERUM128_PREPARE = CLOUD / "octopus_serum128_flow_prepare.sbatch"
SERUM128_SWEEP = CLOUD / "octopus_serum128_flow_sweep.sbatch"
SERUM128_TRAIN = CLOUD / "octopus_serum128_flow_train.sbatch"
SERUM128_JOBS = (SERUM128_PREPARE, SERUM128_SWEEP, SERUM128_TRAIN)


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


def test_pure_train_forwards_phase3_start_only_with_resume() -> None:
    script = PURE_TRAIN.read_text(encoding="utf-8")

    assert "PURE_FLOW_PHASE3_START_UPDATE" in script
    assert 'phase3_start=${PURE_FLOW_PHASE3_START_UPDATE:-}' in script
    assert '[[ ! "$phase3_start" =~ ^[0-9]+$ ]]' in script
    assert '[[ -z "$resume_name" ]]' in script
    assert (
        'resume_args+=(--phase3-start-update "$phase3_start")'
        in script
    )


def test_pure_audition_is_explicit_one_gpu_read_only_job() -> None:
    script = PURE_AUDITION.read_text(encoding="utf-8")
    lowered = script.casefold()

    assert "#SBATCH --partition=gpu1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "srun --kill-on-bad-exit=1" in script
    assert "docker run --rm" in script
    assert (
        "/srv/data-branches/nvme/datasets:/data/datasets:ro"
        in script
    )
    assert (
        "/data/midibrave-zrave-standalone:"
        "/data/midibrave-zrave-standalone:ro"
        in script
    )
    assert "ZRAVE_PURE_FLOW_AUDITION_CHECKPOINT" in script
    assert "checkpoint must be a basename ending in .pt" in script
    assert "octopus_pure_flow_poc.yaml" in script
    assert "-m midibrave.zrave_pure_flow_audition" in script
    assert "pure-flow-poc-v1/auditions" in script
    assert "--midi" not in lowered
    assert "midi_note" not in lowered
    assert "pitch" not in lowered
    assert "clap" not in lowered


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


def test_serum128_jobs_are_pure_zrave_and_keep_sources_read_only() -> None:
    scripts = [path.read_text(encoding="utf-8") for path in SERUM128_JOBS]

    for script in scripts:
        lowered = script.casefold()
        assert "#SBATCH --gres=gpu:8" in script
        assert "srun --kill-on-bad-exit=1" in script
        assert "docker run --rm" in script
        assert "octopus_serum128_pure_flow.yaml" in script
        assert (
            "/data/midibrave-rave-serum-v2:"
            "/data/midibrave-rave-serum-v2:ro"
        ) in script
        assert "/data/datasets/Timbre_A/serum-octopus-v1:/source:ro" in script
        assert "clap" not in lowered
        assert "midi_note" not in lowered
        assert "--pitch-probe" not in lowered
        assert "zhongwei" not in lowered
    assert "latent-cosmos-rave:serum-v2-rate-v3" in scripts[0]
    assert "3a158421567618860808d7f92d301f222b7243591a493ca53bd285cd9ad9bde7" in scripts[0]


def test_serum128_sweep_measures_four_batches_for_100_updates() -> None:
    script = SERUM128_SWEEP.read_text(encoding="utf-8")

    assert "for batch in 64 96 128 160" in script
    assert "--benchmark-warmup 20" in script
    assert "--benchmark-updates 100" in script
    assert "--batches 64 96 128 160" in script
    assert "median_valid_latent_frames_per_second" in script


def test_serum128_training_runs_100k_and_checkpoints_every_5k() -> None:
    config = ZraveFlowConfig.load(SERUM128_CONFIG)
    script = SERUM128_TRAIN.read_text(encoding="utf-8")

    assert config.train.max_updates == 100000
    assert config.train.checkpoint_every == 5000
    assert "--max-updates 100000" in script
    assert "--nproc_per_node=8" in script
    assert "latest-selection.json" in script
