from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
LVZIHAO = ROOT / "scripts" / "lvzihao"
PITCH_TRAIN = LVZIHAO / "pitch_train.sh"
RUNNER = LVZIHAO / "allocation_runner.sh"
QUEUE = LVZIHAO / "pitch_probe.queue.example.tsv"
DOC = ROOT / "docs" / "LVZIHAO_TRAINING.md"


def test_lvzihao_pitch_queue_shell_syntax_is_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(PITCH_TRAIN), str(RUNNER)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pitch_action_reuses_timed_qgpu_container_contract() -> None:
    script = PITCH_TRAIN.read_text(encoding="utf-8")

    assert "assert_config_contract" in script
    assert "max_updates <= 10000" in script
    assert '[[ "$latent_dim" == 128 ]]' in script
    assert "run_timed_gpu_container" in script
    assert "python -m midibrave.zrave_pitch_train" in script
    assert '--config "$config"' in script
    assert '--batch-per-gpu "$LV_PITCH_BATCH_PER_GPU"' in script
    assert '--max-updates "$max_updates"' in script
    assert "docker run" not in script
    assert "qgpu" not in script


def test_pitch_action_resumes_and_requires_qualified_checkpoint() -> None:
    script = PITCH_TRAIN.read_text(encoding="utf-8")

    assert 'latest=$(latest_checkpoint "$checkpoint_host")' in script
    assert 'resume_args=(--resume "$resume_container")' in script
    assert "if (( status == 75 ))" in script
    assert 'report.get("passed") is True' in script
    assert "checkpoint.is_file()" in script
    assert 'report.get("config_sha256") == digest(sys.argv[3])' in script
    assert 'report.get("pack_index_sha256") == digest(sys.argv[4])' in script
    assert 'require_file "$qualification_host"' in script
    assert "verify_qualification ||" in script


def test_allocation_runner_and_example_expose_pitch_train_action() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    queue = QUEUE.read_text(encoding="utf-8")
    documentation = DOC.read_text(encoding="utf-8")

    assert "pitch_train)" in runner
    assert '"$script_dir/pitch_train.sh"' in runner
    assert "\tpitch_train\t" in queue
    assert "lvzihao_serum128_segment248_midi32.yaml" in queue
    assert "qualification.json" in documentation
    assert "LV_PITCH_BATCH_PER_GPU" in documentation
