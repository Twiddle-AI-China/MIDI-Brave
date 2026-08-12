from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
LVZIHAO = ROOT / "scripts" / "lvzihao"
PURE_QUEUE = LVZIHAO / "segment248_pure.queue.example.tsv"
MIDI_QUEUE = LVZIHAO / "segment248_midi.queue.example.tsv"
BASELINE_QUEUE = LVZIHAO / "pure_flow.queue.example.tsv"
DOC = ROOT / "docs" / "LVZIHAO_TRAINING.md"
TRAIN = LVZIHAO / "train.sh"
AUDITION = LVZIHAO / "audition.sh"
SMOKE = LVZIHAO / "smoke.sh"
MIDI_AUDITION = LVZIHAO / "midi_audition.sh"
BUILD_IMAGE = LVZIHAO / "build_image.sh"
DOCKERFILE = ROOT / "Dockerfile.lvzihao"


def _rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    values: list[tuple[str, str, str, str, str]] = []
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"):
        if not row or row[0].startswith("#"):
            continue
        assert len(row) == 5
        values.append(tuple(row))  # type: ignore[arg-type]
    return values


def test_segment248_pure_queue_has_conservative_single_gpu_sequence() -> None:
    config = "configs/zrave/lvzihao_serum128_segment248_pure.yaml"
    run = "runs/segment248-pure"

    assert _rows(PURE_QUEUE) == [
        ("segment248-pure-smoke", "smoke", config, run, "-"),
        (
            "segment248-pure-sweep",
            "sweep",
            config,
            run,
            "1,2,4,8,12,16",
        ),
        ("segment248-pure-train", "train", config, run, "20000"),
        (
            "segment248-pure-audition-005000",
            "audition",
            config,
            run,
            "step-005000.pt",
        ),
        (
            "segment248-pure-audition-010000",
            "audition",
            config,
            run,
            "step-010000.pt",
        ),
        (
            "segment248-pure-audition-015000",
            "audition",
            config,
            run,
            "step-015000.pt",
        ),
        (
            "segment248-pure-audition-020000",
            "audition",
            config,
            run,
            "step-020000.pt",
        ),
    ]
    assert "LV_INITIALIZE_FROM" in PURE_QUEUE.read_text(encoding="utf-8")


def test_segment248_midi_queue_qualifies_before_training() -> None:
    config = "configs/zrave/lvzihao_serum128_segment248_midi32.yaml"
    run = "runs/segment248-midi32"

    assert _rows(MIDI_QUEUE) == [
        (
            "segment248-midi-pitch",
            "pitch_train",
            config,
            run,
            "10000",
        ),
        ("segment248-midi-smoke", "smoke", config, run, "-"),
        ("segment248-midi-sweep", "sweep", config, run, "1,2,4,8"),
        ("segment248-midi-train", "train", config, run, "20000"),
        (
            "segment248-midi-audition-005000",
            "midi_audition",
            config,
            run,
            "step-005000.pt",
        ),
        (
            "segment248-midi-audition-010000",
            "midi_audition",
            config,
            run,
            "step-010000.pt",
        ),
        (
            "segment248-midi-audition-015000",
            "midi_audition",
            config,
            run,
            "step-015000.pt",
        ),
        (
            "segment248-midi-audition-020000",
            "midi_audition",
            config,
            run,
            "step-020000.pt",
        ),
    ]
    assert "LV_INITIALIZE_FROM" in MIDI_QUEUE.read_text(encoding="utf-8")
    assert "LV_PITCH_PROBE" in MIDI_QUEUE.read_text(encoding="utf-8")


def test_queue_configs_bind_the_declared_run_and_conditioning() -> None:
    cases = (
        (PURE_QUEUE, "runs/segment248-pure", False, False),
        (MIDI_QUEUE, "runs/segment248-midi32", True, True),
    )
    for queue, run, pitch, sequence in cases:
        config_relative = _rows(queue)[0][2]
        payload = yaml.safe_load(
            (ROOT / config_relative).read_text(encoding="utf-8")
        )
        assert payload["train"]["output_root"] == (
            "/data/midibrave-zrave-flow-serum128/" + run
        )
        assert payload["train"]["max_updates"] == 20000
        assert payload["model"]["latent_dim"] == 128
        assert payload["model"]["pitch_conditioning"] is pitch
        assert payload["model"]["midi_sequence_conditioning"] is sequence
        assert payload["segment_sampling"] == {
            "enabled": True,
            "divisions": [2, 4, 8],
            "include_first": False,
        }


def test_octopus_85k_queue_is_explicitly_baseline_only() -> None:
    text = BASELINE_QUEUE.read_text(encoding="utf-8")
    documentation = DOC.read_text(encoding="utf-8")

    assert "BASELINE-ONLY" in text
    assert all(row[1] != "train" for row in _rows(BASELINE_QUEUE))
    assert "85k" in documentation
    assert "intentionally has no `train`" in documentation
    assert "segment248_midi.queue.example.tsv" in documentation
    assert 'export LV_INITIALIZE_FROM="$HOST_FLOW_ROOT/' in documentation
    assert 'export LV_PITCH_PROBE="$HOST_FLOW_ROOT/' in documentation
    assert "step-005000.pt" in documentation


def test_new_segment_run_requires_weight_only_initializer() -> None:
    script = TRAIN.read_text(encoding="utf-8")

    assert "segment_enabled=" in script
    assert "new segment runs require LV_INITIALIZE_FROM" in script


def test_pure_audition_requires_the_long_rollout_gate() -> None:
    script = AUDITION.read_text(encoding="utf-8")

    assert "python -m midibrave.zrave_flow_gate" in script
    assert '--output "$partial_container/gate.json"' in script
    assert "--fail-on-reject" in script
    assert 'gate.get("passed") is True' in script


def test_smoke_covers_transition_and_weight_initializer() -> None:
    script = SMOKE.read_text(encoding="utf-8")

    assert "--batch-per-gpu 5" in script
    assert "LV_INITIALIZE_FROM" in script
    assert '--initialize-from "$initializer_container"' in script


def test_midi_audition_strict_contract_and_gate() -> None:
    script = MIDI_AUDITION.read_text(encoding="utf-8")
    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")

    assert "LV_SOURCE_ROOT is required" in script
    assert "LV_PITCH_PROBE is required" in script
    assert "LV_INITIALIZE_FROM is required" in script
    assert 'checkpoint_sha=$(sha256sum "$checkpoint"' in script
    assert 'initializer_sha=$(sha256sum "$LV_INITIALIZE_FROM"' in script
    assert "render_zrave_midi_flow_audition.py" in script
    assert '--expected-checkpoint-sha256 "$checkpoint_sha"' in script
    assert '--expected-initializer-sha256 "$initializer_sha"' in script
    assert "python -m midibrave.zrave_flow_gate" in script
    assert '--output "$partial_container/gate.json"' in script
    assert "--fail-on-reject" in script
    assert "does not yet" in script
    assert "midi_audition)" in runner
    assert '"$script_dir/midi_audition.sh"' in runner


def test_image_build_keeps_official_default_and_explicit_local_fallback() -> None:
    build = BUILD_IMAGE.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    documentation = DOC.read_text(encoding="utf-8")

    assert "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:" in build
    assert '"EXPECTED_TORCH_PREFIX=$LV_EXPECTED_TORCH_PREFIX"' in build
    assert "EXPECTED_TORCH_PREFIX=2.9.1" in dockerfile
    assert "uv pip install --python" in dockerfile
    assert "c7391f0e1b06c723486015aa53641883cd6a46ba167cf13918d4acadc6c75d77" in documentation
    assert "LV_EXPECTED_TORCH_PREFIX=2.10" in documentation
