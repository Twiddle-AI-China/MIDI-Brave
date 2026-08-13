from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

from midibrave.zrave_flow_config import ZraveFlowConfig

ROOT = Path(__file__).parents[1]
LVZIHAO = ROOT / "scripts" / "lvzihao"
GENERATOR = LVZIHAO / "generate_category_matrix.py"
RECIPES = LVZIHAO / "tiny_midi_category_recipes.yaml"
QUEUE = LVZIHAO / "tiny_midi_categories_v1.queue.tsv"
CLAP_QUEUE = LVZIHAO / "tiny_midi_categories_v1.clap.queue.tsv"
CONFIG_ROOT = (
    ROOT / "configs" / "zrave" / "generated" / "lvzihao_tiny_midi_categories_v1"
)
PURE_CONFIG_ROOT = (
    ROOT / "configs" / "zrave" / "generated" / "lvzihao_tiny_categories_v1"
)
EXPECTED_FAMILIES = {
    "arp": ["Arp"],
    "bass": ["Bass"],
    "chord_pilot": ["Chord"],
    "keys_pilot": ["Keys"],
    "lead": ["Lead"],
    "pad": ["Pad"],
    "pluck": ["Pluck"],
    "synth": ["Synth"],
}


def _rows() -> list[tuple[str, str, str, str, str]]:
    return [
        tuple(row)  # type: ignore[misc]
        for row in csv.reader(
            QUEUE.read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
        if row and not row[0].startswith("#")
    ]


def _queue_rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    return [
        tuple(row)  # type: ignore[misc]
        for row in csv.reader(
            path.read_text(encoding="utf-8").splitlines(),
            delimiter="\t",
        )
        if row and not row[0].startswith("#")
    ]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "test_tiny_midi_category_generator",
        GENERATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tiny_midi_generated_outputs_are_current() -> None:
    import subprocess

    subprocess.run(
        ["python", str(GENERATOR), "--recipes", str(RECIPES), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.stem for path in CONFIG_ROOT.glob("*.yaml")} == set(EXPECTED_FAMILIES)


def test_tiny_midi_configs_match_same_family_pure_initializer() -> None:
    output_roots: set[str] = set()
    for family, categories in EXPECTED_FAMILIES.items():
        midi_path = CONFIG_ROOT / f"{family}.yaml"
        pure_path = PURE_CONFIG_ROOT / f"{family}.yaml"
        midi_raw = yaml.safe_load(midi_path.read_text(encoding="utf-8"))
        pure_raw = yaml.safe_load(pure_path.read_text(encoding="utf-8"))
        config = ZraveFlowConfig.load(midi_path)

        assert midi_raw["data"]["sources"][0]["allowed_categories"] == categories
        assert pure_raw["data"]["sources"][0]["allowed_categories"] == categories
        assert config.model.profile == "tiny"
        assert (
            config.model.d_model,
            config.model.context_layers,
            config.model.future_layers,
            config.model.heads,
            config.model.feedforward_dim,
        ) == (128, 2, 4, 4, 512)
        assert config.model.pitch_conditioning is True
        assert config.model.midi_sequence_conditioning is True
        assert config.segment_sampling.divisions == (2, 4, 8)
        assert config.train.max_updates == 10000
        assert config.train.checkpoint_every == 1000
        assert config.train.validation_every == 1000
        assert config.train.output_root.endswith(
            f"/runs/tiny-midi-categories-v1/{family}"
        )
        output_roots.add(config.train.output_root)

        for key in (
            "latent_dim",
            "profile",
            "context_frames",
            "future_frames",
            "d_model",
            "context_layers",
            "future_layers",
            "heads",
            "feedforward_dim",
        ):
            assert midi_raw["model"][key] == pure_raw["model"][key]
        assert pure_raw["model"]["pitch_conditioning"] is False
        assert pure_raw["model"]["midi_sequence_conditioning"] is False
    assert len(output_roots) == len(EXPECTED_FAMILIES)


def test_tiny_midi_queue_uses_per_family_initializer_and_delays_reports() -> None:
    rows = _rows()
    assert len(rows) == 8 * 3 + 8 * 3
    assert len({row[0] for row in rows}) == len(rows)
    first_report = next(index for index, row in enumerate(rows) if "audition" in row[1])
    assert first_report == 8 * 3
    assert all("audition" not in row[1] for row in rows[:first_report])
    assert all(row[1] == "midi_audition_report_from" for row in rows[first_report:])

    train_rows = [row for row in rows if row[1] == "midi_train_from"]
    assert len(train_rows) == len(EXPECTED_FAMILIES)
    for row in train_rows:
        family = row[3].rsplit("/", 1)[-1]
        assert row[4] == (
            f"10000|runs/tiny-categories-v1/{family}/checkpoints/step-010000.pt"
        )
    assert all(
        row[4].startswith("16,32,64,96,128,160|")
        for row in rows
        if row[1] == "midi_sweep_from"
    )
    for family in EXPECTED_FAMILIES:
        reports = [
            row
            for row in rows
            if row[1] == "midi_audition_report_from"
            and row[3] == f"runs/tiny-midi-categories-v1/{family}"
        ]
        assert [row[4].split("|", 1)[0] for row in reports] == [
            "step-001000.pt",
            "step-005000.pt",
            "step-010000.pt",
        ]
        assert {row[4].split("|", 1)[1] for row in reports} == {
            f"runs/tiny-categories-v1/{family}/checkpoints/step-010000.pt"
        }


def test_generated_clap_queue_is_independent_and_covers_each_audition() -> None:
    audition_ids = [row[0] for row in _rows() if "audition" in row[1]]
    clap_rows = _queue_rows(CLAP_QUEUE)

    assert len(clap_rows) == len(audition_ids) == 24
    assert [row[4] for row in clap_rows] == audition_ids
    assert all(
        row == (f"{audition_id}-clap", "clap_report", "-", "-", audition_id)
        for row, audition_id in zip(clap_rows, audition_ids, strict=True)
    )
    assert all(row[1] != "clap_report" for row in _rows())


def test_all_generated_research_matrices_have_current_clap_queues() -> None:
    for recipe_name in (
        "tiny_category_recipes.yaml",
        "tiny_latent_proxy_recipes.yaml",
        "tiny_midi_category_recipes.yaml",
    ):
        recipe = yaml.safe_load((LVZIHAO / recipe_name).read_text(encoding="utf-8"))
        main_queue = ROOT / recipe["queue_output"]
        suffix = ".queue.tsv"
        clap_queue = main_queue.with_name(
            main_queue.name[: -len(suffix)] + ".clap.queue.tsv"
        )
        audition_ids = [
            row[0] for row in _queue_rows(main_queue) if "audition" in row[1]
        ]
        assert [row[4] for row in _queue_rows(clap_queue)] == audition_ids


def test_generator_rejects_initializer_path_traversal(tmp_path: Path) -> None:
    module = _load_generator()
    module.ROOT = tmp_path
    recipe = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))
    recipe["template"] = "configs/template.yaml"
    recipe["config_output_dir"] = "configs/generated"
    recipe["queue_output"] = "scripts/queue.tsv"
    recipe["initializer_config_root"] = "../foreign"
    recipe_path = tmp_path / "scripts/recipes.yaml"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")

    with pytest.raises(ValueError, match="must not escape"):
        module.materialize(recipe_path)


def test_midi_matrix_shell_contract_keeps_soft_and_hard_gates() -> None:
    import subprocess

    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")
    audition = (LVZIHAO / "midi_audition.sh").read_text(encoding="utf-8")
    common = (LVZIHAO / "common.sh").read_text(encoding="utf-8")
    smoke = (LVZIHAO / "smoke.sh").read_text(encoding="utf-8")
    sweep = (LVZIHAO / "sweep.sh").read_text(encoding="utf-8")
    train = (LVZIHAO / "train.sh").read_text(encoding="utf-8")

    assert "bind_initializer_contract" in runner
    assert "initializer path/hash contract changed" in runner
    assert "per-row initializer actions forbid global LV_INITIALIZE_FROM" in runner
    assert "per-row MIDI actions require global LV_PITCH_PROBE" in runner
    assert "midi_train_from)" in runner
    assert "midi_audition_from|midi_audition_report_from)" in runner
    assert '[[ "$action" == midi_audition_from ]]' in runner
    assert "LV_EXPECTED_INITIALIZER_SHA256" in common
    assert '--initialize-from "$initializer_container"' in smoke
    assert '--initialize-from "$initializer_container"' in sweep
    assert '--initialize-from "$initialize_container"' in train
    assert '--expected-initializer-sha256 "$expected_initializer_sha256"' in train
    assert '--expected-initializer-update "$expected_initializer_update"' in train
    assert "LV_MIDI_AUDITION_FAIL_ON_REJECT" in audition
    assert "--no-fail-on-reject" in audition
    assert "--fail-on-reject" in audition
    assert "--expected-initializer-update" in audition
    assert '--categories "${ALLOWED_CATEGORIES[@]}"' in audition
    subprocess.run(
        [
            "bash",
            "-n",
            str(LVZIHAO / "common.sh"),
            str(LVZIHAO / "allocation_runner.sh"),
            str(LVZIHAO / "smoke.sh"),
            str(LVZIHAO / "sweep.sh"),
            str(LVZIHAO / "train.sh"),
            str(LVZIHAO / "midi_audition.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_runtime_initializer_resolution_hashes_exact_checkpoint(
    tmp_path: Path,
) -> None:
    import subprocess

    checkpoint = (
        tmp_path
        / "runs"
        / "tiny-categories-v1"
        / "arp"
        / "checkpoints"
        / "step-010000.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"same-family-pure-checkpoint")
    relative = checkpoint.relative_to(tmp_path).as_posix()
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; resolve_persistent_initializer "$2"; '
                "printf '%s\\n%s\\n%s\\n' "
                '"$RESOLVED_INITIALIZER_PATH" '
                '"$RESOLVED_INITIALIZER_SHA256" '
                '"$RESOLVED_INITIALIZER_UPDATE"'
            ),
            "bash",
            str(LVZIHAO / "common.sh"),
            relative,
        ],
        env={**os.environ, "LV_WORK_ROOT": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        str(checkpoint),
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "10000",
    ]


def test_runtime_initializer_resolution_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    import subprocess

    outside = tmp_path / "outside" / "step-010000.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"wrong-family-or-root")
    link = tmp_path / "runs" / "arp" / "checkpoints" / "step-010000.pt"
    link.parent.mkdir(parents=True)
    os.symlink(outside, link)
    command = 'source "$1"; resolve_persistent_initializer "$2"'

    traversal = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(LVZIHAO / "common.sh"),
            "runs/arp/../../outside/step-010000.pt",
        ],
        env={**os.environ, "LV_WORK_ROOT": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )
    symlink = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(LVZIHAO / "common.sh"),
            link.relative_to(tmp_path).as_posix(),
        ],
        env={**os.environ, "LV_WORK_ROOT": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert traversal.returncode == 2
    assert "relative path traversal is not allowed" in traversal.stderr
    assert symlink.returncode == 2
    assert "resolves outside LV_WORK_ROOT/runs" in symlink.stderr
