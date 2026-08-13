from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from midibrave.zrave_flow_config import ZraveFlowConfig

ROOT = Path(__file__).parents[1]
LVZIHAO = ROOT / "scripts" / "lvzihao"
GENERATOR = LVZIHAO / "generate_category_matrix.py"
RECIPES = LVZIHAO / "tiny_category_recipes.yaml"
QUEUE = LVZIHAO / "tiny_categories_v1.queue.tsv"
TEMPLATE = ROOT / "configs" / "zrave" / ("lvzihao_serum128_tiny_category_template.yaml")
CONFIG_ROOT = ROOT / "configs" / "zrave" / "generated" / ("lvzihao_tiny_categories_v1")
EXPECTED_FAMILIES = {
    "arp": ["Arp"],
    "bass": ["Bass"],
    "fx": ["FX"],
    "lead": ["Lead"],
    "pad": ["Pad"],
    "pluck": ["Pluck"],
    "synth": ["Synth"],
    "chord_pilot": ["Chord"],
    "keys_pilot": ["Keys"],
    "lead_pluck": ["Lead", "Pluck"],
    "keys_harmony": ["Arp", "Chord", "Keys"],
}


def _rows() -> list[tuple[str, str, str, str, str]]:
    rows = csv.reader(
        QUEUE.read_text(encoding="utf-8").splitlines(),
        delimiter="\t",
    )
    return [
        tuple(row)  # type: ignore[misc]
        for row in rows
        if row and not row[0].startswith("#")
    ]


def test_category_matrix_generated_outputs_are_current() -> None:
    subprocess.run(
        ["python", str(GENERATOR), "--recipes", str(RECIPES), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.stem for path in CONFIG_ROOT.glob("*.yaml")} == set(EXPECTED_FAMILIES)


def test_category_matrix_accepts_json_recipes(tmp_path: Path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "test_category_matrix_generator",
        GENERATOR,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    module.ROOT = tmp_path
    template = tmp_path / "configs/zrave/template.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    recipe = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))
    recipe["template"] = "configs/zrave/template.yaml"
    recipe["config_output_dir"] = "configs/zrave/generated"
    recipe["queue_output"] = "scripts/lvzihao/category.queue.tsv"
    recipe_path = tmp_path / "scripts/lvzihao/recipes.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    outputs = module.materialize(recipe_path)

    assert tmp_path / "configs/zrave/generated/bass.yaml" in outputs
    assert tmp_path / "scripts/lvzihao/category.queue.tsv" in outputs


def test_category_recipes_materialize_tiny_pure_independent_runs() -> None:
    recipe = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))

    assert {
        value["id"]: value["categories"] for value in recipe["families"]
    } == EXPECTED_FAMILIES
    assert recipe["maximum_updates"] == 10000
    assert recipe["checkpoints"] == [1000, 5000, 10000]
    assert recipe["sweep_batches"] == [16, 32, 64, 96, 128, 160]
    output_roots: set[str] = set()
    for family, categories in EXPECTED_FAMILIES.items():
        path = CONFIG_ROOT / f"{family}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ZraveFlowConfig.load(path)
        source = raw["data"]["sources"]
        assert len(source) == 1
        assert source[0]["allowed_categories"] == categories
        assert config.model.profile == "tiny"
        assert (
            config.model.d_model,
            config.model.context_layers,
            config.model.future_layers,
            config.model.heads,
            config.model.feedforward_dim,
        ) == (128, 2, 4, 4, 512)
        assert config.model.pitch_conditioning is False
        assert config.model.midi_sequence_conditioning is False
        assert config.segment_sampling.enabled is True
        assert config.train.max_updates == 10000
        assert config.train.short_future_updates == 1000
        assert config.train.checkpoint_every == 1000
        assert config.train.validation_every == 1000
        assert config.train.output_root.endswith(f"/runs/tiny-categories-v1/{family}")
        output_roots.add(config.train.output_root)
    assert len(output_roots) == len(EXPECTED_FAMILIES)


def test_category_queue_trains_every_family_before_any_gate() -> None:
    rows = _rows()
    identifiers = [row[0] for row in rows]
    assert len(identifiers) == len(set(identifiers))
    first_gate = next(
        index for index, row in enumerate(rows) if row[1] == "audition_report"
    )
    assert all(row[1] != "audition_report" for row in rows[:first_gate])
    assert all(row[1] == "audition_report" for row in rows[first_gate:])
    train_rows = [row for row in rows if row[1] == "train"]
    assert [row[3].rsplit("/", 1)[-1] for row in train_rows] == list(EXPECTED_FAMILIES)
    assert all(row[4] == "10000" for row in train_rows)
    assert all(row[4] == "16,32,64,96,128,160" for row in rows if row[1] == "sweep")
    for family in EXPECTED_FAMILIES:
        gates = [
            row
            for row in rows
            if row[1] == "audition_report"
            and row[3] == f"runs/tiny-categories-v1/{family}"
        ]
        assert [row[4] for row in gates] == [
            "step-001000.pt",
            "step-005000.pt",
            "step-010000.pt",
        ]


def test_audition_reads_ordered_categories_from_single_source_config() -> None:
    command = (
        f"source {LVZIHAO / 'common.sh'}; "
        'load_single_source_allowed_categories "$0"; '
        "printf '%s\\n' \"${ALLOWED_CATEGORIES[@]}\""
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            str(CONFIG_ROOT / "keys_harmony.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["Arp", "Chord", "Keys"]
    audition = (LVZIHAO / "audition.sh").read_text(encoding="utf-8")
    assert 'load_single_source_allowed_categories "$host_config"' in audition
    assert '--categories "${ALLOWED_CATEGORIES[@]}"' in audition
    assert "--categories Pad Lead Bass Pluck Keys Synth" not in audition


def test_tiny_pure_shell_contract_is_scratch_only() -> None:
    train = (LVZIHAO / "train.sh").read_text(encoding="utf-8")
    smoke = (LVZIHAO / "smoke.sh").read_text(encoding="utf-8")

    assert "non-standard pure profiles train from scratch" in train
    assert "non-standard pure profiles train from scratch" in smoke
    assert "same-profile pure LV_INITIALIZE_FROM" in train
    assert "same-profile pure LV_INITIALIZE_FROM" in smoke
    subprocess.run(
        [
            "bash",
            "-n",
            str(LVZIHAO / "common.sh"),
            str(LVZIHAO / "audition.sh"),
            str(LVZIHAO / "smoke.sh"),
            str(LVZIHAO / "train.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_category_audio_gate_is_report_only_and_keeps_hard_action() -> None:
    audition = (LVZIHAO / "audition.sh").read_text(encoding="utf-8")
    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")

    assert "LV_AUDITION_FAIL_ON_REJECT" in audition
    assert "--no-fail-on-reject" in audition
    assert "--fail-on-reject" in audition
    assert "audition_report)" in runner
    assert 'LV_AUDITION_FAIL_ON_REJECT=0 "$script_dir/audition.sh"' in runner
