from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from midibrave.zrave_flow_config import ZraveFlowConfig

ROOT = Path(__file__).parents[1]
LVZIHAO = ROOT / "scripts" / "lvzihao"
GENERATOR = LVZIHAO / "generate_category_matrix.py"
RECIPES = LVZIHAO / "tiny_latent_proxy_recipes.yaml"
QUEUE = LVZIHAO / "tiny_latent_proxies_v1.queue.tsv"
CONFIG_ROOT = (
    ROOT / "configs" / "zrave" / "generated" / "lvzihao_tiny_latent_proxies_v1"
)
VALIDATOR = LVZIHAO / "validate_timbre_taxonomy.py"
EXPECTED_BUCKETS = (
    "latent_slow_proxy",
    "latent_fast_proxy",
    "latent_soft_onset_proxy",
    "latent_hard_onset_proxy",
    "latent_static_proxy",
    "latent_moving_proxy",
    "latent_smooth_proxy",
    "latent_irregular_proxy",
)
FORMAL_CATEGORIES = ["Arp", "Bass", "FX", "Lead", "Pad", "Pluck", "Synth"]
CONTAINER_TAXONOMY = "/data/midibrave-zrave-flow-serum128/taxonomies/serum128-latent-v1"


def _rows() -> list[tuple[str, str, str, str, str]]:
    reader = csv.reader(QUEUE.read_text(encoding="utf-8").splitlines(), delimiter="\t")
    return [
        tuple(row)  # type: ignore[misc]
        for row in reader
        if row and not row[0].startswith("#")
    ]


def _load_validator():
    module_spec = importlib.util.spec_from_file_location(
        "test_lvzihao_taxonomy_validator",
        VALIDATOR,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _write_taxonomy_fixture(root: Path) -> tuple[Path, Path]:
    pack = root / "pack"
    taxonomy = root / "taxonomy"
    buckets_root = taxonomy / "buckets"
    pack.mkdir(parents=True)
    buckets_root.mkdir(parents=True)
    index = pack / "index.json"
    index.write_text('{"records":1}\n', encoding="utf-8")
    sequences = pack / "sequences.jsonl"
    sequences.write_text('{"sample_id":"sample-1"}\n', encoding="utf-8")
    index_digest = hashlib.sha256(index.read_bytes()).hexdigest()
    sequences_digest = hashlib.sha256(sequences.read_bytes()).hexdigest()
    feature_version = "zrave-timbre-proxy-v1"
    source_hashes = {
        "index": {"sha256": index_digest},
        "sequences": {"sha256": sequences_digest},
    }
    report_buckets = {}
    for bucket in EXPECTED_BUCKETS:
        relative = f"buckets/{bucket}.json"
        report_buckets[bucket] = {"json": relative}
        (taxonomy / relative).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "feature_source": "latent",
                    "feature_version": feature_version,
                    "bucket": bucket,
                    "source_hashes": source_hashes,
                    "preset_ids": [f"serum:{bucket}"],
                }
            ),
            encoding="utf-8",
        )
    (taxonomy / "taxonomy.report.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "zrave-preset-timbre-proxy-taxonomy",
                "feature_source": "latent",
                "feature_version": feature_version,
                "source_hashes": source_hashes,
                "buckets": report_buckets,
            }
        ),
        encoding="utf-8",
    )
    return pack, taxonomy


def test_latent_proxy_generated_outputs_are_current() -> None:
    subprocess.run(
        ["python", str(GENERATOR), "--recipes", str(RECIPES), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.stem for path in CONFIG_ROOT.glob("*.yaml")} == set(EXPECTED_BUCKETS)


def test_latent_proxy_configs_are_independent_tiny_allowlist_runs() -> None:
    recipe = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))
    assert recipe["maximum_updates"] == 10000
    assert recipe["checkpoints"] == [1000, 5000, 10000]
    assert recipe["sweep_batches"] == [16, 32, 64, 96, 128, 160]
    assert recipe["audition_reports"] is True
    assert [family["id"] for family in recipe["families"]] == list(EXPECTED_BUCKETS)

    output_roots: set[str] = set()
    for bucket in EXPECTED_BUCKETS:
        path = CONFIG_ROOT / f"{bucket}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ZraveFlowConfig.load(path)
        source = raw["data"]["sources"][0]
        assert source["allowed_categories"] == FORMAL_CATEGORIES
        assert source["preset_allowlist"] == (
            f"{CONTAINER_TAXONOMY}/buckets/{bucket}.json"
        )
        assert config.model.profile == "tiny"
        assert config.model.pitch_conditioning is False
        assert config.model.midi_sequence_conditioning is False
        assert config.segment_sampling.enabled is True
        assert config.train.max_updates == 10000
        assert config.train.checkpoint_every == 1000
        assert config.train.validation_every == 1000
        assert config.train.output_root.endswith(
            f"/runs/tiny-latent-proxies-v1/{bucket}"
        )
        output_roots.add(config.train.output_root)
    assert len(output_roots) == len(EXPECTED_BUCKETS)


def test_latent_proxy_queue_builds_taxonomy_trains_all_then_reports_all() -> None:
    rows = _rows()
    assert rows[0] == (
        "tinyproxy-v1-taxonomy",
        "taxonomy",
        "-",
        "-",
        "taxonomies/serum128-latent-v1",
    )
    training_end = 1 + 3 * len(EXPECTED_BUCKETS)
    assert len(rows) == training_end + 3 * len(EXPECTED_BUCKETS)
    assert all(row[1] != "audition_report" for row in rows[:training_end])
    assert all(row[1] == "audition_report" for row in rows[training_end:])
    for index, bucket in enumerate(EXPECTED_BUCKETS):
        group = rows[1 + 3 * index : 1 + 3 * (index + 1)]
        assert [row[1] for row in group] == ["smoke", "sweep", "train"]
        assert {row[3] for row in group} == {f"runs/tiny-latent-proxies-v1/{bucket}"}
        assert group[1][4] == "16,32,64,96,128,160"
        assert group[2][4] == "10000"
        reports = [
            row
            for row in rows[training_end:]
            if row[3] == f"runs/tiny-latent-proxies-v1/{bucket}"
        ]
        assert [row[4] for row in reports] == [
            "step-001000.pt",
            "step-005000.pt",
            "step-010000.pt",
        ]


def test_taxonomy_validator_accepts_current_exact_eight_buckets(
    tmp_path: Path,
) -> None:
    module = _load_validator()
    pack, taxonomy = _write_taxonomy_fixture(tmp_path)

    report = module.validate_taxonomy(pack, taxonomy)

    assert report["valid"] is True
    assert set(report["bucket_preset_counts"]) == set(EXPECTED_BUCKETS)


def test_taxonomy_validator_rejects_stale_pack_and_extra_bucket(
    tmp_path: Path,
) -> None:
    module = _load_validator()
    pack, taxonomy = _write_taxonomy_fixture(tmp_path)
    (pack / "index.json").write_text('{"records":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="pack index SHA-256 is stale"):
        module.validate_taxonomy(pack, taxonomy)

    pack, taxonomy = _write_taxonomy_fixture(tmp_path / "extra")
    (taxonomy / "buckets/extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the eight bucket JSONs"):
        module.validate_taxonomy(pack, taxonomy)


def test_taxonomy_validator_rejects_stale_sequences(tmp_path: Path) -> None:
    module = _load_validator()
    pack, taxonomy = _write_taxonomy_fixture(tmp_path)
    (pack / "sequences.jsonl").write_text(
        '{"sample_id":"changed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pack sequences SHA-256 is stale"):
        module.validate_taxonomy(pack, taxonomy)


def test_taxonomy_shell_and_allocation_action_are_well_formed() -> None:
    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")
    taxonomy = (LVZIHAO / "taxonomy.sh").read_text(encoding="utf-8")
    assert "taxonomy)" in runner
    assert '"$script_dir/taxonomy.sh" "$spec"' in runner
    assert "validate_current_taxonomy" in taxonomy
    assert "python -m midibrave.zrave_timbre_taxonomy" in taxonomy
    subprocess.run(
        [
            "bash",
            "-n",
            str(LVZIHAO / "taxonomy.sh"),
            str(LVZIHAO / "allocation_runner.sh"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
