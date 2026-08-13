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
RECIPES = LVZIHAO / "tiny_audio_proxy_recipes.yaml"
QUEUE = LVZIHAO / "tiny_audio_proxies_v1.queue.tsv"
CLAP_QUEUE = LVZIHAO / "tiny_audio_proxies_v1.clap.queue.tsv"
CONFIG_ROOT = ROOT / "configs" / "zrave" / "generated" / "lvzihao_tiny_audio_proxies_v1"
VALIDATOR = LVZIHAO / "validate_imported_audio_taxonomy.py"
EXPECTED_BUCKETS = (
    "audio_dark_proxy",
    "audio_bright_proxy",
    "audio_soft_attack_proxy",
    "audio_hard_attack_proxy",
    "audio_static_proxy",
    "audio_moving_proxy",
    "audio_clean_proxy",
    "audio_noisy_proxy",
)
FORMAL_CATEGORIES = ["Arp", "Bass", "FX", "Lead", "Pad", "Pluck", "Synth"]
CONTAINER_TAXONOMY = "/data/midibrave-zrave-flow-serum128/taxonomies/serum128-audio-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    reader = csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t")
    return [
        tuple(row)  # type: ignore[misc]
        for row in reader
        if row and not row[0].startswith("#")
    ]


def _load_validator():
    module_spec = importlib.util.spec_from_file_location(
        "test_lvzihao_imported_audio_taxonomy_validator",
        VALIDATOR,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _write_import_fixture(
    root: Path,
) -> tuple[object, Path, Path, Path]:
    module = _load_validator()
    pack = root / "pack"
    taxonomy = root / "taxonomy"
    buckets_root = taxonomy / "buckets"
    audio_root = root / "raw-audio"
    pack.mkdir(parents=True)
    buckets_root.mkdir(parents=True)
    audio_root.mkdir(parents=True)

    audio = audio_root / "sample-1.wav"
    audio.write_bytes(b"authoritative-octopus-audio-fixture")
    manifest = root / "audio-manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "sample-1", "audio_path": str(audio)}) + "\n",
        encoding="utf-8",
    )
    sequences = pack / "sequences.jsonl"
    sequences.write_text(
        json.dumps({"sample_id": "sample-1"}) + "\n",
        encoding="utf-8",
    )
    shard = pack / "shard-00000.npz"
    shard.write_bytes(b"packed-latent-shard-fixture")
    index = pack / "index.json"
    index.write_text(
        json.dumps(
            {
                "records": 1,
                "shard_sha256": {shard.name: _sha256(shard)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    audio_digest = _sha256(audio)
    aggregate = hashlib.sha256()
    aggregate.update(b"sample-1\0")
    aggregate.update(audio_digest.encode("ascii"))
    aggregate.update(b"\n")
    source_hashes = {
        "pack_root": str(pack.resolve()),
        "index": {"path": str(index.resolve()), "sha256": _sha256(index)},
        "sequences": {
            "path": str(sequences.resolve()),
            "sha256": _sha256(sequences),
            "records": 1,
        },
        "shards": {shard.name: _sha256(shard)},
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": _sha256(manifest),
        },
        "audio_files": {
            "count": 1,
            "unique_files": 1,
            "sample_id_content_sha256": aggregate.hexdigest(),
        },
    }
    feature_version = "zrave-timbre-proxy-v1"
    counts = {"presets": 1, "records": 1}
    report_buckets = {}
    for bucket in EXPECTED_BUCKETS:
        relative = f"buckets/{bucket}.json"
        report_buckets[bucket] = {"json": relative, "counts": counts}
        (taxonomy / relative).write_text(
            json.dumps(
                {
                    "schema": 1,
                    "feature_source": "audio",
                    "feature_version": feature_version,
                    "bucket": bucket,
                    "counts": counts,
                    "source_hashes": source_hashes,
                    "preset_ids": [f"serum:{bucket}"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (taxonomy / "taxonomy.report.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "zrave-preset-timbre-proxy-taxonomy",
                "feature_source": "audio",
                "feature_version": feature_version,
                "tool": {"module_sha256": hashlib.sha256(b"fixture").hexdigest()},
                "source_hashes": source_hashes,
                "buckets": report_buckets,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return module, pack, taxonomy, audio


def test_audio_proxy_generated_outputs_are_current() -> None:
    subprocess.run(
        ["python", str(GENERATOR), "--recipes", str(RECIPES), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.stem for path in CONFIG_ROOT.glob("*.yaml")} == set(EXPECTED_BUCKETS)


def test_audio_proxy_configs_are_independent_tiny_allowlist_runs() -> None:
    recipe = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))
    assert recipe["taxonomy_relative"] == "taxonomies/serum128-audio-v1"
    assert recipe["taxonomy_action"] == "taxonomy_validate_audio"
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
        assert tuple(config.segment_sampling.divisions) == (2, 4, 8)
        assert config.train.max_updates == 10000
        assert config.train.checkpoint_every == 1000
        assert config.train.validation_every == 1000
        assert config.train.output_root.endswith(
            f"/runs/tiny-audio-proxies-v1/{bucket}"
        )
        output_roots.add(config.train.output_root)
    assert len(output_roots) == len(EXPECTED_BUCKETS)


def test_audio_proxy_queue_validates_then_trains_all_then_reports_all() -> None:
    rows = _rows(QUEUE)
    assert rows[0] == (
        "tinyaudioproxy-v1-taxonomy",
        "taxonomy_validate_audio",
        "-",
        "-",
        "taxonomies/serum128-audio-v1",
    )
    training_end = 1 + 3 * len(EXPECTED_BUCKETS)
    assert len(rows) == training_end + 3 * len(EXPECTED_BUCKETS)
    assert all(row[1] != "audition_report" for row in rows[:training_end])
    assert all(row[1] == "audition_report" for row in rows[training_end:])
    for index, bucket in enumerate(EXPECTED_BUCKETS):
        group = rows[1 + 3 * index : 1 + 3 * (index + 1)]
        assert [row[1] for row in group] == ["smoke", "sweep", "train"]
        assert group[1][4] == "16,32,64,96,128,160"
        assert group[2][4] == "10000"
        reports = [
            row
            for row in rows[training_end:]
            if row[3] == f"runs/tiny-audio-proxies-v1/{bucket}"
        ]
        assert [row[4] for row in reports] == [
            "step-001000.pt",
            "step-005000.pt",
            "step-010000.pt",
        ]

    clap_rows = _rows(CLAP_QUEUE)
    assert len(clap_rows) == 3 * len(EXPECTED_BUCKETS)
    assert all(row[1:4] == ("clap_report", "-", "-") for row in clap_rows)
    assert [row[4] for row in clap_rows] == [row[0] for row in rows[training_end:]]


def test_octopus_seal_and_import_validation_bind_raw_audio_and_pack(
    tmp_path: Path,
) -> None:
    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path)

    sealed = module.seal_octopus_export(pack, taxonomy)
    validated = module.validate_imported_audio_taxonomy(pack, taxonomy)

    assert sealed == validated
    assert validated["valid"] is True
    assert validated["feature_source"] == "audio"
    assert set(validated["bucket_preset_counts"]) == set(EXPECTED_BUCKETS)
    assert (taxonomy / module.PROVENANCE_NAME).is_file()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.seal_octopus_export(pack, taxonomy)


def test_audio_import_rejects_unsealed_stale_or_tampered_provenance(
    tmp_path: Path,
) -> None:
    module, pack, taxonomy, audio = _write_import_fixture(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="octopus-export"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)

    audio.write_bytes(b"changed-after-taxonomy-build")
    with pytest.raises(ValueError, match="raw audio content collection changed"):
        module.seal_octopus_export(pack, taxonomy)

    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path / "stale")
    module.seal_octopus_export(pack, taxonomy)
    (pack / "sequences.jsonl").write_text(
        json.dumps({"sample_id": "changed"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pack sequences SHA-256 is stale"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)

    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path / "bucket")
    module.seal_octopus_export(pack, taxonomy)
    bucket_path = taxonomy / "buckets" / f"{EXPECTED_BUCKETS[0]}.json"
    bucket = json.loads(bucket_path.read_text(encoding="utf-8"))
    bucket["source_hashes"]["audio_files"]["sample_id_content_sha256"] = "0" * 64
    bucket_path.write_text(json.dumps(bucket), encoding="utf-8")
    with pytest.raises(ValueError, match="source provenance mismatches report"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)

    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path / "index")
    module.seal_octopus_export(pack, taxonomy)
    index = json.loads((pack / "index.json").read_text(encoding="utf-8"))
    (pack / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pack index SHA-256 is stale"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)

    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path / "origin")
    module.seal_octopus_export(pack, taxonomy)
    provenance_path = taxonomy / module.PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_cluster"] = "not-octopus"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="Octopus export provenance"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)

    module, pack, taxonomy, _audio = _write_import_fixture(tmp_path / "extra")
    module.seal_octopus_export(pack, taxonomy)
    (taxonomy / "buckets" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly eight audio bucket JSONs"):
        module.validate_imported_audio_taxonomy(pack, taxonomy)


def test_audio_taxonomy_action_only_validates_exact_import() -> None:
    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")
    shell = (LVZIHAO / "taxonomy_validate_audio.sh").read_text(encoding="utf-8")
    assert "taxonomy_validate_audio)" in runner
    assert '"$script_dir/taxonomy_validate_audio.sh" "$spec"' in runner
    assert "taxonomies/serum128-audio-v1" in shell
    assert "validate_imported_audio_taxonomy.py" in shell
    assert "zrave_timbre_taxonomy" not in shell
    assert "sync $taxonomy_relative before training" in shell
    subprocess.run(
        [
            "bash",
            "-n",
            str(LVZIHAO / "taxonomy_validate_audio.sh"),
            str(LVZIHAO / "allocation_runner.sh"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
