#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping


EXPECTED_AUDIO_BUCKETS = (
    "audio_dark_proxy",
    "audio_bright_proxy",
    "audio_soft_attack_proxy",
    "audio_hard_attack_proxy",
    "audio_static_proxy",
    "audio_moving_proxy",
    "audio_clean_proxy",
    "audio_noisy_proxy",
)
PROVENANCE_NAME = "octopus-export.provenance.json"
PROVENANCE_KIND = "zrave-octopus-audio-taxonomy-export-provenance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _source_member(
    source_hashes: Mapping[str, object],
    member: str,
) -> dict[str, Any]:
    return _mapping(source_hashes.get(member), f"source_hashes.{member}")


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _pack_contract(pack: Path) -> dict[str, object]:
    index_path = pack / "index.json"
    sequences_path = pack / "sequences.jsonl"
    index = _object(index_path, "pack index")
    rows = _jsonl_objects(sequences_path)
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("pack sequences require non-empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("pack sequence sample_id values must be unique")
    if index.get("records") != len(rows):
        raise ValueError("pack index records do not match sequences")
    raw_shards = _mapping(index.get("shard_sha256"), "pack index shard_sha256")
    if not raw_shards:
        raise ValueError("pack index shard_sha256 must not be empty")
    shards = {
        str(name): _valid_sha(digest, f"pack shard {name}")
        for name, digest in raw_shards.items()
    }
    if any(
        not name or Path(name).is_absolute() or ".." in Path(name).parts
        for name in shards
    ):
        raise ValueError("pack index contains an unsafe shard path")
    return {
        "index_sha256": _sha256(index_path),
        "sequences_sha256": _sha256(sequences_path),
        "records": len(rows),
        "sample_ids": sample_ids,
        "shards": shards,
    }


def _validate_source_hashes(
    report: Mapping[str, object],
    pack_contract: Mapping[str, object],
) -> dict[str, Any]:
    source_hashes = _mapping(report.get("source_hashes"), "report source_hashes")
    expected_members = {
        "pack_root",
        "index",
        "sequences",
        "shards",
        "manifest",
        "audio_files",
    }
    if set(source_hashes) != expected_members:
        raise ValueError(
            "audio taxonomy source_hashes must contain exactly pack_root, index, "
            "sequences, shards, manifest, and audio_files"
        )
    source_pack_root = source_hashes["pack_root"]
    if (
        not isinstance(source_pack_root, str)
        or not Path(source_pack_root).is_absolute()
    ):
        raise ValueError("audio taxonomy source pack_root must be absolute")

    index = _source_member(source_hashes, "index")
    if set(index) != {"path", "sha256"} or not Path(str(index["path"])).is_absolute():
        raise ValueError("audio taxonomy index provenance is invalid")
    if (
        _valid_sha(index.get("sha256"), "taxonomy index")
        != pack_contract["index_sha256"]
    ):
        raise ValueError("audio taxonomy pack index SHA-256 is stale")

    sequences = _source_member(source_hashes, "sequences")
    if set(sequences) != {"path", "sha256", "records"}:
        raise ValueError("audio taxonomy sequences provenance is invalid")
    if not Path(str(sequences["path"])).is_absolute():
        raise ValueError("audio taxonomy sequences path must be absolute")
    if (
        _valid_sha(sequences.get("sha256"), "taxonomy sequences")
        != pack_contract["sequences_sha256"]
    ):
        raise ValueError("audio taxonomy pack sequences SHA-256 is stale")
    if sequences.get("records") != pack_contract["records"]:
        raise ValueError("audio taxonomy sequences record count is stale")

    shards = _mapping(source_hashes.get("shards"), "source_hashes.shards")
    normalized_shards = {
        str(name): _valid_sha(digest, f"taxonomy shard {name}")
        for name, digest in shards.items()
    }
    if normalized_shards != pack_contract["shards"]:
        raise ValueError("audio taxonomy shard provenance mismatches current pack")

    manifest = _source_member(source_hashes, "manifest")
    if set(manifest) != {"path", "sha256"}:
        raise ValueError("audio taxonomy manifest provenance is invalid")
    manifest_path = manifest.get("path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
        raise ValueError("audio taxonomy manifest path must be absolute")
    _valid_sha(manifest.get("sha256"), "audio manifest")

    audio_files = _source_member(source_hashes, "audio_files")
    if set(audio_files) != {
        "count",
        "unique_files",
        "sample_id_content_sha256",
    }:
        raise ValueError("audio taxonomy content collection provenance is invalid")
    count = audio_files.get("count")
    unique_files = audio_files.get("unique_files")
    if count != pack_contract["records"]:
        raise ValueError("audio taxonomy content collection count is stale")
    if (
        not isinstance(unique_files, int)
        or isinstance(unique_files, bool)
        or not 0 < unique_files <= int(count)
    ):
        raise ValueError("audio taxonomy unique audio file count is invalid")
    _valid_sha(
        audio_files.get("sample_id_content_sha256"),
        "audio sample/content collection",
    )
    return source_hashes


def _expected_provenance(
    report_path: Path,
    report: Mapping[str, object],
    source_hashes: Mapping[str, object],
) -> dict[str, object]:
    index = _source_member(source_hashes, "index")
    sequences = _source_member(source_hashes, "sequences")
    manifest = _source_member(source_hashes, "manifest")
    audio_files = _source_member(source_hashes, "audio_files")
    tool = _mapping(report.get("tool"), "taxonomy report tool")
    return {
        "schema": 1,
        "kind": PROVENANCE_KIND,
        "source_cluster": "octopus",
        "feature_source": "audio",
        "feature_version": report["feature_version"],
        "taxonomy_report_sha256": _sha256(report_path),
        "taxonomy_tool_sha256": _valid_sha(
            tool.get("module_sha256"),
            "taxonomy tool module",
        ),
        "source_pack_root": source_hashes["pack_root"],
        "source_manifest_path": manifest["path"],
        "source_hashes": {
            "pack_index_sha256": index["sha256"],
            "pack_sequences_sha256": sequences["sha256"],
            "audio_manifest_sha256": manifest["sha256"],
            "audio_sample_id_content_sha256": audio_files["sample_id_content_sha256"],
        },
    }


def _validate_core(
    pack_root: Path,
    taxonomy_root: Path,
    *,
    require_provenance: bool,
) -> tuple[dict[str, object], dict[str, Any], dict[str, object]]:
    pack = pack_root.expanduser().resolve()
    taxonomy = taxonomy_root.expanduser().resolve()
    if not pack.is_dir():
        raise FileNotFoundError(pack)
    if not taxonomy.is_dir():
        raise FileNotFoundError(taxonomy)
    pack_contract = _pack_contract(pack)
    report_path = taxonomy / "taxonomy.report.json"
    report = _object(report_path, "taxonomy report")
    if report.get("schema") != 1:
        raise ValueError("audio taxonomy report schema must be 1")
    if report.get("kind") != "zrave-preset-timbre-proxy-taxonomy":
        raise ValueError("audio taxonomy report kind is invalid")
    if report.get("feature_source") != "audio":
        raise ValueError("audio taxonomy report feature_source must be audio")
    tool = _mapping(report.get("tool"), "audio taxonomy report tool")
    if set(tool) != {"module_sha256"}:
        raise ValueError("audio taxonomy report tool contract is invalid")
    _valid_sha(tool.get("module_sha256"), "audio taxonomy tool module")
    feature_version = report.get("feature_version")
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("audio taxonomy feature_version is invalid")
    source_hashes = _validate_source_hashes(report, pack_contract)

    report_buckets = _mapping(report.get("buckets"), "taxonomy report buckets")
    expected = set(EXPECTED_AUDIO_BUCKETS)
    if set(report_buckets) != expected:
        raise ValueError(
            "taxonomy report must contain exactly eight audio proxy buckets"
        )
    buckets_root = taxonomy / "buckets"
    if not buckets_root.is_dir():
        raise FileNotFoundError(buckets_root)
    observed_json = {path.stem for path in buckets_root.glob("*.json")}
    if observed_json != expected:
        raise ValueError(
            "taxonomy buckets directory must contain exactly eight audio bucket JSONs"
        )

    preset_counts: dict[str, int] = {}
    for bucket in EXPECTED_AUDIO_BUCKETS:
        relative = f"buckets/{bucket}.json"
        summary = _mapping(report_buckets[bucket], f"report bucket {bucket}")
        if summary.get("json") != relative:
            raise ValueError(f"taxonomy report path mismatch for {bucket}")
        payload = _object(taxonomy / relative, f"bucket {bucket}")
        if payload.get("schema") != 1:
            raise ValueError(f"bucket {bucket} schema must be 1")
        if payload.get("feature_source") != "audio":
            raise ValueError(f"bucket {bucket} feature_source must be audio")
        if payload.get("feature_version") != feature_version:
            raise ValueError(f"bucket {bucket} feature_version mismatch")
        if payload.get("bucket") != bucket:
            raise ValueError(f"bucket identity mismatch for {bucket}")
        if payload.get("source_hashes") != source_hashes:
            raise ValueError(f"bucket {bucket} source provenance mismatches report")
        preset_ids = payload.get("preset_ids")
        if (
            not isinstance(preset_ids, list)
            or not preset_ids
            or any(not isinstance(value, str) or not value for value in preset_ids)
            or len(preset_ids) != len(set(preset_ids))
        ):
            raise ValueError(f"bucket {bucket} preset_ids are invalid")
        counts = _mapping(payload.get("counts"), f"bucket {bucket} counts")
        if counts.get("presets") != len(preset_ids):
            raise ValueError(f"bucket {bucket} preset count mismatches allowlist")
        if summary.get("counts") != counts:
            raise ValueError(f"bucket {bucket} counts mismatch report")
        preset_counts[bucket] = len(preset_ids)

    expected_provenance = _expected_provenance(report_path, report, source_hashes)
    if require_provenance:
        provenance = _object(
            taxonomy / PROVENANCE_NAME,
            "Octopus export provenance",
        )
        if provenance != expected_provenance:
            raise ValueError(
                "Octopus export provenance does not bind the current audio taxonomy"
            )
    return (
        report,
        source_hashes,
        {
            "pack": pack_contract,
            "taxonomy_root": str(taxonomy),
            "bucket_preset_counts": preset_counts,
            "expected_provenance": expected_provenance,
        },
    )


def _audio_collection_from_manifest(
    manifest_path: Path,
    sample_ids: list[str],
) -> dict[str, object]:
    paths: dict[str, Path] = {}
    for row in _jsonl_objects(manifest_path):
        sample_id = str(row.get("sample_id") or "")
        audio_value = row.get("audio_path")
        if not sample_id or not isinstance(audio_value, str) or not audio_value:
            raise ValueError("audio manifest rows require sample_id and audio_path")
        if sample_id in paths:
            raise ValueError(f"duplicate audio manifest sample_id: {sample_id}")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        paths[sample_id] = audio_path
    missing = sorted(set(sample_ids) - set(paths))
    if missing:
        raise ValueError(
            "audio manifest lacks packed samples: " + ", ".join(missing[:10])
        )
    content_hashes: dict[Path, str] = {}
    aggregate = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        path = paths[sample_id]
        if path not in content_hashes:
            content_hashes[path] = _sha256(path)
        aggregate.update(sample_id.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content_hashes[path].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "count": len(sample_ids),
        "unique_files": len(content_hashes),
        "sample_id_content_sha256": aggregate.hexdigest(),
    }


def seal_octopus_export(pack_root: Path, taxonomy_root: Path) -> dict[str, object]:
    report, source_hashes, context = _validate_core(
        pack_root,
        taxonomy_root,
        require_provenance=False,
    )
    taxonomy = taxonomy_root.expanduser().resolve()
    provenance_path = taxonomy / PROVENANCE_NAME
    if provenance_path.exists():
        raise FileExistsError(
            f"provenance already exists; refusing overwrite: {provenance_path}"
        )
    manifest = _source_member(source_hashes, "manifest")
    manifest_path = Path(str(manifest["path"])).expanduser().resolve()
    if _sha256(manifest_path) != manifest["sha256"]:
        raise ValueError("audio manifest SHA-256 changed since taxonomy build")
    pack_contract = _mapping(context["pack"], "pack contract")
    observed_audio_files = _audio_collection_from_manifest(
        manifest_path,
        list(pack_contract["sample_ids"]),
    )
    if observed_audio_files != source_hashes["audio_files"]:
        raise ValueError("raw audio content collection changed since taxonomy build")
    provenance = _expected_provenance(
        taxonomy / "taxonomy.report.json",
        report,
        source_hashes,
    )
    payload = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    temporary = taxonomy / f".{PROVENANCE_NAME}.tmp-{secrets.token_hex(8)}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, provenance_path)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_imported_audio_taxonomy(pack_root, taxonomy_root)


def validate_imported_audio_taxonomy(
    pack_root: Path,
    taxonomy_root: Path,
) -> dict[str, object]:
    report, source_hashes, context = _validate_core(
        pack_root,
        taxonomy_root,
        require_provenance=True,
    )
    pack_contract = _mapping(context["pack"], "pack contract")
    manifest = _source_member(source_hashes, "manifest")
    audio_files = _source_member(source_hashes, "audio_files")
    provenance_path = taxonomy_root.expanduser().resolve() / PROVENANCE_NAME
    return {
        "schema": 1,
        "kind": "zrave-imported-audio-taxonomy-validation",
        "valid": True,
        "feature_source": "audio",
        "feature_version": report["feature_version"],
        "taxonomy_root": context["taxonomy_root"],
        "pack_index_sha256": pack_contract["index_sha256"],
        "pack_sequences_sha256": pack_contract["sequences_sha256"],
        "audio_manifest_sha256": manifest["sha256"],
        "audio_sample_id_content_sha256": audio_files["sample_id_content_sha256"],
        "octopus_export_provenance_sha256": _sha256(provenance_path),
        "bucket_preset_counts": context["bucket_preset_counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seal an audio taxonomy on Octopus or validate its immutable import "
            "on lvzihao. Validation never builds or overwrites a taxonomy."
        )
    )
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--taxonomy-root", required=True, type=Path)
    parser.add_argument(
        "--seal-octopus-export",
        action="store_true",
        help=(
            "Rehash the authoritative manifest/raw audio and create the one-time "
            f"{PROVENANCE_NAME} sidecar."
        ),
    )
    args = parser.parse_args()
    report = (
        seal_octopus_export(args.pack_root, args.taxonomy_root)
        if args.seal_octopus_export
        else validate_imported_audio_taxonomy(args.pack_root, args.taxonomy_root)
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
