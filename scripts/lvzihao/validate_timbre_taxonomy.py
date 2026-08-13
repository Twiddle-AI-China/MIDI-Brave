#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_LATENT_BUCKETS = (
    "latent_slow_proxy",
    "latent_fast_proxy",
    "latent_soft_onset_proxy",
    "latent_hard_onset_proxy",
    "latent_static_proxy",
    "latent_moving_proxy",
    "latent_smooth_proxy",
    "latent_irregular_proxy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _source_sha(payload: dict[str, Any], name: str, member: str) -> str:
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError(f"{name}.source_hashes must be an object")
    source = source_hashes.get(member)
    if not isinstance(source, dict):
        raise ValueError(f"{name}.source_hashes.{member} must be an object")
    digest = source.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} has an invalid {member} SHA-256")
    return digest


def validate_taxonomy(pack_root: Path, taxonomy_root: Path) -> dict[str, object]:
    pack = pack_root.expanduser().resolve()
    taxonomy = taxonomy_root.expanduser().resolve()
    index_path = pack / "index.json"
    sequences_path = pack / "sequences.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not sequences_path.is_file():
        raise FileNotFoundError(sequences_path)
    current_index_sha256 = _sha256(index_path)
    current_sequences_sha256 = _sha256(sequences_path)

    report_path = taxonomy / "taxonomy.report.json"
    report = _object(report_path, "taxonomy report")
    if report.get("schema") != 1:
        raise ValueError("taxonomy report schema must be 1")
    if report.get("kind") != "zrave-preset-timbre-proxy-taxonomy":
        raise ValueError("taxonomy report kind is invalid")
    if report.get("feature_source") != "latent":
        raise ValueError("taxonomy report feature_source must be latent")
    feature_version = report.get("feature_version")
    if not isinstance(feature_version, str) or not feature_version:
        raise ValueError("taxonomy report feature_version is invalid")
    if _source_sha(report, "taxonomy report", "index") != current_index_sha256:
        raise ValueError("taxonomy report pack index SHA-256 is stale")
    report_sequences_sha256 = _source_sha(
        report,
        "taxonomy report",
        "sequences",
    )
    if report_sequences_sha256 != current_sequences_sha256:
        raise ValueError("taxonomy report pack sequences SHA-256 is stale")

    report_buckets = report.get("buckets")
    if not isinstance(report_buckets, dict):
        raise ValueError("taxonomy report buckets must be an object")
    expected = set(EXPECTED_LATENT_BUCKETS)
    if set(report_buckets) != expected:
        raise ValueError(
            "taxonomy report must contain exactly the eight latent proxy buckets"
        )
    buckets_root = taxonomy / "buckets"
    if not buckets_root.is_dir():
        raise FileNotFoundError(buckets_root)
    observed_json = {path.stem for path in buckets_root.glob("*.json")}
    if observed_json != expected:
        raise ValueError(
            "taxonomy buckets directory must contain exactly the eight bucket JSONs"
        )

    preset_counts: dict[str, int] = {}
    for bucket in EXPECTED_LATENT_BUCKETS:
        relative = f"buckets/{bucket}.json"
        summary = report_buckets[bucket]
        if not isinstance(summary, dict) or summary.get("json") != relative:
            raise ValueError(f"taxonomy report path mismatch for {bucket}")
        payload = _object(taxonomy / relative, f"bucket {bucket}")
        if payload.get("schema") != 1:
            raise ValueError(f"bucket {bucket} schema must be 1")
        if payload.get("feature_source") != "latent":
            raise ValueError(f"bucket {bucket} feature_source must be latent")
        if payload.get("feature_version") != feature_version:
            raise ValueError(f"bucket {bucket} feature_version mismatch")
        if payload.get("bucket") != bucket:
            raise ValueError(f"bucket identity mismatch for {bucket}")
        if _source_sha(payload, f"bucket {bucket}", "index") != current_index_sha256:
            raise ValueError(f"bucket {bucket} pack index SHA-256 is stale")
        if (
            _source_sha(payload, f"bucket {bucket}", "sequences")
            != report_sequences_sha256
        ):
            raise ValueError(f"bucket {bucket} sequences SHA-256 mismatches report")
        preset_ids = payload.get("preset_ids")
        if (
            not isinstance(preset_ids, list)
            or not preset_ids
            or any(not isinstance(value, str) or not value for value in preset_ids)
            or len(preset_ids) != len(set(preset_ids))
        ):
            raise ValueError(f"bucket {bucket} preset_ids are invalid")
        preset_counts[bucket] = len(preset_ids)

    return {
        "schema": 1,
        "feature_source": "latent",
        "feature_version": feature_version,
        "pack_index_sha256": current_index_sha256,
        "pack_sequences_sha256": current_sequences_sha256,
        "taxonomy_root": str(taxonomy),
        "bucket_preset_counts": preset_counts,
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the lvzihao latent-proxy taxonomy preflight contract."
    )
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--taxonomy-root", required=True, type=Path)
    args = parser.parse_args()
    report = validate_taxonomy(args.pack_root, args.taxonomy_root)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
