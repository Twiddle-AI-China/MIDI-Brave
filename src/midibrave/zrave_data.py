from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .zrave_config import ZraveConfig


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL file: {source}")
    return rows


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, *values: object) -> bytes:
    payload = ":".join([str(seed), "zrave-balanced50", *map(str, values)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _preset_id(row: dict[str, Any]) -> str:
    value = (
        row.get("preset_id")
        or row.get("timbre_id")
        or row.get("instrument_id")
    )
    if not value:
        raise ValueError("row is missing preset_id/timbre_id/instrument_id")
    return str(value)


def _category_lookup(
    preset_rows: Iterable[dict[str, Any]],
    categories: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    canonical = {category.casefold(): category for category in categories}
    result: dict[str, dict[str, str]] = {}
    for row in preset_rows:
        preset_id = _preset_id(row)
        category_raw = str(row.get("category") or "").strip()
        category = canonical.get(category_raw.casefold())
        if category is None:
            continue
        if preset_id in result:
            raise ValueError(f"duplicate preset metadata: {preset_id}")
        result[preset_id] = {
            "category": category,
            "bank": str(row.get("bank") or preset_id),
        }
    return result


def _diverse_order(
    candidates: list[str],
    banks: dict[str, str],
    seed: int,
    category: str,
) -> list[str]:
    by_bank: dict[str, list[str]] = defaultdict(list)
    for preset_id in candidates:
        by_bank[banks[preset_id]].append(preset_id)
    for bank, values in by_bank.items():
        values.sort(key=lambda value: _stable_key(seed, category, bank, value))
    bank_order = sorted(
        by_bank,
        key=lambda bank: _stable_key(seed, category, "bank", bank),
    )
    ordered: list[str] = []
    round_index = 0
    while len(ordered) < len(candidates):
        added = False
        for bank in bank_order:
            values = by_bank[bank]
            if round_index < len(values):
                ordered.append(values[round_index])
                added = True
        if not added:
            break
        round_index += 1
    return ordered


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_balanced_manifest(config: ZraveConfig) -> dict[str, object]:
    data = config.data
    preset_rows = read_jsonl(data.preset_metadata)
    eligible_rows = read_jsonl(data.eligible_manifest)
    preset_info = _category_lookup(preset_rows, data.categories)

    rows_by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        preset_id = _preset_id(row)
        if preset_id in preset_info:
            rows_by_preset[preset_id].append(row)

    complete: dict[str, list[dict[str, Any]]] = {}
    for preset_id, rows in rows_by_preset.items():
        conditions: set[tuple[int, int]] = set()
        sample_ids: set[str] = set()
        valid = True
        for row in rows:
            try:
                condition = (int(row["midi_note"]), int(row["velocity"]))
                sample_id = str(row["sample_id"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            conditions.add(condition)
            sample_ids.add(sample_id)
        if (
            valid
            and len(rows) == data.conditions_per_preset
            and len(sample_ids) == data.conditions_per_preset
            and len(conditions) == data.conditions_per_preset
        ):
            complete[preset_id] = rows

    category_candidates: dict[str, list[str]] = {
        category: [] for category in data.categories
    }
    banks: dict[str, str] = {}
    for preset_id in complete:
        info = preset_info[preset_id]
        category_candidates[info["category"]].append(preset_id)
        banks[preset_id] = info["bank"]

    selected_by_category: dict[str, list[str]] = {}
    for category in data.categories:
        ordered = _diverse_order(
            category_candidates[category],
            banks,
            config.seed,
            category,
        )
        if len(ordered) < data.presets_per_category:
            raise ValueError(
                f"category {category} has {len(ordered)} complete presets; "
                f"need {data.presets_per_category}"
            )
        selected_by_category[category] = ordered[: data.presets_per_category]

    split_by_preset: dict[str, str] = {}
    selected_metadata: dict[str, list[dict[str, str]]] = {}
    for category in data.categories:
        selected_metadata[category] = []
        for index, preset_id in enumerate(selected_by_category[category]):
            split = (
                "train"
                if index < 8
                else "validation"
                if index == 8
                else "test"
            )
            split_by_preset[preset_id] = split
            selected_metadata[category].append(
                {
                    "preset_id": preset_id,
                    "bank": banks[preset_id],
                    "split": split,
                }
            )

    selected_rows: list[dict[str, Any]] = []
    for category in data.categories:
        for preset_id in selected_by_category[category]:
            rows = sorted(
                complete[preset_id],
                key=lambda row: (
                    int(row["midi_note"]),
                    int(row["velocity"]),
                    str(row["sample_id"]),
                ),
            )
            for source in rows:
                row = dict(source)
                row["preset_id"] = preset_id
                row["split"] = split_by_preset[preset_id]
                row["zrave_category"] = category
                row["zrave_bank"] = banks[preset_id]
                selected_rows.append(row)

    output_manifest = Path(data.selected_manifest)
    output_metadata = Path(data.metadata_output)
    _atomic_jsonl(output_manifest, selected_rows)

    split_samples = Counter(str(row["split"]) for row in selected_rows)
    split_presets = Counter(split_by_preset.values())
    category_counts = {
        category: len(selected_by_category[category])
        for category in sorted(data.categories)
    }
    duration_seconds = sum(
        float(row.get("duration_seconds") or 0.0)
        for row in selected_rows
    )
    report: dict[str, object] = {
        "presets": len(split_by_preset),
        "samples": len(selected_rows),
        "categories": category_counts,
        "splits": {
            split: split_presets[split]
            for split in ("train", "validation", "test")
        },
        "duration_hours": duration_seconds / 3600.0,
    }
    metadata = {
        "schema": 1,
        "profile": "serum-balanced50-zrave",
        "seed": config.seed,
        **report,
        "preset_metadata": str(Path(data.preset_metadata).resolve()),
        "preset_metadata_sha256": _sha256_file(data.preset_metadata),
        "eligible_manifest": str(Path(data.eligible_manifest).resolve()),
        "eligible_manifest_sha256": _sha256_file(data.eligible_manifest),
        "selected_manifest": str(output_manifest.resolve()),
        "selected_manifest_sha256": _sha256_file(output_manifest),
        "selected_presets": selected_metadata,
        "split_samples": {
            split: split_samples[split]
            for split in ("train", "validation", "test")
        },
        "completeness_rule": {
            "unique_sample_ids": data.conditions_per_preset,
            "unique_midi_note_velocity_pairs": data.conditions_per_preset,
            "rows": data.conditions_per_preset,
        },
        "warmup_frames": data.warmup_frames,
        "latent_hop": data.latent_hop,
    }
    _atomic_json(output_metadata, metadata)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the balanced Z-RAVE latent dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ZraveConfig.load(args.config)
    if args.command == "select":
        print(
            json.dumps(
                select_balanced_manifest(config),
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
