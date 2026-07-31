from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .zrave_flow_config import FlowSourceConfig, ZraveFlowConfig


_FIELDS = (
    "sample_id",
    "source_name",
    "audio_path",
    "canonical_preset_id",
    "split",
    "midi_note",
    "velocity",
    "articulation_id",
    "category",
    "duration_seconds",
    "maximum_future_frames",
)
_CATEGORY_ALIASES = {
    "pad": "Pad",
    "pads": "Pad",
    "lead": "Lead",
    "leads": "Lead",
    "bass": "Bass",
    "pluck": "Pluck",
    "keys": "Keys",
    "arp": "Arp",
    "chord": "Chord",
    "synth": "Synth",
    "atmosphere": "Atmosphere",
    "drums": "Drums",
    "fx": "FX",
    "vocal": "Vocal",
}


class _MissingAudioError(ValueError):
    """A manifest row references audio that is absent from its data tier."""


@dataclass(frozen=True)
class FlowManifestReport:
    rows: int
    hours: float
    by_source: dict[str, int]
    by_category: dict[str, int]
    by_note: dict[str, int]
    by_split: dict[str, int]
    manifest_sha256: str
    source_sha256: dict[str, str]
    missing_audio_by_source: dict[str, int]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}:{line_number}: expected a JSON object"
                )
            yield value


def read_flow_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = list(_read_jsonl(path))
    if not rows:
        raise ValueError(f"empty flow manifest: {path}")
    expected = set(_FIELDS)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"flow manifest row {index} fields differ: "
                f"{sorted(set(row) ^ expected)}"
            )
    return rows


def preset_split(canonical_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"{seed}:{canonical_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _number(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}: {value!r}") from error


def canonical_preset_id(
    source_name: str,
    row: dict[str, Any],
) -> str:
    if source_name in {"serum_full", "serum_balanced"}:
        raw = str(row.get("preset_id") or row.get("preset_key") or "")
        match = re.fullmatch(
            r"(?:serum_s|xfer/serum:|serum:)(\d+)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"invalid Serum preset_id: {raw!r}")
        return f"serum:{int(match.group(1)):06d}"
    if source_name == "pianobook_pitch":
        raw = str(row.get("preset_id") or "")
        match = re.fullmatch(
            r"(?:pianobook_|pianobook:)(\d+)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"invalid Pianobook preset_id: {raw!r}")
        return f"pianobook:{int(match.group(1)):06d}"
    if source_name == "dexed_surge_broad":
        index = _number(row.get("preset_index"), "preset_index")
        synth = str(row.get("canonical_synth_id") or "").casefold()
        if "dexed" in synth:
            return f"dexed:{index:04d}"
        if "surge" in synth:
            return f"surge:{index:06d}"
        raise ValueError(f"unknown broad synth: {synth!r}")
    raise ValueError(f"unknown source name: {source_name}")


def _category(value: object) -> str:
    category = _CATEGORY_ALIASES.get(str(value or "").strip().casefold())
    if category is None:
        raise ValueError(f"unknown Serum category: {value!r}")
    return category


def _resolve_audio(root_value: str, relative_value: object) -> str:
    root = Path(root_value).resolve()
    relative = Path(str(relative_value or ""))
    candidate = (
        relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"audio path escapes audio_root: {relative_value!r}"
        ) from error
    if not candidate.is_file():
        raise _MissingAudioError(f"missing audio: {candidate}")
    return str(candidate)


def _preset_categories(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    categories: dict[str, str] = {}
    for row in _read_jsonl(path):
        raw = str(row.get("preset_id") or "")
        if not raw:
            raise ValueError("preset metadata row lacks preset_id")
        category = _category(row.get("category"))
        previous = categories.setdefault(raw, category)
        if previous != category:
            raise ValueError(f"conflicting preset metadata: {raw}")
    return categories


def _column(
    columns: set[str],
    *candidates: str,
) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"registry lacks required column; tried {', '.join(candidates)}"
    )


def _registry_rows(source: FlowSourceConfig) -> Iterator[dict[str, Any]]:
    path = Path(source.manifest)
    uri = f"file:{path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )
        ]
        if {"data_management", "auxiliary_info"}.issubset(tables):
            query = """
                SELECT
                    dm.data_item_id AS sample_id,
                    dm.audio_path AS audio_path,
                    dm.source_id AS preset_id,
                    ai.pitch_midi AS midi_note,
                    ai.velocity AS velocity,
                    'steady' AS articulation_id,
                    dm.source_meta_json AS source_meta_json,
                    dm.duration_sec AS duration_seconds
                FROM data_management AS dm
                JOIN auxiliary_info AS ai
                    ON ai.data_item_id = dm.data_item_id
                WHERE dm.dataset_id = ?
                  AND dm.status = 'active'
                  AND dm.is_duplicate = 0
                  AND dm.is_silent = 0
                  AND ai.pitch_midi BETWEEN 21 AND 109
                  AND ai.velocity BETWEEN 0 AND 127
                  AND dm.duration_sec > 0
                ORDER BY dm.data_item_id
            """
            for result in connection.execute(
                query,
                (source.registry_dataset_id,),
            ):
                row = dict(result)
                try:
                    metadata = json.loads(
                        str(row.pop("source_meta_json") or "{}")
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid registry source_meta_json for "
                        f"{row.get('sample_id')}"
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "registry source_meta_json must be an object"
                    )
                row["category"] = metadata.get("category")
                yield row
            return
        selected: tuple[str, dict[str, str]] | None = None
        for table in tables:
            escaped = table.replace('"', '""')
            columns = {
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{escaped}")'
                )
            }
            try:
                aliases = {
                    "sample_id": _column(columns, "sample_id", "id"),
                    "dataset_id": _column(columns, "dataset_id"),
                    "audio_path": _column(
                        columns,
                        "audio_path",
                        "wav_path",
                        "file_path",
                        "path",
                    ),
                    "preset_id": _column(
                        columns,
                        "preset_id",
                        "timbre_id",
                        "instrument_id",
                    ),
                    "midi_note": _column(columns, "midi_note", "note"),
                    "velocity": _column(columns, "velocity"),
                    "category": _column(
                        columns,
                        "category",
                        "class_name",
                    ),
                    "duration": _column(
                        columns,
                        "duration_seconds",
                        "duration_sec",
                        "duration",
                    ),
                    "status": _column(columns, "status"),
                    "duplicate": _column(columns, "is_duplicate"),
                    "silent": _column(columns, "is_silent"),
                }
            except ValueError:
                continue
            aliases["articulation"] = (
                "articulation_id"
                if "articulation_id" in columns
                else ""
            )
            selected = (table, aliases)
            break
        if selected is None:
            raise ValueError("registry has no compatible sample table")
        table, aliases = selected

        def quoted(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        articulation = (
            quoted(aliases["articulation"])
            if aliases["articulation"]
            else "'steady'"
        )
        select = [
            f"{quoted(aliases['sample_id'])} AS sample_id",
            f"{quoted(aliases['audio_path'])} AS audio_path",
            f"{quoted(aliases['preset_id'])} AS preset_id",
            f"{quoted(aliases['midi_note'])} AS midi_note",
            f"{quoted(aliases['velocity'])} AS velocity",
            f"{articulation} AS articulation_id",
            f"{quoted(aliases['category'])} AS category",
            f"{quoted(aliases['duration'])} AS duration_seconds",
        ]
        query = (
            f"SELECT {', '.join(select)} FROM {quoted(table)} "
            f"WHERE {quoted(aliases['dataset_id'])}=? "
            f"AND {quoted(aliases['status'])}='active' "
            f"AND {quoted(aliases['duplicate'])}=0 "
            f"AND {quoted(aliases['silent'])}=0 "
            f"AND {quoted(aliases['midi_note'])} BETWEEN 21 AND 109 "
            f"AND {quoted(aliases['velocity'])} BETWEEN 0 AND 127 "
            f"AND {quoted(aliases['duration'])}>0 "
            "ORDER BY sample_id"
        )
        for row in connection.execute(
            query,
            (source.registry_dataset_id,),
        ):
            yield dict(row)
    finally:
        connection.close()


def _source_rows(source: FlowSourceConfig) -> Iterator[dict[str, Any]]:
    if source.kind == "registry":
        yield from _registry_rows(source)
    else:
        yield from _read_jsonl(source.manifest)


def _adapt_row(
    source: FlowSourceConfig,
    row: dict[str, Any],
    seed: int,
    preset_categories: dict[str, str],
) -> dict[str, object] | None:
    source_name = source.name
    if source_name == "serum_balanced":
        preset_index = _number(row.get("preset_index"), "preset_index")
        note = _number(row.get("midi_note"), "midi_note")
        velocity = _number(row.get("midi_velocity"), "midi_velocity")
        native_id = f"{preset_index:06d}-n{note:03d}-v{velocity:03d}"
        audio_value = row.get("source_path")
        duration_value = row.get("duration_seconds") or 5.0
        category = _CATEGORY_ALIASES.get(
            str(row.get("category") or "").strip().casefold()
        )
        if category is None or category not in source.allowed_categories:
            return None
        articulation = "steady"
    elif source_name == "dexed_surge_broad":
        synth_id = str(row.get("canonical_synth_id") or "").casefold()
        synth = "dexed" if "dexed" in synth_id else "surge"
        native_id = (
            f"broad-{synth}-{_number(row.get('preset_index'), 'preset_index'):06d}"
            f"-n{_number(row.get('midi_note'), 'midi_note'):03d}"
            f"-v{_number(row.get('velocity'), 'velocity'):03d}"
        )
        audio_value = row.get("wav_path")
        duration_value = row.get("duration_sec")
        category = "Dexed" if synth == "dexed" else "Surge"
        articulation = "steady"
    elif source_name == "pianobook_pitch":
        native_id = str(row.get("sample_id") or "")
        audio_value = row.get("audio_path")
        duration_value = row.get("duration_seconds")
        articulation = str(row.get("articulation_id") or "steady")
        category = "Pianobook"
    else:
        native_id = str(row.get("sample_id") or "")
        audio_value = row.get("audio_path")
        duration_value = row.get("duration_seconds")
        articulation = str(row.get("articulation_id") or "steady")
        if source_name == "serum_full":
            category = _CATEGORY_ALIASES.get(
                str(row.get("category") or "").strip().casefold()
            )
            if category is None:
                return None
            if category not in source.allowed_categories:
                return None
        else:
            raise ValueError(f"unknown source name: {source_name}")
    if not native_id:
        raise ValueError(f"{source_name} row lacks sample_id")
    note = _number(row.get("midi_note"), "midi_note")
    velocity = _number(
        row.get("midi_velocity")
        if source_name == "serum_balanced"
        else row.get("velocity"),
        "velocity",
    )
    try:
        duration = float(duration_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid duration: {duration_value!r}") from error
    if not 21 <= note <= 109:
        raise ValueError(f"midi_note outside 21-109: {note}")
    if not 0 <= velocity <= 127:
        raise ValueError(f"velocity outside 0-127: {velocity}")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"duration must be finite and positive: {duration}")
    canonical = canonical_preset_id(source_name, row)
    split = (
        str(row.get("split") or "")
        if source_name == "serum_balanced"
        else preset_split(canonical, seed)
    )
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid split: {split!r}")
    return {
        "sample_id": f"{source_name}__{native_id}",
        "source_name": source_name,
        "audio_path": _resolve_audio(source.audio_root, audio_value),
        "canonical_preset_id": canonical,
        "split": split,
        "midi_note": note,
        "velocity": velocity,
        "articulation_id": articulation,
        "category": category,
        "duration_seconds": duration,
        "maximum_future_frames": source.maximum_future_frames,
    }


def build_flow_manifest(config: ZraveFlowConfig) -> FlowManifestReport:
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    categories_by_preset: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    missing_audio: Counter[str] = Counter()
    for source in config.data.sources:
        preset_categories = _preset_categories(source.preset_metadata)
        source_hashes[source.name] = _sha256_file(source.manifest)
        for raw in _source_rows(source):
            try:
                row = _adapt_row(
                    source,
                    raw,
                    config.seed,
                    preset_categories,
                )
            except _MissingAudioError:
                missing_audio[source.name] += 1
                continue
            if row is None:
                continue
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            canonical = str(row["canonical_preset_id"])
            category = str(row["category"])
            previous = categories_by_preset.setdefault(canonical, category)
            if previous != category and canonical.startswith("serum:"):
                raise ValueError(
                    f"conflicting category for {canonical}: "
                    f"{previous} != {category}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("unified flow manifest would be empty")
    rows.sort(
        key=lambda row: (
            str(row["source_name"]),
            str(row["sample_id"]),
        )
    )
    manifest_path = Path(config.data.unified_manifest)
    _atomic_jsonl(manifest_path, rows)
    report = FlowManifestReport(
        rows=len(rows),
        hours=sum(float(row["duration_seconds"]) for row in rows) / 3600.0,
        by_source=dict(
            sorted(Counter(str(row["source_name"]) for row in rows).items())
        ),
        by_category=dict(
            sorted(Counter(str(row["category"]) for row in rows).items())
        ),
        by_note=dict(
            sorted(Counter(str(row["midi_note"]) for row in rows).items())
        ),
        by_split=dict(
            sorted(Counter(str(row["split"]) for row in rows).items())
        ),
        manifest_sha256=_sha256_file(manifest_path),
        source_sha256=source_hashes,
        missing_audio_by_source=dict(sorted(missing_audio.items())),
    )
    _atomic_json(Path(config.data.manifest_report), report.__dict__)
    return report
