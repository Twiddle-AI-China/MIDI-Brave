#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SERUM_CATEGORIES = {
    "Arp",
    "Atmosphere",
    "Bass",
    "Chord",
    "Drums",
    "FX",
    "Keys",
    "Lead",
    "Pad",
    "Pluck",
    "Synth",
    "Vocal",
}
REQUIRED_RECIPE_KEYS = {
    "schema",
    "template",
    "config_output_dir",
    "queue_output",
    "run_root",
    "experiment_prefix",
    "initialization",
    "maximum_updates",
    "checkpoints",
    "sweep_batches",
    "families",
}
OPTIONAL_RECIPE_KEYS = {
    "environment_prefix",
    "taxonomy_relative",
    "taxonomy_action",
    "audition_reports",
    "initializer_config_root",
    "initializer_run_root",
    "initializer_checkpoint",
}
FAMILY_REQUIRED_KEYS = {"id", "categories"}
FAMILY_OPTIONAL_KEYS = {"preset_allowlist"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _relative_path(value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{name} must not escape the repository")
    resolved = (ROOT / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"{name} must stay below the repository root")
    return relative.as_posix(), resolved


def _load_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return _mapping(value, "recipes")


def _validate_template(
    template: dict[str, Any],
    *,
    initialization: str,
) -> None:
    data = _mapping(template.get("data"), "template.data")
    sources = data.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("template must contain exactly one data source")
    source = _mapping(sources[0], "template.data.sources[0]")
    if source.get("name") != "serum_balanced":
        raise ValueError("template source must be serum_balanced")
    model = _mapping(template.get("model"), "template.model")
    expected_model = {
        "latent_dim": 128,
        "profile": "tiny",
        "context_frames": 32,
        "future_frames": 64,
        "d_model": 128,
        "context_layers": 2,
        "future_layers": 4,
        "heads": 4,
        "feedforward_dim": 512,
        "pitch_conditioning": initialization == "same_family_pure",
        "midi_sequence_conditioning": initialization == "same_family_pure",
    }
    for name, expected in expected_model.items():
        if model.get(name) != expected:
            raise ValueError(f"template model.{name} must be {expected!r}")
    segment = _mapping(
        template.get("segment_sampling"),
        "template.segment_sampling",
    )
    if segment.get("enabled") is not True:
        raise ValueError("template must enable segment sampling")
    train = _mapping(template.get("train"), "template.train")
    for name in (
        "short_future_updates",
        "checkpoint_every",
        "validation_every",
    ):
        if train.get(name) != 1000:
            raise ValueError(f"template train.{name} must be 1000")


def _validate_recipe(
    recipe: dict[str, Any],
) -> tuple[list[dict[str, object]], list[int], list[int]]:
    unknown = set(recipe) - REQUIRED_RECIPE_KEYS - OPTIONAL_RECIPE_KEYS
    missing = REQUIRED_RECIPE_KEYS - set(recipe)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise ValueError("invalid recipe keys: " + " ".join(details))
    if recipe["schema"] != 1:
        raise ValueError("recipes schema must be 1")
    environment_prefix = recipe.get("environment_prefix", "LV")
    if environment_prefix not in {"LV", "SP"}:
        raise ValueError("environment_prefix must be LV or SP")
    initialization = recipe["initialization"]
    if initialization not in {"scratch", "same_family_pure"}:
        raise ValueError("initialization must be scratch or same_family_pure")
    initializer_keys = {
        "initializer_config_root",
        "initializer_run_root",
        "initializer_checkpoint",
    }
    if initialization == "scratch":
        unexpected = initializer_keys & set(recipe)
        if unexpected:
            raise ValueError(
                "scratch recipes must not declare initializer keys: "
                + ", ".join(sorted(unexpected))
            )
    else:
        missing_initializer = initializer_keys - set(recipe)
        if missing_initializer:
            raise ValueError(
                "same_family_pure recipes are missing initializer keys: "
                + ", ".join(sorted(missing_initializer))
            )
        _relative_path(
            recipe["initializer_config_root"],
            "initializer_config_root",
        )
        initializer_run_root = recipe["initializer_run_root"]
        if (
            not isinstance(initializer_run_root, str)
            or PurePosixPath(initializer_run_root).is_absolute()
            or ".." in PurePosixPath(initializer_run_root).parts
            or "." in PurePosixPath(initializer_run_root).parts
            or not initializer_run_root.startswith("runs/")
        ):
            raise ValueError("initializer_run_root must be a safe runs/ relative path")
        initializer_checkpoint = recipe["initializer_checkpoint"]
        if (
            not isinstance(initializer_checkpoint, str)
            or re.fullmatch(r"step-[0-9]{6}\.pt", initializer_checkpoint) is None
        ):
            raise ValueError("initializer_checkpoint must be a step-NNNNNN.pt basename")
    audition_reports = recipe.get("audition_reports", True)
    if not isinstance(audition_reports, bool):
        raise TypeError("audition_reports must be a boolean")
    taxonomy_relative = recipe.get("taxonomy_relative")
    if taxonomy_relative is not None:
        taxonomy_relative, _taxonomy_path = _relative_path(
            taxonomy_relative,
            "taxonomy_relative",
        )
        if not taxonomy_relative.startswith("taxonomies/"):
            raise ValueError("taxonomy_relative must be below taxonomies/")
    taxonomy_action = recipe.get("taxonomy_action", "taxonomy")
    if taxonomy_action not in {"taxonomy", "taxonomy_validate_audio"}:
        raise ValueError("taxonomy_action must be taxonomy or taxonomy_validate_audio")
    if taxonomy_relative is None and "taxonomy_action" in recipe:
        raise ValueError("taxonomy_action requires taxonomy_relative")
    if (
        taxonomy_action == "taxonomy_validate_audio"
        and taxonomy_relative != "taxonomies/serum128-audio-v1"
    ):
        raise ValueError(
            "taxonomy_validate_audio requires taxonomies/serum128-audio-v1"
        )
    maximum_updates = _positive_integer(
        recipe["maximum_updates"],
        "maximum_updates",
    )
    checkpoints_value = recipe["checkpoints"]
    if not isinstance(checkpoints_value, list):
        raise TypeError("checkpoints must be a list")
    checkpoints = [
        _positive_integer(value, "checkpoint") for value in checkpoints_value
    ]
    if (
        not checkpoints
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] > maximum_updates
    ):
        raise ValueError(
            "checkpoints must be sorted, unique, and within maximum_updates"
        )
    batches_value = recipe["sweep_batches"]
    if not isinstance(batches_value, list):
        raise TypeError("sweep_batches must be a list")
    batches = [_positive_integer(value, "sweep batch") for value in batches_value]
    if not batches or batches != sorted(set(batches)):
        raise ValueError("sweep_batches must be sorted and unique")
    prefix = recipe["experiment_prefix"]
    if not isinstance(prefix, str) or not SAFE_ID.fullmatch(prefix):
        raise ValueError("experiment_prefix is unsafe")

    raw_families = recipe["families"]
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("families must be a non-empty list")
    families: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw_families):
        family = _mapping(value, f"families[{index}]")
        unknown_family = set(family) - FAMILY_REQUIRED_KEYS - FAMILY_OPTIONAL_KEYS
        missing_family = FAMILY_REQUIRED_KEYS - set(family)
        if unknown_family or missing_family:
            raise ValueError(
                f"families[{index}] must contain id/categories and may contain "
                "preset_allowlist"
            )
        family_id = family["id"]
        if not isinstance(family_id, str) or not SAFE_ID.fullmatch(family_id):
            raise ValueError(f"families[{index}].id is unsafe")
        if family_id in seen_ids:
            raise ValueError(f"duplicate family id: {family_id}")
        categories = family["categories"]
        if (
            not isinstance(categories, list)
            or not categories
            or any(not isinstance(item, str) for item in categories)
            or len(categories) != len(set(categories))
        ):
            raise ValueError(f"family {family_id} categories are invalid")
        unknown_categories = set(categories) - SERUM_CATEGORIES
        if unknown_categories:
            raise ValueError(
                f"family {family_id} has unknown categories: "
                + ", ".join(sorted(unknown_categories))
            )
        preset_allowlist = family.get("preset_allowlist")
        if preset_allowlist is not None:
            if not isinstance(preset_allowlist, str) or not preset_allowlist:
                raise ValueError(
                    f"family {family_id} preset_allowlist must be a non-empty path"
                )
            allowlist_path = PurePosixPath(preset_allowlist)
            if not allowlist_path.is_absolute() or ".." in allowlist_path.parts:
                raise ValueError(
                    f"family {family_id} preset_allowlist must be absolute"
                )
        seen_ids.add(family_id)
        families.append(
            {
                "id": family_id,
                "categories": list(categories),
                "preset_allowlist": preset_allowlist,
            }
        )
    has_allowlists = [family["preset_allowlist"] is not None for family in families]
    if taxonomy_relative is not None and not all(has_allowlists):
        raise ValueError("taxonomy recipes require preset_allowlist for every family")
    if taxonomy_relative is None and any(has_allowlists):
        raise ValueError("preset_allowlist recipes require taxonomy_relative")
    return families, checkpoints, batches


def materialize(recipes_path: Path) -> dict[Path, str]:
    recipe = _load_payload(recipes_path)
    families, checkpoints, batches = _validate_recipe(recipe)
    template_relative, template_path = _relative_path(
        recipe["template"],
        "template",
    )
    output_relative, output_root = _relative_path(
        recipe["config_output_dir"],
        "config_output_dir",
    )
    _queue_relative, queue_path = _relative_path(
        recipe["queue_output"],
        "queue_output",
    )
    run_root = recipe["run_root"]
    if (
        not isinstance(run_root, str)
        or PurePosixPath(run_root).is_absolute()
        or ".." in PurePosixPath(run_root).parts
        or not run_root.startswith("runs/")
    ):
        raise ValueError("run_root must be a safe runs/ relative path")
    prefix = str(recipe["experiment_prefix"])
    maximum_updates = int(recipe["maximum_updates"])
    template = _mapping(
        yaml.safe_load(template_path.read_text(encoding="utf-8")),
        "template",
    )
    initialization = str(recipe["initialization"])
    _validate_template(template, initialization=initialization)
    checkpoint_every = int(template["train"]["checkpoint_every"])
    if any(update % checkpoint_every for update in checkpoints):
        raise ValueError(
            "all reported checkpoints must align with template train.checkpoint_every"
        )
    packed_root = PurePosixPath(str(template["data"]["packed_root"]))
    container_root = packed_root.parent.parent
    taxonomy_relative = recipe.get("taxonomy_relative")
    if taxonomy_relative is not None:
        taxonomy_relative = PurePosixPath(str(taxonomy_relative)).as_posix()
    audition_reports = bool(recipe.get("audition_reports", True))
    environment_prefix = str(recipe.get("environment_prefix", "LV"))
    taxonomy_action = str(recipe.get("taxonomy_action", "taxonomy"))
    initializer_config_root: Path | None = None
    initializer_run_root: str | None = None
    initializer_checkpoint: str | None = None
    initializer_update: int | None = None
    if initialization == "same_family_pure":
        _initializer_config_relative, initializer_config_root = _relative_path(
            recipe["initializer_config_root"],
            "initializer_config_root",
        )
        initializer_run_root = str(recipe["initializer_run_root"]).rstrip("/")
        initializer_checkpoint = str(recipe["initializer_checkpoint"])
        initializer_update = int(initializer_checkpoint[5:11])
        if initializer_update <= 0:
            raise ValueError("initializer checkpoint update must be positive")

    outputs: dict[Path, str] = {}
    prelude_rows: list[tuple[str, str, str, str, str]] = []
    family_rows: list[tuple[str, str, str, str, str]] = []
    gate_rows: list[tuple[str, str, str, str, str]] = []
    if taxonomy_relative is not None:
        prelude_rows.append(
            (
                f"{prefix}-taxonomy",
                taxonomy_action,
                "-",
                "-",
                taxonomy_relative,
            )
        )
    for family in families:
        family_id = str(family["id"])
        categories = list(family["categories"])
        config_relative = f"{output_relative}/{family_id}.yaml"
        config_path = output_root / f"{family_id}.yaml"
        run_relative = f"{run_root.rstrip('/')}/{family_id}"
        config = copy.deepcopy(template)
        config["data"]["sources"][0]["allowed_categories"] = categories
        preset_allowlist = family["preset_allowlist"]
        if preset_allowlist is not None:
            allowlist_path = PurePosixPath(str(preset_allowlist))
            if not allowlist_path.is_relative_to(container_root):
                raise ValueError(
                    f"family {family_id} preset_allowlist must be below "
                    f"{container_root}"
                )
            expected_taxonomy = container_root / str(taxonomy_relative)
            if not allowlist_path.is_relative_to(expected_taxonomy / "buckets"):
                raise ValueError(
                    f"family {family_id} preset_allowlist must be below "
                    f"{expected_taxonomy}/buckets"
                )
            config["data"]["sources"][0]["preset_allowlist"] = str(allowlist_path)
        config["train"]["output_root"] = str(container_root / run_relative)
        config["train"]["max_updates"] = maximum_updates
        initializer_relative: str | None = None
        if initialization == "same_family_pure":
            assert initializer_config_root is not None
            assert initializer_run_root is not None
            assert initializer_checkpoint is not None
            assert initializer_update is not None
            pure_config_path = initializer_config_root / f"{family_id}.yaml"
            if not pure_config_path.is_file():
                raise ValueError(
                    f"family {family_id} initializer config does not exist: "
                    f"{pure_config_path}"
                )
            pure = _mapping(
                yaml.safe_load(pure_config_path.read_text(encoding="utf-8")),
                f"initializer config for {family_id}",
            )
            pure_data = _mapping(pure.get("data"), "initializer.data")
            pure_sources = pure_data.get("sources")
            if not isinstance(pure_sources, list) or len(pure_sources) != 1:
                raise ValueError(f"family {family_id} initializer must have one source")
            pure_source = _mapping(
                pure_sources[0],
                f"initializer source for {family_id}",
            )
            if pure_source.get("allowed_categories") != categories:
                raise ValueError(f"family {family_id} initializer categories mismatch")
            if pure_source.get("preset_allowlist") != preset_allowlist:
                raise ValueError(
                    f"family {family_id} initializer preset allowlist mismatch"
                )
            pure_model = _mapping(
                pure.get("model"),
                f"initializer model for {family_id}",
            )
            target_model = _mapping(
                config.get("model"),
                f"target model for {family_id}",
            )
            shared_model_fields = (
                "latent_dim",
                "profile",
                "context_frames",
                "future_frames",
                "d_model",
                "context_layers",
                "future_layers",
                "heads",
                "feedforward_dim",
            )
            if any(
                pure_model.get(name) != target_model.get(name)
                for name in shared_model_fields
            ):
                raise ValueError(
                    f"family {family_id} initializer profile/architecture mismatch"
                )
            if (
                pure_model.get("pitch_conditioning") is not False
                or pure_model.get("midi_sequence_conditioning") is not False
            ):
                raise ValueError(f"family {family_id} initializer must be pure flow")
            pure_train = _mapping(
                pure.get("train"),
                f"initializer train config for {family_id}",
            )
            expected_pure_output = str(
                container_root / initializer_run_root / family_id
            )
            if pure_train.get("output_root") != expected_pure_output:
                raise ValueError(f"family {family_id} initializer output_root mismatch")
            pure_maximum = _positive_integer(
                pure_train.get("max_updates"),
                f"initializer {family_id} max_updates",
            )
            pure_checkpoint_every = _positive_integer(
                pure_train.get("checkpoint_every"),
                f"initializer {family_id} checkpoint_every",
            )
            if (
                initializer_update > pure_maximum
                or initializer_update % pure_checkpoint_every
            ):
                raise ValueError(
                    f"family {family_id} cannot produce {initializer_checkpoint}"
                )
            initializer_relative = (
                f"{initializer_run_root}/{family_id}/checkpoints/"
                f"{initializer_checkpoint}"
            )
        config_text = (
            "# Generated by scripts/lvzihao/generate_category_matrix.py\n"
            f"# Source recipe: {recipes_path.relative_to(ROOT).as_posix()}\n"
            f"# Source template: {template_relative}\n"
            + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
        outputs[config_path] = config_text
        if initialization == "scratch":
            family_rows.extend(
                (
                    (
                        f"{prefix}-{family_id}-smoke",
                        "smoke",
                        config_relative,
                        run_relative,
                        "-",
                    ),
                    (
                        f"{prefix}-{family_id}-sweep",
                        "sweep",
                        config_relative,
                        run_relative,
                        ",".join(str(value) for value in batches),
                    ),
                    (
                        f"{prefix}-{family_id}-train",
                        "train",
                        config_relative,
                        run_relative,
                        str(maximum_updates),
                    ),
                )
            )
        else:
            assert initializer_relative is not None
            family_rows.extend(
                (
                    (
                        f"{prefix}-{family_id}-smoke",
                        "midi_smoke_from",
                        config_relative,
                        run_relative,
                        f"-|{initializer_relative}",
                    ),
                    (
                        f"{prefix}-{family_id}-sweep",
                        "midi_sweep_from",
                        config_relative,
                        run_relative,
                        (
                            f"{','.join(str(value) for value in batches)}|"
                            f"{initializer_relative}"
                        ),
                    ),
                    (
                        f"{prefix}-{family_id}-train",
                        "midi_train_from",
                        config_relative,
                        run_relative,
                        f"{maximum_updates}|{initializer_relative}",
                    ),
                )
            )
        if audition_reports:
            gate_rows.extend(
                (
                    f"{prefix}-{family_id}-audition-{update:06d}",
                    (
                        "audition_report"
                        if initialization == "scratch"
                        else "midi_audition_report_from"
                    ),
                    config_relative,
                    run_relative,
                    (
                        f"step-{update:06d}.pt"
                        if initializer_relative is None
                        else f"step-{update:06d}.pt|{initializer_relative}"
                    ),
                )
                for update in checkpoints
            )
    rows = [*prelude_rows, *family_rows, *gate_rows]
    if any("audition" in row[1] for row in family_rows):
        raise AssertionError("training phase unexpectedly contains a gate")
    queue_text = (
        "# Generated by scripts/lvzihao/generate_category_matrix.py; do not edit.\n"
        f"# Source recipe: {recipes_path.relative_to(ROOT).as_posix()}\n"
        + (
            "# Pure tiny category models train from scratch; "
            f"{environment_prefix}_INITIALIZE_FROM must be unset.\n"
            if initialization == "scratch"
            else "# Tiny MIDI models use the per-row same-family pure "
            "initializer; global LV_INITIALIZE_FROM is forbidden.\n"
            "# LV_PITCH_PROBE must point to one globally qualified probe.\n"
        )
        + "# All model training completes before the first audio gate.\n"
        + (
            "# Audition reports are intentionally omitted: the current renderer "
            "selects by category but cannot constrain canonical preset IDs to "
            "preset_allowlist. The 1k/5k/10k checkpoints are still produced.\n"
            if not audition_reports
            else ""
        )
        + "# id\taction\tconfig relative to MidiBrave-v2\t"
        "run relative to HOST_FLOW_ROOT\tspec\n"
        + "".join("\t".join(row) + "\n" for row in rows)
    )
    outputs[queue_path] = queue_text
    if audition_reports:
        suffix = ".queue.tsv"
        if not queue_path.name.endswith(suffix):
            raise ValueError("queue_output must end in .queue.tsv")
        clap_queue_path = queue_path.with_name(
            queue_path.name[: -len(suffix)] + ".clap.queue.tsv"
        )
        clap_queue_text = (
            "# Generated by scripts/lvzihao/generate_category_matrix.py; "
            "do not edit.\n"
            f"# Source recipe: {recipes_path.relative_to(ROOT).as_posix()}\n"
            "# Report-only CLAP jobs are deliberately separate from training "
            "and audition queues.\n"
            "# id\taction\tconfig relative to MidiBrave-v2\t"
            "run relative to HOST_FLOW_ROOT\tspec\n"
            + "".join(
                "\t".join(
                    (
                        f"{row[0]}-clap",
                        "clap_report",
                        "-",
                        "-",
                        row[0],
                    )
                )
                + "\n"
                for row in gate_rows
            )
        )
        outputs[clap_queue_path] = clap_queue_text
    return outputs


def _write(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def _check(outputs: dict[Path, str]) -> None:
    stale = [
        path.relative_to(ROOT).as_posix()
        for path, expected in outputs.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != expected
    ]
    if stale:
        raise SystemExit("stale generated category matrix: " + ", ".join(stale))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny Serum category configs and serial queue."
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path(__file__).with_name("tiny_category_recipes.yaml"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipes_path = args.recipes.resolve()
    if not recipes_path.is_relative_to(ROOT):
        raise ValueError("recipes must stay below the repository root")
    outputs = materialize(recipes_path)
    if args.check:
        _check(outputs)
    else:
        _write(outputs)
        for path in outputs:
            print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
