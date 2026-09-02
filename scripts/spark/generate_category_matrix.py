#!/usr/bin/env python3
"""Materialize the Spark-only tiny Serum category matrix."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPES = Path(__file__).with_name("tiny_category_recipes.yaml")
LVZIHAO_GENERATOR = ROOT / "scripts/lvzihao/generate_category_matrix.py"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "_shared_category_matrix_generator", LVZIHAO_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared generator: {LVZIHAO_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_spark_recipe(path: Path) -> None:
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "template": "configs/zrave/spark_serum128_tiny_category_template.yaml",
        "config_output_dir": "configs/zrave/generated/spark_tiny_categories_v1",
        "queue_output": "scripts/spark/tiny_categories_v1.queue.tsv",
        "run_root": "runs/spark-tiny-categories-v1",
        "initialization": "scratch",
        "environment_prefix": "SP",
    }
    for name, value in expected.items():
        if recipe.get(name) != value:
            raise ValueError(f"Spark recipe {name} must be {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    recipes = args.recipes.resolve()
    if not recipes.is_relative_to(ROOT / "scripts/spark"):
        raise ValueError("Spark recipes must stay below scripts/spark")
    _validate_spark_recipe(recipes)
    generator = _generator()
    outputs = generator.materialize(recipes)
    if args.check:
        generator._check(outputs)
    else:
        generator._write(outputs)
        for path in outputs:
            print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
