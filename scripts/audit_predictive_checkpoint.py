#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from midibrave.config import Config
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.trainer import PredictiveStage, configure_predictive_stage


def tensor_stats(value: torch.Tensor) -> dict[str, float | int | bool]:
    item = value.detach().float()
    finite = torch.isfinite(item)
    finite_values = item[finite]
    return {
        "elements": item.numel(),
        "nonfinite": int((~finite).sum().item()),
        "maximum_absolute": (
            float(finite_values.abs().max().item()) if finite_values.numel() else 0.0),
        "rms": (
            float(finite_values.square().mean().sqrt().item())
            if finite_values.numel() else 0.0),
    }


def merge_group(groups: dict[str, dict[str, Any]], root: str, name: str,
                value: torch.Tensor) -> None:
    stats = tensor_stats(value)
    group = groups.setdefault(root, {
        "tensors": 0, "elements": 0, "nonfinite": 0,
        "maximum_absolute": 0.0, "maximum_tensor": "",
    })
    group["tensors"] += 1
    group["elements"] += stats["elements"]
    group["nonfinite"] += stats["nonfinite"]
    if stats["maximum_absolute"] > group["maximum_absolute"]:
        group["maximum_absolute"] = stats["maximum_absolute"]
        group["maximum_tensor"] = name


def audit(config: Config, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)
    configure_predictive_stage(model, PredictiveStage.RAVE)
    trainable_names = [name for name, value in model.named_parameters()
                       if value.requires_grad]

    model_groups: dict[str, dict[str, Any]] = {}
    tensor_ranking = []
    for name, value in checkpoint["model"].items():
        if not isinstance(value, torch.Tensor):
            continue
        stats = tensor_stats(value)
        merge_group(model_groups, name.split(".", 1)[0], name, value)
        tensor_ranking.append((stats["maximum_absolute"], name, stats["rms"]))

    optimizer = checkpoint["optimizer"]
    parameter_ids = optimizer["param_groups"][0]["params"]
    parameter_names = dict(zip(parameter_ids, trainable_names))
    optimizer_groups: dict[str, dict[str, Any]] = {}
    optimizer_ranking = []
    for parameter_id, state in optimizer["state"].items():
        name = parameter_names.get(parameter_id, f"unknown:{parameter_id}")
        root = name.split(".", 1)[0]
        for state_name in ("exp_avg", "exp_avg_sq"):
            value = state.get(state_name)
            if not isinstance(value, torch.Tensor):
                continue
            label = f"{name}:{state_name}"
            stats = tensor_stats(value)
            merge_group(optimizer_groups, root, label, value)
            optimizer_ranking.append(
                (stats["maximum_absolute"], label, stats["rms"]))

    return {
        "checkpoint": str(checkpoint_path),
        "stage_update": int(checkpoint["stage_update"]),
        "scaler": checkpoint["scaler"],
        "model_groups": model_groups,
        "optimizer_groups": optimizer_groups,
        "largest_model_tensors": [
            {"name": name, "maximum_absolute": maximum, "rms": rms}
            for maximum, name, rms in sorted(tensor_ranking, reverse=True)[:20]
        ],
        "largest_optimizer_tensors": [
            {"name": name, "maximum_absolute": maximum, "rms": rms}
            for maximum, name, rms in sorted(optimizer_ranking, reverse=True)[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.predictive is None:
        raise ValueError("predictive checkpoint audit requires a predictive config")
    reports = [audit(config, Path(path)) for path in args.checkpoints]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
