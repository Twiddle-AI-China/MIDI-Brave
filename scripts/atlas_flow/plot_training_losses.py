from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    ("stage1", "Stage 1 · audio trajectory reconstruction", "#2563eb"),
    ("flow", "Stage 2 · Flow Matching", "#7c3aed"),
    ("joint", "Stage 3 · joint refinement", "#059669"),
)


def _read_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    by_update: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        update = row.get("update")
        loss = row.get("loss")
        if isinstance(update, int) and isinstance(loss, (int, float)) and math.isfinite(loss):
            by_update[update] = float(loss)
    if not by_update:
        raise ValueError(f"no finite loss records in {path}")
    updates = np.asarray(sorted(by_update), dtype=np.int64)
    losses = np.asarray([by_update[int(update)] for update in updates], dtype=np.float64)
    return updates, losses


def _rolling_median(values: np.ndarray) -> np.ndarray:
    window = max(5, min(101, len(values) // 80 * 2 + 1))
    if len(values) < window:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sanitized Atlas Flow training losses")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 9.0))
    figure.suptitle("midiBrave Atlas Flow v5 · Three-stage training loss", fontsize=16, weight="bold")

    for axis, (stage, title, color) in zip(axes, STAGES):
        updates, losses = _read_metrics(args.metrics_dir / f"{stage}.jsonl")
        stride = max(1, len(updates) // 1_500)
        axis.plot(
            updates[::stride], losses[::stride],
            color=color, alpha=0.20, linewidth=0.75, label="logged loss",
        )
        axis.plot(
            updates, _rolling_median(losses),
            color=color, linewidth=1.8, label="rolling median",
        )
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlabel("Effective update")
        axis.set_ylabel("Training loss")
        axis.grid(True, color="#d1d5db", alpha=0.55, linewidth=0.6)
        axis.legend(loc="lower right", frameon=False, fontsize=8)
        axis.text(
            0.995, 0.95,
            f"first {losses[0]:.4f}  →  final {losses[-1]:.4f}",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            color="#374151",
        )

    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.text(
        0.01, 0.015,
        "Joint loss mixes reconstruction and flow terms and is not directly comparable to either preceding stage.",
        fontsize=8, color="#4b5563",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
