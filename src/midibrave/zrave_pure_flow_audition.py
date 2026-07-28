from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .zrave_flow_model import (
    FlowStatistics,
    sample_pure_flow_block,
)


_DEFAULT_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")


def rollout_pure_flow(
    model: object,
    statistics: FlowStatistics,
    history: Tensor,
    frames: int,
    *,
    generation_seed: int,
    temperature: float,
    wander_delay_frames: int,
    solver_steps: int,
    sample_block: Callable[..., Tensor] = sample_pure_flow_block,
) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if history.ndim != 3 or history.shape[1:] != (32, 16):
        raise ValueError("history must have shape [batch, 32, 16]")
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    block_index = 0
    while remaining:
        generated = sample_block(
            model,
            statistics,
            current,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            solver_steps=solver_steps,
        )
        expected = (history.shape[0], 64, 16)
        if generated.shape != expected:
            raise ValueError(
                "pure flow block sampler must return [batch, 64, 16]"
            )
        take = min(remaining, 64)
        chunk = generated[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[:, -32:].detach()
        remaining -= take
        block_index += 1
    return torch.cat(chunks, dim=1)


def select_audition_cases(
    sequence_manifest: str | Path,
    *,
    categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
    seed: int = 20260728,
) -> list[dict[str, Any]]:
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be non-empty and unique")
    selected: dict[str, tuple[bytes, dict[str, Any]]] = {}
    path = Path(sequence_manifest)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            category = str(row.get("category", ""))
            if (
                category not in categories
                or str(row.get("split")) != "test"
                or str(row.get("source_name")) != "serum_full"
                or int(row.get("active_frames", 0)) < 96
                or int(row.get("maximum_future_frames", 0)) < 64
            ):
                continue
            key = (
                f"{seed}:{category}:"
                f"{row.get('canonical_preset_id')}:"
                f"{row.get('sample_id')}"
            )
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            current = selected.get(category)
            if current is None or digest < current[0]:
                selected[category] = (digest, row)
    missing = [category for category in categories if category not in selected]
    if missing:
        raise ValueError(
            "test split lacks eligible audition rows: "
            + ", ".join(missing)
        )
    return [selected[category][1] for category in categories]


def validate_pure_checkpoint(
    payload: object,
    config: object,
    *,
    index_hash: str,
    statistics_hash: str,
    expected_update: int | None = None,
) -> None:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("pure flow checkpoint format mismatch")
    if payload.get("architecture") != "zrave_pure_flow_transformer_v1":
        raise ValueError("checkpoint is not a pure flow model")
    model = getattr(config, "model")
    if bool(getattr(model, "pitch_conditioning")):
        raise ValueError("audition config is not pure flow")
    if expected_update is not None and int(payload.get("update", -1)) != int(
        expected_update
    ):
        raise ValueError("pure flow checkpoint update mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("pure flow checkpoint lacks contract")
    expected = {
        "pack_index_sha256": index_hash,
        "statistics_sha256": statistics_hash,
        "latent_dim": int(getattr(model, "latent_dim")),
        "context_frames": int(getattr(model, "context_frames")),
        "future_frames": int(getattr(model, "future_frames")),
        "pitch_conditioning": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"pure flow checkpoint {name} mismatch")
