from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedSelection:
    index: int
    latent: np.ndarray
    sample_id: str
    distance: float


class SeedBank:
    def __init__(self, latents: np.ndarray, clap: np.ndarray, notes: np.ndarray,
                 velocities: np.ndarray, sample_ids: list[str],
                 metadata: dict[str, Any]):
        self.latents = np.asarray(latents, dtype=np.float32)
        self.clap = np.asarray(clap, dtype=np.float32)
        self.notes = np.asarray(notes, dtype=np.int64)
        self.velocities = np.asarray(velocities, dtype=np.float32)
        self.sample_ids = list(sample_ids)
        self.metadata = dict(metadata)
        count = self.latents.shape[0] if self.latents.ndim == 3 else 0
        if not count:
            raise ValueError("seed bank cannot be empty")
        if self.clap.ndim != 2 or self.clap.shape[0] != count:
            raise ValueError("seed CLAP matrix must align with latents")
        if self.notes.shape != (count,) or self.velocities.shape != (count,):
            raise ValueError("seed MIDI vectors must align with latents")
        if len(self.sample_ids) != count:
            raise ValueError("seed sample ids must align with latents")
        if not np.isfinite(self.latents).all() or not np.isfinite(self.clap).all():
            raise ValueError("seed bank rejects non-finite values")
        contracts = {
            "latent_dim": self.latents.shape[1],
            "history_frames": self.latents.shape[2],
        }
        for field, value in contracts.items():
            if field in self.metadata and self.metadata[field] != value:
                raise ValueError(f"seed bank {field} mismatch")

    def select(self, control: np.ndarray, top_k: int,
               random_seed: int) -> SeedSelection:
        control = np.asarray(control, dtype=np.float32)
        if control.shape != (self.clap.shape[1],):
            raise ValueError("CLAP control dimension does not match seed bank")
        norm = float(np.linalg.norm(control))
        bank_norm = np.linalg.norm(self.clap, axis=1)
        if norm <= 0.0 or np.any(bank_norm <= 0.0):
            raise ValueError("CLAP controls must have non-zero norm")
        similarity = (self.clap @ control) / (bank_norm * norm)
        candidates = np.argsort(-similarity, kind="stable")[:max(1, min(top_k, len(similarity)))]
        rng = np.random.default_rng(random_seed)
        index = int(candidates[int(rng.integers(0, len(candidates)))])
        return SeedSelection(index, self.latents[index].copy(), self.sample_ids[index],
                             float(1.0 - similarity[index]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = path.with_suffix(".npz")
        description = path.with_suffix(".json")
        arrays_tmp = arrays.with_suffix(".npz.tmp")
        json_tmp = description.with_suffix(".json.tmp")
        with arrays_tmp.open("wb") as output:
            np.savez_compressed(output, latents=self.latents, clap=self.clap,
                                notes=self.notes, velocities=self.velocities)
        json_tmp.write_text(json.dumps({
            "metadata": self.metadata, "sample_ids": self.sample_ids,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arrays_tmp.replace(arrays)
        json_tmp.replace(description)

    @classmethod
    def load(cls, path: str | Path, expected: dict[str, Any]) -> "SeedBank":
        path = Path(path)
        description = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        metadata = description["metadata"]
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"seed bank {field} mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            return cls(arrays["latents"], arrays["clap"], arrays["notes"],
                       arrays["velocities"], description["sample_ids"], metadata)
