from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .zrave_flow_config import ZraveFlowConfig


@dataclass(frozen=True)
class FlowBatch:
    history: Tensor
    future: Tensor
    future_mask: Tensor
    midi_note: Tensor
    source_code: Tensor
    category_code: Tensor
    wander_delay_frames: Tensor
    history_midi_note: Tensor
    pitch_transition_mask: Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _largest_remainder(
    total: int,
    weights: tuple[float, ...],
) -> list[int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("allocation weights must have positive sum")
    raw = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


class GpuFlowSampler:
    def __init__(
        self,
        *,
        latents: Tensor,
        lengths: Tensor,
        active_frames: Tensor,
        notes: Tensor,
        velocities: Tensor,
        split_codes: Tensor,
        source_codes: Tensor,
        category_codes: Tensor,
        maximum_future_frames: Tensor,
        pitch_pairs: Tensor,
        source_weights: tuple[float, ...],
        context_frames: int,
        future_frames: int,
        wander_delays: tuple[int, ...],
        pitch_transition_fraction: float,
        seed: int,
        device: str | torch.device,
        allowed_split_code: int | None = None,
    ) -> None:
        if latents.ndim != 3:
            raise ValueError("latents must be [records, frames, channels]")
        records = latents.shape[0]
        metadata = (
            lengths,
            active_frames,
            notes,
            velocities,
            split_codes,
            source_codes,
            category_codes,
            maximum_future_frames,
            pitch_pairs,
        )
        if any(tensor.numel() != records for tensor in metadata):
            raise ValueError("sampler metadata length mismatch")
        if context_frames <= 0 or future_frames <= 0:
            raise ValueError("context and future frames must be positive")
        if future_frames != 64:
            raise ValueError("future_frames must be 64")
        if tuple(wander_delays) != (16, 32, 48):
            raise ValueError("wander delays must be 16, 32, 48")
        if not 0.0 <= pitch_transition_fraction <= 1.0:
            raise ValueError("pitch transition fraction must be in [0, 1]")
        if len(source_weights) != 4 or not math.isclose(
            sum(source_weights),
            1.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("source_weights must contain four values summing to 1")
        self.device = torch.device(device)
        self.latents = latents.to(self.device)
        self.lengths = lengths.to("cpu", dtype=torch.long)
        self.active_frames = active_frames.to("cpu", dtype=torch.long)
        self.notes = notes.to("cpu", dtype=torch.long)
        self.velocities = velocities.to("cpu", dtype=torch.long)
        self.split_codes = split_codes.to("cpu", dtype=torch.long)
        self.source_codes = source_codes.to("cpu", dtype=torch.long)
        self.category_codes = category_codes.to("cpu", dtype=torch.long)
        self.maximum_future_frames = maximum_future_frames.to(
            "cpu",
            dtype=torch.long,
        )
        self.pitch_pairs = pitch_pairs.to("cpu", dtype=torch.long)
        self.source_weights = tuple(float(value) for value in source_weights)
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.wander_delays = tuple(int(value) for value in wander_delays)
        self.pitch_transition_fraction = pitch_transition_fraction
        self.allowed_split_code = allowed_split_code
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        if torch.any(self.lengths > latents.shape[1]):
            raise ValueError("length exceeds padded latent frames")
        if torch.any(self.active_frames > self.lengths):
            raise ValueError("active_frames exceeds length")
        valid_pairs = self.pitch_pairs >= 0
        if torch.any(self.pitch_pairs[valid_pairs] >= records):
            raise ValueError("pitch pair index is out of range")

    @classmethod
    def from_pack(
        cls,
        config: ZraveFlowConfig,
        split: str,
        device: str | torch.device,
        seed: int,
    ) -> "GpuFlowSampler":
        split_codes = {"train": 0, "validation": 1, "test": 2}
        if split not in split_codes:
            raise ValueError(f"invalid split: {split}")
        root = Path(config.data.packed_root)
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["rave_checkpoint_sha256"] != config.rave.expected_sha256:
            raise ValueError("pack codec hash does not match config")
        shard_paths = [
            root / name for name in sorted(index["shard_sha256"])
        ]
        loaded: list[dict[str, np.ndarray]] = []
        maximum_length = 0
        for path in shard_paths:
            if _sha256_file(path) != index["shard_sha256"][path.name]:
                raise ValueError(f"pack shard hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as shard:
                values = {name: shard[name].copy() for name in shard.files}
            loaded.append(values)
            maximum_length = max(
                maximum_length,
                values["latents"].shape[1],
            )
        records = sum(values["latents"].shape[0] for values in loaded)
        latents = np.zeros(
            (records, maximum_length, config.model.latent_dim),
            dtype=np.float16,
        )
        names = (
            "lengths",
            "active_frames",
            "notes",
            "velocities",
            "split_codes",
            "source_codes",
            "category_codes",
            "maximum_future_frames",
        )
        combined: dict[str, list[np.ndarray]] = {
            name: [] for name in names
        }
        offset = 0
        for values in loaded:
            count, frames, _channels = values["latents"].shape
            latents[offset : offset + count, :frames] = values["latents"]
            for name in names:
                combined[name].append(values[name])
            offset += count
        pitch_path = root / "pitch-pairs.npy"
        if _sha256_file(pitch_path) != index["pitch_pairs_sha256"]:
            raise ValueError("pitch-pairs hash mismatch")
        return cls(
            latents=torch.from_numpy(latents),
            **{
                name: torch.from_numpy(np.concatenate(parts))
                for name, parts in combined.items()
            },
            pitch_pairs=torch.from_numpy(
                np.load(pitch_path, allow_pickle=False)
            ),
            source_weights=tuple(
                source.weight for source in config.data.sources
            ),
            context_frames=config.model.context_frames,
            future_frames=config.model.future_frames,
            wander_delays=config.model.wander_delay_frames,
            pitch_transition_fraction=(
                config.train.pitch_transition_fraction
            ),
            seed=seed,
            device=device,
            allowed_split_code=split_codes[split],
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if set(state) != {"generator_state"}:
            raise ValueError("invalid sampler state")
        self.generator.set_state(state["generator_state"].cpu())

    def _eligible(
        self,
        source: int,
        require_full_future: bool,
    ) -> Tensor:
        mask = self.source_codes == source
        if self.allowed_split_code is not None:
            mask &= self.split_codes == self.allowed_split_code
        if require_full_future:
            mask &= self.active_frames >= (
                self.context_frames + self.future_frames
            )
            mask &= self.maximum_future_frames >= self.future_frames
        else:
            mask &= self.active_frames >= self.context_frames + 1
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _draw(self, pool: Tensor, count: int) -> Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        if pool.numel() == 0:
            raise ValueError("requested sampler bucket has no eligible rows")
        offsets = torch.randint(
            pool.numel(),
            (count,),
            generator=self.generator,
        )
        return pool[offsets]

    def _draw_source(
        self,
        source: int,
        count: int,
        *,
        require_full_future: bool,
        transition_count: int,
    ) -> tuple[Tensor, Tensor]:
        pool = self._eligible(source, require_full_future)
        if source == 0 and count:
            category_values = sorted(
                {
                    int(value)
                    for value in self.category_codes[pool].tolist()
                }
            )
            category_counts = _largest_remainder(
                count,
                tuple(1.0 for _ in category_values),
            )
            selected = torch.cat(
                [
                    self._draw(
                        pool[self.category_codes[pool] == category],
                        category_count,
                    )
                    for category, category_count in zip(
                        category_values,
                        category_counts,
                        strict=True,
                    )
                ]
            )
            return selected, torch.zeros(count, dtype=torch.bool)
        if transition_count:
            transition_pool = pool[
                self.pitch_pairs[pool] >= 0
            ]
            partner = self.pitch_pairs[transition_pool]
            transition_pool = transition_pool[
                self.active_frames[partner] >= self.context_frames
            ]
            transition_rows = self._draw(
                transition_pool,
                transition_count,
            )
            normal_rows = self._draw(pool, count - transition_count)
            return (
                torch.cat([transition_rows, normal_rows]),
                torch.cat(
                    [
                        torch.ones(transition_count, dtype=torch.bool),
                        torch.zeros(
                            count - transition_count,
                            dtype=torch.bool,
                        ),
                    ]
                ),
            )
        return self._draw(pool, count), torch.zeros(count, dtype=torch.bool)

    def sample(
        self,
        batch_size: int,
        maximum_valid_future: int,
        require_full_future: bool = False,
    ) -> FlowBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 1 <= maximum_valid_future <= self.future_frames:
            raise ValueError("maximum_valid_future must be in [1, 64]")
        weights = list(self.source_weights)
        if require_full_future:
            weights[2] = 0.0
        source_counts = _largest_remainder(batch_size, tuple(weights))
        transition_total = (
            0
            if require_full_future
            else math.floor(batch_size * self.pitch_transition_fraction)
        )
        dense_total = source_counts[1] + source_counts[2]
        if transition_total > dense_total:
            raise ValueError("not enough dense source slots for transitions")
        dense_transition_counts = _largest_remainder(
            transition_total,
            (
                float(source_counts[1]),
                float(source_counts[2]),
            ),
        )
        transition_by_source = {
            1: dense_transition_counts[0],
            2: dense_transition_counts[1],
        }

        selected_parts: list[Tensor] = []
        transition_parts: list[Tensor] = []
        for source, count in enumerate(source_counts):
            rows, transition = self._draw_source(
                source,
                count,
                require_full_future=require_full_future,
                transition_count=transition_by_source.get(source, 0),
            )
            selected_parts.append(rows)
            transition_parts.append(transition)
        selected = torch.cat(selected_parts)
        transitions = torch.cat(transition_parts)
        permutation = torch.randperm(
            batch_size,
            generator=self.generator,
        )
        selected = selected[permutation]
        transitions = transitions[permutation]

        history = torch.empty(
            batch_size,
            self.context_frames,
            self.latents.shape[-1],
            dtype=torch.float32,
            device=self.device,
        )
        future = torch.zeros(
            batch_size,
            self.future_frames,
            self.latents.shape[-1],
            dtype=torch.float32,
            device=self.device,
        )
        future_mask = torch.zeros(
            batch_size,
            self.future_frames,
            dtype=torch.bool,
            device=self.device,
        )
        history_indices = selected.clone()
        history_indices[transitions] = self.pitch_pairs[
            selected[transitions]
        ]
        for batch_index in range(batch_size):
            target = int(selected[batch_index])
            history_index = int(history_indices[batch_index])
            valid_future = min(
                maximum_valid_future,
                int(self.maximum_future_frames[target]),
                int(self.active_frames[target]) - self.context_frames,
            )
            if require_full_future and valid_future != self.future_frames:
                raise ValueError("full-future sampler selected a short row")
            if valid_future <= 0:
                raise ValueError("sampled row has no valid future")
            maximum_start = min(
                int(self.active_frames[target])
                - self.context_frames
                - valid_future,
                int(self.active_frames[history_index])
                - self.context_frames,
            )
            if maximum_start < 0:
                raise ValueError("pitch partner has insufficient active frames")
            start = int(
                torch.randint(
                    maximum_start + 1,
                    (1,),
                    generator=self.generator,
                )
            )
            history[batch_index] = self.latents[
                history_index,
                start : start + self.context_frames,
            ].float()
            future[batch_index, :valid_future] = self.latents[
                target,
                start
                + self.context_frames : start
                + self.context_frames
                + valid_future,
            ].float()
            future_mask[batch_index, :valid_future] = True
        wander_indices = torch.randint(
            len(self.wander_delays),
            (batch_size,),
            generator=self.generator,
        )
        wander = torch.tensor(
            self.wander_delays,
            dtype=torch.long,
        )[wander_indices]
        return FlowBatch(
            history=history,
            future=future,
            future_mask=future_mask,
            midi_note=self.notes[selected].to(self.device),
            source_code=self.source_codes[selected].to(self.device),
            category_code=self.category_codes[selected].to(self.device),
            wander_delay_frames=wander.to(self.device),
            history_midi_note=self.notes[history_indices].to(self.device),
            pitch_transition_mask=transitions.to(self.device),
        )
