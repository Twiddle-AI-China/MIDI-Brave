from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
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
    velocity: Tensor | None = None
    history_velocity: Tensor | None = None
    history_mask: Tensor | None = None
    segment_division: Tensor | None = None
    segment_index: Tensor | None = None
    absolute_start: Tensor | None = None
    history_midi_sequence: Tensor | None = None
    future_midi_sequence: Tensor | None = None
    history_velocity_sequence: Tensor | None = None
    future_velocity_sequence: Tensor | None = None
    midi_event_frame: Tensor | None = None


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


def build_different_note_pair_graph(
    rows: list[dict[str, Any]],
) -> Tensor:
    """Map each row to a deterministic same-preset, different-note row.

    The packed graph remains one integer per row, so existing packs and
    checkpoints do not change shape.  Unlike the legacy graph, eligibility is
    not restricted to an octave difference.  Nearest pitch is preferred and
    stable sample identifiers break ties.
    """

    required = {
        "source_name",
        "split",
        "canonical_preset_id",
        "velocity",
        "articulation_id",
        "midi_note",
        "sample_id",
    }
    buckets: dict[
        tuple[str, str, str, int, str],
        list[tuple[int, int, str]],
    ] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                "pitch pair row is missing: "
                + ", ".join(sorted(missing))
            )
        key = (
            str(row["source_name"]),
            str(row["split"]),
            str(row["canonical_preset_id"]),
            int(row["velocity"]),
            str(row["articulation_id"]),
        )
        buckets[key].append(
            (index, int(row["midi_note"]), str(row["sample_id"]))
        )

    result = torch.full((len(rows),), -1, dtype=torch.long)
    for members in buckets.values():
        for index, note, _sample_id in members:
            candidates = [
                candidate
                for candidate in members
                if candidate[1] != note
            ]
            if candidates:
                partner, _partner_note, _partner_id = min(
                    candidates,
                    key=lambda candidate: (
                        abs(candidate[1] - note),
                        candidate[1],
                        candidate[2],
                        candidate[0],
                    ),
                )
                result[index] = partner
    return result


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
        record_eligible: Tensor | None = None,
        segment_sampling_enabled: bool = False,
        segment_divisions: tuple[int, ...] = (2, 4, 8),
        segment_include_first: bool = False,
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
        if record_eligible is not None and record_eligible.numel() != records:
            raise ValueError("record eligibility length mismatch")
        if context_frames <= 0 or future_frames <= 0:
            raise ValueError("context and future frames must be positive")
        if future_frames != 64:
            raise ValueError("future_frames must be 64")
        if tuple(wander_delays) != (16, 32, 48):
            raise ValueError("wander delays must be 16, 32, 48")
        if not 0.0 <= pitch_transition_fraction <= 1.0:
            raise ValueError("pitch transition fraction must be in [0, 1]")
        if not isinstance(segment_sampling_enabled, bool):
            raise ValueError("segment_sampling_enabled must be boolean")
        if not isinstance(segment_include_first, bool):
            raise ValueError("segment_include_first must be boolean")
        segment_divisions = tuple(segment_divisions)
        if (
            not segment_divisions
            or tuple(sorted(set(segment_divisions))) != segment_divisions
            or any(value not in {2, 4, 8} for value in segment_divisions)
        ):
            raise ValueError(
                "segment_divisions must be an ordered, unique subset "
                "of 2, 4, 8"
            )
        if segment_include_first:
            raise ValueError(
                "first-segment sampling requires an explicit "
                "beginning-of-sequence history contract"
            )
        if not source_weights or not math.isclose(
            sum(source_weights),
            1.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("source_weights must sum to 1")
        if torch.any(source_codes < 0) or torch.any(
            source_codes >= len(source_weights)
        ):
            raise ValueError("source code is outside source_weights")
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
        self.record_eligible = (
            torch.ones(records, dtype=torch.bool)
            if record_eligible is None
            else record_eligible.to("cpu", dtype=torch.bool)
        )
        self.source_weights = tuple(float(value) for value in source_weights)
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.wander_delays = tuple(int(value) for value in wander_delays)
        self.pitch_transition_fraction = pitch_transition_fraction
        self.allowed_split_code = allowed_split_code
        self.segment_sampling_enabled = segment_sampling_enabled
        self.segment_divisions = segment_divisions
        self.segment_include_first = segment_include_first
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.transition_credit = 0.0
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
        source_vocab = tuple(str(value) for value in index["source_vocab"])
        configured_sources = tuple(
            source.name for source in config.data.sources
        )
        if source_vocab != configured_sources:
            raise ValueError(
                "pack source vocabulary does not match configured sources"
            )
        category_vocab = tuple(
            str(value) for value in index["category_vocab"]
        )
        category_codes_by_name = {
            name: code for code, name in enumerate(category_vocab)
        }
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
        pitch_pairs = torch.from_numpy(
            np.load(pitch_path, allow_pickle=False)
        )
        sequence_path = root / "sequences.jsonl"
        if sequence_path.exists() and not bool(torch.any(pitch_pairs >= 0)):
            rows = [
                json.loads(line)
                for line in sequence_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            if len(rows) != records:
                raise ValueError(
                    "pack sequence metadata length does not match shards"
                )
            pitch_pairs = build_different_note_pair_graph(rows)
        combined_arrays = {
            name: np.concatenate(parts)
            for name, parts in combined.items()
        }
        record_eligible = np.ones(records, dtype=np.bool_)
        for source_code, source in enumerate(config.data.sources):
            if not source.allowed_categories:
                continue
            unknown = set(source.allowed_categories) - set(category_vocab)
            if unknown:
                raise ValueError(
                    "configured categories are absent from pack: "
                    + ", ".join(sorted(unknown))
                )
            allowed_codes = np.asarray(
                [
                    category_codes_by_name[name]
                    for name in source.allowed_categories
                ],
                dtype=combined_arrays["category_codes"].dtype,
            )
            source_rows = combined_arrays["source_codes"] == source_code
            record_eligible[source_rows] = np.isin(
                combined_arrays["category_codes"][source_rows],
                allowed_codes,
            )
        return cls(
            latents=torch.from_numpy(latents),
            **{
                name: torch.from_numpy(values)
                for name, values in combined_arrays.items()
            },
            pitch_pairs=pitch_pairs,
            record_eligible=torch.from_numpy(record_eligible),
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
            segment_sampling_enabled=config.segment_sampling.enabled,
            segment_divisions=config.segment_sampling.divisions,
            segment_include_first=config.segment_sampling.include_first,
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "generator_state": self.generator.get_state().clone(),
            "transition_credit": torch.tensor(
                self.transition_credit,
                dtype=torch.float64,
            ),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if set(state) not in (
            {"generator_state"},
            {"generator_state", "transition_credit"},
        ):
            raise ValueError("invalid sampler state")
        self.generator.set_state(state["generator_state"].cpu())
        credit = state.get("transition_credit")
        self.transition_credit = (
            0.0 if credit is None else float(credit.item())
        )
        if not 0.0 <= self.transition_credit < 1.0:
            raise ValueError("invalid sampler transition credit")

    def _eligible(
        self,
        source: int,
        require_full_future: bool,
    ) -> Tensor:
        mask = self.record_eligible & (self.source_codes == source)
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

    def _segment_tasks(self) -> tuple[tuple[int, int], ...]:
        first = 0 if self.segment_include_first else 1
        return tuple(
            (division, index)
            for division in self.segment_divisions
            for index in range(first, division)
        )

    def _segment_pool(
        self,
        source: int,
        division: int,
        segment_index: int,
        *,
        maximum_valid_future: int,
        require_full_future: bool,
        transition: bool = False,
    ) -> Tensor:
        mask = self.record_eligible & (self.source_codes == source)
        if self.allowed_split_code is not None:
            mask &= self.split_codes == self.allowed_split_code
        pool = torch.nonzero(mask, as_tuple=False).flatten()
        if pool.numel() == 0:
            return pool
        active = self.active_frames[pool]
        start = torch.div(
            active * segment_index,
            division,
            rounding_mode="floor",
        )
        end = torch.div(
            active * (segment_index + 1),
            division,
            rounding_mode="floor",
        )
        valid = end - start
        eligible = (start > 0) & (valid > 0)
        eligible &= valid <= self.future_frames
        eligible &= valid <= self.maximum_future_frames[pool]
        if require_full_future:
            eligible &= valid == self.future_frames
        else:
            eligible &= valid <= maximum_valid_future
        pool = pool[eligible]
        if not transition or pool.numel() == 0:
            return pool
        pool = pool[valid[eligible] > 1]
        if pool.numel() == 0:
            return pool
        pool = pool[self.pitch_pairs[pool] >= 0]
        if pool.numel() == 0:
            return pool
        ends = torch.div(
            self.active_frames[pool] * (segment_index + 1),
            division,
            rounding_mode="floor",
        )
        partners = self.pitch_pairs[pool]
        return pool[self.active_frames[partners] >= ends]

    def _segment_eligible(
        self,
        source: int,
        *,
        maximum_valid_future: int,
        require_full_future: bool,
        transition: bool = False,
    ) -> Tensor:
        pools = [
            self._segment_pool(
                source,
                division,
                segment_index,
                maximum_valid_future=maximum_valid_future,
                require_full_future=require_full_future,
                transition=transition,
            )
            for division, segment_index in self._segment_tasks()
        ]
        nonempty = [pool for pool in pools if pool.numel()]
        if not nonempty:
            return torch.empty(0, dtype=torch.long)
        return torch.unique(torch.cat(nonempty), sorted=True)

    def _draw_balanced_pool(
        self,
        source: int,
        pool: Tensor,
        count: int,
    ) -> Tensor:
        if count == 0 or source != 0:
            return self._draw(pool, count)
        categories = sorted(
            {int(value) for value in self.category_codes[pool].tolist()}
        )
        counts = _largest_remainder(
            count,
            tuple(1.0 for _ in categories),
        )
        return torch.cat(
            [
                self._draw(
                    pool[self.category_codes[pool] == category],
                    category_count,
                )
                for category, category_count in zip(
                    categories,
                    counts,
                    strict=True,
                )
                if category_count
            ]
        )

    def _draw_segment_group(
        self,
        source: int,
        count: int,
        *,
        maximum_valid_future: int,
        require_full_future: bool,
        transition: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if count == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone()
        task_pools = [
            (
                division,
                segment_index,
                self._segment_pool(
                    source,
                    division,
                    segment_index,
                    maximum_valid_future=maximum_valid_future,
                    require_full_future=require_full_future,
                    transition=transition,
                ),
            )
            for division, segment_index in self._segment_tasks()
        ]
        task_pools = [
            item for item in task_pools if item[2].numel()
        ]
        if not task_pools:
            kind = "transition " if transition else ""
            raise ValueError(
                f"requested {kind}segment bucket has no eligible rows"
            )
        task_counts = _largest_remainder(
            count,
            tuple(1.0 for _ in task_pools),
        )
        counted_tasks = list(zip(task_pools, task_counts, strict=True))
        selected: list[Tensor] = []
        divisions: list[Tensor] = []
        indices: list[Tensor] = []
        for (division, segment_index, pool), task_count in counted_tasks:
            if not task_count:
                continue
            selected.append(
                self._draw_balanced_pool(source, pool, task_count)
            )
            divisions.append(
                torch.full((task_count,), division, dtype=torch.long)
            )
            indices.append(
                torch.full(
                    (task_count,),
                    segment_index,
                    dtype=torch.long,
                )
            )
        return (
            torch.cat(selected),
            torch.cat(divisions),
            torch.cat(indices),
        )

    def _draw_segment_source(
        self,
        source: int,
        count: int,
        *,
        maximum_valid_future: int,
        require_full_future: bool,
        transition_count: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rows, divisions, indices = self._draw_segment_group(
            source,
            count,
            maximum_valid_future=maximum_valid_future,
            require_full_future=require_full_future,
            transition=False,
        )
        transition_mask = torch.zeros(count, dtype=torch.bool)
        if transition_count:
            eligible_slots = torch.tensor(
                [
                    slot
                    for slot, (division, index) in enumerate(
                        zip(divisions, indices, strict=True)
                    )
                    if self._segment_pool(
                        source,
                        int(division),
                        int(index),
                        maximum_valid_future=maximum_valid_future,
                        require_full_future=require_full_future,
                        transition=True,
                    ).numel()
                ],
                dtype=torch.long,
            )
            if eligible_slots.numel() < transition_count:
                raise ValueError(
                    "balanced segment tasks lack pitch-transition slots"
                )
            chosen = eligible_slots[
                torch.randperm(
                    eligible_slots.numel(),
                    generator=self.generator,
                )[:transition_count]
            ]
            transition_mask[chosen] = True
        pools = [
            self._segment_pool(
                source,
                int(division),
                int(index),
                maximum_valid_future=maximum_valid_future,
                require_full_future=require_full_future,
                transition=bool(is_transition),
            )
            for division, index, is_transition in zip(
                divisions,
                indices,
                transition_mask,
                strict=True,
            )
        ]
        if source == 0 and count:
            category_assignment = self._balanced_categories_for_pools(pools)
            rows = torch.stack(
                [
                    self._draw(
                        pool[
                            self.category_codes[pool] == category
                        ],
                        1,
                    )[0]
                    for pool, category in zip(
                        pools,
                        category_assignment,
                        strict=True,
                    )
                ]
            )
        elif count:
            rows = torch.stack([self._draw(pool, 1)[0] for pool in pools])
        return (
            rows,
            transition_mask,
            divisions,
            indices,
        )

    def _balanced_categories_for_pools(
        self,
        pools: list[Tensor],
    ) -> Tensor:
        """Jointly satisfy whole-batch category quotas and task eligibility."""

        if not pools:
            return torch.empty(0, dtype=torch.long)
        categories = sorted(
            {
                int(category)
                for pool in pools
                for category in self.category_codes[pool].tolist()
            }
        )
        if not categories:
            raise ValueError("segment task pools have no categories")
        available = [
            {int(value) for value in self.category_codes[pool].tolist()}
            for pool in pools
        ]
        base, remainder = divmod(len(pools), len(categories))
        extra_choices = list(
            itertools.combinations(range(len(categories)), remainder)
        )
        choice_order = torch.randperm(
            len(extra_choices),
            generator=self.generator,
        ).tolist()
        for choice_index in choice_order:
            extras = set(extra_choices[choice_index])
            tokens = [
                category
                for category_index, category in enumerate(categories)
                for _ in range(base + int(category_index in extras))
            ]
            token_order = torch.randperm(
                len(tokens),
                generator=self.generator,
            ).tolist()
            tokens = [tokens[index] for index in token_order]
            token_to_slot = [-1] * len(tokens)
            slot_to_token = [-1] * len(pools)

            def assign(
                slot: int,
                seen: set[int],
                *,
                resolved_tokens: list[int] = tokens,
                resolved_token_to_slot: list[int] = token_to_slot,
                resolved_slot_to_token: list[int] = slot_to_token,
            ) -> bool:
                for token, category in enumerate(resolved_tokens):
                    if token in seen or category not in available[slot]:
                        continue
                    seen.add(token)
                    previous = resolved_token_to_slot[token]
                    if previous == -1 or assign(previous, seen):
                        resolved_token_to_slot[token] = slot
                        resolved_slot_to_token[slot] = token
                        return True
                return False

            slot_order = torch.randperm(
                len(pools),
                generator=self.generator,
            ).tolist()
            if all(assign(slot, set()) for slot in slot_order):
                return torch.tensor(
                    [tokens[token] for token in slot_to_token],
                    dtype=torch.long,
                )
        raise ValueError("cannot jointly balance segment tasks and categories")

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
        if self.segment_sampling_enabled:
            eligible_by_source = [
                self._segment_eligible(
                    source,
                    maximum_valid_future=maximum_valid_future,
                    require_full_future=require_full_future,
                )
                for source in range(len(self.source_weights))
            ]
        else:
            eligible_by_source = [
                self._eligible(source, require_full_future)
                for source in range(len(self.source_weights))
            ]
        weights = [
            weight if eligible.numel() else 0.0
            for weight, eligible in zip(
                self.source_weights,
                eligible_by_source,
                strict=True,
            )
        ]
        if not any(weights):
            mode = "segment " if self.segment_sampling_enabled else ""
            raise ValueError(f"{mode}sampler has no eligible rows")
        source_counts = _largest_remainder(batch_size, tuple(weights))
        if require_full_future:
            transition_total = 0
        else:
            expected_transitions = (
                self.transition_credit
                + batch_size * self.pitch_transition_fraction
            )
            transition_total = math.floor(expected_transitions + 1.0e-12)
            self.transition_credit = expected_transitions - transition_total
            if self.transition_credit >= 1.0 - 1.0e-12:
                transition_total += 1
                self.transition_credit = 0.0
        if self.segment_sampling_enabled:
            transition_sources = [
                source
                for source, count in enumerate(source_counts)
                if count
                and self._segment_eligible(
                    source,
                    maximum_valid_future=maximum_valid_future,
                    require_full_future=require_full_future,
                    transition=True,
                ).numel()
            ]
        else:
            transition_sources = [
                source
                for source, (count, eligible) in enumerate(
                    zip(source_counts, eligible_by_source, strict=True)
                )
                if count
                and torch.any(self.pitch_pairs[eligible] >= 0)
            ]
        transition_capacity = sum(
            source_counts[source] for source in transition_sources
        )
        if transition_total > transition_capacity:
            raise ValueError(
                "not enough pitch-paired source slots for transitions"
            )
        transition_by_source: dict[int, int] = {}
        if transition_total:
            allocated = _largest_remainder(
                transition_total,
                tuple(
                    float(source_counts[source])
                    for source in transition_sources
                ),
            )
            transition_by_source = dict(
                zip(transition_sources, allocated, strict=True)
            )

        selected_parts: list[Tensor] = []
        transition_parts: list[Tensor] = []
        division_parts: list[Tensor] = []
        segment_index_parts: list[Tensor] = []
        for source, count in enumerate(source_counts):
            if self.segment_sampling_enabled:
                rows, transition, divisions, segment_indices = (
                    self._draw_segment_source(
                        source,
                        count,
                        maximum_valid_future=maximum_valid_future,
                        require_full_future=require_full_future,
                        transition_count=transition_by_source.get(source, 0),
                    )
                )
                division_parts.append(divisions)
                segment_index_parts.append(segment_indices)
            else:
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
        segment_division = (
            torch.cat(division_parts)
            if self.segment_sampling_enabled
            else None
        )
        segment_index = (
            torch.cat(segment_index_parts)
            if self.segment_sampling_enabled
            else None
        )
        permutation = torch.randperm(
            batch_size,
            generator=self.generator,
        )
        selected = selected[permutation]
        transitions = transitions[permutation]
        if segment_division is not None and segment_index is not None:
            segment_division = segment_division[permutation]
            segment_index = segment_index[permutation]

        history = torch.zeros(
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
        history_mask = (
            torch.zeros(
                batch_size,
                self.context_frames,
                dtype=torch.bool,
                device=self.device,
            )
            if self.segment_sampling_enabled
            else None
        )
        absolute_start = (
            torch.empty(batch_size, dtype=torch.long, device=self.device)
            if self.segment_sampling_enabled
            else None
        )
        history_midi_sequence = torch.full(
            (batch_size, self.context_frames),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        future_midi_sequence = torch.full(
            (batch_size, self.future_frames),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        history_velocity_sequence = torch.zeros(
            batch_size,
            self.context_frames,
            dtype=torch.long,
            device=self.device,
        )
        future_velocity_sequence = torch.zeros(
            batch_size,
            self.future_frames,
            dtype=torch.long,
            device=self.device,
        )
        midi_event_frame = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        history_indices = selected.clone()
        history_indices[transitions] = self.pitch_pairs[
            selected[transitions]
        ]
        for batch_index in range(batch_size):
            target = int(selected[batch_index])
            history_index = int(history_indices[batch_index])
            if self.segment_sampling_enabled:
                assert segment_division is not None
                assert segment_index is not None
                assert history_mask is not None
                assert absolute_start is not None
                division = int(segment_division[batch_index])
                index = int(segment_index[batch_index])
                active = int(self.active_frames[target])
                target_start = active * index // division
                target_end = active * (index + 1) // division
                valid_future = target_end - target_start
                if not 1 <= valid_future <= self.future_frames:
                    raise ValueError("sampled segment does not fit future")
                if valid_future > maximum_valid_future:
                    raise ValueError(
                        "sampled segment exceeds maximum_valid_future"
                    )
                if require_full_future and valid_future != self.future_frames:
                    raise ValueError(
                        "full-future sampler selected a short segment"
                    )
                history_frames = min(self.context_frames, target_start)
                if history_frames <= 0:
                    raise ValueError(
                        "causal segment has no preceding history"
                    )
                history_start = target_start - history_frames
                history[batch_index, -history_frames:] = self.latents[
                    history_index,
                    history_start:target_start,
                ].float()
                history_mask[batch_index, -history_frames:] = True
                history_midi_sequence[batch_index, -history_frames:] = int(
                    self.notes[history_index]
                )
                history_velocity_sequence[
                    batch_index, -history_frames:
                ] = int(self.velocities[history_index])
                if bool(transitions[batch_index]):
                    event = int(
                        torch.randint(
                            1,
                            valid_future,
                            (1,),
                            generator=self.generator,
                        )
                    )
                    future[batch_index, :event] = self.latents[
                        history_index,
                        target_start : target_start + event,
                    ].float()
                    future[batch_index, event:valid_future] = self.latents[
                        target,
                        target_start + event : target_end,
                    ].float()
                    future_midi_sequence[batch_index, :event] = int(
                        self.notes[history_index]
                    )
                    future_midi_sequence[
                        batch_index, event:valid_future
                    ] = int(self.notes[target])
                    future_velocity_sequence[batch_index, :event] = int(
                        self.velocities[history_index]
                    )
                    future_velocity_sequence[
                        batch_index, event:valid_future
                    ] = int(self.velocities[target])
                    midi_event_frame[batch_index] = event
                else:
                    future[batch_index, :valid_future] = self.latents[
                        target,
                        target_start:target_end,
                    ].float()
                    future_midi_sequence[
                        batch_index, :valid_future
                    ] = int(self.notes[target])
                    future_velocity_sequence[
                        batch_index, :valid_future
                    ] = int(self.velocities[target])
                future_mask[batch_index, :valid_future] = True
                absolute_start[batch_index] = target_start
                continue
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
            history_midi_sequence[batch_index] = int(
                self.notes[history_index]
            )
            history_velocity_sequence[batch_index] = int(
                self.velocities[history_index]
            )
            future[batch_index, :valid_future] = self.latents[
                target,
                start
                + self.context_frames : start
                + self.context_frames
                + valid_future,
            ].float()
            future_midi_sequence[batch_index, :valid_future] = int(
                self.notes[target]
            )
            future_velocity_sequence[batch_index, :valid_future] = int(
                self.velocities[target]
            )
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
            velocity=self.velocities[selected].to(self.device),
            history_velocity=self.velocities[history_indices].to(
                self.device
            ),
            history_mask=history_mask,
            segment_division=(
                segment_division.to(self.device)
                if segment_division is not None
                else None
            ),
            segment_index=(
                segment_index.to(self.device)
                if segment_index is not None
                else None
            ),
            absolute_start=absolute_start,
            history_midi_sequence=history_midi_sequence,
            future_midi_sequence=future_midi_sequence,
            history_velocity_sequence=history_velocity_sequence,
            future_velocity_sequence=future_velocity_sequence,
            midi_event_frame=midi_event_frame,
        )
