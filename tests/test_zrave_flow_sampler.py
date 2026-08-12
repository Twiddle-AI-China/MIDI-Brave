from __future__ import annotations

import pytest
import torch

from midibrave.zrave_flow_sampler import (
    GpuFlowSampler,
    build_different_note_pair_graph,
)


def _sampler(seed: int) -> GpuFlowSampler:
    sources: list[int] = []
    categories: list[int] = []
    notes: list[int] = []
    maximum_futures: list[int] = []
    pitch_pairs: list[int] = []

    for category in range(8):
        sources.append(0)
        categories.append(category)
        notes.append(60)
        maximum_futures.append(64)
        pitch_pairs.append(-1)

    for source in (1, 2):
        first = len(sources)
        for note in (48, 60, 72, 84):
            sources.append(source)
            categories.append(8 + source)
            notes.append(note)
            maximum_futures.append(20 if source == 2 else 64)
            pitch_pairs.append(-1)
        for offset in range(4):
            partner = offset + 1 if offset % 2 == 0 else offset - 1
            pitch_pairs[first + offset] = first + partner

    for _ in range(2):
        sources.append(3)
        categories.append(11)
        notes.append(60)
        maximum_futures.append(64)
        pitch_pairs.append(-1)

    records = len(sources)
    latents = torch.arange(
        records * 100 * 16,
        dtype=torch.float32,
    ).reshape(records, 100, 16)
    return GpuFlowSampler(
        latents=latents,
        lengths=torch.full((records,), 100),
        active_frames=torch.full((records,), 100),
        notes=torch.tensor(notes),
        velocities=torch.full((records,), 100),
        split_codes=torch.zeros(records, dtype=torch.uint8),
        source_codes=torch.tensor(sources),
        category_codes=torch.tensor(categories),
        maximum_future_frames=torch.tensor(maximum_futures),
        pitch_pairs=torch.tensor(pitch_pairs),
        source_weights=(0.55, 0.20, 0.15, 0.10),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.20,
        seed=seed,
        device="cpu",
    )


def test_sampler_honors_source_weights_and_category_balance() -> None:
    sampler = _sampler(seed=7)

    batch = sampler.sample(batch_size=100, maximum_valid_future=64)

    counts = torch.bincount(batch.source_code, minlength=4)
    assert counts.tolist() == [55, 20, 15, 10]
    serum_categories = batch.category_code[batch.source_code == 0]
    category_counts = torch.bincount(serum_categories, minlength=8)[:8]
    assert int(category_counts.max() - category_counts.min()) <= 1


def test_sampler_filters_prepacked_records_by_configured_category() -> None:
    sampler = _sampler(seed=11)
    sampler.record_eligible = torch.isin(
        sampler.category_codes,
        torch.tensor([1, 3, 5, 7, 9, 10, 11])
    )

    batch = sampler.sample(batch_size=100, maximum_valid_future=64)

    serum_categories = batch.category_code[batch.source_code == 0]
    assert set(serum_categories.tolist()) == {1, 3, 5, 7}


def test_dense_serum_masks_release_tail_and_sampling_repeats() -> None:
    first = _sampler(seed=19).sample(20, maximum_valid_future=64)
    second = _sampler(seed=19).sample(20, maximum_valid_future=64)

    assert torch.equal(first.history, second.history)
    assert torch.equal(first.future, second.future)
    assert torch.equal(first.future_mask, second.future_mask)
    dense = first.source_code == 2
    assert torch.all(first.future_mask[dense, :20])
    assert not torch.any(first.future_mask[dense, 20:])
    assert torch.equal(first.velocity, torch.full((20,), 100))
    assert torch.equal(first.history_velocity, torch.full((20,), 100))
    assert first.history_mask is None
    assert first.segment_division is None
    assert first.segment_index is None
    assert first.absolute_start is None


def test_exposure_sampling_returns_only_complete_64_frame_futures() -> None:
    batch = _sampler(seed=23).sample(
        32,
        maximum_valid_future=64,
        require_full_future=True,
    )

    assert torch.all(batch.future_mask)
    assert not torch.any(batch.source_code == 2)
    assert not torch.any(batch.pitch_transition_mask)


def test_sampler_builds_real_same_preset_pitch_transitions() -> None:
    batch = _sampler(seed=31).sample(100, maximum_valid_future=64)
    transition = batch.pitch_transition_mask

    assert int(transition.sum()) == 20
    assert torch.all(
        torch.abs(
            batch.history_midi_note[transition] - batch.midi_note[transition]
        )
        == 12
    )
    assert torch.all(
        (batch.source_code[transition] == 1)
        | (batch.source_code[transition] == 2)
    )


def test_sampler_state_restores_the_next_batch() -> None:
    first = _sampler(seed=41)
    first.sample(20, maximum_valid_future=32)
    state = first.state_dict()
    expected = first.sample(20, maximum_valid_future=32)
    resumed = _sampler(seed=999)
    resumed.load_state_dict(state)

    actual = resumed.sample(20, maximum_valid_future=32)

    assert torch.equal(expected.history, actual.history)
    assert torch.equal(expected.future, actual.future)


def test_sampler_accepts_three_real_octopus_sources() -> None:
    records = 9
    sampler = GpuFlowSampler(
        latents=torch.randn(records, 100, 16),
        lengths=torch.full((records,), 100),
        active_frames=torch.full((records,), 100),
        notes=torch.tensor([48, 60, 72] * 3),
        velocities=torch.full((records,), 100),
        split_codes=torch.zeros(records, dtype=torch.uint8),
        source_codes=torch.tensor([0] * 3 + [1] * 3 + [2] * 3),
        category_codes=torch.tensor([0] * 3 + [1] * 3 + [2] * 3),
        maximum_future_frames=torch.full((records,), 64),
        pitch_pairs=torch.full((records,), -1),
        source_weights=(0.65, 0.10, 0.25),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.0,
        seed=53,
        device="cpu",
    )

    batch = sampler.sample(20, maximum_valid_future=64)

    assert torch.bincount(batch.source_code, minlength=3).tolist() == [
        13,
        2,
        5,
    ]


def _segment_sampler(seed: int = 61) -> GpuFlowSampler:
    frame_values = torch.arange(96, dtype=torch.float32)
    latents = frame_values[:, None].expand(-1, 2).unsqueeze(0)
    return GpuFlowSampler(
        latents=latents,
        lengths=torch.tensor([96]),
        active_frames=torch.tensor([96]),
        notes=torch.tensor([67]),
        velocities=torch.tensor([83]),
        split_codes=torch.zeros(1, dtype=torch.uint8),
        source_codes=torch.zeros(1, dtype=torch.uint8),
        category_codes=torch.zeros(1, dtype=torch.uint16),
        maximum_future_frames=torch.tensor([64]),
        pitch_pairs=torch.tensor([-1]),
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.0,
        seed=seed,
        device="cpu",
        segment_sampling_enabled=True,
        segment_divisions=(2, 4, 8),
    )


def test_segment_sampler_balances_causal_division_position_tasks() -> None:
    batch = _segment_sampler().sample(33, maximum_valid_future=64)

    assert batch.segment_division is not None
    assert batch.segment_index is not None
    assert batch.absolute_start is not None
    assert batch.history_mask is not None
    assert torch.all(batch.segment_index > 0)
    assert torch.all(batch.segment_index < batch.segment_division)
    tasks = torch.stack(
        (batch.segment_division, batch.segment_index),
        dim=1,
    )
    _unique, counts = torch.unique(tasks, dim=0, return_counts=True)
    assert len(_unique) == 11
    assert int(counts.max() - counts.min()) <= 1
    tail = batch.segment_index == batch.segment_division - 1
    assert int(tail.sum()) == 9


def test_segment_sampler_short_curriculum_uses_four_and_eight_parts() -> None:
    batch = _segment_sampler().sample(30, maximum_valid_future=32)

    tasks = torch.stack(
        (batch.segment_division, batch.segment_index),
        dim=1,
    )
    unique, counts = torch.unique(tasks, dim=0, return_counts=True)
    assert set(unique[:, 0].tolist()) == {4, 8}
    assert len(unique) == 10
    assert int(counts.max() - counts.min()) <= 1


def test_segment_sampler_pads_targets_and_right_aligns_causal_history() -> None:
    batch = _segment_sampler().sample(33, maximum_valid_future=64)

    assert batch.segment_division is not None
    assert batch.segment_index is not None
    assert batch.absolute_start is not None
    assert batch.history_mask is not None
    for row in range(33):
        division = int(batch.segment_division[row])
        index = int(batch.segment_index[row])
        start = 96 * index // division
        end = 96 * (index + 1) // division
        valid = end - start
        history_frames = min(32, start)
        assert int(batch.absolute_start[row]) == start
        assert int(batch.future_mask[row].sum()) == valid
        assert torch.all(batch.future[row, :valid, 0] == torch.arange(start, end))
        assert not torch.any(batch.future[row, valid:])
        assert int(batch.history_mask[row].sum()) == history_frames
        assert torch.all(batch.history_mask[row, -history_frames:])
        assert not torch.any(batch.history_mask[row, :-history_frames])
        assert torch.all(
            batch.history[row, -history_frames:, 0]
            == torch.arange(start - history_frames, start)
        )
        assert not torch.any(batch.history[row, :-history_frames])


def test_segment_sampler_is_state_resumable_with_metadata() -> None:
    sampler = _segment_sampler(seed=71)
    sampler.sample(9, maximum_valid_future=64)
    state = sampler.state_dict()
    expected = sampler.sample(33, maximum_valid_future=64)
    resumed = _segment_sampler(seed=999)
    resumed.load_state_dict(state)

    actual = resumed.sample(33, maximum_valid_future=64)

    assert torch.equal(expected.history, actual.history)
    assert torch.equal(expected.future, actual.future)
    assert torch.equal(expected.segment_division, actual.segment_division)
    assert torch.equal(expected.segment_index, actual.segment_index)
    assert torch.equal(expected.absolute_start, actual.absolute_start)


def test_segment_sampler_jointly_balances_small_batch_categories() -> None:
    records = 12
    latents = torch.randn(records, 96, 2)
    sampler = GpuFlowSampler(
        latents=latents,
        lengths=torch.full((records,), 96),
        active_frames=torch.full((records,), 96),
        notes=torch.full((records,), 62),
        velocities=torch.full((records,), 108),
        split_codes=torch.zeros(records, dtype=torch.uint8),
        source_codes=torch.zeros(records, dtype=torch.uint8),
        category_codes=torch.arange(records),
        maximum_future_frames=torch.full((records,), 64),
        pitch_pairs=torch.full((records,), -1),
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.0,
        seed=72,
        device="cpu",
        segment_sampling_enabled=True,
        segment_divisions=(2, 4, 8),
    )

    batch = sampler.sample(16, maximum_valid_future=64)

    counts = torch.bincount(batch.category_code, minlength=records)
    assert set(counts.tolist()) == {1, 2}
    assert int((counts > 0).sum()) == records


def test_segment_midi_small_batch_balances_tasks_categories_and_pairs() -> None:
    categories = torch.arange(8).repeat_interleave(2)
    records = len(categories)
    pitch_pairs = torch.arange(records).reshape(-1, 2).flip(1).reshape(-1)
    sampler = GpuFlowSampler(
        latents=torch.randn(records, 96, 2),
        lengths=torch.full((records,), 96),
        active_frames=torch.full((records,), 96),
        notes=torch.tensor([62, 82] * 8),
        velocities=torch.full((records,), 108),
        split_codes=torch.zeros(records, dtype=torch.uint8),
        source_codes=torch.zeros(records, dtype=torch.uint8),
        category_codes=categories,
        maximum_future_frames=torch.full((records,), 64),
        pitch_pairs=pitch_pairs,
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.20,
        seed=73,
        device="cpu",
        segment_sampling_enabled=True,
        segment_divisions=(2, 4, 8),
    )

    batch = sampler.sample(16, maximum_valid_future=64)

    tasks = torch.stack((batch.segment_division, batch.segment_index), dim=1)
    _unique, task_counts = torch.unique(tasks, dim=0, return_counts=True)
    category_counts = torch.bincount(batch.category_code, minlength=8)
    assert len(_unique) == 11
    assert int(task_counts.max() - task_counts.min()) <= 1
    assert category_counts.tolist() == [2] * 8
    assert int(batch.pitch_transition_mask.sum()) == 3
    assert torch.all(
        batch.history_midi_note[batch.pitch_transition_mask]
        != batch.midi_note[batch.pitch_transition_mask]
    )


def test_segment_fractional_transition_credit_covers_batch_one() -> None:
    categories = torch.arange(8).repeat_interleave(2)
    records = len(categories)
    pitch_pairs = torch.arange(records).reshape(-1, 2).flip(1).reshape(-1)
    sampler = GpuFlowSampler(
        latents=torch.randn(records, 96, 2),
        lengths=torch.full((records,), 96),
        active_frames=torch.full((records,), 96),
        notes=torch.tensor([62, 82] * 8),
        velocities=torch.full((records,), 108),
        split_codes=torch.zeros(records, dtype=torch.uint8),
        source_codes=torch.zeros(records, dtype=torch.uint8),
        category_codes=categories,
        maximum_future_frames=torch.full((records,), 64),
        pitch_pairs=pitch_pairs,
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.20,
        seed=74,
        device="cpu",
        segment_sampling_enabled=True,
        segment_divisions=(2, 4, 8),
    )

    transitions = sum(
        int(sampler.sample(1, 64).pitch_transition_mask.sum())
        for _ in range(5)
    )

    assert transitions == 1
    assert sampler.transition_credit == pytest.approx(0.0)


def test_segment_transition_splices_latent_and_framewise_midi_event() -> None:
    frames = torch.arange(96, dtype=torch.float32)
    latents = torch.stack(
        (
            frames[:, None].expand(-1, 2),
            (1000.0 + frames)[:, None].expand(-1, 2),
        )
    )
    sampler = GpuFlowSampler(
        latents=latents,
        lengths=torch.full((2,), 96),
        active_frames=torch.full((2,), 96),
        notes=torch.tensor([62, 82]),
        velocities=torch.full((2,), 108),
        split_codes=torch.zeros(2, dtype=torch.uint8),
        source_codes=torch.zeros(2, dtype=torch.uint8),
        category_codes=torch.zeros(2, dtype=torch.long),
        maximum_future_frames=torch.full((2,), 64),
        pitch_pairs=torch.tensor([1, 0]),
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.20,
        seed=76,
        device="cpu",
        segment_sampling_enabled=True,
        segment_divisions=(2, 4, 8),
    )

    batches = [sampler.sample(1, 64) for _ in range(5)]
    batch = next(value for value in batches if value.pitch_transition_mask.item())
    assert batch.future_midi_sequence is not None
    assert batch.history_midi_sequence is not None
    assert batch.midi_event_frame is not None
    event = int(batch.midi_event_frame.item())
    valid = int(batch.future_mask.sum())
    history_note = int(batch.history_midi_note.item())
    target_note = int(batch.midi_note.item())
    history_offset = 0.0 if history_note == 62 else 1000.0
    target_offset = 0.0 if target_note == 62 else 1000.0
    start = int(batch.absolute_start.item())

    assert 0 < event < valid
    assert torch.all(batch.future_midi_sequence[0, :event] == history_note)
    assert torch.all(
        batch.future_midi_sequence[0, event:valid] == target_note
    )
    torch.testing.assert_close(
        batch.future[0, :event, 0],
        history_offset + torch.arange(start, start + event),
    )
    torch.testing.assert_close(
        batch.future[0, event:valid, 0],
        target_offset + torch.arange(start + event, start + valid),
    )


@pytest.mark.parametrize("seed", [1, 5, 7, 8])
def test_joint_category_matcher_searches_feasible_remainder_quotas(
    seed: int,
) -> None:
    sampler = _segment_sampler(seed=seed)
    sampler.category_codes = torch.tensor([0, 0, 1])
    pools = [
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1, 2]),
    ]

    assignment = sampler._balanced_categories_for_pools(pools)

    assert torch.bincount(assignment, minlength=2).tolist() == [2, 1]


def test_sampler_loads_legacy_state_without_transition_credit() -> None:
    sampler = _segment_sampler(seed=75)
    legacy = {"generator_state": sampler.generator.get_state().clone()}

    sampler.load_state_dict(legacy)

    assert sampler.transition_credit == 0.0


def test_different_note_pair_graph_is_not_limited_to_octaves() -> None:
    common = {
        "source_name": "serum",
        "split": "train",
        "canonical_preset_id": "serum:1",
        "velocity": 100,
        "articulation_id": "steady",
    }
    rows = [
        common | {"midi_note": 48, "sample_id": "a"},
        common | {"midi_note": 55, "sample_id": "b"},
        common | {"midi_note": 60, "sample_id": "c"},
        common | {"midi_note": 55, "sample_id": "d"},
        common
        | {
            "canonical_preset_id": "serum:2",
            "midi_note": 48,
            "sample_id": "e",
        },
    ]

    pairs = build_different_note_pair_graph(rows)

    assert pairs.tolist() == [1, 2, 1, 2, -1]
    for index, partner in enumerate(pairs.tolist()):
        if partner >= 0:
            assert rows[index]["canonical_preset_id"] == rows[partner][
                "canonical_preset_id"
            ]
            assert rows[index]["midi_note"] != rows[partner]["midi_note"]
