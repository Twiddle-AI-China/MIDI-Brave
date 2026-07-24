from __future__ import annotations

import torch

from midibrave.zrave_flow_sampler import GpuFlowSampler


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


def test_dense_serum_masks_release_tail_and_sampling_repeats() -> None:
    first = _sampler(seed=19).sample(20, maximum_valid_future=64)
    second = _sampler(seed=19).sample(20, maximum_valid_future=64)

    assert torch.equal(first.history, second.history)
    assert torch.equal(first.future, second.future)
    assert torch.equal(first.future_mask, second.future_mask)
    dense = first.source_code == 2
    assert torch.all(first.future_mask[dense, :20])
    assert not torch.any(first.future_mask[dense, 20:])


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
