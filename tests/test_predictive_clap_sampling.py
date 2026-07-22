from __future__ import annotations

from types import SimpleNamespace

import torch

from midibrave.trainer import (
    _predictive_clap_due,
    _predictive_clap_indices,
    _predictive_control_clap_due,
    _sampled_predictive_clap_auxiliary,
)


def _config(*, interval: int = 3, batch_size: int = 2):
    return SimpleNamespace(
        loss=SimpleNamespace(
            clap_every_updates=interval,
            clap_batch_size=batch_size,
        ),
        predictive=SimpleNamespace(
            predictor_control_start_updates=1,
            predictor_control_every_updates=2,
        ),
    )


def test_predictive_clap_cadence_counts_stage_and_control_events():
    config = _config(interval=3)

    assert [_predictive_clap_due(config, update) for update in range(6)] == [
        False, False, True, False, False, True,
    ]
    # Control is active at updates 1, 3, 5. CLAP is therefore due on the
    # third control event (update 5), rather than never firing due to parity.
    assert [_predictive_control_clap_due(config, update) for update in range(6)] == [
        False, False, False, False, False, True,
    ]


def test_predictive_clap_indices_limit_rotate_and_accept_empty_masks():
    mask = torch.tensor([True, False, True, True, False])

    first = _predictive_clap_indices(mask, limit=2, event=0, rank=0)
    next_event = _predictive_clap_indices(mask, limit=2, event=1, rank=0)
    next_rank = _predictive_clap_indices(mask, limit=2, event=0, rank=1)

    assert first.tolist() == [0, 2]
    assert next_event.tolist() == [3, 0]
    assert next_rank.tolist() == [3, 0]
    empty = _predictive_clap_indices(
        torch.zeros(5, dtype=torch.bool), limit=2, event=0, rank=0)
    assert empty.dtype == torch.long
    assert empty.numel() == 0


def test_predictive_clap_auxiliary_scatter_uses_sample_mean_and_cadence_scale():
    waveform = torch.zeros(4, 1, 3, requires_grad=True)
    indices = torch.tensor([1, 3])
    selected_gradients = torch.ones(2, 1, 3)

    auxiliary = _sampled_predictive_clap_auxiliary(
        waveform, indices, selected_gradients,
        objective_weight=2.0, expectation_scale=3.0,
    )
    (waveform * auxiliary).sum().backward()

    # weight * cadence / selected_count = 2 * 3 / 2 = 3
    assert torch.count_nonzero(waveform.grad[0]).item() == 0
    assert torch.all(waveform.grad[1] == 3.0)
    assert torch.count_nonzero(waveform.grad[2]).item() == 0
    assert torch.all(waveform.grad[3] == 3.0)
