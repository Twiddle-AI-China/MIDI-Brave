from __future__ import annotations

import math

import torch
from torch import nn

from midibrave.zrave_flow_loss import (
    PitchWeightController,
    distributed_gradient_l2_norm,
    make_flow_training_pair,
    zrave_flow_loss,
    zrave_pure_flow_loss,
)
from midibrave.zrave_flow_model import FlowStatistics
from midibrave.zrave_pitch_probe import PitchProbeOutput


def _statistics(
    mean: float = 0.0,
    latent_std: float = 1.0,
) -> FlowStatistics:
    return FlowStatistics(
        mean=torch.full((16,), mean),
        latent_std=torch.full((16,), latent_std),
        delta_std=torch.ones(16),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )


class _RecordingPitchProbe(nn.Module):
    note_min = 21
    note_max = 109

    def __init__(self) -> None:
        super().__init__()
        self.seen_windows = torch.empty(0)

    def forward(self, latents: torch.Tensor) -> PitchProbeOutput:
        self.seen_windows = latents
        signal = latents.mean(dim=(1, 2), keepdim=False)
        offsets = torch.linspace(
            -1.0,
            1.0,
            89,
            device=latents.device,
        )
        logits = signal.unsqueeze(1) * offsets.unsqueeze(0)
        probabilities = logits.float().softmax(dim=-1)
        notes = torch.arange(
            21,
            110,
            device=latents.device,
            dtype=torch.float32,
        )
        return PitchProbeOutput(
            logits=logits,
            expected_midi=(probabilities * notes).sum(dim=-1),
        )


def _loss_fixture(
    *,
    clean_future: torch.Tensor | None = None,
    valid_frames: int = 64,
    generator_seed: int = 7,
    statistics: FlowStatistics | None = None,
    pitch_probe: nn.Module | None = None,
    perfect_velocity: bool = False,
) -> dict[str, object]:
    raw_future = torch.randn(2, 64, 16) if clean_future is None else clean_future
    resolved_statistics = statistics or _statistics()
    mask = torch.arange(64).unsqueeze(0).expand(2, -1) < valid_frames
    generator = torch.Generator().manual_seed(generator_seed)
    pair = make_flow_training_pair(
        raw_future,
        mask,
        temperature=torch.ones(2),
        wander_delay_frames=torch.tensor([16, 48]),
        statistics=resolved_statistics,
        generator=generator,
    )
    predicted_velocity = (
        pair.target_velocity.clone()
        if perfect_velocity
        else torch.zeros_like(pair.target_velocity)
    )
    return {
        "predicted_velocity": predicted_velocity,
        "pair": pair,
        "history": torch.randn(2, 32, 16),
        "future_mask": mask,
        "midi_note": torch.tensor([60, 72]),
        "pitch_probe": pitch_probe or _RecordingPitchProbe(),
        "statistics": resolved_statistics,
        "pitch_weight": 0.3,
    }


def test_flow_loss_ignores_masked_release_tail() -> None:
    clean = torch.randn(2, 64, 16)
    changed_clean = clean.clone()
    changed_clean[:, 20:] = 10000.0

    base = zrave_flow_loss(
        **_loss_fixture(
            clean_future=clean,
            valid_frames=20,
            generator_seed=19,
        )
    )
    changed = zrave_flow_loss(
        **_loss_fixture(
            clean_future=changed_clean,
            valid_frames=20,
            generator_seed=19,
        )
    )

    torch.testing.assert_close(base.total, changed.total)
    for name in base.components:
        torch.testing.assert_close(
            base.components[name],
            changed.components[name],
        )


def test_training_pair_absolute_offset_keeps_late_noise_hot() -> None:
    future = torch.zeros(2, 64, 16)
    mask = torch.ones(2, 64, dtype=torch.bool)
    arguments = {
        "future": future,
        "future_mask": mask,
        "temperature": torch.ones(2),
        "wander_delay_frames": torch.full((2,), 32.0),
        "statistics": _statistics(),
    }
    anchored = make_flow_training_pair(
        **arguments,
        schedule_offset_frames=0,
        generator=torch.Generator().manual_seed(23),
    )
    wandering = make_flow_training_pair(
        **arguments,
        schedule_offset_frames=64,
        generator=torch.Generator().manual_seed(23),
    )

    torch.testing.assert_close(
        wandering.target_velocity[:, 0],
        anchored.target_velocity[:, 0] / 0.15,
    )


def test_loss_has_only_approved_components() -> None:
    report = zrave_flow_loss(**_loss_fixture())

    assert set(report.components) == {
        "flow",
        "pitch",
        "boundary",
        "statistics",
    }
    assert "future" not in report.components
    assert "delta" not in report.components
    assert "acceleration" not in report.components
    assert torch.isfinite(report.total)


def test_pure_loss_has_no_pitch_component() -> None:
    arguments = _loss_fixture()
    for name in ("midi_note", "pitch_probe", "pitch_weight"):
        arguments.pop(name)

    report = zrave_pure_flow_loss(**arguments)

    assert set(report.components) == {
        "flow",
        "boundary",
        "statistics",
    }
    torch.testing.assert_close(
        report.total,
        report.components["flow"]
        + 0.10 * report.components["boundary"]
        + 0.02 * report.components["statistics"],
    )


def test_masked_mean_weights_variable_length_segments_per_sample() -> None:
    from midibrave.zrave_flow_loss import _masked_mean

    values = torch.zeros(2, 64, 1)
    values[0, :54] = 1.0
    values[1, :13] = 3.0
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[0, :54] = True
    mask[1, :13] = True

    result = _masked_mean(values, mask)

    torch.testing.assert_close(result, torch.tensor(2.0))


def test_boundary_uses_last_visible_history_frame() -> None:
    from midibrave.zrave_flow_loss import _boundary_loss

    history = torch.zeros(1, 32, 2)
    history[:, 5] = 3.0
    history[:, -1] = 1000.0
    history_mask = torch.zeros(1, 32, dtype=torch.bool)
    history_mask[:, :6] = True
    target = torch.full((1, 64, 2), 4.0)
    mask = torch.ones(1, 64, dtype=torch.bool)

    loss = _boundary_loss(target, target, history, mask, history_mask)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_short_segment_is_padded_for_pitch_supervision() -> None:
    from midibrave.zrave_flow_loss import _pitch_windows

    values = torch.arange(13).view(1, 13, 1).expand(-1, -1, 4).float()
    estimate = torch.zeros(1, 64, 4)
    estimate[:, :13] = values
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :13] = True

    windows, notes = _pitch_windows(estimate, mask, torch.tensor([62]))

    assert windows is not None and notes is not None
    assert windows.shape == (1, 16, 4)
    assert notes.tolist() == [62]
    assert torch.all(windows[:, 13:] == 12)


def test_pitch_windows_follow_framewise_midi_event_labels() -> None:
    from midibrave.zrave_flow_loss import _pitch_windows

    estimate = torch.randn(1, 64, 4)
    mask = torch.ones(1, 64, dtype=torch.bool)
    notes = torch.full((1, 64), 62)
    notes[:, 20:] = 82

    windows, labels = _pitch_windows(estimate, mask, notes)

    assert windows is not None and labels is not None
    assert windows.shape == (3, 16, 4)
    assert labels.tolist() == [62, 82, 82]
    # No pitch-probe window is allowed to cross the note event at frame 20.
    torch.testing.assert_close(windows[0], estimate[0, :16])
    torch.testing.assert_close(windows[1], estimate[0, 20:36])
    torch.testing.assert_close(windows[2], estimate[0, 36:52])


def test_short_transition_runs_are_independently_padded() -> None:
    from midibrave.zrave_flow_loss import _pitch_windows

    estimate = torch.zeros(1, 64, 2)
    estimate[:, :12] = torch.arange(12).view(1, 12, 1).expand(-1, -1, 2).float()
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :12] = True
    notes = torch.full((1, 64), 62)
    notes[:, 5:12] = 82

    windows, labels = _pitch_windows(estimate, mask, notes)

    assert windows is not None and labels is not None
    assert windows.shape == (2, 16, 2)
    assert labels.tolist() == [62, 82]
    assert torch.all(windows[0, 5:] == 4)
    assert torch.all(windows[1, 7:] == 11)


def test_temporal_loss_detects_one_frozen_sample_inside_moving_batch() -> None:
    from midibrave.zrave_flow_loss import _temporal_motion_loss

    target = torch.zeros(2, 64, 4)
    target[0] = torch.arange(64).view(64, 1) * 0.1
    target[1] = torch.arange(64).view(64, 1) * 0.2
    prediction = target.clone()
    prediction[0] = prediction[0, :1]
    mask = torch.ones(2, 64, dtype=torch.bool)

    loss = _temporal_motion_loss(
        prediction,
        target,
        mask,
        torch.ones(4),
    )

    assert loss > 0.1


def test_temporal_loss_does_not_force_motion_on_static_target() -> None:
    from midibrave.zrave_flow_loss import _temporal_motion_loss

    value = torch.ones(2, 64, 4)
    mask = torch.ones(2, 64, dtype=torch.bool)

    loss = _temporal_motion_loss(
        value,
        value,
        mask,
        torch.ones(4),
    )

    assert loss < 1.0e-6


def test_exploration_pure_loss_reports_weighted_temporal_component() -> None:
    arguments = _loss_fixture()
    for name in ("midi_note", "pitch_probe", "pitch_weight"):
        arguments.pop(name)

    report = zrave_pure_flow_loss(
        **arguments,
        temporal_weight=0.05,
    )

    assert set(report.components) == {
        "flow",
        "boundary",
        "statistics",
        "temporal",
    }
    torch.testing.assert_close(
        report.total,
        report.components["flow"]
        + 0.10 * report.components["boundary"]
        + 0.02 * report.components["statistics"]
        + 0.05 * report.components["temporal"],
    )


def test_auxiliary_losses_receive_raw_codec_coordinates() -> None:
    statistics = _statistics(mean=10.0, latent_std=2.0)
    probe = _RecordingPitchProbe()

    zrave_flow_loss(
        **_loss_fixture(
            clean_future=torch.full((2, 64, 16), 12.0),
            statistics=statistics,
            pitch_probe=probe,
            perfect_velocity=True,
        )
    )

    assert torch.allclose(
        probe.seen_windows,
        torch.full_like(probe.seen_windows, 12.0),
    )


def test_pitch_weight_controller_respects_ratio_and_bounds() -> None:
    controller = PitchWeightController(
        initial=0.30,
        minimum=0.10,
        maximum=1.00,
        target_minimum=0.20,
        target_maximum=0.35,
        ema_decay=0.90,
    )

    assert controller.update(1.0, 10.0, update=100) < 0.30
    for update in range(200, 5000, 100):
        controller.update(1.0, 0.001, update=update)

    assert controller.value == 1.00
    restored = PitchWeightController()
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_distributed_gradient_norm_uses_unscaled_l2_norm() -> None:
    parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
    loss = (parameter.square()).sum()

    norm = distributed_gradient_l2_norm(loss, [parameter])

    assert math.isclose(norm, 10.0)
