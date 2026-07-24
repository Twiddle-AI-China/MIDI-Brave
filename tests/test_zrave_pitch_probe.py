from __future__ import annotations

import pytest
import torch

from midibrave.zrave_pitch_probe import (
    LatentPitchProbe,
    freeze_pitch_probe,
    pitch_probe_loss,
)


def test_pitch_probe_predicts_note_logits_and_expectation() -> None:
    probe = LatentPitchProbe()

    output = probe(torch.randn(5, 16, 16))

    assert output.logits.shape == (5, 89)
    assert output.expected_midi.shape == (5,)
    assert torch.all(output.expected_midi >= 21)
    assert torch.all(output.expected_midi <= 109)


def test_pitch_probe_loss_and_freeze_contract() -> None:
    probe = LatentPitchProbe()
    notes = torch.tensor([36, 48, 60, 72])

    report = pitch_probe_loss(probe(torch.randn(4, 16, 16)), notes)

    assert set(report.components) == {
        "cross_entropy",
        "midi_smooth_l1",
    }
    assert torch.isfinite(report.total)
    report.total.backward()
    assert any(parameter.grad is not None for parameter in probe.parameters())

    frozen = freeze_pitch_probe(probe)
    assert frozen is probe
    assert not frozen.training
    assert all(
        not parameter.requires_grad for parameter in frozen.parameters()
    )


@pytest.mark.parametrize(
    "latents",
    [
        torch.randn(16, 16),
        torch.randn(2, 15, 16),
        torch.randn(2, 16, 15),
    ],
)
def test_pitch_probe_rejects_wrong_latent_shape(
    latents: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="latents"):
        LatentPitchProbe()(latents)


@pytest.mark.parametrize(
    "notes",
    [
        torch.tensor([20, 60]),
        torch.tensor([60, 110]),
        torch.tensor([60.0, 60.5]),
        torch.tensor([[60, 72]]),
    ],
)
def test_pitch_probe_loss_rejects_invalid_notes(
    notes: torch.Tensor,
) -> None:
    output = LatentPitchProbe()(torch.randn(2, 16, 16))

    with pytest.raises(ValueError, match="midi_note"):
        pitch_probe_loss(output, notes)
