from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class PitchProbeOutput(NamedTuple):
    logits: Tensor
    expected_midi: Tensor


@dataclass(frozen=True)
class PitchProbeLoss:
    total: Tensor
    components: dict[str, Tensor]


class _ResidualPitchBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.network(inputs)


class LatentPitchProbe(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        note_min: int = 21,
        note_max: int = 109,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if note_min < 0 or note_max < note_min:
            raise ValueError("invalid MIDI note range")
        self.latent_dim = latent_dim
        self.note_min = note_min
        self.note_max = note_max
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_projection = nn.Conv1d(
            latent_dim,
            128,
            kernel_size=5,
            padding=2,
        )
        self.residual_blocks = nn.Sequential(
            *(
                _ResidualPitchBlock(128, dilation)
                for dilation in (1, 2, 4, 8)
            )
        )
        self.output_norm = nn.GroupNorm(8, 128)
        self.output = nn.Linear(128, note_max - note_min + 1)

    def forward(self, latents: Tensor) -> PitchProbeOutput:
        expected = (16, self.latent_dim)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                "latents must have shape "
                f"[batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(latents.shape)}"
            )
        hidden = self.input_norm(latents.float()).transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.residual_blocks(hidden)
        hidden = functional.gelu(self.output_norm(hidden))
        logits = self.output(hidden.mean(dim=-1))
        probabilities = logits.float().softmax(dim=-1)
        note_values = torch.arange(
            self.note_min,
            self.note_max + 1,
            device=logits.device,
            dtype=torch.float32,
        )
        expected_midi = torch.sum(probabilities * note_values, dim=-1)
        return PitchProbeOutput(
            logits=logits,
            expected_midi=expected_midi,
        )


def _validated_notes(
    midi_note: Tensor,
    *,
    batch_size: int,
    note_min: int,
    note_max: int,
) -> Tensor:
    if midi_note.ndim != 1 or midi_note.shape[0] != batch_size:
        raise ValueError(
            f"midi_note must have shape ({batch_size},), "
            f"got {tuple(midi_note.shape)}"
        )
    if midi_note.is_floating_point():
        if not torch.isfinite(midi_note).all():
            raise ValueError("midi_note contains non-finite values")
        if not torch.equal(midi_note, midi_note.round()):
            raise ValueError("midi_note must contain integer values")
    notes = midi_note.to(dtype=torch.long)
    if torch.any(notes < note_min) or torch.any(notes > note_max):
        raise ValueError(
            f"midi_note must be in [{note_min}, {note_max}]"
        )
    return notes


def pitch_probe_loss(
    output: PitchProbeOutput,
    midi_note: Tensor,
    *,
    note_min: int = 21,
    note_max: int = 109,
) -> PitchProbeLoss:
    classes = note_max - note_min + 1
    if output.logits.ndim != 2 or output.logits.shape[1] != classes:
        raise ValueError(
            f"output.logits must have shape [batch, {classes}]"
        )
    if output.expected_midi.shape != (output.logits.shape[0],):
        raise ValueError(
            "output.expected_midi must have shape [batch]"
        )
    notes = _validated_notes(
        midi_note,
        batch_size=output.logits.shape[0],
        note_min=note_min,
        note_max=note_max,
    ).to(device=output.logits.device)
    cross_entropy = functional.cross_entropy(
        output.logits.float(),
        notes - note_min,
    )
    midi_smooth_l1 = functional.smooth_l1_loss(
        output.expected_midi.float(),
        notes.float(),
    )
    return PitchProbeLoss(
        total=cross_entropy + 0.25 * midi_smooth_l1,
        components={
            "cross_entropy": cross_entropy,
            "midi_smooth_l1": midi_smooth_l1,
        },
    )


def freeze_pitch_probe(model: LatentPitchProbe) -> LatentPitchProbe:
    model.eval()
    model.requires_grad_(False)
    return model
