from __future__ import annotations

import hashlib
import inspect

import pytest
import torch

from midibrave.zrave_flow_exploration import trailing_history_mask
from midibrave.zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    derive_block_seed,
    retention_curve,
    sample_flow_block,
    sample_pure_flow_block,
    temperature_curve,
)


def _unit_statistics() -> FlowStatistics:
    return FlowStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        latent_norm_p01=torch.tensor(0.5),
        latent_norm_p99=torch.tensor(8.0),
    )


def _small_model(
    *,
    pitch_conditioning: bool = True,
) -> tuple[ZraveFlowTransformer, FlowStatistics]:
    statistics = _unit_statistics()
    model = ZraveFlowTransformer(
        statistics=statistics,
        latent_dim=16,
        context_frames=32,
        future_frames=64,
        d_model=32,
        context_layers=1,
        future_layers=1,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
        pitch_conditioning=pitch_conditioning,
    )
    return model, statistics


def test_retention_and_temperature_have_approved_endpoints() -> None:
    delays = torch.tensor([16, 32, 48])

    retention = retention_curve(delays, 64)
    temperature = temperature_curve(torch.ones(3), delays, 64)

    assert torch.allclose(retention[:, 0], torch.ones(3))
    assert torch.allclose(
        retention[:, -1],
        torch.full((3,), 0.15),
    )
    assert torch.allclose(
        temperature[:, 0],
        torch.full((3,), 0.15),
    )
    assert torch.allclose(temperature[:, -1], torch.ones(3))
    assert torch.all(retention[:, 1:] <= retention[:, :-1])
    assert torch.all(temperature[:, 1:] >= temperature[:, :-1])


def test_absolute_schedule_does_not_reset_at_second_commit() -> None:
    delay = torch.tensor([32.0])

    first_temperature = temperature_curve(
        torch.tensor([1.0]),
        delay,
        16,
        offset_frames=0,
    )
    second_temperature = temperature_curve(
        torch.tensor([1.0]),
        delay,
        16,
        offset_frames=16,
    )
    second_retention = retention_curve(
        delay,
        16,
        offset_frames=16,
    )

    assert second_temperature[0, 0] > first_temperature[0, 0]
    assert second_retention[0, 0] < 1.0


def test_zero_absolute_offset_preserves_legacy_schedule() -> None:
    delays = torch.tensor([16.0, 32.0, 48.0])
    temperatures = torch.tensor([0.7, 1.0, 1.3])

    torch.testing.assert_close(
        retention_curve(delays, 64, offset_frames=0),
        retention_curve(delays, 64),
    )
    torch.testing.assert_close(
        temperature_curve(
            temperatures,
            delays,
            64,
            offset_frames=0,
        ),
        temperature_curve(temperatures, delays, 64),
    )


def test_flow_forward_and_sampling_contract() -> None:
    model, statistics = _small_model()
    model.eval()
    history = torch.randn(2, 32, 16)
    noisy = torch.randn(2, 64, 16)
    velocity = model(
        noisy,
        torch.tensor([0.25, 0.75]),
        history,
        torch.tensor([48, 72]),
        retention_curve(torch.tensor([16, 48]), 64),
    )
    assert velocity.shape == noisy.shape

    arguments = dict(
        model=model,
        statistics=statistics,
        history=history[:1],
        midi_note=torch.tensor([60]),
        block_index=3,
        temperature=1.0,
        wander_delay_frames=32,
        pitch_guidance=3.0,
        solver_steps=4,
    )
    first = sample_flow_block(generation_seed=99, **arguments)
    second = sample_flow_block(generation_seed=99, **arguments)
    different = sample_flow_block(generation_seed=100, **arguments)

    assert first.shape == (1, 64, 16)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_pure_forward_has_no_midi_argument() -> None:
    signature = inspect.signature(ZraveFlowTransformer.forward_pure)

    assert "midi_note" not in signature.parameters
    assert "context_present" not in signature.parameters


def test_pure_forward_and_sampling_replay_seed_and_branch() -> None:
    model, statistics = _small_model(pitch_conditioning=False)
    model.eval()
    history = torch.randn(1, 32, 16)
    noisy = torch.randn(1, 64, 16)
    velocity = model.forward_pure(
        noisy,
        torch.tensor([0.5]),
        history,
        retention_curve(torch.tensor([32]), 64),
    )

    arguments = {
        "model": model,
        "statistics": statistics,
        "history": history,
        "block_index": 3,
        "temperature": 1.0,
        "wander_delay_frames": 32,
        "solver_steps": 4,
    }
    first = sample_pure_flow_block(generation_seed=99, **arguments)
    replay = sample_pure_flow_block(generation_seed=99, **arguments)
    branch = sample_pure_flow_block(generation_seed=100, **arguments)

    assert velocity.shape == noisy.shape
    assert first.shape == (1, 64, 16)
    assert torch.equal(first, replay)
    assert not torch.equal(first, branch)
    assert all(
        "midi" not in name
        for name, _parameter in model.named_parameters()
    )


def test_pure_sampling_accepts_continuous_exploration_schedule() -> None:
    model, statistics = _small_model(pitch_conditioning=False)
    model.eval()

    result = sample_pure_flow_block(
        model,
        statistics,
        torch.randn(1, 32, 16),
        generation_seed=31,
        block_index=2,
        temperature=1.1,
        wander_delay_frames=24.0,
        solver_steps=4,
        schedule_offset_frames=48,
        visible_history_frames=16,
    )

    assert result.shape == (1, 64, 16)
    assert torch.isfinite(result).all()


def test_masked_old_history_cannot_change_pure_velocity() -> None:
    model, _statistics = _small_model(pitch_conditioning=False)
    model.eval()
    noisy = torch.randn(1, 64, 16)
    history = torch.randn(1, 32, 16)
    changed = history.clone()
    changed[:, :24] = torch.randn_like(changed[:, :24]) * 1000.0
    history_mask = trailing_history_mask(torch.tensor([8]), 32)
    retention = retention_curve(torch.tensor([32.0]), 64)

    first = model.forward_pure(
        noisy,
        torch.tensor([0.5]),
        history,
        retention,
        history_mask=history_mask,
    )
    second = model.forward_pure(
        noisy,
        torch.tensor([0.5]),
        changed,
        retention,
        history_mask=history_mask,
    )

    torch.testing.assert_close(first, second)


def test_history_mask_requires_one_visible_frame_per_sample() -> None:
    model, _statistics = _small_model(pitch_conditioning=False)

    with pytest.raises(ValueError, match="visible history"):
        model.forward_pure(
            torch.randn(1, 64, 16),
            torch.tensor([0.5]),
            torch.randn(1, 32, 16),
            retention_curve(torch.tensor([32.0]), 64),
            history_mask=torch.zeros(1, 32, dtype=torch.bool),
        )


def test_history_mask_support_does_not_change_state_dict_contract() -> None:
    model, statistics = _small_model(pitch_conditioning=False)
    expected = {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
    }
    replacement = ZraveFlowTransformer(
        statistics=statistics,
        latent_dim=16,
        context_frames=32,
        future_frames=64,
        d_model=32,
        context_layers=1,
        future_layers=1,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
        pitch_conditioning=False,
    )

    replacement.load_state_dict(model.state_dict(), strict=True)

    assert {
        name: tuple(value.shape)
        for name, value in replacement.state_dict().items()
    } == expected


def test_context_and_pitch_conditions_can_drop_independently() -> None:
    model, _statistics = _small_model()
    noisy = torch.randn(2, 64, 16)
    history = torch.randn(2, 32, 16)
    retention = retention_curve(torch.tensor([32, 32]), 64)

    output = model(
        noisy,
        torch.full((2,), 0.5),
        history,
        torch.tensor([-1, 60]),
        retention,
        context_present=torch.tensor([True, False]),
    )

    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()


def test_masked_future_noise_cannot_change_valid_velocity() -> None:
    model, _statistics = _small_model()
    model.eval()
    noisy = torch.randn(1, 64, 16)
    changed = noisy.clone()
    changed[:, 20:] = torch.randn_like(changed[:, 20:]) * 1000.0
    mask = torch.arange(64).unsqueeze(0) < 20
    arguments = (
        torch.tensor([0.5]),
        torch.randn(1, 32, 16),
        torch.tensor([60]),
        retention_curve(torch.tensor([32]), 64),
    )

    first = model(noisy, *arguments, future_mask=mask)
    second = model(changed, *arguments, future_mask=mask)

    torch.testing.assert_close(first[:, :20], second[:, :20])


def test_sampling_uses_sha256_seed_without_consuming_global_rng() -> None:
    model, statistics = _small_model()
    expected_seed = int.from_bytes(
        hashlib.sha256(b"123:7").digest()[:8],
        byteorder="big",
        signed=False,
    )
    assert derive_block_seed(123, 7) == expected_seed
    before = torch.random.get_rng_state().clone()

    sample_flow_block(
        model,
        statistics,
        torch.zeros(1, 32, 16),
        torch.tensor([60]),
        generation_seed=123,
        block_index=7,
        temperature=0.0,
        wander_delay_frames=16,
        pitch_guidance=1.0,
        solver_steps=4,
    )

    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    ("temperature", "delay", "guidance", "steps"),
    [
        (-0.1, 32, 3.0, 8),
        (1.0, 24, 3.0, 8),
        (1.0, 32, 0.9, 8),
        (1.0, 32, 5.1, 8),
        (1.0, 32, 3.0, 6),
    ],
)
def test_sampling_rejects_unsupported_controls(
    temperature: float,
    delay: int,
    guidance: float,
    steps: int,
) -> None:
    model, statistics = _small_model()

    with pytest.raises(ValueError):
        sample_flow_block(
            model,
            statistics,
            torch.zeros(1, 32, 16),
            torch.tensor([60]),
            generation_seed=1,
            block_index=0,
            temperature=temperature,
            wander_delay_frames=delay,
            pitch_guidance=guidance,
            solver_steps=steps,
        )
