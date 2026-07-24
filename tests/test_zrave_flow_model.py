from __future__ import annotations

import hashlib

import pytest
import torch

from midibrave.zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    derive_block_seed,
    retention_curve,
    sample_flow_block,
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


def _small_model() -> tuple[ZraveFlowTransformer, FlowStatistics]:
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
