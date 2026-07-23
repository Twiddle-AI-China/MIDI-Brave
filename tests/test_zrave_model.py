from __future__ import annotations

import torch

from midibrave.zrave_config import ZraveLossConfig, ZraveModelConfig
from midibrave.zrave_model import (
    ZraveStatistics,
    ZraveTransformer,
    zrave_prediction_loss,
)


def _unit_statistics(latent_dim: int) -> ZraveStatistics:
    return ZraveStatistics(
        mean=torch.zeros(latent_dim),
        latent_std=torch.ones(latent_dim),
        delta_std=torch.ones(latent_dim),
        acceleration_std=torch.ones(latent_dim),
    )


def _small_model() -> tuple[ZraveTransformer, ZraveStatistics]:
    config = ZraveModelConfig(
        latent_dim=16,
        context_frames=128,
        horizon_frames=16,
        d_model=96,
        layers=2,
        heads=4,
        feedforward_dim=192,
        dropout=0.0,
    )
    statistics = _unit_statistics(config.latent_dim)
    return ZraveTransformer(config, statistics), statistics


def test_transformer_predicts_direct_future_differences() -> None:
    model, _ = _small_model()
    history = torch.randn(3, 128, 16)

    result = model(history)

    assert result.delta.shape == (3, 16, 16)
    assert result.latent.shape == (3, 16, 16)
    torch.testing.assert_close(
        result.latent,
        history[:, -1:] + result.delta.cumsum(dim=1),
    )


def test_loss_uses_only_history_prediction_and_target() -> None:
    model, statistics = _small_model()
    history = torch.randn(2, 128, 16)
    target = torch.randn(2, 16, 16)

    prediction = model(history)
    loss = zrave_prediction_loss(
        prediction,
        history,
        target,
        statistics,
        ZraveLossConfig(),
    )

    assert set(loss.components) == {"future", "delta", "acceleration"}
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_zero_error_has_zero_loss() -> None:
    _, statistics = _small_model()
    history = torch.zeros(2, 128, 16)
    target = torch.zeros(2, 16, 16)
    model, _ = _small_model()
    prediction = model(history)
    zero_prediction = type(prediction)(
        delta=torch.zeros_like(prediction.delta),
        latent=torch.zeros_like(prediction.latent),
    )

    loss = zrave_prediction_loss(
        zero_prediction,
        history,
        target,
        statistics,
        ZraveLossConfig(),
    )

    torch.testing.assert_close(loss.total, torch.tensor(0.0))


def test_transformer_rejects_wrong_context_length() -> None:
    model, _ = _small_model()

    try:
        model(torch.randn(1, 127, 16))
    except ValueError as error:
        assert "history" in str(error)
        assert "128" in str(error)
    else:
        raise AssertionError("wrong context length was accepted")
