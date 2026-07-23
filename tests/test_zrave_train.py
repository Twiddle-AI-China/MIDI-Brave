from __future__ import annotations

from pathlib import Path

import torch

from midibrave.zrave_config import (
    ZraveLossConfig,
    ZraveModelConfig,
)
from midibrave.zrave_model import (
    ZraveStatistics,
    ZraveTransformer,
    zrave_prediction_loss,
)
from midibrave.zrave_train import (
    GpuWindowSampler,
    load_zrave_checkpoint,
    save_zrave_checkpoint,
    summarize_benchmark,
)


def _sampler(seed: int = 17) -> GpuWindowSampler:
    latents = torch.arange(
        5 * 180 * 16,
        dtype=torch.float32,
    ).reshape(5, 180, 16)
    lengths = torch.tensor([150, 160, 170, 175, 180])
    splits = torch.tensor([0, 0, 1, 2, 0])
    return GpuWindowSampler(
        latents,
        lengths,
        splits,
        context_frames=128,
        horizon_frames=16,
        split_code=0,
        seed=seed,
    )


def _model() -> tuple[ZraveTransformer, ZraveStatistics]:
    config = ZraveModelConfig(
        d_model=32,
        layers=1,
        heads=4,
        feedforward_dim=64,
    )
    statistics = ZraveStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        acceleration_std=torch.ones(16),
    )
    return ZraveTransformer(config, statistics), statistics


def test_gpu_sampler_stays_inside_training_sequences() -> None:
    sampler = _sampler()

    history, target = sampler.sample(32)

    assert history.shape == (32, 128, 16)
    assert target.shape == (32, 16, 16)
    assert torch.isfinite(history).all()
    assert set(sampler.last_sequence_indices.tolist()) <= {0, 1, 4}
    assert torch.all(
        sampler.last_start_indices + 144
        <= sampler.lengths[sampler.last_sequence_indices]
    )


def _update(
    model: ZraveTransformer,
    optimizer: torch.optim.Optimizer,
    sampler: GpuWindowSampler,
    statistics: ZraveStatistics,
) -> float:
    history, target = sampler.sample(2)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(history)
    loss = zrave_prediction_loss(
        prediction,
        history,
        target,
        statistics,
        ZraveLossConfig(),
    ).total
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def test_checkpoint_resume_reproduces_next_update(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model, statistics = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    sampler = _sampler(seed=29)
    _update(model, optimizer, sampler, statistics)
    checkpoint = tmp_path / "update-1.pt"
    contract = {
        "config_sha256": "config",
        "packed_index_sha256": "index",
        "statistics_sha256": "statistics",
    }
    save_zrave_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        update=1,
        contract=contract,
        world_size=1,
        batch_per_gpu=2,
        best_validation_metric=1.0,
        validations_without_improvement=0,
    )
    expected_loss = _update(model, optimizer, sampler, statistics)
    expected_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    resumed_model, resumed_statistics = _model()
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=1.0e-4,
    )
    resumed_sampler = _sampler(seed=999)
    restored = load_zrave_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=None,
        sampler=resumed_sampler,
        expected_contract=contract,
    )
    actual_loss = _update(
        resumed_model,
        resumed_optimizer,
        resumed_sampler,
        resumed_statistics,
    )

    assert restored["update"] == 1
    assert actual_loss == expected_loss
    for name, expected in expected_state.items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_benchmark_summary_contains_selection_evidence() -> None:
    report = summarize_benchmark(
        batch_per_gpu=256,
        world_size=8,
        durations_seconds=[0.8, 1.0, 1.2],
        peak_memory_mib=12345.0,
        total_memory_mib=32768.0,
        nonfinite_updates=0,
    )

    assert report["batch_per_gpu"] == 256
    assert report["world_size"] == 8
    assert report["global_batch"] == 2048
    assert report["measured_updates"] == 3
    assert report["median_windows_per_second"] == 2048.0
    assert report["p10_windows_per_second"] > 0.0
    assert report["peak_memory_mib"] == 12345.0
    assert report["status"] == "ok"
