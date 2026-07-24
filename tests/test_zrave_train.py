from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from midibrave.zrave_config import (
    ZraveLossConfig,
    ZraveModelConfig,
    ZraveTrainConfig,
)
from midibrave.zrave_model import (
    ZravePrediction,
    ZraveStatistics,
    ZraveTransformer,
    zrave_prediction_loss,
)
from midibrave.zrave_train import (
    GpuWindowSampler,
    allowed_rollout_depth,
    condition_rollout_history,
    load_zrave_checkpoint,
    load_zrave_warm_start,
    rollout_training_frames,
    sample_rollout_depth,
    save_zrave_checkpoint,
    select_rollout_target,
    should_checkpoint,
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


def test_validation_improvement_always_triggers_checkpoint() -> None:
    assert should_checkpoint(
        update=2750,
        checkpoint_every=500,
        final_due=False,
        stopped_early=False,
        validation_improved=True,
    )
    assert not should_checkpoint(
        update=2750,
        checkpoint_every=500,
        final_due=False,
        stopped_early=False,
        validation_improved=False,
    )


def test_rollout_depth_curriculum_is_bounded() -> None:
    assert allowed_rollout_depth(0, 7, 500) == 2
    assert allowed_rollout_depth(250, 7, 500) == 4
    assert allowed_rollout_depth(500, 7, 500) == 7
    assert allowed_rollout_depth(900, 7, 500) == 7
    assert allowed_rollout_depth(0, 0, 500) == 0


def test_rollout_depth_sampling_is_reproducible() -> None:
    first = torch.Generator().manual_seed(31)
    second = torch.Generator().manual_seed(31)

    actual = [
        sample_rollout_depth(500, 7, 500, 0.25, first, "cpu")
        for _ in range(20)
    ]
    expected = [
        sample_rollout_depth(500, 7, 500, 0.25, second, "cpu")
        for _ in range(20)
    ]

    assert actual == expected
    assert set(actual) <= set(range(8))
    assert 0 in actual
    assert any(depth > 0 for depth in actual)


def test_rollout_config_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="rollout_teacher_probability"):
        ZraveTrainConfig(
            output_root="run",
            rollout_teacher_probability=1.1,
        )


class _IncrementModel:
    config = SimpleNamespace(context_frames=4, horizon_frames=2)

    def __call__(self, history: torch.Tensor) -> ZravePrediction:
        latent = history[:, -1:] + torch.arange(
            1,
            3,
            dtype=history.dtype,
            device=history.device,
        )[None, :, None]
        delta = torch.diff(
            torch.cat((history[:, -1:], latent), dim=1),
            dim=1,
        )
        return ZravePrediction(delta=delta, latent=latent)


def test_condition_rollout_history_detaches_generated_chunks() -> None:
    history = torch.zeros(2, 4, 16, requires_grad=True)

    conditioned = condition_rollout_history(
        _IncrementModel(),
        history,
        depth=2,
    )

    assert conditioned.shape == history.shape
    assert conditioned.requires_grad is False
    torch.testing.assert_close(
        conditioned[:, -1],
        torch.full((2, 16), 4.0),
    )


def test_select_rollout_target_uses_matching_chunk() -> None:
    target = torch.arange(2 * 64 * 16).reshape(2, 64, 16)

    actual = select_rollout_target(
        target,
        depth=3,
        horizon_frames=8,
    )

    torch.testing.assert_close(actual, target[:, 24:32])


def test_rollout_training_window_covers_every_depth_target() -> None:
    assert rollout_training_frames(
        horizon_frames=8,
        maximum_depth=7,
    ) == 64
    assert rollout_training_frames(
        horizon_frames=16,
        maximum_depth=0,
    ) == 16


def test_warm_start_loads_only_compatible_model_weights(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    source, _ = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-4)
    sampler = _sampler()
    checkpoint = tmp_path / "source.pt"
    contract = {
        "packed_index_sha256": "index",
        "statistics_sha256": "statistics",
        "latent_dim": 16,
        "context_frames": 128,
        "horizon_frames": 16,
    }
    save_zrave_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        update=7,
        contract=contract,
        world_size=1,
        batch_per_gpu=2,
        best_validation_metric=1.0,
        validations_without_improvement=0,
    )
    torch.manual_seed(43)
    target, _ = _model()
    fresh_optimizer = torch.optim.AdamW(target.parameters(), lr=2.0e-4)

    report = load_zrave_warm_start(
        checkpoint,
        model=target,
        expected_contract=contract,
    )

    assert report["source_update"] == 7
    assert fresh_optimizer.state == {}
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], expected)
