from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from midibrave.zrave_flow_loss import PitchWeightController
from midibrave.zrave_flow_train import (
    load_flow_checkpoint,
    maximum_valid_future,
    roll_exposure_history,
    save_flow_checkpoint,
    summarize_flow_benchmark,
    use_exposure_batch,
)


class _TinySampler:
    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.randn(
            batch_size,
            1,
            generator=self.generator,
        )
        return inputs, 2.0 * inputs

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        self.generator.set_state(state["generator_state"])


def _run_tiny_training(
    root: Path,
    *,
    updates: int,
    save: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    torch.manual_seed(12)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    sampler = _TinySampler(seed=44)
    controller = PitchWeightController()
    checkpoint = root / "flow.pt"
    contract = {
        "world_size": 1,
        "batch_per_gpu": 2,
        "config_sha256": "a" * 64,
        "pack_index_sha256": "b" * 64,
        "statistics_sha256": "c" * 64,
        "pitch_checkpoint_sha256": "d" * 64,
        "pitch_qualification_sha256": "e" * 64,
    }
    start = 0
    if resume:
        restored = load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=controller,
            expected_contract=contract,
        )
        start = restored["update"]

    for _update in range(start, start + updates):
        inputs, target = sampler.sample()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()

    if save:
        save_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=controller,
            update=start + updates,
            contract=contract,
        )
    next_inputs, next_target = sampler.sample()
    next_loss = (model(next_inputs) - next_target).square().mean()
    return {
        "next_loss": float(next_loss.detach()),
        "next_parameters": [
            parameter.detach().tolist() for parameter in model.parameters()
        ],
    }


def test_future_curriculum_switches_at_5000() -> None:
    assert maximum_valid_future(update=0, short_updates=5000) == 32
    assert maximum_valid_future(update=4999, short_updates=5000) == 32
    assert maximum_valid_future(update=5000, short_updates=5000) == 64


def test_exposure_batches_start_only_in_final_twenty_percent() -> None:
    generator = torch.Generator().manual_seed(3)

    assert not use_exposure_batch(
        79999,
        100000,
        0.8,
        1.0,
        generator,
    )
    assert use_exposure_batch(
        80000,
        100000,
        0.8,
        1.0,
        generator,
    )


def test_exposure_rolls_generated_prefix_into_history() -> None:
    history = (
        torch.arange(32)
        .view(1, 32, 1)
        .repeat(1, 1, 16)
        .float()
    )
    generated = (
        torch.arange(32, 64)
        .view(1, 32, 1)
        .repeat(1, 1, 16)
        .float()
    )

    rolled = roll_exposure_history(history, generated)

    assert torch.equal(rolled, generated)
    assert not rolled.requires_grad


def test_flow_checkpoint_resume_reproduces_next_update(
    tmp_path: Path,
) -> None:
    continuous = _run_tiny_training(tmp_path / "a", updates=3)
    _run_tiny_training(tmp_path / "b", updates=2, save=True)
    resumed = _run_tiny_training(
        tmp_path / "b",
        updates=1,
        resume=True,
    )

    assert continuous["next_loss"] == resumed["next_loss"]
    assert (
        continuous["next_parameters"]
        == resumed["next_parameters"]
    )


def test_benchmark_reports_valid_frame_throughput_and_exposure() -> None:
    report = summarize_flow_benchmark(
        batch_per_gpu=128,
        world_size=8,
        durations_seconds=[2.0] * 100,
        global_valid_frames=[32768] * 100,
        exposure_safety_updates=5,
        peak_memory_mib=1000.0,
        total_memory_mib=16000.0,
        nonfinite_updates=0,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        pitch_checkpoint_sha256="c" * 64,
        git_commit="d" * 40,
    )

    assert report["status"] == "ok"
    assert report["measured_updates"] == 100
    assert report["exposure_safety_updates"] == 5
    assert report["median_windows_per_second"] == 512.0
    assert (
        report["median_valid_latent_frames_per_second"]
        == 16384.0
    )
