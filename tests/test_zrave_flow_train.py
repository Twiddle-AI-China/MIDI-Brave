from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from midibrave.zrave_flow_config import ZraveFlowConfig
from midibrave.zrave_flow_loss import PitchWeightController
from midibrave.zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
)
from midibrave.zrave_flow_sampler import FlowBatch
from midibrave.zrave_flow_train import (
    _parser,
    _run_flow_update,
    build_flow_checkpoint_contract,
    load_flow_checkpoint,
    maximum_valid_future,
    roll_exposure_history,
    save_flow_checkpoint,
    summarize_flow_benchmark,
    use_exposure_batch,
)

ROOT = Path(__file__).parents[1]
PURE_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_pure_flow_poc.yaml"
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


class _PureFlowSampler:
    def __init__(self, seed: int = 71) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def sample(
        self,
        batch_size: int,
        *,
        maximum_valid_future: int,
        require_full_future: bool,
    ) -> FlowBatch:
        valid_frames = 64 if require_full_future else maximum_valid_future
        history = torch.randn(
            batch_size,
            32,
            16,
            generator=self.generator,
        )
        future = torch.randn(
            batch_size,
            64,
            16,
            generator=self.generator,
        )
        future_mask = (
            torch.arange(64).unsqueeze(0) < valid_frames
        ).expand(batch_size, -1)
        zeros = torch.zeros(batch_size, dtype=torch.long)
        return FlowBatch(
            history=history,
            future=future,
            future_mask=future_mask,
            midi_note=zeros,
            source_code=zeros,
            category_code=zeros,
            wander_delay_frames=torch.full(
                (batch_size,),
                32,
                dtype=torch.long,
            ),
            history_midi_note=zeros,
            pitch_transition_mask=torch.zeros(
                batch_size,
                dtype=torch.bool,
            ),
        )


def _unit_statistics() -> FlowStatistics:
    return FlowStatistics(
        mean=torch.zeros(16),
        latent_std=torch.ones(16),
        delta_std=torch.ones(16),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )


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


def test_pure_cli_does_not_require_pitch_probe() -> None:
    arguments = _parser().parse_args(["--config", str(PURE_CONFIG)])

    assert arguments.pitch_probe is None


def test_pure_runtime_contract_has_no_pitch_hashes() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=8,
        batch_per_gpu=128,
        maximum_updates=100000,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )

    assert "pitch_checkpoint_sha256" not in contract
    assert "pitch_qualification_sha256" not in contract
    assert contract["pitch_conditioning"] is False


def test_pure_update_reports_only_pure_losses() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
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
        pitch_conditioning=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    result = _run_flow_update(
        training_model=model,
        sampler=_PureFlowSampler(),
        pitch_probe=None,
        statistics=statistics,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        controller=None,
        config=config,
        update=0,
        maximum_updates=100,
        batch_per_gpu=2,
        device=torch.device("cpu"),
        rank=0,
        force_exposure=False,
    )

    assert set(result.components) == {
        "flow",
        "boundary",
        "statistics",
    }
    assert result.pitch_metrics == {}
    assert result.pitch_weight == 0.0
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_pure_checkpoint_omits_pitch_state(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=5)
    config = ZraveFlowConfig.load(PURE_CONFIG)
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=1,
        batch_per_gpu=2,
        maximum_updates=100,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )
    checkpoint = tmp_path / "pure.pt"

    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=1,
        contract=contract,
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert payload["architecture"] == "zrave_pure_flow_transformer_v1"
    assert "pitch_weight_controller" not in payload
    restored = load_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        expected_contract=contract,
    )
    assert restored["update"] == 1


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


def test_flow_checkpoint_restores_only_explicit_gate_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(5)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=6)
    controller = PitchWeightController()
    contract = {
        "world_size": 1,
        "batch_per_gpu": 2,
        "config_sha256": "a" * 64,
        "pack_index_sha256": "b" * 64,
        "statistics_sha256": "c" * 64,
        "pitch_checkpoint_sha256": "d" * 64,
        "pitch_qualification_sha256": "e" * 64,
    }
    checkpoint = tmp_path / "gated.pt"
    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=controller,
        update=5000,
        contract=contract,
        latest_gate_report_sha256="f" * 64,
        consecutive_gate_passes=2,
    )

    restored = load_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=controller,
        expected_contract=contract,
    )

    assert restored["latest_gate_report_sha256"] == "f" * 64
    assert restored["consecutive_gate_passes"] == 2
