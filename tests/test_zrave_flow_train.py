from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import midibrave.zrave_flow_train as flow_train
from midibrave.zrave_flow_config import ZraveFlowConfig
from midibrave.zrave_flow_loss import PitchWeightController
from midibrave.zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
)
from midibrave.zrave_flow_sampler import FlowBatch
from midibrave.zrave_flow_train import (
    _git_commit,
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
EXPLORATION_CONFIG = (
    ROOT
    / "configs"
    / "zrave"
    / "octopus_serum128_exploration_flow.yaml"
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


def _pure_model() -> ZraveFlowTransformer:
    return ZraveFlowTransformer(
        statistics=_unit_statistics(),
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


def _numbered_flow_batch(batch_size: int = 2) -> FlowBatch:
    history = torch.arange(32).view(1, 32, 1).repeat(
        batch_size,
        1,
        16,
    ).float()
    future = torch.arange(32, 96).view(1, 64, 1).repeat(
        batch_size,
        1,
        16,
    ).float()
    zeros = torch.zeros(batch_size, dtype=torch.long)
    return FlowBatch(
        history=history,
        future=future,
        future_mask=torch.ones(batch_size, 64, dtype=torch.bool),
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


def test_pure_cli_accepts_phase3_start_update() -> None:
    arguments = _parser().parse_args(
        [
            "--config",
            str(PURE_CONFIG),
            "--resume",
            "step-020000.pt",
            "--phase3-start-update",
            "20000",
        ]
    )

    assert arguments.phase3_start_update == 20000


def test_pure_cli_accepts_weight_only_initializer() -> None:
    arguments = _parser().parse_args(
        [
            "--config",
            str(EXPLORATION_CONFIG),
            "--initialize-from",
            "step-085000.pt",
        ]
    )

    assert arguments.initialize_from == "step-085000.pt"
    assert arguments.resume is None


def test_resume_and_weight_initializer_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--config",
                str(EXPLORATION_CONFIG),
                "--resume",
                "resume.pt",
                "--initialize-from",
                "initial.pt",
            ]
        )


def test_legacy_checkpoint_can_enter_phase3_at_resume_update() -> None:
    assert (
        flow_train.resolve_phase3_start_update(
            maximum_updates=100000,
            default_fraction=0.8,
            resumed_update=20000,
            requested=20000,
            restored=None,
        )
        == 20000
    )


def test_persisted_phase3_start_rejects_conflicting_override() -> None:
    assert (
        flow_train.resolve_phase3_start_update(
            maximum_updates=100000,
            default_fraction=0.8,
            resumed_update=25000,
            requested=None,
            restored=20000,
        )
        == 20000
    )

    with pytest.raises(
        ValueError,
        match="phase3 start conflicts with checkpoint",
    ):
        flow_train.resolve_phase3_start_update(
            maximum_updates=100000,
            default_fraction=0.8,
            resumed_update=25000,
            requested=80000,
            restored=20000,
        )


def test_git_commit_marks_only_exact_repo_as_safe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    class _Result:
        stdout = "d" * 40 + "\n"

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return _Result()

    monkeypatch.setattr(
        "midibrave.zrave_flow_train.subprocess.run",
        fake_run,
    )

    assert _git_commit(tmp_path) == "d" * 40
    assert observed["command"] == [
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
        "rev-parse",
        "HEAD",
    ]
    assert observed["cwd"] == tmp_path.resolve()


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
    assert contract["exploration_enabled"] is False


def test_exploration_runtime_contract_selects_v2_architecture() -> None:
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG)
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=8,
        batch_per_gpu=128,
        maximum_updates=20000,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )

    assert contract["exploration_enabled"] is True
    assert flow_train._checkpoint_architecture(contract) == (
        "zrave_pure_flow_transformer_v2"
    )


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
    assert result.applied
    assert result.amp_scale == 1.0
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_exploration_update_reports_controls_and_temporal_loss() -> None:
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG)
    statistics = _unit_statistics()
    model = _pure_model()
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
        "temporal",
    }
    assert result.exposure_depth == 0
    assert result.visible_history_frames in {8, 16, 32}
    assert result.schedule_offset_frames in {0, 16, 32, 64, 128}
    assert 0.0 <= result.exploration <= 1.0


def test_nonfinite_gradient_backs_off_amp_without_optimizer_step() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.0)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square()).backward()
    scaler.unscale_(optimizer)
    assert parameter.grad is not None
    parameter.grad.fill_(float("inf"))
    gradient_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    before = parameter.detach().clone()

    applied, amp_scale = flow_train._apply_amp_optimizer_step(
        optimizer=optimizer,
        scaler=scaler,
        gradient_norm=gradient_norm,
    )

    assert not applied
    assert torch.equal(parameter.detach(), before)
    assert parameter.grad is None
    assert amp_scale == 8.0


def test_eight_consecutive_nonfinite_gradients_are_fatal() -> None:
    streak = 0
    for _ in range(7):
        streak = flow_train._next_nonfinite_gradient_streak(
            streak,
            applied=False,
            maximum=8,
        )
    assert streak == 7
    assert (
        flow_train._next_nonfinite_gradient_streak(
            streak,
            applied=True,
            maximum=8,
        )
        == 0
    )

    with pytest.raises(
        FloatingPointError,
        match="8 consecutive non-finite flow gradients",
    ):
        flow_train._next_nonfinite_gradient_streak(
            streak,
            applied=False,
            maximum=8,
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
    assert restored["phase3_start_update"] is None


def test_legacy_pure_checkpoint_without_exploration_flag_still_resumes(
    tmp_path: Path,
) -> None:
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
    checkpoint = tmp_path / "legacy-pure.pt"
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
    payload["contract"].pop("exploration_enabled")
    torch.save(payload, checkpoint)

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


def test_pure_checkpoint_persists_phase3_start(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=5)
    config = ZraveFlowConfig.load(PURE_CONFIG)
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=1,
        batch_per_gpu=2,
        maximum_updates=100000,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )
    checkpoint = tmp_path / "phase3.pt"

    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=20000,
        contract=contract,
        phase3_start_update=20000,
    )
    restored = load_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        expected_contract=contract,
    )

    assert restored["update"] == 20000
    assert restored["phase3_start_update"] == 20000


def test_distributed_checkpoint_uses_explicit_cpu_process_group(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=5)
    checkpoint_group = object()
    observed_groups: list[object] = []

    monkeypatch.setattr(
        "midibrave.zrave_flow_train.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "midibrave.zrave_flow_train.dist.get_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "midibrave.zrave_flow_train.dist.get_rank",
        lambda: 1,
    )

    def gather_object(
        _source,
        _destination,
        *,
        dst: int,
        group,
    ) -> None:
        assert dst == 0
        observed_groups.append(group)

    monkeypatch.setattr(
        "midibrave.zrave_flow_train.dist.gather_object",
        gather_object,
    )

    save_flow_checkpoint(
        tmp_path / "distributed.pt",
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=5000,
        contract={
            "world_size": 2,
            "batch_per_gpu": 2,
            "config_sha256": "a" * 64,
            "pack_index_sha256": "b" * 64,
            "statistics_sha256": "c" * 64,
            "pitch_conditioning": False,
        },
        checkpoint_process_group=checkpoint_group,
    )

    assert observed_groups == [checkpoint_group, checkpoint_group]


def test_checkpoint_process_group_uses_gloo(monkeypatch) -> None:
    checkpoint_group = object()
    observed: dict[str, object] = {}

    def new_group(*, backend: str):
        observed["backend"] = backend
        return checkpoint_group

    monkeypatch.setattr(flow_train.dist, "new_group", new_group)

    assert flow_train._new_checkpoint_process_group(1) is None
    assert (
        flow_train._new_checkpoint_process_group(8)
        is checkpoint_group
    )
    assert observed == {"backend": "gloo"}


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


def test_exposure_batches_honor_resumed_phase3_start() -> None:
    generator = torch.Generator().manual_seed(3)

    assert use_exposure_batch(
        20000,
        100000,
        0.8,
        1.0,
        generator,
        start_update=20000,
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


def test_exploration_exposure_depth_ramps_from_zero_to_three() -> None:
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG).exploration

    assert flow_train.allowed_exploration_exposure_depth(999, config) == 0
    assert flow_train.allowed_exploration_exposure_depth(1000, config) == 1
    assert flow_train.allowed_exploration_exposure_depth(3000, config) == 2
    assert flow_train.allowed_exploration_exposure_depth(5000, config) == 3

    generator = torch.Generator().manual_seed(4)
    for update in (999, 1000, 3000, 5000):
        sampled = flow_train.exploration_exposure_depth(
            update,
            config,
            generator,
        )
        assert 0 <= sampled <= flow_train.allowed_exploration_exposure_depth(
            update,
            config,
        )


def test_depth_three_exposure_keeps_final_sixteen_real_targets(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_sample(
        model,
        statistics,
        history,
        **kwargs,
    ) -> torch.Tensor:
        del model, statistics
        calls.append(kwargs)
        value = float(len(calls) * 100)
        return torch.full(
            (history.shape[0], 64, history.shape[2]),
            value,
        )

    monkeypatch.setattr(
        flow_train,
        "sample_pure_flow_block",
        fake_sample,
    )
    batch = _numbered_flow_batch()

    prepared = flow_train._prepare_exploration_exposure_batch(
        _pure_model(),
        batch,
        generation_seed=7,
        block_index=0,
        depth=3,
        stride_frames=16,
        exploration=1.0,
        schedule_offset_frames=0,
    )

    assert prepared.future_mask.sum(dim=1).tolist() == [16, 16]
    torch.testing.assert_close(
        prepared.future[:, :16],
        batch.future[:, 48:64],
    )
    assert torch.all(prepared.history[:, :16] == 200.0)
    assert torch.all(prepared.history[:, 16:] == 300.0)
    assert [call["schedule_offset_frames"] for call in calls] == [
        0,
        16,
        32,
    ]
    assert all(call["visible_history_frames"] == 8 for call in calls)


def test_weight_only_initializer_loads_v1_model_strictly(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    source = _pure_model()
    checkpoint = tmp_path / "step-085000.pt"
    torch.save(
        {
            "format": 1,
            "architecture": "zrave_pure_flow_transformer_v1",
            "model": source.state_dict(),
            "update": 85000,
            "contract": {
                "pack_index_sha256": "b" * 64,
                "statistics_sha256": "c" * 64,
                "latent_dim": 16,
                "context_frames": 32,
                "future_frames": 64,
                "pitch_conditioning": False,
            },
        },
        checkpoint,
    )
    target = _pure_model()
    for parameter in target.parameters():
        parameter.data.zero_()
    expected_contract = {
        "pack_index_sha256": "b" * 64,
        "statistics_sha256": "c" * 64,
        "latent_dim": 16,
        "context_frames": 32,
        "future_frames": 64,
        "pitch_conditioning": False,
        "exploration_enabled": True,
    }

    metadata = flow_train.load_flow_initial_weights(
        checkpoint,
        model=target,
        expected_contract=expected_contract,
    )

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert metadata["source_update"] == 85000
    assert metadata["source_architecture"] == (
        "zrave_pure_flow_transformer_v1"
    )
    assert metadata["checkpoint_sha256"] == flow_train._sha256_file(
        checkpoint
    )

    with pytest.raises(ValueError, match="pack_index_sha256"):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=target,
            expected_contract={
                **expected_contract,
                "pack_index_sha256": "d" * 64,
            },
        )


def test_training_start_uses_initializer_without_resume_state(
    monkeypatch,
) -> None:
    metadata = {
        "source_update": 85000,
        "source_architecture": "zrave_pure_flow_transformer_v1",
        "checkpoint_sha256": "f" * 64,
    }
    observed: dict[str, object] = {}

    def fake_initialize(path, *, model, expected_contract):
        observed["path"] = path
        observed["model"] = model
        observed["contract"] = expected_contract
        return metadata

    monkeypatch.setattr(
        flow_train,
        "load_flow_initial_weights",
        fake_initialize,
    )
    model = nn.Linear(1, 1)
    contract = {"exploration_enabled": True}

    state = flow_train._load_flow_training_start(
        SimpleNamespace(
            resume=None,
            initialize_from="step-085000.pt",
        ),
        {
            "training_model": model,
            "contract": contract,
        },
    )

    assert state == {
        "update": 0,
        "restored_phase3_start": None,
        "latest_gate_hash": None,
        "consecutive_gate_passes": 0,
        "initialization": metadata,
    }
    assert observed == {
        "path": "step-085000.pt",
        "model": model,
        "contract": contract,
    }


def test_checkpoint_persists_initialization_lineage(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=5)
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG)
    contract = build_flow_checkpoint_contract(
        config=config,
        world_size=1,
        batch_per_gpu=2,
        maximum_updates=20000,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )
    initialization = {
        "source_update": 85000,
        "source_architecture": "zrave_pure_flow_transformer_v1",
        "checkpoint_sha256": "f" * 64,
    }
    checkpoint = tmp_path / "exploration.pt"

    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=1,
        contract=contract,
        initialization=initialization,
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    restored = load_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        expected_contract=contract,
    )

    assert payload["architecture"] == "zrave_pure_flow_transformer_v2"
    assert payload["initialization"] == initialization
    assert restored["initialization"] == initialization


def test_exploration_update_logs_control_telemetry() -> None:
    class _Writer:
        def __init__(self) -> None:
            self.values: dict[str, float] = {}

        def add_scalar(self, name, value, update) -> None:
            assert update == 7
            self.values[name] = float(value)

    writer = _Writer()
    result = flow_train.FlowUpdateResult(
        loss=1.0,
        components={"flow": 0.8, "temporal": 0.2},
        pitch_weight=0.0,
        gradient_norm=0.5,
        amp_scale=1024.0,
        global_valid_frames=128,
        duration_seconds=1.0,
        pitch_metrics={},
        exposure=True,
        exposure_depth=3,
        exploration=2.0 / 3.0,
        visible_history_frames=16,
        schedule_offset_frames=64,
        applied=True,
    )

    flow_train._log_update(writer, result, 7)

    assert writer.values["train/temporal"] == 0.2
    assert writer.values["train/exploration"] == pytest.approx(2.0 / 3.0)
    assert writer.values["train/visible_history_frames"] == 16.0
    assert writer.values["train/schedule_offset_frames"] == 64.0
    assert writer.values["health/exposure_depth"] == 3.0


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
