from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import midibrave.zrave_flow_train as flow_train
from midibrave.zrave_flow_config import (
    ZraveFlowConfig,
    data_selection_sha256,
)
from midibrave.zrave_flow_loss import PitchWeightController
from midibrave.zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
)
from midibrave.zrave_pitch_probe import PitchProbeOutput
from midibrave.zrave_flow_sampler import FlowBatch, GpuFlowSampler
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
PURE_CONFIG = ROOT / "configs" / "zrave" / "octopus_pure_flow_poc.yaml"
EXPLORATION_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_serum128_exploration_flow.yaml"
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
        require_full_future: bool = False,
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
        future_mask = (torch.arange(64).unsqueeze(0) < valid_frames).expand(
            batch_size, -1
        )
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

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator_state": self.generator.get_state().clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])


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


def _tiny_config_with_categories(
    categories: tuple[str, ...],
) -> ZraveFlowConfig:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    sources = tuple(
        replace(
            source,
            allowed_categories=(
                categories if index == 0 else source.allowed_categories
            ),
        )
        for index, source in enumerate(config.data.sources)
    )
    return replace(
        config,
        data=replace(config.data, sources=sources),
        model=replace(
            config.model,
            profile="tiny",
            d_model=128,
            context_layers=2,
            future_layers=4,
            heads=4,
            feedforward_dim=512,
        ),
        train=replace(
            config.train,
            checkpoint_every=1000,
            validation_every=1000,
            short_future_updates=1000,
        ),
    )


def _numbered_flow_batch(batch_size: int = 2) -> FlowBatch:
    history = (
        torch.arange(32)
        .view(1, 32, 1)
        .repeat(
            batch_size,
            1,
            16,
        )
        .float()
    )
    future = (
        torch.arange(32, 96)
        .view(1, 64, 1)
        .repeat(
            batch_size,
            1,
            16,
        )
        .float()
    )
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
            "--expected-initializer-sha256",
            "f" * 64,
            "--expected-initializer-update",
            "85000",
        ]
    )

    assert arguments.initialize_from == "step-085000.pt"
    assert arguments.resume is None
    assert arguments.expected_initializer_sha256 == "f" * 64
    assert arguments.expected_initializer_update == 85000


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
    assert contract["model_profile"] == "standard"
    assert "data_selection_sha256" not in contract
    assert tuple(contract[name] for name in flow_train._MODEL_PROFILE_FIELDS) == (
        384,
        4,
        8,
        8,
        1536,
    )


def test_checkpoint_contract_records_small_profile_dimensions() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    model = replace(
        config.model,
        profile="small",
        d_model=256,
        context_layers=3,
        future_layers=6,
        heads=8,
        feedforward_dim=1024,
    )
    contract = build_flow_checkpoint_contract(
        config=replace(
            config,
            model=model,
            train=replace(
                config.train,
                checkpoint_every=1000,
                validation_every=1000,
                short_future_updates=1000,
            ),
        ),
        world_size=1,
        batch_per_gpu=16,
        maximum_updates=20000,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )

    assert contract["model_profile"] == "small"
    assert tuple(contract[name] for name in flow_train._MODEL_PROFILE_FIELDS) == (
        256,
        3,
        6,
        8,
        1024,
    )


def test_tiny_data_selection_hash_is_canonical_and_category_sensitive() -> None:
    first = _tiny_config_with_categories(("Pad", "Bass"))
    reordered = _tiny_config_with_categories(("Bass", "Pad", "Pad"))
    changed = _tiny_config_with_categories(("Lead",))

    first_hash = data_selection_sha256(first)

    assert first_hash == data_selection_sha256(reordered)
    assert first_hash != data_selection_sha256(changed)
    assert first_hash is not None
    assert len(first_hash) == 64


def test_data_selection_hash_tracks_allowlist_contents_not_path(
    tmp_path: Path,
) -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    first_path = tmp_path / "first.ids.txt"
    second_path = tmp_path / "second.ids.txt"
    first_path.write_text("serum:a\nserum:b\n", encoding="utf-8")
    second_path.write_text("serum:a\nserum:b\n", encoding="utf-8")

    def selected(path: Path) -> ZraveFlowConfig:
        sources = (
            replace(
                config.data.sources[0],
                preset_allowlist=str(path),
            ),
            *config.data.sources[1:],
        )
        return replace(config, data=replace(config.data, sources=sources))

    first_hash = data_selection_sha256(selected(first_path))
    assert first_hash == data_selection_sha256(selected(second_path))

    second_path.write_text("serum:a\nserum:c\n", encoding="utf-8")
    assert first_hash != data_selection_sha256(selected(second_path))


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


def test_pure_validation_snapshot_restores_sampler_rng_and_mode() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    sampler = _PureFlowSampler(seed=91)
    model = _pure_model()
    model.train()
    sampler_state = sampler.state_dict()["generator_state"].clone()
    torch_state = torch.get_rng_state().clone()

    validation, diversity = flow_train._validation_snapshot(
        model=model,
        sampler=sampler,
        pitch_probe=None,
        statistics=_unit_statistics(),
        config=config,
        batch_per_gpu=2,
        device=torch.device("cpu"),
    )

    assert set(validation) == {"total", "flow", "boundary", "statistics"}
    assert all(torch.isfinite(torch.tensor(value)) for value in validation.values())
    assert diversity["normalized_clean_std"] > 0.0
    assert torch.equal(
        sampler.state_dict()["generator_state"],
        sampler_state,
    )
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert model.training

    second_validation, second_diversity = flow_train._validation_snapshot(
        model=model,
        sampler=sampler,
        pitch_probe=None,
        statistics=_unit_statistics(),
        config=config,
        batch_per_gpu=2,
        device=torch.device("cpu"),
    )
    assert second_validation == validation
    assert second_diversity == diversity
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_validation_sampler_shares_resident_pack_but_not_rng_state() -> None:
    train = GpuFlowSampler(
        latents=torch.zeros(3, 96, 16),
        lengths=torch.full((3,), 96),
        active_frames=torch.full((3,), 96),
        notes=torch.tensor([36, 62, 82]),
        velocities=torch.tensor([54, 108, 54]),
        split_codes=torch.tensor([0, 1, 2]),
        source_codes=torch.zeros(3, dtype=torch.long),
        category_codes=torch.zeros(3, dtype=torch.long),
        maximum_future_frames=torch.full((3,), 64),
        pitch_pairs=torch.full((3,), -1),
        source_weights=(1.0,),
        context_frames=32,
        future_frames=64,
        wander_delays=(16, 32, 48),
        pitch_transition_fraction=0.0,
        seed=10,
        device="cpu",
        allowed_split_code=0,
    )

    validation = flow_train._shared_validation_sampler(train, seed=20)

    assert validation.latents.data_ptr() == train.latents.data_ptr()
    assert validation.allowed_split_code == 1
    assert train.allowed_split_code == 0
    assert not torch.equal(
        validation.generator.get_state(),
        train.generator.get_state(),
    )
    validation.transition_credit = 0.75
    assert train.transition_credit == 0.0


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
    assert result.learning_rate == pytest.approx(
        config.optimizer.learning_rate / config.train.warmup_updates
    )
    assert result.diagnostics["batch/future_valid_fraction"] == 0.5
    assert "target/latent_frame_norm_p99" in result.diagnostics
    assert "target/motion_normalized_delta_rms_p95" in result.diagnostics
    assert "model/predicted_velocity_std" in result.diagnostics
    assert not any(name.startswith("midi/") for name in result.diagnostics)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_flow_batch_diagnostics_are_masked_and_cover_segment_midi() -> None:
    future = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.1], [900.0, 900.0], [900.0, 900.0]],
            [[1.0, 1.0], [1.0, 1.0], [3.0, 1.0], [900.0, 900.0]],
        ]
    )
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    prediction = torch.full_like(future, 2.0)
    prediction.masked_fill_(~mask.unsqueeze(-1), 700.0)
    batch = FlowBatch(
        history=torch.zeros(2, 2, 2),
        future=future,
        future_mask=mask,
        midi_note=torch.tensor([36, 62]),
        source_code=torch.zeros(2, dtype=torch.long),
        category_code=torch.zeros(2, dtype=torch.long),
        wander_delay_frames=torch.full((2,), 32, dtype=torch.long),
        history_midi_note=torch.tensor([36, 62]),
        pitch_transition_mask=torch.tensor([True, False]),
        velocity=torch.tensor([54, 108]),
        history_velocity=torch.tensor([54, 108]),
        segment_division=torch.tensor([2, 4]),
        segment_index=torch.tensor([1, 3]),
        future_midi_sequence=torch.tensor([[36, 36, 109, 109], [62, 62, 82, 109]]),
        future_velocity_sequence=torch.tensor(
            [[54, 54, 127, 127], [108, 108, 108, 127]]
        ),
        midi_event_frame=torch.tensor([1, -1]),
    )
    statistics = FlowStatistics(
        mean=torch.zeros(2),
        latent_std=torch.ones(2),
        delta_std=torch.tensor([1.0, 2.0]),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )

    metrics = flow_train._flow_batch_diagnostics(
        batch=batch,
        predicted_velocity=prediction,
        estimated_future=future,
        statistics=statistics,
        pitch_conditioning=True,
        midi_sequence_conditioning=True,
        note_min=21,
        note_max=109,
        pitch_present=torch.tensor([True, False]),
    )

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["batch/future_valid_fraction"] == pytest.approx(5.0 / 8.0)
    assert metrics["batch/effective_future_frames_mean"] == 2.5
    assert metrics["target/latent_mean"] == pytest.approx(0.82)
    assert metrics["target/near_static_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["target/normalized_coordinate_near_zero_fraction"] == pytest.approx(
        4.0 / 10.0
    )
    # This is a normalized latent-coordinate distribution proxy, not a claim
    # about audible event density or acoustic sparsity.
    assert metrics["target/normalized_coordinate_near_zero_threshold"] == 0.1
    assert "target/channel_observed_reference_std_ratio_p10" in metrics
    assert "target/channel_observed_reference_std_ratio_p50" in metrics
    assert "target/channel_observed_reference_std_ratio_p90" in metrics
    assert metrics["model/predicted_velocity_mean"] == 2.0
    assert metrics["model/predicted_velocity_std"] == 0.0
    assert metrics["estimate/latent_mean"] == pytest.approx(0.82)
    assert metrics["estimate/near_static_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["segment/division_2_fraction"] == 0.5
    assert metrics["segment/division_4_fraction"] == 0.5
    assert metrics["segment/index_1_fraction"] == 0.5
    assert metrics["segment/index_3_fraction"] == 0.5
    assert metrics["segment/division_2_index_1_fraction"] == 0.5
    assert metrics["segment/division_4_index_3_fraction"] == 0.5
    assert metrics["coverage/source_code_0_fraction"] == 1.0
    assert metrics["coverage/category_code_0_fraction"] == 1.0
    assert metrics["midi/transition_batch_fraction"] == 0.5
    assert metrics["midi/condition_present_fraction"] == 0.5
    assert metrics["midi/note_change_frame_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["midi/event_present_fraction"] == 0.5
    assert metrics["midi/event_frame_p50"] == 1.0
    assert metrics["midi/event_frame_p95"] == 1.0
    assert metrics["midi/note_036_fraction"] == pytest.approx(2.0 / 5.0)
    assert metrics["midi/note_062_fraction"] == pytest.approx(2.0 / 5.0)
    assert metrics["midi/note_082_fraction"] == pytest.approx(1.0 / 5.0)
    assert metrics["midi/velocity_048_063_fraction"] == pytest.approx(2.0 / 5.0)
    assert metrics["midi/velocity_096_111_fraction"] == pytest.approx(3.0 / 5.0)


def test_flow_batch_diagnostics_handle_sparse_mask_and_scalar_midi() -> None:
    mask = torch.tensor([[True, False, True, False], [True, True, True, False]])
    batch = FlowBatch(
        history=torch.zeros(2, 2, 2),
        future=torch.zeros(2, 4, 2),
        future_mask=mask,
        midi_note=torch.tensor([36, 62]),
        source_code=torch.tensor([0, 2]),
        category_code=torch.tensor([3, 3]),
        wander_delay_frames=torch.full((2,), 32, dtype=torch.long),
        history_midi_note=torch.tensor([36, 62]),
        pitch_transition_mask=torch.tensor([False, True]),
        velocity=torch.tensor([54, 108]),
        history_velocity=torch.tensor([54, 108]),
    )
    statistics = FlowStatistics(
        mean=torch.zeros(2),
        latent_std=torch.ones(2),
        delta_std=torch.ones(2),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )

    metrics = flow_train._flow_batch_diagnostics(
        batch=batch,
        predicted_velocity=torch.zeros_like(batch.future),
        estimated_future=torch.zeros_like(batch.future),
        statistics=statistics,
        pitch_conditioning=True,
        midi_sequence_conditioning=False,
        note_min=21,
        note_max=109,
        pitch_present=torch.ones(2, dtype=torch.bool),
    )

    assert metrics["midi/note_036_fraction"] == pytest.approx(2.0 / 5.0)
    assert metrics["midi/note_062_fraction"] == pytest.approx(3.0 / 5.0)
    assert metrics["midi/velocity_048_063_fraction"] == pytest.approx(2.0 / 5.0)
    assert metrics["midi/velocity_096_111_fraction"] == pytest.approx(3.0 / 5.0)
    assert metrics["coverage/source_code_0_fraction"] == 0.5
    assert metrics["coverage/source_code_1_fraction"] == 0.0
    assert metrics["coverage/source_code_2_fraction"] == 0.5
    assert metrics["target/motion_normalized_delta_rms_mean"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_flow_batch_diagnostics_allow_all_null_notes_and_no_deltas() -> None:
    mask = torch.tensor([[True, False, True, False]])
    batch = FlowBatch(
        history=torch.zeros(1, 2, 2),
        future=torch.zeros(1, 4, 2),
        future_mask=mask,
        midi_note=torch.tensor([-1]),
        source_code=torch.tensor([0]),
        category_code=torch.tensor([0]),
        wander_delay_frames=torch.tensor([32]),
        history_midi_note=torch.tensor([-1]),
        pitch_transition_mask=torch.tensor([False]),
        velocity=torch.tensor([0]),
        history_velocity=torch.tensor([0]),
    )
    statistics = FlowStatistics(
        mean=torch.zeros(2),
        latent_std=torch.ones(2),
        delta_std=torch.ones(2),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )

    metrics = flow_train._flow_batch_diagnostics(
        batch=batch,
        predicted_velocity=torch.zeros_like(batch.future),
        estimated_future=torch.zeros_like(batch.future),
        statistics=statistics,
        pitch_conditioning=True,
        midi_sequence_conditioning=False,
        note_min=21,
        note_max=109,
        pitch_present=torch.tensor([False]),
    )

    assert metrics["midi/note_null_fraction"] == 1.0
    assert metrics["midi/note_sounding_fraction"] == 0.0
    assert metrics["midi/note_mean"] == 0.0
    assert metrics["midi/note_std"] == 0.0
    assert metrics["target/motion_normalized_delta_rms_p95"] == 0.0
    assert metrics["target/near_static_fraction"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_pitch_diagnostics_never_labels_a_cross_event_window() -> None:
    class _Probe(nn.Module):
        note_min = 21

        def __init__(self) -> None:
            super().__init__()
            self.windows = torch.empty(0)

        def forward(self, latents: torch.Tensor) -> PitchProbeOutput:
            self.windows = latents.detach().clone()
            logits = torch.full((latents.shape[0], 89), -100.0)
            logits[:, 36 - self.note_min] = 100.0
            return PitchProbeOutput(
                logits=logits,
                expected_midi=torch.full((latents.shape[0],), 36.0),
            )

    probe = _Probe()
    statistics = FlowStatistics(
        mean=torch.zeros(2),
        latent_std=torch.ones(2),
        delta_std=torch.ones(2),
        latent_norm_p01=torch.tensor(0.1),
        latent_norm_p99=torch.tensor(100.0),
    )
    mask = torch.ones(1, 32, dtype=torch.bool)
    pair = SimpleNamespace(
        noisy_future=torch.cat(
            (torch.ones(1, 20, 2), torch.full((1, 12, 2), 9.0)),
            dim=1,
        ),
        flow_time=torch.ones(1),
    )

    metrics = flow_train._pitch_diagnostics(
        pair,
        torch.zeros(1, 32, 2),
        mask,
        torch.tensor([[36] * 20 + [62] * 12]),
        torch.tensor([True]),
        probe,
        statistics,
    )

    assert torch.all(probe.windows == 1.0)
    assert metrics["pitch_transition_accuracy"] == 1.0


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


def test_resume_rejects_changed_preset_allowlist_contents(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=5)
    config = ZraveFlowConfig.load(PURE_CONFIG)
    allowlist = tmp_path / "bucket.ids.txt"
    allowlist.write_text("serum:a\n", encoding="utf-8")
    sources = (
        replace(
            config.data.sources[0],
            preset_allowlist=str(allowlist),
        ),
        *config.data.sources[1:],
    )
    config = replace(config, data=replace(config.data, sources=sources))

    def contract() -> dict[str, object]:
        return build_flow_checkpoint_contract(
            config=config,
            world_size=1,
            batch_per_gpu=2,
            maximum_updates=100,
            config_sha256="a" * 64,
            pack_index_sha256="b" * 64,
            statistics_sha256="c" * 64,
        )

    original_contract = contract()
    checkpoint = tmp_path / "allowlisted.pt"
    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=1,
        contract=original_contract,
    )
    allowlist.write_text("serum:b\n", encoding="utf-8")

    changed_contract = contract()
    assert (
        changed_contract["data_selection_sha256"]
        != (original_contract["data_selection_sha256"])
    )
    with pytest.raises(ValueError, match="data_selection_sha256"):
        load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=None,
            expected_contract=changed_contract,
        )


def test_legacy_pure_checkpoint_without_new_false_flags_still_resumes(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=6)
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
    checkpoint = tmp_path / "legacy-new-flags.pt"
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["contract"].pop("midi_sequence_conditioning")
    payload["contract"].pop("segment_sampling")
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


def test_legacy_checkpoint_without_profile_fields_is_standard_only(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = _TinySampler(seed=8)
    config = ZraveFlowConfig.load(PURE_CONFIG)
    standard = build_flow_checkpoint_contract(
        config=config,
        world_size=1,
        batch_per_gpu=2,
        maximum_updates=100,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )
    checkpoint = tmp_path / "legacy-standard.pt"
    save_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        update=1,
        contract=standard,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name in flow_train._MODEL_PROFILE_CONTRACT_FIELDS:
        payload["contract"].pop(name)
    torch.save(payload, checkpoint)

    restored = load_flow_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        sampler=sampler,
        pitch_weight_controller=None,
        expected_contract=standard,
    )
    assert restored["update"] == 1

    small_model = replace(
        config.model,
        profile="small",
        d_model=256,
        context_layers=3,
        future_layers=6,
        heads=8,
        feedforward_dim=1024,
    )
    small = build_flow_checkpoint_contract(
        config=replace(
            config,
            model=small_model,
            train=replace(
                config.train,
                checkpoint_every=1000,
                validation_every=1000,
                short_future_updates=1000,
            ),
        ),
        world_size=1,
        batch_per_gpu=2,
        maximum_updates=100,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        statistics_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="model profile mismatch"):
        load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=None,
            expected_contract=small,
        )


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
    assert flow_train._new_checkpoint_process_group(8) is checkpoint_group
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
    history = torch.arange(32).view(1, 32, 1).repeat(1, 1, 16).float()
    generated = torch.arange(32, 64).view(1, 32, 1).repeat(1, 1, 16).float()

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
        assert (
            0
            <= sampled
            <= flow_train.allowed_exploration_exposure_depth(
                update,
                config,
            )
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
        expected_initializer_sha256=flow_train._sha256_file(checkpoint),
        expected_initializer_update=85000,
    )

    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert metadata["source_update"] == 85000
    assert metadata["source_architecture"] == ("zrave_pure_flow_transformer_v1")
    assert metadata["checkpoint_sha256"] == flow_train._sha256_file(checkpoint)

    with pytest.raises(ValueError, match="initializer SHA-256 mismatch"):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=target,
            expected_contract=expected_contract,
            expected_initializer_sha256="0" * 64,
            expected_initializer_update=85000,
        )
    with pytest.raises(ValueError, match="initializer update mismatch"):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=target,
            expected_contract=expected_contract,
            expected_initializer_sha256=flow_train._sha256_file(checkpoint),
            expected_initializer_update=84999,
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


def test_midi_sequence_initializer_loads_shared_pure_backbone(
    tmp_path: Path,
) -> None:
    source = _pure_model()
    checkpoint = tmp_path / "pure.pt"
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
    target = ZraveFlowTransformer(
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
        pitch_conditioning=True,
        midi_sequence_conditioning=True,
    )

    metadata = flow_train.load_flow_initial_weights(
        checkpoint,
        model=target,
        expected_contract={
            "pack_index_sha256": "b" * 64,
            "statistics_sha256": "c" * 64,
            "latent_dim": 16,
            "context_frames": 32,
            "future_frames": 64,
            "pitch_conditioning": True,
            "midi_sequence_conditioning": True,
            "segment_sampling": True,
        },
    )

    for name, value in source.state_dict().items():
        if (
            name in target.state_dict()
            and target.state_dict()[name].shape == value.shape
        ):
            torch.testing.assert_close(target.state_dict()[name], value)
    assert metadata["source_update"] == 85000
    assert target.midi_sequence_conditioner is not None
    assert torch.count_nonzero(target.midi_sequence_conditioner.projection.weight) == 0


def test_weight_initializer_rejects_profile_and_shared_shape_mismatch(
    tmp_path: Path,
) -> None:
    source = _pure_model()
    checkpoint = tmp_path / "pure-standard.pt"
    legacy_standard = {
        "pack_index_sha256": "b" * 64,
        "statistics_sha256": "c" * 64,
        "latent_dim": 16,
        "context_frames": 32,
        "future_frames": 64,
        "pitch_conditioning": False,
    }
    torch.save(
        {
            "format": 1,
            "architecture": "zrave_pure_flow_transformer_v1",
            "model": source.state_dict(),
            "update": 85000,
            "contract": legacy_standard,
        },
        checkpoint,
    )
    small_contract = {
        **legacy_standard,
        "model_profile": "small",
        "d_model": 256,
        "context_layers": 3,
        "future_layers": 6,
        "heads": 8,
        "feedforward_dim": 1024,
    }
    with pytest.raises(ValueError, match="initializer model profile mismatch"):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=source,
            expected_contract=small_contract,
        )

    wider = ZraveFlowTransformer(
        statistics=_unit_statistics(),
        latent_dim=16,
        context_frames=32,
        future_frames=64,
        d_model=64,
        context_layers=1,
        future_layers=1,
        heads=4,
        feedforward_dim=64,
        dropout=0.0,
        pitch_conditioning=True,
        midi_sequence_conditioning=True,
    )
    with pytest.raises(ValueError, match="shared parameter shape mismatch"):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=wider,
            expected_contract={
                **legacy_standard,
                "pitch_conditioning": True,
                "midi_sequence_conditioning": True,
                "segment_sampling": True,
            },
        )


def test_tiny_initializer_rejects_different_category_selection(
    tmp_path: Path,
) -> None:
    source_config = _tiny_config_with_categories(("Bass",))
    target_config = _tiny_config_with_categories(("Lead",))

    def contract(config: ZraveFlowConfig) -> dict[str, object]:
        return build_flow_checkpoint_contract(
            config=config,
            world_size=1,
            batch_per_gpu=2,
            maximum_updates=10000,
            config_sha256="a" * 64,
            pack_index_sha256="b" * 64,
            statistics_sha256="c" * 64,
        )

    source_contract = contract(source_config)
    target_contract = contract(target_config)
    checkpoint = tmp_path / "tiny-bass.pt"
    model = _pure_model()
    torch.save(
        {
            "format": 1,
            "architecture": "zrave_pure_flow_transformer_v1",
            "model": model.state_dict(),
            "update": 10000,
            "contract": source_contract,
        },
        checkpoint,
    )

    assert (
        source_contract["data_selection_sha256"]
        != (target_contract["data_selection_sha256"])
    )
    with pytest.raises(
        ValueError,
        match="initializer data_selection_sha256 mismatch",
    ):
        flow_train.load_flow_initial_weights(
            checkpoint,
            model=model,
            expected_contract=target_contract,
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

    def fake_initialize(
        path,
        *,
        model,
        expected_contract,
        expected_initializer_sha256,
        expected_initializer_update,
    ):
        observed["path"] = path
        observed["model"] = model
        observed["contract"] = expected_contract
        observed["expected_initializer_sha256"] = expected_initializer_sha256
        observed["expected_initializer_update"] = expected_initializer_update
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
            expected_initializer_sha256="f" * 64,
            expected_initializer_update=85000,
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
        "expected_initializer_sha256": "f" * 64,
        "expected_initializer_update": 85000,
    }


def test_benchmark_forwards_expected_initializer_lineage(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_initialize(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return {
            "source_update": 10000,
            "source_architecture": "zrave_pure_flow_transformer_v2",
            "checkpoint_sha256": "f" * 64,
        }

    monkeypatch.setattr(flow_train, "load_flow_initial_weights", fake_initialize)
    monkeypatch.setattr(
        flow_train,
        "_build_runtime",
        lambda _args: {
            "config": SimpleNamespace(
                segment_sampling=SimpleNamespace(enabled=False),
                model=SimpleNamespace(midi_sequence_conditioning=False),
            ),
            "rank": 0,
            "device": torch.device("cpu"),
            "training_model": nn.Linear(1, 1),
            "contract": {},
        },
    )
    args = SimpleNamespace(
        initialize_from="step-010000.pt",
        expected_initializer_sha256="f" * 64,
        expected_initializer_update=10000,
        benchmark_warmup=-1,
        benchmark_updates=1,
        benchmark_exposure_updates=0,
    )

    with pytest.raises(ValueError, match="invalid benchmark"):
        flow_train._benchmark(args)

    assert observed["expected_initializer_sha256"] == "f" * 64
    assert observed["expected_initializer_update"] == 10000


def test_training_resume_forwards_expected_initializer_lineage(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_resume(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return {
            "update": 1000,
            "phase3_start_update": None,
            "latest_gate_report_sha256": None,
            "consecutive_gate_passes": 0,
            "initialization": {
                "source_update": 10000,
                "source_architecture": "zrave_pure_flow_transformer_v1",
                "checkpoint_sha256": "f" * 64,
            },
        }

    monkeypatch.setattr(flow_train, "load_flow_checkpoint", fake_resume)
    runtime = {
        "training_model": object(),
        "optimizer": object(),
        "scaler": object(),
        "train_sampler": object(),
        "controller": object(),
        "contract": {"midi_sequence_conditioning": True},
    }
    state = flow_train._load_flow_training_start(
        SimpleNamespace(
            resume="step-001000.pt",
            initialize_from=None,
            expected_initializer_sha256="f" * 64,
            expected_initializer_update=10000,
        ),
        runtime,
    )

    assert state["update"] == 1000
    assert observed["path"] == "step-001000.pt"
    assert observed["expected_initializer_sha256"] == "f" * 64
    assert observed["expected_initializer_update"] == 10000

    with pytest.raises(
        ValueError,
        match="requires expected initializer SHA-256 and update",
    ):
        flow_train._load_flow_training_start(
            SimpleNamespace(
                resume="step-001000.pt",
                initialize_from=None,
            ),
            runtime,
        )


def test_training_start_allows_nonstandard_pure_scratch_resume(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_resume(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return {
            "update": 1000,
            "phase3_start_update": None,
            "latest_gate_report_sha256": None,
            "consecutive_gate_passes": 0,
            "initialization": None,
        }

    monkeypatch.setattr(flow_train, "load_flow_checkpoint", fake_resume)
    state = flow_train._load_flow_training_start(
        SimpleNamespace(
            resume="step-001000.pt",
            initialize_from=None,
            expected_initializer_sha256=None,
            expected_initializer_update=None,
        ),
        {
            "training_model": object(),
            "optimizer": object(),
            "scaler": object(),
            "train_sampler": object(),
            "controller": None,
            "contract": {
                "model_profile": "tiny",
                "pitch_conditioning": False,
                "segment_sampling": True,
            },
        },
    )

    assert state["update"] == 1000
    assert state["initialization"] is None
    assert observed["expected_initializer_sha256"] is None
    assert observed["expected_initializer_update"] is None


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
        expected_initializer_sha256="f" * 64,
        expected_initializer_update=85000,
    )

    assert payload["architecture"] == "zrave_pure_flow_transformer_v2"
    assert payload["initialization"] == initialization
    assert restored["initialization"] == initialization

    with pytest.raises(ValueError, match="initializer SHA-256 mismatch"):
        load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=None,
            expected_contract=contract,
            expected_initializer_sha256="e" * 64,
            expected_initializer_update=85000,
        )
    with pytest.raises(ValueError, match="initializer update mismatch"):
        load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=None,
            expected_contract=contract,
            expected_initializer_sha256="f" * 64,
            expected_initializer_update=10000,
        )

    payload.pop("initialization")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="lacks initialization lineage"):
        load_flow_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            sampler=sampler,
            pitch_weight_controller=None,
            expected_contract=contract,
            expected_initializer_sha256="f" * 64,
            expected_initializer_update=85000,
        )


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
        learning_rate=1.0e-4,
        diagnostics={"batch/future_valid_fraction": 0.75},
    )

    flow_train._log_update(writer, result, 7)

    assert writer.values["train/temporal"] == 0.2
    assert writer.values["train/exploration"] == pytest.approx(2.0 / 3.0)
    assert writer.values["train/visible_history_frames"] == 16.0
    assert writer.values["train/schedule_offset_frames"] == 64.0
    assert writer.values["train/learning_rate"] == 1.0e-4
    assert writer.values["diagnostics/batch/future_valid_fraction"] == 0.75
    assert writer.values["health/exposure_depth"] == 3.0


def test_metrics_jsonl_is_finite_append_only_and_resume_safe(
    tmp_path: Path,
) -> None:
    result = flow_train.FlowUpdateResult(
        loss=1.0,
        components={"flow": 0.8, "statistics": 0.2},
        pitch_weight=0.0,
        gradient_norm=0.5,
        amp_scale=1024.0,
        global_valid_frames=128,
        duration_seconds=2.0,
        pitch_metrics={},
        exposure=False,
        exposure_depth=0,
        exploration=0.0,
        visible_history_frames=32,
        schedule_offset_frames=0,
        applied=True,
        learning_rate=1.0e-4,
        diagnostics={"batch/future_valid_fraction": 0.5},
    )
    path = tmp_path / "metrics.jsonl"

    tail = flow_train._append_update_metrics(
        path,
        result,
        20,
        previous_update=None,
        wall_time=1_700_000_000.0,
    )
    assert tail == 20
    assert flow_train._last_metrics_update(path) == 20
    payload = json.loads(path.read_text().strip())
    assert payload["schema"] == 1
    assert payload["update"] == 20
    assert payload["timestamp_utc"] == "2023-11-14T22:13:20Z"
    assert payload["scalars"]["train/loss"] == 1.0
    assert payload["scalars"]["train/learning_rate"] == 1.0e-4
    assert payload["scalars"]["health/gradient_norm"] == 0.5
    assert payload["scalars"]["diagnostics/batch/future_valid_fraction"] == 0.5

    with pytest.raises(ValueError, match="strictly newer"):
        flow_train._append_update_metrics(
            path,
            result,
            20,
            previous_update=tail,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        flow_train._append_update_metrics(
            path,
            replace(result, loss=float("nan")),
            21,
            previous_update=tail,
        )


def test_metrics_jsonl_marks_checkpoint_rollback_without_deleting_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"schema":1,"update":980,"scalars":{"train/loss":0.4}}\n'
        '{"schema":1,"update":1020,"scalars":{"train/loss":0.3}}\n',
        encoding="utf-8",
    )

    tail = flow_train._append_metrics_resume_boundary(
        path,
        checkpoint_update=1000,
        previous_tail_update=1020,
        wall_time=1_700_000_000.0,
    )

    assert tail == 1000
    assert flow_train._last_metrics_update(path) == 1000
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["update"] for row in rows] == [980, 1020, 1000]
    assert rows[-1]["event"] == "resume_rollback_to_checkpoint"
    assert rows[-1]["orphaned_tail_update"] == 1020
    assert rows[-1]["scalars"] == {}
    with pytest.raises(ValueError, match="updates are invalid"):
        flow_train._append_metrics_resume_boundary(
            path,
            checkpoint_update=1020,
            previous_tail_update=1020,
        )


def test_metrics_jsonl_appends_finite_validation_and_health_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"

    flow_train._append_metrics_event(
        path,
        event="validation",
        update=1000,
        scalars={"validation/total": 0.25, "diversity/normalized_clean_std": 0.8},
        wall_time=1_700_000_000.0,
    )
    flow_train._append_metrics_event(
        path,
        event="nonfinite_gradient_skip",
        update=1000,
        scalars={"health/nonfinite_gradient_skips": 1},
        wall_time=1_700_000_001.0,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "validation",
        "nonfinite_gradient_skip",
    ]
    assert rows[0]["scalars"]["validation/total"] == 0.25
    assert flow_train._last_metrics_update(path) == 1000
    with pytest.raises(FloatingPointError, match="non-finite"):
        flow_train._append_metrics_event(
            path,
            event="validation",
            update=1001,
            scalars={"validation/total": float("nan")},
        )


def test_final_checkpoint_alias_is_repaired_after_crash_window(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    maximum = checkpoint_root / "step-010000.pt"
    maximum.write_bytes(b"complete maximum checkpoint")

    final = flow_train._ensure_final_checkpoint(checkpoint_root, 10000)
    assert final.read_bytes() == maximum.read_bytes()

    final.write_bytes(b"stale final checkpoint")
    repaired = flow_train._ensure_final_checkpoint(checkpoint_root, 10000)
    assert repaired.read_bytes() == maximum.read_bytes()

    with pytest.raises(RuntimeError, match="maximum-update checkpoint"):
        flow_train._ensure_final_checkpoint(tmp_path / "missing", 10000)


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
    assert continuous["next_parameters"] == resumed["next_parameters"]


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
    assert report["median_valid_latent_frames_per_second"] == 16384.0


def test_segment_benchmark_is_valid_without_legacy_exposure() -> None:
    report = summarize_flow_benchmark(
        batch_per_gpu=8,
        world_size=1,
        durations_seconds=[1.0] * 10,
        global_valid_frames=[1024] * 10,
        exposure_safety_updates=0,
        peak_memory_mib=1000.0,
        total_memory_mib=16000.0,
        nonfinite_updates=0,
        config_sha256="a" * 64,
        pack_index_sha256="b" * 64,
        pitch_checkpoint_sha256=None,
        git_commit="d" * 40,
        require_exposure_safety=False,
    )

    assert report["status"] == "ok"
    assert report["exposure_safety_updates"] == 0
    assert report["exposure_safety_required"] is False


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
