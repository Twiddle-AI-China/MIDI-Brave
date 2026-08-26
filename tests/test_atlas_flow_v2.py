from __future__ import annotations

import inspect
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from midibrave.atlas_flow_atlas import TimbreAtlas, canonical_anchor
from midibrave.atlas_flow_config import AtlasLossConfig, AtlasModelConfig
from midibrave.atlas_flow_loss import atlas_flow_matching_loss
from midibrave.atlas_flow_manifest import (
    PAD_DUPLICATE_DROPS, pad_unique_top50, render_rows, split_presets,
)
from midibrave.atlas_flow_model import (
    AtlasConditionedTrajectoryFlow, FlowBatch, TemporalTimbreEncoder,
)
from midibrave.atlas_flow_evaluate import AUDITION_NOTES, _atomic_json, _json_native
from midibrave.atlas_flow_runtime import AtlasFlowLiveSession, PlaybackLayer, RenderPlan
from midibrave.atlas_flow_train import (
    _batch_contract, _gradient_buckets, _update_nonfinite_counters,
    _is_out_of_memory_error, _numeric_recovery_reason, _recovery_profile, _resume_position,
    _scheduled_learning_rate, _write_tensorboard_record,
)


def test_pad_selection_deduplicates_refills_and_splits_by_preset() -> None:
    selected = pad_unique_top50()
    assert len(selected) == len(set(selected)) == 50
    assert not set(PAD_DUPLICATE_DROPS).intersection(selected)
    assert selected[-5:] == (
        "serum_s091504", "serum_s043574", "serum_s047877",
        "serum_s003030", "serum_s000303",
    )
    split = split_presets(selected)
    assert list(split.values()).count("train") == 45
    assert list(split.values()).count("validation") == 2
    assert list(split.values()).count("test") == 3


def test_render_contract_has_2700_exact_lifecycle_rows() -> None:
    rows = render_rows()
    assert len(rows) == 2_700
    assert len({(row["preset_index"], row["config_id"]) for row in rows}) == 2_700
    assert {row["midi_velocity"] for row in rows} == {127}
    assert {row["note_start_seconds"] for row in rows} == {0.1}
    assert {row["note_off_seconds"] for row in rows} == {2.6}
    assert {row["render_duration_seconds"] for row in rows} == {5.0}


def test_temporal_encoder_maps_96d_hop512_to_128d_hop2048() -> None:
    model = TemporalTimbreEncoder()
    result = model(torch.randn(2, 96, 432))
    assert result.shape[0:2] == (2, 128)
    assert 107 <= result.shape[-1] <= 109
    result.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flow_has_no_midi_input_and_all_four_losses_backpropagate() -> None:
    config = AtlasModelConfig()
    model = AtlasConditionedTrajectoryFlow(config)
    assert "midi" not in inspect.signature(model.forward).parameters
    batch = 1
    flow_batch = FlowBatch(
        history=torch.randn(batch, 32, 128),
        history_anchor=torch.randn(batch, 32, 128),
        target=torch.randn(batch, 64, 128),
        anchor_path=torch.randn(batch, 64, 128),
        atlas_path=torch.randn(batch, 64, 8),
        lifecycle=torch.randn(batch, 64, 4),
        history_mask=torch.ones(batch, 32, dtype=torch.bool),
    )
    result = atlas_flow_matching_loss(model, flow_batch, AtlasLossConfig())
    assert set(result.components) == {"flow", "boundary", "statistics", "temporal_motion"}
    assert torch.isfinite(result.total)
    result.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_atlas_pca_is_train_only_but_transforms_holdout_and_projects_locally(tmp_path) -> None:
    rng = np.random.default_rng(7)
    values = {
        f"preset-{index:02d}": rng.normal(size=(4, 108, 128)).astype(np.float32)
        for index in range(12)
    }
    atlas = TimbreAtlas.fit(values, fit_presets=list(values)[:9])
    assert atlas.anchors.shape == (12, 128)
    assert atlas.coordinates.shape == (12, 8)
    projected, anchor = atlas.project_local(np.full(8, 100.0, dtype=np.float32))
    assert projected.shape == (8,)
    assert anchor.shape == (128,)
    assert np.isfinite(projected).all() and np.isfinite(anchor).all()
    path = tmp_path / "atlas.npz"
    atlas.save(path)
    loaded = TimbreAtlas.load(path)
    np.testing.assert_allclose(loaded.coordinates, atlas.coordinates)


def test_canonical_anchor_is_robust_to_one_extreme_frame() -> None:
    trajectories = np.zeros((4, 10, 128), dtype=np.float32)
    trajectories[0, 0] = 1_000.0
    np.testing.assert_allclose(canonical_anchor(trajectories), 0.0)


def test_evaluation_json_boundary_converts_numpy_and_rejects_nan(tmp_path) -> None:
    destination = tmp_path / "evaluation.json"
    payload = {
        "gate": np.bool_(True),
        "count": np.int64(3),
        "score": np.float32(0.25),
        "array": np.asarray([True, False], dtype=np.bool_),
        "missing": float("nan"),
    }
    native = _json_native(payload)
    assert native == {
        "gate": True,
        "count": 3,
        "score": 0.25,
        "array": [True, False],
        "missing": None,
    }
    _atomic_json(destination, payload)
    assert json.loads(destination.read_text(encoding="utf-8")) == native
    assert not list(tmp_path.glob("*.tmp"))


def test_atlas_graph_path_is_component_safe_and_projection_is_auditable() -> None:
    preset_ids = ("a", "b", "c", "d")
    anchors = np.eye(4, 128, dtype=np.float32)
    coordinates = np.zeros((4, 8), dtype=np.float32)
    coordinates[:, 0] = np.asarray((0.0, 1.0, 5.0, 6.0), dtype=np.float32)
    adjacency = np.asarray(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.bool_,
    )
    atlas = TimbreAtlas(
        preset_ids, anchors, np.zeros(128, dtype=np.float32),
        np.zeros((8, 128), dtype=np.float32), coordinates,
        np.ones(8, dtype=np.float32), adjacency,
    )
    assert atlas.connected_components() == ((0, 1), (2, 3))
    assert atlas.graph_path(0, 1) == (0, 1)
    assert atlas.graph_path(0, 3) is None
    details = atlas.project_details(np.zeros(8, dtype=np.float32), component=0)
    assert details["component"] == 0
    assert set(details["preset_ids"]).issubset({"a", "b"})
    assert np.isclose(np.asarray(details["weights"]).sum(), 1.0)


def test_live_solver_can_use_four_steps_without_changing_offline_default() -> None:
    config = AtlasModelConfig()
    model = AtlasConditionedTrajectoryFlow(config).eval()
    history = torch.zeros(1, 32, 128)
    anchor_path = torch.zeros(1, 64, 128)
    atlas_path = torch.zeros(1, 64, 8)
    lifecycle = torch.zeros(1, 64, 4)
    mask = torch.ones(1, 32, dtype=torch.bool)
    four = model.sample(
        history, history, anchor_path, atlas_path, lifecycle,
        seed=1, temperature=0.0, history_mask=mask, solver_steps=4,
    )
    assert four.shape == anchor_path.shape and torch.isfinite(four).all()
    with pytest.raises(ValueError, match="solver_steps"):
        model.sample(
            history, history, anchor_path, atlas_path, lifecycle,
            seed=1, temperature=0.0, history_mask=mask, solver_steps=9,
        )


def test_nonfinite_guard_counts_lifetime_events_but_aborts_only_a_streak() -> None:
    total = consecutive = 0
    for _ in range(20):
        total, consecutive, abort = _update_nonfinite_counters(
            finite=False, total=total, consecutive=consecutive,
        )
        assert not abort
        total, consecutive, abort = _update_nonfinite_counters(
            finite=True, total=total, consecutive=consecutive,
        )
        assert not abort and consecutive == 0
    assert total == 20

    for index in range(8):
        total, consecutive, abort = _update_nonfinite_counters(
            finite=False, total=total, consecutive=consecutive,
        )
        assert abort is (index == 7)


def test_numeric_recovery_escalates_on_repeated_nonfinite_or_large_gradients() -> None:
    assert _numeric_recovery_reason(
        finite=False,
        consecutive_nonfinite=2,
        recent_nonfinite=[True, False, False],
        gradient_norm=float("nan"),
        recent_gradient_norms=[],
    ) == "two_consecutive_nonfinite_gradients"
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[False, True, False, True, False, True, False],
        gradient_norm=1.0,
        recent_gradient_norms=[1.0],
    ) == "four_nonfinite_gradients_in_200_attempts"
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True],
        gradient_norm=1_001.0,
        recent_gradient_norms=[1_001.0],
    ) == "gradient_norm_above_1000"
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True],
        gradient_norm=101.0,
        recent_gradient_norms=[101.0, 102.0, 103.0, 104.0],
    ) == "four_gradient_norms_above_100_in_50_updates"
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True],
        gradient_norm=101.0,
        recent_gradient_norms=[101.0, 102.0, 103.0, 104.0],
        warmup_complete=False,
    ) is None
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True],
        gradient_norm=104.0,
        recent_gradient_norms=[101.0, 102.0, 103.0, 104.0],
        repeated_large_gradient_threshold=500.0,
    ) is None
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True],
        gradient_norm=504.0,
        recent_gradient_norms=[501.0, 502.0, 503.0, 504.0],
        repeated_large_gradient_threshold=500.0,
    ) == "four_gradient_norms_above_500_in_50_updates"
    assert _numeric_recovery_reason(
        finite=True,
        consecutive_nonfinite=0,
        recent_nonfinite=[True] * 200,
        gradient_norm=1.0,
        recent_gradient_norms=[1.0] * 50,
    ) is None


def test_oom_guard_recognizes_native_and_nccl_wrapped_cuda_failures() -> None:
    assert _is_out_of_memory_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert _is_out_of_memory_error(
        RuntimeError("ncclUnhandledCudaError: Cuda failure 2 'out of memory'")
    )
    assert not _is_out_of_memory_error(RuntimeError("NCCL connection closed"))


def test_recovery_profiles_lower_flow_lr_and_disable_amp_by_tier() -> None:
    safe = _recovery_profile("safe_amp")
    rescue = _recovery_profile("fp32_rescue")
    low = _recovery_profile("fp32_low_lr")
    assert safe.use_amp and safe.amp_initial_scale == safe.amp_max_scale == 128.0
    assert not rescue.use_amp and not low.use_amp
    assert (
        safe.flow_lr_multiplier,
        rescue.flow_lr_multiplier,
        low.flow_lr_multiplier,
    ) == (0.5, 0.25, 0.125)
    assert (safe.grad_clip, rescue.grad_clip, low.grad_clip) == (1.0, 1.0, 0.5)

    from midibrave.atlas_flow_config import AtlasTrainConfig

    config = SimpleNamespace(train=AtlasTrainConfig(output_root="test"))
    base = _scheduled_learning_rate("flow", 20_000, 100_000, config)
    assert np.isclose(_scheduled_learning_rate("flow", 20_000, 100_000, config, safe), base * 0.5)
    assert np.isclose(_scheduled_learning_rate("flow", 20_000, 100_000, config, rescue), base * 0.25)
    assert np.isclose(_scheduled_learning_rate("flow", 20_000, 100_000, config, low), base * 0.125)


def test_resume_position_skips_only_within_the_current_epoch() -> None:
    assert _resume_position(0, 11) == (0, 0)
    assert _resume_position(10, 11) == (0, 10)
    assert _resume_position(11, 11) == (1, 0)
    assert _resume_position(35, 11) == (3, 2)


def test_gradient_buckets_respect_cap_dtype_and_device() -> None:
    gradients = [
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float64),
    ]
    buckets = _gradient_buckets(gradients, bucket_cap_bytes=20)
    assert [[tensor.numel() for tensor in bucket] for bucket in buckets] == [[4], [3], [2]]
    assert all(len({tensor.dtype for tensor in bucket}) == 1 for bucket in buckets)


def test_tensorboard_record_exposes_loss_optimizer_stability_and_runtime() -> None:
    class Writer:
        def __init__(self) -> None:
            self.values: dict[str, tuple[float, int]] = {}
            self.flushed = False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.values[tag] = (float(value), step)

        def flush(self) -> None:
            self.flushed = True

    writer = Writer()
    _write_tensorboard_record(writer, {  # type: ignore[arg-type]
        "update": 20,
        "loss": 0.5,
        "components": {"flow/flow": 0.4},
        "module_gradient_norms": {"flow": 1.0},
        "gradient_norm": 1.5,
        "learning_rate": 1.0e-4,
        "amp_scale": 4.0,
        "nonfinite_skips": 0,
        "consecutive_nonfinite_skips": 0,
        "updates_per_second": 8.0,
        "peak_memory_mib": 1024.0,
        "peak_memory_reserved_mib": 1200.0,
    })
    assert writer.values["loss/total"] == (0.5, 20)
    assert writer.values["loss/components/flow/flow"] == (0.4, 20)
    assert writer.values["optimizer/learning_rate"] == (1.0e-4, 20)
    assert writer.values["runtime/updates_per_second"] == (8.0, 20)
    assert writer.flushed


def test_spark_accumulation_preserves_all_octopus_global_batches() -> None:
    assert _batch_contract("stage1", 32, 8, 1)["audio_global_batch"] == 256
    assert _batch_contract("flow", 96, 8, 1)["flow_global_batch"] == 768
    joint = _batch_contract("joint", 48, 8, 1)
    assert joint["audio_global_batch"] == 192
    assert joint["flow_global_batch"] == 96

    # Every approved OOM fallback keeps the same mathematical batch.
    assert _batch_contract("stage1", 16, 16, 1)["audio_global_batch"] == 256
    assert _batch_contract("stage1", 8, 32, 1)["audio_global_batch"] == 256
    assert _batch_contract("flow", 48, 16, 1)["flow_global_batch"] == 768
    assert _batch_contract("flow", 24, 32, 1)["flow_global_batch"] == 768
    assert _batch_contract("joint", 24, 16, 1)["audio_global_batch"] == 192
    assert _batch_contract("joint", 12, 32, 1)["flow_global_batch"] == 96

    for micro, accum in ((256, 1), (128, 2), (64, 4), (32, 8)):
        assert _batch_contract("stage1", micro, accum, 1)["audio_global_batch"] == 256
    for micro, accum in ((768, 1), (384, 2), (192, 4), (96, 8)):
        assert _batch_contract("flow", micro, accum, 1)["flow_global_batch"] == 768
    for micro, accum in ((384, 1), (192, 2), (96, 4), (48, 8)):
        contract = _batch_contract("joint", micro, accum, 1)
        assert contract["audio_global_batch"] == 192
        assert contract["flow_global_batch"] == 96


def _qualification_report(
    micro: int, accum: int, throughput: float, *, headroom: float = 30_000.0,
) -> dict[str, object]:
    return {
        "passed": True,
        "stage": "stage1",
        "updates_requested": 5,
        "updates_completed": 5,
        "effective_examples_per_second": throughput,
        "seconds_per_atomic_update": 256.0 / throughput,
        "memory_headroom_mib": headroom,
        "system_available_memory_mib": 50_000.0,
        "peak_memory_allocated_mib": 40_000.0,
        "peak_memory_reserved_mib": 42_000.0,
        "batch_contract": _batch_contract("stage1", micro, accum, 1),
        "host": "spark",
        "device": "NVIDIA GB10",
        "compute_capability": [12, 1],
        "torch_version": "test",
        "cuda_version": "test",
    }


def test_spark_selector_rejects_low_headroom_and_prefers_larger_micro_in_tie() -> None:
    script = Path(__file__).parents[1] / "scripts" / "atlas_flow" / "select_spark_batch_contract.py"
    select_contract = runpy.run_path(str(script))["select_contract"]
    result = select_contract(
        [
            _qualification_report(32, 8, 20.0),
            _qualification_report(64, 4, 25.0),
            _qualification_report(128, 2, 24.5),
            _qualification_report(256, 1, 28.0, headroom=10_000.0),
        ],
        stage="stage1", baseline=(32, 8),
    )
    assert result["selected"]["micro_batch"] == 128
    assert result["selection_reason"] == "larger_micro_within_tie"
    rejected = next(row for row in result["candidates"] if row["micro_batch"] == 256)
    assert "insufficient_memory_headroom" in rejected["rejection_reasons"]


def test_spark_selector_keeps_baseline_when_gain_is_below_five_percent() -> None:
    script = Path(__file__).parents[1] / "scripts" / "atlas_flow" / "select_spark_batch_contract.py"
    select_contract = runpy.run_path(str(script))["select_contract"]
    result = select_contract(
        [_qualification_report(32, 8, 20.0), _qualification_report(64, 4, 20.8)],
        stage="stage1", baseline=(32, 8),
    )
    assert result["selected"]["micro_batch"] == 32
    assert result["selection_reason"] == "gain_below_minimum"


def test_loss_divided_per_micro_matches_one_full_batch_optimizer_update() -> None:
    torch.manual_seed(11)
    inputs = torch.randn(8, 3)
    targets = torch.randn(8, 2)
    full = torch.nn.Linear(3, 2)
    accumulated = torch.nn.Linear(3, 2)
    accumulated.load_state_dict(full.state_dict())
    full_optimizer = torch.optim.SGD(full.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)

    full_optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(full(inputs), targets).backward()
    full_optimizer.step()

    accumulated_optimizer.zero_grad(set_to_none=True)
    for micro_inputs, micro_targets in zip(inputs.chunk(4), targets.chunk(4)):
        (torch.nn.functional.mse_loss(accumulated(micro_inputs), micro_targets) / 4).backward()
    accumulated_optimizer.step()

    for expected, actual in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(actual, expected)


def test_training_gate_reports_sparse_lifetime_skips_without_rejecting_them(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(json.dumps(row) for row in (
            {"update": 1, "loss": 2.0, "nonfinite_skips": 0},
            {
                "update": 200,
                "loss": 1.0,
                "nonfinite_skips": 8,
                "consecutive_nonfinite_skips": 0,
            },
        )) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "gate.json"
    script = Path(__file__).parents[1] / "scripts" / "atlas_flow" / "check_training_gate.py"
    namespace = runpy.run_path(str(script))
    original = sys.argv
    try:
        sys.argv = [
            str(script), "--metrics", str(metrics), "--minimum-update", "200",
            "--minimum-relative-decrease", "0.1", "--report", str(report),
        ]
        namespace["main"]()
    finally:
        sys.argv = original
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["passed"]
    assert result["maximum_nonfinite_skips"] == 8
    assert result["maximum_consecutive_nonfinite_skips"] == 0


def _fake_live_engine() -> SimpleNamespace:
    data = SimpleNamespace(
        sample_rate=1_000,
        note_on_sample=100,
        note_off_sample=2_600,
        render_samples=5_000,
    )
    coordinates = np.zeros((2, 8), dtype=np.float32)
    anchors = np.zeros((2, 128), dtype=np.float32)
    return SimpleNamespace(
        default_normalized=np.zeros(8, dtype=np.float32),
        atlas=SimpleNamespace(
            components=np.eye(8, 128, dtype=np.float32),
            coordinates=coordinates,
            anchors=anchors,
        ),
        component_membership=np.zeros(2, dtype=np.int64),
        center_index=0,
        config=SimpleNamespace(data=data),
    )


def _fake_render_plan(*, revision: int = 1, note: int = 50) -> RenderPlan:
    frames = 108
    waveform = np.ones((5_000, 2), dtype=np.float32)
    return RenderPlan(
        revision=revision,
        note=note,
        trajectory=torch.zeros(frames, 128),
        anchor_timeline=torch.zeros(frames, 128),
        coordinate_timeline=torch.zeros(frames, 8),
        waveform=waveform,
        requested_normalized=np.zeros(8, dtype=np.float32),
        projected_normalized=np.zeros(8, dtype=np.float32),
        projected_coordinate=np.zeros(8, dtype=np.float32),
        projected_anchor=np.zeros(128, dtype=np.float32),
        component=0,
        plan_ms=10.0,
        mode="graph_plan",
        morph_seconds=2.0,
        restart=False,
    )


def test_live_voice_sustains_past_canonical_five_seconds_until_note_off() -> None:
    session = AtlasFlowLiveSession(_fake_live_engine())
    session.active = PlaybackLayer(_fake_render_plan(), cursor=100)
    session.state = "held_sustain"

    blocks = [session.render_block()[0] for _ in range(6)]

    assert all(np.isfinite(block).all() and np.any(block) for block in blocks)
    assert session.state == "held_sustain"
    assert session.loop_start <= session.active.cursor < session.loop_end
    session.note_off()
    session.render_block()
    assert session.state == "idle"
    assert session.active is None


def test_live_replanning_does_not_interrupt_current_audio() -> None:
    session = AtlasFlowLiveSession(_fake_live_engine())
    session.active = PlaybackLayer(_fake_render_plan(), cursor=1_300)
    session.state = "held_sustain"
    session.planned_revision = 0

    accepted = session.request_control(
        seq=1,
        pca_normalized=[0.25] * 8,
        note=50,
        velocity=1.0,
        temperature=0.0,
        morph_seconds=2.0,
    )
    audio, _ = session.render_block()

    assert accepted and session.plan_pending()
    assert np.any(audio)
    assert session.active is not None
    assert session.state == "held_sustain"


def test_runtime_protocol_accepts_eight_pca_axes_and_clamps_controls() -> None:
    pytest.importorskip("aiohttp")
    from midibrave.atlas_flow_runtime_server import parse_message

    parsed = parse_message({
        "type": "control",
        "seq": 9,
        "pcaNormalized": [-2, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 2],
        "note": 90,
        "velocity": 2.0,
        "temperature": -1.0,
        "morphSeconds": 9.0,
    })
    assert parsed["pca_normalized"] == [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]
    assert parsed["note"] == 71
    assert parsed["velocity"] == 1.0
    assert parsed["temperature"] == 0.0
    assert parsed["morph_seconds"] == 5.0
    with pytest.raises(ValueError, match="exactly eight"):
        parse_message({
            "type": "start", "pcaNormalized": [0.0, 0.0], "note": 50,
            "velocity": 1.0, "temperature": 0.0, "seed": 1,
        })


def test_evaluation_audition_notes_are_six_representative_pitches() -> None:
    assert AUDITION_NOTES == (36, 43, 50, 57, 64, 71)
