from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midibrave.zrave_flow_config import (
    FlowModelConfig,
    FlowSegmentSamplingConfig,
    FlowTrainConfig,
    ZraveFlowConfig,
    load_preset_allowlist,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "zrave" / "octopus_flow.yaml"
PURE_CONFIG = ROOT / "configs" / "zrave" / "octopus_pure_flow_poc.yaml"
SERUM128_CONFIG = ROOT / "configs" / "zrave" / "octopus_serum128_pure_flow.yaml"
EXPLORATION_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_serum128_exploration_flow.yaml"
)
SEGMENT_PURE_CONFIG = (
    ROOT / "configs" / "zrave" / "lvzihao_serum128_segment248_pure.yaml"
)
SEGMENT_MIDI_CONFIG = (
    ROOT / "configs" / "zrave" / "lvzihao_serum128_segment248_midi32.yaml"
)


def test_flow_config_locks_runtime_and_data_contract() -> None:
    config = ZraveFlowConfig.load(CONFIG)

    assert config.model.latent_dim == 16
    assert config.model.context_frames == 32
    assert config.model.future_frames == 64
    assert config.model.note_min == 21
    assert config.model.note_max == 109
    assert config.model.wander_delay_frames == (16, 32, 48)
    assert config.model.solver_steps == 8
    assert config.data.shard_records == 4096
    assert config.data.maximum_audio_seconds == 5.0
    assert [source.name for source in config.data.sources] == [
        "serum_full",
        "pianobook_pitch",
        "dexed_surge_broad",
    ]
    assert [source.weight for source in config.data.sources] == [
        0.65,
        0.10,
        0.25,
    ]
    assert config.train.checkpoint_every == 5000
    assert config.train.pitch_transition_fraction == 0.20
    assert config.model.midi_sequence_conditioning is False
    assert config.model.profile == "standard"
    assert config.segment_sampling.enabled is False
    assert config.segment_sampling.divisions == (2, 4, 8)
    assert config.segment_sampling.include_first is False


def test_pure_flow_config_disables_pitch_conditioning() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)

    assert config.model.pitch_conditioning is False
    assert config.model.context_frames == 32
    assert config.model.future_frames == 64
    assert config.model.solver_steps == 8
    assert config.train.checkpoint_every == 5000
    assert config.train.pitch_transition_fraction == 0.0
    assert config.train.output_root.endswith("/pure-flow-poc-v1")
    assert config.exploration.enabled is False


def test_serum128_config_binds_the_balanced_rave_codec() -> None:
    config = ZraveFlowConfig.load(SERUM128_CONFIG)

    assert config.model.latent_dim == 128
    assert config.model.pitch_conditioning is False
    assert config.model.context_frames == 32
    assert config.model.future_frames == 64
    assert config.rave.checkpoint.endswith("/codec/serum-balanced-rave128-full.ts")
    assert config.rave.expected_sha256 == (
        "69e8a2a133151eb896842100c993c6f4904936f08ee72e430f1c9c2556b6f9b4"
    )
    assert [source.name for source in config.data.sources] == ["serum_balanced"]
    assert config.data.sources[0].weight == 1.0
    assert config.data.sources[0].manifest.endswith(
        "/midibrave-rave-serum-v2/corpus/manifest.jsonl"
    )
    assert config.train.output_root.endswith("/runs/pure-flow-v1")
    assert config.exploration.enabled is False


def test_serum128_exploration_config_locks_v2_contract() -> None:
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG)

    assert config.model.latent_dim == 128
    assert config.model.pitch_conditioning is False
    assert config.model.context_frames == 32
    assert config.model.future_frames == 64
    assert config.exploration.enabled is True
    assert config.exploration.visible_history_frames == (8, 16, 32)
    assert config.exploration.rollout_stride_frames == 16
    assert config.exploration.exposure_max_depth == 3
    assert config.exploration.temporal_loss_weight == 0.05
    assert config.exploration.exposure_probability == 0.50
    assert config.train.max_updates == 20000
    assert config.train.output_root.endswith("/runs/exploration-v2")


def test_flow_config_rejects_unknown_and_inconsistent_values(
    tmp_path: Path,
) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        text.replace(
            "  future_frames: 64\n",
            "  future_frames: 63\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="future_frames must be 64"):
        ZraveFlowConfig.load(bad)

    config = ZraveFlowConfig.load(CONFIG)
    with pytest.raises(ValueError, match="source weights must sum to 1"):
        replace(
            config.data,
            sources=(
                replace(config.data.sources[0], weight=0.54),
                *config.data.sources[1:],
            ),
        )


def test_flow_config_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "unknown.yaml"
    bad.write_text(
        text.replace(
            "  future_frames: 64",
            "  future_frames: 64\n  mystery_option: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model config keys"):
        ZraveFlowConfig.load(bad)


def test_segment_sampling_config_loads_as_an_opt_in_task(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "segment.yaml"
    bad.write_text(
        PURE_CONFIG.read_text(encoding="utf-8")
        + "\nsegment_sampling:\n"
        + "  enabled: true\n"
        + "  divisions: [2, 8]\n"
        + "  include_first: false\n",
        encoding="utf-8",
    )

    config = ZraveFlowConfig.load(bad)

    assert config.segment_sampling.enabled is True
    assert config.segment_sampling.divisions == (2, 8)
    assert config.segment_sampling.include_first is False


def test_lvzihao_segment_configs_define_separate_weight_families() -> None:
    pure = ZraveFlowConfig.load(SEGMENT_PURE_CONFIG)
    midi = ZraveFlowConfig.load(SEGMENT_MIDI_CONFIG)

    assert pure.segment_sampling.enabled
    assert pure.model.pitch_conditioning is False
    assert len(pure.data.sources[0].allowed_categories) == 12
    assert pure.train.output_root.endswith("/runs/segment248-pure")
    assert midi.segment_sampling.enabled
    assert midi.model.pitch_conditioning is True
    assert midi.model.midi_sequence_conditioning is True
    assert midi.data.sources[0].allowed_categories == (
        "Arp",
        "Bass",
        "Chord",
        "Keys",
        "Lead",
        "Pad",
        "Pluck",
        "Synth",
    )
    assert midi.train.output_root.endswith("/runs/segment248-midi32")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"divisions": (4, 2)}, "ordered, unique subset"),
        ({"divisions": (2, 2)}, "ordered, unique subset"),
        ({"divisions": (2, 3)}, "ordered, unique subset"),
        ({"include_first": True}, "beginning-of-sequence"),
    ],
)
def test_segment_sampling_rejects_ambiguous_or_noncausal_contracts(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FlowSegmentSamplingConfig(**values)


def test_midi_sequence_conditioning_requires_pitch_conditioning() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)

    with pytest.raises(ValueError, match="requires pitch_conditioning"):
        replace(config.model, midi_sequence_conditioning=True)


def test_exploration_rejects_midi_until_rollout_is_condition_aware() -> None:
    config = ZraveFlowConfig.load(EXPLORATION_CONFIG)
    train = replace(config.train, pitch_transition_fraction=0.20)

    with pytest.raises(ValueError, match="MIDI conditioning"):
        replace(
            config,
            model=replace(config.model, pitch_conditioning=True),
            train=train,
        )

    with pytest.raises(ValueError, match="MIDI conditioning"):
        replace(
            config,
            model=replace(
                config.model,
                pitch_conditioning=True,
                midi_sequence_conditioning=True,
            ),
            train=train,
        )


@pytest.mark.parametrize(
    ("profile", "dimensions"),
    [
        ("standard", (384, 4, 8, 8, 1536)),
        ("small", (256, 3, 6, 8, 1024)),
        ("tiny", (128, 2, 4, 4, 512)),
    ],
)
def test_flow_model_accepts_only_named_exact_profiles(
    profile: str,
    dimensions: tuple[int, int, int, int, int],
) -> None:
    model = FlowModelConfig(
        profile=profile,
        d_model=dimensions[0],
        context_layers=dimensions[1],
        future_layers=dimensions[2],
        heads=dimensions[3],
        feedforward_dim=dimensions[4],
    )

    assert (
        model.d_model,
        model.context_layers,
        model.future_layers,
        model.heads,
        model.feedforward_dim,
    ) == dimensions


def test_flow_model_rejects_unnamed_or_mixed_profile_dimensions() -> None:
    with pytest.raises(ValueError, match="model.profile"):
        FlowModelConfig(profile="custom")

    with pytest.raises(ValueError, match="profile small requires"):
        FlowModelConfig(profile="small")

    with pytest.raises(ValueError, match="profile tiny requires"):
        FlowModelConfig(
            profile="tiny",
            d_model=128,
            context_layers=2,
            future_layers=4,
            heads=8,
            feedforward_dim=512,
        )


def test_train_intervals_accept_only_controlled_values() -> None:
    for interval in (1000, 5000):
        train = FlowTrainConfig(
            output_root="/tmp/profile-test",
            checkpoint_every=interval,
            validation_every=interval,
            short_future_updates=interval,
        )
        assert train.checkpoint_every == interval

    with pytest.raises(ValueError, match="must be 1000 or 5000"):
        FlowTrainConfig(
            output_root="/tmp/profile-test",
            checkpoint_every=2000,
        )


@pytest.mark.parametrize(
    ("profile", "dimensions"),
    [
        ("small", (256, 3, 6, 8, 1024)),
        ("tiny", (128, 2, 4, 4, 512)),
    ],
)
def test_small_profiles_require_dense_checkpoint_and_curriculum_intervals(
    profile: str,
    dimensions: tuple[int, int, int, int, int],
) -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)
    model = replace(
        config.model,
        profile=profile,
        d_model=dimensions[0],
        context_layers=dimensions[1],
        future_layers=dimensions[2],
        heads=dimensions[3],
        feedforward_dim=dimensions[4],
    )
    train = replace(
        config.train,
        max_updates=10000,
        checkpoint_every=1000,
        validation_every=1000,
        short_future_updates=1000,
    )

    resolved = replace(config, model=model, train=train)

    assert resolved.train.max_updates == 10000
    assert resolved.train.checkpoint_every == 1000
    with pytest.raises(
        ValueError,
        match=f"must be 1000 for model.profile {profile}",
    ):
        replace(resolved, train=replace(train, checkpoint_every=5000))


def test_standard_profile_keeps_5000_intervals() -> None:
    config = ZraveFlowConfig.load(PURE_CONFIG)

    with pytest.raises(
        ValueError,
        match="must be 5000 for model.profile standard",
    ):
        replace(
            config,
            train=replace(
                config.train,
                checkpoint_every=1000,
                validation_every=1000,
                short_future_updates=1000,
            ),
        )


def test_preset_allowlist_loads_taxonomy_json_jsonl_and_text(
    tmp_path: Path,
) -> None:
    object_path = tmp_path / "bucket.json"
    object_path.write_text(
        '{"preset_ids":["serum:a","serum:b","serum:a"]}',
        encoding="utf-8",
    )
    array_path = tmp_path / "bucket-array.json"
    array_path.write_text(
        '["serum:c", {"canonical_preset_id":"serum:d"}]',
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "bucket.jsonl"
    jsonl_path.write_text(
        '{"canonical_preset_id":"serum:e","split":"train"}\n{"preset_id":"serum:f"}\n',
        encoding="utf-8",
    )
    text_path = tmp_path / "bucket.ids.txt"
    text_path.write_text(
        "# taxonomy bucket\nserum:g\n\nserum:h\n",
        encoding="utf-8",
    )

    assert load_preset_allowlist(object_path) == (
        "serum:a",
        "serum:b",
    )
    assert load_preset_allowlist(array_path) == (
        "serum:c",
        "serum:d",
    )
    assert load_preset_allowlist(jsonl_path) == (
        "serum:e",
        "serum:f",
    )
    assert load_preset_allowlist(text_path) == (
        "serum:g",
        "serum:h",
    )


def test_preset_allowlist_config_and_loader_reject_empty_values(
    tmp_path: Path,
) -> None:
    source = ZraveFlowConfig.load(PURE_CONFIG).data.sources[0]

    with pytest.raises(ValueError, match="non-empty path"):
        replace(source, preset_allowlist="")

    empty = tmp_path / "empty.ids.txt"
    empty.write_text("# no presets\n", encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist is empty"):
        load_preset_allowlist(empty)
