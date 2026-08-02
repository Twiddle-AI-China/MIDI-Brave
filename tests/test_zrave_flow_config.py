from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midibrave.zrave_flow_config import ZraveFlowConfig


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "zrave" / "octopus_flow.yaml"
PURE_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_pure_flow_poc.yaml"
)
SERUM128_CONFIG = (
    ROOT / "configs" / "zrave" / "octopus_serum128_pure_flow.yaml"
)
EXPLORATION_CONFIG = (
    ROOT
    / "configs"
    / "zrave"
    / "octopus_serum128_exploration_flow.yaml"
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
    assert config.rave.checkpoint.endswith(
        "/codec/serum-balanced-rave128-full.ts"
    )
    assert config.rave.expected_sha256 == (
        "69e8a2a133151eb896842100c993c6f4904936f08ee72e430f1c9c2556b6f9b4"
    )
    assert [source.name for source in config.data.sources] == [
        "serum_balanced"
    ]
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
