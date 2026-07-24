from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midibrave.zrave_flow_config import ZraveFlowConfig


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "zrave" / "octopus_flow.yaml"


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
    assert [source.weight for source in config.data.sources] == [
        0.55,
        0.20,
        0.15,
        0.10,
    ]
    assert config.train.checkpoint_every == 5000
    assert config.train.pitch_transition_fraction == 0.20


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
