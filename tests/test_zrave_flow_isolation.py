from __future__ import annotations

from pathlib import Path

from midibrave.zrave_config import ZraveConfig


ROOT = Path(__file__).parents[1]


def test_flow_modules_keep_midi_local_and_do_not_import_clap_stack() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for path in (ROOT / "src" / "midibrave").glob(
            "zrave_flow_*.py"
        )
    )

    assert "laion_clap" not in source
    assert "predictive_model" not in source
    assert "predictivemidibrave" not in source
    assert "zrave_prediction_loss" not in source
    assert "midi_sequence_conditioning" in source


def test_legacy_standalone_config_is_unchanged() -> None:
    config = ZraveConfig.load(
        ROOT
        / "configs"
        / "zrave"
        / "octopus_standalone50_rollout.yaml"
    )

    assert config.model.context_frames == 32
    assert config.model.horizon_frames == 8
    assert config.loss.future == 1.0
    assert config.loss.delta == 0.5
