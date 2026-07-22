from pathlib import Path

import torch

from midibrave.config import Config
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.trainer import (
    PredictiveStage,
    configure_predictive_stage,
    load_predictive_warm_start,
    predictive_checkpoint_contract,
    save_predictive_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _model(config: Config) -> PredictiveMidiBrave:
    assert config.predictive is not None
    return PredictiveMidiBrave(
        config.model,
        config.predictive,
        config.data.window_samples,
        config.data.sample_rate,
    )


def test_rollout_checkpoint_can_warm_start_corrective_rollout(tmp_path: Path):
    config = Config.load(ROOT / "configs/v3/smoke.yaml")
    source = _model(config)
    configure_predictive_stage(source, PredictiveStage.ROLLOUT)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    checkpoint = tmp_path / "rollout.pt"
    save_predictive_checkpoint(
        checkpoint,
        source,
        optimizer,
        scaler,
        predictive_checkpoint_contract(
            config, PredictiveStage.ROLLOUT, "stats", "calibration", False
        ),
        stage_update=1_500,
        epoch=4,
        microbatch_offset=3,
    )

    target = _model(config)
    load_predictive_warm_start(
        checkpoint, target, config, PredictiveStage.ROLLOUT
    )

    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)
