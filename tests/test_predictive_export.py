from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path
import hashlib

from midibrave.config import Config
from midibrave.export_predictive import PredictiveRuntime, export_predictive_runtime
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.rave_encoder import RaveEncoder
from midibrave.seed_bank import SeedBank
from midibrave.trainer import PredictiveStage, predictive_checkpoint_contract


def _runtime() -> PredictiveRuntime:
    config = Config.load("configs/v3/smoke.yaml")
    assert config.predictive is not None
    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)
    rng = np.random.default_rng(7)
    bank = SeedBank(
        rng.normal(size=(3, 16, 16)).astype(np.float32),
        rng.normal(size=(3, 512)).astype(np.float32),
        np.asarray([48, 60, 72]), np.asarray([80, 100, 120], dtype=np.float32),
        ["a", "b", "c"],
        {"architecture": "predictive_rave_v1", "latent_dim": 16,
         "history_frames": 16, "samples_per_latent": 128,
         "checkpoint_hash": "rave-checkpoint"})
    return PredictiveRuntime.from_training_model(model, bank)


def test_runtime_contains_no_training_audio_encoder():
    runtime = _runtime()
    assert not any(isinstance(module, RaveEncoder) for module in runtime.modules())
    assert not any("encoder" in name or "clap_audio" in name
                   for name, _ in runtime.named_modules())
    assert not any("encoder" in name or "clap_audio" in name
                   for name in runtime.state_dict())


def test_runtime_seed_and_four_frame_step_are_reproducible():
    runtime = _runtime().eval()
    clap = torch.randn(1, 512)
    first = runtime.initial_state(clap, 17)
    second = runtime.initial_state(clap, 17)
    assert torch.equal(first, second)
    assert first.shape == (1, 16, 16)

    audio, history = runtime.step(
        first, clap, torch.tensor([60]), torch.tensor([100.0]))
    assert audio.shape == (1, 1, 4 * 128)
    assert history.shape == (1, 16, 16)
    diagnostics = runtime.diagnostics()
    assert diagnostics.shape == (3,)
    assert diagnostics[2].item() == 1.0


def test_runtime_is_torchscript_serializable():
    runtime = _runtime().eval()
    scripted = torch.jit.script(runtime)
    clap = torch.randn(1, 512)
    history = scripted.initial_state(clap, 9)
    audio, next_history = scripted.step(
        history, clap, torch.tensor([64]), torch.tensor([90.0]))
    assert audio.shape[-1] == 512
    assert next_history.shape[-1] == 16


def test_export_writes_audited_encoder_free_artifact(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/v3/smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_root"] = str(tmp_path / "cache")
    config_path = tmp_path / "export.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = Config.load(config_path)
    assert config.predictive is not None
    cache = Path(config.data.cache_root)
    cache.mkdir(parents=True)
    stats_path = cache / "rave-statistics.npz"
    np.savez_compressed(
        stats_path, latent_std=np.ones(16, dtype=np.float32),
        delta_std=np.ones(16, dtype=np.float32),
        acceleration_std=np.ones(16, dtype=np.float32),
        checkpoint_hash=np.asarray("rave-checkpoint"),
        samples_per_latent=np.asarray(128, dtype=np.int64))
    stats_hash = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)
    contract = predictive_checkpoint_contract(
        config, PredictiveStage.ROLLOUT, stats_hash, "calibration", True)
    checkpoint = tmp_path / "rollout.pt"
    torch.save({"format": 5, "predictive_contract": contract,
                "model": model.state_dict()}, checkpoint)
    rng = np.random.default_rng(3)
    bank = SeedBank(
        rng.normal(size=(2, 16, 16)).astype(np.float32),
        rng.normal(size=(2, 512)).astype(np.float32),
        np.asarray([60, 64]), np.asarray([90, 100], dtype=np.float32),
        ["one", "two"],
        {"architecture": "predictive_rave_v1", "latent_dim": 16,
         "history_frames": 16, "samples_per_latent": 128,
         "checkpoint_hash": "rave-checkpoint"})
    bank_path = tmp_path / "bank"
    bank.save(bank_path)
    output = tmp_path / "runtime.pt"
    result = export_predictive_runtime(
        config_path, checkpoint, bank_path, output)
    assert output.is_file()
    assert Path(str(output) + ".json").is_file()
    assert result["encoder_free"] is True
    loaded = torch.jit.load(str(output))
    assert not any("encoder" in key for key in loaded.state_dict())
