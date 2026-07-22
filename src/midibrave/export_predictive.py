from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config
from .predictive_model import PredictiveMidiBrave
from .seed_bank import SeedBank


class _RuntimeDecoderStage(nn.Module):
    def __init__(self, ratio: int, blocks: nn.ModuleList, anti_alias: nn.Module,
                 projection: nn.Module, static_midi: bool):
        super().__init__()
        self.ratio = ratio
        self.blocks = copy.deepcopy(blocks)
        self.anti_alias = copy.deepcopy(anti_alias)
        self.projection = copy.deepcopy(projection)
        self.static_midi = static_midi

    def forward(self, value: Tensor, midi: Tensor, excitation: Tensor) -> Tensor:
        if self.ratio > 1:
            value = F.interpolate(value, scale_factor=float(self.ratio), mode="nearest")
            value = self.anti_alias(value)
        value = F.silu(self.projection(value))
        midi_level = (midi[..., :1] if self.static_midi else
                      F.interpolate(midi, size=value.shape[-1], mode="nearest"))
        for block in self.blocks:
            value = block(value, midi_level, excitation.to(dtype=value.dtype),
                          static_midi=self.static_midi)
        return value


class _RuntimeDecoder(nn.Module):
    """TorchScript-friendly structural copy of BraveDecoder."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.rave_latent_dim = int(source.rave_latent_dim)
        self.total_ratio = int(source.total_ratio)
        self.fusion = copy.deepcopy(source.fusion)
        self.excitation_downsamplers = copy.deepcopy(source.excitation_downsamplers)
        self.stages = nn.ModuleList([
            _RuntimeDecoderStage(
                int(ratio), blocks, anti_alias, projection,
                bool(source.static_condition_fast_path))
            for ratio, blocks, anti_alias, projection in zip(
                source.ratios, source.blocks, source.anti_alias, source.projections)
        ])
        self.output = copy.deepcopy(source.output)
        self.pqmf = copy.deepcopy(source.pqmf)

    def forward(self, z_clap: Tensor, z_midi: Tensor, excitation: Tensor,
                output_samples: int, z_rave: Tensor) -> Tensor:
        if (z_rave.ndim != 3 or z_rave.shape[1] != self.rave_latent_dim
                or z_rave.shape[0] != z_midi.shape[0]
                or z_rave.shape[2] != z_midi.shape[2]):
            raise ValueError("runtime RAVE and MIDI latent sequences do not align")
        if (excitation.shape[1] != self.pqmf.bands
                or excitation.shape[2] != z_midi.shape[2] * self.total_ratio):
            raise ValueError("runtime excitation does not align with latent frames")
        reversed_levels = torch.jit.annotate(list[Tensor], [])
        reversed_levels.append(excitation)
        current = excitation
        for downsampler in self.excitation_downsamplers:
            current = downsampler(current)
            reversed_levels.append(current)
        value = self.fusion(z_clap, z_midi, False, z_rave)
        level_count = len(reversed_levels)
        for index, stage in enumerate(self.stages):
            value = stage(value, z_midi, reversed_levels[level_count - index - 1])
        subbands = self.output(value)
        waveform = self.pqmf.synthesis(subbands.float(), output_samples)
        return torch.tanh(waveform)


class _RuntimeHarmonicExcitation(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.sample_rate = float(source.sample_rate)
        self.target_rms = float(source.target_rms)
        self.register_buffer("harmonics", source.harmonics.detach().clone())

    def forward(self, note: Tensor, samples: int) -> Tensor:
        frequency = 440.0 * torch.pow(2.0, (note.float() - 69.0) / 12.0)
        time = torch.arange(1, samples + 1, device=note.device, dtype=torch.float32)
        phase = (2.0 * torch.pi / self.sample_rate) * frequency[:, None] * time[None, :]
        harmonic = self.harmonics.to(note.device).view(1, -1, 1)
        keep = frequency[:, None, None] * harmonic <= self.sample_rate / 2.0
        excitation = ((torch.sin(phase[:, None, :] * harmonic) / harmonic) * keep).sum(dim=1)
        measured = excitation.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (excitation * (self.target_rms / measured)).unsqueeze(1)


class PredictiveRuntime(nn.Module):
    """Deployable continuation graph; deliberately owns no audio encoder."""

    def __init__(self, model: PredictiveMidiBrave, bank: SeedBank,
                 latent_std: Tensor | None = None):
        super().__init__()
        predictive = model.predictive_config
        expected = {
            "architecture": predictive.architecture,
            "latent_dim": predictive.rave_latent_dim,
            "history_frames": predictive.history_frames,
            "samples_per_latent": predictive.samples_per_latent,
        }
        for field, value in expected.items():
            if bank.metadata.get(field) != value:
                raise ValueError(f"seed bank {field} mismatch")
        # Deep-copy only deployable children. Keeping the training parent here
        # would silently retain the RAVE encoder in serialized state.
        self.clap_projection = copy.deepcopy(model.clap_projection)
        self.midi = copy.deepcopy(model.midi)
        self.predictor = copy.deepcopy(model.predictor)
        self.decoder = _RuntimeDecoder(model.decoder)
        self.excitation = _RuntimeHarmonicExcitation(model.excitation)
        self.history_frames = predictive.history_frames
        self.horizon_frames = predictive.horizon_frames
        self.stride_frames = predictive.stride_frames
        self.samples_per_latent = predictive.samples_per_latent
        self.rave_latent_dim = predictive.rave_latent_dim
        self.clap_dim = model.model_config.clap_dim
        self.midi_dim = model.model_config.midi_dim
        self.pqmf_fp32 = model.model_config.pqmf_dtype == "fp32"
        self.register_buffer("seed_latents", torch.from_numpy(bank.latents.copy()))
        self.register_buffer("seed_clap", torch.from_numpy(bank.clap.copy()))
        scale = (torch.ones(self.rave_latent_dim) if latent_std is None
                 else latent_std.detach().float().cpu())
        if scale.shape != (self.rave_latent_dim,):
            raise ValueError("runtime latent statistics have the wrong shape")
        self.register_buffer("latent_std", scale.clamp_min(1e-4))
        noise_frames = ((predictive.history_frames + predictive.stride_frames)
                        * self.decoder.total_ratio)
        if model.model_config.stochastic_excitation:
            with torch.no_grad():
                noise = model.stochastic_excitation(
                    32, noise_frames, torch.device("cpu"),
                    torch.arange(32, dtype=torch.long) + model.model_config.stochastic_seed)
        else:
            noise = torch.zeros(1, model.model_config.pqmf_bands, noise_frames)
        self.register_buffer("noise_table", noise)
        self.register_buffer("step_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_seed_distance", torch.zeros(()))
        self.register_buffer("last_latent_variance", torch.zeros(()))
        self.register_buffer("last_finite", torch.ones(()))

    @classmethod
    def from_training_model(cls, model: PredictiveMidiBrave, bank: SeedBank,
                            latent_std: Tensor | None = None) -> "PredictiveRuntime":
        return cls(model, bank, latent_std)

    def _project_clap(self, clap: Tensor) -> Tensor:
        if clap.ndim == 2:
            if clap.shape[1] != self.clap_dim:
                raise ValueError("CLAP control dimension mismatch")
            return self.clap_projection(clap)
        if clap.ndim != 3 or clap.shape[1] != self.clap_dim:
            raise ValueError("CLAP trajectory must have shape [batch, clap_dim, frames]")
        normalized = F.normalize(clap.transpose(1, 2), dim=-1)
        return self.clap_projection.net(normalized).transpose(1, 2)

    @torch.jit.export
    def initial_state(self, clap: Tensor, random_seed: int, top_k: int = 8) -> Tensor:
        if clap.ndim == 1:
            clap = clap.unsqueeze(0)
        if clap.ndim != 2 or clap.shape[1] != self.seed_clap.shape[1]:
            raise ValueError("initial CLAP control does not match the seed bank")
        controls = F.normalize(clap.float(), dim=1)
        bank = F.normalize(self.seed_clap.float(), dim=1)
        similarity = controls @ bank.transpose(0, 1)
        count = min(max(1, top_k), self.seed_latents.shape[0])
        values, candidates = torch.topk(similarity, count, dim=1)
        rows = torch.arange(clap.shape[0], device=clap.device, dtype=torch.long)
        choices = torch.remainder(rows * 1_000_003 + random_seed, count)
        selected = candidates.gather(1, choices[:, None]).squeeze(1)
        selected_similarity = values.gather(1, choices[:, None]).squeeze(1)
        self.last_seed_distance.copy_((1.0 - selected_similarity).mean())
        self.step_index.zero_()
        history = self.seed_latents.index_select(0, selected).to(clap.device)
        self.last_latent_variance.copy_(
            (history / self.latent_std.view(1, -1, 1)).var(unbiased=False))
        self.last_finite.copy_(torch.isfinite(history).all().to(self.last_finite))
        return history

    def _future_clap(self, clap: Tensor, batch: int) -> Tensor:
        projected = self._project_clap(clap)
        if projected.ndim == 2:
            return projected.unsqueeze(-1).expand(-1, -1, self.horizon_frames)
        if projected.shape[0] != batch or projected.shape[-1] != self.horizon_frames:
            raise ValueError("CLAP future must contain exactly K frames")
        return projected

    def _excitation_bands(self, note: Tensor, samples: int) -> Tensor:
        waveform = self.excitation(note, samples)
        bands = self.decoder.pqmf.analysis(
            waveform.float() if self.pqmf_fp32 else waveform)
        indices = torch.remainder(
            torch.arange(note.shape[0], device=note.device, dtype=torch.long)
            + self.step_index, self.noise_table.shape[0])
        noise = self.noise_table.index_select(0, indices)
        if noise.shape[-1] != bands.shape[-1]:
            raise ValueError("runtime noise table does not match decoder context")
        bands = bands.float() + noise
        return bands

    @torch.jit.export
    def step(self, history: Tensor, clap_future: Tensor, note: Tensor,
             velocity: Tensor) -> tuple[Tensor, Tensor]:
        if (history.ndim != 3 or history.shape[1] != self.rave_latent_dim
                or history.shape[2] != self.history_frames):
            raise ValueError("runtime history must have shape [batch, rave_dim, C]")
        if (note.ndim != 1 or velocity.ndim != 1
                or note.shape[0] != history.shape[0]
                or velocity.shape[0] != history.shape[0]):
            raise ValueError("MIDI controls must contain one value per runtime voice")
        z_clap_future = self._future_clap(clap_future, history.shape[0])
        z_midi_future = self.midi(
            note, velocity, self.horizon_frames, static_condition=False)
        prediction = self.predictor(history, z_clap_future, z_midi_future).latent
        consumed = prediction[..., :self.stride_frames]
        new_history = torch.cat((history, consumed), dim=-1)[..., -self.history_frames:]

        context = torch.cat((history, consumed), dim=-1)
        historical_clap = z_clap_future[..., :1].expand(
            -1, -1, self.history_frames)
        context_clap = torch.cat(
            (historical_clap, z_clap_future[..., :self.stride_frames]), dim=-1)
        context_midi = self.midi(
            note, velocity, context.shape[-1], static_condition=False)
        samples = context.shape[-1] * self.samples_per_latent
        excitation = self._excitation_bands(note, samples)
        decoded = self.decoder(
            context_clap, context_midi, excitation, samples, z_rave=context)
        emitted_samples = self.stride_frames * self.samples_per_latent
        audio = decoded[..., -emitted_samples:]
        finite = torch.isfinite(audio).all() & torch.isfinite(new_history).all()
        self.last_finite.copy_(finite.to(self.last_finite))
        self.last_latent_variance.copy_(
            (new_history / self.latent_std.view(1, -1, 1)).var(unbiased=False))
        self.step_index.add_(1)
        return audio, new_history

    @torch.jit.export
    def diagnostics(self) -> Tensor:
        return torch.stack((self.last_seed_distance, self.last_latent_variance,
                            self.last_finite))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_predictive_runtime(config_path: str | Path, checkpoint_path: str | Path,
                              seed_bank_path: str | Path, output_path: str | Path,
                              device: str = "cpu") -> dict[str, object]:
    """Audit and atomically export an encoder-free scripted runtime."""
    from .trainer import load_predictive_statistics

    config = Config.load(config_path)
    if config.predictive is None:
        raise ValueError("predictive export requires a v3 config")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 5:
        raise ValueError("predictive export requires checkpoint format 5")
    contract = payload.get("predictive_contract")
    if not isinstance(contract, dict) or contract.get("stage") not in {"rollout", "gan"}:
        raise ValueError("export requires a rollout-validated or GAN checkpoint")
    for field, value in {
        "architecture": config.predictive.architecture,
        "history_frames": config.predictive.history_frames,
        "horizon_frames": config.predictive.horizon_frames,
        "stride_frames": config.predictive.stride_frames,
    }.items():
        if contract.get(field) != value:
            raise ValueError(f"checkpoint {field.replace('_', ' ')} mismatch")
    statistics, rave_checkpoint_hash, statistics_hash = load_predictive_statistics(
        Path(config.data.cache_root) / "rave-statistics.npz", config)
    if contract.get("latent_statistics_hash") != statistics_hash:
        raise ValueError("checkpoint latent statistics hash mismatch")
    bank = SeedBank.load(seed_bank_path, {
        "architecture": config.predictive.architecture,
        "latent_dim": config.predictive.rave_latent_dim,
        "history_frames": config.predictive.history_frames,
        "samples_per_latent": config.predictive.samples_per_latent,
        "checkpoint_hash": rave_checkpoint_hash,
    })
    training_model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate)
    training_model.load_state_dict(payload["model"])
    runtime = PredictiveRuntime.from_training_model(
        training_model, bank, statistics.latent_std).to(device).eval()
    forbidden = [name for name, _ in runtime.named_modules()
                 if "encoder" in name.lower() or "clap_audio" in name.lower()]
    forbidden += [name for name in runtime.state_dict()
                  if "encoder" in name.lower() or "clap_audio" in name.lower()]
    if forbidden:
        raise RuntimeError(f"encoder-free export audit failed: {sorted(set(forbidden))}")
    with torch.no_grad():
        clap = torch.from_numpy(bank.clap[:1]).to(device)
        history = runtime.initial_state(clap, 0)
        note = torch.from_numpy(bank.notes[:1]).to(device)
        velocity = torch.from_numpy(bank.velocities[:1]).to(device)
        for _ in range(2):
            audio, history = runtime.step(history, clap, note, velocity)
            if audio.shape[-1] != config.predictive.stride_frames * config.predictive.samples_per_latent:
                raise RuntimeError("runtime emitted the wrong audio block size")
        if runtime.diagnostics()[2].item() != 1.0:
            raise RuntimeError("runtime smoke test produced non-finite values")
    scripted = torch.jit.script(runtime.cpu())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    scripted.save(str(temporary))
    temporary.replace(output)
    metadata = {
        "format": 1, "architecture": config.predictive.architecture,
        "history_frames": config.predictive.history_frames,
        "horizon_frames": config.predictive.horizon_frames,
        "stride_frames": config.predictive.stride_frames,
        "samples_per_latent": config.predictive.samples_per_latent,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed_bank_npz_sha256": _sha256(Path(seed_bank_path).with_suffix(".npz")),
        "seed_bank_json_sha256": _sha256(Path(seed_bank_path).with_suffix(".json")),
        "latent_statistics_sha256": statistics_hash,
        "runtime_sha256": _sha256(output),
        "sample_rate": config.data.sample_rate,
        "encoder_free": True,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    return {"runtime": str(output.resolve()), "metadata": str(metadata_path.resolve()),
            **metadata}
