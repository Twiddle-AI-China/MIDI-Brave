from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig, PredictiveConfig
from .latent_predictor import LatentPrediction, MultiHorizonPredictor
from .model import (BraveDecoder, HarmonicExcitation, MidiConditioner,
                    PitchAdversary, StochasticBandExcitation, TimbreAdapter,
                    _GradientReverse)
from .rave_encoder import RaveEncoder, RavePosterior


@dataclass(frozen=True)
class PredictiveReconstruction:
    audio: Tensor
    posterior: RavePosterior
    clap: Tensor
    midi: Tensor
    excitation: Tensor


class PredictiveMidiBrave(nn.Module):
    """Training model whose deployable continuation path excludes its encoder."""

    def __init__(self, model_config: ModelConfig, predictive_config: PredictiveConfig,
                 output_samples: int, sample_rate: int):
        super().__init__()
        if output_samples % predictive_config.samples_per_latent:
            raise ValueError("output_samples must be divisible by samples_per_latent")
        if predictive_config.clap_control_dim != model_config.timbre_dim:
            raise ValueError("CLAP control dimension must equal model timbre dimension")
        self.model_config = model_config
        self.predictive_config = predictive_config
        self.output_samples = output_samples
        self.samples_per_latent = predictive_config.samples_per_latent
        self.encoder = RaveEncoder(
            model_config.pqmf_bands, predictive_config.rave_latent_dim,
            model_config.ratios, model_config.capacity, model_config.pqmf_taps)
        self.clap_projection = TimbreAdapter(
            model_config.clap_dim, predictive_config.clap_control_dim)
        self.midi = MidiConditioner(model_config.midi_dim)
        self.predictor = MultiHorizonPredictor(
            predictive_config.rave_latent_dim, predictive_config.clap_control_dim,
            model_config.midi_dim, predictive_config.predictor_hidden_dim,
            predictive_config.history_frames, predictive_config.horizon_frames)
        self.rave_pitch_adversary = PitchAdversary(
            predictive_config.rave_latent_dim, classes=128)
        self.decoder = BraveDecoder(model_config, predictive_config.rave_latent_dim)
        self.excitation = HarmonicExcitation(
            sample_rate, model_config.excitation_harmonics, model_config.excitation_rms)
        self.stochastic_excitation = StochasticBandExcitation(
            model_config.pqmf_bands, sample_rate, model_config.pqmf_bands,
            model_config.stochastic_excitation_rms,
            model_config.stochastic_modulation_hz, model_config.stochastic_seed)

    def encode_audio(self, audio: Tensor, sample: bool = True) -> RavePosterior:
        return self.encoder(audio, sample=sample)

    def project_clap(self, clap: Tensor, frames: int) -> Tensor:
        if clap.ndim == 2:
            projected = self.clap_projection(clap)
            return projected.unsqueeze(-1).expand(-1, -1, frames)
        if clap.ndim != 3 or clap.shape[1] != self.model_config.clap_dim:
            raise ValueError("CLAP control must have shape [batch, clap_dim] or [batch, clap_dim, frames]")
        if clap.shape[-1] != frames:
            raise ValueError("CLAP control trajectory length does not match latent frames")
        sequence = F.normalize(clap.transpose(1, 2), dim=-1)
        return self.clap_projection.net(sequence).transpose(1, 2)

    def midi_control(self, note: Tensor, velocity: Tensor, frames: int) -> Tensor:
        return self.midi(note, velocity, frames, static_condition=False)

    def _excitation_bands(self, note: Tensor, samples: int,
                          seed: Tensor | None) -> Tensor:
        waveform = self.excitation(note, samples)
        if self.model_config.pqmf_dtype == "fp32":
            with torch.autocast(device_type=waveform.device.type, enabled=False):
                bands = self.decoder.pqmf.analysis(waveform.float())
        else:
            bands = self.decoder.pqmf.analysis(waveform)
        if self.model_config.stochastic_excitation:
            noise = self.stochastic_excitation(
                note.shape[0], bands.shape[-1], note.device, seed)
            bands = bands.float() + noise
        return bands

    def decode_latents(self, z_rave: Tensor, clap: Tensor, note: Tensor,
                       velocity: Tensor, excitation_seed: Tensor | None = None) -> Tensor:
        if z_rave.ndim != 3 or z_rave.shape[1] != self.predictive_config.rave_latent_dim:
            raise ValueError(
                f"RAVE latent must have {self.predictive_config.rave_latent_dim} channels")
        frames = z_rave.shape[-1]
        samples = frames * self.samples_per_latent
        z_clap = self.project_clap(clap, frames)
        z_midi = self.midi_control(note, velocity, frames)
        excitation = self._excitation_bands(note, samples, excitation_seed)
        return self.decoder(z_clap, z_midi, excitation, samples, z_rave=z_rave)

    def predict_future(self, history: Tensor, clap: Tensor, note: Tensor,
                       velocity: Tensor) -> LatentPrediction:
        frames = self.predictive_config.horizon_frames
        return self.predictor(
            history, self.project_clap(clap, frames),
            self.midi_control(note, velocity, frames))

    def rave_pitch_logits(self, latent: Tensor, reversal_scale: float = 1.0) -> Tensor:
        if latent.ndim != 3 or latent.shape[1] != self.predictive_config.rave_latent_dim:
            raise ValueError("RAVE latent shape does not match the pitch adversary")
        pooled = latent.mean(dim=-1)
        return self.rave_pitch_adversary(
            _GradientReverse.apply(pooled, reversal_scale))

    def forward_reconstruction(self, audio: Tensor, clap: Tensor, note: Tensor,
                               velocity: Tensor, sample_encoder: bool = True,
                               excitation_seed: Tensor | None = None,
                               ) -> PredictiveReconstruction:
        posterior = self.encode_audio(audio, sample=sample_encoder)
        frames = posterior.latent.shape[-1]
        z_clap = self.project_clap(clap, frames)
        z_midi = self.midi_control(note, velocity, frames)
        excitation = self._excitation_bands(note, audio.shape[-1], excitation_seed)
        generated = self.decoder(
            z_clap, z_midi, excitation, audio.shape[-1], z_rave=posterior.latent)
        return PredictiveReconstruction(
            generated, posterior, z_clap, z_midi, excitation)
