from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import Config
from .calibration import (calibrate_loss_weights, cap_auxiliary_gradient,
                          loss_gradient_norm)
from .data import PairDataset
from .losses import (BraveMultiScaleDiscriminator, ClapHealth,
                     ClapWaveformGradient,
                     FrozenClapReconstructionObjective,
                     MultiResolutionSTFTLoss, ReconstructionLoss,
                     SpectralPitchObjective,
                     discriminator_hinge, feature_matching, generator_adversarial)
from .model import ChannelRMSNorm, MidiBrave
from .latent_predictor import rollout_blocks
from .predictive_model import PredictiveMidiBrave
from .predictive_losses import (LatentStatistics, overlap_loss,
                                prediction_loss)


class PredictiveStage(str, Enum):
    RAVE = "rave"
    PREDICTOR = "predictor"
    ROLLOUT = "rollout"
    GAN = "gan"


@dataclass(frozen=True)
class PredictiveLossSchedule:
    rave_kl: float = 0.0
    rave_pitch_adversary: float = 0.0
    rollout: float = 0.0
    teacher_forcing: float = 1.0
    gan_adversarial: float = 0.0
    gan_feature_matching: float = 0.0


def counterfactual_control_assignment(
        preset_ids: list[str], notes: Tensor, velocities: Tensor,
        device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Split a batch and choose pitch-aligned cross-preset timbre targets."""
    count = len(preset_ids)
    if notes.shape != (count,) or velocities.shape != (count,):
        raise ValueError("counterfactual notes and velocities must match the batch")
    indices = torch.arange(count, device=device)
    midi = indices < (count + 1) // 2
    cross_preset = torch.tensor(
        [[source != candidate for candidate in preset_ids]
         for source in preset_ids], device=device, dtype=torch.bool)
    note = notes.to(device=device, dtype=torch.float32)
    velocity = velocities.to(device=device, dtype=torch.float32)
    # CLAP contains pitch as well as timbre. Prefer the same note, then the
    # closest velocity, so a timbre edit does not quietly fight fixed MIDI.
    score = (note[:, None] - note[None, :]).abs() * 256.0
    score = score + (velocity[:, None] - velocity[None, :]).abs()
    score = score.masked_fill(~cross_preset, torch.inf)
    best = score.argmin(dim=1)
    target = torch.where(cross_preset.any(dim=1), best, indices)
    timbre = ~midi & target.ne(indices)
    midi = ~timbre
    return midi, timbre, target


def midi_swap_example_weights(
        note_a: Tensor, note_b: Tensor, active: Tensor) -> Tensor:
    """Emphasize pitch extremes and large swaps without changing batch scale."""
    active_float = active.to(dtype=torch.float32)
    value = (1.0 + note_b.lt(48).to(torch.float32)
             + note_b.ge(60).to(torch.float32)
             + note_b.sub(note_a).abs().ge(12).to(torch.float32)) * active_float
    return value / value.sum().clamp_min(1.0) * active_float.sum().clamp_min(1.0)


@dataclass(frozen=True)
class PredictiveLatentObjective:
    total: Tensor
    components: dict[str, Tensor]
    prediction: Tensor


@dataclass(frozen=True)
class PredictorRolloutStabilityObjective:
    total: Tensor
    future: Tensor
    delta: Tensor
    acceleration: Tensor
    style: Tensor
    rollout: Tensor


@dataclass(frozen=True)
class _LatentStyleStatistics:
    mean: Tensor
    rms: Tensor
    delta_rms: Tensor
    covariance: Tensor


def _latent_style_statistics(
        latent: Tensor, statistics: LatentStatistics) -> _LatentStyleStatistics:
    if latent.ndim != 3 or latent.shape[-1] < 2:
        raise ValueError("latent style requires [batch, channels, frames>=2]")
    # Moment and covariance reductions overflow easily in fp16 once a free
    # rollout starts diverging. Preserve gradients while forcing the complete
    # reduction path to fp32 even when the caller is inside autocast.
    with torch.autocast(device_type=latent.device.type, enabled=False):
        value = latent.float()
        latent_scale, delta_scale, _ = statistics.scales(
            value.shape[1], value.device, value.dtype)
        normalized = value / latent_scale
        mean = normalized.mean(dim=-1)
        centered = normalized - mean[..., None]
        rms = centered.square().mean(dim=-1).add(1e-8).sqrt()
        delta = value.diff(dim=-1) / delta_scale
        delta_rms = delta.square().mean(dim=-1).add(1e-8).sqrt()
        covariance = (
            torch.matmul(centered, centered.transpose(1, 2)) / value.shape[-1]
        )
    return _LatentStyleStatistics(mean, rms, delta_rms, covariance)


def _latent_style_interpolation_loss(
        generated: Tensor, source: Tensor, target: Tensor, alpha: Tensor,
        statistics: LatentStatistics) -> Tensor:
    """Match order-independent RAVE style moments along a timbre path."""
    if generated.shape[0] != source.shape[0] or source.shape[0] != target.shape[0]:
        raise ValueError("latent style batches must align")
    if alpha.shape != (generated.shape[0],):
        raise ValueError("latent style alpha must have shape [batch]")
    generated_style = _latent_style_statistics(generated, statistics)
    source_style = _latent_style_statistics(source, statistics)
    target_style = _latent_style_statistics(target, statistics)
    blend = alpha.to(dtype=generated_style.mean.dtype)

    def distance(name: str) -> Tensor:
        actual = getattr(generated_style, name)
        start = getattr(source_style, name)
        end = getattr(target_style, name)
        shape = (blend.shape[0],) + (1,) * (actual.ndim - 1)
        desired = torch.lerp(start, end, blend.view(shape))
        return F.smooth_l1_loss(actual, desired)

    return (distance("mean") + distance("rms")
            + 0.25 * distance("delta_rms")
            + 0.1 * distance("covariance"))


def _latent_style_matching_loss(
        generated: Tensor, target: Tensor,
        statistics: LatentStatistics) -> Tensor:
    """Match same-recording moments without forcing framewise averaging."""
    if generated.shape != target.shape:
        raise ValueError("latent style matching requires equal shapes")
    generated_style = _latent_style_statistics(generated, statistics)
    target_style = _latent_style_statistics(target, statistics)
    return (
        F.smooth_l1_loss(generated_style.mean, target_style.mean)
        + F.smooth_l1_loss(generated_style.rms, target_style.rms)
        + 0.25 * F.smooth_l1_loss(
            generated_style.delta_rms, target_style.delta_rms)
        + 0.1 * F.smooth_l1_loss(
            generated_style.covariance, target_style.covariance)
    )


def _seed_washout_loss(
        first: Tensor, second: Tensor, statistics: LatentStatistics,
        window_frames: int, maximum_ratio: float = 0.9) -> tuple[Tensor, Tensor]:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("seed washout rollouts must share [batch, channels, frames]")
    if not 0 < window_frames <= first.shape[-1] // 2:
        raise ValueError("seed washout window must fit twice inside the rollout")
    if not 0.0 < maximum_ratio <= 1.0:
        raise ValueError("seed washout ratio must be in (0,1]")
    with torch.autocast(device_type=first.device.type, enabled=False):
        first_value = first.float()
        second_value = second.float()
        latent_scale, _, _ = statistics.scales(
            first_value.shape[1], first_value.device, first_value.dtype)
        normalized_difference = (first_value - second_value) / latent_scale
        # vector_norm defines the zero-vector subgradient as zero. A direct
        # sqrt(mean(square)) has an infinite derivative at exact seed
        # agreement and produces 0 * inf = NaN through the washout hinge.
        distance = (
            torch.linalg.vector_norm(normalized_difference, dim=1)
            / math.sqrt(first_value.shape[1])
        )
        early = distance[..., :window_frames].mean(dim=-1)
        late = distance[..., -window_frames:].mean(dim=-1)
        loss = F.relu(late - maximum_ratio * early).mean()
        ratio = (late / early.clamp_min(1e-8)).mean()
    return loss, ratio


@dataclass(frozen=True)
class PredictiveStageObjective:
    total: Tensor
    components: dict[str, Tensor]
    generated_audio: Tensor | None = None
    target_audio: Tensor | None = None
    diagnostic_tensors: dict[str, Tensor] | None = None


@dataclass(frozen=True)
class PredictiveControlObjective:
    total: Tensor
    components: dict[str, Tensor]
    diagnostic_tensors: dict[str, Tensor]


@dataclass(frozen=True)
class PredictorCounterfactualRollout:
    audio: Tensor
    rollout: Tensor
    latent_sequence: Tensor
    midi_mask: Tensor
    timbre_mask: Tensor
    target_index: Tensor
    target_clap: Tensor
    source_clap: Tensor
    target_note: Tensor
    timbre_alpha: Tensor


def predictor_counterfactual_rollout(
    model: PredictiveMidiBrave, batch: dict[str, Any], config: Config,
) -> PredictorCounterfactualRollout:
    """Roll out one causal future under disjoint MIDI and timbre edits."""
    if config.predictive is None:
        raise ValueError("predictor counterfactual rollout requires a v3 config")
    required = {"rave_a", "clap_a", "note_a", "note_b", "velocity_a", "preset_id"}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"predictor counterfactual batch is missing: {sorted(missing)}")
    predictive = config.predictive
    source_latent = batch["rave_a"]
    if (source_latent.ndim != 3
            or source_latent.shape[1] != predictive.rave_latent_dim
            or source_latent.shape[-1] < predictive.history_frames):
        raise ValueError("rave_a does not contain a valid predictor history")
    history = source_latent[..., :predictive.history_frames]
    midi_mask, timbre_mask, target_index = counterfactual_control_assignment(
        list(batch["preset_id"]), batch["note_a"], batch["velocity_a"],
        history.device)
    target_clap = batch["clap_a"].clone()
    timbre_alpha = target_clap.new_zeros(target_clap.shape[0])
    timbre_positions = timbre_mask.nonzero().flatten()
    if timbre_positions.numel():
        interpolation_steps = predictive.predictor_timbre_interpolation_steps
        alpha = ((torch.arange(timbre_positions.numel(), device=history.device)
                  % interpolation_steps) + 1).to(target_clap) / interpolation_steps
        timbre_alpha[timbre_positions] = alpha
        source = F.normalize(
            batch["clap_a"].index_select(0, timbre_positions).float(), dim=-1)
        target = F.normalize(batch["clap_a"].index_select(
            0, target_index[timbre_positions]).float(), dim=-1)
        interpolated = F.normalize(
            torch.lerp(source, target, alpha[:, None].float()), dim=-1)
        target_clap[timbre_positions] = interpolated.to(target_clap)
    target_note = torch.where(midi_mask, batch["note_b"], batch["note_a"])
    frames = predictive.predictor_control_rollout_frames
    result = rollout_blocks(
        model.predictor, history, model.project_clap(target_clap, frames),
        model.midi_control(target_note, batch["velocity_a"], frames),
        predictive.stride_frames)
    sequence = torch.cat((history, result.latent), dim=-1)
    audio = model.decode_latents(
        sequence, target_clap, target_note, batch["velocity_a"],
        batch.get("excitation_seed_a"))
    return PredictorCounterfactualRollout(
        audio=audio, rollout=result.latent, latent_sequence=sequence,
        midi_mask=midi_mask, timbre_mask=timbre_mask, target_index=target_index,
        target_clap=target_clap, source_clap=batch["clap_a"],
        target_note=target_note, timbre_alpha=timbre_alpha)


def _predictor_rollout_stability_objective(
        model: PredictiveMidiBrave, batch: dict[str, Tensor], config: Config,
        statistics: LatentStatistics) -> PredictorRolloutStabilityObjective:
    """Free-run against only the same recording's cached future."""
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("predictor rollout stability requires a v3 config")
    predictive = config.predictive
    latent = batch["rave_a"]
    history = latent[..., :predictive.history_frames]
    available = latent.shape[-1] - predictive.history_frames
    frames = min(predictive.predictor_control_rollout_frames, available)
    frames = frames // predictive.stride_frames * predictive.stride_frames
    if frames < predictive.stride_frames:
        raise ValueError("cached RAVE window is too short for predictor stability")
    result = rollout_blocks(
        model.predictor, history, model.project_clap(batch["clap_a"], frames),
        model.midi_control(batch["note_a"], batch["velocity_a"], frames),
        predictive.stride_frames)
    target = latent[..., predictive.history_frames:predictive.history_frames + frames]
    with torch.autocast(device_type=latent.device.type, enabled=False):
        losses = prediction_loss(
            result.latent.float(), target.float(), history.float(), statistics,
            discount=1.0)
    weights = config.latent_loss
    total = (weights.future * losses.future + weights.delta * losses.delta
             + weights.acceleration * losses.acceleration)
    style = (_latent_style_matching_loss(result.latent, target, statistics)
             if weights.predictor_rollout_style > 0.0 else total.new_zeros(()))
    return PredictorRolloutStabilityObjective(
        total, losses.future, losses.delta, losses.acceleration, style,
        result.latent)


def _linear_warmup(update: int, updates: int) -> float:
    if update < 0:
        raise ValueError("update must be non-negative")
    return min(1.0, update / max(1, updates))


def _predictor_control_active(config: Config, update: int) -> bool:
    if config.predictive is None:
        return False
    predictive = config.predictive
    return (update >= predictive.predictor_control_start_updates
            and (update - predictive.predictor_control_start_updates)
            % predictive.predictor_control_every_updates == 0)


def _empty_predictive_control_components(zero: Tensor) -> dict[str, Tensor]:
    return {
        "predictor_midi_swap_pitch": zero,
        "predictor_midi_swap_source_rejection": zero,
        "predictor_midi_control_total": zero,
        "predictor_timbre_pitch_preservation": zero,
        "predictor_pitch_control_total": zero,
        "predictor_midi_style_preservation": zero,
        "predictor_timbre_style_control": zero,
        "predictor_seed_washout": zero,
        "predictor_seed_washout_ratio": zero,
        "predictor_latent_style_control_total": zero,
        "predictor_midi_clap_preservation_total": zero,
        "predictor_midi_clap_preservation_cosine": zero,
        "predictor_clap_control_total": zero,
        "predictor_clap_target_cosine": zero,
        "predictor_clap_source_cosine": zero,
        "predictor_clap_following": zero,
        "predictor_control_total": zero,
        "predictor_control_gradient_fraction_limit": zero,
        "predictor_control_gradient_scale": zero,
    }


def _predictive_control_objective(
    model: PredictiveMidiBrave, batch: dict[str, Any], config: Config,
    update: int, statistics: LatentStatistics, primary_total: Tensor,
    gradient_parameters: list[Tensor], swap_pitch: nn.Module | None,
    clap_objective: FrozenClapReconstructionObjective | None,
) -> PredictiveControlObjective:
    """Apply the same bounded MIDI/CLAP controls in predictor and rollout stages."""
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("predictive controls require a v3 config")
    if swap_pitch is None:
        raise ValueError("predictive MIDI control requires a pitch objective")
    required = {"note_b", "pitch_confidence_a", "pitch_valid_mask_a", "preset_id"}
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"predictive control batch is missing: {sorted(missing)}")
    losses = config.latent_loss
    zero = primary_total.new_zeros(())
    counterfactual = predictor_counterfactual_rollout(model, batch, config)
    midi_mask = counterfactual.midi_mask
    changed = batch["note_a"].ne(batch["note_b"]) & midi_mask
    weights = midi_swap_example_weights(
        batch["note_a"], batch["note_b"], changed)
    pitch = swap_pitch(
        counterfactual.audio[midi_mask], batch["note_b"][midi_mask],
        batch["pitch_confidence_a"][midi_mask]
        * changed[midi_mask, None].to(batch["pitch_confidence_a"])
        * weights[midi_mask, None].to(batch["pitch_confidence_a"]),
        batch["pitch_valid_mask_a"][midi_mask])
    source_rejection = swap_pitch.source_rejection(
        counterfactual.audio[midi_mask], batch["note_b"][midi_mask],
        batch["note_a"][midi_mask], batch["pitch_confidence_a"][midi_mask],
        batch["pitch_valid_mask_a"][midi_mask], weights[midi_mask])
    midi_control = (
        config.loss.cross_pitch * config.loss.analytic_pitch * pitch
        + losses.rave_swap_source_rejection * source_rejection)
    timbre_mask = counterfactual.timbre_mask
    timbre_pitch = zero
    if bool(timbre_mask.any().item()):
        timbre_pitch = swap_pitch(
            counterfactual.audio[timbre_mask], batch["note_a"][timbre_mask],
            batch["pitch_confidence_a"][timbre_mask],
            batch["pitch_valid_mask_a"][timbre_mask])
    pitch_control = (
        midi_control + config.loss.cross_pitch
        * config.loss.analytic_pitch * timbre_pitch)
    midi_style = zero
    if losses.predictor_midi_style > 0.0 and bool(midi_mask.any().item()):
        source = batch["rave_a"][midi_mask]
        midi_style = _latent_style_interpolation_loss(
            counterfactual.rollout[midi_mask], source, source,
            torch.zeros(source.shape[0], device=source.device, dtype=source.dtype),
            statistics)
    timbre_style = zero
    seed_washout = zero
    seed_washout_ratio = zero
    if bool(timbre_mask.any().item()) and (
            losses.predictor_timbre_style > 0.0
            or losses.predictor_seed_washout > 0.0):
        target_latent = batch["rave_a"].index_select(
            0, counterfactual.target_index)[timbre_mask]
        source_latent = batch["rave_a"][timbre_mask]
        if losses.predictor_timbre_style > 0.0:
            timbre_style = _latent_style_interpolation_loss(
                counterfactual.rollout[timbre_mask], source_latent,
                target_latent, counterfactual.timbre_alpha[timbre_mask],
                statistics)
        if losses.predictor_seed_washout > 0.0:
            frames = config.predictive.predictor_control_rollout_frames
            alternate = rollout_blocks(
                model.predictor,
                target_latent[..., :config.predictive.history_frames],
                model.project_clap(
                    counterfactual.target_clap[timbre_mask], frames),
                model.midi_control(
                    counterfactual.target_note[timbre_mask],
                    batch["velocity_a"][timbre_mask], frames),
                config.predictive.stride_frames).latent
            seed_washout, seed_washout_ratio = _seed_washout_loss(
                counterfactual.rollout[timbre_mask], alternate, statistics,
                window_frames=min(
                    config.predictive.history_frames, frames // 2))
    latent_style_control = (
        losses.predictor_midi_style * midi_style
        + losses.predictor_timbre_style * timbre_style
        + losses.predictor_seed_washout * seed_washout)
    midi_clap_loss = zero
    midi_clap_cosine = zero
    timbre_clap_loss = zero
    clap_surrogate = zero
    clap_health = None
    clap_target_cosine = zero
    clap_source_cosine = zero
    clap_following = zero
    if losses.clap_control > 0.0 or losses.clap_counterfactual > 0.0:
        if clap_objective is None:
            raise ValueError("predictive CLAP control requires a frozen CLAP objective")
        if losses.clap_control > 0.0 and bool(midi_mask.any().item()):
            midi_audio = counterfactual.audio[midi_mask]
            valid = torch.full(
                (midi_audio.shape[0],), midi_audio.shape[-1],
                device=midi_audio.device, dtype=torch.long)
            preservation = clap_objective.waveform_gradients_to_embeddings(
                midi_audio, counterfactual.source_clap[midi_mask],
                counterfactual.source_clap[midi_mask], valid, margin=0.0)
            midi_clap_loss = preservation.losses.mean()
            midi_clap_cosine = preservation.target_cosine.mean()
            health = _clap_health_vector(preservation.health, midi_audio.device)
            clap_health = health if clap_health is None else _merge_clap_health(
                clap_health, health)
            if not bool(preservation.health.skipped.item()):
                requested = (preservation.gradients * losses.clap_control
                             / midi_audio.shape[0])
                clap_surrogate = (
                    clap_surrogate
                    + (midi_audio.float() * requested.float()).sum())
        if losses.clap_counterfactual > 0.0 and bool(timbre_mask.any().item()):
            timbre_audio = counterfactual.audio[timbre_mask]
            valid = torch.full(
                (timbre_audio.shape[0],), timbre_audio.shape[-1],
                device=timbre_audio.device, dtype=torch.long)
            clap_result = clap_objective.waveform_gradients_to_embeddings(
                timbre_audio, counterfactual.target_clap[timbre_mask],
                counterfactual.source_clap[timbre_mask], valid)
            timbre_clap_loss = clap_result.losses.mean()
            clap_target_cosine = clap_result.target_cosine.mean()
            clap_source_cosine = clap_result.source_cosine.mean()
            clap_following = clap_result.following.float().mean()
            health = _clap_health_vector(clap_result.health, timbre_audio.device)
            clap_health = health if clap_health is None else _merge_clap_health(
                clap_health, health)
            if not bool(clap_result.health.skipped.item()):
                requested = (clap_result.gradients * losses.clap_counterfactual
                             / timbre_audio.shape[0])
                clap_surrogate = (
                    clap_surrogate
                    + (timbre_audio.float() * requested.float()).sum())
    weighted_midi_clap = losses.clap_control * midi_clap_loss
    weighted_timbre_clap = losses.clap_counterfactual * timbre_clap_loss
    weighted_clap = weighted_midi_clap + weighted_timbre_clap
    control_auxiliary = (pitch_control + latent_style_control
                         + clap_surrogate - clap_surrogate.detach())
    control_progress = _linear_warmup(
        update - config.predictive.predictor_control_start_updates + 1,
        config.predictive.predictor_control_warmup_updates)
    gradient_fraction = (
        config.predictive.predictor_control_gradient_fraction_max
        * control_progress)
    capped_control, control_scale = _cap_auxiliary_parameter_gradient(
        control_auxiliary, primary_total, gradient_parameters, gradient_fraction)
    displayed_control = (pitch_control.detach() + latent_style_control.detach()
                         + weighted_clap.detach())
    control_total = capped_control - capped_control.detach() + displayed_control
    components = {
        "predictor_midi_swap_pitch": pitch,
        "predictor_midi_swap_source_rejection": source_rejection,
        "predictor_midi_control_total": midi_control,
        "predictor_timbre_pitch_preservation": timbre_pitch,
        "predictor_pitch_control_total": pitch_control,
        "predictor_midi_style_preservation": midi_style,
        "predictor_timbre_style_control": timbre_style,
        "predictor_seed_washout": seed_washout,
        "predictor_seed_washout_ratio": seed_washout_ratio,
        "predictor_latent_style_control_total": latent_style_control,
        "predictor_midi_clap_preservation_total": weighted_midi_clap,
        "predictor_midi_clap_preservation_cosine": midi_clap_cosine,
        "predictor_clap_control_total": weighted_clap,
        "predictor_clap_target_cosine": clap_target_cosine,
        "predictor_clap_source_cosine": clap_source_cosine,
        "predictor_clap_following": clap_following,
        "predictor_control_total": displayed_control,
        "predictor_control_gradient_fraction_limit": zero.new_tensor(
            gradient_fraction),
        "predictor_control_gradient_scale": control_scale,
    }
    diagnostics = {
        "counterfactual_audio": counterfactual.audio,
        "counterfactual_latent_rollout": counterfactual.rollout,
        "counterfactual_latent_sequence": counterfactual.latent_sequence,
        "counterfactual_midi_mask": counterfactual.midi_mask,
        "counterfactual_timbre_mask": counterfactual.timbre_mask,
        "counterfactual_target_index": counterfactual.target_index,
        "counterfactual_target_clap": counterfactual.target_clap,
        "counterfactual_source_clap": counterfactual.source_clap,
        "counterfactual_target_note": counterfactual.target_note,
        "counterfactual_timbre_alpha": counterfactual.timbre_alpha,
        **({"predictor_control_clap_health": clap_health}
           if clap_health is not None else {}),
    }
    return PredictiveControlObjective(control_total, components, diagnostics)


def _should_log_predictive_component(name: str, regular_interval: bool,
                                     control_active: bool) -> bool:
    """Keep sparse control metrics off inactive zero-valued updates."""
    continuous = (name == "predictor_continuation_total"
                  or name.startswith("predictor_rollout_stability_"))
    sparse_control = name.startswith("predictor_") and not continuous
    return control_active if sparse_control else regular_interval


def predictive_loss_schedule(config: Config, stage: PredictiveStage | str,
                             update: int) -> PredictiveLossSchedule:
    """Return stage-local loss weights; counters restart at each stage."""
    stage = PredictiveStage(stage)
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("predictive loss scheduling requires a v3 config")
    predictive = config.predictive
    losses = config.latent_loss
    if stage is PredictiveStage.RAVE:
        return PredictiveLossSchedule(
            rave_kl=losses.rave_kl * _linear_warmup(
                update, predictive.kl_warmup_updates),
            rave_pitch_adversary=losses.rave_pitch_adversary * _linear_warmup(
                update, predictive.pitch_adversary_warmup_updates),
        )
    if stage is PredictiveStage.ROLLOUT:
        progress = _linear_warmup(update, predictive.rollout_warmup_updates)
        return PredictiveLossSchedule(
            rollout=losses.rollout * progress,
            teacher_forcing=1.0 - progress * (1.0 - predictive.teacher_forcing_floor),
        )
    if stage is PredictiveStage.GAN:
        return PredictiveLossSchedule(
            teacher_forcing=predictive.teacher_forcing_floor,
            gan_adversarial=losses.gan_adversarial,
            gan_feature_matching=losses.gan_feature_matching,
        )
    return PredictiveLossSchedule()


def configure_predictive_stage(model: PredictiveMidiBrave,
                               stage: PredictiveStage | str,
                               rollout_gate_passed: bool = False) -> set[str]:
    """Freeze every module outside the ownership boundary of one stage."""
    stage = PredictiveStage(stage)
    owned = {
        PredictiveStage.RAVE: {
            "encoder", "rave_pitch_adversary", "clap_projection", "midi", "decoder"},
        PredictiveStage.PREDICTOR: {"predictor"},
        PredictiveStage.ROLLOUT: {"predictor", "clap_projection", "midi", "decoder"},
        PredictiveStage.GAN: {"decoder", "predictor"} if rollout_gate_passed else {"decoder"},
    }[stage]
    trainable = set()
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        parameter.requires_grad_(root in owned)
        if parameter.requires_grad:
            trainable.add(name)
    return trainable


def predictive_latent_objective(
    model: PredictiveMidiBrave, batch: dict[str, Tensor], config: Config,
    statistics: LatentStatistics,
    calibrated_weights: dict[str, float] | None = None,
) -> PredictiveLatentObjective:
    """Direct K-step supervision from one recording, with block overlap."""
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("latent prediction requires a v3 config")
    p = config.predictive
    losses = config.latent_loss
    latent = batch["rave_a"]
    required = p.history_frames + p.horizon_frames
    if latent.ndim != 3 or latent.shape[1] != p.rave_latent_dim or latent.shape[-1] < required:
        raise ValueError(
            f"cached RAVE latent must contain at least {required} aligned frames")
    history = latent[..., :p.history_frames]
    target = latent[..., p.history_frames:required]
    result = model.predict_future(
        history, batch["clap_a"], batch["note_a"], batch["velocity_a"])
    terms = prediction_loss(
        result.latent, target, history, statistics, losses.horizon_discount)
    components = {
        "future": terms.future,
        "delta": terms.delta,
        "acceleration": terms.acceleration,
    }
    second_stop = p.history_frames + p.stride_frames + p.horizon_frames
    if latent.shape[-1] >= second_stop and p.stride_frames < p.horizon_frames:
        shifted_history = latent[..., p.stride_frames:p.history_frames + p.stride_frames]
        shifted = model.predict_future(
            shifted_history, batch["clap_a"], batch["note_a"], batch["velocity_a"])
        components["overlap"] = overlap_loss(
            result.latent, shifted.latent, p.stride_frames, statistics)
    else:
        components["overlap"] = result.latent.new_zeros(())
    weights = calibrated_weights or {
        "future": losses.future, "delta": losses.delta,
        "acceleration": losses.acceleration, "overlap": losses.overlap,
    }
    if set(weights) != set(components):
        raise ValueError("calibrated latent weights must cover every prediction loss")
    total = sum(weights[name] * value for name, value in components.items())
    return PredictiveLatentObjective(total, components, result.latent)


def predictive_stage_objective(
    model: PredictiveMidiBrave, batch: dict[str, Tensor], config: Config,
    stage: PredictiveStage | str, update: int,
    statistics: LatentStatistics | None = None,
    calibrated_weights: dict[str, float] | None = None,
    stft: MultiResolutionSTFTLoss | None = None,
    swap_pitch: nn.Module | None = None,
    clap_objective: FrozenClapReconstructionObjective | None = None,
) -> PredictiveStageObjective:
    """Compute one generator objective for any pre-GAN predictive stage."""
    stage = PredictiveStage(stage)
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("predictive stage objective requires a v3 config")
    schedule = predictive_loss_schedule(config, stage, update)
    losses = config.latent_loss
    if stage is PredictiveStage.RAVE:
        reconstruction = model.forward_reconstruction(
            batch["audio_a"], batch["clap_a"], batch["note_a"],
            batch["velocity_a"], sample_encoder=True,
            excitation_seed=batch.get("excitation_seed_a"))
        waveform = F.l1_loss(reconstruction.audio, batch["audio_a"])
        spectral = (stft(reconstruction.audio, batch["audio_a"], batch.get("valid_samples_a"))
                    if stft is not None else waveform.new_zeros(()))
        pitch = F.cross_entropy(
            model.rave_pitch_logits(reconstruction.posterior.latent),
            batch["note_a"].long().clamp(0, 127))
        components = {
            "waveform": waveform, "spectral": spectral,
            "rave_kl": reconstruction.posterior.kl,
            "rave_pitch_adversary": pitch,
        }
        midi_swap_audio = None
        counterfactual_diagnostics: dict[str, Tensor] = {}
        if config.loss.cross_pitch > 0.0:
            if swap_pitch is None:
                raise ValueError("RAVE MIDI swap supervision requires a pitch objective")
            required = {"note_b", "pitch_confidence_a", "pitch_valid_mask_a"}
            missing = required - batch.keys()
            if missing:
                raise ValueError(f"RAVE MIDI swap batch is missing: {sorted(missing)}")
            use_mixed = (losses.rave_swap_source_rejection > 0.0
                         or losses.clap_counterfactual > 0.0)
            if use_mixed:
                if "preset_id" not in batch:
                    raise ValueError("counterfactual controls require preset_id")
                midi_mask, timbre_mask, target_index = counterfactual_control_assignment(
                    list(batch["preset_id"]), batch["note_a"],
                    batch["velocity_a"], reconstruction.audio.device)
                clap = batch["clap_a"].clone()
                clap[timbre_mask] = batch["clap_a"].index_select(
                    0, target_index[timbre_mask])
                note = torch.where(midi_mask, batch["note_b"], batch["note_a"])
                counterfactual_audio = model.decode_latents(
                    reconstruction.posterior.latent, clap, note,
                    batch["velocity_a"], batch.get("excitation_seed_a"))
                changed = batch["note_a"].ne(batch["note_b"]) & midi_mask
                weights = midi_swap_example_weights(
                    batch["note_a"], batch["note_b"], changed)
                components["midi_swap_pitch"] = swap_pitch(
                    counterfactual_audio[midi_mask], batch["note_b"][midi_mask],
                    batch["pitch_confidence_a"][midi_mask]
                    * changed[midi_mask, None].to(batch["pitch_confidence_a"])
                    * weights[midi_mask, None].to(batch["pitch_confidence_a"]),
                    batch["pitch_valid_mask_a"][midi_mask])
                components["midi_swap_source_rejection"] = swap_pitch.source_rejection(
                    counterfactual_audio[midi_mask], batch["note_b"][midi_mask],
                    batch["note_a"][midi_mask],
                    batch["pitch_confidence_a"][midi_mask],
                    batch["pitch_valid_mask_a"][midi_mask], weights[midi_mask])
                counterfactual_diagnostics = {
                    "counterfactual_audio": counterfactual_audio,
                    "counterfactual_midi_mask": midi_mask,
                    "counterfactual_timbre_mask": timbre_mask,
                    "counterfactual_target_index": target_index,
                    "counterfactual_target_clap": clap,
                    "counterfactual_source_clap": batch["clap_a"],
                }
            else:
                midi_swap_audio = model.decode_latents(
                    reconstruction.posterior.latent, batch["clap_a"], batch["note_b"],
                    batch["velocity_a"], batch.get("excitation_seed_a"))
                changed = batch["note_a"].ne(batch["note_b"]).to(
                    batch["pitch_confidence_a"].dtype)[:, None]
                components["midi_swap_pitch"] = swap_pitch(
                    midi_swap_audio, batch["note_b"],
                    batch["pitch_confidence_a"] * changed,
                    batch["pitch_valid_mask_a"])
                components["midi_swap_source_rejection"] = waveform.new_zeros(())
        else:
            components["midi_swap_pitch"] = waveform.new_zeros(())
            components["midi_swap_source_rejection"] = waveform.new_zeros(())
        total = (waveform + config.loss.self_stft * spectral
                 + schedule.rave_kl * components["rave_kl"]
                 + schedule.rave_pitch_adversary * pitch
                 + config.loss.cross_pitch * config.loss.analytic_pitch
                 * components["midi_swap_pitch"]
                 + losses.rave_swap_source_rejection
                 * components["midi_swap_source_rejection"])
        return PredictiveStageObjective(
            total, components, reconstruction.audio, batch["audio_a"], {
                "input_audio": batch["audio_a"],
                "input_clap": batch["clap_a"],
                "rave_mean": reconstruction.posterior.mean,
                "rave_logvar": reconstruction.posterior.logvar,
                "rave_latent": reconstruction.posterior.latent,
                "clap_control": reconstruction.clap,
                "midi_control": reconstruction.midi,
                "excitation": reconstruction.excitation,
                "decoder_audio": reconstruction.audio,
                **({"midi_swap_audio": midi_swap_audio}
                   if midi_swap_audio is not None else {}),
                **counterfactual_diagnostics,
            })
    if statistics is None:
        raise ValueError(f"{stage.value} stage requires latent statistics")
    latent_objective = predictive_latent_objective(
        model, batch, config, statistics, calibrated_weights)
    if stage is PredictiveStage.PREDICTOR:
        components = dict(latent_objective.components)
        zero = latent_objective.total.new_zeros(())
        stability = None
        primary_total = latent_objective.total
        if (losses.predictor_rollout_stability > 0.0
                or losses.predictor_rollout_style > 0.0):
            stability = _predictor_rollout_stability_objective(
                model, batch, config, statistics)
            primary_total = (primary_total
                             + losses.predictor_rollout_stability * stability.total
                             + losses.predictor_rollout_style * stability.style)
        components.update({
            "predictor_continuation_total": latent_objective.total,
            "predictor_rollout_stability_total": (
                stability.total if stability is not None else zero),
            "predictor_rollout_stability_future": (
                stability.future if stability is not None else zero),
            "predictor_rollout_stability_delta": (
                stability.delta if stability is not None else zero),
            "predictor_rollout_stability_acceleration": (
                stability.acceleration if stability is not None else zero),
            "predictor_rollout_stability_style": (
                stability.style if stability is not None else zero),
        })
        components.update(_empty_predictive_control_components(zero))
        if not _predictor_control_active(config, update):
            return PredictiveStageObjective(primary_total, components)
        control = _predictive_control_objective(
            model, batch, config, update, statistics, primary_total,
            list(model.predictor.parameters()), swap_pitch, clap_objective)
        components.update(control.components)
        return PredictiveStageObjective(
            primary_total + control.total, components,
            diagnostic_tensors=control.diagnostic_tensors)

    p = config.predictive
    latent = batch["rave_a"]
    history = latent[..., :p.history_frames]
    available = latent.shape[-1] - p.history_frames
    rollout_frames = (available // p.stride_frames) * p.stride_frames
    if rollout_frames <= 0:
        raise ValueError("cached RAVE window is too short for rollout")
    clap_future = model.project_clap(batch["clap_a"], rollout_frames)
    midi_future = model.midi_control(
        batch["note_a"], batch["velocity_a"], rollout_frames)
    current = history
    chunks = []
    for start in range(0, rollout_frames, p.stride_frames):
        stop = min(rollout_frames, start + p.horizon_frames)
        clap_window = clap_future[..., start:stop]
        midi_window = midi_future[..., start:stop]
        if clap_window.shape[-1] < p.horizon_frames:
            pad = p.horizon_frames - clap_window.shape[-1]
            clap_window = torch.cat(
                (clap_window, clap_window[..., -1:].expand(-1, -1, pad)), dim=-1)
            midi_window = torch.cat(
                (midi_window, midi_window[..., -1:].expand(-1, -1, pad)), dim=-1)
        prediction = model.predictor(current, clap_window, midi_window).latent
        consumed = prediction[..., :p.stride_frames]
        chunks.append(consumed)
        true = latent[..., p.history_frames + start:
                      p.history_frames + start + p.stride_frames]
        mixed = schedule.teacher_forcing * true + (1.0 - schedule.teacher_forcing) * consumed
        current = torch.cat((current, mixed), dim=-1)[..., -p.history_frames:]
    rollout = torch.cat(chunks, dim=-1)
    target_rollout = latent[..., p.history_frames:p.history_frames + rollout_frames]
    latent_scale, _, _ = statistics.scales(
        latent.shape[1], latent.device, latent.dtype)
    rollout_error = F.smooth_l1_loss(
        (rollout - target_rollout) / latent_scale,
        torch.zeros_like(rollout))
    sequence = torch.cat((history, rollout), dim=-1)
    generated = model.decode_latents(
        sequence, batch["clap_a"], batch["note_a"], batch["velocity_a"],
        batch.get("excitation_seed_a"))
    samples = sequence.shape[-1] * p.samples_per_latent
    target = batch["audio_a"][..., :samples]
    future_start = p.history_frames * p.samples_per_latent
    generated_future = generated[..., future_start:]
    target_future = target[..., future_start:]
    audio = F.l1_loss(generated_future, target_future)
    spectral = (stft(generated_future, target_future)
                if stft is not None else audio.new_zeros(()))
    components = dict(latent_objective.components)
    components.update({"rollout": rollout_error, "predicted_audio": audio,
                       "predicted_spectral": spectral})
    primary_total = (
        latent_objective.total + schedule.rollout * rollout_error
        + losses.predicted_audio * (audio + config.loss.self_stft * spectral))
    zero = primary_total.new_zeros(())
    components.update(_empty_predictive_control_components(zero))
    diagnostics = None
    # Direct objective callers may omit the optional analytic-pitch helper.
    # The production trainer always supplies it for rollout, which enables the
    # inherited counterfactual controls while preserving the lightweight API.
    if _predictor_control_active(config, update) and swap_pitch is not None:
        control = _predictive_control_objective(
            model, batch, config, update, statistics, primary_total,
            [parameter for parameter in model.parameters()
             if parameter.requires_grad], swap_pitch, clap_objective)
        primary_total = primary_total + control.total
        components.update(control.components)
        diagnostics = control.diagnostic_tensors
    return PredictiveStageObjective(
        primary_total, components, generated_future, target_future, diagnostics)


def predictive_checkpoint_contract(config: Config, stage: PredictiveStage | str,
                                   latent_statistics_hash: str | None,
                                   calibration_hash: str | None,
                                   rollout_gate_passed: bool) -> dict[str, Any]:
    if config.predictive is None:
        raise ValueError("predictive checkpoint contract requires a v3 config")
    predictive = config.predictive
    manifest = Path(config.data.manifest)
    return {
        "architecture": predictive.architecture,
        "stage": PredictiveStage(stage).value,
        "encoder_frozen": PredictiveStage(stage) is not PredictiveStage.RAVE,
        "rollout_gate_passed": bool(rollout_gate_passed),
        "latent_statistics_hash": latent_statistics_hash,
        "calibration_hash": calibration_hash,
        "history_frames": predictive.history_frames,
        "horizon_frames": predictive.horizon_frames,
        "stride_frames": predictive.stride_frames,
        "config_hash": _artifact_sha256(config.source_path),
        "manifest_hash": _artifact_sha256(manifest) if manifest.is_file() else None,
    }


def validate_predictive_resume(payload: dict[str, Any],
                               expected_contract: dict[str, Any]) -> None:
    if int(payload.get("format", 0)) != 5:
        raise ValueError("predictive exact resume requires checkpoint format 5")
    actual = payload.get("predictive_contract")
    if not isinstance(actual, dict):
        raise ValueError("checkpoint is missing predictive contract")
    for key, expected in expected_contract.items():
        if actual.get(key) != expected:
            label = key.replace("_", " ")
            raise ValueError(
                f"checkpoint {label} mismatch: {actual.get(key)!r} != {expected!r}")


def save_predictive_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, contract: dict[str, Any], *,
    stage_update: int, epoch: int, microbatch_offset: int,
    scheduler_state: dict[str, Any] | None = None,
    sampler_state: dict[str, Any] | None = None,
    batch_per_gpu: int | None = None,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Atomically save every state required for an exact v3 stage resume."""
    if stage_update < 0 or epoch < 0 or microbatch_offset < 0:
        raise ValueError("predictive checkpoint counters must be non-negative")
    if batch_per_gpu is not None and batch_per_gpu <= 0:
        raise ValueError("predictive checkpoint batch_per_gpu must be positive")
    local_rng = _rng_state()
    if dist.is_available() and dist.is_initialized():
        rng_states: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rng_states, local_rng)
        if dist.get_rank() != 0:
            return
    else:
        rng_states = [local_rng]
    payload: dict[str, Any] = {
        "format": 5,
        "predictive_contract": dict(contract),
        "stage_update": stage_update,
        "epoch": epoch,
        "microbatch_offset": microbatch_offset,
        "world_size": (dist.get_world_size()
                       if dist.is_available() and dist.is_initialized() else 1),
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler_state": scheduler_state or {},
        "sampler_state": sampler_state or {},
        "rng_by_rank": rng_states,
    }
    if batch_per_gpu is not None:
        payload["batch_per_gpu"] = int(batch_per_gpu)
    if discriminator is not None:
        payload["discriminator"] = unwrap(discriminator).state_dict()
        payload["discriminator_arch"] = getattr(
            unwrap(discriminator), "architecture_id", type(unwrap(discriminator)).__name__)
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_predictive_checkpoint(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, expected_contract: dict[str, Any],
    discriminator: nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    allow_world_size_change: bool = False,
    batch_size_changed: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("predictive checkpoint must be a mapping")
    validate_predictive_resume(payload, expected_contract)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    checkpoint_world_size = int(payload.get("world_size", 1))
    world_size_changed = checkpoint_world_size != world_size
    if world_size_changed and not allow_world_size_change:
        raise ValueError("predictive exact resume requires the same DDP world size")
    if discriminator is not None:
        missing = {"discriminator", "discriminator_optimizer", "discriminator_arch"} - payload.keys()
        if missing or discriminator_optimizer is None:
            raise ValueError(f"GAN exact resume is missing state: {sorted(missing)}")
        expected_arch = getattr(unwrap(discriminator), "architecture_id", None)
        if payload["discriminator_arch"] != expected_arch:
            raise ValueError("checkpoint discriminator architecture mismatch")
    unwrap(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    if discriminator is not None and discriminator_optimizer is not None:
        unwrap(discriminator).load_state_dict(payload["discriminator"])
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rng_states = payload.get("rng_by_rank")
    if world_size_changed or batch_size_changed:
        epoch = int(payload["epoch"]) + 1
        microbatch_offset = 0
        rng = None
    else:
        epoch = int(payload["epoch"])
        microbatch_offset = int(payload["microbatch_offset"])
        rng = rng_states[rank] if rng_states else payload.get("rng")
    return {
        "stage_update": payload["stage_update"], "epoch": epoch,
        "microbatch_offset": microbatch_offset,
        "scheduler_state": payload["scheduler_state"],
        "sampler_state": payload["sampler_state"],
        "rng": rng, "world_size_changed": world_size_changed,
        "batch_size_changed": bool(batch_size_changed),
        "checkpoint_world_size": checkpoint_world_size,
        "world_size": world_size,
    }


def load_predictive_warm_start(path: str | Path, model: PredictiveMidiBrave,
                               config: Config, target_stage: PredictiveStage | str) -> None:
    """Load model weights for a new stage or a fresh corrective fine-tune."""
    target_stage = PredictiveStage(target_stage)
    expected_sources = {
        PredictiveStage.RAVE: {PredictiveStage.RAVE},
        PredictiveStage.PREDICTOR: {
            PredictiveStage.RAVE, PredictiveStage.PREDICTOR},
        PredictiveStage.ROLLOUT: {PredictiveStage.PREDICTOR},
        PredictiveStage.GAN: {PredictiveStage.ROLLOUT},
    }[target_stage]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 5:
        raise ValueError("predictive warm start requires checkpoint format 5")
    contract = payload.get("predictive_contract")
    source_stage = contract.get("stage") if isinstance(contract, dict) else None
    if source_stage not in {stage.value for stage in expected_sources}:
        expected = " or ".join(sorted(stage.value for stage in expected_sources))
        raise ValueError(
            f"{target_stage.value} warm start requires a {expected} checkpoint")
    expected = predictive_checkpoint_contract(config, target_stage, None, None, False)
    for key in ("architecture", "history_frames", "horizon_frames", "stride_frames"):
        if contract.get(key) != expected[key]:
            raise ValueError(f"warm-start {key.replace('_', ' ')} mismatch")
    model.load_state_dict(payload["model"])


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA; CPU is reserved for unit tests and preprocessing")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()}


def mask_audio(audio: Tensor, valid_samples: Tensor | None) -> Tensor:
    if valid_samples is None:
        return audio
    positions = torch.arange(audio.shape[-1], device=audio.device)
    mask = (positions[None, :] < valid_samples[:, None]).to(audio)[:, None]
    return audio * mask


def cosine_lr(step: int, total: int, warmup: int, high: float, low: float,
              hold_until: int | None = None) -> float:
    if step < warmup:
        return high * (step + 1) / max(1, warmup)
    hold_until = warmup if hold_until is None else max(warmup, hold_until)
    if step < hold_until:
        return high
    progress = min(1.0, (step - hold_until) / max(1, total - hold_until))
    return low + 0.5 * (high - low) * (1 + math.cos(math.pi * progress))


def _rng_state() -> dict[str, Any]:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(path: Path, phase: int, step: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                    epoch: int, microbatch_offset: int, manifest_hash: str,
                    discriminator: nn.Module | None = None,
                    discriminator_optimizer: torch.optim.Optimizer | None = None,
                    *, loop_step: int | None = None, generator_updates: int | None = None,
                    discriminator_updates: int = 0, config_hash: str | None = None,
                    manifest_metadata_hash: str | None = None,
                    scheduler_state: dict[str, Any] | None = None,
                    precision_state: dict[str, Any] | None = None,
                    sampler_state: dict[str, Any] | None = None) -> None:
    local_rng = _rng_state()
    if dist.is_available() and dist.is_initialized():
        rng_states: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(rng_states, local_rng)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rng_states = [local_rng]
        rank = 0
        world_size = 1
    if rank != 0:
        return
    if generator_updates is None:
        generator_updates = step + 1
    if loop_step is None:
        loop_step = generator_updates
    payload = {
        "format": 4, "phase": phase, "step": step, "loop_step": loop_step,
        "generator_updates": generator_updates,
        "discriminator_updates": discriminator_updates,
        "world_size": world_size,
        "epoch": epoch, "microbatch_offset": microbatch_offset,
        "manifest_hash": manifest_hash, "manifest_metadata_hash": manifest_metadata_hash,
        "config_hash": config_hash,
        "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "rng_by_rank": rng_states,
        "scheduler_state": scheduler_state or {},
        "precision_state": precision_state or {},
        "sampler_state": sampler_state or {},
    }
    if discriminator is not None:
        payload["discriminator"] = unwrap(discriminator).state_dict()
        payload["discriminator_arch"] = getattr(
            unwrap(discriminator), "architecture_id", type(unwrap(discriminator)).__name__)
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, phase: int,
                    discriminator: nn.Module | None = None,
                    discriminator_optimizer: torch.optim.Optimizer | None = None,
                    manifest_hash: str | None = None,
                    config_hash: str | None = None,
                    manifest_metadata_hash: str | None = None,
                    precision_state: dict[str, Any] | None = None,
                    ) -> tuple[int, int, int, int, int, dict[str, Any] | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format", 0)) == 5:
        raise ValueError(
            "checkpoint format 5 belongs to predictive v3; use --stage instead of --phase")
    same_phase = int(payload["phase"]) == phase
    if same_phase and int(payload.get("format", 0)) != 4:
        raise ValueError("exact resume requires checkpoint format 4; use the old checkpoint only as a warm start")
    if same_phase:
        if phase == 2:
            if discriminator is None or discriminator_optimizer is None:
                raise ValueError("exact Phase 2 resume requires a discriminator and its optimizer")
            missing = {"discriminator", "discriminator_optimizer", "discriminator_arch"} - payload.keys()
            if missing:
                raise ValueError(
                    f"exact Phase 2 resume is missing checkpoint state: {sorted(missing)}"
                )
        unwrap(model).load_state_dict(payload["model"])
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if int(payload.get("world_size", 1)) != world_size:
            raise ValueError("exact resume requires the same DDP world size")
        if manifest_hash is not None and payload.get("manifest_hash") != manifest_hash:
            raise ValueError("checkpoint manifest hash does not match current data")
        if config_hash is not None and payload.get("config_hash") != config_hash:
            raise ValueError("checkpoint config hash does not match current config")
        if (manifest_metadata_hash is not None
                and payload.get("manifest_metadata_hash") != manifest_metadata_hash):
            raise ValueError("checkpoint manifest metadata hash does not match current data contract")
        if precision_state is not None and payload.get("precision_state") != precision_state:
            raise ValueError("checkpoint precision policy does not match current config")
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        if discriminator is not None and "discriminator" in payload:
            expected_arch = getattr(unwrap(discriminator), "architecture_id", None)
            if payload.get("discriminator_arch") != expected_arch:
                raise ValueError(
                    f"discriminator architecture mismatch: {payload.get('discriminator_arch')} "
                    f"!= {expected_arch}"
                )
            unwrap(discriminator).load_state_dict(payload["discriminator"])
        if discriminator_optimizer is not None and "discriminator_optimizer" in payload:
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        rng_states = payload.get("rng_by_rank")
        rng = rng_states[rank] if rng_states else payload.get("rng")
        return (int(payload.get("loop_step", payload["step"] + 1)),
                int(payload.get("generator_updates", payload["step"] + 1)),
                int(payload.get("discriminator_updates", 0)),
                int(payload.get("epoch", 0)), int(payload.get("microbatch_offset", 0)), rng)

    if int(payload["phase"]) != 1 or phase != 2:
        raise ValueError("cross-phase warm start is supported only from Phase 1 into Phase 2")
    if manifest_hash is not None and payload.get("manifest_hash") not in {None, manifest_hash}:
        raise ValueError("warm-start manifest hash does not match current data")
    if config_hash is not None and payload.get("config_hash") not in {None, config_hash}:
        raise ValueError("warm-start config hash does not match current config")
    if (manifest_metadata_hash is not None
            and payload.get("manifest_metadata_hash") not in {None, manifest_metadata_hash}):
        raise ValueError("warm-start manifest metadata hash does not match current data contract")
    current = unwrap(model).state_dict()
    compatible = {name: value for name, value in payload["model"].items()
                  if name in current and current[name].shape == value.shape}
    incompatible = sorted(set(payload["model"]) - set(compatible))
    unwrap(model).load_state_dict(compatible, strict=False)
    if incompatible and set(incompatible) - {"pitch_adversary.net.2.weight",
                                              "pitch_adversary.net.2.bias"}:
        raise ValueError(f"warm start has incompatible generator tensors: {incompatible}")
    return 0, 0, 0, 0, 0, None


def make_generator_optimizer(model: MidiBrave, config: Config, phase: int) -> torch.optim.Optimizer:
    optimizer_kwargs: dict[str, Any] = {"betas": (0.8, 0.99)}
    if config.train.fused_adamw:
        optimizer_kwargs["fused"] = True
    if phase == 1:
        return torch.optim.AdamW(model.parameters(), lr=config.train.lr, **optimizer_kwargs)
    condition_parameters = list(model.timbre.parameters()) + list(model.midi.parameters())
    condition_ids = {id(parameter) for parameter in condition_parameters}
    synthesis_parameters = [parameter for parameter in model.parameters() if id(parameter) not in condition_ids]
    return torch.optim.AdamW([
        {"params": condition_parameters, "lr": config.train.phase2_condition_lr},
        {"params": synthesis_parameters, "lr": config.train.phase2_generator_lr},
    ], **optimizer_kwargs)


def _global_all_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _tracked_tensor(module: nn.Module, preferred: str) -> Tensor:
    named = dict(unwrap(module).named_parameters())
    if preferred in named:
        return named[preferred]
    parameters = [parameter for parameter in unwrap(module).parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("cannot track a module without trainable parameters")
    return min(parameters, key=lambda parameter: parameter.numel())


def _parameter_delta(parameter: Tensor, before: Tensor) -> float:
    return float((parameter.detach().float() - before).abs().max().item())


def _nan_audit_tensor(value: Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    item = value.detach().float()
    finite = torch.isfinite(item)
    valid = item[finite]
    return {
        "shape": list(item.shape),
        "finite": bool(finite.all().item()),
        "nonfinite": int((~finite).sum().item()),
        "minimum": float(valid.min().item()) if valid.numel() else None,
        "maximum": float(valid.max().item()) if valid.numel() else None,
        "maximum_absolute": float(valid.abs().max().item()) if valid.numel() else None,
    }


def _write_nan_audit_failure(
        directory: Path, error: BaseException, rank: int, local_rank: int,
        loop_step: int, generator_updates: int, include_self: bool, self_scale: float,
        scaler: torch.amp.GradScaler, batch: dict[str, Any], output: Any,
        losses: Any, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensor_metadata = {}
    for name, value in batch.items():
        if isinstance(value, Tensor) and name not in {"audio_a", "audio_b", "clap_a", "clap_b"}:
            tensor_metadata[name] = value.detach().cpu().tolist()
        elif not isinstance(value, Tensor):
            tensor_metadata[name] = value
    parameter_nonfinite = []
    gradient_nonfinite = []
    for name, parameter in unwrap(model).named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            parameter_nonfinite.append(name)
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            gradient_nonfinite.append(name)
    payload = {
        "schema": 1,
        "error_type": type(error).__name__,
        "error": str(error),
        "rank": rank,
        "local_rank": local_rank,
        "loop_step_before_increment": loop_step,
        "generator_updates": generator_updates,
        "include_self": include_self,
        "self_scale": self_scale,
        "scaler": float(scaler.get_scale()),
        "batch": tensor_metadata,
        "inputs": {
            "audio_a": _nan_audit_tensor(batch.get("audio_a")),
            "audio_b": _nan_audit_tensor(batch.get("audio_b")),
            "clap_a": _nan_audit_tensor(batch.get("clap_a")),
            "clap_b": _nan_audit_tensor(batch.get("clap_b")),
        },
        "outputs": ({
            "self_audio": _nan_audit_tensor(output.self_audio),
            "cross_audio": _nan_audit_tensor(output.cross_audio),
            "timbre": _nan_audit_tensor(output.timbre),
            "target_timbre": _nan_audit_tensor(output.target_timbre),
            "pitch_logits": _nan_audit_tensor(output.pitch_logits),
        } if output is not None else None),
        "losses": ({name: float(value.detach().float().item())
                    for name, value in losses.values.items()}
                   if losses is not None else None),
        "parameter_nonfinite": parameter_nonfinite,
        "gradient_nonfinite": gradient_nonfinite,
    }
    temporary_json = directory / f"failure-rank-{rank}.json.tmp"
    final_json = directory / f"failure-rank-{rank}.json"
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(final_json)
    tensor_batch = {
        name: (value.detach().cpu() if isinstance(value, Tensor) else value)
        for name, value in batch.items()
    }
    temporary_batch = directory / f"failure-batch-rank-{rank}.pt.tmp"
    final_batch = directory / f"failure-batch-rank-{rank}.pt"
    torch.save(tensor_batch, temporary_batch)
    temporary_batch.replace(final_batch)
    # Every rank may be the first one on which anomaly mode raises. Save the
    # synchronized model plus that rank's exact batch without requiring rank 0
    # to reach the exception handler or any collective.
    temporary_state = directory / f"failure-state-rank-{rank}.pt.tmp"
    final_state = directory / f"failure-state-rank-{rank}.pt"
    torch.save({
        "model": unwrap(model).state_dict(),
        "scaler": scaler.state_dict(),
        "loop_step": loop_step,
        "generator_updates": generator_updates,
    }, temporary_state)
    temporary_state.replace(final_state)


def _split_discriminator_batch(output, first_batch: int):
    first = []
    second = []
    for score, features in output:
        first.append((score[:first_batch], [feature[:first_batch] for feature in features]))
        second.append((score[first_batch:], [feature[first_batch:] for feature in features]))
    return first, second


def _split_discriminator_segments(output, sizes: list[int]):
    segments = []
    offset = 0
    for size in sizes:
        current = []
        for score, features in output:
            current.append((score[offset:offset + size],
                            [feature[offset:offset + size] for feature in features]))
        segments.append(current)
        offset += size
    return segments


def _detach_discriminator_output(output):
    return [(score.detach(), [feature.detach() for feature in features])
            for score, features in output]


def _self_branch_schedule(config: Config, phase: int,
                          generator_updates: int) -> tuple[bool, float]:
    probability = config.train.self_probability
    if not 0.0 < probability <= 1.0:
        raise ValueError("train.self_probability must be in (0, 1]")
    if not 0.0 <= config.train.self_full_fraction <= 1.0:
        raise ValueError("train.self_full_fraction must be in [0, 1]")
    if phase == 1:
        full_updates = round(config.train.phase1_steps * config.train.self_full_fraction)
        if generator_updates < full_updates:
            return True, 1.0
        relative_step = generator_updates - full_updates
    else:
        relative_step = generator_updates
    if probability == 1.0:
        return True, 1.0
    include = (math.floor((relative_step + 1) * probability)
               > math.floor(relative_step * probability))
    return include, (1.0 / probability if include else 0.0)


def _pitch_adversary_scale(config: Config, phase: int, generator_updates: int) -> float:
    # Phase 2 is a warm start from a fully ramped Phase 1 model. Its local update
    # counter intentionally restarts at zero, so reusing the Phase 1 threshold
    # would silently disable the disentanglement constraint for most of Phase 2.
    if phase == 2:
        return 1.0
    if generator_updates < config.train.pitch_adversary_start:
        return 0.0
    return min(
        1.0,
        (generator_updates - config.train.pitch_adversary_start + 1)
        / max(1, config.train.pitch_adversary_ramp),
    )


def _scheduled_clap_weight(weight: float, generator_updates: int,
                           warmup_steps: int) -> float:
    if weight <= 0.0:
        return 0.0
    if warmup_steps <= 0:
        return float(weight)
    return float(weight) * min(1.0, (generator_updates + 1) / warmup_steps)


def _clap_due(config: Config, generator_updates: int) -> bool:
    enabled = config.loss.self_clap > 0.0 or config.loss.cross_clap > 0.0
    if not enabled:
        return False
    interval = config.loss.clap_every_updates
    if interval <= 0:
        raise ValueError("loss.clap_every_updates must be positive")
    if config.loss.clap_batch_size <= 0:
        raise ValueError("loss.clap_batch_size must be positive")
    return (generator_updates + 1) % interval == 0


def _clap_health_vector(health: ClapHealth, device: torch.device) -> Tensor:
    """Pack one local CLAP evaluation into counts followed by maxima."""
    return torch.stack((
        torch.ones((), device=device),
        health.skipped.to(device),
        health.target_failures.to(device),
        health.generated_failures.to(device),
        health.input_nonfinite_count.to(device),
        health.raw_embedding_nonfinite_count.to(device),
        health.normalized_embedding_nonfinite_count.to(device),
        health.loss_nonfinite_count.to(device),
        health.waveform_gradient_nonfinite_count.to(device),
        health.input_peak.to(device),
        health.input_rms.to(device),
    )).float()


def _distributed_clap_health(health: ClapHealth, device: torch.device) -> Tensor:
    return _distributed_clap_health_vector(_clap_health_vector(health, device))


def _distributed_clap_health_vector(vector: Tensor) -> Tensor:
    if vector.shape != (11,):
        raise ValueError("CLAP health vector must have shape [11]")
    vector = vector.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vector[:9], op=dist.ReduceOp.SUM)
        dist.all_reduce(vector[9:], op=dist.ReduceOp.MAX)
    return vector


def _merge_clap_health(total: Tensor, call: Tensor) -> Tensor:
    if total.shape != (11,) or call.shape != (11,):
        raise ValueError("CLAP health vectors must have shape [11]")
    merged = total.clone()
    merged[:9] += call[:9]
    merged[9:] = torch.maximum(merged[9:], call[9:])
    return merged


def _clap_health_metrics(health: Tensor, prefix: str) -> dict[str, float]:
    if health.shape != (11,):
        raise ValueError("CLAP health vector must have shape [11]")
    values = [float(value.item()) for value in health]
    evaluations, skips = values[:2]
    return {
        f"{prefix}_evaluations": evaluations,
        f"{prefix}_skips": skips,
        f"{prefix}_skip_ratio": skips / max(1.0, evaluations),
        f"{prefix}_target_failures": values[2],
        f"{prefix}_generated_failures": values[3],
        f"{prefix}_input_nonfinite_count": values[4],
        f"{prefix}_raw_embedding_nonfinite_count": values[5],
        f"{prefix}_normalized_embedding_nonfinite_count": values[6],
        f"{prefix}_loss_nonfinite_count": values[7],
        f"{prefix}_waveform_gradient_nonfinite_count": values[8],
        f"{prefix}_failed_input_peak_max": values[9],
        f"{prefix}_failed_input_rms_max": values[10],
    }


def _tensor_health_metrics(tensors: dict[str, Tensor],
                           distributed: bool = False) -> dict[str, float]:
    """Summarize existing pipeline tensors without retaining their graphs."""
    metrics: dict[str, float] = {}
    for name in sorted(tensors):
        value = tensors[name].detach().float()
        finite = torch.isfinite(value)
        finite_values = torch.where(finite, value, torch.zeros_like(value))
        finite_count = finite.sum()
        vector = torch.stack((
            (~finite).sum().float(),
            finite_values.abs().max(),
            (finite_values.square().sum() / finite_count.clamp_min(1)).sqrt(),
        ))
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(vector[:1], op=dist.ReduceOp.SUM)
            dist.all_reduce(vector[1:], op=dist.ReduceOp.MAX)
        metrics[f"pipeline/{name}_nonfinite_count"] = float(vector[0].item())
        metrics[f"pipeline/{name}_peak"] = float(vector[1].item())
        metrics[f"pipeline/{name}_rms"] = float(vector[2].item())
    return metrics


def _predictive_clap_auxiliary(
        result: ClapWaveformGradient, total: Tensor, generated: Tensor,
        weight: float, maximum_fraction: float) -> Tensor:
    """Return zero immediately when CLAP already rejected this local waveform."""
    if bool(result.health.skipped.item()):
        return torch.zeros_like(generated)
    reference_gradient, = torch.autograd.grad(
        total, generated, retain_graph=True)
    auxiliary = result.gradients * weight / generated.shape[0]
    return cap_auxiliary_gradient(
        auxiliary, reference_gradient, maximum_fraction)


def _predictive_clap_auxiliaries(
        total: Tensor, waveforms: tuple[Tensor, ...],
        auxiliaries: tuple[Tensor, ...], maximum_fraction: float,
        ) -> tuple[Tensor, ...]:
    """Cap several CLAP injections against one shared base-gradient budget."""
    if len(waveforms) != len(auxiliaries) or not waveforms:
        raise ValueError("CLAP waveforms and auxiliaries must be equally non-empty")
    references = torch.autograd.grad(
        total, waveforms, retain_graph=True, allow_unused=True)
    references = tuple(
        torch.zeros_like(waveform) if reference is None else reference
        for waveform, reference in zip(waveforms, references))
    sizes = [value.numel() for value in auxiliaries]
    combined_auxiliary = torch.cat([value.reshape(-1) for value in auxiliaries])
    combined_reference = torch.cat([value.reshape(-1) for value in references])
    combined_capped = cap_auxiliary_gradient(
        combined_auxiliary, combined_reference, maximum_fraction)
    values = []
    offset = 0
    for auxiliary, size in zip(auxiliaries, sizes):
        values.append(combined_capped[offset:offset + size].reshape_as(auxiliary))
        offset += size
    return tuple(values)


def _cap_auxiliary_parameter_gradient(
        auxiliary: Tensor, reference: Tensor, parameters: list[Tensor],
        maximum_fraction: float) -> tuple[Tensor, Tensor]:
    """Cap one auxiliary objective against a reference parameter gradient."""
    if not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError("maximum parameter gradient fraction must be between zero and one")
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("parameter gradient cap requires trainable parameters")
    reference_gradients = torch.autograd.grad(
        reference, trainable, retain_graph=True, allow_unused=True)

    def gradient_norm(values: tuple[Tensor | None, ...]) -> Tensor | None:
        square = reference.new_zeros((), dtype=torch.float32)
        for value in values:
            if value is None:
                continue
            value = value.detach().float()
            if not bool(torch.isfinite(value).all().item()):
                return None
            square = square + value.square().sum()
        return square.sqrt()

    reference_norm = gradient_norm(reference_gradients)
    if reference_norm is None:
        raise ValueError("parameter gradient cap rejects non-finite values")
    # Measuring an uncapped control objective can itself overflow an fp16
    # intermediate before we have computed the cap. Retry the norm probe with
    # exact power-of-two loss scaling, then undo that scale in fp32. This does
    # not change the eventual capped objective or its gradient direction.
    auxiliary_norm = None
    for probe_scale in (1.0, 2.0 ** -8, 2.0 ** -16, 2.0 ** -24):
        auxiliary_gradients = torch.autograd.grad(
            auxiliary * probe_scale, trainable, retain_graph=True,
            allow_unused=True)
        probed_norm = gradient_norm(auxiliary_gradients)
        if probed_norm is not None:
            auxiliary_norm = probed_norm / probe_scale
            break
    if auxiliary_norm is None or not bool(torch.isfinite(auxiliary_norm).item()):
        raise ValueError("parameter gradient cap rejects non-finite values")
    limit = reference_norm * maximum_fraction
    if bool(auxiliary_norm.le(limit).item()):
        scale = auxiliary_norm.new_ones(())
    else:
        scale = limit / auxiliary_norm.clamp_min(1e-12)
    scale = scale.detach().to(device=auxiliary.device, dtype=auxiliary.dtype)
    return auxiliary * scale, scale


def _predictive_throughput(
        applied_updates: int, elapsed_seconds: float, batch_per_gpu: int,
        world_size: int) -> tuple[float, float]:
    """Return applied updates/s and global training samples/s."""
    if applied_updates < 0:
        raise ValueError("applied_updates must be non-negative")
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be positive")
    if batch_per_gpu <= 0:
        raise ValueError("batch_per_gpu must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    updates_per_second = applied_updates / elapsed_seconds
    samples_per_second = updates_per_second * batch_per_gpu * world_size
    return updates_per_second, samples_per_second


def _predictive_scaler_step(scaler: Any, optimizer: Any, *,
                            globally_finite: bool,
                            previous_scale: float) -> bool:
    """Keep optimizer decisions identical across manually reduced DDP ranks."""
    if globally_finite:
        scaler.step(optimizer)
        scaler.update()
        return True
    scaler.update(new_scale=max(1.0, previous_scale * 0.5))
    return False


@torch.no_grad()
def _clap_prepass(model: nn.Module, batch: dict[str, Any], include_self: bool,
                  indices: Tensor) -> dict[str, Tensor]:
    """Recompute the normal decoder output without retaining its activation graph."""
    value = unwrap(model)(
        batch["clap_a"], batch["note_a"], batch["velocity_a"],
        batch["note_b"], batch["velocity_b"], batch["clap_b"],
        grl_scale=0.0, include_self=include_self,
        source_excitation_seed=batch.get("excitation_seed_a"),
        target_excitation_seed=batch.get("excitation_seed_b"),
    )
    selected = {"cross": value.cross_audio.index_select(0, indices).detach()}
    if include_self:
        assert value.self_audio is not None
        selected["self"] = value.self_audio.index_select(0, indices).detach()
    return selected


def _prepare_clap_injections(
        model: nn.Module, objective: FrozenClapReconstructionObjective,
        batch: dict[str, Any], config: Config, include_self: bool, self_scale: float,
        generator_updates: int, rank: int,
        ) -> tuple[dict[str, dict[str, Tensor | float]], ClapHealth]:
    batch_size = int(batch["audio_a"].shape[0])
    selected_count = min(config.loss.clap_batch_size, batch_size)
    event = (generator_updates + 1) // config.loss.clap_every_updates
    start = (event * selected_count + rank * selected_count) % batch_size
    indices = torch.tensor(
        [(start + offset) % batch_size for offset in range(selected_count)],
        device=batch["audio_a"].device, dtype=torch.long,
    )
    with torch.autocast("cuda", dtype=torch.float16):
        predictions = _clap_prepass(model, batch, include_self, indices)

    names = ["cross"]
    if include_self and config.loss.self_clap > 0.0:
        names.append("self")
    names = [name for name in names if getattr(config.loss, f"{name}_clap") > 0.0]
    if not names:
        raise RuntimeError("CLAP injection requested without an enabled branch")
    generated = torch.cat([predictions[name] for name in names], dim=0)
    targets = torch.cat([
        batch["audio_b" if name == "cross" else "audio_a"].index_select(0, indices)
        for name in names
    ], dim=0)
    valid = torch.cat([
        batch["valid_samples_b" if name == "cross" else "valid_samples_a"].index_select(
            0, indices)
        for name in names
    ], dim=0)
    result = objective.waveform_gradients(generated, targets, valid)
    injections: dict[str, dict[str, Tensor | float]] = {}
    offset = 0
    for name in names:
        stop = offset + selected_count
        raw_loss = result.losses[offset:stop].mean()
        branch_scale = self_scale if name == "self" else 1.0
        weight = _scheduled_clap_weight(
            getattr(config.loss, f"{name}_clap"), generator_updates,
            config.loss.clap_warmup_steps,
        )
        # The trainer divides each microbatch loss by grad_accum. CLAP is
        # evaluated on one microbatch every K updates, so G*K restores the
        # expectation of the full per-update objective. selected_count is the
        # Monte-Carlo batch mean; Self reuses its existing 1/p correction.
        gradient_scale = (
            weight * config.train.grad_accum * config.loss.clap_every_updates
            * branch_scale / selected_count
        )
        injections[name] = {
            "indices": indices,
            "reference": predictions[name],
            "gradient": result.gradients[offset:stop] * gradient_scale,
            "raw_loss": raw_loss,
            "raw_gradient_norm": result.gradient_norms[offset:stop].mean(),
            "clipped_gradient_norm": result.clipped_gradient_norms[offset:stop].mean(),
            "weight": weight,
            "branch_scale": branch_scale,
        }
        offset = stop
    return injections, result.health


def _inject_clap_gradients(output: Any, injections: dict[str, dict[str, Tensor | float]],
                           grad_accum: int) -> tuple[Tensor, dict[str, Tensor]]:
    addition = output.cross_audio.new_zeros(())
    diagnostics: dict[str, Tensor] = {}
    for name, injection in injections.items():
        prediction = output.cross_audio if name == "cross" else output.self_audio
        if prediction is None:
            raise RuntimeError(f"missing {name} output required by CLAP injection")
        indices = injection["indices"]
        assert isinstance(indices, Tensor)
        selected = prediction.index_select(0, indices)
        gradient = injection["gradient"]
        reference = injection["reference"]
        raw_loss = injection["raw_loss"]
        assert isinstance(gradient, Tensor) and isinstance(reference, Tensor)
        assert isinstance(raw_loss, Tensor)
        surrogate = (selected.float() * gradient.float()).sum()
        weight = float(injection["weight"])
        branch_scale = float(injection["branch_scale"])
        # The detached value keeps logged totals on the ordinary objective
        # scale. Only the surrogate carries the sparse, probability-corrected
        # waveform gradient.
        displayed = raw_loss * weight * branch_scale * grad_accum
        addition = addition + surrogate - surrogate.detach() + displayed.detach()
        diagnostics[f"{name}_clap"] = raw_loss
        diagnostics[f"{name}_clap_waveform_grad_norm"] = injection["raw_gradient_norm"]
        diagnostics[f"{name}_clap_waveform_grad_norm_clipped"] = (
            injection["clipped_gradient_norm"])
        diagnostics[f"{name}_clap_recompute_max_abs_error"] = (
            selected.detach().float() - reference.float()).abs().amax()
        diagnostics[f"{name}_clap_effective_weight"] = raw_loss.new_tensor(weight)
    return addition, diagnostics


def _artifact_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_predictor_warm_start_cache(
        path: str | Path, cache_checkpoint_hash: str,
        statistics_hash: str) -> None:
    """Bind RAVE starts by artifact and predictor restarts by latent statistics."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload.get("predictive_contract") if isinstance(payload, dict) else None
    source_stage = contract.get("stage") if isinstance(contract, dict) else None
    if source_stage == PredictiveStage.RAVE.value:
        if _artifact_sha256(path) != cache_checkpoint_hash:
            raise ValueError(
                "RAVE cache checkpoint hash does not match predictor warm start")
        return
    if source_stage == PredictiveStage.PREDICTOR.value:
        if contract.get("latent_statistics_hash") != statistics_hash:
            raise ValueError(
                "predictor warm-start latent statistics do not match the cache")
        return
    raise ValueError("predictor warm start requires a RAVE or predictor checkpoint")


def load_predictive_statistics(path: str | Path, config: Config,
                               ) -> tuple[LatentStatistics, str, str]:
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("latent statistics require a v3 config")
    with np.load(path, allow_pickle=False) as values:
        required = {"latent_std", "delta_std", "acceleration_std",
                    "checkpoint_hash", "samples_per_latent"}
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"latent statistics are missing fields: {sorted(missing)}")
        hop = int(values["samples_per_latent"].item())
        if hop != config.predictive.samples_per_latent:
            raise ValueError("latent statistics samples per latent mismatch")
        arrays = [torch.from_numpy(values[name].astype(np.float32, copy=True))
                  for name in ("latent_std", "delta_std", "acceleration_std")]
        checkpoint_hash = str(values["checkpoint_hash"].item())
    statistics = LatentStatistics(*arrays, floor=config.latent_loss.statistic_floor)
    statistics.scales(config.predictive.rave_latent_dim, torch.device("cpu"), torch.float32)
    return statistics, checkpoint_hash, _artifact_sha256(path)


def save_predictive_calibration(path: str | Path, weights: dict[str, float],
                                statistics_hash: str, batches: int) -> str:
    if batches <= 0 or not weights:
        raise ValueError("predictive calibration is empty")
    payload = {"format": 1, "statistics_hash": statistics_hash,
               "batches": batches, "weights": weights}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return _artifact_sha256(destination)


def load_predictive_calibration(path: str | Path, statistics_hash: str,
                                ) -> tuple[dict[str, float], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != 1:
        raise ValueError("unsupported predictive calibration format")
    if payload.get("statistics_hash") != statistics_hash:
        raise ValueError("calibration statistics hash mismatch")
    weights = payload.get("weights")
    required = {"future", "delta", "acceleration", "overlap"}
    if not isinstance(weights, dict) or set(weights) != required:
        raise ValueError("calibration weights do not match predictive losses")
    parsed = {name: float(value) for name, value in weights.items()}
    if any(not math.isfinite(value) or value <= 0 for value in parsed.values()):
        raise ValueError("calibration weights must be finite and positive")
    return parsed, _artifact_sha256(path)


def calibrate_predictive_latent_weights(
    model: PredictiveMidiBrave, batches: Any, config: Config,
    statistics: LatentStatistics, device: torch.device,
) -> tuple[dict[str, float], int]:
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("predictive calibration requires a v3 config")
    initial = {
        "future": config.latent_loss.future,
        "delta": config.latent_loss.delta,
        "acceleration": config.latent_loss.acceleration,
        "overlap": config.latent_loss.overlap,
    }
    total_initial = sum(initial.values())
    shares = {name: value / total_initial for name, value in initial.items()}
    samples: dict[str, list[float]] = {name: [] for name in initial}
    count = 0
    for raw_batch in batches:
        batch = move_batch(raw_batch, device)
        objective = predictive_latent_objective(model, batch, config, statistics)
        for name, component in objective.components.items():
            norm = loss_gradient_norm(component, objective.prediction)
            samples[name].append(float(norm.detach().cpu()))
        count += 1
        if count >= config.predictive.calibration_batches:
            break
    if count != config.predictive.calibration_batches:
        raise ValueError(
            f"calibration requires {config.predictive.calibration_batches} batches, got {count}")
    return calibrate_loss_weights(samples, shares, initial, anchor="future"), count


def _predictive_stage_steps(config: Config, stage: PredictiveStage) -> int:
    return {
        PredictiveStage.RAVE: config.train.rave_steps,
        PredictiveStage.PREDICTOR: config.train.predictor_steps,
        PredictiveStage.ROLLOUT: config.train.rollout_steps,
        PredictiveStage.GAN: config.train.gan_steps,
    }[stage]


def _repeat_batches(loader: DataLoader, dataset: PairDataset,
                    sampler: DistributedSampler, start_epoch: int = 0):
    epoch = start_epoch
    while True:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def train_predictive(
    config_path: str, stage: PredictiveStage | str,
    max_steps: int | None = None, resume: str | None = None,
    warm_start: str | None = None, statistics_path: str | None = None,
    calibration_path: str | None = None, rollout_gate_passed: bool = False,
    allow_world_size_change: bool = False,
    batch_per_gpu: int | None = None,
) -> None:
    """Four-stage CUDA trainer for encoder-free predictive deployment."""
    config = Config.load(config_path)
    configured_batch_per_gpu = config.train.batch_per_gpu
    requested_batch_per_gpu = (configured_batch_per_gpu if batch_per_gpu is None
                               else int(batch_per_gpu))
    if requested_batch_per_gpu <= 0:
        raise ValueError("--batch-per-gpu must be positive")
    resume_payload: dict[str, Any] | None = None
    checkpoint_batch_per_gpu = configured_batch_per_gpu
    if resume:
        loaded = torch.load(resume, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError("predictive checkpoint must be a mapping")
        resume_payload = loaded
        checkpoint_batch_per_gpu = int(
            loaded.get("batch_per_gpu", configured_batch_per_gpu))
    batch_size_changed = bool(
        resume and requested_batch_per_gpu != checkpoint_batch_per_gpu)
    if requested_batch_per_gpu != configured_batch_per_gpu:
        config = replace(
            config, train=replace(
                config.train, batch_per_gpu=requested_batch_per_gpu))
    stage = PredictiveStage(stage)
    if config.predictive is None or config.latent_loss is None:
        raise ValueError("--stage requires a predictive v3 config")
    if resume and warm_start:
        raise ValueError("exact resume and warm start are mutually exclusive")
    if stage is not PredictiveStage.RAVE and not resume and not warm_start:
        raise ValueError(f"{stage.value} stage requires --warm-start or --resume")
    rank, _, world_size, device = distributed_setup()
    seed_everything(config.seed, rank)
    torch.autograd.set_detect_anomaly(
        os.environ.get("MIDIBRAVE_DETECT_ANOMALY", "0") == "1",
        check_nan=True,
    )
    torch.backends.cudnn.benchmark = True

    stats: LatentStatistics | None = None
    cache_checkpoint_hash: str | None = None
    statistics_hash: str | None = None
    if stage is not PredictiveStage.RAVE:
        statistics_path = statistics_path or str(
            Path(config.data.cache_root) / "rave-statistics.npz")
        stats, cache_checkpoint_hash, statistics_hash = load_predictive_statistics(
            statistics_path, config)
        stats = LatentStatistics(
            stats.latent_std.to(device), stats.delta_std.to(device),
            stats.acceleration_std.to(device), stats.floor)
        if stage is PredictiveStage.PREDICTOR and warm_start:
            assert statistics_hash is not None
            _validate_predictor_warm_start_cache(
                warm_start, cache_checkpoint_hash, statistics_hash)

    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate).to(device)
    if warm_start:
        load_predictive_warm_start(warm_start, model, config, stage)
    configure_predictive_stage(model, stage, rollout_gate_passed)
    if world_size > 1:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)

    load_cache = stage is not PredictiveStage.RAVE
    dataset = PairDataset(
        config.data, config.seed, config.predictive, cache_checkpoint_hash,
        load_rave_cache=load_cache)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=config.seed, drop_last=True)
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.train.batch_per_gpu, "sampler": sampler,
        "num_workers": config.data.num_workers, "pin_memory": True,
        "drop_last": True,
    }
    if config.data.num_workers:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=config.data.prefetch_factor)
    loader = DataLoader(dataset, **loader_kwargs)
    if not len(loader):
        raise ValueError("predictive training has no complete batches")

    calibrated_weights: dict[str, float] | None = None
    calibration_hash: str | None = None
    if stage is not PredictiveStage.RAVE:
        assert statistics_hash is not None and stats is not None
        calibration_path = calibration_path or str(
            Path(config.data.cache_root) / "predictive-calibration.json")
        if Path(calibration_path).is_file():
            calibrated_weights, calibration_hash = load_predictive_calibration(
                calibration_path, statistics_hash)
        elif stage is PredictiveStage.PREDICTOR and not resume:
            if rank == 0:
                calibrated_weights, batches = calibrate_predictive_latent_weights(
                    model, _repeat_batches(loader, dataset, sampler), config, stats, device)
                save_predictive_calibration(
                    calibration_path, calibrated_weights, statistics_hash, batches)
            if world_size > 1:
                dist.barrier()
            calibrated_weights, calibration_hash = load_predictive_calibration(
                calibration_path, statistics_hash)
        else:
            raise FileNotFoundError(
                f"missing fixed predictive calibration artifact: {calibration_path}")

    contract = predictive_checkpoint_contract(
        config, stage, statistics_hash, calibration_hash, rollout_gate_passed)
    if resume:
        assert resume_payload is not None
        # Deliberately validate immutable contracts before optimizer creation.
        validate_predictive_resume(resume_payload, contract)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"{stage.value} stage has no trainable parameters")
    optimizer_kwargs: dict[str, Any] = {"betas": (0.8, 0.99)}
    if config.train.fused_adamw:
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(trainable, lr=config.train.lr, **optimizer_kwargs)
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=config.train.grad_scaler_init_scale,
        growth_interval=config.train.grad_scaler_growth_interval)
    discriminator: nn.Module | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if stage is PredictiveStage.GAN:
        discriminator = BraveMultiScaleDiscriminator().to(device)
        if world_size > 1:
            discriminator = DDP(discriminator, device_ids=[device.index],
                                output_device=device.index, broadcast_buffers=False)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=config.train.discriminator_lr,
            betas=(0.8, 0.99))

    stage_update = 0
    epoch = 0
    microbatch_offset = 0
    if resume:
        restored = load_predictive_checkpoint(
            resume, model, optimizer, scaler, contract,
            discriminator, discriminator_optimizer,
            allow_world_size_change=allow_world_size_change,
            batch_size_changed=batch_size_changed)
        stage_update = int(restored["stage_update"])
        epoch = int(restored["epoch"])
        microbatch_offset = int(restored["microbatch_offset"])
        dataset.load_sampler_state_dict(restored["sampler_state"])
        if restored["rng"] is not None:
            _restore_rng(restored["rng"])
        if rank == 0 and (restored["world_size_changed"]
                          or restored["batch_size_changed"]):
            print(json.dumps({
                "event": "predictive_elastic_resume",
                "checkpoint_world_size": restored["checkpoint_world_size"],
                "world_size": restored["world_size"],
                "checkpoint_batch_per_gpu": checkpoint_batch_per_gpu,
                "batch_per_gpu": config.train.batch_per_gpu,
                "batch_size_changed": restored["batch_size_changed"],
                "stage_update": stage_update,
                "epoch": epoch,
                "microbatch_offset": microbatch_offset,
            }, sort_keys=True), flush=True)

    configured_steps = _predictive_stage_steps(config, stage)
    target_updates = configured_steps if max_steps is None else min(configured_steps, max_steps)
    if stage_update > target_updates:
        raise ValueError("checkpoint exceeds requested predictive update limit")
    run_dir = Path(config.train.output_dir) / config.train.run_name / stage.value
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)
    else:
        writer = None
    stft = MultiResolutionSTFTLoss((2048, 1024, 512, 256, 128)).to(device)
    swap_pitch = (SpectralPitchObjective(config.data.sample_rate).to(device)
                  if stage in {
                      PredictiveStage.RAVE, PredictiveStage.PREDICTOR,
                      PredictiveStage.ROLLOUT}
                  and config.loss.cross_pitch > 0
                  else None)
    clap_objective = None
    if ((config.latent_loss.clap_control > 0
         or config.latent_loss.clap_counterfactual > 0)
            and config.data.clap_checkpoint):
        clap_objective = FrozenClapReconstructionObjective(
            config.data.clap_checkpoint, config.data.sample_rate, device,
            maximum_gradient_norm=0.0)

    updates_at_start = stage_update
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    clap_health_cumulative = torch.zeros(11, device=device)
    clap_health_interval = torch.zeros(11, device=device)
    clap_warning_events = 0
    nonfinite_updates = 0
    while stage_update < target_updates:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(loader):
            if batch_index < microbatch_offset:
                continue
            batch = move_batch(raw_batch, device)
            lr = cosine_lr(
                stage_update, target_updates, config.train.warmup_steps,
                config.train.lr, config.train.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast("cuda", dtype=torch.float16):
                objective = predictive_stage_objective(
                    unwrap(model), batch, config, stage, stage_update, stats,
                    calibrated_weights, stft, swap_pitch, clap_objective)
                total = objective.total
                if discriminator is not None:
                    assert objective.generated_audio is not None
                    discriminator.requires_grad_(False)
                    fake_outputs = discriminator(objective.generated_audio)
                    with torch.no_grad():
                        real_outputs = discriminator(objective.target_audio)
                    schedule = predictive_loss_schedule(config, stage, stage_update)
                    adversarial = generator_adversarial(fake_outputs)
                    matching = feature_matching(real_outputs, fake_outputs)
                    total = (total + schedule.gan_adversarial * adversarial
                             + schedule.gan_feature_matching * matching)
                    discriminator.requires_grad_(True)
                else:
                    adversarial = matching = total.new_zeros(())

            diagnostics = objective.diagnostic_tensors or {}
            if "predictor_control_clap_health" in diagnostics:
                global_control_health = _distributed_clap_health_vector(
                    diagnostics["predictor_control_clap_health"])
                clap_health_cumulative = _merge_clap_health(
                    clap_health_cumulative, global_control_health)
                clap_health_interval = _merge_clap_health(
                    clap_health_interval, global_control_health)
                if int(global_control_health[1].item()):
                    clap_warning_events += 1
                    pipeline_health = _tensor_health_metrics(
                        diagnostics, distributed=True)
                    if rank == 0 and (clap_warning_events <= 10
                                      or clap_warning_events % 100 == 0):
                        warning = _clap_health_metrics(
                            global_control_health, "clap_nonfinite_call")
                        warning.update({
                            "event": "predictor_control_clap_nonfinite_skipped",
                            "stage": stage.value,
                            "stage_update": stage_update,
                            "warning_event": clap_warning_events,
                        })
                        warning.update(pipeline_health)
                        print(json.dumps(warning, sort_keys=True), flush=True)

            clap_loss = total.new_zeros(())
            if clap_objective is not None and objective.generated_audio is not None:
                generated = objective.generated_audio
                target = objective.target_audio
                assert target is not None
                valid = torch.full(
                    (generated.shape[0],), generated.shape[-1], device=device,
                    dtype=torch.long)
                clap_result = clap_objective.waveform_gradients(generated, target, valid)
                global_clap_health = _distributed_clap_health(
                    clap_result.health, device)
                clap_health_cumulative = _merge_clap_health(
                    clap_health_cumulative, global_clap_health)
                clap_health_interval = _merge_clap_health(
                    clap_health_interval, global_clap_health)
                if int(global_clap_health[1].item()):
                    clap_warning_events += 1
                    pipeline_health = _tensor_health_metrics(
                        objective.diagnostic_tensors or {"decoder_audio": generated},
                        distributed=True)
                    if rank == 0 and (clap_warning_events <= 10
                                      or clap_warning_events % 100 == 0):
                        warning = _clap_health_metrics(
                            global_clap_health, "clap_nonfinite_call")
                        warning.update({
                            "event": "clap_nonfinite_embedding_skipped",
                            "stage": stage.value,
                            "stage_update": stage_update,
                            "warning_event": clap_warning_events,
                        })
                        warning.update(pipeline_health)
                        print(json.dumps(warning, sort_keys=True), flush=True)
                clap_loss = clap_result.losses.mean()
                reconstruction_auxiliary = (
                    torch.zeros_like(generated) if bool(clap_result.health.skipped.item())
                    else (clap_result.gradients * config.latent_loss.clap_control
                          / generated.shape[0]))
                waveforms = [generated]
                requested_auxiliaries = [reconstruction_auxiliary]
                counterfactual_loss = total.new_zeros(())
                diagnostics = objective.diagnostic_tensors or {}
                if (config.latent_loss.clap_counterfactual > 0.0
                        and "predictor_clap_control_total" not in objective.components
                        and "counterfactual_audio" in diagnostics):
                    counterfactual = diagnostics["counterfactual_audio"]
                    timbre_mask = diagnostics["counterfactual_timbre_mask"].bool()
                    if bool(timbre_mask.any().item()):
                        timbre_audio = counterfactual[timbre_mask]
                        timbre_valid = torch.full(
                            (timbre_audio.shape[0],), timbre_audio.shape[-1],
                            device=device, dtype=torch.long)
                        control_result = clap_objective.waveform_gradients_to_embeddings(
                            timbre_audio,
                            diagnostics["counterfactual_target_clap"][timbre_mask],
                            diagnostics["counterfactual_source_clap"][timbre_mask],
                            timbre_valid)
                        global_control_health = _distributed_clap_health(
                            control_result.health, device)
                        clap_health_cumulative = _merge_clap_health(
                            clap_health_cumulative, global_control_health)
                        clap_health_interval = _merge_clap_health(
                            clap_health_interval, global_control_health)
                        counterfactual_loss = control_result.losses.mean()
                        objective.components["clap_counterfactual"] = counterfactual_loss
                        objective.components["clap_counterfactual_following"] = (
                            control_result.following.float().mean())
                        objective.components["clap_counterfactual_target_cosine"] = (
                            control_result.target_cosine.mean())
                        objective.components["clap_counterfactual_source_cosine"] = (
                            control_result.source_cosine.mean())
                        counterfactual_auxiliary = torch.zeros_like(counterfactual)
                        if not bool(control_result.health.skipped.item()):
                            counterfactual_auxiliary[timbre_mask] = (
                                control_result.gradients
                                * config.latent_loss.clap_counterfactual
                                / timbre_audio.shape[0])
                        waveforms.append(counterfactual)
                        requested_auxiliaries.append(counterfactual_auxiliary)
                capped = _predictive_clap_auxiliaries(
                    total, tuple(waveforms), tuple(requested_auxiliaries),
                    config.predictive.clap_gradient_fraction_max)
                surrogate = sum(
                    (waveform.float() * auxiliary.float()).sum()
                    for waveform, auxiliary in zip(waveforms, capped))
                total = (total + surrogate - surrogate.detach()
                         + (config.latent_loss.clap_control * clap_loss).detach()
                         + (config.latent_loss.clap_counterfactual
                            * counterfactual_loss).detach())

            scaler.scale(total / config.train.grad_accum).backward()
            accumulated += 1
            microbatch_offset = batch_index + 1
            if accumulated < config.train.grad_accum:
                continue
            scaler.unscale_(optimizer)
            if world_size > 1:
                for parameter in trainable:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad.div_(world_size)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, config.train.grad_clip)
            globally_finite = _global_all_true(
                bool(torch.isfinite(gradient_norm).item()), device)
            previous_scale = scaler.get_scale()
            step_applied = _predictive_scaler_step(
                scaler, optimizer, globally_finite=globally_finite,
                previous_scale=previous_scale)
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            if not step_applied:
                nonfinite_updates += 1
                if rank == 0 and (nonfinite_updates <= 10
                                  or nonfinite_updates % 100 == 0):
                    print(json.dumps({
                        "event": "predictive_nonfinite_update_rejected",
                        "stage": stage.value,
                        "stage_update": stage_update,
                        "rejected_updates": nonfinite_updates,
                        "gradient_norm": float(gradient_norm.float().item()),
                        "scale_before": float(previous_scale),
                        "scale_after": float(scaler.get_scale()),
                    }, sort_keys=True), flush=True)
                continue

            if discriminator is not None and discriminator_optimizer is not None:
                assert objective.generated_audio is not None and objective.target_audio is not None
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    real_outputs = discriminator(objective.target_audio.detach())
                    fake_outputs = discriminator(objective.generated_audio.detach())
                    discriminator_loss = discriminator_hinge(real_outputs, fake_outputs)
                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.train.grad_clip)
                discriminator_optimizer.step()
            else:
                discriminator_loss = total.new_zeros(())

            stage_update += 1
            dataset.set_training_update(stage_update)
            regular_log = stage_update % config.train.log_every == 0
            control_log = (stage in {
                               PredictiveStage.PREDICTOR, PredictiveStage.ROLLOUT}
                           and _predictor_control_active(config, stage_update - 1))
            if writer is not None and (regular_log or control_log):
                if regular_log:
                    updates_per_second, samples_per_second = _predictive_throughput(
                        stage_update - updates_at_start,
                        time.perf_counter() - training_started,
                        config.train.batch_per_gpu * config.train.grad_accum,
                        world_size)
                    writer.add_scalar("loss/total", float(total.detach()), stage_update)
                    writer.add_scalar("loss/clap_control", float(clap_loss), stage_update)
                    writer.add_scalar(
                        "loss/adversarial", float(adversarial.detach()), stage_update)
                    writer.add_scalar(
                        "loss/feature_matching", float(matching.detach()), stage_update)
                    writer.add_scalar(
                        "loss/discriminator", float(discriminator_loss.detach()), stage_update)
                    writer.add_scalar("train/lr", lr, stage_update)
                    writer.add_scalar(
                        "train/updates_per_second", updates_per_second,
                        stage_update)
                    writer.add_scalar(
                        "train/samples_per_second", samples_per_second,
                        stage_update)
                    writer.add_scalar(
                        "train/amp_scale", float(scaler.get_scale()), stage_update)
                    writer.add_scalar(
                        "health/nonfinite_updates", nonfinite_updates, stage_update)
                for name, value in objective.components.items():
                    if _should_log_predictive_component(
                            name, regular_log, control_log):
                        writer.add_scalar(
                            f"loss/{name}", float(value.detach()), stage_update)
                if regular_log:
                    for name, value in _clap_health_metrics(
                            clap_health_cumulative, "health/clap").items():
                        writer.add_scalar(name, value, stage_update)
                    for name, value in _clap_health_metrics(
                            clap_health_interval, "health/clap_interval").items():
                        writer.add_scalar(name, value, stage_update)
            if regular_log:
                clap_health_interval.zero_()
            should_checkpoint = (
                stage_update == target_updates
                or (config.train.checkpoint_every > 0
                    and stage_update % config.train.checkpoint_every == 0))
            if should_checkpoint:
                save_predictive_checkpoint(
                    run_dir / f"update-{stage_update:08d}.pt", model, optimizer,
                    scaler, contract, stage_update=stage_update, epoch=epoch,
                    microbatch_offset=microbatch_offset,
                    scheduler_state={"lr": lr, "stage_update": stage_update},
                    sampler_state=dataset.sampler_state_dict(),
                    batch_per_gpu=config.train.batch_per_gpu,
                    discriminator=discriminator,
                    discriminator_optimizer=discriminator_optimizer)
            if stage_update >= target_updates:
                break
        if stage_update >= target_updates:
            break
        epoch += 1
        microbatch_offset = 0
    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def train(config_path: str, phase: int, max_steps: int | None = None,
          resume: str | None = None) -> None:
    config = Config.load(config_path)
    if config.train.precision != "amp_fp16":
        raise ValueError("Octopus V100 training supports precision=amp_fp16 only")
    if config.train.ddp_static_graph and config.train.self_probability < 1.0:
        raise ValueError(
            "DDP static_graph is incompatible with conditional Self-branch sampling"
        )
    if config.loss.clap_warmup_steps < 0:
        raise ValueError("loss.clap_warmup_steps must be non-negative")
    if config.loss.clap_gradient_norm < 0:
        raise ValueError("loss.clap_gradient_norm must be non-negative")
    clap_enabled = config.loss.self_clap > 0.0 or config.loss.cross_clap > 0.0
    if clap_enabled and not config.data.clap_checkpoint:
        raise ValueError("CLAP reconstruction requires data.clap_checkpoint")
    rank, local_rank, world_size, device = distributed_setup()
    seed_everything(config.seed, rank)
    deterministic = os.environ.get("MIDIBRAVE_DETERMINISTIC", "0") == "1"
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    dataset = PairDataset(config.data, config.seed)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=config.seed, drop_last=True)
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.train.batch_per_gpu,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": True,
        "persistent_workers": config.data.num_workers > 0,
        "drop_last": True,
    }
    if config.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    if not len(loader):
        raise ValueError("dataset is too small for the configured DDP world size and batch")

    model: nn.Module = MidiBrave(
        config.model, config.data.window_samples, config.data.sample_rate).to(device)
    unwrap(model).prepare_runtime_caches()
    activation_run_max = torch.zeros((), device=device)
    activation_hooks = []
    if os.environ.get("MIDIBRAVE_TRACK_ACTIVATIONS", "0") == "1":
        def track_activation(_module, inputs):
            nonlocal activation_run_max
            value = inputs[0].detach().float().abs().amax()
            activation_run_max = torch.maximum(activation_run_max, value)
        for module in unwrap(model).modules():
            if isinstance(module, ChannelRMSNorm):
                activation_hooks.append(module.register_forward_pre_hook(track_activation))
    if config.train.compile_decoder:
        decoder = unwrap(model).decoder
        if not hasattr(decoder, "compile"):
            raise RuntimeError("this PyTorch build does not support nn.Module.compile")
        decoder.compile(mode=config.train.compile_mode)
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank], broadcast_buffers=False,
            gradient_as_bucket_view=config.train.ddp_gradient_as_bucket_view,
            static_graph=config.train.ddp_static_graph,
            bucket_cap_mb=config.train.ddp_bucket_cap_mb,
        )
    generator_optimizer = make_generator_optimizer(unwrap(model), config, phase)
    discriminator: nn.Module | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if phase == 2:
        discriminator = BraveMultiScaleDiscriminator().to(device)
        if config.train.compile_discriminator:
            if not hasattr(discriminator, "compile"):
                raise RuntimeError("this PyTorch build does not support nn.Module.compile")
            discriminator.compile(mode=config.train.compile_mode)
        if world_size > 1:
            discriminator = DDP(
                discriminator, device_ids=[local_rank], broadcast_buffers=False,
                gradient_as_bucket_view=config.train.ddp_gradient_as_bucket_view,
                static_graph=config.train.ddp_static_graph,
                bucket_cap_mb=config.train.ddp_bucket_cap_mb,
            )
        discriminator_kwargs: dict[str, Any] = {"betas": (0.5, 0.9)}
        if config.train.fused_adamw:
            discriminator_kwargs["fused"] = True
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=config.train.discriminator_lr,
            **discriminator_kwargs)
    reconstruction = ReconstructionLoss(
        config.loss, config.data.sample_rate, config.model.pitch_backend,
        config.data.pitch_hop_length, config.model.pqmf_bands, config.model.pqmf_taps,
    ).to(device)
    clap_objective = (
        FrozenClapReconstructionObjective(
            config.data.clap_checkpoint, config.data.sample_rate, device,
            config.loss.clap_gradient_norm,
        )
        if clap_enabled else None
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=config.train.grad_scaler_init_scale,
        growth_interval=config.train.grad_scaler_growth_interval,
    )

    manifest_hash = hashlib.sha256(Path(config.data.manifest).read_bytes()).hexdigest()
    config_hash = hashlib.sha256(Path(config.source_path).read_bytes()).hexdigest()
    metadata_hash = None
    if config.data.manifest_metadata:
        metadata_hash = hashlib.sha256(Path(config.data.manifest_metadata).read_bytes()).hexdigest()
    loop_step = generator_updates = discriminator_updates = epoch = microbatch_offset = 0
    precision_state = {
        "precision": config.train.precision,
        "decoder_fp32_tail": config.model.decoder_fp32_tail,
        "pqmf_dtype": config.model.pqmf_dtype,
        "norm_reduction_dtype": config.model.norm_reduction_dtype,
        "phase_accumulation_dtype": config.model.phase_accumulation_dtype,
    }
    resume_rng = None
    if resume:
        (loop_step, generator_updates, discriminator_updates, epoch,
         microbatch_offset, resume_rng) = load_checkpoint(
            resume, model, generator_optimizer, scaler, phase, discriminator,
            discriminator_optimizer, manifest_hash, config_hash, metadata_hash,
            precision_state,
        )
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        dataset.load_sampler_state_dict(resume_payload.get("sampler_state", {}))
    configured_updates = config.train.phase1_steps if phase == 1 else config.train.phase2_steps
    total_updates = min(configured_updates, max_steps) if max_steps is not None else configured_updates
    if generator_updates > total_updates:
        raise ValueError("checkpoint already exceeds the requested effective update limit")

    run_dir = Path(config.train.output_dir) / config.train.run_name / f"phase{phase}"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            Path(config.source_path).read_text(encoding="utf-8"), encoding="utf-8")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)
    else:
        writer = None
    metrics_path = run_dir / "metrics.jsonl"

    dataset.set_epoch(epoch)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(microbatch_offset):
        try:
            next(iterator)
        except StopIteration as error:
            raise ValueError("checkpoint microbatch offset exceeds epoch length") from error
    if resume_rng is not None:
        _restore_rng(resume_rng)
    generator_optimizer.zero_grad(set_to_none=True)
    if discriminator_optimizer is not None:
        discriminator_optimizer.zero_grad(set_to_none=True)

    started = time.time()
    updates_at_start = generator_updates
    global_pairs_per_update = world_size * config.train.batch_per_gpu * config.train.grad_accum
    executed_audio_windows = 0
    maximum_loops = loop_step + max(1000, (total_updates - generator_updates) * 10)
    nan_audit_directory_text = os.environ.get("MIDIBRAVE_NAN_AUDIT_DIR", "")
    nan_audit_directory = (Path(nan_audit_directory_text)
                           if nan_audit_directory_text else None)
    nan_audit_start = int(os.environ.get("MIDIBRAVE_NAN_AUDIT_START_UPDATE", "0"))
    nan_audit_capture_scale_max = float(
        os.environ.get("MIDIBRAVE_NAN_AUDIT_CAPTURE_SCALE_MAX", "4.0")
    )
    max_consecutive_nonfinite = int(
        os.environ.get("MIDIBRAVE_MAX_CONSECUTIVE_NONFINITE", "16")
    )
    if max_consecutive_nonfinite <= 0:
        raise ValueError("MIDIBRAVE_MAX_CONSECUTIVE_NONFINITE must be positive")
    consecutive_nonfinite = 0
    anomaly_enabled = False
    clap_health_cumulative = torch.zeros(11, device=device)
    clap_health_interval = torch.zeros(11, device=device)
    clap_warning_events = 0
    while generator_updates < total_updates:
        if loop_step >= maximum_loops:
            raise RuntimeError("too many skipped updates; refusing to count overflow loops as training")
        if (nan_audit_directory is not None and generator_updates >= nan_audit_start
                and not anomaly_enabled):
            # Global anomaly mode records the forward traceback for the exact
            # backward operator that first returns NaN. It is intentionally
            # enabled only near the known failure region because it is slow.
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            anomaly_enabled = True
        loop_started = time.perf_counter()
        if phase == 1:
            lr = cosine_lr(generator_updates, configured_updates, config.train.warmup_steps,
                           config.train.lr, config.train.min_lr,
                           config.train.lr_hold_until)
            for group in generator_optimizer.param_groups:
                group["lr"] = lr
        aggregate: dict[str, Tensor] = {}
        data_wait_seconds = 0.0
        include_self, self_scale = _self_branch_schedule(config, phase, generator_updates)

        for microstep in range(config.train.grad_accum):
            data_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                dataset.set_epoch(epoch)
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                microbatch_offset = 0
                batch = next(iterator)
            data_wait_seconds += time.perf_counter() - data_started
            microbatch_offset += 1
            batch = move_batch(batch, device)
            sync = microstep == config.train.grad_accum - 1
            sync_context = nullcontext() if sync or world_size == 1 else model.no_sync()
            grl_scale = _pitch_adversary_scale(config, phase, generator_updates)
            clap_injections = {}
            if (sync and clap_objective is not None
                    and _clap_due(config, generator_updates)):
                clap_injections, local_clap_health = _prepare_clap_injections(
                    model, clap_objective, batch, config, include_self, self_scale,
                    generator_updates, rank,
                )
                global_clap_health = _distributed_clap_health(local_clap_health, device)
                clap_health_cumulative = _merge_clap_health(
                    clap_health_cumulative, global_clap_health)
                clap_health_interval = _merge_clap_health(
                    clap_health_interval, global_clap_health)
                skipped_ranks = int(global_clap_health[1].item())
                if skipped_ranks:
                    clap_warning_events += 1
                    if rank == 0 and (clap_warning_events <= 10
                                      or clap_warning_events % 100 == 0):
                        warning = _clap_health_metrics(
                            global_clap_health, "clap_nonfinite_call")
                        warning.update({
                            "event": "clap_nonfinite_embedding_skipped",
                            "generator_updates": generator_updates,
                            "warning_event": clap_warning_events,
                        })
                        print(json.dumps(warning, sort_keys=True), flush=True)

            with sync_context:
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch["clap_a"], batch["note_a"], batch["velocity_a"],
                                   batch["note_b"], batch["velocity_b"], batch["clap_b"],
                                   grl_scale=grl_scale, include_self=include_self,
                                   source_excitation_seed=batch.get("excitation_seed_a"),
                                   target_excitation_seed=batch.get("excitation_seed_b"))
                    losses = reconstruction(
                        output, batch, output.target_timbre, grl_scale > 0,
                        self_scale=self_scale, generator_update=generator_updates,
                    )
                    generator_loss = losses.total

                if clap_injections:
                    clap_addition, clap_diagnostics = _inject_clap_gradients(
                        output, clap_injections, config.train.grad_accum)
                    generator_loss = generator_loss + clap_addition
                    losses.total = generator_loss
                    losses.values.update(clap_diagnostics)

                discriminator_loss: Tensor | None = None
                if discriminator is not None and discriminator_optimizer is not None:
                    discriminator.requires_grad_(True)
                    discriminator_sync = (nullcontext() if sync or world_size == 1
                                          else discriminator.no_sync())
                    pair_batch = batch["audio_a"].shape[0]
                    with discriminator_sync:
                        with torch.autocast("cuda", dtype=torch.float16):
                            discriminator_inputs = [mask_audio(
                                batch["audio_b"], batch.get("valid_samples_b"))]
                            segment_names = ["real_cross"]
                            if include_self:
                                discriminator_inputs.append(mask_audio(
                                    batch["audio_a"], batch.get("valid_samples_a")))
                                segment_names.append("real_self")
                            discriminator_inputs.append(mask_audio(
                                output.cross_audio.detach(), batch.get("valid_samples_b")))
                            segment_names.append("fake_cross")
                            if include_self:
                                assert output.self_audio is not None
                                discriminator_inputs.append(mask_audio(
                                    output.self_audio.detach(), batch.get("valid_samples_a")))
                                segment_names.append("fake_self")
                            discriminator_train_output = discriminator(
                                torch.cat(discriminator_inputs, dim=0))
                            split = _split_discriminator_segments(
                                discriminator_train_output,
                                [pair_batch for _ in discriminator_inputs],
                            )
                            segments = dict(zip(segment_names, split))
                            discriminator_loss = 0.5 * discriminator_hinge(
                                segments["real_cross"], segments["fake_cross"])
                            if include_self:
                                discriminator_loss = (
                                    discriminator_loss
                                    + 0.5 * self_scale * discriminator_hinge(
                                        segments["real_self"], segments["fake_self"])
                                )
                            real_cross = _detach_discriminator_output(segments["real_cross"])
                            real_self = (_detach_discriminator_output(segments["real_self"])
                                         if include_self else None)
                        scaler.scale(
                            discriminator_loss / config.train.grad_accum).backward()
                    del discriminator_train_output, segments, split
                    discriminator.requires_grad_(False)
                    with torch.autocast("cuda", dtype=torch.float16):
                        if include_self:
                            assert output.self_audio is not None and real_self is not None
                            fake_combined = discriminator(torch.cat((
                                mask_audio(output.self_audio, batch.get("valid_samples_a")),
                                mask_audio(output.cross_audio, batch.get("valid_samples_b")),
                            ), dim=0))
                            fake_self, fake_cross = _split_discriminator_batch(
                                fake_combined, pair_batch)
                            adversarial = (self_scale * generator_adversarial(fake_self)
                                           + 0.25 * generator_adversarial(fake_cross))
                            matching = (self_scale * feature_matching(real_self, fake_self)
                                        + 0.25 * feature_matching(real_cross, fake_cross))
                        else:
                            fake_cross = discriminator(mask_audio(
                                output.cross_audio, batch.get("valid_samples_b")))
                            adversarial = 0.25 * generator_adversarial(fake_cross)
                            matching = 0.25 * feature_matching(real_cross, fake_cross)
                        generator_loss = (generator_loss
                                          + config.loss.adversarial * adversarial
                                          + config.loss.feature_matching * matching)
                        losses.values["adversarial"] = adversarial
                        losses.values["feature_matching"] = matching
                    discriminator.requires_grad_(True)

                try:
                    scaler.scale(generator_loss / config.train.grad_accum).backward()
                except RuntimeError as error:
                    if nan_audit_directory is not None:
                        _write_nan_audit_failure(
                            nan_audit_directory, error, rank, local_rank,
                            loop_step, generator_updates, include_self, self_scale,
                            scaler, batch, output, losses, model, generator_optimizer,
                        )
                    raise

            if discriminator_loss is not None:
                aggregate["discriminator"] = (
                    aggregate.get("discriminator", discriminator_loss.new_zeros(()))
                    + discriminator_loss.detach() / config.train.grad_accum)
            for name, value in losses.values.items():
                # Sparse CLAP metrics occur only on the final accumulation
                # microbatch. Undo the ordinary 1/G metric averaging; the
                # gradient itself was already corrected when prepared.
                metric_scale = (config.train.grad_accum if "_clap" in name else 1.0)
                aggregate[name] = (aggregate.get(name, value.new_zeros(()))
                                   + value.detach() * metric_scale / config.train.grad_accum)
            aggregate["total"] = (aggregate.get("total", generator_loss.new_zeros(()))
                                  + generator_loss.detach() / config.train.grad_accum)

            if microbatch_offset == len(loader):
                epoch += 1
                dataset.set_epoch(epoch)
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                microbatch_offset = 0

        scale_before = float(scaler.get_scale())
        generator_tracked = _tracked_tensor(model, "decoder.output.weight")
        generator_before = generator_tracked.detach().clone()
        scaler.unscale_(generator_optimizer)
        generator_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.grad_clip)
        local_finite = bool(torch.isfinite(generator_grad_norm).item())
        discriminator_grad_norm: Tensor | None = None
        discriminator_tracked: Tensor | None = None
        discriminator_before: Tensor | None = None
        if discriminator is not None and discriminator_optimizer is not None:
            discriminator_tracked = _tracked_tensor(discriminator, "scales.0.output.weight")
            discriminator_before = discriminator_tracked.detach().clone()
            scaler.unscale_(discriminator_optimizer)
            discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), config.train.grad_clip)
            local_finite = local_finite and bool(torch.isfinite(discriminator_grad_norm).item())

        # Autograd anomaly mode raises on NaN-producing backward operators, but
        # it does not reliably raise for every Inf gradient. Capture the first
        # low-scale overflow after unscale as a second diagnostic route. At
        # scale <= 4 this is no longer merely an aggressive GradScaler growth
        # probe; the underlying unscaled graph is near or beyond FP16 range.
        if (nan_audit_directory is not None and not local_finite
                and scale_before <= nan_audit_capture_scale_max):
            error = RuntimeError(
                "non-finite gradient after unscale at low AMP scale "
                f"{scale_before:g}"
            )
            _write_nan_audit_failure(
                nan_audit_directory, error, rank, local_rank,
                loop_step, generator_updates, include_self, self_scale,
                scaler, batch, output, losses, model, generator_optimizer,
            )
            raise error

        step_applied = _global_all_true(local_finite, device)
        if step_applied:
            consecutive_nonfinite = 0
            scaler.step(generator_optimizer)
            if discriminator_optimizer is not None:
                scaler.step(discriminator_optimizer)
            scaler.update()
            generator_updates += 1
            executed_audio_windows += global_pairs_per_update * (1 + int(include_self))
            if discriminator_optimizer is not None:
                discriminator_updates += 1
        else:
            consecutive_nonfinite += 1
            scaler.update(new_scale=max(1.0, scale_before * 0.5))
        generator_optimizer.zero_grad(set_to_none=True)
        if discriminator_optimizer is not None:
            discriminator_optimizer.zero_grad(set_to_none=True)
        loop_step += 1

        generator_delta = _parameter_delta(generator_tracked, generator_before)
        discriminator_delta = 0.0
        if discriminator_tracked is not None and discriminator_before is not None:
            discriminator_delta = _parameter_delta(discriminator_tracked, discriminator_before)
        scale_after = float(scaler.get_scale())
        should_log = (loop_step % config.train.log_every == 0
                      or not step_applied or generator_updates == total_updates)
        reported_activation: Tensor | None = None
        if should_log and activation_hooks:
            reported_activation = activation_run_max.detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(reported_activation, op=dist.ReduceOp.MAX)
        if should_log:
            torch.cuda.synchronize(device)
        if rank == 0 and should_log:
            values = {name: float(value.float().item()) for name, value in aggregate.items()}
            reconstruction_weights = {
                "self_stft": config.loss.self_stft,
                "self_envelope": config.loss.self_envelope,
                "self_pitch": config.loss.self_pitch,
                "self_rms": config.loss.self_rms,
                "self_spectral_flux": config.loss.self_spectral_flux,
                "self_band_statistics": config.loss.self_band_statistics,
                "cross_stft": config.loss.cross_stft,
                "cross_envelope": config.loss.cross_envelope,
                "cross_pitch": config.loss.cross_pitch,
                "cross_rms": config.loss.cross_rms,
                "cross_spectral_flux": config.loss.cross_spectral_flux,
                "cross_band_statistics": config.loss.cross_band_statistics,
                "velocity_rank": config.loss.velocity_rank,
                "velocity_delta": config.loss.velocity_delta,
                "self_clap": config.loss.self_clap,
                "cross_clap": config.loss.cross_clap,
                "timbre_pair": config.loss.timbre_pair,
                "distribution": config.loss.distribution,
                "pitch_adversary": config.loss.pitch_adversary,
                "adversarial": config.loss.adversarial,
                "feature_matching": config.loss.feature_matching,
            }
            for name, weight in reconstruction_weights.items():
                if name in values:
                    sampling_scale = (self_scale if name.startswith("self_")
                                      or name in ("velocity_rank", "velocity_delta")
                                      else 1.0)
                    values[f"weighted_{name}"] = values[name] * weight * sampling_scale
            elapsed = time.time() - started
            effective_pairs = (generator_updates - updates_at_start) * global_pairs_per_update
            values.update({
                "loop_step": loop_step,
                "generator_updates": generator_updates,
                "discriminator_updates": discriminator_updates,
                "generator_step_applied": int(step_applied),
                "discriminator_step_applied": int(step_applied and discriminator is not None),
                "gradients_finite": int(step_applied),
                "rank0_gradients_finite": int(local_finite),
                "consecutive_nonfinite_updates": consecutive_nonfinite,
                "generator_grad_norm": float(generator_grad_norm.float().item()),
                "discriminator_grad_norm": (float(discriminator_grad_norm.float().item())
                                                if discriminator_grad_norm is not None else 0.0),
                "generator_parameter_delta": generator_delta,
                "discriminator_parameter_delta": discriminator_delta,
                "self_branch_executed": int(include_self),
                "self_sampling_scale": self_scale,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "data_wait_ms": data_wait_seconds * 1000.0,
                "step_wall_ms": (time.perf_counter() - loop_started) * 1000.0,
                "seconds": elapsed,
                "effective_updates_per_second": generator_updates / max(1e-9, elapsed),
                "effective_pairs_per_second": effective_pairs / max(1e-9, elapsed),
                "generated_audio_seconds_per_second": (
                    executed_audio_windows * config.data.window_samples / config.data.sample_rate
                    / max(1e-9, elapsed)),
                "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "generator_lr_min": min(group["lr"] for group in generator_optimizer.param_groups),
                "generator_lr_max": max(group["lr"] for group in generator_optimizer.param_groups),
                "discriminator_lr": (discriminator_optimizer.param_groups[0]["lr"]
                                     if discriminator_optimizer is not None else 0.0),
            })
            values.update(_clap_health_metrics(
                clap_health_cumulative, "health/clap"))
            values.update(_clap_health_metrics(
                clap_health_interval, "health/clap_interval"))
            if reported_activation is not None:
                values["residual_activation_absmax"] = float(reported_activation.item())
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(values, sort_keys=True) + "\n")
            assert writer is not None
            for name, value in values.items():
                if name not in {"loop_step", "generator_updates", "discriminator_updates", "seconds"}:
                    writer.add_scalar(name, value, generator_updates)
            print(json.dumps(values, sort_keys=True), flush=True)

        if should_log:
            clap_health_interval.zero_()

        if not step_applied and consecutive_nonfinite >= max_consecutive_nonfinite:
            raise RuntimeError(
                "refusing to continue after "
                f"{consecutive_nonfinite} consecutive non-finite updates"
            )

        checkpoint_updates = set(config.train.checkpoint_updates or [])
        should_archive_checkpoint = (
            generator_updates in checkpoint_updates
            or (config.train.checkpoint_every > 0
                and generator_updates % config.train.checkpoint_every == 0)
            or generator_updates == total_updates
        )
        should_roll_checkpoint = (
            config.train.rolling_checkpoint_every > 0
            and generator_updates % config.train.rolling_checkpoint_every == 0
        )
        if step_applied and (should_archive_checkpoint or should_roll_checkpoint):
            checkpoint_path = (
                run_dir / f"step-{generator_updates:09d}.pt"
                if should_archive_checkpoint else run_dir / "latest.pt"
            )
            save_checkpoint(
                checkpoint_path, phase,
                generator_updates - 1, model, generator_optimizer, scaler, epoch,
                microbatch_offset, manifest_hash, discriminator, discriminator_optimizer,
                loop_step=loop_step, generator_updates=generator_updates,
                discriminator_updates=discriminator_updates, config_hash=config_hash,
                manifest_metadata_hash=metadata_hash,
                scheduler_state={"generator_updates": generator_updates,
                                 "lr": [group["lr"] for group in generator_optimizer.param_groups]},
                precision_state=precision_state,
                sampler_state=dataset.sampler_state_dict(),
            )
            if rank == 0 and should_archive_checkpoint and should_roll_checkpoint:
                temporary_latest = run_dir / "latest.tmp"
                shutil.copyfile(checkpoint_path, temporary_latest)
                temporary_latest.replace(run_dir / "latest.pt")

    if writer is not None:
        writer.close()
    for hook in activation_hooks:
        hook.remove()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="MidiBrave v2/v3 DDP trainer")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", type=int, choices=(1, 2))
    mode.add_argument("--stage", choices=tuple(stage.value for stage in PredictiveStage))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-effective-updates", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--warm-start")
    parser.add_argument("--latent-statistics")
    parser.add_argument("--calibration")
    parser.add_argument("--rollout-gate-passed", action="store_true")
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--batch-per-gpu", type=int)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_effective_updates is not None:
        parser.error("use only one of --max-steps and --max-effective-updates")
    limit = (args.max_effective_updates
             if args.max_effective_updates is not None else args.max_steps)
    if args.stage:
        train_predictive(
            args.config, args.stage, limit, args.resume, args.warm_start,
            args.latent_statistics, args.calibration,
            args.rollout_gate_passed, args.allow_world_size_change,
            args.batch_per_gpu)
    else:
        if (args.warm_start or args.latent_statistics or args.calibration
                or args.rollout_gate_passed or args.allow_world_size_change
                or args.batch_per_gpu is not None):
            parser.error("predictive stage options cannot be used with --phase")
        assert args.phase is not None
        train(args.config, args.phase, limit, args.resume)


if __name__ == "__main__":
    main()
