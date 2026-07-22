from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .config import Config
from .data import (PairDataset, configured_roots, load_audio, load_manifest,
                   midi_to_hz, record_audio_path)
from .losses import (FrozenClapReconstructionObjective,
                     MultiResolutionSTFTLoss, rms_db)
from .model import MidiBrave
from .latent_predictor import rollout_blocks
from .predictive_losses import LatentStatistics
from .predictive_model import PredictiveMidiBrave


PREDICTIVE_HORIZONS = (1, 8, 32, 128)


def predictive_rave_quality_gate(
        metrics: dict[str, dict[str, float | int]],
        nonfinite_count: int) -> dict[str, Any]:
    """Gate the RAVE representation before caching it for predictor training."""
    thresholds = {
        "reconstruction_f0_median_cents": 50.0,
        "reconstruction_f0_p90_cents": 100.0,
        "swap_f0_median_cents": 100.0,
        "swap_f0_p90_cents": 200.0,
        "midi_swap_following": 0.90,
        "reconstruction_clap_cosine": 0.80,
        "clap_control_following": 0.90,
    }
    failures = []
    reconstruction = metrics.get("reconstruction_f0_absolute_cents", {})
    swapped = metrics.get("swap_f0_absolute_cents", {})
    following = metrics.get("midi_swap_following", {})
    reconstruction_clap = metrics.get("reconstruction_clap_cosine", {})
    clap_following = metrics.get("clap_control_following", {})
    if float(reconstruction.get("median", math.inf)) > thresholds[
            "reconstruction_f0_median_cents"]:
        failures.append("reconstruction_f0_median")
    if float(reconstruction.get("p90", math.inf)) > thresholds[
            "reconstruction_f0_p90_cents"]:
        failures.append("reconstruction_f0_p90")
    if float(swapped.get("median", math.inf)) > thresholds["swap_f0_median_cents"]:
        failures.append("swap_f0_median")
    if float(swapped.get("p90", math.inf)) > thresholds["swap_f0_p90_cents"]:
        failures.append("swap_f0_p90")
    if float(following.get("mean", -math.inf)) < thresholds["midi_swap_following"]:
        failures.append("midi_swap_following")
    if float(reconstruction_clap.get("median", -math.inf)) < thresholds[
            "reconstruction_clap_cosine"]:
        failures.append("reconstruction_clap_cosine")
    if float(clap_following.get("mean", -math.inf)) < thresholds[
            "clap_control_following"]:
        failures.append("clap_control_following")
    if nonfinite_count:
        failures.append("non_finite")
    return {"passed": not failures, "failures": sorted(failures),
            "thresholds": thresholds, "non_finite_count": int(nonfinite_count)}


def predictive_control_quality_gate(
        metrics: dict[str, dict[str, float | int]],
        nonfinite_count: int) -> dict[str, Any]:
    """Require independent MIDI/timbre control and a healthy causal rollout."""
    thresholds = {
        "swap_f0_p90_cents": 200.0,
        "midi_swap_following": 0.90,
        "midi_timbre_preservation_cosine": 0.80,
        "clap_control_following": 0.90,
        "timbre_f0_p90_cents": 200.0,
        "timbre_path_monotonicity": 0.75,
        "seed_washout_ratio_max": 0.90,
        "rollout_variance_ratio_min": 0.25,
        "rollout_variance_ratio_max": 4.0,
    }
    swapped = metrics.get("swap_f0_absolute_cents", {})
    midi_following = metrics.get("midi_swap_following", {})
    midi_timbre = metrics.get("midi_timbre_preservation_cosine", {})
    clap_following = metrics.get("clap_control_following", {})
    timbre_f0 = metrics.get("timbre_f0_absolute_cents", {})
    timbre_path = metrics.get("timbre_path_monotonicity", {})
    seed_washout = metrics.get("seed_washout_ratio", {})
    variance = float(metrics.get("rollout_variance_ratio", {}).get(
        "mean", math.nan))
    failures = []
    if float(swapped.get("p90", math.inf)) > thresholds["swap_f0_p90_cents"]:
        failures.append("swap_f0_p90")
    if float(midi_following.get("mean", -math.inf)) < thresholds[
            "midi_swap_following"]:
        failures.append("midi_swap_following")
    if float(midi_timbre.get("median", -math.inf)) < thresholds[
            "midi_timbre_preservation_cosine"]:
        failures.append("midi_timbre_preservation_cosine")
    if float(clap_following.get("mean", -math.inf)) < thresholds[
            "clap_control_following"]:
        failures.append("clap_control_following")
    if float(timbre_f0.get("p90", math.inf)) > thresholds[
            "timbre_f0_p90_cents"]:
        failures.append("timbre_f0_p90")
    if float(timbre_path.get("mean", -math.inf)) < thresholds[
            "timbre_path_monotonicity"]:
        failures.append("timbre_path_monotonicity")
    if float(seed_washout.get("mean", math.inf)) > thresholds[
            "seed_washout_ratio_max"]:
        failures.append("seed_washout_ratio")
    if (not math.isfinite(variance)
            or variance < thresholds["rollout_variance_ratio_min"]
            or variance > thresholds["rollout_variance_ratio_max"]):
        failures.append("rollout_variance_ratio")
    if nonfinite_count:
        failures.append("non_finite")
    return {
        "passed": not failures,
        "failures": sorted(failures),
        "thresholds": thresholds,
        "non_finite_count": int(nonfinite_count),
    }


def _finite_scalar(value: Tensor) -> float | None:
    item = float(value.detach().float().cpu())
    return item if math.isfinite(item) else None


def _dominant_frequency(audio: Tensor, sample_rate: int) -> Tensor:
    value = audio.float().squeeze(1)
    spectrum = torch.fft.rfft(value, dim=-1).abs()
    if spectrum.shape[-1] > 1:
        spectrum[..., 0] = 0
    index = spectrum.argmax(dim=-1)
    return index.float() * sample_rate / max(1, value.shape[-1])


def predictive_rollout_report(
    predicted_latent: Tensor, reference_latent: Tensor,
    predicted_audio: Tensor, reference_audio: Tensor,
    statistics: LatentStatistics, stride_frames: int,
    samples_per_latent: int, elapsed_seconds: float,
    sample_rate: int = 44100,
    clap_embeddings: dict[int, tuple[Tensor, Tensor]] | None = None,
    runtime_underruns: int = 0,
) -> dict[str, Any]:
    """Aggregate fixed-horizon rollout metrics and safety-only gates."""
    if predicted_latent.shape != reference_latent.shape or predicted_latent.ndim != 3:
        raise ValueError("rollout latents must share [batch, channels, frames]")
    if predicted_audio.shape != reference_audio.shape or predicted_audio.ndim != 3:
        raise ValueError("rollout audio must share [batch, 1, samples]")
    if stride_frames <= 0 or samples_per_latent <= 0 or elapsed_seconds < 0:
        raise ValueError("invalid rollout timing contract")
    required_frames = max(PREDICTIVE_HORIZONS) * stride_frames
    required_samples = required_frames * samples_per_latent
    if (predicted_latent.shape[-1] < required_frames
            or predicted_audio.shape[-1] < required_samples):
        raise ValueError("rollout does not cover the 128-step evaluation horizon")
    latent_scale, delta_scale, _ = statistics.scales(
        predicted_latent.shape[1], predicted_latent.device, predicted_latent.dtype)
    horizons: dict[str, dict[str, float | int | None]] = {}
    all_variance_ratios = []
    total_non_finite = 0
    for steps in PREDICTIVE_HORIZONS:
        frames = steps * stride_frames
        samples = frames * samples_per_latent
        predicted_z = predicted_latent[..., :frames]
        reference_z = reference_latent[..., :frames]
        predicted_wave = predicted_audio[..., :samples]
        reference_wave = reference_audio[..., :samples]
        non_finite = int((~torch.isfinite(predicted_z)).sum().item()
                         + (~torch.isfinite(predicted_wave)).sum().item())
        total_non_finite = max(total_non_finite, non_finite)
        safe_z = torch.nan_to_num(predicted_z)
        safe_wave = torch.nan_to_num(predicted_wave)
        latent_error = ((safe_z - reference_z) / latent_scale).square().mean().sqrt()
        if frames > 1:
            delta_error = ((safe_z.diff(dim=-1) - reference_z.diff(dim=-1))
                           / delta_scale).square().mean().sqrt()
        else:
            delta_error = latent_error.new_zeros(())
        predicted_variance = safe_z.var(dim=-1, unbiased=False).mean()
        reference_variance = reference_z.var(dim=-1, unbiased=False).mean().clamp_min(1e-8)
        variance_ratio = predicted_variance / reference_variance
        all_variance_ratios.append(float(variance_ratio.detach().cpu()))
        predicted_spectrum = torch.fft.rfft(safe_wave.float(), dim=-1).abs().clamp_min(1e-7)
        reference_spectrum = torch.fft.rfft(reference_wave.float(), dim=-1).abs().clamp_min(1e-7)
        stft = (predicted_spectrum.log() - reference_spectrum.log()).abs().mean()
        predicted_f0 = _dominant_frequency(safe_wave, sample_rate).clamp_min(1e-6)
        reference_f0 = _dominant_frequency(reference_wave, sample_rate).clamp_min(1e-6)
        cents = (1200.0 * torch.log2(predicted_f0 / reference_f0)).abs().mean()
        rms_error = (rms_db(safe_wave) - rms_db(reference_wave)).abs().mean()
        clap_cosine = None
        if clap_embeddings is not None and steps in clap_embeddings:
            predicted_clap, reference_clap = clap_embeddings[steps]
            clap_cosine = _finite_scalar(F.cosine_similarity(
                predicted_clap.float(), reference_clap.float(), dim=-1).mean())
        audio_seconds = samples / sample_rate
        proportional_elapsed = elapsed_seconds * steps / max(PREDICTIVE_HORIZONS)
        horizons[str(steps)] = {
            "normalized_latent_error": _finite_scalar(latent_error),
            "normalized_delta_error": _finite_scalar(delta_error),
            "variance_ratio": _finite_scalar(variance_ratio),
            "stft": _finite_scalar(stft),
            "f0_cents": _finite_scalar(cents),
            "rms_error_db": _finite_scalar(rms_error),
            "clap_cosine": clap_cosine,
            "non_finite_count": non_finite,
            "realtime_factor": proportional_elapsed / max(audio_seconds, 1e-12),
        }
    failures = []
    if total_non_finite:
        failures.append("non_finite")
    if any(not 0.25 <= value <= 4.0 for value in all_variance_ratios):
        failures.append("variance_ratio")
    if stride_frames != 4:
        failures.append("control_stride")
    if runtime_underruns:
        failures.append("runtime_underrun")
    return {
        "schema": 1,
        "control_stride_frames": stride_frames,
        "control_stride_samples": stride_frames * samples_per_latent,
        "horizons": horizons,
        "gate": {
            "passed": not failures,
            "failures": failures,
            "variance_ratio_range": [0.25, 4.0],
            "non_finite_count": total_non_finite,
            "runtime_underruns": runtime_underruns,
        },
    }


def _fixed_predictive_control_rollout(
        model: PredictiveMidiBrave, history: Tensor, clap: Tensor,
        note: Tensor, velocity: Tensor, frames: int, stride_frames: int,
        excitation_seed: Tensor) -> tuple[Tensor, Tensor]:
    """Generate a fixed future while accepting a directly manipulated CLAP state."""
    result = rollout_blocks(
        model.predictor, history, model.project_clap(clap, frames),
        model.midi_control(note, velocity, frames), stride_frames)
    sequence = torch.cat((history, result.latent), dim=-1)
    decoded = model.decode_latents(
        sequence, clap, note, velocity, excitation_seed)
    future_start = history.shape[-1] * model.samples_per_latent
    return result.latent, decoded[..., future_start:]


def _predictive_control_partners(records: list[Any], source: Any) -> tuple[Any, Any] | None:
    midi_candidates = [
        record for record in records
        if (record.preset_id == source.preset_id
            and record.velocity == source.velocity
            and record.midi_note != source.midi_note)
    ]
    timbre_candidates = [
        record for record in records
        if (record.preset_id != source.preset_id
            and record.velocity == source.velocity
            and record.midi_note == source.midi_note)
    ]
    if not midi_candidates or not timbre_candidates:
        return None
    midi = max(midi_candidates, key=lambda record: (
        abs(record.midi_note - source.midi_note), record.sample_id))
    timbre = min(timbre_candidates, key=lambda record: record.sample_id)
    return midi, timbre


@torch.no_grad()
def evaluate_predictive(config_path: str, checkpoint_path: str, output_path: str,
                        sequences: int = 8, device_name: str = "cuda",
                        examples: int = 8) -> dict[str, Any]:
    """Evaluate true cached continuations for 1/8/32/128 rolling steps."""
    from .trainer import load_predictive_statistics

    config = Config.load(config_path)
    if config.predictive is None:
        raise ValueError("predictive evaluation requires a v3 config")
    if sequences <= 0:
        raise ValueError("predictive evaluation sequence count must be positive")
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 5:
        raise ValueError("predictive evaluation requires checkpoint format 5")
    contract = payload.get("predictive_contract")
    if not isinstance(contract, dict) or contract.get("stage") not in {
            "predictor", "rollout", "gan"}:
        raise ValueError(
            "predictive evaluation requires a predictor, rollout, or GAN checkpoint")
    statistics, rave_checkpoint_hash, statistics_hash = load_predictive_statistics(
        Path(config.data.cache_root) / "rave-statistics.npz", config)
    if contract.get("latent_statistics_hash") != statistics_hash:
        raise ValueError("evaluation statistics do not match the checkpoint")
    statistics = LatentStatistics(
        statistics.latent_std.to(device), statistics.delta_std.to(device),
        statistics.acceleration_std.to(device), statistics.floor)
    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    records = load_manifest(config.data.manifest)
    roots = configured_roots(config.data, config.data.manifest)
    p = config.predictive
    rollout_frames = max(PREDICTIVE_HORIZONS) * p.stride_frames
    total_frames = p.history_frames + rollout_frames
    control_frames = p.predictor_control_rollout_frames
    control_steps = control_frames // p.stride_frames
    if str(control_steps) not in {str(value) for value in PREDICTIVE_HORIZONS}:
        raise ValueError(
            "predictor control rollout must match a reported evaluation horizon")
    clap_evaluator = FrozenClapReconstructionObjective(
        config.data.clap_checkpoint, config.data.sample_rate, device,
        maximum_gradient_norm=0.0)
    control_metrics = MetricStore()
    control_rows = []
    control_nonfinite_count = 0
    reports = []
    seed_rows = []
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    audition_root = destination / "examples"
    audition_rows: list[dict[str, Any]] = []
    if examples > 0:
        audition_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        cache_path = Path(config.data.cache_root) / "rave" / f"{record.cache_id}.npz"
        if not cache_path.is_file():
            continue
        with np.load(cache_path, allow_pickle=False) as cached:
            latent = cached["latent"].astype(np.float32, copy=True)
            if (str(cached["checkpoint_hash"].item()) != rave_checkpoint_hash
                    or int(cached["hop"].item()) != p.samples_per_latent
                    or str(cached["sample_id"].item()) != record.sample_id
                    or latent.shape[0] != p.rave_latent_dim
                    or latent.shape[-1] < total_frames):
                continue
        audio = load_audio(record_audio_path(record, roots), config.data.sample_rate)
        required_samples = total_frames * p.samples_per_latent
        if len(audio) < required_samples:
            continue
        clap = np.load(
            Path(config.data.cache_root) / "clap" / f"{record.cache_id}.npy").astype(np.float32)
        history = torch.from_numpy(latent[:, :p.history_frames]).unsqueeze(0).to(device)
        clap_tensor = torch.from_numpy(clap).unsqueeze(0).to(device)
        note = torch.tensor([record.midi_note], device=device)
        velocity = torch.tensor([record.velocity], device=device, dtype=torch.float32)
        chunks = []
        current = history
        started = time.perf_counter()
        for _ in range(max(PREDICTIVE_HORIZONS)):
            prediction = model.predict_future(current, clap_tensor, note, velocity).latent
            consumed = prediction[..., :p.stride_frames]
            chunks.append(consumed)
            current = torch.cat((current, consumed), dim=-1)[..., -p.history_frames:]
        predicted_latent = torch.cat(chunks, dim=-1)
        decoded = model.decode_latents(
            torch.cat((history, predicted_latent), dim=-1),
            clap_tensor, note, velocity)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        future_start = p.history_frames * p.samples_per_latent
        predicted_audio = decoded[..., future_start:]
        reference_audio = torch.from_numpy(
            audio[future_start:required_samples].copy()).view(1, 1, -1).to(device)
        reference_latent = torch.from_numpy(
            latent[:, p.history_frames:total_frames]).unsqueeze(0).to(device)
        report = predictive_rollout_report(
            predicted_latent, reference_latent, predicted_audio, reference_audio,
            statistics, p.stride_frames, p.samples_per_latent, elapsed,
            config.data.sample_rate)
        partners = _predictive_control_partners(records, record)
        if partners is None:
            continue
        midi_record, timbre_record = partners
        timbre_cache_path = (
            Path(config.data.cache_root) / "rave" / f"{timbre_record.cache_id}.npz")
        timbre_clap_path = (
            Path(config.data.cache_root) / "clap" / f"{timbre_record.cache_id}.npy")
        if not timbre_cache_path.is_file() or not timbre_clap_path.is_file():
            continue
        with np.load(timbre_cache_path, allow_pickle=False) as cached:
            timbre_latent = cached["latent"].astype(np.float32, copy=True)
            if (str(cached["checkpoint_hash"].item()) != rave_checkpoint_hash
                    or timbre_latent.shape[0] != p.rave_latent_dim
                    or timbre_latent.shape[-1] < p.history_frames):
                continue
        target_clap = np.load(timbre_clap_path).astype(np.float32)
        source_clap = F.normalize(clap_tensor.float(), dim=-1)
        target_clap_tensor = F.normalize(
            torch.from_numpy(target_clap).unsqueeze(0).to(device).float(), dim=-1)
        interpolation_count = p.predictor_timbre_interpolation_steps + 1
        alpha = torch.linspace(
            0.0, 1.0, interpolation_count, device=device,
            dtype=source_clap.dtype)
        path_clap = F.normalize(torch.lerp(
            source_clap.expand(interpolation_count, -1),
            target_clap_tensor.expand(interpolation_count, -1),
            alpha[:, None]), dim=-1)
        path_history = history.expand(interpolation_count, -1, -1).contiguous()
        path_note = note.expand(interpolation_count)
        path_velocity = velocity.expand(interpolation_count)
        seed_value = int.from_bytes(
            hashlib.sha256(record.sample_id.encode()).digest()[:4], "little")
        path_seed = torch.full(
            (interpolation_count,), seed_value, device=device, dtype=torch.long)
        path_latent, path_audio = _fixed_predictive_control_rollout(
            model, path_history, path_clap, path_note, path_velocity,
            control_frames, p.stride_frames, path_seed)
        midi_note = torch.tensor([midi_record.midi_note], device=device)
        midi_latent, midi_audio = _fixed_predictive_control_rollout(
            model, history, source_clap, midi_note, velocity,
            control_frames, p.stride_frames, path_seed[:1])
        alternate_history = torch.from_numpy(
            timbre_latent[:, :p.history_frames]).unsqueeze(0).to(device)
        alternate_latent, _ = _fixed_predictive_control_rollout(
            model, alternate_history, target_clap_tensor, note, velocity,
            control_frames, p.stride_frames, path_seed[:1])

        control_nonfinite_count += int((~torch.isfinite(path_latent)).sum().item())
        control_nonfinite_count += int((~torch.isfinite(path_audio)).sum().item())
        control_nonfinite_count += int((~torch.isfinite(midi_latent)).sum().item())
        control_nonfinite_count += int((~torch.isfinite(midi_audio)).sum().item())
        control_nonfinite_count += int((~torch.isfinite(alternate_latent)).sum().item())
        path_latent = torch.nan_to_num(path_latent.float())
        path_audio = torch.nan_to_num(path_audio.float())
        midi_audio = torch.nan_to_num(midi_audio.float())
        alternate_latent = torch.nan_to_num(alternate_latent.float())
        washout_frames = min(p.history_frames, max(0, control_frames - 32))
        washout_samples = washout_frames * p.samples_per_latent
        path_eval_audio = path_audio[..., washout_samples:]
        midi_eval_audio = midi_audio[..., washout_samples:]
        valid_path = torch.full(
            (interpolation_count,), path_eval_audio.shape[-1],
            device=device, dtype=torch.long)
        path_embedding = clap_evaluator._embedding(
            path_eval_audio, valid_path, role="generated")
        midi_embedding = clap_evaluator._embedding(
            midi_eval_audio, valid_path[:1], role="generated")
        midi_timbre_cosine = F.cosine_similarity(
            midi_embedding, source_clap, dim=-1)
        endpoint_target_cosine = F.cosine_similarity(
            path_embedding[-1:], target_clap_tensor, dim=-1)
        endpoint_source_cosine = F.cosine_similarity(
            path_embedding[-1:], source_clap, dim=-1)
        direction = target_clap_tensor - source_clap
        progress = ((path_embedding - source_clap) * direction).sum(-1)
        progress = progress / direction.square().sum(-1).clamp_min(1e-8)
        monotonicity = progress.diff().gt(0.0).float().mean()
        desired_path = path_clap
        control_metrics.add("midi_timbre_preservation_cosine", midi_timbre_cosine)
        control_metrics.add("clap_control_target_cosine", endpoint_target_cosine)
        control_metrics.add("clap_control_source_cosine", endpoint_source_cosine)
        control_metrics.add(
            "clap_control_following",
            endpoint_target_cosine.gt(endpoint_source_cosine).float())
        control_metrics.add("timbre_path_monotonicity", monotonicity)
        control_metrics.add("timbre_path_target_cosine", F.cosine_similarity(
            path_embedding, desired_path, dim=-1))

        midi_errors, midi_medians, midi_periodicity = pitch_measurements_by_sample(
            midi_eval_audio, midi_note, config.data.sample_rate,
            config.data.pitch_hop_length)
        add_pitch_metrics(
            control_metrics, "swap", midi_errors, midi_periodicity,
            midi_note, velocity)
        for median in midi_medians:
            if median is None:
                continue
            source_offset = 100.0 * float(midi_record.midi_note - record.midi_note)
            control_metrics.add(
                "midi_swap_following",
                float(abs(median) < abs(median + source_offset)))
        timbre_notes = note.expand(interpolation_count - 1)
        timbre_velocities = velocity.expand(interpolation_count - 1)
        timbre_errors, _, timbre_periodicity = pitch_measurements_by_sample(
            path_eval_audio[1:], timbre_notes, config.data.sample_rate,
            config.data.pitch_hop_length)
        add_pitch_metrics(
            control_metrics, "timbre", timbre_errors, timbre_periodicity,
            timbre_notes, timbre_velocities)

        latent_scale, _, _ = statistics.scales(
            p.rave_latent_dim, device, path_latent.dtype)
        seed_distance = ((path_latent[-1:] - alternate_latent)
                         / latent_scale).square().mean(dim=1).sqrt()
        seed_window = min(p.history_frames, control_frames // 2)
        early_seed_distance = seed_distance[..., :seed_window].mean()
        late_seed_distance = seed_distance[..., -seed_window:].mean()
        seed_washout_ratio = late_seed_distance / early_seed_distance.clamp_min(1e-8)
        control_metrics.add("seed_washout_ratio", seed_washout_ratio)
        variance_ratio = report["horizons"][str(control_steps)]["variance_ratio"]
        if variance_ratio is not None:
            control_metrics.add("rollout_variance_ratio", float(variance_ratio))
        control_rows.append({
            "sample_id": record.sample_id,
            "midi_target_sample_id": midi_record.sample_id,
            "timbre_target_sample_id": timbre_record.sample_id,
            "source_note": int(record.midi_note),
            "midi_target_note": int(midi_record.midi_note),
            "path_alpha": [float(value) for value in alpha.cpu()],
            "path_progress": [float(value) for value in progress.cpu()],
            "seed_washout_ratio": float(seed_washout_ratio.cpu()),
        })
        if len(audition_rows) < examples:
            audition_frames = max(control_frames, 512)
            audition_samples = audition_frames * p.samples_per_latent
            source_rollout, source_render = _fixed_predictive_control_rollout(
                model, history, source_clap, note, velocity,
                audition_frames, p.stride_frames, path_seed[:1])
            _, midi_render = _fixed_predictive_control_rollout(
                model, history, source_clap, midi_note, velocity,
                audition_frames, p.stride_frames, path_seed[:1])
            _, timbre_render = _fixed_predictive_control_rollout(
                model, history, target_clap_tensor, note, velocity,
                audition_frames, p.stride_frames, path_seed[:1])

            def reference_segment(item: Any) -> np.ndarray:
                value = load_audio(
                    record_audio_path(item, roots), config.data.sample_rate)
                segment = value[future_start:future_start + audition_samples]
                if len(segment) < audition_samples:
                    segment = np.pad(segment, (0, audition_samples - len(segment)))
                return segment.astype(np.float32, copy=False)

            stem = f"{len(audition_rows):02d}-{record.sample_id}"
            waveforms = {
                "source-reference": reference_segment(record),
                "source-rollout": source_render[0, 0].float().cpu().numpy(),
                "midi-reference": reference_segment(midi_record),
                "midi-swap": midi_render[0, 0].float().cpu().numpy(),
                "timbre-reference": reference_segment(timbre_record),
                "timbre-direct": timbre_render[0, 0].float().cpu().numpy(),
            }
            tracks = []
            labels = {
                "source-reference": "原始延续",
                "source-rollout": "模型延续",
                "midi-reference": "目标音高真值",
                "midi-swap": "MIDI 改音高",
                "timbre-reference": "目标音色真值",
                "timbre-direct": "直接 CLAP 音色",
            }
            for kind, waveform in waveforms.items():
                filename = f"{stem}-{kind}.wav"
                sf.write(audition_root / filename, waveform,
                         config.data.sample_rate, subtype="FLOAT")
                tracks.append({
                    "kind": kind, "label": labels[kind],
                    "file": f"examples/{filename}",
                })
            audition_rows.append({
                "index": len(audition_rows),
                "source_id": record.sample_id,
                "source_preset": record.preset_id,
                "source_note": int(record.midi_note),
                "midi_target_id": midi_record.sample_id,
                "midi_target_note": int(midi_record.midi_note),
                "timbre_target_id": timbre_record.sample_id,
                "timbre_target_preset": timbre_record.preset_id,
                "duration_seconds": audition_samples / config.data.sample_rate,
                "tracks": tracks,
            })
        reports.append(report)
        seed_rows.append({
            "sample_id": record.sample_id,
            "seed_washout_ratio": float(seed_washout_ratio.cpu()),
        })
        if len(reports) >= sequences:
            break
    if not reports:
        raise ValueError("no cached render is long enough for a 128-step rollout")
    horizons: dict[str, dict[str, float | int | None]] = {}
    metric_names = tuple(reports[0]["horizons"]["1"])
    for horizon in (str(value) for value in PREDICTIVE_HORIZONS):
        aggregate: dict[str, float | int | None] = {}
        for name in metric_names:
            values = [report["horizons"][horizon][name] for report in reports]
            finite = [float(value) for value in values
                      if value is not None and math.isfinite(float(value))]
            aggregate[name] = (sum(finite) / len(finite) if finite else None)
        horizons[horizon] = aggregate
    control_summary = control_metrics.summary()
    control_gate = predictive_control_quality_gate(
        control_summary, control_nonfinite_count)
    continuation_failures = {
        failure for report in reports for failure in report["gate"]["failures"]}
    failures = sorted(continuation_failures | set(control_gate["failures"]))
    output = {
        "schema": 2, "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "latent_statistics_sha256": statistics_hash,
        "calibration_sha256": contract.get("calibration_hash"),
        "evaluated_sequences": len(reports), "seeds": seed_rows,
        "control_stride_frames": p.stride_frames,
        "control_stride_samples": p.stride_frames * p.samples_per_latent,
        "control_rollout_frames": control_frames,
        "control_seed_washout_frames": min(
            p.history_frames, max(0, control_frames - 32)),
        "horizons": horizons,
        "control_metrics": control_summary,
        "control_gate": control_gate,
        "gate": {"passed": not failures, "failures": failures,
                 "variance_ratio_range": [0.25, 4.0],
                 "control_thresholds": control_gate["thresholds"]},
        "notes": {
            "pitch_independence": (
                "MIDI changes target pitch while source CLAP remains fixed; timbre-path "
                "changes keep the source MIDI note fixed"),
            "timbre_navigation": (
                "CLAP controls are directly interpolated at fixed MIDI and identical "
                "source history; monotonic target-direction progress is gated"),
            "seed_washout": (
                "the same target controls roll from source and target-preset histories; "
                "late/early normalized latent distance must decrease"),
        },
    }
    (destination / "predictive-metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "predictive-control-paths.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in control_rows),
        encoding="utf-8")
    (destination / "audition-manifest.json").write_text(
        json.dumps({
            "schema": 1,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "sample_rate": config.data.sample_rate,
            "examples": audition_rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return output


class MetricStore:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, values: Tensor | list[float] | float) -> None:
        if isinstance(values, Tensor):
            items = values.detach().float().flatten().cpu().tolist()
        elif isinstance(values, list):
            items = values
        else:
            items = [values]
        self.values.setdefault(name, []).extend(
            float(value) for value in items if math.isfinite(float(value)))

    def summary(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self.values.items()):
            if not values:
                output[name] = {"count": 0}
                continue
            array = np.asarray(values, dtype=np.float64)
            output[name] = {
                "count": len(values),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "p90": float(np.quantile(array, 0.9)),
            }
        return output


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for name, value in batch.items()}


def mask_audio(audio: Tensor, valid_samples: Tensor | None) -> Tensor:
    if valid_samples is None:
        return audio
    positions = torch.arange(audio.shape[-1], device=audio.device)
    return audio * (positions[None, :] < valid_samples[:, None]).to(audio)[:, None]


def spectral_metrics(prediction: Tensor, target: Tensor, sample_rate: int) -> tuple[Tensor, Tensor]:
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        window = torch.hann_window(2048, device=prediction.device)
        predicted = torch.stft(prediction.float().squeeze(1), 2048, 512, 2048, window,
                               return_complex=True, pad_mode="constant").abs().clamp_min(1e-7)
        reference = torch.stft(target.float().squeeze(1), 2048, 512, 2048, window,
                               return_complex=True, pad_mode="constant").abs().clamp_min(1e-7)
        lsd = ((20.0 * (predicted.log10() - reference.log10())).square()
               .mean(dim=(-1, -2)).sqrt())
        frequency = torch.fft.rfftfreq(2048, 1.0 / sample_rate).to(prediction.device)
        high = frequency >= 0.8 * (sample_rate / 2.0)
        predicted_energy = predicted[:, high].square().mean(dim=(-1, -2)).clamp_min(1e-12)
        reference_energy = reference[:, high].square().mean(dim=(-1, -2)).clamp_min(1e-12)
        upper_band_error_db = (10.0 * torch.log10(predicted_energy / reference_energy)).abs()
    return lsd, upper_band_error_db


def transient_metrics(prediction: Tensor, target: Tensor, sample_rate: int) -> tuple[Tensor, Tensor, Tensor]:
    predicted_difference = prediction.float().diff(dim=-1).abs().squeeze(1)
    target_difference = target.float().diff(dim=-1).abs().squeeze(1)
    median = target_difference.median(dim=-1).values
    deviation = (target_difference - median[:, None]).abs().median(dim=-1).values
    threshold = (median + 12.0 * deviation).clamp_min(1e-4)
    generated_clicks = (predicted_difference > threshold[:, None]).float().sum(-1)
    target_clicks = (target_difference > threshold[:, None]).float().sum(-1)
    seconds = prediction.shape[-1] / sample_rate
    generated_rate = generated_clicks / seconds
    target_rate = target_clicks / seconds
    crest = (prediction.float().abs().amax(dim=(-1, -2))
             / prediction.float().square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-7))
    target_crest = (target.float().abs().amax(dim=(-1, -2))
                    / target.float().square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-7))
    return generated_rate, target_rate, (crest - target_crest).abs()


def ripple_error(prediction: Tensor, target: Tensor) -> Tensor:
    predicted = F.avg_pool1d(prediction.float().square(), 256, 128).clamp_min(1e-8).sqrt()
    reference = F.avg_pool1d(target.float().square(), 256, 128).clamp_min(1e-8).sqrt()
    predicted_curvature = predicted[..., 2:] - 2 * predicted[..., 1:-1] + predicted[..., :-2]
    reference_curvature = reference[..., 2:] - 2 * reference[..., 1:-1] + reference[..., :-2]
    return (predicted_curvature - reference_curvature).abs().mean(dim=(-1, -2))


@torch.no_grad()
def pitch_measurements_by_sample(
        audio: Tensor, notes: Tensor, sample_rate: int,
        hop_length: int) -> tuple[list[list[float]], list[float | None], list[list[float]]]:
    import torchcrepe

    # torchcrepe's public preprocessing contract is [1, time], and it flattens
    # a larger leading dimension into one temporal sequence. Decode each signal
    # independently so Viterbi state never crosses waveform boundaries. Calling
    # at the source sample rate deliberately reuses torchcrepe's official
    # resampy path, matching the dataset F0-cache implementation exactly.
    pitches: list[Tensor] = []
    periodicities: list[Tensor] = []
    for signal in audio.float().squeeze(1):
        item_pitch, item_periodicity = torchcrepe.predict(
            signal.unsqueeze(0), sample_rate, hop_length,
            50.0, 2000.0, "tiny", batch_size=1024,
            device=audio.device, return_periodicity=True,
        )
        pitches.append(item_pitch.squeeze(0))
        periodicities.append(item_periodicity.squeeze(0))
    pitch = torch.stack(pitches)
    periodicity = torch.stack(periodicities)
    target = midi_to_hz(notes).to(pitch)[:, None]
    cents = 1200.0 * torch.log2((pitch + 1e-7) / (target + 1e-7))
    frame_errors: list[list[float]] = []
    medians: list[float | None] = []
    frame_periodicity: list[list[float]] = []
    for index in range(audio.shape[0]):
        # Generated audio with poor periodicity is a model failure that must be
        # measured, not silently removed from the F0 denominator.  Retain every
        # finite CREPE frame for F0/octave/MIDI-following and expose periodicity
        # separately so pitch accuracy and voicing quality remain distinguishable.
        valid = torch.isfinite(cents[index]) & torch.isfinite(periodicity[index])
        values = cents[index, valid]
        periodicity_values = periodicity[index, valid]
        if values.numel():
            frame_errors.append(values.cpu().tolist())
            frame_periodicity.append(periodicity_values.cpu().tolist())
            medians.append(float(values.median().item()))
        else:
            frame_errors.append([])
            frame_periodicity.append([])
            medians.append(None)
    return frame_errors, medians, frame_periodicity


@torch.no_grad()
def pitch_measurements(audio: Tensor, notes: Tensor, sample_rate: int,
                       hop_length: int) -> tuple[list[float], list[float | None], list[float]]:
    errors, medians, periodicity = pitch_measurements_by_sample(
        audio, notes, sample_rate, hop_length)
    return ([value for item in errors for value in item], medians,
            [value for item in periodicity for value in item])


def _pitch_band(note: int) -> str:
    if note <= 47:
        return "low"
    if note <= 59:
        return "mid"
    return "high"


def add_pitch_metrics(metrics: MetricStore, prefix: str,
                      errors: list[list[float]], periodicity: list[list[float]],
                      notes: Tensor, velocities: Tensor) -> None:
    for index, (sample_errors, sample_periodicity) in enumerate(zip(errors, periodicity)):
        absolute = [abs(value) for value in sample_errors]
        octave = [float(abs(value) > 600.0) for value in sample_errors]
        low_periodicity = [float(value < 0.5) for value in sample_periodicity]
        metrics.add(f"{prefix}_f0_signed_cents", sample_errors)
        metrics.add(f"{prefix}_f0_absolute_cents", absolute)
        metrics.add(f"{prefix}_f0_octave_error", octave)
        metrics.add(f"{prefix}_f0_periodicity", sample_periodicity)
        metrics.add(f"{prefix}_f0_low_periodicity_rate", low_periodicity)
        band = _pitch_band(int(notes[index].item()))
        velocity = int(velocities[index].item())
        metrics.add(f"{prefix}_f0_absolute_cents_{band}", absolute)
        metrics.add(f"{prefix}_f0_octave_error_{band}", octave)
        metrics.add(f"{prefix}_f0_periodicity_{band}", sample_periodicity)
        metrics.add(f"{prefix}_f0_absolute_cents_v{velocity}", absolute)
        metrics.add(f"{prefix}_f0_periodicity_v{velocity}", sample_periodicity)


def save_examples(root: Path, offset: int, batch: dict[str, Any], self_audio: Tensor,
                  cross_audio: Tensor, sample_rate: int, maximum: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    saved = offset
    for index in range(self_audio.shape[0]):
        if saved >= maximum:
            break
        stem = f"{saved:04d}-{batch['sample_id_a'][index]}-to-{batch['sample_id_b'][index]}"
        values = {
            "target-a": batch["audio_a"][index], "generated-self": self_audio[index],
            "target-b": batch["audio_b"][index], "generated-cross": cross_audio[index],
        }
        for suffix, waveform in values.items():
            sf.write(root / f"{stem}-{suffix}.wav",
                     waveform.detach().float().squeeze().cpu().numpy(), sample_rate,
                     subtype="FLOAT")
        saved += 1
    return saved


@torch.no_grad()
def evaluate_predictive_rave(
    config_path: str, checkpoint_path: str, output_path: str,
    pairs: int = 32, batch_size: int = 2, examples: int = 8,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Evaluate reconstruction and counterfactual MIDI control in the RAVE stage."""
    if pairs <= 0 or batch_size <= 0 or examples < 0:
        raise ValueError("predictive RAVE evaluation counts must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("predictive RAVE evaluation requires an allocated CUDA device")
    config = Config.load(config_path)
    if config.predictive is None:
        raise ValueError("predictive RAVE evaluation requires a v3 config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = checkpoint.get("predictive_contract", {})
    if int(checkpoint.get("format", 0)) != 5 or contract.get("stage") != "rave":
        raise ValueError("predictive RAVE evaluation requires a format-5 RAVE checkpoint")

    # The current Pad50 manifest contains train rows only.  Keep this a fixed,
    # deterministic diagnostic subset and label it honestly in the report.
    data_config = replace(config.data, repeats=1, num_workers=2)
    dataset = PairDataset(
        data_config, config.seed + 30000, config.predictive, load_rave_cache=False)
    pitch_modes = {"pitch", "pitch_velocity"}
    indices = [index for index in range(len(dataset))
               if PairDataset.PAIR_SEQUENCE[index % len(PairDataset.PAIR_SEQUENCE)]
               in pitch_modes][:pairs]
    if not indices:
        raise ValueError("predictive RAVE evaluation found no cross-note validation pairs")
    evaluation_batch_size = max(batch_size, 8)
    loader = DataLoader(
        Subset(dataset, indices), batch_size=evaluation_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True)

    model = PredictiveMidiBrave(
        config.model, config.predictive, config.data.window_samples,
        config.data.sample_rate).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stft = MultiResolutionSTFTLoss().to(device)
    clap = FrozenClapReconstructionObjective(
        config.data.clap_checkpoint, config.data.sample_rate, device,
        maximum_gradient_norm=0.0)
    metrics = MetricStore()
    output = Path(output_path)
    example_root = output / "examples"
    example_root.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    saved = 0
    nonfinite_count = 0
    rows = []

    for raw_batch in loader:
        if evaluated >= pairs:
            break
        batch = move_batch(raw_batch, device)
        current = min(batch["audio_a"].shape[0], pairs - evaluated)
        audio = batch["audio_a"][:current]
        note = batch["note_a"][:current]
        swap_note = batch["note_b"][:current]
        velocity = batch["velocity_a"][:current]
        clap_control = batch["clap_a"][:current]
        preset_ids = list(raw_batch["preset_id"][:current])
        valid = batch["valid_samples_a"][:current]
        seed = batch.get("excitation_seed_a")
        if seed is not None:
            seed = seed[:current]
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            posterior = model.encode_audio(audio, sample=False)
            reconstruction = model.decode_latents(
                posterior.latent, clap_control, note, velocity, seed)
            swapped = model.decode_latents(
                posterior.latent, clap_control, swap_note, velocity, seed)
            target_index = torch.arange(current, device=device)
            for index, preset_id in enumerate(preset_ids):
                for offset in range(1, current):
                    candidate = (index + offset) % current
                    if preset_ids[candidate] != preset_id:
                        target_index[index] = candidate
                        break
            timbre_valid_mask = target_index.ne(
                torch.arange(current, device=device))
            timbre_control = clap_control.index_select(0, target_index)
            counterfactual_timbre = model.decode_latents(
                posterior.latent, timbre_control, note, velocity, seed)

        nonfinite_count += int((~torch.isfinite(reconstruction)).sum().item())
        nonfinite_count += int((~torch.isfinite(swapped)).sum().item())
        nonfinite_count += int((~torch.isfinite(
            counterfactual_timbre[timbre_valid_mask])).sum().item())
        safe_reconstruction = torch.nan_to_num(reconstruction.float())
        safe_swapped = torch.nan_to_num(swapped.float())
        masked_target = mask_audio(audio.float(), valid)
        masked_reconstruction = mask_audio(safe_reconstruction, valid)
        masked_swapped = mask_audio(safe_swapped, valid)
        metrics.add("reconstruction_l1", (masked_reconstruction - masked_target)
                    .abs().mean(dim=(-1, -2)))
        metrics.add("reconstruction_mr_stft", stft(
            masked_reconstruction, masked_target, valid))
        lsd, upper = spectral_metrics(
            masked_reconstruction, masked_target, config.data.sample_rate)
        metrics.add("reconstruction_lsd_db", lsd)
        metrics.add("reconstruction_upper_band_energy_error_db", upper)
        metrics.add("reconstruction_rms_error_db", (
            rms_db(masked_reconstruction, valid) - rms_db(masked_target, valid)).abs())

        pitch_results = {}
        for prefix, waveform, pitch_note in (
                ("target", masked_target, note),
                ("reconstruction", masked_reconstruction, note),
                ("swap", masked_swapped, swap_note)):
            errors, medians, periodicity = pitch_measurements_by_sample(
                waveform, pitch_note, config.data.sample_rate,
                config.data.pitch_hop_length)
            add_pitch_metrics(metrics, prefix, errors, periodicity,
                              pitch_note, velocity)
            pitch_results[prefix] = (errors, medians, periodicity)

        for index, median in enumerate(pitch_results["swap"][1]):
            if median is None:
                continue
            source_offset = 100.0 * float((swap_note[index] - note[index]).item())
            metrics.add("midi_swap_following",
                        float(abs(median) < abs(median + source_offset)))
            rows.append({
                "sample_id": raw_batch["sample_id_a"][index],
                "source_note": int(note[index].item()),
                "swap_note": int(swap_note[index].item()),
                "reconstruction_median_cents": pitch_results[
                    "reconstruction"][1][index],
                "swap_median_cents": median,
            })

        target_embedding = clap._embedding(masked_target, valid, role="target")
        reconstruction_embedding = clap._embedding(
            masked_reconstruction, valid, role="generated")
        swapped_embedding = clap._embedding(masked_swapped, valid, role="generated")
        metrics.add("reconstruction_clap_cosine", F.cosine_similarity(
            reconstruction_embedding, target_embedding, dim=-1))
        metrics.add("swap_clap_cosine", F.cosine_similarity(
            swapped_embedding, reconstruction_embedding, dim=-1))
        if bool(timbre_valid_mask.any().item()):
            masked_counterfactual = mask_audio(
                torch.nan_to_num(counterfactual_timbre.float()), valid)
            generated_timbre_embedding = clap._embedding(
                masked_counterfactual[timbre_valid_mask],
                valid[timbre_valid_mask], role="generated")
            target_timbre_embedding = F.normalize(
                timbre_control[timbre_valid_mask].float(), dim=-1)
            source_timbre_embedding = F.normalize(
                clap_control[timbre_valid_mask].float(), dim=-1)
            target_cosine = F.cosine_similarity(
                generated_timbre_embedding, target_timbre_embedding, dim=-1)
            source_cosine = F.cosine_similarity(
                generated_timbre_embedding, source_timbre_embedding, dim=-1)
            metrics.add("clap_control_target_cosine", target_cosine)
            metrics.add("clap_control_source_cosine", source_cosine)
            metrics.add("clap_control_following", target_cosine.gt(source_cosine).float())
        metrics.add("rave_pitch_adversary_accuracy", model.rave_pitch_logits(
            posterior.latent.float(), reversal_scale=0.0).argmax(-1).eq(note).float())

        while saved < min(examples, evaluated + current):
            index = saved - evaluated
            stem = (f"{saved:04d}-{raw_batch['sample_id_a'][index]}-"
                    f"n{int(note[index])}-to-n{int(swap_note[index])}")
            for suffix, waveform in (("target", masked_target[index]),
                                     ("reconstruction", masked_reconstruction[index]),
                                     ("swap", masked_swapped[index])):
                sf.write(example_root / f"{stem}-{suffix}.wav",
                         waveform.squeeze().cpu().numpy(), config.data.sample_rate,
                         subtype="FLOAT")
            saved += 1
        evaluated += current

    summary = metrics.summary()
    gate = predictive_rave_quality_gate(summary, nonfinite_count)
    report = {
        "schema": 1,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "stage_update": int(checkpoint["stage_update"]),
        "evaluation_split": data_config.split,
        "independent_holdout": False,
        "evaluated_pairs": evaluated,
        "metrics": summary,
        "gate": gate,
        "listening_examples": saved,
        "notes": {
            "swap": "same RAVE latent and CLAP control; only MIDI note is replaced",
            "clap_control": "same RAVE latent and MIDI note; CLAP control is replaced by a different preset",
            "gate": "early Phase-1 quality gate; failures require diagnosis before caching",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "pitch_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


@torch.no_grad()
def generate_grid(model: MidiBrave, dataset: PairDataset, config: Config,
                  output: Path, preset_count: int) -> dict[str, Any]:
    preset_rows = {}
    if config.data.preset_manifest:
        with Path(config.data.preset_manifest).open(encoding="utf-8") as handle:
            preset_rows = {row["preset_id"]: row for row in map(json.loads, handle)}
    by_preset: dict[str, Any] = {}
    for record in dataset.records:
        by_preset.setdefault(record.preset_id, record)
    selected = []
    used_categories = set()
    for preset_id, record in sorted(by_preset.items()):
        category = preset_rows.get(preset_id, {}).get("category", "unknown")
        if category not in used_categories:
            selected.append((record, category))
            used_categories.add(category)
        if len(selected) == preset_count:
            break
    for preset_id, record in sorted(by_preset.items()):
        if len(selected) == preset_count:
            break
        if all(existing.preset_id != preset_id for existing, _ in selected):
            selected.append((record, preset_rows.get(preset_id, {}).get("category", "unknown")))

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    device = next(model.parameters()).device
    notes = list(range(36, 72))
    for record, category in selected:
        clap = torch.from_numpy(np.load(
            Path(config.data.cache_root) / "clap" / f"{record.cache_id}.npy"
        ).astype(np.float32)).unsqueeze(0).to(device)
        z_timbre = model.timbre(clap)
        conditions = [(note, velocity) for note in notes for velocity in (50, 127)]
        for start in range(0, len(conditions), 4):
            chunk = conditions[start:start + 4]
            note = torch.tensor([item[0] for item in chunk], device=device)
            velocity = torch.tensor([item[1] for item in chunk], device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.float16):
                waveform = model.decode(z_timbre.expand(len(chunk), -1), note, velocity)
            for index, (midi_note, midi_velocity) in enumerate(chunk):
                relative = Path(record.preset_id) / f"n{midi_note:03d}-v{midi_velocity:03d}.wav"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, waveform[index].float().squeeze().cpu().numpy(),
                         config.data.sample_rate, subtype="FLOAT")
                rows.append({
                    "preset_id": record.preset_id, "category": category,
                    "reference_sample_id": record.sample_id,
                    "midi_note": midi_note, "velocity": midi_velocity,
                    "audio_path": str(relative),
                })
    manifest = output / "grid.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")
    return {"presets": len(selected), "audio_files": len(rows), "manifest": str(manifest)}


@torch.no_grad()
def evaluate(config_path: str, checkpoint_path: str, output_path: str,
             pairs: int, batch_size: int, examples: int, grid_presets: int,
             velocity_pairs: int = 64) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    config = Config.load(config_path)
    device = torch.device("cuda")
    data_config = replace(config.data, split="validation", repeats=1, num_workers=2)
    dataset = PairDataset(data_config, config.seed + 991)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    model = MidiBrave(config.model, config.data.window_samples, config.data.sample_rate).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics = MetricStore()
    stft = MultiResolutionSTFTLoss().to(device)
    timbre_a: list[Tensor] = []
    timbre_b: list[Tensor] = []
    preset_ids: list[str] = []
    pitch_rows: list[dict[str, Any]] = []
    evaluated = 0
    saved = 0
    output = Path(output_path)
    example_root = output / "examples"
    for raw_batch in loader:
        if evaluated >= pairs:
            break
        batch = move_batch(raw_batch, device)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(
                batch["clap_a"], batch["note_a"], batch["velocity_a"],
                batch["note_b"], batch["velocity_b"], batch["clap_b"], grl_scale=0.0,
                source_excitation_seed=batch.get("excitation_seed_a"),
                target_excitation_seed=batch.get("excitation_seed_b"))
        current = min(result.self_audio.shape[0], pairs - evaluated)
        self_audio = result.self_audio[:current]
        cross_audio = result.cross_audio[:current]
        audio_a = batch["audio_a"][:current]
        audio_b = batch["audio_b"][:current]
        note_a = batch["note_a"][:current]
        note_b = batch["note_b"][:current]

        valid_a = batch.get("valid_samples_a")
        valid_b = batch.get("valid_samples_b")
        metrics.add("self_mr_stft", stft(self_audio, audio_a, valid_a))
        metrics.add("cross_mr_stft", stft(cross_audio, audio_b, valid_b))
        for prefix, prediction, target, valid_samples in (
                ("self", self_audio, audio_a, valid_a),
                ("cross", cross_audio, audio_b, valid_b)):
            prediction = mask_audio(prediction, valid_samples)
            target = mask_audio(target, valid_samples)
            lsd, upper = spectral_metrics(prediction, target, config.data.sample_rate)
            generated_click, target_click, crest_error = transient_metrics(
                prediction, target, config.data.sample_rate)
            metrics.add(f"{prefix}_lsd_db", lsd)
            metrics.add(f"{prefix}_upper_band_energy_error_db", upper)
            metrics.add(f"{prefix}_generated_clicks_per_second", generated_click)
            metrics.add(f"{prefix}_target_clicks_per_second", target_click)
            metrics.add(f"{prefix}_crest_factor_error", crest_error)
            metrics.add(f"{prefix}_envelope_ripple_error", ripple_error(prediction, target))
            metrics.add(f"{prefix}_rms_error_db",
                        (rms_db(prediction, valid_samples) - rms_db(target, valid_samples)).abs())

        branch_pitch: dict[str, tuple[list[list[float]], list[float | None],
                                      list[list[float]]]] = {}
        for prefix, generated, target, notes, velocities, sample_ids in (
                ("self", mask_audio(self_audio, valid_a), mask_audio(audio_a, valid_a),
                 note_a, batch["velocity_a"][:current],
                 raw_batch["sample_id_a"][:current]),
                ("cross", mask_audio(cross_audio, valid_b), mask_audio(audio_b, valid_b),
                 note_b, batch["velocity_b"][:current],
                 raw_batch["sample_id_b"][:current])):
            errors, medians, periodicity = pitch_measurements_by_sample(
                generated, notes, config.data.sample_rate, config.data.pitch_hop_length)
            add_pitch_metrics(metrics, prefix, errors, periodicity, notes, velocities)
            for item in errors:
                metrics.add("f0_signed_cents", item)
                metrics.add("f0_absolute_cents", [abs(value) for value in item])
                metrics.add("f0_octave_error", [float(abs(value) > 600.0) for value in item])
            for item in periodicity:
                metrics.add("f0_periodicity", item)
                metrics.add("f0_low_periodicity_rate", [float(value < 0.5) for value in item])
            branch_pitch[prefix] = (errors, medians, periodicity)

            target_errors, _, target_periodicity = pitch_measurements_by_sample(
                target, notes, config.data.sample_rate, config.data.pitch_hop_length)
            add_pitch_metrics(metrics, f"target_{prefix}", target_errors,
                              target_periodicity, notes, velocities)
            for item in target_errors:
                metrics.add("target_f0_absolute_cents", [abs(value) for value in item])
                metrics.add("target_f0_octave_error",
                            [float(abs(value) > 600.0) for value in item])
            for item in target_periodicity:
                metrics.add("target_f0_periodicity", item)
                metrics.add("target_f0_low_periodicity_rate",
                            [float(value < 0.5) for value in item])

            for index, (sample_errors, sample_periodicity) in enumerate(
                    zip(errors, periodicity)):
                absolute = np.abs(np.asarray(sample_errors, dtype=np.float64))
                period = np.asarray(sample_periodicity, dtype=np.float64)
                pitch_rows.append({
                    "branch": prefix,
                    "sample_id": sample_ids[index],
                    "note": int(notes[index].item()),
                    "velocity": int(velocities[index].item()),
                    "frames": int(absolute.size),
                    "f0_absolute_median": (float(np.median(absolute))
                                           if absolute.size else None),
                    "f0_absolute_p90": (float(np.quantile(absolute, 0.9))
                                        if absolute.size else None),
                    "periodicity_median": (float(np.median(period)) if period.size else None),
                    "low_periodicity_rate": (float(np.mean(period < 0.5))
                                             if period.size else None),
                })

        cross_medians = branch_pitch["cross"][1]
        for index, median in enumerate(cross_medians):
            if median is None or note_a[index] == note_b[index]:
                continue
            source_offset = 100.0 * float((note_b[index] - note_a[index]).item())
            metrics.add("midi_swap_following", float(abs(median) < abs(median + source_offset)))

        target_delta = (batch["velocity_reference_rms_db_b"]
                        - batch["velocity_reference_rms_db_a"])
        prediction_delta = rms_db(cross_audio, valid_b) - rms_db(self_audio, valid_a)
        velocity_mask = (note_a.eq(note_b)
                         & batch["velocity_a"][:current].ne(batch["velocity_b"][:current])
                         & batch.get("velocity_known_a", torch.ones_like(note_a, dtype=torch.bool))[:current]
                         & batch.get("velocity_known_b", torch.ones_like(note_b, dtype=torch.bool))[:current]
                         & target_delta.abs().ge(config.loss.velocity_margin_db))
        if velocity_mask.any():
            direction = torch.sign(target_delta[velocity_mask])
            directed = direction * prediction_delta[velocity_mask]
            metrics.add("velocity_direction_accuracy", directed.gt(0).float())
            metrics.add("velocity_margin_accuracy",
                        directed.ge(config.loss.velocity_margin_db).float())
            metrics.add("velocity_delta_error_db",
                        (prediction_delta[velocity_mask] - target_delta[velocity_mask]).abs())

        metrics.add("same_preset_timbre_cosine", F.cosine_similarity(
            result.timbre[:current], result.target_timbre[:current], dim=-1))
        metrics.add("pitch_adversary_accuracy", result.pitch_logits[:current].argmax(-1)
                    .eq(note_a).float())
        timbre_a.append(result.timbre[:current].float().cpu())
        timbre_b.append(result.target_timbre[:current].float().cpu())
        preset_ids.extend(raw_batch["preset_id"][:current])
        saved = save_examples(example_root, saved, batch, self_audio, cross_audio,
                              config.data.sample_rate, examples)
        evaluated += current

    targeted_velocity = 0
    if velocity_pairs > 0:
        # One validation traversal contains fewer than 64 pairs whose rendered
        # loudness difference exceeds the 1 dB validity margin.  Extend only
        # this deterministic pair stream; the main 128/256-pair contract above
        # remains unchanged.  Higher indices use new deterministic RNG seeds.
        velocity_dataset = PairDataset(
            replace(data_config, repeats=max(4, data_config.repeats)),
            config.seed + 991,
        )
        velocity_indices = list(range(
            2, len(velocity_dataset), len(PairDataset.PAIR_SEQUENCE)))
        velocity_loader = DataLoader(
            Subset(velocity_dataset, velocity_indices), batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True)
        for raw_batch in velocity_loader:
            if targeted_velocity >= velocity_pairs:
                break
            batch = move_batch(raw_batch, device)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(
                    batch["clap_a"], batch["note_a"], batch["velocity_a"],
                    batch["note_b"], batch["velocity_b"], batch["clap_b"], grl_scale=0.0,
                    source_excitation_seed=batch.get("excitation_seed_a"),
                    target_excitation_seed=batch.get("excitation_seed_b"))
            target_delta = (batch["velocity_reference_rms_db_b"]
                            - batch["velocity_reference_rms_db_a"])
            prediction_delta = (rms_db(result.cross_audio, batch.get("valid_samples_b"))
                                - rms_db(result.self_audio, batch.get("valid_samples_a")))
            active = (batch["note_a"].eq(batch["note_b"])
                      & batch["velocity_a"].ne(batch["velocity_b"])
                      & batch.get("velocity_known_a", torch.ones_like(
                          batch["note_a"], dtype=torch.bool))
                      & batch.get("velocity_known_b", torch.ones_like(
                          batch["note_b"], dtype=torch.bool))
                      & target_delta.abs().ge(config.loss.velocity_margin_db))
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            remaining = velocity_pairs - targeted_velocity
            active_indices = active_indices[:remaining]
            if active_indices.numel():
                direction = torch.sign(target_delta.index_select(0, active_indices))
                target = target_delta.index_select(0, active_indices)
                prediction = prediction_delta.index_select(0, active_indices)
                directed = direction * prediction
                metrics.add("targeted_velocity_direction_accuracy", directed.gt(0).float())
                metrics.add("targeted_velocity_margin_accuracy",
                            directed.ge(config.loss.velocity_margin_db).float())
                metrics.add("targeted_velocity_delta_error_db", (prediction - target).abs())
                targeted_velocity += int(active_indices.numel())

    left = F.normalize(torch.cat(timbre_a), dim=-1)
    right = F.normalize(torch.cat(timbre_b), dim=-1)
    nearest = (left @ right.T).argmax(dim=-1)
    retrieval = [float(preset_ids[index] == preset_ids[candidate])
                 for index, candidate in enumerate(nearest.tolist())]
    metrics.add("timbre_preset_retrieval_at_1", retrieval)
    grid = generate_grid(model, dataset, config, output / "midi_grid", grid_presets)
    report = {
        "schema": 2,
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_phase": int(checkpoint["phase"]),
        "checkpoint_generator_updates": int(checkpoint["generator_updates"]),
        "evaluated_pairs": evaluated,
        "targeted_velocity_pairs": targeted_velocity,
        "metrics": metrics.summary(),
        "listening_examples": saved,
        "midi_grid": grid,
        "notes": {
            "upper_band_energy_error_db": "high-frequency reconstruction proxy, not a direct alias detector",
            "click_rate": "derivative outliers relative to each target waveform's robust threshold",
            "f0": "generated self+cross frames are unconditional; target_f0 is the matched evaluator control",
            "velocity": "target delta uses complete-render RMS because crop offsets are not model inputs",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "pitch_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pitch_rows),
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--grid-presets", type=int, default=6)
    parser.add_argument("--velocity-pairs", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.is_predictive:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        stage = checkpoint.get("predictive_contract", {}).get("stage")
        if stage == "rave":
            evaluate_predictive_rave(
                args.config, args.checkpoint, args.output, args.pairs,
                args.batch_size, args.examples, args.device)
        else:
            evaluate_predictive(
                args.config, args.checkpoint, args.output, args.pairs, args.device,
                args.examples)
    else:
        evaluate(args.config, args.checkpoint, args.output, args.pairs,
                 args.batch_size, args.examples, args.grid_presets, args.velocity_pairs)


if __name__ == "__main__":
    main()
