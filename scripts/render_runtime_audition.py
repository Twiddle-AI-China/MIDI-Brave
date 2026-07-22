from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from midibrave.config import Config
from midibrave.data import SampleRecord, load_manifest
from midibrave.export_predictive import PredictiveRuntime
from midibrave.latent_cache import load_latent_window
from midibrave.predictive_model import PredictiveMidiBrave
from midibrave.seed_bank import SeedBank
from midibrave.trainer import load_predictive_statistics


def _load_clap(cache_root: Path, record: SampleRecord) -> np.ndarray:
    value = np.load(cache_root / "clap" / f"{record.cache_id}.npy").astype(
        np.float32, copy=False)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"invalid CLAP cache for {record.sample_id}")
    return value.copy()


def _choose_diverse_timbres(
        groups: dict[str, list[SampleRecord]], cache_root: Path, count: int,
        rng: np.random.Generator,
) -> list[tuple[str, SampleRecord, np.ndarray]]:
    candidates: list[tuple[str, SampleRecord, np.ndarray]] = []
    for preset, records in sorted(groups.items()):
        record = next((item for item in records
                       if (cache_root / "clap" / f"{item.cache_id}.npy").is_file()), None)
        if record is not None:
            candidates.append((preset, record, _load_clap(cache_root, record)))
    if len(candidates) < count:
        raise ValueError(f"need {count} cached timbres, found {len(candidates)}")
    matrix = np.stack([item[2] for item in candidates])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
    selected = [int(rng.integers(0, len(candidates)))]
    while len(selected) < count:
        similarity = matrix @ matrix[selected].T
        nearest = similarity.max(axis=1)
        nearest[selected] = np.inf
        selected.append(int(np.argmin(nearest)))
    return [candidates[index] for index in selected]


def _choose_seed_records(
        records: list[SampleRecord], cache_root: Path, count: int,
        rng: np.random.Generator,
) -> list[SampleRecord]:
    eligible = [
        record for record in records
        if (cache_root / "rave" / f"{record.cache_id}.npz").is_file()
        and (cache_root / "clap" / f"{record.cache_id}.npy").is_file()
    ]
    if len(eligible) < count:
        raise ValueError(f"need {count} cached seeds, found {len(eligible)}")
    indices = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[int(index)] for index in indices]


def _audition_bank(
        config: Config, checkpoint_hash: str, seed_records: list[SampleRecord],
) -> SeedBank:
    if config.predictive is None:
        raise ValueError("runtime audition requires a predictive config")
    cache_root = Path(config.data.cache_root)
    p = config.predictive
    latents = [
        load_latent_window(
            cache_root / "rave" / f"{record.cache_id}.npz",
            record.sample_id, 0, p.history_frames, p.rave_latent_dim,
            p.samples_per_latent, checkpoint_hash,
        )
        for record in seed_records
    ]
    claps = [_load_clap(cache_root, record) for record in seed_records]
    return SeedBank(
        np.stack(latents), np.stack(claps),
        np.asarray([record.midi_note for record in seed_records]),
        np.asarray([record.velocity for record in seed_records]),
        [record.sample_id for record in seed_records],
        {
            "schema": 1,
            "architecture": p.architecture,
            "latent_dim": p.rave_latent_dim,
            "history_frames": p.history_frames,
            "samples_per_latent": p.samples_per_latent,
            "checkpoint_hash": checkpoint_hash,
        },
    )


@torch.inference_mode()
def render_runtime_audition(
        config_path: str, checkpoint_path: str, output_path: str,
        seed_count: int, timbre_count: int, note_count: int,
        duration_seconds: float, warmup_seconds: float, random_seed: int,
        device_name: str,
) -> dict[str, object]:
    if min(seed_count, timbre_count, note_count) <= 0:
        raise ValueError("audition axis counts must be positive")
    if duration_seconds <= 0.0 or warmup_seconds < 0.0:
        raise ValueError("audition durations are invalid")
    config = Config.load(config_path)
    if config.predictive is None:
        raise ValueError("runtime audition requires a predictive config")
    p = config.predictive
    statistics, rave_checkpoint_hash, statistics_hash = load_predictive_statistics(
        Path(config.data.cache_root) / "rave-statistics.npz", config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 5:
        raise ValueError("runtime audition requires a format-5 checkpoint")
    contract = payload.get("predictive_contract", {})
    if contract.get("stage") not in {"predictor", "rollout", "gan"}:
        raise ValueError("runtime audition requires a predictive checkpoint")
    records = load_manifest(config.data.manifest)
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        groups[record.preset_id].append(record)
    rng = np.random.default_rng(random_seed)
    cache_root = Path(config.data.cache_root)
    seeds = _choose_seed_records(records, cache_root, seed_count, rng)
    timbres = _choose_diverse_timbres(groups, cache_root, timbre_count, rng)
    bank = _audition_bank(config, rave_checkpoint_hash, seeds)

    available_notes = np.asarray(sorted({record.midi_note for record in records}), dtype=np.int64)
    note_indices = np.rint(np.linspace(0, len(available_notes) - 1, note_count)).astype(int)
    notes = available_notes[note_indices].tolist()
    velocity = int(max(record.velocity for record in records))

    device = torch.device(device_name)
    training_model = PredictiveMidiBrave(
        config.model, p, config.data.window_samples, config.data.sample_rate)
    training_model.load_state_dict(payload["model"])
    training_model.to(device).eval()
    runtime = PredictiveRuntime.from_training_model(
        training_model, bank, statistics.latent_std).to(device).eval()

    combinations = [
        (seed_index, timbre_index, note)
        for seed_index in range(seed_count)
        for timbre_index in range(timbre_count)
        for note in notes
    ]
    batch = len(combinations)
    seed_indices = torch.tensor(
        [item[0] for item in combinations], device=device, dtype=torch.long)
    note_tensor = torch.tensor(
        [item[2] for item in combinations], device=device, dtype=torch.long)
    velocity_tensor = torch.full(
        (batch,), float(velocity), device=device, dtype=torch.float32)
    target_clap = torch.from_numpy(np.stack([
        timbres[item[1]][2] for item in combinations
    ])).to(device)

    rave_history = runtime.seed_latents.index_select(0, seed_indices)
    seed_clap = runtime.seed_clap.index_select(0, seed_indices)
    clap_history = runtime._project_clap(seed_clap).unsqueeze(-1).expand(
        -1, -1, p.history_frames).clone()
    seed_notes = torch.from_numpy(bank.notes).to(device).index_select(0, seed_indices)
    seed_velocities = torch.from_numpy(bank.velocities).to(device).index_select(
        0, seed_indices)
    midi_history = runtime.midi(
        seed_notes, seed_velocities, p.history_frames, static_condition=False)
    clap_future = runtime._future_clap(target_clap, batch)
    midi_future = runtime.midi(
        note_tensor, velocity_tensor, p.horizon_frames, static_condition=False)

    emitted_samples = p.stride_frames * p.samples_per_latent
    render_steps = math.ceil(
        duration_seconds * config.data.sample_rate / emitted_samples)
    warmup_steps = math.ceil(
        warmup_seconds * config.data.sample_rate / emitted_samples)
    total_steps = warmup_steps + render_steps
    total_frames = p.history_frames + total_steps * p.stride_frames
    total_samples = total_frames * p.samples_per_latent
    excitation = training_model._excitation_bands(
        note_tensor, total_samples,
        torch.full((batch,), random_seed, device=device, dtype=torch.long))
    excitation_ratio = runtime.decoder.total_ratio
    context_frames = p.history_frames + p.stride_frames
    context_samples = context_frames * p.samples_per_latent
    context_excitation_frames = context_frames * excitation_ratio

    rendered: list[torch.Tensor] = []
    for step in range(total_steps):
        prediction = runtime.predictor(rave_history, clap_future, midi_future).latent
        consumed = prediction[..., :p.stride_frames]
        context_rave = torch.cat((rave_history, consumed), dim=-1)
        context_clap = torch.cat(
            (clap_history, clap_future[..., :p.stride_frames]), dim=-1)
        context_midi = torch.cat(
            (midi_history, midi_future[..., :p.stride_frames]), dim=-1)
        excitation_start = step * p.stride_frames * excitation_ratio
        context_excitation = excitation[
            ..., excitation_start:excitation_start + context_excitation_frames]
        decoded = runtime.decoder(
            context_clap, context_midi, context_excitation,
            context_samples, context_rave)
        rave_history = context_rave[..., -p.history_frames:]
        clap_history = context_clap[..., -p.history_frames:]
        midi_history = context_midi[..., -p.history_frames:]
        if step >= warmup_steps:
            rendered.append(decoded[..., -emitted_samples:].float().cpu())

    audio = torch.cat(rendered, dim=-1).numpy()[:, 0]
    wanted_samples = round(duration_seconds * config.data.sample_rate)
    audio = audio[:, :wanted_samples]
    if not np.isfinite(audio).all():
        raise RuntimeError("runtime audition produced NaN/Inf audio")

    destination = Path(output_path)
    wav_root = destination / "runtime-examples"
    wav_root.mkdir(parents=True, exist_ok=True)
    clips = []
    for row, (seed_index, timbre_index, note) in enumerate(combinations):
        filename = f"s{seed_index}-t{timbre_index}-n{note:03d}.wav"
        sf.write(wav_root / filename, audio[row], config.data.sample_rate, subtype="FLOAT")
        clips.append({
            "seed": seed_index,
            "timbre": timbre_index,
            "note": note,
            "file": f"runtime-examples/{filename}",
        })

    manifest = {
        "schema": 2,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(payload.get("stage_update", payload.get("update", 0))),
        "sample_rate": config.data.sample_rate,
        "duration_seconds": wanted_samples / config.data.sample_rate,
        "architecture": {
            "encoder_free": True,
            "rave_dim": p.rave_latent_dim,
            "clap_dim": config.model.clap_dim,
            "clap_control_dim": p.clap_control_dim,
            "midi_dim": config.model.midi_dim,
            "history_frames": p.history_frames,
            "horizon_frames": p.horizon_frames,
            "stride_frames": p.stride_frames,
            "samples_per_latent": p.samples_per_latent,
            "aligned_control_history": True,
        },
        "latent_statistics_sha256": statistics_hash,
        "seeds": [
            {"id": index, "label": f"Seed {chr(65 + index)}", "sample_id": record.sample_id}
            for index, record in enumerate(seeds)
        ],
        "timbres": [
            {"id": index, "label": f"Timbre {index + 1}", "preset_id": preset}
            for index, (preset, _record, _clap) in enumerate(timbres)
        ],
        "notes": notes,
        "velocity": velocity,
        "clips": clips,
    }
    (destination / "runtime-audition-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination.resolve()),
        "clips": len(clips),
        "seeds": seed_count,
        "timbres": timbre_count,
        "notes": notes,
        "duration_seconds": manifest["duration_seconds"],
        "finite": True,
    }, ensure_ascii=False, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an encoder-free runtime audition grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timbres", type=int, default=4)
    parser.add_argument("--notes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--warmup", type=float, default=0.4)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_runtime_audition(
        args.config, args.checkpoint, args.output,
        args.seeds, args.timbres, args.notes,
        args.duration, args.warmup, args.random_seed, args.device)


if __name__ == "__main__":
    main()
