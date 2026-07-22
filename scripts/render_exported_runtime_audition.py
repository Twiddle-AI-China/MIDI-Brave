from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

from midibrave.config import Config
from midibrave.data import SampleRecord, load_manifest
from midibrave.seed_bank import SeedBank


TimbreCandidate = tuple[str, str, np.ndarray]


def select_notes(available_notes: Iterable[int], count: int) -> list[int]:
    notes = sorted({int(note) for note in available_notes})
    if count <= 0 or count > len(notes):
        raise ValueError("note count must fit the available training notes")
    indices = np.rint(np.linspace(0, len(notes) - 1, count)).astype(np.int64)
    selected = [notes[int(index)] for index in indices]
    if len(set(selected)) != count:
        raise ValueError("note selection did not produce unique controls")
    return selected


def select_diverse_timbres(
        candidates: Sequence[TimbreCandidate], count: int,
        random_seed: int,
) -> list[TimbreCandidate]:
    if count <= 0 or count > len(candidates):
        raise ValueError("timbre count must fit the cached Pad candidates")
    matrix = np.stack([np.asarray(item[2], dtype=np.float32) for item in candidates])
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("CLAP candidates must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("CLAP candidates must have non-zero norm")
    normalized = matrix / norms
    rng = np.random.default_rng(random_seed)
    selected = [int(rng.integers(0, len(candidates)))]
    while len(selected) < count:
        nearest_similarity = (normalized @ normalized[selected].T).max(axis=1)
        nearest_similarity[selected] = np.inf
        selected.append(int(np.argmin(nearest_similarity)))
    return [candidates[index] for index in selected]


def resolve_seed_selections(
        bank: SeedBank, seed_anchor_clap: np.ndarray, seed_count: int,
        random_seed: int, top_k: int = 8,
) -> list[dict[str, object]]:
    if seed_count <= 0:
        raise ValueError("seed count must be positive")
    anchor = torch.from_numpy(
        np.asarray(seed_anchor_clap, dtype=np.float32)).view(1, -1)
    controls = torch.nn.functional.normalize(anchor, dim=1)
    candidates = torch.nn.functional.normalize(
        torch.from_numpy(bank.clap.copy()).float(), dim=1)
    similarities = controls @ candidates.transpose(0, 1)
    count = min(max(1, top_k), bank.clap.shape[0])
    values, indices = torch.topk(similarities, count, dim=1)
    if seed_count > count:
        raise ValueError("seed count exceeds the distinct top-k candidates")
    selected: list[dict[str, object]] = []
    for seed_id in range(seed_count):
        choice = (random_seed + seed_id) % count
        bank_index = int(indices[0, choice].item())
        selected.append({
            "id": seed_id,
            "label": f"Seed {chr(65 + seed_id)}",
            "bank_index": bank_index,
            "sample_id": bank.sample_ids[bank_index],
            "distance": max(0.0, 1.0 - float(values[0, choice].item())),
        })
    return selected


def validate_runtime_metadata(path: str | Path) -> dict[str, object]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("encoder_free") is not True:
        raise ValueError("runtime metadata must prove encoder_free=true")
    for field in (
            "stride_frames", "samples_per_latent", "checkpoint_sha256",
            "seed_bank_npz_sha256", "seed_bank_json_sha256",
            "latent_statistics_sha256", "architecture", "history_frames",
            "horizon_frames", "runtime_sha256", "sample_rate"):
        if field not in metadata:
            raise ValueError(f"runtime metadata is missing {field}")
    return metadata


def build_clip_manifest(
        seed_count: int, timbres: Sequence[TimbreCandidate], notes: Sequence[int],
) -> tuple[list[dict[str, int]], list[dict[str, object]]]:
    combinations: list[dict[str, int]] = []
    clips: list[dict[str, object]] = []
    for seed in range(seed_count):
        for timbre in range(len(timbres)):
            for note in notes:
                row = {"seed": seed, "timbre": timbre, "note": int(note)}
                combinations.append(row)
                clips.append({
                    **row,
                    "file": f"runtime-examples/s{seed}-t{timbre}-n{int(note):03d}.wav",
                })
    return combinations, clips


@torch.inference_mode()
def _render_audio_matrix(
        runtime, claps: np.ndarray, seed_anchor_clap: np.ndarray,
        combinations: Sequence[dict[str, int]],
        duration_samples: int, warmup_samples: int, block_samples: int,
        velocity: float, random_seed: int, device: torch.device,
) -> np.ndarray:
    batch = len(combinations)
    if batch == 0 or claps.shape[0] != batch:
        raise ValueError("CLAP controls must align with audition combinations")
    if duration_samples <= 0 or block_samples <= 0:
        raise ValueError("audition sample counts must be positive")
    clap_tensor = torch.from_numpy(
        np.asarray(claps, dtype=np.float32)).to(device=device)
    anchor = torch.from_numpy(
        np.asarray(seed_anchor_clap, dtype=np.float32)).to(device=device)
    if anchor.ndim != 1 or anchor.shape[0] != clap_tensor.shape[1]:
        raise ValueError("seed anchor CLAP dimension does not match target controls")
    anchor = anchor.unsqueeze(0)
    warmup_blocks = math.ceil(warmup_samples / block_samples)
    render_blocks = math.ceil(duration_samples / block_samples)
    rows: list[np.ndarray] = []
    for index, combination in enumerate(combinations):
        history = runtime.initial_state(
            anchor, random_seed + int(combination["seed"]), 8)
        target_clap = clap_tensor[index:index + 1]
        note = torch.tensor(
            [combination["note"]], device=device, dtype=torch.long)
        target_velocity = torch.full(
            (1,), float(velocity), device=device, dtype=torch.float32)
        rendered: list[torch.Tensor] = []
        for step in range(warmup_blocks + render_blocks):
            block, history = runtime.step(
                history, target_clap, note, target_velocity)
            if tuple(block.shape) != (1, 1, block_samples):
                raise RuntimeError("runtime emitted an invalid audio block shape")
            if not torch.isfinite(block).all() or not torch.isfinite(history).all():
                raise RuntimeError("runtime audition produced NaN/Inf")
            if step >= warmup_blocks:
                rendered.append(block[0, 0].float().cpu())
        rows.append(torch.cat(rendered).numpy()[:duration_samples])
    audio = np.stack(rows)
    if audio.shape != (batch, duration_samples) or not np.isfinite(audio).all():
        raise RuntimeError("runtime audition produced invalid or NaN/Inf audio")
    return audio


def _load_clap(cache_root: Path, record: SampleRecord) -> np.ndarray:
    path = cache_root / "clap" / f"{record.cache_id}.npy"
    value = np.load(path).astype(np.float32, copy=False)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"invalid CLAP cache for {record.sample_id}")
    return value.copy()


def _candidate_timbres(
        records: Sequence[SampleRecord], cache_root: Path,
) -> list[TimbreCandidate]:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        groups[record.preset_id].append(record)
    candidates: list[TimbreCandidate] = []
    for preset_id, group in sorted(groups.items()):
        ordered = sorted(
            group, key=lambda item: (abs(int(item.midi_note) - 60),
                                     -float(item.velocity), item.sample_id))
        record = next((item for item in ordered if (
            cache_root / "clap" / f"{item.cache_id}.npy").is_file()), None)
        if record is not None:
            candidates.append((preset_id, record.sample_id,
                               _load_clap(cache_root, record)))
    return candidates


def render_exported_runtime_audition(
        config_path: str, runtime_path: str, runtime_metadata_path: str,
        seed_bank_path: str, output_path: str, seed_count: int,
        timbre_count: int, note_count: int, duration_seconds: float,
        warmup_seconds: float, random_seed: int, device_name: str,
) -> dict[str, object]:
    if min(seed_count, timbre_count, note_count) <= 0:
        raise ValueError("audition axis counts must be positive")
    if duration_seconds <= 0.0 or warmup_seconds < 0.0:
        raise ValueError("audition durations are invalid")
    config = Config.load(config_path)
    if config.predictive is None:
        raise ValueError("runtime audition requires a predictive config")
    metadata = validate_runtime_metadata(runtime_metadata_path)
    measured_runtime_hash = hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest()
    if measured_runtime_hash != metadata["runtime_sha256"]:
        raise ValueError("runtime metadata runtime_sha256 does not match runtime.pt")
    if int(metadata["sample_rate"]) != int(config.data.sample_rate):
        raise ValueError("runtime metadata and config disagree on sample_rate")
    for field, expected in {
        "architecture": config.predictive.architecture,
        "history_frames": config.predictive.history_frames,
        "horizon_frames": config.predictive.horizon_frames,
        "stride_frames": config.predictive.stride_frames,
        "samples_per_latent": config.predictive.samples_per_latent,
    }.items():
        if metadata[field] != expected:
            raise ValueError(f"runtime metadata and config disagree on {field}")
    seed_stem = Path(seed_bank_path)
    seed_npz = seed_stem.with_suffix(".npz")
    seed_json = seed_stem.with_suffix(".json")
    for path, field in (
            (seed_npz, "seed_bank_npz_sha256"),
            (seed_json, "seed_bank_json_sha256")):
        measured = hashlib.sha256(path.read_bytes()).hexdigest()
        if measured != metadata[field]:
            raise ValueError(f"runtime metadata {field} does not match the seed bank")
    bank = SeedBank.load(seed_stem, {
        "architecture": config.predictive.architecture,
        "latent_dim": config.predictive.rave_latent_dim,
        "history_frames": config.predictive.history_frames,
        "samples_per_latent": config.predictive.samples_per_latent,
    })
    block_samples = int(metadata["stride_frames"]) * int(
        metadata["samples_per_latent"])
    records = load_manifest(config.data.manifest)
    notes = select_notes((record.midi_note for record in records), note_count)
    cache_root = Path(config.data.cache_root)
    timbres = select_diverse_timbres(
        _candidate_timbres(records, cache_root), timbre_count, random_seed)
    seed_anchor = timbres[0][2]
    seed_selections = resolve_seed_selections(
        bank, seed_anchor, seed_count, random_seed)
    combinations, clips = build_clip_manifest(seed_count, timbres, notes)
    clap_matrix = np.stack([
        timbres[row["timbre"]][2] for row in combinations
    ]).astype(np.float32, copy=False)
    device = torch.device(device_name)
    runtime = torch.jit.load(str(runtime_path), map_location=device).to(device).eval()
    anchor_tensor = torch.from_numpy(seed_anchor).to(device=device).unsqueeze(0)
    for selection in seed_selections:
        history = runtime.initial_state(
            anchor_tensor, random_seed + int(selection["id"]), 8)
        expected = torch.from_numpy(
            bank.latents[int(selection["bank_index"])]).to(device=device)
        if not torch.allclose(history[0], expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError("runtime seed selection does not match the audited seed bank")
    duration_samples = round(duration_seconds * config.data.sample_rate)
    warmup_samples = round(warmup_seconds * config.data.sample_rate)
    velocity = float(max(record.velocity for record in records))
    audio = _render_audio_matrix(
        runtime, clap_matrix, seed_anchor, combinations, duration_samples,
        warmup_samples, block_samples, velocity, random_seed, device)

    destination = Path(output_path)
    wav_root = destination / "runtime-examples"
    wav_root.mkdir(parents=True, exist_ok=True)
    for row, clip in enumerate(clips):
        sf.write(destination / str(clip["file"]), audio[row],
                 config.data.sample_rate, subtype="FLOAT")
    shutil.copy2(runtime_metadata_path, destination / "runtime.pt.json")
    manifest: dict[str, object] = {
        "schema": 3,
        "encoder_free": True,
        "checkpoint": f"sha256:{metadata['checkpoint_sha256']}",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "runtime_sha256": measured_runtime_hash,
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "runtime_metadata": "runtime.pt.json",
        "sample_rate": int(config.data.sample_rate),
        "duration_seconds": duration_samples / config.data.sample_rate,
        "warmup_seconds": warmup_samples / config.data.sample_rate,
        "warmup_blocks": math.ceil(warmup_samples / block_samples),
        "block_samples": block_samples,
        "velocity": int(velocity),
        "random_seed": int(random_seed),
        "architecture": {
            field: metadata[field] for field in (
                "architecture", "history_frames", "horizon_frames",
                "stride_frames", "samples_per_latent")
        },
        "seed_anchor": {
            "timbre": 0, "preset_id": timbres[0][0],
            "sample_id": timbres[0][1],
        },
        "seeds": seed_selections,
        "timbres": [
            {"id": index, "label": f"Pad {index + 1}",
             "preset_id": item[0], "sample_id": item[1]}
            for index, item in enumerate(timbres)
        ],
        "notes": notes,
        "clips": clips,
    }
    if len(clips) != seed_count * timbre_count * note_count:
        raise RuntimeError("runtime audition matrix is incomplete")
    manifest_path = destination / "runtime-audition-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(destination.resolve()), "clips": len(clips),
        "notes": notes, "finite": True, "encoder_free": True,
    }, ensure_ascii=False, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an audition matrix from an encoder-free runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--runtime-metadata")
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timbres", type=int, default=4)
    parser.add_argument("--notes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--warmup", type=float, default=0.4)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runtime_metadata = args.runtime_metadata or str(
        Path(args.runtime).with_suffix(Path(args.runtime).suffix + ".json"))
    render_exported_runtime_audition(
        args.config, args.runtime, runtime_metadata, args.seed_bank, args.output,
        args.seeds, args.timbres, args.notes, args.duration, args.warmup,
        args.random_seed, args.device)


if __name__ == "__main__":
    main()
