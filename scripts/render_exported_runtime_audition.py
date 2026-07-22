from __future__ import annotations

import argparse
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


def validate_runtime_metadata(path: str | Path) -> dict[str, object]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("encoder_free") is not True:
        raise ValueError("runtime metadata must prove encoder_free=true")
    for field in ("stride_frames", "samples_per_latent", "checkpoint_sha256"):
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
        runtime, claps: np.ndarray, combinations: Sequence[dict[str, int]],
        duration_samples: int, warmup_samples: int, block_samples: int,
        velocity: float, random_seed: int, device: torch.device,
) -> np.ndarray:
    batch = len(combinations)
    if batch == 0 or claps.shape[0] != batch:
        raise ValueError("CLAP controls must align with audition combinations")
    clap_tensor = torch.from_numpy(
        np.asarray(claps, dtype=np.float32)).to(device=device)
    histories = []
    for index, combination in enumerate(combinations):
        histories.append(runtime.initial_state(
            clap_tensor[index:index + 1],
            random_seed + int(combination["seed"]), 8,
        ))
    history = torch.cat(histories, dim=0)
    notes = torch.tensor(
        [row["note"] for row in combinations], device=device, dtype=torch.long)
    velocities = torch.full(
        (batch,), float(velocity), device=device, dtype=torch.float32)
    warmup_blocks = math.ceil(warmup_samples / block_samples)
    render_blocks = math.ceil(duration_samples / block_samples)
    rendered: list[torch.Tensor] = []
    for step in range(warmup_blocks + render_blocks):
        block, history = runtime.step(history, clap_tensor, notes, velocities)
        if tuple(block.shape) != (batch, 1, block_samples):
            raise RuntimeError("runtime emitted an invalid audio block shape")
        if not torch.isfinite(block).all() or not torch.isfinite(history).all():
            raise RuntimeError("runtime audition produced NaN/Inf")
        if step >= warmup_blocks:
            rendered.append(block[:, 0].float().cpu())
    audio = torch.cat(rendered, dim=-1).numpy()[:, :duration_samples]
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
        output_path: str, seed_count: int, timbre_count: int, note_count: int,
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
    metadata = validate_runtime_metadata(runtime_metadata_path)
    block_samples = int(metadata["stride_frames"]) * int(
        metadata["samples_per_latent"])
    records = load_manifest(config.data.manifest)
    notes = select_notes((record.midi_note for record in records), note_count)
    cache_root = Path(config.data.cache_root)
    timbres = select_diverse_timbres(
        _candidate_timbres(records, cache_root), timbre_count, random_seed)
    combinations, clips = build_clip_manifest(seed_count, timbres, notes)
    clap_matrix = np.stack([
        timbres[row["timbre"]][2] for row in combinations
    ]).astype(np.float32, copy=False)
    device = torch.device(device_name)
    runtime = torch.jit.load(str(runtime_path), map_location=device).to(device).eval()
    duration_samples = round(duration_seconds * config.data.sample_rate)
    warmup_samples = round(warmup_seconds * config.data.sample_rate)
    velocity = float(max(record.velocity for record in records))
    audio = _render_audio_matrix(
        runtime, clap_matrix, combinations, duration_samples, warmup_samples,
        block_samples, velocity, random_seed, device)

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
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "runtime_metadata": "runtime.pt.json",
        "sample_rate": int(config.data.sample_rate),
        "duration_seconds": duration_samples / config.data.sample_rate,
        "velocity": int(velocity),
        "random_seed": int(random_seed),
        "architecture": metadata,
        "seeds": [
            {"id": index, "label": f"Seed {chr(65 + index)}"}
            for index in range(seed_count)
        ],
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
        args.config, args.runtime, runtime_metadata, args.output,
        args.seeds, args.timbres, args.notes, args.duration, args.warmup,
        args.random_seed, args.device)


if __name__ == "__main__":
    main()
