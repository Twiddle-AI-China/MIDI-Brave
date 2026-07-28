from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn

from .zrave_audition import decode_prerolled_latent
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_data import _load_rave_audio
from .zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    sample_pure_flow_block,
)


_DEFAULT_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")
_GENERATION_SEEDS = (17, 29)
_CHECKPOINT_NAME = re.compile(r"^step-(\d+)\.pt$")


def rollout_pure_flow(
    model: object,
    statistics: FlowStatistics,
    history: Tensor,
    frames: int,
    *,
    generation_seed: int,
    temperature: float,
    wander_delay_frames: int,
    solver_steps: int,
    sample_block: Callable[..., Tensor] = sample_pure_flow_block,
) -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if history.ndim != 3 or history.shape[1:] != (32, 16):
        raise ValueError("history must have shape [batch, 32, 16]")
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    block_index = 0
    while remaining:
        generated = sample_block(
            model,
            statistics,
            current,
            generation_seed=generation_seed,
            block_index=block_index,
            temperature=temperature,
            wander_delay_frames=wander_delay_frames,
            solver_steps=solver_steps,
        )
        expected = (history.shape[0], 64, 16)
        if generated.shape != expected:
            raise ValueError(
                "pure flow block sampler must return [batch, 64, 16]"
            )
        take = min(remaining, 64)
        chunk = generated[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[:, -32:].detach()
        remaining -= take
        block_index += 1
    return torch.cat(chunks, dim=1)


def select_audition_cases(
    sequence_manifest: str | Path,
    *,
    categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
    seed: int = 20260728,
) -> list[dict[str, Any]]:
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be non-empty and unique")
    selected: dict[str, tuple[bytes, dict[str, Any]]] = {}
    path = Path(sequence_manifest)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            category = str(row.get("category", ""))
            if (
                category not in categories
                or str(row.get("split")) != "test"
                or str(row.get("source_name")) != "serum_full"
                or int(row.get("active_frames", 0)) < 96
                or int(row.get("maximum_future_frames", 0)) < 64
            ):
                continue
            key = (
                f"{seed}:{category}:"
                f"{row.get('canonical_preset_id')}:"
                f"{row.get('sample_id')}"
            )
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            current = selected.get(category)
            if current is None or digest < current[0]:
                selected[category] = (digest, row)
    missing = [category for category in categories if category not in selected]
    if missing:
        raise ValueError(
            "test split lacks eligible audition rows: "
            + ", ".join(missing)
        )
    return [selected[category][1] for category in categories]


def validate_pure_checkpoint(
    payload: object,
    config: object,
    *,
    index_hash: str,
    statistics_hash: str,
    expected_update: int | None = None,
) -> None:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("pure flow checkpoint format mismatch")
    if payload.get("architecture") != "zrave_pure_flow_transformer_v1":
        raise ValueError("checkpoint is not a pure flow model")
    model = getattr(config, "model")
    if bool(getattr(model, "pitch_conditioning")):
        raise ValueError("audition config is not pure flow")
    if expected_update is not None and int(payload.get("update", -1)) != int(
        expected_update
    ):
        raise ValueError("pure flow checkpoint update mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("pure flow checkpoint lacks contract")
    expected = {
        "pack_index_sha256": index_hash,
        "statistics_sha256": statistics_hash,
        "latent_dim": int(getattr(model, "latent_dim")),
        "context_frames": int(getattr(model, "context_frames")),
        "future_frames": int(getattr(model, "future_frames")),
        "pitch_conditioning": False,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"pure flow checkpoint {name} mismatch")


def audition_file_names(
    case_index: int,
    category: str,
) -> dict[str, str]:
    if case_index < 0:
        raise ValueError("case index must be non-negative")
    slug = category.casefold().replace(" ", "-")
    if not slug or not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("category cannot form a safe file name")
    prefix = f"{case_index:02d}-{slug}"
    return {
        "source": f"wavs/{prefix}-source.wav",
        "rave_direct": f"wavs/{prefix}-rave-direct.wav",
        "seed17_short": f"wavs/{prefix}-seed17-short.wav",
        "seed29_short": f"wavs/{prefix}-seed29-short.wav",
        "seed17_long": f"wavs/{prefix}-seed17-long.wav",
        "seed29_long": f"wavs/{prefix}-seed29-long.wav",
    }


def audio_diagnostics(
    audio: Tensor,
    *,
    sample_rate: int,
) -> dict[str, int | float]:
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    waveform = audio.detach().float().reshape(-1).cpu()
    if waveform.numel() == 0:
        raise ValueError("audio is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("non-finite audio")
    return {
        "samples": waveform.numel(),
        "seconds": waveform.numel() / sample_rate,
        "peak": float(waveform.abs().max()),
        "rms": float(waveform.square().mean().sqrt()),
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_statistics(root: Path) -> FlowStatistics:
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
        return FlowStatistics(
            mean=torch.from_numpy(values["mean"].copy()),
            latent_std=torch.from_numpy(values["latent_std"].copy()),
            delta_std=torch.from_numpy(values["delta_std"].copy()),
            latent_norm_p01=torch.as_tensor(
                values["latent_norm_p01"].copy()
            ),
            latent_norm_p99=torch.as_tensor(
                values["latent_norm_p99"].copy()
            ),
        )


def _load_manifest_rows(
    path: Path,
    indices: set[int],
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in indices:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index + 1} is not an object")
            selected[index] = row
            if len(selected) == len(indices):
                break
    missing = sorted(indices - set(selected))
    if missing:
        raise ValueError(f"unified manifest lacks indices: {missing}")
    return selected


def _load_sequence_latent(
    packed_root: Path,
    row: dict[str, Any],
) -> Tensor:
    with np.load(
        packed_root / str(row["shard"]),
        allow_pickle=False,
    ) as values:
        latent = values["latents"][
            int(row["shard_row"]),
            : int(row["length"]),
        ].astype(np.float32)
    tensor = torch.from_numpy(latent)
    if tensor.ndim != 2 or tensor.shape[1] != 16:
        raise ValueError("packed audition latent must be [frames, 16]")
    if not torch.isfinite(tensor).all():
        raise ValueError("packed audition latent is non-finite")
    return tensor


def _load_codec(
    config: ZraveFlowConfig,
    device: torch.device,
) -> tuple[nn.Module, str]:
    path = Path(config.rave.checkpoint)
    digest = _sha256_file(path)
    if digest != config.rave.expected_sha256:
        raise ValueError("RAVE codec hash mismatch")
    codec = torch.jit.load(str(path), map_location=device).to(device).eval()
    latent_size = getattr(codec, "latent_size")
    if isinstance(latent_size, Tensor):
        latent_size = int(latent_size.flatten()[0])
    elif hasattr(latent_size, "__len__"):
        latent_size = int(latent_size[0])
    else:
        latent_size = int(latent_size)
    if latent_size != config.model.latent_dim:
        raise ValueError("RAVE codec latent size mismatch")
    return codec, digest


def _decode_latent(
    codec: nn.Module,
    latent: Tensor,
    *,
    random_seed: int,
    device: torch.device,
    latent_hop: int,
    decoder_preroll_frames: int,
) -> Tensor:
    if latent.ndim != 2 or latent.shape[1] != 16:
        raise ValueError("decode latent must be [frames, 16]")
    if latent_hop <= 0 or decoder_preroll_frames <= 0:
        raise ValueError("decoder pre-roll contract must be positive")
    layout = (
        latent.to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    preroll = (
        layout[:, :, :1]
        .expand(-1, -1, decoder_preroll_frames)
        .contiguous()
    )
    decoded = torch.from_numpy(
        decode_prerolled_latent(
            codec,
            preroll,
            layout,
            random_seed,
            latent_hop=latent_hop,
        )
    ).float()
    if not torch.isfinite(decoded).all():
        raise ValueError("RAVE decoder produced non-finite audio")
    return decoded.cpu()


def _write_wav(
    output: Path,
    relative_path: str,
    audio: Tensor,
    *,
    sample_rate: int,
) -> dict[str, object]:
    diagnostics = audio_diagnostics(audio, sample_rate=sample_rate)
    path = output / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        audio.detach().float().cpu().numpy(),
        sample_rate,
        subtype="FLOAT",
    )
    return {
        "wav": relative_path,
        "sha256": _sha256_file(path),
        **diagnostics,
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def render_pure_flow_audition(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, object]:
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("audition output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    config = ZraveFlowConfig.load(config_file)
    if config.model.pitch_conditioning:
        raise ValueError("pure audition config enables pitch conditioning")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    index_hash = _sha256_file(index_path)
    statistics_hash = _sha256_file(statistics_path)
    config_hash = _sha256_file(config_file)
    match = _CHECKPOINT_NAME.fullmatch(checkpoint_file.name)
    if match is None:
        raise ValueError("checkpoint must be named step-NNNNNN.pt")
    expected_update = int(match.group(1))
    payload = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    validate_pure_checkpoint(
        payload,
        config,
        index_hash=index_hash,
        statistics_hash=statistics_hash,
        expected_update=expected_update,
    )
    contract = payload["contract"]
    assert isinstance(contract, dict)
    if contract.get("config_sha256") != config_hash:
        raise ValueError("pure flow checkpoint config_sha256 mismatch")
    statistics = _load_statistics(packed_root)
    model = ZraveFlowTransformer(config.model, statistics).to(selected_device)
    model.load_state_dict(payload["model"])
    model.eval()
    codec, codec_hash = _load_codec(config, selected_device)

    cases = select_audition_cases(
        packed_root / "sequences.jsonl",
        seed=20260728,
    )
    manifest_indices = {int(row["manifest_index"]) for row in cases}
    source_rows = _load_manifest_rows(
        Path(config.data.unified_manifest),
        manifest_indices,
    )
    sequences = [
        _load_sequence_latent(packed_root, row)
        for row in cases
    ]
    history = torch.stack(
        [sequence[:32] for sequence in sequences],
        dim=0,
    ).to(selected_device)
    generated_by_seed: dict[int, Tensor] = {}
    for generation_seed in _GENERATION_SEEDS:
        generated = rollout_pure_flow(
            model,
            statistics,
            history,
            320,
            generation_seed=generation_seed,
            temperature=1.0,
            wander_delay_frames=32,
            solver_steps=8,
        )
        replay = rollout_pure_flow(
            model,
            statistics,
            history,
            64,
            generation_seed=generation_seed,
            temperature=1.0,
            wander_delay_frames=32,
            solver_steps=8,
        )
        if not torch.equal(replay, generated[:, :64]):
            raise ValueError("pure flow generation is not reproducible")
        if not torch.isfinite(generated).all():
            raise ValueError("pure flow generation is non-finite")
        generated_by_seed[generation_seed] = generated
    branch_rms = float(
        (
            generated_by_seed[17].float()
            - generated_by_seed[29].float()
        )
        .square()
        .mean()
        .sqrt()
    )
    if not math.isfinite(branch_rms) or branch_rms <= 0.0:
        raise ValueError("generation seeds did not branch")

    sample_rate = config.rave.sample_rate
    hop = config.rave.latent_hop
    decoder_seed = 20260728
    decoder_preroll_frames = 32
    file_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for case_index, (case, sequence) in enumerate(
        zip(cases, sequences, strict=True)
    ):
        category = str(case["category"])
        names = audition_file_names(case_index, category)
        source = source_rows[int(case["manifest_index"])]
        source_audio = _load_rave_audio(
            Path(str(source["audio_path"])),
            sample_rate=sample_rate,
            maximum_seconds=config.data.maximum_audio_seconds,
        )
        source_tensor = torch.from_numpy(
            source_audio[: 96 * hop].copy()
        )
        real_history = sequence[:32]
        real_future = sequence[32:96]
        decoded: dict[str, Tensor] = {
            "source": source_tensor,
            "rave_direct": _decode_latent(
                codec,
                torch.cat((real_history, real_future), dim=0),
                random_seed=decoder_seed,
                device=selected_device,
                latent_hop=hop,
                decoder_preroll_frames=decoder_preroll_frames,
            ),
        }
        for generation_seed in _GENERATION_SEEDS:
            generated = generated_by_seed[generation_seed][case_index].cpu()
            decoded[f"seed{generation_seed}_short"] = _decode_latent(
                codec,
                torch.cat((real_history, generated[:64]), dim=0),
                random_seed=decoder_seed,
                device=selected_device,
                latent_hop=hop,
                decoder_preroll_frames=decoder_preroll_frames,
            )
            decoded[f"seed{generation_seed}_long"] = _decode_latent(
                codec,
                torch.cat((real_history, generated), dim=0),
                random_seed=decoder_seed,
                device=selected_device,
                latent_hop=hop,
                decoder_preroll_frames=decoder_preroll_frames,
            )
        for role, relative_path in names.items():
            row = _write_wav(
                output,
                relative_path,
                decoded[role],
                sample_rate=sample_rate,
            )
            row.update(
                {
                    "case_index": case_index,
                    "category": category,
                    "role": role,
                    "generation_seed": (
                        int(role[4:6])
                        if role.startswith("seed")
                        else None
                    ),
                }
            )
            file_rows.append(row)
        case_rows.append(
            {
                "case_index": case_index,
                "category": category,
                "sample_id": case["sample_id"],
                "canonical_preset_id": case["canonical_preset_id"],
                "midi_note_metadata_only": int(case["midi_note"]),
                "velocity_metadata_only": int(case["velocity"]),
                "source_audio_path": source["audio_path"],
                "packed_index": int(case["packed_index"]),
                "latent_shard": case["shard"],
                "latent_shard_row": int(case["shard_row"]),
            }
        )
    if len(file_rows) != len(_DEFAULT_CATEGORIES) * 6:
        raise ValueError("audition did not produce six files per category")
    manifest: dict[str, object] = {
        "format": 1,
        "architecture": "zrave_pure_flow_transformer_v1",
        "checkpoint": str(checkpoint_file),
        "checkpoint_update": expected_update,
        "checkpoint_sha256": _sha256_file(checkpoint_file),
        "config_sha256": config_hash,
        "pack_index_sha256": index_hash,
        "statistics_sha256": statistics_hash,
        "codec_sha256": codec_hash,
        "sample_rate": sample_rate,
        "latent_dim": config.model.latent_dim,
        "latent_hop": hop,
        "history_frames": 32,
        "short_future_frames": 64,
        "long_future_frames": 320,
        "seed_boundary_seconds": 32 * hop / sample_rate,
        "decoder_preroll_frames": decoder_preroll_frames,
        "decoder_preroll_mode": "repeat_first_audible_frame",
        "temperature": 1.0,
        "wander_delay_frames": 32,
        "solver_steps": 8,
        "generation_seeds": list(_GENERATION_SEEDS),
        "latent_branch_rms": branch_rms,
        "cases": case_rows,
        "files": file_rows,
    }
    _atomic_json(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        "Pure Z-RAVE Flow audition, checkpoint step 45,000.\n"
        "No MIDI or CLAP is used by the model or renderer.\n"
        "Each category contains source, direct RAVE reconstruction, and "
        "two stochastic continuations.\n"
        f"The first 32 latent frames are real; generation begins at "
        f"{32 * hop / sample_rate:.6f} seconds.\n"
        "The decoder is warmed with 32 repeats of the first audible latent "
        "frame, then that pre-roll is cropped exactly.\n"
        "Short files contain 64 generated frames. Long files contain "
        "320 block-autoregressive generated frames.\n"
        "WAV files are unnormalized 32-bit float audio.\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a pure Z-RAVE Flow listening comparison."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    torch.use_deterministic_algorithms(True)
    manifest = render_pure_flow_audition(
        args.config,
        args.checkpoint,
        args.output,
        device=args.device,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
