from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor

from .zrave_codec import decode_with_seed
from .zrave_flow_config import ZraveFlowConfig
from .zrave_flow_model import (
    FlowStatistics,
    ZraveFlowTransformer,
    sample_pure_flow_block,
)


def rollout_pure_flow(
    model: Any,
    statistics: FlowStatistics | object,
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
    expected_history = (
        history.shape[0] if history.ndim else 0,
        int(model.context_frames),
        int(model.latent_dim),
    )
    if history.ndim != 3 or tuple(history.shape) != expected_history:
        raise ValueError(
            f"history must have shape {expected_history}, "
            f"got {tuple(history.shape)}"
        )
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
        expected_block = (
            history.shape[0],
            int(model.future_frames),
            int(model.latent_dim),
        )
        if tuple(generated.shape) != expected_block:
            raise ValueError(
                f"flow sampler must return {expected_block}, "
                f"got {tuple(generated.shape)}"
            )
        take = min(remaining, int(model.future_frames))
        chunk = generated[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[
            :, -int(model.context_frames) :
        ].detach()
        remaining -= take
        block_index += 1
    return torch.cat(chunks, dim=1)


def select_audition_rows(
    rows: Iterable[dict[str, Any]],
    *,
    categories: Sequence[str],
    split: str,
    seed: int,
    minimum_active_frames: int,
) -> list[dict[str, Any]]:
    if minimum_active_frames <= 0:
        raise ValueError("minimum_active_frames must be positive")
    available = list(rows)
    selected: list[dict[str, Any]] = []
    for category in categories:
        candidates = [
            row
            for row in available
            if str(row.get("category", "")).casefold()
            == category.casefold()
            and str(row.get("split", "")) == split
            and int(row.get("active_frames", 0))
            >= minimum_active_frames
        ]
        if not candidates:
            raise ValueError(
                f"no {split} {category} row has at least "
                f"{minimum_active_frames} active frames"
            )
        candidates.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:serum128-flow-audition:{category}:"
                    f"{row['sample_id']}"
                ).encode("utf-8")
            ).digest()
        )
        selected.append(candidates[0])
    return selected


def match_rms(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    peak_limit: float = 0.98,
) -> np.ndarray:
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if not np.isfinite(reference).all():
        raise ValueError("reference audio is not finite")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate audio is not finite")
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate))))
    if reference_rms <= 1.0e-12 or candidate_rms <= 1.0e-12:
        return np.zeros_like(candidate, dtype=np.float32)
    matched = candidate * np.float32(reference_rms / candidate_rms)
    peak = float(np.max(np.abs(matched)))
    if peak > peak_limit:
        matched *= np.float32(peak_limit / peak)
    return matched.astype(np.float32, copy=False)


def validate_pure_checkpoint_payload(
    payload: object,
    *,
    latent_dim: int,
    context_frames: int,
    future_frames: int,
    config_sha256: str,
    pack_index_sha256: str,
    statistics_sha256: str,
) -> int:
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("checkpoint must use format 1")
    if payload.get("architecture") != "zrave_pure_flow_transformer_v1":
        raise ValueError("checkpoint architecture is not pure flow")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no contract")
    expected: dict[str, object] = {
        "latent_dim": latent_dim,
        "context_frames": context_frames,
        "future_frames": future_frames,
        "pitch_conditioning": False,
        "config_sha256": config_sha256,
        "pack_index_sha256": pack_index_sha256,
        "statistics_sha256": statistics_sha256,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"checkpoint {name} contract mismatch")
    update = payload.get("update")
    if not isinstance(update, int) or isinstance(update, bool) or update <= 0:
        raise ValueError("checkpoint update must be positive")
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no model state")
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint model state is malformed")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor is non-finite: {name}")
    return update


def _audio_cell(title: str, payload: dict[str, Any]) -> str:
    path = escape(str(payload["matched_wav"]), quote=True)
    label = escape(title)
    return (
        '<div class="take">'
        f'<div class="take__label">{label}</div>'
        f'<audio controls preload="none" src="{path}"></audio>'
        "</div>"
    )


def render_index_html(manifest: dict[str, Any]) -> str:
    checkpoint = manifest["checkpoint"]
    cards: list[str] = []
    for example in manifest["examples"]:
        takes = [
            _audio_cell("Source", example["source"]),
            _audio_cell("RAVE Direct", example["direct"]),
        ]
        takes.extend(
            _audio_cell(str(rollout["label"]), rollout)
            for rollout in example["rollouts"]
        )
        cards.append(
            '<section class="instrument">'
            '<header class="instrument__head">'
            f'<h2>{escape(str(example["category"]))}</h2>'
            f'<code>{escape(str(example["sample_id"]))}</code>'
            "</header>"
            '<div class="takes">'
            + "".join(takes)
            + "</div></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Serum128 flow audition</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #edf1f5;
      --ink: #17243a;
      --muted: #657286;
      --rule: #bac5d1;
      --rave: #286aa6;
      --flow: #c56b17;
      --panel: #f8fafc;
      --focus: #704f91;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Segoe UI", sans-serif;
    }}
    main {{ width: min(1480px, calc(100% - 32px)); margin: 0 auto 64px; }}
    .masthead {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(420px, 1.45fr);
      gap: 36px;
      padding: 48px 0 32px;
      border-bottom: 2px solid var(--ink);
    }}
    .eyebrow, code, .take__label {{
      font-family: "Cascadia Mono", Consolas, monospace;
      letter-spacing: .04em;
    }}
    .eyebrow {{ color: var(--flow); font-weight: 700; text-transform: uppercase; }}
    h1, h2 {{ font-family: Bahnschrift, "Arial Narrow", sans-serif; margin: 0; }}
    h1 {{ font-size: clamp(42px, 7vw, 94px); line-height: .86; letter-spacing: -.045em; }}
    .facts {{ align-self: end; color: var(--muted); font-size: 15px; line-height: 1.55; }}
    .facts strong {{ color: var(--ink); }}
    .timeline {{ display: grid; grid-template-columns: 1fr repeat(5, 2fr); gap: 4px; margin-top: 24px; }}
    .timeline span {{ padding: 10px 8px; font: 12px "Cascadia Mono", monospace; text-align: center; }}
    .timeline__seed {{ background: var(--rave); color: white; }}
    .timeline__flow {{ background: var(--flow); color: white; }}
    .legend {{ display: flex; gap: 20px; margin-top: 10px; font: 12px "Cascadia Mono", monospace; }}
    .instrument {{ border-bottom: 1px solid var(--rule); padding: 28px 0 30px; }}
    .instrument__head {{ display: flex; align-items: baseline; gap: 18px; margin-bottom: 18px; }}
    .instrument__head h2 {{ font-size: 30px; }}
    .instrument__head code {{ color: var(--muted); font-size: 11px; }}
    .takes {{ display: grid; grid-template-columns: repeat(3, minmax(250px, 1fr)); gap: 10px; }}
    .take {{ background: var(--panel); border-left: 4px solid var(--rule); padding: 12px 14px 14px; }}
    .take:nth-child(2) {{ border-left-color: var(--rave); }}
    .take:nth-child(n+3) {{ border-left-color: var(--flow); }}
    .take__label {{ min-height: 32px; font-size: 12px; font-weight: 650; }}
    audio {{ width: 100%; height: 36px; }}
    audio:focus-visible {{ outline: 3px solid var(--focus); outline-offset: 3px; }}
    .note {{ margin-top: 18px; color: var(--muted); max-width: 80ch; }}
    @media (max-width: 900px) {{
      .masthead {{ grid-template-columns: 1fr; }}
      .takes {{ grid-template-columns: 1fr; }}
      .timeline span {{ font-size: 9px; padding-inline: 2px; }}
    }}
  </style>
</head>
<body>
<main>
  <header class="masthead">
    <div>
      <div class="eyebrow">Pure Z-RAVE / listening proof</div>
      <h1>Does the latent keep moving?</h1>
    </div>
    <div class="facts">
      <strong>Checkpoint {int(checkpoint['update']):,}</strong><br>
      128D Serum Balanced RAVE · 44.1 kHz · hop 2048<br>
      Compare the source, codec ceiling, then stochastic continuation.
      <div class="timeline">
        <span class="timeline__seed">32 real seed</span>
        <span class="timeline__flow">64</span><span class="timeline__flow">64</span>
        <span class="timeline__flow">64</span><span class="timeline__flow">64</span>
        <span class="timeline__flow">64</span>
      </div>
      <div class="legend"><span>32 real seed</span><span>5 × 64 generated blocks</span></div>
    </div>
  </header>
  <p class="note">Players use source-RMS-matched files for fair loudness. The package also contains untouched float WAVs under <code>raw/</code>.</p>
  {''.join(cards)}
</main>
</body>
</html>
"""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _load_statistics(path: str | Path) -> FlowStatistics:
    with np.load(path, allow_pickle=False) as values:
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


def _packed_latent(root: Path, row: dict[str, Any]) -> np.ndarray:
    with np.load(root / str(row["shard"]), allow_pickle=False) as values:
        latent = values["latents"][
            int(row["shard_row"]), : int(row["length"])
        ].astype(np.float32)
    if latent.ndim != 2 or not np.isfinite(latent).all():
        raise ValueError(f"invalid packed latent for {row['sample_id']}")
    return latent


def _mono_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    audio, actual_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"invalid source audio: {path}")
    if actual_rate != sample_rate:
        divisor = math.gcd(actual_rate, sample_rate)
        mono = resample_poly(
            mono,
            sample_rate // divisor,
            actual_rate // divisor,
        ).astype(np.float32)
        expected_samples = round(audio.shape[0] * sample_rate / actual_rate)
        mono = mono[:expected_samples]
    if not np.isfinite(mono).all():
        raise ValueError(f"non-finite resampled source audio: {path}")
    return np.ascontiguousarray(mono, dtype=np.float32)


def _audio_facts(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("rendered audio must be non-empty and finite")
    return {
        "samples": int(audio.size),
        "seconds": float(audio.size / sample_rate),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
    }


def _write_audio_pair(
    output: Path,
    stem: str,
    source_reference: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    raw_path = output / "raw" / f"{stem}.wav"
    matched_path = output / "matched" / f"{stem}.wav"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(audio, dtype=np.float32).reshape(-1)
    matched = match_rms(source_reference, raw)
    sf.write(raw_path, raw, sample_rate, subtype="FLOAT")
    sf.write(matched_path, matched, sample_rate, subtype="PCM_16")
    return {
        "raw_wav": raw_path.relative_to(output).as_posix(),
        "matched_wav": matched_path.relative_to(output).as_posix(),
        "raw": _audio_facts(raw, sample_rate),
        "matched": _audio_facts(matched, sample_rate),
    }


def _decode_latent(
    codec: Any,
    latent: np.ndarray | Tensor,
    *,
    device: torch.device,
    random_seed: int,
) -> np.ndarray:
    tensor = torch.as_tensor(
        latent,
        device=device,
        dtype=torch.float32,
    )
    if tensor.ndim != 2:
        raise ValueError("decode latent must be [frames, channels]")
    decoded = decode_with_seed(
        codec,
        tensor.transpose(0, 1).unsqueeze(0).contiguous(),
        random_seed,
    )
    audio = decoded.detach().float().cpu().reshape(-1).numpy()
    if not np.isfinite(audio).all():
        raise ValueError("RAVE decoder produced non-finite audio")
    return audio


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "sample"


def _codec_latent_size(codec: Any) -> int:
    value = getattr(codec, "latent_size")
    if isinstance(value, Tensor):
        return int(value.flatten()[0].item())
    if hasattr(value, "__len__"):
        return int(value[0])
    return int(value)


def render_flow_audition(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    categories: Sequence[str] = (
        "Pad",
        "Lead",
        "Bass",
        "Pluck",
        "Keys",
        "Synth",
    ),
    temperatures: Sequence[float] = (0.7, 1.0),
    generation_seeds: Sequence[int] = (17, 71),
    generated_frames: int = 320,
    selection_seed: int = 20260802,
    device_name: str = "cuda",
) -> dict[str, Any]:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    output = Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audition output is not empty: {output}")
    if generated_frames <= 0:
        raise ValueError("generated_frames must be positive")
    if not temperatures or any(value < 0.0 for value in temperatures):
        raise ValueError("temperatures must be non-empty and non-negative")
    if not generation_seeds or any(value < 0 for value in generation_seeds):
        raise ValueError("generation seeds must be non-empty and non-negative")

    config = ZraveFlowConfig.load(config_path)
    if config.model.pitch_conditioning:
        raise ValueError("audition requires a pure flow config")
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    update = validate_pure_checkpoint_payload(
        payload,
        latent_dim=config.model.latent_dim,
        context_frames=config.model.context_frames,
        future_frames=config.model.future_frames,
        config_sha256=_sha256_file(config_path),
        pack_index_sha256=_sha256_file(index_path),
        statistics_sha256=_sha256_file(statistics_path),
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    statistics = _load_statistics(statistics_path)
    statistics.validate(config.model.latent_dim)
    device = torch.device(device_name)
    model = ZraveFlowTransformer(config.model, statistics).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    del payload

    codec_path = Path(config.rave.checkpoint)
    codec_hash = _sha256_file(codec_path)
    if codec_hash != config.rave.expected_sha256:
        raise ValueError("RAVE codec hash mismatch")
    codec = torch.jit.load(str(codec_path), map_location=device).eval()
    if _codec_latent_size(codec) != config.model.latent_dim:
        raise ValueError("RAVE codec latent dimension mismatch")

    sequence_rows = _read_jsonl(packed_root / "sequences.jsonl")
    selected = select_audition_rows(
        sequence_rows,
        categories=categories,
        split="test",
        seed=selection_seed,
        minimum_active_frames=config.model.context_frames,
    )
    manifest_rows = {
        str(row["sample_id"]): row
        for row in _read_jsonl(config.data.unified_manifest)
    }
    latents: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, row in enumerate(selected):
            sample_id = str(row["sample_id"])
            source_row = manifest_rows.get(sample_id)
            if source_row is None:
                raise ValueError(f"manifest lacks {sample_id}")
            latent = _packed_latent(packed_root, row)
            if latent.shape[1] != config.model.latent_dim:
                raise ValueError(f"latent dimension mismatch for {sample_id}")
            source = _mono_audio(
                str(source_row["audio_path"]),
                config.rave.sample_rate,
            )
            direct = _decode_latent(
                codec,
                latent,
                device=device,
                random_seed=config.seed + index,
            )[: source.size]
            stem = f"{index:02d}-{_slug(str(row['category']))}"
            examples.append(
                {
                    "category": str(row["category"]),
                    "sample_id": sample_id,
                    "canonical_preset_id": str(
                        row["canonical_preset_id"]
                    ),
                    "midi_note": int(row["midi_note"]),
                    "velocity": int(row["velocity"]),
                    "source": _write_audio_pair(
                        output,
                        f"{stem}-source",
                        source,
                        source,
                        config.rave.sample_rate,
                    ),
                    "direct": _write_audio_pair(
                        output,
                        f"{stem}-rave-direct",
                        source,
                        direct,
                        config.rave.sample_rate,
                    ),
                    "rollouts": [],
                }
            )
            latents.append(latent)
            sources.append(source)

        history = torch.from_numpy(
            np.stack(
                [latent[: config.model.context_frames] for latent in latents]
            )
        ).to(device)
        for temperature in temperatures:
            for generation_seed in generation_seeds:
                generated = rollout_pure_flow(
                    model,
                    statistics,
                    history,
                    generated_frames,
                    generation_seed=generation_seed,
                    temperature=float(temperature),
                    wander_delay_frames=32,
                    solver_steps=config.model.solver_steps,
                )
                complete = torch.cat((history, generated), dim=1)
                for index, example in enumerate(examples):
                    audio = _decode_latent(
                        codec,
                        complete[index],
                        device=device,
                        random_seed=(
                            config.seed + generation_seed * 100 + index
                        ),
                    )
                    temperature_name = str(float(temperature)).replace(
                        ".", "p"
                    )
                    stem = (
                        f"{index:02d}-{_slug(str(example['category']))}"
                        f"-flow-t{temperature_name}-s{generation_seed}"
                    )
                    rendered = _write_audio_pair(
                        output,
                        stem,
                        sources[index],
                        audio,
                        config.rave.sample_rate,
                    )
                    rendered.update(
                        {
                            "label": (
                                f"Flow · T{float(temperature):g} · "
                                f"Seed {generation_seed}"
                            ),
                            "temperature": float(temperature),
                            "generation_seed": int(generation_seed),
                        }
                    )
                    example["rollouts"].append(rendered)

    manifest: dict[str, Any] = {
        "schema": 1,
        "title": "Serum128 pure-flow audition",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "update": update,
            "sha256": checkpoint_hash,
        },
        "codec": {
            "path": str(codec_path.resolve()),
            "sha256": codec_hash,
            "latent_dim": config.model.latent_dim,
        },
        "sample_rate": config.rave.sample_rate,
        "latent_hop": config.rave.latent_hop,
        "context_frames": config.model.context_frames,
        "generated_frames": generated_frames,
        "solver_steps": config.model.solver_steps,
        "wander_delay_frames": 32,
        "temperatures": [float(value) for value in temperatures],
        "generation_seeds": [int(value) for value in generation_seeds],
        "selection_seed": selection_seed,
        "examples": examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(
        render_index_html(manifest),
        encoding="utf-8",
    )
    return manifest
