from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from .data import load_audio
from .zrave_config import ZraveConfig
from .zrave_data import read_jsonl
from .zrave_codec import decode_with_seed, encode_posterior_mean
from .zrave_model import ZraveStatistics, ZraveTransformer


_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")
_CODEC_KIND = "standalone_torchscript_rave"


def _rows_by_sample_id(
    rows: Iterable[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("audition manifest row is missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate audition sample_id: {sample_id}")
        result[sample_id] = row
    return result


def select_audition_rows(
    index: dict[str, object],
    manifest_rows: Iterable[dict[str, object]],
    categories: Iterable[str],
) -> list[dict[str, object]]:
    sequences = index.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("packed index must contain a sequences list")
    manifest_by_id = _rows_by_sample_id(manifest_rows)
    category_order = tuple(categories)
    if not category_order:
        raise ValueError("audition categories must not be empty")

    groups: list[dict[tuple[int, int], list[dict[str, object]]]] = []
    labels: list[tuple[str, str]] = []
    for category in category_order:
        for split in ("validation", "test"):
            matching = [
                sequence
                for sequence in sequences
                if isinstance(sequence, dict)
                and sequence.get("category") == category
                and sequence.get("split") == split
            ]
            if not matching:
                raise ValueError(
                    f"missing held-out audition data for {category}/{split}"
                )
            preset_id = min(str(row["preset_id"]) for row in matching)
            conditions: dict[
                tuple[int, int],
                list[dict[str, object]],
            ] = {}
            for sequence in matching:
                if str(sequence["preset_id"]) != preset_id:
                    continue
                sample_id = str(sequence["sample_id"])
                manifest_row = manifest_by_id.get(sample_id)
                if manifest_row is None:
                    raise ValueError(
                        f"packed sample is absent from manifest: {sample_id}"
                    )
                condition = (
                    int(manifest_row["midi_note"]),
                    int(manifest_row["velocity"]),
                )
                joined = dict(manifest_row)
                joined.update(
                    {
                        "category": category,
                        "split": split,
                        "preset_id": preset_id,
                        "sequence_index": int(sequence["index"]),
                        "length": int(sequence["length"]),
                    }
                )
                conditions.setdefault(condition, []).append(joined)
            if not conditions:
                raise ValueError(
                    f"preset has no audition conditions: {category}/{split}"
                )
            groups.append(conditions)
            labels.append((category, split))

    common = set(groups[0])
    for group in groups[1:]:
        common.intersection_update(group)
    if not common:
        raise ValueError("held-out presets have no shared note condition")
    chosen_condition = min(
        common,
        key=lambda value: (
            -value[1],
            abs(value[0] - 60),
            value[0],
        ),
    )

    selected: list[dict[str, object]] = []
    for (category, split), group in zip(labels, groups, strict=True):
        candidates = sorted(
            group[chosen_condition],
            key=lambda row: str(row["sample_id"]),
        )
        selected.append(dict(candidates[0]))
    return selected


def deterministic_start(
    sample_id: str,
    length: int,
    required_frames: int,
    seed: int,
) -> int:
    if required_frames <= 0:
        raise ValueError("required frames must be positive")
    if length < required_frames:
        raise ValueError(
            f"latent sequence is shorter than {required_frames}: "
            f"{sample_id} has {length}"
        )
    payload = f"{seed}:zrave-audition:{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (length - required_frames + 1)


def _mono_float32(value: np.ndarray, name: str) -> np.ndarray:
    clip = np.asarray(value, dtype=np.float32)
    if clip.ndim != 1 or clip.size == 0:
        raise ValueError(f"{name} audio must be a non-empty mono array")
    if not np.isfinite(clip).all():
        raise ValueError(f"{name} audio contains non-finite values")
    return np.ascontiguousarray(clip)


def shared_peak_gain(
    *clips: np.ndarray,
    target_peak: float = 0.95,
) -> float:
    if not clips:
        raise ValueError("at least one audio clip is required")
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target peak must be in (0, 1]")
    peak = max(float(np.max(np.abs(clip))) for clip in clips)
    if peak < 1.0e-8:
        raise ValueError("audition audio is silent")
    return target_peak / peak


def prepare_triplet(
    source: np.ndarray,
    direct: np.ndarray,
    predicted: np.ndarray,
    target_peak: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    clips = (
        _mono_float32(source, "source"),
        _mono_float32(direct, "direct"),
        _mono_float32(predicted, "predicted"),
    )
    if len({clip.shape[0] for clip in clips}) != 1:
        raise ValueError("audition clips must have identical sample counts")
    gain = shared_peak_gain(*clips, target_peak=target_peak)
    rendered = tuple(np.ascontiguousarray(clip * gain) for clip in clips)
    if any(float(np.max(np.abs(clip))) > 1.0 for clip in rendered):
        raise ValueError("normalized audition audio exceeds [-1, 1]")
    return rendered[0], rendered[1], rendered[2], gain


def rollout_latents(
    model: Any,
    history: Tensor,
    frames: int,
) -> Tensor:
    if frames <= 0:
        raise ValueError("rollout frames must be positive")
    context_frames = int(model.config.context_frames)
    if history.ndim != 3 or history.shape[1] != context_frames:
        raise ValueError(
            f"history must have {context_frames} latent frames"
        )
    chunks: list[Tensor] = []
    current = history
    remaining = frames
    while remaining:
        prediction = model(current).latent
        take = min(remaining, prediction.shape[1])
        chunk = prediction[:, :take]
        chunks.append(chunk)
        current = torch.cat((current, chunk), dim=1)[:, -context_frames:]
        remaining -= take
    return torch.cat(chunks, dim=1)


def crop_decoded_future(
    decoded: np.ndarray,
    *,
    context_frames: int,
    future_frames: int,
    latent_hop: int,
) -> np.ndarray:
    clip = _mono_float32(decoded, "decoded")
    if min(context_frames, future_frames, latent_hop) <= 0:
        raise ValueError("crop frame counts and latent hop must be positive")
    start = context_frames * latent_hop
    end = start + future_frames * latent_hop
    if clip.shape[0] < end:
        raise ValueError(
            f"decoded audio is shorter than required crop: "
            f"{clip.shape[0]} < {end}"
        )
    return np.ascontiguousarray(clip[start:end])


def crop_source_future(
    source: np.ndarray,
    *,
    warmup_frames: int,
    window_start: int,
    context_frames: int,
    future_frames: int,
    latent_hop: int,
) -> np.ndarray:
    clip = _mono_float32(source, "source")
    if warmup_frames < 0 or window_start < 0:
        raise ValueError("source crop offsets must be non-negative")
    start = (
        warmup_frames + window_start + context_frames
    ) * latent_hop
    end = start + future_frames * latent_hop
    if clip.shape[0] < end:
        raise ValueError(
            f"source audio is shorter than required crop: "
            f"{clip.shape[0]} < {end}"
        )
    return np.ascontiguousarray(clip[start:end])


def duration_to_frames(
    duration_seconds: float,
    sample_rate: int,
    latent_hop: int,
) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if sample_rate <= 0 or latent_hop <= 0:
        raise ValueError("sample rate and latent hop must be positive")
    return int(math.ceil(duration_seconds * sample_rate / latent_hop))


def select_random_seed_rows(
    index: dict[str, object],
    manifest_rows: Iterable[dict[str, object]],
    *,
    count: int,
    required_frames: int,
    seed: int,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("random seed row count must be positive")
    sequences = index.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("packed index must contain a sequences list")
    manifest_by_id = _rows_by_sample_id(manifest_rows)
    candidates: list[dict[str, object]] = []
    for sequence in sequences:
        if (
            not isinstance(sequence, dict)
            or sequence.get("split") not in {"validation", "test"}
            or int(sequence.get("length") or 0) < required_frames
        ):
            continue
        sample_id = str(sequence.get("sample_id") or "")
        source = manifest_by_id.get(sample_id)
        if source is None:
            raise ValueError(
                f"packed sample is absent from manifest: {sample_id}"
            )
        joined = dict(source)
        joined.update(
            {
                "category": sequence.get("category"),
                "split": sequence.get("split"),
                "preset_id": sequence.get("preset_id"),
                "sequence_index": int(sequence["index"]),
                "length": int(sequence["length"]),
            }
        )
        candidates.append(joined)
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:random-zrave-head:{row['sample_id']}".encode("utf-8")
        ).digest()
    )
    if len(candidates) < count:
        raise ValueError(
            f"need {count} held-out random seed rows, found {len(candidates)}"
        )
    return candidates[:count]


def validate_audition_manifest(
    manifest: dict[str, object],
    categories: Iterable[str],
) -> None:
    if manifest.get("schema") != 1:
        raise ValueError("unsupported audition manifest schema")
    if manifest.get("conditioning") != []:
        raise ValueError("audition conditioning must be empty")
    codec = manifest.get("codec")
    if (
        not isinstance(codec, dict)
        or codec.get("kind") != _CODEC_KIND
    ):
        raise ValueError("audition must use one standalone RAVE codec")
    sample_rate = int(manifest.get("sample_rate") or 0)
    latent_hop = int(manifest.get("latent_hop") or 0)
    context_frames = int(manifest.get("context_frames") or 0)
    if min(sample_rate, latent_hop, context_frames) <= 0:
        raise ValueError("audition timing contract is invalid")

    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("audition comparisons must be a list")
    expected = Counter(
        (category, split)
        for category in categories
        for split in ("validation", "test")
    )
    actual: Counter[tuple[object, object]] = Counter()
    for row in comparisons:
        if not isinstance(row, dict):
            raise ValueError("audition comparison must be an object")
        actual[(row.get("category"), row.get("split"))] += 1
        audio = row.get("audio")
        if not isinstance(audio, dict) or set(audio) != {
            "original",
            "direct",
            "predicted",
        }:
            raise ValueError("comparison must contain three audio variants")
        if not all(isinstance(path, str) and path for path in audio.values()):
            raise ValueError("comparison audio paths must not be empty")
    if actual != expected:
        raise ValueError(
            "manifest must contain validation and test rows for every class"
        )

    random_rows = manifest.get("random_continuations")
    if not isinstance(random_rows, list) or len(random_rows) != 2:
        raise ValueError("manifest must contain exactly two random continuations")
    for row in random_rows:
        if not isinstance(row, dict) or not str(row.get("file") or ""):
            raise ValueError("random continuation file is required")
        if int(row.get("seed_frames") or 0) != context_frames:
            raise ValueError("random continuation seed length mismatch")
        transition = float(row.get("transition_seconds") or 0.0)
        if not math.isfinite(transition) or transition <= 0.0:
            raise ValueError("random continuation transition is invalid")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_statistics(root: Path) -> ZraveStatistics:
    with np.load(root / "statistics.npz", allow_pickle=False) as values:
        return ZraveStatistics(
            mean=torch.from_numpy(values["mean"].copy()),
            latent_std=torch.from_numpy(values["latent_std"].copy()),
            delta_std=torch.from_numpy(values["delta_std"].copy()),
            acceleration_std=torch.from_numpy(
                values["acceleration_std"].copy()
            ),
        )


def _load_transformer(
    config: ZraveConfig,
    checkpoint_path: Path,
    index_path: Path,
    statistics_path: Path,
    device: torch.device,
) -> tuple[ZraveTransformer, dict[str, object]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("audition requires a format-1 Z-RAVE checkpoint")
    if payload.get("architecture") != "zrave_transformer_v1":
        raise ValueError("audition Z-RAVE architecture mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("audition Z-RAVE checkpoint has no contract")
    expected = {
        "packed_index_sha256": _sha256_file(index_path),
        "statistics_sha256": _sha256_file(statistics_path),
        "latent_dim": config.model.latent_dim,
        "context_frames": config.model.context_frames,
        "horizon_frames": config.model.horizon_frames,
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"audition Z-RAVE checkpoint {name} mismatch")
    model = ZraveTransformer(
        config.model,
        _load_statistics(index_path.parent),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


def _load_codec(
    config: ZraveConfig,
    device: torch.device,
) -> Any:
    checkpoint = Path(config.rave.checkpoint)
    actual_hash = _sha256_file(checkpoint)
    expected_hash = config.rave.expected_sha256
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            f"standalone RAVE SHA-256 mismatch: "
            f"{actual_hash} != {expected_hash}"
        )
    codec = torch.jit.load(
        str(checkpoint),
        map_location=device,
    ).to(device).eval()
    sample_rate_value = codec.sr
    if hasattr(sample_rate_value, "__len__"):
        sample_rate_value = sample_rate_value[0]
    if int(sample_rate_value) != config.rave.sample_rate:
        raise ValueError("standalone RAVE sample rate mismatch")
    if int(codec.latent_size) != config.model.latent_dim:
        raise ValueError("standalone RAVE latent dimension mismatch")

    probe_frames = 4
    probe_audio = torch.zeros(
        1,
        1,
        probe_frames * config.data.latent_hop,
        device=device,
    )
    probe_latent = encode_posterior_mean(codec, probe_audio)
    probe_decoded = decode_with_seed(codec, probe_latent, config.seed)
    if tuple(probe_latent.shape) != (
        1,
        config.model.latent_dim,
        probe_frames,
    ):
        raise ValueError("standalone RAVE encode layout/hop mismatch")
    if tuple(probe_decoded.shape) != (
        1,
        1,
        probe_frames * config.data.latent_hop,
    ):
        raise ValueError("standalone RAVE decode layout/hop mismatch")
    if not torch.isfinite(probe_latent).all() or not torch.isfinite(
        probe_decoded
    ).all():
        raise ValueError("standalone RAVE codec probe is non-finite")
    return codec


def _decode_latent(
    codec: Any,
    latent: Tensor,
    random_seed: int,
) -> np.ndarray:
    decoded = (
        decode_with_seed(codec, latent, random_seed)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    if decoded.ndim != 3 or decoded.shape[:2] != (1, 1):
        raise ValueError(
            f"standalone RAVE returned invalid audio shape: "
            f"{tuple(decoded.shape)}"
        )
    return _mono_float32(decoded[0, 0], "decoded")


def _audio_path(
    config: ZraveConfig,
    row: dict[str, object],
) -> Path:
    return (
        Path(config.data.audio_root) / str(row["audio_path"])
    ).resolve()


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        subtype="PCM_16",
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def render_audition(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    duration_seconds: float = 2.5,
    random_seed: int = 20260724,
    device_name: str = "cuda",
) -> dict[str, object]:
    config = ZraveConfig.load(config_path)
    device = torch.device(device_name)
    packed_root = Path(config.data.packed_root)
    index_path = packed_root / "index.json"
    statistics_path = packed_root / "statistics.npz"
    latents_path = packed_root / "latents.npy"
    checkpoint = Path(checkpoint_path)
    codec_path = Path(config.rave.checkpoint)
    for required in (
        index_path,
        statistics_path,
        latents_path,
        checkpoint,
        codec_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing audition artifact: {required}")

    destination = Path(output_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"audition output directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audio").mkdir(parents=True, exist_ok=True)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != 1:
        raise ValueError("unsupported packed index schema")
    codec_hash = _sha256_file(codec_path)
    if index.get("rave_codec") != _CODEC_KIND:
        raise ValueError("packed latents did not come from a standalone codec")
    if index.get("rave_latent_encoding") != (
        "posterior_mean_temp0_reset_v1"
    ):
        raise ValueError("packed latents are not deterministic posterior means")
    if index.get("streaming_state_reset_per_clip") is not True:
        raise ValueError("packed latents contain cross-clip streaming state")
    if index.get("rave_checkpoint_sha256") != codec_hash:
        raise ValueError("packed latents and standalone codec do not match")
    if int(index.get("latent_hop") or 0) != config.data.latent_hop:
        raise ValueError("packed latent hop does not match config")

    manifest_rows = read_jsonl(config.data.selected_manifest)
    selected = select_audition_rows(index, manifest_rows, _CATEGORIES)
    sample_rate = config.rave.sample_rate
    latent_hop = config.data.latent_hop
    future_frames = duration_to_frames(
        duration_seconds,
        sample_rate,
        latent_hop,
    )
    context_frames = config.model.context_frames
    required_frames = context_frames + future_frames
    random_rows = select_random_seed_rows(
        index,
        manifest_rows,
        count=2,
        required_frames=required_frames,
        seed=random_seed,
    )

    transformer, transformer_payload = _load_transformer(
        config,
        checkpoint,
        index_path,
        statistics_path,
        device,
    )
    codec = _load_codec(config, device)
    packed = np.load(latents_path, mmap_mode="r", allow_pickle=False)

    audio_by_id: dict[str, np.ndarray] = {}
    for row in [*selected, *random_rows]:
        sample_id = str(row["sample_id"])
        if sample_id in audio_by_id:
            continue
        audio_by_id[sample_id] = load_audio(
            _audio_path(config, row),
            sample_rate,
        )

    comparisons: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, 1):
        sample_id = str(row["sample_id"])
        start = deterministic_start(
            sample_id,
            int(row["length"]),
            required_frames,
            random_seed,
        )
        sequence = torch.from_numpy(
            np.asarray(
                packed[
                    int(row["sequence_index"]),
                    : int(row["length"]),
                ],
                dtype=np.float32,
            ).copy()
        ).to(device).unsqueeze(0)
        future_start = start + context_frames
        future_end = future_start + future_frames
        history = sequence[:, start:future_start]
        predicted_future = rollout_latents(
            transformer,
            history,
            future_frames,
        )
        direct_latent = sequence[:, :future_end].transpose(1, 2)
        predicted_latent = torch.cat(
            (
                sequence[:, :future_start],
                predicted_future,
            ),
            dim=1,
        ).transpose(1, 2)
        codec_seed = random_seed + row_index
        direct_full = _decode_latent(
            codec,
            direct_latent,
            codec_seed,
        )
        predicted_full = _decode_latent(
            codec,
            predicted_latent,
            codec_seed,
        )
        decoded_sample_start = future_start * latent_hop
        decoded_sample_end = future_end * latent_hop
        source_sample_start = (
            config.data.warmup_frames + future_start
        ) * latent_hop
        source_sample_end = source_sample_start + future_frames * latent_hop
        direct = direct_full[
            decoded_sample_start:decoded_sample_end
        ]
        predicted = predicted_full[
            decoded_sample_start:decoded_sample_end
        ]
        original = audio_by_id[sample_id][
            source_sample_start:source_sample_end
        ]
        original, direct, predicted, gain = prepare_triplet(
            original,
            direct,
            predicted,
        )

        stem = (
            f"{row_index:02d}-"
            f"{str(row['category']).casefold()}-{row['split']}"
        )
        files = {
            "original": f"audio/{stem}-original.wav",
            "direct": f"audio/{stem}-direct.wav",
            "predicted": f"audio/{stem}-predicted.wav",
        }
        for name, clip in (
            ("original", original),
            ("direct", direct),
            ("predicted", predicted),
        ):
            _write_wav(destination / files[name], clip, sample_rate)
        comparisons.append(
            {
                "id": stem,
                "category": row["category"],
                "split": row["split"],
                "preset_id": row["preset_id"],
                "sample_id": sample_id,
                "sequence_index": int(row["sequence_index"]),
                "window_start": start,
                "source_sample_start": source_sample_start,
                "sample_count": int(original.shape[0]),
                "shared_gain": gain,
                "codec_random_seed": codec_seed,
                "audio": files,
            }
        )

    continuations: list[dict[str, object]] = []
    for continuation_index, row in enumerate(random_rows, 1):
        sample_id = str(row["sample_id"])
        start = deterministic_start(
            sample_id,
            int(row["length"]),
            required_frames,
            random_seed + 1000 + continuation_index,
        )
        sequence = torch.from_numpy(
            np.asarray(
                packed[
                    int(row["sequence_index"]),
                    : int(row["length"]),
                ],
                dtype=np.float32,
            ).copy()
        ).to(device).unsqueeze(0)
        future_start = start + context_frames
        history = sequence[:, start:future_start]
        prediction = rollout_latents(
            transformer,
            history,
            future_frames,
        )
        latent = torch.cat(
            (history, prediction),
            dim=1,
        ).transpose(1, 2)
        codec_seed = random_seed + 100 + continuation_index
        decoded = _decode_latent(codec, latent, codec_seed)
        expected_samples = required_frames * latent_hop
        if decoded.shape[0] < expected_samples:
            raise ValueError("random continuation decode is too short")
        decoded = decoded[:expected_samples]
        gain = shared_peak_gain(decoded)
        decoded = np.ascontiguousarray(decoded * gain)
        stem = (
            f"random-{continuation_index:02d}-"
            f"{str(row['category']).casefold()}-{row['split']}"
        )
        relative_path = f"audio/{stem}.wav"
        _write_wav(destination / relative_path, decoded, sample_rate)
        continuations.append(
            {
                "id": stem,
                "category": row["category"],
                "split": row["split"],
                "preset_id": row["preset_id"],
                "sample_id": sample_id,
                "sequence_index": int(row["sequence_index"]),
                "window_start": start,
                "seed_frames": context_frames,
                "predicted_frames": future_frames,
                "transition_seconds": (
                    context_frames * latent_hop / sample_rate
                ),
                "sample_count": int(decoded.shape[0]),
                "gain": gain,
                "codec_random_seed": codec_seed,
                "file": relative_path,
            }
        )

    manifest: dict[str, object] = {
        "schema": 1,
        "title": "Standalone RAVE Sequence Transformer Audition",
        "categories": list(_CATEGORIES),
        "sample_rate": sample_rate,
        "latent_hop": latent_hop,
        "latent_dim": config.model.latent_dim,
        "context_frames": context_frames,
        "horizon_frames": config.model.horizon_frames,
        "future_frames": future_frames,
        "audible_duration_seconds": (
            future_frames * latent_hop / sample_rate
        ),
        "random_seed": random_seed,
        "codec_random_seed_policy": (
            "paired direct/predicted decodes use identical seeds"
        ),
        "conditioning": [],
        "codec": {
            "kind": _CODEC_KIND,
            "path": str(codec_path.resolve()),
            "sha256": codec_hash,
            "sample_rate": sample_rate,
            "latent_dim": config.model.latent_dim,
            "latent_hop": latent_hop,
        },
        "transformer": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256_file(checkpoint),
            "update": int(transformer_payload["update"]),
            "architecture": transformer_payload["architecture"],
        },
        "packed_index_sha256": _sha256_file(index_path),
        "comparisons": comparisons,
        "random_continuations": continuations,
    }
    validate_audition_manifest(manifest, _CATEGORIES)
    _atomic_json(destination / "audition-manifest.json", manifest)
    return manifest
