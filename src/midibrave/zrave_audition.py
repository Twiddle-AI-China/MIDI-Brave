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
from scipy.signal import resample_poly
from torch import Tensor

from .config import Config
from .data import load_audio
from .predictive_model import PredictiveMidiBrave
from .zrave_config import ZraveConfig
from .zrave_data import read_jsonl
from .zrave_model import ZraveStatistics, ZraveTransformer


_CATEGORIES = ("Pad", "Bass", "Lead", "Pluck", "Keys")


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
                tuple[int, int], list[dict[str, object]]
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
        raise ValueError("held-out presets have no shared MIDI condition")
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
        row = dict(candidates[0])
        row["category"] = category
        row["split"] = split
        selected.append(row)
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
    if min(context_frames, future_frames, latent_hop) <= 0:
        raise ValueError("source crop dimensions must be positive")
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
    if duration_seconds <= 0.0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0 or latent_hop <= 0:
        raise ValueError("sample rate and latent hop must be positive")
    return math.ceil(duration_seconds * sample_rate / latent_hop)


def select_random_seed_rows(
    index: dict[str, object],
    manifest_rows: Iterable[dict[str, object]],
    *,
    count: int,
    required_frames: int,
    seed: int,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("random seed count must be positive")
    sequences = index.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("packed index must contain a sequences list")
    manifest_by_id = _rows_by_sample_id(manifest_rows)
    eligible: list[dict[str, object]] = []
    for sequence in sequences:
        if not isinstance(sequence, dict):
            continue
        if sequence.get("split") not in {"validation", "test"}:
            continue
        if int(sequence.get("length") or 0) < required_frames:
            continue
        sample_id = str(sequence.get("sample_id") or "")
        manifest_row = manifest_by_id.get(sample_id)
        if manifest_row is None:
            raise ValueError(
                f"packed sample is absent from manifest: {sample_id}"
            )
        row = dict(manifest_row)
        row.update(
            {
                "category": str(sequence.get("category") or ""),
                "split": str(sequence["split"]),
                "preset_id": str(sequence["preset_id"]),
                "sequence_index": int(sequence["index"]),
                "length": int(sequence["length"]),
            }
        )
        eligible.append(row)
    eligible.sort(key=lambda row: str(row["sample_id"]))
    if len(eligible) < count:
        raise ValueError(
            f"need {count} eligible random seed rows, found {len(eligible)}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eligible), size=count, replace=False)
    return [dict(eligible[int(index)]) for index in indices]


def validate_audition_manifest(
    manifest: dict[str, object],
    categories: Iterable[str],
) -> None:
    if manifest.get("schema") != 1:
        raise ValueError("unsupported audition manifest schema")
    sample_rate = int(manifest.get("sample_rate") or 0)
    context_frames = int(manifest.get("context_frames") or 0)
    if sample_rate <= 0 or context_frames <= 0:
        raise ValueError("manifest timing metadata is invalid")
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("manifest comparisons must be a list")
    expected = Counter(
        (category, split)
        for category in categories
        for split in ("validation", "test")
    )
    actual: Counter[tuple[str, str]] = Counter()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("manifest comparison must be an object")
        actual[
            (
                str(comparison.get("category") or ""),
                str(comparison.get("split") or ""),
            )
        ] += 1
        if int(comparison.get("sample_count") or 0) <= 0:
            raise ValueError("comparison sample count must be positive")
        audio = comparison.get("audio")
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


def _load_decoder(
    config: ZraveConfig,
    device: torch.device,
) -> tuple[PredictiveMidiBrave, Config, dict[str, object]]:
    source_config = Config.load(config.rave.source_config)
    if source_config.predictive is None:
        raise ValueError("audition decoder config must be predictive")
    if source_config.data.sample_rate <= 0:
        raise ValueError("audition decoder sample rate is invalid")
    payload = torch.load(
        config.rave.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or int(payload.get("format", 0)) != 5:
        raise ValueError("audition decoder requires a format-5 checkpoint")
    contract = payload.get("predictive_contract")
    if not isinstance(contract, dict) or contract.get("stage") not in {
        "predictor",
        "rollout",
        "gan",
    }:
        raise ValueError("audition decoder checkpoint stage is unsupported")
    model = PredictiveMidiBrave(
        source_config.model,
        source_config.predictive,
        source_config.data.window_samples,
        source_config.data.sample_rate,
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, source_config, payload


def _load_clap_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Any:
    import laion_clap

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing CLAP checkpoint: {checkpoint}")
    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel="HTSAT-base",
        device=str(device),
    )
    model.load_ckpt(str(checkpoint))
    model.eval()
    return model


def _clap_embedding(
    model: Any,
    audio: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    if sample_rate != 48000:
        common = math.gcd(sample_rate, 48000)
        audio = resample_poly(
            audio,
            48000 // common,
            sample_rate // common,
        ).astype(np.float32)
    tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
    embedding = model.get_audio_embedding_from_data(
        tensor,
        use_tensor=True,
    )
    embedding = torch.nn.functional.normalize(
        embedding.float(),
        dim=-1,
    )
    result = embedding[0].detach().cpu().numpy().astype(np.float32)
    if result.shape != (512,) or not np.isfinite(result).all():
        raise ValueError("CLAP embedding must be finite and 512-dimensional")
    return result


def _audio_path(config: ZraveConfig, row: dict[str, object]) -> Path:
    relative = Path(str(row["audio_path"]))
    path = (
        relative
        if relative.is_absolute()
        else Path(config.data.audio_root) / relative
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing source audio: {path}")
    return path


def _decode_latents(
    decoder: PredictiveMidiBrave,
    latent: Tensor,
    clap: np.ndarray,
    note: int,
    velocity: int,
    excitation_seed: int,
    device: torch.device,
) -> np.ndarray:
    clap_tensor = torch.from_numpy(clap).unsqueeze(0).to(device)
    note_tensor = torch.tensor([note], device=device, dtype=torch.long)
    velocity_tensor = torch.tensor(
        [float(velocity)],
        device=device,
        dtype=torch.float32,
    )
    seed_tensor = torch.tensor(
        [excitation_seed],
        device=device,
        dtype=torch.long,
    )
    decoded = decoder.decode_latents(
        latent,
        clap_tensor,
        note_tensor,
        velocity_tensor,
        seed_tensor,
    )
    result = decoded[0].detach().float().cpu().numpy().reshape(-1)
    return _mono_float32(result, "decoded")


def _write_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    clip = _mono_float32(audio, path.name)
    if float(np.max(np.abs(clip))) > 1.0:
        raise ValueError(f"audio exceeds [-1, 1]: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, clip, sample_rate, subtype="FLOAT")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_audio_and_clap(
    config: ZraveConfig,
    rows: Iterable[dict[str, object]],
    source_sample_rate: int,
    clap_checkpoint: str | Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    unique = {
        str(row["sample_id"]): row
        for row in rows
    }
    clap_model = _load_clap_model(clap_checkpoint, device)
    audio_by_id: dict[str, np.ndarray] = {}
    clap_by_id: dict[str, np.ndarray] = {}
    for sample_id, row in unique.items():
        sample_rate = int(row.get("sample_rate") or source_sample_rate)
        if sample_rate != source_sample_rate:
            raise ValueError(
                f"source sample rate mismatch for {sample_id}: "
                f"{sample_rate} != {source_sample_rate}"
            )
        audio = load_audio(_audio_path(config, row), sample_rate)
        audio_by_id[sample_id] = audio
        clap_by_id[sample_id] = _clap_embedding(
            clap_model,
            audio,
            sample_rate,
            device,
        )
    del clap_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return audio_by_id, clap_by_id


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
    for required in (
        index_path,
        statistics_path,
        latents_path,
        checkpoint,
        Path(config.rave.checkpoint),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing audition artifact: {required}")

    destination = Path(output_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"audition output directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    audio_root = destination / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != 1:
        raise ValueError("unsupported packed index schema")
    if index.get("rave_checkpoint_sha256") != _sha256_file(
        config.rave.checkpoint
    ):
        raise ValueError("packed latents and decoder checkpoint do not match")
    manifest_rows = read_jsonl(config.data.selected_manifest)
    selected = select_audition_rows(
        index,
        manifest_rows,
        _CATEGORIES,
    )
    sample_rate = 44100
    latent_hop = config.data.latent_hop
    future_frames = duration_to_frames(
        duration_seconds,
        sample_rate,
        latent_hop,
    )
    required_frames = config.model.context_frames + future_frames
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
    decoder, source_config, decoder_payload = _load_decoder(config, device)
    if source_config.predictive is None:
        raise AssertionError("predictive decoder config disappeared")
    sample_rate = source_config.data.sample_rate
    if sample_rate != 44100:
        raise ValueError("audition decoder must use the 44.1 kHz project rate")
    if source_config.predictive.samples_per_latent != latent_hop:
        raise ValueError("decoder and packed latent hops do not match")
    if source_config.predictive.rave_latent_dim != config.model.latent_dim:
        raise ValueError("decoder and Transformer latent dimensions differ")
    if not source_config.data.clap_checkpoint:
        raise ValueError("decoder source config has no CLAP checkpoint")

    audio_by_id, clap_by_id = _source_audio_and_clap(
        config,
        [*selected, *random_rows],
        sample_rate,
        source_config.data.clap_checkpoint,
        device,
    )
    latents = np.load(latents_path, mmap_mode="r", allow_pickle=False)
    comparisons: list[dict[str, object]] = []
    context_frames = config.model.context_frames
    for row_index, row in enumerate(selected, 1):
        sample_id = str(row["sample_id"])
        start = deterministic_start(
            sample_id,
            int(row["length"]),
            required_frames,
            random_seed,
        )
        sequence_index = int(row["sequence_index"])
        true_sequence = torch.from_numpy(
            np.asarray(
                latents[
                    sequence_index,
                    start : start + required_frames,
                ],
                dtype=np.float32,
            ).copy()
        ).to(device)
        history = true_sequence[:context_frames].unsqueeze(0)
        predicted_future = rollout_latents(
            transformer,
            history,
            future_frames,
        )[0]
        direct_latent = true_sequence.transpose(0, 1).unsqueeze(0)
        predicted_latent = torch.cat(
            (history[0], predicted_future),
            dim=0,
        ).transpose(0, 1).unsqueeze(0)
        excitation_seed = random_seed + row_index
        direct_full = _decode_latents(
            decoder,
            direct_latent,
            clap_by_id[sample_id],
            int(row["midi_note"]),
            int(row["velocity"]),
            excitation_seed,
            device,
        )
        predicted_full = _decode_latents(
            decoder,
            predicted_latent,
            clap_by_id[sample_id],
            int(row["midi_note"]),
            int(row["velocity"]),
            excitation_seed,
            device,
        )
        direct = crop_decoded_future(
            direct_full,
            context_frames=context_frames,
            future_frames=future_frames,
            latent_hop=latent_hop,
        )
        predicted = crop_decoded_future(
            predicted_full,
            context_frames=context_frames,
            future_frames=future_frames,
            latent_hop=latent_hop,
        )
        original = crop_source_future(
            audio_by_id[sample_id],
            warmup_frames=config.data.warmup_frames,
            window_start=start,
            context_frames=context_frames,
            future_frames=future_frames,
            latent_hop=latent_hop,
        )
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
                "midi_note": int(row["midi_note"]),
                "velocity": int(row["velocity"]),
                "sequence_index": sequence_index,
                "window_start": start,
                "source_sample_start": (
                    config.data.warmup_frames + start + context_frames
                ) * latent_hop,
                "sample_count": int(original.shape[0]),
                "shared_gain": gain,
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
        sequence_index = int(row["sequence_index"])
        true_history = torch.from_numpy(
            np.asarray(
                latents[
                    sequence_index,
                    start : start + context_frames,
                ],
                dtype=np.float32,
            ).copy()
        ).to(device).unsqueeze(0)
        prediction = rollout_latents(
            transformer,
            true_history,
            future_frames,
        )[0]
        latent = torch.cat(
            (true_history[0], prediction),
            dim=0,
        ).transpose(0, 1).unsqueeze(0)
        decoded = _decode_latents(
            decoder,
            latent,
            clap_by_id[sample_id],
            int(row["midi_note"]),
            int(row["velocity"]),
            random_seed + 100 + continuation_index,
            device,
        )
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
                "midi_note": int(row["midi_note"]),
                "velocity": int(row["velocity"]),
                "sequence_index": sequence_index,
                "window_start": start,
                "seed_frames": context_frames,
                "predicted_frames": future_frames,
                "transition_seconds": (
                    context_frames * latent_hop / sample_rate
                ),
                "sample_count": int(decoded.shape[0]),
                "gain": gain,
                "file": relative_path,
            }
        )

    manifest: dict[str, object] = {
        "schema": 1,
        "title": "Z-RAVE Sequence Transformer Audition",
        "categories": list(_CATEGORIES),
        "sample_rate": sample_rate,
        "latent_hop": latent_hop,
        "context_frames": context_frames,
        "future_frames": future_frames,
        "audible_duration_seconds": (
            future_frames * latent_hop / sample_rate
        ),
        "random_seed": random_seed,
        "models": {
            "transformer": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256_file(checkpoint),
                "update": int(transformer_payload["update"]),
                "architecture": transformer_payload["architecture"],
            },
            "decoder": {
                "path": str(Path(config.rave.checkpoint).resolve()),
                "sha256": _sha256_file(config.rave.checkpoint),
                "update": int(
                    decoder_payload.get(
                        "stage_update",
                        decoder_payload.get("update", 0),
                    )
                ),
                "pad_focused": True,
            },
            "clap": {
                "path": str(
                    Path(source_config.data.clap_checkpoint).resolve()
                ),
                "sha256": _sha256_file(
                    source_config.data.clap_checkpoint
                ),
            },
        },
        "packed_index_sha256": _sha256_file(index_path),
        "comparisons": comparisons,
        "random_continuations": continuations,
    }
    validate_audition_manifest(manifest, _CATEGORIES)
    _atomic_json(destination / "audition-manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(destination.resolve()),
                "comparisons": len(comparisons),
                "random_continuations": len(continuations),
                "wav_files": len(comparisons) * 3 + len(continuations),
                "future_frames": future_frames,
                "finite": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest
