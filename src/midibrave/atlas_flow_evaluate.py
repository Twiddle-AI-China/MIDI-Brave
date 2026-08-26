from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .atlas_flow_atlas import TimbreAtlas
from .atlas_flow_config import load_atlas_flow_config
from .atlas_flow_data import feature_path, lifecycle_frames, resolve_audio
from .atlas_flow_model import AtlasFlowSystem
from .data import load_manifest


AUDITION_NOTES = (36, 43, 50, 57, 64, 71)


def _json_native(value: object) -> object:
    """Convert evaluation values to strict, portable JSON values.

    NumPy comparisons deliberately return ``np.bool_`` and aggregate functions
    return NumPy scalar types.  Letting those values reach ``json.dumps`` made
    the post-training chain fail after all audio had already been rendered.
    Keep this conversion at the report boundary so metric code remains free to
    use NumPy naturally while every persisted/public value has a stable type.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_native(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    raise TypeError(f"unsupported evaluation JSON type: {type(value).__name__}")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    native = _json_native(payload)
    encoded = json.dumps(native, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _rms_db(value: np.ndarray) -> float:
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))), 1.0e-8)))


def _spectral_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    total = 0.0
    for size in (2_048, 1_024, 512, 256):
        window = torch.hann_window(size, device=prediction.device)
        a = torch.stft(prediction.float(), size, size // 4, size, window, return_complex=True).abs().clamp_min(1.0e-6)
        b = torch.stft(target.float(), size, size // 4, size, window, return_complex=True).abs().clamp_min(1.0e-6)
        total += float((a.log() - b.log()).abs().mean())
    return total / 4


def _midi_estimate(audio: np.ndarray, sample_rate: int, start: int, stop: int) -> float:
    value = audio[start:stop].astype(np.float64)
    if value.size < 4_096:
        return float("nan")
    value = value - value.mean()
    spectrum = np.abs(np.fft.rfft(value * np.hanning(value.size)))
    frequencies = np.fft.rfftfreq(value.size, 1.0 / sample_rate)
    candidates = np.arange(36, 72)
    scores = []
    for note in candidates:
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        score = 0.0
        for harmonic in range(1, 9):
            index = int(np.argmin(np.abs(frequencies - f0 * harmonic)))
            score += spectrum[index] / harmonic
        scores.append(score)
    return float(candidates[int(np.argmax(scores))])


def _boundary_ratio(audio: np.ndarray, sample: int, width: int = 2_048) -> float:
    local = audio[max(0, sample - width):min(audio.size, sample + width)]
    baseline = np.median(np.abs(np.diff(local))) + 1.0e-8
    return float(abs(float(audio[sample] - audio[sample - 1])) / baseline)


@torch.no_grad()
def _generate_trajectory(
    system: AtlasFlowSystem,
    atlas: TimbreAtlas,
    preset_id: str,
    frames: int,
    device: torch.device,
    *,
    seed: int,
    temperature: float,
) -> torch.Tensor:
    index = atlas.preset_ids.index(preset_id)
    anchor = torch.from_numpy(atlas.anchors[index]).to(device)
    coordinate = torch.from_numpy(atlas.coordinates[index]).to(device)
    context = system.flow.config.context_frames
    history = anchor.view(1, 1, -1).expand(1, context, -1).clone()
    history_anchor = history.clone()
    history_mask = torch.zeros(1, context, dtype=torch.bool, device=device)
    history_mask[:, -1] = True
    chunks: list[torch.Tensor] = []
    produced = 0
    while produced < frames:
        future = system.flow.config.future_frames
        anchor_path = anchor.view(1, 1, -1).expand(1, future, -1)
        atlas_path = coordinate.view(1, 1, -1).expand(1, future, -1)
        lifecycle = torch.from_numpy(lifecycle_frames(produced, future, system.instrument.data_config)).to(device)[None]
        generated = system.flow.sample(
            history, history_anchor, anchor_path, atlas_path, lifecycle,
            seed=seed + produced, temperature=temperature, history_mask=history_mask,
        )
        chunks.append(generated[0])
        combined = torch.cat((history, generated), 1)
        history = combined[:, -context:]
        history_anchor = anchor.view(1, 1, -1).expand_as(history)
        history_mask = torch.ones_like(history_mask)
        produced += future
    return torch.cat(chunks, 0)[:frames].T.contiguous()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    config = load_atlas_flow_config(args.config)
    device = torch.device(args.device)
    system = AtlasFlowSystem(config.model, config.data).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    system.load_state_dict(checkpoint["system"], strict=True)
    system.eval()
    atlas = TimbreAtlas.load(config.data.atlas_path)
    output = Path(args.output)
    audio_root = output / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    records = [
        record for record in load_manifest(config.data.manifest)
        if record.split == "test" and record.sample_id.endswith("_r0")
    ]
    by_key: dict[tuple[str, int], object] = {}
    for record in records:
        key = (record.preset_id, record.midi_note)
        if key in by_key:
            raise RuntimeError(f"duplicate deterministic test render: {key}")
        by_key[key] = record
    presets = sorted({key[0] for key in by_key})
    expected = {(preset, note) for preset in presets for note in range(36, 72)}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        raise RuntimeError(f"test population is incomplete: {missing[:5]}")
    metric_rows: list[dict[str, object]] = []
    audition_rows: list[dict[str, object]] = []
    generated_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]] = {}
    for preset_id, note in sorted(by_key):
        record_value = by_key[(preset_id, note)]
        record = record_value
        features = torch.from_numpy(np.load(feature_path(record, config.data)).astype(np.float32))[None].to(device)
        reference, rate = sf.read(resolve_audio(record, config.data), dtype="float32")
        trajectory = system.instrument.encode(features)
        static = trajectory.mean(-1, keepdim=True).expand_as(trajectory)
        note_tensor = torch.tensor([note], device=device)
        dynamic_audio, _ = system.instrument.decode(trajectory, note_tensor, config.data.render_samples)
        static_audio, _ = system.instrument.decode(static, note_tensor, config.data.render_samples)
        if preset_id not in generated_cache:
            generated = _generate_trajectory(
                system, atlas, preset_id, trajectory.shape[-1], device,
                seed=20260822, temperature=0.0,
            )
            generated_again = _generate_trajectory(
                system, atlas, preset_id, trajectory.shape[-1], device,
                seed=20260822, temperature=0.0,
            )
            stochastic_a = _generate_trajectory(
                system, atlas, preset_id, trajectory.shape[-1], device,
                seed=1, temperature=1.0,
            )
            stochastic_b = _generate_trajectory(
                system, atlas, preset_id, trajectory.shape[-1], device,
                seed=2, temperature=1.0,
            )
            generated_cache[preset_id] = (
                generated, stochastic_a, stochastic_b,
                bool(torch.equal(generated, generated_again)),
                float((stochastic_a - stochastic_b).float().square().mean().sqrt()),
            )
        generated, _stochastic_a, _stochastic_b, deterministic, stochastic_distance = generated_cache[preset_id]
        flow_audio, _ = system.instrument.decode(
            generated[None], note_tensor, config.data.render_samples,
        )
        dynamic_np = dynamic_audio[0, 0].float().cpu().numpy()
        static_np = static_audio[0, 0].float().cpu().numpy()
        flow_np = flow_audio[0, 0].float().cpu().numpy()
        target_tensor = torch.from_numpy(reference).to(device)
        dynamic_error = _spectral_error(dynamic_audio[0, 0], target_tensor)
        static_error = _spectral_error(static_audio[0, 0], target_tensor)
        flow_error = _spectral_error(flow_audio[0, 0], target_tensor)
        row = {
            "preset_id": preset_id,
            "sample_id": record.sample_id,
            "note": note,
            "dynamic_spectral_error": dynamic_error,
            "static_spectral_error": static_error,
            "dynamic_improvement_fraction": (static_error - dynamic_error) / max(static_error, 1.0e-8),
            "flow_spectral_error": flow_error,
            "flow_rms_error_db": abs(_rms_db(flow_np) - _rms_db(reference)),
            "flow_tail_rms_drift_db": abs(_rms_db(flow_np[-22_050:]) - _rms_db(reference[-22_050:])),
            "flow_boundary_ratio": _boundary_ratio(flow_np, config.data.note_off_sample),
            "flow_silent": float(np.max(np.abs(flow_np))) < 0.01,
            "estimated_midi": _midi_estimate(
                flow_np, rate, config.data.note_on_sample + 22_050, config.data.note_off_sample - 22_050
            ),
            "deterministic_exact": deterministic,
            "stochastic_latent_distance": stochastic_distance,
        }
        metric_rows.append(row)
        if note in AUDITION_NOTES:
            prefix = f"{preset_id}-n{note:03d}"
            for name, value in (
                ("source", reference), ("dynamic", dynamic_np),
                ("static", static_np), ("flow", flow_np),
            ):
                sf.write(audio_root / f"{prefix}-{name}.wav", value, rate, subtype="PCM_16")
            audition_rows.append({
                **row,
                "audio": {
                    name: f"audio/{prefix}-{name}.wav"
                    for name in ("source", "dynamic", "static", "flow")
                },
            })
    improvements = [float(row["dynamic_improvement_fraction"]) for row in metric_rows]
    rms_errors = sorted(float(row["flow_rms_error_db"]) for row in metric_rows)
    following = [
        abs(float(row["estimated_midi"]) - int(row["note"])) <= 0.5
        for row in metric_rows
    ]
    gates: dict[str, bool] = {
        "midi_following_95pct": bool(np.mean(following) >= 0.95),
        "dynamic_median_5pct": bool(np.median(improvements) >= 0.05),
        "dynamic_benefit_70pct": bool(np.mean(np.asarray(improvements) > 0) >= 0.70),
        "silence_le_1pct": bool(np.mean([bool(row["flow_silent"]) for row in metric_rows]) <= 0.01),
        "tail_drift_le_12db": bool(max(float(row["flow_tail_rms_drift_db"]) for row in metric_rows) <= 12.0),
        "boundary_ratio_le_4": bool(max(float(row["flow_boundary_ratio"]) for row in metric_rows) <= 4.0),
        "rms_median_le_6db": bool(np.median(rms_errors) <= 6.0),
        "rms_p90_le_12db": bool(np.percentile(rms_errors, 90) <= 12.0),
        "deterministic": bool(all(bool(row["deterministic_exact"]) for row in metric_rows)),
    }
    report: dict[str, object] = {
        "schema": "midibrave.atlas-flow.evaluation.v2",
        "pipeline_state": "complete",
        "quality_state": "pass" if all(gates.values()) else "fail",
        "checkpoint": str(args.checkpoint),
        "test_presets": len(presets),
        "population": {
            "split": "test",
            "presets": len(presets),
            "notes": list(range(36, 72)),
            "records": len(metric_rows),
            "render_selector": "r0",
        },
        "metric_rows": metric_rows,
        "audition": {
            "notes": list(AUDITION_NOTES),
            "records": len(audition_rows),
            "rows": audition_rows,
        },
        # Compatibility for the v1 static dashboard during atomic rollout.
        "rows": audition_rows,
        "gates": gates,
        "passed": all(gates.values()),
        "limitations": [
            "This automatic run covers Pad test presets; four-class and listening-study gates remain pending.",
            "Real-time P99 and 10-minute soak are reported by the separate runtime probe.",
        ],
    }
    native_report = _json_native(report)
    if not isinstance(native_report, dict):
        raise RuntimeError("evaluation report must be a JSON object")
    _atomic_json(output / "evaluation.json", native_report)
    return native_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Atlas Flow and produce an audition bundle.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
