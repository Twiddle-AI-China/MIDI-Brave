from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import soundfile as sf


PAD_RANKED_55 = (
    "serum_s089661", "serum_s113668", "serum_s007608", "serum_s048540", "serum_s089009",
    "serum_s047927", "serum_s100654", "serum_s062227", "serum_s003148", "serum_s066203",
    "serum_s006630", "serum_s066204", "serum_s025385", "serum_s048007", "serum_s002724",
    "serum_s016782", "serum_s048543", "serum_s062086", "serum_s048063", "serum_s000639",
    "serum_s072458", "serum_s065410", "serum_s018095", "serum_s048066", "serum_s014377",
    "serum_s016790", "serum_s113014", "serum_s063514", "serum_s067453", "serum_s023971",
    "serum_s072507", "serum_s100767", "serum_s065228", "serum_s100765", "serum_s113170",
    "serum_s086903", "serum_s004065", "serum_s007435", "serum_s113234", "serum_s078108",
    "serum_s065837", "serum_s065995", "serum_s073090", "serum_s112100", "serum_s006632",
    "serum_s098623", "serum_s006355", "serum_s046519", "serum_s096923", "serum_s056601",
    "serum_s091504", "serum_s043574", "serum_s047877", "serum_s003030", "serum_s000303",
)
PAD_DUPLICATE_DROPS = {
    "serum_s007608": "serum_s113668",
    "serum_s003148": "serum_s062227",
    "serum_s113234": "serum_s007435",
    "serum_s078108": "serum_s007435",
    "serum_s006632": "serum_s112100",
}
BROAD_NOTES = tuple(range(36, 72))
REPEATED_NOTES = (36, 43, 50, 57, 64, 71)
SEED = 20260822


def pad_unique_top50() -> tuple[str, ...]:
    selected = tuple(item for item in PAD_RANKED_55 if item not in PAD_DUPLICATE_DROPS)
    if len(selected) != 50 or len(set(selected)) != 50:
        raise RuntimeError("Pad dedupe/refill contract did not produce 50 unique presets")
    return selected


def split_presets(presets: Iterable[str], seed: int = SEED) -> dict[str, str]:
    ordered = sorted(
        presets,
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).digest(),
    )
    if len(ordered) != 50 or len(set(ordered)) != 50:
        raise ValueError("split requires exactly 50 unique presets")
    held_out = {item: "validation" for item in ordered[:2]}
    held_out.update({item: "test" for item in ordered[2:5]})
    return {item: held_out.get(item, "train") for item in presets}


def preset_index(preset_id: str) -> int:
    if not preset_id.startswith("serum_s") or not preset_id[7:].isdigit():
        raise ValueError(f"invalid Serum preset id: {preset_id}")
    return int(preset_id[7:])


def render_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for preset in pad_unique_top50():
        for note in BROAD_NOTES:
            repeats = 4 if note in REPEATED_NOTES else 1
            for render_id in range(repeats):
                rows.append({
                    "preset_index": preset_index(preset),
                    "action": "render",
                    "config_id": f"n{note:03d}_r{render_id}",
                    "midi_input_note": note,
                    "midi_velocity": 127,
                    "duration_mode": "fixed",
                    "note_start_seconds": 0.1,
                    "note_off_seconds": 2.6,
                    "render_duration_seconds": 5.0,
                    # Kept for backward-compatible database columns. The
                    # fixed renderer treats note_off_seconds as authoritative.
                    "note_on_seconds": 2.6,
                    "persistent_cap_seconds": 5.0,
                })
    if len(rows) != 2_700:
        raise RuntimeError(f"expected 2700 render rows, got {len(rows)}")
    return rows


def expected_filename(preset: str, note: int, render_id: int) -> str:
    return (
        f"{preset_index(preset):06d}_n{note:03d}_v127_on2600_"
        f"n{note:03d}_r{render_id}.wav"
    )


def training_rows(wav_root: str | Path, check_audio: bool = True) -> list[dict[str, object]]:
    root = Path(wav_root)
    splits = split_presets(pad_unique_top50())
    rows: list[dict[str, object]] = []
    for preset in pad_unique_top50():
        for note in BROAD_NOTES:
            repeats = 4 if note in REPEATED_NOTES else 1
            for render_id in range(repeats):
                filename = expected_filename(preset, note, render_id)
                path = root / filename
                if check_audio:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    info = sf.info(path)
                    if info.samplerate != 44_100 or info.channels != 1 or info.frames != 220_500:
                        raise ValueError(
                            f"{path}: expected 44100Hz mono/220500, got "
                            f"{info.samplerate}Hz/{info.channels}ch/{info.frames}"
                        )
                sample_id = f"{preset}_n{note:03d}_r{render_id}"
                rows.append({
                    "sample_id": sample_id,
                    "audio_path": filename,
                    "source_id": "serum-octopus-v2",
                    "dataset_id": "atlas-flow-pad-v1",
                    "preset_id": preset,
                    "timbre_id": preset,
                    "articulation_id": "fixed_on0100_off2600_end5000",
                    "midi_note": note,
                    "midi_note_sent": note,
                    "transpose_semitones": 0,
                    "velocity": 127,
                    "velocity_known": True,
                    "sample_rate": 44_100,
                    "num_samples": 220_500,
                    "duration_seconds": 5.0,
                    "a4_tuning_hz": 440.0,
                    "render_or_recording": "render",
                    "render_gain_db": 0.0,
                    "render_id": render_id,
                    "class_name": "pad",
                    "split": splits[preset],
                })
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def materialize(output_root: str | Path, wav_root: str | Path | None, check_audio: bool) -> dict[str, object]:
    root = Path(output_root)
    render_manifest = root / "pad-top50-render.jsonl"
    _write_jsonl(render_manifest, render_rows())
    selected = pad_unique_top50()
    splits = split_presets(selected)
    report: dict[str, object] = {
        "schema": "midibrave.atlas-flow.pad-selection.v1",
        "seed": SEED,
        "selected": list(selected),
        "source_ranks": {item: PAD_RANKED_55.index(item) + 1 for item in selected},
        "duplicate_drops": PAD_DUPLICATE_DROPS,
        "split_counts": {name: list(splits.values()).count(name) for name in ("train", "validation", "test")},
        "render_rows": 2_700,
        "render_contract": {
            "sample_rate": 44_100, "channels": 1, "duration_seconds": 5.0,
            "note_start_seconds": 0.1, "note_off_seconds": 2.6, "velocity": 127,
        },
    }
    if wav_root is not None:
        manifest = root / "pad-top50-training.jsonl"
        rows = training_rows(wav_root, check_audio=check_audio)
        _write_jsonl(manifest, rows)
        report["training_manifest"] = str(manifest)
        report["training_rows"] = len(rows)
    report_path = root / "pad-top50-selection.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the deduplicated Pad Top50 render contract.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wav-root")
    parser.add_argument("--skip-audio-check", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(materialize(args.output_root, args.wav_root, not args.skip_audio_check), indent=2))


if __name__ == "__main__":
    main()
