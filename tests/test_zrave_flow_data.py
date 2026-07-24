from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn

from midibrave.zrave_flow_config import ZraveFlowConfig
from midibrave.zrave_flow_data import (
    encode_flow_shards,
    finalize_flow_pack,
)


ROOT = Path(__file__).parents[1]
FORMAL_CONFIG = ROOT / "configs" / "zrave" / "octopus_flow.yaml"


class _FakeStandaloneCodec(nn.Module):
    sr = 44100
    latent_size = 16

    def encode(self, audio: Tensor, _temperature: float) -> Tensor:
        frames = audio.shape[-1] // 2048
        code = torch.round(audio[:, :, 0] * 10.0)
        values = torch.where(
            code == 1,
            torch.ones_like(code),
            torch.where(
                code == 2,
                torch.full_like(code, 100.0),
                torch.where(
                    code == -2,
                    torch.full_like(code, -100.0),
                    torch.full_like(code, 2.0),
                ),
            ),
        )
        return values.view(-1, 1, 1).expand(-1, 16, frames).clone()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_config(
    tmp_path: Path,
    *,
    shard_records: int,
) -> ZraveFlowConfig:
    config = ZraveFlowConfig.load(FORMAL_CONFIG)
    data = replace(
        config.data,
        unified_manifest=str(tmp_path / "manifest.jsonl"),
        manifest_report=str(tmp_path / "manifest-report.json"),
        cache_root=str(tmp_path / "cache"),
        packed_root=str(tmp_path / "pack"),
        shard_records=shard_records,
    )
    rave = replace(config.rave, checkpoint=str(tmp_path / "fake-codec.ts"))
    return replace(config, data=data, rave=rave)


def _audio(
    path: Path,
    *,
    code: float,
    frames: int = 100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(frames * 2048, code, dtype=np.float32)
    sf.write(path, samples, 44100, subtype="FLOAT")


def _row(
    index: int,
    audio_path: Path,
    *,
    source_name: str,
    split: str,
    note: int,
    preset: str,
    category: str,
    maximum_future: int,
    velocity: int = 100,
) -> dict[str, object]:
    return {
        "sample_id": f"{source_name}__sample-{index}",
        "source_name": source_name,
        "audio_path": str(audio_path.resolve()),
        "canonical_preset_id": preset,
        "split": split,
        "midi_note": note,
        "velocity": velocity,
        "articulation_id": "steady",
        "category": category,
        "duration_seconds": 100 * 2048 / 44100,
        "maximum_future_frames": maximum_future,
    }


def test_shards_preserve_masks_notes_and_contract(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, shard_records=2)
    sources = [
        "serum_full",
        "dexed_dense",
        "serum_dense",
        "dexed_surge_broad",
        "dexed_dense",
    ]
    notes = [36, 62, 48, 72, 60]
    rows: list[dict[str, object]] = []
    for index, (source, note) in enumerate(zip(sources, notes, strict=True)):
        audio = tmp_path / "audio" / f"{index}.wav"
        _audio(audio, code=0.3)
        rows.append(
            _row(
                index,
                audio,
                source_name=source,
                split="validation" if index == 0 else "train",
                note=note,
                preset=f"preset:{index}",
                category="Pad" if source.startswith("serum") else "Dexed",
                maximum_future=20 if source == "serum_dense" else 64,
            )
        )
    _write_jsonl(Path(config.data.unified_manifest), rows)

    report = encode_flow_shards(
        config,
        rank=0,
        world_size=1,
        device="cpu",
        codec=_FakeStandaloneCodec(),
    )
    final = finalize_flow_pack(config)

    assert report["shards"] == 3
    assert final["records"] == 5
    assert final["shard_records"] == [2, 2, 1]
    pitch_pairs = np.load(Path(config.data.packed_root) / "pitch-pairs.npy")
    assert pitch_pairs.dtype == np.int64
    assert pitch_pairs.shape == (5,)
    seed_bank = np.load(Path(config.data.packed_root) / "seed-bank.npz")
    assert seed_bank["history"].shape[1:] == (32, 16)
    with np.load(
        Path(config.data.packed_root) / "shard-000000.npz"
    ) as shard:
        assert shard["latents"].dtype == np.float16
        assert shard["latents"].shape[2] == 16
        assert shard["notes"].tolist() == [36, 62]
        assert shard["maximum_future_frames"].tolist() == [64, 64]
        assert shard["active_frames"].dtype == np.int16


def test_pack_statistics_ignore_validation_and_test(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, shard_records=4)
    specs = [
        ("train", 0.1, 1.0),
        ("validation", 0.2, 100.0),
        ("test", -0.2, -100.0),
    ]
    rows: list[dict[str, object]] = []
    codec = _FakeStandaloneCodec()
    for index, (split, audio_code, _latent_value) in enumerate(specs):
        audio = tmp_path / "audio" / f"stats-{index}.wav"
        _audio(audio, code=audio_code)
        rows.append(
            _row(
                index,
                audio,
                source_name="dexed_dense",
                split=split,
                note=60,
                preset=f"dexed:{index:04d}",
                category="Dexed",
                maximum_future=64,
            )
        )
    _write_jsonl(Path(config.data.unified_manifest), rows)

    encode_flow_shards(
        config,
        rank=0,
        world_size=1,
        device="cpu",
        codec=codec,
    )
    finalize_flow_pack(config)

    with np.load(Path(config.data.packed_root) / "statistics.npz") as stats:
        assert np.allclose(stats["mean"], 1.0)
        assert stats["latent_norm_p01"] < 10.0
        assert stats["latent_norm_p99"] < 10.0
