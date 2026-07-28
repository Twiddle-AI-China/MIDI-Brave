from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from midibrave.zrave_codec_qualification import (
    aligned_codec_window,
    qualification_file_names,
    render_codec_qualification,
    select_qualification_cases,
    shared_listening_gain,
    validate_latent_pair,
    write_qualification_case,
)


def test_aligned_codec_window_uses_real_preroll_and_matching_source() -> None:
    sequence = torch.arange(107 * 16).reshape(107, 16).float()
    source = np.arange(107 * 4, dtype=np.float32)

    preroll, audible, reference = aligned_codec_window(
        sequence,
        source,
        latent_hop=4,
    )

    assert torch.equal(preroll, sequence[:32])
    assert torch.equal(audible, sequence[32:107])
    assert np.array_equal(reference, source[32 * 4 : 107 * 4])


def test_aligned_codec_window_rejects_insufficient_real_context() -> None:
    with pytest.raises(ValueError, match="107"):
        aligned_codec_window(
            torch.zeros(106, 16),
            np.zeros(107 * 4, dtype=np.float32),
            latent_hop=4,
        )

    with pytest.raises(ValueError, match="source"):
        aligned_codec_window(
            torch.zeros(107, 16),
            np.zeros(107 * 4 - 1, dtype=np.float32),
            latent_hop=4,
        )


def test_validate_latent_pair_accepts_float16_storage_error() -> None:
    fresh = torch.linspace(-2.0, 2.0, 107 * 16).reshape(107, 16)
    packed = fresh.half().float()

    report = validate_latent_pair(packed, fresh)

    assert report["cosine"] >= 0.99999
    assert report["rmse"] < 0.005
    assert report["maximum_absolute_error"] < 0.005


def test_validate_latent_pair_rejects_wrong_source_binding() -> None:
    packed = torch.ones(107, 16)

    with pytest.raises(ValueError, match="pairing"):
        validate_latent_pair(packed, -packed)


def test_shared_listening_gain_is_one_shared_attenuation() -> None:
    source = np.array([-0.5, 0.5], dtype=np.float32)
    direct = np.array([-2.0, 1.0], dtype=np.float32)

    assert shared_listening_gain(source, direct) == pytest.approx(0.475)
    assert shared_listening_gain(source, source) == 1.0


def test_qualification_file_names_define_codec_only_pair() -> None:
    assert qualification_file_names(2, "Lead") == {
        "source": "wavs/02-lead-source.wav",
        "rave_direct": "wavs/02-lead-rave-direct.wav",
    }

    with pytest.raises(ValueError, match="safe"):
        qualification_file_names(0, "../Lead")


def test_select_qualification_cases_requires_107_stored_frames(
    tmp_path,
) -> None:
    manifest = tmp_path / "sequences.jsonl"
    rows = [
        {
            "split": "test",
            "source_name": "serum_full",
            "category": "Pad",
            "length": 106,
            "active_frames": 106,
            "canonical_preset_id": "serum:short",
            "sample_id": "short",
        },
        {
            "split": "test",
            "source_name": "serum_full",
            "category": "Pad",
            "length": 107,
            "active_frames": 99,
            "canonical_preset_id": "serum:eligible",
            "sample_id": "eligible",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = select_qualification_cases(
        manifest,
        categories=("Pad",),
        seed=20260728,
    )

    assert [row["sample_id"] for row in selected] == ["eligible"]


def test_write_qualification_case_records_aligned_pcm16_pair(
    tmp_path,
) -> None:
    latent_hop = 2048
    samples = 75 * latent_hop
    phase = np.arange(samples, dtype=np.float64) / 44100.0
    source = (0.5 * np.sin(2.0 * np.pi * 220.0 * phase)).astype(
        np.float32
    )
    direct = source * 0.8
    packed = torch.linspace(-2.0, 2.0, 107 * 16).reshape(107, 16)
    fresh = packed.half().float()

    report = write_qualification_case(
        tmp_path,
        case_index=0,
        category="Pad",
        source=source,
        direct=direct,
        packed=packed,
        fresh=fresh,
        sample_rate=44100,
        latent_hop=latent_hop,
        preroll_frames=32,
        audible_end_frame=107,
    )

    assert report["decoder_preroll_frames"] == 32
    assert report["source_start_frame"] == 32
    assert report["audible_frames"] == 75
    assert report["pairing"]["cosine"] >= 0.99999
    assert report["audio_metrics"]["spectral_cosine"] > 0.99
    assert report["listening_gain"] == 1.0
    assert set(report["audio"]) == {"source", "rave_direct"}
    for role in ("source", "rave_direct"):
        item = report["audio"][role]
        info = sf.info(tmp_path / item["wav"])
        assert info.channels == 1
        assert info.samplerate == 44100
        assert info.frames == samples
        assert info.subtype == "PCM_16"
        assert len(item["sha256"]) == 64


def test_write_qualification_case_rejects_dominant_cold_start(
    tmp_path,
) -> None:
    latent_hop = 2048
    samples = 75 * latent_hop
    source = np.ones(samples, dtype=np.float32) * 0.5
    direct = source.copy()
    direct[: 4 * latent_hop] = 2.0
    packed = torch.ones(107, 16)

    with pytest.raises(ValueError, match="cold-start"):
        write_qualification_case(
            tmp_path,
            case_index=0,
            category="Pad",
            source=source,
            direct=direct,
            packed=packed,
            fresh=packed,
            sample_rate=44100,
            latent_hop=latent_hop,
            preroll_frames=32,
            audible_end_frame=107,
        )


def test_write_qualification_case_allows_a_naturally_silent_tail(
    tmp_path,
) -> None:
    latent_hop = 2048
    samples = 75 * latent_hop
    phase = np.arange(samples, dtype=np.float64) / 44100.0
    source = (0.5 * np.sin(2.0 * np.pi * 220.0 * phase)).astype(
        np.float32
    )
    source[-3 * latent_hop :] = 0.0
    packed = torch.ones(107, 16)

    report = write_qualification_case(
        tmp_path,
        case_index=0,
        category="Pad",
        source=source,
        direct=source,
        packed=packed,
        fresh=packed,
        sample_rate=44100,
        latent_hop=latent_hop,
    )

    assert report["audio_metrics"]["spectral_cosine"] > 0.99
    assert len(report["audio_metrics"]["chunks"]) == 1


def test_renderer_has_no_transformer_checkpoint_and_rejects_nonempty_output(
    tmp_path,
) -> None:
    assert "checkpoint" not in inspect.signature(
        render_codec_qualification
    ).parameters
    output = tmp_path / "qualification"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        render_codec_qualification(
            tmp_path / "missing.yaml",
            output,
            device="cpu",
        )


class _IdentityQualificationCodec(nn.Module):
    sr = 44100

    def encode(
        self,
        audio: torch.Tensor,
        _: float,
    ) -> torch.Tensor:
        return torch.ones(
            audio.shape[0],
            16,
            audio.shape[2] // 2048,
            device=audio.device,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latent.shape[0], 1, latent.shape[2] * 2048),
            0.2,
            device=latent.device,
        )


def test_render_codec_qualification_writes_codec_only_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    packed_root = tmp_path / "pack"
    packed_root.mkdir()
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("qualification fixture\n", encoding="utf-8")
    unified_manifest = tmp_path / "manifest.jsonl"
    categories = ("Pad", "Bass", "Lead", "Pluck", "Keys")
    sequence_rows = []
    source_rows = []
    for index, category in enumerate(categories):
        audio_path = audio_root / f"{category.casefold()}.wav"
        sf.write(
            audio_path,
            np.full(107 * 2048, 0.2, dtype=np.float32),
            44100,
            subtype="FLOAT",
        )
        source_rows.append({"audio_path": str(audio_path)})
        sequence_rows.append(
            {
                "split": "test",
                "source_name": "serum_full",
                "category": category,
                "length": 107,
                "active_frames": 107,
                "canonical_preset_id": f"serum:{index}",
                "sample_id": f"sample-{index}",
                "manifest_index": index,
                "shard": "audition.npz",
                "shard_row": index,
            }
        )
    unified_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    (packed_root / "sequences.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sequence_rows),
        encoding="utf-8",
    )
    np.savez(
        packed_root / "audition.npz",
        latents=np.ones((5, 107, 16), dtype=np.float16),
    )
    codec_hash = "c" * 64
    (packed_root / "index.json").write_text(
        json.dumps(
            {
                "rave_checkpoint_sha256": codec_hash,
                "sample_rate": 44100,
                "latent_hop": 2048,
                "latent_dim": 16,
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            packed_root=str(packed_root),
            unified_manifest=str(unified_manifest),
            maximum_audio_seconds=5.0,
        ),
        rave=SimpleNamespace(
            sample_rate=44100,
            latent_hop=2048,
        ),
        model=SimpleNamespace(latent_dim=16),
    )
    monkeypatch.setattr(
        "midibrave.zrave_codec_qualification.ZraveFlowConfig.load",
        lambda _: config,
    )
    monkeypatch.setattr(
        "midibrave.zrave_codec_qualification._load_codec",
        lambda _config, _device: (
            _IdentityQualificationCodec(),
            codec_hash,
        ),
    )
    output = tmp_path / "qualification"

    manifest = render_codec_qualification(
        config_path,
        output,
        device="cpu",
    )

    assert manifest["codec_only"] is True
    assert manifest["transformer_checkpoint"] is None
    assert manifest["decoder_preroll_mode"] == "preceding_real_latent"
    assert manifest["decoder_preroll_frames"] == 32
    assert manifest["source_start_sample"] == 32 * 2048
    assert manifest["audible_frames"] == 75
    assert len(manifest["cases"]) == 5
    assert len(list((output / "wavs").glob("*.wav"))) == 10
    assert "No Transformer checkpoint" in (
        output / "README.txt"
    ).read_text(encoding="utf-8")
