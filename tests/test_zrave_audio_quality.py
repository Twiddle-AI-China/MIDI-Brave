from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from midibrave.zrave_audio_quality import (
    audio_pair_metrics,
    evaluate_audition_audio,
)


def _sine(samples: int, sample_rate: int, frequency: float) -> np.ndarray:
    phase = np.arange(samples, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * phase).astype(np.float32)


def _write_three_pair_fixture(root: Path) -> Path:
    audio_root = root / "audio"
    audio_root.mkdir()
    sample_rate = 44100
    latent_hop = 2048
    future_frames = 54
    samples = future_frames * latent_hop
    comparisons = []
    for index, tail_gain in enumerate((1.0, 0.5, 0.1), 1):
        direct = _sine(samples, sample_rate, 110.0 * index)
        predicted = direct.copy()
        predicted[-8 * latent_hop :] *= tail_gain
        direct_path = audio_root / f"{index}-direct.wav"
        predicted_path = audio_root / f"{index}-predicted.wav"
        sf.write(direct_path, direct, sample_rate, subtype="PCM_16")
        sf.write(predicted_path, predicted, sample_rate, subtype="PCM_16")
        comparisons.append(
            {
                "id": str(index),
                "category": "Pad",
                "split": "test",
                "audio": {
                    "direct": f"audio/{direct_path.name}",
                    "predicted": f"audio/{predicted_path.name}",
                },
            }
        )
    manifest = root / "audition-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "latent_hop": latent_hop,
                "future_frames": future_frames,
                "comparisons": comparisons,
                "transformer": {"update": 17},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_audio_pair_metrics_detects_tail_washout() -> None:
    sample_rate = 44100
    latent_hop = 2048
    samples = 54 * latent_hop
    direct = _sine(samples, sample_rate, 220.0)
    washed = direct.copy()
    washed[-8 * latent_hop :] *= 0.1

    identical = audio_pair_metrics(
        direct,
        direct,
        sample_rate,
        latent_hop=latent_hop,
        chunk_frames=8,
    )
    degraded = audio_pair_metrics(
        direct,
        washed,
        sample_rate,
        latent_hop=latent_hop,
        chunk_frames=8,
    )

    assert identical["spectral_cosine"] == pytest.approx(1.0, abs=1.0e-6)
    assert identical["tail_rms_drift_db"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )
    assert degraded["spectral_cosine"] < identical["spectral_cosine"]
    assert degraded["tail_rms_drift_db"] == pytest.approx(-20.0, abs=0.1)
    assert len(degraded["chunks"]) == 7


def test_audition_audio_report_has_robust_aggregates(
    tmp_path: Path,
) -> None:
    manifest = _write_three_pair_fixture(tmp_path)

    report = evaluate_audition_audio(manifest)

    summary = report["summary"]["spectral_cosine"]
    assert set(summary) == {
        "median",
        "p10",
        "p90",
        "worst_decile_mean",
        "minimum",
        "maximum",
    }
    assert report["pairs"] == 3
    assert report["finite"] is True
    assert report["checkpoint_update"] == 17
    assert len(report["rows"]) == 3
