from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.build_zrave_blind_audition import build_blind_audition


def _write_matching_auditions(root: Path) -> tuple[Path, Path]:
    sample_rate = 44_100
    manifests: list[Path] = []
    for model_index, model_name in enumerate(("old-model", "new-model")):
        model_root = root / model_name
        audio_root = model_root / "audio"
        audio_root.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for row_index, row_id in enumerate(("one", "two")):
            phase = np.linspace(
                0.0,
                8.0 * np.pi,
                4096,
                endpoint=False,
                dtype=np.float32,
            )
            direct_audio = np.sin(phase + row_index).astype(np.float32)
            predicted_audio = np.sin(
                phase + row_index + model_index * 0.25
            ).astype(np.float32)
            direct = audio_root / f"{row_id}-direct.wav"
            predicted = audio_root / f"{row_id}-predicted.wav"
            sf.write(direct, direct_audio, sample_rate, subtype="PCM_16")
            sf.write(
                predicted,
                predicted_audio,
                sample_rate,
                subtype="PCM_16",
            )
            rows.append(
                {
                    "id": row_id,
                    "category": "Pad",
                    "split": "test",
                    "sample_count": direct_audio.size,
                    "shared_gain": 1.0,
                    "audio": {
                        "direct": f"audio/{direct.name}",
                        "predicted": f"audio/{predicted.name}",
                    },
                }
            )
        manifest = model_root / "audition-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "sample_rate": sample_rate,
                    "latent_hop": 2048,
                    "future_frames": 2,
                    "comparisons": rows,
                    "transformer": {
                        "path": f"/secret/{model_name}.pt",
                        "sha256": model_name,
                        "update": model_index,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)
    return manifests[0], manifests[1]


def test_blind_builder_is_deterministic_and_hides_identity(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_matching_auditions(tmp_path)

    first = build_blind_audition(
        baseline,
        candidate,
        tmp_path / "first",
        seed=20260724,
    )
    second = build_blind_audition(
        baseline,
        candidate,
        tmp_path / "second",
        seed=20260724,
    )

    assert first["rows"] == second["rows"]
    assignments = {
        row["assignment"]["A"] for row in first["answer_key"]["rows"]
    }
    assert assignments == {"baseline", "candidate"}
    public = json.dumps(first["public_manifest"]).casefold()
    assert "baseline" not in public
    assert "candidate" not in public
    assert "old-model" not in public
    assert "new-model" not in public
    for row in first["public_manifest"]["rows"]:
        for relative in row["audio"].values():
            assert (tmp_path / "first" / relative).is_file()


def test_blind_page_reveals_answer_only_after_explicit_click(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_matching_auditions(tmp_path)

    build_blind_audition(
        baseline,
        candidate,
        tmp_path / "output",
        seed=17,
    )

    html = (tmp_path / "output" / "index.html").read_text(encoding="utf-8")
    assert "blind-manifest.json" in html
    assert "blind-answer-key.json" in html
    assert 'id="revealButton"' in html
    assert 'addEventListener("click", revealAnswers)' in html
    assert "switchVariant" in html
    assert "currentTime" in html
    assert "<script src=" not in html
    assert "<link rel=" not in html


def test_blind_builder_rebalances_model_dependent_render_gains(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_matching_auditions(tmp_path)
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_row = candidate_payload["comparisons"][0]
    candidate_row["shared_gain"] = 0.5
    for variant in ("direct", "predicted"):
        path = candidate.parent / candidate_row["audio"][variant]
        audio, sample_rate = sf.read(path, dtype="float32")
        sf.write(path, audio * 0.5, sample_rate, subtype="PCM_16")
    candidate.write_text(
        json.dumps(candidate_payload),
        encoding="utf-8",
    )

    result = build_blind_audition(
        baseline,
        candidate,
        tmp_path / "output",
        seed=20260724,
    )

    public_row = result["public_manifest"]["rows"][0]
    private_row = result["answer_key"]["rows"][0]
    output_gain = public_row["shared_gain"]
    for label in ("A", "B"):
        role = private_row["assignment"][label]
        source_manifest = baseline if role == "baseline" else candidate
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_row = source_payload["comparisons"][0]
        source_audio, _ = sf.read(
            source_manifest.parent / source_row["audio"]["predicted"],
            dtype="float32",
        )
        output_audio, _ = sf.read(
            tmp_path / "output" / public_row["audio"][label],
            dtype="float32",
        )
        np.testing.assert_allclose(
            output_audio / output_gain,
            source_audio / source_row["shared_gain"],
            atol=1.5e-4,
        )


def test_blind_builder_rejects_mismatched_comparisons(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_matching_auditions(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["comparisons"][0]["id"] = "different"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="comparison IDs"):
        build_blind_audition(
            baseline,
            candidate,
            tmp_path / "output",
            seed=17,
        )
