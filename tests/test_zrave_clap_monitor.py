from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.nn import functional as F

from midibrave import zrave_clap_monitor as clap_monitor


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/zrave/clap_audio_monitor_v1.json"
LVZIHAO = ROOT / "scripts/lvzihao"
OFFICIAL_CHECKPOINT_SHA256 = (
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
)
OFFICIAL_CHECKPOINT_BYTES = 2_352_471_003


class _FakeFrozenClapObjective:
    def __init__(self) -> None:
        self.calls = 0

    def _embedding(
        self,
        audio: torch.Tensor,
        valid_samples: torch.Tensor,
        role: str = "generated",
    ) -> torch.Tensor:
        assert role == "target"
        self.calls += 1
        index = torch.arange(512, device=audio.device, dtype=torch.float32)
        rows = []
        for batch_index in range(audio.shape[0]):
            length = int(valid_samples[batch_index].item())
            waveform = audio[batch_index, 0, :length]
            delta = waveform[1:] - waveform[:-1]
            features = torch.stack(
                (
                    waveform.mean(),
                    waveform.square().mean().sqrt(),
                    waveform.abs().mean(),
                    delta.abs().mean(),
                )
            )
            value = (
                torch.ones_like(index)
                + features[0] * torch.sin(index * 0.013)
                + features[1] * torch.cos(index * 0.017)
                + features[2] * torch.sin(index * 0.029)
                + features[3] * torch.cos(index * 0.037)
            )
            rows.append(value)
        return F.normalize(torch.stack(rows), dim=-1)


class _FailIfEmbedded:
    def _embedding(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("valid CLAP cache should avoid encoder calls")


class _RandomCropFakeObjective(_FakeFrozenClapObjective):
    def _embedding(
        self,
        audio: torch.Tensor,
        valid_samples: torch.Tensor,
        role: str = "generated",
    ) -> torch.Tensor:
        value = super()._embedding(audio, valid_samples, role)
        random_feature = (
            random.random() + float(np.random.random()) + float(torch.rand(()).item())
        )
        value = value.clone()
        value[:, 0] += random_feature
        return F.normalize(value, dim=-1)


def _tone(sample_rate: int, frequency: float, seconds: float = 0.4) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float64) / sample_rate
    envelope = np.minimum(time / 0.02, 1.0)
    return (0.3 * envelope * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _write(path: Path, audio: np.ndarray, sample_rate: int = 8000) -> str:
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path.name


def _audition_manifest(tmp_path: Path) -> Path:
    source = _write(tmp_path / "source.wav", _tone(8000, 220.0))
    direct = _write(
        tmp_path / "direct.wav",
        0.98 * _tone(8000, 224.0),
    )
    generated_a = _write(tmp_path / "generated-a.wav", _tone(8000, 310.0))
    generated_b = _write(tmp_path / "generated-b.wav", _tone(8000, 420.0))
    payload = {
        "schema": 1,
        "latent_dim": 128,
        "sample_rate": 8000,
        "triplets": [
            {
                "id": "pad-seed-17",
                "sample_id": "pad-a",
                "category": "Pad",
                "generation_seed": 17,
                "source": source,
                "direct": direct,
                "generated": generated_a,
            },
            {
                "id": "pad-seed-71",
                "sample_id": "pad-a",
                "category": "Pad",
                "generation_seed": 71,
                "source": source,
                "direct": direct,
                "generated": generated_b,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _official_sparse_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "music_audioset_epoch_15_esc_90.14.pt"
    with path.open("wb") as handle:
        handle.truncate(OFFICIAL_CHECKPOINT_BYTES)
    return path


def _patch_checkpoint_digest(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> None:
    original = clap_monitor.sha256_file

    def digest(path: str | Path) -> str:
        if Path(path).resolve() == checkpoint.resolve():
            return OFFICIAL_CHECKPOINT_SHA256
        return original(path)

    monkeypatch.setattr(clap_monitor, "sha256_file", digest)


def _runtime_dependencies() -> dict[str, str]:
    return {
        "laion-clap": "1.1.7",
        "transformers": "5.13.0",
        "torchlibrosa": "0.1.0",
        "librosa": "0.11.0",
        "ftfy": "6.3.1",
        "torch": torch.__version__,
        "torchaudio": "matching-test-double",
        "torchvision": "matching-test-double",
    }


def test_clap_monitor_config_freezes_authoritative_audio_contract() -> None:
    config = clap_monitor.ClapMonitorConfig.load(CONFIG)

    assert config.implementation == "laion-clap-1.1.7/HTSAT-base/no-fusion"
    assert config.transformers_version == "5.13.0"
    assert config.torchlibrosa_version == "0.1.0"
    assert config.librosa_version == "0.11.0"
    assert config.ftfy_version == "6.3.1"
    assert config.checkpoint_filename == "music_audioset_epoch_15_esc_90.14.pt"
    assert config.checkpoint_bytes == OFFICIAL_CHECKPOINT_BYTES
    assert config.checkpoint_sha256 == OFFICIAL_CHECKPOINT_SHA256
    assert config.clap_sample_rate == 48000
    assert config.clap_input_samples == 480000
    assert config.data_truncating == "rand_trunc"
    assert config.data_filling == "repeatpad"
    assert config.crop_seed == (
        "first_63_bits_of_source_audio_sha256/python_numpy_torch_cuda_scoped_restore"
    )
    assert config.import_time_hf_assets == [
        "bert-base-uncased-tokenizer",
        "roberta-base-tokenizer-and-model",
        "facebook/bart-base-tokenizer",
    ]
    assert config.embedding_dimension == 512
    assert config.batch_size == 1
    assert config.caption_evaluation is False


def test_clap_monitor_reports_audio_cosines_hashes_and_codec_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)
    objective = _FakeFrozenClapObjective()
    cache = tmp_path / "cache"

    report = clap_monitor.evaluate_clap_audition(
        manifest,
        checkpoint_path=checkpoint,
        config_path=CONFIG,
        cache_root=cache,
        device="cpu",
        expected_checkpoint_sha256=OFFICIAL_CHECKPOINT_SHA256,
        objective=objective,
        runtime_package_version="1.1.7",
        runtime_dependencies=_runtime_dependencies(),
    )

    assert objective.calls == 4
    assert report["report_only"] is True
    assert report["hard_thresholds"] is None
    assert report["claims"]["text_semantic_accuracy_evaluated"] is False
    assert report["claims"]["timbre_identity_ground_truth"] is False
    contract = report["clap_contract"]
    assert contract["checkpoint_sha256"] == OFFICIAL_CHECKPOINT_SHA256
    assert len(contract["config_sha256"]) == 64
    assert contract["text_embedding_used"] is False
    runtime_fingerprint = contract["runtime_fingerprint"]
    assert runtime_fingerprint["dependencies"] == _runtime_dependencies()
    assert len(runtime_fingerprint["monitor_code_sha256"]) == 64
    assert len(runtime_fingerprint["sha256"]) == 64
    assert (
        report["cache"]["runtime_fingerprint_sha256"] == (runtime_fingerprint["sha256"])
    )
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert set(row["audio_paths"]) == {"source", "direct", "generated"}
        assert set(row["audio_sha256"]) == {"source", "direct", "generated"}
        assert set(row["embedding_sha256"]) == {
            "source",
            "direct",
            "generated",
        }
        for name in (
            "generated_vs_source_audio_cosine",
            "generated_vs_direct_audio_cosine",
            "direct_vs_source_codec_ceiling_audio_cosine",
        ):
            assert -1.0 <= row[name] <= 1.0
    summary = report["summary"]
    for name in (
        "generated_vs_source_audio_cosine",
        "generated_vs_direct_audio_cosine",
    ):
        assert set(summary[name]) == {
            "count",
            "p10",
            "median",
            "p90",
            "worst",
            "worst_id",
        }
        assert summary[name]["count"] == 2
    assert summary["direct_vs_source_codec_ceiling_audio_cosine"]["count"] == 1
    assert report["cache"]["writes"] == 4
    assert report["cache"]["hits"] == 0
    assert len(report["cache"]["used_embedding_collection_sha256"]) == 64
    assert report["runtime"] == {
        "device": "cpu",
        "embedding_batch_size": 1,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_allocated_gib": 0.0,
    }
    json.dumps(report, allow_nan=False)


def test_clap_monitor_reuses_complete_hash_bound_embedding_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)
    cache = tmp_path / "cache"
    kwargs = {
        "checkpoint_path": checkpoint,
        "config_path": CONFIG,
        "cache_root": cache,
        "device": "cpu",
        "expected_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "runtime_package_version": "1.1.7",
        "runtime_dependencies": _runtime_dependencies(),
    }
    clap_monitor.evaluate_clap_audition(
        manifest,
        objective=_FakeFrozenClapObjective(),
        **kwargs,
    )

    report = clap_monitor.evaluate_clap_audition(
        manifest,
        objective=_FailIfEmbedded(),
        **kwargs,
    )

    assert report["cache"]["hits"] == 4
    assert report["cache"]["writes"] == 0
    metadata = list((cache / "clap").glob("*.json"))
    assert len(metadata) == 4
    for path in metadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["clap_checkpoint_sha256"] == OFFICIAL_CHECKPOINT_SHA256
        assert len(payload["source_audio_sha256"]) == 64
        assert len(payload["embedding_sha256"]) == 64
        assert isinstance(payload["preprocessing_crop_seed"], int)
        assert payload["preprocessing_rngs"] == [
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        ]
        assert payload["preprocessing_rng_scope"] == "save_seed_restore"
        fingerprint = payload["runtime_fingerprint"]
        assert fingerprint["dependencies"] == _runtime_dependencies()
        assert len(fingerprint["monitor_code_sha256"]) == 64
        assert len(fingerprint["sha256"]) == 64


def test_clap_cache_misses_when_runtime_dependency_fingerprint_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)
    cache = tmp_path / "cache"
    common = {
        "checkpoint_path": checkpoint,
        "config_path": CONFIG,
        "cache_root": cache,
        "device": "cpu",
        "expected_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "runtime_package_version": "1.1.7",
    }
    first_dependencies = _runtime_dependencies()
    clap_monitor.evaluate_clap_audition(
        manifest,
        objective=_FakeFrozenClapObjective(),
        runtime_dependencies=first_dependencies,
        **common,
    )

    changed_dependencies = {**first_dependencies, "torch": "different-torch"}
    objective = _FakeFrozenClapObjective()
    report = clap_monitor.evaluate_clap_audition(
        manifest,
        objective=objective,
        runtime_dependencies=changed_dependencies,
        **common,
    )

    assert objective.calls == 4
    assert report["cache"]["hits"] == 0
    assert report["cache"]["writes"] == 4
    assert (
        report["clap_contract"]["runtime_fingerprint"]["dependencies"]["torch"]
        == "different-torch"
    )


def test_clap_shell_reuse_validator_rehashes_current_wav_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)
    report = clap_monitor.evaluate_clap_audition(
        manifest,
        checkpoint_path=checkpoint,
        config_path=CONFIG,
        cache_root=tmp_path / "cache",
        device="cpu",
        expected_checkpoint_sha256=OFFICIAL_CHECKPOINT_SHA256,
        objective=_FakeFrozenClapObjective(),
        runtime_package_version="1.1.7",
        runtime_dependencies=_runtime_dependencies(),
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    shell = (LVZIHAO / "clap_monitor.sh").read_text(encoding="utf-8")
    validator = shell.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    arguments = (
        str(report_path),
        str(manifest),
        clap_monitor.sha256_file(manifest),
        OFFICIAL_CHECKPOINT_SHA256,
        clap_monitor.sha256_file(CONFIG),
    )

    valid = subprocess.run(
        [sys.executable, "-", *arguments],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr

    _write(tmp_path / "generated-a.wav", _tone(8000, 880.0))
    stale = subprocess.run(
        [sys.executable, "-", *arguments],
        input=validator,
        text=True,
        capture_output=True,
        check=False,
    )
    assert stale.returncode == 1, stale.stderr


def test_clap_monitor_audio_hash_seed_makes_random_crop_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)
    common = {
        "checkpoint_path": checkpoint,
        "config_path": CONFIG,
        "device": "cpu",
        "expected_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "runtime_package_version": "1.1.7",
        "runtime_dependencies": _runtime_dependencies(),
    }

    random.seed(1001)
    np.random.seed(1002)
    torch.manual_seed(1003)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    first = clap_monitor.evaluate_clap_audition(
        manifest,
        cache_root=tmp_path / "first-cache",
        objective=_RandomCropFakeObjective(),
        **common,
    )
    assert random.getstate() == python_state
    observed_numpy_state = np.random.get_state()
    assert observed_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy_state[1], numpy_state[1])
    assert observed_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_states is not None:
        assert all(
            torch.equal(observed, expected)
            for observed, expected in zip(
                torch.cuda.get_rng_state_all(),
                cuda_states,
                strict=True,
            )
        )

    reversed_payload = json.loads(manifest.read_text(encoding="utf-8"))
    reversed_payload["triplets"].reverse()
    reversed_manifest = tmp_path / "manifest-reversed.json"
    reversed_manifest.write_text(json.dumps(reversed_payload), encoding="utf-8")
    random.seed(7001)
    np.random.seed(7002)
    torch.manual_seed(7003)
    second_python_state = random.getstate()
    second_numpy_state = np.random.get_state()
    second_torch_state = torch.random.get_rng_state().clone()
    second_cuda_states = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    second = clap_monitor.evaluate_clap_audition(
        reversed_manifest,
        cache_root=tmp_path / "second-cache",
        objective=_RandomCropFakeObjective(),
        **common,
    )
    assert random.getstate() == second_python_state
    observed_numpy_state = np.random.get_state()
    assert observed_numpy_state[0] == second_numpy_state[0]
    np.testing.assert_array_equal(observed_numpy_state[1], second_numpy_state[1])
    assert observed_numpy_state[2:] == second_numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), second_torch_state)
    if second_cuda_states is not None:
        assert all(
            torch.equal(observed, expected)
            for observed, expected in zip(
                torch.cuda.get_rng_state_all(),
                second_cuda_states,
                strict=True,
            )
        )

    def embedding_map(report: dict[str, object]) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in report["rows"]:  # type: ignore[index]
            for role, audio_sha256 in row["audio_sha256"].items():
                result[audio_sha256] = row["embedding_sha256"][role]
        return result

    assert embedding_map(first) == embedding_map(second)
    assert first["summary"] == second["summary"]


def test_clap_monitor_rejects_wrong_checkpoint_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _audition_manifest(tmp_path)
    checkpoint = _official_sparse_checkpoint(tmp_path)
    _patch_checkpoint_digest(monkeypatch, checkpoint)

    with pytest.raises(ValueError, match="caller CLAP checkpoint SHA-256"):
        clap_monitor.evaluate_clap_audition(
            manifest,
            checkpoint_path=checkpoint,
            config_path=CONFIG,
            cache_root=tmp_path / "cache",
            device="cpu",
            expected_checkpoint_sha256="0" * 64,
            objective=_FakeFrozenClapObjective(),
            runtime_dependencies=_runtime_dependencies(),
        )


def test_clap_action_is_optional_and_never_added_to_training_queues() -> None:
    runner = (LVZIHAO / "allocation_runner.sh").read_text(encoding="utf-8")
    script = (LVZIHAO / "clap_monitor.sh").read_text(encoding="utf-8")
    assert "clap_report)" in runner
    assert '"$script_dir/clap_monitor.sh" "$experiment_id" "$spec"' in runner
    assert OFFICIAL_CHECKPOINT_SHA256 in script
    assert "python -m midibrave.zrave_clap_monitor" in script
    assert "import transformers" in script
    assert 'BertTokenizer.from_pretrained("bert-base-uncased"' in script
    assert 'RobertaModel.from_pretrained("roberta-base"' in script
    assert 'BartTokenizer.from_pretrained("facebook/bart-base"' in script
    assert "local_files_only=True" in script
    assert 'row.get("audio_paths")' in script
    assert "digest(resolved) != expected" in script
    dockerfile = (ROOT / "Dockerfile.lvzihao-clap").read_text(encoding="utf-8")
    assert "laion-clap==1.1.7" in dockerfile
    assert "transformers==5.13.0" in dockerfile
    assert "torchlibrosa==0.1.0" in dockerfile
    assert "import torchaudio" in dockerfile
    assert "import torchvision" in dockerfile
    for queue in (
        LVZIHAO / "tiny_categories_v1.queue.tsv",
        LVZIHAO / "tiny_latent_proxies_v1.queue.tsv",
    ):
        assert "\tclap_report\t" not in queue.read_text(encoding="utf-8")
