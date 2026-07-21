from pathlib import Path

import numpy as np

from scripts.verify_phase2_predictor import validate_statistics_binding


def test_predictor_restart_binds_cache_by_statistics_hash(tmp_path: Path):
    warm_start = tmp_path / "predictor.pt"
    warm_start.write_bytes(b"predictor-checkpoint")
    statistics = tmp_path / "rave-statistics.npz"
    np.savez(statistics, checkpoint_hash=np.asarray("phase1-rave-hash"))

    import hashlib
    statistics_hash = hashlib.sha256(statistics.read_bytes()).hexdigest()
    source = {
        "predictive_contract": {
            "stage": "predictor",
            "latent_statistics_hash": statistics_hash,
        }
    }

    binding = validate_statistics_binding(source, warm_start, statistics)

    assert binding["source_stage"] == "predictor"
    assert binding["statistics_sha256"] == statistics_hash
    assert binding["cache_checkpoint_sha256"] == "phase1-rave-hash"
