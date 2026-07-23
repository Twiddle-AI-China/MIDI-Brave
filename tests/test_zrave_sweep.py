from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "cloud"
    / "select_zrave_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("select_zrave_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
select_candidate = MODULE.select_candidate


def _candidate(
    batch: int,
    throughput: float,
    *,
    status: str = "ok",
    nonfinite: int = 0,
    peak: float = 12000.0,
) -> dict[str, object]:
    return {
        "batch_per_gpu": batch,
        "world_size": 8,
        "global_batch": batch * 8,
        "measured_updates": 100,
        "median_windows_per_second": throughput,
        "p10_windows_per_second": throughput * 0.9,
        "peak_memory_mib": peak,
        "total_memory_mib": 32768.0,
        "nonfinite_updates": nonfinite,
        "status": status,
    }


def test_sweep_uses_two_percent_smaller_batch_tie_rule() -> None:
    winner = select_candidate(
        [
            _candidate(128, 10000.0),
            _candidate(256, 10150.0),
            _candidate(384, 9000.0),
            _candidate(512, 8000.0),
        ]
    )

    assert winner["batch_per_gpu"] == 128


def test_sweep_rejects_oom_nonfinite_and_overcommitted_candidates() -> None:
    winner = select_candidate(
        [
            _candidate(128, 5000.0, status="failed"),
            _candidate(256, 6000.0, nonfinite=1),
            _candidate(384, 7000.0, peak=33000.0),
            _candidate(512, 4000.0),
        ]
    )

    assert winner["batch_per_gpu"] == 512


def test_sweep_fails_when_no_candidate_is_valid() -> None:
    with pytest.raises(ValueError, match="no valid"):
        select_candidate(
            [
                _candidate(batch, 0.0, status="failed")
                for batch in (128, 256, 384, 512)
            ]
        )
