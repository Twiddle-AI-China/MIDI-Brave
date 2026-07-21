from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_digest(state: dict[str, torch.Tensor], root: str) -> str:
    digest = hashlib.sha256()
    matched = 0
    for name in sorted(state):
        if name == root or name.startswith(root + "."):
            value = state[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
            matched += 1
    if not matched:
        raise ValueError(f"checkpoint has no {root} tensors")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    args = parser.parse_args()

    source = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(trained.get("format", 0)) != 5:
        raise ValueError("Phase 2 checkpoint must use format 5")
    contract = trained.get("predictive_contract", {})
    if contract.get("stage") != "predictor" or not contract.get("encoder_frozen"):
        raise ValueError("Phase 2 checkpoint contract is not predictor-only")
    if int(trained.get("stage_update", -1)) != args.expected_update:
        raise ValueError("Phase 2 checkpoint update does not match the smoke target")
    source_state = source["model"]
    trained_state = trained["model"]
    if set(source_state) != set(trained_state):
        raise ValueError("warm start and Phase 2 model states have different keys")
    if any(not torch.isfinite(value).all() for value in trained_state.values()):
        raise ValueError("Phase 2 checkpoint contains a non-finite model tensor")

    frozen_roots = (
        "encoder", "clap_projection", "midi", "decoder",
        "rave_pitch_adversary", "excitation",
    )
    frozen = {}
    for root in frozen_roots:
        before = module_digest(source_state, root)
        after = module_digest(trained_state, root)
        if before != after:
            raise ValueError(f"frozen module changed during Phase 2 smoke: {root}")
        frozen[root] = after
    predictor_changed = any(
        not torch.equal(source_state[name], value)
        for name, value in trained_state.items()
        if name == "predictor" or name.startswith("predictor.")
    )
    if not predictor_changed:
        raise ValueError("predictor did not change during Phase 2 smoke")

    with np.load(args.statistics, allow_pickle=False) as statistics:
        cache_checkpoint_hash = str(statistics["checkpoint_hash"].item())
    if cache_checkpoint_hash != sha256(args.warm_start):
        raise ValueError("RAVE cache is not bound to the Phase 1.5 warm start")
    print(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "frozen_module_sha256": frozen,
        "predictor_changed": predictor_changed,
        "stage_update": int(trained["stage_update"]),
        "warm_start_sha256": cache_checkpoint_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
