from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def full_latent_fidelity_curve(
    latent_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a monotonic curve whose 0.999 export rank is full width."""
    if latent_dim < 2:
        raise ValueError("latent_dim must be at least 2")
    return torch.linspace(
        0.0,
        1.0,
        steps=latent_dim,
        dtype=dtype,
        device=device,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_full_latent_codec(
    *,
    source_checkpoint: Path,
    source_config: Path,
    output: Path,
    expected_source_sha256: str,
    expected_global_step: int,
    latent_dim: int,
    sample_rate: int,
    latent_hop: int,
) -> dict[str, Any]:
    """Export one exact RAVE checkpoint without post-hoc PCA cropping."""
    import cached_conv as cc
    import gin
    import rave
    import torch.nn as nn
    from scripts.export import VariationalScriptedRAVE

    source_hash = _sha256(source_checkpoint)
    if source_hash != expected_source_sha256.casefold():
        raise ValueError(
            "source checkpoint SHA-256 mismatch: "
            f"{source_hash} != {expected_source_sha256}"
        )
    cc.use_cached_conv(False)
    gin.clear_config()
    gin.parse_config_file(str(source_config))
    pretrained = rave.RAVE()
    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    if int(checkpoint.get("global_step", -1)) != expected_global_step:
        raise ValueError(
            "source checkpoint global_step mismatch: "
            f"{checkpoint.get('global_step')} != {expected_global_step}"
        )
    pretrained.load_state_dict(checkpoint["state_dict"], strict=False)
    pretrained.eval()
    if int(pretrained.latent_size) != latent_dim:
        raise ValueError(
            f"source latent size is {pretrained.latent_size}, "
            f"expected {latent_dim}"
        )
    for module in pretrained.modules():
        if hasattr(module, "weight_g"):
            nn.utils.remove_weight_norm(module)
    with torch.no_grad():
        pretrained.fidelity.copy_(
            full_latent_fidelity_curve(
                latent_dim,
                dtype=pretrained.fidelity.dtype,
                device=pretrained.fidelity.device,
            )
        )
    scripted = VariationalScriptedRAVE(
        pretrained=pretrained,
        fidelity=0.999,
    )
    if (
        int(scripted.latent_size) != latent_dim
        or int(scripted.full_latent_size) != latent_dim
    ):
        raise ValueError(
            "exported codec did not preserve the full latent width"
        )
    probe_frames = 8
    probe = torch.zeros(1, 1, probe_frames * latent_hop)
    encoded = scripted.encode(probe)
    decoded = scripted.decode(encoded)
    if tuple(encoded.shape) != (1, latent_dim, probe_frames):
        raise ValueError(f"unexpected encoded shape: {tuple(encoded.shape)}")
    if decoded.shape[-1] != probe.shape[-1]:
        raise ValueError("exported codec latent hop mismatch")
    if int(scripted.sr) != sample_rate:
        raise ValueError("exported codec sample-rate mismatch")
    if not torch.isfinite(encoded).all() or not torch.isfinite(decoded).all():
        raise ValueError("exported codec probe is non-finite")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output.parent / (output.name + ".exporting")
    temporary_root.mkdir(parents=True, exist_ok=False)
    temporary_output = temporary_root / output.name
    try:
        scripted.export_to_ts(str(temporary_output))
        os.replace(temporary_output, output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()
        temporary_root.rmdir()
    artifact_hash = _sha256(output)
    manifest: dict[str, Any] = {
        "schema": 1,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_hash,
        "source_global_step": int(checkpoint["global_step"]),
        "source_epoch": int(checkpoint["epoch"]),
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "latent_dim": latent_dim,
        "latent_space": "full_pca_posterior_mean",
        "sample_rate": sample_rate,
        "latent_hop": latent_hop,
        "export_fidelity_argument": 0.999,
        "exported_codec": str(output),
        "exported_codec_sha256": artifact_hash,
    }
    manifest_path = output.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an exact RAVE checkpoint at its full latent width."
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--latent-hop", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = export_full_latent_codec(
        source_checkpoint=args.source_checkpoint,
        source_config=args.source_config,
        output=args.output,
        expected_source_sha256=args.expected_source_sha256,
        expected_global_step=args.expected_global_step,
        latent_dim=args.latent_dim,
        sample_rate=args.sample_rate,
        latent_hop=args.latent_hop,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
