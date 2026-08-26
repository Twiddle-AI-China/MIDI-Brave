from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from midibrave.atlas_flow_config import load_atlas_flow_config
from midibrave.atlas_flow_loss import atlas_flow_matching_loss
from midibrave.atlas_flow_model import AtlasFlowSystem, FlowBatch


def norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config = load_atlas_flow_config(args.config)
    device = torch.device("cuda:0")
    system = AtlasFlowSystem(config.model, config.data).to(device)
    features = torch.randn(1, 96, 432, device=device)
    note = torch.tensor([50], device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        trajectory = system.instrument.encode(features)
        waveform, _ = system.instrument.decode(
            trajectory, note, 8_192, window_start=torch.tensor([4_410], device=device)
        )
        audio_loss = waveform.float().square().mean() + trajectory.float().square().mean()
    audio_loss.backward()
    audio_gradients = {
        "encoder": norm(system.instrument.encoder),
        "adapter": norm(system.instrument.adapter),
        "decoder": norm(system.instrument.decoder),
        "pitch_conditioner": norm(system.instrument.pitch_conditioner),
        "output_gain": norm(system.instrument.output_gain),
    }
    system.zero_grad(set_to_none=True)
    batch = 2
    flow_batch = FlowBatch(
        history=torch.randn(batch, 32, 128, device=device),
        history_anchor=torch.randn(batch, 32, 128, device=device),
        target=torch.randn(batch, 64, 128, device=device),
        anchor_path=torch.randn(batch, 64, 128, device=device),
        atlas_path=torch.randn(batch, 64, 8, device=device),
        lifecycle=torch.randn(batch, 64, 4, device=device),
        history_mask=torch.ones(batch, 32, dtype=torch.bool, device=device),
    )
    with torch.autocast("cuda", dtype=torch.float16):
        flow_result = atlas_flow_matching_loss(system.flow, flow_batch, config.loss)
    flow_result.total.backward()
    flow_gradient = norm(system.flow)
    passed = (
        torch.isfinite(waveform).all()
        and torch.isfinite(flow_result.total)
        and all(value > 0 for value in audio_gradients.values())
        and flow_gradient > 0
    )
    report = {
        "schema": "midibrave.atlas-flow.gpu-smoke.v1",
        "passed": bool(passed),
        "waveform_shape": list(waveform.shape),
        "trajectory_shape": list(trajectory.shape),
        "audio_gradients": audio_gradients,
        "flow_gradient": flow_gradient,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "device": torch.cuda.get_device_name(device),
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
