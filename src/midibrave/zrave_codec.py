from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def reset_streaming_state(codec: Any) -> int:
    reset = 0
    with torch.no_grad():
        for _, module in codec.named_modules():
            for state_name in ("pad", "cache"):
                try:
                    state = getattr(module, state_name)
                except (AttributeError, RuntimeError):
                    continue
                if isinstance(state, Tensor):
                    state.zero_()
                    reset += 1
    return reset


def encode_posterior_mean(codec: Any, audio: Tensor) -> Tensor:
    reset_streaming_state(codec)
    return codec.encode(audio, 0.0)


def decode_with_seed(
    codec: Any,
    latent: Tensor,
    random_seed: int,
) -> Tensor:
    reset_streaming_state(codec)
    torch.manual_seed(random_seed)
    if latent.is_cuda:
        torch.cuda.manual_seed_all(random_seed)
    return codec.decode(latent)
