from __future__ import annotations

import inspect
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
    encode = codec.encode
    schema = getattr(encode, "schema", None)
    if schema is not None:
        input_count = len(
            [argument for argument in schema.arguments if argument.name != "self"]
        )
    else:
        signature = inspect.signature(encode)
        input_count = len(
            [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
        )
    if input_count == 1:
        return encode(audio)
    if input_count == 2:
        return encode(audio, 0.0)
    raise TypeError(
        "RAVE encode must accept audio or audio plus temperature, "
        f"got {input_count} inputs"
    )


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
