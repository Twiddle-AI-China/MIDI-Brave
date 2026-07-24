from __future__ import annotations

import torch
from torch import nn

from midibrave.zrave_codec import (
    decode_with_seed,
    encode_posterior_mean,
    reset_streaming_state,
)


class _StatefulCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("pad", torch.ones(1))
        self.register_buffer("cache", torch.full((1,), 2.0))
        self.temperatures: list[float] = []

    def encode(
        self,
        audio: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.temperatures.append(float(temperature))
        value = audio[..., ::2] + self.pad
        self.pad.fill_(7.0)
        return value

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        value = torch.randn(
            latent.shape[0],
            1,
            latent.shape[-1] * 2,
        ) + self.pad
        self.pad.fill_(9.0)
        return value


def test_reset_streaming_state_zeros_all_pad_buffers() -> None:
    codec = nn.Sequential(_StatefulCodec(), _StatefulCodec())

    reset = reset_streaming_state(codec)

    assert reset == 4
    assert all(
        torch.count_nonzero(module.pad).item() == 0
        and torch.count_nonzero(module.cache).item() == 0
        for module in codec
    )


def test_posterior_mean_encoding_resets_state_and_uses_zero_temperature() -> None:
    codec = _StatefulCodec()
    audio = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)

    first = encode_posterior_mean(codec, audio)
    second = encode_posterior_mean(codec, audio)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first, audio[..., ::2])
    assert codec.temperatures == [0.0, 0.0]


def test_seeded_decode_resets_state_before_every_render() -> None:
    codec = _StatefulCodec()
    latent = torch.zeros(1, 16, 4)

    first = decode_with_seed(codec, latent, 20260724)
    second = decode_with_seed(codec, latent, 20260724)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
