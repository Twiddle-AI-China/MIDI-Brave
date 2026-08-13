#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/../.." && pwd)

: "${LV_AUDIO_EVAL_IMAGE:=midibrave:lvzihao-cu128-audio-eval-v1}"
: "${LV_AUDIO_EVAL_BASE_IMAGE:=midibrave:lvzihao-cu128-v1}"
command -v docker >/dev/null 2>&1 || {
  printf 'docker is required\n' >&2
  exit 2
}

base_versions=$(docker run --rm --network none --entrypoint python \
  "$LV_AUDIO_EVAL_BASE_IMAGE" -c '
import importlib.metadata
import torch
print("\t".join((
    torch.__version__,
    importlib.metadata.version("torchaudio"),
)))
')
IFS=$'\t' read -r detected_torch detected_torchaudio <<<"$base_versions"
: "${LV_AUDIO_EVAL_EXPECTED_TORCH_PREFIX:=$detected_torch}"
: "${LV_AUDIO_EVAL_EXPECTED_TORCHAUDIO_PREFIX:=$detected_torchaudio}"
[[ -n "$LV_AUDIO_EVAL_EXPECTED_TORCH_PREFIX" && -n "$LV_AUDIO_EVAL_EXPECTED_TORCHAUDIO_PREFIX" ]] || {
  printf 'failed to audit audio-eval base torch stack: %s\n' "$base_versions" >&2
  exit 2
}

docker build \
  --file "$code_root/Dockerfile.lvzihao-audio-eval" \
  --build-arg "BASE_IMAGE=$LV_AUDIO_EVAL_BASE_IMAGE" \
  --build-arg "EXPECTED_TORCH_PREFIX=$LV_AUDIO_EVAL_EXPECTED_TORCH_PREFIX" \
  --build-arg "EXPECTED_TORCHAUDIO_PREFIX=$LV_AUDIO_EVAL_EXPECTED_TORCHAUDIO_PREFIX" \
  --tag "$LV_AUDIO_EVAL_IMAGE" \
  "$code_root"

docker image inspect \
  --format 'image={{.RepoTags}} id={{.Id}} size={{.Size}} created={{.Created}}' \
  "$LV_AUDIO_EVAL_IMAGE"
