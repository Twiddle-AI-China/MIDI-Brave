#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/../.." && pwd)

: "${LV_CLAP_IMAGE:=midibrave:lvzihao-cu128-clap-v1}"
: "${LV_CLAP_BASE_IMAGE:=midibrave:lvzihao-cu128-v1}"
command -v docker >/dev/null 2>&1 || {
  printf 'docker is required\n' >&2
  exit 2
}

base_versions=$(docker run --rm --network none --entrypoint python \
  "$LV_CLAP_BASE_IMAGE" -c '
import importlib.metadata
import torch
print("\t".join((
    torch.__version__,
    importlib.metadata.version("torchaudio"),
    importlib.metadata.version("torchvision"),
)))
')
IFS=$'\t' read -r detected_torch detected_torchaudio detected_torchvision <<<"$base_versions"
: "${LV_CLAP_EXPECTED_TORCH_PREFIX:=$detected_torch}"
: "${LV_CLAP_EXPECTED_TORCHAUDIO_PREFIX:=$detected_torchaudio}"
: "${LV_CLAP_EXPECTED_TORCHVISION_PREFIX:=$detected_torchvision}"
[[ -n "$LV_CLAP_EXPECTED_TORCH_PREFIX" && -n "$LV_CLAP_EXPECTED_TORCHAUDIO_PREFIX" && -n "$LV_CLAP_EXPECTED_TORCHVISION_PREFIX" ]] || {
  printf 'failed to audit CLAP base torch stack: %s\n' "$base_versions" >&2
  exit 2
}

docker build \
  --file "$code_root/Dockerfile.lvzihao-clap" \
  --build-arg "BASE_IMAGE=$LV_CLAP_BASE_IMAGE" \
  --build-arg "EXPECTED_TORCH_PREFIX=$LV_CLAP_EXPECTED_TORCH_PREFIX" \
  --build-arg "EXPECTED_TORCHAUDIO_PREFIX=$LV_CLAP_EXPECTED_TORCHAUDIO_PREFIX" \
  --build-arg "EXPECTED_TORCHVISION_PREFIX=$LV_CLAP_EXPECTED_TORCHVISION_PREFIX" \
  --tag "$LV_CLAP_IMAGE" \
  "$code_root"

docker image inspect \
  --format 'image={{.RepoTags}} id={{.Id}} size={{.Size}} created={{.Created}}' \
  "$LV_CLAP_IMAGE"
