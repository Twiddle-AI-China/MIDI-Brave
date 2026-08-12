#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/../.." && pwd)

: "${LV_IMAGE:=midibrave:lvzihao-cu128-v1}"
: "${LV_BASE_IMAGE:=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385}"

command -v docker >/dev/null 2>&1 || {
  printf 'docker is required\n' >&2
  exit 2
}

docker build \
  --file "$code_root/Dockerfile.lvzihao" \
  --build-arg "BASE_IMAGE=$LV_BASE_IMAGE" \
  --tag "$LV_IMAGE" \
  "$code_root"

docker image inspect \
  --format 'image={{.RepoTags}} id={{.Id}} created={{.Created}}' \
  "$LV_IMAGE"
