#!/usr/bin/env bash

set -euo pipefail

LV_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${LV_IMAGE:=midibrave:lvzihao-cu128-v1}"
: "${HOST_PROJECT_ROOT:=/home/twiddle/Developer/Latent-Cosmos-Synth-rave-midi-integration}"
: "${HOST_REPO_ROOT:=$HOST_PROJECT_ROOT/repo}"
: "${HOST_FLOW_ROOT:=$HOST_PROJECT_ROOT/runtime/serum128}"
: "${LV_PROJECT_ROOT:=$HOST_REPO_ROOT}"
: "${LV_CODE_ROOT:=$LV_PROJECT_ROOT/MidiBrave-v2}"
: "${LV_WORK_ROOT:=$HOST_FLOW_ROOT}"
: "${LV_STATE_ROOT:=$LV_WORK_ROOT/lvzihao-state}"
: "${LV_CONTAINER_PROJECT_ROOT:=/workspace/Latent-Cosmos-Synth}"
: "${LV_CONTAINER_CODE_ROOT:=$LV_CONTAINER_PROJECT_ROOT/MidiBrave-v2}"
: "${LV_CONTAINER_WORK_ROOT:=/data/midibrave-zrave-flow-serum128}"
: "${LV_CONTAINER_RAVE_ROOT:=/data/midibrave-rave-serum-v2}"
: "${LV_CONTAINER_SOURCE_ROOT:=/source}"
: "${LV_PACK_RELATIVE:=packs/serum-balanced}"
: "${LV_SHM_SIZE:=8g}"
: "${LV_DOCKER_CPUS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${LV_DOCKER_MEMORY:=30g}"
: "${LV_DOCKER_NETWORK:=none}"
: "${LV_SHUTDOWN_MARGIN_SECONDS:=900}"

export HOST_PROJECT_ROOT HOST_REPO_ROOT HOST_FLOW_ROOT
export LV_SCRIPT_DIR LV_CODE_ROOT LV_PROJECT_ROOT LV_IMAGE
export LV_WORK_ROOT LV_STATE_ROOT LV_CONTAINER_PROJECT_ROOT
export LV_CONTAINER_CODE_ROOT LV_CONTAINER_WORK_ROOT
export LV_CONTAINER_RAVE_ROOT LV_CONTAINER_SOURCE_ROOT
export LV_PACK_RELATIVE LV_SHM_SIZE LV_DOCKER_CPUS LV_DOCKER_MEMORY
export LV_DOCKER_NETWORK LV_SHUTDOWN_MARGIN_SECONDS

die() {
  printf 'lvzihao: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_directory() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

assert_persistent_work_root() {
  [[ "$HOST_PROJECT_ROOT" == /home/twiddle/Developer/* ]] ||
    die "HOST_PROJECT_ROOT must be below /home/twiddle/Developer"
  [[ "$LV_PROJECT_ROOT" == "$HOST_PROJECT_ROOT"/* ]] ||
    die "HOST_REPO_ROOT must be below HOST_PROJECT_ROOT"
  [[ "$LV_WORK_ROOT" == "$HOST_PROJECT_ROOT"/* ]] ||
    die "HOST_FLOW_ROOT must be below HOST_PROJECT_ROOT"
  [[ "$LV_STATE_ROOT" == "$LV_WORK_ROOT"/* ]] ||
    die "LV_STATE_ROOT must be below LV_WORK_ROOT"
}

assert_clean_checkout() {
  require_file "$LV_CODE_ROOT/pyproject.toml"
  require_command git
  git -C "$LV_PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null
  if [[ -n "$(git -C "$LV_PROJECT_ROOT" status --porcelain=v1)" ]]; then
    die "the training checkout must be clean; commit or remove local changes"
  fi
}

assert_single_gpu_allocation() {
  local allocated_gpus=${SLURM_GPUS_ON_NODE:-1}
  [[ -n "${SLURM_JOB_ID:-}" ]] ||
    die "GPU work must run inside a qgpu/SLURM allocation"
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] ||
    die "CUDA_VISIBLE_DEVICES is missing from the allocation"
  [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] ||
    die "exactly one GPU is required, got: $CUDA_VISIBLE_DEVICES"
  [[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ ]] ||
    die "unsupported CUDA_VISIBLE_DEVICES value: $CUDA_VISIBLE_DEVICES"
  if [[ "$allocated_gpus" =~ ^[0-9]+$ ]]; then
    (( allocated_gpus == 1 )) ||
      die "exactly one SLURM GPU is required"
  fi
}

validate_relative_path() {
  local value=$1
  [[ -n "$value" && "$value" != /* ]] ||
    die "expected a non-empty relative path: $value"
  [[ "/$value/" != *'/../'* && "/$value/" != *'/./'* ]] ||
    die "relative path traversal is not allowed: $value"
}

verify_initializer_environment() {
  local initializer_path=$1 actual_sha expected_sha
  require_file "$initializer_path"
  require_command sha256sum
  actual_sha=$(sha256sum "$initializer_path" | awk '{print $1}')
  expected_sha=${LV_EXPECTED_INITIALIZER_SHA256:-}
  if [[ -n "$expected_sha" ]]; then
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] ||
      die "LV_EXPECTED_INITIALIZER_SHA256 must be a lowercase SHA-256"
    [[ "$actual_sha" == "$expected_sha" ]] ||
      die "initializer SHA-256 changed: $initializer_path"
  fi
  INITIALIZER_SHA256=$actual_sha
}

resolve_persistent_initializer() {
  local relative=$1 candidate resolved expected work_root_real source_update
  validate_relative_path "$relative"
  [[ "$relative" =~ ^runs/[A-Za-z0-9._/-]+/checkpoints/step-[0-9]{6}\.pt$ ]] ||
    die "initializer must be a runs/.../checkpoints/step-NNNNNN.pt path"
  [[ "$relative" != *//* ]] || die "initializer path is malformed"
  require_command realpath
  candidate="$LV_WORK_ROOT/$relative"
  require_file "$candidate"
  work_root_real=$(realpath "$LV_WORK_ROOT")
  resolved=$(realpath "$candidate")
  expected="$work_root_real/$relative"
  [[ "$resolved" == "$work_root_real"/runs/* ]] ||
    die "initializer resolves outside LV_WORK_ROOT/runs: $relative"
  [[ "$resolved" == "$expected" ]] ||
    die "initializer path must not traverse symlinks: $relative"
  verify_initializer_environment "$resolved"
  source_update=$(basename -- "$resolved")
  source_update=${source_update#step-}
  source_update=${source_update%.pt}
  [[ "$source_update" =~ ^[0-9]{6}$ ]] ||
    die "resolved initializer basename is not step-NNNNNN.pt"
  source_update=$((10#$source_update))
  (( source_update > 0 )) || die "initializer update must be positive"
  RESOLVED_INITIALIZER_PATH=$resolved
  RESOLVED_INITIALIZER_SHA256=$INITIALIZER_SHA256
  RESOLVED_INITIALIZER_UPDATE=$source_update
}

host_config_path() {
  local relative=$1
  validate_relative_path "$relative"
  printf '%s/%s\n' "$LV_CODE_ROOT" "$relative"
}

container_config_path() {
  local relative=$1
  validate_relative_path "$relative"
  printf '%s/%s\n' "$LV_CONTAINER_CODE_ROOT" "$relative"
}

assert_config_contract() {
  local config_relative=$1
  local run_relative=$2
  local config output expected packed expected_pack
  validate_relative_path "$run_relative"
  validate_relative_path "$LV_PACK_RELATIVE"
  config=$(host_config_path "$config_relative")
  require_file "$config"
  output=$(awk '$1 == "output_root:" {print $2; exit}' "$config")
  expected="$LV_CONTAINER_WORK_ROOT/$run_relative"
  [[ "$output" == "$expected" ]] ||
    die "config output_root is $output; expected $expected for this queue row"
  packed=$(awk '$1 == "packed_root:" {print $2; exit}' "$config")
  expected_pack="$LV_CONTAINER_WORK_ROOT/$LV_PACK_RELATIVE"
  [[ "$packed" == "$expected_pack" ]] ||
    die "config packed_root is $packed; expected $expected_pack"
}

load_single_source_allowed_categories() {
  local config=$1 source_count category list_sections
  require_file "$config"
  source_count=$(awk '$1 == "kind:" {count++} END {print count + 0}' "$config")
  [[ "$source_count" == 1 ]] ||
    die "audition config must contain exactly one data source"
  ALLOWED_CATEGORIES=()
  list_sections=$(awk '$1 == "allowed_categories:" {count++} END {print count + 0}' "$config")
  [[ "$list_sections" == 1 ]] ||
    die "config must contain exactly one allowed_categories list"
  while IFS= read -r category; do
    [[ "$category" =~ ^[A-Za-z0-9_-]+$ ]] ||
      die "unsafe allowed category in $config: $category"
    ALLOWED_CATEGORIES+=("$category")
  done < <(
    awk '
      $1 == "allowed_categories:" {
        sections++
        capture = 1
        next
      }
      capture && $1 == "-" {
        value = $0
        sub(/^[[:space:]]*-[[:space:]]*/, "", value)
        sub(/[[:space:]]*#.*$/, "", value)
        sub(/[[:space:]]+$/, "", value)
        print value
        values++
        next
      }
      capture && $0 !~ /^[[:space:]]*$/ {capture = 0}
      END {if (sections != 1 || values == 0) exit 2}
    ' "$config"
  )
  ((${#ALLOWED_CATEGORIES[@]} > 0)) ||
    die "config allowed_categories must not be empty"
}

latest_checkpoint() {
  local checkpoint_root=$1
  local checkpoints=()
  shopt -s nullglob
  checkpoints=("$checkpoint_root"/step-*.pt)
  shopt -u nullglob
  ((${#checkpoints[@]} > 0)) || return 1
  printf '%s\n' "${checkpoints[@]}" | LC_ALL=C sort -V | tail -n 1
}

host_path_to_container_work_path() {
  local host_path=$1
  [[ "$host_path" == "$LV_WORK_ROOT"/* ]] ||
    die "path must be below LV_WORK_ROOT: $host_path"
  printf '%s/%s\n' \
    "$LV_CONTAINER_WORK_ROOT" "${host_path#"$LV_WORK_ROOT"/}"
}

remaining_seconds() {
  local now deadline remaining
  now=$(date +%s)
  deadline=${LV_DEADLINE_EPOCH:-$((now + 70 * 60 * 60))}
  [[ "$deadline" =~ ^[0-9]+$ ]] || die "LV_DEADLINE_EPOCH must be an epoch"
  remaining=$((deadline - now - LV_SHUTDOWN_MARGIN_SECONDS))
  (( remaining >= 60 )) || return 75
  printf '%s\n' "$remaining"
}

prepare_docker_args() {
  local docker_memory_gib allocated_memory_gib runtime_user
  assert_persistent_work_root
  assert_single_gpu_allocation
  require_command docker
  require_command timeout
  require_directory "$LV_PROJECT_ROOT"
  [[ "$LV_DOCKER_CPUS" =~ ^[1-9][0-9]*$ ]] ||
    die "LV_DOCKER_CPUS must be a positive integer"
  (( LV_DOCKER_CPUS <= 28 )) ||
    die "LV_DOCKER_CPUS cannot exceed 28"
  if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
    (( LV_DOCKER_CPUS <= SLURM_CPUS_PER_TASK )) ||
      die "LV_DOCKER_CPUS exceeds the SLURM CPU allocation"
  fi
  [[ "$LV_DOCKER_MEMORY" =~ ^([1-9][0-9]*)[gG]$ ]] ||
    die "LV_DOCKER_MEMORY must use integer gigabytes"
  docker_memory_gib=${BASH_REMATCH[1]}
  (( docker_memory_gib <= 32 )) ||
    die "LV_DOCKER_MEMORY cannot exceed the qgpu limit of 32G"
  allocated_memory_gib=${LV_QGPU_MEMORY_GIB:-32}
  [[ "$allocated_memory_gib" =~ ^[1-9][0-9]*$ ]] ||
    die "LV_QGPU_MEMORY_GIB must be a positive integer"
  (( docker_memory_gib <= allocated_memory_gib )) ||
    die "LV_DOCKER_MEMORY exceeds the qgpu memory request"
  runtime_user=$(id -un)
  [[ "$runtime_user" =~ ^[A-Za-z0-9._-]+$ ]] ||
    die "host user name is unsafe for the container environment"
  mkdir -p \
    "$LV_WORK_ROOT/cache/huggingface" \
    "$LV_WORK_ROOT/cache/torch" \
    "$LV_WORK_ROOT/cache/xdg" \
    "$LV_WORK_ROOT/runtime-home" \
    "$LV_STATE_ROOT"

  DOCKER_ARGS=(
    run --rm
    --gpus "device=$CUDA_VISIBLE_DEVICES"
    --cpus "$LV_DOCKER_CPUS"
    --memory "$LV_DOCKER_MEMORY"
    --memory-swap "$LV_DOCKER_MEMORY"
    --shm-size "$LV_SHM_SIZE"
    --ulimit memlock=-1
    --network "$LV_DOCKER_NETWORK"
    --user "$(id -u):$(id -g)"
    -v "$LV_PROJECT_ROOT:$LV_CONTAINER_PROJECT_ROOT:ro"
    -v "$LV_WORK_ROOT:$LV_CONTAINER_WORK_ROOT"
    -e "PYTHONPATH=$LV_CONTAINER_CODE_ROOT/src"
    -e PYTHONDONTWRITEBYTECODE=1
    -e "USER=$runtime_user"
    -e "LOGNAME=$runtime_user"
    -e CUDA_VISIBLE_DEVICES=0
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -e "HOME=$LV_CONTAINER_WORK_ROOT/runtime-home"
    -e "HF_HOME=$LV_CONTAINER_WORK_ROOT/cache/huggingface"
    -e "TORCH_HOME=$LV_CONTAINER_WORK_ROOT/cache/torch"
    -e "XDG_CACHE_HOME=$LV_CONTAINER_WORK_ROOT/cache/xdg"
    -w "$LV_CONTAINER_CODE_ROOT"
  )

  if [[ -n "${LV_RAVE_ROOT:-}" ]]; then
    require_directory "$LV_RAVE_ROOT"
    DOCKER_ARGS+=(
      -v "$LV_RAVE_ROOT:$LV_CONTAINER_RAVE_ROOT:ro"
    )
  fi
  if [[ -n "${LV_SOURCE_ROOT:-}" ]]; then
    require_directory "$LV_SOURCE_ROOT"
    DOCKER_ARGS+=(
      -v "$LV_SOURCE_ROOT:$LV_CONTAINER_SOURCE_ROOT:ro"
    )
  fi
}

run_gpu_container() {
  prepare_docker_args
  docker "${DOCKER_ARGS[@]}" "$LV_IMAGE" "$@"
}

run_timed_gpu_container() {
  local seconds status
  seconds=$(remaining_seconds) || return $?
  prepare_docker_args
  if timeout --foreground --signal=TERM --kill-after=120 \
    "${seconds}s" docker "${DOCKER_ARGS[@]}" "$LV_IMAGE" "$@"; then
    return 0
  else
    status=$?
  fi
  if (( status == 124 )); then
    return 75
  fi
  return "$status"
}

optional_pitch_probe_args() {
  PITCH_PROBE_ARGS=()
  if [[ -n "${LV_PITCH_PROBE:-}" ]]; then
    local container_probe
    require_file "$LV_PITCH_PROBE/qualification.json"
    container_probe=$(host_path_to_container_work_path "$LV_PITCH_PROBE")
    PITCH_PROBE_ARGS=(--pitch-probe "$container_probe")
  fi
}
