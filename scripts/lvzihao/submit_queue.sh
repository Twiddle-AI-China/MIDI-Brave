#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( $# < 1 || $# > 2 )); then
  printf 'usage: %s QUEUE_TSV [ALLOCATION_COUNT]\n' "$0" >&2
  exit 2
fi

queue=$1
allocations=${2:-${LV_ALLOCATIONS:-8}}
: "${LV_CPUS:=8}"
: "${LV_MEMORY:=32G}"
: "${LV_TIME_LIMIT:=72:00:00}"
: "${LV_JOB_PREFIX:=${USER:-twiddle}_midibrave}"

[[ -f "$queue" ]] || {
  printf 'queue not found: %s\n' "$queue" >&2
  exit 2
}
queue=$(cd -- "$(dirname -- "$queue")" && pwd)/$(basename -- "$queue")
runner=$(cd -- "$script_dir" && pwd)/allocation_runner.sh

[[ "$allocations" =~ ^[1-9][0-9]*$ ]] && (( allocations <= 64 )) || {
  printf 'allocation count must be between 1 and 64\n' >&2
  exit 2
}
[[ "$LV_CPUS" =~ ^[1-9][0-9]*$ ]] && (( LV_CPUS <= 28 )) || {
  printf 'LV_CPUS must be between 1 and 28\n' >&2
  exit 2
}
[[ "$LV_MEMORY" =~ ^([1-9][0-9]*)G$ ]] || {
  printf 'LV_MEMORY must use integer gigabytes, for example 32G\n' >&2
  exit 2
}
memory_gib=${BASH_REMATCH[1]}
(( memory_gib <= 32 )) || {
  printf 'LV_MEMORY cannot exceed the measured qgpu limit of 32G\n' >&2
  exit 2
}
[[ "$LV_TIME_LIMIT" == 72:00:00 ]] || {
  printf 'LV_TIME_LIMIT must be 72:00:00\n' >&2
  exit 2
}
[[ "$LV_JOB_PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || {
  printf 'LV_JOB_PREFIX contains unsafe characters\n' >&2
  exit 2
}
[[ -z "${SLURM_JOB_ID:-}" ]] || {
  printf 'submit_queue.sh must run on the login host, not in an allocation\n' >&2
  exit 2
}
export LV_QGPU_MEMORY_GIB=$memory_gib

if [[ "${LV_SUBMIT_DRY_RUN:-0}" != 1 ]]; then
  command -v qgpu >/dev/null 2>&1 || {
    printf 'qgpu is required\n' >&2
    exit 2
  }
fi

for ((index = 1; index <= allocations; index++)); do
  job_name=$(printf '%s_%02d' "$LV_JOB_PREFIX" "$index")
  command_args=(
    qgpu -n "$job_name"
    -c "$LV_CPUS"
    -m "$LV_MEMORY"
    -t "$LV_TIME_LIMIT"
    -- bash "$runner" "$queue"
  )
  if [[ "${LV_SUBMIT_DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${command_args[@]}"
    printf '\n'
  else
    "${command_args[@]}"
  fi
done
