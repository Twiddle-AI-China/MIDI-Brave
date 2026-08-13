#!/usr/bin/env bash
#SBATCH --job-name=serum128-audio-taxonomy
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jnzhang/slurm_logs/serum128-audio-taxonomy-%j.out
#SBATCH --error=/home/jnzhang/slurm_logs/serum128-audio-taxonomy-%j.err

set -euo pipefail

code_root=/home/jnzhang/Developer/Latent-Cosmos-Synth-rave-midi-integration/MidiBrave-v2
artifact_root=/home/jnzhang/Developer/Latent-Cosmos-Synth-rave-midi-integration/artifacts
taxonomy_host="$artifact_root/serum128-audio-v1"
container_code=/workspace/MidiBrave-v2
container_taxonomy=/output/serum128-audio-v1
image=midibrave:v0.10.0-zrave-flow

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  printf 'taxonomy must run in a SLURM GPU allocation\n' >&2
  exit 2
}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
  printf 'expected exactly one allocated GPU, got %s\n' "$CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
[[ -f "$code_root/src/midibrave/zrave_timbre_taxonomy.py" ]] || exit 2
[[ -f "$code_root/scripts/lvzihao/validate_imported_audio_taxonomy.py" ]] || exit 2
[[ -f /data/midibrave-zrave-flow-serum128/packs/serum-balanced/index.json ]] || exit 2
[[ -f /data/midibrave-zrave-flow-serum128/manifests/serum-balanced.jsonl ]] || exit 2
[[ -d /data/datasets/Timbre_A/serum-octopus-v1 ]] || exit 2
mkdir -p "$artifact_root"

docker_args=(
  run --rm --network none
  --gpus "device=$CUDA_VISIBLE_DEVICES"
  --shm-size=4g
  --cpus "${SLURM_CPUS_PER_TASK:-12}"
  --memory 14g
  -v "$code_root:$container_code:ro"
  -v /data/midibrave-zrave-flow-serum128:/data/midibrave-zrave-flow-serum128:ro
  -v /data/datasets/Timbre_A/serum-octopus-v1:/source:ro
  -v "$artifact_root:/output"
  -e "PYTHONPATH=$container_code/src"
  -w "$container_code"
  "$image"
)

if [[ ! -d "$taxonomy_host" ]]; then
  srun --kill-on-bad-exit=1 \
    docker "${docker_args[@]}" \
    python -m midibrave.zrave_timbre_taxonomy \
      --pack-root /data/midibrave-zrave-flow-serum128/packs/serum-balanced \
      --feature-source audio \
      --manifest /data/midibrave-zrave-flow-serum128/manifests/serum-balanced.jsonl \
      --output-root "$container_taxonomy" \
      --workers "${SLURM_CPUS_PER_TASK:-12}"
fi

if [[ ! -f "$taxonomy_host/octopus-export.provenance.json" ]]; then
  srun --kill-on-bad-exit=1 \
    docker "${docker_args[@]}" \
    python scripts/lvzihao/validate_imported_audio_taxonomy.py \
      --pack-root /data/midibrave-zrave-flow-serum128/packs/serum-balanced \
      --taxonomy-root "$container_taxonomy" \
      --seal-octopus-export
else
  srun --kill-on-bad-exit=1 \
    docker "${docker_args[@]}" \
    python scripts/lvzihao/validate_imported_audio_taxonomy.py \
      --pack-root /data/midibrave-zrave-flow-serum128/packs/serum-balanced \
      --taxonomy-root "$container_taxonomy"
fi

sha256sum \
  "$taxonomy_host/taxonomy.report.json" \
  "$taxonomy_host/octopus-export.provenance.json"
