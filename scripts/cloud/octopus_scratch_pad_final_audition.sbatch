#!/usr/bin/env bash
#SBATCH --job-name=mb-pad-audition
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/midibrave-v3/scratch-pad-final-audition/logs/render-%j.out

set -euo pipefail

root=/data/midibrave-v3/scratch-pad-final-audition
repo=$root/repo
config=$root/config.yaml
checkpoint=$root/input/update-00001500.pt
seed_bank=$root/cache/seed-bank
runtime=$root/output/runtime.pt
artifact=$root/output/audition
image=midibrave:v0.10.0-clap-recon

check_sha256() {
  local expected=$1
  local path=$2
  local measured
  measured=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$measured" != "$expected" ]]; then
    printf 'SHA-256 mismatch for %s: expected %s, measured %s\n' \
      "$path" "$expected" "$measured" >&2
    return 1
  fi
}

check_sha256 174f15307bb0054c89f075ceb5e2d01744ad8128e8aefa04f085c79aff4d33b9 "$checkpoint"
check_sha256 0533a1b0c77ae8025e16679f9340cec00cd5874048324e329ddde7c2027e5ad5 "$root/cache/rave-statistics.npz"
check_sha256 1d22daf0bb5ea3d7c22c552a4abab0ccd8b5a16d6a1014cfd8a4916599744110 "$seed_bank.npz"
check_sha256 99d6481f705df1e02433f65508fada6b1d3f6b8606b55f2f5156ae752e07c818 "$seed_bank.json"
test -f "$config"
test -f /data/midibrave-v3/manifests/pad50-strict.jsonl
test -d "$root/cache/clap"
test ! -e "$runtime"
test ! -e "$artifact/runtime-audition-manifest.json"
mkdir -p "$root/output" "$artifact"

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" --shm-size=16g \
  -v /data:/data -e PYTHONPATH="$repo/src" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w "$repo" "$image" \
  python -m midibrave.cli export-predictive \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --seed-bank "$seed_bank" \
    --output "$runtime" \
    --device cuda

srun --kill-on-bad-exit=1 --cpu-bind=cores docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" --shm-size=16g \
  -v /data:/data -e PYTHONPATH="$repo/src" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w "$repo" "$image" \
  python scripts/render_exported_runtime_audition.py \
    --config "$config" \
    --runtime "$runtime" \
    --seed-bank "$seed_bank" \
    --output "$artifact" \
    --seeds 3 --timbres 4 --notes 4 \
    --duration 2.5 --warmup 0.4 \
    --random-seed 20260722 --device cuda

cp "$repo/audition/scratch-pad-final/index.html" "$artifact/index.html"
test "$(find "$artifact/runtime-examples" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 48
grep -q '"encoder_free": true' "$artifact/runtime-audition-manifest.json"
grep -q '"runtime_sha256"' "$artifact/runtime-audition-manifest.json"
printf 'audition complete: %s\n' "$artifact"
