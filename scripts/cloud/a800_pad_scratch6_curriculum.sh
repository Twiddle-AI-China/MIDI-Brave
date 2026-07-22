#!/usr/bin/env bash
# Run inside the six-A800 SLURM allocation. The first RAVE stage has no
# checkpoint input: every trainable module is initialized from the configured
# random seed. Later stages consume only checkpoints made by this curriculum.

set -Eeuo pipefail

ROOT=${MIDIBRAVE_ROOT:-/data/run01/scwc257/latent-cosmos-synth}
RUN_ROOT=${MIDIBRAVE_RUN_ROOT:-$ROOT/midibrave-v3/scratch-pad-20260722}
DATA_ROOT=${MIDIBRAVE_PAD_AUDIO_ROOT:-$ROOT/datasets/latent-cosmos-synth/serum-dataset}
ENV_ROOT=${MIDIBRAVE_ENV:-$ROOT/envs/midibrave-py311-torch271-cu128}
REPO=${MIDIBRAVE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${MIDIBRAVE_PYTHON:-$ENV_ROOT/bin/python}
TORCHRUN=${MIDIBRAVE_TORCHRUN:-$ENV_ROOT/bin/torchrun}
WORLD_SIZE=6

RAVE_RECON_STEPS=${RAVE_RECON_STEPS:-500}
RAVE_RAMP_STEPS=${RAVE_RAMP_STEPS:-500}
RAVE_FULL_STEPS=${RAVE_FULL_STEPS:-7500}
PREDICTOR_STEPS=${PREDICTOR_STEPS:-2500}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1500}

MANIFEST=$RUN_ROOT/pad50-strict.jsonl
METADATA=$RUN_ROOT/pad50-strict.meta.json
CACHE=$RUN_ROOT/cache/pad50
SHARED_FEATURE_CACHE=${MIDIBRAVE_SHARED_FEATURE_CACHE:-$ROOT/midibrave/cache/serum_strict_1822}
STATISTICS=$CACHE/rave-statistics.npz
CALIBRATION=$CACHE/predictive-calibration.json
SEED_BANK=$CACHE/seed-bank.pt
CONFIG_ROOT=$RUN_ROOT/configs
LOG_ROOT=$RUN_ROOT/logs

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "This script must run inside a SLURM GPU allocation." >&2
  exit 2
fi
[[ -d "$DATA_ROOT" ]] || { echo "missing Pad audio root: $DATA_ROOT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "missing Python environment: $PYTHON" >&2; exit 1; }
[[ -d "$REPO/src/midibrave" ]] || { echo "missing MidiBrave package: $REPO" >&2; exit 1; }

mkdir -p "$RUN_ROOT" "$CACHE" "$CACHE/rave" "$CONFIG_ROOT" \
  "$LOG_ROOT" "$RUN_ROOT/monitor"
export PYTHONPATH=$REPO/src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
if (( ${#GPU_IDS[@]} < WORLD_SIZE )); then
  echo "six visible GPUs are required; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 1
fi

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 10 > "$RUN_ROOT/monitor/gpu-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
tensorboard_pid=
if [[ ${START_TENSORBOARD:-1} == 1 ]]; then
  "$ENV_ROOT/bin/tensorboard" --logdir "$RUN_ROOT/runs" --host 0.0.0.0 \
    --port "${TENSORBOARD_PORT:-6006}" > "$LOG_ROOT/tensorboard-${SLURM_JOB_ID}.out" 2>&1 &
  tensorboard_pid=$!
fi
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  [[ -z "$tensorboard_pid" ]] || kill "$tensorboard_pid" 2>/dev/null || true
}
trap cleanup EXIT

find_clap_checkpoint() {
  local requested=${MIDIBRAVE_CLAP_CHECKPOINT:-$ROOT/model_weights/laion-clap/music_audioset_epoch_15_esc_90.14.pt}
  if [[ -f "$requested" ]]; then
    printf '%s\n' "$requested"
    return
  fi
  echo "CLAP checkpoint not found: $requested; set MIDIBRAVE_CLAP_CHECKPOINT" >&2
  return 1
}
CLAP_CHECKPOINT=$(find_clap_checkpoint)

# The Pad manifest is an already-frozen server artifact. Do not select a new
# subset here: only prove the 50-preset contract and that the audio resolves.
[[ -f "$MANIFEST" ]] || { echo "missing fixed Pad manifest: $MANIFEST" >&2; exit 1; }
"$PYTHON" - "$DATA_ROOT" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

audio_root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
with manifest.open(encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

def preset(row):
    return row.get("preset_id") or row.get("timbre_id") or row.get("instrument_id")

presets = {}
for row in rows:
    current_preset = preset(row)
    presets.setdefault(current_preset, []).append(row)
if len(rows) != 3600 or len(presets) != 50:
    raise SystemExit(
        f"fixed Pad manifest is {len(rows)} records/{len(presets)} presets, expected 3600/50")
bad = {key: len(value) for key, value in presets.items() if len(value) != 72}
if bad:
    raise SystemExit(f"Pad manifest must contain 72 MIDI configurations per preset: {bad}")
print(json.dumps({"manifest": str(manifest), "records": 3600,
                  "presets": 50, "records_per_preset": 72}, sort_keys=True))
PY

# Runtime copies keep the checked-in templates immutable while allowing the
# discovered CLAP checkpoint to live anywhere under the server project root.
for name in \
  a800_pad50_scratch_rave_reconstruction \
  a800_pad50_scratch_rave_control_ramp \
  a800_pad50_scratch_rave_full \
  a800_pad50_scratch_predictor \
  a800_pad50_scratch_rollout; do
  "$PYTHON" - "$REPO/configs/v3/$name.yaml" "$CONFIG_ROOT/$name.yaml" \
    "$CLAP_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination, clap = map(Path, sys.argv[1:])
raw = yaml.safe_load(source.read_text(encoding="utf-8"))
raw["data"]["clap_checkpoint"] = str(clap.resolve())
destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
done

RECON_CONFIG=$CONFIG_ROOT/a800_pad50_scratch_rave_reconstruction.yaml
RAMP_CONFIG=$CONFIG_ROOT/a800_pad50_scratch_rave_control_ramp.yaml
FULL_CONFIG=$CONFIG_ROOT/a800_pad50_scratch_rave_full.yaml
PREDICTOR_CONFIG=$CONFIG_ROOT/a800_pad50_scratch_predictor.yaml
ROLLOUT_CONFIG=$CONFIG_ROOT/a800_pad50_scratch_rollout.yaml

expected=$(wc -l < "$MANIFEST")
[[ -d "$SHARED_FEATURE_CACHE" ]] || {
  echo "missing shared serum feature cache: $SHARED_FEATURE_CACHE" >&2
  exit 1
}
for stage in audio clap pitch; do
  [[ -d "$SHARED_FEATURE_CACHE/$stage" ]] || {
    echo "missing shared cache stage: $SHARED_FEATURE_CACHE/$stage" >&2
    exit 1
  }
  if [[ ! -e "$CACHE/$stage" && ! -L "$CACHE/$stage" ]]; then
    ln -s "$SHARED_FEATURE_CACHE/$stage" "$CACHE/$stage"
  elif [[ ! -L "$CACHE/$stage" || $(readlink -f "$CACHE/$stage") != $(readlink -f "$SHARED_FEATURE_CACHE/$stage") ]]; then
    echo "refusing to replace existing scratch cache path: $CACHE/$stage" >&2
    exit 1
  fi
done
"$PYTHON" - "$DATA_ROOT" "$MANIFEST" "$CACHE" <<'PY'
import sys
from pathlib import Path
from midibrave.data import load_manifest

audio_root, manifest, cache = map(Path, sys.argv[1:])
by_preset = {}
for record in load_manifest(manifest):
    by_preset.setdefault(record.preset_id, []).append(record)
missing = []
checked = 0
for records in by_preset.values():
    records.sort(key=lambda record: record.sample_id)
    for record in (records[0], records[-1]):
        raw = Path(record.audio_path)
        raw = raw if raw.is_absolute() else audio_root / raw
        paths = [raw]
        paths.extend(
            cache / stage / f"{record.cache_id}{suffix}"
            for stage, suffix in (("audio", ".npz"), ("clap", ".npy"),
                                  ("pitch", ".npz")))
        missing.extend(str(path) for path in paths if not path.is_file())
        checked += 1
if missing:
    raise SystemExit(
        f"sampled Pad audio/cache validation misses {len(missing)} artifacts; first={missing[0]}")
print(f"sampled Pad audio/cache verified: {checked} endpoint records across {len(by_preset)} presets")
PY

torchrun6() {
  if [[ -x "$TORCHRUN" ]]; then
    "$TORCHRUN" --standalone --nproc_per_node="$WORLD_SIZE" "$@"
  else
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" "$@"
  fi
}

tensorboard_throughput() {
  local run_dir=$1 elapsed_ms=$2 batch=$3
  "$PYTHON" - "$run_dir" "$elapsed_ms" "$batch" "$WORLD_SIZE" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
elapsed_ms, batch, world = map(int, sys.argv[2:])
score = None
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = sorted(run_dir.glob("events.out.tfevents.*"),
                         key=lambda path: path.stat().st_mtime, reverse=True)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        values = accumulator.Scalars("train/samples_per_second")
        if values:
            score = float(values[-1].value)
            break
except Exception:
    pass
if score is None:
    score = 20.0 * batch * world * 1000.0 / max(1, elapsed_ms)
print(f"{score:.6f}")
PY
}

sweep_stage_batch() {
  local label=$1 config=$2 stage=$3 warm_start=$4
  shift 4
  local candidates=("$@")
  local sweep_root=$RUN_ROOT/${label}-batch-sweep
  local winner_file=$sweep_root/winner.txt
  local results=$sweep_root/results.tsv
  if [[ -s "$winner_file" ]]; then
    cat "$winner_file"
    return
  fi
  mkdir -p "$sweep_root"
  : > "$results"
  for batch in "${candidates[@]}"; do
    local run_name=pad50_scratch_${label}_sweep_b${batch}
    local sweep_config=$CONFIG_ROOT/${run_name}.yaml
    "$PYTHON" - "$config" "$sweep_config" "$run_name" "$batch" "$stage" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination = map(Path, sys.argv[1:3])
raw = yaml.safe_load(source.read_text(encoding="utf-8"))
raw["train"]["run_name"] = sys.argv[3]
raw["train"]["batch_per_gpu"] = int(sys.argv[4])
raw["train"][f"{sys.argv[5]}_steps"] = 20
raw["train"]["log_every"] = 20
raw["train"]["checkpoint_every"] = 20
destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
    local run_dir=$RUN_ROOT/runs/$run_name/$stage
    local log=$LOG_ROOT/${label}-sweep-b${batch}.out
    local start_args=()
    if [[ -n "$warm_start" ]]; then
      [[ -f "$warm_start" ]] || {
        echo "missing sweep warm start: $warm_start" >&2
        return 1
      }
      start_args=(--warm-start "$warm_start")
    fi
    local started ended elapsed_ms
    started=$(date +%s%N)
    if torchrun6 -m midibrave.trainer --config "$sweep_config" --stage "$stage" \
        --max-steps 20 --batch-per-gpu "$batch" "${start_args[@]}" > "$log" 2>&1; then
      ended=$(date +%s%N)
      elapsed_ms=$(( (ended - started) / 1000000 ))
      local score
      score=$(tensorboard_throughput "$run_dir" "$elapsed_ms" "$batch")
      printf '%s\t%s\tOK\t%s\n' "$batch" "$score" "$elapsed_ms" \
        | tee -a "$results" >&2
    elif grep -Eqi 'CUDA.*out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$log"; then
      printf '%s\t0\tOOM\t0\n' "$batch" | tee -a "$results" >&2
    else
      printf '%s\t0\tFAIL\t0\n' "$batch" | tee -a "$results" >&2
      echo "$label batch sweep failed for batch=$batch; continuing; inspect $log" >&2
    fi
  done
  local winner
  winner=$(awk '$3 == "OK" && $2 + 0 > best {best=$2 + 0; winner=$1} END {print winner}' "$results")
  [[ -n "$winner" ]] || {
    echo "no successful $label batch candidate" >&2
    return 1
  }
  printf '%s\n' "$winner" > "$winner_file"
  printf '%s\n' "$winner"
}

checkpoint_path() {
  local run_name=$1 stage=$2 updates=$3
  printf '%s/runs/%s/%s/update-%08d.pt\n' "$RUN_ROOT" "$run_name" "$stage" "$updates"
}

run_stage() {
  local config=$1 stage=$2 run_name=$3 target=$4 warm_start=${5:-} batch=${6:-}
  local run_dir=$RUN_ROOT/runs/$run_name/$stage
  local final
  final=$(checkpoint_path "$run_name" "$stage" "$target")
  if [[ -f "$final" ]]; then
    echo "=== $run_name/$stage already complete: $final ==="
    return
  fi
  mkdir -p "$run_dir"
  local latest
  latest=$(find "$run_dir" -maxdepth 1 -type f -name 'update-*.pt' 2>/dev/null | sort | tail -n 1 || true)
  local start_args=()
  if [[ -n "$latest" ]]; then
    start_args=(--resume "$latest")
  elif [[ -n "$warm_start" ]]; then
    [[ -f "$warm_start" ]] || { echo "missing warm start: $warm_start" >&2; exit 1; }
    start_args=(--warm-start "$warm_start")
  fi
  local batch_args=()
  [[ -z "$batch" ]] || batch_args=(--batch-per-gpu "$batch")
  echo "=== train $run_name/$stage target=$target start=${start_args[*]:-random} ==="
  torchrun6 -m midibrave.trainer --config "$config" --stage "$stage" \
    --max-steps "$target" "${batch_args[@]}" "${start_args[@]}"
  [[ -f "$final" ]] || { echo "stage did not create final checkpoint: $final" >&2; exit 1; }
}

RECON_FINAL=$(checkpoint_path pad50_scratch_rave_reconstruction rave "$RAVE_RECON_STEPS")
RAMP_FINAL=$(checkpoint_path pad50_scratch_rave_control_ramp rave "$RAVE_RAMP_STEPS")
RAVE_FINAL=$(checkpoint_path pad50_scratch_rave_full rave "$RAVE_FULL_STEPS")
PREDICTOR_FINAL=$(checkpoint_path pad50_scratch_predictor predictor "$PREDICTOR_STEPS")
ROLLOUT_FINAL=$(checkpoint_path pad50_scratch_rollout rollout "$ROLLOUT_STEPS")

# Only the sweep candidates and this first formal stage initialize randomly.
# Use the full-control graph for sizing: reconstruction-only batches do not
# include the counterfactual decoder, analytic pitch, or CLAP waveform graph.
RAVE_BATCH=$(sweep_stage_batch rave-full "$FULL_CONFIG" rave "" 16 32 64 96)
echo "=== RAVE batch sweep winner: $RAVE_BATCH per GPU ==="
run_stage "$RECON_CONFIG" rave pad50_scratch_rave_reconstruction \
  "$RAVE_RECON_STEPS" "" "$RAVE_BATCH"
run_stage "$RAMP_CONFIG" rave pad50_scratch_rave_control_ramp \
  "$RAVE_RAMP_STEPS" "$RECON_FINAL" "$RAVE_BATCH"
run_stage "$FULL_CONFIG" rave pad50_scratch_rave_full \
  "$RAVE_FULL_STEPS" "$RAMP_FINAL" "$RAVE_BATCH"

actual_rave=$(find "$CACHE/rave" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
if [[ ! -f "$STATISTICS" || $actual_rave -ne $expected ]]; then
  echo "=== rebuild RAVE cache and latent statistics from $RAVE_FINAL ==="
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} "$PYTHON" -m midibrave.cli cache-rave \
    --config "$PREDICTOR_CONFIG" --checkpoint "$RAVE_FINAL" --device cuda
fi

"$PYTHON" - "$RAVE_FINAL" "$STATISTICS" "$CACHE/rave" "$expected" <<'PY'
import hashlib
import sys
from pathlib import Path
import numpy as np

checkpoint, statistics, cache, expected = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
with np.load(statistics, allow_pickle=False) as values:
    cached = str(values["checkpoint_hash"].item())
if cached != digest:
    raise SystemExit(f"RAVE statistics checkpoint mismatch: {cached} != {digest}")
actual = len(list(cache.glob("*.npz")))
if actual != expected:
    raise SystemExit(f"RAVE cache incomplete: {actual}/{expected}")
print(f"RAVE cache verified: {actual}/{expected}, checkpoint={digest}")
PY

rave_hash=$(sha256sum "$RAVE_FINAL" | awk '{print $1}')
if [[ ! -f "$SEED_BANK" ]]; then
  "$PYTHON" -m midibrave.cli build-seed-bank --config "$PREDICTOR_CONFIG" \
    --checkpoint-hash "$rave_hash" --output "$SEED_BANK"
fi

# The first predictor sweep candidate creates CALIBRATION on rank zero before
# its first update; all later candidates and the formal run reuse that artifact.
PREDICTOR_BATCH=$(sweep_stage_batch predictor "$PREDICTOR_CONFIG" predictor \
  "$RAVE_FINAL" 32 64 96 128)
echo "=== predictor batch sweep winner: $PREDICTOR_BATCH per GPU ==="
run_stage "$PREDICTOR_CONFIG" predictor pad50_scratch_predictor \
  "$PREDICTOR_STEPS" "$RAVE_FINAL" "$PREDICTOR_BATCH"
[[ -f "$CALIBRATION" ]] || { echo "missing predictive calibration: $CALIBRATION" >&2; exit 1; }

ROLLOUT_BATCH=$(sweep_stage_batch rollout "$ROLLOUT_CONFIG" rollout \
  "$PREDICTOR_FINAL" 8 16 24 32)
echo "=== rollout batch sweep winner: $ROLLOUT_BATCH per GPU ==="
run_stage "$ROLLOUT_CONFIG" rollout pad50_scratch_rollout \
  "$ROLLOUT_STEPS" "$PREDICTOR_FINAL" "$ROLLOUT_BATCH"

cat <<EOF
=== scratch Pad curriculum complete ===
RAVE:      $RAVE_FINAL
Predictor: $PREDICTOR_FINAL
Rollout:   $ROLLOUT_FINAL
Stats:     $STATISTICS
Seed bank: $SEED_BANK
TensorBoard logdir: $RUN_ROOT/runs
EOF
