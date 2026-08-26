#!/bin/bash
set -euo pipefail
umask 027

ROOT=/data/atlas-flow-pad-v1
CODE=/data/projects/latent-cosmos-synth/atlas-flow-6d143b2-20260822/MidiBrave-v2
CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
IMAGE=midibrave:atlas-flow-v1
RENDER_IMAGE=serum-render:atlas-flow-v1
REPLAY=/data/serum-timbreclap/attestation/serum-fxp-v2-replays.db
SNAPSHOT=/data/serum-timbreclap/attestation/serum-corpus-v1-snapshot.db
DATASET=/data/datasets/Timbre_A/serum-octopus-v2/atlas-flow-pad-v1
WORK=$DATASET/work
WAV=$DATASET/wav
RENDER_MANIFEST=$ROOT/manifests/pad-top50-render.jsonl
TRAIN_MANIFEST=$ROOT/manifests/pad-top50-training.jsonl

for path in "$CODE" "$REPLAY" "$SNAPSHOT"; do
  [[ -e "$path" && ! -L "$path" ]] || { echo "required input missing or symbolic: $path" >&2; exit 70; }
done
available=$(df -P /data | awk 'NR==2{print $4}')
total=$(df -P /data | awk 'NR==2{print $2}')
(( available * 100 >= total * 20 )) || { echo "/data has less than 20% free" >&2; exit 70; }
mkdir -p "$ROOT/manifests" "$ROOT/contracts" "$ROOT/logs" "$WORK" "$WAV"

run_cpu_image() {
  docker run --rm --network none --cpus 80 --memory 110g --memory-swap 110g \
    --pids-limit 16384 --user "$(id -u):$(id -g)" \
    --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
    "$IMAGE" "$@"
}

run_cpu_image python -m midibrave.atlas_flow_manifest --output-root "$ROOT/manifests"

python3 - "$REPLAY" "$RENDER_MANIFEST" <<'PY'
import json, sqlite3, sys
replay, manifest = sys.argv[1:]
selected = sorted({json.loads(line)["preset_index"] for line in open(manifest, encoding="utf-8")})
connection = sqlite3.connect(f"file:{replay}?mode=ro&immutable=1", uri=True)
placeholders = ",".join("?" for _ in selected)
available = {row[0] for row in connection.execute(
    f"select distinct preset_index from preset_replays where plugin_id='xfer/serum' and preset_index in ({placeholders})",
    selected,
)}
connection.close()
missing = sorted(set(selected) - available)
if missing:
    raise SystemExit(f"replay coverage missing {len(missing)} selected presets: {missing[:10]}")
print(f"replay coverage OK: {len(available)}/50")
PY

if [[ ! -s "$WORK/corpus.db" ]]; then
  cp --reflink=auto -- "$SNAPSHOT" "$WORK/corpus.db"
fi
chmod 0600 "$WORK/corpus.db"

docker run --rm --name atlas-flow-pad-render --network none --cap-drop ALL \
  --security-opt no-new-privileges=true --user 1004:1004 \
  --cpus 80 --memory 110g --memory-swap 110g --pids-limit 16384 \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=2g,mode=1777 \
  --mount "type=bind,src=$WORK,dst=/work" \
  --mount "type=bind,src=$WAV,dst=/wavs" \
  --mount "type=bind,src=$RENDER_MANIFEST,dst=/inputs/config.jsonl,readonly" \
  --mount "type=bind,src=$REPLAY,dst=/inputs/replay.db,readonly" \
  "$RENDER_IMAGE" \
  --output /work/corpus.db --wav-dir /wavs --render-only \
  --midi-config-manifest /inputs/config.jsonl --replay-db /inputs/replay.db \
  --sample-rate 44100 --workers 72 --resume

run_cpu_image python -m midibrave.atlas_flow_manifest \
  --output-root "$ROOT/manifests" --wav-root "$WAV"
run_cpu_image python -m midibrave.atlas_flow_train preprocess \
  --config "$CONFIG" --workers 80

python3 - "$TRAIN_MANIFEST" "$WAV" "$ROOT/contracts/render-qc.json" <<'PY'
import hashlib, json, sys, wave
from collections import Counter, defaultdict
from pathlib import Path
manifest, wav_root, output = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
hashes = defaultdict(list)
for row in rows:
    path = wav_root / row["audio_path"]
    with wave.open(str(path), "rb") as handle:
        facts = (handle.getframerate(), handle.getnchannels(), handle.getnframes(), handle.getsampwidth())
    if facts != (44100, 1, 220500, 2):
        raise SystemExit(f"WAV contract failed: {path} {facts}")
    if row["render_id"] > 0 or row["midi_note"] in (36,43,50,57,64,71):
        hashes[(row["preset_id"], row["midi_note"])].append(hashlib.sha256(path.read_bytes()).hexdigest())
distinct = [len(set(value)) for value in hashes.values() if len(value) == 4]
report = {
    "schema": "midibrave.atlas-flow.render-qc.v1",
    "records": len(rows), "wav_contract_passed": True,
    "repeat_groups": len(distinct),
    "repeat_groups_with_variation": sum(value > 1 for value in distinct),
    "distinct_hash_count_distribution": dict(Counter(distinct)),
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

sha256sum "$RENDER_MANIFEST" "$TRAIN_MANIFEST" \
  "$ROOT/manifests/pad-top50-selection.json" > "$ROOT/contracts/input-sha256.txt"
touch "$ROOT/contracts/cpu-prepare.complete"

if [[ ! -s "$ROOT/contracts/gpu-job-id.txt" ]]; then
  job_id=$(sbatch --parsable "$CODE/scripts/atlas_flow/octopus_pad_v1_gpu8.sbatch")
  printf '%s\n' "$job_id" > "$ROOT/contracts/gpu-job-id.txt"
  echo "submitted GPU pipeline job $job_id"
else
  echo "GPU pipeline already submitted as $(cat "$ROOT/contracts/gpu-job-id.txt")"
fi
