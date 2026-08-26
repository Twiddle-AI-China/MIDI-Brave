#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --job-name=atlas-flow-eval-v5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=/data/atlas-flow-pad-v1/logs/evaluate-v5-%j.out

set -euo pipefail
readonly IMAGE=midibrave:atlas-flow-v5
readonly ROOT=/data/atlas-flow-pad-v1
readonly CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
readonly CHECKPOINT=$ROOT/runs/pad-v1/joint/checkpoints/step-000030000.pt
readonly OUTPUT=$ROOT/evaluation/pad-v1
readonly CONTRACT=$ROOT/contracts/evaluation-v5

mkdir -p "$OUTPUT" "$CONTRACT" "$ROOT/logs"
[[ -s "$CHECKPOINT" ]] || { echo "joint checkpoint is missing" >&2; exit 72; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "SLURM did not allocate a GPU" >&2; exit 70; }
[[ $(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES") -eq 1 ]] || { echo "expected exactly one GPU" >&2; exit 70; }

docker run --rm --name "atlas-flow-eval-v5-${SLURM_JOB_ID}" \
  --network none --ipc private --shm-size 3g --cpus 8 \
  --memory 14g --memory-swap 14g --pids-limit 2048 \
  --security-opt no-new-privileges:true --cap-drop ALL \
  --user "$(id -u):$(id -g)" \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
  "$IMAGE" python -m midibrave.atlas_flow_evaluate \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" --device cuda:0

python3 - "$OUTPUT/evaluation.json" "$CONTRACT/gpu-evaluation.complete" "$SLURM_JOB_ID" <<'PY'
import json, os, pathlib, sys, time
report_path, marker_path, job_id = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report.get("schema") == "midibrave.atlas-flow.evaluation.v2"
assert report.get("pipeline_state") == "complete"
assert isinstance(report.get("passed"), bool)
assert report.get("population", {}).get("records") == 108
assert report.get("audition", {}).get("records") == 18
assert report.get("audition", {}).get("notes") == [36, 43, 50, 57, 64, 71]
payload = {
    "schema": "midibrave.atlas-flow.pipeline-marker.v1",
    "stage": "gpu-evaluation",
    "state": "complete",
    "quality_state": report.get("quality_state"),
    "metric_records": 108,
    "audition_records": 18,
    "slurm_job_id": str(job_id),
    "created_unix": time.time(),
}
temporary = marker_path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
PY
