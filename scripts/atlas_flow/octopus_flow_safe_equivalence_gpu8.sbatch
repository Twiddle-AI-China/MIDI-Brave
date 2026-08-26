#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --job-name=atlas-flow-safe-eq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/data/atlas-flow-pad-v1/logs/flow-safe-equivalence-%j.out

set -euo pipefail
umask 027

IMAGE=midibrave:atlas-flow-v3
CONFIG=/opt/midibrave/configs/atlas_flow/octopus_pad_v1.yaml
ROOT=/data/atlas-flow-pad-v1
CHECKPOINT=$ROOT/runs/pad-v1/flow/checkpoints/latest.pt
LEGACY=/data/projects/latent-cosmos-synth/atlas-flow-stage1-finish-20260824/MidiBrave-v2/src/midibrave/atlas_flow_train.py
CANDIDATE=/data/projects/latent-cosmos-synth/atlas-flow-optimization-20260824/MidiBrave-v2/src/midibrave/atlas_flow_train.py
REPORT_DIR=$ROOT/contracts/octopus-v100
EXPECTED_LEGACY_SHA=a6c48b46569a9640457679c89e92613a4369bd1c0b4f2e9fa750cb4e36fd49df
EXPECTED_CANDIDATE_SHA=dc3dc772e8c61cdd5f590a6bac3822bef9e6edb64a178e9ae109e704e427c4f4

mkdir -p "$REPORT_DIR" "$ROOT/logs"
[[ $(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-}") -eq 8 ]]
[[ $(sha256sum "$LEGACY" | awk '{print $1}') == "$EXPECTED_LEGACY_SHA" ]]
[[ $(sha256sum "$CANDIDATE" | awk '{print $1}') == "$EXPECTED_CANDIDATE_SHA" ]]

run_case() {
  local label=$1
  local source=$2
  local report=$3
  local name="atlas-flow-safe-eq-${SLURM_JOB_ID}-${label}"
  docker run --rm --name "$name" --network none --ipc private --shm-size 16g \
    --cpus 92 --memory 106g --memory-swap 106g --pids-limit 16384 \
    --user "$(id -u):$(id -g)" \
    --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" \
    --env MIDIBRAVE_GRAD_SYNC=per_parameter \
    --env NCCL_P2P_LEVEL=NVL --env NCCL_IB_DISABLE=1 --env NCCL_TIMEOUT=1800 \
    --mount type=bind,src=/data,dst=/data,bind-propagation=rprivate \
    --mount type=bind,src="$source",dst=/opt/midibrave/src/midibrave/atlas_flow_train.py,readonly \
    "$IMAGE" torchrun --standalone --nproc_per_node=8 \
      -m midibrave.atlas_flow_train qualify \
      --config "$CONFIG" --stage flow --checkpoint "$CHECKPOINT" \
      --batch-per-gpu 96 --grad-accum 1 --warmup-updates 5 --updates 40 \
      --report "$report"
}

LEGACY_REPORT=$REPORT_DIR/flow-safe-equivalence-legacy-${SLURM_JOB_ID}.json
CANDIDATE_REPORT=$REPORT_DIR/flow-safe-equivalence-candidate-${SLURM_JOB_ID}.json
COMPARISON=$REPORT_DIR/flow-safe-equivalence-comparison-${SLURM_JOB_ID}.json

run_case legacy "$LEGACY" "$LEGACY_REPORT"
sleep 5
run_case candidate "$CANDIDATE" "$CANDIDATE_REPORT"

docker run --rm --network none --cpus 2 --memory 4g --memory-swap 4g --pids-limit 512 \
  --user "$(id -u):$(id -g)" --mount type=bind,src=/data,dst=/data \
  "$IMAGE" python -c "import json; a=json.load(open('$LEGACY_REPORT')); b=json.load(open('$CANDIDATE_REPORT')); first=abs(a['loss_first']-b['loss_first']); last=abs(a['loss_last']-b['loss_last']); out={'legacy_examples_per_second':a['effective_examples_per_second'],'candidate_examples_per_second':b['effective_examples_per_second'],'relative_gain':b['effective_examples_per_second']/a['effective_examples_per_second']-1,'loss_first_abs_diff':first,'loss_last_abs_diff':last,'legacy_passed':a['passed'],'candidate_passed':b['passed'],'passed':a['passed'] and b['passed'] and first<1e-7 and last<1e-7}; open('$COMPARISON','w').write(json.dumps(out,indent=2)+'\\n'); print(json.dumps(out,indent=2)); assert out['passed']"
