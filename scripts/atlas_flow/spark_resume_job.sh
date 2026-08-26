#!/bin/bash
set -euo pipefail

readonly PAUSE_MARKER=/data/atlas-flow-pad-v1/contracts/spark-gb10/pause-requested
readonly JOB_SCRIPT=/data/projects/latent-cosmos-synth/atlas-flow-spark/MidiBrave-v2/scripts/atlas_flow/spark_pad_v1_gpu1.sbatch

[[ -s "$JOB_SCRIPT" ]] || { echo "Spark job script is missing: $JOB_SCRIPT" >&2; exit 66; }
rm -f "$PAUSE_MARKER"
sbatch "$JOB_SCRIPT"
