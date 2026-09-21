#!/usr/bin/env bash
# Push the demo to Spark as soon as it answers again, then restart the job.
#
# The natapp tunnel accepts TCP even when the far end is dead, so ssh can hang
# past ConnectTimeout; perl's alarm is the hard cap.
set -u

REPO=${ATLAS_DEMO_REPO:-/Users/rolf/Desktop/twiddle-research/midi-brave}
REMOTE=${ATLAS_DEMO_REMOTE:-/home/rolf/projects/atlas-flow-demo}
TRIES=${ATLAS_DEMO_TRIES:-60}

reach() { perl -e 'alarm 90; exec @ARGV' ssh -o BatchMode=yes -o ConnectTimeout=10 spark true; }

for attempt in $(seq 1 "$TRIES"); do
  if reach; then
    echo "$(date '+%F %T') spark reachable on attempt $attempt, deploying"
    rsync -az --partial --delete --exclude '__pycache__' -e ssh \
      "$REPO/src" "$REPO/atlas-flow-web-demo" "$REPO/scripts" "$REMOTE/repo/" || exit 1
    ssh spark "old=\$(squeue -h -n rolf_atlas_flow_demo -o '%i'); [ -n \"\$old\" ] && scancel \$old; sleep 8
      cd $REMOTE && SBATCH_SIGNAL=B:TERM@180 qgpu -n rolf_atlas_flow_demo -c 8 -m 32G -t 7-00:00:00 -- \
        bash $REMOTE/repo/scripts/atlas_flow/spark_web_demo.sh" || exit 1
    for wait in $(seq 20); do
      sleep 10
      job=$(ssh spark "squeue -h -n rolf_atlas_flow_demo -o '%i'" 2>/dev/null)
      if ssh spark "docker ps --format '{{.Names}}' | grep -qx atlas-flow-web-demo-$job \
          && curl -fsS -m 3 http://127.0.0.1:18795/api/health" >/dev/null 2>&1; then
        echo "$(date '+%F %T') job $job serving"
        exit 0
      fi
    done
    echo "$(date '+%F %T') deployed but health check never passed" >&2
    exit 1
  fi
  sleep 60
done
echo "$(date '+%F %T') spark never came back in $TRIES attempts" >&2
exit 1
