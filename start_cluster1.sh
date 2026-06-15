#!/bin/bash
# Launch full experiment pipeline on Cluster1 (masterk8s, 4 workers)
# Run from inside tmux: tmux new -s exp && ./start_cluster1.sh

export NTFY_TOPIC="tnsm-faraz-2026"
export CLUSTER_NAME="C1-4workers"

cd "$(dirname "$0")"
source /home/studente/rl_autoscaler_project/venv/bin/activate

./run_all_experiments.sh http://192.168.122.2:30001/
