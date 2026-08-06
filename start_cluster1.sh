#!/bin/bash
# Launch full experiment pipeline on Cluster1 (masterk8s, 4 workers)
# Run from inside tmux: tmux new -s exp && ./start_cluster1.sh

export CLUSTER_NAME="C1-4workers"

cd "$(dirname "$0")"
source ~/rl_autoscaler_project/venv/bin/activate

./run_all_experiments.sh "${1:?usage: start_cluster1.sh <service-url>}"
