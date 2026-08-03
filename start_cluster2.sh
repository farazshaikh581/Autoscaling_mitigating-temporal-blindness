#!/bin/bash
# Launch full experiment pipeline on Cluster2 (masterk8s2, 2 workers)
# Run from Cluster1 — SSHes into cluster2 and starts pipeline in tmux there
#
# Usage: ./start_cluster2.sh <cluster2-host> <cluster2-service-url>

C2_HOST="${1:?usage: start_cluster2.sh <cluster2-host> <cluster2-service-url>}"
C2_URL="${2:?usage: start_cluster2.sh <cluster2-host> <cluster2-service-url>}"

ssh "$C2_HOST" bash <<EOF
cd ~/rl_autoscaler_project1/Autoscaling_mitigating-temporal-blindness
source ~/rl_autoscaler_project1/venv/bin/activate
export CLUSTER_NAME="C2-2workers"
tmux new-session -d -s exp "./run_all_experiments.sh $C2_URL 2>&1 | tee results/pipeline_remote.log"
echo "Started on cluster2. Attach with: ssh $C2_HOST then: tmux attach -t exp"
EOF
