#!/bin/bash
# Launch full experiment pipeline on Cluster2 (masterk8s2, 2 workers)
# Run from Cluster1 — SSHes into cluster2 and starts pipeline in tmux there

ssh studente@192.168.122.20 bash <<'EOF'
cd ~/rl_autoscaler_project1/Autoscaling_mitigating-temporal-blindness
source ~/rl_autoscaler_project1/venv/bin/activate
export NTFY_TOPIC="tnsm-faraz-2026"
export CLUSTER_NAME="C2-2workers"
tmux new-session -d -s exp "./run_all_experiments.sh http://192.168.122.20:31401/ 2>&1 | tee results/pipeline_remote.log"
echo "Started on cluster2. Attach with: ssh studente@192.168.122.20 then: tmux attach -t exp"
EOF
