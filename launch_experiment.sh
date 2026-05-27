#!/bin/bash

PROFILE=${1:-perf_focused}
RUN_ID=${2:-1}
SEED=${3:-42}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${PROFILE}_run${RUN_ID}_seed${SEED}_${TIMESTAMP}"
RUN_DIR="results/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

# Write metadata file
cat > "${RUN_DIR}/metadata.json" << METADATA
{
  "profile":    "${PROFILE}",
  "run_id":     ${RUN_ID},
  "seed":       ${SEED},
  "timestamp":  "${TIMESTAMP}",
  "run_name":   "${RUN_NAME}",
  "mode":       "train",
  "script":     "run_simulation.py",
  "cluster":    "minikube-2node",
  "worker_config": "1_worker",
  "notes":      "journal_extension_v1"
}
METADATA

echo "================================================"
echo "  EXPERIMENT LAUNCH"
echo "================================================"
echo "  Profile   : ${PROFILE}"
echo "  Run ID    : ${RUN_ID}"
echo "  Seed      : ${SEED}"
echo "  Timestamp : ${TIMESTAMP}"
echo "  Output    : ${RUN_DIR}"
echo "================================================"

# Run training — log to both console and file
python run_simulation.py \
  --mode train \
  --profile ${PROFILE} \
  2>&1 | tee "${RUN_DIR}/train.log"

# Write completion marker
echo "COMPLETED at $(date)" > "${RUN_DIR}/DONE"
echo ""
echo "Training complete. Results in: ${RUN_DIR}"
