#!/bin/bash
# Usage: ./launch_experiment.sh <agent> <url> [mode] [seed]
# agent: double-lstm | single-lstm | ddqn | static-hpa
# url:   http://<service-ip>:8080
# mode:  train (default) | test
# seed:  42 (default)

AGENT=${1:-double-lstm}
URL=${2:-http://localhost:8080}
MODE=${3:-train}
SEED=${4:-42}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="results/${AGENT}_${MODE}_seed${SEED}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "================================================"
echo "  EXPERIMENT LAUNCH"
echo "  Agent     : ${AGENT}"
echo "  URL       : ${URL}"
echo "  Mode      : ${MODE}"
echo "  Seed      : ${SEED}"
echo "  Output    : ${RUN_DIR}"
echo "================================================"

case "$AGENT" in
  double-lstm)
    python double-lstm_agent.py --mode "$MODE" --url "$URL" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  single-lstm)
    python single-lstm_agent.py --mode "$MODE" --url "$URL" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  ddqn)
    python ddqn_agent.py --mode "$MODE" --url "$URL" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  static-hpa)
    python static-hpa50.py --url "$URL" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  *)
    echo "Unknown agent: $AGENT"
    echo "Choose from: double-lstm | single-lstm | ddqn | static-hpa"
    exit 1
    ;;
esac

echo "Done. Log saved to ${RUN_DIR}/run.log"
