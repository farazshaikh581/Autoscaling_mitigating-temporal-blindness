#!/bin/bash
# Usage: ./launch_experiment.sh <agent> <url> [mode] [seed] [window] [no_aux]
# agent:   double-lstm | single-lstm | static-hpa | keda
# url:     http://<node-ip>:<port>/
# mode:    train (default) | test
# seed:    42 (default) — vary for multi-seed evaluation
# window:  8 (default, double-lstm only) — sensitivity: 3 10 30
# no_aux:  0 (default) | 1 — disable throughput multiplier/enhancement (fair HPA comparison)

AGENT=${1:-double-lstm}
URL=${2:-http://192.168.122.2:30001/}
MODE=${3:-train}
SEED=${4:-42}
WINDOW=${5:-8}
NO_AUX=${6:-0}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="results/${AGENT}_${MODE}_seed${SEED}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "================================================"
echo "  EXPERIMENT LAUNCH"
echo "  Agent     : ${AGENT}"
echo "  URL       : ${URL}"
echo "  Mode      : ${MODE}"
echo "  Seed      : ${SEED}"
echo "  Window    : ${WINDOW}"
echo "  No-Aux    : ${NO_AUX}"
echo "  Output    : ${RUN_DIR}"
echo "================================================"

NO_AUX_FLAG=""
if [ "${NO_AUX}" = "1" ]; then NO_AUX_FLAG="--no-aux"; fi

case "$AGENT" in
  double-lstm)
    python double-lstm_agent.py --mode "$MODE" --url "$URL" \
      --seed "$SEED" --window "$WINDOW" $NO_AUX_FLAG \
      --log-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  single-lstm)
    python single-lstm_agent.py --mode "$MODE" --url "$URL" \
      --seed "$SEED" --log-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  static-hpa)
    python static-hpa50.py --url "$URL" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  ddqn)
    python ddqn_agent.py --mode "$MODE" --url "$URL" \
      --seed "$SEED" $NO_AUX_FLAG \
      --log-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  drqn)
    python drqn_agent.py --mode "$MODE" --url "$URL" \
      --seed "$SEED" $NO_AUX_FLAG \
      --log-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  keda)
    python keda_baseline.py --url "$URL" --log-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"
    ;;
  *)
    echo "Unknown agent: $AGENT"
    echo "Choose from: double-lstm | single-lstm | ddqn | static-hpa | keda"
    exit 1
    ;;
esac

echo "Done. Log saved to ${RUN_DIR}/run.log"
