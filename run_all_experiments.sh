#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Full TNSM revision experiment pipeline
#
# Runs all agents in series on a single cluster. Run the identical script on
# both clusters simultaneously — the only difference is the URL and cluster
# name. Seeds are identical across clusters for direct comparison.
#
# Usage:
#   export CLUSTER_NAME="Cluster1-1worker"     # or "Cluster2-4workers"
#
#   # Run inside tmux so it survives SSH disconnect:
#   tmux new -s exp
#   ./run_all_experiments.sh http://<cluster-ip>:<port>/
#   # Ctrl+B D to detach; tmux attach -t exp to return
#
# Order: Static HPA (1 run) → DDQN (6 seeds) → Single-LSTM (6 seeds) → Double-LSTM (6 seeds)
# Seeds: 123  456  789  1337  2024  42  (same on both clusters)
#
# Results layout:
#   results/
#     static_hpa/seed42/     test_log_<ts>.csv  run.log
#     ddqn/seed42/           train_log_<ts>.csv  test_log_<ts>.csv  tb/
#     single_lstm/seed42/    train_log_<ts>.csv  test_log_<ts>.csv  tb/  *.log
#     double_lstm/seed42/    train_log_<ts>.csv  test_log_<ts>.csv  attn_weights_<ts>.csv  tb/
# =============================================================================

URL="${1:?usage: run_all_experiments.sh <service-url>}"
# 6 seeds for RL methods (Double-LSTM, Single-LSTM, DDQN); 42 is last (reference seed).
# Static HPA is deterministic — run once only (seed=42 for RNG in day selection).
SEEDS=(123 456 789 1337 2024 42)
HPA_SEED=42
RESULTS="results"
TRACE="AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
CLUSTER_NAME="${CLUSTER_NAME:-Cluster}"

# ── time helpers ─────────────────────────────────────────────────────────────
dur() {
    local s=$1
    local d=$(( s / 86400 ))
    local h=$(( (s % 86400) / 3600 ))
    local m=$(( (s % 3600) / 60 ))
    if   [ "$d" -gt 0 ]; then echo "${d}d ${h}h ${m}m"
    elif [ "$h" -gt 0 ]; then echo "${h}h ${m}m"
    else echo "${m}m"; fi
}

eta() {
    # eta <elapsed_total_secs> <seeds_done> <seeds_left>
    local elapsed=$1 done=$2 left=$3
    [ "$done" -eq 0 ] && echo "calculating..." && return
    local avg=$(( elapsed / done ))
    dur $(( avg * left ))
}

# ── logging ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
sep() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ── run helper: run a python command, tee to logfile, return exit code ────────
run_step() {
    local logfile="$1"; shift
    python3 "$@" 2>&1 | tee "$logfile"
    return "${PIPESTATUS[0]}"
}

# ── pre-flight ────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS"
PIPELINE_LOG="$RESULTS/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

[ -f "$TRACE" ] || { log "ERROR: Trace file missing: $TRACE"; exit 1; }

python3 -c "
import requests, sys
try:
    r = requests.get('${URL}factor?n=1', timeout=5)
    assert r.status_code == 200
except Exception as e:
    print(f'Service check failed: {e}'); sys.exit(1)
" || { log "ERROR: Service unreachable at $URL"; exit 1; }

PIPELINE_START=$SECONDS
# Estimated total wall-clock time at 60s/step:
#  Static HPA:   1 run  × 1000 steps =  1000 steps ≈  17h
#  DDQN:         6 seeds × 3500 steps = 21000 steps ≈ 352h
#  Single-LSTM:  6 seeds × 3500 steps = 21000 steps ≈ 352h
#  Double-LSTM:  6 seeds × 3500 steps = 21000 steps ≈ 352h
#  Total ≈ 1073h ≈ 45 days

sep "EXPERIMENT PIPELINE START"
log "Cluster   : $CLUSTER_NAME"
log "URL       : $URL"
log "RL Seeds  : ${SEEDS[*]}"
log "HPA seed  : $HPA_SEED (1 run)"
log "Est. total: ~45 days (running unattended)"
log "Pipeline log: $PIPELINE_LOG"

# =============================================================================
# PHASE 1 — Static HPA  (evaluation only, ~17h per seed)
# =============================================================================
sep "PHASE 1/4: Static HPA (1 run — deterministic baseline)"
PHASE1_START=$SECONDS

OUT="$RESULTS/static_hpa/seed${HPA_SEED}"
mkdir -p "$OUT"

log "Static HPA | seed=$HPA_SEED | Start"

if run_step "$OUT/run.log" static-hpa50.py \
        --url "$URL" --seed "$HPA_SEED" --log-dir "$OUT"; then
    ELAPSED=$(( SECONDS - PHASE1_START ))
    log "Static HPA done in $(dur $ELAPSED)"
else
    log "WARNING: Static HPA failed — continuing"
fi

log "Phase 1 complete in $(dur $(( SECONDS - PHASE1_START )))"

# =============================================================================
# PHASE 2 — DDQN  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 2/4: DDQN — Double Deep Q-Network (4 seeds, ~232h total)"
PHASE2_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/ddqn/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "DDQN | seed=$SEED | TRAIN"

    if run_step "$OUT/train.log" ddqn_agent.py \
            --mode train --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "DDQN seed=$SEED training done in $(dur $TRAIN_TIME)"
    else
        log "WARNING: DDQN seed=$SEED training failed — skipping test"
        continue
    fi

    log "DDQN | seed=$SEED | TEST"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" ddqn_agent.py \
            --mode test --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE2_START ))
        log "DDQN seed=$SEED complete in $(dur $SEED_TIME)"
    else
        log "WARNING: DDQN seed=$SEED test failed"
    fi
done

log "Phase 2 complete in $(dur $(( SECONDS - PHASE2_START )))"

# =============================================================================
# PHASE 3 — Single-LSTM  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 3/4: Single-LSTM ablation (4 seeds, ~232h total)"
PHASE3_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/single_lstm/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "Single-LSTM | seed=$SEED | TRAIN"

    if run_step "$OUT/train.log" single-lstm_agent.py \
            --mode train --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "Single-LSTM seed=$SEED training done in $(dur $TRAIN_TIME)"
    else
        log "WARNING: Single-LSTM seed=$SEED training failed — skipping test"
        continue
    fi

    log "Single-LSTM | seed=$SEED | TEST"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" single-lstm_agent.py \
            --mode test --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE3_START ))
        log "Single-LSTM seed=$SEED complete in $(dur $SEED_TIME)"
    else
        log "WARNING: Single-LSTM seed=$SEED test failed"
    fi
done

log "Phase 3 complete in $(dur $(( SECONDS - PHASE3_START )))"

# =============================================================================
# PHASE 4 — Double-LSTM  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 4/4: Double-LSTM proposed agent (4 seeds, ~232h total)"
PHASE4_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/double_lstm/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "Double-LSTM | seed=$SEED | TRAIN"

    if run_step "$OUT/train.log" double-lstm_agent.py \
            --mode train --url "$URL" --seed "$SEED" --window 8 --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "Double-LSTM seed=$SEED training done in $(dur $TRAIN_TIME)"
    else
        log "WARNING: Double-LSTM seed=$SEED training failed — skipping test"
        continue
    fi

    log "Double-LSTM | seed=$SEED | TEST"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" double-lstm_agent.py \
            --mode test --url "$URL" --seed "$SEED" --window 8 --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE4_START ))
        log "Double-LSTM seed=$SEED complete in $(dur $SEED_TIME)"
    else
        log "WARNING: Double-LSTM seed=$SEED test failed"
    fi
done

log "Phase 4 complete in $(dur $(( SECONDS - PHASE4_START )))"

# =============================================================================
sep "ALL EXPERIMENTS COMPLETE"
TOTAL=$(( SECONDS - PIPELINE_START ))
log "Total wall time: $(dur $TOTAL)"
log "Results:"
find "$RESULTS" -name "*.csv" | sort | sed 's/^/  /'
log "Pipeline log: $PIPELINE_LOG"
log "TensorBoard:  tensorboard --logdir $RESULTS --port 6006"
