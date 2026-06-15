#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Full TNSM revision experiment pipeline
#
# Runs all agents in series on a single cluster. Run the identical script on
# both clusters simultaneously — the only difference is the URL and cluster
# name. Seeds are identical across clusters for direct comparison.
#
# Usage:
#   # Set your ntfy topic once (subscribe to it on Android via the ntfy app):
#   export NTFY_TOPIC="tnsm-faraz-2026"        # pick anything unique
#   export CLUSTER_NAME="Cluster1-1worker"     # or "Cluster2-4workers"
#
#   # Run inside tmux so it survives SSH disconnect:
#   tmux new -s exp
#   ./run_all_experiments.sh http://192.168.122.2:30001/
#   # Ctrl+B D to detach; tmux attach -t exp to return
#
# Order: Static HPA (5 seeds) → DDQN (seed 42) → Single-LSTM (5 seeds) → Double-LSTM (5 seeds)
# Seeds: 42  123  456  789  1337  (same on both clusters)
#
# Results layout:
#   results/
#     static_hpa/seed42/     test_log_<ts>.csv  run.log
#     ddqn/seed42/           train_log_<ts>.csv  test_log_<ts>.csv  tb/
#     single_lstm/seed42/    train_log_<ts>.csv  test_log_<ts>.csv  tb/  *.log
#     double_lstm/seed42/    train_log_<ts>.csv  test_log_<ts>.csv  attn_weights_<ts>.csv  tb/
# =============================================================================

URL="${1:-http://192.168.122.2:30001/}"
# 4 seeds: 42 (primary, used by DDQN convergence test) + 3 others for multi-seed stats
SEEDS=(42 123 456 789)
RESULTS="results"
TRACE="AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

# ── ntfy push notifications (Android) ────────────────────────────────────────
# Install the "ntfy" app on your phone, subscribe to $NTFY_TOPIC.
# No account or server setup needed — uses the free ntfy.sh public service.
NTFY_TOPIC="${NTFY_TOPIC:-tnsm-faraz-2026}"
CLUSTER_NAME="${CLUSTER_NAME:-Cluster}"
NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"

notify() {
    local title="$1"
    local body="$2"
    local priority="${3:-default}"   # min low default high urgent
    local tags="${4:-bell}"
    curl -s \
        -H "Title: [${CLUSTER_NAME}] ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${body}" \
        "${NTFY_URL}" >/dev/null 2>&1 || true   # never let ntfy fail the pipeline
}

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

[ -f "$TRACE" ] || { log "ERROR: Trace file missing: $TRACE"; notify "STARTUP FAILED" "Trace file not found. Pipeline aborted." "urgent" "rotating_light,sos"; exit 1; }

python3 -c "
import requests, sys
try:
    r = requests.get('${URL}factor?n=1', timeout=5)
    assert r.status_code == 200
except Exception as e:
    print(f'Service check failed: {e}'); sys.exit(1)
" || { log "ERROR: Service unreachable at $URL"; notify "STARTUP FAILED" "factorizator unreachable at $URL" "urgent" "rotating_light,sos"; exit 1; }

PIPELINE_START=$SECONDS
# Estimated total wall-clock time at 60s/step:
#  Static HPA:   4 seeds × 1000 steps =  4000 steps ≈  68h
#  DRQN:         4 seeds × 3500 steps = 14000 steps ≈ 232h
#  Single-LSTM:  4 seeds × 3500 steps = 14000 steps ≈ 232h
#  Double-LSTM:  4 seeds × 3500 steps = 14000 steps ≈ 232h
#  Total ≈ 764h ≈ 32 days

sep "EXPERIMENT PIPELINE START"
log "Cluster   : $CLUSTER_NAME"
log "URL       : $URL"
log "Seeds     : ${SEEDS[*]}"
log "ntfy topic: $NTFY_TOPIC"
log "Est. total: ~32 days (running unattended)"
log "Pipeline log: $PIPELINE_LOG"

notify "Pipeline STARTED" "URL: $URL | Seeds: ${SEEDS[*]} | Est. ~32 days | Order: HPA → DDQN → Single-LSTM → Double-LSTM" "default" "rocket"

# =============================================================================
# PHASE 1 — Static HPA  (evaluation only, ~17h per seed)
# =============================================================================
sep "PHASE 1/4: Static HPA (4 seeds, ~68h total)"
notify "Phase 1/4 START: Static HPA" "4 seeds × ~17h each | Est. ~68h" "default" "hourglass_flowing_sand"
PHASE1_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/static_hpa/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "Static HPA | seed=$SEED | Start"
    notify "HPA seed=$SEED START" "Seed $((i+1))/${#SEEDS[@]} | ETA this seed ~17h" "low" "hourglass"

    if run_step "$OUT/run.log" static-hpa50.py \
            --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        ELAPSED=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE1_START ))
        notify "HPA seed=$SEED DONE ✓" "Took $(dur $ELAPSED) | $(( i+1 ))/${#SEEDS[@]} seeds | ETA remaining: $(eta $PHASE_ELAPSED $(( i+1 )) $(( ${#SEEDS[@]} - i - 1 )))" "default" "white_check_mark"
        log "Static HPA seed=$SEED done in $(dur $ELAPSED)"
    else
        notify "HPA seed=$SEED FAILED ✗" "Check $OUT/run.log — pipeline continuing" "high" "warning"
        log "WARNING: Static HPA seed=$SEED failed — continuing"
    fi
done

notify "Phase 1/4 COMPLETE: Static HPA" "All 4 seeds done | Took $(dur $(( SECONDS - PHASE1_START )))" "default" "white_check_mark,tada"
log "Phase 1 complete in $(dur $(( SECONDS - PHASE1_START )))"

# =============================================================================
# PHASE 2 — DDQN  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 2/4: DDQN — Double Deep Q-Network (4 seeds, ~232h total)"
notify "Phase 2/4 START: DDQN" "4 seeds × ~58h | Fixed action space + target network | Est. ~232h (~10 days)" "default" "hourglass_flowing_sand"
PHASE2_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/ddqn/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "DDQN | seed=$SEED | TRAIN"
    notify "DDQN seed=$SEED TRAIN START" "Seed $((i+1))/${#SEEDS[@]} | ~41h" "low" "hourglass"

    if run_step "$OUT/train.log" ddqn_agent.py \
            --mode train --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "DDQN seed=$SEED training done in $(dur $TRAIN_TIME)"
        notify "DDQN seed=$SEED TRAIN DONE ✓" "Took $(dur $TRAIN_TIME) | Starting test" "low" "white_check_mark"
    else
        notify "DDQN seed=$SEED TRAIN FAILED ✗" "Check $OUT/train.log — skipping test for this seed" "high" "warning"
        log "WARNING: DDQN seed=$SEED training failed — skipping test"
        continue
    fi

    log "DDQN | seed=$SEED | TEST"
    notify "DDQN seed=$SEED TEST START" "~17h" "low" "hourglass"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" ddqn_agent.py \
            --mode test --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE2_START ))
        notify "DDQN seed=$SEED DONE ✓" "Took $(dur $SEED_TIME) | Seed $((i+1))/${#SEEDS[@]} | ETA remaining: $(eta $PHASE_ELAPSED $(( i+1 )) $(( ${#SEEDS[@]} - i - 1 )))" "default" "white_check_mark"
        log "DDQN seed=$SEED complete in $(dur $SEED_TIME)"
    else
        notify "DDQN seed=$SEED TEST FAILED ✗" "Check $OUT/test.log" "high" "warning"
        log "WARNING: DDQN seed=$SEED test failed"
    fi
done

notify "Phase 2/4 COMPLETE: DDQN" "All 4 seeds done | Took $(dur $(( SECONDS - PHASE2_START )))" "default" "white_check_mark,tada"
log "Phase 2 complete in $(dur $(( SECONDS - PHASE2_START )))"

# =============================================================================
# PHASE 3 — Single-LSTM  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 3/4: Single-LSTM ablation (4 seeds, ~232h total)"
notify "Phase 3/4 START: Single-LSTM" "4 seeds × ~58h each | Est. ~232h (~10 days)" "default" "hourglass_flowing_sand"
PHASE3_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/single_lstm/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "Single-LSTM | seed=$SEED | TRAIN"
    notify "Single-LSTM seed=$SEED TRAIN START" "Seed $((i+1))/${#SEEDS[@]} | ~41h" "low" "hourglass"

    if run_step "$OUT/train.log" single-lstm_agent.py \
            --mode train --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "Single-LSTM seed=$SEED training done in $(dur $TRAIN_TIME)"
        notify "Single-LSTM seed=$SEED TRAIN DONE ✓" "Took $(dur $TRAIN_TIME) | Starting test" "low" "white_check_mark"
    else
        notify "Single-LSTM seed=$SEED TRAIN FAILED ✗" "Check $OUT/train.log — skipping test for this seed" "high" "warning"
        log "WARNING: Single-LSTM seed=$SEED training failed — skipping test"
        continue
    fi

    log "Single-LSTM | seed=$SEED | TEST"
    notify "Single-LSTM seed=$SEED TEST START" "~17h" "low" "hourglass"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" single-lstm_agent.py \
            --mode test --url "$URL" --seed "$SEED" --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE3_START ))
        notify "Single-LSTM seed=$SEED DONE ✓" "Took $(dur $SEED_TIME) | Seed $((i+1))/${#SEEDS[@]} | ETA remaining seeds: $(eta $PHASE_ELAPSED $(( i+1 )) $(( ${#SEEDS[@]} - i - 1 )))" "default" "white_check_mark"
        log "Single-LSTM seed=$SEED complete in $(dur $SEED_TIME)"
    else
        notify "Single-LSTM seed=$SEED TEST FAILED ✗" "Check $OUT/test.log" "high" "warning"
        log "WARNING: Single-LSTM seed=$SEED test failed"
    fi
done

notify "Phase 3/4 COMPLETE: Single-LSTM" "All 4 seeds done | Took $(dur $(( SECONDS - PHASE3_START )))" "default" "white_check_mark,tada"
log "Phase 3 complete in $(dur $(( SECONDS - PHASE3_START )))"

# =============================================================================
# PHASE 4 — Double-LSTM  (4 seeds, ~58h each, ~232h total)
# =============================================================================
sep "PHASE 4/4: Double-LSTM proposed agent (4 seeds, ~232h total)"
notify "Phase 4/4 START: Double-LSTM" "4 seeds × ~58h each | Est. ~232h (~10 days) | FINAL PHASE" "high" "hourglass_flowing_sand"
PHASE4_START=$SECONDS

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="$RESULTS/double_lstm/seed${SEED}"
    mkdir -p "$OUT"
    SEED_START=$SECONDS

    log "Double-LSTM | seed=$SEED | TRAIN"
    notify "Double-LSTM seed=$SEED TRAIN START" "Seed $((i+1))/${#SEEDS[@]} | ~41h | Window=8" "low" "hourglass"

    if run_step "$OUT/train.log" double-lstm_agent.py \
            --mode train --url "$URL" --seed "$SEED" --window 8 --log-dir "$OUT"; then
        TRAIN_TIME=$(( SECONDS - SEED_START ))
        log "Double-LSTM seed=$SEED training done in $(dur $TRAIN_TIME)"
        notify "Double-LSTM seed=$SEED TRAIN DONE ✓" "Took $(dur $TRAIN_TIME) | Starting test" "low" "white_check_mark"
    else
        notify "Double-LSTM seed=$SEED TRAIN FAILED ✗" "Check $OUT/train.log — skipping test for this seed" "high" "warning"
        log "WARNING: Double-LSTM seed=$SEED training failed — skipping test"
        continue
    fi

    log "Double-LSTM | seed=$SEED | TEST"
    notify "Double-LSTM seed=$SEED TEST START" "~17h | Attention weights will be exported" "low" "hourglass"
    TEST_START=$SECONDS

    if run_step "$OUT/test.log" double-lstm_agent.py \
            --mode test --url "$URL" --seed "$SEED" --window 8 --log-dir "$OUT"; then
        SEED_TIME=$(( SECONDS - SEED_START ))
        PHASE_ELAPSED=$(( SECONDS - PHASE4_START ))
        notify "Double-LSTM seed=$SEED DONE ✓" "Took $(dur $SEED_TIME) | Seed $((i+1))/${#SEEDS[@]} | ETA remaining: $(eta $PHASE_ELAPSED $(( i+1 )) $(( ${#SEEDS[@]} - i - 1 )))" "default" "white_check_mark"
        log "Double-LSTM seed=$SEED complete in $(dur $SEED_TIME)"
    else
        notify "Double-LSTM seed=$SEED TEST FAILED ✗" "Check $OUT/test.log" "high" "warning"
        log "WARNING: Double-LSTM seed=$SEED test failed"
    fi
done

notify "Phase 4/4 COMPLETE: Double-LSTM" "All 4 seeds done | Took $(dur $(( SECONDS - PHASE4_START )))" "default" "white_check_mark,tada"
log "Phase 4 complete in $(dur $(( SECONDS - PHASE4_START )))"

# =============================================================================
sep "ALL EXPERIMENTS COMPLETE"
TOTAL=$(( SECONDS - PIPELINE_START ))
log "Total wall time: $(dur $TOTAL)"
log "Results:"
find "$RESULTS" -name "*.csv" | sort | sed 's/^/  /'
log "Pipeline log: $PIPELINE_LOG"
log "TensorBoard:  tensorboard --logdir $RESULTS --port 6006"

notify "ALL EXPERIMENTS COMPLETE" "Total time: $(dur $TOTAL) | Results in $RESULTS/ | Run: tensorboard --logdir $RESULTS --port 6006 to view training curves" "urgent" "tada,trophy,white_check_mark"
