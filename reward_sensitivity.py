"""
reward_sensitivity.py — Post-hoc reward weight sensitivity analysis (R1/R3)

Uses already-completed test CSVs (no re-simulation needed).
Recomputes per-step reward components from raw metrics and applies
±50% perturbations to each weight, one at a time.

Usage (after experiments complete):
    python reward_sensitivity.py --results-dir results/ --seed 42

Output:
    reward_sensitivity_table.csv   — mean reward per agent per weight config
    reward_sensitivity.png         — heatmap figure for the paper
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Reward constants (must match ddqn_agent.py / dlstm_agent.py / keda_baseline.py) ──
# CSV stores latency in ms (agents multiply by 1000 before writing),
# so convert L_TARGET and L_THRESH to ms here.
L_TARGET          = 20.0    # ms  (0.020 s in agent code)
L_THRESH          = 50.0    # ms  (0.050 s in agent code)
NATIVE_HPA_TARGET = 50.0    # CPU% target used when a CSV has no HPA_Target column (KEDA)

# Weights as actually implemented (2026-07-30 audit): the reward formula is
# 0.55*r_sla + 0.25*r_cpu + 0.08*r_stab + 0.12*r_succ everywhere (ddqn_agent.py,
# dlstm_agent.py, keda_baseline.py). There is no standalone forecast reward term
# any more -- dlstm_agent.py:248 computes "0.50*r_slo + ... + 0.05*r_slo", i.e. the
# weight that used to belong to a forecast-quality term (W_FCST) was redistributed
# onto r_sla itself, not onto a genuine forecast signal (see
# double_lstm_forecast_analysis memory: the forecast state feature has no dedicated
# reward incentive). The previous version of this script still modeled a separate
# W_FCST weight and an r_fcst() component -- that no longer corresponds to anything
# in the actual reward computation, so it's removed rather than perturbed.
BASELINE_WEIGHTS  = {
    "W_SLA":  0.55,
    "W_CPU":  0.25,
    "W_STAB": 0.08,
    "W_SUCC": 0.12,
}

PERTURBATIONS = [-0.50, 0.0, +0.50]   # −50 %, baseline, +50 %

# DDQN removed 2026-07-30 (replaced by KEDA, see response_R4.2_DDQN_justification.md).
# "double_lstm" (old vanilla PPO version) superseded by "dlstm" (Double-LSTM-v2,
# DQN + AttentionDoubleLSTM extractor, see dlstm_ddqn_variant_jul19 memory).
AGENTS = ["static_hpa", "single_lstm", "dlstm", "keda"]
AGENT_LABELS = {
    "static_hpa":  "Static HPA",
    "single_lstm": "Single-LSTM",
    "dlstm":       "Double-LSTM-v2 (ours)",
    "keda":        "KEDA",
}

# ── Component reward functions ────────────────────────────────────────────────
def r_sla(lat):
    if lat <= L_TARGET:
        return 1.0
    elif lat <= L_THRESH:
        return 0.5 + 0.5 * (L_THRESH - lat) / (L_THRESH - L_TARGET)
    else:
        return max(-1.0, -0.5 * (lat - L_THRESH) / 0.1)

def r_cpu(cpu, hpa_target):
    if abs(cpu - hpa_target) <= 10.0:
        return 1.0
    return float(np.exp(-((cpu - hpa_target) / 50.0) ** 2))

def r_stab(delta):
    if delta <= 2:
        return -0.1 * delta
    return -0.5 * delta

def r_succ(succ):
    if succ >= 0.99:
        return 1.0
    return float(np.log(max(succ, 1e-6)))

def compute_components(df):
    """Return dict of per-step component arrays from a test CSV."""
    lat  = df["Latency_P90"].values
    cpu  = df["CPU_Pct"].values
    reps = df["Replicas"].values
    succ = df["Success"].values
    # KEDA's CSV has no HPA_Target column (it has no CPU target of its own) --
    # keda_baseline.py's own reward computation centers r_cpu on the native-HPA
    # default (50%) in that case, so we do the same here for consistency.
    hpa_t = df["HPA_Target"].values if "HPA_Target" in df.columns \
        else np.full(len(df), NATIVE_HPA_TARGET)

    prev_reps = np.concatenate([[reps[0]], reps[:-1]])
    delta     = np.abs(reps - prev_reps)

    return {
        "r_sla":  np.array([r_sla(l) for l in lat]),
        "r_cpu":  np.array([r_cpu(c, t) for c, t in zip(cpu, hpa_t)]),
        "r_stab": np.array([r_stab(d) for d in delta]),
        "r_succ": np.array([r_succ(s) for s in succ]),
    }

def weighted_reward(components, w_sla, w_cpu, w_stab, w_succ):
    return (w_sla  * components["r_sla"]
          + w_cpu  * components["r_cpu"]
          + w_stab * components["r_stab"]
          + w_succ * components["r_succ"])

# ── Load best test CSV per agent (seed 42 by default) ────────────────────────
def _find_csv(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_agent_components(results_dir, agent, seed=42):
    """Path layout differs by agent -- static_hpa/single_lstm/dlstm share
    results/{agent}/seed{N}/*test*.csv; KEDA writes to a differently-shaped
    results_log_keda/seed{N}_target{N}_mult{N}/keda_test_*.csv (see
    keda_baseline.py's own log_dir construction)."""
    if agent == "keda":
        # target-rpm/load-multiplier are script defaults (100, 3.0) -- glob over
        # them rather than hardcoding, in case a run used non-default values.
        pattern = os.path.join(results_dir, "..", "results_log_keda",
                                f"seed{seed}_target*_mult*", "keda_test_*.csv")
        path = _find_csv(pattern)
    else:
        prefix = {"static_hpa": "test_log", "single_lstm": "test_log", "dlstm": "dlstm_test"}[agent]
        pattern = os.path.join(results_dir, agent, f"seed{seed}", f"{prefix}_*.csv")
        path = _find_csv(pattern)

    if not path:
        print(f"  WARNING: no test CSV found for {agent}/seed{seed} (pattern: {pattern}) — skipping")
        return None
    return compute_components(pd.read_csv(path))

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Path to results/ directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Which seed's test CSV to use")
    parser.add_argument("--out-dir", default=".",
                        help="Where to save table and figure")
    args = parser.parse_args()

    print(f"Loading test CSVs from: {args.results_dir}  (seed={args.seed})")

    agent_components = {}
    for agent in AGENTS:
        comps = load_agent_components(args.results_dir, agent, args.seed)
        if comps is not None:
            agent_components[agent] = comps
            print(f"  Loaded {agent}: {len(comps['r_sla'])} steps")

    if not agent_components:
        print("No CSVs found — run experiments first.")
        return

    rows = []
    weight_keys = list(BASELINE_WEIGHTS.keys())

    for perturbed_key in weight_keys:
        for delta in PERTURBATIONS:
            # Build weight set: perturb one weight, keep others at baseline
            w = dict(BASELINE_WEIGHTS)
            w[perturbed_key] = w[perturbed_key] * (1.0 + delta)

            label = (f"{perturbed_key}×{1+delta:.0%}"
                     if delta != 0.0 else "Baseline")

            row = {"config": label, "perturbed_weight": perturbed_key,
                   "delta": delta}
            for agent, comps in agent_components.items():
                rewards = weighted_reward(
                    comps,
                    w_sla=w["W_SLA"], w_cpu=w["W_CPU"], w_stab=w["W_STAB"],
                    w_succ=w["W_SUCC"],
                )
                row[AGENT_LABELS[agent]] = float(np.mean(rewards))
            rows.append(row)

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "reward_sensitivity_table.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nTable saved: {out_csv}")
    print(results_df.to_string(index=False))

    # ── Figure: heatmap of mean reward per agent per config ──────────────────
    agent_cols = [AGENT_LABELS[a] for a in AGENTS if a in agent_components]
    baseline_row = results_df[results_df["delta"] == 0.0].iloc[0]
    baseline_vals = baseline_row[agent_cols].values.astype(float)

    # Normalise to baseline so the heatmap shows relative change
    pivot_data = []
    config_labels = []
    for _, row in results_df.iterrows():
        if row["perturbed_weight"] == weight_keys[0] and row["delta"] == 0.0:
            # Only include baseline once (at the top)
            pivot_data.append((row[agent_cols].values.astype(float)
                               - baseline_vals) / (np.abs(baseline_vals) + 1e-9))
            config_labels.append("Baseline")
        elif row["delta"] != 0.0:
            rel = (row[agent_cols].values.astype(float)
                   - baseline_vals) / (np.abs(baseline_vals) + 1e-9)
            pivot_data.append(rel)
            config_labels.append(row["config"])

    heat = pd.DataFrame(pivot_data, columns=agent_cols, index=config_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Relative change from baseline"})
    ax.set_title("Reward Weight Sensitivity (±50% perturbation, post-hoc on trained policy)",
                 fontsize=11)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Weight configuration")
    plt.tight_layout()

    out_fig = os.path.join(args.out_dir, "reward_sensitivity.png")
    plt.savefig(out_fig, dpi=150)
    print(f"Figure saved: {out_fig}")
    plt.close()

if __name__ == "__main__":
    main()
