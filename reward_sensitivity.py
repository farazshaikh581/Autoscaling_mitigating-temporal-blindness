"""
reward_sensitivity.py — Post-hoc reward weight sensitivity analysis (R1/R3)

Uses already-completed test CSVs (no re-simulation needed).
Recomputes per-step reward components from raw metrics and applies
±50% perturbations to each weight, one at a time.

Usage (after experiments complete):
    python reward_sensitivity.py --results-dir results/

Output:
    reward_sensitivity_table.csv   — mean reward per agent per weight config
    reward_sensitivity.png         — heatmap figure for the paper
"""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Reward constants (must match double-lstm_agent.py) ───────────────────────
L_TARGET          = 0.020
L_THRESH          = 0.050
NOMINAL_CAP       = 100.0   # req/min per replica at target CPU

BASELINE_WEIGHTS  = {
    "W_SLA":  0.50,
    "W_CPU":  0.25,
    "W_STAB": 0.08,
    "W_FCST": 0.05,
    "W_SUCC": 0.12,
}

PERTURBATIONS = [-0.50, 0.0, +0.50]   # −50 %, baseline, +50 %

AGENTS = ["static_hpa", "drqn", "single_lstm", "double_lstm"]
AGENT_LABELS = {
    "static_hpa":   "Static HPA",
    "drqn":         "DRQN",
    "single_lstm":  "Single-LSTM",
    "double_lstm":  "Double-LSTM (ours)",
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

def r_fcst(eff_req, forecast, C=NOMINAL_CAP):
    err = (eff_req - forecast) / max(C, 1e-6)
    return -(err ** 2)

def compute_components(df):
    """Return dict of per-step component arrays from a test CSV."""
    lat   = df["Latency_P90"].values
    cpu   = df["CPU_Pct"].values
    reps  = df["Replicas"].values
    succ  = df["Success"].values
    hpa_t = df["HPA_Target"].values
    ereq  = (df["Requests"] * df["Throughput"]).values
    fcast = df["Forecast"].values

    prev_reps = np.concatenate([[reps[0]], reps[:-1]])
    delta     = np.abs(reps - prev_reps)

    return {
        "r_sla":  np.array([r_sla(l) for l in lat]),
        "r_cpu":  np.array([r_cpu(c, t) for c, t in zip(cpu, hpa_t)]),
        "r_stab": np.array([r_stab(d) for d in delta]),
        "r_succ": np.array([r_succ(s) for s in succ]),
        "r_fcst": np.array([r_fcst(e, f) for e, f in zip(ereq, fcast)]),
    }

def weighted_reward(components, w_sla, w_cpu, w_stab, w_fcst, w_succ):
    return (w_sla  * components["r_sla"]
          + w_cpu  * components["r_cpu"]
          + w_stab * components["r_stab"]
          + w_fcst * components["r_fcst"]
          + w_succ * components["r_succ"])

# ── Load best test CSV per agent (seed 42 by default) ────────────────────────
def load_agent_components(results_dir, agent, seed=42):
    pattern = os.path.join(results_dir, agent, f"seed{seed}", "test_log_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no test CSV found for {agent}/seed{seed} — skipping")
        return None
    df = pd.read_csv(files[-1])
    return compute_components(df)

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
                    w_fcst=w["W_FCST"], w_succ=w["W_SUCC"],
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
