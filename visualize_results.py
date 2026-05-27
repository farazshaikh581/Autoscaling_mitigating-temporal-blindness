import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "Results-CSVs"
SLA_MS = 15  # SLA threshold in milliseconds

FILES = {
    "Double-LSTM (Proposed)": "test_log_Double-LSTM.csv",
    "Single-LSTM":            "test_log_Single-LSTM.csv",
    "DDQN":                   "test_log_DDQN.csv",
    "Static HPA":             "test_log_static-hpa.csv",
}

sns.set_theme(style="whitegrid")


def load_data():
    data = {}
    for label, fname in FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        data[label] = pd.read_csv(path)
    return data


def plot_reward(data):
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        ax.plot(df["Step"], df["Reward"], label=label, alpha=0.8)
    ax.set_title("Reward per Step (Test Phase)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reward_comparison.png", dpi=150)
    print("Saved reward_comparison.png")
    plt.close()


def plot_latency(data):
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        ax.plot(df["Step"], df["Latency_P90"], label=f"{label} (P90)", alpha=0.8)
    ax.axhline(SLA_MS, color="red", linestyle="--", linewidth=1.5, label=f"SLA = {SLA_MS} ms")
    ax.set_title("P90 Latency per Step (Test Phase)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency P90 (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("latency_comparison.png", dpi=150)
    print("Saved latency_comparison.png")
    plt.close()


def plot_replicas(data):
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, df in data.items():
        ax.plot(df["Step"], df["Replicas"], label=label, alpha=0.8)
    ax.set_title("Replica Count per Step (Test Phase)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Replicas")
    ax.legend()
    plt.tight_layout()
    plt.savefig("replicas_comparison.png", dpi=150)
    print("Saved replicas_comparison.png")
    plt.close()


def print_summary(data):
    print("\n" + "=" * 70)
    print(f"{'Method':<28} {'Avg Reward':>10} {'Avg P90 (ms)':>14} {'SLA Comp.%':>11} {'Avg Replicas':>13}")
    print("-" * 70)
    for label, df in data.items():
        avg_reward   = df["Reward"].mean()
        avg_lat      = df["Latency_P90"].mean()
        sla_pct      = (df["Latency_P90"] <= SLA_MS).mean() * 100
        avg_replicas = df["Replicas"].mean()
        print(f"{label:<28} {avg_reward:>10.4f} {avg_lat:>14.2f} {sla_pct:>10.1f}% {avg_replicas:>13.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    data = load_data()
    if not data:
        print(f"No CSV files found in {RESULTS_DIR}/")
    else:
        print_summary(data)
        plot_reward(data)
        plot_latency(data)
        plot_replicas(data)
