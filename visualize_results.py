import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
SLA_LATENCY_S = 0.015
RESULTS_FILE = 'test_results.csv'

# --- Styling ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

def plot_performance_overview(df):
    """ Plots Reward, Forecast Error, and Requests, similar to Fig 5/6 in the paper. """
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    
    df['Forecast_Error'] = (df['Forecast'] - df['Requests']) / df['Requests'].replace(0, 1)
    sns.lineplot(x=df.index, y='Forecast_Error', data=df, ax=axes[0], color='crimson', label='Normalized Forecast Error')
    axes[0].set_title('Normalized Forecast Error per Step (Testing Phase)', fontsize=16)
    axes[0].set_ylabel('Normalized Error')
    axes[0].axhline(0, color='k', linestyle='--', lw=1)
    axes[0].set_ylim(-1.5, 1.5)

    sns.lineplot(x=df.index, y='HPA_Target', data=df, ax=axes[1], color='navy', label='HPA CPU Target (%)')
    sns.lineplot(x=df.index, y='CPU_Usage', data=df, ax=axes[1], color='skyblue', label='Actual Avg CPU Usage (%)')
    axes[1].set_title('HPA Target and Actual CPU Usage', fontsize=16)
    axes[1].set_ylabel('CPU (%)')
    axes[1].legend()

    sns.lineplot(x=df.index, y='Requests', data=df, ax=axes[2], color='green', label='Actual Requests')
    axes[2].set_title('Number of Actual Requests per Step (Testing Phase)', fontsize=16)
    axes[2].set_ylabel('Actual Requests')
    axes[2].set_xlabel('Simulation Step')

    plt.tight_layout()
    plt.savefig('performance_overview.png')
    plt.show()

def plot_latency_distribution(df):
    """ Plots latency distribution, similar to Fig 2 in the paper. """
    plt.figure(figsize=(12, 7))
    sns.histplot(df['Latency'], bins=50, kde=True, color='purple')
    
    mean_lat = df['Latency'].mean()
    median_lat = df['Latency'].median()
    
    plt.axvline(SLA_LATENCY_S, color='red', linestyle='--', lw=2, label=f'SLA Threshold ({SLA_LATENCY_S*1000:.0f} ms)')
    plt.axvline(mean_lat, color='orange', linestyle='-', lw=2, label=f'Mean Latency ({mean_lat*1000:.2f} ms)')
    plt.axvline(median_lat, color='green', linestyle='-', lw=2, label=f'Median Latency ({median_lat*1000:.2f} ms)')
    
    plt.title('Latency Distribution (Testing Phase)', fontsize=16)
    plt.xlabel('Latency (s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('latency_distribution.png')
    plt.show()

def print_summary_metrics(df):
    """ Prints a summary table similar to Table IV in the paper. """
    avg_latency = df['Latency'].mean()
    median_latency = df['Latency'].median()
    max_latency = df['Latency'].max()
    avg_success_ratio = df['Success_Ratio'].mean()
    min_success_ratio = df['Success_Ratio'].min()
    sla_compliance = (df['Latency'] <= SLA_LATENCY_S).mean() * 100

    print("\n" + "="*50)
    print("      PERFORMANCE SUMMARY (TESTING PHASE)      ")
    print("="*50)
    print(f"{'Metric':<25} | {'Value':>20}")
    print("-"*50)
    print(f"{'Average Latency (s)':<25} | {avg_latency:>20.4f}")
    print(f"{'Median Latency (s)':<25} | {median_latency:>20.4f}")
    print(f"{'Maximum Latency (s)':<25} | {max_latency:>20.4f}")
    print(f"{'Average Success Ratio':<25} | {avg_success_ratio:>20.4f}")
    print(f"{'Minimum Success Ratio':<25} | {min_success_ratio:>20.4f}")
    print(f"{'SLA Compliance (%)':<25} | {sla_compliance:>20.2f}%")
    print("="*50 + "\n")


if __name__ == '__main__':
    try:
        df_results = pd.read_csv(RESULTS_FILE)
        print(f"Successfully loaded results from '{RESULTS_FILE}'.")
        
        print_summary_metrics(df_results)
        plot_performance_overview(df_results)
        plot_latency_distribution(df_results)
        
        print(f"Charts saved to 'performance_overview.png' and 'latency_distribution.png'.")

    except FileNotFoundError:
        print(f"Error: The results file '{RESULTS_FILE}' was not found. Please run the simulation in 'test' mode first.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
