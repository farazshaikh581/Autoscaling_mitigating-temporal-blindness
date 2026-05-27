import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.options.mode.chained_assignment = None # default='warn'

# --- Configuration ---
INVOCATION_FILE = "invocations_per_function_md.anon.d01.csv"
MIN_TOTAL_INVOCATIONS = 100000  # Filter out functions with very low activity
MIN_ACTIVE_DAYS = 5             # Ensure the function is reasonably consistent

# --- Analysis Functions ---
def load_and_preprocess_data(filepath):
    """Loads the Alibaba trace and adds datetime columns."""
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None, names=['Timestamp', 'HashOwner', 'HashApp', 'HashFunction', 'Invoke'])
    
    # The timestamp is in minutes from the start of the trace.
    df['time'] = pd.to_datetime(df['Timestamp'], unit='m')
    df['day'] = df['time'].dt.dayofyear
    df['minute_of_day'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    return df

def calculate_stats(df):
    """Calculates summary statistics for each function."""
    logging.info("Calculating statistics for each function. This may take a moment...")
    
    # Calculate daily statistics first
    daily_invokes = df.groupby(['HashFunction', 'day'])['Invoke'].sum().reset_index()
    
    # Calculate summary stats per function
    stats = daily_invokes.groupby('HashFunction')['Invoke'].agg([
        'sum', 
        'mean', 
        'std',
        lambda x: x.count()
    ]).rename(columns={'<lambda_0>': 'active_days', 'sum': 'total_invocations'})
    
    # Calculate Coefficient of Variation (Volatility)
    stats['coeff_variation'] = stats['std'] / stats['mean']
    
    # Filter based on minimum requirements
    stats = stats[
        (stats['total_invocations'] > MIN_TOTAL_INVOCATIONS) &
        (stats['active_days'] > MIN_ACTIVE_DAYS)
    ]
    
    # Calculate Autocorrelation at 24-hour lag (1440 minutes)
    autocorrelations = {}
    for func_hash in stats.index:
        func_series = df[df['HashFunction'] == func_hash].set_index('time')['Invoke']
        # Resample to ensure a complete time series, filling gaps with 0
        func_series = func_series.resample('min').sum().fillna(0)
        # Calculate autocorrelation for 1 day lag
        autocorr = func_series.autocorr(lag=1440)
        autocorrelations[func_hash] = autocorr if not np.isnan(autocorr) else 0

    stats['autocorr_lag_1440'] = stats.index.map(autocorrelations)
    
    return stats.sort_values(by='total_invocations', ascending=False)

def plot_function_trace(df, func_hash):
    """Plots the invocation trace for a specific function."""
    logging.info(f"Plotting trace for function: {func_hash}")
    func_data = df[df['HashFunction'] == func_hash]
    
    plt.figure(figsize=(15, 6))
    plt.plot(func_data['time'], func_data['Invoke'], label=f"Invocations for {func_hash[:10]}...")
    plt.title(f"Invocation Trace for Function {func_hash[:10]}...")
    plt.xlabel("Time")
    plt.ylabel("Invocations per Minute")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"trace_{func_hash[:10]}.png")
    logging.info(f"Saved plot to trace_{func_hash[:10]}.png")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    full_df = load_and_preprocess_data(INVOCATION_FILE)
    function_stats = calculate_stats(full_df)
    
    logging.info("Top 10 most active functions:")
    print(function_stats.head(10))
    
    # --- Find Candidates for Each Profile ---
    print("\n" + "="*50)
    print("CANDIDATE SEARCH")
    print("="*50)

    # Profile 1: Predictable Diurnal Workload (Low Volatility, High Autocorrelation)
    predictable_candidates = function_stats[function_stats['autocorr_lag_1440'] > 0.5].sort_values(by='coeff_variation', ascending=True)
    print("\n[Profile 1: Predictable Diurnal Candidates (Low Volatility, High Autocorrelation)]")
    print(predictable_candidates.head(5))

    # Profile 2: Volatile & Spiky Workload (High Volatility, Low Autocorrelation)
    volatile_candidates = function_stats[function_stats['autocorr_lag_1440'] < 0.2].sort_values(by='coeff_variation', ascending=False)
    print("\n[Profile 2: Volatile & Spiky Candidates (High Volatility, Low Autocorrelation)]")
    print(volatile_candidates.head(5))
    
    # Profile 3: Balanced Real-World Scenario
    balanced_candidates = function_stats[
        (function_stats['coeff_variation'].between(0.8, 1.5)) &
        (function_stats['autocorr_lag_1440'].between(0.2, 0.5))
    ].sort_values(by='total_invocations', ascending=False)
    print("\n[Profile 3: Balanced Real-World Candidates (Moderate Volatility & Autocorrelation)]")
    print(balanced_candidates.head(5))

    # --- Plotting selected candidates ---
    # After reviewing the printed tables, select one hash from each profile to visualize.
    # Replace these with your chosen hashes.
    
    # Example selection
    PREDICTABLE_HASH = predictable_candidates.index[0]
    VOLATILE_HASH = volatile_candidates.index[0]
    BALANCED_HASH = balanced_candidates.index[0] # The one from the previous prompt often fits here
    
    plot_function_trace(full_df, PREDICTABLE_HASH)
    plot_function_trace(full_df, VOLATILE_HASH)
    plot_function_trace(full_df, BALANCED_HASH)
