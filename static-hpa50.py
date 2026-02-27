# -*- coding: utf-8 -*-
# === STATIC HPA BENCHMARK (Exact Workload Match) ===
# Logic: Kubernetes HPA (CPU Target 50%)
# Workload: EXACT COPY of PPO's run_hey and day selection logic

import os
import sys
import math
import numpy as np
import pandas as pd
import subprocess
import random
import re
import time
import logging
import csv
import requests
import datetime

# ============================================
# === CONFIGURATION ===
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants matching RL scripts
application = "factorizator"
app_env = "factorizator"
MINUTES_PER_DAY = 500 
SEED = SEED

# Although several seed values were explored during our experiments, all reported results in the paper correspond to runs with the random seed fixed at 42.

# Global Seeding
np.random.seed(SEED)
random.seed(SEED)

# ============================================
# === UTILS (Copied from PPO Script) ===
# ============================================
def add_day_column(df, minutes_per_day=MINUTES_PER_DAY):
    t0 = df.end_timestamp.min()
    df['minute'] = (df.end_timestamp - t0) // 60
    df['day'] = (df['minute'] // minutes_per_day).astype(int)
    return df

def get_random_days(df, n_days=7, train_days=5, test_days=2, seed=SEED):
    # EXACT COPY from PPO script
    all_days = sorted(df.day.unique())
    n_days = min(n_days, len(all_days))
    
    # CRITICAL: Reset seed internally to ensure same choice regardless of global state
    random.seed(seed)
    selected_days = random.sample(list(all_days), n_days)
    
    return selected_days[:train_days], selected_days[train_days:train_days+test_days]

def wait_for_service_availability(url, max_retries=5, wait_sec=2):
    logging.info(f"🔍 PRE-FLIGHT CHECK: Testing connection to {url}...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}factor?n=1", timeout=2)
            if response.status_code == 200:
                logging.info(f"✅ Service is UP. (Attempt {i+1}/{max_retries})")
                return True
        except Exception:
            logging.warning(f"⏳ Connection failed. Retrying in {wait_sec}s...")
        time.sleep(wait_sec)
    logging.error("❌ CRITICAL: Service unreachable. Check your --url argument.")
    return False

# ============================================
# === STATIC ENV CLASS ===
# ============================================
class StaticBenchmark:
    def __init__(self, invocation_file, test_days, service_url, log_file):
        self.service_url = service_url
        self.df = pd.read_csv(invocation_file)
        self.df = add_day_column(self.df)
        
        self.day_list = test_days 
        self.max_minutes = MINUTES_PER_DAY
        self.invocation_matrix = self.make_invocation_matrix()
        
        # Counters
        self.steps = 0
        self.days = 0 
        
        # Logging
        self.log_file = log_file
        self.headers = ["Step", "Reward", "Latency_P90", "Latency_Avg", "Replicas", "CPU_Pct", "RAM_Pct", 
                        "Requests", "Total_CPU", "Total_RAM", "Success", "HPA_Target", "Throughput", "Enhancement", "Forecast"]
        
        with open(self.log_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.headers).writeheader()
            
        logging.info(f"✅ Static Baseline Initialized. Test Days: {self.day_list}")

    def make_invocation_matrix(self):
        matrix = np.zeros((len(self.day_list), self.max_minutes))
        for i, day in enumerate(self.day_list):
            mask = self.df['day'] == day
            minutes = self.df.loc[mask, 'minute'] % self.max_minutes
            for m in minutes:
                if 0 <= int(m) < self.max_minutes: matrix[i, int(m)] += 1
        return matrix

    def get_resources_usage(self):
        try:
            output = subprocess.check_output(['kubectl', 'top', 'pod', '-l', f'app={application}', '-n', app_env], encoding='utf-8', stderr=subprocess.DEVNULL)
            lines = output.strip().split('\n')[1:]
            cpu_vals, ram_vals = [], []
            for line in lines:
                cols = line.split()
                if len(cols) >= 3:
                    cpu_vals.append(float(cols[1].replace('m',''))/1000 if 'm' in cols[1] else float(cols[1]))
                    ram_vals.append(float(cols[2].replace('Mi','')) if 'Mi' in cols[2] else float(cols[2]))
            
            cpu_u = min((sum(cpu_vals)/len(cpu_vals))/0.25*100, 200) if cpu_vals else 1 
            ram_u = min((sum(ram_vals)/len(ram_vals))/128*100, 200) if ram_vals else 1
            return cpu_u, ram_u, sum(cpu_vals)*1000, sum(ram_vals)
        except: return 1, 1, 1, 1

    def run_hey(self):
        # --- Day/Step cycling ---
        # When we finish one day (500 minutes), move to next test day
        if self.steps >= self.max_minutes:
            self.days += 1
            self.steps = 0
    
        # If all test days are finished, behave like idle (end-of-trace)
        if self.days >= len(self.day_list):
            logging.info("All test days completed.")
            time.sleep(60)
            return 0.0, 0.005, 0.005, 1.0
    
        current_day_idx = self.days
        current_min_idx = self.steps
    
        # 1) Get raw demand
        try:
            raw_requests = float(self.invocation_matrix[current_day_idx, current_min_idx])
        except Exception:
            raw_requests = 100.0
    
        # 2) Idle mode
        if raw_requests < 1.0:
            logging.info(f"Step {self.steps}: Trace=0 | Idle Mode (Sleeping 60s)")
            time.sleep(60)
            return 0.0, 0.005, 0.005, 1.0
    
        # 3) Active mode (same pacing structure; throughput fixed at 1.0)
        concurrency = max(1, min(int(raw_requests / 10), 10))
        throughput_multiplier = 1.0
        target_qps = (raw_requests * throughput_multiplier) / 60.0
        work_param = 1000000016000000063
    
        command = (
            f"hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s "
            f"-m GET {self.service_url}factor?n={work_param}"
        )
    
        logging.info(
            f"Step {self.steps}: Trace={raw_requests:.0f} | "
            f"DayIdx={current_day_idx} MinIdx={current_min_idx} | "
            f"Workers={concurrency} | QPS={target_qps:.2f}"
        )
    
        lat_p90, lat_avg, success = 1.0, 1.0, 0.0
        try:
            output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=80)
            lat_p90, lat_avg, success = self._parse_hey(output.stdout + "\n" + output.stderr)
        except Exception as e:
            logging.error(f"Hey error: {e}")
    
        return raw_requests, lat_p90, lat_avg, success

    def _parse_hey(self, output):
        try:
            if "Response time histogram" not in output: return 0.5, 0.5, 0.0
            
            p90 = re.search(r"90%\s+in\s+([\d.]+)\s+secs", output)
            avg = re.search(r"Average:\s+([\d.]+)\s+secs", output)
            l90 = float(p90.group(1)) if p90 else 0.05
            lavg = float(avg.group(1)) if avg else 0.05

            status = re.findall(r"\[(\d+)\]\s+(\d+)\s+responses", output)
            counts = {int(k): int(v) for k, v in status}
            succ = (counts.get(200, 0) / sum(counts.values())) if counts else (1.0 if "Error" not in output else 0.0)
            return l90, lavg, succ
        except: return 0.05, 0.05, 0.0

    def reset_cluster(self):
        logging.info("♻️  Resetting Cluster for Static Benchmark...")
        # 1. Reset Deployment
        subprocess.run(['kubectl', 'scale', 'deploy', application, '-n', app_env, '--replicas=1'], stdout=subprocess.DEVNULL)
        # 2. Reset HPA (Delete RL HPA, Create Static HPA)
        subprocess.run(['kubectl', 'delete', 'hpa', application, '-n', app_env], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        # 3. Create Static HPA (Target 50%, Min 1, Max 30)
        subprocess.run(['kubectl', 'autoscale', 'deploy', application, '-n', app_env, '--cpu-percent=50', '--min=1', '--max=30'], stdout=subprocess.DEVNULL)
        logging.info("✅ Static HPA (50%) Applied.")
        time.sleep(10)


  
    def run_loop(self):
        total_steps = len(self.day_list) * self.max_minutes
        logging.info(f"🚀 Starting Static Benchmark for {total_steps} steps...")
        
        self.reset_cluster()

        for i in range(total_steps):
            # 1. Run Workload
            reqs, lat_p90, lat_avg, succ = self.run_hey()
            
            # 2. Get Metrics
            cpu, ram, t_cpu, t_ram = self.get_resources_usage()
            try: 
                reps = int(subprocess.check_output(['kubectl', 'get', 'deploy', application, '-n', app_env, '-o', 'jsonpath={.spec.replicas}'], text=True))
            except: reps = 1

            # 3. Log to CSV
            with open(self.log_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.headers)
                w.writerow({
                    "Step": i,
                    "Reward": 0.0, 
                    "Latency_P90": lat_p90 * 1000,
                    "Latency_Avg": lat_avg * 1000,
                    "Replicas": reps,
                    "CPU_Pct": cpu, "RAM_Pct": ram,
                    "Requests": reqs, "Total_CPU": t_cpu, "Total_RAM": t_ram,
                    "Success": succ, 
                    "HPA_Target": 50,
                    "Throughput": 1.0, "Enhancement": 0, "Forecast": 0.0
                })
            
            self.steps += 1

# ============================================
# === MAIN ===
# ============================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Target Service URL")
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    
    # Setup Directory
    log_dir = "results_log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = f"{log_dir}/test-StaticHPA50_CORRECTED_{timestamp}.csv"
    
    if not wait_for_service_availability(args.url): sys.exit(1)

    # Load Data
    df = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df = add_day_column(df)
    
    # === CRITICAL: Get same days as PPO ===
    _, test_days = get_random_days(df, seed=SEED)
    logging.info(f"📅 SELECTED TEST DAYS: {test_days}")
    
    # Run Benchmark
    benchmark = StaticBenchmark("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", test_days, args.url, csv_file)
    benchmark.run_loop()
    
    logging.info(f"✅ Benchmark Complete. Results: {csv_file}")
