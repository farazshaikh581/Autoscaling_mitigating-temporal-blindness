# -*- coding: utf-8 -*-
# === DQN BASELINE (Strict Baseline: No Forecast) ===
# Comparison: Uses Standard Double DQN (MlpPolicy)
# Workload: Aligned with PPO, but State is 13-dim (No Forecast)

import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import subprocess
import random
import re
import json
from collections import deque
import logging
import time
import warnings
import csv
import requests

# === IMPORTS FOR DQN ===
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from prometheus_client import start_http_server

# ============================================
# === CONFIGURATION ===
# ============================================
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("WARNING: CUDA GPU not found, running on CPU.")

# CONSTANTS
application = "factorizator"
app_env = "factorizator"
cpu_target_percentage = 50
MIN_REPLICAS = 1
MAX_REPLICAS = 30 
MINUTES_PER_DAY = 500 
FORECAST_WINDOW = 3
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================
# === UTILS ===
# ============================================
def add_day_column(df, minutes_per_day=MINUTES_PER_DAY):
    t0 = df.end_timestamp.min()
    df['minute'] = (df.end_timestamp - t0) // 60
    df['day'] = (df['minute'] // minutes_per_day).astype(int)
    return df

def get_random_days(df, n_days=7, train_days=5, test_days=2, seed=42):
    all_days = sorted(df.day.unique())
    n_days = min(n_days, len(all_days))
    random.seed(seed)
    selected_days = random.sample(list(all_days), n_days)
    return selected_days[:train_days], selected_days[train_days:train_days+test_days]

def wait_for_service_availability(url, max_retries=5, wait_sec=2):
    logging.info(f"?? PRE-FLIGHT CHECK: Testing connection to {url}...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}factor?n=1", timeout=2)
            if response.status_code == 200:
                logging.info(f"? Service is UP. (Attempt {i+1}/{max_retries})")
                return True
        except Exception:
            logging.warning(f"? Connection failed. Retrying in {wait_sec}s...")
        time.sleep(wait_sec)
    logging.error("? CRITICAL: Service unreachable. Check your --url argument.")
    return False

import gymnasium as gym

# ============================================
# === DQN ENV CLASS ===
# ============================================
class DQNClusterEnv(gym.Env):
    def __init__(self, invocation_file, day_list, service_url):
        super().__init__()
        self.service_url = service_url
        self.df = pd.read_csv(invocation_file)
        self.df = add_day_column(self.df)
        
        self.action_space = gym.spaces.Discrete(3)
        
        self.day_list = day_list
        self.max_minutes = MINUTES_PER_DAY 
        self.invocation_matrix = self.make_invocation_matrix()
        self.days_train = len(day_list)
        
        self.steps = 0; self.days = 0; self.global_step = 0
        self.current_replicas = 1
        self.hpa_target = 50 
        self.throughput_multiplier = 1.0
        self.enhancement = 0
        
        self.reward = 0.0
        self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 1.0
        
        # === OBSERVATION SPACE: 13 DIMS (Matches Saved Model) ===
        # Removed the 14th dimension (Forecast)
        self.low_base = np.array([0.001, 1, 0, 0, 0, 0, 0, 0, 1, 1.0, 0, -1, -1], dtype=np.float32)
        self.high_base = np.array([0.15, 200.0, 200.0, 100, 3600, 1800, 1800, 1, 95, 3.0, 2, 1, 1], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=self.low_base, high=self.high_base, dtype=np.float32)
        
        self.state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 0, 1, 0], dtype=np.float32)

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
            output = subprocess.check_output(
                ['kubectl', 'top', 'pod', '-l', f'app={application}', '-n', app_env],
                encoding='utf-8', stderr=subprocess.DEVNULL
            )
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
        except:
            return 1, 1, 1, 1

    def run_hey(self):
        # 1. Day/Step Cycling
        if self.steps + self.days * self.max_minutes >= len(self.day_list) * self.max_minutes:
            self.days += 1
            self.steps = 0

        current_day_idx = (self.steps // self.max_minutes) % len(self.day_list)
        current_min_idx = self.steps % self.max_minutes

        # 2. Get Raw Demand
        # Robust check to prevent index errors
        try:
            raw_requests = float(self.invocation_matrix[current_day_idx, current_min_idx])
        except IndexError:
             # Fallback if matrix math is slightly off
             raw_requests = 100.0

        if raw_requests < 1.0:
            logging.info(f"Step {self.steps}: Trace=0 | Idle Mode (Sleeping 60s)")
            time.sleep(60)
            self._latency_p90 = 0.005 
            self._latency_avg = 0.005
            self._success_ratio = 1.0
            return 0.0, 0.005, 1.0

        concurrency = max(1, min(int(raw_requests / 10), 10))
        target_qps = raw_requests / 60.0
        work_param = 1000000016000000063
        
        command = f"hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s -m GET {self.service_url}factor?n={work_param}"
        logging.info(f"Step {self.steps}: Trace={raw_requests:.0f} | Workers={concurrency} | QPS={target_qps:.2f}")

        try:
            output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=80)
            full_output = output.stdout + "\n" + output.stderr
            self._parse_hey_output(full_output, raw_requests)
        except Exception as e:
            logging.error(f"Hey execution exception: {e}")
            self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 0.0

        return raw_requests, self._latency_p90, self._success_ratio

    def _parse_hey_output(self, output, raw_requests):
        try:
            if "Response time histogram" not in output:
                if "connection refused" in output or "dial tcp" in output:
                    logging.error("? CRITICAL: Service unreachable.")
                    self._latency_p90 = 0.5; self._latency_avg = 0.5; self._success_ratio = 0.0; return

            p90 = re.search(r"90%\s+in\s+([\d.]+)\s+secs", output)
            avg = re.search(r"Average:\s+([\d.]+)\s+secs", output)
            self._latency_p90 = float(p90.group(1)) if p90 else 0.05
            self._latency_avg = float(avg.group(1)) if avg else 0.05

            status = re.findall(r"\[(\d+)\]\s+(\d+)\s+responses", output)
            counts = {int(k): int(v) for k, v in status}
            if counts:
                ok = counts.get(200, 0)
                self._success_ratio = ok / sum(counts.values())
            else:
                self._success_ratio = 1.0 if "Error distribution" not in output else 0.0
        except Exception:
            self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 0.0

    def get_new_state(self, reqs, lat, succ):
        # Generates the 13-dim vector (Removed Forecast)
        cpu, ram, t_cpu, t_ram = self.get_resources_usage()
        try: reps = int(subprocess.check_output(['kubectl', 'get', 'deploy', application, '-n', app_env, '-o', 'jsonpath={.spec.replicas}'], text=True))
        except: reps = 1
        
        eff_req = reqs * self.throughput_multiplier
        angle = (self.steps / self.max_minutes) * 2 * math.pi
        
        # 13 Dimensions:
        return np.array([lat, reps, cpu, ram, eff_req, t_cpu, t_ram, succ, 
                        self.hpa_target, self.throughput_multiplier, self.enhancement, 
                        math.cos(angle), math.sin(angle)], dtype=np.float32)

    def compute_reward(self):
        lat = self._latency_p90
        cpu = self.state[2]
        replicas = self.state[1]
        succ = self._success_ratio
        
        if lat <= 0.020: r_sla = 1.0
        elif lat <= 0.050: r_sla = 0.5 + 0.5 * (0.050 - lat) / 0.030
        else: r_sla = max(-1.0, -0.5 * (lat - 0.050) / 0.1)
        
        r_cpu = 1.0 if abs(cpu - 50) <= 10 else np.exp(-((cpu - 50)/50)**2)
        r_cost = -0.05 * replicas
        r_fail = -5.0 if succ < 0.95 else 0.0
        
        reward = 0.5*r_sla + 0.25*r_cpu + 0.15*succ + r_cost + r_fail
        self.reward = reward
        return reward

    def step(self, action):
        scale_change = int(action) - 1 
        target_reps = max(MIN_REPLICAS, min(MAX_REPLICAS, self.current_replicas + scale_change))
        
        if target_reps != self.current_replicas:
            subprocess.run(f"kubectl scale deployment {application} --replicas={target_reps} -n {app_env}", shell=True, stdout=subprocess.DEVNULL)
            self.current_replicas = target_reps
            time.sleep(2)
        
        reqs, lat, succ = self.run_hey()
        
        raw_state = self.get_new_state(reqs, lat, succ)
        
        raw_state[0] = np.clip(raw_state[0], self.low_base[0], self.high_base[0])
        self.state = np.clip(raw_state, self.low_base, self.high_base)
        
        self._latency_p90 = lat; self._success_ratio = succ
        reward = self.compute_reward()
        
        self.steps += 1; self.global_step += 1
        term = self.steps >= self.days_train * self.max_minutes
        
        return self.state, float(reward), term, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.days = 0; self.steps = 0
        
        logging.info("Resetting Cluster (Delete HPA, Scale to 1)...")
        subprocess.run(f"kubectl delete hpa {application} -n {app_env}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"kubectl scale deployment {application} --replicas=1 -n {app_env}", shell=True, stdout=subprocess.DEVNULL)
        time.sleep(5)
        self.current_replicas = 1
        
        self.state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 0, 1, 0], dtype=np.float32)
        return self.state, {}

# ============================================
# === CALLBACKS ===
# ============================================
class CSVCallback(BaseCallback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.headers = ["Step", "Reward", "Latency_P90", "Latency_Avg", "Replicas", "CPU_Pct", "Requests", "Success"]
        
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f: 
                csv.DictWriter(f, fieldnames=self.headers).writeheader()

    def _on_step(self) -> bool:
        env = self.training_env.unwrapped.envs[0]
        s = env.state
        if self.csv_path:
            with open(self.csv_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.headers)
                w.writerow({
                    "Step": self.num_timesteps, 
                    "Reward": env.reward, 
                    "Latency_P90": env._latency_p90*1000, 
                    "Latency_Avg": env._latency_avg*1000,
                    "Replicas": int(env.current_replicas), 
                    "CPU_Pct": s[2], 
                    "Requests": s[4], 
                    "Success": env._success_ratio
                })
        return True

# ============================================
# === MAIN ===
# ============================================
if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--url", type=str, required=True, help="Target Service URL")
    args = parser.parse_args()
    
    log_dir = "results_log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = f"{log_dir}/dqn_{args.mode}_log_{timestamp}.csv"

    try: start_http_server(9091) 
    except: pass
    
    if not wait_for_service_availability(args.url): sys.exit(1)

    df = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df = add_day_column(df)
    tr_days, te_days = get_random_days(df, seed=SEED)
    
    def build_env(days): 
        return DQNClusterEnv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", days, args.url)

    if args.mode == "train":
        logging.info(f"?? Starting DQN Training on {len(tr_days)} days...")
        logging.info(f"?? Logging to: {csv_file}")
        
        env = DummyVecEnv([lambda: build_env(tr_days)])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4, 
            gamma=0.99, 
            buffer_size=50000, 
            exploration_fraction=0.2,
            tensorboard_log=f"{log_dir}/tb_dqn",
            device=device
        )
        
        cb = CSVCallback(csv_file)
        model.learn(total_timesteps=len(tr_days)*MINUTES_PER_DAY, callback=cb)
        
        model.save(f"{log_dir}/dqn_model_final")
        env.save(f"{log_dir}/vecnorm_dqn.pkl")
        logging.info("? DQN Training Complete.")

    elif args.mode == "test":
        logging.info("?? Starting DQN Testing...")
        
        env = DummyVecEnv([lambda: build_env(te_days)])
        
        if os.path.exists(f"{log_dir}/vecnorm_dqn.pkl"):
            env = VecNormalize.load(f"{log_dir}/vecnorm_dqn.pkl", env)
            env.training = False; env.norm_reward = False
        else:
            logging.warning("?? No normalization stats found. Running without VecNormalize.")
        
        model = DQN.load(f"{log_dir}/dqn_model_final")
        
        headers = ["Step", "Reward", "Latency_P90", "Latency_Avg", "Replicas", "CPU_Pct", "Requests", "Success"]
        with open(csv_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
        
        obs = env.reset()
        total_steps = len(te_days) * MINUTES_PER_DAY
        
        for i in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            raw_env = env.envs[0]
            s = raw_env.state
            
            with open(csv_file, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writerow({
                    "Step": i, 
                    "Reward": float(reward[0]), 
                    "Latency_P90": raw_env._latency_p90*1000, 
                    "Latency_Avg": raw_env._latency_avg*1000,
                    "Replicas": int(raw_env.current_replicas), 
                    "CPU_Pct": s[2], 
                    "Requests": s[4], 
                    "Success": raw_env._success_ratio
                })
            
            if i % 10 == 0:
                logging.info(f"Step {i}: Req={raw_env.state[4]:.0f} | Rep={raw_env.current_replicas} | Lat={raw_env._latency_p90*1000:.1f}ms")