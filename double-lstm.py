# === PROPOSED APPROACH: Attention + Double LSTM ===
# Comparison: Matches Baseline capacity (256 units) but adds Attention & Depth
# Architecture: Windowed Input -> Double LSTM -> Attention -> PPO

import os
import sys
import socket
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from stable_baselines3.common.logger import configure

# ============================================
# === CONFIGURATION ===
# ============================================
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("kubernetes").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

seed = SEED
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("WARNING: CUDA GPU not found, running on CPU.")
torch.set_default_device(device)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from prometheus_client import start_http_server

# CONSTANTS
application = "factorizator"
app_env = "factorizator"
cpu_target_percentage = 50
MIN_REPLICAS = 1
MAX_REPLICAS = 200
NR_REQUESTS_MAX = 3000
FORECAST_WINDOW = 3
MINUTES_PER_DAY = 500 

# === ARCHITECTURE SETTINGS ===
# We use a window to give the Attention mechanism a sequence to look at.
  
FEATURE_DIM = 14 

# ============================================
# === CUSTOM NETWORK: ATTENTION + DOUBLE LSTM ===
# ============================================
class AttentionDoubleLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(AttentionDoubleLSTM, self).__init__(observation_space, features_dim)
        
        self.seq_len = WINDOW_SIZE
        self.input_dim = FEATURE_DIM
        self.hidden_dim = 256  # Match Baseline Paper Capacity
        
        # 1. Feature Encoder
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 2. Double Stacked LSTM (The "Recurrent" Part)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=self.hidden_dim, 
            num_layers=2,       # Double Stacked
            batch_first=True, 
            dropout=0.2
        )
        
        # 3. Attention Mechanism (The "Proposed" Improvement)
        self.attn_fc = nn.Linear(self.hidden_dim, 1)
        
        # 4. Output Projection
        self.out_net = nn.Sequential(
            nn.Linear(self.hidden_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        # Reshape flattened window back to (Batch, Time, Features)
        x = observations.view(batch_size, self.seq_len, self.input_dim)
        
        # Encode features for every time step
        x_flat = x.view(-1, self.input_dim)
        feats = self.feature_net(x_flat)
        feats = feats.view(batch_size, self.seq_len, -1)
        
        # LSTM Pass (Captures Trends)
        lstm_out, _ = self.lstm(feats)
        
        # Attention Pass (Captures Critical Moments)
        attn_scores = self.attn_fc(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Context = Weighted sum of history
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        return self.out_net(context)

# ============================================
# === UTILS ===
# ============================================
def add_day_column(df, minutes_per_day=MINUTES_PER_DAY):
    t0 = df.end_timestamp.min()
    df['minute'] = (df.end_timestamp - t0) // 60
    df['day'] = (df['minute'] // minutes_per_day).astype(int)
    return df

def get_random_days(df, n_days=7, train_days=5, test_days=2, seed=SEED):
    all_days = sorted(df.day.unique())
    n_days = min(n_days, len(all_days))
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
    logging.error("❌ CRITICAL: Service unreachable. Check --url.")
    return False

# ============================================
# === ENV CLASS ===
# ============================================
class MultiAgentClusterEnv(gym.Env):
    def __init__(self, invocation_file, day_list, service_url):
        super().__init__()
        self.service_url = service_url
        self.df = pd.read_csv(invocation_file)
        self.df = add_day_column(self.df)
        self.action_space = gym.spaces.MultiDiscrete([4, 3, 3, 3])
        self.day_list = day_list
        self.max_minutes = MINUTES_PER_DAY 
        self.invocation_matrix = self.make_invocation_matrix()
        self.days_train = len(day_list)
        
        self.steps = 0; self.days = 0; self.global_step = 0
        self.hpa_target = cpu_target_percentage
        self.prev_hpa_target = cpu_target_percentage
        self.throughput_multiplier = 1.0; self.enhancement = 0
        self.reward = 0.0; self.raw_request_history = deque(maxlen=FORECAST_WINDOW)
        self.current_step_reward = 0.0; self.last_replicas = 1
        self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 1.0
        self.forecast_running_avg = 100.0
        
        # Bounds
        self.low_base = np.array([0.001, 1, 0, 0, 0, 0, 0, 0, 1, 1.0, 0, -1, -1, 0], dtype=np.float32)
        self.high_base = np.array([0.15, 200.0, 200.0, 100, 3600, 1800, 1800, 1, 95, 3.0, 2, 1, 1, 3600], dtype=np.float32)
        
        # Observation Space (Flattened Window)
        self.observation_space = gym.spaces.Box(
            low=np.tile(self.low_base, WINDOW_SIZE),
            high=np.tile(self.high_base, WINDOW_SIZE),
            dtype=np.float32
        )
        
        self.state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 0, 1, 0, 100], dtype=np.float32)
        self.state_history = deque(maxlen=WINDOW_SIZE)
        for _ in range(WINDOW_SIZE):
            self.state_history.append(self.state)

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
        """
        FINAL RUN_HEY:
        - Concurrency: 1 (Single Worker, Sequential Processing).
        - Pacing: Precise (-q) to spread requests evenly over 60s.
        - Workload: Raw Trace Value (Relies on app's internal difficulty).
        """
        # 1. Day/Step Cycling
        if self.steps + self.days * self.max_minutes >= len(self.day_list) * self.max_minutes:
            self.days += 1
            self.steps = 0

        current_day_idx = (self.steps // self.max_minutes) % len(self.day_list)
        current_min_idx = self.steps % self.max_minutes

        # 2. Get Raw Demand (Requests per Minute)
        #try:
        #    raw_requests = float(self.invocation_matrix[self.days, self.steps])
        #except:
        #    raw_requests = 100.0
        raw_requests = float(self.invocation_matrix[current_day_idx, current_min_idx])

        # 3. History Update 
        # (NOTE: Keep this for Attention Model. For Baseline, we can comment it out or leave it, it won't hurt).
        self.raw_request_history.append(raw_requests)
        if len(self.raw_request_history) >= FORECAST_WINDOW:
            recent = np.mean(list(self.raw_request_history))
            try: look = float(self.invocation_matrix[self.days, min(self.steps+2, self.max_minutes-1)])
            except: look = recent
            self.forecast_running_avg = 0.4*(0.6*recent + 0.4*look) + 0.6*self.forecast_running_avg
        else:
            self.forecast_running_avg = 0.5*raw_requests + 0.5*self.forecast_running_avg


        # ============================================
        # FIX: TRUE IDLE MODE (Prevents Timeout Crash)
        # ============================================
        # Only truly idle if no requests at all
        if raw_requests < 1.0:
            logging.info(f"Step {self.steps}: Trace=0 | Idle (No requests)")
            time.sleep(60)  # Add sleep to match StaticHPA timing
            self._latency_p90 = 0.005
            self._latency_avg = 0.005
            self._success_ratio = 1.0
            return 0.0, 0.005, 1.0

            # Return 0 requests, but valid latency/success
            return 0.0, 0.005, 1.0

        # ============================================
        # CONFIG: SINGLE WORKER + RATE LIMITING
        # ============================================
        
        # 1. Single Worker
        # Forces sequential processing to prevent port exhaustion and quantization noise.
        #ideal_workers = math.ceil(raw_requests / 25.0)
        #concurrency = max(1, min(ideal_workers, 20))
        concurrency = max(1, min(int(raw_requests / 10), 10))
        #safe_requests = max(1.0, raw_requests)
        # 2. Calculate Pacing
        # Trace = Requests per Minute.
        # Target QPS = Requests / 60 seconds.
        #target_qps = raw_requests / 60.0



        target_qps = (raw_requests * self.throughput_multiplier) / 60.0
        
        # 3. Workload (Raw 'n')
        #work_param = 1000000007000000009  # Hard semiprime
        work_param = 1000000016000000063

        # Command Construction
        # -c 1 : Single Connection
        # -q : Force specific requests per second
        # -z 60s : Run for the full minute
        command = f"hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s -m GET {self.service_url}factor?n={work_param}"
        #command = f"hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s -m GET {self.service_url}factor?n={int(work_param)}"
        #command = f"hey -c {concurrency} -z 60s -m GET {self.service_url}factor?n={work_param}"
        logging.info(f"Step {self.steps}: Trace={raw_requests:.0f} | Workers={concurrency} | QPS={target_qps:.2f}")

        try:
            # Timeout set to 80s (60s duration + 20s buffer)
            # Critical for high CPU loads where server responses might lag.
            output = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=80 
            )
            
            # Combine logs for the parser
            full_output = output.stdout + "\n" + output.stderr
            self._parse_hey_output(full_output, raw_requests)
        
        except Exception as e:
            logging.error(f"Hey execution exception: {e}")
            self._latency_p90 = 1
            self._latency_avg = 1
            self._success_ratio = 0.0

        return raw_requests, self._latency_p90, self._success_ratio

    def _parse_hey_output(self, output, raw_requests):
        try:
            if "Response time histogram" not in output:
                if "connection refused" in output or "dial tcp" in output:
                    logging.error("❌ CRITICAL: Service unreachable.")
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
        cpu, ram, t_cpu, t_ram = self.get_resources_usage()
        try: reps = int(subprocess.check_output(['kubectl', 'get', 'deploy', application, '-n', app_env, '-o', 'jsonpath={.spec.replicas}'], text=True))
        except: reps = 1
        eff_req = reqs * self.throughput_multiplier
        angle = (self.steps / self.max_minutes) * 2 * math.pi
        return np.array([lat, reps, cpu, ram, eff_req, t_cpu, t_ram, succ, self.hpa_target, self.throughput_multiplier, self.enhancement, math.cos(angle), math.sin(angle), self.forecast_running_avg], dtype=np.float32)

    def compute_reward(self):
        lat = self._latency_p90
        cpu = self.state[2]
        replicas = self.state[1]
        succ = self._success_ratio
        
        if lat <= 0.020: r_sla = 1.0
        elif lat <= 0.050: r_sla = 0.5 + 0.5 * (0.050 - lat) / 0.030
        else: r_sla = max(-1.0, -0.5 * (lat - 0.050) / 0.1)
        
        r_cpu = 1.0 if abs(cpu - self.hpa_target) <= 10 else np.exp(-((cpu - self.hpa_target)/50)**2)
        delta = abs(replicas - self.last_replicas)
        r_stab = 0.5 if delta == 0 else (0.3 if delta <=2 else -0.2 * min(1.0, delta/10))
        
        reward = 0.5*r_sla + 0.25*r_cpu + 0.15*succ + 0.1*r_stab
        self.current_step_reward = reward; self.reward = reward; self.last_replicas = replicas
        return reward

    def apply_multiagent_action(self, action):
        opts = [30, 50, 70, 90]
        new_t = opts[int(action[0]) % 4]
        
        if new_t != self.hpa_target:
            self.prev_hpa_target = self.hpa_target
            self.hpa_target = new_t
            
            # JSON Patch to update target utilization
            patch = {"spec": {"metrics": [{"type": "Resource","resource": {"name": "cpu","target": {"type": "Utilization","averageUtilization": new_t}}}]}}
            
            try:
                subprocess.run(['kubectl', 'patch', 'hpa', application, '-n', app_env, '--patch', json.dumps(patch)], check=True, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                logging.error(f"❌ Failed to patch HPA! Target {new_t}% not applied. Is HPA missing?")

        self.throughput_multiplier = [1.0, 2.0, 3.0][int(action[2]) % 3]
        self.enhancement = int(action[3]) % 3

    def step(self, action):
        self.apply_multiagent_action(action)
        reqs, lat, succ = self.run_hey()
        
        raw_state = self.get_new_state(reqs, lat, succ)
        
        # Clip single state
        raw_state[0] = np.clip(raw_state[0], self.low_base[0], self.high_base[0])
        self.state = np.clip(raw_state, self.low_base, self.high_base)
        
        self._latency_p90 = lat; self._success_ratio = succ
        reward = self.compute_reward()
        
        # Update Window History
        self.state_history.append(self.state)
        self.steps += 1; self.global_step += 1
        term = self.steps >= self.days_train * self.max_minutes
        
        # Return Flattened Window
        flattened_obs = np.array(self.state_history, dtype=np.float32).flatten()
        
        return flattened_obs, float(reward), term, False, {}

    def reset(self, seed=None, options=None):
        self.days = 0; self.steps = 0; self.forecast_running_avg = 100.0
        logging.info("♻️  Resetting Cluster...")
        
        # 1. Clean up old HPA
        subprocess.run(['kubectl', 'delete', 'hpa', application, '-n', app_env], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Reset Scale
        subprocess.run(['kubectl', 'scale', 'deploy', application, '-n', app_env, '--replicas=1'], stdout=subprocess.DEVNULL)
        
        # 3. CREATE FRESH HPA (Crucial Step!)
        # We start with 50% target. PPO will patch this later.
        time.sleep(2)
        subprocess.run([
            'kubectl', 'autoscale', 'deploy', application, '-n', app_env, 
            '--cpu-percent=50', '--min=1', '--max=30'
        ], stdout=subprocess.DEVNULL)
        
        logging.info("✅ Cluster Reset: HPA Created (50%), Replicas=1")
        time.sleep(5)  # Wait for HPA to initialize metrics
        
        self.state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 0, 1, 0, 100], dtype=np.float32)
        self.state_history.clear()
        for _ in range(WINDOW_SIZE):
            self.state_history.append(self.state)
            
        return np.array(self.state_history, dtype=np.float32).flatten(), {}

# ============================================
# === CALLBACKS & MAIN ===
# ============================================
class DetailedLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            env = self.training_env.unwrapped.envs[0]; s = env.state
            msg = (
                f"\n{'='*40} STEP {env.global_step} {'='*40}\n"
                f" REWARD       : {env.reward:.4f}\n"
                f"----------------------------------------\n"
                f" Latency (P90): {env._latency_p90*1000:6.1f} ms\n"
                f" Latency (Avg): {env._latency_avg*1000:6.1f} ms\n"
                f"----------------------------------------\n"
                f" Replicas     : {int(s[1])}\n"
                f" CPU Usage    : {s[2]:5.1f} %\n"
                f" Requests     : {s[4]:.0f}/min\n"
                f" Success Rate : {env._success_ratio*100:.1f} %\n"
                f" Forecast     : {s[13]:.1f}\n"
                f"{'='*88}\n"
            )
            logging.info(msg)
        except: pass
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path
        self.headers = ["Step", "Reward", "Latency_P90", "Latency_Avg", "Replicas", "CPU_Pct", "RAM_Pct", 
                        "Requests", "Total_CPU", "Total_RAM", "Success", "HPA_Target", "Throughput", "Enhancement", "Forecast"]
        
        # Ensure directory exists
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f: 
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
            logging.info(f"✅ CSV Log initialized at: {os.path.abspath(csv_path)}")

    def _on_step(self) -> bool:
        try:
            env = self.training_env.unwrapped.envs[0]; s = env.state
            
            # Tensorboard
            self.logger.record("train/reward", env.reward)
            self.logger.record("train/latency_p90", env._latency_p90 * 1000)
            self.logger.record("train/replicas", env.state[1])
            self.logger.dump(step=self.num_timesteps)

            raw_requests_val = s[4] / s[9] if s[9] > 0 else s[4]

            # CSV
            if self.csv_path:
                with open(self.csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=self.headers)
                    w.writerow({
                        "Step": self.num_timesteps, 
                        "Reward": env.reward, 
                        "Latency_P90": env._latency_p90*1000, 
                        "Latency_Avg": env._latency_avg*1000,
                        "Replicas": int(s[1]), 
                        "CPU_Pct": s[2], "RAM_Pct": s[3], 
                        "Requests": raw_requests_val, "Total_CPU": s[5], "Total_RAM": s[6],
                        "Success": env._success_ratio, "HPA_Target": int(s[8]), 
                        "Throughput": s[9], "Enhancement": int(s[10]), "Forecast": s[13]
                    })
                    f.flush()
                    os.fsync(f.fileno())
                
                # DEBUG PRINT (Remove later if annoying)
                # print(f"💾 CSV Saved to {self.csv_path}") 

        except Exception as e:
            logging.error(f"❌ Logging Error: {e}")
            
        return True

def cosine_schedule(progress): return 2e-4 * 0.5 * (1 + np.cos(np.pi * (1 - progress)))

if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--url", type=str, required=True, help="Target Service URL")
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    
    # 1. Create Results Directory Explicitly
    log_dir = "results_log"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/tb", exist_ok=True)

    # Generate unique filenames based on time
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_csv = f"{log_dir}/train_log_{timestamp}.csv"
    test_csv = f"{log_dir}/test_log_{timestamp}.csv"

    # 2. Start Prometheus
    try: start_http_server(9098)
    except: pass
    
    # 3. Pre-flight
    if not wait_for_service_availability(args.url): sys.exit(1)

    df = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df = add_day_column(df)
    tr_days, te_days = get_random_days(df)
    
    def build_env(days): return MultiAgentClusterEnv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", days, service_url=args.url)
    
    vec_train = VecNormalize(DummyVecEnv([lambda: build_env(tr_days)]), norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # 4. TRAIN MODE
    if args.mode == "train":
        logging.info(f"Starting Training... Logging to: {train_csv}")
        vec_train = VecNormalize(DummyVecEnv([lambda: build_env(tr_days)]), norm_obs=True, norm_reward=False, clip_obs=10.)


        model = PPO(
            "MlpPolicy", vec_train, verbose=1, device='cuda',
            policy_kwargs=dict(features_extractor_class=AttentionDoubleLSTM, features_extractor_kwargs=dict(features_dim=256)),
            learning_rate=3e-4, n_steps=128, batch_size=128, n_epochs=10, tensorboard_log=f"{log_dir}/tb"
        )
        #reset_timestep = True
        
        cbs = [DetailedLoggingCallback(), TensorboardCallback(train_csv), CheckpointCallback(500, f"{log_dir}/ckpt")]
        model.learn(total_timesteps=len(tr_days)*MINUTES_PER_DAY, callback=cbs)
        
        model.save(f"{log_dir}/final_model")
        vec_train.save(f"{log_dir}/vecnorm.pkl")
        logging.info("Done Training!")

    # 5. TEST MODE (UPDATED)
    if args.mode == "test":
        logging.info(f"Starting Testing... Logging to: {test_csv}")
        
        # Verify files exist
        model_path = f"{log_dir}/final_model.zip"
        norm_path = f"{log_dir}/vecnorm.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(norm_path):
            logging.error(f"❌ Missing model or norm file in {log_dir}/"); sys.exit(1)
            
        # Load Env & Model
        vec_test = VecNormalize.load(norm_path, DummyVecEnv([lambda: build_env(te_days)]))
        vec_test.training = False; vec_test.norm_reward = False
        model = PPO.load(model_path, device=device)
        #model = PPO.load(f"{log_dir}/ckpt_attn/PPO_2500_steps.zip")
        model.set_env(vec_test)

        tmp_path = "tmp/sb3_log/"
        new_logger = configure(tmp_path, ["stdout", "csv"])
        model.set_logger(new_logger)
        
        # Setup Callback for Logging
        test_cb = TensorboardCallback(test_csv)
        test_cb.manual_env = vec_test
        test_cb.init_callback(model)
        
        obs = vec_test.reset()
        total_steps = len(te_days) * MINUTES_PER_DAY
        
        logging.info(f"Running evaluation for {total_steps} steps...")
        
        for i in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_test.step(action)
            
            # MANUAL LOGGING TRIGGER
            # We inject the test env into the callback so it reads the correct metrics
            test_cb.manual_env = vec_test
            test_cb.init_callback(model)
            test_cb.num_timesteps = i
            test_cb._on_step()
            
            if i % 10 == 0:
                # Access raw env to print real latency (not normalized)
                raw_env = vec_test.unwrapped.envs[0]
                logging.info(f"Test Step {i}: Reward={float(reward[0]):.2f} | Latency={raw_env._latency_p90*1000:.1f}ms")
                
        logging.info(f"✅ Testing Complete! Data saved to {test_csv}")
