# Changes from Double-LSTM: 
#  1. Single LSTM Layer (256 units)
#  2. No Explicit Forecast Input (The LSTM must learn the trend)
#  3. No History Stacking (The LSTM state handles memory)

import os
import sys
import socket
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
# add seed
seed = SEED
# Although several seed values were explored during our experiments, all reported results in the paper correspond to runs with the random seed fixed at 42.

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
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from prometheus_client import start_http_server

# CONSTANTS
application = "factorizator"
app_env = "factorizator"
cpu_target_percentage = 50
MIN_REPLICAS = 1
MAX_REPLICAS = 200
NR_REQUESTS_MAX = 3000
MINUTES_PER_DAY = 500 



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
    logging.error("❌ CRITICAL: Service unreachable. Check your --url argument.")
    return False

# ============================================
# === ENV CLASS (Baseline) ===
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
        
        # State
        self.steps = 0; self.days = 0; self.global_step = 0
        self.hpa_target = cpu_target_percentage
        self.prev_hpa_target = cpu_target_percentage
        self.throughput_multiplier = 1.0; self.enhancement = 0
        self.reward = 0.0
        self.current_step_reward = 0.0; self.last_replicas = 1
        
        # Metrics
        self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 1.0
        
        # === BASELINE INPUT: 13 dims (Removed Forecast) ===
        # 0:Lat, 1:Reps, 2:Cpu, 3:Ram, 4:EffReq, 5:TotCpu, 6:TotRam, 7:Succ, 8:Hpa, 9:Thru, 10:Enh, 11:Cos, 12:Sin
        low = np.array([0.001, 1, 0, 0, 0, 0, 0, 0, 1, 1.0, 0, -1, -1], dtype=np.float32)
        high = np.array([0.15, 200.0, 200.0, 100, 3600, 1800, 1800, 1, 95, 3.0, 2, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
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
        """Run hey WITHOUT forecasting logic (Fixed AttributeError)."""
        if self.steps >= self.max_minutes:
            self.days += 1
            self.steps = 0

        #try: raw_requests = float(self.invocation_matrix[self.days, self.steps])
        #except: raw_requests = 100.0

        current_day_idx = min(self.days, len(self.day_list) - 1)
        current_min_idx = self.steps
        raw_requests = float(self.invocation_matrix[current_day_idx, current_min_idx])


        # ============================================
        # FIX: TRUE IDLE MODE (Prevents Timeout Crash)
        # ============================================
        if raw_requests < 1.0:
            logging.info(f"Step {self.steps}: Trace=0 | Idle Mode (Sleeping 60s)")
            time.sleep(60)
            # Perfect metrics for idle state
            self._latency_p90 = 0.005 
            self._latency_avg = 0.005
            self._success_ratio = 1.0
            # Return 0 requests, but valid latency/success
            return 0.0, 0.005, 1.0
        
        # === FIX: REMOVED HISTORY APPEND LOGIC ===
        # This ensures we don't call self.raw_request_history (which doesn't exist in this class)

        #ideal_workers = math.ceil(raw_requests / 50.0)
        #concurrency = max(1, min(ideal_workers, 20))
        concurrency = max(1, min(int(raw_requests / 10), 10))
        #safe_requests = max(1.0, raw_requests)
        #target_qps = raw_requests / 60.0

        target_qps = (raw_requests * self.throughput_multiplier) / 60.0
        q_per_worker = target_qps / concurrency
        
        work_param = 1000000016000000063
        
        command = f"hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s -m GET {self.service_url}factor?n={work_param}"
        #command = f"hey -c {concurrency} -q {q_per_worker:.4f} -z 60s -m GET {self.service_url}factor?n={work_param}"
        logging.info(f"Step {self.steps}: Trace={raw_requests:.0f} | Workers={concurrency} | QPS={target_qps:.2f}")

        try:
            output = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=80)
            self._parse_hey_output(output.stdout + "\n" + output.stderr, raw_requests)
        except Exception as e:
            logging.error(f"Hey error: {e}")
            self._latency_p90 = 1; self._latency_avg = 1; self._success_ratio = 0.0

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
        
        # --- RETURN 13 DIMS (No Forecast) ---
        return np.array([lat, reps, cpu, ram, eff_req, t_cpu, t_ram, succ, self.hpa_target, self.throughput_multiplier, self.enhancement, math.cos(angle), math.sin(angle)], dtype=np.float32)

    def compute_reward(self):
        lat = float(self._latency_p90)
        cpu = float(self.state[2])
        replicas = int(self.state[1])
        succ = float(self._success_ratio)
    
        # SLA piecewise
        L_TARGET = 0.020
        L_THRESH = 0.050
        if lat <= L_TARGET:
            r_sla = 1.0
        elif lat <= L_THRESH:
            r_sla = 0.5 + 0.5 * (L_THRESH - lat) / (L_THRESH - L_TARGET)
        else:
            r_sla = max(-1.0, -0.5 * (lat - L_THRESH) / 0.1)
    
        # CPU centered at current HPA target
        thpa = float(self.hpa_target)
        if abs(cpu - thpa) <= 10.0:
            r_cpu = 1.0
        else:
            r_cpu = float(np.exp(-((cpu - thpa) / 50.0) ** 2))
    
        # Stability penalty
        delta = abs(replicas - int(self.last_replicas))
        if delta <= 2:
            r_stab = -0.1 * delta
        else:
            r_stab = -0.5 * delta
    
        # Success term
        if succ >= 0.99:
            r_succ = 1.0
        else:
            r_succ = float(np.log(max(succ, 1e-6)))
    
        # No-forecast: renormalize weights (drop 0.05 forecast)
        W_SLA  = 0.50 / 0.95
        W_CPU  = 0.25 / 0.95
        W_STAB = 0.08 / 0.95
        W_SUCC = 0.12 / 0.95
    
        reward = W_SLA * r_sla + W_CPU * r_cpu + W_STAB * r_stab + W_SUCC * r_succ
    
        self.current_step_reward = float(reward)
        self.reward = float(reward)
        self.last_replicas = replicas
        return float(reward)

    def apply_multiagent_action(self, action):
        opts = [30, 50, 70, 90]
        new_t = opts[int(action[0]) % 4]
        if new_t != self.hpa_target:
            self.prev_hpa_target = self.hpa_target; self.hpa_target = new_t
            patch = {"spec": {"metrics": [{"type": "Resource","resource": {"name": "cpu","target": {"type": "Utilization","averageUtilization": new_t}}}]}}
            try: subprocess.run(['kubectl', 'patch', 'hpa', application, '-n', app_env, '--patch', json.dumps(patch)], stdout=subprocess.DEVNULL)
            except: pass
        self.throughput_multiplier = [1.0, 2.0, 3.0][int(action[2]) % 3]
        self.enhancement = int(action[3]) % 3

    def step(self, action):
        self.apply_multiagent_action(action)
        reqs, lat, succ = self.run_hey()
        
        raw_state = self.get_new_state(reqs, lat, succ)
        # Clip 13 dim vector
        raw_state[0] = np.clip(raw_state[0], self.observation_space.low[0], self.observation_space.high[0])
        
        self._latency_p90 = lat; self._success_ratio = succ
        
        # === BASELINE: Single State (No History Stacking) ===
        self.state = np.clip(raw_state, self.observation_space.low, self.observation_space.high)
        
        reward = self.compute_reward()
        self.steps += 1; self.global_step += 1
        term = self.global_step >= (self.days_train * self.max_minutes)        
        return self.state, float(reward), term, False, {}

    def reset(self, seed=None, options=None):
        self.days = 0
        self.steps = 0
        self.global_step = 0
        self.last_replicas = 1
    
        logging.info("Resetting Cluster (Delete HPA, Scale=1, Recreate HPA@50)...")
    
        subprocess.run(["kubectl", "delete", "hpa", application, "-n", app_env],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        subprocess.run(["kubectl", "scale", "deploy", application, "-n", app_env, "--replicas=1"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        time.sleep(2)
    
        subprocess.run(["kubectl", "autoscale", "deploy", application, "-n", app_env,
                        "--cpu-percent=50", "--min=1", "--max=200"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        time.sleep(5)
    
        self.hpa_target = 50
        self.throughput_multiplier = 1.0
        self.enhancement = 0
    
        self.state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 0, 1, 0], dtype=np.float32)
        return self.state, {}

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
                f"{'='*88}\n"
            )
            logging.info(msg)
        except: pass
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path
        self.manual_env = None  # Needed for testing
        
        self.headers = ["Step", "Reward", "Latency_P90", "Latency_Avg", "Replicas", "CPU_Pct", "RAM_Pct", 
                        "Requests", "Total_CPU", "Total_RAM", "Success", "HPA_Target", "Throughput", "Enhancement", "Forecast"]
        
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w', newline='') as f: 
                csv.DictWriter(f, fieldnames=self.headers).writeheader()
            logging.info(f"✅ CSV Log initialized at: {os.path.abspath(csv_path)}")

    def _on_step(self) -> bool:
        # 1. Determine Environment Source (Train vs Test)
        if self.manual_env:
            env = self.manual_env.unwrapped.envs[0]
        else:
            env = self.training_env.unwrapped.envs[0]

        s = env.state
        
        # 2. Safe Feature Extraction
        # If state has 13 items, Forecast (index 13) doesn't exist.
        forecast_val = s[13] if len(s) > 13 else 0.0

        # 3. Tensorboard Logging
        try:
            self.logger.record("train/reward", env.reward)
            self.logger.record("train/latency_p90", env._latency_p90 * 1000)
            self.logger.record("train/replicas", env.state[1])
            self.logger.dump(step=self.num_timesteps)
        except: pass
        
        raw_requests_val = s[4] / s[9] if s[9] > 0 else s[4]

        # 4. CSV Logging
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
                    "Throughput": s[9], "Enhancement": int(s[10]), 
                    "Forecast": forecast_val  # <--- FIXED HERE
                })
                f.flush(); os.fsync(f.fileno())
        return True

lr_schedule = lambda p: cosine_schedule(p, lr_start=2e-4, lr_end=0.0)

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
    
    current_dir = os.getcwd()
    
    # 1. Create Results Directory
    log_dir = "results_log"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/tb", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_csv = f"{log_dir}/train_log_{timestamp}.csv"
    test_csv = f"{log_dir}/test_log_{timestamp}.csv"

    # 2. Start Prometheus (Port 9091)
    try: start_http_server(9091) 
    except: pass
    
    # 3. Pre-flight
    if not wait_for_service_availability(args.url): sys.exit(1)

    df = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df = add_day_column(df)
    tr_days, te_days = get_random_days(df)
    
    def build_env(days): return MultiAgentClusterEnv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", days, service_url=args.url)
    
    vec_train = VecNormalize(DummyVecEnv([lambda: build_env(tr_days)]), norm_obs=True, norm_reward=False, clip_obs=10.)
    
    if args.mode == "train":
        logging.info("Starting Baseline LSTM Training...")
        logging.info(f"📊 Logging to: {train_csv}")
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            vec_train,
            n_steps=128, batch_size=128,
            gamma=0.99, gae_lambda=0.95,
            learning_rate=lr_schedule, 
            policy_kwargs={
                'n_lstm_layers': 1,          # Single Layer
                'lstm_hidden_size': 256,     # 256 Units
                'net_arch': dict(pi=[64, 64], vf=[64, 64]) # 2x64 MLP
            }, 
            verbose=1, tensorboard_log=f"{log_dir}/tb", n_epochs=10, device='cuda'
        )
        
        cbs = [
            DetailedLoggingCallback(), 
            TensorboardCallback(train_csv), 
            CheckpointCallback(500, f"{log_dir}/ckpt_baseline")
        ]
        model.learn(total_timesteps=len(tr_days)*MINUTES_PER_DAY, callback=cbs)
        
        model.save(f"{log_dir}/final_model_baseline")
        vec_train.save(f"{log_dir}/vecnorm_baseline.pkl")
        logging.info("Done Training!")

    if args.mode == "test":
        logging.info("Starting Testing...")
        if not os.path.exists(f"{log_dir}/final_model_baseline.zip"): sys.exit(1)
            
        vec_test = VecNormalize.load(f"{log_dir}/vecnorm_baseline.pkl", DummyVecEnv([lambda: build_env(te_days)]))
        vec_test.training = False; vec_test.norm_reward = False
        model = RecurrentPPO.load(f"{log_dir}/final_model_baseline")
        
        test_cb = TensorboardCallback(test_csv)
        test_cb.manual_env = vec_test
        obs = vec_test.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        total_steps = len(te_days) * MINUTES_PER_DAY
        
        for i in range(total_steps):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, done, info = vec_test.step(action)
            episode_starts = done

            test_cb.num_timesteps = i
            test_cb._on_step()
            
            #test_cb.training_env = vec_test; test_cb.num_timesteps = i; test_cb._on_step()
            if i % 10 == 0: logging.info(f"Test Step {i}: Reward={float(reward[0]):.2f}")
        logging.info("Done Testing!")
