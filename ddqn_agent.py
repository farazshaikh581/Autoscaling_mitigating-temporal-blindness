# -*- coding: utf-8 -*-
# === DOUBLE DQN BASELINE ===
# Off-policy value-based RL baseline (no forecast, 13-dim state).
# Runs in train or test mode; supports --seed, --log-dir, --no-aux.

import os, sys, math, torch, numpy as np, pandas as pd
import subprocess, random, re, json, logging, time, warnings, csv, requests
import datetime, argparse, glob

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
MINUTES_PER_DAY = 500
application = "factorizator"
app_env = "factorizator"
MIN_REPLICAS = 1
MAX_REPLICAS = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")

# ============================================================
# UTILS
# ============================================================
def add_day_column(df, minutes_per_day=MINUTES_PER_DAY):
    t0 = df.end_timestamp.min()
    df['minute'] = (df.end_timestamp - t0) // 60
    df['day']    = (df['minute'] // minutes_per_day).astype(int)
    return df

def get_random_days(df, n_days=7, train_days=5, test_days=2, seed=None):
    seed = seed if seed is not None else SEED
    all_days = sorted(df.day.unique())
    random.seed(seed)
    selected = random.sample(list(all_days), min(n_days, len(all_days)))
    return selected[:train_days], selected[train_days:train_days + test_days]

def wait_for_service(url, max_retries=20):
    for i in range(max_retries):
        try:
            if requests.get(f"{url}factor?n=1", timeout=8).status_code == 200:
                logging.info("Service UP.")
                return True
        except Exception:
            pass
        time.sleep(3)
    logging.error("Service unreachable.")
    return False

# ============================================================
# ENVIRONMENT
# ============================================================
import gymnasium as gym

class DQNClusterEnv(gym.Env):
    def __init__(self, invocation_file, day_list, service_url, no_aux=False):
        super().__init__()
        self.service_url = service_url
        self.no_aux = no_aux
        self.df = pd.read_csv(invocation_file)
        self.df = add_day_column(self.df)
        self.day_list = day_list

        # [4 HPA targets] x [3 throughput multipliers] = 12
        # Dead learning-rate action removed (was action[1] in original [4,3,3,3] encoding).
        # Dead enhancement action/state field removed (never modified scaling behavior).
        self.action_space = gym.spaces.Discrete(4 * 3)

        self.invocation_matrix = self._make_matrix()
        self.days_train = len(day_list)

        self.steps = 0; self.days = 0; self.global_step = 0
        self.hpa_target = 50
        self.throughput_multiplier = 1.0
        self.last_replicas = 1
        self.reward = 0.0
        self._latency_p90 = 0.05; self._latency_avg = 0.05; self._success_ratio = 1.0

        # 12-dim state (no forecast signal — ablation vs Double-LSTM's 13-dim)
        low  = np.array([0.001, 1,   0,   0,   0,    0,    0,    0, 1,  1.0, -1, -1], dtype=np.float32)
        high = np.array([0.150, 200, 200, 100, 3600, 1800, 1800, 1, 95, 3.0,  1,  1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._init_state = np.array([0.05, 1, 30, 40, 100, 1500, 2000, 1, 50, 1.0, 1, 0], dtype=np.float32)
        self.state = self._init_state.copy()

    def _make_matrix(self):
        matrix = np.zeros((len(self.day_list), MINUTES_PER_DAY))
        for i, day in enumerate(self.day_list):
            mask = self.df['day'] == day
            minutes = self.df.loc[mask, 'minute'] % MINUTES_PER_DAY
            for m in minutes:
                if 0 <= int(m) < MINUTES_PER_DAY:
                    matrix[i, int(m)] += 1
        return matrix

    def decode_action(self, a: int):
        # Decode flat index → [hpa_idx, throughput_idx]
        a = int(a) % 12
        a0 = a // 3  # HPA target:  0–3
        a1 = a % 3   # Throughput:  0–2
        return a0, a1

    def apply_action(self, a0, a1):
        new_t = [30, 50, 70, 90][a0]
        if new_t != self.hpa_target:
            self.hpa_target = new_t
            patch = {"spec": {"metrics": [{"type": "Resource", "resource": {
                "name": "cpu", "target": {"type": "Utilization",
                "averageUtilization": new_t}}}]}}
            try:
                subprocess.run(['kubectl', 'patch', 'hpa', application, '-n', app_env,
                                '--patch', json.dumps(patch)], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        if not self.no_aux:
            self.throughput_multiplier = [1.0, 2.0, 3.0][a1]

    def _run_hey(self):
        day_idx = min(self.days, len(self.day_list) - 1)
        raw_req = float(self.invocation_matrix[day_idx, self.steps])
        if raw_req < 1.0:
            time.sleep(60)
            self._latency_p90 = 0.005; self._latency_avg = 0.005; self._success_ratio = 1.0
            return 0.0

        target_qps = (raw_req * self.throughput_multiplier) / 60.0
        concurrency = max(1, min(int(raw_req / 10), 10))
        cmd = (f'hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z 60s -m GET '
               f'{self.service_url}factor?n=1000000016000000063')
        logging.info(f"Step {self.steps}: Trace={raw_req:.0f} | QPS={target_qps:.2f}")
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=80)
            output = out.stdout + "\n" + out.stderr
        except Exception as e:
            logging.error(f"hey error: {e}")
            self._latency_p90 = 1.0; self._latency_avg = 1.0; self._success_ratio = 0.0
            return raw_req

        p90 = re.search(r"90%\s+in\s+([\d.]+)\s+secs", output)
        avg = re.search(r"Average:\s+([\d.]+)\s+secs", output)
        self._latency_p90 = float(p90.group(1)) if p90 else 0.05
        self._latency_avg = float(avg.group(1)) if avg else 0.05
        status = re.findall(r"\[(\d+)\]\s+(\d+)\s+responses", output)
        counts = {int(k): int(v) for k, v in status}
        self._success_ratio = (counts.get(200, 0) / sum(counts.values())) if counts else 1.0
        return raw_req

    def _get_state(self, reqs):
        try:
            out = subprocess.check_output(
                ['kubectl', 'top', 'pod', '-l', f'app={application}', '-n', app_env],
                encoding='utf-8', stderr=subprocess.DEVNULL)
            lines = out.strip().split('\n')[1:]
            cpu_vals, ram_vals = [], []
            for line in lines:
                cols = line.split()
                if len(cols) >= 3:
                    cpu_vals.append(float(cols[1].replace('m', '')) / 1000 if 'm' in cols[1] else float(cols[1]))
                    ram_vals.append(float(cols[2].replace('Mi', '')) if 'Mi' in cols[2] else float(cols[2]))
            cpu = min((sum(cpu_vals) / len(cpu_vals)) / 0.25 * 100, 200) if cpu_vals else 1
            ram = min((sum(ram_vals) / len(ram_vals)) / 128 * 100, 200) if ram_vals else 1
            t_cpu, t_ram = sum(cpu_vals) * 1000, sum(ram_vals)
        except Exception:
            cpu, ram, t_cpu, t_ram = 1, 1, 1, 1
        try:
            reps = int(subprocess.check_output(
                ['kubectl', 'get', 'deploy', application, '-n', app_env,
                 '-o', 'jsonpath={.spec.replicas}'], text=True).strip())
        except Exception:
            reps = 1
        angle = (self.steps / MINUTES_PER_DAY) * 2 * math.pi
        return np.array([self._latency_p90, reps, cpu, ram,
                         reqs * self.throughput_multiplier, t_cpu, t_ram,
                         self._success_ratio, self.hpa_target,
                         self.throughput_multiplier,
                         math.cos(angle), math.sin(angle)], dtype=np.float32)

    def _compute_reward(self):
        lat = self._latency_p90
        reps = int(self.state[1])
        cpu = float(self.state[2])
        succ = self._success_ratio

        r_slo = (1.0 if lat <= 0.020 else
                 0.5 + 0.5 * (0.050 - lat) / 0.030 if lat <= 0.050 else
                 max(-1.0, -0.5 * (lat - 0.050) / 0.1))
        r_cpu = (1.0 if abs(cpu - self.hpa_target) <= 10
                 else float(np.exp(-((cpu - self.hpa_target) / 50.0) ** 2)))
        delta = abs(reps - self.last_replicas)
        r_stab = -0.1 * delta if delta <= 2 else -0.5 * delta
        r_succ = 1.0 if succ >= 0.99 else float(np.log(max(succ, 1e-6)))

        # Rt = w_slo*R_SLO + w_cpu*R_CPU + w_succ*R_Succ + w_stab*R_Stab (no forecast term)
        reward = 0.55 * r_slo + 0.25 * r_cpu + 0.12 * r_succ + 0.08 * r_stab
        self.reward = float(reward)
        self.last_replicas = reps
        return self.reward

    def step(self, action):
        a0, a1 = self.decode_action(int(action))
        self.apply_action(a0, a1)
        reqs = self._run_hey()
        raw = self._get_state(reqs)
        low = self.observation_space.low; high = self.observation_space.high
        self.state = np.clip(raw, low, high)
        reward = self._compute_reward()
        self.steps += 1
        self.global_step += 1
        if self.steps >= MINUTES_PER_DAY:
            self.days += 1; self.steps = 0
        done = self.global_step >= self.days_train * MINUTES_PER_DAY
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.days = 0; self.steps = 0; self.global_step = 0
        self.hpa_target = 50; self.throughput_multiplier = 1.0
        self.last_replicas = 1
        subprocess.run(['kubectl', 'delete', 'hpa', application, '-n', app_env],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['kubectl', 'scale', 'deploy', application, '-n', app_env, '--replicas=1'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        subprocess.run(['kubectl', 'autoscale', 'deploy', application, '-n', app_env,
                        '--cpu-percent=50', f'--min={MIN_REPLICAS}', f'--max={MAX_REPLICAS}'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        self.state = self._init_state.copy()
        return self.state, {}

# ============================================================
# CHECKPOINT HELPERS
# ============================================================
class CheckpointWithVecNorm(BaseCallback):
    """Saves model + VecNormalize stats together every save_freq steps."""
    def __init__(self, save_freq, save_path, vec_env):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(f"{self.save_path}/ckpt_{self.num_timesteps}_steps")
            self.vec_env.save(f"{self.save_path}/vecnorm_{self.num_timesteps}_steps.pkl")
            logging.info(f"Checkpoint saved at step {self.num_timesteps}")
        return True

def find_latest_checkpoint(ckpt_dir):
    """Returns (model_path, vecnorm_path, steps_done) for the best complete checkpoint, or (None, None, 0).
    Iterates from largest to smallest step count so a missing vecnorm on the latest checkpoint
    falls back to the previous one rather than forcing a full restart."""
    zips = glob.glob(f"{ckpt_dir}/ckpt_*_steps.zip")
    if not zips:
        return None, None, 0
    steps = sorted(
        [int(re.search(r'ckpt_(\d+)_steps', z).group(1)) for z in zips],
        reverse=True
    )
    for n in steps:
        m = f"{ckpt_dir}/ckpt_{n}_steps.zip"
        v = f"{ckpt_dir}/vecnorm_{n}_steps.pkl"
        if os.path.exists(m) and os.path.exists(v):
            return m, v, n
    return None, None, 0

# ============================================================
# CALLBACK
# ============================================================
class CSVCallback(BaseCallback):
    FIELDS = ["Step", "Reward", "Latency_P90", "Latency_Avg",
              "Replicas", "CPU_Pct", "Requests", "Success"]

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def _on_step(self) -> bool:
        env = self.training_env.unwrapped.envs[0]
        s = env.state
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow({
                "Step": self.num_timesteps,
                "Reward": env.reward,
                "Latency_P90": env._latency_p90 * 1000,
                "Latency_Avg": env._latency_avg * 1000,
                "Replicas": int(s[1]),
                "CPU_Pct": float(s[2]),
                "Requests": float(s[4]),
                "Success": float(env._success_ratio),
            })
        return True

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double DQN baseline for Kubernetes autoscaling")
    parser.add_argument("--mode",    default="train", choices=["train", "test"])
    parser.add_argument("--url",     type=str, required=True)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="results_log_ddqn")
    parser.add_argument("--no-aux",  action="store_true",
                        help="Fix throughput multiplier=1.0 (fair HPA comparison)")
    args = parser.parse_args()

    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    suffix = "_no_aux" if args.no_aux else ""
    log_dir = os.path.join(args.log_dir, f"seed{SEED}{suffix}")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not wait_for_service(args.url):
        sys.exit(1)

    df = pd.read_csv("AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")
    df = add_day_column(df)
    tr_days, te_days = get_random_days(df, seed=SEED)

    if args.mode == "train":
        final_model_path = os.path.join(log_dir, "ddqn_model.zip")
        if os.path.exists(final_model_path):
            logging.info("Training already complete (final model exists). Skipping.")
        else:
            logging.info(f"Starting Double DQN training | seed={SEED} | log={log_dir}")
            ckpt_dir = os.path.join(log_dir, "ckpt")
            ckpt_model, ckpt_vecnorm, ckpt_steps = find_latest_checkpoint(ckpt_dir)

            base_env = DummyVecEnv([lambda: DQNClusterEnv(
                "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt",
                tr_days, args.url, no_aux=args.no_aux)])

            total_steps = len(tr_days) * MINUTES_PER_DAY   # 5 × 500 = 2500
            remaining_steps = total_steps - ckpt_steps

            if ckpt_model and ckpt_vecnorm:
                logging.info(f"Resuming from checkpoint at step {ckpt_steps} ({remaining_steps} steps left)")
                env = VecNormalize.load(ckpt_vecnorm, base_env)
                model = DQN.load(ckpt_model, env=env, device=device)
                model.set_env(env)
            else:
                env = VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
                model = DQN(
                    "MlpPolicy", env,
                    verbose=1,
                    learning_rate=1e-4,
                    gamma=0.99,
                    buffer_size=min(total_steps, 5000),
                    batch_size=32,
                    learning_starts=50,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=500,
                    exploration_fraction=0.4,
                    exploration_final_eps=0.05,
                    optimize_memory_usage=False,
                    tensorboard_log=os.path.join(log_dir, "tb"),
                    device=device,
                )

            cbs = [CSVCallback(os.path.join(log_dir, f"ddqn_train_{ts}.csv")),
                   CheckpointWithVecNorm(500, ckpt_dir, env)]
            model.learn(total_timesteps=remaining_steps, callback=cbs,
                        reset_num_timesteps=(ckpt_steps == 0))
            model.save(os.path.join(log_dir, "ddqn_model"))
            env.save(os.path.join(log_dir, "vecnorm.pkl"))
            logging.info("Double DQN training complete.")

    elif args.mode == "test":
        logging.info(f"Starting Double DQN evaluation | seed={SEED} | log={log_dir}")
        env = DummyVecEnv([lambda: DQNClusterEnv(
            "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt",
            te_days, args.url, no_aux=args.no_aux)])
        vn_path = os.path.join(log_dir, "vecnorm.pkl")
        if os.path.exists(vn_path):
            env = VecNormalize.load(vn_path, env)
            env.training = False; env.norm_reward = False
        model = DQN.load(os.path.join(log_dir, "ddqn_model"), env=env, device=device)

        headers = ["Step", "Reward", "Latency_P90", "Latency_Avg",
                   "Replicas", "CPU_Pct", "Requests", "Success"]
        out_csv = os.path.join(log_dir, f"ddqn_test_{ts}.csv")
        with open(out_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

        obs = env.reset()
        total_steps = len(te_days) * MINUTES_PER_DAY
        for i in range(total_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            raw_env = env.envs[0]
            s = raw_env.state
            with open(out_csv, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=headers).writerow({
                    "Step": i,
                    "Reward": float(reward[0]),
                    "Latency_P90": raw_env._latency_p90 * 1000,
                    "Latency_Avg": raw_env._latency_avg * 1000,
                    "Replicas": int(s[1]),
                    "CPU_Pct": float(s[2]),
                    "Requests": float(s[4]),
                    "Success": float(raw_env._success_ratio),
                })
            if i % 10 == 0:
                logging.info(f"Step {i}: Reps={int(s[1])} | Lat={raw_env._latency_p90*1000:.1f}ms | "
                             f"Succ={raw_env._success_ratio:.3f}")
        logging.info(f"Double DQN evaluation complete. Results: {out_csv}")
