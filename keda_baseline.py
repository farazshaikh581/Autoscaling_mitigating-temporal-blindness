# KEDA HTTP add-on autoscaling baseline. Pure reactive scaler, no learning.
# Reward uses the same weights/formula as the RL agents (r_cpu centered on the
# native HPA's 50% target since KEDA has no CPU target of its own) but it's a
# summary stat only — latency/SLA/replicas are the real comparison.
# Client mode (concurrency formula, keepalive, load-multiplier amplification)
# matches ddqn_agent.py/dlstm_agent.py so the comparison is apples to apples.
#
# Install KEDA before running (see keda-httpscaledobject.yaml header).

import os
import sys
import math
import subprocess
import re
import json
import csv
import logging
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MINUTES_PER_DAY = 500
application     = "factorizator"
app_env         = "factorizator"
INTERCEPTOR_URL = "http://<cluster-ip>:30002/"   # KEDA interceptor NodePort
KEDA_MANIFEST   = os.path.join(os.path.dirname(__file__), "keda-httpscaledobject.yaml")
WORK_PARAM      = 1000000016000000063

def add_day_column(df, minutes_per_day=MINUTES_PER_DAY):
    t0 = df.end_timestamp.min()
    df['minute'] = (df.end_timestamp - t0) // 60
    df['day']    = (df['minute'] // minutes_per_day).astype(int)
    return df

def get_test_days(df, n_days=7, train_days=5, test_days=2, seed=42):
    import random
    all_days = sorted(df.day.unique())
    random.seed(seed)
    selected = random.sample(list(all_days), min(n_days, len(all_days)))
    return selected[train_days:train_days + test_days]

def get_replicas():
    try:
        out = subprocess.check_output(
            ['kubectl', 'get', 'deploy', application, '-n', app_env,
             '-o', 'jsonpath={.spec.replicas}'], text=True)
        return int(out.strip())
    except:
        return 1

def get_resources_usage():
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
        cpu_u = min((sum(cpu_vals) / len(cpu_vals)) / 0.25 * 100, 200) if cpu_vals else 1
        ram_u = min((sum(ram_vals) / len(ram_vals)) / 128 * 100, 200) if ram_vals else 1
        return cpu_u, ram_u, sum(cpu_vals) * 1000, sum(ram_vals)
    except:
        return 1, 1, 1, 1

def parse_hey(output):
    p90 = re.search(r"90%\s+in\s+([\d.]+)\s+secs", output)
    avg = re.search(r"Average:\s+([\d.]+)\s+secs", output)
    lat_p90 = float(p90.group(1)) if p90 else 0.05
    lat_avg = float(avg.group(1)) if avg else 0.05
    # hey prints 10/25/50/75/90/95/99% lines; the original parser kept only 90%.
    # Falls back to P90 when a bucket is unpopulated on a low-sample idle step.
    def pct(tag):
        m = re.search(r"%s%%\s+in\s+([\d.]+)\s+secs" % tag, output)
        return float(m.group(1)) if m else lat_p90
    lat_p50, lat_p95, lat_p99 = pct("50"), pct("95"), pct("99")
    status = re.findall(r"\[(\d+)\]\s+(\d+)\s+responses", output)
    counts = {int(k): int(v) for k, v in status}
    if counts:
        succ = counts.get(200, 0) / sum(counts.values())
    else:
        succ = 1.0 if "Error distribution" not in output else 0.0
    return lat_avg, lat_p50, lat_p90, lat_p95, lat_p99, succ

def run_hey(url, raw_requests, host_header=False, duration="60s", load_multiplier=1.0):
    """Same concurrency/QPS formula as the RL agents' client. Returns
    (effective_reqs, lat_avg, lat_p50, lat_p90, lat_p95, lat_p99, succ)."""
    if raw_requests < 1.0:
        time.sleep(60)
        return 0.0, 0.005, 0.005, 0.005, 0.005, 0.005, 1.0
    concurrency = max(1, min(int(raw_requests / 10), 10))
    target_qps  = (raw_requests * load_multiplier) / 60.0
    # hey ignores Host set via -H, needs -host or the interceptor 404s everything
    host = '-host "factorizator.local" ' if host_header else ''
    cmd = (f'hey -c {concurrency} -q {max(0.001, target_qps):.4f} -z {duration} '
           f'{host}-m GET {url}factor?n={WORK_PARAM}')
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=80)
        output = result.stdout + "\n" + result.stderr
    except Exception as e:
        logging.error(f"hey error: {e}")
        return raw_requests * load_multiplier, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
    lat_avg, lat_p50, lat_p90, lat_p95, lat_p99, succ = parse_hey(output)
    return (raw_requests * load_multiplier, lat_avg, lat_p50, lat_p90,
            lat_p95, lat_p99, succ)

def calibrate_interceptor_overhead(app_url, interceptor_url, rounds, log_dir):
    """Direct-vs-interceptor latency at fixed load (60 req/min), 1 replica,
    no autoscaler active. Saves per-round numbers + mean deltas to a JSON file."""
    logging.info(f"CALIBRATION: {rounds}x direct vs {rounds}x interceptor rounds (60 req/min, 1 replica)")
    subprocess.run(['kubectl', 'scale', 'deploy', application, '-n', app_env, '--replicas=1'],
                   stdout=subprocess.DEVNULL)
    time.sleep(10)
    results = {'direct': [], 'interceptor': []}
    for i in range(rounds):
        _, da, _, dp, _, _, _ = run_hey(app_url, 60.0)
        results['direct'].append({'avg_ms': da * 1000, 'p90_ms': dp * 1000})
        logging.info(f"  round {i+1} direct     : avg={da*1000:.1f}ms p90={dp*1000:.1f}ms")
        _, ia, _, ip, _, _, _ = run_hey(interceptor_url, 60.0, host_header=True)
        results['interceptor'].append({'avg_ms': ia * 1000, 'p90_ms': ip * 1000})
        logging.info(f"  round {i+1} interceptor: avg={ia*1000:.1f}ms p90={ip*1000:.1f}ms")
    mean = lambda k, f: float(np.mean([r[f] for r in results[k]]))
    summary = {
        'direct_avg_ms': mean('direct', 'avg_ms'), 'direct_p90_ms': mean('direct', 'p90_ms'),
        'interceptor_avg_ms': mean('interceptor', 'avg_ms'), 'interceptor_p90_ms': mean('interceptor', 'p90_ms'),
    }
    summary['overhead_avg_ms'] = summary['interceptor_avg_ms'] - summary['direct_avg_ms']
    summary['overhead_p90_ms'] = summary['interceptor_p90_ms'] - summary['direct_p90_ms']
    out = {'rounds': results, 'summary': summary,
           'note': 'overhead = interceptor proxy hop cost; report alongside KEDA latency in the paper'}
    path = os.path.join(log_dir, 'interceptor_calibration.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    logging.info(f"CALIBRATION result: proxy overhead avg={summary['overhead_avg_ms']:.1f}ms "
                 f"p90={summary['overhead_p90_ms']:.1f}ms -> {path}")
    return summary

def keda_setup(target_rpm):
    """Delete the native HPA and apply the HTTPScaledObject with the requested
    per-replica target (expressed in req/min via granularity 1m so integers work)."""
    logging.info(f"KEDA setup: removing native HPA, applying HTTPScaledObject (target {target_rpm} req/min/replica)...")
    subprocess.run(['kubectl', 'delete', 'hpa', application, '-n', app_env],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(KEDA_MANIFEST) as f:
        manifest = f.read()
    manifest = re.sub(r'granularity:\s*\S+', 'granularity: 1m', manifest)
    manifest = re.sub(r'targetValue:\s*\d+', f'targetValue: {int(target_rpm)}', manifest)
    subprocess.run(['kubectl', 'apply', '-f', '-'], input=manifest, text=True,
                   check=True, stdout=subprocess.DEVNULL)
    for _ in range(20):
        out = subprocess.run(['kubectl', 'get', 'hpa', '-n', app_env, '--no-headers'],
                             capture_output=True, text=True)
        if 'keda-hpa' in out.stdout:
            logging.info("KEDA internal HPA is active.")
            return
        time.sleep(3)
    logging.warning("KEDA HPA did not appear within 60s — proceeding anyway.")

def keda_teardown(restore_max):
    logging.info("KEDA teardown: removing HTTPScaledObject, restoring native HPA...")
    subprocess.run(['kubectl', 'delete', 'httpscaledobject', 'factorizator-keda',
                    '-n', app_env], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    subprocess.run(['kubectl', 'scale', 'deploy', application, '-n', app_env,
                    '--replicas=1'], stdout=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(['kubectl', 'autoscale', 'deploy', application, '-n', app_env,
                    '--cpu-percent=50', '--min=1', f'--max={restore_max}'],
                   stdout=subprocess.DEVNULL)
    logging.info(f"Native HPA restored (max={restore_max}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KEDA HTTP autoscaling baseline — evaluation only")
    parser.add_argument("--url",         type=str, default=INTERCEPTOR_URL,
                        help="KEDA interceptor URL")
    parser.add_argument("--app-url",     type=str, default=None,
                        help="Direct app URL, enables interceptor overhead calibration")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Test-day seed, matches the RL agents")
    parser.add_argument("--target-rpm",  type=int, default=100,
                        help="KEDA per-replica target, req/min")
    parser.add_argument("--load-multiplier", type=float, default=3.0,
                        help="QPS amplification, matches the RL agents' throughput multiplier")
    parser.add_argument("--restore-max", type=int, default=200,
                        help="maxReplicas for the restored native HPA")
    parser.add_argument("--calib-rounds", type=int, default=3,
                        help="calibration rounds (0 disables, requires --app-url)")
    parser.add_argument("--log-dir",     type=str, default="results_log_keda")
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, f"seed{args.seed}_target{args.target_rpm}_mult{args.load_multiplier:g}")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv   = f"{log_dir}/keda_test_{timestamp}.csv"

    # Calibration BEFORE any autoscaler is active (native HPA removed first)
    calib = None
    if args.calib_rounds > 0 and args.app_url:
        subprocess.run(['kubectl', 'delete', 'hpa', application, '-n', app_env],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        calib = calibrate_interceptor_overhead(args.app_url, args.url, args.calib_rounds, log_dir)
    elif args.calib_rounds > 0:
        logging.warning("Calibration skipped: --app-url not given.")

    keda_setup(args.target_rpm)
    time.sleep(5)

    logging.info(f"Pre-flight: checking interceptor at {args.url}")
    for _ in range(10):
        try:
            r = requests.get(f"{args.url}factor?n=1", timeout=5,
                             headers={"Host": "factorizator.local"})
            if r.status_code == 200:
                logging.info("Interceptor UP and routing correctly.")
                break
        except Exception as e:
            logging.warning(f"Interceptor not ready yet: {e}")
            time.sleep(3)
    else:
        logging.error("Interceptor unreachable after 30s."); keda_teardown(args.restore_max); sys.exit(1)

    trace_file = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    df = add_day_column(pd.read_csv(trace_file))
    te_days = get_test_days(df, seed=args.seed)
    logging.info(f"Test days for seed {args.seed}: {te_days}")

    matrix = np.zeros((len(te_days), MINUTES_PER_DAY))
    for i, day in enumerate(te_days):
        mask    = df['day'] == day
        minutes = df.loc[mask, 'minute'] % MINUTES_PER_DAY
        for m in minutes:
            if 0 <= int(m) < MINUTES_PER_DAY:
                matrix[i, int(m)] += 1

    headers = ["Step", "Reward", "Latency_P50", "Latency_P90", "Latency_P95",
               "Latency_P99", "Latency_Avg", "Replicas",
               "CPU_Pct", "RAM_Pct", "Base_Requests", "Requests", "Total_CPU", "Total_RAM",
               "Success", "Throughput", "Target_RPM"]
    with open(out_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()

    total_steps = len(te_days) * MINUTES_PER_DAY
    logging.info(f"KEDA baseline evaluation: {total_steps} steps | target={args.target_rpm} req/min | "
                 f"load_multiplier={args.load_multiplier} | log={out_csv}")

    last_replicas = 1
    for step in range(total_steps):
        day_idx = step // MINUTES_PER_DAY
        min_idx = step % MINUTES_PER_DAY
        raw_req = float(matrix[day_idx, min_idx])

        reqs, lat_avg, lat_p50, lat_p90, lat_p95, lat_p99, succ = run_hey(args.url, raw_req, host_header=True,
                                               load_multiplier=args.load_multiplier)
        cpu, ram, t_cpu, t_ram = get_resources_usage()
        reps = get_replicas()

        # same reward formula/weights as ddqn_agent.py & dlstm_agent.py
        L_TARGET, L_THRESH = 0.020, 0.050
        if lat_p90 <= L_TARGET:
            r_sla = 1.0
        elif lat_p90 <= L_THRESH:
            r_sla = 0.5 + 0.5 * (L_THRESH - lat_p90) / (L_THRESH - L_TARGET)
        else:
            r_sla = max(-1.0, -0.5 * (lat_p90 - L_THRESH) / 0.1)
        r_cpu  = 1.0 if abs(cpu - 50.0) <= 10 else float(np.exp(-((cpu - 50.0) / 50.0) ** 2))
        delta  = abs(reps - last_replicas)
        r_stab = -0.1 * delta if delta <= 2 else -0.5 * delta
        r_succ = 1.0 if succ >= 0.99 else float(np.log(max(succ, 1e-6)))
        reward = 0.55 * r_sla + 0.25 * r_cpu + 0.08 * r_stab + 0.12 * r_succ
        last_replicas = reps

        row = {"Step": step, "Reward": reward,
               "Latency_P50": lat_p50 * 1000, "Latency_P90": lat_p90 * 1000,
               "Latency_P95": lat_p95 * 1000, "Latency_P99": lat_p99 * 1000,
               "Latency_Avg": lat_avg * 1000, "Replicas": reps,
               "CPU_Pct": cpu, "RAM_Pct": ram, "Base_Requests": raw_req, "Requests": reqs,
               "Total_CPU": t_cpu, "Total_RAM": t_ram, "Success": succ,
               "Throughput": args.load_multiplier, "Target_RPM": args.target_rpm}
        with open(out_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writerow(row)

        if step % 10 == 0:
            logging.info(f"Step {step}: Replicas={reps} | P90={lat_p90*1000:.1f}ms | "
                         f"Avg={lat_avg*1000:.1f}ms | Succ={succ:.3f}")

    logging.info(f"KEDA baseline complete. Results: {out_csv}")
    keda_teardown(args.restore_max)
