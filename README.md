# Mitigating Temporal Blindness in Kubernetes Autoscaling

**Paper:** *Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework*

Code, configuration files, and result logs for the experiments in the paper. Four autoscaling approaches are implemented and compared on a real Kubernetes cluster using the Azure Functions Invocation Trace (2021).

---

## Approaches

| Script | Method | Description |
|---|---|---|
| `double-lstm_agent.py` | Proposed | Attention + Double LSTM + PPO |
| `single-lstm_agent.py` | Baseline | Single LSTM + PPO (no attention) |
| `ddqn_agent.py` | Baseline | Double DQN (MlpPolicy, no forecast) |
| `static-hpa50.py` | Benchmark | Kubernetes HPA at 50% CPU target |

---

## Repository Structure

```
.
├── double-lstm_agent.py        # Proposed: Attention + Double LSTM + PPO
├── single-lstm_agent.py        # Baseline: Single LSTM + PPO
├── ddqn_agent.py               # Baseline: Double DQN
├── static-hpa50.py             # Benchmark: Static HPA @ 50% CPU
│
├── app.py                      # Factorizator Flask app (the workload being scaled)
├── Dockerfile                  # Container image for the factorizator app
├── energy_model.py             # Energy consumption model
├── analyze_traces.py           # Azure trace analysis
├── visualize_results.py        # Plot results from CSV logs
├── launch_experiment.sh        # Script to run an experiment
│
├── factorizator-namespace.yaml # K8s namespace
├── factorizator-deployment.yaml# K8s deployment
├── factorizator-service.yaml   # K8s service
├── hpa.yaml                    # HorizontalPodAutoscaler
├── RBAC.yaml                   # RBAC for metrics access
│
├── requirements.txt            # Python dependencies
├── Results-CSVs/               # Test logs for all four methods
└── Invocation Traces           # Link to Azure dataset download
```

---

## Prerequisites

- Python 3.9+
- Kubernetes cluster (tested with Minikube 2-node)
- `kubectl` configured for your cluster
- [`hey`](https://github.com/rakyll/hey) load generator on `$PATH`
- Docker
- Prometheus and metrics-server in the cluster

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/farazshaikh581/Autoscaling_mitigating-temporal-blindness.git
cd Autoscaling_mitigating-temporal-blindness
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Azure invocation trace

The dataset is not included in the repository due to its size (305 MB). Download it from the link in the `Invocation Traces` file:

```
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md
```

Place the downloaded file in the repo root:
```
AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
```

### 3. Build and deploy the factorizator workload

```bash
# Build and push the container image
docker build -t <your-dockerhub-username>/factorizator:latest .
docker push <your-dockerhub-username>/factorizator:latest

# Update the image field in factorizator-deployment.yaml to match your registry, then deploy
kubectl apply -f factorizator-namespace.yaml
kubectl apply -f factorizator-deployment.yaml
kubectl apply -f factorizator-service.yaml
kubectl apply -f RBAC.yaml
kubectl apply -f hpa.yaml

# Check pods are running
kubectl get pods -n factorizator
```

### 4. Expose the service

```bash
kubectl port-forward svc/factorizator 8080:8080 -n factorizator &
curl http://localhost:8080/health
```

---

## Running Experiments

All scripts take `--url` (the factorizator service URL) and `--mode train` or `--mode test`.

### Proposed: Attention + Double LSTM

```bash
python double-lstm_agent.py --mode train --url http://localhost:8080
python double-lstm_agent.py --mode test  --url http://localhost:8080
```

### Baseline: Single LSTM

```bash
python single-lstm_agent.py --mode train --url http://localhost:8080
python single-lstm_agent.py --mode test  --url http://localhost:8080
```

### Baseline: Double DQN

```bash
python ddqn_agent.py --mode train --url http://localhost:8080
python ddqn_agent.py --mode test  --url http://localhost:8080
```

### Benchmark: Static HPA @ 50%

```bash
python static-hpa50.py --url http://localhost:8080
```

### Launch script

```bash
# Usage: ./launch_experiment.sh <agent> <url> [mode] [seed]
chmod +x launch_experiment.sh
./launch_experiment.sh double-lstm http://localhost:8080 train 42
./launch_experiment.sh double-lstm http://localhost:8080 test 42
```

---

## Results

Test logs (1000 steps each) are in `Results-CSVs/`:

| File | Method |
|---|---|
| `test_log_Double-LSTM.csv` | Proposed (Attention + Double LSTM) |
| `test_log_Single-LSTM.csv` | Single LSTM baseline |
| `test_log_DDQN.csv` | Double DQN baseline |
| `test_log_static-hpa.csv` | Static HPA benchmark |

To plot the results:

```bash
python visualize_results.py
```

CSV columns (RL agents): `Step, Reward, Latency_P90, Latency_Avg, Replicas, CPU_Pct, RAM_Pct, Requests, Total_CPU, Total_RAM, Success, HPA_Target, Throughput, Enhancement, Forecast`

---

## Reproducibility

All results in the paper use random seed 42 (`SEED = 42` in all scripts). 7 days are sampled from the trace (5 train, 2 test), with `MINUTES_PER_DAY = 500` steps per simulated day.

---

## Citation

```bibtex
@article{shaikh2025temporal,
  title     = {Mitigating Temporal Blindness in Kubernetes Autoscaling:
               An Attention-Double-LSTM Framework},
  author    = {Shaikh, Faraz, Reali, Gianluca, and Femminella, Mauro},
  journal   = {(under review / to appear)},
  year      = {2026}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
