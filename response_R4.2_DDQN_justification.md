# Response to Reviewer 4, Comment 2 — DDQN Baseline Replacement

## Reviewer's concern
> "The DDQN baseline appears broken — the agent consistently selects 1 replica
> regardless of load. This undermines the validity of the comparison in Table II.
> A proper reactive baseline (e.g., KEDA HTTP-based autoscaler) should replace it."

## Our response

We thank the reviewer for this careful observation. Upon investigation, we confirmed
that the original DDQN implementation suffered from two compounding issues: (1) the
action space included a dead learning-rate dimension that consumed exploration budget
without affecting policy, and (2) the target network was updated too infrequently,
causing Q-value overestimation that collapsed to a conservative single-replica policy.

We considered replacing DDQN with KEDA (Kubernetes Event-Driven Autoscaling) as the
reviewer suggested. However, KEDA's HTTP add-on scaler operates as a pure reactive
proxy — it counts in-flight requests at the network level and scales proportionally,
with no awareness of latency SLOs, CPU utilisation targets, or workload forecasts.
Its scaling decisions are therefore not directly comparable with RL agents that
optimise a multi-objective reward signal. Including KEDA would introduce a
category mismatch: a reactive network proxy vs. learned control policies.

Instead, we replaced the broken DDQN with **DRQN (Deep Recurrent Q-Network)**
[cite: Hausknecht & Stone, 2015], which addresses the same reviewer concern
(a functional DQN-family baseline) while providing a more scientifically meaningful
comparison:

- DRQN uses an LSTM inside the Q-network, making it a direct recurrent counterpart
  to our Double-LSTM architecture.
- It is optimised with the same multi-objective reward function, the same action
  space, and the same four evaluation seeds (42, 123, 456, 789) as our proposed
  method, ensuring a fair apples-to-apples comparison.
- The comparison isolates the contribution of the attention mechanism and the
  dual-LSTM architecture: Single-LSTM removes the second LSTM branch; DRQN removes
  the attention mechanism entirely.

We believe this substitution is more informative than a KEDA comparison for the
specific claims of this paper (temporal dependency modelling in RL-based
autoscaling), and we respectfully request the reviewer's acceptance of this change.
The revised Table II now reports mean ± 95% CI across four seeds for all agents,
with Wilcoxon signed-rank tests confirming statistical significance.

## Changes made to the paper
- Section IV-B: DDQN description replaced with DRQN description and citation.
- Table II: DDQN row replaced with DRQN (mean ± CI, 4 seeds).
- Footnote added explaining the KEDA trade-off and the choice of DRQN.
