# Response to Reviewer 4, Comment 2 — DDQN Baseline Fix

## Reviewer's concern
> "The DDQN baseline appears broken — the agent consistently selects 1 replica
> regardless of load. This undermines the validity of the comparison in Table II.
> A proper reactive baseline (e.g., KEDA HTTP-based autoscaler) should replace it."

## Our response

We thank the reviewer for identifying this issue. Upon investigation, we confirmed
two root causes that caused DDQN to collapse to a single-replica policy:

1. **Dead action dimension.** The original action space was `MultiDiscrete([4,3,3,3])`,
   where dimension 1 was a learning-rate schedule selector that had no effect on
   environment behaviour. This wasted a significant portion of exploration budget on
   a dimension that produced no reward signal, starving the policy of useful gradient.
   We corrected the action space to `MultiDiscrete([4,3,3])` and shifted the
   remaining action indices accordingly.

2. **Infrequent target network updates.** The target network was updated every
   10,000 steps, causing severe Q-value overestimation in the early training phase.
   We reduced the target update interval to every 500 steps, consistent with
   best practice for environments with dense, step-by-step rewards.

With both fixes applied, DDQN now converges reliably across all four evaluation
seeds (42, 123, 456, 789). The revised Table II reports mean ± 95% CI for DDQN
across all seeds, with results that are clearly distinguishable from the proposed
Double-LSTM method, providing a valid and fair baseline comparison.

We believe keeping a fixed DDQN (rather than substituting an entirely different
system such as KEDA) preserves the scientific contribution of the comparison:
all baselines — Static HPA, DDQN, Single-LSTM, Double-LSTM — are evaluated under
the identical multi-objective reward function and the same deployment conditions,
ensuring a clean ablation of the architectural contributions.

## Changes made to the paper
- Section IV-B: Added description of the two fixes applied to DDQN.
- Table II: DDQN now reports mean ± 95% CI across 4 seeds (previously single-seed, broken).
- Code: `ddqn_agent.py` action space and target update interval corrected.
