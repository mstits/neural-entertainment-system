# Phase 3, hazard-masked PPO: FAIL at −100% relative (documented negative)

**Gate (pre-registered):** ≥20% relative improvement in strict honest
clear rate over ≥100 episodes across two seeds, versus an unmasked
control. **Result: control 31/100, masked 0/100, relative −1.0.**
Receipts: `runs/phase3/verdict.json`, `runs/phase3/eval2_*_seed*.json`,
arm configs differing in exactly one key block (asserted at generation).

Per the synthesis's own instruction, the substrate experiment is
TERMINATED. No threshold tuning, level change, or budget extension was
applied to rescue the result.

## The run that had to be voided first

The first masked arm produced control 0.28/0.34 and masked 0.28/0.34 —
byte-identical outcomes. Diffing checkpoints: only the critic differed;
the actor was untouched after 200 iters. Cause: `-inf` veto logits made
the entropy term compute `0 × -inf = NaN`, and the NaN guard silently
skipped every actor update. Additionally `eval_game` never armed the
veto, so the treatment was absent from its own measurement. Both fixed
(`NEG_MASK = -1e9`, veto wrapped after checkpoint load) and the arm was
re-run from the same seed policy. Without the void being caught, the
verdict would have been "0% improvement" — a mechanism killed by a NaN.

## Why the veto fails: the hazard is off-policy

Measured on 6,000 real 1-2 states (`runs/phase3/veto_overlap.txt`):
the escape hatch fires in 13.5% of states; a partial veto is active in
20.4%; and **within veto-active states, the control's greedy action is
blocked 77.4% of the time**. Movement actions are vetoed roughly
uniformly (14–17%) while passive actions stay legal (3–4%).

The labels are micro-forks with random continuations, so the model
learned P(death | action, then random play). Under random play the pit
jump is lethal; under the trained policy it is the winning line. The
veto therefore removes precisely the skilled actions, leaves the policy
passivity, and the predicted death arrives anyway — masked episodes are
SHORTER than control (mean 301 vs 653). The C-index of 0.9170 was a true
measurement of discrimination on the fork distribution; it says nothing
about action quality under the improved policy. Discrimination
transferred; the decision rule did not.

## Standing conclusions

1. Hazard-as-veto on Tier-1 labels is falsified for this class. Any
   revival (hazard as reward shaping, policy-conditioned relabeling,
   veto-with-alternative-ranking) is a NEW experiment.
2. The winner's-curse discipline held: every number here is ≥100
   episodes, two seeds, strict predicate.
3. This is the falsification that justifies a Deep Research round; the
   question travels with measured artifacts, not hunches.
