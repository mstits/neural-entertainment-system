# v26 adjudication — probe PASSES its gate, but detection was never the wall

v26 (responses/20260824T020108Z_v26_gru_learning_failure.md) diagnosed
the recurrent arm's collapse (stateless-BC pathology + frame-stack
gradient competition + inherited hyperparameters + rung-277 hidden-state
distribution shift) and prescribed exactly one follow-up: a supervised
stick-detection probe (v27_PROBE) gating any further recurrent RL spend.

The probe was already built and run the same evening
(scripts/stick_probe.py, receipts runs/gru_ab/stick_probe*.json), in
both the v26 spec form (single frame) and the policy-faithful form
(4-frame stack). 24k steps from the banked 0.76 control under sticky
0.25; divergent-stick base rate 0.034 (sticky repeats of an
already-repeated action diverge nothing); held-out-episode AUC:

| probe | single frame | 4-frame stack |
|---|---|---|
| stateless MLP (+ intent onehot) | 0.77 | 0.83 |
| GRUCell (+ intent onehot)       | 0.82 | 0.87 |

## Gate result

**PASS by v26's own condition** (robust predictive power far above the
random baseline): a minimal GRU internalizes execution noise. The
recurrent line is not killed by v26's kill-condition.

## The strategic override the full matrix forces

v26's PASS branch "authorizes the massive budget to fix the RL
pipeline (sequence-BC, tuning, R2D2 methods)." The matrix says that
budget would chase a mechanism whose absence is not binding:

- The STATELESS probe already detects sticks at 0.83 with the stack —
  and 0.77 from a single frame (the tile features carry velocity
  scalars; intent + velocity largely reveals a stick).
- The feedforward class therefore already HAD AUC-0.83-grade access to
  stick information — and still plateaued at 0.21–0.51 honest.
- Recurrence adds ≈ +0.05 AUC everywhere. Nothing here is the
  night-and-day observability gap v25's POMDP story requires.

Conclusion: **detection is not the sticky wall's binding constraint.**
The v25 mechanism claim survives in the narrow sense (recurrence CAN
carry the signal) and dies in the strategic sense (the signal was
never missing). The expensive recurrent overhaul is NOT authorized on
this evidence.

## Next registered experiment: the recovery assay

The wall's next falsifiable hypothesis is RESPONSE, not detection —
the A/B pre-registration's tertiary metric, now promoted to primary:

From honest-protocol rollouts of the banked control, snapshot the
post-stick state at every divergent stick; tag whether that episode
subsequently died. For a sample of death-preceding stick states, run
the SOLVER from the snapshot (go_explore burst, minutes per state):

- If search finds a clear from ≥ most death-preceding sticks, recovery
  EXISTS and the gap is a learnable-response problem (train on
  post-stick states; solver-as-teacher curriculum fuel).
- If search itself cannot recover, the stick was already fatal when it
  landed — a game-mechanics funnel no policy class can fix, and the
  honest ceiling interpretation (v25 option c, minus the wrapper
  tricks) is the truth of the matter.

Cheap (savestate forking + existing solver), purity-clean (labels from
our own rollouts), and it decides between "train recovery" and "accept
per-level ceilings" with receipts either way.
