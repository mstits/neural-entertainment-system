# Reward-component mix shift — evidence notes

Dimension assigned: does a penalty term (time penalty, death penalty) come to
dominate the positive terms late in a run, silently changing what the policy
optimizes? Does wavefront/PBRS shaping telescope badly late in a run?

All numbers reproducible via the scripts in this directory, run from the repo
root with `.venv/bin/python`.

## What's actually logged

`metrics.jsonl` for all 8 runs carries exactly three `reward_*` keys, no more,
no fewer:

    reward_forward, reward_completion, reward_time_penalty

Verified by scanning every row of every run's metrics.jsonl for any key
containing "reward" or "death" — `extract_reward_mix.py` output confirms the
same 3-key set in 8/8 files.

## Confirmed structural fact #1: death penalty is invisible in metrics.jsonl (code-certain, not correlational)

`death_penalty: -15.0` is a live, non-zero weight in all 8 configs
(`configs/mario_1_1_v{27,28}_seed{0..3}.yaml`), and the Rust reward struct
does add it to the breakdown on every death:

    nes_core/src/rewards.rs:1476   acc.add("death", self.death_penalty);

But the trainer's per-iteration metrics emission reads each env's
*cumulative-since-last-reset* breakdown at the single instant the rollout
ends:

    src/training/trainer.py:8246-8260   reward_breakdown_emit[f"reward_{k}"] += float(v)  (summed over reward_fns, divided by num_envs)

and every death-terminated episode calls `reward_fns[i].reset()` in the SAME
step, before the loop advances (both the curriculum warm-start-reload branch
and the stage-0 inline-restart branch do this):

    src/training/trainer.py:7489    reward_fns[i].reset()   (after curriculum warm-start reload)
    src/training/trainer.py:7661    reward_fns[i].reset()   (after stage-0 inline restart)

`reset()` clears the breakdown dict in Rust:

    nes_core/src/python.rs:1147-1150
        fn reset(&mut self) {
            self.inner.reset();
            self.breakdown.clear();
        }

So the "death" entry that gets added at the exact step of death is wiped
before the end-of-rollout aggregation ever reads it. This is deterministic —
it happens on every death, in every env, in every iteration, in all 8 runs,
because they share this trainer.py code path. `.venv/bin/python
runs/peak_instability/reward_mix/extract_reward_mix.py` confirms zero
occurrences of a death-related key across all 2000 (8 runs x 250 iters) rows.

**Answer to the assigned question, for the death term specifically: not
computable from what was logged.** The instrumentation gap is the emission
site reading a since-last-reset snapshot instead of a true per-iteration sum
(or reading the breakdown BEFORE calling reset() on the same step). Fixing
it means snapshotting `reward_fns[i].breakdown` at the moment `done` fires
and accumulating that snapshot into the iteration total, before the reset
call — not something derivable from the existing logs after the fact.

## Confirmed structural fact #2: wavefront/PBRS is OFF in all 8 runs (clean null, not a lead)

`trainer.py` only activates the wavefront potential when
`reinforce.wavefront_reward.enabled` is true and a `dmap` path is given:

    src/training/trainer.py:5230-5243

None of the 8 configs set `wavefront_reward` at all:

    grep -c wavefront_reward configs/mario_1_1_v27_seed*.yaml configs/mario_1_1_v28_seed*.yaml   -> 0 for all 8

and none of the 8 training logs contain the "WAVEFRONT reward ON" line the
code emits when it activates:

    grep -l WAVEFRONT runs/v27_fresh_recovery/train_seed*.log runs/v28_capacity/train_seed*.log  -> no matches

So the wavefront-telescoping mechanism cannot be operating in this dataset.
This is a clean, confirmed null on that sub-question — reported as such, not
as "unexamined."

## Mix shift among the 3 components that ARE logged

`share_time_penalty = |reward_time_penalty| / (|forward|+|completion|+|time_penalty|)`,
computed per iteration per run (`analyze_reward_mix.py`,
`analyze_reward_mix2.py`; full table in `summary_table.txt`).

- `forward` accounts for 90.6%-91.6% of magnitude at iteration 0, rising to
  98.7%-99.2% at each run's own peak iteration, and 99.4%-99.8% at iteration
  240, in all 8/8 runs.
- `share_time_penalty` never exceeds 0.74% in ANY run at ANY of the 250
  logged iterations (max across all 8 runs: 0.74%, v28/seed0 iter 245; the
  smallest of the 8 per-run maxima is 0.57%).
- No sign flips are possible or observed: `reward_forward` and
  `reward_completion` are 0 in every single one of the 1721 valid rows
  (`check_sign_flips.py`, `sign_flip_check.txt`) and `reward_time_penalty` is
  never positive — the reward struct's own construction (progress deltas
  clamped to >=0, a constant negative per-step term) makes a flip
  impossible by design, not just empirically absent.
- The per-run minimum share_time_penalty does NOT co-locate with that run's
  own peak iteration in any of the 8 runs (`is_min@peak` is False 8/8); share
  stays flat within [0.13%, 0.22%] from peak through peak+40 in all 8 runs.

**Conclusion for the logged components: refuted.** Within what's actually
recorded, no penalty term approaches dominance at any point in any of the 8
runs, before or after peak. Whatever drives the collapse is not visible as a
reward-component mix shift among forward/completion/time_penalty.

## A secondary, hedged observation (lead, not a finding)

`net_logged_reward` (= forward + completion + time_penalty, the same
since-last-reset snapshot, NOT a true per-iteration total — flagged
explicitly as a biased proxy, since it reflects whichever fragment of
whichever episode happens to be live for each of the up-to-60 envs at the
single instant of aggregation, and that fragment's length/starting position
is itself shaped by curriculum warm-starts) has a HIGHER cohort mean in the
post-peak block (peak_iter..249) than the pre-peak block (0..peak_iter-1) in
7 of 8 runs (`cohort_means.py`, `cohort_means_pre_post_peak.txt`) — even
though the honest-eval score has collapsed to ~0.00-0.05 by iteration 240.
The one exception is v28/seed2 (1985.0 pre -> 1424.7 post), which is also
the run with the sharpest `vanilla_ppo_clears` collapse (43.8 -> 7.8 mean
clears/iter) — i.e. the one run where this proxy DOES track the real
decline is the one run where the decline is largest, which is at least
internally consistent rather than contradictory.

This is a hedged lead, not a finding: it does not mean training-time reward
"stays healthy" in the other 7. `vanilla_ppo_clears` (a real, non-snapshot
per-iteration count of completed level-clears, immune to the reset-timing
issue above) declines from pre-peak to post-peak cohort mean in 8/8 runs, by
varying magnitude (mild in several, e.g. v28/seed0: 36.6 -> 31.3; severe in
others, e.g. v28/seed2: 43.8 -> 7.8, v27/seed3: 35.0 -> 16.6). The
net-reward-snapshot staying flat-to-elevated in most runs is best read as an
artifact of curriculum-warm-started fragments still banking large raw
forward-pixel deltas even as genuine full-episode competence (clears)
degrades — not evidence that the reward function itself is fine late in
training. Named confound: fragment start position is not comparable across
iterations once curriculum/warm-start is active, so `reward_forward`
magnitude alone cannot be read as a play-quality signal.

`share_time_penalty`'s cohort mean does tick up post-peak in 6 of 8 runs
(e.g. v28/seed0: 0.0019 -> 0.0026; v27/seed2: 0.0020 -> 0.0025), flat in the
remaining 2 (v27/seed3, v28/seed1, both ~0.0020 -> ~0.0021). So there is a
real, consistent-direction (6-8 of 8) small uptick in the time-penalty
share after peak — but the largest post-peak cohort mean observed anywhere
is 0.0029 (0.29%). The direction is a genuine, repeatable lead; the
magnitude is nowhere near "dominates," which is the question actually
asked.

## An incidental, quantified finding: `reward_completion` goes missing far more often post-peak

`reward_completion` (never `reward_forward` or `reward_time_penalty`) is
entirely ABSENT from the emitted metrics row in 279/2000 rows. This happens
because the emission loop only writes a key when at least one live
(not-yet-reset-this-iteration) env's breakdown contains it — `RewardAccum.add`
only records a signal when its per-step amount is non-zero
(`nes_core/src/rewards.rs:356-364`), and a completion event is far rarer per
in-flight fragment than per-step forward/time-penalty accrual.

Split by each run's own peak iteration (`analyze_completion_missing.py`,
`completion_missing_pre_post_peak.txt`):

    pre-peak:  5/600   = 0.8%  missing reward_completion
    post-peak: 274/1400 = 19.6% missing reward_completion

8/8 runs show a higher missing-rate post-peak than pre-peak (pre-peak misses
are 0 in 5 of the 8 runs outright). Read as a symptom (it occurs strictly
after the honest-eval peak in every run, so temporal ordering rules it out as
a leading cause): late in training, an increasing fraction of iterations
have NO env, out of up to 60, currently mid-episode with any completion event
banked since its last reset — consistent with, but not proof of, the same
behavioral degradation `vanilla_ppo_clears` shows more directly. This is
offered as corroboration for other agents' dimensions (entropy, exploration
collapse), not as a reward-mix finding — it's an absence-of-signal artifact
of the same snapshot-timing issue as the death-penalty gap above, not a
change in weights or in what's being optimized.

## Using `scripts/analyze.py` on this dimension, and a critique

Ran it directly on the reward fields, e.g.:

    .venv/bin/python scripts/analyze.py \
      --metrics checkpoints/mario_1_1_v28_capacity_seed2/metrics.jsonl \
      --split "generation>=120" --top 30

(output saved: `analyze_py_v28seed2_postpeak.txt`, generation 120 is this
run's own peak_iter). It correctly reproduces the entropy story
(`ppo_entropy` effect -2.1, mean 0.80 pre -> 0.15 post) and the clear-rate
collapse (`vanilla_ppo_clears` effect -2.01, 43.8 -> 7.8) at the top of the
ranking, and it does surface `reward_completion` mid-list (effect -1.13) —
so the tool works and is worth using for this dimension.

The concrete critique, visible in that same output: `reward_completion`'s
row reads `n_A/tot = 71/130` — the tool's `split_rows()` deliberately drops
any row missing the field from BOTH cohorts before computing means
(`analyze.py` `split_rows`, "Rows missing `field` entirely are excluded...
a row that never recorded X isn't evidence about X, it's just missing
data"). That's the right call for `clear_rate`-style fields where
"missing" really is unrelated missing data. But for `reward_completion`
specifically, ABSENCE IS THE SIGNAL — as shown above, the field goes
missing in a rising share of post-peak rows (0.8% pre-peak -> 19.6%
post-peak, pooled across all 8 runs) precisely because it means "no env
had a live completion event." Folding those 59 missing post-peak rows
(out of 130) out of the mean, rather than treating the 0.8%-vs-19.6%
missingness-rate gap itself as a divergent field, means the tool's ranked
effect size (-1.13, mid-table, well below entropy and clears) understates
how divergent this field's behavior actually is between cohorts — the
coverage columns (`n_A/tot`, `n_B/tot`) carry that information but it is
not folded into the ranking statistic, so it is easy to skim past. A
general fix: emit a synthetic pseudo-field per column (`<field>_coverage`
= present/total per row's cohort) and rank ITS effect size alongside the
value-based one, so a field whose presence rate — not just its value —
diverges between cohorts surfaces on its own rather than only being
visible to someone who reads the coverage columns closely.

Two smaller, lower-confidence observations against the tool, not verified
as bugs, just noted: (1) the "both cohorts constant at different values ->
+-inf" rule (docstring) means any field that is legitimately 0.0 in both
cohorts but for unrelated reasons (`demo_anchor_coef`, `vanilla_ppo_count_bonus_mean`
in the run above) reports effect `0`, correctly — verified fine, not a bug,
included here only because it was the other edge case named in the
docstring and worth confirming empirically. (2) the tool ranks by
`|effect size|` alone with no significance/sample-size gate beyond the
printed `n/total` columns, so a field measured on very few rows in one
cohort (e.g. a rare debug counter that only exists for 3 iterations) could
in principle rank above a robust, fully-covered field purely from small-n
noise; not observed happening in this run's output, but nothing in the
ranking penalizes it if it did.

## What would make the death-penalty question answerable

Snapshot `dict(reward_fns[i].breakdown)` into a per-iteration ACCUMULATOR the
instant `done` fires (before calling `.reset()`), and sum that accumulator
into the emitted `reward_*` metrics instead of reading the live (post-reset)
breakdown once at rollout end. That change, plus a re-run, would make the
death-penalty share directly measurable per iteration going forward; it
cannot be reconstructed from the 8 runs' existing metrics.jsonl.
