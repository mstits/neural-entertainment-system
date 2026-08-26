# Peak instability: forensics on 8 from-scratch runs (no mechanism established)

**Bottom line up front: the campaign did not identify the cause. It
demoted the prime suspect, killed six candidates outright, and surfaced
one new structural finding that reframes the question — the collapse
looks like TWO phenomena, not one, and the data on disk cannot separate
the top three candidates for the second phase.** This document says so
plainly rather than promoting the least-damaged candidate.

Eighteen analyses ran against 8 completed runs, all reanalysis of data
already on disk plus two small eval-only probes. Scripts and raw output
are under `runs/peak_instability/<topic>/`, one directory per analysis.

## A warning about this document's own evidence base

Only **4 of the 18 analyses carry a recorded adversarial verification
pass. All 4 came back WEAKENED. None came back SURVIVES.** The verified
four are: death-position forensics, GAE/advantage estimation, the decay-
curve probe, and the protocol ablation. The other fourteen are
single-pass and unaudited.

That base rate is the single most important caveat here. A 4-for-4
WEAKENED record on the audited subset means the unaudited fourteen
should be read as provisional — the prior that a given unverified claim
survives contact with a hostile reviewer is, on this campaign's own
record, poor. Where a finding below is unverified, it is marked
`[unverified]`. Rankings weight verified evidence above unverified
evidence, which is why the entropy hypothesis — supported almost
entirely by unverified analyses and contradicted by checks I ran
directly — is not ranked first.

---

## 1. The phenomenon

All 8 from-scratch runs (v27 = 48k params × 4 seeds; v28 = 72k params ×
4 seeds; 250 iterations each) peak early and decay to near-zero. Honest
protocol: cold entrance, greedy, sticky 0.25, jitter ±16, 100 episodes
pooled over eval seeds {0,1}.

| run | peak@iter | % through | honest@peak | honest@final |
|---|---|---|---|---|
| v27 seed0 | 60 | 24% | 0.040 | 0.020 |
| v27 seed1 | 50 | 20% | 0.290 | 0.020 |
| v27 seed2 | 90 | 36% | 0.530 | 0.000 |
| v27 seed3 | 60 | 24% | 0.170 | 0.010 |
| v28 seed0 | 70 | 28% | 0.450 | 0.000 |
| v28 seed1 | 60 | 24% | 0.230 | 0.050 |
| v28 seed2 | 120 | 48% | 0.370 | 0.000 |
| v28 seed3 | 90 | 36% | 0.670 | 0.000 |

Peak iters re-derived from `checkpoints/<run>/winners/best.json`
(`source_iter`), confirmed to match. The banked all-time control on this
level is 0.767 — see §3 for why that bar is softer than it looks.
Preserve-on-peak is the only reason these experiments have numbers;
without it all 8 report ~0.01.

### 1.1 The decay curve is a plateau, not a spike (n=2 runs, verified WEAKENED)

Twenty honest evals on checkpoints already on disk, n=30, eval seed 0.
Full table: `runs/peak_instability/decay_curve/decay_curve_table.csv`.

| iter | v28 s3 honest | v28 s3 train_sr | v28 s0 honest | v28 s0 train_sr |
|---|---|---|---|---|
| 50 | 0.600 | 0.709 | 0.200 | 0.582 |
| 70 | 0.233 | 0.662 | **0.467** (peak) | 0.691 |
| 80 | 0.700 | 0.821 | 0.433 | 0.756 |
| 90 | **0.667** (peak) | 1.000 | 0.733 | 0.795 |
| 100 | 0.533 | 0.921 | — | — |
| 120 | 0.333 | 0.950 | 0.400 | 0.622 |
| 160 | 0.033 | 0.049 | 0.367 | 0.360 |
| 200 | 0.000 | 0.000 | 0.000 | 0.000 |
| 240 | 0.000 | 0.000 | 0.000 | 0.000 |

The selection-noise analysis pre-registered a prediction before this ran:
median `honest(peak±10)/honest(peak)` would land in **0.4–0.9**. Measured
ratios were 1.049, 0.800, 1.000, 0.929 — **median 0.965, above the
predicted band.** The peak is a genuinely broad multi-checkpoint
capability band, not a lucky draw. Winner's curse is refuted as an
explanation of the *honest* peak (see §3).

Caveats, stated because they are load-bearing: n=30 single-seed is a
shape probe with roughly ±9 points of binomial error, not a gate number.
The adversarial pass rated this WEAKENED for stating conclusions (a) and
(b) with more confidence than N=2 supports, and for not cross-checking
against the free `train_sr` column — which I have now done above.

**The v28 seed0 row at iter 90 is a live problem.** It honest-scores
0.733 — higher than the banked peak at iter 70 (0.467) — and it also
carries the highest in-training `success_rate` of any probed checkpoint
in that run (0.795). Both metrics agree that iter 90 is the better
checkpoint, and the winner-selector banked iter 70 anyway. The
non-monotonic dip at iter 80 is probably n=30 noise; the iter-90 miss
probably is not.

### 1.2 The collapse is real capability loss, not a protocol artifact (verified WEAKENED)

Full 2×2×2 grid on v28 seed3, n=30, eval seed 0
(`runs/peak_instability/protocol_ablation/`):

| checkpoint | sticky | select | clear_rate |
|---|---|---|---|
| peak | 0.00 | greedy | **1.000** |
| peak | 0.00 | sampled | 0.833 |
| peak | 0.25 | greedy | 0.667 |
| peak | 0.25 | sampled | 0.667 |
| final | 0.00 | greedy | **0.033** |
| final | 0.00 | sampled | 0.000 |
| final | 0.25 | greedy | 0.000 |
| final | 0.25 | sampled | 0.033 |

**The key cell: final checkpoint, stickiness removed entirely, greedy
argmax — the single most favorable protocol setting available — clears
1/30.** Capability at iter 240 is genuinely gone. The collapse is not a
stickiness-robustness artifact of the honest gate.

Harness validation: the honest-setting cells reproduce the banked
100-episode pooled gate (peak 0.667 vs 0.670; final 0.000 vs 0.000).

The adversarial pass rated this WEAKENED on two grounds worth carrying
forward. First, N=1 run; when the verifier extended it to v27 seed2, a
naive replication at `eval_workers=2` returned 0.167 (5/30) — an
apparent counterexample — which resolved to 0.04 (2/50) only once
`eval_rng`/`eval_workers` were matched to that run's own historical gate
protocol. **A 4× swing in a near-zero clear-rate estimate driven purely
by eval-harness RNG configuration is a real, unflagged confound**, and it
sits exactly where this document's central claims live. Second, under
sticky=0/greedy the setup is near-deterministic: the peak cell has
exactly 1 distinct `max_gx` value across all 30 episodes, so "n=30"
overstates the effective sample size.

### 1.3 The structural finding: this looks like two phases, not one

This is the campaign's most consequential result and it emerged from the
verification pass, not the original analyses. Cross-referencing the
honest decay curve against in-training `success_rate` — a free column in
the same `metrics.jsonl` files — shows the two decouple.

**Phase 1 (v28 seed3, iters 90→120): honest greedy falls by half
(0.667 → 0.333) while in-training `success_rate` stays at ceiling
(1.000 → 0.950).** The policy under its own sampling distribution is
still fully capable. Only the argmax has degraded.

**Phase 2 (iters 120→160): `success_rate` craters 0.950 → 0.049.** Now
capability is gone under any decoding, which §1.2 confirms independently
at iter 240.

v28 seed0 shows a weaker version of the same shape: over iters 90→120,
honest falls 45% (0.733 → 0.400) while `train_sr` falls 22%
(0.795 → 0.622).

Confounds separating these two curves, checked in the configs: sticky is
**matched** at 0.25 in training and eval (`sticky_action_prob: 0.25`,
commented "train under the protocol the gate measures"). Training jitter
uses a 160-frame restart window versus the gate's ±16 — training is the
*wider* perturbation, which cuts against an artifact explanation. The
remaining difference is **greedy versus sampled action selection**, which
is precisely the discriminator proposed in §5.

**N=2 of 8.** Only two runs have dense honest evals. This is the
strongest lead in the campaign and it is not a finding.

---

## 2. Ranked candidate mechanisms

Ranked by support *after* verification, weighting verified evidence above
unverified. No candidate is established.

### Rank 1 — Policy sharpening driving argmax/decoding degradation (Phase 1)

**Evidence for.** Logit magnitude grows without bound and never
saturates: `logit_abs_max` runs 21.6 → 200.1 (v27 seed0), 29.9 → 125.8
(v28 seed3) from peak to iter 240, 8/8 runs. Actor-head weight norm
grows a further 2.03–3.57× *after* the preserved peak; `actor.weight`
reaches 95% of final norm only at iter 220–230 in all 8 runs, while
`critic.weight` plateaus at iter 70–150. Behaviorally, only **29.1–39.4%
of argmax decisions agree** between the peak checkpoint and iter 240 on
a fixed batch of 800 real states (chance ≈16.7% for 6 actions) — the
network is re-deciding most states, not merely sharpening. The dominant
shift is unanimous across 8/8: the run+jump action's argmax share falls
0.463 → 0.225 while bare "right" rises ~0.025 → ~0.12. Given SMB's death
structure, eroding jump commitment is a directly plausible route to
zero. Layered on top: the Phase-1 decoupling in §1.3.
`[unverified except §1.3]`

**Adversarial verdict.** The underlying probe was not audited. §1.3,
which is the load-bearing part, came out of a verification pass.

**Causal or symptomatic.** Candidate cause for Phase 1; explicitly *not*
a candidate for the timing of the turn. Logit and weight-norm growth are
smooth and linear straight through the peak with no inflection in any of
the 8 runs — a background ratchet, not an event. Anything that explains
*when* a run turns over must come from elsewhere.

**Holds across all 8?** The weight/logit/argmax evidence: yes, 8/8. The
Phase-1 decoupling that makes it interesting: 2/2 of the runs with dense
honest data, 2 of 8 overall.

**Falsifiable prediction.** In the Phase-1 window (post-peak iterations
where `train_sr` remains ≥0.85 but honest greedy has fallen ≥40% from
peak), honest **sampled** eval will exceed honest **greedy** eval by a
factor ≥1.5 in ≥6 of 8 runs, and sampled will land within ~15 points of
the peak's own greedy score. Rivals 2, 3 and 4 all predict sampled ≈
greedy, both depressed. **Falsifier: if sampled ≈ greedy (ratio
0.85–1.15) in ≥6 of 8 runs, this candidate is dead and Phase 1 is real
capability loss.**

### Rank 2 — Critic degradation as the Phase-2 driver

**Evidence for.** Whole-run mean `ppo_value_loss` versus honest peak is
**Spearman ρ = −0.786 across the 8 runs** — the strongest non-tautological
cross-run predictor of outcome found anywhere in the campaign. Near-peak
value-loss floor versus honest peak: ρ = −0.69, p = 0.058. v27 seed0
(worst run, 0.040) is a categorical outlier: its value loss never drops
below 27.7 even at its own best moment, where every other run reaches
12.9–19.6. Value loss shows a U-shape in 7/8 runs, minimum at/near peak,
rising through collapse. Structurally, the shared trunk means the value
term is **99.3–99.5% of the combined gradient magnitude** (median, all
2000 iteration-rows) — the same shape as the historical bug commit
`1c7ef1f` fixed, at 10–50 scale rather than 300–660. `[unverified]`

**Adversarial verdict.** Not audited. The analysis was self-critical and
reported its own confound (below), which is a point in its favor.

**Causal or symptomatic.** Unresolved, and the analysis says so. The
decisive complication it found itself: **entropy at the moment of
value-loss rise onset is tightly clustered at 0.27–0.41 (CV ≈14%) across
all 8 runs, while entropy at each run's actual peak spans 0.32–0.97
(CV ≈30%).** Value-loss destabilization fires at nearly the same entropy
level regardless of when or how well a run peaks. That is the signature
of a fast-reacting readout entangled with the entropy trajectory, not an
independently-timed driver. Direction of causation between "policy got
worse so returns got noisier" and "critic failed so policy broke" is not
separable from data on disk. Note also that the 99.3% gradient-dominance
figure is **flat across the entire run** — chronic condition, not a
trigger.

**Holds across all 8?** The gradient-dominance structure: 8/8. The
U-shape: 7/8 (v27 seed0 excepted). The lead/lag timing: 7/8 at a loose
1.5× rise threshold, collapsing to **4/8 at a 2.0× threshold** — a coin
flip once you require an unambiguous blowup.

**Falsifiable prediction.** Measured as *explained variance* rather than
raw Huber loss (which conflates "targets got harder" with "fit got
worse"), critic degradation onset precedes `train_sr` collapse by >10
iterations in ≥6 of 8 runs. Intervention form: freezing the critic or
dropping `value_coef` at iter ~100 delays the Phase-2 crash by >30
iterations without changing Phase 1. Rival 3 predicts no effect from any
critic-side intervention.

### Rank 3 — Entropy collapse as a unitary, threshold-triggered driver

**Demoted from prime suspect. The direct checks contradict it, and I ran
them myself against `metrics.jsonl` rather than relying on the
analyses.**

**Evidence for.** Entropy declines monotonically in 8/8 runs and is by a
wide margin the largest, most consistent field in a hypothesis-free
sweep of every numeric column (median effect 3.70; the next
non-trivial field is less than half that). Terminal near-zero entropy is
universal. The mechanism is coherent — sharpening logits is the same
process as Rank 1.

**Evidence against, measured directly across all 8 runs:**

| run | peak | entropy@peak | iter `train_sr` floors | entropy there | iter entropy sustains <0.10 | lag |
|---|---|---|---|---|---|---|
| v27 s0 | 60 | 0.614 | 142 | 0.137 | 215 | +73 |
| v27 s1 | 50 | 0.972 | 147 | 0.411 | never | — |
| v27 s2 | 90 | 0.528 | 183 | 0.309 | never | — |
| v27 s3 | 60 | 0.756 | 120 | 0.227 | 209 | +89 |
| v28 s0 | 70 | 0.688 | 179 | 0.111 | 194 | +15 |
| v28 s1 | 60 | 0.645 | 159 | 0.114 | 206 | +47 |
| v28 s2 | 120 | 0.316 | 139 | 0.182 | never | — |
| v28 s3 | 90 | 0.449 | 155 | 0.197 | 198 | +43 |

Three independent problems. **(a) No fixed threshold exists.** Entropy at
peak spans 0.316–0.972, a 3.1× range; entropy at the moment behavior
floors spans 0.111–0.411, a 3.7× range. **(b) Deep collapse lags
behavior in every case where it happens at all.** In the 5 runs where
entropy sustains below 0.10, it does so **15 to 89 iterations after**
`train_sr` has already floored; in 3 of 8 runs entropy never sustains
below 0.10 despite total collapse. **(c) A clean counterexample:** v28
seed2 crossed 0.5 at iter 78 and bottomed at 0.264 around iter 108 —
then recorded its all-time-best checkpoint at iter 120, *under* that low
entropy. v28 seed3 is a softer version, crossing 0.5 eight iterations
before its peak.

The prompt's originating observation — "peak sits near where entropy
crosses ~0.3–0.5" — holds loosely for the 0.5 crossing (median lag +11.5
iters) and fails badly for 0.3 (median +41 iters, by which point
performance has already fallen). It was one correlation on one seed and
it does not generalize.

**Adversarial verdict.** The dedicated skeptic analysis was assigned to
refute and concluded "not refuted outright, but three sub-claims needed
to call it a driver rather than a correlate all fail." My direct checks
above agree and go further. Separately, the death-position analysis —
one of the four that *was* audited — set out to test entropy's corollary
that a collapsed policy converges on one deterministic wrong action, and
found the **opposite**: death-position entropy *rises* 8/8 peak→final
(mean +0.126), and the tightest single-location pileups in the entire
dataset belong to *peak* checkpoints (v28 seed1: 51% of failures in one
50-unit bin). That audit was itself rated WEAKENED for a level-geometry
confound — a competent policy's rare failures naturally concentrate at
the level's one hard obstacle — so it does not cleanly refute anything.
But it certainly does not support the entropy story either.

**Causal or symptomatic.** On the evidence above: **lagging correlate for
the deep-collapse phase.** Not dead as a mechanism — it is the same
sharpening process as Rank 1 and remains a plausible *substrate* — but
dead as a threshold-triggered driver.

**Holds across all 8?** The monotone decline: 8/8. Any threshold or
timing claim: no.

**Falsifiable prediction.** `BackwardEntropyGuard` exists in
`trainer.py:320`, is wired to `configs/mario_1_2_backward.yaml` and
`mario_1_3_backward.yaml`, and has **never been applied to any 1-1
config**. Arm it at floor ≈0.30, engaging only after tau reaches 0
(~iter 25). If entropy is held ≥0.30 through iter 240 and honest score
still collapses to ~0, **entropy is refuted as driver, full stop.** If
instead the honest plateau extends past iter 150 in ≥3 of 4 seeds, this
candidate is resurrected as causal. The floor value matters: the
2026-07-20 attempt at `entropy_floor: 0.5` from cold *degraded* a working
policy (10/10 deterministic → 0/20), which is why that knob has been
`0.0` in every 1-1 config since, inherited unexamined into all 8 runs
here.

### Rank 4 — Adam second-moment ratchet on the output-head biases

**Evidence for.** Within the actor and critic heads specifically, the
bias/weight effective-LR ratio grows from ~1× to **2.03–4.04× over
training in 4/4 runs checked**, while trunk layers show nothing
(`fc1.bias/fc1.weight` stays flat at ~1.02×, despite a far more extreme
parameter-count ratio — so this is not a size artifact). Mechanistically
consistent with sharpening: a near-deterministic action distribution
gives the bias term less gradient variance, shrinking `v`, inflating its
effective LR — a self-reinforcing ratchet on the state-independent
baseline logit. Cross-layer imbalance roughly doubles (~25× → ~55–60×)
over every run. `[unverified]`

**Adversarial verdict.** Not audited. Only 4 of 8 runs were checked.

**Causal or symptomatic.** Amplifier at best. The rise is smooth across
the entire run with no step change at peak, so the data cannot order it
against entropy decline; the analysis says so explicitly and names the
intervention needed instead.

**Holds across all 8?** Unknown — 4 of 8 checked, unanimous within those.

**Falsifiable prediction.** Reset optimizer state (`exp_avg`,
`exp_avg_sq` zeroed) at iter ~100 while preserving weights. Under this
candidate, the reset buys another ~40–60 iterations of plateau. Under
Ranks 1 and 3 it does nothing, because the logits and weights are
already sharp and the reset does not touch them. This is a clean
discriminator and the checkpoints carry optimizer state already.

### Rank 5 — Advantage-normalization outliers (weakened; retained only for completeness)

**Evidence for.** Advantage normalization is a single global z-score per
iteration with no running statistics, `symlog_rewards` is dead on this
code path, and PPO's clip bounds the ratio but not the advantage
magnitude. Elevated-magnitude policy-loss events are exclusively a
post-peak phenomenon in 8/8 runs. `[verified WEAKENED]`

**Adversarial verdict. WEAKENED, on three counts.** The cited source
location was stale — the normalization block that actually ran is in
`src/training/ppo_updater.py`, not the `trainer.py` lines given. The
analysis's own supporting tally was backwards: of the 5 explosion events,
3 (not 2) occur at near-total failure. And it never checked
`clip_grad_norm_(..., 1.0)`, which runs before **every** optimizer step
and caps the actual parameter-space damage identically regardless of
loss magnitude — undercutting the "unbounded mechanism" framing.

**Causal or symptomatic.** Symptom. The one full temporal trace (v27
seed3 iter 113, `policy_loss` = 1777) shows the decline already 20+
iterations underway with no acceleration afterward.

**Holds across all 8? No — and this is disqualifying.** Explosion events
appear in only **4 of 8 runs**. The other four collapse just as
completely without a single qualifying event. Whatever this measures, it
is not necessary for the phenomenon.

**Falsifiable prediction.** Runs with zero explosion events should
collapse more slowly. **Already falsified** — v27 seeds 1 and 2 have zero
events and collapse fully.

---

## 3. Mechanisms ruled out

**Dormant-neuron death — closed before this campaign, re-confirmed.**
Exactly **2000 ReDo checks across the 8 runs, every one reporting
`dormant fc1 0/96 fc2 0/32 recycled 0`.** Zero recycles at both widths.
Verified directly from the logs, not taken on report.

**Action-space collapse — clean null, 8/8.** At iter 240 every run still
uses all 6 actions on a fixed real-state batch, with the most-common
action never exceeding 37.8% share (range 0.296–0.378). The classic
"collapses to a constant function" mode does not describe any of these
runs.

**Network freezing — clean null, 8/8.** Weight motion at iter 240 is
still 12.8–18.2% relative L2 change per 10 iterations, cosine similarity
0.984–0.992. The optimizer never settles; it keeps moving at ~60% of its
early-training rate through the entire second half.

**Curriculum non-stationarity — refuted.** The ladder reaches the true
entrance (tau=0) at **iters 22–29 in all 8 runs** and never reopens
(`at_entrance_reopened_after_first: false`, 8/8). **88.4–91.2% of every
run trains on a byte-identical, frozen start-state distribution.** The
collapse happens on a provably stationary task; there is no moving
goalpost left to blame. The honest peak arrives 25–91 iterations *after*
the ladder freezes, so the simplest version of the hypothesis — "peak
occurs when the ladder stops" — is also refuted.

**RND intrinsic decay — clean null, 8/8.** Intrinsic reward crosses 10%
of its initial value at iters 34–46, which is **before** each run's own
peak in 8/8 (median 23 iterations before), then sits at the noise floor
(≤0.003, three orders below start) flat through the entire collapse. Its
magnitude was never large: 4–6% of per-step extrinsic forward reward even
at maximum. The coefficient is a static 0.1 — both `trainer.py` anneal
blocks are gated behind `smb_curriculum`, which is `false` in all 8
configs, so they are provably dead code here. The historical RND
round-trip bug (commit `9865c4c`) predates these runs and no mid-run
resume occurred.

**Reward-mix shift — refuted.** `reward_forward` is 90.6–91.6% of reward
magnitude at iteration 0, rising to 98.7–99.8% by peak and staying there.
`share_time_penalty` never exceeds **0.74%** at any iteration of any run.
Sign flips are structurally impossible (0 occurrences in 1721 valid
rows). No penalty term approaches dominance at any point. Wavefront/PBRS
is off in all 8 (confirmed by config absence and log absence).

**GAE horizon drift as trigger — refuted.** The trace horizon is 16.8
env-steps against episodes of 324–593 steps, so the mismatch is chronic —
but it is **worst at peak** (ratio 0.028–0.043) and *improves* to
0.07–0.09 by the final iteration. It does not drift in the direction that
would explain collapse timing.

**Training-reward specialization — refuted.** Mean reward declines
−51.2% median in training telemetry and −50.2% median in the honest gate:
two disjoint measurement pipelines, matched declines. Binary success
falls −98.3% (training) against a 0.00–0.05 honest floor. The one number
that "stays high" — `best_fitness`/`max_x`, a batch **max** over ~150–220
rollouts — is fully explained by a binomial null model
(`P(≥1 clear) = 1-(1-p)^n`) reproducing observed frequency to a mean
absolute gap of 0.088. No hidden reward channel is needed.

**Winner's curse as an explanation of the honest peak — refuted by
measurement.** The decay curve tested the pre-registered prediction and
the peak neighbourhood held up at median ratio 0.965 (§1.1).

**But winner's curse as an explanation of the *training proxy* is
confirmed, and it is severe.** At the argmax-selected peak, the proxy
metric and the honest metric diverge by a mean of **+0.615**; at the
non-selected iter-240 checkpoint, the same two metrics agree to within a
mean of **−0.005**. Same pipeline both times — so essentially the entire
gap is the act of taking a max over ~24 noisy candidates.
`entrance_trailing_rate` spans only 0.867–1.000 across the 8 runs (range
0.133) while honest spans 0.040–0.670 (range 0.630): a **4.7×
compression**. The selector tells you roughly which runs are better and
is nearly useless for how much better.

**Capacity — established as not the primary lever.** 48k → 72k raised
peaks (best-of-4 0.530 → 0.670) without changing the collapse shape. The
v27/v28 configs differ in exactly two keys (`tile_hidden_dim: 96`,
`tile_trunk_dim: 32`), verified by diff.

### The 0.767 control is not a different regime — it is a softer measurement

Load-bearing, because 0.767 is the PASS/FAIL bar these experiments were
scored against. `configs/mario_1_1_backward.yaml` diffs against
`mario_1_1_v27_seed0.yaml` with **no difference** in architecture, reward,
`lr` (3e-4), `entropy_coef` (0.005), `value_coef` (0.25), `rollout_steps`,
or trainer mode. Same recipe.

And the identical config, run to completion at matched width, collapses:
`runs/mario_1_1_backward_seed0.log` is a full 250-iteration run whose
train-time clear rate sits at **0.000–0.005 for the entire back half
(iters 140–249)**, peaking at ~32% through the budget.

Four process differences produced 0.767, all biasing the same direction:

1. **The run stopped at iter 159**, never reaching 250 — before its own
   decline became visible.
2. **`num_envs: 24` in the run manifest** against `num_envs: 60` declared
   in the config and used by every sibling run. The CLI override is
   unrecoverable.
3. **Hand-picked checkpoint.** The automation selected iter 120
   (`winners/best.json`, `entrance_trailing_rate` = 0.7667). Every
   downstream document cites **iter 140** instead — a different
   checkpoint the automation never chose.
4. **Single eval seed, never cross-checked.** I read every row of
   `checkpoints/mario_1_1_backward/eval.jsonl`: every evaluation of
   `backward_1_1_seed3_iter140.pt` uses `eval_seed: 0` — at n=30 (0.867),
   n=60 (0.767), and n=2/4/5 spot checks. **`eval_seed: 1` has never been
   run on this checkpoint.** v27/v28 were scored on 100 episodes pooled
   over two seeds.

Note the coincidence worth flagging: **0.7667 appears twice** — as the
iter-120 training-time selector metric and as the iter-140 60-episode
single-seed honest eval. Two different numbers for two different
checkpoints that happen to collide.

Project precedent exists and was not carried forward: a 2026-08-16
receipt (commit `32e86eb`) already retracted a circulating "0.76" for a
different 1-1 checkpoint, finding the honest strict rate measured 43/100.

**This does not mean 0.767 is fake.** It means the bar was set by a
shorter run, at an undocumented env count, on a hand-picked checkpoint,
under a narrower eval protocol — and v28 seed3's rigorously-measured
0.670 is already close to it.

---

## 4. Instrumentation gaps

Each is cheap, and each compounds across every future experiment.

**Never computed anywhere in the codebase:**

- **Clip fraction.** `ppo_losses()` builds the `clipped` tensor and
  discards the per-sample indicator.
- **Approximate KL.** Computed only inside the `kl_anchor_loss_coef`
  path, which is off in all 8 configs.
- **Gradient norm.** `clip_grad_norm_` necessarily computes it and the
  return value is discarded at all four call sites
  (`ppo_updater.py:551-553`, `:559-561`, `trainer.py:4688`, `:4781`).
- **Advantage mean and std.** The per-iteration global z-score is
  computed and thrown away. This is the single number that would settle
  whether late-training advantage distributions degenerate.
- **Critic explained variance.** Only raw Huber loss is logged, which
  conflates "targets got harder to fit" with "fit quality got worse" —
  and that conflation is exactly what blocks Rank 2 from being resolved.

**Logged, but wrong or misleading:**

- **`ppo_policy_loss` / `ppo_value_loss` / `ppo_entropy` are
  last-minibatch snapshots, not epoch means.** `_last_policy_t` is
  overwritten inside the innermost loop, so the logged value is 1 sample
  of ~2,400 minibatches per iteration. Every spike analysis in this
  campaign is downstream of this.
- **Death penalty is invisible.** `death_penalty: -15.0` is live and does
  enter the reward breakdown, but every death-terminated episode calls
  `reward_fns[i].reset()` (`trainer.py:7489`, `:7661`) which clears the
  Rust-side breakdown dict (`python.rs:1147-1150`) *before* the
  once-per-iteration aggregation reads it. **Zero death-related keys
  appear across all 2000 rows.** Fix: accumulate before reset.
- **`entrance_trailing_rate` is only logged on a new best.** The
  authoritative value at non-winning checkpoints is never written, which
  is why §1.1's neighbour analysis had to fall back to a lower-bound log
  proxy. Fix: emit `bwd_sched.snapshot()`'s rate at every
  `it % 10 == 0`, not only on `save_winner`.
- **The telemetry trap itself.** The `[backward] iter N: ... trailing
  A/30=R` log line is a **lower bound** — a `bwd_sched.record()`
  force-completion pass runs after it prints and before the winner-save
  block reads the window. Log-derived trailing rates and
  `winners/best.json` disagree by 2–25× (e.g. v27 seed0 iter 60 logs
  0.53, `best.json` records 0.867).

**Not collected at all:**

- **No honest-eval curve.** Only 2 points per run (peak, final) existed
  before this campaign; §1.1 added a 10-point curve for 2 of 8 runs.
  Everything else had to proxy through in-training telemetry.
- **No per-step action traces in eval receipts.** This is why the
  death-position analysis cannot separate "policy repeats one wrong
  action and harness noise scatters the outcome" from "policy behavior is
  genuinely varied."
- **No per-episode value loss split by outcome** (death vs clear), which
  would test whether the critic's rise is driven by returns going
  bimodal as the policy sharpens.
- **`eval_rng`/`eval_workers` are not pinned across receipts.** §1.2's
  verification found a 4× swing in a near-zero clear rate from this alone.
  Fix: pin both in the gate protocol and record them in every receipt.

The first five items are one-line additions. Given the 4-for-4 WEAKENED
record in §0, adding them before the next training run is worth more than
another pass of reanalysis over the same logs.

---

## 5. The single most informative cheap experiment

**Greedy-versus-sampled honest eval across the post-peak window, on
checkpoints already on disk. Zero training compute.**

This is the right experiment because it discriminates Rank 1 from Ranks
2, 3 and 4 in a single measurement, it directly tests the §1.3
two-phase finding that reframes the whole question, and every checkpoint
it needs already exists. `scripts/eval_game.py` supports `action_select`
sampled today — no new machinery.

**Protocol.** For all 8 runs, at 4 checkpoints spanning peak → peak+60
(10-iter grid): honest eval at n=50, eval seed 0, cold entrance, sticky
0.25, jitter ±16 — **once greedy, once sampled**. 64 evals. Pin
`eval_workers` and `eval_rng` and record both in every receipt, per §4.

**Pre-registered decision rule:**

- **If median(sampled/greedy) ≥ 1.5 in ≥6 of 8 runs** — Phase 1 is
  decoding degradation, not capability loss. Rank 1 is promoted to
  established. The immediate consequence is large: part of what this
  project has been calling "collapse" is an artifact of scoring the
  argmax of an over-sharpened policy, and the fix is checkpoint selection
  and decode policy, not training dynamics.
- **If sampled ≈ greedy (0.85–1.15) in ≥6 of 8 runs** — Rank 1 is dead.
  Phase 1 is real capability loss, and the campaign's attention moves to
  Ranks 2 and 4, whose discriminating interventions (critic freeze at
  iter 100; optimizer-moment reset at iter 100) are the next two
  experiments in that order.
- **If the result splits** — report the split. With N=8 a 4/4 outcome is
  a real result about seed heterogeneity, not a failed experiment.

**Second, run concurrently because it costs nothing:** add the five
missing scalars from §4 (`clip_fraction`, `approx_kl`, `grad_norm`,
`adv_mean`/`adv_std`, critic explained variance) to the metrics sink
before the next training run of any kind.

**Third, and separately: re-score the 0.767 control** through the
canonical two-seed 100-episode gate. It is one eval invocation and it
currently sets the PASS/FAIL bar for this entire line of work on a single
eval seed at n=60.

---

## Standing conclusions

1. **No mechanism is established.** Three candidates remain live for
   Phase 2 (critic degradation, the Adam bias ratchet, and sharpening as
   substrate) and the data on disk cannot separate them. The
   discriminating interventions are named in §2 and none of them is
   reanalysis — they all require running something.
2. **Entropy collapse is demoted from prime suspect to lagging
   correlate.** No fixed threshold exists (3.1× spread at peak, 3.7× at
   behavioral floor); deep collapse trails behavioral collapse by 15–89
   iterations wherever it occurs at all; 3 of 8 runs never reach it
   despite collapsing completely; and v28 seed2 recorded its best
   checkpoint *below* the proposed danger zone. It is not dead as a
   substrate — it is dead as a trigger.
3. **The collapse is real.** De-stickified greedy at the final checkpoint
   clears 1/30. This is not a protocol artifact.
4. **The peak is real too.** Neighbours at ±10 iterations score at median
   0.965 of the peak, above the range selection-noise theory predicted.
   The training-proxy number, by contrast, is inflated by +0.615 and
   compressed 4.7×, and should stop being quoted as a capability figure.
5. **The two-phase structure is the most consequential lead and it is
   N=2.** It came from a verification pass, not from the analysis that
   owned the dimension — which is an argument for auditing the other
   fourteen.
6. **Six mechanisms are closed with evidence** (§3): dormancy, action
   collapse, network freezing, curriculum non-stationarity, RND decay,
   reward-mix shift — plus GAE horizon drift and training-reward
   specialization as refuted-as-trigger.
7. **The 0.767 bar should be re-measured before it adjudicates anything
   else.** Same recipe, shorter run, undocumented env count, hand-picked
   checkpoint, single eval seed.
