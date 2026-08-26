# Peak instability: forensics on 8 from-scratch runs (no mechanism established)

**Bottom line up front: the campaign did not identify the cause. Its
own top-ranked candidate was falsified by an experiment that has since
run. Eight mechanisms are closed with evidence. Three candidates remain
live and the data on disk cannot separate them — and a late finding
shows the phenomenon table's `peak@iter` column is itself wrong in 4 of
5 runs checked, which re-anchors every peak-relative statistic in this
document.** This says so plainly rather than promoting the
least-damaged candidate.

Eighteen analyses ran against 8 completed runs, all reanalysis of data
already on disk plus three eval-only probes. Scripts and raw output are
under `runs/peak_instability/<topic>/`, one directory per analysis.

## A warning about this document's own evidence base

All 18 analyses now carry a recorded adversarial verification pass. I
hold verdicts for 14 of them:

| verdict | count | analyses |
|---|---|---|
| SURVIVES | **2** | train-vs-honest divergence, RND intrinsic decay |
| WEAKENED | **12** | entropy skeptic, value/critic health, curriculum ladder, policy-update trust region, hypothesis-free field sweep, checkpoint autopsy, prior-art review, external literature, death-position forensics, GAE/advantage, decay curve, protocol ablation |
| REFUTED | 0 | — |

**Both SURVIVES verdicts are on analyses that returned a null.** Every
analysis that proposed a positive mechanism came back WEAKENED. That
asymmetry is the single most important fact about this evidence base:
this campaign has been reliable at killing hypotheses and unreliable at
establishing them. Rankings below weight that accordingly.

Where I re-derived a number myself rather than taking it on report, it
is marked `[re-derived]`. New work done for this document is under
`runs/peak_instability/crash_window_ordering/` and
`runs/peak_instability/p1_falsifier/compile_p1.py`.

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
(`source_iter`); all 8 confirmed to match `[re-derived]`. The banked
all-time control is 0.767 — see §3 for why that bar is softer than it
looks. Preserve-on-peak is the only reason these experiments have
numbers; without it all 8 report ~0.01.

### 1.1 CORRECTION: the `peak@iter` column is a proxy artifact, and the true honest maximum is later

**This is the most consequential correction in the document and it
invalidates the anchor used by roughly half the campaign's analyses.**

`winners/best.json`'s `metric_name` is `entrance_trailing_rate` — an
in-training trailing-window statistic, **not** the honest gate
`[re-derived]`. The P1 probe (`runs/peak_instability/p1_falsifier/`, 37
receipts, 5 of 8 runs, n=50, eval seed 0, one internally consistent
harness config so no cross-receipt confound applies) measured honest
greedy on a checkpoint grid spanning peak → peak+60:

| run | banked peak_it | greedy@peak | best probed it | best greedy | gain |
|---|---|---|---|---|---|
| v27 seed0 | 60 | 0.040 | **100** | **0.160** | 4.00× |
| v27 seed1 | 50 | 0.260 | **70** | **0.520** | 2.00× |
| v27 seed2 | 90 | 0.500 | **110** | **0.600** | 1.20× |
| v27 seed3 | 60 | 0.340 | 60 | 0.340 | 1.00× |
| v28 seed0 | 70 | 0.500 | **90** | **0.700** | 1.40× |

**The winner-selected peak is not the honest-greedy maximum in 4 of 5
runs probed.** For v27 seed0 the banked "peak" is the *worst* of the
four checkpoints measured — capability is 4× higher 40 iterations later
than at the iteration this campaign has been calling its peak. v28
seed0's true maximum at iter 90 is confirmed twice independently: 0.700
here (n=50) and 0.733 in the decay curve (n=30), against a banked 0.450
at iter 70.

Consequences, stated plainly:

- The `honest@peak` column **understates** true capability, by up to 4×.
- The `peak@iter` column is **systematically early**, so decay begins
  later and is shorter than the table implies.
- Every analysis that split cohorts at `source_iter` — the field sweep,
  the checkpoint probe, the optimizer forensics, the value-loss timing —
  is anchored at the wrong iteration by 0–40 iters. This does not
  automatically void those results but it does mean none of their
  peak-relative timing claims should be quoted at face value.

The 3 unprobed runs (v27 seed1 is probed; v28 seeds 1, 2, 3 are not)
are the gap. Closing it is the zero-compute prerequisite in §5.

### 1.2 The decay curve is a plateau, not a spike (n=2 runs, verified WEAKENED)

Twenty honest evals on checkpoints already on disk, n=30, eval seed 0.
Full table: `runs/peak_instability/decay_curve/decay_curve_table.csv`.

| iter | v28 s3 honest | v28 s3 train_sr | v28 s0 honest | v28 s0 train_sr |
|---|---|---|---|---|
| 50 | 0.600 | 0.709 | 0.200 | 0.582 |
| 70 | 0.233 | 0.662 | **0.467** (banked) | 0.691 |
| 80 | 0.700 | 0.821 | 0.433 | 0.756 |
| 90 | **0.667** (banked) | 1.000 | **0.733** | 0.795 |
| 100 | 0.533 | 0.921 | — | — |
| 120 | 0.333 | 0.950 | 0.400 | 0.622 |
| 160 | 0.033 | 0.049 | 0.367 | 0.360 |
| 200 | 0.000 | 0.000 | 0.000 | 0.000 |
| 240 | 0.000 | 0.000 | 0.000 | 0.000 |

The selection-noise analysis pre-registered a prediction before this
ran: median `honest(peak±10)/honest(peak)` would land in **0.4–0.9**.
Measured ratios were 1.049, 0.800, 1.000, 0.929 — **median 0.965, above
the predicted band.** The peak is a broad multi-checkpoint capability
band, not a lucky draw. Winner's curse is refuted as an explanation of
the *honest* peak (see §3).

Caveats, load-bearing: n=30 single-seed is a shape probe with roughly
±9 points of binomial error, not a gate number. The adversarial pass
rated this WEAKENED for stating its conclusions with more confidence
than N=2 supports and for not cross-checking the free `train_sr` column
— done above.

### 1.3 The collapse is real capability loss, not a protocol artifact (verified WEAKENED)

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
argmax — the most favorable protocol setting available — clears 1/30.**
Capability at iter 240 is genuinely gone. Whatever §1.1 corrects about
*when* the peak sits, the terminal collapse itself is real.

Harness validation: the honest-setting cells reproduce the banked
100-episode pooled gate (peak 0.667 vs 0.670; final 0.000 vs 0.000).

WEAKENED on two grounds worth carrying. N=1 run; when the verifier
extended to v27 seed2, a naive replication at `eval_workers=2` returned
0.167 (5/30), resolving to 0.04 (2/50) only once `eval_rng`/
`eval_workers` matched that run's historical protocol. **A 4× swing in
a near-zero clear rate driven purely by eval-harness RNG configuration
is a real confound** and it sits exactly where this document's central
claims live — see §1.5. Second, under sticky=0/greedy the setup is
near-deterministic: the peak cell has exactly 1 distinct `max_gx` value
across all 30 episodes, so "n=30" overstates effective sample size.

### 1.4 The two-phase structure — and the falsification of its leading explanation

Cross-referencing the honest decay curve against in-training
`success_rate` shows the two decouple:

**Phase 1 (v28 seed3, iters 90→120): honest greedy falls by half
(0.667 → 0.333) while `success_rate` stays at ceiling (1.000 → 0.950).**
**Phase 2 (iters 120→160): `success_rate` craters 0.950 → 0.049.**

v28 seed0 shows a weaker version: over 90→120, honest falls 45% while
`train_sr` falls 22%.

The leading explanation was decoding degradation — that an
over-sharpened policy's argmax degrades while the sampled policy stays
capable. **The P1 falsifier tested this directly and it failed.**

| cell | sampled/greedy ratio |
|---|---|
| at the peak checkpoint (n=5 runs) | median **1.47** |
| post-peak, +20 to +60 (n=14 cells) | median **1.29** |

Sampled beats greedy nearly everywhere — 16 of 18 complete cells — so
there is a real, persistent argmax penalty of roughly 30–50%. **But it
is a constant offset present at the peak checkpoint too, and it does
not grow as the run collapses; if anything it shrinks.** Only 2 of 14
post-peak cells reach the 1.5× the pre-registered rule required, and
that rule required ≥1.5× in ≥6 of 8 runs. `[re-derived,
`compile_p1.py`]`

(The 6.50× outlier at v27 seed0's peak cell is an artifact of a
2/50 denominator, not signal.)

**Phase 1 is therefore real capability loss, not a decode artifact, and
it remains unexplained.** The remaining measured difference between
`train_sr` and the honest gate is the start-state distribution width —
training restarts draw from a 160-frame window, the gate uses ±16 —
not the decode rule. That is now the open question Phase 1 poses.

### 1.5 Two campaign-wide confounds found late, both previously unflagged

**(a) The v27 and v28 cohorts were measured with different eval harness
settings.** `[re-derived]`

| cohort | `eval_workers` | `eval_rng` |
|---|---|---|
| v27 (all 4 seeds) | **1** | **shared-stream** |
| v28 (all 4 seeds) | **8** | **per-episode** |

Given §1.3's verified finding that these two knobs alone swing a
near-zero clear rate by 4×, **the cross-cohort comparison that grounds
"capacity is not the primary lever" (best-of-4 0.530 → 0.670) spans a
harness change and is confounded.** The within-cohort collapse shape is
unaffected — all 8 collapse regardless — so the conclusion that capacity
does not change the *shape* stands. The claim that 72k *raised peaks* does
not.

**(b) The 8 runs were not run on identical code — 6 distinct git shas.**
`[re-derived]`

| run | sha | | run | sha |
|---|---|---|---|---|
| v27 s0 | `9bbd65f` | | v28 s0 | `a720d81` |
| v27 s1–s3 | `7747da2` | | v28 s1 | `4e9432b` |
| | | | v28 s2 | `789868c` |
| | | | v28 s3 | `01e56f1` |

`src/training/ppo.py`, `trainer.py`, `exploration_controller.py` and
`checkpoint_manager.py` all changed within that span. I checked the one
that matters most: the `ppo.py` diff splits `entropy_reported` from
`entropy_for_loss`, but the scaled path is reached only when
`entropy_weights is not None`, which requires `commitment_options` —
**absent from all 8 configs (`grep -c` = 0 in every one)** `[re-derived]`.
So `entropy_for_loss == entropy_reported` in every run and the change is
inert here. The confound is real, was never checked by the campaign, and
happens to resolve benign for the field this document leans on hardest.
It has not been cleared for the other three files.

---

## 2. Ranked candidate mechanisms

Ranked by support **after** verification. No candidate is established.
The candidate this campaign ranked first has since been falsified and
is listed as such rather than quietly dropped.

### FALSIFIED — Decoding/argmax degradation (was Rank 1)

**Killed by the experiment its own predecessor document proposed.** The
prediction was that sampled eval would exceed greedy by ≥1.5× in the
post-peak window in ≥6 of 8 runs, distinguishing it from every rival
(which predicted sampled ≈ greedy, both depressed). Measured: the
sampled/greedy ratio is a **constant ~1.3–1.5× offset present at the
peak checkpoint too**, median 1.29 post-peak versus 1.47 at peak — flat
to slightly shrinking, in the wrong direction. 2/14 post-peak cells
reach 1.5×. §1.4.

Two supporting planks also failed verification independently. The
checkpoint autopsy's "the network is re-deciding most states" (29–39%
argmax agreement peak→240) was rated WEAKENED once churn was measured at
every consecutive checkpoint pair rather than three sampled points:
**early healthy training re-decides more than post-peak training does.**
`agree(iter10, iter20)` is 0.182 for v28 seed0 and 0.599–0.642 for two
others, against 0.84–0.88 per 10 iters right after peak `[re-derived,
`checkpoint_probe/argmax_drift_output.txt`]`. Argmax churn is a
background property of a small fast-training net at every stage, not a
collapse signature. The same analysis's logit-growth multipliers also
failed to reproduce (claimed 5–10× and 3–5×; actual 9–21.5× and
3.8–10.4×, with its own cited example contradicting its stated range).

What survives from it: logit magnitude and actor-head weight norm do
grow without bound and without inflection at the peak, 8/8. That is real
— it is just a smooth background ratchet, and it does not predict decode
degradation, which is now measured and absent.

### Rank 1 — Critic degradation

**Evidence for.** Whole-run `ppo_value_loss` versus honest peak is the
strongest non-tautological cross-run predictor found anywhere in the
campaign. The verifier attacked the mean-aggregated version for
leave-one-out fragility, and was right — but the attack does not survive
switching to an outlier-robust aggregator `[re-derived,
`value_loss_corr_robustness.py`]`:

| field / aggregator | Spearman ρ | LOO range | LOO drops below n=8 significance |
|---|---|---|---|
| `ppo_value_loss` (mean) | −0.786 | −0.857 … −0.679 | **4/8** |
| `ppo_value_loss` (**median**) | **−0.857** | −0.893 … −0.786 | **0/8** |
| `ppo_policy_loss` (mean) | −0.381 | −0.607 … −0.107 | 8/8 |
| `ppo_entropy` (mean) | +0.667 | +0.500 … +0.786 | 6/8 |

Median aggregation is the correct choice here anyway, because the logged
value is a last-minibatch snapshot (§4) and the mean is outlier-driven:
v27 seed3's `ppo_policy_loss` mean is 7.231 against a median of 0.0227,
from a single 1776.98 spike. Under the median, the value-loss
correlation strengthens to ρ=−0.857 and **survives every leave-one-out**.

Supporting: v27 seed0 (worst run) is a categorical outlier whose value
loss never drops below 27.7 even at its best moment, where every other
run reaches 12.9–19.6. Value loss shows a U-shape in 7/8, minimum near
peak. Structurally, the shared trunk means the value term is 99.3–99.5%
of combined gradient magnitude (median, all 2000 rows).

**Adversarial verdict: WEAKENED.** Its Finding 4 summary ("7 of 8
positive, median +12.5") contradicted its own displayed table (6/8,
median 9.0), and its one highlighted counterexample (v28 seed2, lag −34)
was an artifact of anchoring on the log trailing line instead of the
authoritative peak — re-anchored, that run's lag is +3, erasing the
counterexample. The verifier also showed its headline confound argument
(entropy clusters tightly at value-loss-rise onset, CV≈14%) is explained
by entropy being a near-seed-invariant function of iteration number:
CV is 15.2% at *any* run-blind iteration near 114.

**Causal or symptomatic. Unresolved, and this is the central problem.**
Raw Huber loss conflates "targets got harder to fit" with "fit quality
degraded." A policy that is dying produces noisier, more bimodal returns
purely mechanically, which inflates value loss with no critic pathology
at all. Nothing on disk separates these. The 99.3% gradient dominance is
also **flat across the entire run** — chronic condition, not a trigger.
And ρ=−0.857 was selected from a scan of ~10 fields at N=8 with no
multiple-comparison correction; a Šidák-corrected threshold at 10
comparisons needs |ρ|≈0.88, so it lands just short even at its best.

**Holds across all 8?** Gradient-dominance structure: 8/8. U-shape: 7/8.
Lead/lag timing: 7/8 at a loose 1.5× rise threshold, **4/8 at 2.0×** — a
coin flip once an unambiguous blowup is required.

**Falsifiable prediction.** Measured as **explained variance** rather
than raw Huber loss, critic fit degradation precedes `train_sr` collapse
by >10 iterations in ≥6 of 8 runs. Intervention form: dropping
`value_coef` from 0.25 to ~0.05 at iter 100 delays the Phase-2 crash by
>30 iterations. **Rivals 2 and 3 predict no effect from any critic-side
intervention.** Falsifier: if explained variance is flat or improving
through the crash window, the value-loss rise is target noise and this
candidate is dead.

### Rank 2 — Adam second-moment ratchet on the output-head biases

**Evidence for.** Within the actor and critic heads specifically, the
bias/weight effective-LR ratio grows from ~1× to **2.03–4.04× over
training in 4/4 runs checked**, while trunk layers show nothing
(`fc1.bias/fc1.weight` flat at ~1.02× despite a far more extreme
parameter-count ratio — so not a size artifact). Mechanistically
self-reinforcing: a sharpening action distribution gives the
state-independent baseline logit less gradient variance, shrinking `v`,
inflating its effective LR, sharpening further. Cross-layer imbalance
roughly doubles (~25× → ~55–60×) over every run.

Also from that analysis: the RND predictor's first layer has 99–100% of
`exp_avg_sq` below 1e-10 by iter 40–60, and 1–2% of `fc1.weight`
entries sit at literal zero or subnormal from iter ~20 onward, giving
those entries effective LR ≈ `lr/eps` = 30,000. That is weight-level,
not neuron-level, so it is not in tension with the ReDo null (§3) —
different granularity, and ReDo's activation-based check cannot see it.

**Adversarial verdict: no verdict recorded.** This is the one analysis
whose verification block I do not hold. Treat as unaudited — and note
§0's base rate for unaudited positive claims.

**Causal or symptomatic.** Amplifier at best. The rise is smooth across
the entire run with no step change at peak, so the data cannot order it
against anything; the analysis says so explicitly.

**Holds across all 8?** **Unknown — 4 of 8 checked**, unanimous within
those. This is the weakest coverage of any live candidate.

**Falsifiable prediction.** Reset optimizer state (`exp_avg`,
`exp_avg_sq` zeroed) at iter ~100 while preserving weights. Under this
candidate the reset buys ≥40 iterations of additional plateau. **Under
Ranks 1 and 3 it does nothing**, because weights and logits are already
sharp and the reset does not touch them. Checkpoints already carry
optimizer state, so no re-derivation is needed. This is the cleanest
discriminator in the document.

### Rank 3 — Sharpening as substrate (entropy as its readout, not a trigger)

**Heavily demoted. Entropy collapse is dead as a threshold-triggered
driver; I ran the decisive test directly.**

**Evidence for.** Entropy declines monotonically in 8/8 and is by a wide
margin the largest, most consistent field in a hypothesis-free sweep of
every numeric column (median effect 3.70; next non-trivial field less
than half). Terminal near-zero entropy is universal. Logit and weight
norms grow without bound, 8/8 — the same underlying process.

**Evidence against — the clock test.** Entropy declines near-monotonically
over all 250 iters, so it correlates with anything that happens late.
The discriminating question is whether its decline **rate** changes at
the behavioral crash. For each run I took the crash window (smoothed
`success_rate` falling from 80% to 15% of its post-peak local max) and
an equal-length control window immediately before it
`[re-derived, `entropy_clock_test.py`]`:

| run | pre-crash window | crash window | pre slope | crash slope | accel |
|---|---|---|---|---|---|
| v27 s0 | 11–69 | 69–127 | −0.01805 | −0.00672 | 0.37× |
| v27 s1 | 46–96 | 96–146 | −0.01100 | −0.00227 | 0.21× |
| v27 s2 | 162–171 | 171–180 | −0.00460 | **+0.00170** | −0.37× |
| v27 s3 | 72–96 | 96–120 | −0.00901 | −0.00390 | 0.43× |
| v28 s0 | 48–113 | 113–178 | −0.00855 | −0.00383 | 0.45× |
| v28 s1 | 79–111 | 111–143 | −0.00588 | −0.00384 | 0.65× |
| v28 s2 | 108–124 | 124–140 | **+0.00401** | −0.01150 | −2.87× |
| v28 s3 | 96–120 | 120–144 | −0.00621 | −0.00481 | 0.77× |

**Entropy's decline decelerates at the crash in 8 of 8 runs (median
0.40×). Zero runs accelerate.** Entropy was falling roughly 2.5× *faster*
during the healthy, improving phase than during the collapse. A driver
should accelerate at or before the event it drives; this does the
opposite, unanimously.

**A direct counterexample.** v28 seed2's entropy roughly doubles from
~0.075 (iters 210–222) to a sustained 0.15–0.26 plateau (iters 223–249)
while `success_rate` stays pinned at 0.000–0.022 throughout
`[re-derived]`. **Entropy recovered and performance did not.** A second:
v28 seed2 crossed 0.5 at iter 78, bottomed at 0.264 near iter 108, then
recorded its best banked checkpoint at iter 120 — *under* that low
entropy.

The originating observation — "peak sits near where entropy crosses
0.3–0.5" — was one correlation on one seed. Across 8 runs entropy at
peak spans 0.316–0.972 (3.1×) and entropy where behavior floors spans
0.111–0.411 (3.7×). **No fixed threshold exists.**

**Adversarial verdict: WEAKENED** (the dedicated skeptic, assigned to
refute, concluded "not refuted outright, but three sub-claims needed to
call it a driver rather than a correlate all fail"). Its arithmetic
reproduced exactly; its causal framing did not survive. Two other
verifiers independently found entropy flat-or-rising across sharp crash
windows.

**Causal or symptomatic. Lagging readout of the sharpening process, not
a trigger.** Not dead as a *substrate* — unbounded logit growth is real,
8/8 — but dead as the thing that decides when a run turns over.

**Holds across all 8?** Monotone decline: 8/8. Deceleration at crash:
8/8. Any threshold or trigger claim: no.

**Falsifiable prediction.** `BackwardEntropyGuard` exists in
`trainer.py`, is wired to `mario_1_2_backward.yaml` and
`mario_1_3_backward.yaml`, and — **correction to the prior-art analysis,
which the verifier caught** — has *not* been "never tried on 1-1":
`configs/mario_generalist_w1.yaml` sets `entropy_floor: 0.3` and trains
1-1, and in both executed runs the 1-1 cold probe sat at 0.00 for
essentially the entire 500–900 iterations. That is a different failure
shape (never peaked at all, so nothing to lose) and does not settle the
question, but the clean "untested knob" framing was wrong.

Armed at floor ≈0.30 after tau reaches 0: if entropy is held ≥0.30
through iter 240 and honest score still collapses, **entropy is refuted
as driver, full stop.** Given the 8/8 deceleration result above, that is
the expected outcome, which is precisely why this is no longer the
experiment worth running first.

### Rank 4 — Advantage-normalization outliers (retained only for completeness)

**Evidence for.** Advantage normalization is a single global z-score per
iteration with no running statistics; PPO's clip bounds the ratio but
not the advantage magnitude. Elevated policy-loss events are exclusively
post-peak, 8/8. `[verified WEAKENED]`

**Adversarial verdict: WEAKENED on three counts.** The cited source
location was stale — the block that actually ran is in
`ppo_updater.py`, not the `trainer.py` lines given. Its own tally was
backwards (3 of 5 explosion events at near-total failure, not 2). And it
never checked `clip_grad_norm_(..., 1.0)`, which runs before every
optimizer step and caps parameter-space damage identically regardless of
loss magnitude — undercutting the "unbounded mechanism" framing.

**Holds across all 8? No — and this is disqualifying.** Explosion events
appear in only **4 of 8 runs**. The other four collapse just as
completely without a single qualifying event.

**Falsifiable prediction: already falsified.** Runs with zero events
should collapse more slowly; v27 seeds 1 and 2 have zero events and
collapse fully.

---

## 3. Mechanisms ruled out

**Dormant-neuron death — closed before this campaign, re-confirmed.**
Exactly **2000 ReDo checks across the 8 runs, every one reporting
`dormant fc1 0/96 fc2 0/32 recycled 0`.** Zero recycles at both widths.
Note the scope correction from the prior-art review: this closes
Sokar-style dormant-unit capacity loss. It does **not** close
Nikishin-style primacy bias, whose paper diagnoses via learning-curve
behavior under manipulated data order, never mentions dormant units, and
studies only replay-buffer off-policy algorithms — so it has no direct
analogue in a buffer-free on-policy learner anyway. The registration
doc's phrasing ("cannot be attributed to primacy bias *because* ReDo…")
is not a valid inference and should be corrected if reused.

**Action-space collapse — clean null, 8/8.** At iter 240 every run still
uses all 6 actions on a fixed real-state batch, most-common action never
exceeding 37.8% (range 0.296–0.378).

**Network freezing — clean null, 8/8.** Weight motion at iter 240 is
still 12.8–18.2% relative L2 per 10 iterations, cosine 0.984–0.992. The
optimizer never settles.

**Curriculum non-stationarity — refuted.** The ladder reaches the true
entrance (tau=0) at **iters 22–29 in all 8 runs** and never reopens
(`at_entrance_reopened_after_first: false`, 8/8). **88.4–91.2% of every
run trains on a byte-identical, frozen start-state distribution.** The
collapse happens on a provably stationary task. WEAKENED only on
secondary arithmetic (a stated "+9.4/iter" truncation rise recomputes to
8.22) and on two correlational side-leads vulnerable to an
extreme-value confound; the central null is untouched.

**RND intrinsic decay — clean null, 8/8. `[SURVIVES]`** Intrinsic reward
crosses 10% of initial at iters 34–46, **before** each run's own peak in
8/8 (median 23 iterations before), then sits at the noise floor flat
through the entire collapse. Magnitude never large: 4–6% of per-step
extrinsic forward reward at maximum. Coefficient is a static 0.1 — both
`trainer.py` anneal blocks are gated behind `smb_curriculum`, `false` in
all 8 configs. The verifier additionally confirmed
`vanilla_ppo_count_bonus_mean` is exactly 0.0 across all 250 iterations
of all 8 runs, closing the one back door the analysis had left open.

**Reward-mix shift — refuted.** `reward_forward` is 90.6–91.6% of reward
magnitude at iteration 0, rising to 98.7–99.8% by peak.
`share_time_penalty` never exceeds **0.74%** at any iteration of any run.
Zero sign flips in 1721 valid rows.

**GAE horizon drift as trigger — refuted.** Trace horizon is 16.8
env-steps against episodes of 324–593 steps, so the mismatch is chronic
— but it is **worst at peak** (0.028–0.043) and *improves* to 0.07–0.09
by the final iteration. It does not drift in the direction that would
explain collapse timing.

**Training-reward specialization — refuted. `[SURVIVES]`** Mean reward
declines −51.2% median in training telemetry against −50.2% median in
the honest gate: two disjoint pipelines, matched declines, verified at
the individual-run level for all 8 with no run showing the
"train stays high while honest craters" signature. The one number that
"stays high" — `best_fitness`/`max_x`, a batch **max** over ~150–220
rollouts — is fully explained by a binomial null
(`P(≥1 clear) = 1-(1-p)^n`) reproducing observed frequency to a mean
absolute gap of 0.088. No hidden reward channel is needed.

**Feature-rank collapse (Moalla et al. 2024) — refuted for this
architecture, 8/8.** Effective rank of the shared trunk **rises** through
collapse; it does not decline. Peak-iter rank is below final-iter rank in
every run. WEAKENED because rank is essentially flat through the entire
peak-relevant window and only jumps in the same 10-iter bin as the
`success_rate` crash — so the whole-trajectory correlation framing
overstates a two-regime step process — but the direction is unambiguous
and the mechanism does not transfer here.

**Winner's curse as an explanation of the honest peak — refuted by
measurement** (§1.2, median neighbour ratio 0.965).

**But winner's curse on the *training proxy* is confirmed, and severe.**
At the argmax-selected peak, proxy and honest diverge by a mean of
**+0.615**; at the non-selected iter-240 checkpoint the same two metrics
agree to within **−0.005**. `entrance_trailing_rate` spans 0.867–1.000
across the 8 runs (range 0.133) while honest spans 0.040–0.670 (range
0.630): a **4.7× compression**. This is the same defect §1.1 shows
mis-selecting the peak checkpoint outright.

**Capacity — not the primary lever, with a caveat now attached.** All 8
collapse regardless of width; the v27/v28 configs differ in exactly two
keys (`tile_hidden_dim`, `tile_trunk_dim`). But per §1.5(a) the
cross-cohort *peak-height* comparison spans an eval-harness change and
should not be quoted until re-measured.

### The 0.767 control is not a different regime — it is a softer measurement

Load-bearing, because 0.767 is the PASS/FAIL bar these experiments were
scored against. `configs/mario_1_1_backward.yaml` diffs against
`mario_1_1_v27_seed0.yaml` with **no difference** in architecture,
reward, `lr` (3e-4), `entropy_coef` (0.005), `value_coef` (0.25),
`rollout_steps`, or trainer mode. Same recipe. And the identical config
run to completion at matched width collapses:
`runs/mario_1_1_backward_seed0.log` is a full 250-iteration run whose
train-time clear rate sits at **0.000–0.005 for the entire back half.**

Four process differences produced 0.767, all biasing the same direction:

1. **The run stopped at iter 159**, before its own decline was visible.
2. **`num_envs: 24` in the run manifest** against `num_envs: 60` in the
   config and every sibling run. The CLI override is unrecoverable.
3. **Hand-picked checkpoint.** The automation selected iter 120
   (`entrance_trailing_rate` = 0.7667); every downstream document cites
   **iter 140**, a checkpoint the automation never chose.
4. **Single eval seed.** Every evaluation of
   `backward_1_1_seed3_iter140.pt` uses `eval_seed: 0` — at n=30 (0.867),
   n=60 (0.767), and n=2/4/5 spot checks. **`eval_seed: 1` has never been
   run on this checkpoint.** v27/v28 were scored on 100 episodes pooled
   over two seeds.

Note the coincidence: **0.7667 appears twice** — as the iter-120
training-time selector metric and as the iter-140 60-episode single-seed
honest eval. Two different numbers for two different checkpoints that
collide.

Precedent exists and was not carried forward: a 2026-08-16 receipt
(commit `32e86eb`) already retracted a circulating "0.76" for a different
1-1 checkpoint, finding the honest strict rate measured 43/100.

**This does not mean 0.767 is fake.** It means the bar was set by a
shorter run, at an undocumented env count, on a hand-picked checkpoint,
under a narrower protocol — and per §1.1, v27/v28's own honest maxima
are higher than their banked peaks, so the gap to 0.767 is smaller than
the phenomenon table implies in both directions at once.

---

## 4. Instrumentation gaps

Each is cheap, and each compounds across every future experiment.

**Never computed anywhere in the codebase:**

- **Clip fraction.** `ppo_losses()` builds the `clipped` tensor and
  discards the per-sample indicator.
- **Approximate KL.** Computed only inside the `kl_anchor_loss_coef`
  path, off in all 8 configs.
- **Gradient norm.** `clip_grad_norm_` necessarily computes it; the
  return value is discarded at all four call sites
  (`ppo_updater.py:551-553`, `:559-561`, `trainer.py:4688`, `:4781`).
- **Advantage mean and std.** The per-iteration global z-score is
  computed and thrown away.
- **Critic explained variance.** Only raw Huber loss is logged, which
  conflates "targets got harder to fit" with "fit quality got worse" —
  **and that conflation is exactly what blocks Rank 1 from resolving.**
  This is the single highest-value missing scalar in the codebase.

**Logged, but wrong or misleading:**

- **`ppo_policy_loss` / `ppo_value_loss` / `ppo_entropy` are
  last-minibatch snapshots, not epoch means.** `_last_policy_t` is
  overwritten inside the innermost loop, so the logged value is 1 sample
  of ~2,400 minibatches per iteration. Every spike analysis in this
  campaign is downstream of this, and it is why median aggregation beats
  mean aggregation in §2 Rank 1.
- **Death penalty is invisible.** `death_penalty: -15.0` is live and
  enters the reward breakdown, but every death-terminated episode calls
  `reward_fns[i].reset()` (`trainer.py:7489`, `:7661`), clearing the
  Rust-side breakdown dict (`python.rs:1147-1150`) *before* the
  once-per-iteration aggregation reads it. **Zero death-related keys
  appear across all 2000 rows.** Fix: accumulate before reset.
- **`entrance_trailing_rate` is only logged on a new best**, so the
  authoritative value at non-winning checkpoints is never written. Fix:
  emit `bwd_sched.snapshot()`'s rate at every `it % 10 == 0`.
- **The telemetry trap.** The `[backward] iter N: ... trailing A/30=R`
  log line is a **lower bound** — a `bwd_sched.record()` force-completion
  pass runs after it prints and before the winner-save block reads the
  window. Log-derived rates and `winners/best.json` disagree by 2–25×
  (v27 seed0 iter 60 logs 0.53; `best.json` records 0.867). At least two
  analyses anchored on the wrong one and had findings reversed by it.
- **`winners/best.json` records a proxy but is universally read as the
  peak.** Per §1.1 it mis-selects the honest maximum in 4 of 5 runs
  checked. Fix: either select on a periodic honest eval, or rename the
  field so it stops being read as a capability number.

**Not collected at all:**

- **No honest-eval curve.** Only 2 points per run existed before this
  campaign; §1.2 added 10 points for 2 runs and §1.4 added 4 points for
  5 runs. Everything else proxies through in-training telemetry.
- **No per-step action traces in eval receipts**, which is why the
  death-position analysis cannot separate "policy repeats one wrong
  action and harness noise scatters the outcome" from genuinely varied
  behavior.
- **No per-episode value loss split by outcome** (death vs clear), which
  would test whether the critic's rise is driven by returns going
  bimodal as the policy sharpens — the exact ambiguity blocking Rank 1.
- **`eval_rng`/`eval_workers` are not pinned across receipts.** §1.3
  found a 4× swing from this alone and §1.5(a) shows the v27 and v28
  cohorts were measured with different values. Fix: pin both, record
  both in every receipt, and re-measure anything compared across cohorts.
- **Git sha is recorded per run but never checked for uniformity.** §1.5(b):
  6 distinct shas across 8 nominally identical runs. Fix: assert sha
  equality across arms at experiment registration.

The first five scalars are one-line additions. Given §0's 12-WEAKENED /
2-SURVIVES record, adding them before the next training run is worth
more than another pass of reanalysis over the same logs.

---

## 5. The single most informative cheap experiment

**A three-arm resume from a common iter-100 checkpoint.** This is the
right experiment because each live candidate makes a *different*
prediction about a *different* arm, so one run discriminates all three —
and because the alternative (more reanalysis) has now been tried
eighteen times and produced two SURVIVES verdicts, both of them nulls.

### Mandatory zero-compute prerequisite

**Do not run the discriminator until the honest checkpoint ladder is
complete for all 8 runs.** §1.1 shows the banked peak is not the honest
maximum in 4 of 5 runs probed, by up to 4×. Every arm below is scored by
"does the plateau extend," which requires knowing where the plateau
actually is. Running a discriminator against a peak that is wrong by 40
iterations wastes the compute.

Cost: ~24 checkpoints × 8 runs, honest protocol, n=100 pooled over eval
seeds {0,1}, **with `eval_workers` and `eval_rng` pinned and recorded**
(§1.5a). Zero training compute; `scripts/eval_game.py` does this today.
Add the five §4 scalars to the metrics sink in the same pass.

### The discriminator

Resume from each run's iter-100 checkpoint (optimizer state included,
already on disk), train to 240, honest-eval every 10 iters:

| arm | intervention | Rank 1 predicts | Rank 2 predicts | Rank 3 predicts |
|---|---|---|---|---|
| **A** control | plain resume | collapse | collapse | collapse |
| **B** optimizer reset | zero `exp_avg`, `exp_avg_sq`; weights untouched | no effect | **plateau extends ≥40 iters** | no effect |
| **C** critic relief | `value_coef` 0.25 → 0.05 | **crash delayed >30 iters** | no effect | no effect |
| **D** norm control | rescale actor/critic head norms to their iter-100 values each iter | no effect | no effect | **plateau extends** |

Arm D is optional if compute is tight; Ranks 1 and 2 are the two with
the most support and B/C alone separate them. Two seeds per arm.

**Pre-registered decision rule.** An arm "wins" if its honest plateau
(iterations with honest ≥ 0.7 × that run's true maximum, per the
prerequisite ladder) extends ≥30 iterations beyond arm A in **both**
seeds. If exactly one arm wins, that candidate is promoted to
established. If none wins, all three are dead as interventions and the
mechanism is something none of the 18 analyses proposed — which, given
§0's record, is a live possibility that should be stated in advance
rather than explained away afterward. If two win, report the pair; they
are not mutually exclusive, since Rank 2 is explicitly an amplifier.

**Why not the entropy guard.** It was the prior document's headline
recommendation. §2 Rank 3's 8/8 deceleration result plus the v28 seed2
counterexample (entropy doubled, performance did not recover) make its
outcome near-certain in advance, and the prior-art correction shows a
1-1 config with `entropy_floor: 0.3` already exists and already failed
differently. Running it now would buy a confirmation, not a
discrimination.

**Third, separately: re-score the 0.767 control** through the canonical
two-seed 100-episode gate with pinned harness settings. It is one eval
invocation and it currently sets the PASS/FAIL bar for this entire line
of work on a single eval seed at n=60.

---

## Standing conclusions

1. **No mechanism is established.** Three candidates remain live —
   critic degradation, the Adam bias ratchet, and sharpening as
   substrate — and **the data on disk cannot separate them.** What would
   separate them is §5: critic explained variance (zero compute,
   resolves Rank 1's central ambiguity) and the B/C/D resume arms, each
   of which has a unique positive prediction. None of this is
   reanalysis.
2. **The campaign's top-ranked candidate was falsified.** Decoding/argmax
   degradation predicted a growing sampled/greedy gap post-peak;
   measured, the gap is a constant ~1.3–1.5× offset present at the peak
   checkpoint too, flat-to-shrinking as runs collapse. Phase 1 is real
   capability loss and is now unexplained.
3. **Entropy collapse is demoted from prime suspect to lagging readout.**
   Its decline **decelerates** at the behavioral crash in 8/8 runs
   (median 0.40×, zero accelerate); no fixed threshold exists (3.1×
   spread at peak, 3.7× at behavioral floor); and v28 seed2's entropy
   doubled over the last 27 iterations while performance stayed at zero.
   It is not dead as a substrate. It is dead as a trigger.
4. **The `peak@iter` column is wrong in 4 of 5 runs checked**, by up to
   4× in honest score and up to 40 iterations in position. This is a
   selector defect, not a measurement defect, and it re-anchors every
   peak-relative claim in the campaign. Fixing it is the prerequisite to
   the next experiment, not a follow-up to it.
5. **The collapse is real.** De-stickified greedy at the final checkpoint
   clears 1/30. Terminal capability loss is not a protocol artifact,
   whatever §1.1 corrects about the peak.
6. **Ten mechanisms are closed with evidence** (§3): dormancy, action
   collapse, network freezing, curriculum non-stationarity, RND decay,
   reward-mix shift, GAE horizon drift, training-reward specialization,
   feature-rank collapse, and winner's curse on the honest peak.
7. **Two campaign-wide confounds were found late and were never checked
   by the campaign itself**: the v27 and v28 cohorts were measured under
   different eval-harness settings, and the 8 runs span 6 git shas. The
   first invalidates the cross-cohort peak-height comparison. The second
   is benign for `ppo.py` (verified inert) and uncleared elsewhere.
8. **The 0.767 bar should be re-measured before it adjudicates anything
   else.** Same recipe, shorter run, undocumented env count, hand-picked
   checkpoint, single eval seed.
