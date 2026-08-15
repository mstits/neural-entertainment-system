# Eval RNG regimes forensics — why the campaign probes read 14.4% and the definitive eval read 2.0%

Date: 2026-08-15. Method: code reading + analysis of already-recorded artifacts only
(no rollouts, no evals, no training). Statistics computed by
`scripts/eval_regime_stats.py` (tested: `tests/test_eval_regime_stats.py`, 15 passed).

**Headline finding up front: the 14.4%-vs-2.0% gap is NOT an RNG-regime effect.
The two numbers are different *clear predicates* scored on statistically
indistinguishable trajectory distributions.** Held to a common predicate, the two
regimes agree (Fisher p = 0.60 / 0.81 / 1.00 depending on the slice); held to the
banked cross-predicate comparison, the gap is significant (p = 0.002) — because it
compares "advanced the level chain past 1-2" against "flagpole/castle success latch".
Details and receipts below.

---

## 1. The two RNG regimes, mechanically

All references are to `scripts/eval_game.py` at its 2026-08-15 working-tree state
(1,285 lines; the regimes are declared at `eval_game.py:99-130`).

### 1.1 Which draws exist at all

Three and only three consumers of randomness exist in the episode loop
(`run_consumes_randomness`, `eval_game.py:133-148`):

1. **Start-jitter offset** (Machado no-op starts): one
   `randint(0, start_jitter + 1)` per episode from the **numpy** stream —
   serial at `eval_game.py:583-585`, parallel at `eval_game.py:429-432`.
   Gated on `start_jitter > 0`.
2. **Sticky coin flip**: one `rng.random()` from the **numpy** stream at *every*
   loop step with `step > 0` when `sticky_prob > 0` — serial at
   `eval_game.py:625` (`sticky_prob > 0.0 and step > 0 and _sticky_rng.random() < sticky_prob`;
   the first two conjuncts are constant per run, so the draw is consumed every
   step ≥ 1 whether or not the repeat fires), parallel identically at
   `eval_game.py:451-455`.
3. **Sampled-action draw**: one `torch.multinomial` per step from the **torch**
   generator, only in `--action-select sampled` (`select_action`,
   `eval_game.py:165-183`; greedy is pure argmax at `eval_game.py:178-179` and
   never touches a generator).

Numpy draws (jitter + sticky) and torch draws (sampled actions) come from two
*separate* streams in every regime; a greedy run never advances the torch stream.

### 1.2 shared-stream (default; the historical / "B-series canonical" regime)

Serial executor only (`_run_episodes_serial`):

- ONE `np.random.RandomState(eval_seed)` built once before the episode loop
  (`eval_game.py:557`) and ONE `torch.Generator` seeded `manual_seed(eval_seed)`
  (`eval_game.py:563-564`). Both are threaded through **all** episodes in order.
- Draw accounting per episode *i*: 1 jitter draw + (L_i − 1) sticky draws, where
  L_i is the number of loop iterations episode *i* ran. Episode *i*'s position in
  the stream is therefore `Σ_{j<i} (1 + L_j − 1)` — **data-dependent**. Episode
  *i* is not a function of `(eval_seed, i)`; it depends on how long every earlier
  episode ran (documented in-file at `eval_game.py:104-110`).
- **Statistical independence**: episodes still consume *disjoint, sequential
  segments* of one i.i.d. uniform stream, and each episode's length is a stopping
  time measurable w.r.t. the draws consumed so far. For an ideal PRNG this makes
  the episodes' randomness mutually independent *in distribution* — shared-stream
  costs reproducibility-per-episode-index, not independence. (This theoretical
  point is confirmed empirically in §2.4: no streak structure.)
- Reproducibility: bit-exact replay of the whole ordered run from `eval_seed`;
  no per-episode replay.

### 1.3 per-episode

- Seed derivation (`episode_rng_seeds`, `eval_game.py:151-162`):
  `np.random.SeedSequence([eval_seed, episode_index]).generate_state(2)` yields
  two decorrelated uint32s → `(numpy_seed, torch_seed)`. Episode *i* is a pure
  function of `(eval_seed, i)`.
- Serial: both streams re-seeded at the top of each episode
  (`eval_game.py:567-570`). Parallel: each lane builds its own
  `RandomState(np_seed)` + `torch.Generator(torch_seed)` at lane creation
  (`eval_game.py:418-421`).
- Episodes are independent by construction *and* individually replayable; the
  result is invariant to how episodes are distributed over workers
  (`eval_game.py:111-119`).
- NOT byte-comparable to shared-stream receipts: same protocol distribution,
  different draws (`eval_game.py:118-119, 1199-1202`).

### 1.4 Executor selection and the guard

- The **effective lane count alone** picks the executor (`eval_game.py:754-760`,
  clamp at `eval_game.py:931-936`, dispatch at `eval_game.py:1032-1035`). Both
  executors honor both RNG modes.
- **A stochastic shared-stream run cannot be parallel**: `eval_one_game` raises
  (`eval_game.py:742-753`) and the CLI pre-checks the same predicate
  (`eval_game.py:1235-1258`). So *every* stochastic parallel receipt is
  per-episode by construction — including all campaign probes — and every
  shared-stream stochastic receipt is serial. The 14.4-vs-2.0 comparison is
  therefore per-episode/parallel vs shared-stream/serial *by necessity*, and the
  RNG mode is confounded with the executor in any naive reading.

### 1.5 Parallel-path differences beyond RNG (audit)

Checked line-by-line against the serial loop:

- **Wave scheduling**: episodes are handed out in blocks of `lanes`; each wave
  starts with `pool.reset_all()` plus the same per-worker `load_worker_state`
  warm start the serial loop performs (`eval_game.py:410-417` vs `571-577`).
  Nothing carries across waves; finished lanes are parked via `set_worker_done`
  (`eval_game.py:405-408, 436-437`).
- **Boundary no-ops**: serial burns 1 mandatory post-reset no-op step + the
  jitter no-ops in a pre-loop (`eval_game.py:579-585`); a lane counts
  `noops_left = 1 + jitter` and interleaves them (action 0) with other lanes
  (`eval_game.py:429-433, 442-443, 465-477`). Same count, same order, same draw.
- **Sticky boundary reset**: identical guards — no sticky at step 0
  (`step > 0` serial :625, `lane.step > 0` parallel :453), `prev_action`
  initialized to 0 in both (`eval_game.py:608` vs `:327`). The GRU hidden state
  is reset per episode in both (`eval_game.py:589` vs `:468-469`).
- **Per-lane state**: each lane owns its policy wrapper/stacker, reward fn,
  tracker, and RNG pair (`eval_game.py:422-425`); serial builds one of each and
  resets per episode (`eval_game.py:555-556, 578`). Pool workers are independent
  emulators.
- **max_steps**: serial `for step in range(max_steps)` + `length = step + 1`;
  parallel finishes on `lane.step + 1 >= max_steps` — both cap length at
  `max_steps` (`eval_game.py:609, 659-661` vs `:496-499, 341-346`).
- **Aggregation**: one shared fold site (`eval_game.py:1017-1056`); records are
  returned in episode order in both (`eval_game.py:402, 501-504, 565-566`).

No mechanical divergence found beyond the RNG derivation itself; the equivalence
is also what `tests/test_eval_parallel.py` asserts (stub + ROM-gated differential,
per the docstrings at `eval_game.py:27-40, 384-396`).

---

## 2. The data

### 2.1 Provenance of the two banked numbers

**14.4%** = pooled 13/90 over attempt-8's three campaign probes
(`runs/online_1_2/campaign.jsonl` lines 3, 4, 7; ledger
`runs/online_1_2_attempt_ledger.md`, "Attempt 8" section):

| probe | checkpoint | eval seed | banked "clear_rate" |
|---|---|---|---|
| line 3 | iter_00860 | 20260815 | 6/30 (0.200) |
| line 4 | iter_00910 | 20260816 | 5/30 (0.167) |
| line 7 (final_probe) | iter_00910 | 20260817 | 2/30 (0.067) |

Probe protocol (`runs/online_1_2/manifest.json` config block;
`scripts/run_online_campaign.py:79-81, 518-519, 529-530, 844, 929`):
`--sequential --level-clear`, sticky 0.25, jitter 16, greedy, 30 eps,
`--max-steps 3000`, `--eval-workers 5 --eval-rng per-episode`, cold from
`stage_03.state`, seed = 20260814 + probe index. **The controller banks
`seq_clear_rate` under the key `clear_rate`**
(`run_online_campaign.py:270-279`: `clear = eval_json.get("seq_clear_rate")`,
falling back to `clear_rate` only when absent).

**2.0%** = 2/100 from the definitive eval of `online_v2_FINAL_consolidated.pt`
(= iter_00910 per ledger), `runs/online_1_2/final_eval_seed7.json` (0/50) +
`final_eval_seed101.json` (2/50): serial (`eval_workers: 1`), shared-stream,
greedy, sticky 0.25, jitter 16, same `stage_03.state` start, **not**
`--sequential` — so its `clear_rate` is `episode_success()`. The sibling pair
`final_eval_i860_seed{7,101}.json` (iter_00860) also reads 2/100.

**The two protocols differ in three ways at once**: RNG regime (per-episode
parallel vs shared-stream serial), seeds (20260815-17 vs 7/101), and — decisive —
**clear predicate** (`seq_clear_rate` vs `episode_success`).

### 2.2 The two predicates are different events

- `episode_success()` = the Rust reward's durable `cleared_any` latch, set ONLY
  on a flagpole grab (`float_state == 3`, `nes_core/src/rewards.rs:1528-1540`) or
  an F52-guarded castle clear (`rewards.rs:1370-1376`); predicate defined at
  `rewards.rs:1552-1562`.
- `seq_clear_rate` under `--level-clear` = `LevelClearTracker.level_cleared`:
  the warp-guarded sequential chain advanced strictly past the warm-start level —
  i.e. displayed `(world, level)` reached 1-3
  (`src/training/smb_sequential.py:218-252, 277+`; RAM bytes `$075F/$075C/$0760`,
  `smb_sequential.py:41-46`). In sequential mode the episode **terminates the
  instant this fires** (`eval_game.py:648-654` serial, `:491-497` parallel).

**Calibration on the probe rows themselves** (`checkpoints/mario_1_2_online_v2/eval.jsonl`
lines 28-30, which are the three probes — same timestamps as campaign.jsonl):
the seq_clear count equals `#(max_gx ≥ 3267)` **exactly** in all three rows,
while `episode_success` on the *same episodes* reads 1/90:

| eval.jsonl row | ckpt / seed | seq_clear | #(gx ≥ 3267) | episode_success |
|---|---|---|---|---|
| 28 | 860 / 20260815 | 6/30 | 6/30 | 1/30 |
| 29 | 910 / 20260816 | 5/30 | 5/30 | 0/30 |
| 30 | 910 / 20260817 | 2/30 | 2/30 | 0/30 |

So on identical episodes the two predicates disagree 13 : 1. `max_gx` clusters
*exactly* at 3267/3268 in the clearing episodes (never higher), consistent with
the tracker firing at a fixed x (the 1-2 exit transition; the coordinate frame
changes past it, so the max latches) — and with the sequential loop freezing the
episode right there, before any flag animation could latch `cleared_any`.
Crucially, the **non-sequential** definitive runs show the same under-latch:
episodes reaching gx ≥ 3267 continue playing there, yet only 2/9 (iter910) and
2/24 (iter860) of them ever latch `episode_success`. The divergence is a
predicate property, not a sequential-mode termination artifact.

### 2.3 Same event, both regimes: no gap

Using `gx ≥ 3267` (⇔ tracker-clear, calibrated above) as the common yardstick:

| checkpoint | per-episode / parallel probes | shared-stream / serial definitive | Fisher p |
|---|---|---|---|
| iter_00910 | 7/60 (11.7%) | 9/100 (9.0%), CI [4.0, 15.0]% | **0.596** |
| iter_00860 | 6/30 (20.0%) | 24/100 (24.0%), CI [16.0, 33.0]% | **0.806** |

Using `episode_success` as the common yardstick:
probes 1/90 (1.1%) vs definitive 2/100 (2.0%) — Fisher p = **1.000**.

Full-distribution check (two-sample KS on per-episode `max_gx`, same checkpoint,
across regimes): iter910 D = 0.100, p = 0.83; iter860 D = 0.103, p = 0.96.
Medians/means and past-x2674 fractions also line up (910: median 1784 vs 2059;
860: 1511 vs 1731 — differences well inside resampling noise at these n).

The regimes produce statistically indistinguishable trajectory distributions.

### 2.4 Episode correlation in shared-stream: none detected

Wald-Wolfowitz runs test (z) and lag-1 autocorrelation on the episode-ordered
outcome sequences of each 50-episode shared-stream run
(binary = past-bottleneck x ≥ 2674; also gx ≥ 3267 and raw gx):

| run | runs-z (x≥2674) | runs-z (x≥3267) | lag-1 r (gx) |
|---|---|---|---|
| final_eval_seed7 | −1.02 | +0.65 | +0.09 |
| final_eval_seed101 | +1.73 | +0.82 | −0.32 |
| final_eval_i860_seed7 | −0.47 | −1.36 | −0.03 |
| final_eval_i860_seed101 | +0.08 | −0.06 | +0.06 |

All |z| < 1.96, signs mixed; the per-episode probe rows (the control, rows 28-30)
show the same magnitudes (|z| ≤ 0.95, lag-1 |r| ≤ 0.09). This matches the theory
in §1.2: shared-stream episodes consume disjoint segments of one i.i.d. stream
and are independent in distribution; the mode costs per-episode replayability,
not independence.

### 2.5 Bootstrap CIs for the banked rates (20k resamples)

- Campaign probe pooled `seq_clear`: 13/90 = **14.4%, 95% CI [7.8, 22.2]%**
- Definitive `episode_success` (FINAL): 2/100 = **2.0%, 95% CI [0.0, 5.0]%**
- Same-event shared-stream flag-x rate: iter910 9/100 = 9.0% CI [4.0, 15.0]%;
  iter860 24/100 = 24.0% CI [16.0, 33.0]%
- Naive banked comparison 13/90 vs 2/100: Fisher p = 0.0020 (and per-checkpoint:
  910: 7/60 vs 2/100 p = 0.027; 860: 6/30 vs 2/100 p = 0.0019) — the gap is real
  *between predicates*, and only between predicates.

Seed-lottery check (b): flag-x splits across the two definitive seeds are
binomial-compatible (FINAL 4/50 vs 5/50, p = 1.00; i860 10/50 vs 14/50,
p = 0.48; episode_success 0/50 vs 2/50, p = 0.49). n=2 streams is thin, but the
observed between-seed spread shows no evidence of a seed-level lottery beyond
binomial noise.

---

## 3. Verdict

**(a) episode correlation in shared-stream — REJECTED.** Theoretically absent
(§1.2 stopping-time argument) and empirically absent (§2.4: all runs-test |z| <
1.96, mixed signs, control-matched).

**(b) seed lottery at n=2 streams — REJECTED as the driver.** Between-seed
splits are binomial-compatible (§2.5), and the same-event cross-regime agreement
(§2.3) leaves no residual gap for a lottery to explain. (n=2 is still too few
streams to *bank* a rate on; see protocol below.)

**(c) mechanical difference in the parallel path — CONFIRMED, but not where the
question pointed.** The parallel executor's RNG/boundary handling is equivalent
to serial (§1.5) and the trajectory distributions match (§2.3 KS). The mechanical
difference is upstream of the executor: **the campaign controller banks
`seq_clear_rate` from a `--sequential --level-clear` probe under the key
`clear_rate`** (`run_online_campaign.py:270-279, 518-519`), while the definitive
eval banks `episode_success`. "Advanced the displayed-level chain past 1-2"
(fires at gx ≈ 3267, terminating the probe episode) and "flagpole/castle success
latch" disagree 13:1 on the same episodes (§2.2).

**(d) genuinely different noise-process distributions — REJECTED.** KS p = 0.83
/ 0.96 on max-gx across regimes at fixed checkpoint; same-event rates agree
(p = 0.60 / 0.81); same-predicate rates agree (p = 1.00).

**Best-supported explanation: (c), as predicate mismatch.** 14.4% and 2.0% are
both faithfully computed — they measure different events on the same underlying
policy behavior. Under the transition event the policy's honest rate is ~9-24%
(checkpoint-dependent); under the flag-latch event it is ~1-2%. Which event
counts as "the honest 1-2 clear" is a *semantics* decision that was never pinned:
the ledger's "reconciling the gap is a documented open item" is answered — there
is no RNG gap to reconcile, only a predicate to choose.

### What makes the banked rate defensible

1. **Pin ONE clear predicate and name it in every receipt.** If "clear" means
   completing the level (flag grab), the level-clear probe must not both
   (i) terminate the episode at the level-byte flip and (ii) be reported under a
   key named `clear_rate` — report `seq_clear_rate` and `episode_success`
   side-by-side and stop overloading the name in campaign events
   (`run_online_campaign.py:279` is the single line that created this incident).
   The 13:1 same-episode disagreement also deserves its own follow-up: whether
   the level-byte flip at gx ≈ 3267 precedes the flag, and whether policies that
   reach the transition fail to convert the remainder, is decidable with one
   instrumented replay of a recorded clearing episode (out of scope under the
   no-rollouts constraint).
2. **Per-episode RNG as canon for stochastic evals.** Episodes become pure
   functions of `(seed, i)`: individually replayable, worker-count invariant,
   and pooling across seeds is clean. Shared-stream remains for bit-reproducing
   historical receipts only. (No statistical validity difference — §2.4 — the
   argument is auditability, not bias.)
3. **≥3 disjoint eval seeds × ≥30 episodes, pooled with per-seed breakdown and
   a bootstrap CI**, per checkpoint, both predicates reported. At true rates of
   2-15%, 90-100 episodes give CIs spanning a factor of ~3 (§2.5); a banked
   headline rate should quote the CI, not the point estimate.
4. **Emit `--max-steps` in the result row.** The result dict
   (`eval_game.py:1060-1103`) omits it; the definitive evals' mean_length 1564.5
   (> the 1500 default) proves a non-default value was passed, but its equality
   with the probes' 3000 is unverifiable from the receipts alone.

---

## 4. Artifact index

- Code: `scripts/eval_game.py` (regimes §1); `scripts/run_online_campaign.py:79-81,
  270-279, 499-530, 844, 929` (probe protocol + banking);
  `nes_core/src/rewards.rs:1370-1376, 1528-1562` (episode_success);
  `src/training/smb_sequential.py:218-289` (LevelClearTracker).
- Data: `runs/online_1_2/campaign.jsonl` (lines 3, 4, 7);
  `checkpoints/mario_1_2_online_v2/eval.jsonl` (lines 28-30 = the three probes;
  full per-episode `max_gx_per_episode` arrays);
  `runs/online_1_2/final_eval{,_i860}_seed{7,101}.json` (50-ep gx arrays each);
  `runs/online_1_2/manifest.json`; `runs/online_1_2_attempt_ledger.md`.
- Analysis code: `scripts/eval_regime_stats.py` +
  `tests/test_eval_regime_stats.py` (15 passed, pytest).
