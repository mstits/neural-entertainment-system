# The learning track, audited against one question — 2026-08-27

**Could this instrument have returned a different answer?**

Applied to the rest of this project over the preceding two days, that
question found `is_clear` compiled to `() > ()` — False always — on 152 of
155 profiles, so millions of steps of `solutions: 0` were a compile-time
constant. It found `area()` returning literal 0, so `n_area == 1` measured
the YAML and not the ROM. It found 186 green detector tests guarding a
detector that could not fire on two witnessed clears, because `coord`
needed a 300-unit drop against observables spanning 1 and 32 units. It
found six vacuous gates, the latest written by work that was holding the
previous five in its brief. None of it was found by looking for it.

The learning track is where the flagship claim lives and it had never had
this treatment. This document is that treatment.

---

## The answer, first

**43 claims examined: 18 stand, 10 weakened, 15 withdrawn.**

**The flagship numbers survive, and one of them is not the number the
ledger has been quoting.**

| Flagship claim | Verdict |
| --- | --- |
| 1-1 LEARNABLE at 0.76 honest | **STANDS.** Re-measured under the canonical protocol: eval seed 0 reads 0.76, reproducing the banked figure. |
| 1-2 BANKED at 2/100 | **SUPERSEDED BY THIS REPO'S OWN LEDGER.** Correct for its own predicate, and stale: `CLAIMS.md` records the arc 0.0 → 2.0% → **38.0%**, and 38/100 reproduces bit-exactly. |
| The 1-2 policy class formally falsified | **WITHDRAWN AS SCOPED.** The same policy class, from the same entrance, at twice the width, clears 38/100. What failed was the CGSA-PPO *recipe*, and that failure stands. |
| Offline imitation for 1-2 CLOSED at 0.0 clears | **STANDS**, and it is the strongest result in the repository. |
| v27 FAIL 0.530 / v28 FAIL 0.670 | **BOTH STAND**, by three independent selectors. |

Said plainly, because the 1-2 negative is quoted as a contribution: **two
of its three parts hold and the third does not.** The offline-imitation
closure holds, and holds by the strongest evidence in this repository. The
CGSA-PPO signpost failures hold. The *generalization from those to the
policy class* does not, and it is refuted by this repo's own later receipt
rather than by any argument made here. The 2/100 headline is not wrong —
it is stale, superseded by a 38/100 the ledger already banked and its
documents never reconciled. Correcting that is the opposite of
over-withdrawal: it restores a positive, not removes one.

The instrument itself came out **sound**. Every defect found is in the
layer *around* the number: what the receipt records, what the test suite
can detect, what the selector ranks, and what the prose says happened.

---

## 1. The instrument: measured, not read

Eleven components of the honest protocol were verified by counting the
events the protocol is *defined* by, not by reading the code that
implements them.

The mechanism: a stub policy whose argmax is always `(prev_action + 1) % n`.
The freshly chosen action can then never equal the previous executed one,
which makes `executed == previous` exactly equivalent to "the sticky roll
fired" — turning an unobservable coin flip into a directly countable rate.

| Component | Measured |
| --- | --- |
| Sticky at the requested p | 0.2545 serial (7,960 eligible), 0.2520 shared-stream and 0.2504 per-episode (59,700 each, SE 0.0018). p=0.0 → exactly 0/7960; p=1.0 → exactly 7960/7960 |
| Sticky `step > 0` guard | First action of every episode is the policy's own; no repeat crosses an episode boundary |
| Sticky granularity | Repeats the previous **executed** action at agent-step granularity (frame_skip 4) — the unit the trainer sticks on and the unit Machado et al. define |
| Start-jitter | Uniform draw on 0..16 inclusive: min 0, max 16, all 17 values present, mean 7.85–8.03 over 300–400 episodes |
| Jitter actually desynchronizes | From each of the three banked start blobs, the 17 jitter values produce 17/17 distinct RAM states after an identical action sequence. Mario's own x is unchanged — it perturbs environment phase, not the player, which is the Machado intent |
| `--eval-rng per-episode` does not collude | Over eval seeds {0,1} × 50 episodes, all 100 (numpy, torch) seed pairs and all 200 individual seeds distinct; zero shared draws |
| The two RNG modes are equivalent | Distributionally identical (sticky 0.2520 vs 0.2504, jitter mean 7.85 vs 8.03) |
| Serial ≡ parallel on **real** emulation | 1-2 consol2, 50 eps, eval seed 7: clear_rate 0.28 and mean_length 622.0 at both 1 and 2 workers |
| `episode_success` | Strict `$001D == 3` flagpole latch or a warp-guarded castle world increment; reset per episode; `first_step` re-syncs prev_world/level so a warm start cannot latch on frame 1 |
| Single-life denominator | player_state 0x0B/0x06 or any lives decrement sets `done`; no respawn can be played into a later clear |
| The 1-1 cold entrance | `entrance_start.state` loads to world 0, area 0, x 40, lives 2, timer 400, score 000000 — power-on equivalent |

Both headline receipts reproduced **bit-for-bit eleven days later** from
the committed checkpoint, start state and harness: 1-2 consol2 seed 7 →
0.48 / mean_length 676.16 and seed 101 → 0.28 / 704.26, matching
`runs/consol2/peak_eval_seed{7,101}.json` to the digit; v28 seed 0 peak →
0.5 / 624.08 at a different worker count.

Across 224 receipts and 11,200 episodes, `clear_rate × n` equals
`count(max_gx >= 3161)` **exactly** in every single receipt, spanning 0.00
to 0.72 with 2,029 positive events, and `max_byte_seen == 0` everywhere.
Nothing in this instrument is vacuous.

### The predicate errs conservative, and that direction matters

`episode_success` misses real completions. Four of fifty episodes in the
banked seed-7 run climb to x=3266/3267, hold player_state 0x05 (autowalk)
for 65–95 agent-steps, then advance `$075C` 1 → 2 with the world byte
unchanged and no life lost — a displayed 1-2 → 1-3 completion the latch
never saw. That is not a warp; warps increment the world byte.

The direction is decisive. A conservative predicate **cannot inflate
anything**: 40/40 solver tapes latch correctly, die-respawn cannot fake it,
and every FAIL comparison applies it to both arms. So every banked positive
is a floor. It convicts the 2/100 artifact — whose true completion rate was
~9–11/100 — and does not overturn the 38%, where `runs/consol2/campaign.jsonl`
shows the chain-vs-strict gap at the peak is at most 1/30 in both legs.

---

## 2. What the instrument could not detect about itself

This is the finding that matters, and it is the exact shape of the
compile-time-constant `is_clear` defect sitting under the flagship claim.

**The two numbers the honest protocol IS could be deleted from the code and
the eval-harness tests stayed green.** Six mutants of `scripts/eval_game.py`,
loaded under the real module name via a pytest plugin with no repo file
touched. Counts below are passing tests across the twelve test files that
exercise this harness (`test_eval_parallel`, `test_eval_action_select`,
`test_sticky_actions`, `test_cold_probe`, `test_consol2`,
`test_smb_sequential`, `test_b6_gate_repair`, `test_soft_distill_cure`,
`test_interference_falsifier`, `test_engine_driver`, `test_online_campaign`,
`test_segment_probe`) — not the whole 5,000-test suite:

| Mutant | Passing, of the harness tests it should have failed |
| --- | --- |
| M1 — sticky removed from **both** executors | 534 pass |
| M2 — jitter removed from **both** executors | 141 pass |
| M6 — jitter range collapsed to {0,1} | 79 pass |
| M3 — sticky at half the requested p | caught only incidentally |
| M4/M5 — RNG derivation broken | caught properly |

The existing coverage is a **serial-vs-parallel differential**, which is
blind by construction to a mutation applied to both branches — the file's
own docstring worries about the one-branch case and stops there. An
independent reproduction found one real-ROM test that catches M1, but only
via an anti-vacuity side-effect (`len(set(max_gx)) > 1`) on a gitignored
local checkpoint, absent on a clean checkout, and it does **not** catch the
jitter deletion at all (47/47 green).

And the receipt could not have told anyone either: `sticky_prob: 0.25`,
`start_jitter: 16` and `stochastic` were all computed from **argv**. A run
with the mechanism physically removed emitted a byte-identical receipt to a
correct one.

Production code was never actually broken — the full git history of these
lines was walked, and every revision that produced a banked receipt has the
guards intact. This was forward risk, not retroactive doubt. It is now
closed on both sides (§5).

> **Live near-miss, recorded for the ledger.** During this audit a
> sticky-disabling mutation was briefly present in the shared working tree.
> Any honest eval invoked in that window would have written a receipt
> reporting `sticky_prob: 0.25` while the mechanism did nothing. It was
> reverted and the tree verified clean. This is the hypothetical, observed.

---

## 3. What the receipt did not record

| Field | State before this audit |
| --- | --- |
| `max_steps` | **0 of 921** honest receipts. Registered at 1500 (v28), 3000 (consol2), 2400 (configs) — and it moves the number: 1-2 reads 0.15 at 1000 vs 0.20 at 1500/3000 |
| `profile` | **0 of 921**, though it fixes the action space, the encoder and the reward id — i.e. what `episode_success` *means* |
| ROM identity | ~0 of 921 (the configs carry `rom_hashes`; the receipts did not) |
| Measured sticky rate | No field. The receipt echoed the request |
| Measured jitter draws | No field. The receipt echoed the request |
| Delivered episode count | `n_episodes` and the `clear_rate` denominator were both the **request** |
| `warp_rate` | Emitted only under `--sequential`, which the registered v27/v28 gate command never passes — so a PASS could not have been scored against its own registered `warp_rate 0.0` |

`runs/online_1_2/final_eval_seed101.json` records `mean_length 1564.5`,
which is arithmetically impossible under the CLI default cap of 1500. That
is proof the banked run used a value its own receipt does not state. It was
recoverable only by re-running until `max_gx` matched — it did, exactly, at
3000; at 1500 the same episodes diverge immediately.

Two receipts with identical visible fields could be different protocols.
No banked number retracts on this — both 1-2 headline receipts reconstruct
intact — but that was luck, not instrumentation.

`warp_rate 0.0` is substantively satisfied everywhere: `max_byte_seen`
retro-certifies it for every banked receipt (1-1 reads 0, 1-2 reads 2 and
3, all below 16). Nothing was wired to check that.

---

## 4. The selector: EXHIBITION outranking LEARNED, in code

Two defects, both now fixed.

**Protocol-blind.** `_best_from_eval_log` keyed on `(clear_rate, timestamp)`
with no protocol filter and no episode weighting — and `eval_game.py`
appends *every* run to that same `eval.jsonl`, including deterministic ones.
Measured across 62 checkpoint dirs: in 8, the selector was decided by a
non-honest row while honest rows existed. `mario_1_2_online_v2` picked
`clear_rate 1.0, sticky 0.0, jitter 0, n=10` over the best honest
`0.633, n=30`. That is the ledger's own EXHIBITION/LEARNED boundary being
crossed by code rather than by prose.

**Ceiling-locked.** `save_winner` skipped on `prev_val >= metric_value`,
and `entrance_trailing_rate` is successes/30 — maximum exactly 1.0. Once a
run recorded 1.0 the gate became mathematically unsatisfiable and the
winner froze. v27 seed 2 and v28 seed 3 both froze at iter 90 of 250.
Across all 8 runs the selector's last save lands at iter 50–120 (median 65)
of 250, with 3–6 saves per run.

This is sharper than the briefed "argmax over a saturated statistic is
near-arbitrary": it is **deterministic, one-directional under-selection**,
and it explains the recorded +0.08..+0.21 shortfall without appealing to
noise. Both v27 and v28 *registered* the correct rule — "ties → later iter"
— and neither implemented it.

---

## 5. What was fixed

| Fix | Where |
| --- | --- |
| Receipt now measures its own protocol: `sticky_applied`, `sticky_eligible`, `sticky_measured`, `jitter_hist` | `scripts/eval_game.py` |
| Receipt now records protocol identity: `max_steps`, `profile`, `rom`, `rom_sha256` | `scripts/eval_game.py` |
| `clear_rate` denominator is the **delivered** episode count; `n_episodes_delivered` emitted | `scripts/eval_game.py` |
| Selector is protocol-aware — when any honest row exists, only honest rows are eligible; ties broken by sample size | `src/training/checkpointing.py` |
| Ceiling lock broken — equal metric with a later `source_iter` now wins, the rule both campaigns registered | `src/training/checkpointing.py` |
| Pooling reads the receipt's real key and refuses a short pool | `scripts/gru_ab_eval.py` |

A verification run on the banked 1-2 checkpoint (8 eps, eval seed 7,
per-episode, 2 workers) now emits `sticky_measured: 0.2572` against a
requested 0.25 (1,427 of 5,549 eligible; SE 0.0058), a real jitter spread
`{3:2, 4:1, 7:3, 9:1, 13:1}`, `max_steps: 3000`, the profile path and the
ROM sha256. A stubbed mechanism can no longer emit a correct-looking
receipt.

Every fix carries a test that fails when the fix is removed — the same
discipline §6 applies to the new checks, applied to the repairs. Reverting
the protocol filter and the ceiling tie-break together fails three tests in
`tests/test_winner_checkpoints.py`
(`..._never_promotes_a_deterministic_replay_over_an_honest_row`,
`..._does_not_treat_a_sampled_unperturbed_run_as_honest`,
`..._equal_metric_prefers_the_later_iter`), and the receipt fields are
pinned by an exact key-surface assertion in `tests/test_eval_pixel.py`, so
the schema cannot drift silently again.

One deliberate scoping note on the honest-row filter: it keys on
**environment perturbation** (`sticky_prob`/`start_jitter`), not on
`stochastic`. `run_consumes_randomness` sets `stochastic` for any sampled
action draw, including on an unperturbed environment — that is not a replay,
but it is not the honest protocol either, and promoting it as one would have
been the same category error in a new place.

---

## 6. The mechanical check

Six vacuous gates proved a text rule does not work here. Two checks now
enforce the two halves mechanically.

### Layer 1 — static reachability (`config_schema.py`)

The existing registry check is **one-way**: it enforces "every key the
trainer consumes is registered", which is why all 34 schema tests pass
green while a dozen keys in the flagship recipe are inert.

`inert_reinforce_keys_under_vanilla_ppo()` derives the other direction
from the AST. `Trainer.run()` dispatches with an early return, so the whole
GA loop below it is dead in the mode every banked learning-track run uses.
The walk takes `run()`'s body up to and including that branch, closes over
`self.foo(...)` calls, then reports every `reinforce` key whose consumption
sites all lie outside that set. Sibling modules that reach into the trainer
by attribute (`ppo_updater.py` does `t.ppo_clip_eps`) are scanned too, so
the analysis errs toward silence — a false "this knob is inert" is the
costlier error.

**Result: 20 keys inert under `vanilla_ppo`. The flagship 1-1 recipe and
both v27 and v28 seed configs declare 12 of them each.**

```
bc_replay_enabled  bc_replay_epochs  bc_replay_every_gens
bc_replay_max_buffer  bc_replay_train_window  episodes_per_genome
warmup_gens_ga_only  preserve_elite_diversity  freeze_pre_ppo_elite
symlog_rewards  enabled  max_steps_per_traj  top_k  async_pipeline
autocast_fp16  torch_compile  vmap_forward  drq_aug  drq_pad
recurrent_env_minibatch
```

`symlog_rewards` is the calibration case: found by hand on 2026-08-26 and
annotated in the v28 config. Deriving it from source reproduces that
finding, which is what makes the other nineteen credible. `ppo_clip_eps`
and `rnd_loss_coef` are the calibration case in the other direction — a
trainer-only scan calls them dead, and they are live via `ppo_updater.py`.

The trainer already carries a hand-written "this knob is a no-op, warn and
disable" guard for `preserve_elite_diversity`/`freeze_pre_ppo_elite`. It
tests `trainer_mode == 'pure_ppo'` and therefore stays silent in the only
mode any banked run used. Deriving the check from source is what stops it
drifting again.

### Layer 2 — runtime fired-or-VOID (`scripts/check_mechanism_receipt.py`)

> A mechanism that is armed and whose own counter never moves for an entire
> run makes that run **VOID** for any claim that mechanism was a variable in.

VOID is not FAIL. A VOID run's other arms may be perfectly sound — v27 and
v28 both keep their FAIL verdicts, because neither depended on ReDo doing
anything. What VOID forbids is the *sentence*: "we tested X and it did not
help" when X never ran.

Four verdicts, and the split between the last two is the checker's own
anti-vacuity property:

- `NOT_ARMED` — not enabled in this run. Nothing to say.
- `FIRED` — armed, counter moved. Certified live.
- `INERT` — armed, counter observed N>0 times, never moved. → **VOID**
- `UNAUDITABLE` — armed, and **no counter exists**. Not a pass.

A counter at zero is only evidence when the counter was actually read.
Zero *observations* of a counter is ignorance, not a null result, and the
checker reports it as ignorance. `HazardMask` accumulates
`actions_vetoed`/`n_fully_vetoed` on every `apply()` and its own docstring
says *"a veto you cannot audit is a veto you cannot trust"* — and nothing
reads them. An armed run prints `[hazard-mask] ARMED` and then never
reports a single veto. That reads `UNAUDITABLE`, which is a defect.

Wired as `make mechanism-check RUN=<dir> [REQUIRE=redo,sil]`, alongside
`clear-lint` — which exists for the same reason, one layer down (it refuses
a solve profile that declares clear machinery it cannot fire). Run against
the real receipts:

```
$ make mechanism-check RUN=runs/v27_fresh_recovery REQUIRE=redo
   [VOID] redo         INERT        peak 0 over 1000 observations
   [ok  ] backward     FIRED        20 distinct values over 1000 observations
exit 2

$ scripts/check_mechanism_receipt.py checkpoints/mario_1_2_online_v2
   [ok  ] backward     FIRED   (held at the terminal rung 0 — armed and arrived)
   [ok  ] kl_anchor    FIRED   109 non-zero observations
   [ok  ] sil          FIRED   peak 3596 over 109 observations
exit 0
```

The second command is the **positive control**, and it is not decoration: a
checker hard-wired to return INERT would pass every negative test in the
suite, and the 1-2 campaign — whose mechanisms were genuinely live — would
be defamed by its own tooling. The ladder distinction is load-bearing too:
entrance-pinned consolidation legitimately holds `tau=0` for a whole run
(armed and *arrived*), while a ladder frozen at a non-zero rung is the real
defect.

#### What it found on its first sweep

Run across 40 learning-track run directories (36 readable), the check
returns:

| Reading | Count |
| --- | ---: |
| `backward` FIRED | 15 |
| **`redo` INERT** | **8** — every v27 and v28 seed, one dir at a time |
| **`backward` INERT** | **2** — NEW, see below |
| `redo` FIRED | 1 |
| `kl_anchor` FIRED | 1 |

**The single `redo` FIRED is the finding that makes the eight INERTs
precise.** It is `checkpoints/mario_1_1_v27_preflight_redo_forced`, the
repo's own forced-recycle sweep, reading `peak 90 over 4 observations`. So
ReDo's plumbing works in this codebase and can recycle 90 units when the
threshold is reachable. "INERT on v27/v28" is therefore a statement about
the **registered operating point**, not about broken wiring — which is
exactly the distinction §7 argues, now produced mechanically by a tool
rather than by a human reading a sweep log.

**Two `backward` INERT runs are new and were not part of any prior
finding.** `checkpoints/mario_1_1_backward_consol` armed the backward
curriculum with 758 states and logged `tau=757/757` on all 140 of its
observations; `checkpoints/mario_1_1_recovery_ppo` armed 27 states and
logged `tau=26/26` across 60. **Neither ladder took a single step.** Both
sat at the rung furthest from the entrance for the whole run, which is the
opposite of the entrance-pinned case the checker deliberately passes.

Scoped honestly: neither run backs a banked headline, and neither is
re-adjudicated here. But `configs/mario_1_1_recovery_ppo.yaml` is named in
`CAPABILITY_REPORT_2026-08-24.md` as the *pattern* v27 was built from
("fresh run with recovery states in the curriculum FROM THE START"), so a
pilot whose curriculum never walked is worth knowing about before the next
campaign inherits it again. This is the check paying for itself on its
first run: two armed-and-inert mechanisms that no audit had looked at.

### The checks were tested by removing them

`tests/test_mechanism_receipt.py` (23 tests, 0.10 s) and
`tests/test_inert_key_reachability.py` (10 tests). Five mutants of the
checker, loaded under the real module name with no repo file modified:

| Mutant | Result |
| --- | --- |
| M1 — the INERT verdict removed (armed always counts as fired) | **8 failed**, including the real-v27 regression |
| M2 — INERT and UNAUDITABLE collapsed | **2 failed** |
| M3 — anti-vacuity removed (an unreadable run certifies clean) | **2 failed** |
| M4 — always INERT (the "defame everything" mutant) | **6 failed** — caught by the positive control |
| M5 — armed-with-no-counter waved through | **1 failed** |
| unmutated control, same harness | **23 passed** |

One entry in the registry failed its own standard and was caught before
shipping, which is worth recording rather than quietly fixing. The
wavefront descriptor originally used `wave_truncations` as **both** its
arming signal and its activity counter — so a non-zero value read as
armed-and-fired and a zero value read as never-armed, making the entry
structurally incapable of ever returning INERT. A vacuous entry inside an
anti-vacuity checker. It now arms on the independent `[wavefront] N cells`
build line, and is renamed `wave_terminal` for what its counter actually
measures: `wave_truncations` counts the terminal rule firing and says
nothing about the monotone-invariant PBRS shaping term, which emits no
counter and stays unaudited. Reading a field by its name rather than its
definition is already an entry in `MISTAKES.md`; it very nearly became one
here.

Layer 1 carries the same guard: `vanilla_reachable_methods` **raises**
rather than returning an empty set when it cannot find the dispatch it
parses. An analysis that reports "nothing is inert" because a refactor
moved the code it reads is a vacuous check, and this project has already
shipped six of those.

### Layer 3 — not shipped, and named

A counter at zero cannot distinguish "the phenomenon did not occur" from
"the threshold is outside the statistic's range". Any mechanism with a
threshold should log the **distribution** of the statistic the threshold
cuts, not only the count of crossings, so a pre-registration can carry a
*reachability condition* checked at the **registered** operating point.
This requires trainer changes and is left as the named next step.

---

## 7. The treatments that were registered and never ran

ReDo is the case that reaches a registration, and it upgrades from "never
fired" to **"could not fire, knowably, before launch"**.

The dormancy statistic normalizes post-activation magnitudes by the layer
mean. `TilePolicyNetwork` applies LayerNorm immediately **before** the
SiLU, renormalizing every forward pass and holding the statistic near 1 —
ReDo is calibrated on un-normalized ReLU nets. Confirmed two ways: the
repo's own pre-launch sweep recycles **zero** units at every tau ≤ 0.20 and
first fires 5 units at tau = 0.25, **ten times the registered 0.025**; and
in the trained v27/v28 checkpoints the minimum LayerNorm gain is 0.18–0.53,
so nothing is silenced by the affine either.

`isolate_tau0.35.log` was written at 00:19:42. The first line of
`train_seed0.log` is 00:22:03. **The evidence that would have voided the
amendment was on disk 141 seconds before the 8-run budget started**, and
was read as a fresh-net artifact.

The pre-registered V7 armed-evidence gate is vacuous in this ledger's exact
sense: its conditions were all evaluated at tau = 0.5 — twenty times the
experimental value — and nothing in it required the **registered** operating
point to be reachable. That is the seventh vacuous gate, same shape as the
six already on the ledger.

**Honest restatement: v27 and v28 each tested ONE variable, not two.** The
FAIL verdicts are untouched; neither depended on ReDo doing anything.

The GA-path knobs (§6, Layer 1) are numerically harmless — each is inert
identically in the baseline and in every comparison arm, so all
like-for-like comparisons stand. What they damage is the recipe text. The
`"Phase-A recipe, verbatim"` header on the flagship config and on all eight
v27/v28 seed configs describes machinery that did not run. Stated plainly:
**the banked 1-1 runs had no clear-anchoring mechanism active at all** —
those configs declare no `sil:` block either, so it is not that the wrong
anchor fired instead of the right one.

---

## 8. The verdicts, re-adjudicated

Both registrations name a selection rule — the peak trailing entrance rate
in the printed `[backward]` telemetry, ties → later — and both adjudications
instead used `winners/best.pt`. Those are different quantities: v27 seed 0's
iter-60 log line prints `trailing 16/30=0.53` while `winners/best.json` for
the same iter records `entrance_trailing_rate=0.8667`, because a second
force-completion pass runs after the telemetry print and before the winner
block reads the window.

The re-adjudication registered its rule in full **before** any new eval ran.
The selection statistic consumes **zero** evaluation data, so it carries no
winner's curse.

| | v27 | v28 |
| --- | --- | --- |
| Banked | 0.530 | 0.670 |
| Registration-literal re-selection | **0.500** | **0.670** (identical) |
| Split-sample cross-fit (v28 ladder) | — | 0.670 or 0.720 |
| Adversarial one-sided upper bound (v27) | **0.592** | — |
| FAIL bar | ≤0.767 | ≤0.767 |

**Three independent selectors, one verdict: FAIL.** `CLAIMS.md` ADDENDUM
V-2's worry that a corrected v27 best-of-4 "could plausibly approach 0.7" is
now measured and refuted.

Two corrections fall out. v27's per-seed field was mostly selector noise:
corrected it is 0.03 / 0.50 / 0.48 / 0.46, so three of four seeds are a
three-way tie and only seed 0 is genuinely bad — which finishes the
withdrawal of "telemetry ranks seeds correctly".

And larger: **the 0.767 bar had never been measured under the protocol it
gates.** It is 46/60 at eval seed 0 only, shared-stream, one worker.
Re-measured canonically: es0 0.76 (reproducing the banked number), es1 0.60,
**pooled 0.68**. The registered thresholds do **not** move — that would be
the goalpost move this ledger treats as fabrication. But the narrative
clause both verdict docs attach, *"no seed's better checkpoint reached the
banked control's own 0.767"*, is not supported for v28, whose 0.670 is
statistically indistinguishable from a same-protocol control of 0.680.

Also recorded: all 16 v27 gate receipts are `shared-stream`/1 worker while
all 16 v28 receipts are `per-episode`/8, so `CLAIMS.md`'s "identical to the
v27 gate in every respect (… `--eval-rng per-episode`)" is false. Measured,
it is not a bias — 53/200 vs 57/200 on matched weights — and the
adjudication reads only training-log metrics, so it is uncontaminated. The
forensics found this on 2026-08-25 and it never propagated.

---

## 9. The 1-2 negative, held to the bar demanded of a positive

The brief called this the strongest result in the repository. Held to that
bar, **as a package it does not hold** — and the thing that refutes it is
this repo's own later receipt, not an auditor's skepticism.

### What survives, and why

**The offline imitation closure is the strongest result here and nothing in
this audit dents it.** It is **predicate-independent**: A7's episodes top
out at `max_gx` 976 of 3,266, median death 646, so it reads 0.0 under the
flagpole predicate, under the level-advance predicate, and under any
x-threshold. §1's conservative-predicate finding cannot rescue it — which
is exactly what makes it strong. It reproduced to the last digit at HEAD,
on a dataset that verifies exactly (40,785 transitions, 422 progressed /
422 failed, terminals labelled, sha256 provenance for the start state and
all four tapes), against a ≥0.60 fit gate pre-registered two hours before
the runs, with a genuinely matched chi-vs-unweighted ablation pair. **That
is a negative built to the standard of a positive.**

"The CGSA-PPO recipe failed its own pre-registered signposts on three
seeds" survives intact, and the SPRT machinery under it is real: correct
Wald thresholds, welds accruing only at p ≥ target, 96–99% welded at full
0.25 protocol noise, median 14–29 completed windows each — not the
3-episode mirage.

The 2/100 figure is correct **for its own named predicate** and reproduces
bit-exactly.

### What fails

The generalization — *"the failure is a property of the policy class, not
of the training procedure"* — is refuted by a **training-procedure change
on the same policy class**:

| | falsification checkpoint | banked 38/100 checkpoint |
| --- | --- | --- |
| Architecture | 2-layer LayerNorm MLP, 712-d tile obs | same |
| Params | 95,943 (`fc1` 128×712, `fc2` 32×128) | 200,071 (`fc1` 256×712, `fc2` 64×256) |
| Entrance | `stage_03.state` | same |
| Honest 1-2 | 0/100 | **38/100** |

Same class, same entrance, twice the width, 0 → 38. Reproduced today by two
independent verifiers and corroborated by the campaign's own per-episode
probe reading 0.40 strict at n=30.

The literature claim fails **as cited**: the source artifact ("external deep
research, 2026-07-23") does not exist in the repo or in
`research-consult/responses/`, whose earliest artifact is 2026-07-27. And
the protocol is bespoke, which makes the claim close to true-by-construction.
It must be weakened to "we are aware of no published per-level 1-2 clear
rate under Machado sticky-0.25, in either direction".

And the 2/100 headline is stale twice over: superseded by 38/100, and
under-counting its own artifact by ~4.5× (§1).

**This is re-scoping, not withdrawal, and it is the opposite of
over-withdrawal: the correction restores a POSITIVE this ledger already
banked and its documents never reconciled.** The audit brief for this very
work quoted the stale falsification as standing — the propagation caught in
the act.

### The same bar, applied to the positive

The 38/100 is banked as "definitive eval on fresh seeds under the canonical
protocol" while both receipts read `shared-stream` / 1 worker; the ledger's
LEARNED canon is `per-episode`. It is not an RNG-mode artifact (the
campaign's own per-episode probe reads 0.40 at n=30, and a 10-episode
per-episode run today read 0.30), but the sentence is wrong and the pooled
100 should be re-run per-episode before it is restored.

And 38% describes **one preserved checkpoint** at a transient probe spike
whose neighbours at iters 1070/1180/1230 read 0.067/0.033/0.0, on a run
killed by its own collapse rule. The 100-episode eval is an unbiased
estimate of that fixed artifact — which is what makes it citable — but the
training recipe is not shown to reach 38% again.

---

## 10. One rejection, and it mattered

A verifier reported that identical parallel per-episode evals return
different clear rates (0.35 then 0.20) on a fixed command. Had that stood,
it would have contaminated **every** multi-worker receipt, including the
entire v28 gate.

Re-run on a clean tree, the same command twice: byte-identical receipts
except the timestamp. The split is best explained by the sticky-disabling
mutation live in the shared working tree during that window — which is
itself the live demonstration of the §2 receipt defect. **Rejected on
measurement.**

---

## 11. The count

| Layer | Claims | Stand | Weakened | Withdrawn |
| --- | ---: | ---: | ---: | ---: |
| The instrument (§1) | 11 | 11 | 0 | 0 |
| Protocol as described in the ledger (§1, §3) | 3 | 0 | 3 | 0 |
| Receipt integrity (§3) | 5 | 0 | 1 | 4 |
| Selector (§4) | 3 | 0 | 0 | 3 |
| Treatments (§7) | 5 | 0 | 1 | 4 |
| Verdicts and the bar (§8) | 6 | 2 | 1 | 3 |
| Flagship results (§9) | 10 | 5 | 4 | 1 |
| **Total** | **43** | **18** | **10** | **15** |

**Track health: SOUND INSTRUMENT, DEFECTIVE RECORD.** The core measurement
survived the question that broke everything else — it could not be made to
return the same answer regardless of the world. Verdict damage is nil: v27
FAIL, v28 FAIL, the offline-imitation closure and both 1-2 replications all
stand. The prose damage is broad and **one-directional**: the documents
consistently describe a stricter, more canonical, better-instrumented
experiment than the one the code ran. That is a more tractable failure mode
than the one this project just finished digging out of.

---

## 12. Next

1. **Re-run the banked 1-2 pooled 100 under `--eval-rng per-episode`** so
   the "canonical protocol" sentence becomes true. Two evals, ~30 minutes on
   a quiet box. This is the highest-value single action: it converts the
   repository's strongest positive from *nearly* canonical to canonical.
2. Ship Layer 3 — threshold mechanisms must log the distribution of the
   statistic they cut, and every pre-registration must carry a reachability
   condition checked at the **registered** operating point.
3. Wire Layer 1 into `check_profile` as a warning, and annotate the 12 dead
   keys in the flagship and v27/v28 configs the way `symlog_rewards` already
   was.
4. Wire the `max_byte_seen` → `warp_rate 0.0` retro-certification into the
   gate scorer, or emit a real warp counter outside `--sequential`.

---

## Receipts

- Code: `scripts/eval_game.py`, `src/training/checkpointing.py`,
  `src/training/config_schema.py`, `scripts/gru_ab_eval.py`
- Checks: `scripts/check_mechanism_receipt.py` (`make mechanism-check`),
  `config_schema.inert_reinforce_keys_under_vanilla_ppo()`
- Tests: `tests/test_honest_protocol_measured.py`,
  `tests/test_mechanism_receipt.py`, `tests/test_inert_key_reachability.py`,
  `tests/test_winner_checkpoints.py` (selector), `tests/test_eval_pixel.py`
  (receipt key surface)
- Re-adjudication: `runs/v27_readjudication_2026-08-27/`
- Prior art this builds on: `CHECKPOINT_SELECTION_DEFECT_2026-08-26.md`,
  `F0_CORRECTED_PEAK_LADDER_2026-08-26.md`,
  `PEAK_INSTABILITY_FORENSICS_2026-08-25.md`,
  `INERT_CONFIG_KEYS_2026-08-26.md`, `LEDGER_AUDIT_2026-08-26.md`

**CPU:** this audit was reads, static analysis and replayed receipts. All
evals ran at `--eval-workers` 1 or 2 against two sibling emulator workflows.
Throughput 389 agent-steps/s at 2 workers (~194/s/worker, ~1,556 NES
frames/s, ~26× realtime) against the repo's ~3,500 samples/s uncontended
figure — this reads as a contended box, as it should.
