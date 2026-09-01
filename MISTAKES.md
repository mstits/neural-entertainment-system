# MISTAKES

Evidence log. Newest first. Not meant to be reloaded each session — this is the
archive the enforced rules in the project instruction file get audited against.

Rules below are **drafts, not enforced.** A root cause graduates to an enforced
one-line invariant only after recurring across 4–5 separate entries.

**Graduation watch** — DERIVED, not hand-maintained. Verify with
`.venv/bin/python scripts/mistakes_tally.py`; `--check` fails on drift.

| root cause | entries | deterministic enforcement |
|---|---|---|
| `[unverified-claim]` | **13** | **PROMOTED 2026-08-28** (project instruction file) + **PROMOTED 2026-08-29** (cross-derivation: a claim gating an irreversible/expensive action is re-derived by a non-producer before the action — 4 same-day catches; operator-approved); no — judgement, working form is the two-session split |
| `[vacuous-gate]` | **14** | **PROMOTED 2026-08-28** (project instruction file) + **SHIPPED** — `scripts/anti_vacuity_scan.py` + registry test `tests/test_anti_vacuity_gates.py` (collected by the full suite); also: ask what the mechanism preserves *by construction* before registering a check on it; emit the symmetric difference between a negative control's rows and the positive control's, VOIDing when it is empty; a regression test must call the changed function, never reimplement its effect beside it; a test whose assertion is a literal string or an object identity should assert the behavior it stands in for; and before trusting an adjudicator's verdict, confirm it ran in the mode the artifact under test actually used |
| `[weak-eval]` | **8** | **PROMOTED 2026-08-28** (project instruction file); partial — enforce min-n at the gate; and emit rows/cells per contrasted region, refusing to grade two regions against separately-estimated nulls when their n differs by more than a registered factor |
| `[purity-leak]` | 3 | **SHIPPED** — `make purity-check` (derived scanner + provenance registry + `WIN_WITNESS_LEDGER`) |
| `[inert-treatment]` | **8** | **PROMOTED 2026-08-28** (project instruction file); **partial** — `scripts/check_mechanism_receipt.py` VOIDs an armed mechanism whose counter never moves, `scripts/redo_arm_gate.py` + `_REDO_ARM_DEADLINE_ITERS` kill an armed-but-never-firing run at iter 25; blind to a mechanism nothing imports, to one armed at a reachable-but-wrong dose, and — newest — to an instrument a registration ADOPTED and never wrote at all, which leaves the same receipt as one that ran and found nothing |
| `[stale-artifact]` | **9** | **PROMOTED 2026-08-28** (project instruction file); candidate — hash the loaded artifact against the built one; never default a harness output path to a live receipt; assert on the bytes written, not the values in hand (**shipped for transition banks**: `assert_bank_wellformed`'s chain invariant); and a derived threshold should be computed from its inputs at adjudication time, not hand-copied at registration time, so an escalation ladder that moves the inputs also moves the derived value |
| `[process]` | **20** | **PROMOTED 2026-08-28** (project instruction file); candidate — an orchestrator may only record a verdict it can prove was measured; a missing or unparseable receipt writes `INFRASTRUCTURE-ERROR`, which is not a verdict; a config file is code — adding or copying a profile runs the full suite, not the subset that covers it; and log to this file as part of the fix commit, not as a followup someone has to ask about |
| `[start-state]` | 2 | — |
| `[false-alarm]` | 2 | — (new category: a guard that fires on legitimate data. Candidate — run any new guard once on a known-good artifact from the real pipeline before arming it on a grid) |
| `[measurement]` | 2 | — |
| `[git]` | 1 | — |
| `[reward-exploit]` | 1 | — |
| `[confound]` | 1 | — (new category 2026-08-29: the contrast itself is invalid — arms differ in a variable the registration never names) |

Bold = at or past the 4-entry threshold, awaiting a call. Six categories were promoted to the enforced ruleset on 2026-08-28 and cross-derivation on 2026-08-29; see the project instruction file.

**This table was hand-maintained until 2026-08-27 and had drifted on all six
categories it listed** (claiming 9 `[unverified-claim]` against 6 real, 5
`[stale-artifact]` against 3), while 12 entries carried no tag and were
invisible to it. That is the defect the engine purity sweep named the same day
— *enforcement must be DERIVED from the declaration, never listed beside it* —
committed inside the log that records it. It is now derived.
---|---|---|
---

## 2026-09-01 — [process] The 13:52:50 backfill commit landed before its own pytest gate returned, and the failure it should have stopped for is dated after the commit
- **What happened:** `6a77a9c` (this file's own backfill commit) was authored at 13:52:50, while the `pytest tests/ -q -m "not slow"` run it was supposed to gate on did not finish until 13:54 — `do3_pytest_receipt.log`, `do3_pytest_full.log` and `DO3_FAILURE_REPORT.txt` are all timestamped after the commit they should have preceded. The run it should have waited for returned 1 failed, 6026 passed (35 short of the 6061-pass baseline): `tests/test_anti_vacuity_gates.py::test_registered_gates_demonstrate_both_polarities` fails with `ImportError: cannot import name 'behavioural_lives_verdict'` — the same DO-2 WIP-not-on-main gap the 2026-09-01 execution pass had already found and named ("main cannot pass its own suite without this WIP"). The failure report even states the operating rule verbatim — "any other pass count is a stop, report it, do not commit" — and was written to disk two minutes too late to stop anything.
- **Root cause:** "Receipts before committing" was followed for the cheap `make` gates (13:45) but not for pytest: the commit was made while the slow suite was still running instead of after it returned, so there was no pass count to check against the rule at commit time even though the rule required one.
- **Consequence:** No wrong claim banked in the backfilled entries themselves — the DO-2 gap is pre-existing and independently documented in the MASTER-LIST, not something this commit caused or hid. Cost: a stop condition existed and was violated anyway, on the one file whose entire subject that day was committing without following rules that had just been promoted.
- **Rule (draft):** A commit gated on a suite run does not exist until the suite's own exit code does — stage and hold, run the suite, then commit; a `git log` timestamp earlier than its own cited receipt file's mtime is the mechanical tell that this happened.

---

## 2026-09-01 — [stale-artifact] The unsafe-inventory gate was red on `main` for four days and nothing looked at it
- **What happened:** `scripts/check_unsafe_inventory.py` — built 2026-08-25 so
  `SECURITY.md`'s per-file `unsafe` counts could never drift silently again —
  has failed on HEAD since `e9dafa8` (2026-08-28): the corrected
  `step_all_native` ordering comment added two lines containing the word
  `unsafe` to `pool.rs` (75 → 77; crate-wide 155 → 157). The gate is a
  prerequisite of `make test` (`Makefile:161`), so `make test` has been red on
  `main` for four days. The 2026-09-01 all-skills review found it by running
  the script; nothing in the repo had run it since the drift.
- **Root cause:** A mechanism's counter moved and nothing watched the counter —
  the inert-treatment shape, applied to the gate itself. The commit that moved
  the count did not run the gate that guards the file it touched.
- **Consequence:** No wrong claim banked; the drift is two comment lines. Cost:
  four days of a red `make test` on `main`, and a document whose purpose is to
  be exact being off by two — the same "document that exists to track this
  drifted out from under itself" the script's own docstring was written about.
- **Rule (draft):** A fix commit re-runs every gate that guards a file it
  touches, and a moved inventory count is part of that commit's receipt.

---

## 2026-09-01 — [process] Five audit-sourced fixes shipped in one 51-second batch with no entry here, the day after "log in the fix commit" graduated to enforced
- **What happened:** `a0a7664`, `e4aa905`, `0d87216`, `7924352`, `6b19aa8` (2026-08-29 13:48:25–13:49:14) each fixed a defect an external audit had named, each with a regression test, and none carried the entry the rule at the top of this file requires. The five entries below are that backfill, written 2026-09-01 from the commit messages; each names its fix commit.
- **Root cause:** The rule was promoted on 2026-08-28 and applied forward by habit, not by a check; a batch of fixes landed under time pressure the next day and the habit did not fire five times in a row.
- **Consequence:** No wrong claim banked. Cost: the evidence log under-counted four categories for three days, and the derived watch table was accurate only because nothing had been logged to drift from.
- **Rule (draft):** `mistakes_tally.py --check` (already in `make test`) should also fail when a commit whose subject starts with `fix:` or names an audit touches no entry here — the rule is machine-checkable at the commit boundary even if the entry's content is not.

---

## 2026-09-01 — [process] Two dated concurrent-writer incidents on shared paths with no mutual exclusion (backfill: fixed 2026-08-29 in `a0a7664`)
- **What happened:** A `git update-ref` race on `refs/heads/main` between worktrees, and duplicate chain watchers running the same 192-eval ladder against shared receipt paths (2026-08-29) — harmless only because eval writes are deterministic.
- **Root cause:** `train_game.py` had a PID + `ps -o lstart=` fingerprinted `.run.lock` since its rewrite; every other writer class re-invented it or skipped it. Ownership of `runs/` and `checkpoints/` was a directory-naming convention enforced by nobody.
- **Consequence:** No data lost, by luck of the domain. `a0a7664` extracted the lock into `src/utils/run_lock.py` with a CLI wrapper; at ship time it guarded two writers. The remaining writers (`go_explore_chain.py`, `go_explore_solve.py`, `run_v31_eval_ladder.py`, `run_online_campaign.py`, `merge_recovery_ladder.py`, `watch_asm.py`, `discover_observables.py`) are still unguarded as of this entry, and nothing locks git refs.
- **Rule (draft):** A script that writes under `runs/` or `checkpoints/` acquires `run_lock` on its output path before its first write; a census of writers that never import it is a test, not a grep someone remembers to run.

---

## 2026-09-01 — [measurement] `vanilla_ppo_approx_kl` reached literal JSON `Infinity` in `metrics.jsonl` and no consumer could see it (backfill: fixed 2026-08-29 in `e4aa905`)
- **What happened:** The k3 estimator `((ratio-1) - log(ratio))` returns `+inf` the moment one importance ratio underflows to 0; the raw mean landed in `metrics.jsonl` as `Infinity` — up to 84 of 250 generations (33.6 %) on one v32 seed.
- **Root cause:** No clamp on the estimator and no non-finite check on the sink. `jq` reads `Infinity` as `DBL_MAX`; `analyze.py`'s finite filter dropped the rows. The single most alarming reading a PPO run can produce was invisible to every consumer.
- **Consequence:** The v32 campaign under study carried the signal that mattered most and no tool surfaced it. `e4aa905` clamps to ±1e4 and counts clamped rows (`kl_clamped_rows`).
- **Rule (draft):** Every metric written to the record is finite by construction or counted when it is not; a sink that can write `Infinity` has no consumers that can read it.

---

## 2026-09-01 — [vacuous-gate] `analyze.py`, built to surface surprising fields, filtered out the surprise (backfill: fixed 2026-08-29 in `0d87216`)
- **What happened:** `_is_number()`'s `math.isfinite` treated `Infinity` as "not a number", so the tool excluded exactly the rows it existed to rank. Pointed at v32 seed 1 it computed cohort stats over 66 finite rows and said nothing about the 84 `Infinity` rows beside them.
- **Root cause:** A type filter written for robustness doubled as a silencer; nothing asked what the filter *removed*.
- **Consequence:** A green analysis over a campaign whose defining defect was in the dropped rows. `0d87216` gives non-finite fields a dedicated top-ranked row instead.
- **Rule (draft):** A tool that ranks anomalies reports what it filtered out, as a count, next to what it ranked — a filter's discard pile is part of its output.

---

## 2026-09-01 — [inert-treatment] The engine's documented kill switch was checked everywhere and set nowhere (backfill: fixed 2026-08-29 in `7924352`)
- **What happened:** The circuit breaker only appended a string to the journal, and `state["halted"]` was read in `guard_reasons()` but assigned by no code path in the tree — a kill switch that did not exist, on the one subsystem designed to run unattended for weeks (external audit 2026-08-29).
- **Root cause:** The guard was written against a state the design intended; the writer of that state was never built, and the guard passing was indistinguishable from the guard working.
- **Consequence:** Silent halts with no notification. `7924352` pushes one macOS notification per blocked episode (latched, re-armed on clear) and makes `--halt "reason"` / `--clear-halt` real and journaled.
- **Rule (draft):** A guard that reads a state field is registered together with the code that writes it; a field read by a guard and written by nothing is a VOID mechanism (the existing `check_mechanism_receipt.py` shape, applied to state, not counters).

---

## 2026-09-01 — [process] Only the engine scheduler checked free disk; the processes that actually write did not, and ENOSPC degraded into a warning (backfill: fixed 2026-08-29 in `6b19aa8`)
- **What happened:** `DISK_FLOOR_GB=40` lived in `guard_reasons()` only. A hand-launched `train_game.py` or `run_online_campaign.py` on the 91 %-full volume started anyway, and an ENOSPC during a checkpoint save was caught by `except Exception: log.warning` — training continued indefinitely with durability silently off.
- **Root cause:** The floor was placed on the scheduler, the least disk-hungry process, and the save path's catch-all treated a durability failure like a transient.
- **Consequence:** A three-week run could have burned its compute for output that would not survive a crash. `6b19aa8` adds the floor to both launchers (fail-closed on measured low disk, `--allow-low-disk` override) and halts after 3 consecutive failed saves. `go_explore_chain.py` and `night2_runner.py` still have no floor as of this entry.
- **Rule (draft):** Every long-running entry point checks the disk floor at launch, and a failed durability write is an escalation, never a warning.

---

## 2026-09-01 — [process] Orphaned child on wall-clock timeout: the reaper killed the group leader and left the trainer running (backfill: fixed 2026-08-28 in `9bfe035`)
- **What happened:** `detach.py` launches with `start_new_session=True`, making the managed pid a session/process-group leader; `reap()`'s `os.kill(pid, 9)` killed only the leader, leaving `run_online_campaign.py`'s own `train_game.py` child running as an orphan after a timeout. `emulator_busy()` then reported that lane "held externally" forever — a silent stall with no further diagnostic.
- **Root cause:** Process-group semantics were not part of the reaper's contract, and the journal logged legitimate holds and leaked orphans with the identical line, so nothing could flag the stall as anomalous.
- **Consequence:** Exactly the failure a multi-day unattended run exists to survive. Fixed to `os.killpg(getpgid(pid), 9)` with a plain-kill fallback. The first regression test called `os.killpg` directly instead of through `reap()` and passed with the bug in place — a `[vacuous-gate]` in miniature, caught by revert-verify and rewritten to exercise `reap()`.
- **Rule (draft):** A regression test calls the changed function, never re-implements its effect beside it (already drafted under `[vacuous-gate]`; this is its fourth instance).

---

## 2026-09-01 — [inert-treatment] `trunc_buf` was correctly populated and never reached GAE on a live config — truncation bootstrapped as death (backfill: fixed 2026-08-28 in `9bfe035`)
- **What happened:** The rung-budget cut populated `trunc_buf` independent of `wave_monotone`, but all three `batched_gae`/`ppo_updater` call sites gated *passing* it on that unrelated flag. `configs/mario_1_2_backward.yaml` never sets `wave_monotone`, so every rung-budget truncation on that config was bootstrapped as a terminal state (the Pardo time-limit bug) instead of bootstrapping `V(s)`.
- **Root cause:** A correctly built mechanism wired behind the wrong condition; nothing asserted the buffer's contents ever reached the consumer.
- **Consequence:** Biased value targets on a live learning config for as long as that gate stood. Passing `trunc_buf` unconditionally is behavior-preserving wherever nothing truncated (an all-`False` array is `None` to `ppo.py`'s check) and correct everywhere else. Revert-verified. Not yet training-verified: `mario_1_2_backward` has not trained since 2026-08-14.
- **Rule (draft):** A buffer populated for a consumer carries a test that the consumer received non-trivial contents at least once under the config that populates it.

---

## 2026-09-01 — [process] Timeout ordering inversion: the outer reaper fired 1,500 s before the inner timeout could report (backfill: fixed 2026-08-28 in `9bfe035`)
- **What happened:** `suite_check`'s `timeout_h=1.0` (outer wall-clock reaper) was tighter than its own `--timeout 5400` (`run_suite_check.py`'s inner subprocess timeout), so a merely-slow suite was SIGKILLed 1,500 s before its own timeout could fire and report cleanly, and counted as an action-level failure.
- **Root cause:** Two timeouts on the same job, owned by two layers, with no rule that the outer must exceed the inner plus reporting time.
- **Consequence:** Slow-but-healthy suites recorded as failures with no diagnostic. Fixed by ordering the two.
- **Rule (draft):** When a job has an inner timeout, the outer reaper is derived from it (inner + margin) at launch, never hand-set beside it.

---

## 2026-08-29 — [confound] A pre-registration was internally airtight and still did not isolate the variable it named
- **What happened:** The substrate experiment's registration survived a
  hostile cross-review (which caught and closed a real threshold bias),
  ran clean, and produced a decisive FAILED (42 vs 158, dual-derived
  with exact agreement) — and only then did the verifier notice the
  design compared a shared-trunk net trained by OFFLINE BC against
  baselines trained by ONLINE RL with curricula. Two variables changed
  at once; the result cannot attribute the collapse to sharing. The
  registered claim scope said "interference-only" while the design
  could not deliver interference-only.
- **Root cause:** Both reviewers audited the registration's internal
  consistency (threshold provenance, composition, claim wording) and
  neither asked the isolation question: what OTHER difference between
  the arms could produce the registered outcome? The original pooled
  falsifier had the same structure and was rescued only accidentally,
  by a within-run contrast (pooled BC beat one specialist) this run did
  not produce.
- **Consequence:** No wrong claim banked — the confound was caught at
  verdict time by the independent re-derivation, before the verdict
  doc. Cost: the main run answers a weaker question than registered,
  and a control run (registered as Addendum 2, single-variable:
  sharing) is owed to disambiguate.
- **Rule (draft):** A registration names, in its own text, every
  variable that differs between the arms and why each cannot produce
  the registered outcome; a reviewer's checklist includes "what else
  differs?" — internal airtightness is not isolation. New category:
  the failure is neither the eval's power ([weak-eval]) nor the
  numbers' provenance ([unverified-claim]); it is the contrast itself.

## 2026-08-29 — [unverified-claim] Greenlit an archival target on a from-memory claim about where its receipts live
- **What happened:** Asked whether `runs/ng_odo_*` (188G) could enter the
  archival sweep, this session answered "the odometer certification's
  receipts are banked under docs/receipts, so raw ng_odo runs are exactly
  what your battery is for." Re-derivation by the sweep's executor found
  the opposite: docs/receipts holds only odometer swap/cohort surveys;
  `RECEIPTS_INDEX_2026-08-24.md` names `runs/ng_odo_scene/` the
  "Headline receipt" and `runs/ng_odo_doa/` the receipt run for a FORGE
  claim; the capability report attributes best_score 74,783 to two more
  of the dirs. The run dirs ARE the receipts. Verified conceding:
  3429d8a banked the odometer CODE and cert script, no receipt copies.
- **Root cause:** A memory summary ("odometer SHIPPED, certified 5/5")
  was inflated at recall time into an artifact-location claim the memory
  never contained, and shipped as a greenlight without an `ls`. Fourth
  same-day instance of the pattern the cross-derivation entry records —
  and the first where the flipped claim was this session's own
  greenlight of someone else's irreversible-adjacent action.
- **Consequence:** None on disk — the executor re-derived before acting
  (the rule working), held 188G hot, and the veto stands.
- **Rule (draft):** A claim that an artifact exists at a path is made by
  listing the path, never from a remembered summary; a greenlight is a
  claim that gates an action and gets the same re-derivation as any
  other.

## 2026-08-29 — [process] The archival sweep could not see root-level-file layouts, and said nothing
- **What happened:** The sweep that evaluated runs/ for archival walked
  SUBDIRECTORIES only. `ng_odo_*` keeps its ~188G in five root-level
  files (traces.pkl, archive.pkl, progress.jsonl, roots.json,
  archive.stats.json) with one tiny subdir — so it was never evaluated,
  never logged as skipped, and absent from the output entirely. Caught
  by the executing session only because it went looking for why ng_odo
  never appeared.
- **Root cause:** The sweep's walker assumed bulk lives in subdirs; a
  layout class outside the assumption produced silence
  indistinguishable from "evaluated, nothing to do."
- **Consequence:** None this time — the gap was found before any verdict
  relied on the silence. But every "the sweep found nothing else" claim
  to date implicitly excluded this layout class without saying so.
- **Rule (draft):** A sweep names what it cannot see. Any tool whose
  verdict can be "nothing found" enumerates the classes it skipped
  (layouts, extensions, depths) in its own output — silence licensed by
  coverage, never by construction. (The repo's "no silent caps"
  workflow principle, applied to local tooling.)

## 2026-08-29 — [process] A banked hash never named its domain, and read as corruption of the flagship artifact
- **What happened:** `docs/receipts/full_run/receipts.json` attests the
  complete-game tape with `tape_sha256 = 38ae326e…`. A file-level
  verification (`shasum full_tape.npy`) produced `034f6eb7…` — twice, two
  tools — and the flagship's banked deliverable was reported as failing
  its own attestation, freezing a 127G archival decision. Re-derivation
  against the producer (`assemble_full_run.py:223`) resolved it: the hash
  is over `tape_arr.tobytes()` — the raw 31,202 uint8 action bytes — and
  the `.npy` file adds a ~128-byte numpy header. The tape is intact; the
  attestation is valid; exact match on re-derivation in the correct
  domain.
- **Root cause:** The receipt recorded a hash without recording what the
  hash is OVER. Any verifier without the producer's source to hand picks
  the obvious domain (the file) and reads integrity failure on an intact
  artifact.
- **Consequence:** A correct STOP on the archival (right call given the
  information), two sessions' verification round, and a flagship
  "corruption" scare that was a labeling gap. Fixed same day: the
  assembler now writes `tape_sha256_domain` into every future receipt,
  `docs/receipts/full_run/VERIFY.md` documents the recipe beside the
  banked (unmodified) receipt, and
  `tests/test_full_run_receipt_integrity.py` verifies the tape in the
  receipt's own domain every suite run — real corruption now reds a test
  instead of hiding under an ambiguous hash.
- **Rule (draft):** A hash in a receipt names its domain in the receipt.
  A hash whose domain must be recovered from the producer's source code
  attests nothing to anyone else.
- **Amendment (2026-08-29, same day — the verifier-side twin):** During
  independent verification of the ng_odo bank, the verifying session
  regexed BANKING_2026-08-29.md for sha/path pairs, found none, and
  nearly reported a gap — the shas live in rederivation_2026-08-29.json,
  exactly where the manifest's own line 17 points. Self-caught and
  disclosed. Same failure shape from the other side: the producer must
  name the domain/location, and the verifier must FOLLOW the named
  pointer to its referent before verdicting a gap. Two same-day
  instances of "checked the wrong artifact" (tape file vs array bytes;
  manifest prose vs pointed-to JSON).

## 2026-08-29 — [process] Three same-day catches, one shape: a claim gating an action, re-derived by a session that didn't produce it
- **What happened:** Three times today a load-bearing claim changed on
  independent re-derivation: (1) a 204-verdict receipt log read as a
  completed successful experiment — 100% mock test pollution; (2) a
  pre-registered baseline threshold (153) carried a 5-point bias in the
  hypothesis's favour — the like-for-like figure is 158, caught in
  cross-review before compute; (3) the flagship tape read as failing its
  attestation — a hash-domain labeling gap, artifact intact (previous
  entry). Different producers, different defects, one detection
  mechanism: the session that did NOT produce the claim re-derived it
  before anyone acted on it.
- **Root cause:** Producer-side verification shares the producer's blind
  spots — the assumptions baked into the artifact are baked into the
  check. All three defects were invisible from inside and cheap to see
  from outside.
- **Consequence:** None of the three reached a ledger, a deletion, or a
  campaign launch. The pattern is currently a habit two sessions happen
  to share, not a rule anything enforces.
- **Rule (draft):** A claim that gates an irreversible or expensive
  action — a deletion, a campaign launch, a ledger bank — is re-derived
  from its underlying artifacts by a session (or at minimum a fresh
  derivation path) that did not produce it, before the action. The
  producer's own receipt is an input to that check, never a substitute
  for it.
- **Amendment (2026-08-29, same day):** Two further catches where the
  split changed a CONCLUSION rather than confirming a number — the
  substrate confound (both pre-run reviews passed; the blind verifier
  caught the two-variable contrast at verdict time) and the control's
  direction finding (+20 transfer on 1-1 hidden inside a −12 aggregate
  net). Both times the raw per-leg data carried the finding and the
  aggregate hid it. Corollary: derive PER-UNIT before deriving the
  aggregate — an aggregate can be correct and still describe nothing
  real.

## 2026-08-29 — [stale-artifact] A test wrote mock verdicts to a production receipt path for 12 days
- **What happened:** `tests/test_eval_shared_substrate.py::test_run_never_
  invokes_a_real_subprocess` passed the PRODUCTION config through `run()`
  unredirected. Every full-suite execution — the engine's recurring
  suite_check included — appended 4 mock legs (checkpoints under
  `/fake/dir/`, every clear_rate exactly 0.5) plus one mock SUPERSEDES
  verdict (delta +47) to `runs/shared_substrate/eval_shared_substrate.jsonl`
  and overwrote `manifest.json`. 204 fabricated verdicts accumulated from
  2026-08-17 to 2026-08-29; the directory read as a completed, successful
  trunk-plus-heads experiment. The real eval has never run (its head
  checkpoint dir does not exist).
- **Root cause:** The test's two tmp-redirected siblings set
  `receipt_log`/`manifest_out` into `tmp_path`; this one reused the module
  CONFIG verbatim because its assertion was about subprocess calls, not
  writes — and nothing checked that a test leaves production paths alone.
- **Consequence:** A production receipt path held nothing but fabricated
  measurement-shaped bytes; it burned two sessions' attention as an
  "unidentified writer" during a disk audit, and its verdict was briefly
  relayed as a real result (own entry). Fixed same day: test redirected,
  autouse guard added that fails any test in the module touching the
  production paths (revert-verified: restoring CONFIG trips the guard),
  polluted files quarantined with a README.
- **Rule (draft):** A test that exercises a writer takes the writer's
  OUTPUT PATHS as part of its fixture, never from production config; and
  a module whose subject writes to `runs/` ships an autouse guard
  asserting the production paths are byte-identical after every test.

## 2026-08-29 — [unverified-claim] Relayed a receipt-log verdict to a peer before re-deriving it
- **What happened:** Asked to identify the writer of `runs/shared_substrate/`,
  the directory's manifest and final verdict row (SUPERSEDES, 200 vs 153)
  were relayed to a peer session — with a re-derive-before-quoting caveat
  attached, but relayed nonetheless — as "at-rest receipts of a completed,
  apparently-successful experiment." Re-derivation an hour later showed
  the log was 100% mock test output (previous entry).
- **Root cause:** Treated a well-formed manifest + verdict row as evidence
  the measurement happened. The invariant already says the opposite: a
  verdict may only be recorded from a receipt that PROVES it was measured
  — and proving means re-deriving the aggregate from the underlying rows,
  which took one script and immediately exposed `/fake/dir/`.
- **Consequence:** A correction message to the peer before it reached the
  project owner as good news; no artifact or ledger was touched with the
  false number. Self-caught by running the standing invariant — but after
  the relay, not before.
- **Rule (draft):** Re-derivation happens BEFORE the first relay, not
  before the ledger write; a caveat attached to an unverified number does
  not license sending the number.

## 2026-08-29 — [process] The citation-grep deletion rule was almost inverted into a selector
- **What happened:** During the approved disk sweep, the rule "grep the
  candidate path against CLAIMS.md and docs/ before deleting" correctly
  vetoed 4 of 5 approved deletions (507G worth: a glob dependency, a
  banked receipt pointer, a 2-day-old witness ledger, a config-named
  observables receipt). A salvage plan was then drafted to prune each
  vetoed directory down to its CITED subdir and delete the uncited bulk.
  Caught before execution: in each directory the uncited bulk was NEWER
  than the cited core (stage1_v8 postdates cited stage1_v6; ll_2_2_*
  postdates cited ll_1_1_transfer). The plan would have deleted the
  newest work in each line and kept the oldest.
- **Root cause:** Reading citation as a liveness marker. Citation marks
  what got WRITTEN UP; the newest work in an active line is precisely the
  part not yet cited anywhere.
- **Consequence:** None — self-caught before any deletion. Recorded
  because the inversion is the natural next step for anyone holding the
  rule, and it fails worst on the most active directories.
- **Rule (draft):** The citation-grep rule is a VETO on deletion, never a
  selector for it. "Uncited" licenses nothing; recency inverts citation.
  Positive selection for deletion needs its own evidence (age + zero
  refs + owner sign-off), per item.
- **Amendment (2026-08-29, same day):** A literal-path grep also cannot
  see GLOB coverage — `assemble_full_run.py` reaches `lvl_*`,
  `clear_detect.py:3733` and `engine_status.py:119` walk `runs/**`
  recursively, none of which contain the candidate path as a literal.
  Executing the sweep with a glob-aware battery (literal refs + glob
  patterns + open handles + campaign/solution markers, re-asserted at
  execution time) caught four additional directories the literal grep
  had passed. The veto check must resolve globs, not just grep strings.

## 2026-08-29 — [process] v33 configs minted without a full-suite run; the roster gate caught it 19 hours late
- **What happened:** The four v33 capacity configs (af054bf, minted
  2026-08-28 18:58) declare `reward_id: mario` and were never appended to
  `tests/reward_dispatch_baseline.json` — the documented one-batch-per-
  addition mechanism every prior post-freeze profile set (v31, v32) went
  through. `test_roster_dispatch_matches_the_frozen_pre_change_baseline`
  failed on the next full-suite run, 19 hours later, after the campaign
  had already trained and been adjudicated.
- **Root cause:** The mint ran targeted checks (construct-validity
  preflight, smoke, redo-disabled sanity) but not the full suite. The
  drafted rule for exactly this existed at mint time: "a config file is
  code — adding or copying a profile runs the full suite, not the subset
  that covers it."
- **Consequence:** No wrong result — the arm the configs get is the one
  they explicitly declare, and the fix is the sanctioned 4-row append —
  but the suite carried a red test through a full campaign, its
  adjudication, and six unrelated commits, any of which could have been
  blamed for it. Separately measured while diagnosing: 4 OTHER tests
  (timing, highlight ring, learning-regression, consolidate smoke) fail
  under two concurrent full suites and pass quiet — concurrent full
  suites on one machine produce contention artifacts that mimic real
  regressions.
- **Rule (draft):** Minting or editing any config runs the FULL suite
  before the commit lands (recurrence of the standing [process] draft —
  count it); and a suite verdict is only readable if no other full suite
  was running concurrently on the machine.

## 2026-08-29 — [stale-artifact] `git checkout <file>` as mutation-restore wiped an unstaged fix
- **What happened:** During revert-verification of the approx-KL clamp, the
  mutation was applied to `src/training/ppo.py` and then "restored" with
  `git checkout src/training/ppo.py` — which restored from the INDEX, where
  the fix had never been staged. The entire unstaged fix was destroyed, not
  just the mutation; the test run that followed errored on a missing symbol
  instead of passing.
- **Root cause:** Used a whole-file VCS restore to undo a one-line mutation
  on a file whose current state was ahead of both index and HEAD. The
  mutation-testing loop (patch → run → restore) needs a restore that
  targets the mutation, not the file.
- **Consequence:** ~2 minutes: the fix was re-applied from the still-open
  patch script and re-verified green. Zero loss only because the exact
  patch text existed in the session; on hand-typed edits this deletes real
  work.
- **Rule (draft):** During mutation-verification of uncommitted work,
  restore by reversing the mutation string (or stage/stash the good state
  first); never `git checkout`/`git restore` a file that carries unstaged
  fix work.

## 2026-08-29 — [process] Two chain watchers ran the same eval pipeline concurrently
- **What happened:** A background chain watcher believed dead (its parent
  task had been cleaned up and an earlier pkill was assumed to have reached
  it) had survived; when a second instance was launched directly, both ran
  the 192-eval ladder against the same receipt paths simultaneously.
- **Root cause:** Assumed a process was dead from the disappearance of its
  parent task instead of verifying the process itself; then launched a
  duplicate without a lock or a liveness check.
- **Consequence:** None measured — verified rather than assumed: 192/192
  receipts, zero bad, zero missing, zero extras, and the evals are
  deterministic per (checkpoint, eval-seed) so double-writes rewrote
  identical bytes. The waste was duplicate compute, not corruption.
- **Rule (draft):** A pipeline that writes shared paths takes a lockfile or
  verifies no sibling is running before starting; "its parent is gone" is
  not evidence a process is.

## 2026-08-28 — [unverified-claim] The registration asserted the configs' state without reading the configs
- **What happened:** V33's protocol section stated "ReDo stays at its
  schema-default inert setting in all four configs." The configs, minted as
  two-line diffs from v28, inherited v28's `redo_enabled: true` at the inert
  tau — and the v32-era arming deadline (added after v28 ran) VOIDed all
  four seeds at iteration 40, across all six supervisor restarts. ~2.2 h.
- **Root cause:** Wrote what the configs SHOULD contain into the
  registration and verified the intended variable (`tile_hidden_dim`) but
  not the asserted invariant (`redo_enabled`); a diff-based mint inherits
  everything the diff does not name.
- **Consequence:** One full campaign attempt VOID before any grid existed.
  The guard chain worked exactly as designed — this is the first time an
  inert-armed run was refused BEFORE burning its 7 h, which is what the
  deadline was built for. Configs conformed to the registration; attempt-1
  artifacts quarantined; partial checkpoints deleted before relaunch.
- **Rule (draft):** Every invariant a registration asserts about its configs
  is checked against the minted files before launch — the same preflight
  discipline the dose gets, applied to the things held constant.

## 2026-08-28 — [false-alarm] A preflight check failed twice on a healthy mechanism
- **What happened:** The v33 dose-reachability smoke "failed" twice — at 5
  and then 10 iterations — before passing at 11. Both failures were in the
  CHECK (checkpoints save at `it % 10 == 0` with `it > 0`, and `--iters N`
  runs iters 0..N-1, so 11 is the save minimum), not in the dose; the
  mechanism was healthy throughout.
- **Root cause:** Wrote the check's expectations from assumption (a save
  will exist after a short run) instead of from the save condition in the
  code it was checking.
- **Consequence:** Minutes lost, both false starts recorded in the receipt
  rather than erased — a preflight that hides its own false alarms teaches
  the next author the same wrong assumption.
- **Rule (draft):** Before asserting an artifact exists, read the condition
  that produces it; a checker's expectations come from the code, not from
  what would be convenient for the check.

## 2026-08-28 — [process] Recommended a pre-registered consequent whose antecedent never occurred
- **What happened:** Recommended retiring ReDo off a VOID campaign. The
  registration grants retirement to exactly two outcomes — a FAIL (no Theta
  exists) or a second Phase-R NO-GO (it GOed) — and states VOID takes no
  branch. The recommendation took the most terminal branch available.
- **Root cause:** Treated "conservative direction" as exempt from the
  moved-goalpost rule; a null from a design pre-declared blind to the
  plausible effect size felt conclusive.
- **Consequence:** Caught by the commissioned direction review before any
  compute or ledger change; the retirement was overturned.
- **Rule (draft):** The goalpost rule is symmetric — lifting a pre-written
  conclusion out of its registered antecedent is fabrication in the
  conservative direction too.

## 2026-08-28 — [unverified-claim] Attributed a five-campaign failure to the architecture without testing the operator
- **What happened:** Drafted a stopping statement blaming Linear→LayerNorm→SiLU
  for ReDo's failure. The review measured the actual cause: recycle() pins the
  recycled unit's LayerNorm gain at 1.0 with zero gradient (both head columns
  zeroed), and the dormancy score is a rank-readout of that gain (Spearman
  +0.93) — the operator deposits its own output at the bottom of the statistic
  that selects it. Verified on raw checkpoints: gain exactly 1.000, columns
  exactly 0.000.
- **Root cause:** Five campaigns varied dose and cadence; the reset operator
  itself was never treated as a variable, and the stopping statement inherited
  that blind spot.
- **Consequence:** A wrong causal claim nearly banked as a two-registration
  stopping statement; corrected to "the cadence-and-threshold search retires,
  the hypothesis does not."
- **Rule (draft):** Before blaming the substrate, enumerate the operator's own
  fixed choices and check whether one of them produces the failure by
  construction.

## 2026-08-28 — [unverified-claim] Built a pivot on a delta with a corrected numerator and an uncorrected denominator
- **What happened:** Cited v28's +0.14 over v27 as capacity evidence. The
  correction (F0 ladder) was only ever run on v28; no v27 ladder exists, both
  v27 points ever spot-corrected moved UP (+0.08, +0.21), and v27 seed 2 — the
  0.530 anchoring the delta — was never re-scored. The sign of the effect is
  unmeasured.
- **Root cause:** Quoted a comparison without checking that both sides had
  passed through the same correction, on a defect (selector under-selection)
  already known to bias every uncorrected number low.
- **Consequence:** The capacity pivot was suspended pending the v27 corrected
  ladder, with the fork registered before the numbers.
- **Rule (draft):** A delta may only be cited when both of its ends were
  measured under the same estimator; a one-sided correction is a new number,
  not a comparison.

## 2026-08-28 — [vacuous-gate] Five registrations moved the schedule around a selection statistic the treatment itself sets
- **What happened:** v27, v28, v30, v31 and v32 all searched the ReDo *delivery
  schedule* — `tau` inert, then a fixed tau, then a surgical tau, then a
  rank-based bottom-k — against a dormancy score that is, on this trunk, a
  rank-readout of the learned LayerNorm gain. Spearman(`norm2.weight`, fc2
  score) = +0.932 / +0.943 / +0.905 / +0.773 across the four v32 seeds, with
  trained gains spanning 0.477–11.454. `recycle()` sets that gain to exactly
  1.0 and zeroes the recycled unit's actor AND critic columns plus their Adam
  moments; with both head columns at zero the unit gets no gradient, so the
  gain stays pinned at 1.0. The mechanism deposits its own recycled unit at the
  bottom of the statistic that selects it, then re-selects it: 91.41% still
  rank-bottom-4 one check later over 384 unit-observations, median rank 2 of 32.
  All sixteen units recycled at the final check, across all four seeds, read
  gain exactly 1.000 and mean |actor column| exactly 0.000.
- **Root cause:** No registration ever checked that its selection statistic was
  independent of a quantity its own treatment writes. The check is a Spearman
  over one checkpoint and one banked receipt — minutes, zero compute — and it
  was never run, in five registrations, because every one of them reasoned from
  the write-ups rather than from the checkpoints.
- **Consequence:** ~21 h of training compute (v27 7.14 h + v28 7.37 h + v30
  ~0.70 h + v31 ~0.56 h + v32 5.33 h) across five campaigns, and a proposed
  stopping statement that blamed the architecture for what the operator
  explains. Found only by a commissioned adversarial review going to
  `checkpoints/` instead of `docs/`.
- **Rule (draft):** Before registering any experiment, demonstrate offline that
  the SELECTION STATISTIC is not a deterministic readout of a quantity the
  TREATMENT sets. It costs minutes. Attach the demonstration to the
  registration or the registration is inadmissible.

## 2026-08-28 — [weak-eval] The arming preflight GOed at n=44 and the campaign it licensed refuted it at n=384
- **What happened:** v32's Phase R2 ran at exactly the campaign's operating
  point (`mode=bottom_k k=4 every_iters=10`, in `phase_r2_stdout.log`) and read
  75.0% re-selection on 44 observations, which was adjudicated GO and licensed
  the full 4-seed campaign. The campaign, at identical settings, reads 91.41%
  on 384 observations. Fisher two-sided p = 0.0024.
- **Root cause:** A preflight sized for a go/no-go on a small pilot was read as
  an estimate of the quantity it gates. No minimum-n was registered for the
  Phase R statistic, and no confidence interval was reported beside the point
  estimate, so a swing from 90.9% to 75.0% across 12 checks read as improvement
  rather than as noise.
- **Consequence:** 5.33 h of training compute launched on a preflight number
  the campaign then contradicted at p = 0.0024, ending VOID-UNDERPOWERED with
  no Theta.
- **Rule (draft):** A preflight that gates compute must report an interval, not
  a point, and must declare its minimum n before it runs. If the gating
  statistic's interval spans the no-go threshold, that is a NO-GO.

## 2026-08-28 — [process] A mechanism receipt registered "in every branch, including STOP" was never computed
- **What happened:** §7 of the v32 registration requires the recovery curve as
  a mechanism receipt in every branch, and §10.4 makes it the one thing a VOID
  does license. Both pilots wrote one (`phase_r/recovery_curve.json`,
  `phase_r2/recovery_curve.json`). The full campaign — 4 seeds, 8× the
  observations of both pilots combined — wrote none, and the campaign write-up
  proposed a stopping statement about recovery without it.
- **Root cause:** The receipt was produced by the pilot harness as a side
  effect, never by the campaign runner, and nothing checked for its presence at
  adjudication. A registered deliverable with no automated existence check is a
  deliverable only when someone remembers.
- **Consequence:** The one measurement that could adjudicate the campaign's own
  stopping statement sat uncomputed on disk while the statement was drafted
  from the pilots. Computing it (offline, seconds) refuted the statement's
  central clause.
- **Rule (draft):** Every artifact a registration names as owed "in every
  branch" gets an existence assertion in the adjudicator, and adjudication
  fails on its absence. A receipt that only the happy path writes is not
  registered, it is hoped for.


## 2026-08-28 — [vacuous-gate] Called an adjudicator without its mode flag, silently ran the wrong check
- **What happened:** The v32 campaign runner called `redo_arm_gate.py` on all
  four seeds with no flags. `--bottom-k` is opt-in; without it the script
  silently falls through to the threshold-based adjudicator, which checks
  `tau` against a default of 0.25 -- a parameter the registration explicitly
  documents as NOT READ under bottom-k mode. All four seeds reported VOID
  on a check that could never have been satisfied, for a reason unrelated to
  anything the campaign actually measured.
- **Root cause:** Wrote the invocation from memory of what the tool should do
  rather than reading its own CLI surface; a tool built to serve two modes
  silently defaults to the wrong one for the mode actually in use.
- **Consequence:** A correctly-run campaign initially reported four false
  VOIDs. Caught by reading the registration text ("not read on this path")
  against the arm-gate's actual complaint before accepting the verdict.
- **Rule (draft):** Before trusting an adjudicator's verdict, confirm it was
  invoked in the mode the artifact under test actually used, not the tool's
  default.

## 2026-08-28 — [stale-artifact] An escalation ladder moved k and C but not the floor derived from them
- **What happened:** B1's event floor (`>= 48`, `cum_recycled == 2 x events`)
  was written for rung 1 (k=2, C=5: 50 checks over 250 iters, 48 = 50-2
  slack). When the registered ladder escalated to rung 2 (k=4, C=10: 25
  checks is the structural maximum), the floor was never updated --
  `>= 48` became arithmetically unreachable, so any rung-2 run would
  VOID-NOT-REACHED regardless of mechanism behavior.
- **Root cause:** A derived numeral (the floor) was hand-copied at
  registration time from the rung it was first written against, instead of
  being expressed as a function of the numerals it derives from (checks per
  seed, minus a fixed slack count) so an escalation propagates automatically.
- **Consequence:** Would have VOIDed a healthy campaign on a defect in the
  adjudicator, not the mechanism. Caught before any verdict was accepted;
  corrected via dated addendum (23 = 25 - 2, same slack magnitude as rung 1)
  in the registration itself, disclosed as written after seeing the raw
  counts. A companion note already existed in CLAIMS.md proposing 24 (one
  event of slack) for the same fix, written before the campaign ran -- the
  two numbers disagree by one and neither changes any of the four verdicts,
  since all four seeds hit exactly 25/25 events.
- **Rule (draft):** A derived threshold should be computed from its inputs at
  adjudication time, not hand-copied at registration time -- an escalation
  ladder that moves the inputs and not the derived value is the same defect
  as a config key nobody re-checks after the code around it changes.

## 2026-08-28 — [vacuous-gate] Wrote a test that called the fix's own primitive instead of the code under test
- **What happened:** First regression test for the orphaned-child fix called
  `os.killpg` directly to kill a session leader, then asserted its child died.
  It always used killpg regardless of what `engine_driver.reap()` actually did
  -- it never exercised `reap()` at all.
- **Root cause:** Wrote a test that demonstrates the KERNEL semantics
  (killpg kills a process group) instead of a test that exercises the
  function under test and checks its effect.
- **Consequence:** Caught before commit by the standing revert-verify habit:
  reverted the fix in `engine_driver.py`, re-ran the test, it still passed.
  Rewritten to call `ed.reap()` itself; re-verified it then fails on revert.
- **Rule (draft):** A regression test must call the changed function, not
  reimplement its intended effect beside it.

## 2026-08-28 — [vacuous-gate] Two tests pinned a bug as the expected contract
- **What happened:** `test_trainer_wires_the_monotone_rule` asserted the exact
  buggy source line (`trunc_buf=(trunc_buf if wave_monotone else None)`) as
  required text. `test_monotone_lost_cut_truncates_in_real_loop` asserted
  `tb is None` -- identity on an implementation detail -- when the intended,
  behavioral contract was "no truncation was flagged" (`not tb.any()`, which
  `None` and an all-False array satisfy identically downstream).
- **Root cause:** Both tests encoded a specific IMPLEMENTATION (a literal
  string; a specific object identity) as the assertion, rather than the
  BEHAVIOR the implementation was supposed to produce.
- **Consequence:** Both passed for as long as the bug existed, then failed the
  moment the bug was fixed (`e9dafa8`) -- a full suite run caught it,
  targeted file runs during development would not have. Fixed in `5f699f4`,
  re-verified the rewritten assertions still fail if the mechanism they
  actually care about regresses.
- **Rule (draft):** When a test's assertion is a literal string or an object
  identity, ask what behavior it stands in for; assert the behavior.

## 2026-08-28 — [process] Fixes from an external audit went uncommitted to MISTAKES.md
- **What happened:** Three findings above, all from the same external-audit
  response session, were fixed and committed without a MISTAKES.md entry
  until asked directly whether they were logged.
- **Root cause:** Treated "fix it, test it, commit it" as the complete loop
  and stopped there; logging was a separate step that had no forcing function
  once the audit thread itself was closed out.
- **Consequence:** None on the fixes' correctness; the log undercounted for
  one full exchange, and the pattern (two vacuous-gate instances in one
  sitting) went unrecorded until a direct question surfaced it.
- **Rule (draft):** Log to MISTAKES.md as part of the fix commit, not as a
  followup -- a fix and its lesson are one unit of work, not two.

## 2026-08-28 — [weak-eval] A positive control that could not pass
- **What happened:** The on-policy `V_adv` arc requires its positive control
  `PC_B5` to read LIVE at an iterate before any signature is declarable
  (gate A2). On the repaired, physically sound banks it read **COLLAPSED at
  26 of 26 iterates**, |A| = 0, and the registration's §11 retirement fired.
  The published reading attributes that to critic training exposure — the
  critic trained on `WALL` and never on the PC rungs — and that reading is
  well supported. It is not the whole mechanism. Computed over
  `arc_scored.json`: `WALL` carries a median **41,346 rows in 197 action
  cells**, `PC_B5` **3,106 in 47** — a **13.3×** asymmetry nothing in the
  registration controls. A permutation null shrinks with n, so the two
  regions were graded against different bars: median q97.5 **0.026** for
  `WALL` against **0.059** for `PC_B5`, on effect sizes that broadly overlap
  (η² medians 0.0689 vs 0.0420, within 0.02 at 10 of 26 iterates).
  **`PC_B5`'s own η² would clear the LIVE bar at 24 of 26 iterates if scored
  against `WALL`'s null.**
- **Root cause:** **The registration checked every threshold's acting range
  and never checked the control's power.** The asymmetry is not incidental —
  it is baked into the rollout protocol the same registration specifies: 40
  `WALL_SRC` episodes per iterate, all truncated at a median 1,041 steps
  while pinned at gx 2674, against 24 `PC_SRC` episodes ending in a median
  190 steps because they travel and clear. A region that goes nowhere
  accumulates rows; a region that succeeds does not. That was knowable from
  the protocol before a single checkpoint was loaded.
- **Consequence:** The instrument retired on a run that was sound in every
  physical respect, and the retirement is correct — but for a reason one step
  deeper than the write-up gives. It matters because it changes what the
  retirement forecloses: a successor that only gave the critic exposure on
  the PC rungs would still grade the control against a null built from a
  seventh of the data, and would fail the same way. Nine vacuous gates were
  gates that could not fail; this is the first recorded gate that could not
  **pass**.
- **Rule (draft):** **Power the control, not just the treatment.** Before
  scoring, emit rows and cells per contrasted region and refuse to grade two
  regions against separately-estimated nulls when their n differs by more
  than a registered factor; equalize by subsampling, by matching cells, or by
  pooling the null across regions. A control graded on a seventh of the data
  is not a control, in the same way that a control whose row set equals the
  positive control's is not one.

## 2026-08-28 — [false-alarm] A guard asserted a trajectory invariant on a filtered table
- **What happened:** `assert_bank_wellformed`, shipped in `e89091e` to catch
  the aliasing defect that voided the previous run, asserts the chain
  property — the successor recorded at row `i` IS the antecedent recorded at
  row `i+1`. On its first live run it **raised `CHAIN BROKEN` at iter 30 on a
  perfectly good bank.** A `PC_SRC` rung-1013 episode traverses the WALL
  band, and the registration's own cross-population drop removes those
  interior rows, leaving a legitimate mid-episode gap between two adjacent
  *recorded* rows. Fifteen of 26 iterates carry such drops, so most of the
  grid would have tripped.
- **Root cause:** **The invariant was true of the trajectory and was applied
  to the artifact, after the pipeline had filtered it.** Row adjacency in the
  written table and step adjacency in the rollout are different relations,
  and the guard was written as if they were the same. The acting range of the
  check was never exercised against output the *registered* pipeline actually
  produces — only against synthetic ideal banks in unit tests, where nothing
  is ever dropped.
- **Consequence:** Caught at iter 30 of 260 by the guard's own first live
  run, so the cost was a partial collection and a recollect rather than a
  false verdict; the grid was recollected under the fix. Fix (`5d0e016`):
  `row_step` is recorded into the bank so the gap is visible in the artifact
  itself, and the chain is asserted only across step-adjacent pairs — full
  strength against the aliasing class, which corrupts every adjacent pair,
  with zero tolerance added. Revert-verified at 6 of 51 tests failing,
  including the exact iter-30 reproduction. Had it not tripped early it would
  have aborted a 0.86 h collection near its end.
- **Rule (draft):** **Run a new guard once on a known-good artifact from the
  real pipeline before arming it on a grid.** Unit tests establish that a
  check fires on the defect; only a live pass establishes that it does not
  fire on legitimate output. The standing rule — every threshold checked
  against its acting range on the data it will see — has two directions, and
  this log had only ever recorded one of them.

## 2026-08-28 — [process] Five configs shipped without running the suite that guards them
- **What happened:** The v32 seed-1/2/3 profiles and both Phase R profiles
  declare `reward_id: mario` and were never appended to the frozen
  reward-dispatch roster, so each tripped
  `test_roster_dispatch_matches_the_frozen_pre_change_baseline`'s "profile
  gained a specialized reward it never had" assertion. They were committed
  anyway. The full suite then returned **two** failures against a baseline
  that permits exactly one known-environmental failure — a red suite standing
  on `main` across two commits until the next session ran it.
- **Root cause:** **A session shipped config files and re-ran only the tests
  it had just written.** The roster test exists precisely to catch a new
  profile silently acquiring a specialized reward; it is cheap, it is in the
  default suite, and it was not run. Same shape as the 2026-08-27
  "subset test run called no regression" entry, one artifact class over —
  there a shared hot method, here a frozen roster.
- **Consequence:** No result was corrupted (the roster test guards dispatch,
  and dispatch was correct for these profiles) but the project's single
  integrity gate was red on `main`, which is the state in which a *real*
  regression is invisible. Fixed as the test's own documented policy
  prescribes: five appended rows carrying a batch comment naming the
  authorizing registration, expected count moved 126+4+1 → 126+4+1+5, and the
  **frozen rows not regenerated** — which that test correctly treats as
  unforgivable. Revert-verified: deleting one appended row fails two tests.
- **Rule (draft):** **A config file is code.** Adding or copying a profile
  runs the full suite before the commit, not the subset that covers the
  feature the profile was written for — the tests that break on a new profile
  are by construction the ones nobody was thinking about.

## 2026-08-28 — [stale-artifact] Every gate passed on a bank whose rows were not transitions
- **What happened:** The on-policy `V_adv` re-score
  (`docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md`, verdict
  `docs/research/TWO_REGISTERED_TESTS_2026-08-27.md` Part I) collected 26
  transition banks — ~1.6 M rows, 0.86 h — in which **`state` was bit-identical
  to `next_state` on 100 % of rows.** The antecedent was never recorded.
  `TileFeatureStacker._flatten_oldest_to_newest`
  (`src/emulation/frame_utils.py:190-204`) returns its own reused `_out` buffer
  as a BC-pretrain optimisation; the collector held `lane.obs` as an alias of it
  and then called `push()`, which mutates that buffer in place and returns the
  same object, so `lane.obs is s_new` and both `.copy()` calls captured the
  **successor**. The registered estimator `Â = r + γV(s') − V(s)` therefore
  collapsed to `(γ−1)·V(s')` — a function of ONE state carrying zero action
  information by construction (`WALL` raw 3.93e-09 against
  `Var_batch[V] = 332.73`). Cross-check that settles it: `PC_SRC` episodes
  clear the level, spanning `gx` 0 → 3266, yet `gx(s) == gx(s')` on 100 % of
  22,845 PC rows.
- **Root cause:** **Every check ran on the values the loop held in hand; none
  ran on the bytes written to the file.** All eight admissibility gates passed
  and were individually correct — A3's zeroed-critic stub returns 0.0 whatever
  the successors are, A5's `Var_batch[V]` is computed on states (which were
  valid), A7 injects an effect four orders of magnitude above the real signal
  and detects it, A8 compares widths. Sharpest of all, **the level-identity
  purity guard ran on the true antecedents and passed with 0 violations across
  ~1.6 M transitions — it certified transitions that were never the ones
  written to disk.** Guard subject and artifact content diverged. The four
  revert-verified anti-vacuity tests all pass *distinct* arrays into
  `append_transition_row`, so none exercised the aliasing: the unit was correct
  and the integration was not. Same family as the 2026-04-22 stale-`.so` entry,
  one level up — there the loaded artifact was not the built one; here the
  written artifact was not the collected one.
- **Consequence:** A 2.13 h job of which ~1.27 h of scoring measured nothing,
  and — worse — a **result that looked complete**: 26 iterates, |A| = 24
  admissible, a published R curve, an arc verdict. Had the adjudication trusted
  the report, `V_adv` would have been **retired** under the registration's own
  §11 rule on a null a bug manufactured, and B5 would have been recorded as
  un-adjudicable by an instrument that never ran. What survives is only what
  was read off `s_new` directly and is uncorrupted: the no-penetration
  measurement (1,040 rung-893 episodes, `gx` never above 2674) and the rung-933
  probe.
- **Rule (draft):** **Assert on the artifact, not on the loop.** A transition
  bank ships a chain invariant — the successor recorded at step `i` IS the
  antecedent recorded at step `i+1`, checked on the arrays about to be written.
  It is exact, threshold-free, one pass, and it is now enforced in
  `assert_bank_wellformed` (`scripts/collect_onpolicy_bank.py`), revert-verified
  at 4 of 45 tests failing when neutered. Corollary for any buffer-reusing
  producer: a function documented as returning a reused buffer must have every
  consumer's copy made load-bearing in a comment, or the copy gets optimised
  away by the next reader.

## 2026-08-28 — [vacuous-gate] The negative control was a byte-copy of the positive control
- **What happened:** `NC-b` (`NEG_gx_frozen`) selects cells within `PC_B5`
  where no tried action moves `gx`. Because the aliasing defect above made
  `gx(s) == gx(s')` universally, the frozen mask was always true and the
  `moving` exclusion set was empty, so **`NEG_gx_frozen` returned
  bit-identical to `PC_B5` at 26 of 26 iterates** — same `n_cells`, same
  `n_rows`, same `raw`, same `η²`, same null median, same q97.5, every digit.
  The run report cited the single iterate where the cap fired as evidence the
  control "did real, falsifiable work here," reading the bug's own signature as
  the safety mechanism working.
- **Root cause:** **A control's acting range was never checked against the data
  it would actually see.** The standing rule from the previous seven vacuous
  gates is exactly that, and it was applied to every *threshold* in this
  registration and to no *region definition*. Structurally the damage was total
  and pre-compute: `NEG == PC_B5` implies (`NEG` LIVE ⟺ `PC_B5` LIVE); both
  registered signatures require `PC_B5` LIVE; the driver caps to INDETERMINATE
  whenever `NEG` is not COLLAPSED and a signature would otherwise be declared —
  so **no signature was declarable at any iterate before a single checkpoint was
  loaded**, and the reported `frac_mis = frac_cap = 0.00` was an arithmetic
  identity of the pipeline rather than a reading.
- **Consequence:** The ninth vacuous gate, the second in two days found inside a
  registration whose explicit brief was preventing the previous ones — and the
  first to sit directly on the primary verdict path rather than beside it. It
  did not merely fail to catch something; it made the intended verdict
  unreachable while printing a plausible one.
- **Rule (draft):** **A negative control must be shown to differ from the
  positive control on the actual rows, before either is scored.** Emit
  `|rows(NEG) Δ rows(PC)|` on every reading and VOID when the symmetric
  difference is empty — a control whose row set equals the positive control's
  is not a control. Re-derive it on the repaired bank *before* scoring: if
  `NEG_gx_frozen` is again a near-copy on real transitions, it is unusable on
  on-policy data and must be re-specified in a written addendum, not left in
  place to cap the verdict.

## 2026-08-28 — [process] An automation wrote a registered NO-GO from a missing directory
- **What happened:** The v31 campaign orchestrator redirected training stdout
  with `> checkpoints/mario_1_1_v31_redo_seed0/run.log` but did not create that
  directory first — its `mkdir` came *after* the training call. The shell
  redirect failed, `train_game.py` never started, the Phase M adjudicator
  crashed on a missing file, and the script wrote a **"Phase M NO-GO"** status
  marker. The marker and the launch are timestamped in the **same second**
  (22:27:49). Fixing it surfaced a second defect on the same path: the trainer
  attaches its own `FileHandler` to `<checkpoint_dir>/run.log` in **truncating**
  mode (`mode="w"`), so a shell redirect to that same path produces two writers
  at independent offsets and corrupts the very log Phase M is adjudicated from —
  observed interleaving live, killed that run, redirected to a separate path.
- **Root cause:** **A registered decision was delegated to a script that could
  not distinguish "the measurement returned NO-GO" from "the measurement did not
  run."** Every non-zero exit looked the same to it. The registration was
  careful about the opposite direction — it deliberately does *not* let the
  script pick a ladder rung unattended — but nothing stopped it from
  manufacturing the input to that decision.
- **Consequence:** Caught before adjudication, so nothing was published; had it
  not been, a ladder rung would have been taken on an infrastructure error and
  the campaign's registered verdict would have rested on a missing directory.
  Cost was minutes. The near-miss is the entry.
- **Rule (draft):** **An orchestrator may only record a verdict it can prove was
  measured.** Require the receipt (a log with the expected `[redo] ENABLED
  tau=` line, a non-empty telemetry series) before writing any PASS/FAIL/NO-GO
  marker; on a missing or unparseable receipt write `INFRASTRUCTURE-ERROR`,
  which is not a verdict and takes no branch. And never point a shell redirect
  at a path the program under test opens itself.

## 2026-08-27 — [inert-treatment] An instrument adopted in a pre-registration and never built leaves the same receipt as one that ran and found nothing
- **What happened:** `v15_d1` diagnosed the confound behind the 1-2 backward
  curriculum — *the advance gate is reachability-based while the reward is
  progress-based, so a progress-flat bottleneck is a mis-specification, not a
  capability wall* (verdicts file line 13) — and **adopted the instrument that
  separates the two**: `V_adv = E_s[Var_a(Â)]`, the state-conditioned variance
  of the advantage across actions (ADOPT bullet, line 382). **It was never
  implemented.** Parameter drift was substituted in as the fifth instrument,
  which does not answer that question. B5 then CLOSED under the once-rule on
  `trailing 0/30, entrance 0/717` — a symptom **both** live hypotheses predict —
  and **`THIS IS A REAL CAPABILITY WALL at gx ~2674-2872`
  (`docs/research/B5_PREREG_2026-08-08.md:420`, section RUN 3 FINAL VERDICT —
  `:414` before today's addendum shifted the file) **stood un-retracted for 17
  days**,
  with the rung-relative wavefront amendment deferred behind it since 2026-08-11
  and explicitly gated on a written addendum re-opening B5 that never came
  (verdicts file line 525). Built and run 2026-08-27 (`f8e3536`, registration
  `docs/proposals/VADV_PREREG_2026-08-27.md`, report
  `docs/research/VADV_B5_2026-08-27.md`): admissible instrument, controls
  separated, **verdict VOID** — `R = 0.279` inside the pre-declared indeterminate
  band. The verdict is neither re-opened nor corroborated.
- **Root cause:** **A registration records what an instrument is supposed to do;
  only a receipt records what it did — and an absent receipt reads as a silent
  pass.** Nothing in the chain distinguished "adopted and never written" from
  "written, run, returned nothing," so the once-rule closed a question the
  adopted discriminator had never been pointed at. This is the same defect the
  derived graduation table fixed one level up (*enforcement must be DERIVED from
  the declaration, never listed beside it*), here applied to instruments rather
  than to tags. **Fourth instance today, spanning the whole spectrum from never
  written to written-and-mute:** (1) **ReDo** — built and armed at `tau=0.025`
  against a `>=0.25` firing threshold, never fired, so v27/v28's FAILs were never
  tests of plasticity; (2) **v22 semi-MDP GAE** (`src/training/smdp_gae.py`) —
  implemented, unit-tested, **imported by nothing but its own test**, while an
  options A/B was recorded as FAIL crediting it; (3) **the confluence
  `solve.null_rates` gate** — armed in code with **no profile supplying the
  rates**, so `DEGENERATE` can never trigger, and a Contra VOID surfaced only
  after 2.4M worker-steps; (4) **`V_adv`**, this entry.
- **Consequence:** Not a wrong verdict — an **unfalsifiable** one. A never-retracted
  capability wall taints the reading of every backward-curriculum null downstream
  of it (v27, v28, v29 and the 1-3 campaign are all backward-curriculum runs), and
  it froze a named follow-on mechanism behind a gate that could only be opened by
  an instrument nobody had written. The cost of finally answering it was **hours
  of offline scoring over checkpoints that already existed** — no emulator, no
  training. The cheapest item in the corpus was the one blocking a standing verdict.
  Sixth entry on this tag.
- **Rule (draft):** A pre-registration that ADOPTS an instrument must, at
  publication of the verdict it informs, cite **either its receipt or its
  non-execution** — "instrument X adopted, NOT RUN" is an acceptable line and a
  silent omission is not. Deterministic form: a verdict may not close under the
  once-rule while any instrument its own registration adopted lacks a receipt
  path; scan the ADOPT bullets against `runs/` and `grep -rn <symbol> src/ scripts/`
  before publishing. **And the sharper half — an adopted instrument is not built
  until something other than its own test imports it**, which is exactly what
  would have caught `smdp_gae` and the mute `null_rates` gate as well as this one.

## 2026-08-27 — [inert-treatment] The corrected operating point was wrong the same way, one rung down
- **What happened:** The entry below — *"Registered a treatment at an operating
  point its own statistic could not reach"* — diagnosed v27/v28 arming ReDo at
  `tau=0.025` against a mechanism that first fires at 0.25. The registration
  written to fix it (`docs/proposals/V30_REDO_ARMED_2026-08-27.md`) picked
  `tau=0.25` and justified it as *"the SMALLEST threshold on the sweep that
  fires at all: the minimum dose,"* explicitly accepting *"the inert-side risk,
  deliberately."* Run for 20 iterations instead of 2
  (`runs/v30_premise_falsifier_2026-08-27/`,
  `docs/research/REDO_ACTUALLY_FIRES_2026-08-27.md`), tau=0.25 settles at a
  **median 20 of 32 trunk units re-initialized every single iteration from iter
  5 onward — 62.5%.** That is the regime the same registration named at
  tau=0.50, called *"a per-iteration partial reset of two-thirds of the trunk …
  the 'network reset' family the DR ruled INCOMPATIBLE,"* and wrote **"RISK I
  REFUSE"** against. **The registered operating point landed in the registered
  forbidden regime, and the risk actually taken was the opposite of the one
  declared.** "0.25 is the smallest tau that fires" is also false: tau=0.15
  fires from iter 4, and the untreated control's fc2 score minimum falls below
  0.10 on 10 of 26 iterations.
- **Root cause:** Not the threshold — the **measurement horizon**. Both
  registrations set a threshold from a **2-iteration sweep taken near orthogonal
  initialization** and then spent a 250-iteration budget against it. The
  dormancy tail drifts monotonically downward through training (fc2 min 0.285 at
  iter 0 → 0.127 at iter 5 → 0.101 at iter 15 → 0.079 at iter 24, **still
  falling** where the measurement stops), so a *fixed* threshold is
  mis-specified by construction: too high near init is inert, and the same
  number becomes an ever-larger dose as training proceeds. The previous entry's
  rule — check reachability *at the registered value* — was honoured here and
  was not enough, because it was still checked **at iteration 0**.
- **Consequence:** Every treatment arm is **VOID, not FAIL** — a FAIL at 62% of
  the trunk reset per iteration cannot distinguish "plasticity was not the
  barrier" from "ReDo damaged a healthy network," which is the exact disease the
  registration existed to cure, arriving one layer deeper. Caught for ~42
  minutes of pilot compute instead of the registered 12 h. The plasticity-loss
  hypothesis remains **UNTESTED** — not by v27, not by v28, and not by v30.
  Fifth entry on this tag.
- **Rule (draft):** A reachability check at iteration 0 certifies iteration 0.
  Any threshold on a statistic that **drifts with training** must be swept over
  a horizon comparable to the run it governs (40–60 iterations, not 2) — or,
  better, the registered variable should be the **dose** (recycle the bottom-k
  units) with the threshold *derived* from it, which is stable under the drift
  where no fixed threshold is. Enforcement shipped:
  `_REDO_ARM_DEADLINE_ITERS = 25` raises `RuntimeError` in
  `src/training/trainer.py` when an armed run reaches iter 25 with
  `cum_recycled == 0` — verified by running the inert case, not by reasoning
  about it (the full 60-env v27 recipe at tau=0.025 aborted at iter 26 with zero
  verdict/eval/`clear_rate` lines in its log) — and `scripts/redo_arm_gate.py`
  exits 2 with VOID on all 8 banked v27/v28 logs, structurally incapable of
  printing PASS or FAIL for an unarmed seed.

## 2026-08-27 — [vacuous-gate] An identity check the mechanism satisfies by construction
- **What happened:** Abort A4 of the v30 registration voids the arm if the
  median greedy-argmax agreement over the first 50 recycle events falls below
  **0.60**, on the reasoning that a low-agreement step is *"a partial reset, not
  a surgical intervention."* Measured over 20 iterations: **0.856** at width 64,
  **0.901** at width 96, **0.950** at tau=0.15 — every arm passing comfortably
  **while 38–62% of the trunk was re-initialized every iteration.**
- **Root cause:** `recycle()` **zeroes the outgoing actor and critic weight
  columns by construction**, so the network's output is approximately preserved
  *however many hidden units were destroyed*. Agreement is structurally
  insensitive to the dose. The gate measured a quantity the mechanism is
  designed to hold constant, so it could not fail on the failure mode it names,
  at any dose. The eighth vacuous gate on this ledger — written into a
  registration whose entire brief was the previous seven, by work that quoted
  them.
- **Consequence:** Had the pilot not run, A4 would have certified a
  62%-of-trunk-per-iteration partial network reset as a "surgical intervention"
  and the 12 h campaign would have produced an uninterpretable FAIL that read as
  evidence against the plasticity-loss hypothesis.
- **Rule (draft):** Before registering a check, ask what the mechanism
  *guarantees* about the quantity being checked. If the mechanism preserves it
  by construction, the check is decorative — measure the thing the mechanism
  does **not** control instead. Closed by V6 in `scripts/redo_arm_gate.py`:
  median recycled fraction **of the worst-hit layer** ≤ 0.25, which VOIDs all
  three treatment arms. Revert-verified, executed not asserted
  (`tests/test_redo_armed_gate.py`, 22 tests): delete the `cum_recycled == 0`
  branch → 10/22 fail; delete V6 → 3/22 fail; **pool fc1+fc2 instead of taking
  the worst-hit layer → 5/22 fail** — that last being the vacuity V6's own first
  draft contained, reporting 20/96 = 21% *passing* for an event that reset 20 of
  32 trunk units, because fc1 never goes dormant and a pooled denominator is
  permanent ballast that can never contribute to the numerator.

## 2026-08-27 — [inert-treatment] A mechanism can pass every arming check by never being imported
- **What happened:** The DR never-executed audit
  (`docs/research/DR_NEVER_EXECUTED_2026-08-27.md`) traced 51 prescriptions across
  10 Deep Research rounds to code. `src/training/smdp_gae.py` implements v22's
  semi-MDP advantage estimator exactly as prescribed, is covered by
  `tests/test_smdp_gae.py`, and passes. **The only file in the repo that imports
  it is that test.** `ppo_updater.py:153` calls `batched_gae` with an identical
  argument list whether `commitment_options` is on or off.
  `OPTIONS_PREREG_2026-08-22.md` lists the mechanic as adopted;
  `ZELDA_VISION_AGENT_AUDIT_2026-08-25.md:166` names `smdp_gae.py` as part of what
  "FAILED its gate." Two of the same registration's four adopted mechanics were
  likewise absent: the dense-critic auxiliary pass, and all three forms of the
  eval-argmax overcommitment mitigation whose stand-in the registration asserted
  was "already standing in the campaign machinery."
- **Root cause:** Every signal we check for aliveness is emitted *by the mechanism
  itself*. `scripts/check_mechanism_receipt.py` VOIDs an armed mechanism whose
  counter never moves — but a module nothing imports has no counter to move, emits
  no telemetry, and therefore cannot fail an aliveness check. A green unit test, a
  written docstring and a name in an adopted-mechanics list are each read as
  evidence the mechanic shipped, and none of the three touches the import graph.
- **Consequence:** The standing OPTIONS MECHANISM FAIL (control 8/100 vs treatment
  0/100) survives as a number but not as a scope — it tested fixed-duration,
  open-loop, *unprotected* options under a per-step estimator and an untrained
  held-state critic, not temporal abstraction. At λ<1 the substituted estimator
  weights each held-state critic value into the advantage by `γ(1−λ)(γλ)^(i−1)`,
  zero in the semi-MDP form and non-zero only for k≥2 — a duration-dependent bias
  in the exact quantity a run that failed by duration overcommitment (k=4 in 93.6%
  of states) adjudicated. A downstream line (v23 Castlevania options) inherits
  that FAIL. Fourth entry on this tag.
- **Rule (draft):** A registration that names a module as adopted must cite the
  **production import path** that reaches it — file and line, not the module name.
  Extend `check_mechanism_receipt.py` to VOID any named mechanism reachable only
  from `tests/`. Aliveness proven by the mechanism's own output cannot detect a
  mechanism that produces none.

## 2026-08-27 — [stale-artifact] The test suite wrote verdicts into the production receipt log
- **What happened:** `runs/shared_substrate/eval_shared_substrate.jsonl` — the
  canonical receipt for the trunk-plus-heads experiment — holds 860 rows. 688 carry
  `/fake/dir/` fixture paths. **The other 172 are repeated
  `{"verdict": "SUPERSEDES", "aggregate": {"baseline_sum": 153, "shared_sum": 200,
  "delta": 47, "beats_baseline": true}}` records**, written by
  `tests/test_eval_shared_substrate.py` through the module-level default at
  `scripts/eval_shared_substrate.py:121`, which points at the real log. The
  experiment has never trained: `manifest.json` reads `"status": "pending"` and
  `checkpoints/shared_substrate_v1` does not exist. The rows accumulate on every
  `pytest` run — 11 on 08-18, 48 on 08-26, 62 on 08-27.
- **Root cause:** Several tests correctly redirect `receipt_log` to `tmp_path`; at
  least one path falls through to the module default. A default that points at a
  live artifact turns an un-monkeypatched test into a writer of production
  receipts, and the failure grows monotonically instead of announcing itself.
- **Consequence:** Compounded with `docs/proposals/README.md` §10 — committed
  2026-08-25, two days after `PROCESS_AUDIT_2026-08-23.md:122` recorded the
  experiment as "distinct and unscheduled" — marking the round
  `COMPLETED/ACTIONED` and the ranking "shipped as commit `740cac6`", an auditor
  who checks the status index and the receipts concludes the experiment ran and
  won. Only the manifest `status` field and that one audit dissent. No claim was
  made from it, so nothing is retracted; the near miss is the finding.
- **Rule (draft):** A harness's default output path must never be a live receipt.
  Default to a run-scoped temp path and require the real log to be passed
  explicitly, so a test that forgets to redirect writes nowhere that matters.

## 2026-08-27 — [purity-leak] The quarantine covered the declarative layer only
- **What happened:** The 994-entry `configs/` sweep retracted 7 entries and stated
  its own scope limit in its commit message: "Quarantining the YAML retracts the
  DOCUMENTATION claim, NOT the Rust constant." A third sweep took that sentence as
  the specification and swept the executing layer: **134 RAM-address constants
  across `nes_core/src/` plus 23 non-address constants carrying semantics.** 21
  findings were ruled SEMANTIC-and-UNWITNESSED, covering **27 constants across 11
  games**, all now annotated. Reward arithmetic changed: **0**. For **Kid Icarus
  (`$0130`)** and **Double Dragon (`$0030`)** the sentence the YAML had retracted
  was found alive *verbatim* in `rewards.rs`.
- **Root cause:** A claim written in one layer and executed in another. Retracting
  it in the declarative layer feels like retracting it, produces a clean diff, and
  leaves behaviour untouched — and the layer that got corrected is also the layer
  everyone reads, so the failure is silent by construction.
- **The retraction made it worse before it made it better,** which is the part
  worth carrying forward. Before the config sweep both layers carried the same
  wrong claim: consistent, and discoverable by reading either one. After it they
  disagreed, and the authoritative-looking half was the wrong one. **A partial
  retraction is not a partial fix; it is a new defect class.**
- **Consequence: none banked.** All 27 are unfired — not one sits under a quoted
  number, all belong to games with no witnessed clear, and no boss defeat has ever
  been witnessed on any game here. SMB is byte-identical (no existing executable
  line in `rewards.rs` was removed or modified) and is now positively marked
  `PURITY: WITNESSED`. That the engine came back mostly clean is evidence the
  first two sweeps worked, not a wasted pass.
- **Second-order finding, and the reason this entry closes the class:** the guards
  written that morning to enforce the retraction were themselves vacuous when
  reverted — 19 of 24 constants survived deletion of their own tag because the
  check searched a 60-line window and accepted a *neighbour's* tag. Two more had
  the same shape (a 50-line lookback; a naming convention standing in for the
  address). **A guard that locates its evidence by proximity can be satisfied by a
  neighbour.** All three now scope to the artifact they guard, and all 27
  single-tag deletions are caught.
- **Rule — mechanical, not written:** `make purity-check` derives quarantined
  addresses from the `quarantined_external_knowledge:` blocks themselves and
  ownership from the source's own dispatch table, so neither can drift from what
  it guards; `WIN_WITNESS_LEDGER` classifies all 17 reward arms and five Rust
  tests drive each one through the byte its row names. The written form of this
  rule ("retract in every layer, not just the declarative one") is exactly the
  kind of instruction that has failed seven times here. Enforce it or do not
  claim it.

## 2026-08-27 — [purity-leak] Unwitnessed events annotated as measured fact
- **What happened:** A tree-wide sweep of all 101 configs carrying a live
  `ram_mapping` (994 int-parseable entries) found 17 annotations asserting what a
  byte does at a clear, a win, or a boss death on a ROM where this repo has never
  witnessed one — plus 2 more outside the inline-comment scan. `contra.yaml`:
  "increments on a real stage clear" and "0->1 when the current stage boss dies",
  on a game with zero witnessed stage transitions across 6 archives and a boss
  reading already falsified as a multiplexing artifact. `punchout.yaml`:
  "VERIFIED WIN LATCH ... nonzero ONLY at the winning KO/TKO. THE win" — on a
  profile whose only go_explore archive is empty and whose own receipt calls the
  claim "a purity leak dressed as an empirical find". 7 entries quarantined, 24
  downgraded, 963 left alone.
- **Root cause:** The certification tag was written from the address table rather
  than from an observation. "VERIFIED" recorded that somebody believed the label,
  not that anybody watched the byte move. Nothing distinguishes, at the point of
  writing, a byte watched across an event from a byte whose event never happened —
  so an unfalsified claim reads identically to a confirmed one.
- **Consequence:** Documentation-only in the config layer: no `solve.level_key`
  and no `reward_weights.*_addr` referenced a quarantined address, so no banked
  clear, gate, or reward is retracted. But two of the seven are still live in
  `nes_core/src/rewards.rs` as hardcoded constants (`RAM_MATCH_ID = 0x0001`
  drives Punch-Out's `episode_success()`; `RAM_BOSS_HEALTH = 0x06C1` drives Mega
  Man's boss term). Neither has ever fired, so nothing rests on them today — but
  the sweep covered `configs/` and the same class of claim lives uncovered in the
  executing layer.
- **NOT a recurrence of the exclusion-set inversion.** Worth recording precisely,
  because the brief that launched this work asserted it was. The
  `excluded |= known` fold in `scripts/observatory.py` closed for Zelda in
  `222ae25` is **still closed**: `_mapping_bytes()` is private, reaches `main()`
  only behind an `is_known()` predicate that cannot be unioned into anything, and
  the receipt logs the `ram_mapping` region with `"excludes": False`. No config in
  this tree was steering the discovery instrument. What recurred is the root cause
  one level up — external, unwitnessed semantics living in a live `ram_mapping` —
  not the mechanism that made it bite. Logging the mechanism as recurrent would
  have been a fabricated recurrence.
- **Rule (draft):** A semantic claim tied to an event names the observation that
  witnessed it, or it is written as a hypothesis. "Could not be driven to
  increment, so the increment is unverified" is the correct form; "it can only
  rise on a real floor clear" is not, because a false-positive rate cannot be
  derived from zero observations. Unfalsified is not verified.

## 2026-08-27 — [inert-treatment] Registered a treatment at an operating point its own statistic could not reach
- **What happened:** ReDo was registered at `tau=0.025` as one of two variables in
  v27's AMENDMENT 1. Its dormancy statistic normalises post-activation magnitudes
  by the layer mean, but `TilePolicyNetwork` LayerNorms immediately *before* the
  SiLU, pinning the statistic near 1 — the mechanism is calibrated on
  un-normalised ReLU nets. The repo's own forced-recycle sweep recycles **zero**
  units at every tau ≤ 0.20 and first fires at 0.25, ten times the registered
  value. `isolate_tau0.35.log` was written at 00:19:42; `train_seed0.log` opens at
  00:22:03. **The evidence was on disk 141 seconds before the 8-run budget
  started** and was read as a fresh-net artifact.
- **Root cause:** The pre-registered V7 armed-evidence gate checked only safety
  conditions, and checked all of them at `tau=0.5` — twenty times the experimental
  value. Nothing in it required the REGISTERED operating point to be reachable, so
  it could not have failed whatever tau the experiment actually used.
- **Consequence:** The seventh vacuous gate on this ledger. v27 and v28 were each
  single-variable arms and were not described as such; a registration amendment
  and eight training runs measured one variable while claiming two. Both FAIL
  verdicts survive — neither depended on ReDo acting — but the framing did not.
- **Rule (draft):** A pre-registration that names a threshold must carry a
  *reachability* condition — the registered operating point lies at or above the
  1st percentile of its own statistic on a trained net — checked at the REGISTERED
  value, at iteration 0 of the first run. Safety conditions checked at a different
  operating point certify nothing.

## 2026-08-27 — [unverified-claim] Adjudicated on the selection rule as remembered, not as written
- **What happened:** Both the v27 and v28 registrations name the selection
  statistic as "the checkpoint with the peak trailing entrance rate in the
  `[backward]` telemetry (ties → later iter)". Both adjudications instead used
  `checkpoints/*/winners/best.pt` and described it *as* that quantity. They are
  different numbers: v27 seed 0's iter-60 log line prints `trailing 16/30=0.53`
  while `winners/best.json` for the same iter records
  `entrance_trailing_rate=0.8667`, because a force-completion pass runs after the
  telemetry print and before the winner block reads the window.
- **Root cause:** The registration's selection sentence was not re-read at
  adjudication time; a familiar artifact was substituted for the named statistic.
- **Consequence:** Both headlines happen to survive the literal rule (v28 lands on
  0.670 exactly, v27 on 0.500 vs a banked 0.530), so nothing was retracted — but
  that was luck. The rule changed the selected checkpoint on 3 of 4 seeds in each
  campaign.
- **Rule (draft):** At adjudication, quote the registration's selection sentence
  verbatim into the verdict and compute from it. If a stored artifact is used
  instead, prove it equals the named quantity before calling it that.

## 2026-08-27 — [unverified-claim] Findings stayed in the document that found them
- **What happened:** Three separate cases in one track. (a) Two configs annotated
  the GA-knob inertness on 2026-08-10/11; it was never back-ported to the flagship
  config it was copied from, to the later v27/v28 experiments, or to CLAIMS.md.
  (b) `PEAK_INSTABILITY_FORENSICS_2026-08-25.md` §1.5 found that the v27 and v28
  gates ran under different `--eval-rng` modes; CLAIMS.md kept asserting they were
  "identical in every respect (… per-episode)". (c) The 1-2 "policy class
  falsified" paragraph survived the commit that banked a 38/100 result on the same
  policy class, and survived the 2026-08-26 ledger audit.
- **Root cause:** A finding was treated as delivered once its own document was
  written. Nothing required the *claim it contradicts* to be edited in the same
  commit.
- **Consequence:** The brief commissioning this very audit quoted the superseded
  falsification as standing and cited the stale 2/100 rather than the banked
  38/100 — the propagation caught in the act, one level up.
- **Rule (draft):** A finding that contradicts a live claim is not landed until
  that claim carries the annotation, in the same commit. Grep the ledger for the
  sentence you just falsified before closing the task.

## 2026-08-27 — [weak-eval] A gate threshold was never measured under the protocol it gates
- **What happened:** `0.767` was the FAIL bar for two full campaigns (v27, v28,
  eight training runs, 32 gate receipts). It is 46/60 measured at eval seed 0
  only, shared-stream, one worker — not the canonical two-seed 100-episode
  per-episode protocol it was gating. `V29_STABILITY` named the two-seed
  re-measurement as an F0 deliverable; F0 never ran it and V29 was withdrawn.
  Measured on 2026-08-27: es0 0.76, es1 0.60, pooled **0.68**.
- **Root cause:** A number from one campaign's convenience measurement was
  promoted to a threshold without re-measuring it under the protocol it would
  adjudicate.
- **Consequence:** The registered thresholds do not move — moving them now would
  be the goalpost move this ledger treats as fabrication — but the narrative
  clause both verdict docs attach ("no seed reached the banked control's own
  0.767") is unsupported for v28, whose 0.670 is statistically indistinguishable
  from a same-protocol control of 0.680. Two campaigns were described as clearly
  below a control that had never been measured beside them.
- **Rule (draft):** A threshold must be measured under the exact protocol it will
  gate, before it gates anything. A bar inherited from a different harness is a
  bar with an unknown value.

## 2026-08-27 — [stale-artifact] Resumed an archive from before the mechanism existed
- **What happened:** A Rygar run resumed `runs/rygar_campaign/R1-14/extend` with
  the new blank-run transition axis armed. **493 of that archive's 1,647 trace
  records are 7-tuples**, written before the axis added a 9th element carrying
  each lineage's occupied-area set. Every lineage restored from one therefore
  started with an empty `seen` and re-banked the first area it arrived in — the
  archive shows **9 cells at `sect >= 1` whose arriving area key is the room the
  run starts in**.
- **Root cause:** The trace record's arity is a schema, and the resume path
  compared everything else about the two archives — key arity, room-index
  alphabet, axis config — but not whether the record carried the field the new
  mechanism reads.
- **Consequence:** The novelty gate **fabricated** arrivals, which is the one
  direction it may never fail in. It also pinned that run's `max_sect` at 1 and
  made its whole transition stream non-comparable to the two cold runs it was
  meant to be read beside. Caught in adjudication by reading the banked archive,
  not by anything the run itself reported.
- **Rule (draft):** When a mechanism adds a field to a persisted record, the
  resume path must refuse records that lack it — not default it. A default is a
  guess about history, and here the guess fabricates.

## 2026-08-27 — [weak-eval] Ran three falsifier searches that banked no replayable tape
- **What happened:** Three of seven runs in the Rygar transition campaign
  (359,829 steps between them) were standalone harnesses that wrote only a
  summary JSON. Their conclusion — no fourth area from inside the frontier room
  — is the one that fires the campaign's pre-registered falsifier, and it is the
  one nobody can re-derive.
- **Root cause:** The harnesses were written to answer a question, not to leave a
  receipt, and a *null* felt like it had nothing to preserve.
- **Consequence:** Four sibling tapes were replayed at landing and all four
  reproduced their filed terminals exactly; these three could not be checked at
  all. A null with no tape cannot be distinguished from a harness that was
  silently not searching.
- **Rule (draft):** A search that reports "found nothing" must bank a tape too.
  The null is exactly the result whose harness most needs to be replayable.

## 2026-08-27 — [unverified-claim] Called a subset test run "no regression" on a shared hot method
- **What happened:** Three parallel lanes each added an attribute read to
  `Solver._refresh_sel_cache` / `observe` and each reported a green subset (546,
  641 and 85 passed). The full suite at landing came back with **7 new
  `AttributeError` failures** in `test_room_router.py`, `test_terminal_stasis.py`
  and `test_gate_k0_reforge.py` — three of the **four** test files that carry
  duck-typed `SimpleNamespace` Solver stand-ins. Only one of the four had been
  updated.
- **Root cause:** Each lane ran the tests it knew about. A bare `self.<attr>` in
  a method that four independent stand-in families call is a change whose blast
  radius is not visible from any one lane's subset.
- **Consequence:** The mechanism would have landed red. Fixed by moving to
  `getattr(self, ..., default)` — the form the sibling site in the same commit
  already used, for the same reason, one day earlier.
- **Rule (draft):** "No regression" is a claim about the whole suite. A subset
  pass count is not evidence for it, least of all on a method reached by
  duck-typed stand-ins the lane never sees.

## 2026-08-27 — [vacuous-gate] Shipped a CLI mode that printed as armed and did nothing
- **What happened:** `--lock-objective latch` landed on `main` as a declared
  argparse choice with no dispatch branch anywhere in the solver. It parsed, the
  progress line printed `lock_mode: latch` with a non-zero `lock_cells`, and it
  changed not one draw — measured at 3,000 selections byte-identical to `off`.
- **Root cause:** Four lanes built one shared `--lock-objective` flag family in
  the same working-tree file at the same time. Three implementations landed on
  `main`; the fourth (LEX-LATCH) landed on an unmerged branch — but the shared
  choices tuple that landed carried its *name*. Nobody owned the roster.
- **Consequence:** An operator running it would have read a null as "the
  objective did not help" when nothing ran. This is the seventh vacuity in a
  campaign whose own brief held the previous six, and it shipped inside the work
  that was auditing for exactly this.
- **Rule (draft):** A flag's value list is a claim that every value does
  something. Guard it behaviourally — each declared mode must measurably change
  the thing it names, and the guard must be shown to fail on a fabricated name.

## 2026-08-27 — [unverified-claim] Read a telemetry field by its name, not its definition
- **What happened:** `lock_armed_secs` was `round(now - _pin_time)` — time since
  the *frontier* last moved, which starts accruing `--lock-pin-secs` before the
  objective steers anything. Two of four campaign reports quoted it as armed
  time and overstated their runs by 2.4-2.5x (26.5 min claimed vs ~21.5 actual;
  512 s claimed vs ~212).
- **Root cause:** The field's name asserted a semantic its one-line definition
  did not have, and four readers in a row trusted the name. The sibling arm in
  the same file already called the same quantity `pinned_secs`, honestly.
- **Consequence:** Two published durations wrong by a factor of 2.4-2.5 in the
  one number that decides whether a negative means "the mechanism failed" or
  "the mechanism barely ran". Both were caught in adjudication, not in review.
- **Rule (draft):** Before quoting a telemetry field, read the line that
  computes it. If a field's name implies a gate, cross-check it in test against
  the predicate the code actually gates on.

## 2026-08-27 — [unverified-claim] A falsifier generalised its own harness defect to everyone else's receipts
- **What happened:** A commissioned falsifier found its own concatenated tapes
  died on replay (first life lost at step 73-116, gx capped at 486-1266) and
  published the headline that *every* reached-gx-3072 claim in the whole body of
  work was checkpoint continuation rather than a single unbroken life.
- **Root cause:** It verified the defect in its own tapes and then generalised
  without replaying anybody else's — the one check that would have separated
  "my bookkeeping is broken" from "the campaign's receipts are broken".
- **Consequence:** A correct local finding became a false global claim that, if
  landed, would have retroactively voided a characterisation campaign. Refuted
  by replaying 24 solver tapes (and 12 again at landing): all reach gx exactly
  3072 from the declared start state with zero life losses.
- **Rule (draft):** A defect found in your own harness is a claim about your own
  harness until you have run the same check against someone else's artifact.

## 2026-08-26 — [process] Killed a workflow that was already self-correcting
- **What happened:** Verifiers flagged a 23-profile arming commit as unsupportable.
  I stopped the workflow and started a revert, before reading what its land phase
  had already produced.
- **Root cause:** Treated verifier output as a verdict I had to act on, when it was
  input the workflow had commissioned and was mid-way through acting on. I checked
  the committed state and not the working tree.
- **Consequence:** Nearly reverted a commit whose fix was already 834 uncommitted
  lines in the tree — a reproducer script and a roster test whose first case is
  `test_the_policy_can_still_fail()`. Resumed instead; a few minutes lost.
- **Rule (draft):** Read the working tree before reverting committed work. A
  process that commissioned its own critique deserves the chance to answer it.

## 2026-08-26 — [vacuous-gate] 23 profiles armed on death evidence
- **What happened:** `scene_cut` armed across the odometer cohort. 7 profiles had
  `has_non_death_candidate: false` — every blank run observed was a death — and
  were armed anyway; 11 at `scene_min: 1`, which the signal's own docstring states
  in writing must never be used; 3 with death vetoes reading a placeholder byte, a
  documented 0↔255 flicker artifact, and a measured null.
- **Root cause:** Arming was a judgement dressed as a measurement — the receipt's
  `reason` strings read affirmatively next to fields recording the disqualifying
  numbers, and the survey script was never committed, so nothing was reproducible.
- **Consequence:** 23 profiles moved UNREACHABLE → FIREABLE on evidence that in
  most cases showed only that the game can die. This is the same defect that left
  26 games with a dead clear hook, and the brief warned against it explicitly.
- **Rule (draft):** A profile may arm a signal only above its own measured null,
  with the survey that measured it committed as a reproducer.

## 2026-08-26 — [process] Built this file twice in the wrong shape
- **What happened:** Wrote a 169-line prose `MISTAKES.md`, then rewrote it as a
  terse `mistakes.md` with a category/context/rule format, plus a pointer added
  to the enforced ruleset.
  Both were wrong; the spec arrived after each attempt.
- **Root cause:** Started producing before the format was specified, twice.
- **Consequence:** Two discarded commits; a premature rule added to the enforced
  ruleset and reverted.
- **Rule (draft):** When asked whether an artifact exists, answer first and
  confirm shape before authoring it.

## 2026-08-26 — [unverified-claim] Briefed a campaign from withdrawn numbers
- **What happened:** Launched a workflow citing Contra's "odometer 162 vs 163
  cross-validated" hours after an audit withdrew it. The 163 has no receipt
  anywhere in the tree.
- **Root cause:** Wrote the brief from recalled prior-session facts instead of
  re-checking claims recent commits had touched.
- **Consequence:** Stopped mid-flight and relaunched; ~10 min of agent work lost.
- **Rule (draft):** Re-verify any number a recent commit or audit could have moved
  before putting it in a brief.

## 2026-08-26 — [process] Overruled a skeptic I commissioned, twice
- **What happened:** A designated-opposition agent argued breadth across games no
  policy can play was motion, not progress. Overruled both times.
- **Root cause:** Treated a commissioned falsifier's verdict as an obstacle to
  route around rather than as the evidence it was commissioned to produce.
- **Consequence:** Two campaigns (~70 and ~30 agents) returned 0 confirmed
  predicates and 0 forward progress.
- **Rule (draft):** When a commissioned skeptic says stop, stop or write down the
  specific evidence that beats it. "Proceed anyway" is not a rebuttal.

## 2026-08-26 — [measurement] Reported a probe artifact as a fact about the game
- **What happened:** Stated "Rygar dies at 138 steps" and excluded Contra as
  SIGNAL UNUSABLE on "20 distinct in 69 steps."
- **Root cause:** 138 was an undodging scripted hold walking into a hazard (real
  window 3,865–4,000 actions); Contra's 69 samples were the post-truncation
  remnant of 1,131 and cannot support a 32-distinct threshold.
- **Consequence:** Excluded a viable candidate game from a campaign; corrected
  only when a later agent re-derived it.
- **Rule (draft):** Never issue a verdict on a window too small to support it;
  terminate probe holds at death before scoring.

## 2026-08-26 — [git] `git update-ref` on shared main with concurrent writers
- **What happened:** An automated lane forcibly repointed `refs/heads/main` twice
  in a checkout with ~15 concurrent writers.
- **Root cause:** Raw ref plumbing bypasses the locking that normal commits rely on.
- **Consequence:** A race dropped a sibling's class (`757a476`); repaired by
  `a66cc74`. Nothing lost — verified by fsck/reflog — but by luck.
- **Rule (draft):** Parallel lanes get receipts-only or a worktree. No raw ref
  plumbing and no `git stash` on shared `main`.

## 2026-08-26 — [unverified-claim] Ledger entries dropped their receipts' scope
- **What happened:** 15 of 39 adjudicated claims overstated their receipts.
- **Root cause:** The scoping qualifier got dropped when the headline was quoted
  forward — "Stage 1-2 validated" became "the engine validated."
- **Consequence:** 15 weakened claims; no fabrication, but the ledger read
  stronger than its evidence.
- **Rule (draft):** Quote a receipt's scope alongside its number.

## 2026-08-26 — [vacuous-gate] Observatory purity guard asserted against itself
- **What happened:** The guard for the quarantine-exclusion fix asserted against a
  copy of the fixed line pasted into its own test body.
- **Root cause:** Test verified text, not behaviour; production decision was never
  driven by a test.
- **Consequence:** Restoring the bug left all three tests green. Caught only by a
  dedicated verify pass.
- **Rule (draft):** A gate's test must drive the real decision path and fail when
  the mechanism is reverted.

## 2026-08-26 — [vacuous-gate] `progress_signal_gate` camera-static override
- **What happened:** An `rx==0/ry==0` branch deleted the "too coarse" finding and
  forced `passed=true` whenever OAM churn showed the agent moving.
- **Root cause:** "The agent moved" was accepted as evidence the odometer *can*
  report a positive.
- **Consequence:** 40 vacuous passes across 13 profiles, all reading
  `distinct=1, min=0, max=0`. Kung Fu's "skill wall" verdict rested on one.
- **Rule (draft):** Agent activity is not instrument capability. Prove the
  instrument can return a positive on that profile.

## 2026-08-26 — [vacuous-gate] Constants quoted as measurements
- **What happened:** `is_clear` opened with `level_key(ram) > tuple(start_key)`;
  with `level_key: []` that is `() > ()`, False always, on 152/155 profiles.
  `area()` returns literal `0` when unconfigured.
- **Root cause:** No check that the expression producing a zero could be non-zero.
- **Consequence:** Every banked `solutions: 0` — millions of steps, up to 443,419
  cells — was a compile-time constant read as a search result. `n_area == 1` was a
  YAML property read as a fact about the ROM.
- **Rule (draft):** Before quoting a zero, evaluate its expression on random input
  and confirm it can be non-zero.

## 2026-08-26 — [weak-eval] Detector suite tested only its own shape
- **What happened:** 186 tests green while the detector could not fire on two
  witnessed clears (Bubble Bobble round 69→70, a banked 4,329-action Tetris-B win).
- **Root cause:** Every fixture was SMB or a synthetic stream built to the
  detector's design centre. `coord` required a ≥300-unit drop against observables
  spanning 1 and 32 units — arithmetically impossible, never tested.
- **Consequence:** 41 profiles carried nulls that measured nothing; Gradius's
  clear hook was silently dead for 18 days.
- **Rule (draft):** At least one fixture must be a real positive from outside the
  instrument's design centre.

## 2026-08-26 — [purity-leak] Reward dispatch on an incidental string
- **What happened:** `build_reward` selected on `name.contains("zelda"|"mario")`.
- **Root cause:** Dispatch keyed on a display name rather than a declared field.
- **Consequence:** Failed both directions — a text-clean profile silently
  inherited a quarantined disassembly-sourced win predicate; `smb_4_4_micro`
  declared nine Mario-only weights and got none of them.
- **Rule (draft):** Route on an explicit declared key with a safe default; remove
  the incidental string from the dispatch path entirely.

## 2026-08-26 — [inert-treatment] Mechanisms unguarded, signals wired to nothing
- **What happened:** `_dead_mm` (death-blip debounce): 3 occurrences in source, 0
  in tests. Separately, six detector signals were built while the live vote
  remained `tally + coord`.
- **Root cause:** No check that a mechanism has a test, or that a finished
  mechanism has a caller.
- **Consequence:** Correct code nothing would notice breaking; six signals reaching
  no production path.
- **Rule (draft):** Grep the mechanism's identifier in `tests/` (zero = defect) and
  confirm the production path reaches it.

## 2026-08-26 — [weak-eval] Checkpoint selection and scoring on the same data
- **What happened:** Selector used `argmax entrance_trailing_rate`, a
  ceiling-saturated metric (0.867–1.000, SE ~0.09 on 30 episodes).
- **Root cause:** Selecting and scoring on one sample.
- **Consequence:** Under-selected 20–40 iters on 4/4 runs; recorded peaks low by
  +0.08…+0.21. Correcting it then invited winner's curse — measured at 0.05.
- **Rule (draft):** Fix the estimator in advance; split-sample — select on one
  seed, score on the held-out other.

## 2026-08-25 — [vacuous-gate] `reaches_empty` satisfied by a zero start
- **What happened:** A lives-candidate gate accepted any byte starting at 0.
- **Root cause:** `start == 0` trivially satisfies "reaches empty."
- **Consequence:** Bad lives nominations passed; one profile's search collapsed
  774 cells → 2, another 1096 → 24.
- **Rule (draft):** A gate clause must reject at least one realistic candidate.

## 2026-08-25 — [vacuous-gate] `spends_its_stock` satisfied by a 3-step regime
- **What happened:** The stock-depletion clause accepted a 3-step regime.
- **Root cause:** No minimum regime duration or refill-rate bound.
- **Consequence:** Oscillating bytes nominated as lives counters.
- **Rule (draft):** Bound regime duration and refill count explicitly.

---

# Project history (pre-2026-08-25)

## 2026-08-25 — [inert-treatment] Three treatments armed, wired, and never fired
- **What happened:** ReDo dormant-neuron recycling logged `dormant fc1 0/96 fc2
  0/32 recycled 0 cum 0` on all ~2,000 per-iteration checks across 8 runs.
  `symlog_rewards: true` and an entropy floor were likewise inert.
- **Root cause:** "Armed" was verified by preflight reading config, never by
  observing the mechanism act.
- **Consequence:** A whole registration amendment (tau-swept, preflight-verified)
  measured nothing; a verdict quoted ReDo as guaranteeing all 48k params active.
  It guaranteed nothing — it never ran.
- **Rule (draft):** Assert a mechanism's own counter is non-zero at least once per
  run, or report it as not-exercised.

## 2026-07-23 — [weak-eval] 3-episode acceptance passed welds with true rate 0/400
- **What happened:** Weld acceptance used 3-episode sticky evaluation. Welds it
  passed measured 0/400 when re-tested from deep inside the "welded" basin.
- **Root cause:** n=3 cannot distinguish a real basin from noise.
- **Consequence:** Distrust of all few-episode weld claims project-wide; a whole
  ladder of accepted welds invalidated.
- **Rule (draft):** Acceptance needs Wilson ≥50% at 95% on ≥20 episodes. Never
  accept on a sample too small to reject.

## 2026-07-23 — [reward-exploit] Five shaping exploits, each farmable
- **What happened:** Farmable negative Φ; transition-frame aliasing poisoning
  area-2 entry; x-only projection enabling a ceiling route; a Φ=0 sanctuary
  allowing peak-charge-and-lose; a probe harness evaluating random-init nets.
- **Root cause:** Shaping was checked for correctness on intended trajectories,
  never for what an optimizer could extract from unintended ones.
- **Consequence:** Months of PPO results shaped by exploitable objectives.
- **Rule (draft):** Prove shaping non-farmable — non-completing episodes must net
  ≤0 no matter how they end.

## 2026-07-16 — [weak-eval] Die-respawn inflated clear rates
- **What happened:** Five root bugs in the clear-evaluation path, chief among them
  counting a die-and-respawn as a level clear.
- **Root cause:** The success predicate did not distinguish reaching the next area
  by winning from reaching it by dying.
- **Consequence:** All clear claims before 2026-07-16 are untrustworthy and are
  marked so.
- **Rule (draft):** A success predicate must be tested against the failure that
  most resembles success.

## 2026-05-28 — [unverified-claim] Acted on an agent's impact estimate
- **What happened:** A review pass claimed a minibatch change would save ~170ms
  per iteration. Micro-benchmarked after landing: 29ms (1.2x), ~2% of an iteration.
- **Root cause:** The estimate measured the operation's total cost and mislabeled
  it as savings; it was quoted forward without re-derivation.
- **Consequence:** A change justified on a 6x-overstated number.
- **Rule (draft):** Re-derive any performance claim before quoting it. An
  operation's cost is not the saving from optimising it.

## 2026-05-28 — [start-state] Trained against an attract-mode demo
- **What happened:** A profile declared no `start_state_path`, so the emulator
  cold-booted to the title screen, where the demo auto-plays and ignores all
  controller input.
- **Root cause:** No validation that the environment was under agent control.
- **Consequence:** Every "PPO won't learn / exploration wall" symptom traced here:
  entropy pinned at 95% of ln(A), 57/80 iterations returning an identical 669,
  12 envs byte-identical. Diagnosed as an algorithm problem for an extended period.
- **Rule (draft):** Verify the agent controls the game before diagnosing learning
  — force half the envs to a different action and confirm they diverge.

## 2026-04-23 — [start-state] Chased a memory leak that was a corrupt input file
- **What happened:** Reported "14 GB/gen" leak. Investigated allocators, views,
  tensor lifecycles. Actual cause: a corrupt start-state file.
- **Root cause:** Began at the most technically interesting hypothesis instead of
  the cheapest one — swapping out the input.
- **Consequence:** Extensive jemalloc/pre-allocation work on a non-existent leak.
  Measured after: 90 MB/gen cold-boot vs 6,700 MB with the corrupt file; the pool
  itself was leak-free at +1 MB across 8 generations.
- **Rule (draft):** Rule out corrupt inputs before instrumenting the system. Move
  the state file aside and re-run first.

## 2026-04-22 — [stale-artifact] Measured a build that was never loaded
- **What happened:** Ran a parity harness against what was believed to be a
  perturbed-palette build. Tests passed because Python was loading the previous
  binary — `maturin` does not always replace the site-packages `.so`.
- **Root cause:** Assumed a successful build implies the new artifact is loaded.
- **Consequence:** ~15 minutes lost to a green result that measured old code.
  Found only by MD5-diffing the installed `.so` against the fresh dylib.
- **Rule (draft):** After any native rebuild, hash the loaded artifact against the
  built one before trusting any measurement.

## (recurring, undated) — [stale-artifact] Stale PGO profile read as a regression
- **What happened:** Benchmarked hot-path changes against existing `.profdata`;
  and separately mixed PGO and plain builds without `cargo clean`.
- **Root cause:** `pgo_build.sh apply` reuses stale profiles and a plain
  `maturin develop --release` produces a non-PGO wheel — neither substitutes for a
  fresh regeneration, and mixed histories corrupt the cache.
- **Consequence:** Stale profiles masquerade as regressions; mixed builds overstate
  ceilings. Ceilings and golden hashes generated in that window are untrustworthy.
- **Rule (draft):** Regenerate PGO from scratch after any hot-path change, and
  `cargo clean` between build modes before measuring.

