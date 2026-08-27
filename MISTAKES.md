# MISTAKES

Evidence log. Newest first. Not meant to be reloaded each session — this is the
archive the enforced rules in the project instruction file get audited against.

Rules below are **drafts, not enforced.** A root cause graduates to an enforced
one-line invariant only after recurring across 4–5 separate entries.

**Graduation watch:** `[vacuous-gate]` now stands at **4 entries** (2026-08-26 ×3,
2026-08-25 ×1) — at threshold, awaiting a call. A lint for `passed = not <coll>`
would enforce it deterministically and is preferable to a text rule.

---

## 2026-08-26 — Built this file twice in the wrong shape
- **What happened:** Wrote a 169-line prose `MISTAKES.md`, then rewrote it as a
  terse `mistakes.md` with a category/context/rule format, plus a pointer added
  to the enforced ruleset.
  Both were wrong; the spec arrived after each attempt.
- **Root cause:** Started producing before the format was specified, twice.
- **Consequence:** Two discarded commits; a premature rule added to the enforced
  ruleset and reverted.
- **Rule (draft):** When asked whether an artifact exists, answer first and
  confirm shape before authoring it.

## 2026-08-26 — Briefed a campaign from withdrawn numbers
- **What happened:** Launched a workflow citing Contra's "odometer 162 vs 163
  cross-validated" hours after an audit withdrew it. The 163 has no receipt
  anywhere in the tree.
- **Root cause:** Wrote the brief from recalled prior-session facts instead of
  re-checking claims recent commits had touched.
- **Consequence:** Stopped mid-flight and relaunched; ~10 min of agent work lost.
- **Rule (draft):** Re-verify any number a recent commit or audit could have moved
  before putting it in a brief.

## 2026-08-26 — Overruled a skeptic I commissioned, twice
- **What happened:** A designated-opposition agent argued breadth across games no
  policy can play was motion, not progress. Overruled both times.
- **Root cause:** Treated a commissioned falsifier's verdict as an obstacle to
  route around rather than as the evidence it was commissioned to produce.
- **Consequence:** Two campaigns (~70 and ~30 agents) returned 0 confirmed
  predicates and 0 forward progress.
- **Rule (draft):** When a commissioned skeptic says stop, stop or write down the
  specific evidence that beats it. "Proceed anyway" is not a rebuttal.

## 2026-08-26 — Reported a probe artifact as a fact about the game
- **What happened:** Stated "Rygar dies at 138 steps" and excluded Contra as
  SIGNAL UNUSABLE on "20 distinct in 69 steps."
- **Root cause:** 138 was an undodging scripted hold walking into a hazard (real
  window 3,865–4,000 actions); Contra's 69 samples were the post-truncation
  remnant of 1,131 and cannot support a 32-distinct threshold.
- **Consequence:** Excluded a viable candidate game from a campaign; corrected
  only when a later agent re-derived it.
- **Rule (draft):** Never issue a verdict on a window too small to support it;
  terminate probe holds at death before scoring.

## 2026-08-26 — `git update-ref` on shared main with concurrent writers
- **What happened:** An automated lane forcibly repointed `refs/heads/main` twice
  in a checkout with ~15 concurrent writers.
- **Root cause:** Raw ref plumbing bypasses the locking that normal commits rely on.
- **Consequence:** A race dropped a sibling's class (`592ea8a`); repaired by
  `698f142`. Nothing lost — verified by fsck/reflog — but by luck.
- **Rule (draft):** Parallel lanes get receipts-only or a worktree. No raw ref
  plumbing and no `git stash` on shared `main`.

## 2026-08-26 — Ledger entries dropped their receipts' scope
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

## 2026-08-26 — Constants quoted as measurements
- **What happened:** `is_clear` opened with `level_key(ram) > tuple(start_key)`;
  with `level_key: []` that is `() > ()`, False always, on 152/155 profiles.
  `area()` returns literal `0` when unconfigured.
- **Root cause:** No check that the expression producing a zero could be non-zero.
- **Consequence:** Every banked `solutions: 0` — millions of steps, up to 443,419
  cells — was a compile-time constant read as a search result. `n_area == 1` was a
  YAML property read as a fact about the ROM.
- **Rule (draft):** Before quoting a zero, evaluate its expression on random input
  and confirm it can be non-zero.

## 2026-08-26 — Detector suite tested only its own shape
- **What happened:** 186 tests green while the detector could not fire on two
  witnessed clears (Bubble Bobble round 69→70, a banked 4,329-action Tetris-B win).
- **Root cause:** Every fixture was SMB or a synthetic stream built to the
  detector's design centre. `coord` required a ≥300-unit drop against observables
  spanning 1 and 32 units — arithmetically impossible, never tested.
- **Consequence:** 41 profiles carried nulls that measured nothing; Gradius's
  clear hook was silently dead for 18 days.
- **Rule (draft):** At least one fixture must be a real positive from outside the
  instrument's design centre.

## 2026-08-26 — Reward dispatch on an incidental string
- **What happened:** `build_reward` selected on `name.contains("zelda"|"mario")`.
- **Root cause:** Dispatch keyed on a display name rather than a declared field.
- **Consequence:** Failed both directions — a text-clean profile silently
  inherited a quarantined disassembly-sourced win predicate; `smb_4_4_micro`
  declared nine Mario-only weights and got none of them.
- **Rule (draft):** Route on an explicit declared key with a safe default; remove
  the incidental string from the dispatch path entirely.

## 2026-08-26 — Mechanisms unguarded, signals wired to nothing
- **What happened:** `_dead_mm` (death-blip debounce): 3 occurrences in source, 0
  in tests. Separately, six detector signals were built while the live vote
  remained `tally + coord`.
- **Root cause:** No check that a mechanism has a test, or that a finished
  mechanism has a caller.
- **Consequence:** Correct code nothing would notice breaking; six signals reaching
  no production path.
- **Rule (draft):** Grep the mechanism's identifier in `tests/` (zero = defect) and
  confirm the production path reaches it.

## 2026-08-26 — Checkpoint selection and scoring on the same data
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
