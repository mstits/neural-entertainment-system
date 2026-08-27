# mistakes

Read at session start. Decision and process errors only. Terse by design.

The question none of these instruments was asking:
**what does this report when the mechanism is absent?**

---

## [verification] Gates that cannot fail
- Context: shipping quality gates for lives/progress/purity discovery.
- Mistake: success condition was `passed = not findings` — "nothing objected"
  instead of "something was demonstrated." Four shipped in one week; the fourth
  was written during the pass fixing the previous three.
- Rule: ALWAYS ship a gate with the mutation that reddens a named test, and RUN
  it — revert the mechanism, watch the test fail, restore. Green both ways = no test.

## [verification] Quoting a constant as a measurement
- Context: reading `solutions: 0` and `n_area == 1` as search/game results.
- Mistake: `is_clear` was `() > ()` (False always) for 152/155 profiles; `area()`
  returns literal 0 when unconfigured. Millions of steps of "evidence" were
  compile-time constants.
- Rule: BEFORE quoting a zero, evaluate the expression that produced it on random
  input and confirm it CAN be non-zero.

## [measurement] Reporting a probe artifact as a fact about the subject
- Context: I said "Rygar dies at 138 steps" and excluded Contra as SIGNAL UNUSABLE.
- Mistake: 138 was an undodging scripted hold walking into a hazard (real window:
  3,865–4,000 actions); Contra's verdict came from a 69-sample window that cannot
  support a 32-distinct threshold. I excluded a candidate game on it.
- Rule: NEVER issue a verdict on a window too small to support it, and ALWAYS
  terminate probe holds at death before scoring.

## [verification] Suites that only test an instrument against its own shape
- Context: 186 detector tests green while it could not fire on two witnessed clears.
- Mistake: every fixture was SMB or a synthetic stream built to the detector's
  design centre, so nothing could discover the detector was shaped wrong.
- Rule: ALWAYS include at least one real positive from OUTSIDE the instrument's
  design centre.

## [architecture] Dispatch on an incidental string
- Context: `build_reward` selected on `name.contains("zelda"|"mario")`.
- Mistake: failed both directions — a text-clean profile silently inherited a
  quarantined win predicate; another declared nine Mario weights and got none.
- Rule: ALWAYS route on an explicit declared key with a safe default; DELETE the
  incidental string from the dispatch path so matching is structurally impossible.

## [verification] Mechanisms nobody guards, and signals wired to nothing
- Context: `_dead_mm` debounce (3 hits in source, 0 in tests); six detector
  signals built while the live vote stayed `tally + coord`.
- Mistake: correct code with no test, and finished code reaching no caller.
- Rule: ALWAYS grep your mechanism's identifier in `tests/` (zero = defect) AND
  confirm the production path reaches it. Existing is not wiring.

## [reporting] Dropping the receipt's scope when quoting its headline
- Context: 15 of 39 adjudicated ledger claims.
- Mistake: the receipt was consistently more honest than the entry summarising it
  — "Stage 1-2 validated" became "the engine validated." No fabrication needed.
- Rule: ALWAYS quote a receipt's scope alongside its number.

## [statistics] Selecting and scoring on the same data
- Context: checkpoint selection by `argmax entrance_trailing_rate`.
- Mistake: under-selected 20–40 iters on 4/4 runs; correcting it then invited
  reading the max off the corrected table, which is winner's curse (measured 0.05).
- Rule: ALWAYS fix the estimator in advance and split-sample — select on one seed,
  score on the held-out other.

## [process] Overruling the designated skeptic
- Context: an opposition agent argued breadth across unplayable games was motion,
  not progress. I overruled it twice; both campaigns returned ~nothing.
- Mistake: treated a commissioned falsifier's verdict as an obstacle to route
  around rather than as the evidence I commissioned it to produce.
- Rule: WHEN a skeptic I commissioned says stop, either stop or write down the
  specific evidence that beats it. "Proceed anyway" is not a rebuttal.

## [process] Briefing work from facts that were already withdrawn
- Context: launched a campaign citing Contra's "162 vs 163 cross-validated"
  hours after an audit withdrew it (the 163 has no receipt in the tree).
- Mistake: wrote a brief from memory of prior sessions instead of re-checking
  claims that recent work had touched.
- Rule: BEFORE briefing agents with a number, re-verify any claim a recent commit
  or audit could have moved.

## [git] Raw ref plumbing on shared main
- Context: parallel lanes committing to one checkout; a race dropped a sibling's
  class.
- Mistake: `git update-ref` forcibly repointing `refs/heads/main` with concurrent
  writers. Recovered by luck, not design.
- Rule: NEVER use raw ref plumbing or `git stash` on shared `main`. Parallel lanes
  get receipts-only or a worktree.
