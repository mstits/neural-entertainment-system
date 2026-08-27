# Mistakes

Defect *classes* this project has actually shipped, each with a real instance and
the check that catches it. Not general advice — every entry cost real work.

The unifying question, which nothing was asking:

> **What does this report when the mechanism is absent?**

A check that cannot answer that has certified nothing, however green it is.

---

## 1. The vacuous gate

A gate whose success condition is *"nothing objected"* rather than *"something was
demonstrated."* The idiom is `passed = not findings`.

**Four shipped in one week. The fourth was written during the pass that fixed the
previous three, with the previous three in front of it.** That is the measure of
how natural the mistake is.

| instance | why it could not fail |
|---|---|
| `reaches_empty` in the lives gate | `start == 0` satisfied it for free |
| `spends_its_stock` | a 3-step regime satisfied it |
| `progress_signal_gate` camera-static branch | forced `passed=true` whenever OAM churn showed motion — **40 vacuous passes across 13 profiles**, all reading `distinct=1, min=0, max=0` |
| the observatory purity guard | asserted against *a copy of the fixed line pasted into its own test body*; restoring the bug left all three tests green |

**The check:** ship every gate with the mutation that reddens a named test, and
*run it* — revert the mechanism, watch the test fail, restore. If it stays green,
you have not written a test.

---

## 2. A constant read as a measurement

The most expensive class here, because the output looks exactly like data.

- `is_clear` opened with `level_key(ram) > tuple(start_key)`. With `level_key: []`
  that is `() > ()` — **False in Python, always.** 152 of 155 profiles have an
  empty `level_key`, so every banked `solutions: 0` (millions of steps, up to
  443,419 cells on one profile) was a **compile-time constant**, never evidence
  that a search tried and failed.
- `area()` returns literal `0` when a profile declares no `solve.area`. So
  `n_area == 1` — read repeatedly as *"this game has no stage structure"* — was a
  property of the YAML, not of the ROM.
- Item-semantics Stage 3a read `cap_hist` from archives minted before the field
  existed: every edge yielded `{}`, so `total_leads: 0` at any budget or seed.

**The check:** before quoting a zero, evaluate the expression that produced it on
a random input and confirm it *can* be non-zero.

---

## 3. A probe artifact reported as a fact about the game

The probe's own behaviour gets attributed to the subject.

- **"Rygar's live window is 138 steps."** That is the survival of an *undodging
  scripted `right` hold from `lives=1`* walking into a hazard. Real windows:
  uniform-random survives a median **677** steps; the solver runs **3,865–4,000
  actions in one continuous life with zero deaths.**
- **Contra "SIGNAL UNUSABLE — only 20 distinct in 69 steps (<32)."** Those 69
  steps were all that survived truncating 1,131 post-death steps. **A 69-sample
  window cannot demonstrate a 32-distinct threshold.** Correct verdict:
  INCONCLUSIVE. Contra was excluded from a campaign on this.
- **Post-death tails: 28 of 45 profiles.** A probe that keeps stepping after
  death scores the game-over animation as progress. Contra's banked "162 distinct"
  was **~94% game-over animation.**

**The check:** terminate every hold at death, and never issue a verdict on a
window too small to support it.

---

## 4. A green suite that only tests the instrument against its own shape

**186 detector tests passed while the detector could not fire on two witnessed
clears** — Bubble Bobble round 69→70 and a banked 4,329-action Tetris-B win,
both visible on screen. Every case in that suite was either SMB or a synthetic
stream built to the detector's own shape.

The cause was arithmetic, not stochastic: `coord` required a ≥300-unit drop
against progress observables spanning **1 and 32 units**. It was *guaranteed
absent* on 26 odometer profiles — `odo_fold_frame` re-anchors without integrating
across a scene cut, with a unit test asserting exactly that.

**The check:** at least one fixture must be a *real* positive from outside the
instrument's design centre. A suite that cannot discover the instrument is shaped
wrong is decoration.

---

## 5. Dispatch on an incidental string

`build_reward` selected on `name.contains("zelda")` / `contains("mario")`.

It failed in **both** directions at once. `legend_of_zelda.yaml` — text-clean, no
`ram_mapping`, no address of any kind — silently inherited a **disassembly-sourced
win predicate** it never declared. Meanwhile `smb_4_4_micro.yaml` (contains "smb",
not "mario") declared **nine Mario-only weights** and got the generic reward; all
nine inert, on the flagship game, on a profile with live-show receipts.

**The check:** route on an explicit declared key with a safe default. Delete the
incidental string from the dispatch path entirely, so substring matching becomes
structurally impossible rather than merely discouraged.

---

## 6. A wired mechanism nobody guards

`_dead_mm`, the death-blip debounce: **3 occurrences in source, 0 in tests.** It
was correct — but nothing would have noticed if it stopped being correct.

Its inverse also shipped: **six clear-detector signals built and wired to
nothing**, while the live vote remained `tally + coord`.

**The check:** grep your mechanism's identifier in `tests/`. Zero hits is a defect.
And confirm the production path *reaches* it — existing somewhere is not wiring.

---

## 7. The summary drops the receipt's own caveat

Found across **15 of 39** adjudicated ledger claims. In every case **the receipt
was more honest than the entry summarising it.** Not fabrication — the scoping
qualifier got dropped when the headline was quoted forward. "Stage 1–2 validated"
became "the engine validated."

**The check:** when quoting a receipt, quote its scope with it.

---

## 8. Selecting and scoring on the same data

The checkpoint selector (`argmax entrance_trailing_rate`) under-selected by
20–40 iterations on **4 of 4** runs tested; recorded peaks were low by
+0.08…+0.21. The metric was ceiling-saturated (0.867–1.000, SE ~0.09 on a 30-episode
window), making the argmax near-arbitrary.

Correcting it does **not** license reading the max off the corrected table — that
is winner's curse. Split-sample (select on one eval seed, score on the held-out
other) landed on **0.670 exactly**, the recorded number. Measured curse: **0.05**.

**The check:** fix the estimator *in advance* and budget the curse.

---

## 9. Shared `main` with concurrent writers

An automated pass used raw `git update-ref` to forcibly repoint `refs/heads/main`
in a checkout it knew had concurrent writers. A race dropped a sibling's class
(`592ea8a`, repaired by `698f142`). Nothing was lost — verified via `fsck`,
reflog and content checks — but by luck, not design.

**The rule:** parallel lanes get receipts-only or a worktree. Never raw ref
plumbing on shared `main`. (And never `git stash` — stash refs are shared, so
parallel lanes swap diffs.)

---

## The meta-lesson

Every entry above is the same shape: **something was measuring itself, or
measuring nothing, and reporting a number anyway.** The instruments were trusted
because they produced output, and output resembles evidence.

Ask of any number before quoting it: *could this have come out differently?*
