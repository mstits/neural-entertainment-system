# MISTAKES

Evidence log. Newest first. Not meant to be reloaded each session — this is the
archive the enforced rules in the project instruction file get audited against.

Rules below are **drafts, not enforced.** A root cause graduates to an enforced
one-line invariant only after recurring across 4–5 separate entries.

**Graduation watch** — root causes at or past the 4–5 entry threshold, awaiting a call:

| root cause | entries | deterministic enforcement available? |
|---|---|---|
| `[vacuous-gate]` a check that cannot fail | **5** | yes — lint for `passed = not <coll>` |
| `[stale-artifact]` measured the old binary/profile | **5** | yes — hash the loaded artifact against the built one in CI-less `make` target |
| `[inert-treatment]` armed, wired, never fired | **4** | **SHIPPED 2026-08-27** — `scripts/check_mechanism_receipt.py` returns VOID for any armed mechanism whose counter never moves |
| `[weak-eval]` protocol too weak to detect its own failure | **6** | partly — enforce min-n at the gate |
| `[unverified-claim]` trusted a number without re-deriving it | **9** | no — judgement |
| `[purity-leak]` external/unwitnessed semantics in a live map | **3** | **SHIPPED 2026-08-27** — `make purity-check` (derived scanner + 27-row provenance registry + `WIN_WITNESS_LEDGER`) |

Nothing has been promoted; the enforced ruleset is untouched. `[vacuous-gate]`,
`[stale-artifact]`, `[weak-eval]` and `[unverified-claim]` are all at or past the
threshold as of 2026-08-27 and are awaiting a call.

`[purity-leak]` stands at **3 entries** and the counter is now backfilled, which
the 2026-08-27 note above declined to do. The three: the **Zelda quarantine**
(2026-08-25, two independent paths — the reward struct and the name-substring
dispatch sites), the **994-entry config sweep** (2026-08-27, 7 quarantined), and
the **engine sweep** (2026-08-27, 27 constants annotated). Backfilling is the
honest call because the third entry proves the class is one root cause and not
three: the config sweep's own scope note ("quarantining the YAML retracts the
DOCUMENTATION claim, not the Rust constant") predicted the engine entry in
writing, and the engine entry then found the *same* retracted sentences alive one
layer down.

**Threshold NOT reached — 3 against 4–5 — and the normal reason to wait does not
apply here, so this row is a deliberate exception worth stating.** The usual
blocker is that no deterministic enforcement exists; for this root cause it now
does, and it is already wired into `make test`. So the call is:

> **Do not promote a written rule. The enforcement IS the mechanical check.**

Seven vacuous gates have shipped in this project, which is direct evidence that a
written invariant does not hold *here* — and this very root cause supplies the
sharpest proof. On 2026-08-27 three guards written the same morning to enforce
this class were themselves defective when independently reverted: a 60-line
proximity window let **19 of 24** constants survive deletion of their own
provenance tag; a stale-binary guard compared only reward-id *sets*, so the exact
edits it existed to catch passed; and a 50-line lookback let all **8** retracted
sentences be restored silently in the very headers they were withdrawn from. Each
was written in good faith by someone who had just read the rule. A fourth entry
would not make a written rule more likely to hold; it would only mean the class
recurred again while an enforceable check sat available.

`[inert-treatment]` is the first root cause whose enforcement actually exists.
`check_mechanism_receipt.py` reads a run's own artifacts and returns **VOID** (not
FAIL) for any mechanism that announced itself and whose counter never moved, and
distinguishes that from **UNAUDITABLE** — armed with no counter at all, which is
what `[hazard-mask] ARMED` runs are. Run against `runs/v27_fresh_recovery` it
reports `redo INERT, peak 0 over 1000 observations`; run against
`checkpoints/mario_1_2_online_v2` it reports every registered mechanism FIRED.
The second is the positive control, without which a checker hard-wired to say
INERT would pass its whole suite. Verified by deleting the check: five mutants
kill 8 / 2 / 2 / 6 / 1 tests respectively, and the unmutated control passes 23/23
on the identical harness. The static half — a config key the trainer parses but
cannot reach under `trainer_mode: vanilla_ppo` — is derived from the AST in
`config_schema.inert_reinforce_keys_under_vanilla_ppo()` and raises rather than
returning an empty set when it cannot find the dispatch it parses.

---

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
  `2e6014f` is **still closed**: `_mapping_bytes()` is private, reaches `main()`
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

## 2026-08-26 — Killed a workflow that was already self-correcting
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

