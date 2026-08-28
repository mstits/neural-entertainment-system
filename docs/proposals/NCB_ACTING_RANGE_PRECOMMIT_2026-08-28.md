# NC-b acting-range adjudication — pre-commitment

Written 2026-08-28, **before any Lane A v2 scoring output exists**
(`runs/vadv_onpolicy_v2/` not yet created at commit time; the commit timestamp
is the proof). Prompted by external review (second-session reviewer of the
NC-b derivation), whose two requests this file implements.

## Why this exists

Last round NEG_gx_frozen tracked PC_B5 **bit-identically at 26/26 iterates** —
and that one statistic was ambiguous between "perfect tracking" and "degenerate
input," because the banks were aliased (`state == next_state` on 100% of rows).
Perfect tracking and a degenerate input are identical in the tracking number
alone. This file separates them in advance.

## 1. Ordering rule (binding on the receipts)

Every NC-b receipt reports, **first and per bank**, before any tracking number:

- `chain_guard`: pass/fail of the exact antecedent-chain check (b2e806b).
  A failing bank is VOID; no tracking number is read from it.
- `alias_rate`: fraction of rows where `state == next_state` **bitwise**.
  Legal nonzero (frozen frames inside a 4-frame stack), but 1.00 was the defect.

Only after both may NEG/PC tracking be reported.

## 2. Information gate — AMENDED before any scoring output exists

The original gate (median alias_rate > 0.50 pooled) is REPLACED: a median hides
a bimodal split, and aliasing is a per-bank property of how each bank was
recorded, so bimodal is the expected failure shape — six banks at 0.02, 0.02,
0.03, 0.98, 0.99, 1.00 would have squeaked through the median while half the
evidence was degenerate. (Amendment prompted by the same external reviewer;
`runs/vadv_onpolicy_v2/` still does not exist at amendment time.)

- **Per-bank gate:** a bank with `alias_rate > 0.50` (or a failing chain guard)
  is VOID and drops out. No pooled statistic is computed over VOID banks.
- **Survivor floor, pre-stated:** the overall NC-b verdict requires **>= 20 of
  26 banks surviving** the per-bank gate. Fewer survivors = **VOID-THIN-BASIS**,
  regardless of how clean the survivors look.
- **Receipts carry the per-bank `alias_rate` vector**, all 26 values, not a
  summary — the distribution must be visible without recomputation.

## 2b. Unit clarification and rounding — pre-data, prompted by review

**BANK == ITERATE on this protocol, and there are exactly 26 banks.** Verified
from the v1 structure, not asserted: 26 npz files (`iter_00010` .. `iter_00260`),
one per checkpoint's own rollout, each containing ALL source populations
(src_rung 893 / PC rungs / entrance) in a single file. The per-bank gate in §2
therefore gates per-iterate-bank, and the two axes the reviewer asked to
distinguish are structurally the same axis here. Were a future protocol to
record multiple banks per iterate, §2 gates BANKS (recording units) and the §3
criteria count ITERATES within surviving banks — stated now so the distinction
survives a protocol change.

**Rounding, stated before any number exists:** the §3(b) thresholds are
`ceil((14/26) * S)` and `ceil((20/26) * S)` where S = surviving banks. This
preserves the original ba34605 numerals exactly at full survival (S=26 gives 14
and 20) and rounds UP — the conservative direction, since §3 certifies a
control as usable and the burden is on the control. At the S=20 floor the
thresholds are 11 and 16.

## 3. Usability criterion (written before the number)

On sound banks (chain guard pass, information gate pass), **NC-b is USABLE on
on-policy data iff BOTH:**

- (a) NEG_gx_frozen is not bit-identical to PC_B5 at **any SURVIVING bank**
  (n_cells, n_rows, raw, eta2, null median, q975 — all six fields).
  **Quantified over survivors only, same scope as (b)** — resolved pre-data:
  aliasing is the known cause of the NEG==PC collapse, so on a bank already
  VOID for aliasing, bit-identity is the expected symptom of the
  already-excluded cause and carries no further information; quantifying over
  all 26 would punish one defect twice and silently reduce the §2 six-void
  tolerance to zero. On a SOUND bank the reading is the opposite and the
  clause stays global in force: six-field bit-identity there cannot come from
  the known cause, so a SINGLE surviving bank exhibiting it is a pipeline
  arithmetic identity and disqualifies NC-b outright, not just that bank; and
- (b) NEG's and PC_B5's eta2 95% bootstrap CIs are **disjoint** (CI-vs-CI, the
  stricter of the two available tests — the original CI-vs-point-estimate
  treated PC as noiseless and was the more permissive; tightened pre-data,
  which is admissible where loosening would not be) at **>= 54% of surviving
  banks**, AND NEG reads not-LIVE at **>= 77% of surviving banks** at which
  PC_B5 reads LIVE. (Fractions of survivors, with the >= 20 survivor floor
  from §2, replace the original absolute 14/26 and 20/26.)

Anything less: **NC-b is UNUSABLE on on-policy data** and must be re-specified
by written addendum before any primary verdict is issued. A middling result is
UNUSABLE — the burden is on the control to demonstrate it can collapse.

## 4. Precedence

If the Lane A precondition agent registered its own NC-b criterion **before
scoring began**, that registration governs and this file is the fallback. Both
must predate scoring; a criterion written after seeing the numbers governs
nothing.

## 5. Supersession, precedence, and a disclosure — written 2026-08-28 ~02:2x, before any adjudication

**The full chain, in one place:**
`ba34605` (01:18:56) → guard definition superseded by `6d700c5` (01:21:15) →
`f89e6c9` (01:27:28) → `59a8a38` (01:29:20) → `94b533d` (01:30:39).

**Guard supersession, accepted.** §1 named "the exact antecedent-chain check
(b2e806b), no tolerance." That guard's first live run false-positived at iter
30 on a PC_SRC rung-1013 episode: the episode traverses the WALL band, the
**registered** cross-population drop removed those interior rows, and the
legitimate gap read as CHAIN BROKEN. The guard as registered was inconsistent
with the protocol as registered — a broken instrument, not an inconvenient
bar. `6d700c5` records the within-episode step index BEFORE the drop and
asserts the chain only across step-consecutive pairs; aliasing corrupts every
adjacent pair, so the guard loses nothing against the defect class it exists
for. This is the loosening direction and does not get the tightening free
pass; it is admitted as an instrument repair on three grounds: the false
positive was against the registration's own mandated drop, the fix is
revert-verified, and it predates every bank in the operative grid (first bank
01:24:04). **The operative guard is 6d700c5 wherever this file says b2e806b.**

**Precedence, stated plainly per §4's own rule:** the Lane A precondition
agent registered its own addendum at `77b8549` (01:11:07) — BEFORE ba34605 —
including an NC-b acting-range decision rule (MEASURABLE-per-iterate with a
>=70% floor). **77b8549 governs wherever it speaks; this file is the
fallback**, exactly as §4 provides. An adjudicator applies 77b8549's rule
first and this file's §2/§3 machinery where 77b8549 is silent.

**Disclosure of a defect in this file's own process.** Each amendment here
claimed "window verified open" against `runs/vadv_onpolicy_v2/` — but the
workflow wrote to `runs/vadv_onpolicy_rerun/`. **The window check was vacuous:
it could not fail**, because it watched a path the pipeline never used. The
pre-data status of the amendments is instead established by artifact
timestamps, which happen to hold:

| event | time |
|---|---|
| ba34605 (base precommit) | 01:18:56 — pre-collection entirely |
| first bank (iter_00010.npz) | 01:24:04 |
| f89e6c9 / 59a8a38 / 94b533d | 01:27:28 / 01:29:20 / 01:30:39 |
| collection complete (collect_summary) | 02:13:14 |
| first scoring output (arc_scored.jsonl) | 02:19:17 |

So the three amendments landed after 1 of 26 banks existed and before the
other 25, before the collection summary, and ~49 minutes before any scoring
output. Their content depends on nothing observable in one bank, and the
author did not read it — but "did not read" is asserted, not provable, so the
adjudicator may discount f89e6c9→94b533d as mid-collection if they judge that
inadmissible. **ba34605 is fully pre-collection and stands regardless.**

**Receipts path correction:** wherever this file says `runs/vadv_onpolicy_v2/`,
the operative path is `runs/vadv_onpolicy_rerun/`.

## 6. Boundary map — how 77b8549 and this file compose. A record; no numeral changes

Written after reading 77b8549 §2.3 in full, before any adjudication.

**77b8549 governs the VERDICT PATH for NC-b, entirely.** Its rule: NC-b is
MEASURABLE at an iterate iff (a) its row set is not identical to PC_B5's and
(b) it has >= 6 qualifying rows in >= 2 cells. **The 70% floor is over
COLLECTED iterates — all 26 — not survivors** (it is critic-free and predates
this file's bank-soundness machinery). Its two branches, including the
pre-decided re-specification for the UNMEASURABLE-majority case ("an
UNMEASURABLE NC-b does not cap; a MEASURABLE non-collapsed NC-b caps at
INDETERMINATE"), are the operative cap semantics. Nothing in this file may
override them.

**77b8549 is silent on, and this file therefore supplies:**
- the receipts ORDERING (§1: chain-guard verdict and bitwise alias_rate per
  bank, before any tracking number) and the full 26-value alias vector;
- BANK soundness (§2: per-bank alias gate, >= 20-survivor floor) — 77b8549
  says "repaired banks"; this file defines what sound means;
- §3's CI-vs-CI and not-LIVE tests. **77b8549 nowhere speaks to CI separation
  — so §3(b) is not replaced; but it is also not on the verdict path.** It is
  published with the receipts as a control-quality characterisation, informing
  any future re-specification, capping nothing.

**Where both speak to identity:** 77b8549's per-iterate ROW-SET identity is
upstream and governs the verdict path (identity → UNMEASURABLE at that
iterate). This file's §3(a) six-field bit-identity on a SOUND bank demotes,
under precedence, from outright disqualifier to REPORTED ANOMALY requiring
investigation before any verdict is released — a sound bank exhibiting it
would almost certainly also trip 77b8549's row-set rule, so the practical
difference is the mandatory investigation, not the cap.

## 7. The cause-split — closing the gap where the defect's recurrence grants permission

**Provenance, disclosed before the rule:** this section postdates the existence
of scoring artifacts (arc_scored.jsonl exists as of ~02:21) and predates any
READING of them — at registration time neither the author nor the external
reviewer has read collect_summary.json, arc_scored.jsonl, or any sidecar, and
the workflow's own run/adjudication phases have not returned (verified:
1 of 7 agents returned, a precondition agent). It is a TIGHTENING (it converts
a permissive outcome to VOID). Because it postdates artifact existence:
**if this section is outcome-determinative, the adjudication publishes BOTH
readings — with and without the split — with this provenance note beside
them.** That commitment removes any benefit post-hoc tailoring could confer.

**The gap (found by the external reviewer):** MEASURABLE(a) requires NC-b's
row set to differ from PC_B5's, identity being the 2026-08-27 defect's own
signature. So a recurrence of that defect — in a form evading the 6d700c5
chain guard, which aborts at collection — makes iterates UNMEASURABLE; an
UNMEASURABLE majority takes 77b8549's no-cap branch; and the recurrence of
the exact defect this rerun exists to eliminate would REMOVE the ceiling
rather than voiding the run. The instrument's failure mode would produce the
permissive outcome. Unmeasurability here is not silence; its dominant cause
is the pipeline defect under repair, and treating defect-driven and genuine
control-absence as one outcome is the error.

**The rule (Gap 2):** the UNMEASURABLE branch splits by cause, using the bank
soundness machinery 77b8549 predates and could not have consulted:
- UNMEASURABLE at an iterate whose bank is SOUND (chain guard pass,
  alias_rate under the §2 gate): genuine control-absence. 77b8549's no-cap
  branch applies unchanged.
- UNMEASURABLE at an iterate whose bank is VOID: instrument failure, not
  control-absence. **Does not count toward the no-cap majority. If VOID-bank
  iterates are what push past the 30%, the verdict is VOID, not uncapped.**
This overrides 77b8549 nowhere it speaks: its rule assumed "repaired banks"
as premise; the split supplies the distinction its critic-free, pre-soundness
text could not draw.

**Gap 1, pre-decided:** a §2 survivor-floor failure (< 20 sound banks)
BLOCKS the verdict path — VOID-THIN-BASIS, not merely reported. Bank
soundness is the admissibility premise 77b8549 relies on implicitly
("repaired banks"); fewer than 20 sound banks means that premise fails and
no verdict issues, whatever the MEASURABLE table says.

## 8. Operative path, confirmed with the adjudicator pre-data — final

The adjudicating session stated this path back and it is confirmed correct;
a future reader needs no other summary:

1. Admissibility: per-bank alias/chain gate; survivor floor 20 of 26; failure
   is VOID-THIN-BASIS and no verdict issues, whatever the MEASURABLE table says.
2. Measurability per 77b8549: row set not identical to PC_B5, >= 6 qualifying
   rows in >= 2 cells; 70% floor over all 26 collected iterates.
3. Cause-split on UNMEASURABLE (§7): sound bank = genuine control-absence;
   void bank counts nothing toward the no-cap majority, and the verdict is
   VOID if void banks are what carry it past 30%.
4. Branch per 77b8549: unmeasurable majority does not cap; measurable
   non-collapsed caps at INDETERMINATE.
5. §3 CI tests reported as control-quality characterisation, capping nothing;
   the six-field clause as anomaly-requiring-investigation, not a cap.
6. If the cause-split is outcome-determinative, both readings publish side by
   side with §7's provenance note.

**Known property, recorded so it is not a discovery later:** a bank VOID for
aliasing can still be MEASURABLE — alias identity and NC-b-vs-PC_B5 identity
are different comparisons — so void banks may contribute to the 70% MEASURABLE
numerator while being excluded from the no-cap majority. The asymmetry errs
RESTRICTIVE (it can only push toward the capping branch) and is deliberately
left in place.

Chain of record: ba34605 → f89e6c9 → 59a8a38 → 94b533d → guard 6d700c5 →
3248ca7 → 78a3250 → b55e9ac → this section. Governed first by 77b8549.

## 9. Post-adjudication record: why alias_rate is exactly 0.0 (item 5, resolved)

The adjudicator asked why a statistic predicted to have a lawful nonzero floor
came back exactly 0.0 at 26/26. Measured answer, from the banks themselves:
**the obs embeds a clock.** Dim 711 — the last scalar of the newest frame in
the 4x178 stack — advances +2 per agent-step and wraps mod 128. In bank 10,
65 rows differ from their successor at ONLY dim 711: those are precisely the
frozen-screen rows that would have been bitwise-identical without the clock.
Same structure in banks 130 and 260.

So the earlier "lawful nonzero floor" prediction was WRONG FOR THIS ENCODING —
it was reasoned from raw-frame intuition. Under this encoding the statistic is
effectively BINARY: healthy collection gives exactly 0.0 (the clock always
moves); the buffer-aliasing defect gives exactly 1.0 (both copies are the same
buffer, clock included). The 0.50 gate is therefore generous but harmless, and
the unanimous 0.0 is the expected-by-construction healthy reading, not an
anomaly. The near-identity histogram (Hamming-1 rows all at dim 711) is the
positive demonstration that frozen-frame rows exist and were separated by the
clock — the statistic's discriminating power is intact, indeed sharper than
registered.

## 10. The 180-row regularity — explained, and §9's clock statement corrected

The adjudicator's fifth finding: 180 mid-episode rows per bank violate the
"+2 per step" clock cadence, in a deterministic 60/60/60 pattern identical
across banks. Explained by measurement, one census per group, all three
**one row per ENTR_SRC episode (60 per bank) at a FIXED within-episode
position**, replicated at the same positions in banks 10, 130 and 260:

- **delta +1 at position 27**: the counter misses one tick (a slip).
- **delta +3 at position 30**: the counter regains it (catch-up). Net drift
  over the pair: zero.
- **raw −120 at position 115** (the group read as "+8 mod 128"): the clock
  series reads ...118, 120 → 0. **The counter is RESET to zero, not
  advanced** — the +8 was an arithmetic coincidence (128−120). This also
  proves dim 711 is a game-writable counter, not a pure frame derivation.

All three are action-independent (fixed positions across 60 episodes with
per-episode seeds and sampled actions), i.e. a scripted phase of the entrance
opening drives the counter's cadence, not agent behaviour. WALL_SRC and
PC_SRC episodes start elsewhere and never exhibit them.

**Correction to §9:** the clock statement "advances +2 per agent-step" holds
for 99.68% of rows; the full census of exceptions is the above plus wrap
(−126) and episode-boundary rows. **The load-bearing property for the
retirement is the weaker and fully verified one: the per-row delta is never
0, so consecutive observations are never bitwise identical.** Any future
check must build on "never 0," not on "+2 always."

