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
