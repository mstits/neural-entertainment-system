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

## 3. Usability criterion (written before the number)

On sound banks (chain guard pass, information gate pass), **NC-b is USABLE on
on-policy data iff BOTH:**

- (a) NEG_gx_frozen is not bit-identical to PC_B5 at **any** iterate
  (n_cells, n_rows, raw, eta2, null median, q975 — all six fields); and
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
