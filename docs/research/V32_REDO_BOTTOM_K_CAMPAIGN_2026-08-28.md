# v32 — ReDo at the rank-based bottom-k dose. Campaign result: 1/4 ARMED, 3/4 VOID-NO-TURNOVER. No Theta computed.

Registration: `docs/proposals/V32_REDO_BOTTOM_K_2026-08-28.md`, escalated at
compute time from rung 1 (k=2, C=5, VOID-NO-TURNOVER on Phase R) to rung 2
(k=4, C=10, GO on Phase R). Four seeds, 250 iterations each, run
2026-08-28 05:26–10:46. All four checkpoints, logs and manifests preserved
under `checkpoints/mario_1_1_v32_redo_bottom_k_seed{0..3}/`.

## Two adjudication defects found and corrected before reading a verdict

Both are logged in `MISTAKES.md`. Neither changes B2, B3, or B4 — only
whether a seed is admitted to be scored at all.

1. **The campaign runner called `redo_arm_gate.py` with no mode flag.**
   `--bottom-k` is opt-in; without it, `main()` silently falls through to the
   threshold-based `adjudicate()` path, which checks the run's `tau` against
   a default of 0.25 — a parameter the registration explicitly documents as
   **not read** under bottom-k ("`redo_tau` is not read on this path and is
   pinned to its schema default... a live tau numeral on a rank-rule run
   would be exactly the kind of dead knob this repo has shipped twenty of").
   All four seeds initially reported `VOID` on this basis. Re-run with
   `--bottom-k --k 4 --cadence 10`, which is what the seeds actually ran.

2. **B1's event floor (`>= 48`) is rung-1 arithmetic, never updated for rung
   2.** C=5 over 250 iterations gives 50 checks; 48 = 50 − 2 slack. At rung
   2's C=10, 25 checks is the *structural maximum* over 250 iterations —
   `>= 48` cannot be satisfied by any rung-2 run regardless of mechanism
   behavior. Corrected to 25 − 2 = 23 (`ADDENDUM 2026-08-28` in the
   registration, `1be16b2`). **Disclosed:** this correction was written after
   seeing all four seeds' raw event counts, not before. The number is derived
   from the registered rung-2 structure, not reverse-engineered to pass any
   particular seed — but the ordering is stated plainly rather than hidden.

## The result, B1–B4 exactly as registered, no numeral moved on B4

| seed | B1 (events/dose) | B2 (artifact) | B3 (dose) | B4 (turnover) | verdict |
|---|---|---|---|---|---|
| 0 | 25 events, 100 units | 100% match | 0.125 every check | `repeat_rate=0.958` | **ARMED** |
| 1 | 25 events, 100 units | 100% match | 0.125 every check | `repeat_rate=1.000` | VOID-NO-TURNOVER |
| 2 | 25 events, 100 units | 100% match | 0.125 every check | `repeat_rate=1.000` | VOID-NO-TURNOVER |
| 3 | 25 events, 100 units | 100% match | 0.125 every check | `repeat_rate=1.000` | VOID-NO-TURNOVER |

Full per-seed JSON: `docs/receipts/v32_redo_bottom_k/seed{0..3}_armgate.json`.

`repeat_rate == 1.00` is applied exactly as written in the registration's
§6.2: *"VOID at `repeat_rate == 1.00` exactly. Nothing else is gated."* No
reinterpretation was applied after seeing the results — the criterion was
binary and pre-committed, and it fires on three of four seeds.

**This is not a surprise the design failed to anticipate.** The
registration's own §6.3, written *before* the campaign, states: *"The guard
fires on every real ReDo trace this repository has ever produced... as of
registration time it is a gate that 100% of prior evidence fails."* Both
banked v31 Phase M logs (tau=0.075, tau=0.10) already VOIDed on this same
criterion. Three of four seeds joining that base rate is consistent with,
not contradicting, what was already known and registered with eyes open.

## What is NOT reported here, and why

**No Theta was computed.** The registration's design is `Theta = best-of-4
over seeds`, and its own text is explicit: *"Any of B1–B4 failing is VOID...
VOID enters no aggregate and takes no branch of the fork."* With 3 of 4 seeds
VOID, the design has one surviving seed. The registration does not specify
what `Theta`, `Theta_adj`, or the `Delta` secondary mean at n=1 — it was
written assuming a `best-of-4` denominator, and the ladder (§8) governs
Phase-R (pre-campaign pilot) outcomes, not post-campaign per-seed attrition
at this granularity. Computing a number from seed 0 alone and calling it
`Theta` would satisfy the letter of "run the eval" while violating the
purpose the cross-fit, best-of-4 design exists for: a single seed's honest
rate cannot be distinguished from seed-level noise, and this project has
independently measured that exact failure mode before (the checkpoint-
selection defect, the winner's-curse budget). No honest-eval compute has been
spent on any of the four seeds' checkpoints as of this writing.

## The plasticity hypothesis, status

**Still not tested to a PASS/FAIL bar**, and this campaign does not close
that gap — it closes a different one. What v32 establishes, for the first
time with real evidence rather than a two-rung pilot:

- Rank-based bottom-k ReDo, escalated exactly per its own registered ladder
  to the only permitted rung, **still fails to sustain turnover in 3 of 4
  seeds over a full 250-iteration run**, even though the same operating
  point passed a 120-iteration Phase-R pilot cleanly. The pilot-vs-full-run
  gap is the same shape v31 found for fixed-tau ReDo, one level up: a short
  window can look fine while the full run does not.
- One seed (0) is genuinely armed — fires at the registered dose, artifact-
  matched, and turns over (repeat_rate 0.958, 16 distinct fc2 indices touched
  over 25 events). The mechanism is not universally degenerate; it is
  seed-dependent, and the registration's own honest-disclosure section
  already flagged this as the likely outcome.

## Recommended next step — not executed here, requires its own registration

Two options exist and neither is executed by this document:

1. **Score seed 0 alone, explicitly as an n=1 exploratory reading**, with a
   pre-registered statement of what that number can and cannot support
   (almost certainly: "consistent with" or "inconsistent with" a large
   effect, nothing finer). This requires deciding, in writing, before running
   any eval, what an n=1 result is licensed to say.
2. **Retire rank-based ReDo for this stack**, following the spirit of v31's
   own pre-written stopping statement, generalized: *"a re-initialized trunk
   unit does not reliably climb out of the rank-bottom of the dormancy
   distribution even at the only registered rank-rule operating point that
   passed its own pilot."* This would close the ReDo line of inquiry (v27,
   v28, v30, v31, v32) with a banked two-registration receipt, matching the
   discipline v31 already established for the fixed-tau family.

Both are legitimate; neither should be decided by continuing to spend
compute without writing the choice down first.
