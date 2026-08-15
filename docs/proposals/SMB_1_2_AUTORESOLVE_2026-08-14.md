# SMB 1-2 Learned-Policy Investigation — DEFINITIVE VERDICT AND DECISION

Date: 2026-08-14
Status: **CLOSED — DECISION: BANK.** 1-2 is recorded as the documented open
hard-exploration wall for learned policies. No variation in this workflow, nor
any prior method in either paradigm, produced a single honest clear.

This document is the terminal record for the current line. It is deliberately
non-promotional: where the number is 0.0, it says 0.0. No 0.0 has been rounded
up, and no training-telemetry number is presented as capability.

**Honest protocol** (the only protocol that counts): cold start from the 1-2
entrance, greedy (argmax) action selection, sticky-0.25 action repeat,
jitter-16 frame-offset randomization, `--level-clear` predicate.
**Harness control:** a known-good 1-1 net scores **0.76** on this exact
harness. The harness registers clears when a policy can produce them. Every
0.0 below is a real 0.0, not a plumbing artifact.

Lineage: this closes the sequence begun in
`docs/proposals/HONEST_1_2_FINDINGS_2026-08-14.md` (imitation paradigm
eliminated; addendum named "online RL that practices sticky execution at the
bottleneck" as the only remaining idea). This workflow ran exactly that idea,
as a swept on-policy campaign. Result: 0.0 across every rung.

---

## (A) Complete elimination table — both paradigms, every honest number

### Paradigm 1: Imitation / distillation (ELIMINATED prior to this workflow)

| method | fit metric | honest clear |
|---|---|---|
| 1-1 known-good net (**harness positive control**) | — | **0.76** |
| BC, narrow funnel | train_acc 0.82 | 0.0 |
| BC, diverse 458k-pair funnel (14k-param baseline) | train_acc 0.475 | 0.0 |
| BC capacity sweep cap_h64 | 0.475 | 0.0 |
| BC capacity sweep cap_h128 | 0.531 | 0.0 |
| BC capacity sweep cap_h256 | 0.577 | 0.0 |
| BC capacity sweep cap_h512 | 0.612 | 0.0 |
| IQ-Learn distill (soft-Bellman + DQfD margin, stabilized) | argmax 0.374 (data ceiling 0.829) | 0.0 |
| BC + a_{t-1} feature (largest fit jump of any lever) | train_acc 0.531 → **0.747** | **0.0** (eval seeds 0/1/9) |

Imitation verdict (standing): capacity, observability, aliasing, and
loss-function explanations were each tested and eliminated. Better fit —
including the a_{t-1} jump to 0.747 — produced zero additional honest clears.
DQfD lives in this table twice: as the margin term inside the IQ-Learn distill
and inside the JAVI recipe below.

### Paradigm 2: On-policy sticky RL

| method | scale | honest clear | max area byte | notes |
|---|---|---|---|---|
| Online JAVI (full recipe: DQfD margin + non-farmable wavefront + go_explore restarts) | 3410 iters | 0.0 | — | prior campaign |
| JAVI-minus-anchor (full machinery minus imitation) | iter 150 (~of 5.4M-step run) | 0.0 | 2 | never left area 2 |
| JAVI-minus-anchor | iter 300 | 0.0 | 2 | never left area 2 |
| JAVI-minus-anchor | iter 450 (~5.4M steps) | 0.0 | 2 | never left area 2 |
| **This workflow — rung sweep (below)** | 4 × ~110 iters | **0.0 on all four** | 2.0 on all four | see (B) |

### This workflow's rungs (exact numbers)

| label | config | seed | honest_clear | mean_max_byte | final_iter | train_clears_max |
|---|---|---|---|---|---|---|
| lowent (entropy-sharpen) | `configs/mario_1_2_ar_lowent.yaml` | 0 | **0.0** | 2.0 | 109 | 13 |
| bottleneck (bottleneck-over-practice) | `configs/mario_1_2_ar_bottleneck.yaml` | 0 | **0.0** | 2.0 | 109 | 12 |
| combo (both combined) | `configs/mario_1_2_ar_combo.yaml` | 0 | **0.0** | 2.0 | 109 | 15 |
| seedvar (seed replicate) | `configs/mario_1_2_ar_seedvar.yaml` | 7 | **0.0** | 2.0 | 109 | 12 |

mean_max_byte 2.0 means the policy **never left area 2** (the 1-2 interior)
under the honest harness — the identical signature as the JAVI-minus-anchor
baseline. Not one honest episode escaped the area in any rung.

---

## (B) Did any variation break 0.0? — NO

- **Entropy-sharpen (lowent): 0.0.** Sharpening the action distribution did
  not convert training-time progress into a greedy-executable policy.
- **Bottleneck-over-practice (bottleneck): 0.0.** Concentrating practice on
  the momentum-critical bottleneck did not produce an honest clear.
- **Combined (combo): 0.0.** The combination peaked highest in training
  telemetry (train_clears_max 15) and still scored 0.0 honest.
- **Seed variation (seedvar, seed 7): 0.0.** The all-zero result is not a
  seed-0 artifact.

**On the training clears (13/12/15/12):** these are training-loop telemetry —
clears observed inside the training environment's own regime, not the honest
protocol. The gap (double-digit training clears, 0.0 honest, mean_max_byte
2.0) is the established mirage pattern this project has documented before
(2026-07-15: training telemetry vs cold greedy eval; name your harness). They
are reported here for completeness and claimed as nothing.

**Adversarial verify: not triggered.** The verify protocol runs only on a
non-zero rung. No rung was non-zero (verify record: null). Therefore there is
no CONFIRMED positive result anywhere in this workflow.

---

## (C) DECISION: BANK

No variation is CONFIRMED >0. Everything — two full paradigms, every honest
number above — is 0.0. Accordingly, the autonomous decision is:

**BANK SMB 1-2 as the documented open hard-exploration wall for learned
policies.**

1. **The solver stays the teacher.** The Go-Explore solver clears 1-2 (and
   the whole game) as teacher infrastructure. That capability is unaffected
   and unclaimed as learning.
2. **The learned product stands on the honestly-learnable levels.** The
   honest ledger for learned policies is what it is: 1-1 = 0.76 under the
   identical protocol; 1-2 = 0.0. Publish it that way.
3. **Stop spending against this transition on this line.** Imitation is
   exhausted (BC/DQfD/IQ-Learn/a_{t-1}); on-policy sticky RL with the full
   machinery minus imitation is exhausted at baseline (~5.4M steps) and
   across the entropy-sharpen / bottleneck-over-practice / combo / seed
   variations. Further sweeps of either paradigm are not research; they are
   re-rolls.

**The only remaining research-grade idea** — out of scope for this line — is
a fundamentally different training signal: one that optimizes robust
execution directly rather than expected return or demonstration match.
Concretely, the untested class is adversarial/worst-case execution training
(PR-MDP style: the sticky/perturbation schedule is chosen adversarially, so
the gradient must secure the momentum-critical maneuver under the worst draw,
not the average one). No honest 1-2 number exists for that class; it would be
a new prereg'd line, not a continuation of this one.

---

## (D) Truncation caveats

None. Every rung reached final_iter = 109 (~110 of 110 iterations; the
truncation threshold for flagging is final_iter < 60). All four results are
full-length runs, so the 0.0s cannot be attributed to truncated training.

---

## Receipts

- Rung configs: `configs/mario_1_2_ar_lowent.yaml`,
  `configs/mario_1_2_ar_bottleneck.yaml`, `configs/mario_1_2_ar_combo.yaml`,
  `configs/mario_1_2_ar_seedvar.yaml`
- Prior elimination record (imitation paradigm + harness validation):
  `docs/proposals/HONEST_1_2_FINDINGS_2026-08-14.md`
- Honest-protocol definition and history:
  `docs/research/RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md`,
  `docs/research/DOSSIER_V3_2026-07-23.md`
