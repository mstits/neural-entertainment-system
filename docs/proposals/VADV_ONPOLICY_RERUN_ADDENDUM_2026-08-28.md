# V_adv on-policy re-execution — disclosed addendum to the 2026-08-27 registration

**Written and committed 2026-08-28, before any collection compute and before
any `eta2`, null, `R` or verdict statistic exists in this job.**

**Parent registration:** `docs/proposals/VADV_ONPOLICY_PREREG_2026-08-27.md`
(commit `35f6d60`) — **BINDING AND INHERITED VERBATIM** except for the items
below. **Predecessor execution:** VOID on a collector data-integrity defect
(`docs/research/TWO_REGISTERED_TESTS_2026-08-27.md` Part I): a reused
`TileFeatureStacker._out` buffer recorded `(s', a, s')` on 100 % of rows in
all 26 banks. This is the §1.9 re-run that VOID named. It is a
**re-execution, not a re-design**.

---

## 1. Inherited verbatim — none of these move

The statistic, the estimator, gamma, the cell key and qualification rule,
the band edges (`WALL [2674,2872)`, `PC_B5 [2872,3267)`, `INTERIOR
[2676,2872)`, `EARLY [160,1600)`), LIVE/COLLAPSED rules, the pre-committed
permutation-median null reference, `R` and its bands (MIS-SPEC `<= 0.20`,
CAPABILITY `>= 0.50`), the §7.2 arc rule (`|A| >= 13`, 80 %), A1–A8 with
the binding A1/A3 stop rule, the A7 negative demonstration, the §4 rollout
protocol (checkpoints, SAMPLED temperature 1.0, sticky 0.25 gated on
`step > 0`, hard source partition with cross-population drop, episode caps,
episode counts 40/24/60, purity guard, seeds `20260827 + iter`), the §5.3
no-penetration reading (INTERIOR VOID never COLLAPSED, positive
measurement, evidence for neither hypothesis), the §5.4 rung-933 probe
(diagnostic only), the §10 budget (6.0 h lane, <= 3.0 h compute) and abort
table, and the §11 retirement rule. `n_perm = 1000`, `n_boot = 2000`,
loose cell key.

**A moved goalpost after the first `eta2` is read voids the job.** None of
the numerals above may change.

## 2. What changed, each with its reason

**2.1 The collector is repaired** (commit `e89091e`). Both call sites copy
out of the stacker's reused buffer, and `assert_bank_wellformed` (the
threshold-free chain + non-degeneracy invariants) runs on the exact arrays
handed to `np.savez`. Re-verified today, executed not asserted: 45/45
tests green; neutering the guard fails 4/45 including the exact 2026-08-27
artifact; a five-step live check against the real `TileFeatureStacker`
loop pattern returns `identical_frac = 0.0` and an intact chain.

**2.2 Output directory** is `runs/vadv_onpolicy_rerun/` instead of the
registered `runs/vadv_onpolicy/`. Operational only: the registered path
holds the 2026-08-27 VOID banks under `VOID_DO_NOT_SCORE.txt`, and the
published verdict document cites receipts at those paths. Nothing else
about the artifact schema changes.

**2.3 NC-b's acting range is re-derived on the repaired banks BEFORE
scoring — procedure and decision rule fixed here, in advance.** The
predecessor's ninth vacuous gate: `gx(s) == gx(s')` everywhere made the
frozen mask universally true, `NEG_gx_frozen` came back bit-identical to
`PC_B5` at 26/26 iterates, and the inherited cap rule ("a signature stands
only if NC-b is COLLAPSED") therefore forced INDETERMINATE **by
construction** before any checkpoint loaded.

*Procedure (critic-free, no `eta2`, no checkpoint):* after collection and
before any critic is loaded, compute per iterate, from the bank arrays
alone: the `NEG_gx_frozen` row/cell sets exactly as the scorer constructs
them (frozen rows within `PC_B5` minus cells with any gx-moving row), and
`PC_B5`'s row/cell sets.

*Decision rule (registered now):* NC-b is **MEASURABLE** at an iterate iff
(a) its row set is **not identical** to `PC_B5`'s (identity is the 2026-08-27
defect's own signature), and (b) it has **>= 6 qualifying rows in >= 2
cells** — the registration's own cell-rule minimum, i.e. the smallest data
on which the statistic exists at all.

* If NC-b is MEASURABLE on >= 70 % of collected iterates → the inherited
  disposition applies unchanged: at each iterate a measurable NC-b that is
  not COLLAPSED caps that iterate's reading at INDETERMINATE.
* If NC-b is UNMEASURABLE on more than 30 % → NC-b **cannot act as a
  verdict-path condition on this data**, and the following re-specification
  applies, decided here and not after: **an UNMEASURABLE NC-b at an iterate
  is reported as `NC-b: UNMEASURABLE` and does not cap that iterate; a
  MEASURABLE NC-b that fails to collapse caps exactly as inherited.**
  Rationale: the inherited cap assumed NC-b measurable (offline it had 612
  frozen rows). A control that structurally cannot be measured on the data
  in front of it neither collapses nor fails to collapse; letting its
  absence cap every signature would be the tenth vacuous gate — a
  verdict-path condition whose acting range excludes its own data. The cap
  from a measurable, live NC-b is untouched in every case.

Whatever branch fires, the per-iterate MEASURABLE/UNMEASURABLE table is
published with the verdict.

**2.4 Log hygiene** (Lane B post-mortem carried over): every phase writes
its stdout to a fresh path no other process holds open; any log a phase
must read while another process writes it is copied first.

## 3. Registered expectations, stated before the run

* **The no-penetration branch remains the LIKELY branch** (~1,040 prior
  episodes never exceeded gx 2674, `min == max` at all 26 iterates). If it
  recurs: INTERIOR is VOID, never COLLAPSED; a positive measurement;
  evidence for neither hypothesis. Pre-paid, cannot be spent later.
* **Trajectory identity check (diagnostic, not a gate):** the aliasing
  corrupted recording, not behaviour — action selection consumed the same
  buffer contents either way, and seeds are identical. The re-run's
  per-episode sidecars are therefore expected to match the VOID run's
  (outcomes, `max_gx`, `pen_rate = 0.0`, `gx_max = 2674`). A mismatch is
  reported and investigated as nondeterminism; it moves no verdict.
* **A1 free reproducibility check:** offline PC-1 on
  `runs/interference/success_1_1.npz` must reproduce eta2
  0.6668875582236998 (ten significant figures, twice prior).
* **If this SOUND run is INDETERMINATE under the arc rule, §11 fires and
  V_adv retires from B5** — the registration's own words, applied to the
  first execution in which the instrument actually runs.

## 4. What is on disk when this is done

| path | what |
|---|---|
| `runs/vadv_onpolicy_rerun/iter_*.npz` + `_episodes.json` | 26 repaired per-iterate banks, guard-checked at write time |
| `runs/vadv_onpolicy_rerun/collect_summary.json`, `probe_933.json` | collection + probe receipts |
| `runs/vadv_onpolicy_rerun/ncb_acting_range.json` | the §2.3 pre-scoring derivation, written before any `eta2` |
| `runs/vadv_onpolicy_rerun/A1/`, `arc_scored.json(.jsonl)`, `arc_verdict.json` | gates, per-iterate readings, arc verdict |
| `docs/research/` verdict document | written after the numbers, against the inherited numerals |
