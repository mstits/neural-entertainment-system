# IS-1a — Zelda negative control: VERDICT

**Disposition: FAIL** (registered gate, `ITEM_SEMANTICS_ENGINE_2026-08-25.md` §6, "IS-1a").
Not committed — orchestrator commits.

IS-1a requires BOTH, conjunctively:
1. Zero candidates promoted to `confirmed` over already-banked Zelda data.
2. The `--item-sig-report`-off byte-identity check (IS-0 item 9) reproduced live.

(2) passes cleanly. (1) does not. Full receipts below; raw data in this
directory (`is1a_stage12_3a_receipt.json`, `is1a_rollout_logs.npz`).

## Method

**Data source ("already-banked Zelda probe traces"):** the four real RG-1
Zelda runs' `traces.pkl` (`runs/room_graph/rg1_zelda_seed{0,1}_bias{025,000}/`)
— genuine 90-minute Go-Explore archives, each keyed cell holding the REAL
mined action sequence from `roms/zelda_start_ctrl.state.bin` (the "…bursts"
half of the task) to that cell. `traces.pkl` stores action bytes + metadata,
not RAM — so 12 representative traces (3 per run: shortest / median / longest
by action-sequence length, spanning 5–2573 actions) were replayed through a
real, single-worker `nes_core.Pool` from the real root state to reconstruct
genuine per-step RAM($0000–$07FF) logs. No action was authored — every
sequence replayed is exactly what the real archive already banked.

**Idle-prefilter:** one short (~5 s) single-worker `idle_mask_from_rom` probe
against the same rom/state — 340 addresses excluded before scanning.

**Claimed-address dedup:** `{0x0070, 0x0084, 0x0670}` — `configs/zelda.yaml`'s
own measured `progress.lo` / `y` / `lives`, per `zelda_onboarding_2026-08-10.md`.
Not new addresses; not consulted from the disassembly-sourced `ram_mapping:`
block (which is checked only AFTER the fact, for reporting, never as scanner
input — see the doctrine cross-check below).

**Stage 1–2** (`confirm_across_rollouts`, `confirm_k=3` default) ran twice:
- **RAW** — the 12 replayed rollouts exactly as recorded.
- **DEATH-TRUNCATED** — each rollout cut at the first frame where the
  ALREADY-CLAIMED `lives` byte (`$0670`) reads 0, mirroring
  `discover_observables.py`'s own settle/reset-threshold discipline for this
  exact game (its docstring names Zelda's death→CONTINUE-menu animation as
  its hardest reset-detection case). No new address, no new game knowledge.

**Stage 3a** (`correlate_boundary_edges`) ran over the four real, banked
`room_index.json` files from the RG-1 runs, for every `bit_index` 0–7.

**Byte-identity (IS-0 item 9, live):** wall-clock `--minutes` budgets are not
step-count-reproducible while v28 training contends for CPU on this machine,
so a dedicated harness (`run_byteid_fakeclock.py`) rebinds the module-local
`time` name (never a global monkeypatch) to a fixed-delta synthetic clock,
making the deadline cross after an exact, CPU-speed-independent step count.
Baseline = `git show HEAD:scripts/go_explore_solve.py` (pre-graft, Tasks 1–3
uncommitted); treatment = the working-tree `scripts/go_explore_solve.py`
(post-graft). Same seed(0)/profile/root-state/workers(1); `--item-sig-report`
absent on both.

## Results

### Stage 1–2 — RAW (no truncation)
**607 keys scanned → 51 confirmed, 18 candidate, 538 rejected.**

### Stage 1–2 — DEATH-TRUNCATED (the fair run)
4 of 12 rollouts truncated (a real death occurred in-window).
**224 keys scanned → 13 confirmed, 15 candidate, 196 rejected.**

**Root cause of the RAW→TRUNCATED drop (A):** manual cross-reference shows
several of the 51 raw confirmations share a first-flip step landing ~20–130
steps after `lives` ($0670) drops to 0 in that same rollout — i.e., they are
the RAM/HUD/menu-state rewrite of Zelda's death→CONTINUE-menu animation
(`zelda_onboarding_2026-08-10.md` §2 already names this as the hardest reset
case in this whole codebase), not any capability event. Truncating at death
removes the shared tail and the false positives it drove.

**Root cause of the residual 13 (B):** every one of them is proposed as
"monotone" by 8 of the 8 (or more) rollouts that survived truncation, at
IDENTICAL elapsed-step offsets — 7, 53, or 69 — despite each of those 8
rollouts replaying a genuinely different, independently-mined real action
sequence. A gameplay-contingent flag fires at a time that depends on what the
player did; a flag that fires at the SAME elapsed step regardless of 8
different real trajectories is the signature of a deterministic, root-clock
(engine-init / sound-engine / animation-parity) artifact, not an item pickup.
None of the 13 addresses are anywhere near the rupees/keys region.

**Doctrine cross-check (reporting only — never scanner input):** rupees
(`0x066D`) and keys (`0x066E`), the disassembly-sourced `ram_mapping:`
addresses, were checked post-hoc against all 12 replayed rollouts —
**neither ever changed value in any rollout.** The "no real pickup has ever
been captured" premise this gate is registered against (§6, citing
`zelda_onboarding_2026-08-10.md` §4) holds. The 13 residual false positives
are not a symptom of Zelda secretly having a detectable pickup; they are two
independently-identified, non-item, non-purity RAM-event classes that the
literal Stage 1–2 spec (§9 Task 2) does not yet exclude.

### Stage 3a — real banked `room_index.json` (all 4 RG-1 runs)
| run | edges | `cap_hist` present | leads (any bit) |
|---|---|---|---|
| seed0_bias025 | 2065 | No (pre-graft run) | 0 |
| seed0_bias000 | 1160 | No | 0 |
| seed1_bias025 | 2086 | No | 0 |
| seed1_bias000 | 1399 | No | 0 |

Expected and trivial: these four runs predate the `cap_hist` graft (Task 1)
entirely, so every edge's `cap_hist` reads as the empty default — zero
exposure at any bit, hence zero leads at any `bit_index`. Confirms
`correlate_boundary_edges` degrades safely (no crash, no false lead) on
pre-graft data.

### Byte-identity (IS-0 item 9) — reproduced live
Deterministic-fake-clock run, both sides identical seed/profile/root-state,
`--item-sig-report` absent:

| artifact | result |
|---|---|
| `steps` | 5919 == 5919 |
| `done:` summary (cells/frontier/best_score/records/new_cells/improvements) | identical |
| `archive.pkl` | sha256-identical |
| `traces.pkl` | sha256-identical |
| `progress.jsonl` | byte-identical |
| `archive.stats.json` | byte-identical |
| `roots.json` | byte-identical |
| `room_index.json` | identical on every field of all 3 committed edges EXCEPT the additive `cap_hist` key — post-graft `cap_hist == {"0": count}` on every edge (exactly the traversal count, no other bucket populated); pre-graft has no `cap_hist` key at all |

Matches Task 1's own documented claim exactly: flags-off is byte-identical
everywhere except the intentionally-additive `cap_hist` sidecar. **PASS,
reproduced live** — not just asserted by the unit suite.

## Verdict

| Criterion | Result |
|---|---|
| Zero candidates promoted to `confirmed` | **FAIL** — 51 raw / 13 after a fair, already-claimed-observable-only mitigation |
| `--item-sig-report`-off byte-identity, live | **PASS** |
| **IS-1a overall** | **FAIL** (conjunctive gate; not VOID — the measurement is real, unconfounded, and answers the registered question) |

This is not a purity violation and not evidence Zelda has a detectable item
pickup — the true-negative premise (rupees/keys stay at 0) holds throughout.
It is a real, root-caused finding: Stage 1–2, exactly as specified in §9
Task 2, needs two additional mitigations before its "zero confirmed" bar is
trustworthy on real (non-idle, non-death-excluded) rollout data of a game
with a death→menu animation like Zelda's:

1. **A death/lives-based rollout truncation ahead of `scan_rollout`**,
   mirroring `discover_observables.py`'s own settle/reset-threshold
   discipline for this exact game.
2. **A cross-rollout transition-step consistency check** that treats a
   candidate whose first-flip step is suspiciously identical across
   independently-diverging real lineages as evidence of a root-clock
   artifact, not a promotion vote.

Neither mitigation requires new game knowledge, a new address, or new Rust —
both use only already-claimed observables and already-mined data. Filed here,
not built here; picking this lane back up should start from these two items,
not from IS-1a's raw numbers alone.

## Files

- `is1a_stage12_3a_receipt.json` — full machine-readable ledger (raw +
  truncated Stage 1-2, Stage 3a per-run, byte-identity, verdict).
- `is1a_rollout_logs.npz` — the 12 real replayed RAM logs (for anyone who
  wants to re-derive the root-cause analysis above without re-replaying).
- `run_is1a_stage12_3a.py` — the Stage 1–3a driver (replay + scan + Stage 3a).
- `run_byteid_fakeclock.py` — the deterministic byte-identity driver.
- `fc_pre/`, `fc_post/` — the fake-clock byte-identity run outputs
  (pre-graft / post-graft).
- `byteid_pre/`, `byteid_post/`, `byteid_pre.log`, `byteid_post.log` — an
  earlier, WALL-CLOCK-timed byte-identity attempt, superseded by the
  fake-clock method above because `--minutes` budgets are not
  step-count-reproducible under concurrent v28 training CPU contention
  (cells 213 vs 226 on two runs of the SAME pre-graft code) — kept for the
  record, not used for the verdict.
