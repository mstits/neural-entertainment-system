# ITEM SEMANTICS ENGINE — SYNTHESIS (2026-08-25)

Synthesized from two independent designs: `ITEM_SEMANTICS_MINIMAL_2026-08-25.md` ("MINIMAL," Lens A —
graft item bits onto the already-shipped `state_sig`/`RoomIndex.EdgeStat` machinery) and
`ITEM_SEMANTICS_INDEPENDENT_2026-08-25.md` ("INDEPENDENT," Lens B — a standalone `BitLedger` +
`SpliceHarness` module that touches zero existing files in v1). Both were written independently
(INDEPENDENT explicitly did not have access to MINIMAL's text) against the same HEAD.

All line anchors below re-verified against current HEAD `27c81d0` (`scripts/go_explore_solve.py`,
7401 lines — byte-identical to `163ef54`, the commit both source designs verified against; confirmed
via `git diff 163ef54 HEAD -- scripts/go_explore_solve.py` returning empty). Every citation quoted
from either source design below was independently re-checked against the live file, not trusted:
`RoomIndex.record_edge` :925 (dict-construction :953-956, `e["count"] += 1` :961 — exact),
`Solver._room_step` `fp_edge` staging :3843 (exact), `Solver._room_transit` `record_edge` call :3872
(exact), `RoomIndex.load()` :1012-1044 (per-edge `.get(...)`-defaulted reconstruction, confirmed —
every field already tolerates a missing key), `RoomIndex.to_json()` :983-995 (confirmed the `"adj"`
line serializes each edge's *raw* dict wholesale, so an additive field on the live dict rides along
with no `to_json` edit needed), `GenericGame.cell_fn` :2253-2267 (`sig` built into cell key index
`-3` of a 5-tuple — confirmed by counting: `(area, hp, sig, y_band, gx_bucket)`, so `cur_key[-3]` is
exactly the `sig` slot), `LINEAGE_KEY_AXES` :302-318 (`sig_arity`/`sig_sha` axes present, confirmed),
`RoomIndex.lookup()` :909 (confirmed "FROZEN read... never interns"), `python.rs` `apply_state_guarded`
:37, `get_ram_range` :709-718 (confirmed hard `PyValueError` on `end > 0x0800`), `peek_oam` :766,
`pool.rs` `save_worker_state` :1515 / `load_worker_state` :1542 — all exact. `poke_ram`/`poke_ram_range`:
confirmed zero hits anywhere in `nes_core` or `scripts/`, only in proposal prose. `GenericGame` is used
by every non-SMB profile (`go_explore_solve.py:2483`, `return GenericGame(profile) if "solve" in
profile else SmbGame(profile)`) — confirmed `configs/zelda.yaml` and `configs/metroid.yaml` both run
`GenericGame`, so `state_sig` is a roster-wide mechanism, not a Castlevania-only feature (currently
used by `castlevania.yaml`, `contra.yaml`, `contra_blank.yaml`, `bubble_bobble.yaml`,
`ghosts_n_goblins.yaml`, `tetris_b.yaml`; absent from `zelda.yaml`/`metroid.yaml` today, addable
without a new mechanism). `docs/receipts/games/zelda_onboarding_2026-08-10.md` §4/§8 confirmed: every
banked Zelda start-state (`zelda_start*.state.bin`, including `zelda_sword_start.state.bin`, which
turns out to be a duplicate of the empty-inventory `zelda_start.state.bin`, not a post-pickup state)
reads `rupees`/`keys` HUD `x0` throughout every probe rollout — no pickup has ever been captured.
`docs/receipts/room_graph/RG1_zelda_2026-08-25.md` confirmed: overall disposition PASS (one sub-check
VOID per this project's VOID≠FAIL discipline) — the room layer is genuinely banked, not aspirational.

---

## Part I — Verdict on the two designs

| Axis | MINIMAL (state_sig graft) | INDEPENDENT (BitLedger module) |
|---|---|---|
| (a) Purity compliance | **9/10** | 8.5/10 |
| (b) Implementation risk given real code (10 = safest) | **8/10** | 7/10 |
| (c) P(a gate passes this week, real Zelda) | **~0.6** | ~0.2 |
| (d) Generality across roster | 7/10 | **8/10** |

**Winner: MINIMAL as chassis; INDEPENDENT's behavioral-verification and classification ideas grafted
on.** Rationale:

- **MINIMAL wins on shipping physics, and by a wider margin than its own text argues.** Every
  integration point it cites was re-verified above to the line, and each one is a demonstrably
  additive, non-version-bumping change: `EdgeStat.cap_hist` is a new dict key that `to_json()` already
  serializes for free (raw-dict wholesale passthrough) and that `load()` already tolerates missing
  (every existing field is `.get(...)`-defaulted, not required). No `RoomIndex.VERSION` bump, no
  resume-refusal migration, no key-shape change. Critically, it rides `state_sig` — a mechanism already
  in production for Castlevania, already covered by `LINEAGE_KEY_AXES`'s `sig_arity`/`sig_sha` resume
  guard, and already a `GenericGame`-level (roster-wide) feature rather than a bespoke one. It also has
  the only gate in either design that produces a **real receipt this week with zero new prerequisite**:
  IS-1a runs Stage 1-2 over already-banked Zelda traces and idle captures, no new rollout needed, no
  mint required.
- **INDEPENDENT has the better *detector engineering* and, correctly, identifies MINIMAL's single
  biggest weakness.** Its `BitLedger` idle-prefilter (a mask frozen once per lineage, exactly like
  `room_fp`'s calibrated mask, computed by reusing `Discoverer.idle()`) is a cleaner first pass than
  running the monotonicity scan over everything and filtering after. More importantly, its
  `SpliceHarness.verify_behavioral` — matched real acquire/skip trajectory pairs replayed from a shared
  `save_worker_state` pre-event snapshot, scored against a random-bit control battery drawn
  preferentially from the rejected pool — is a genuinely **stronger, still purity-legal, still
  zero-new-Rust** evidentiary class than MINIMAL's Stage 3, which is pure passive correlation over
  organically-collected `cap_hist` counts. MINIMAL's own §6 names "rarity vs. gating confound" as its
  most important failure mode; INDEPENDENT's Track A is close to a direct answer to exactly that
  failure mode, using only primitives (`save_worker_state`/`load_worker_state`) both designs already
  confirm are shipped. This is the one idea worth paying INDEPENDENT's extra implementation cost for.
- **INDEPENDENT's architectural independence from `room_fp` is a real generality point, but a
  narrower one than its own comparison table claims.** MINIMAL's Stage 3 (the room-edge-gating claim)
  does require `room_fp`/Room-Graph to be armed for the game in question — but Zelda (RG-1 PASS) and
  Metroid (RG-2, report-only but live) both already have it, so for the two games this document's gate
  actually targets, the coupling costs nothing today. The independence argument matters for the *next*
  game onboarded without a Room-Graph profile yet, which is why this synthesis keeps
  INDEPENDENT's odometer-bbox location as a documented fallback (§2, §8) rather than adopting it as v1's
  primary path.
- **INDEPENDENT's self-critique of "presumed Lens A" does not land on the actual MINIMAL design.**
  INDEPENDENT's §0 argues against rekeying `RoomIndex.adj` on `(src, dst, kind, dir, cap_sig)` — a
  real risk, but not what MINIMAL proposes. MINIMAL never touches the adjacency key shape at all; that
  rekey is explicitly named in both designs as v2-deferred (`(dst, cap_sig)` in MINIMAL §7, the "fusion
  join proper" in INDEPENDENT §8). Once the actual text is compared, the risk gap between the two
  designs' v1 surfaces is real but smaller than INDEPENDENT estimated it would be.
- **Purity:** both are careful and roughly tied. MINIMAL is marginally more conservative — it proposes
  no intervention of any kind in v1 (pure read of organically-produced rollouts), while INDEPENDENT's
  Track A actively selects and replays matched trajectory pairs from real prior data (still legal,
  still no fabricated input sequence — `inconclusive` rather than fabrication when no real skip segment
  exists — but a slightly more active experimental design). Both correctly keep the literal hard-splice
  mechanism (`poke_ram`/`poke_ram_range`) — the one input both source docs and the prior SIMCE lineage
  flag as the "most attackable" purity claim — off the table for v1.

---

## Part II — The synthesized design

### §1 Thesis

An item-semantics engine needs three things, none of which require knowing what a byte means: (1) find
candidate capability bits from our own rollouts without being told an address, (2) raise a candidate's
evidentiary status from "correlated with a spatially-displaced effect" to "behaviorally verified to
cause one" using only already-shipped primitives, and (3) write down what was found — address, bit,
confidence, verification method — in a form a router or a receipt can use without ever assigning a
name. This document builds that as one additive graft onto the shipped `RoomIndex`/`state_sig` chassis
(MINIMAL) plus one small, standalone verification harness reusing `save_worker_state`/
`load_worker_state` (INDEPENDENT's Track A) — not a new hot-loop subsystem, not a new Rust accessor.

### §2 Data structures

**A. `EdgeStat.cap_hist` — additive field (MINIMAL, unchanged from its design).**

```python
EdgeStat = {
    ...,                          # every existing field untouched
    "cap_hist": dict[str, int],   # str(cap_sig_int) -> traversal count.
                                   # absent/{} reads as {"0": count} for
                                   # every pre-existing edge — no VERSION
                                   # bump, no resume refusal.
}
```

`cap_sig` is not a new observable: it is the already-computed `sig` slot (index `-3`) of the last
observed cell key, read at the moment an edge stages — the same `{addr, match, mod}` bits a profile's
`state_sig:` block already declares.

**B. `ItemBitCandidate` — offline discovery ledger (MINIMAL's shape, INDEPENDENT's idle-prefilter and
rediscovery-rule bookkeeping folded in).**

```python
# scripts/discover_item_bits.py (offline, no hot-loop presence)
class ItemBitCandidate:
    addr: int; bit: int | None        # bit=None for a raw-value (state_sig-style) candidate,
                                        # set for a bit-plane candidate — both shapes supported,
                                        # promotion target is always a {addr, match, mod} triple
    match: frozenset[int]; mod: int
    idle_excluded: bool                 # True if flipped under the frozen per-lineage idle mask
                                        # (INDEPENDENT §2) — excluded before the expensive scan runs
    reverts_seen: int                   # >0 anywhere ⇒ PERMANENTLY rejected (rediscovery rule:
                                        # stays rejected until re-derived from scratch, the same
                                        # convention metroid_purity_quarantine_2026-08-10.md uses)
    change_rate: float
    monotone_rollouts: int; total_rollouts: int
    status: "candidate" | "confirmed" | "rejected"
    first_seen: dict                    # {rollout_id, step, position: [x, y]}
```

Persisted as `docs/receipts/item_bits/<game>_candidates.json` (raw ledger) + a human receipt
`docs/receipts/item_bits/<game>.md`, mirroring `room_fp_calibrate.py`'s convention.

**C. `verify_behavioral` — Track A verification (grafted from INDEPENDENT, scoped down to the one
track this document ships).**

```python
def verify_behavioral(pool, candidate: ItemBitCandidate, *, pre_state: bytes,
                       acquire_actions: list[int], skip_actions: list[int],
                       probe_actions: list[int], n_trials: int = 20,
                       control_bits: int = 5) -> dict:
    """From a shared pre-flip save-state (Pool.save_worker_state), run n_trials
    x {ACQUIRE, SKIP} pairs — both real, previously-observed action sequences
    mined from the ledger's own rollout history, never authored — each
    followed by probe_actions, plus control_bits matched trials against
    REJECTED-POOL bits (INDEPENDENT §7.9's collision-avoidance rule) as the
    false-positive control. Returns
    {reach_rate_acquire, reach_rate_skip, control_reach_rates,
     verdict: 'validated'|'rejected'|'inconclusive'}.
    Verdict: validated iff reach_rate_acquire - reach_rate_skip >= 0.5 AND
    max(control_reach_rates) - min(control_reach_rates) < 0.2. Returns
    'inconclusive', never a fabricated trajectory, if no real skip segment
    exists in the ledger's rollout history for this candidate."""
```

No new Rust: `pre_state`/`acquire_actions`/`skip_actions`/`probe_actions` all come from
`save_worker_state`/`load_worker_state` (`pool.rs:1515`/`1542`) and real rollout logs. The literal hard
splice (`poke_ram_range`, forcing the bit true in an untouched state) stays deferred to v2 (§8) exactly
as both source designs independently concluded.

**D. Flag vs. counter split (INDEPENDENT's framing, adopted verbatim).** A capability *flag* (the
`ItemBitCandidate`/`cap_hist` target above) is boolean and monotone for the rest of the episode. A
*counted resource* (rupees, stacking keys) goes up and down and is not this module's target at all —
it routes to the existing `discover_observables.find_progress` methodology, run in "resource mode"
(Gate 1 without the flat-under-reversal requirement). One log, two event kinds, correctly discriminated
— never conflated, per Zelda's own onboarding receipt naming this exact ambiguity without resolving it.

### §3 Integration points (file:line, verified at `27c81d0`)

| # | Where | Change |
|---|---|---|
| 1 | `RoomIndex.record_edge` (:925), dict construction (:953-956), count increment (:961) | Add `cap_sig: int = 0` parameter; init `"cap_hist": {}`; after `e["count"] += 1`, add `k = str(int(cap_sig)); e["cap_hist"][k] = e["cap_hist"].get(k, 0) + 1`. Default `0` ⇒ byte-identical for every non-item-sig caller. |
| 2 | `Solver._room_step`, `fp_edge` staging (:3843) | `fp_edge` tuple gains one field (currently 7 elements, indices 0-6 per the verified `record_edge(e[0]..e[4], exemplar_cell=e[5], exemplar_actions=e[6])` call at :3872): append `cap_sig = int(c["cur_key"][-3]) if (self._item_sig_armed and c.get("cur_key")) else 0` as `e[7]`. |
| 3 | `Solver._room_transit`, `record_edge` call (:3872) | Pass `cap_sig=e[7]`. |
| 4 | `RoomIndex.load()` (:1012-1044) | Add `"cap_hist": {str(k): int(v) for k, v in (e.get("cap_hist") or {}).items()}` to the per-edge reconstruction dict. `to_json()` needs no change (wholesale raw-dict passthrough, confirmed). |
| 5 | `Solver.__init__`, beside `_room_psig_off` (:3390) | `self._item_sig_armed = bool(getattr(args, "item_sig_report", False)) and self.game.state_sig_arity > 0`. Default `False`. |
| 6 | argparse, near `--room-*` flags (:7182 region) | `--item-sig-report` (store_true, report-only, no effect on selection/score/keys). |
| 7 | New file `scripts/discover_item_bits.py` | Stages 1-2 (candidate proposal + cross-rollout confirmation, §2B) + Stage 3a (`cap_hist` correlational read, MINIMAL's original design) + Stage 3b (`verify_behavioral`, §2C, called only on Stage-3a `candidate-gated` bits — the correlational read is the cheap first filter, behavioral verification is the expensive confirming step, run only where Stage 3a already found a lead). Reuses `discover_observables.Discoverer` (idle-prefilter, driving primitives) as a library import, never a patch. |
| 8 | `GenericGame` / profile `state_sig:` | No code change. Confirmed candidates are appended to the same YAML list Castlevania already uses. |

**What this deliberately does not touch, carried forward from both source designs:** `_xram`/
`_xram_local` (no pseudo-RAM bytes — every candidate address is real console RAM already read the way
`state_sig` reads it); `room_sig`/`room_id()` (an item-bit address must never enter `room_sig:` —
named again in §6 as the most likely misuse); `RoomIndex.adj`'s key shape / `RoomIndex.VERSION` (no
rekey on `(dst, cap_sig)` in v1 — that is the pre-declared v2 escalation, §8); `nes_core` (zero Rust
changes — `poke_ram_range` stays unbuilt, §8).

### §4 Profile schema

```yaml
solve:
  state_sig:                       # EXISTING mechanism — confirmed item bits are pasted here,
    - {addr: 0x0000, match: [1], mod: 0}   # exactly like Castlevania's stair-climb bit
  item_bits:                       # NEW, optional — thresholds for discover_item_bits.py only;
    max_change_rate: 0.01          # absent ⇒ script defaults apply, nothing else changes
    confirm_k: 3
    confirm_n: 5
    max_item_bits: 8
    verify:
      n_trials: 20
      control_bits: 5
      verdict_gap: 0.5
      control_flatness: 0.2
```

`--item-sig-report` (CLI flag, §3 row 6) is the only new switch that touches the live solver process;
`item_bits:` is read exclusively by the offline `discover_item_bits.py` script.

### §5 Purity audit

| Input | Class | Verdict |
|---|---|---|
| RAM bytes `$0000`-`$07FF` (`get_ram_range`, `python.rs:709`, hard `PyValueError` above `0x0800`) | Hardware surface, core-enforced boundary | LEGAL — and structurally harder than a policy choice: Metroid's disassembly-sourced, quarantined item-flag addresses (`metroid_purity_quarantine_2026-08-10.md`) are unreachable through this accessor regardless of stance |
| Monotonicity / change-rate / idle-prefilter / dedup filters | Computed from our own rollout RAM logs, same class as `find_progress` Gates 1-2 | LEGAL |
| Cross-rollout `K`-of-`N` stability bar | A receipted reproducibility threshold, same convention as RG-0's fixture-count bars | LEGAL |
| `cap_hist` / `RoomIndex` edges | Derived solely from our own traversed rollouts | LEGAL |
| `verify_behavioral`'s acquire/skip/probe action sequences | Mined exclusively from our own prior rollout logs; `inconclusive`, never fabrication, when no real skip segment exists | LEGAL — same discipline as the Contra objective-key falsifier amendment |
| Bit/byte identity | Never assigned a name; ledger stores `(addr, match, mod)` + a confidence receipt only | LEGAL — rediscovery-rule convention |
| Hard splice (`poke_ram_range`, deferred) | Forcing a bit true in a state where it has not yet occurred | Carries forward the "known-outcome amendment" tension both source docs flag as this lineage's most attackable claim — **not built in v1**, re-examined on its own before it ever ships |

### §6 Pre-registered validation gate

Register verbatim in `runs/item_bits/PREREG.md` before any live run (locked once written; corrections
are dated addenda, never edits) — same discipline as `runs/room_graph/PREREG.md`.

**IS-0 — offline falsifier (BINDING, blocks all live runs).** `tests/test_is0_itembits.py`, zero live
compute:
1. Genuine monotone flag (0 for N steps, 1 forever) → proposed as a candidate.
2. Frame-counter-shaped byte (changes every step) → rejected on change-rate.
3. Flickering bit (`1→0→1` once) → rejected on monotonicity/`reverts_seen`.
4. Address already claimed by an existing observable → deduped.
5. Idle-prefilter over the real, already-banked `tests/fixtures/roomgraph/{zelda,metroid}_idle_fs1.npz`
   captures → post-filter candidate count is exactly zero (idle play produces no capability events).
6. Cross-rollout stability: 2-of-5 stays `candidate`; 4-of-5 promotes to `confirmed`.
7. Scripted-oracle fixture: `verify_behavioral` validates a true causal bit and rejects a decoy that
   also flips once, using the control-bit mechanism, before any real `Pool` is involved.
8. `RoomIndex.record_edge`/`load()` round-trip: `cap_hist` survives save/load exactly; a pre-existing
   archive with no `cap_hist` key loads as `{"0": count}`, never a `KeyError`.
9. `--item-sig-report` absent (default) ⇒ `record_edge` called with `cap_sig=0` on every path; the
   existing SMB/Castlevania 16,000-step determinism harness stays byte-identical.

Any failure ⇒ stop, no live compute.

**IS-1a — Zelda negative control (runnable today, zero new prerequisite).** Run Stages 1-3a over the
already-banked Zelda probe traces and `roms/zelda_start_ctrl.state.bin` bursts (`zelda_onboarding_
2026-08-10.md` §4 confirms every existing rollout shows `rupees`/`keys` at `x0` throughout — a
ready-made true-negative). **Pass:** zero candidates promoted to `confirmed`, and the
`--item-sig-report`-off byte-identity check (IS-0 item 9) reproduced live, not just synthetically.

**IS-1b — Zelda real semantic gate (BLOCKED on a named prerequisite, numbers pre-registered now).**

- *Prerequisite (§9 Task 5, not this week):* mint ≥1 Zelda start-state within reach of a real
  overworld rupee or dungeon key, via an idle-vs-active NT-tile volatility diff (`room_fp_calibrate.py`
  method, applied to the HUD digit boxes `zelda_onboarding_2026-08-10.md` §8 already names as
  unconfirmed hypotheses: rupees ≈ rows 23-32 × cols 88-110; keys ≈ rows 38-47).
- **Pass criteria:** over ≥10 rollouts spanning the minted state — sensitivity (confirmed candidate's
  first-flip step within 5 steps of the HUD-tile flip) ≥ 0.8; specificity (no-flip rollouts producing
  zero confirmed candidates) ≥ 0.9; ≥1 confirmed bit reaches `verify_behavioral` verdict `validated`
  against a real dungeon-door/locked-screen edge.
- **Kill criteria:** sensitivity < 0.5 on two independent minted-pickup batches ⇒ RAM-diff proposal
  falsified for Zelda's inventory encoding at this scan granularity (possible split/BCD encoding,
  §7.3) — lane stops with receipt, hard splice remains the fallback, not a re-tune. All IS-1a
  candidates resolve `inconclusive` under `verify_behavioral` ⇒ the probe-location framing needs
  rework, not evidence the bits are wrong.

**IS-2 — Metroid (secondary, report-only, non-blocking).** Same protocol; because Metroid's
disassembly-sourced item bytes are confirmed unreachable through `get_ram_range` (§5), a
zero-candidate result is itself the receipted answer to an open question, not a lane failure. No kill
criterion pre-registered.

### §7 Failure modes named in advance

1. **Confound: rarity vs. gating.** An edge that never fires at `sig=0` may just be far from where
   `sig=0` workers explore, for reasons unrelated to any gate. Mitigated by Stage 3a's boundary-visit
   cross-reference, and materially reduced (not eliminated) by requiring `verify_behavioral` to reach
   `validated` before any bit is claimed gated — a correlational lead alone (Stage 3a) is never
   sufficient for a claim on its own.
2. **Multi-item / partial-key gates.** A door needing two items shows up as `cap_hist` scattered with
   no single bit cleanly separating crossed/uncrossed. `MAX_ITEM_BITS = 8` leaves room for a v1.1
   conjunction test; not built here (also INDEPENDENT §8's "joint/multi-bit" deferral).
3. **Split/BCD-encoded counters.** A digit pair fails the single-byte monotone test as cleanly as a
   single-byte encoding might pass by accident. `mod` support (borrowed from `state_sig`) handles the
   nibble case; a genuine two-byte joint candidate is out of scope for v1 and is IS-1b's named kill
   path, not a silent absorption.
4. **RAM-window blindness (Metroid-class).** `get_ram_range` masks to `$07FF`; any true capability byte
   in bankswitched PRG-RAM is invisible by construction. No amount of tuning fixes this — it needs a
   wider accessor, out of scope.
5. **False-monotone from an unswept lifetime counter.** A slow accumulator monotone across a short test
   rollout could pass Stage 1-2 if every rollout is short relative to its true period. Mitigation: vary
   rollout *length*, not just seed, across confirmation rollouts.
6. **Misuse: an item-bit address added to `room_sig`.** The single most likely profile-authoring
   mistake — corrupts `room_id()`'s transit logic, looks like a burst of new "rooms" at every pickup.
   Documentation-level tripwire, no structural guard proposed (would require inferring authorial
   intent from YAML).
7. **`cur_key` staleness at burst boundaries.** `cur_key` is `None` until the first `observe()` call;
   `cap_sig` defaults to `0` via the guard — under-counts as ungated, the conservative direction.
8. **Control bits that are themselves meaningful.** A "random other bit" used as a control could be a
   real, different capability flag, failing the control-flatness check for the wrong reason. Guard:
   control bits drawn preferentially from the `rejected` pool (permanently-excluded, reverted bits)
   before falling back to untested ones (INDEPENDENT §7.9).
9. **`verify_behavioral`'s skip arm may not exist.** If no real rollout ever walked past the item
   without touching it, the harness returns `inconclusive` rather than fabricating a trajectory — this
   can stall verification on corridor-shaped items with no natural "walk past" geometry. Named
   honestly: a hard splice (v2, needs no skip trajectory) is the fallback for exactly this case.
10. **Telemetry cost.** One additional full-lock scan over `idx.adj` per status line — negligible
    relative to Room-Graph's already-measured perf budget (RG-1d), named per this codebase's habit of
    pre-declaring costs rather than discovering them live.

### §8 Deferred to v2

`poke_ram_range` — a new Rust accessor mirroring `get_ram_range`'s bounds check (`0..0x800`), with an
explicit post-write liveness probe (N steps of NOOP must not panic/hang) since `apply_state_guarded`
(`python.rs:37`) only catches panics from *deserializing* a state blob, not from stepping a live NES
after a targeted in-RAM mutation — a new guard, not a reused one (INDEPENDENT §7.2/§8). Rekeying
`RoomIndex.adj` on `(dst, cap_sig)` so one physical room can route to different destinations under
different item states (MINIMAL §7's "D2" plan). Extending `BitLedger`'s scan window to OAM
(`peek_oam`, `python.rs:766`) for sprite-slot-encoded capability signals (INDEPENDENT §8). A random-bit
false-positive control battery run against real hard splices (20 rollouts/bit). Joint/multi-bit
conjunction verification. A runtime OR-latch register for item bits, should live evidence (not
assumed here) show `state_sig` reads reverting at restore boundaries often enough to need one.
Trigger for escalating to any of this: IS-1b passes cleanly for one game AND a second game's gate
fails specifically for want of a splice-grade causal claim — build the expensive mechanism only once
the cheap one has demonstrably run out of runway.

### §9 Implementation plan (5 tasks)

| Task | Scope | Deps | Done-when |
|---|---|---|---|
| **1 — `EdgeStat.cap_hist` graft** | §3 rows 1-6 (`go_explore_solve.py` + argparse); `tests/test_is0_itembits.py` items 8-9 | — | IS-0 items 8-9 pass; flags-off SMB/Castlevania determinism harness byte-identical to pre-change HEAD |
| **2 — candidate scanner (Stages 1-2)** | `scripts/discover_item_bits.py`: idle-prefilter (reusing `Discoverer.idle()`) + monotonicity/change-rate/dedup filters + cross-rollout confirmation; `tests/test_is0_itembits.py` items 1-6 | — | IS-0 items 1-6 pass on synthetic fixtures + real idle captures |
| **3 — Stage 3 (correlational lead + `verify_behavioral`)** | `discover_item_bits.py` Stage 3a (`cap_hist`/boundary-visit read) + Stage 3b (`verify_behavioral`, §2C, using `save_worker_state`/`load_worker_state`); `tests/test_is0_itembits.py` item 7 | 1, 2 | IS-0 item 7 passes against a scripted oracle; a 5-minute real-`Pool` smoke exercises the save/load round-trip with zero crashes |
| **4 — IS-1a Zelda negative control** | Run Task 2-3 machinery over existing banked Zelda fixtures; file receipt | 1-3 | Zero confirmed candidates; `--item-sig-report`-off byte-identity confirmed live; receipt filed — **the checkpoint achievable this week with no new prerequisite** |
| **5 — IS-1b prerequisite: Zelda pickup mint** | `room_fp_calibrate.py`-style NT-tile-diff probe producing a receipted Zelda start-state adjacent to a real rupee/key; register IS-1b's numbers verbatim in `runs/item_bits/PREREG.md` so the gate is shovel-ready the moment the state exists | 4 (can start in parallel with 2-3; only the *gate launch* depends on 1-4 being green) | `docs/receipts/item_bits/zelda_mint.md` filed; IS-1b registered but explicitly not run by this plan — named so the next agent does not re-derive it |

Checkpoints mirror Room-Graph's own: after Task 1 (byte-identity), after Tasks 2-3 (IS-0 green,
zero live compute), after Task 4 (the only checkpoint runnable this week without a new prerequisite,
entirely against already-banked data). No `nes_core` rebuild anywhere in this plan — fully compatible
with a machine currently saturated by the v28 training campaign.
