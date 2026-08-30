# ITEM SEMANTICS — MINIMAL EXTENSION (2026-08-25)

Design lens: **extend, don't build.** Three prior documents (`V23_SYNTHESIS_2026-08-23.md`,
`MEMORY_ARCHITECTURE_2026-08-23.md`, and the original Deep Research answer they operationalize,
`research-consult/responses/20260817T225101Z_v16_roomgraph_premise_update.md`) converge on the same
"Layer 3" mechanism — SIMCE: (1) 3-probe stability filtering, (2) irreversible-bit tracking, (3)
counterfactual splicing verification (`poke_ram` a candidate bit into a blocked save-state, re-run,
check whether reachability extends). All three docs are unimplemented — zero lines of code, per a
repo-wide grep for `BitLedger`/`cap_sig`/`poke_ram`/`irreversible` that returns hits only inside the
proposal prose itself. `ROOMGRAPH_ENGINE_2026-08-24.md` §8 explicitly defers this whole mechanism to
"its own lane with its own gates... once the room layer is banked" — which, per RG-1's 2026-08-25 PASS
verdict, it now is.

This document is **not** a plan to build that lane. `poke_ram` does not exist in `nes_core` (confirmed:
zero hits outside proposal docs), building it means a Rust accessor plus a rebuild, and this session's
sequencing rule — *no core rebuild while a live compute run is saturating the machine, cheap
premise-falsifiers before big workflows* — rules that out today regardless of merit. This document
instead asks: **what is the smallest graft onto the already-shipped, already-gated `RoomIndex` /
`classify_transition` machinery that gets a real (if weaker) version of "picking up X at P changes what
happens at Q," using only what already runs?** The answer below adds one read-only offline discovery
script, one additive field on `EdgeStat`, an eight-line edit to two existing hot-loop call sites, and
zero new Rust. Everything else is `state_sig` — shipped since before Room-Graph existed — doing the one
job it was already built to do.

All line numbers below verified against HEAD `163ef54598bdd50db798dc4280b6e54ce93da0ee`
(`scripts/go_explore_solve.py`, 7401 lines).

---

## Part I — Why this is the minimal graft, not a smaller SIMCE

### The one-sentence idea

`state_sig` already conditions a Castlevania cell's identity on a discovered `{addr, match}` bit — "is
the player mid-stair-climb" — read fresh from RAM every step, folded into `cell_fn`'s `sig` slot, and
already a first-class lineage axis (`sig_arity`/`sig_sha`, `LINEAGE_KEY_AXES` :304-305). An item flag
("do you have the key") is, mechanically, *the same kind of bit* — a `{addr, match}` predicate over real
console RAM — with one added property: once true, a real item flag never goes back to false for the rest
of the episode (that is what makes it an *item* rather than a *mode*). So the entire "new axis" the task
asks for is: **find `{addr, match}` predicates that are additionally monotone, and feed them into the
mechanism that already exists**, rather than parallel-building a second `state_sig`.

### Why this is *not* just "run `discover_observables.py` again"

`discover_observables.Discoverer` (script:160) already has the RAM-diff, stability-filtering, and
gameplay-vs-noop-gating machinery this needs — Stage 1 below is a new probe class riding that same
harness, not a new harness. What it does *not* have, and what `RoomIndex`/`classify_transition` supply
for free, is a notion of **spatially-displaced effect**: "the bit's value is entangled with what happens
at a specific room edge, not just with how RAM looks." That correlation is the actual deliverable named
in the task ("picking up X at P permanently changes what happens at Q"), and it rides entirely on
telemetry the Room-Graph engine already collects per traversed edge (`exemplar_cell`, `count`,
`frames_mean`) — one more counter alongside those is the whole extension.

### Why this is weaker than SIMCE, on purpose

SIMCE's step 3 is a **causal** claim: splice the bit into a save-state where the edge was previously
blocked, re-run, and watch the classifier's verdict flip. This design's step 3 is a **correlational**
claim over rollouts that were never intervened on: "every observed crossing of this edge co-occurred
with this bit being set; no crossing was ever observed with it unset." That is real, receipt-grade
evidence — the same evidentiary class as everything else in this codebase's purity ledger (probe-
measured, never asserted) — but it is confound-prone in a way splicing is not (§8, failure mode 1). The
document names this trade explicitly rather than smuggling a causal-sounding claim out of correlational
machinery: this is the deliberate cost of shipping a lane with **zero new Rust** this week instead of
a `poke_ram` build next month.

---

## Part II — The design

### §1 Data structures

No new top-level class. Three small additions, one of them a *file*, not code:

```python
# NEW FILE: scripts/discover_item_bits.py (offline, no hot-loop presence)
#
# ItemBitCandidate — a proposed-then-scored predicate, SAME SHAPE as a
# state_sig entry so promotion is a copy-paste, not a translation:
#   addr: int                       # real RAM byte, $0000-$07FF only
#   match: frozenset[int]           # values the byte takes once "true"
#   mod: int                        # 0 = raw value (state_sig convention)
#   first_seen: (rollout_id, step, room_ordinal | None)
#   monotone_rollouts: int          # rollouts where it fired 0->1 and
#                                    # never reverted, of...
#   total_rollouts: int              # ...rollouts it was visible in at all
#   change_rate: float               # fraction of sampled steps where the
#                                    # tested value changed at all (the
#                                    # clock/counter rejector, §2 Stage 1)
#   status: "candidate" | "confirmed" | "rejected"
#   reject_reason: str | None        # populated on rejection (dup address,
#                                    # high change-rate, unstable)
#
# Persisted as docs/receipts/item_bits/<game>_candidates.json (raw ledger,
# every candidate ever proposed, for audit) plus a human receipt
# docs/receipts/item_bits/<game>.md (the room_fp_calibrate.py convention:
# a ready-to-paste YAML block + the evidence that earned it).
```

```python
# scripts/go_explore_solve.py — RoomIndex.EdgeStat, ADDITIVE field only.
# Every existing key ("kind", "dir", "count", "frames_mean",
# "exemplar_cell", "exemplar_actions", "validated", "validate_attempts")
# is untouched. `cap_hist` is new:
EdgeStat = {
    ...,                      # unchanged existing fields
    "cap_hist": dict[str, int],   # str(cap_sig_int) -> traversal count.
                                   # {} / absent-key reads as {"0": count}
                                   # for every edge recorded before this
                                   # change or on a profile with no
                                   # confirmed item bits — NO VERSION BUMP,
                                   # NO resume refusal, because a reader
                                   # that has never seen the key treats a
                                   # missing "cap_hist" as "everything was
                                   # cap_sig 0," which is exactly what was
                                   # true before this field existed.
}
```

No `ItemLedger` runtime class, no per-worker latch register, no new pseudo-RAM bytes, no `_xram` /
`_xram_local` change. `cap_sig` is computed by re-reading the **already-computed** cell key's `sig`
slot at the moment an edge stages (§2 row 2 below) — it is not a new observable, just a new place an
existing observable gets recorded.

### §2 Integration points (file:line, verified at `163ef54`)

| # | Where | Change |
|---|---|---|
| 1 | `RoomIndex.record_edge` — signature :925, `EdgeStat` construction :954-957, count increment :961 | Add parameter `cap_sig: int = 0`. In the dict-construction branch, initialize `"cap_hist": {}`. After the existing `e["count"] += 1` (:961), add `k = str(int(cap_sig)); e["cap_hist"][k] = e["cap_hist"].get(k, 0) + 1`. Four lines total. Default `0` means every non-item-sig caller (SMB, every game without an `item_sig` block) is unaffected — `record_edge(...)` with no `cap_sig` argument behaves byte-identically. |
| 2 | `Solver._room_step`, edge-staging tuple :3843-3845 | `c["fp_edge"]` gains one field: `cap_sig = int(c["cur_key"][-3]) if (self._item_sig_armed and c.get("cur_key")) else 0`, appended as the 8th tuple element. `cur_key[-3]` is `GenericGame.cell_fn`'s `sig` slot (see §1 note on cell-key layout below) — the item bits ride the *already-computed, already-threaded* `sig` value from the last observed cell, not a fresh RAM read, so this line touches no RAM accessor at all. Using the **pre-transition** `cur_key` (one step stale by construction — `cur_key` is set at the end of `observe()`, `_room_step` runs before that step's RAM fetch per §3 row 4 of the Room-Graph doc) is a deliberate choice: it tags the edge with the item state *held approaching the door*, matching the onset-baseline convention already used for `d_odo`/`d_scene` (fp_settle's onset fields, :742-775) rather than whatever the sig reads one step after arrival. |
| 3 | `Solver._room_transit`, `record_edge` call :3872-3874 | Pass `cap_sig=e[7]`. One-line diff. |
| 4 | `RoomIndex.load()`, per-edge reconstruction :1035-1044 | Add `"cap_hist": {str(k): int(v) for k, v in (e.get("cap_hist") or {}).items()}`. `to_json()` (:983-994, the `"adj"` line at :992) needs **no change** — it already serializes each edge's raw dict wholesale (`{str(d): e for d, e in dsts.items()}`), so `cap_hist` rides along automatically once it exists on the live object. |
| 5 | `GenericGame.__init__`, `state_sig` parse :1841-1865 | **No code change.** Confirmed item-bit candidates are appended to the SAME `solve.state_sig` YAML list the profile already declares for mode bits. This is the entire "conditions room/cell identity" requirement — `cell_fn`'s `sig` (built at :2263-2267, one bit per `state_sig` entry) already folds into every archived cell's key, and every archived cell's key already threads room identity via `psig`/`area` (Room-Graph §3 row 6, `observe()` key composition :4150-4157, untouched). `LINEAGE_KEY_AXES`' existing `sig_arity`/`sig_sha` axes (:305-306, `state_sig_sha` :322-336) automatically start refusing cross-lineage resume the moment an item bit is added to a profile's `state_sig` — zero new lineage code. |
| 6 | `Solver.__init__`, beside the room_fp arm block (:3390-3401, the `self._room_psig_off` derivation) | One new attribute: `self._item_sig_armed = bool(getattr(args, "item_sig_report", False)) and self.game.state_sig_arity > 0` — a flag, not a subsystem, that only decides whether row 2's four-line graft fires. Default **False**: a profile with `state_sig` entries but no `--item-sig-report` flag gets ordinary CV-style mode-bit behavior, unchanged; `cap_hist` stays `{"0": N}` for every edge. |
| 7 | `explore()` status-line telemetry, Room-Graph block :6913-6931 | One additive line inside the existing `if room_fp is not None` guard: `line["edges_cap_gated"] = sum(1 for dsts in idx.adj.values() for e in dsts.values() for k, v in e.get("cap_hist", {}).items() if k != "0" and e["cap_hist"].get("0", 0) == 0 and v > 0)` — a live count of "edges whose every recorded traversal carried a nonzero cap_sig and none carried zero," i.e. edges that currently *look* item-gated. Cheap (same lock-held scan the existing telemetry block already does), report-only, no decision rides on it. |
| 8 | argparse, near the existing `--room-*` flags :7182-7201 | `ap.add_argument("--item-sig-report", action="store_true", help="Tag Room-Graph edges with the profile's state_sig value at staging (cap_hist) and surface edges_cap_gated telemetry. No effect on selection, score, or keys — report-only.")`. |

**What this deliberately does *not* touch**, and why each is correct to leave alone:

- `_xram` / `_xram_local` (:3604 / :3653) — no pseudo-RAM bytes are introduced; every candidate address
  is real console RAM in `$0000`-`$07FF`, read the same way `state_sig` already reads Castlevania's
  stair-climb bit. Room-Graph needed pseudo-addresses because odometer integrals and interned ordinals
  have no hardware address; an item flag already has one.
- `room_sig` / `room_id()` (:1662, :2469-2471) — an item-bit address must **never** be added to a
  profile's `room_sig:` list. `room_id()` change is what fires a sect/psig transit (Room-Graph rides
  this for the ordinal exactly the way SMB rides `$074E`); folding an item bit into it would make every
  pickup look like a room transition and corrupt the adjacency layer for a reason with nothing to do
  with rooms. Named again as failure mode 6 below because it is the single easiest way to misuse this
  design.
- `RoomIndex.adj`'s nesting / `RoomIndex.VERSION` — deliberately not bumped. The full "D2" plan (§8 of
  Room-Graph) rekeys `adj` on `(dst, cap_sig)`, which is more expressive (lets the SAME source room have
  *different destinations* under different item states, not just a gated/ungated single destination) but
  is a schema version bump, a resume-refusal migration, and roughly triples the surface touched for a
  case this design's target instances (a locked door either exists or it doesn't) do not need. Named
  explicitly as the v2 escalation path in §7.

### §3 The discovery protocol — proposing and confirming a candidate purely from rollout data

Three stages, each cheap, each running on data the repo already knows how to produce. No stage reads a
disassembly, a wiki, or a byte's "meaning" — every verdict is a statistic over our own RAM traces.

**Stage 1 — candidate proposal (RAM-diff mining, a new `Discoverer` probe class).**

Riding `discover_observables.Discoverer`'s existing harness (script:160: one headless worker, cached
per-probe RAM logs, `RAM_SIZE = 0x800` — the same `$0000`-`$07FF` window every other observable in this
codebase is found in): for a batch of independently-seeded exploration rollouts (not scripted drives —
genuine `go_explore_solve.py` bursts, or, cheaper, replayed banked traces from an existing run's
`traces.pkl`), log every RAM byte at the room_fp settle cadence (or every step if room_fp is off) and
test each of its 8 bit-planes for:

1. **Monotonicity-within-episode.** The bit is 0 for a prefix of the rollout and 1 for the entire
   remainder — a single 0→1 transition, never reverting, for the rest of *that* rollout. (Multi-valued
   counters — e.g. a key/rupee count that only ever increments — are tested the same way against a
   `match = {k, k+1, k+2, ...}` threshold set, exactly `state_sig`'s `(addr, match, mod)` shape.)
2. **Low change-rate.** The tested predicate flips true on fewer than `max_change_rate` (default 1%,
   receipted per game) of sampled steps across the whole rollout. This is the clock/frame-counter
   rejector — the exact trap `find_progress`'s Gate 1 was built to catch for progress bytes (module
   docstring :7-22), reapplied to bit-planes instead of monotone-climbing bytes. A byte that changes on
   nearly every one of 8 bit positions and every step is a counter or an RNG stream, not a flag.
3. **Dedup against already-claimed addresses.** Any address already in the profile's `progress` /
   `room_sig` / `y` / `lives` / `boss_hp` / existing `state_sig` list is excluded — re-deriving a known
   observable is not a new candidate.
4. **Transition-window stability (new, specific to this stage vs. plain `discover_observables`).** The
   candidate's value is additionally required to be flat across every `RoomIndex`-classified `fade`/`pan`
   churn window in the rollout (cross-referenced against the SAME room_index.json the run already
   produces) — a byte that spikes only during screen-transition churn is animation/HUD-redraw noise, not
   an item state, and this check is unique to running the scan on a room_fp-armed rollout rather than a
   bare probe.

Every byte/bit-plane surviving all four checks becomes an `ItemBitCandidate` at `status: "candidate"`.

**Stage 2 — cross-rollout confirmation.**

Repeat Stage 1 over `N` independent rollouts (different seeds/workers; the same "our own rollouts, no
scripted replay of the same trajectory" discipline the room_fp mask calibration already uses). A
candidate promotes `candidate -> confirmed` only if the identical `(addr, match, mod)` triple survives
Stage 1 in `>= K` of `N` rollouts (default `K=3, N=5`, receipted per game, mirroring RG-0's "banked probe
fixtures" reproducibility bar). Confirmed candidates are capped at `MAX_ITEM_BITS = 8` per profile
(ties broken by earliest first-seen step, discovery-order — the exact convention `RoomIndex.intern`
already uses for room ordinals); candidates beyond the cap are logged, never promoted — a `cap_hits`-
style degrade-to-telemetry, not a crash.

**Stage 3 — behavioral-relevance (the "gates a room edge" claim, correlational).**

This is where §2 row 2-4's `cap_hist` graft earns its keep, and it *requires* a rollout run with
`--item-sig-report` and the Stage-2-confirmed bits already pasted into `solve.state_sig`, so the bit
value is actually threaded through `cur_key` into staged edges. Over that run's `room_index.json`, for
every edge with `len(cap_hist) > 1` (traversed under more than one observed `sig` value) or with
`cap_hist == {"nonzero": N}` only (traversed but *never* under `sig=0`):

- **Positive evidence:** an edge whose `cap_hist` never contains key `"0"` despite `N >= min_crossings`
  (default 5) traversals is flagged *candidate-gated* on whichever bit(s) are set in every observed
  `cap_sig` value.
- **Negative evidence (the harder, still-honest half):** cross-reference against telemetry the router
  already collects for free — `_room_sides`/boundary-cell visit counts (Room-Graph §3 row 7,
  `_refresh_sel_cache` :5113) and the archive-wide distribution of `sig` values (a trivial one-pass scan
  of any archive dump, no new tracking). If workers spent a comparable number of visits near that room's
  boundary *while* `sig=0` was common elsewhere in the same archive, and the edge still never committed
  under `sig=0`, that raises confidence the gating is real rather than an artifact of the router simply
  never sending an unconfirmed-item worker there. This is presence/absence correlation over
  already-recorded telemetry, not a new hot-loop write.
- **Uncorrelated-but-confirmed bits are expected and fine.** A confirmed monotone flag that never
  entangles with any edge (most rupee/point counters) stays in the ledger as `confirmed, uncorrelated`
  — it is a legitimate irreversible observable, just not one this rollout happened to show gating
  anything. This is the intended, common case, not a failure.

The receipt (`docs/receipts/item_bits/<game>.md`) records, per confirmed-and-gated bit: the address,
match set, the `(src, dst, kind, dir)` edge(s) it gates, `cap_hist` counts, and the boundary-visit
cross-reference numbers — every field a number this repo actually measured, none a name.

### §4 Purity audit

| Input | Class | Verdict |
|---|---|---|
| RAM bytes `$0000`-`$07FF` (`Discoverer`/`Pool.get_ram`, python.rs:703) | Hardware surface | LEGAL (same window every existing observable uses) |
| Monotonicity / change-rate / dedup filters | Computed from our own rollout RAM logs | LEGAL — experiment-discovered, same class as `find_progress`'s Gate 1/2 |
| Cross-rollout stability threshold (`K` of `N`) | A reproducibility bar we set and receipt per game, same convention as RG-0's fixture-count bars | LEGAL |
| `RoomIndex` edges, `cap_hist`, boundary-visit counts | Derived solely from our own traversed rollouts | LEGAL — same class as the already-shipped edge exemplar/aliasing machinery |
| The `sig`/`cur_key` value fed into `cap_sig` | Already-shipped, already-legal `state_sig` mechanism (Castlevania in production) | LEGAL — no new observable, new *use* of one |
| Bit *names* ("has key", "has bomb") | — | **NEVER ASSIGNED.** The ledger stores `(addr, match, mod)` and a confidence receipt, exactly the `q_*`/rediscovery-rule convention `metroid_purity_quarantine_2026-08-10.md` already established for this exact failure mode. |
| Metroid's disassembly-sourced `item_flags 0x6878` / `missiles 0x6879` (quarantined, `metroid_purity_quarantine_2026-08-10.md` :17-22) | External RAM map | **NOT USED, AND STRUCTURALLY UNREACHABLE.** `get_ram` masks to `$07FF`; the quarantined addresses live above that window. This extension inherits that block for free — it cannot propose a candidate outside `$0000`-`$07FF` — but it also means Metroid's *actual* equipment bitfield (if it is truly at `0x687x`) can never be found by this mechanism at all. Named as failure mode 4. |

### §5 Pre-registered validation gate

Naming convention: **IS-0** (offline falsifier, mirrors RG-0) and **IS-1** (live gate, mirrors RG-1),
"IS" = item-sig. Register verbatim in `runs/item_bits/PREREG.md` before any live run, same discipline
`runs/room_graph/PREREG.md` already uses (locked once written; corrections are dated addenda, never
edits).

**IS-0 — offline falsifier (blocks all live runs; BINDING).** Synthetic byte-sequence fixtures as a
pytest (`tests/test_is0_itembits.py`), covering, at minimum:

1. A genuine monotone flag (0 for `N` steps, 1 forever after) → proposed as a candidate.
2. A free-running frame-counter-shaped byte (changes every step) → rejected on change-rate, never
   proposed.
3. A flickering bit that goes `1 -> 0 -> 1` within one rollout (HUD blink, transient invuln) → rejected
   on monotonicity — this is the fixture that stands in for the "no runtime latch" design decision: if a
   byte reverts even once inside the confirmation window, it must never be trusted bare, and rejecting it
   at Stage 1 is the whole reason no OR-latch is needed for anything that survives to `confirmed`.
4. An address already claimed by an existing observable (progress/room_sig/y/lives) → deduped, never
   proposed as new.
5. Cross-rollout stability: a candidate appearing in exactly 2 of 5 synthetic rollouts stays
   `"candidate"` (below `K=3`); appearing in 4 of 5 promotes to `"confirmed"`.
6. Synthetic edge-traversal log: an edge crossed only when a bit is set (zero crossings at `sig=0`,
   `>=5` at `sig=1`) → flagged `candidate-gated`; a control edge crossed uniformly regardless of the bit
   → correctly **not** flagged.
7. `RoomIndex.record_edge`/`load()` round-trip: an archive saved with `cap_hist` populated, then loaded,
   reproduces every count exactly; an archive saved **before** this change (no `cap_hist` key at all)
   loads with `cap_hist == {"0": count}` reconstructed, never a `KeyError`.
8. `--item-sig-report` absent (default) ⇒ `record_edge` called with `cap_sig=0` on every path;
   `SmbGame`/existing Castlevania profiles' 16,000-step determinism harness stays byte-identical to
   pre-change HEAD (the same regression class RG-1c already runs, reused here rather than re-invented).

Any failure ⇒ stop, no live compute.

**IS-1a — Zelda negative control (runnable today, no new prerequisite).** `zelda_onboarding_2026-08-10.md`
§4 (:255-256) states plainly that every existing Zelda probe rollout shows `rupees`/`keys` HUD counters
reading `x0` throughout — **no pickup has ever occurred in a banked Zelda rollout.** This is a ready-made
true-negative fixture: run Stage 1-2 over the existing banked Zelda probe traces and `roms/
zelda_start_ctrl.state.bin` bursts. **Pass requires:** zero candidates promoted to `confirmed` (there is
nothing to find — any confirmed candidate here is a bug in the filter, not a discovery) **and** the
`--item-sig-report` flag off-by-default byte-identity check (IS-0 item 8) reproduced live, not just in
the synthetic fixture. This also exercises §2 row 6's arming flag end-to-end on real telemetry at zero
risk of a false claim.

**IS-1b — Zelda real semantic gate (BLOCKED on a named prerequisite, not run by this document).**
Register the numbers now so the run is shovel-ready the moment the prerequisite clears:

- **Prerequisite** (not built here): a probe-captured Zelda start-state within a short rollout of a
  genuine overworld rupee or a dungeon key — none exists in `roms/` today (`zelda_start.state.bin` and
  `zelda_start_ctrl.state.bin` both begin with an empty inventory). Minting one is a `room_fp_calibrate.
  py`-style probe task, out of scope for this design doc and for the current live-compute freeze.
- **Independent ground truth (hardware-legal, built once the prerequisite lands, no new mechanism):** an
  NT-tile-diff oracle over the HUD digit boxes `room_fp_calibrate.py`'s volatility-mask method already
  finds automatically — `zelda_onboarding_2026-08-10.md` §8 item 2 (:385-390) already names the
  approximate screen regions (rupees ≈ rows 23-32 × cols 88-110; keys ≈ rows 38-47) as a *to-do*, not a
  claim; this reuses that exact idle-vs-active NT-tile variance trick that already found the heart-HUD
  byte, purely as an evaluation label for the gate below — never as the discovery mechanism itself (that
  stays Stage 1's RAM-diff scan).
- **IS-1b pass criteria (pre-registered numbers):** over `>=10` rollouts spanning the minted pickup
  state, sensitivity (fraction of true HUD-tile-flip rollouts producing a confirmed candidate whose
  first-flip step is within 5 steps of the HUD-tile flip) `>= 0.8`; specificity (fraction of no-flip
  rollouts producing zero confirmed candidates) `>= 0.9`; at least one confirmed bit reaches Stage 3
  `candidate-gated` status against a real dungeon-door or locked-screen edge in the Room-Graph adjacency.
- **Kill criteria:** sensitivity `< 0.5` on two independent minted-pickup batches ⇒ the RAM-diff
  proposal mechanism is falsified for Zelda's inventory encoding (candidate byte may be split across a
  BCD-digit pair the current single-byte scan cannot see — named as failure mode 3) — lane stops with a
  receipt, `poke_ram`/splice-based SIMCE remains the fallback. Zero `candidate-gated` edges after 3
  confirmed, uncorrelated-or-better bits and `>=90` min of Room-Graph-armed exploration ⇒ Stage 3's
  correlational bar is too weak for Zelda's actual door layout (doors may require multiple keys plus a
  boss key at conflated addresses) — not a kill of the whole mechanism, but router/telemetry-side work
  is needed before claiming a gate (see failure mode 2).

**IS-2 — Kirby / secondary (report-only, non-blocking, optional).** Kirby's door problem
(`LEAGUE_ONBOARDING_WAVE1_2026-08-24.md`) is item-free — no key/lock semantics — but running IS-0/IS-1a's
harness against it (once `configs/kirby_roomfp.yaml` exists, itself unbuilt, survey rank #3) is a free
regression check that the candidate scanner reports **zero** confirmed item bits on a game that has none
in its already-probed rollouts, the same "true negative" logic as IS-1a.

### §6 Failure modes named in advance

1. **Confound: rarity vs. gating.** An edge that "never fires at `sig=0`" may simply be far from where
   `sig=0` workers ever explore, for reasons unrelated to any gate (router bias, distance, difficulty) —
   this design's whole weakness relative to splice-based SIMCE. Mitigated, not eliminated, by the
   boundary-visit cross-reference in Stage 3; named explicitly rather than let a `candidate-gated` label
   read as proven-causal. `edges_cap_gated` telemetry (§2 row 7) is a lead, never a claim on its own.
2. **Multi-item / partial-key gates.** A door needing *two* items (a small key AND the boss key) or an
   item whose effect depends on a second, uncorrelated bit (e.g. only usable in daytime) will show up as
   `cap_hist` scattered across several nonzero values with no single bit cleanly separating crossed vs.
   uncrossed — Stage 3's single-bit correlation test under-detects compound gates. `MAX_ITEM_BITS = 8`
   leaves room to test small conjunctions in a v1.1 pass without a schema change; not built here.
3. **Split/BCD-encoded counters.** A HUD digit pair (tens digit, ones digit) stored as two separate
   bytes fails the single-byte "monotone value" test as cleanly as a single BCD byte with two nibbles
   might pass it by accident of encoding. Stage 1's `mod` support (borrowed verbatim from `state_sig`)
   handles the nibble case; a genuine two-byte counter needs a two-address joint candidate, explicitly
   out of scope for v1 (named as an IS-1b kill criterion above, not silently absorbed).
4. **RAM-window blindness (Metroid-class).** `get_ram` masks to `$07FF` (`metroid_purity_quarantine_
   2026-08-10.md` :20-22); any true capability byte living outside that window (mapper-bankswitched
   PRG-RAM, as Metroid's disassembly-sourced-and-quarantined addresses appear to be) is invisible to
   Stage 1 by construction, not by a bug. No amount of tuning the discovery protocol fixes this — it
   needs a wider RAM accessor, which is exactly the kind of core change this design explicitly avoids.
   Ranked #2 in the roster survey partly *because* of this blocker; this document does not remove it.
5. **False-monotone from an unswept lifetime counter.** A byte that happens to be monotone across THIS
   rollout's length but is actually a slow accumulator that would eventually wrap or reset on a much
   longer horizon (step counters, RNG-seed-derived odometers) could pass Stage 1/2 if every test rollout
   is short relative to its true period. Mitigation: vary rollout LENGTH, not just seed, across the `N`
   confirmation rollouts (a one-line change to Stage 2's rollout generator, noted as a build requirement,
   not a code line shown above since it lives entirely in the new offline script).
6. **Misuse: an item-bit address added to `room_sig`.** Named in §2 above as a "does not touch" item,
   repeated here as a failure mode because it is the most likely mistake a profile author makes: folding
   a candidate into `room_sig` instead of `state_sig` corrupts `room_id()`'s transit-firing logic and
   would look, in telemetry, like a burst of new "rooms" at every pickup. No code guard is proposed for
   this (it would require inferring authorial intent from a YAML list); it is a documentation-level
   tripwire, not a structural one — a gap this document accepts rather than hides.
7. **`cur_key` staleness at burst boundaries.** `_room_step`'s edge-staging reads `c["cur_key"]`, which
   is `None` at burst start until the first `observe()` call populates it (the same staleness the
   existing `fp_onset_key` exemplar mechanism already tolerates). `cap_sig` in that narrow window
   defaults to `0` via the `and c.get("cur_key")` guard in §2 row 2 — under-counts as ungated rather than
   over-counting as gated, which is the conservative direction for a correlational claim.
8. **Telemetry cost.** `edges_cap_gated` (§2 row 7) adds one more full-lock scan over `idx.adj` to the
   existing status-line block, which already does an equivalent-cost scan for `edges_pan`/`edges_fade` —
   negligible relative to Room-Graph's already-measured perf budget (RG-1d: SPS ≥ 90% baseline), but
   named per this codebase's habit of pre-declaring perf costs rather than discovering them live.

### §7 Deferred to v2 (unchanged from Room-Graph §8, restated with this design's vocabulary)

Everything SIMCE actually needs beyond this document: a `poke_ram`-class Rust accessor for save-state
RAM splicing; a `BitLedger` that tracks bits found this way as *causally verified* rather than
correlationally suggestive; rekeying `RoomIndex.adj` on `(dst, cap_sig)` so a single physical room can
route to genuinely different destinations under different item states (rather than this design's
single-destination gated/ungated read); a random-bit false-positive control battery run against real
splices (20 rollouts/bit, per V23's own number); a runtime OR-latch register for item bits, should IS-1's
gate reveal that raw `state_sig` reads revert often enough at restore boundaries to need one (no evidence
of this yet — Stage 1's monotonicity filter is the reason no latch is built pre-emptively). Trigger for
escalating to this v2: IS-1b passes cleanly for at least one game AND a second game's gate fails
specifically because a splice-grade causal claim is needed to resolve failure mode 1 or 2 above — i.e.
build the expensive mechanism only once the cheap one has demonstrably run out of runway, matching the
same "D0 chassis, D1/D2 grafts on demand" sequencing Room-Graph itself used.

### §8 Implementation plan

| Task | Scope | Deps | Done-when |
|---|---|---|---|
| **B1 — `EdgeStat.cap_hist` graft** | §2 rows 1-4, 6-8 (`go_explore_solve.py` + argparse); `tests/test_is0_itembits.py` items 7-8 | — | IS-0 items 7-8 pass; flags-off SMB/CV determinism harness byte-identical |
| **B2 — candidate scanner** | `scripts/discover_item_bits.py` Stage 1 (new `Discoverer` probe class); `tests/test_is0_itembits.py` items 1-4 | — | IS-0 items 1-4 pass on synthetic fixtures |
| **B3 — cross-rollout confirmation + Stage 3 correlation** | `discover_item_bits.py` Stages 2-3, reading `room_index.json` + boundary telemetry | B1, B2 | IS-0 items 5-6 pass |
| **B4 — IS-1a negative control** | Run B2-B3 over banked Zelda fixtures; receipt | B1-B3 | Zero confirmed candidates; byte-identity confirmed live |
| **B5 — IS-1b prerequisite (separate lane, not this document)** | Mint a Zelda pickup-adjacent start-state via a probe task; HUD-tile oracle | B4 | Out of scope here — named so the next agent can pick it up without re-deriving the gate |

Checkpoints mirror Room-Graph's own: after B1 (suite + flags-off byte-identity), after B2-B3 (IS-0
green), after B4 (the only checkpoint runnable this week without a new prerequisite). No `nes_core`
rebuild anywhere in this plan.
