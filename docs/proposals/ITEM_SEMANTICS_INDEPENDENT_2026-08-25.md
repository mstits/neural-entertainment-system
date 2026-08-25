# ITEM SEMANTICS — INDEPENDENT SUBSYSTEM (Design Lens B, 2026-08-25)

All line anchors below verified against HEAD `ee39fde` on 2026-08-25 (`scripts/go_explore_solve.py`,
7401 lines; `scripts/discover_observables.py`; `nes_core/src/python.rs`; `nes_core/src/pool.rs`).
Source lineage: `research-consult/responses/20260817T225101Z_v16_roomgraph_premise_update.md` (v16,
Deep Research — the original SAE+PDDL mechanism, never implemented), `docs/proposals/V23_SYNTHESIS_2026-08-23.md`
(SIMCE — operationalizes v16 into a 3-step irreversible-bit-tracking + counterfactual-splice
mechanism, unbuilt), `docs/proposals/MEMORY_ARCHITECTURE_2026-08-23.md` (restates SIMCE as "Layer 3 —
semantics," unbuilt), `docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md` (Layers 1+2, shipped, §8 names
the capability layer as pre-declared v2 work). This document is **Lens B: the independent-subsystem
design** — a standalone module with its own data structures, its own event log, its own gate, that
touches zero existing files in v1 and is fusable with the room-graph engine later only through a
read-only, optional join. A second document (Lens A, fused) is expected to propose grafting the
capability dimension directly onto `RoomIndex`/`EdgeStat`; this document does not assume access to
that text and stands on its own, but §0 states the axis the two lenses are judged against so a future
synthesis (in the shape of ROOMGRAPH_ENGINE's own Part I three-way judgment) can compare them fairly.

---

## §0 Framing — why independent, and what it costs

**The fused alternative (presumed shape of Lens A, from the shared gap analysis).** Extend
`RoomIndex.adj`'s key from `(src, dst, kind, dir)` to `(src, dst, kind, dir, cap_sig)` — an edge is
only traversable/committed under a specific capability-bitmask precondition. This is exactly what
ROOMGRAPH_ENGINE §8 pre-declares as "D2's capability bitmask keying... banked verbatim as the v2
roadmap" and is the natural reading of "later fusable with the room graph." Its advantage: the
capability axis lands exactly where it will ultimately need to act (gating which edges the router
treats as real), so there is no second join step. Its cost: `EdgeStat`'s schema, `record_edge`'s
signature, the lineage axis (`room_fp` config-sha), and every banked `room_index.json` on disk change
shape the moment capability tracking ships, even for games that never use it — and per §2's own
restore-lockstep invariants, a schema change to an append-only, restore-order-independent structure
is the single riskiest class of edit in that file (exactly the class D1's `rgid` key-prefix change was
rejected for in the original Part I judgment, for the same reason).

**This design (Lens B).** A new file, `scripts/item_semantics.py`, with its own classes
(`BitLedger`, `CapabilityEvent`, `SpliceHarness`), its own on-disk artifact
(`runs/item_semantics/<tag>/{bit_ledger.json,events.jsonl}`), its own profile block
(`solve.item_semantics:`), and its own gate (`IS-0`/`IS-1`/`IS-2`, mirroring `RG-0`/`RG-1`/`RG-2`'s
shape exactly). **Zero lines change in `RoomIndex`, `EdgeStat`, `record_edge`, or any lineage axis.**
The module's only dependency on existing code is read-only: `Pool.get_odometer()` (already shipped,
independent of `room_fp`) for spatial context, and `Pool.get_ram_range`/`save_worker_state`/
`load_worker_state` (already shipped) for the detection and verification loops themselves. It can run
on a profile that has never enabled `room_fp` at all — Layer 3 built on top of Layer 1 (the odometer)
alone, never requiring Layer 2 (the room graph) to exist.

**The cost of independence:** a capability event's "location" is, in v1, an odometer bounding box or
raw (x, y), not a room ordinal — coarser than what a fused design gets for free. Fusion is possible
later (§8) as a one-way read: item-semantics can look up `RoomIndex.lookup(hash)` if a profile happens
to have both blocks enabled, but `RoomIndex` never learns that item-semantics exists. This is the
trade this lens makes deliberately: slower to pay off architecturally, but the room-graph engine
(shipped, gated, RG-1 PASS) is never put back into flux to get it, and item-semantics can gate on
Zelda/Metroid (neither has `room_fp` proven live yet — RG-1 is Zelda-only, RG-2 Metroid is
report-only) without waiting on that lane's schedule.

| Axis | Lens B (this doc, independent) | Presumed Lens A (fused) |
|---|---|---|
| Touches `RoomIndex`/`EdgeStat` in v1 | No | Yes (key shape change) |
| Requires `room_fp` enabled to run at all | No | Yes |
| Risk to banked `room_index.json` / RG-1 receipts | None | Schema migration required |
| Location precision for an event | Odometer bbox (coarse) | Room ordinal (exact) |
| Steps to a gated edge precondition | Two (detect, then a later fusion pass) | One |
| New Rust required for the full mechanism | Same either way (§8, `poke_ram`-class accessor) | Same |
| Blast radius of a design defect | One new file + one new gate | The already-shipped room-graph engine |

---

## §1 Thesis

An item/inventory-capability semantics layer needs exactly three things, none of which require
knowing what a byte *means*: (1) a way to find candidate capability bits from our own rollouts without
being told an address, (2) a way to test whether a candidate bit *causes* a later, spatially-displaced
change in what the agent can reach, and (3) a place to write down what was found that survives restarts
and never needs bit *names* to be useful downstream (a planner or a router only needs "bit 41 gates
reaching bbox Q," never "bit 41 is the ladder").

This lens builds those three things as a closed module:

1. **`BitLedger`** — scans the legal 2 KB CPU-RAM window (`get_ram_range(0, 0x800)`, `python.rs:709`,
   hard-masked by the core itself, not a purity choice) at bit granularity (16,384 candidate bits),
   pre-filtered by an idle-stability pass (reusing `discover_observables.Discoverer`'s idle-probe
   methodology, `discover_observables.py:259`) to drop the frame-counter/animation-clock class before
   the expensive monotone scan ever runs, then tracks per-rollout which surviving bits go 0→1 and never
   revert. This is SIMCE step 1+2 made concrete (`V23_SYNTHESIS_2026-08-23.md`'s "3-probe stability
   filtering... we own this" + "irreversible-bit tracking... found structurally, never named").

2. **`SpliceHarness`** — two verification tracks answering the same causal question ("does reverting
   the candidate bit change reachability at a later probed location") at two different cost/rigor
   points:
   - **Track A (behavioral, ships in v1, zero new Rust).** From one shared pre-event save-state
     (`Pool.save_worker_state`, `pool.rs:1515`), run a matched pair of real rollouts — one that takes
     the action sequence known (from ledger evidence) to flip the candidate bit, one that reaches an
     odometer-matched alternate state without flipping it — then drive both toward the same later
     probed location and compare arrival. No RAM is ever written outside what real inputs produce, so
     every state visited is on-manifold by construction.
   - **Track B (hard splice, deferred to v2, §8).** The literal SIMCE mechanism — write the candidate
     bit true into an otherwise-untouched blocked save-state via a new `poke_ram_range` accessor,
     re-run, check for reachability extension, with random-bit false-positive controls. This is the
     cleaner counterfactual (holds *everything else* fixed) but needs a Rust accessor that does not
     exist anywhere in the tree today (confirmed: zero `poke_ram` hits outside proposal prose) and
     carries a real off-manifold risk this project has paid for once already (the 1-2 distill's
     "off-manifold drift barrier," DR v10, cited in-repo at
     `docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md:40`) — a flag forced true
     without the game state that normally accompanies it can desync invariants the ROM's own code
     never defends against, silently or as a crash.

3. **`CapabilityEvent` log** — an append-only JSONL per rollout tag, each record structural only
   (`{addr, bit, first_flip_step, position, rollout_id}` before verification; `{..., validated,
   method, trials, control_result}` after). No record ever carries a human-assigned name; a consumer
   (planner, router, a future fused edge annotation) reads `bit_id = (addr, bit)` and nothing else.

**A load-bearing distinction this design makes that the source docs do not**: "irreversible bit" and
"item/rupee/key counter" are different measurement classes and need different detectors. A capability
*flag* (sword, raft, ladder, magical key possession) is boolean and, once set, stays set for the run —
exactly the 0→1-never-reverts signature `BitLedger` targets. A *counted resource* (rupee total, bomb
count, key count in games where keys stack) goes up **and down** and is not a bit-level phenomenon at
all — it is exactly the monotone-with-Gate-2-saturation byte class `discover_observables.find_progress`
already detects (`discover_observables.py` PROGRESS docstring, Gate 1/Gate 2), except reversibility
must now be treated as *expected*, not disqualifying. This module does not reimplement that detector;
it emits a second event subtype, `"counter-delta"`, sourced from the *existing* monotone-counter
methodology run in "resource mode" (Gate 1 without the flat-under-reversal requirement), so both
classes land in one log with one schema, correctly discriminated. Zelda's own onboarding receipt
already names this ambiguity without resolving it (`docs/receipts/games/zelda_onboarding_2026-08-10.md`
§4, `rupees`/`keys` — "no pickup ever occurred... nothing to calibrate against").

---

## §2 Data structures (new file `scripts/item_semantics.py`; nothing added to `go_explore_solve.py`)

```python
RAM_BITS = 0x800 * 8          # 16,384 candidate bits — the entire get_ram_range(0, 0x800) legal window
IDLE_PREFILTER_STEPS = 300    # reuse Discoverer.idle() cadence; a bit that ever flips under pure NOOP
                               # (after settle_steps()) is a clock/render-timing bit, permanently excluded
NEVER_REVERT_MARGIN = 0       # a single 1->0 anywhere in the observed window disqualifies the bit for
                               # this rollout (conservative; see §7.6 for the "near-miss" logging path)

class BitLedger:
    """Per-tag (game/profile) accumulator, append-only/monotone like RoomIndex (§2 of
    ROOMGRAPH_ENGINE, same restore-lockstep shape): a candidate bit's status can only move
    UNVERIFIED -> CANDIDATE -> {VALIDATED, REJECTED}, never backward, so replaying a rollout out
    of order or resuming mid-run cannot corrupt it.

    idle_mask: np.ndarray[bool], shape (16384,)  # True = excluded by the idle prefilter (frozen
                                                  # per lineage the moment T4's calibration run
                                                  # produces it; NEVER recomputed mid-run, exactly
                                                  # like room_fp's calibrated mask)
    candidates: dict[(addr:int, bit:int), CandidateRecord]
    events: list[CapabilityEvent]      # append-only; flushed to events.jsonl on the same cadence
                                        # as RoomIndex.save() (archive.stats.json tick + exit)
    lock: threading.Lock
    """

class CandidateRecord(TypedDict):
    addr: int                 # 0..0x7FF
    bit: int                  # 0..7
    status: str                # "unverified" | "candidate" | "validated" | "rejected"
    first_seen_flip: dict | None    # {rollout_id, step, position: [x,y], odo_bbox_at_flip}
    reverts_seen: int          # count of 1->0 observations across ALL rollouts (a bit with reverts_seen
                               # > 0 anywhere is permanently excluded from future candidacy — the
                               # rediscovery rule from the Metroid quarantine doc's own convention:
                               # a rejected candidate stays rejected until re-derived from scratch)
    verification: dict | None  # filled by SpliceHarness; see below

class CapabilityEvent(TypedDict):
    kind: str                  # "flag-set" | "counter-delta"
    bit_id: tuple | None       # (addr, bit) for flag-set; None for counter-delta
    counter_addr: tuple | None # (lo, hi|None) for counter-delta, same shape find_progress reports
    rollout_id: str
    step: int
    position: tuple            # (odo_x, odo_y) at the event, ALWAYS present (Layer-1 only dependency)
    room_ordinal: int | None   # OPTIONAL soft join — filled only if the calling profile ALSO has
                               # room_fp active and passes the frozen RoomIndex in for a read-only
                               # RoomIndex.lookup(); None means "room-graph not in this profile" or
                               # "unrecognized hash," never a hard failure
    delta: int | None          # for counter-delta only
```

```python
class SpliceHarness:
    """Owns the counterfactual verification loop. Constructed with a Pool already at hand (the
    caller's, e.g. a standalone item-semantics probe script OR — later, optionally — the live
    solver's pool passed in read-only for a periodic verification pass); never constructs its own
    Pool, so it cannot desync the caller's worker indices.
    """

    def verify_behavioral(self, ledger: BitLedger, bit_id: tuple, *,
                           pre_state: bytes, acquire_actions: list[int],
                           skip_actions: list[int], probe_actions: list[int],
                           n_trials: int = 20, control_bits: int = 5) -> dict:
        """Track A (§1.2). `pre_state` is a save-state from BEFORE the bit's first observed flip
        (captured by the ledger's caller at `first_seen_flip.step - 1`, via save_worker_state — no
        new Rust). Runs n_trials x {ACQUIRE, SKIP} pairs, each followed by `probe_actions` (an
        action sequence previously observed, from real rollouts, to reach the later probed
        location Q when the bit is set) plus `control_bits` matched trials against RANDOMLY
        CHOSEN OTHER already-rejected-or-unrelated bits sharing the same acquire/skip action
        split, as the false-positive control (SIMCE's own requirement, V23_SYNTHESIS mechanism
        step 3). Returns {reach_rate_acquire, reach_rate_skip, control_reach_rates: list,
        verdict: 'validated'|'rejected'|'inconclusive'}. Verdict rule: validated iff
        reach_rate_acquire - reach_rate_skip >= 0.5 AND max(control_reach_rates) - min(...) < 0.2
        (the control arms must show FLAT reachability regardless of their own bit state — if a
        control also swings, the probe location itself is confounded, e.g. by a timer, and the
        whole test is inconclusive, not a false validation)."""

    def verify_splice(self, ledger: BitLedger, bit_id: tuple, *,
                       pre_state: bytes, probe_actions: list[int],
                       n_trials: int = 20, control_bits: int = 5) -> dict:
        """Track B (§1.2, DEFERRED — see §8). Same signature and verdict rule as
        verify_behavioral, but ACQUIRE = poke_ram_range(pre_state, {addr: byte_with_bit_set}) then
        probe_actions directly (no acquire_actions needed — the bit is forced), SKIP = pre_state
        unmodified then probe_actions. Raises NotImplementedError until the `poke_ram_range`
        accessor ships; the method exists now so the verdict-rule code path is identical and
        testable (mocked) before the Rust lands — see IS-0 fixture C, §6."""
```

Why `verify_behavioral` needs *both* `acquire_actions` and `skip_actions` rather than just replaying
the ledger's real trajectory twice: replaying the same trajectory twice is not a counterfactual, it is
a determinism check. The SKIP arm must be a **different, real, previously-observed** rollout segment
from the same pre-state that reaches an odometer-matched position without flipping the bit — the
ledger's own `events.jsonl` is mined for such a segment (a worker that walked past the item without
touching it) before a splice trial is attempted; if none exists, the harness returns `inconclusive`
rather than fabricating one, because fabrication would mean scripting an input sequence never taken by
our own search, which is exactly the authored-trajectory purity violation SIMCE's own Contra amendment
was written to forbid (`V23_SYNTHESIS_2026-08-23.md`, "the 'destroying' trajectory must come from OUR
OWN prior search rollouts... never from a scripted/authored input sequence").

---

## §3 Integration points

| # | Where | Change |
|---|---|---|
| 1 | New file `scripts/item_semantics.py` | `BitLedger`, `CandidateRecord`, `CapabilityEvent`, `SpliceHarness` as specified in §2. Zero imports from `go_explore_solve.py` internals beyond `GenericGame`'s public profile dict, if a profile-driven CLI wrapper is wanted (T4, §9) — not required for the core module. |
| 2 | `scripts/discover_observables.py` | **No changes.** `Discoverer` is imported and reused as-is (its `idle()`, `_step`, `_run`, `settle_steps()` are already the exact idle-prefilter and driving primitives needed); item-semantics is a *consumer*, never a patch. |
| 3 | `nes_core` (Rust) | **No changes in v1.** `poke_ram_range` is named and specified (§8) but not built; the current live-compute constraint (v28 training) and the "no core rebuild under a live prereg run" rule both argue for sequencing this strictly after the current campaign, exactly as ROOMGRAPH_ENGINE sequenced its own zero-Rust v1. |
| 4 | `scripts/go_explore_solve.py` — `Solver`/`GenericGame`/`RoomIndex` | **No changes in v1.** The optional read-only join (`room_ordinal` field, §2) calls `RoomIndex.lookup(hash)` (already a frozen, non-mutating read, `go_explore_solve.py:909` docstring: "FROZEN read (replay paths): ordinal or None, never interns") from *outside* that file — item-semantics receives a `RoomIndex` instance (or `None`) as a constructor argument; `RoomIndex` never imports or references item-semantics. |
| 5 | `Pool` (already-shipped accessors used read-only) | `get_ram_range(0, 0x800)` (`python.rs:709`) for the bit scan; `get_odometer()` (`python.rs`, `set_odometer_enabled`/`get_odometer` block) for position; `save_worker_state`/`load_worker_state` (`pool.rs:1515`/`1542`) for Track A's pre-event snapshot and matched-pair replay. All four already exist and are already used elsewhere in the tree for unrelated purposes — no new binding surface. |
| 6 | CLI: new standalone entry point, e.g. `scripts/item_semantics.py --rom ... --state ... --tag ...` | A probe-driver script in the shape of `discover_observables.py`'s `main()`, run offline/short like a discovery pass, NOT wired into the live `go_explore_solve.py` search loop in v1. This is the sharpest independence guarantee: item-semantics can be exercised and gated (IS-0/IS-1) without the solver's hot loop ever calling into it. |

---

## §4 Profile schema (entirely new block; absent ⇒ nothing in this module runs, nothing elsewhere changes)

```yaml
solve:
  item_semantics:
    idle_prefilter_steps: 300       # Discoverer.idle() cadence reused verbatim
    scan_window: [0, 0x800]         # the ONLY legal window; validated against get_ram_range's own
                                    # hard bound (python.rs:709) at load time, not a design choice
    never_revert_margin: 0
    splice:
      track: behavioral             # "behavioral" (v1, default) | "hard" (v2, requires poke_ram_range;
                                    # refused at load time with a clear error if the accessor is absent)
      n_trials: 20
      control_bits: 5
      verdict_gap: 0.5              # reach_rate_acquire - reach_rate_skip threshold
      control_flatness: 0.2         # max control swing before a probe location is marked confounded
    room_fp_join: optional          # if the SAME profile also sets solve.room_fp, pass RoomIndex in
                                    # read-only for room_ordinal enrichment; if absent, room_ordinal
                                    # is always null and nothing else changes
    log_dir: runs/item_semantics/{tag}/     # events.jsonl + bit_ledger.json, atomic tmp+rename
```

Receipt requirement mirrors `room_fp`'s: any non-default `scan_window`, `never_revert_margin`, or
splice threshold ships with a receipt under `docs/receipts/item_semantics/<game>.md` explaining the
measurement that motivated the deviation — no silent threshold tuning, same discipline as
`room_fp_config_sha`'s canonicalization (`go_explore_solve.py:819`).

---

## §5 Purity audit — every input

| Input | Class | Verdict |
|---|---|---|
| 2 KB internal CPU RAM, `get_ram_range(0, 0x800)` (`python.rs:709`) | Hardware surface, hard-masked by the core itself (`end > 0x0800` is a `PyValueError`, not a convention) | LEGAL — and note this is a *harder* boundary than a purity policy: Metroid's disassembly-sourced item-flag addresses (`0x6878`/`0x6879`, quarantined `docs/receipts/games/metroid_purity_quarantine_2026-08-10.md`) are unreachable through this accessor regardless of purity stance, so this module structurally cannot rediscover those specific bytes even if it wanted to — only whatever capability state happens to be mirrored into $0000-$07FF is visible (§7.4, an open unknown for Metroid, not assumed to resolve favorably). |
| Idle-prefilter mask (which of the 16,384 bits are excluded as clock/render bits) | Computed from our own `Discoverer.idle()` rollouts, same methodology as every existing `discover_observables` gate | LEGAL — experiment-discovered, per-game, receipted |
| Candidate bit identity (`addr`, `bit`) | Never assigned a name; consumers read the tuple only | LEGAL — matches the Contra objective-key precedent ("found structurally, never named," `V23_SYNTHESIS_2026-08-23.md`) |
| `acquire_actions`/`skip_actions`/`probe_actions` for Track A | Mined exclusively from our own prior rollout logs (the ledger's `events.jsonl` and the calling script's own action history) — never authored or scripted | LEGAL — same discipline as the Contra falsifier amendment; enforced structurally (`inconclusive` verdict, not fabrication, when no real skip segment exists) |
| Track B hard splice (`poke_ram_range`, deferred) | Writing a candidate bit that our own rollout already produced naturally elsewhere, into a state where it has not yet occurred, to test its causal role — never inventing a bit that was never observed, never assigning it meaning | LEGAL under the same "known-outcome amendment" `V23_SYNTHESIS`/the shared gap analysis already flag as this project's most attackable purity claim — **carried forward explicitly, not re-litigated here**; Track B remains gated behind its own IS-1-hard follow-on registration (§8) precisely because it is the one input on this table that a future audit should re-examine on its own, separate from Track A's clean bill. |
| Zelda $0070/$0084/HP/rupee-HUD-digit-box row/col ranges | Where used at all, sourced from the banked onboarding receipt (`docs/receipts/games/zelda_onboarding_2026-08-10.md` §8: "rupees ≈ rows 23-32 × cols 88-110; keys ≈ rows 38-47; bombs ≈ rows 48-55") — a *hypothesis for where to look*, not a RAM address; still requires the NT-diff confirmation step (§6, IS-1 prereq) before anything is claimed | LEGAL — same status as every other pre-registered-but-unconfirmed hypothesis in this doctrine |
| RAM maps, walkthroughs, imported item/equipment semantics, the quarantined Metroid cartridge block | — | **NOT USED.** This module never reads, imports, or cross-checks against `configs/metroid.yaml`'s `quarantined_external_knowledge:` block; a candidate bit landing on one of those addresses by coincidence is exactly that — coincidence, logged in a receipt, never treated as confirmation (identical stance to `zelda.yaml`'s own disclaimer on `ram_mapping`). |

Honest-eval untouched: this is a detection/verification module with its own offline probe driver; it
is never wired into the live search loop, the archive cell key, or the confluence clear detector in
v1. No claim from this lane can be conflated with a solver capability claim until a fusion pass (§8)
explicitly wires a validated event into routing or key composition — and that pass gets its own gate.

---

## §6 Pre-registered validation gate (register verbatim in `runs/item_semantics/PREREG.md` before IS-1; no post-hoc edits — same discipline as `runs/room_graph/PREREG.md`)

**IS-0 — offline falsifier (blocks all live runs; cheap-premise-first, BINDING).** Pure pytest, zero
live compute, synthetic + banked fixtures under `tests/fixtures/item_semantics/`:

- **Fixture A (planted-bit recovery).** A synthetic 500-step RAM stream over the legal 2 KB window
  with exactly one bit planted to flip 0→1 at step 40 and hold, plus ≥50 distractor bits: a
  free-running low bit (flips every step), an input-coupled bit (flips high only while a scripted
  button mask is held, reverts on release), and a bit that flips 0→1 then 1→0 once (a near-miss).
  `BitLedger` over this stream, with the idle-prefilter disabled (no idle segment in this synthetic
  fixture), must return **exactly** the planted bit as `status="candidate"` and zero distractors.
- **Fixture B (decoy discrimination).** Two bits both satisfy the raw 0→1-never-reverts test (a true
  causal bit and an unrelated decoy that also flips once, e.g. a menu-open flag), with a scripted
  reachability oracle standing in for real rollout probes. `SpliceHarness.verify_behavioral` against
  the oracle must validate the causal bit and reject the decoy, using the control-bit mechanism —
  proves the verdict-rule arithmetic (§2) before any real Pool is involved.
- **Fixture C (idle-prefilter, real hardware surface).** Reuse the already-banked
  `tests/fixtures/roomgraph/zelda_idle_fs1.npz` / `metroid_idle_fs1.npz` captures (real console RAM
  under real idle input, no scripting) — the idle-prefilter mask computed over them must exclude every
  bit that ever flips in those streams, and the post-filter candidate count over the SAME streams must
  be **zero** (idle play produces no capability events; a nonzero count here is a prefilter defect,
  not a discovery).
- **Fixture D (Track B verdict-rule parity).** `verify_splice`, called with a mocked `poke_ram_range`
  stub returning scripted reachability outcomes identical to Fixture B's oracle, must produce the same
  verdict as Track A on that fixture — proves the two tracks share one decision procedure and diverge
  only in how the ACQUIRE arm is produced, so Track B's eventual arrival (§8) is a drop-in, not a
  redesign.

Any failure ⇒ stop, no live compute, matching RG-0's own binding clause verbatim.

**IS-1 — Zelda (primary, live, gated on IS-0 PASS).**

*Prerequisite (receipted before IS-1 launches, not part of the gate itself):* mint ≥1 Zelda start-state
within reach of a real rupee or key pickup. Every existing fixture (`roms/zelda_start*.state.bin`,
`roms/Legend of Zelda, The (USA) (Rev A)_start.state.bin`) begins with an empty inventory per the
onboarding receipt — none can exercise a pickup as-is. Minting method: an idle-vs-active NT-tile
volatility diff exactly as `scripts/room_fp_calibrate.py` used to locate the HP-heart HUD tile at NT
byte 214, applied to a rollout that walks over a known overworld rupee, to confirm the HUD digit box
actually moves before any RAM candidate is trusted. Receipt: `docs/receipts/item_semantics/zelda_mint.md`.

- **IS-1a (bit discovery):** ≥1 bit reaches `status="candidate"` (survives idle-prefilter,
  0→1-never-reverts across the observed rollout) within 60 unattended minutes, 8 workers, fs4, across
  ≥2 seeds, with `first_seen_flip.position` recorded.
- **IS-1b (behavioral verification):** for every IS-1a candidate, `verify_behavioral` with n_trials=20,
  control_bits=5 reaches a verdict of `validated` or `rejected` (not `inconclusive`) for ≥1 bit; the
  validated bit's `reach_rate_acquire - reach_rate_skip ≥ 0.5` with `max(control) - min(control) < 0.2`.
- **IS-1c (integrity):** `item_semantics` absent from the profile ⇒ byte-identical to pre-branch HEAD
  on the same 16,000-step SMB determinism harness RG-1c used; zero calls into `Pool.save_worker_state`/
  `load_worker_state` beyond what the profile already made when the block is absent.
- **IS-1d (no room-graph disturbance):** if the same run also has `room_fp` active for the soft join,
  RG-0's 9/9 offline falsifier is re-run green afterward and `room_index.json`'s `config_sha` is
  unchanged — proves the read-only join never mutates the room-graph state it reads.
- **Kill criteria (pre-registered):** zero candidates survive IS-0-shaped scrutiny after 90 unattended
  minutes across both seeds ⇒ the bit-ledger mechanism is falsified for Zelda with this scan window,
  lane stops with receipt (the $0000-$07FF window may simply not contain a Zelda inventory flag —
  a real, nameable outcome, not a bug). All IS-1a candidates resolve `inconclusive` under IS-1b ⇒ the
  probe-location/reachability framing needs rework before another attempt; not evidence the bits
  themselves are wrong.

**IS-2 — Metroid (secondary, report-only, non-blocking, explicit low-confidence).** Same protocol as
IS-1a/b against `configs/metroid_roomfp.yaml`'s existing room_fp-enabled fixtures (RG-2 already
established ≥8 fingerprint rooms live). Because the disassembly-sourced Metroid item bytes are
confirmed unreachable through `get_ram_range` (§5), IS-2's honest headline is binary: either some
proxy of missile/bomb/morph-ball state happens to be mirrored into the legal window (a real, useful
discovery if true) or IS-2 reports **zero candidates**, which is itself the receipted answer to an
open question, not a lane failure. No kill criterion is pre-registered for IS-2 because a null result
here is informative rather than falsifying.

Receipts: `docs/receipts/item_semantics/IS1_zelda_<date>.md` (+ IS-2 report), same shape as
`docs/receipts/room_graph/RG1_zelda_2026-08-25.md` — run configs, seeds, ledger stats, verdicts.

---

## §7 Failure modes named in advance

1. **HUD digit counters mistaken for capability flags.** A BCD digit's individual bits do not move
   monotonically as the displayed number climbs (e.g. the tens digit's bit pattern for 0→9→10 is not a
   single sticky bit). Guard: §1's flag-vs-counter split; a HUD digit region is a `counter-delta`
   candidate via the reused monotone-counter path, never fed to `BitLedger` directly.
2. **Off-manifold drift from Track B (hard splice).** Forcing a bit true without its normal
   accompanying state can desync invariants the ROM never defends against — silently (wrong sprite,
   stuck flag) or as a crash. This is the same failure class the project has already paid for once
   (DR v10's off-manifold drift barrier, `docs/proposals/RECOVERY_DISTILL_1_1_2026-08-24.md:40`).
   Guard: Track B stays
   deferred (§8) until `poke_ram_range` ships with its own crash/hang trap (a step-timeout wrapper
   analogous to `apply_state_guarded`'s `catch_unwind`, `python.rs:37` — that function already exists
   for malformed *deserialized* states; a live in-RAM write after a normal load is a different failure
   surface and needs its own guard, not a reused one) and random-bit controls are mandatory, never
   optional, in every Track B trial.
3. **Idle-prefilter mask staleness.** Computed once per lineage and frozen (like `room_fp`'s mask);
   a ROM revision or a mid-run profile edit invalidates it silently unless the mask ships with a
   `config_sha`-equivalent lineage guard (§9, T1).
4. **Temporal confound at a room transition.** Many bits legitimately flip in the same few frames as a
   door/fade transition (palette swaps, sprite-slot reuse) that have nothing to do with inventory.
   Guard: `first_seen_flip.position` records odometer bbox, not room ordinal, precisely so a burst of
   simultaneous flips at a transition boundary is visible as a temporal cluster in the log rather than
   silently attributed to "the room"; a future fusion pass (§8) must explicitly de-duplicate transition-
   coincident flips before trusting any one of them as *the* capability bit.
5. **Metroid's $07FF window may simply not contain the answer.** Named as IS-2's explicit low-
   confidence framing (§6) rather than discovered mid-run as a surprise.
6. **Near-miss bits (one revert, otherwise sticky).** A bit that flips 0→1 then 1→0 exactly once (e.g.
   a temporary invincibility/animation flag that happens to persist unusually long) is permanently
   excluded by `never_revert_margin=0`. This is conservative by design — logged as `rejected` with
   `reverts_seen=1` rather than silently coerced into a candidate — and is a documented limitation, not
   a bug: a genuinely reversible-but-rare-revert capability (does any target game have one?) would need
   a separate near-miss detector, explicitly out of scope for v1.
7. **`BitLedger` growth.** 16,384 candidate slots × per-rollout bookkeeping is small (no `RoomIndex`-
   style unbounded adjacency growth risk), but `events.jsonl` is append-only and unbounded across a long
   unattended run. Guard: same atomic tmp+rename cadence as `RoomIndex.save()`, plus a rotation policy
   (new file per N events) deferred to T4 if IS-1's 90-minute runs show it matters.
8. **Track A's SKIP arm may not exist in the ledger.** If no real rollout ever walked past the item
   without touching it, `verify_behavioral` returns `inconclusive` rather than fabricating a skip
   trajectory (§2) — this can stall verification on games/items with no natural "walk past" geometry
   (a corridor item with no other path). Named honestly: Track B (once built) is the fallback for
   exactly this case, since a hard splice needs no skip trajectory at all.
9. **False-positive control bits happening to be meaningful too.** If a "random other bit" used as a
   control is *itself* a real, different capability flag, the control-flatness check could fail for the
   wrong reason (a real effect, misread as confounding). Guard: control bits are drawn preferentially
   from the `rejected` pool (bits that failed the irreversibility test outright) before falling back to
   untested bits, minimizing this collision; a receipted false-positive-control audit is required before
   any IS-1 verdict is banked (mirrors RG-1a's false-merge audit).
10. **Room-graph join drift.** If a profile enables both `room_fp` and `item_semantics` and the
    `RoomIndex` instance passed in is stale (a resumed run with a different `config_sha`), the read-only
    `lookup()` call returns `None` for hashes it doesn't recognize (§2's own documented behavior) —
    `room_ordinal` silently degrades to `null` rather than aliasing onto the wrong room. Never a hard
    failure, per the independence guarantee in §0.
11. **Perf.** A full 16,384-bit scan every step is a `numpy` diff over 2 KB per worker per step — the
    same order of cost as `RoomIndex`'s NT-hash-per-step and well inside the sub-ms budget
    ROOMGRAPH_ENGINE §7.13 already measured for a comparable per-step memcpy+hash; no perf gate is
    pre-registered for IS-1 beyond "does not visibly regress the existing Zelda room_fp baseline if
    both are enabled together," checked as part of IS-1d.
12. **Confusing a validated bit for a semantic claim.** IS-1b's `validated` verdict says "this
    structurally-discovered bit at this address/bit-index causally gates this odometer bbox," nothing
    more. No CLAIMS.md entry from this lane may say "the sword" or "a key" — the rediscovery-rule
    convention from the Metroid quarantine doc applies here too: a name only enters prose as
    illustrative color in a receipt's discussion section, never as the identity a downstream consumer
    reads.

---

## §8 Deferred to v2 (pre-declared; nothing in v1 forecloses them)

- **`poke_ram_range` (or `poke_ram`) — new Rust accessor.** Writes a caller-supplied byte range into a
  live or freshly-restored worker's system RAM (mirrors `get_ram_range`'s bounds check, `0..0x800`).
  Needed for Track B. Explicit safety requirement beyond a plain write: a post-write liveness probe
  (N steps of NOOP must not panic/hang before the splice trial is trusted) since `apply_state_guarded`
  (`python.rs:37`) only catches panics from *deserializing* a state blob, not from stepping a live NES
  after a targeted in-RAM mutation — a new guard, not a reused one (§7.2). Sequencing: after the current
  v28 training campaign vacates the machine, honoring "no core rebuild under a live prereg run"; this
  document does not request it now.
- **The fusion join proper.** Once ≥1 bit is IS-1-validated AND RG-1-class room-graph data exists for
  the same game, a fusion pass adds a *read-only-derived, optionally-cached* `cap_sig` annotation
  alongside `RoomIndex` edges — computed by item-semantics, attached by a small joiner script, never by
  editing `EdgeStat`'s live schema. Whether that annotation eventually becomes a routing precondition
  (Lens A's proposed mechanism) is a decision for the synthesis pass this document explicitly defers to
  (§0's comparison table), not this document.
- **OAM (256-byte sprite table) bit tracking.** `peek_oam` (`python.rs:766`) is already a hardware
  surface accessor; extending `BitLedger`'s scan window to include OAM would catch capability signals
  encoded as sprite-slot occupancy (a permanently-spawned "you have the X" HUD icon) rather than CPU
  RAM — same mechanism, larger candidate space (256×8 = 2,048 more bits), deferred because no target
  game has yet motivated it with a receipt.
- **Batched Rust bit-diff.** If per-step `numpy` diffing becomes measurable overhead once wired into a
  live loop (it is not wired in v1 at all, §3), a batched Rust accessor analogous to the pre-declared
  `nt_fingerprint_per_worker` (ROOMGRAPH_ENGINE §8) is the same-shaped answer.
- **Joint/multi-bit capability preconditions** (AND-gated puzzles — "needs both the raft and the
  ladder"). v1's `SpliceHarness` tests one bit at a time; a combinatorial verification loop over bit
  *sets* is a real future need for Zelda-class games but is not gated by anything in this document
  shipping first.

---

## §9 Implementation plan — worktree tasks (no `git stash` across lanes)

Dependency order: **T1 → {T2, T3} → T4 → T5.** IS-0 (inside T3) must PASS before T4 launches.

| Task | Scope (files) | Deps | Done-when |
|---|---|---|---|
| **T1 — `BitLedger` core** | `scripts/item_semantics.py`: `CandidateRecord`/`CapabilityEvent` schemas, idle-prefilter reusing `Discoverer.idle()`, the irreversibility scan, atomic save/load (`bit_ledger.json`), lineage guard (a `config_sha`-equivalent over `scan_window`/`never_revert_margin`/prefilter length). Unit tests: synthetic streams with known planted bits + distractors (superset of IS-0 Fixture A). | — | Unit suite green; scan over a synthetic 10,000-step stream with 5 planted bits + 200 distractors returns exactly the 5 planted bits; idle-prefilter over a synthetic all-noop stream excludes every bit that ever flips. |
| **T2 — `SpliceHarness` (Track A only)** | `verify_behavioral`, the ledger-mining step for real skip trajectories (§2), the verdict-rule arithmetic, control-bit selection preferring the `rejected` pool (§7.9). `verify_splice` stubbed to raise `NotImplementedError` with the Track B signature already fixed (IS-0 Fixture D depends on this shape). | T1 | Fixture B + D pass against a scripted oracle; a real `Pool` smoke test (5-min Zelda, `roms/zelda_start_ctrl.state.bin`) exercises `save_worker_state`/`load_worker_state` round-trip with zero crashes. |
| **T3 — fixtures + IS-0** | `tests/fixtures/item_semantics/` (synthetic streams + the reused `roomgraph/*idle*.npz` captures); `tests/test_is0_item_semantics.py` implementing all four IS-0 fixture assertions (§6) as pytest. | T1, T2 | IS-0 pytest 4/4 (or however many assertions the implementation settles on) PASSES; zero live compute consumed. |
| **T4 — Zelda pickup mint + probe driver** | `scripts/room_fp_calibrate.py`-style NT-diff mint script producing a new `roms/zelda_pickup_*.state.bin` fixture with a receipted rupee/key-adjacent start; standalone `item_semantics.py --rom --state --tag` CLI driver (T-shaped, per §3 item 6) that runs a short unattended probe and writes `events.jsonl`. | T3 (IS-0 pass) | `docs/receipts/item_semantics/zelda_mint.md` filed with the NT-diff confirmation; a 10-minute smoke run against the new fixture produces ≥1 `CandidateRecord`, logged, no crash. |
| **T5 — gate IS-1** | Register §6 verbatim in `runs/item_semantics/PREREG.md`; run the pre-registered 60-minute × 2-seed Zelda protocol unattended with the same abort-guard discipline (SPS floor, archive-size cap, RSS guard) RG-1 used; file the receipt with a pass/kill verdict. | T4 | `docs/receipts/item_semantics/IS1_zelda_<date>.md` filed, no post-hoc edits to the registered numbers, matching `RG1_zelda_2026-08-25.md`'s own convention exactly. |

Checkpoints: after T1-T3 (offline-only, zero live compute — can proceed in full during any period the
machine is saturated by an unrelated campaign, exactly like RG-0's own development did not require live
cycles), after T4 (one short smoke, cheap), after T5 (the only checkpoint that consumes pre-registered
compute, and only after T3's IS-0 gate and T4's mint receipt both exist). No nes_core rebuild anywhere
in this plan, so it is fully compatible with a machine currently saturated by v28 training — T1 through
T4 are pure Python/pytest work against banked fixtures and short (5-10 minute) smokes, not solver bursts.
