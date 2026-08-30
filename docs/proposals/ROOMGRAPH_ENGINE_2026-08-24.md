# ROOM-GRAPH ENGINE — SYNTHESIS (2026-08-24)

Synthesized from three independent designs (D0 minimal-diff, D1 graph-first, D2 discovery-first).
All line anchors below re-verified against HEAD `3e9a502` on 2026-08-24 (`scripts/go_explore_solve.py`,
6099 lines): ODO_LO :646, `room_sig` parse :1448, `room_id` :1975-1977, `derive_transition_macros`
:1992, transition-macro wiring :2560, ortho knobs :2664, `_xram` :2866 / `_xram_local` :2892,
`observe` :2952 / key composition :3182, R4 recorder :3192-3204, `seed` :3857, `_refresh_sel_cache`
:4064, `select` :4156, `_articulation_points` :4257, `_assign` :4332, `explore` :5308, macro
injection slot :5359-5378, per-step ram fetch :5421, transit block :5476-5481, lineage guards
:238/:301/:331/:542, `count_wmax` :2069. nes_core: `odo_debug` pool.rs:1709, `peek_nametables`
binding pool.rs:1736 (`nametable_snapshot` ppu.rs:2984), `palette_ram` pool.rs:1070.

---

## Part I — Verdict on the three designs

| Axis | D0 minimal-diff | D1 graph-first | D2 discovery-first |
|---|---|---|---|
| (a) Purity compliance | **9/10** | **9/10** | 8.5/10 |
| (b) P(gate passes this week, real Zelda/Metroid) | **~0.65** | ~0.45 | ~0.15 |
| (c) Generality | 6/10 | 8/10 | **9/10** |
| (d) Implementation risk (10 = safest) | **9/10** | 7/10 | 4/10 |

**Winner: D0 as chassis.** Rationale:

- **D0** wins on shipping physics: zero Rust changes, zero key-schema changes — room identity rides
  the already-battle-tested `area`/`room_sig`/`room_advance`/sect/psig machinery through two pseudo-RAM
  bytes, exactly the way the scene ordinal already rides 0x803. Its transit detector, frontier gating,
  and door-macro injection are shipped code. Weaknesses (all fixed by grafts below): no transition-kind
  classification, so a Zelda/Metroid death warp would mint a navigable graph edge wherever no
  death observable exists; no offline falsifier before live compute (violates the BINDING
  cheap-premise-first sequencing rule); no edge-replay validation; its per-game masks are receipted
  calibration artifacts (fine, but D1's auto-calibrator is the cleaner generator).
- **D1** has the best *detector*: the STABLE/CANDIDATE state machine with pan/fade/flash_warp
  classification from integrated Δodo + Δscene, the Stage-A offline falsifier over banked probe
  fixtures, edge exemplars for replay validation, and the aliasing audit. But its rgid key-prefix
  change breaks every banked archive, touches lineage/arity/resume in ways that must be perfect on
  the first try, and roughly doubles the integration surface for a one-week gate. All of its detector
  ideas transplant cleanly onto D0's chassis; the prefix does not need to.
- **D2** is the right *long-term* answer (capability bitmask keying is how Zelda/Metroid lock-and-key
  progression eventually falls) and its restore-lockstep discipline is the sharpest of the three. But
  it is not a one-week program: four new Rust accessors + a cert harness + the no-core-rebuild-under-
  live-prereg constraint force strict sequencing, its own gate ladder includes a 12h discovery run and
  a 4×6h A/B, `poke_ram` splicing is the newest purity territory (defensible under the v23
  known-outcome amendment, but the most attackable claim of the thirty on the table), and the
  cap-state explosion failure class has already cost 138 GB once. **Banked verbatim as the v2
  roadmap** (§8); nothing in v1 forecloses it.

**Grafts adopted into the synthesis** (source in brackets):
1. Offline falsifier gate RG-0 over banked probe fixtures, mandatory before any live run [D1 Stage A].
2. Transition-kind classifier (pan / fade / warp) from integrated Δodo + Δscene during the churn
   window; warp-classified settles adopt the new room identity but **mint no adjacency edge and are
   never routable** [D1/D2] — closes D0's death-edge gap game-agnostically (Metroid has no probed
   death observable and must not need one).
3. Edge exemplars (archive cell key at onset + last-32 action ring) enabling the sticky-replay edge
   validity audit in the gate [D1].
4. Aliasing audit — (src, kind/dir) with ≥2 distinct dsts each ≥3 traversals ⇒ nodes marked aliased,
   router deprioritizes [D1] — plus the false-merge bbox audit (two rooms' odometer bboxes disjoint
   by >512 px sharing one fingerprint ⇒ gate check fails) [D2].
5. Restore-lockstep discipline stated as invariants: global structures append-only/monotone
   (restore-order independent by construction); per-worker detector state *derived from cell metadata
   on `_assign`, never accumulated across restores* [D2 §1F].
6. U(R) unexplored-boundary term in the router weight: sides with boundary cells but no out-edge of
   that direction [D1], merged with D0's articulation term.
7. Pre-declared, out-of-v1 contingencies: `mirroring_per_worker()` binding + batched Rust NT hash
   (perf) + the whole capability layer [D2]. No Rust this week; no core rebuild under a live prereg run.

---

## Part II — The synthesized design

### §1 Thesis

Room identity = a settled, masked blake2b-64 hash of the 2 KB physical nametable VRAM
(`Pool.peek_nametables`), interned to a discovery-order ordinal, injected into the existing pseudo-RAM
extension at 0x804/0x805, and routed through the **existing** `solve.area` / `solve.room_sig` /
`solve.room_advance` machinery — so room-keyed cells, sect/psig transits, frontier tracking, and
door-entry macros come from shipped code. New solver logic: (a) the settle/intern loop with
transition-kind classification, (b) edge recording (kind-tagged, warp-vetoed, exemplar-carrying) at
the existing transit point, (c) one flag-gated selection arm, (d) lineage/persistence/restore guards.
Everything defaults off; byte-identical when off.

Why fingerprint-settle and not the scene ordinal as identity: the probes measured scene as noisy in
Metroid (spurious bumps at clamp/seam — scene 4 without leaving room 1) and blind by design to Zelda
fades (cave/dungeon entry) and Rygar blank doors. NT-hash-settle was the reliable room edge in both
games; scene is demoted to **classifier evidence, never identity**.

### §2 Data structures (all in `scripts/go_explore_solve.py` unless noted)

```python
ROOM_LO, ROOM_HI = 0x804, 0x805     # ordinal LE in xram, beside ODO_LO=0x800 (:646)
ODO_ALT = 0x806                     # orthogonal odometer axis, (clamp(v,0,0xFFFFFF)>>4)&0xFF
ROOM_UNKNOWN = 0xFFFF

class RoomIndex:                    # global; append-only/monotone; lock shape = R4's _door_lock
    hashes: dict[int, int]          # blake2b64(masked NT) -> ordinal (discovery order)
    ordinals: list[int]             # ordinal -> hash
    meta: dict[int, RoomMeta]       # ordinal -> {visits, bbox odo x/y extremes, aliased: bool}
    adj: dict[int, dict[int, EdgeStat]]   # DIRECTED; EdgeStat = {kind: pan|fade, dir: E|W|N|S|None,
                                    #   count, frames_mean, exemplar_cell, exemplar_actions[<=32],
                                    #   validated, validate_attempts}
    warps: list[WarpEvent]          # warp-classified settles: recorded for telemetry, NEVER edges
    lock: threading.Lock
    cap: int                        # max_rooms; at cap: hold-last + telemetry, never crash
```

Per-worker state: `self._room_ord: np.uint16[workers]` (seeded on `_assign`, read by `_xram`);
burst ctx `c["fp_pend"] = (hash, n_consecutive, onset_odo_xy, onset_scene, onset_step) | None`,
`c["fp_ring"] = deque(maxlen=32)` of action indices.

**Settle + classify rule** (per worker per solver step, immediately before the :5421 `_xram` call):

```
lines = pool.odo_debug(wid)[2]                    # pool.rs:1709 -> (x, y, rendered_lines)
if lines < min_lines: c["fp_pend"] = None; skip   # blanks/fades never sampled
h = blake2b64(peek_nametables(wid) * mask)        # mask: np.uint8[2048] keep-mask
pend advances iff h repeats; on n >= settle and h != current settled hash:
    kind = classify(Δodo integrated since onset, Δscene, frames)
      pan:  |Δodo| in [pan_min, pan_max] on exactly one axis AND Δscene <= 1   (dir = axis+sign)
      warp: Δscene >= warp_scene_min AND |Δodo| < 32                            (measured Zelda death:
                                                    modal 16->272->16, scene +2, odometer flat)
      fade: otherwise (hash flip, odo ~flat, scene 0..1)  # Zelda caves/dungeons, Rygar doors —
                                                          # the class the scene core is blind to
    ord = index.intern(h)                          # under lock; update bbox/visits
    if kind != warp and self._room_ord[wid] != ROOM_UNKNOWN and adoption_guard_ok:
        stage edge (prev_ord -> ord, kind, dir, exemplars from c)   # recorded at the transit point
    if self._room_ord[wid] == ROOM_UNKNOWN: c["p0750"] = None       # adoption != transit
    self._room_ord[wid] = ord
```

Constants pre-registered from the 2026-08-24 probe receipts, frame-skip-adjusted:
`settle = max(2, ceil(10/frame_skip))` solver steps (Zelda pans churn 64 straight frames, Metroid
~127 — mid-pan settle cannot fire); pan window [128, 384] px (both games measured ≈256);
`warp_scene_min = 2`. Hash = `hashlib.blake2b(digest_size=8)` (stdlib C; pure-Python FNV at
2 KB × workers × steps is too slow — instrumentation choice, not doctrine, noted as a deviation).

**Persistence**: `<out>/room_index.json` (version, `config_sha` over the room_fp block, hashes,
adj with kinds/exemplars, warp count), atomic tmp+rename on the `archive.stats.json` cadence + exit.

**Restore-lockstep invariants** (the documented desync trap — odometer/scene ride the v4 savestate
envelope; Python state does not):
- Global structures (`RoomIndex`) are append-only/monotone ⇒ restore-order independent.
- Per-worker state is derived, never accumulated: `_assign` (:4332) seeds `_room_ord[wid]` from the
  cell's threaded psig tail (cell path, :4380 region) or `ROOM_UNKNOWN` (root path); sets
  `c["fp_pend"] = None`. `c["p0750"] = None` at burst start already prevents restore-edge false
  transits; the adoption-from-unknown guard makes a root's first settle transit-free and edge-free.

### §3 Integration points (all verified at 3e9a502)

| # | Where | Change |
|---|---|---|
| 1 | `GenericGame.__init__` — `room_sig` parse :1448, odometer rewire block ~:1303-1335 | Parse `solve.room_fp` (validate mask ranges ⊂ [0,2048), settle ≥1, cap ≤4096). Allow `area`/`room_sig`/`room_advance.addr`/`y` to reference 0x804-0x806 (bounds text only — they read `ram[addr]` blindly). |
| 2 | `Solver.__init__` — beside ortho knobs :2664, pool-odometer enable block | `RoomIndex`, `_room_ord` init, knobs `--room-bias` (default **0.0** = arm off), `--room-artic-weight` (2.0), `--room-exit-weight` (1.0), `--room-recent-k` (4). `room_fp` **forces `self._odo = True`** even for RAM-progress profiles (Zelda): the lines gate, Δodo classifier, and scene evidence need it. |
| 3 | `_xram` :2866 / `_xram_local` :2892 | When `room_fp`: extend by 7 bytes not 4; write `_room_ord[wid]` LE at 0x804/0x805, other-axis odometer at 0x806. `_xram_local` re-derives the ordinal by hashing the replay env's NT against the **frozen** index — unknown hash ⇒ hold-last; a diverging receipt replay is marked UNVERIFIED by the existing comparison, never silently passed. Length switch keyed on `room_fp` presence ⇒ non-fp profiles byte-identical. |
| 4 | `explore()` — before the :5421 ram fetch | The settle+classify block (§2). Only hot-loop insertion. |
| 5 | `explore()` transit block :5476-5481 | **Transit logic unchanged** — `room_id` (:1975-1977) already folds `room_sig` bytes, so ordinal changes fire sect/psig exactly like SMB's $074E. Add ≤5 lines inside `if _transit:` — commit the staged edge (kind, dir, exemplars) under `index.lock`; warp-classified settles commit nothing. Template: R4 recorder :3192-3204. |
| 6 | `observe()` key composition :3182 | **UNTOUCHED.** Room identity rides psig (key[3]) and area (key[-5]); prefix and tail arities unchanged; every banked archive resumes. |
| 7 | `_refresh_sel_cache()` :4064 | Same single scan: `self._room_pools: dict[ord, list[Cell]]` (ordinal = last-2 elements of `c.key[3]`), `V(r)` = Σ times_chosen, boundary sublists (cells within `near·GX_BUCKET` of the room bbox edge per side). Inline `self._room_artic = self._articulation_points(undirected(adj))` (:4257; graph ≤ cap nodes, no thread). Ortho lesson: never overwrite `_sel_cells`; empty pool falls through. |
| 8 | `select()` — between the ortho arm and the count arm (ortho arm starts at the `self._ortho_armed()` guard, ~:4193) | Router arm, gated on `--room-bias > 0`: frontier set F = {ord ≥ max_ord−K} ∪ {degree ≤1} ∪ artic set ∪ {aliased=False rooms with U(r)>0}; weight `w(r) = (1 + artic_w·[r∈artic] + exit_w·U(r)) / sqrt(V(r)+1)` where U(r) = #sides with boundary cells but no out-edge of that dir; sample r, then a cell via the ortho arm's exact rejection loop + barren skip; boundary cells at p=0.40. Pure count prior; score untouched; selection-side only. Aliased rooms weighted ×0.25. |
| 9 | `_assign()` :4332 | Seeding per §2 lockstep invariants; attach `c["route_room"]` and, when routed to a room with U(r)>0, `c["route_dir"]` (sampled from its no-out-edge sides) — the macro roll at :5359-5378 gains one OR-term: routed workers near that side's bbox edge roll the direction-hold macro (derived structurally from the action space, generalizing `derive_transition_macros` :1992). |
| 10 | Lineage: `LINEAGE_KEY_AXES` :301, `key_config_axes` :331, `key_schema_conflicts` :542, `stamp_stats_provenance` :238 | New axis `("room_fp", ...)` = sha8 over canonical {sorted mask ranges, settle, classifier constants, palette_cokey, max_rooms}; `""` when off. Cross-schema resume refused; provenance stamping automatic. |
| 11 | `seed()` resume block :3857 | Load `room_index.json` before the lineage check; hard-refuse resume if archive lineage has room_fp but the file is missing or `config_sha` mismatches. Per-worker state self-heals from cell psig + re-settle. |
| 12 | argparse near `--door-weight` | The three router flags. Telemetry appended to the status line: `rooms`, `edges_pan/fade`, `warps_vetoed`, `artic`, `settle_rejects`, `aliased`, `router_picks`. |

Door routing needs **no code**: profile sets `room_advance: {addr: 0x804, ...}` →
`derive_transition_macros` (:1992) + `at_frontier` gating + the injection roll (:5359-5378) inject
door maneuvers at the newest room's frontier; `max_room` telemetry tracks the deepest ordinal.

### §4 Profile schema (absent block ⇒ feature inert, byte-identical)

```yaml
solve:
  room_fp:
    mask: [[0,256],[1024,1280]]   # NT byte ranges to ZERO before hashing — emitted by
                                  # scripts/room_fp_calibrate.py from OUR OWN idle/walk frames
                                  # (auto volatility mask, D1-style); receipt required per game
                                  # under docs/receipts/room_fp/<game>.md. Attribute bytes
                                  # (960..1024, 1984..2048 per KB) participate unless volatile.
    settle: 3                     # consecutive identical hashes (solver steps)
    min_lines: 200                # odo_debug rendered-lines floor; below => no sampling
    pan_odo: [128, 384]           # classifier constants, probe-receipted, game-agnostic
    warp_scene_min: 2
    palette_cokey: false          # fold palette_ram into the hash (default off: fades must not fork rooms)
    max_rooms: 1024
    sample_every: 1               # perf fallback: hash every Nth step
  area: 0x804                     # ordinal lo -> key[-5] (novelty frontier; wrap >256 = telemetry warn)
  room_sig: [0x804, 0x805]        # exact 16-bit ordinal -> room_id -> sect/psig transit machinery
  room_advance: {addr: 0x804, p: 0.06, near: 24, buttons: [...]}   # buttons from probe receipts
  # Zelda gate profile: progress: {lo: 0x70}, y: 0x84, lives/hp per banked onboarding receipt
  # Metroid gate profile: progress: {source: odometer, axis: x}, y: 0x806
```

### §5 Purity audit — every input

| Input | Class | Verdict |
|---|---|---|
| 2 KB physical NT VRAM (`peek_nametables`, pool.rs:1736) | Hardware surface | LEGAL (v23 ruled explicitly) |
| `odo_debug` lines / odometer / scene (pool.rs:1709; certified core 3429d8a/741a953) | Hardware surface | LEGAL |
| `palette_ram` optional co-key (pool.rs:1070, default off) | Hardware surface | LEGAL |
| NT masks | Computed by `room_fp_calibrate.py` from variance over our own idle/walk frames; no human reads the screen; receipt per game | LEGAL — experiment-discovered |
| Classifier constants (256 px pan, 10-frame settle, death = scene+2/odo-flat) | Measured in our own 2026-08-24 hardware-surface probes (receipts moved into `tests/fixtures/roomgraph/` in T4) | LEGAL — experiment-discovered |
| Ordinals, adjacency, exemplars, aliasing marks | Derived solely from our own rollout transitions/inputs (same class as the receipted R4 door machinery) | LEGAL |
| `room_advance.buttons` (e.g. Metroid shoot-then-enter) | Probe-measured via our own rollouts; receipts cited in the profile comment | LEGAL |
| Zelda $0070/$0084/HP | Banked, `docs/receipts/games/zelda_onboarding_2026-08-10.md` | LEGAL |
| RAM maps, walkthroughs, imported room semantics, disassembly | — | **NOT USED.** Metroid profile = odometer + pseudo-addresses only. |

Honest-eval untouched: search infrastructure only. Headline stays cold sticky greedy. Zelda keeps
`level_key: []` (coverage, never wins; assert clear-count == 0 in every gate run). Room transits are
**never** wired to the confluence clear detector (its Kirby room-transition false-positive mode is
open/unwired — explicit non-goal).

### §6 Pre-registered validation gate (register verbatim in `runs/room_graph/PREREG.md` BEFORE T5; no post-hoc edits)

**RG-0 — offline falsifier (blocks all live runs; cheap-premise-first, BINDING).** Replay the
detector+classifier over the banked probe fixtures as a pytest. PASS requires ALL: Zelda east exit ⇒
pan-E, exactly one new node; Zelda death ⇒ warp, **zero** edges; Zelda 300-idle-frame log post-mask ⇒
exactly 1 hash; Metroid door1/door2 ⇒ exactly 2 pan edges; Metroid spurious scene bumps (scene 4
inside room 1) ⇒ zero extra nodes. Any failure ⇒ stop, no live compute.

**RG-1 — Zelda (primary).** 4 unattended runs × 90 min, 12 workers, fs4,
`roms/zelda_start_ctrl.state.bin`, seeds {0,1} × {`--room-bias 0.25`, `--room-bias 0`}. Abort
guards: SPS floor, archive-size cap, RSS guard.
- **RG-1a validity:** ≥30 distinct settled rooms by 90 min (seed 0, either arm); ≥1 **fade** edge
  banked (cave or dungeon entry — the class scene is blind to); zero warp-minted edges (audited);
  false-merge audit: room pairs with odometer bboxes disjoint by >512 px sharing a fingerprint = 0.
  Stability: 20 random rooms × 3 archived cells from disparate lineages restored on a fresh pool ⇒
  re-settled ordinal == recorded for ≥95% of 60 restores.
- **RG-1b routing lift:** distinct rooms (router ON) ≥ 1.25× (OFF) at 90 min on BOTH seeds;
  tie-break time-to-25-rooms.
- **RG-1c integrity:** `room_fp` absent + `--room-bias 0` ⇒ SMB 1-1 5-min determinism harness
  byte-identical vs pre-branch build; clear events in all Zelda runs == 0.
- **RG-1d perf:** SPS ≥ 90% of room_fp-off Zelda baseline; else engage `sample_every: 2` and
  re-measure before any verdict.
- **RG-1e edge validity:** 20 sampled edges replayed from `exemplar_cell` restore + exemplar actions
  under sticky p=0.25 ⇒ ≥80% reproduce the same (src, dst, kind).
- **Kill criteria (pre-registered):** <10 rooms in any 60-min seed-0 window ⇒ fingerprint design
  falsified for Zelda, lane stops with receipt. Stability <80% or >50% false splits in a 30-node
  manual audit ⇒ mask/settle design falsified. Router lift <1.0× on both seeds ⇒ router ships
  default-off permanently (identity layer survives on RG-1a); not a lane kill.

**RG-2 — Metroid (secondary, report-only, non-blocking).** 90 min, seed 0, odometer profile:
≥8 fingerprint rooms incl. the measured door corridors; the probed 3-room stretch = exactly 3 nodes
(scene noise fully absorbed); fingerprint room count ≤ scene-ordinal count; door-macro injections >0
with ≥1 transit within 30 s of an injection; zero warp edges. Death detection is an honest stub (no
probed observable — the warp veto is the only defense; NG corpse-frontier caveat applies and is why
RG-2 cannot gate).

Receipts: `docs/receipts/room_graph/RG1_zelda_<date>.md` (+ RG-2 report) with run configs, seeds,
archive stats, audit outputs.

### §7 Failure modes named in advance

1. **CHR-bank aliasing / structurally identical rooms** (enemies are OAM, invisible to the hash) —
   under-split/merge. Guards: aliasing audit (src,kind/dir → ≥2 dsts) marks nodes, router ×0.25;
   false-merge bbox audit in RG-1a; `palette_cokey` available; psig/sect still separates lineages.
2. **False splits from unmasked animated tiles** — settle + `max_rooms` cap + hold-last + kill
   criterion + mask recalibration path.
3. **Seamless scrollers** (SMB-class) — hash never settles while moving ⇒ hold-last, zero transits =
   status quo. Correct behavior, named so nobody "fixes" it; feature is profile-gated.
4. **Fake settle in pause/lag** — `min_lines` + settle ≥3; residual audited in RG-1a.
5. **Death/warp minted as navigable** — the warp classifier veto (no edge, never routable); where a
   death observable exists (Zelda lives) lineage kill additionally fires; Zelda death is
   NON-TERMINAL (continue menu) — no predicate assumes termination.
6. **Replay divergence** — `_xram_local` hold-last on unknown hash; diverging receipts marked
   UNVERIFIED by the existing comparison.
7. **Resume desync** — lineage axis + explicit `room_index.json` check ⇒ hard refusal.
8. **Restore desync of per-worker state** — §2 invariants (derive-don't-accumulate); RG-1a stability
   audit is the measurement.
9. **Ordinal wrap >256 in the one-byte area slot** — frontier bias aliases; identity stays exact
   (16-bit in psig); telemetry warns at 200.
10. **Four-screen carts** — 2 KB snapshot aliases cart-RAM nametables; `room_fp` unsupported there,
    documented (none of the target games).
11. **Fade-pair ambiguity** (cave exit is another fade; A→B, B→A are separate directed edges with
    unknown geometry) — router treats fade edges as traversable-by-replay only, never assumes
    reversibility.
12. **Router pinning on a hub** — barren counters + rejection sampling, unchanged from ortho.
13. **Perf** — 96 × (2 KB memcpy + blake2b + odo_debug)/step ≈ sub-ms; pre-registered fallback =
    `sample_every`, then (v2, pre-declared) the batched Rust hash. Not a redesign.

### §8 Deferred to v2 (pre-declared, from D2 — nothing in v1 forecloses them)

`Pool.mirroring_per_worker()` binding (mapper.rs:86) + visible-page hashing if mirroring flips cause
false splits; batched `nt_fingerprint_per_worker` Rust accessor if `sample_every` is not enough;
CHR-bank-register co-key; the full discovery layer (BitLedger, splice verification via `poke_ram`,
`cap_sig` key slot) as its own lane with its own gates (D2 §1D-1E, G-RG2/3/5) once the room layer is
banked. Sequencing rule stands: core work completes + certs pass before any prereg run that uses it.

### §9 Implementation plan — 6 worktree tasks (no `git stash` across lanes)

Dependency order: **T1 → {T2, T3, T4 in parallel} → T5 → T6.** RG-0 (inside T4) must PASS before T5
launches.

| Task | Scope (files) | Deps | Done-when |
|---|---|---|---|
| **T1 — fingerprint core + classifier + schema + lineage** | `RoomIndex` + blake2b/mask helpers + pan/fade/warp classifier; `room_fp` parse (:1448 region); `_xram`/`_xram_local` 7-byte extension + ODO_ALT (:2866/:2892); forced odometer enable; lineage axis (:301/:331/:542); `room_index.json` save/load + resume refusal (:3857); argparse flags. Files: `go_explore_solve.py`, `tests/test_room_fp.py` | — | Unit tests green: mask/hash/settle/intern/classify determinism (synthetic NT streams incl. warp + fade signatures); xram length switch; cross-schema resume refused; flags-off SMB harness byte-identical |
| **T2 — hot-loop wiring** | Settle+classify block before :5421; kind-tagged edge commit + warp veto inside :5476-5481 (template :3192-3204); exemplar ring; `_assign` seeding + lockstep invariants (:4332); adoption-from-unknown guard | T1 | 5-min Zelda smoke: rooms interned, sect/psig transits fire on room change, warp settles mint zero edges (asserted), restore-then-first-step mints zero false transits (asserted) |
| **T3 — router arm + telemetry** | `_room_pools` + V(r) + boundary sublists + inline articulation in `_refresh_sel_cache` (:4064/:4257); arm between ortho and count arms in `select()`; U(r) exit term; aliased down-weight; route_dir macro OR-term (:5359-5378); status-line telemetry | T1 | Arm-off byte-identity; synthetic-archive unit test picks frontier/articulation/U(r)>0 rooms with the count prior; aliased rooms down-weighted |
| **T4 — calibration + fixtures + profiles + RG-0** | `scripts/room_fp_calibrate.py` (auto volatility mask from our own idle/walk frames → mask + receipt md); copy banked probe states/logs into `tests/fixtures/roomgraph/`; RG-0 falsifier as pytest; Zelda + Metroid profiles per §4; 30-min Metroid death-observable probe (stub documented if it fails) | T1 | RG-0 pytest PASSES on all five fixture assertions; masks reproduce probe idle-stability numbers; receipts under `docs/receipts/room_fp/` |
| **T5 — gate RG-1** | Register §6 verbatim in `runs/room_graph/PREREG.md`; run 4×90 min unattended with abort guards; stability + false-merge + edge-replay audits; SMB regression control; perf check | T2+T3+T4 (RG-0 pass) | `docs/receipts/room_graph/RG1_zelda_<date>.md` with pass/kill verdict per pre-registered numbers, no post-hoc edits |
| **T6 — RG-2 + bank/ship** | Metroid secondary run + scene-noise reconciliation; memory update; ship commit via git-workflow skill; router default set per RG-1b outcome | T5 | RG-2 report banked either way; feature merged default-off; CLAIMS.md entry |

Checkpoints: after T1 (suite + parity green, flags-off byte-identity), after T2-T4 (RG-0 pass +
Zelda/SMB smokes), after T5 (gate verdict — the only checkpoint that consumes pre-registered
compute). No nes_core rebuild anywhere in v1, so the no-core-rebuild-under-live-prereg constraint is
never in play and the M4 stays on runs, not builds.
