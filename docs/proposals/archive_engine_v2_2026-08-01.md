# Archive Engine v2 — mmap-backed Go-Explore cell store

Status: proposal (2026-08-01). Author: M. Stits.
Scope: replace the pickle-dict archive behind `GoExploreArchive`
(`src/training/go_explore.py`) and the solver
(`scripts/go_explore_solve.py`) with a memory-mapped, zero-copy cell store
that holds flat sps and non-blocking persistence to 10M cells.

This document maps an already-decided research direction (mmap arena +
file-backed Fenwick tree + WAL + `madvise` tiering) onto **our** code and
**our** measured data. It does not re-argue the direction; it grounds each
piece in the fields, call sites, and receipts we actually have tonight.

---

## 0. Drivers and receipts

### 0.1 What broke tonight (driver-cited, receipted separately)

- Archives reached **1.6M cells / ~14+ GB**.
- Solver throughput degraded **2800 → 780 sps** as cells grew.
- The pickle flush starves workers: **610 MB / 2 min cadence, sps 1563 → 322**
  during a flush.
- Loads take **minutes**.
- Target: **10M cells with flat sps and stall-free persistence.**

### 0.2 What I measured directly (2026-08-01, read-only probes)

Grounding archives (idle dirs; the live `stage1_v7b*` solver was not
touched):

| archive | cells | file | blob size | blob % of file | non-blob /cell |
|---|---|---|---|---|---|
| `runs/breadth_contra/stage1_v6_resume/archive.pkl` | 652,059 | 13.85 GB | 21,164 B (uniform) | **99.6 %** | 79 B |
| `runs/breadth_contra/stage1_v7_doctrine/archive.pkl` | 1,443,275 | 30.7 GB | 21,164 B (uniform) | 99.6 % | — |
| `runs/regress_pre/archive.pkl` (SMB, older code) | 6,536 | 138.8 MB | 21,164 B (uniform) | 99.6 % | — |

Probe receipts (`.venv/bin/python`, this machine, M4 / 128 GB):

- **Sequential read floor** (bounded 2 GiB prefix of the 13.85 GB pkl):
  **7,312 MB/s** → pure-read floor for 13.85 GB ≈ **1.9 s**. I/O is *not*
  the bottleneck.
- **Object-build floor** (50k `Cell` objects, distinct 21 KB blobs, warm):
  **269,404 cells/s** → extrapolated warm object-build 652k ≈ **2.4 s**,
  1.6M ≈ **5.9 s**, 10M ≈ **37 s**. (An earlier run reported 1.05M cells/s;
  that was pickle memo-deduping a shared blob object — corrected here with
  distinct blobs.)
- **Fenwick weighted sample**, pure-Python, 10M weights: **2,967 ns/sample**
  (≈340k samples/s single-thread); tree = **80 MB** (i64) / 40 MB (i32
  fixed-point). Rust/NEON is 10–50× faster (~50–300 ns).

### 0.3 Diagnosis: one root cause, three symptoms

The warm object-build floor (2.4 s for 652k) is *far* below the receipted
"minutes." The gap is the mechanism v2 must kill:

**The archive is a live Python `dict` of N `Cell` objects, each owning a
21 KB `bytes` blob.** Everything scales with N *and* with total blob bytes
held resident:

1. **Load = minutes**, not the 2.4 s floor, because a resume rebuilds 14 GB
   of live Python objects *while the solver already holds ~46 GB RSS*
   (measured on the running process tonight). That pushes the working set
   into `vm_compressor`; pickle's per-object allocation + GC then runs at
   compressor speed. mmap eliminates the object build entirely — there is no
   14 GB of live objects to construct.
2. **sps degrades with N** because selection scans the dict. The archive's
   native weighted pick is `list(cells.values())` + O(N) weight compute +
   O(N) `random.choices` **per sample** (`go_explore.py:215-217`); the
   solver's `_refresh_sel_cache` is a full O(N) scan whose 2%-growth throttle
   exists *only* to amortize that cost (`go_explore_solve.py:686-691`, whose
   own comment records "sps decayed 2378→475"). At 652k cells the v6 resume
   log shows steady-state **sps ≈ 987** (`progress.jsonl`), corroborating
   that per-cell overhead, not emulation, dominates at scale.
3. **Flush stalls** because `flush()` is a synchronous full-rewrite
   `pickle.dump` of the entire `_cells` dict (all blobs) plus the entire
   `traces` dict, on the explore thread, every `flush_secs`
   (`go_explore_solve.py:1024-1026, 1054-1058`). Writing GB synchronously
   blocks `pool.step_all`.

v2 fixes all three at the storage layer: compact columnar metadata that
stays resident, blobs paged out until a cell is chosen, O(log N) sampling,
and append-only incremental persistence.

---

## 1. Storage architecture

An archive is a **directory**, not a file. Layout:

```
<out>/archive/
  manifest.json      schema version, strides, counts, game id, trackers, CRC
  cells/             struct-of-arrays (SoA) metadata columns (see §1.2)
    score.f64  steps.u32  visits.u32  chosen.u32  flags.u8
    barren.u16  sect.u8  area.u8  gx.u16  blob_slot.u32  key_off.u64  key_len.u16
  keys.bin           append-only variable-length key codec (§2)
  keyhash.bin        open-addressing hash index key -> key_id (§2)
  blobs.bin          fixed-stride 21,164 B savestate arena (§1.1)
  traces.bin         variable-length action-trace arena (§4)
  weights.fen        file-backed Fenwick tree over selection weight (§3)
  wal.log            append-only write-ahead log (§1.4)
```

All of `cells/*`, `keys.bin`, `keyhash.bin`, `blobs.bin`, `traces.bin`,
`weights.fen` are `mmap`'d. The resident set is the hot metadata columns
(~50 MB at 652k, ~800 MB at 10M) plus whatever blobs `madvise(WILLNEED)` has
faulted in; the rest stays paged on APFS.

### 1.1 Blob store — SEPARATE fixed-stride arena (decision)

**Decision: a separate `blobs.bin` arena with a fixed stride, not inline.**

Rationale grounded in our data:

- Blobs are **uniform 21,164 B within a run** (measured on three archives
  spanning SMB and Contra). Variable-length blobs are a non-problem in
  practice; the "variable-length" worry in the brief applies to *keys*, not
  blobs. Stride = `manifest.blob_stride`, read once from the first
  `pool.save_worker_state` and asserted constant thereafter.
- Blobs are **99.6 % of the bytes**. If inlined into the cell record, every
  selection scan (§3) and every `_refresh_sel_cache` rebuild would drag
  21 KB/cell through cache to read a 2-byte `gx`. Separation keeps the
  metadata table ~48 B/cell — the columns the hot loop actually touches stay
  resident while blobs stay evicted until a cell is *chosen* for return.
- Zero-copy: a blob is `mmap[slot*stride : slot*stride+stride]`, handed to
  `pool.load_worker_state` as `&[u8]` with no Python `bytes` copy. The Rust
  side already takes a byte slice.

Record → blob: `Cell.blob_slot: u32`; offset = `slot * blob_stride`. The
synthetic start cell (`Cell.state = None`, allowed by the dataclass,
`go_explore.py:79`) uses sentinel `blob_slot = 0xFFFFFFFF`.

**Domination replaces a blob** (`go_explore.py:171-176`,
`go_explore_solve.py:603-609`): a dominating record writes a *new* slot,
repoints `blob_slot`, and pushes the old slot onto a freelist recorded in
the WAL. New cells append. This is the real scaling wall: 10M × 21 KB =
**212 GB of blob arena on disk** (not RAM — APFS + `madvise(DONTNEED)` keep
the resident set small, but disk and blob churn are finite). Mitigations are
staged (§5 stage 4): freelist reuse, offline compaction, optional zstd blob
column, and LRU blob eviction for deep-interior cells that will never be
re-selected (keep the cell row + key for the door graph; drop only the
restorable state).

### 1.2 Cell metadata record — columnar (SoA), not array-of-structs

The `Cell` dataclass fields (`go_explore.py:72-92`) plus the solver's
dynamic `barren` attribute (`go_explore_solve.py:860-866`) become columns:

| column | type | source |
|---|---|---|
| `score` | f64 | `Cell.best_score` — solver score `sect*10000+gx+bonus` exceeds 2^24, needs f64 (`go_explore_solve.py:603`) |
| `steps` | u32 | `Cell.best_steps` |
| `visits` | u32 | `Cell.visits` |
| `chosen` | u32 | `Cell.times_chosen` |
| `flags` | u8 | bit0 = `Cell.explored` |
| `barren` | u16 | solver R2 dynamic attr — promoted to first-class |
| `sect` | u8 | **denormalized** `key[0]` |
| `area` | u8 | **denormalized** `key[-5]` |
| `gx` | u16 | **denormalized** `key[-1]` |
| `blob_slot` | u32 | index into `blobs.bin` (§1.1) |
| `key_off` / `key_len` | u64 / u16 | slice into `keys.bin` (§2) |

`sect`/`area`/`gx` are denormalized out of the key because the selection
band filter reads `key[0]`, `key[-5]`, `key[-1]` on **every** cell every
rebuild (`go_explore_solve.py:698-708, 726-740, 667-675`). As key tuple
indexing in Python that is the dominant rebuild cost; as three contiguous
columns it is a vectorizable (NEON) mask over `sect`/`area`/`gx` — the
band-membership query the code already wants.

**Why SoA over array-of-structs:** the hot scans touch 3–5 fields, not the
whole record. SoA reads only the `gx`/`sect`/`area`/`score`/`chosen` columns
contiguously; AoS would stride a 48-byte record to read 2 bytes. SoA is also
the Apple-Silicon-friendly shape (column scans vectorize) and — see §1.3 —
maps 1:1 to Arrow primitive arrays for the offline analytics path.

`key_id` (the dense row ordinal 0..N-1) is the single identity for: the SoA
row, the Fenwick leaf, the blob-slot lookup, and the door-graph node. This
unifies the solver's ad-hoc `_key_ids` interning (`go_explore_solve.py:464`)
with the archive.

### 1.3 Arrow IPC vs custom fixed-offset — DECISION

**Decision: custom fixed-offset SoA columns, not Arrow IPC**, with an Arrow
*export* hatch for read-only analytics.

Against our actual schema and access pattern:

- Our schema is a handful of fixed-width scalar columns + one fixed-stride
  blob arena + variable-length keys. Arrow's value is nested types,
  cross-language interop, and encodings — none of which is the bottleneck.
- Our hot access is **random point-update + weighted random sample**:
  `chosen += 1`, `explored = True`, score replacement, blob-slot swap,
  append — thousands/s from the explore loop. Arrow arrays are **immutable
  by construction**; in-place mutation means either the not-really-supported
  builder path or a full record-batch rewrite — the exact synchronous
  rewrite v2 exists to eliminate.
- Custom format lets us own the schema-version byte and the exact stride;
  the migration reader (§4) is ~30 lines.
- **Interop is preserved cheaply:** fixed-width SoA columns map 1:1 to Arrow
  primitive arrays, so `observatory.py` / offline analysis can `mmap` the
  same column files as Arrow buffers **read-only** without Arrow's
  immutability tax touching the hot path. We get Arrow where Arrow helps
  (analytics) and avoid it where it hurts (mutation).

### 1.4 Non-blocking persistence

- **Incremental, not full-rewrite.** Every mutation appends a WAL record
  (`{op, key_id, column, value}` / blob-slot allocation) to `wal.log` and
  writes the column cell in place in the mmap. The synchronous
  `pickle.dump(whole dict)` flush disappears.
- **Durability without stalls.** Periodically (the old `flush_secs` cadence)
  call `msync(MS_ASYNC)` on dirty column pages and the blob arena — the
  kernel writes back on its own schedule; the explore thread does not block.
  A `manifest.json` pointer (last durable `key_id`, WAL offset, CRC) is
  written last, `fsync`'d, and renamed atomically — the commit barrier.
- **Double-buffered manifest/pointer region + atomic swap** so a crash mid-
  update leaves either the old consistent pointer or the new one, never a
  torn one. This mirrors the atomic + tear-tolerant checkpoint pattern
  already adopted in the stability audit (2026-07-20).
- **Recovery** = mmap the columns, replay the WAL tail past the manifest's
  durable pointer, rebuild the Fenwick deltas the tail implies. O(tail), not
  O(N).

---

## 2. Key interning + hash index

Receipts on the key shape (measured):

- Keys are **variable-length nested tuples**: arity **11** in current Contra
  (`(sect, tb, kk, psig, loops, route_sig) + cell_fn(5)`,
  `go_explore_solve.py:583-585`), arity **9** in older SMB archives (prefix
  had no `tb`/`kk`). Nested tuples (`psig`, `route_sig`) sit at drifting
  slots (indices 3 & 5 in Contra, 1 & 3 in old SMB). **Key schema has
  already evolved** — the codec must be arity- and layout-agnostic.
- Keys pickle to **~41 B mean** (max 48 in samples); a compact codec gets
  this to ~10–20 B.
- The solver **already interns keys to `u32` ids** in `_key_ids` for the
  door graph (`go_explore_solve.py:464, 595-598`) — v2 makes that interning
  the archive's durable primary index instead of a per-run throwaway.

Design:

- **`keys.bin`** — append-only. Each key encoded with a tag-length-value
  codec (not pickle): varint arity, then per element a tag byte
  (`INT` → zigzag varint | `TUPLE` → recurse). Handles ints and nested
  tuples of arbitrary depth; arity-agnostic, so the 9↔11 drift costs nothing.
  Round-trips exactly (decode == original tuple).
- **`key_id: u32` = the SoA row ordinal**, assigned in insertion order and
  persisted. One id serves the cell row, Fenwick leaf, blob slot, and door
  node. Because it is durable, the door graph `_adj`
  (`go_explore_solve.py:600-601`) survives resume instead of rebuilding from
  scratch each run.
- **`keyhash.bin`** — open-addressing hash table (64-bit hash of the
  canonical key bytes → `key_id`, linear-probe, load factor ≤ 0.7). This
  replaces `self._cells.get(key)`, called every step in `observe`
  (`go_explore_solve.py:604`) and in `record` (`go_explore.py:158`).
  Lookup: hash the incoming key's canonical bytes, probe, byte-compare the
  candidate in `keys.bin`. O(1) expected.
- **Dict semantics preserved.** Python tuples are equal iff element-wise
  equal; the canonical byte encoding is equal iff the tuples are equal, so
  `hash`+byte-compare reproduces `dict` membership exactly (including nested
  tuples). No behavior change to domination/dedup.

---

## 3. Fenwick integration with the EXISTING selection semantics

There are **two** selection surfaces; both must keep byte-for-byte behavior
under a fixed seed or v2 stays behind a flag (§6).

### 3.1 Archive-native weighted selection (trainer path)

`GoExploreArchive.select_return_cell` / `_selection_weight`
(`go_explore.py:183-220`), consumed by the trainer via
`select_return_states` (`trainer.py:6325, 8145, 8261`). This is the pure
O(N) `random.choices(cells, weights, k=1)` scan — the clearest Fenwick win.

Weight (all terms cited):

- `base = 1/sqrt(times_chosen + 1)` (`go_explore.py:199`)
- `frontier_bonus = 2.0 if not explored else 1.0` (`:200`)
- `W_location = W_LOCATION_COEF * (len(neighbors) - h)` where `h` = archived
  horizontal neighbors (`:202-206`, coef `:55`, `horizontal_neighbors`
  `:343-361`).

Mapping:

- Fenwick leaf `i` holds `w_i` as i32 fixed-point (`weight * 2^16`). Sample =
  draw `u ∈ [0, total)`, `lower_bound` on prefix sums = **O(log N)**.
- **decay-on-chosen** (`:218` `times_chosen += 1`, `:219` `explored = True`):
  recompute `w_i`, `Fenwick.add(i, w_i' - w_i)` = O(log N). The
  frontier_bonus 2→1 flip on first `explored` is the same single delta.
- **W_location on insert:** a new cell changes `h` for its ≤2 horizontal
  neighbors → recompute those ≤2 leaves, O(log N) each. Bounded, not O(N).

### 3.2 Solver selection (hot path, `go_explore_solve.py:686-776`)

Three arms:

- **Deep-frontier arm** (`select`, prob `deep_bias`, `:715-743`): a
  *filtered uniform* pick over a precomputed band (cells in
  `max_sect`/`max_area` with `gx ≥ topgx - rand(24)`), **not** weighted. This
  arm does not need Fenwick; it needs a fast **range query** over the SoA
  `sect`/`area`/`gx` columns (§1.2). With denormalized columns the band mask
  is a vectorized pass, so `_refresh_sel_cache`'s 2%-growth throttle
  (`:716-719`) — which exists solely to amortize the O(N) Python rebuild —
  can be loosened or dropped. The `barren` throttle
  (`:733-735`) becomes a column predicate.
- **Count arm** (`sel_mode == "count"`, `:744-773`): rejection sampling of
  `W = 1/sqrt(chosen+1) * (score/maxscore + 0.1)`, optional `door_weight`
  multiplier for articulation-point cells. **Recommend moving this onto the
  Fenwick tree** (same tree as §3.1, weight formula swapped per `sel_mode`):
  rejection sampling degrades badly when `Wmax >> mean` (one deep
  high-score cell starves acceptance and burns the 64-draw cap, `:755`);
  Fenwick is exact and O(log N) regardless of skew. The door multiplier is a
  per-cell weight delta applied on the async door scan over its bounded door
  set (`:767-769`).
- **Legacy arm** (uniform over the cached list, `:774-776`): a single
  `randint` over `key_id ∈ [0, N)` — trivially O(1) on the column store.

### 3.3 R2 barren credit / throttle

`_assign` credits/debits the source cell's `barren` after each burst
(`go_explore_solve.py:856-866`): novelty resets it to 0, a dry burst
increments it. When `barren` crosses `frontier_throttle` the cell is
excluded from selection (`:733-735, 762-764`). In Fenwick terms: cross → set
leaf weight 0 (`add(-w_i)`); reset → restore. Both O(log N), on the
already-per-burst credit path. No new scan.

**Net:** every selection = one O(log N) sample; every mutation = one O(1)
column write + one O(log N) Fenwick update. No O(N) scan on the hot path →
**sps flat as N grows**, which is the whole point.

---

## 4. Migration path

- **Read-old-pickle → write-new on first open.** `Archive.open(dir)`:
  `cells/` + `manifest.json` present → mmap directly; else `archive.pkl`
  present → one-time import: unpickle the dict, stream cells into the SoA
  columns + blob arena + `keys.bin` + Fenwick, write `manifest(schema=2,
  source="pickle")`, rename `archive.pkl` → `archive.pkl.bak`. The importer
  *is* the stage-0/1 read-path tool (§5).
- **resume-archive compatibility** (`--resume-archive`,
  `go_explore_solve.py:657-679`). Today: `archive.load(pkl)` +
  `traces.update(pickle.load)` + a full scan to rebuild `max_area` /
  `max_gx_in_area` / `max_sect` (`:667-675`). v2: format-detect; new-format
  resume is an **mmap open (seconds)** with the trackers persisted in
  `manifest.json`, so the rebuild scan is gone. Old-format resume
  auto-imports once (the minutes-long path happens exactly once, then never
  again).
- **traces.pkl handling.** `traces` is a parallel dict
  `key → (root_id, trace_bytes, loops, route_sig, sect, psig, kills)`
  (`go_explore_solve.py:611`, 596 MB in v6). Two observations drive the
  design: (a) `loops`, `route_sig`, `sect`, `psig` are **already key
  components** (`:585`) — storing them again in traces is redundant and v2
  drops them, deriving from the key; (b) `trace_bytes` is variable-length
  (the action list). Fold traces into the same `key_id` space: scalar
  columns (`root_id` as a small interned id; `kills`) on the cell row, and
  `trace_bytes` in a `traces.bin` arena (offset+len columns) mirroring the
  blob arena. This removes the second full-dict pickle flush entirely.
- **Schema tolerance.** `manifest.schema` version byte; the key codec is
  arity-agnostic (the 9↔11 receipt proves this must hold); optional columns
  (`barren`, trace columns) may be absent in an imported older archive and
  readers default them.
- **Export hatch.** `Archive.to_pickle()` rebuilds the legacy `_cells` dict
  so any unported consumer (the trainer's `select_return_states`,
  `go_explore_2_1.py`, `observatory.py`, `pixel_phase.py`) keeps working
  during the transition. This decouples the storage swap from the consumer
  port (§5 stage 3).

---

## 5. Staged build plan (per-stage receipts)

Stages 0–1 are **pure Python + numpy, read-only, no Rust** — buildable and
receiptable tomorrow **without touching the PGO wheel** (see §6 build-rule
note). The Rust engine (stage 2+) ships in the one attributed rebuild cycle.

- **Stage 0 — codec + writer + importer (offline).** Key TLV codec, blob
  arena writer, SoA column writer, `pickle → v2` importer.
  *Receipt:* import `runs/breadth_contra/stage1_v6_resume/archive.pkl` →
  v2 dir; assert cell count `== 652,059`, every key decodes to the original
  tuple, every blob byte-identical, total bytes within 1 % of 13.85 GB. Runs
  against the idle v6 dir, never the live `v7b` dir.

- **Stage 1 — read-only mmap snapshot + benchmark vs pickle.**
  `Archive.open()` reader (no mutation): `len`, `get(key)`, column iterators,
  `select()` over a Fenwick built at open.
  *Receipt (the headline bench):* on the **real tonight-sized** archive
  (v6_resume 13.85 GB; and v7_doctrine 30.7 GB / 1.44M cells *only once the
  v7b solver has exited after ~07:00*, to avoid the 46 GB-resident memory
  contention that would poison the pickle side) measure —
  (a) open wall-time v2 vs `pickle.load` (target: v2 **< 1 s** vs pickle
  "minutes"); (b) RSS after open (target **< 500 MB** vs ~14 GB);
  (c) 1M `select()` samples wall; (d) 1M `get(key)` lookups. Publish the
  table next to the §0.2 floors.

- **Stage 2 — live single-writer engine (mutation + WAL + async flush).**
  append/update/point-Fenwick, double-buffer + `msync(MS_ASYNC)` + WAL.
  Wire behind `--archive-engine v2` in `go_explore_solve.py`; **default stays
  pickle** until proven.
  *Receipt:* a short bounded solve (**≤ 2 workers**, minutes-capped, a
  *scratch* out-dir — never `breadth_contra`) with v2 vs pickle: assert
  identical cell-count trajectory and solution shas under a fixed seed
  (determinism, §6), and measure flush-time sps (target: no dip vs the
  1563→322 receipt) and steady sps flat past 1M cells.

- **Stage 3 — port consumers, retire pickle.** Move
  `trainer.select_return_states`, `go_explore_2_1.py`, `observatory.py`,
  `pixel_phase.py` onto the v2 reader; keep `to_pickle()` for anything
  unported.
  *Receipt:* `tests/test_go_explore.py` +
  `tests/test_vanilla_ppo_go_explore_smoke.py` + `make parity` green;
  observatory renders from a v2 dir.

- **Stage 4 — blob tiering + compaction.** `madvise(WILLNEED)` on a chosen
  cell's blob, `DONTNEED` after restore; freelist reuse for domination-
  orphaned slots; optional zstd blob column; LRU blob eviction for deep-
  interior non-frontier cells.
  *Receipt:* a 10M-cell synthetic (or a real 5-3/8-4-scale run) holds RSS
  flat; blob disk bounded by compaction.

Each stage gates on the prior stage's receipt.

---

## 6. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| **Crash consistency** (torn write mid-flush) | med | append-only WAL + double-buffered manifest pointer + `fsync`'d atomic rename as the commit barrier; recovery replays WAL tail past the durable pointer (§1.4). Precedent: stability-audit atomic/tear-tolerant checkpoints (2026-07-20). |
| **Schema evolution** | high (already happened: arity 9→11) | `manifest.schema` byte; arity/nesting-agnostic key codec (§2); optional columns default when absent; importer version-tolerant. |
| **Blob churn / disk blowup** (10M×21 KB = 212 GB; domination orphans slots) | high | freelist reuse, offline compaction, optional zstd, LRU blob eviction of deep-interior cells (keep row+key, drop restorable state). This is the true scaling wall — flagged, not hand-waved. |
| **Determinism drift** (v2 selection RNG order ≠ v1) | med | v2 must reproduce the byte-identical selection sequence under a fixed seed, or solution shas change (regression contract, `go_explore_solve.py:152-154`). Gate v2 behind `--archive-engine v2`; do **not** claim solution-sha parity unless the RNG draw structure matches. |
| **Concurrency** (door thread reads live structure) | low | single-writer model matches reality: the explore loop is the sole writer; the door daemon takes a brief snapshot under `_door_lock` (`go_explore_solve.py:838`). Give the door thread a seqlock or the same snapshot lock over the Fenwick/columns. Do **not** design for multi-writer. |
| **Wheel-rebuild coupling** | med | stages 0–1 are pure Python/numpy and receipt the *entire* read + bench story without any Rust, de-risking the design before the wheel is touched (§ build rule). |

### Python / Rust split — recommendation

- **Rust owns storage primitives:** the mmap arena + SoA columns + Fenwick
  tree + WAL, exposed as a PyO3 object (sibling crate, or on `nes_core`).
  Rationale: `observe`'s per-step `get(key)` + record + point-update runs at
  3000+ sps × workers; the pure-Python Fenwick (2,967 ns/sample) is fine for
  the once-per-burst *selection* cadence but the per-step storage calls want
  native. Blobs already cross into Rust (`pool.save_worker_state` /
  `load_worker_state`), so a Rust-owned arena means zero-copy `&[u8]`
  straight into the Pool with no Python `bytes` round-trip.
- **Python owns policy/semantics:** `cell_fn`, key composition (the
  `(sect,tb,kk,psig,loops,route_sig)+cell_fn` assembly stays in
  `go_explore_solve.py:585`), selection-arm orchestration (deep_bias vs
  count vs legacy), the async door scan, and R2 credit. Python calls the
  Rust archive for storage + sample primitives only.
- **Boundary = the `GoExploreArchive` public surface**
  (`record`/`select_return_cell`/`select_return_states`/`save`/`load`/`cells`)
  plus the solver's `observe`/`select` storage calls. Keep the Python
  `GoExploreArchive` as a thin façade over the Rust engine so
  `tests/test_go_explore.py` and the trainer path stay source-compatible.

---

## Appendix — receipt commands (2026-08-01, read-only)

- Structure/blob probe:
  `PYTHONPATH=<repo> .venv/bin/python scratchpad/inspect_archive.py <archive.pkl>`
  → cells, key arity, nested-tuple slots, blob size, state=None %.
- Perf floors:
  `.venv/bin/python scratchpad/probe_perf.py` (I/O floor, Fenwick microbench)
  and `probe_unpickle.py` (distinct-blob object-build cost).
- All probes ran against idle dirs (`stage1_v6_resume`, `regress_pre`,
  `stage1_baseline_collapsed_cells`); the live `stage1_v7b*` solver
  (10 workers, ~46 GB RSS) was not touched.

### Not yet verified (deferred to the rebuild)

- The v2 open/RSS/select numbers are **targets** derived from the §0.2
  floors, not yet measured against a built engine — that is stage 1's
  receipt.
- The "minutes" load and 2800→780 / 1563→322 sps figures are the driver's
  receipts, reproduced in mechanism (memory-pressured object build;
  synchronous full-pickle flush) but not re-timed end-to-end here, because
  doing so would require loading the full 14 GB under the live 46 GB-resident
  solver.
