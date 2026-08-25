# RG-1 / RG-0 / RG-2 pre-registration — room-graph engine (Zelda)

**LOCKED. This file is a pre-registration. Once written it may not be edited to
match results — any correction after a run has started gets a new dated
addendum file, never an edit to the numbers below.**

## Registration provenance (proves this is a registration, not a post-hoc rewrite)

| Field | Value |
|---|---|
| Registered at (UTC) | `2026-08-25T08:43:50Z` |
| Registered against repo HEAD | `9bbd65f811f6ce339c7235dd9c4b71d0a365c129` |
| Source document | `docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md`, §6 |
| Source document sha256 (whole file, at HEAD above) | `f8b2104a3281b747084338126ff8c457fb0457ee3f95310b4769a81d66d27883` |
| §6 section text sha256 (lines 212-249 of the source document, the exact span transcribed verbatim below) | `41d9a3d07114fc4b1fa457079168e3cc0fd12474cf90b18b31e111abfa29b0fb` |
| Transcription method | `sed -n '212,249p' docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md` piped directly into this file's §6 block — no manual retyping, no paraphrase, no threshold edits |
| RG-0 status at registration time | 9/9 PASS (`tests/test_rg0_roomgraph.py`, re-confirmed 2026-08-25 per `CLAIMS.md`) — RG-1 may launch per the BINDING cheap-premise-first sequencing rule |
| RG-1 status at registration time | **NOT YET RUN.** No run has been launched under this registration as of the timestamp above. Any receipt filed after this timestamp is scored against the numbers below exactly as written, with no adjustment. |

Anyone auditing this file for tampering can re-run
`sed -n '212,249p' docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md | shasum -a 256`
against the source doc at commit `9bbd65f` and confirm it reproduces the §6
section checksum above, and can diff the "§6 — verbatim" block below
byte-for-byte against that same span.

---

## §6 — verbatim (transcribed exactly from `docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md`, no paraphrasing, no numeric changes)

> ### §6 Pre-registered validation gate (register verbatim in `runs/room_graph/PREREG.md` BEFORE T5; no post-hoc edits)
>
> **RG-0 — offline falsifier (blocks all live runs; cheap-premise-first, BINDING).** Replay the
> detector+classifier over the banked probe fixtures as a pytest. PASS requires ALL: Zelda east exit ⇒
> pan-E, exactly one new node; Zelda death ⇒ warp, **zero** edges; Zelda 300-idle-frame log post-mask ⇒
> exactly 1 hash; Metroid door1/door2 ⇒ exactly 2 pan edges; Metroid spurious scene bumps (scene 4
> inside room 1) ⇒ zero extra nodes. Any failure ⇒ stop, no live compute.
>
> **RG-1 — Zelda (primary).** 4 unattended runs × 90 min, 12 workers, fs4,
> `roms/zelda_start_ctrl.state.bin`, seeds {0,1} × {`--room-bias 0.25`, `--room-bias 0`}. Abort
> guards: SPS floor, archive-size cap, RSS guard.
> - **RG-1a validity:** ≥30 distinct settled rooms by 90 min (seed 0, either arm); ≥1 **fade** edge
>   banked (cave or dungeon entry — the class scene is blind to); zero warp-minted edges (audited);
>   false-merge audit: room pairs with odometer bboxes disjoint by >512 px sharing a fingerprint = 0.
>   Stability: 20 random rooms × 3 archived cells from disparate lineages restored on a fresh pool ⇒
>   re-settled ordinal == recorded for ≥95% of 60 restores.
> - **RG-1b routing lift:** distinct rooms (router ON) ≥ 1.25× (OFF) at 90 min on BOTH seeds;
>   tie-break time-to-25-rooms.
> - **RG-1c integrity:** `room_fp` absent + `--room-bias 0` ⇒ SMB 1-1 5-min determinism harness
>   byte-identical vs pre-branch build; clear events in all Zelda runs == 0.
> - **RG-1d perf:** SPS ≥ 90% of room_fp-off Zelda baseline; else engage `sample_every: 2` and
>   re-measure before any verdict.
> - **RG-1e edge validity:** 20 sampled edges replayed from `exemplar_cell` restore + exemplar actions
>   under sticky p=0.25 ⇒ ≥80% reproduce the same (src, dst, kind).
> - **Kill criteria (pre-registered):** <10 rooms in any 60-min seed-0 window ⇒ fingerprint design
>   falsified for Zelda, lane stops with receipt. Stability <80% or >50% false splits in a 30-node
>   manual audit ⇒ mask/settle design falsified. Router lift <1.0× on both seeds ⇒ router ships
>   default-off permanently (identity layer survives on RG-1a); not a lane kill.
>
> **RG-2 — Metroid (secondary, report-only, non-blocking).** 90 min, seed 0, odometer profile:
> ≥8 fingerprint rooms incl. the measured door corridors; the probed 3-room stretch = exactly 3 nodes
> (scene noise fully absorbed); fingerprint room count ≤ scene-ordinal count; door-macro injections >0
> with ≥1 transit within 30 s of an injection; zero warp edges. Death detection is an honest stub (no
> probed observable — the warp veto is the only defense; NG corpse-frontier caveat applies and is why
> RG-2 cannot gate).
>
> Receipts: `docs/receipts/room_graph/RG1_zelda_<date>.md` (+ RG-2 report) with run configs, seeds,
> archive stats, audit outputs.

No numeric threshold above (30 rooms, 1.25×, 95%, 90%, 80%, 20 edges, 10 rooms,
80%/50%, 8 rooms, 512 px, etc.) may be changed once this file is committed.
A design defect discovered after launch is a new dated addendum, not an edit
to this block.

---

## Exact invocation — the 4 RG-1 Zelda runs

All four runs use `scripts/go_explore_solve.py` (repo HEAD `9bbd65f`), the
`configs/zelda_roomfp.yaml` profile (the RG-1 profile per its own header
comment: `frame_skip: 4` — this supplies "fs4"; `configs/zelda.yaml` stays
the untouched RG-1c/RG-1d comparison baseline), and root state
`roms/zelda_start_ctrl.state.bin` — confirmed present at that exact path in
the repo at registration time (`ls roms/zelda_start_ctrl.state.bin`
succeeds; no wave-2 rename occurred for this file). 12 workers, 90 minutes
wall-clock each, matrix = seeds {0,1} × room-bias {0.25, 0}.

```
# Run 1/4 — seed 0, router ON
.venv/bin/python scripts/go_explore_solve.py \
  --out runs/room_graph/rg1_zelda_seed0_bias025 \
  --root-state roms/zelda_start_ctrl.state.bin \
  --profile configs/zelda_roomfp.yaml \
  --workers 12 --minutes 90 --seed 0 --room-bias 0.25

# Run 2/4 — seed 0, router OFF (control)
.venv/bin/python scripts/go_explore_solve.py \
  --out runs/room_graph/rg1_zelda_seed0_bias000 \
  --root-state roms/zelda_start_ctrl.state.bin \
  --profile configs/zelda_roomfp.yaml \
  --workers 12 --minutes 90 --seed 0 --room-bias 0

# Run 3/4 — seed 1, router ON
.venv/bin/python scripts/go_explore_solve.py \
  --out runs/room_graph/rg1_zelda_seed1_bias025 \
  --root-state roms/zelda_start_ctrl.state.bin \
  --profile configs/zelda_roomfp.yaml \
  --workers 12 --minutes 90 --seed 1 --room-bias 0.25

# Run 4/4 — seed 1, router OFF (control)
.venv/bin/python scripts/go_explore_solve.py \
  --out runs/room_graph/rg1_zelda_seed1_bias000 \
  --root-state roms/zelda_start_ctrl.state.bin \
  --profile configs/zelda_roomfp.yaml \
  --workers 12 --minutes 90 --seed 1 --room-bias 0
```

`--room-bias` defaults to `0.0` in the argparse definition
(`scripts/go_explore_solve.py`, `ap.add_argument("--room-bias", ...,
default=0.0, ...)`), so the two OFF-arm invocations above pass it
explicitly for the record rather than relying on the default, per §6's
"register verbatim" spirit — the command line itself should say what ran,
not require reading the script to know.

**Abort guards named in §6** ("SPS floor, archive-size cap, RSS guard") are
not exposed as `go_explore_solve.py` CLI flags as of this registration —
grep of the script's `add_argument` calls and of `scripts/*.py` for
`sps_floor` / `archive_size_cap` / `rss_guard` / `abort_guard` finds no
matching flag or wrapper in the repo at HEAD `9bbd65f`. Whoever launches
these 4 runs must supply the abort-guard wrapper (or monitor equivalents
manually) separately; this registration does not invent guard thresholds
that don't exist in the design doc or the code, and does not silently drop
the requirement either.

## Exact receipt paths each run will produce

Per-run raw output (written by `go_explore_solve.py` itself under `--out`,
matching the existing convention — see e.g.
`docs/receipts/games/zelda_onboarding_2026-08-10.md` §6 for the same
script's output shape: periodic JSON status lines, a final `done:` summary,
`room_index.json` on flush/exit):

- `runs/room_graph/rg1_zelda_seed0_bias025/` (stdout/status log, `room_index.json`, archive)
- `runs/room_graph/rg1_zelda_seed0_bias000/` (stdout/status log, `room_index.json`, archive)
- `runs/room_graph/rg1_zelda_seed1_bias025/` (stdout/status log, `room_index.json`, archive)
- `runs/room_graph/rg1_zelda_seed1_bias000/` (stdout/status log, `room_index.json`, archive)

Consolidated gate verdict (per §6's own "Receipts:" line and the §9 T5
done-when column, "no post-hoc edits"):

- `docs/receipts/room_graph/RG1_zelda_<date>.md` — run configs, seeds,
  archive stats, RG-1a/b/c/d/e audit outputs, pass/kill verdict against the
  numbers in the verbatim §6 block above, unedited.
- RG-2 (Metroid, secondary/non-blocking) report banked alongside it per the
  same "Receipts:" line, filename not otherwise specified in the source doc.

---

## RG-1c control harness — status check (does a concrete script exist?)

§6, RG-1c: *"`room_fp` absent + `--room-bias 0` ⇒ SMB 1-1 5-min determinism
harness byte-identical vs pre-branch build."*

**No script or test in the repo is named or documented as "the SMB 1-1 5-min
determinism harness."** What exists instead, checked against the T1-T4
implementation (commit `3601c45`) and its claims/capability-report trail:

- The flags-off byte-identity property that has actually been exercised so
  far is a **16,000-step SMB solve** run ad hoc against pre-branch HEAD
  (sha256-identical RAM/archive/traces across all 8 workers) — cited in the
  `3601c45` commit message, `CLAIMS.md` (FORGE-PENDING-VALIDATION entry),
  and `docs/research/CAPABILITY_REPORT_2026-08-25.md` §1. This is a
  step-count run, not a wall-clock 5-minute run, and it was invoked
  manually — no checked-in script or Makefile target reproduces it on
  demand.
- §9's T1 done-when column names only "flags-off SMB harness byte-identical"
  with no filename; T2's done-when column separately names a "5-min Zelda
  smoke" (Zelda, not SMB). Neither maps onto an "SMB 1-1 5-min" artifact.
  `tests/test_room_fp.py::test_xram_prefix_is_byte_identical_between_the_two_modes`
  and `tests/test_rg0_roomgraph.py` exist and are real, but neither is a
  timed SMB 1-1 live-solve regression against a pre-branch build; they are
  unit-level.
- `make parity` (`tests/parity/`, `pytest -q -m parity`) is a different
  harness entirely (nes_core vs nes-py fidelity), not a pre-branch-vs-
  post-branch room-graph regression check, and is not 5 minutes / not SMB
  1-1-specific by construction.

**Flagged, not fabricated:** RG-1c as pre-registered has no concrete,
nameable command line to invoke today. Before RG-1c can be scored, T5 must
either (a) point to an existing script this search missed, or (b) mint one
— e.g. a `--minutes 5 --room-bias 0` SMB 1-1 run under
`configs/zelda_roomfp.yaml`'s sibling SMB profile with `room_fp` absent,
diffed sha256 against the equivalent pre-branch-HEAD run — and bank it
under a discoverable name (e.g. `scripts/room_fp_smb_determinism_check.py`
or a documented one-liner in the RG-1 receipt) before the RG-1 runs are
scored against it. This registration does not invent that script or claim
one exists; it names the gap so T5 closes it rather than skipping RG-1c
silently.

---

## Numeric thresholds carried forward unchanged from the design doc

30 distinct rooms · 1 fade edge · 0 false-merges · 20×3 / ≥95% of 60
restores · 1.25× routing lift (both seeds) · tie-break time-to-25-rooms ·
byte-identical (RG-1c) · 0 clear events · ≥90% SPS baseline (else
`sample_every: 2` retry) · 20 sampled edges / ≥80% reproduce · kill: <10
rooms in any 60-min seed-0 window · kill: stability <80% or >50% false
splits in a 30-node manual audit · router lift <1.0× on both seeds ⇒
default-off (not a lane kill) · RG-2: ≥8 rooms, exactly 3 nodes on the
probed 3-room stretch, fingerprint count ≤ scene-ordinal count, >0
door-macro injections with ≥1 transit within 30s, 0 warp edges.

None of the above have been touched. This file transcribes; it does not
tune.
