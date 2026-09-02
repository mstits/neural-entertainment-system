# The Forge — specification, revision 2 (2026-09-01)

Status: DRAFT, ratification by Matthew pending; built 2026-09-01/02, commits 0a1b2f5 (a), 661a4e3 (b), 9881f86 (c), 8c0e16a (e), e80c123 (f), 02ddd13 (d), plus the landing commit that carries this file, the two wall manifests and the two FORGE-GRANT entries. This copy is the reports-directory original verbatim except for this status paragraph and one substitution in section 5, where the attribution grep's literal pattern is replaced by a reference so the committed tree never carries the pattern itself.

Pre-registered before the build; revised once, pre-ratification, against the five review
findings (§0). After ratification, corrections are dated addenda, never edits (LG design
rule 6). Every `path:line` is against the committed tree `git show e9fcf13:<path>` in
`/Users/stits/Documents/macos-emulation-and-training`. Research files: SD
(`research/stall-detection-and-diagnosis.md`), LG (`research/ledger-and-gates.md`), PS
(`research/program-search-for-heuristics.md`), LE (`research/llm-guided-exploration.md`),
PA (`research/repo-prior-art.md`), MAP (`00-MAP.md`). Copied verbatim into
`docs/proposals/` by the build.

## 0. Revision 2: what changed, and one new fact

1. **Campaign verdict was unbuildable as written** — `runs/cv_hall_ortho_ctrl/` holds no
   `A*_result.json`, and cv_hall's staleness spans three directories. §2a now defines a
   wall as a family via a manifest, states what each member shape contributes, re-derives
   `MIN_TERMINAL` from disk, and carries cv_hall's own `distinct_roots:1` (not contra's 2).
2. **Citations re-run against `e9fcf13`:** `journal()` is `scripts/engine_driver.py:130`;
   the GATE-OPENER section is `scripts/go_explore_solve.py:3241` (`:3225` is `count_wmax`);
   the ortho FORGE entry starts at `CLAIMS.md:199`. `--ortho :8954` and `stall_flat_windows
   :8393` were right for `e9fcf13` and wrong for the working tree (below).
3. **§4 counts match the bullets:** (a) 7 tests + 2 registry entries (a seventh test
   added); (f) 6 tests + 1 registry entry. **4.** §2c no longer claims field identity with
   `Mechanism`. **5.** §3's repair-round count is marked as ours.

**New fact.** The working tree is dirty: 178 insertions in 8 files, +22 lines in
`scripts/go_explore_solve.py` (numbers shift +2 from `:3117`, +22 from `:8057`); the 16:53
not-slow receipt on that tree reads **10 failed, 6065 passed**, all ten in the WIP-modified
`tests/test_go_explore_chain.py` (`receipts/pytest_receipt.log:527-536`). `PROGRESS.md`'s
6062/0 baseline is clean `e9fcf13`. §5 adds step 0 to pin the tree.

## 1. Claim and scope

**Claim.** The Forge is machinery that lets the solver fleet detect, from its own telemetry,
that a wall has stopped moving; assemble a diagnosis bundle from receipts it already holds;
pick a registered search arm by index; propose, pilot, gate and adversarially review a
candidate mechanism inside a bounded, watchdog-gated block; and register what survives into
the arm library with a ledger entry that states plainly whether its validation gate was
met. Every step emits a receipt that can read VOID; every gate has a fixture proving it can say no.

**Ledgers.** The Forge lives in the FORGE ledger, which classifies machinery, not play
(`CLAIMS.md:124-131`). Any clear a forged arm produces is an EXHIBITION result in its own
ledger; FORGE entries carry no clear rate, episode count or protocol number, and the two are
never merged into one sentence (`CLAIMS.md:152-158`; LG finding 7).

**Tier-3 sentence, instantiated per FORGE entry (LG design rule 4).** *Designing this arm
is LLM guidance of exploration in the plain sense; it is permitted under CLAIMS.md's purity
boundary because the design is game-agnostic machinery, its knob settings are derived only
from this run's own telemetry (named in the entry), and it references no route, map,
disassembly, or game-specific instruction — the test "could this decision have been made by
a party who has never seen the game" holds.* (`CLAIMS.md:168-195`.)

**What it is NOT.** Not learned — *learn, learned, learns, learning, self-taught* are
banned on every surface it writes (`CLAIMS.md:160-166`). Not a guarantee of clears: tonight
passes if every piece produced its receipt and every gate was shown able to fail, not if a
wall fell. Not a cause claim: STALLED is a rate claim only (SD design rules 1, 3). Not a
resurrection of `WallClass.GATED` — falsified vocabulary; citing it in a Forge artifact is a
defect (`src/training/wall_taxonomy.py:8-34,444-453`).

## 2. Architecture — the six pieces

New code lives in `src/forge/` (pure functions, importable by tests the way
`tests/test_anti_vacuity_gates.py` imports real gate functions) with one CLI,
`scripts/forge.py {stall,bundle,select,cycle,block} [--dry-run]`. Receipts go to
`runs/forge/<wall_id>/<cycle_id>/`; the engine-facing verdict stream is
`runs/engine/stall_receipts.jsonl`, sibling of `proposed_claims.jsonl`
(`scripts/engine_driver.py:60`). Nothing writes `CLAIMS.md` except the build's own commit
(§2e). Every threshold-shaped gate is registered in a new
`tests/test_forge_gates.py::FORGE_REGISTRY` with the `(verdict key, positive, negative)`
shape of `tests/test_anti_vacuity_gates.py:235-244`; any site the scanner
(`scripts/anti_vacuity_scan.py:15-23`) finds in `src/forge/` is registered there too, or the
drift test fails by name (`tests/test_anti_vacuity_gates.py:247-263`; LG open question 3).

### (a) Watchdog — the STALLED verdict

**Purpose.** Turn the existing, unacted stall counter into a verdict the engine journals.
Two independent kinds, because one clock cannot see both walls (SD finding 8, design rule
2): `kind:"archive"` is in-run and clock-driven and applies only to a *live* child (a
piece-(f) block); `kind:"campaign"` is cross-run and receipt-driven — **both walls are
campaign-kind tonight** (all three cv_hall runs ended at `stall_flat_windows=0`).

**Wall = family, by manifest** `runs/forge/walls/<wall_id>.json`, written by the build.
Membership is scope, like a config; detection stays self-measured — the verdict reads only
what members recorded. No glob heuristics (SD open question 1, decided). cv_hall's manifest:
`prior_best:767`, `prior_best_replay_verified:false` (receipt
`runs/cv_hall_ortho_a/archive.stats.json`), three `shape:"progress"` members
`runs/cv_hall_{ortho_a,ortho_ctrl,true_frontier_1}`. contra's:
```json
{"wall_id": "contra_wall", "prior_best": 3072, "prior_best_replay_verified": true,
 "prior_best_receipt": "docs/research/CONTRA_WALL_2026-08-27.md:16-22",
 "members": [
  {"dir": "runs/contra_wall/A1", "shape": "receipt", "receipt": "boundary_probe.json", "terminal_field": "beat_3072", "best_field": "global_max_gx", "root_family": "solve20"},
  {"dir": "runs/contra_wall/A4", "shape": "receipt", "receipt": "probe_release_report.json", "terminal_field": null, "best_field": "max_gx_observed", "root_family": "solve20"},
  {"dir": "runs/contra_wall/A7", "shape": "receipt", "receipt": "attack_a7_results.json", "terminal_field": "global_beat_3072", "best_field": "global_max_gx_alive", "root_family": "head_wall"},
  {"dir": "runs/contra_wall/A8", "shape": "receipt", "receipt": "A8_result.json", "terminal_field": "breakthrough_found", "best_field": "max_progress_seen", "root_family": "head_wall"}]}
```
A2, A3, A5, A6 follow A1's shape (`beat_3072`); `A7/*_PASS1_CONTAMINATED.json` is not a
member. `root_family` comes from `CONTRA_WALL_2026-08-27.md:205-206` ("five root from
`solve20/archive.pkl`, three from `head_wall_*.state`"); A7's recorded root paths are dead
scratchpad paths, so the family label is the only root identity the receipts can supply.

**What each member shape contributes.** `shape:"progress"`: the last line of
`progress.jsonl` (60 s cadence; `cells, steps, elapsed_s, solutions, max_gx_in_max_area,
stall_flat_windows`, written at `scripts/go_explore_solve.py:8393`); `best_seen` =
`archive.stats.json["best_score"]`, else the tail's `max_gx_in_max_area`; root id = sha256
of `roots.json` bytes (PA §3.3: root states, not directories); `terminal_no_advance` iff
`solutions == 0 and best_seen <= prior_best`. No `breakthrough_found` exists here and none
is invented. `shape:"receipt"`: the named file; `terminal_no_advance` iff the named
`terminal_field` exists and is boolean `False`; `terminal_field: null` (A4 carries only a
prose `verdict: "FALSIFIED…"`) makes the member **UNMEASURED, listed in `missing`** — prose
is not a receipt field (PA §4); `best_seen` = `best_field`; root id = `root_family`. Any
member: `advance` iff `solutions > 0` or `best_seen > prior_best`.

**Verdict rule** (`MIN_TERMINAL=3`). `terminal_runs` = members with `terminal_no_advance`;
`advances` = members with `advance`. ADVANCING iff `advances > 0`; STALLED iff `advances ==
0 and terminal_runs >= MIN_TERMINAL`; WATCHING iff `advances == 0 and 0 < terminal_runs <
MIN_TERMINAL`; UNMEASURED iff no member parsed. `degraded:true` whenever
`prior_best_replay_verified` is false or any member is UNMEASURED.

**`MIN_TERMINAL` re-derived from disk.** cv_hall: 3 progress-shaped members, identical
`roots.json` (`runs/cv_chain_hw2/entrances/entrance_after_2.state`, so `distinct_roots:1`),
`best_score` 767/767/767, `solutions` 0/0/0, steps 8,972,160 / 9,135,920 / 5,711,310 →
`terminal_runs:3, advances:0`. contra: 7 falsy-boolean members (A1-A3, A5-A8) + A4
unmeasured → `terminal_runs:7, distinct_roots:2`. Three is the largest value both walls
clear; cv_hall clears it exactly, so the boundary test sits at 3. CLAIMS.md's "five arms,
~110M steps" (`:247-248`) counts runs not on disk under `runs/cv_hall_*`; the verdict says so.

**Output** — the cv_hall verdict as it must read tonight:
```json
{"verdict": "STALLED", "kind": "campaign", "wall_id": "cv_hall",
 "source": "runs/forge/walls/cv_hall.json", "measure": "advances==0 and terminal_runs>=MIN_TERMINAL",
 "evidence": {"terminal_runs": 3, "advances": 0, "distinct_roots": 1, "prior_best": 767,
              "best_seen": [767, 767, 767], "solutions": [0, 0, 0],
              "steps": [8972160, 9135920, 5711310], "members_unmeasured": []},
 "threshold": {"MIN_TERMINAL": 3}, "degraded": true,
 "missing": ["replay_verified_frontier: prior_best 767 is the phantom pin; true frontier 751 (PA §3.2)"],
 "t": "2026-09-01T…"}
```
`kind:"archive"` on a live child's tail: STALLED iff `stall_flat_windows >=
FROZEN_WINDOWS_MAX (12)` and `steps >= EFFORT_MIN_STEPS (250_000)`
(`src/training/wall_taxonomy.py:159,149`); WATCHING at `>= 2` (`scripts/go_explore_solve.py:8378`);
ADVANCING otherwise; UNMEASURED on an empty tail. `evidence` holds raw numbers, never a derived ratio (SD rule 7).

**Files.** New `src/forge/stall.py` (`archive_verdict`, `campaign_verdict`, pure;
`archive_verdict` replays the real `update_stall()`, `scripts/go_explore_solve.py:3115`).
Wiring in `scripts/engine_driver.py`: a sibling check `stall_check(state, repo)` called once
`plan()` (`:702`) has chosen a wall-bound action — not a `guard_reasons` member (`:1288`),
because a stalled wall routes that wall's next action, it does not halt the engine (SD rule
5). Latched per wall in `state["stalled_notified"][wall_id]` as `state["blocked_notified"]`
latches (`:1323-1329`); journaled via `journal()` (`:130`) before anything acts on it;
appended to `runs/engine/stall_receipts.jsonl`. Default-off behind `--forge`: flag absent, journal byte-identical to today.

**Tests** (`tests/test_forge_stall.py`, 7 + 2 registry entries, each revert-verified):
- `test_archive_verdict_stalled_at_threshold` — 12 flat rows, steps ≥ 250k → STALLED.
  Corrupt `>=` to `>` → fails at exactly 12.
- `test_archive_verdict_advancing_on_real_cv_hall_tail` — fixture copy of the last 5
  `cv_hall_ortho_ctrl` rows → ADVANCING. Corrupt: hardcode STALLED.
- `test_archive_verdict_unmeasured_on_empty_tail` — zero rows → UNMEASURED, never
  ADVANCING. Corrupt: default the empty case to ADVANCING.
- `test_campaign_stalled_on_progress_shaped_family` — three fixture dirs, identical
  `roots.json`, 767/767/767, 0 solutions → STALLED, `terminal_runs:3, distinct_roots:1`.
  Corrupt: count directories, not root hashes → 3 → fails.
- `test_campaign_stalled_on_receipt_family_with_prose_only_member` — three falsy-boolean
  receipts + one prose-`verdict` receipt → STALLED, `terminal_runs:3,
  members_unmeasured:["A4"]`. Corrupt: parse the prose as falsy → `terminal_runs:4`,
  `missing` empty → fails.
- `test_campaign_advancing_when_one_member_beats_prior` — one member `best_seen:800` →
  ADVANCING. Corrupt: ignore `best_seen` → fails.
- `test_engine_journal_byte_identical_without_forge_flag` — `tick(dry=True)` journal bytes
  equal with the module present and `--forge` absent. Corrupt: emit the stall row
  unconditionally → fails.
- FORGE_REGISTRY entries for `archive_verdict` and `campaign_verdict` (LG rule 8).

### (b) Diagnosis bundle

**Purpose.** Everything a designer may see about a wall, and nothing else
(`CLAIMS.md:168-195`: telemetry in). No game content, captions, or cell renderings (LE
rules 4, 16). **Input:** a STALLED verdict plus the manifest members' receipts. **Output:**
```json
{"wall_id": "…", "verdict": {…piece (a) object…},
 "frontier_shape": {"certainty": "confirmed_by_receipt|candidate|not_probed", "data": {…raw boundary_axis_profile() output…}},
 "cell_rate_history": {"certainty": "…", "data": [{"elapsed_s": 0, "cells": 0, "stall_flat_windows": 0}, …]},
 "ram_observables": {"certainty": "not_probed", "data": null},
 "mechanism_class": [{"class": "KEY_BLIND|SCRIPTED_RELEASE|OBSERVABLE_DEFECT|UNKNOWN",
   "certainty": "…", "receipt": "docs/proposals/gate_opener_arm_2026-08-11.md:139-215"}],
 "arms_tried": [{"arm": "ortho", "knobs": {"mode": "up"}, "verdict": "FIRED",
   "outcome": "MECHANISM_VALIDATED/PREMISE_STALE", "receipt": "CLAIMS.md:223-242"}],
 "missing": ["replay_verified_frontier", "fresh discover_observables run on frontier band"]}
```
`mechanism_class` is a list (contra needs two at once, SD finding 13) from a controlled
vocabulary sourced only from receipts on disk before the cycle starts (SD design rule 6;
open question 4 decided). `cell_rate_history` is the raw trailing series of the newest
progress-shaped member, bounded to the last `WINDOW_RECORDS*6` rows (LE rule 5); empty for
a receipt-only wall, stated in `missing`, which copies `MISSING_TELEMETRY`'s style
(`src/training/wall_taxonomy.py:394`): name the gap, never omit it.

**Files.** New `src/forge/bundle.py::build_bundle(wall_id, verdict)`. Reads
`boundary_axis_profile()` (`src/training/wall_taxonomy.py:962`) against the newest member's
`archive.pkl` when present, the manifest's receipts when not (SD design rule 8).
`discover_observables.py` (`:812,841,940,1094,1556`) is referenced as `ram_observables`, not run tonight — `not_probed`, stated.

**Tests** (`tests/test_forge_bundle.py`, 4):
`test_bundle_cv_hall_reports_key_blind_from_receipt` — fixture archive with six constant
axes and one live 1-bit axis → `KEY_BLIND`, `confirmed_by_receipt`; corrupt: drop the
constant-axis count → `UNKNOWN`. `test_bundle_contra_from_receipts_only_yields_two_classes`
— fixture `A8_result.json` + `A6_RECEIPT.json` → `[SCRIPTED_RELEASE candidate,
OBSERVABLE_DEFECT confirmed]`, `ram_observables.certainty == "not_probed"`; corrupt: default
`ram_observables` to `confirmed_by_receipt`. `test_bundle_never_contains_gated_vocabulary`
— no `GATED`/`saturated` in the serialized bundle; corrupt: add the word to a remedy string.
`test_bundle_string_fields_are_controlled_vocabulary_or_paths` — every string leaf is an
enum member, a repo path, or a `path:line` receipt; corrupt: add a free-text `notes` field.

### (c) Mechanism registry and arm selection

**Purpose.** The library of arms a cycle may select or extend, and an index-only selector
(LE rules 1-3: an index cannot name an arm that does not exist). **Registry entry**
(`src/forge/registry.py::ARMS`, a tuple of a new `Arm` dataclass modeled on the
armed-signal + activity-counter pattern of `Mechanism`,
`scripts/check_mechanism_receipt.py:95-118` — same `activity_kind` vocabulary, new field
names, not the same class):
```json
{"name": "ortho", "kind": "arm", "flag": "--ortho", "off_value": "off",
 "knobs": {"--ortho-pin-secs": 120.0, "--ortho-bias": 0.30, "--ortho-band": 1, "--ortho-weight": 4.0, "--ortho-macro-p": 0.0},
 "armed_signal": "ortho_armed()", "activity_counter": "ortho_selections", "activity_kind": "cumulative",
 "gate_fn": "ortho_cols_improved>=8",
 "inertness_test": "tests/test_go_explore_solve.py::test_the_ortho_arm_stays_inert_at_the_shipped_cli_defaults",
 "forge_entry": "CLAIMS.md:199", "status": "FORGE-PENDING-VALIDATION", "wall_classes": ["KEY_BLIND"], "history": []}
```
Seeded with `ortho` (flag `scripts/go_explore_solve.py:8954`; `ortho_pool :3141`,
`ortho_armed :3150`; counters `ortho_selections`, `ortho_cols_improved` as written in
`runs/cv_hall_ortho_a/progress.jsonl`), `lock` (flag `:9012`; `lock_armed :3167`,
`lock_clocks :3176`, `in_lock_key :3194`; inertness guard
`tests/test_go_explore_solve.py:3561`; counter name read from `progress_line()` at build
time — none found means UNAUDITABLE and unarmable, logged, not patched), and `gate_opener`
(flag `:9084`, shipped choices `{off, enumerate}` — PA §1.3's `{probe,search,off}` is the
proposal's CLI, not what shipped; section `:3241`, `gate_armed :3269`; counters
`gate_armed`/`gate_armed_secs`; status `K0-HALTED`, pickable per PA open question 1: option
B, sound-but-uncertified). No `activity_counter`, no arming (PA §4; `check_mechanism_receipt.py:26-40`).

**Selection.** `select_arm(bundle, ARMS) -> {"index": i, "knobs": {…}, "redundant_with":
[...], "reason_codes": [...]}`. Three bounded calls (LE rule 1): most promising index, knob
values expressible from `bundle` fields only, novelty against `arms_tried` on knob-space
distance (LE rule 14). Tonight: a deterministic table lookup on `mechanism_class`; the agentic version is a two-way door (§6).

**Tests** (`tests/test_forge_registry.py`, 4):
`test_every_registered_arm_has_activity_counter_or_is_unarmable` — corrupt: register
`activity_counter: None` with `status: ARMED`. `test_select_returns_index_in_range_only` —
corrupt: return a name. `test_select_marks_tried_arm_redundant` — `arms_tried` has `ortho
up`, proposal `ortho up` → `redundant_with:["ortho"]`; corrupt: ignore history.
`test_select_on_key_blind_never_picks_spatial_only_arm` (PA §3.1) — corrupt: drop the class
filter.

### (d) Forge loop — §3. Files `src/forge/loop.py` (`forge_cycle`, `forge_gate`), `src/forge/proposal.py`; tests in §3.

### (e) FORGE ledger writer

> **Vocabulary constraint (added 2026-09-01 from the no-ai-slop pass over the repo docs):** the entry template must not emit interpretive metadiscourse. (Corrected 2026-09-01 19:45: the `status_plain` value template below previously read "Status at forging, stated plainly: …", which this constraint bans; the key stays, the rendered line now reads "Status at forging: gate met|not met, <what the gate measured>". Piece (e) renders the corrected form.) In particular no field or phrase of the form "stated plainly", "the honest reading is", "to be clear", no whole-sentence bold, no caps-label lines, no aphorism kicker; the entry states the gate outcome and the receipts and stops. The `check_vocabulary` gate in §2(e) covers these patterns as well as the banned verbs.

**Purpose.** Render every ledger artifact the Forge emits in the shape of the one existing
entry (`CLAIMS.md:199-262`; LG finding 5) and refuse anything that breaks vocabulary.
**Entry** (rendered Markdown; fields in this order, LG rule 3):
```json
{"status": "FORGE-PENDING-VALIDATION|FORGE-VALIDATED-MECHANISM|FORGE-VOID|FORGE-GRANT",
 "arm": "name", "flag": "--x", "commit": "sha", "date": "2026-09-02",
 "detection": "telemetry read + what was ruled out", "mechanism": "full knob surface + composition",
 "review": "specific findings fixed", "gate": "test count + inertness-mirror test name",
 "tier3_sentence": "…§1 verbatim, telemetry named…",
 "status_plain": "Status at forging: gate met|not met, <what the gate measured>",
 "citable_as": "…no clear may be attributed to it…", "addenda": []}
```
**Output paths.** `runs/forge/<wall>/<cycle>/CLAIMS_ENTRY.md` plus a row in
`runs/engine/proposed_claims.jsonl`. Landing into `CLAIMS.md` is a commit under the owner's
name (`CLAIMS.md:140-144`); tonight the build lands exactly two FORGE-GRANT entries (§4),
required by the ruling before any block runs (`PROGRESS.md`); `CLAIMS.md` has none today (0
hits in 4,544 lines). Corrections are appended addenda, WITHDRAWN/STANDS (LG rule 6; PA §4).

**Files.** `src/forge/ledger.py::render_entry`, `::check_vocabulary(text)`
(case-insensitive `\b(learn|learned|learns|learning|self-taught)\b`, plus a refusal of any
digit-bearing "clear rate / episodes / protocol" phrase inside a FORGE entry).

**Tests** (`tests/test_forge_ledger.py`, 5): `test_render_has_all_eight_sections_in_order`
— corrupt: swap `review` and `gate`. `test_banned_verb_refused` — `"the arm learned"` →
refused; corrupt: drop `learned` from the regex. `test_clear_rate_refused_in_forge_entry` —
`"3/10 clears"` → refused; corrupt: allow digits before "clear".
`test_tier3_sentence_present_and_names_telemetry` — corrupt: empty `telemetry` list.
`test_addendum_appends_never_rewrites` — original bytes unchanged after an addendum;
corrupt: rewrite in place.

### (f) Block runner

**Purpose.** The only way a Forge pilot touches the emulator: a bounded, watchdog-gated
unattended block with hard abort, per the phase-3 ruling (`PROGRESS.md`). Piece (d) calls
(f); (f) never calls (d). **Input, a block plan:** `{wall_id, cycle_id, cmd,
root_state_sha, max_secs (1200), max_steps (2_000_000), grant_entry
("CLAIMS.md#FORGE-GRANT-…"), attended_log ("runs/forge/attended.jsonl"),
inject_wrongful_reset (false)}`. The child launches as the engine launches actions (`scripts/engine_driver.py:1073-1081`, via `scripts/detach.py`).

**Wrongful reset, defined.** Between consecutive progress rows or lock reads: `cells[t] <
cells[t-1]`, or the child's root-state hash differs from `root_state_sha`, or `read_lock()`
(`src/utils/run_lock.py:93`) reports a holder pid other than the child's, or `solutions`
decreases. Any one trips the watchdog: SIGTERM the child, `release()` the lock (`:150`), bank nothing.

**Block receipt** (fields fixed by the ruling; LG rule 9):
```json
{"wall_id": "…", "cycle_id": "…", "grant_entry": "CLAIMS.md#FORGE-GRANT-cv_hall-2026-09-01",
 "started": "…", "ended": "…", "stop": "budget|stalled|abort|complete",
 "attended_hours": 0.5, "run_lock_hours": 0.33, "ratio_machine_per_attended": 0.67, "ratio_ok": false,
 "watchdog_trips": 0, "positive_control": {"injected": true, "caught": true, "banked_from_reset": 0},
 "fabricated_clears_unretracted": 0, "banked": ["runs/forge/…/sandbox/receipt.json"],
 "aborted": false, "abort_reason": null}
```
LG open questions 1-2 decided: `watchdog_trips` must be 0 to bank (an archive-kind STALLED
stop from piece (a) is `stop:"stalled"`, clean, not a trip); the ≥6 machine-h per attended-h
ratio is reported with `ratio_ok`, not refused — the grant is judged on the cycle receipt,
not the block. No `grant_entry` anchor in `CLAIMS.md`, no start. The moment a wrongful reset
banks an artifact, the runner writes `GRANT_ENDED` to `runs/forge/grant_state.json` and refuses every later block.

**Addendum, 2026-09-02 (FORGE-FIX-2), superseding the `positive_control` literal above
and the first test paragraph below.** `positive_control` carries six keys, not three:
`injected` (the plan's request flag, meaning unchanged), `injected_done` (the injection
actually wrote bytes), `injected_sha` (the fingerprint the injection left), `root_sha_at_trip`
(the fingerprint the watchdog read at the trip), `caught`, and `banked_from_reset`. `caught`
is no longer a copy of `injected`. `src/forge/block.py::positive_control_caught` computes it,
and it reads true only when the injection fired, the trip reason was `root_state_mismatch`,
and `root_sha_at_trip == injected_sha`. The three added keys let a reviewer re-derive `caught`
from the receipt alone. `test_positive_control_injected_reset_is_caught` no longer proves what
the paragraph below says it proves: its original half runs a child with no `--root-state`, so
the injection cannot fire, and it now asserts `injected_done:false, caught:false`; a second
half with a correct `--root-state` asserts the gate saying yes. Two tests are added beside it,
`test_uninjected_trip_reports_no_positive_control` and
`test_positive_control_caught_requires_the_injections_own_sha`, so the test count in the
heading below is superseded too.

**Files.** `src/forge/block.py` (`run_block`, `wrongful_reset`), extending
`src/utils/run_lock.py` (`acquire :106`, `release :150`). Modeled on
`tests/test_terminal_stasis.py` and `tests/test_stability_audit_guards.py` (LG finding 11).

**Tests** (`tests/test_forge_block.py`, 6 + 1 registry entry):
`test_positive_control_injected_reset_is_caught` — `inject_wrongful_reset` makes a
synthetic child emit cells 100→150→40; receipt reads `caught:true, banked:[]`; corrupt:
compare `<=` instead of `<` (a flat read trips first and masks the decrease) → fails on
`banked_from_reset`; corrupt 2: remove the SIGTERM → fails on `ended`.
`test_clean_block_banks_with_zero_trips` — monotone child → `banked` non-empty,
`watchdog_trips:0`; corrupt: trip on flat. `test_budget_stop_at_max_secs_and_max_steps` —
corrupt: drop either bound. `test_refuses_without_grant_anchor` — corrupt: skip the anchor
grep. `test_grant_ended_refuses_all_later_blocks` — corrupt: ignore `grant_state.json`.
`test_ratio_reported_not_refused` — 0.5 attended vs 0.33 machine → `ratio_ok:false`, block
still ran; corrupt: refuse on ratio. FORGE_REGISTRY entry for `wrongful_reset` (positive:
decreasing cells; negative: monotone).

## 3. The forge loop (piece d)

**Roles.** A **designer** agent produces proposals from the bundle, the registry entry it
was pointed at, and an abstracted arm interface (signatures of `ortho_pool`/`ortho_armed`,
not solver source — LE open question 1 decided, so `classify_transition`'s game branches at
`scripts/go_explore_solve.py:878` never enter context). A **refuter** agent, a separate
call, tries to break each proposal on four counts: purity (any route, map, address, or
game-name-specific instruction), redundancy (knob-space distance to `arms_tried`), vacuity
(a gate clause that cannot fail), budget (cannot arm within `max_secs`). The **gate** is
deterministic code (`forge_gate`), never a model; it decides.

**Proposal** (`runs/forge/<wall>/<cycle>/proposal_<k>.md`, house format of
`docs/proposals/gate_opener_arm_2026-08-11.md` §1-7; PS rule 2), frozen (sha256 recorded)
before the pilot:
```json
{"kind": "knob|arm", "arm_index": 0, "knobs": {"--ortho-pin-secs": 60.0},
 "wall_class_addressed": "KEY_BLIND", "why_tried_arms_dont_fit": "…",
 "patch": null, "tests_added": [], "inertness_mirror": "test name",
 "gate": {"stage1": "activity_counter FIRED, n_obs>0",
          "stage2": {"metric": "cols_improved", "threshold": 8, "vs": "matched_control"},
          "stage3": ["G1a frontier>prior_best replay-verified", "G1b solved exit", "G1c transition"]},
 "kill": ["throughput cost >40%", "archive >3x cells with no frontier motion", "counter INERT"],
 "budget": {"max_secs": 1200, "max_steps": 2000000, "seed": 1, "control": "same seed, flag off"},
 "tier3_sentence": "…"}
```
`kind:"knob"` re-parameterizes a registered arm from telemetry (a run setting, recorded with
the run; not itself a FORGE entry). `kind:"arm"` carries a solver patch plus tests and must
pass default-off byte-identity and the inertness mirror before any pilot
(`CLAIMS.md:145-148`; LG rule 1); schema-supported and fixture-tested tonight, live only if §5 finishes early.

**Sandbox.** `K=3` proposals per cycle (LE rule 9), at most 2 refute/repair rounds each
(the count is ours; LE rule 12 establishes only that the loop is capped, citing Voyager's 4
and OMNI-EPIC's 5), each survivor piloted by one piece-(f) block plus one shared matched
control at identical seed and budget with the flag at its `off_value` (PA §4). Receipts:
the block receipt, the pilot's `progress.jsonl`, and a `check_mechanism_receipt`-style counter reading.

**Gate — three stages, split verdict** (PS rule 4; ortho precedent `CLAIMS.md:223-242`).
1. ARMED: the arm's `activity_counter` read FIRED with `n_obs>0`; INERT or UNAUDITABLE →
VOID, stage 2 never runs. 2. MECHANISM: the pre-registered `stage2.metric` beats the control
by the pre-registered threshold → MECHANISM_VALIDATED, else MECHANISM_FAIL. 3. PREMISE: any
stage-3 clause true on replay → PREMISE_CROSSED, else PREMISE_STALE; reported, never
required for registration. A crossing the control also produces is the taxonomy's own
falsifier, reported as such, not as a pass (PA §4). Anti-vacuity: `forge_gate` is in
FORGE_REGISTRY with a positive fixture (FIRED + metric over threshold → VALIDATED) and a
negative fixture (a null candidate whose knob equals `off_value`, so the counter reads INERT
→ VOID). The null candidate also runs live once per cycle as the in-situ negative control; a cycle in which it does not read VOID is itself VOID.

**Registration and arming.** A MECHANISM_VALIDATED `arm` proposal appends an `ARMS` entry
(§2c) and a FORGE ledger entry via (e) in one commit; a `knob` proposal appends its setting
to the registry entry's `history` and the run's receipt. Discards are logged as `FORGE-VOID`
rows with stage and reason (PS rule 7), never silently retried under a new name. A
registered arm is armed on its wall only by a later block under a dated grant; the FORGE
entry says PENDING-VALIDATION until that block's stage-2 receipt exists.

**Stop conditions.** (1) any proposal reaches MECHANISM_VALIDATED; (2) all proposals
VOID/FAIL after their repair rounds; (3) any block reports `watchdog_trips>0` or
`aborted:true` — the cycle is VOID, nothing registers; (4) cycle budget exhausted: `3 pilots
+ 1 control + 1 null` blocks, ≈100 min machine time; (5) `grant_state.json` reads
`GRANT_ENDED`.

**Tests** (`tests/test_forge_loop.py`, 7): `test_gate_void_when_counter_inert` (corrupt:
skip stage 1); `test_gate_validated_only_over_threshold` (corrupt: `>=` to `>` at exactly
8); `test_gate_reports_control_crossing_as_falsifier` (corrupt: credit the candidate);
`test_cycle_void_when_null_candidate_not_void` (corrupt: drop the null run);
`test_cycle_stops_after_two_repair_rounds` (corrupt: loop to 3);
`test_refuter_rejects_address_literal_in_proposal` (fixture containing `0x0450` → refused;
corrupt: drop the regex); `test_arm_proposal_requires_inertness_mirror_before_pilot`
(corrupt: pilot with `patch` set and `tests_added` empty).

## 4. Pre-registered gates for tonight

| piece | PASS | FAIL | VOID |
|---|---|---|---|
| (a) | cv_hall manifest (3 members) → `campaign` STALLED, `terminal_runs:3, advances:0, distinct_roots:1, degraded:true`; contra manifest (8 members) → STALLED, `terminal_runs:7, distinct_roots:2, members_unmeasured:["A4"]`; engine journal byte-identical without `--forge`; 7 tests + 2 registry entries green | Either wall reads ADVANCING or WATCHING with all members readable, or the byte-identity test fails | A member's files unreadable → member in `missing`; if that drops a wall below `MIN_TERMINAL` the row is VOID (reported), not FAIL |
| (b) | cv_hall bundle → `KEY_BLIND confirmed_by_receipt`; contra → two classes; both `ram_observables: not_probed`; 4 tests green | A bundle contains free text, `GATED`, or a class with no receipt path | `archive.pkl` unreadable → `frontier_shape.certainty: not_probed`, bundle still emitted |
| (c) | 3 seeded arms, each auditable or marked UNAUDITABLE; selection on both bundles returns an index with `redundant_with` populated for cv_hall (`ortho up` tried); 4 tests green | Selector returns a spatial-only arm for `KEY_BLIND`, or a name | — |
| (d) | One full cycle per wall inside grant blocks; null candidate reads VOID; every proposal frozen before its pilot; 7 tests green | Null candidate reads anything but VOID; a proposal edited after pilot start (hash mismatch) | Cycle aborted by (f); or budget exhausted before the null control ran |
| (e) | Two FORGE-GRANT entries landed; vocabulary check refuses the two fixture violations; 5 tests green | Any Forge-written text contains a banned verb or a clear rate | — |
| (f) | Positive control caught on the synthetic child AND once on a real solver child (`inject_wrongful_reset:true` on a 3-min block, `banked:[]`); 6 tests + 1 registry entry green | Injected reset not caught, or anything banked from it — the grant ends before it starts | Real solver child fails to launch (environment), synthetic control still passes: (f) PASS-SYNTHETIC only, live blocks do not run |

**Live-validation protocol (two walls under the ruling).**
1. No block runs before (f) PASS on the real child. Test gate (§5) green first.
2. Build lands `FORGE-GRANT-cv_hall-2026-09-01` and `FORGE-GRANT-contra_wall-2026-09-01`
   in `CLAIMS.md`: manifest path, root-state sha, block bounds (1200 s / 2.0M steps), max
   blocks (5 per wall), the receipt fields, and "one wrongful reset that banks an artifact
   ends the grant."
3. Per wall, in order: (a) verdict → (b) bundle → (c) selection → (d) cycle, each block
   through (f). cv_hall pilots `kind:"knob"` on `ortho` (pin-secs/bias/band from
   `cell_rate_history`) from the shared root `entrance_after_2.state`; contra pilots
   `kind:"knob"` on `lock` from a `_LANDING` root — if its bundle carries `OBSERVABLE_DEFECT
   confirmed`, the cycle is expected to read VOID at stage 1 or 2 and says so (PA §2).
4. Receipts as ruled, per cycle: attended hours beside run-lock hours with the ≥6:1 check,
   `watchdog_trips` sum 0, the positive control, zero unretracted fabricated clears, the
   grant anchor — `runs/forge/<wall>/<cycle>/cycle_receipt.json`, copied to `receipts/`.
5. Any clear during a pilot → EXHIBITION entry in the run's solutions ledger citing the
   flag and commit; the FORGE entry says only whether its gate was met.

## 5. Build order and effort (Phase 2, one live writer)

| # | piece | commit subject | est. |
|---|---|---|---|
| 0 | tree pin | **No stash (denied by policy and this repo's incident class).** The "dirty tree" seen during drafting was the run-lock prerequisite (DO-5) being built; it committed as `fa289d4` with its gate green. Step 0 = wait for DO-5's Pool and Floor commits, verify `git status` clean, run `make test-fast --deselect tests/test_night2_runner.py::test_dry_run_passes_live`, record the count in `PROGRESS.md` as the baseline. Operator-run. | 0.2 h |
| 1 | (a) | `forge: STALLED verdict from wall manifests and the stall watchdog, journaled to the engine` | 2.3 h |
| 2 | (b) | `forge: diagnosis bundle for progress-shaped and receipt-shaped walls` | 2.5 h |
| 3 | (c) | `forge: mechanism registry and index-only arm selection` | 1.5 h |
| 4 | (e) | `forge: FORGE ledger writer with vocabulary and no-clear-rate checks` | 1.0 h |
| 5 | (f) | `forge: bounded block runner with wrongful-reset watchdog and positive control` | 2.0 h |
| 6 | (d) | `forge: propose, pilot, gate and review loop over the registry` | 3.0 h |
| 7 | spec+grants | `forge: land the specification, two wall manifests and two block-grant entries` | 0.5 h |
| 8 | live | two cycles, ≈100 min machine time each, overlapped with 6 where (f) is done | 2.0 h |

Total ≈15.1 h against the window ending 08:27; steps 0-5 must be green by 02:00 or step 6
ships fixture-tested only and the live protocol runs (a)-(c) plus one (f) positive-control
block per wall. (f) precedes (d) because (d)'s pilots are (f) blocks. Step 0 is Matthew's
call at ratification (committing the WIP is allowed) — but the ten
`tests/test_go_explore_chain.py` failures must then be fixed before step 1; they are code,
not environment state, and are not excluded by name.

**Test gate, every commit.** `make test-fast` (`Makefile:164`): the step-0 count (expected
6062 passed, 0 failed on clean `e9fcf13`, per `PROGRESS.md`) + every test added above, zero
failures, zero new skips. `make test` prerequisites (`Makefile:161`: `rust-check
unsafe-inventory-check clear-lint purity-check mistakes-check`) green. The slow suite may
run; `tests/test_night2_runner.py::test_dry_run_passes_live` is excluded **by name** with
`--deselect`, per `PROGRESS.md` (environment state, DO-19); any other slow failure is a
FAIL. Before each commit, two greps at zero hits: the diff for
the attribution patterns the build rules name (this copy does not repeat them; the reports-directory original does), and `check_vocabulary` over every new `.md` and docstring.

## 6. One-way doors vs two-way doors

**One-way.** (1) The FORGE status vocabulary — `FORGE-PENDING-VALIDATION`,
`FORGE-VALIDATED-MECHANISM`, `FORGE-VOID`, `FORGE-GRANT` — and the verdict words
`STALLED/WATCHING/ADVANCING/UNMEASURED`, `MECHANISM_VALIDATED/FAIL`, `PREMISE_CROSSED/STALE`:
once cited elsewhere they cannot be renamed, only addended. (2) The registry entry schema
(§2c) once a second arm is registered — every later gate and receipt keys on those fields.
(3) The FORGE-GRANT entry: its bounds and receipt fields are the ruling's terms; loosening
them is a new ruling, not an edit. (4) The wrongful-reset definition: an artifact banked
under a weaker definition is fabricated under a stronger one. (5) The wall-manifest schema (§2a) once a verdict citing it lands in `CLAIMS.md`.

**Two-way.** Thresholds (`MIN_TERMINAL=3`, `K=3`, repair rounds 2, pilot 1200 s / 2.0M
steps), manifest *membership* (adding a run is an addendum, not a schema change), the
selector's agentic form, `src/forge/` and `runs/forge/` layout, `ratio_ok` reported-not-refused, the step-0 stash, `instrument`-kind proposals (§7).

## 7. Not decided, and the three sentences most likely wrong

**Not decided here.** Whether an instrument repair (contra's broken hp observable,
`A6_RECEIPT.json`) is a Forge mechanism or out of scope (PA open question 3); whether piece
(b) gets a reusable RAM-probe library or each wall keeps needing an A-series script (SD open
question 3); whether the agentic selector replaces the table lookup; the K0 fork (PA open
question 1 — parked as pickable, uncertified); whether cv_hall's manifest grows to the runs
behind CLAIMS.md's "five arms" once they are located.

**Most likely wrong, with the receipt that settles each.**
1. *"`MIN_TERMINAL=3` with `best_seen <= prior_best` is a stall, not three runs that each
   ran out of clock."* All three cv_hall members hit their deadline still finding cells
   (`stall_flat_windows=0`, SD finding 8); "no run beat 767" is what a longer run could
   falsify. Receipt: the first cv_hall pilot block's tail — `max_gx_in_max_area > 767`
   inside 1200 s means the rule confused budget exhaustion with a wall, and `MIN_TERMINAL`
   must also require a per-member `elapsed_s` floor (two-way door).
2. *"A 1200 s / 2.0M-step pilot can produce a non-VOID stage-1 reading on the Castlevania
   hall."* Ortho took 120 s to pin and 5401 s to bank 37,345 selections (`CLAIMS.md:226`;
   PS open question 4). Receipt: the first cv_hall block's counter reading — FIRED with
   `n_obs>0`, or INERT, in which case the budget doubles once.
3. *"On-disk receipts alone let a bundle assign `mechanism_class` without a fresh probe."*
   Both walls' classes came from bespoke A-series scripts (SD findings 13, 15). Receipt:
   `ram_observables.certainty` on both bundles tonight reads `not_probed`; a refuter
   rejecting a proposal for lacking a probe the bundle could not supply means the class was assigned on thinner evidence than the vocabulary implies, and (b) needs the probe library.
