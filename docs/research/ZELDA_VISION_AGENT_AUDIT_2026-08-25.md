# Zelda vision-agent research report — purity audit + repo reconciliation (2026-08-25)

Status: REVIEW ONLY. Nothing in this document has been implemented. It
adjudicates an externally-authored research report against `CLAIMS.md`
and against what this repo already ships, and it ends with a roadmap that
replaces the source report's. Every repo anchor below was re-verified
against HEAD `01e56f1` (`scripts/go_explore_solve.py` is 7,781 lines at
this commit; earlier proposal docs cite anchors from `07f367d`/`09299fa`
and have drifted).

---

## 1. What the source report is, and how it is being treated

The document under review is a 20-page deep-research report titled
"Deep Research Report: Vision-Only Autonomous Agent Architecture for NES
The Legend of Zelda (1986)," supplied by the owner as a PDF at
`/Users/stits/Downloads/Zelda Vision Agent Research Directives.pdf`. It
was not commissioned through this project's own consultation bridge
(`scripts/consult_deep_research.py`), so it arrives unsolicited and
un-briefed — it has never seen `CLAIMS.md`, does not know the purity
line exists, and was written by a party whose default assumption is that
a RAM map and a corpus of human gameplay video are ordinary engineering
inputs. It is nonetheless treated exactly like every other Deep Research
consultation this project banks (DR v10 through v14, the Gemini survey
reconciled in `EXTERNAL_DR_RECONCILIATION_2026-08-23.md`, the maze
rounds in `MAZE_DOSSIER_V3_2026-07-26.md`): read in full, adjudicated
item by item against the binding rules, reconciled against what already
exists, and translated — never implemented as written. Its structure is
Sections A–G plus a five-item prioritized roadmap and a 37-entry works-
cited list. Seventy-two adjudicable items were extracted; the tally came
out **26 PASS / 27 MIXED / 19 BANNED**.

The bottom line, stated once up front so no downstream reader has to
infer it: **the report's architectural patterns are largely sound and
several are already shipped here, while its specific methodology — the
privileged RAM critic as specified, the eleven-checkpoint evaluation
table, and the entire Section D training-data pipeline — is
categorically incompatible with this project's binding rules.**

---

## 2. The test that was applied

From `CLAIMS.md`, "Where the purity line sits," verbatim:

> The same rule binds the human operator: a run whose flags were chosen
> by reading a map is Tier-3 contaminated regardless of who typed them.
> The test in both directions is *could this decision have been made by
> a party who has never seen the game?* If not, it is banned.

And the "Never" clause it enforces:

> **Never.** Routes, maps, walkthroughs, disassembly, RAM semantics
> beyond the observables the system discovers for itself, hand-authored
> input segments, per-game reward shaping, or any instruction naming
> where to go in a specific game. An agent recalling a walkthrough from
> its priors is exactly the Tier-3 injection the clause bans; the ban
> does not weaken because the recall was automated.

Four distinctions did most of the adjudication work, and every verdict
in §4 can be traced to one of them:

1. **NES hardware fact vs. this game's engine logic.** "The PPU renders
   at most 8 sprites per scanline, so games multiplex OAM" is true of
   every cartridge ever pressed and sits inside the owner's explicit
   fidelity/mapper exemption. "Zelda's NMI handler executes
   `OAMIndexOffset += 4` each frame" is a symbol lifted from a
   reverse-engineered source tree. The report reaches the *identical*
   engineering conclusion (stack k=4 frames) by both routes, in items
   B-3 and F-3. B-3 passes. F-3 does not.
2. **Unlabeled RAM as an observation vs. labeled addresses as
   semantics.** Tier 1 explicitly permits "RAM tile observations" with
   disclosure; Tier 3 bans "RAM edits," not RAM reads. The boundary is
   not reading memory — it is *interpreting* memory with labels a human
   looked up.
3. **Self-generated demonstrations vs. human demonstrations.** Tier 1
   permits "backward-algorithm start scheduling along the agent's OWN
   search demos; demo-anchor/self-imitation losses on those demos."
   Tier 3 bans "hand-driven input segments in any training input."
   Section D of the report sits on the wrong side of the single word
   OWN, and pseudo-labeling does not launder it.
4. **A gameable-resistant milestone concept vs. a specific milestone
   list.** The report's evaluation *methodology* is excellent and
   matches this project's honest-eval doctrine. Its *instantiation* is a
   walkthrough table of contents with bitmask receipts. These separate
   cleanly.

---

## 3. Provenance: two citations produce most of the violations

Nineteen of the twenty-two RAM/bitmask assertions in the report trace to
exactly two sources named in its Section F:

- **[34]** "Complete Disassembly of NES The Legend of Zelda (Aldo
  Núñez)" — `github.com/aldonunez/zelda1-disassembly`,
  `computerarcheology.com/NES/Zelda/`
- **[22]** "DataCrystal Zelda 1 RAM & ROM Mapping Architecture" —
  `datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/RAM_map`

Cite [22] is the footnote on the Section E critic, on the Section F
register table, and on *every one of the eleven* Section G checkpoints.
Tier 3's first named ban is "Game-disassembly knowledge"; the purity
line's "Never" clause names "disassembly" and "RAM semantics beyond the
observables the system discovers for itself." Cutting these two sources
at the root removes most of the BANNED verdicts in §5, and in nearly
every case a legal re-derivation exists using tooling this repo already
ships (§6).

There is a second, independent contamination channel that survives even
if [22] and [34] are deleted: Section D is not RAM-sourced at all. Its
load-bearing asset is 500 hours of human gameplay video, banned by a
different Tier-3 clause. Section D fails on its own.

Third structural observation, and the one worth carrying furthest: **the
report never once derives anything from the agent's own telemetry.**
Every fact it needs, it looks up. That single habit — not any individual
address — is what produced 19 BANNED verdicts, and in several cases the
legal path was *cheaper* and would have yielded a stronger, portable
result. This repo's shipped methodology is the exact inverse, with named
instruments: `scripts/discover_observables.py` (`find_progress`,
`find_room_counter`, `find_y`, `find_hp_lives`, `lives_from_death_drives`),
`scripts/verify_ram_map.py` (three probes grading every claimed mapping
from our own rollouts), `scripts/onboard_game.py`,
`scripts/odometer_cert.py`, `scripts/discover_item_bits.py`. The
fight-gate work rediscovered Punch-Out's `0x0398` blind through that
pipeline (`docs/proposals/FIGHTGATE_MECHANISM_2026-08-25.md`, wired
`2e2696a`) — honest scope from `1b65f63`: **1-for-4**, since Kung Fu,
Ice Climber and Galaga were tried and none validated.

---

## 4. Purity verdict table

One row per adjudicable recommendation. Item IDs are section-local
(A-1 … G-7, R-1 … R-5) and are referenced by the same IDs throughout §5
and §6. "Already-built-as" cites the file or receipt that makes the
recommendation redundant, partial, or new here.

### Section A — benchmark and precedent landscape

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| A-1 Cite VideoGameBench; real-time vs. paused inference modality | PASS | Another benchmark's published scores inject nothing. **Adoption hazard, not a verdict:** VGBench's harness feeds each model a per-game "high-level goal description." Use the numbers, never import the prompt field — a party who has never seen the game cannot author it. | Protocol axis built: `scripts/eval_game.py` runs real-time no-pause with Machado perturbations (`sticky_prob`, `start_jitter`). No language-model call exists in any run path (a grep for the major provider SDK names over `src/`+`scripts/` returns 0 hits). |
| A-2 Vision-only policy scored by an out-of-band evaluator | MIXED | The *pattern* (policy sees pixels, verifier scores out-of-band) is exactly what honest eval wants. The *instantiation* — "memory-scanning evaluator" — presumes address-level semantics for the target title, i.e. a RAM wiki. | Built three times, legally: `scripts/eval_game.py` (predicate outside the observation path, subprocessed by `src/training/cold_probe.py`); `scripts/clear_detect.py` `StreamingConfluenceDetector` (4-signal, 3-of-4); the certified PPU scroll odometer. |
| A-3 Modular harness decoupling perception / memory / reasoning | PASS | Pure architectural decomposition; no game content. | Partial. Decoupling exists as injection points (`go_explore.py`'s injected `cell_fn`; `onboard_game.py`'s `verdict_fn`; the `TileObservation` protocol) — but no symbolic scene-graph layer exists. |
| A-4 DarkAutumn Triforce: PPO + a rewind-and-attribute `StateChangeWrapper` | MIXED | The wrapper is a *rule* pointable at any observable — legal. The project's object model (health/inventory/enemy coords over RAM) is a *lookup table about one game* — banned. Adopting the wrapper is legal only if keyed on observables we discovered; the moment its trigger is a wiki address, the mechanism has become the delivery vehicle for the injection. | **Wrapper already built, with a measured negative.** `scripts/hazard_collect.py` micro-forks (commit one action for one tick, observe ≤ horizon−1, right-censor); 104,640 banked labels; `src/training/hazard_model.py` gated at IPCW C-index ≥ 0.85, **achieved 0.9170**. Negative: `docs/research/PHASE3_HAZARD_VETO_NEGATIVE_2026-08-22.md`. |
| A-5 Learned world model from own frames; autoregressive-drift critique | PASS | Trains on telemetry the system generated; the drift critique is a general property of autoregressive simulators. | Partial and explicitly unwired. `src/models/world_model.py` (DreamerV3 RSSM), `src/training/dreamer.py`; `configs/zelda.yaml` even carries a tuned `dreamer:` block. Project doctrine: STOP dreamer welds. |
| A-6 Deadlock taxonomy; 16 px movement quantum; resource-exhaustion macro deadlock | MIXED | Discrete screen transitions are observable. A 16 px quantum is legal **if measured**, banned if asserted. The bomb/secret-wall example asserts that hidden walls exist, are mandatory, and are consumable-gated — a walkthrough fact. | Partial + one cautionary receipt. `src/training/wall_taxonomy.py` — read its header first: `WallClass.GATED` was **removed**, not re-thresholded, after 22 statistics over 103 archives all straddled. Measurement instrument for the quantum exists (`odo_debug`, `nes_core/src/pool.rs:1709`); the histogram has never been run. |
| A-7 Perceptual drift over a 10-hour horizon | PASS | A question about drift; "inventory state" here means a HUD region readable from pixels. | Partial, with a real number: RG-1a stability re-fingerprinted 20 rooms × 3 cells → **86.6% ordinal agreement (58/67)** against a 95% target / 80% kill line. Root cause was *not* perceptual drift — `exemplar_cell` is a mutable archive key and both runs saturated `cap=1024` (`cap_hits: 87`/`3801`). |
| A-8 Minimal params/throughput for a world model to preserve discrete state | MIXED | Scaling question is generic; "bombable wall flags" names a game-specific affordance and an internal flag semantic. | Machinery partial: `src/training/latent_cells.py` (VQ-VAE over RAM windows, codebook id as cell key, no address privileged) exists but is **not wired** into the solver. The scaling question itself is new. |
| A-9 Claims hygiene on zero-shot VLM marketing numbers | PASS | A claims-hygiene finding about other people's numbers; consonant with our own policy. | Doctrine already: `CLAIMS.md` two ledgers + FORGE; "a deterministic number is never shown without the sticky number beside it." |

### Section B — System 1, real-time visual-motor policy

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| B-1 Compact specialized policy beats a large general model under latency | PASS | Game-agnostic architectural finding. **Flagged ingredient:** the cited system's imitation component, if human-sourced, is Section D's ban; the extracted recommendation does not depend on it. | `src/models/tile_policy.py:46` `TilePolicyNetwork` (~14k params). Our imitation sources are own-clears (`src/training/sil.py`) and own-solver demos (`src/training/demo_bank.py`), allowlist-gated. |
| B-2 ImpalaCNN / residual conv backbone | PASS | A backbone choice; no game content. | Partial. A pixel CNN path exists (`src/models/policy_network.py`, `src/models/rnd.py`, 84×84 in `src/emulation/frame_utils.py` with a NEON fast path). Whether it is residual is **uncertain**; the tile MLP is the default path regardless. |
| B-3 8-sprites-per-scanline; OAM multiplexing; flicker is observable | PASS | Console hardware, sourced to NESdev, true of every cartridge — squarely inside the fidelity exemption. The practical consequence is directly visible in our own captures. | Primitives shipped: `peek_oam` (`nes_core/src/python.rs:766`), `peek_nametables` (:759), `odo_debug` rendered-lines. **Nobody has measured a flicker period from our own frames.** |
| B-4 Mixture-of-Experts with a learned router | MIXED | MoE + learned router passes as a policy class. The violation is the *hand-authored expert decomposition*: no naive party can enumerate combat/traversal/item-use regimes or know there are eight sub-weapons. Because the router is goal-conditioned, the labels are also a channel for telling the policy which regime it is in. | Chassis partial: `src/training/multihead_policy.py` (shared trunk + N heads, `export_head(level)` loads unmodified in `eval_game.py`) and `src/training/composite_policy.py` (`HysteresisSwitch(k=2)`) — but that router is **hand-keyed on RAM semantics** (`label_from_ram`), not learned. Anonymous-expert MoE is net-new. |
| B-5 Fixed frame-stacking, k=4 | PASS | Canonical DQN/Atari preprocessing; the justification (sprite multiplexing makes single frames non-Markovian) is NES-generic. | **Shipped at exactly k=4 in both encoders**: `src/emulation/frame_utils.py:99` `FrameStacker(stack_size=4)` and `:158` `TileFeatureStacker(stack_size=4, feature_dim=175)`. Hazard substrate consumes the same encoding (`OBS_DIM = 712`). |
| B-6 Frame-stack vs. GRU vs. causal-transformer trade-off table | PASS | Pure architecture/latency trade-off; no game content. | **Answered by experiment, which beats a table.** `docs/research/RECURRENT_AB_VERDICT_2026-08-23.md`: treatment best-of-4 **0.06** honest sticky vs. control **0.76** — FAIL. Adjudication nuance: the GRU class showed deterministic ≈ sticky ≈ 0 on every seed, so v25's mechanism claim was **never tested**, not refuted. No transformer arm exists. |
| B-7 "Sub-pixel positioning (`$0070`, `$0084`)" | **BANNED** | Two addresses with asserted semantics, footnoted to DataCrystal. This is "RAM semantics beyond the observables the system discovers for itself," verbatim. | — |
| B-8 Lattice-snapping macro/option actions | MIXED | Macro/option actions that snap to a discrete lattice are permitted machinery. The instantiation fails: the 16 px quantum is asserted not measured, the trigger ("prior to entering doorways") is a game-mechanic fact, and it depends on B-7's addresses. | Built three ways, one killed: `derive_transition_macros` (`scripts/go_explore_solve.py:2693`, synthesized from the profile's declared action space); `scripts/macro_mine.py` (n-grams from our own archives); `src/training/commitment_policy.py` + `smdp_gae.py` — **FAILED its gate**, `docs/research/OPTIONS_NEGATIVE_2026-08-23.md`, 0/100 vs 8/100, k=4 chosen 93.6% of states. |
| B-9 MoE routing jitter; frame-stack vs. hybrid-GRU gradient stability | PASS | Architecture-internal empirical questions answerable on our own runs. | Q2 answered at the bottom-line level (B-6), not at the gradient level. Q1 has instrument analogues (`HysteresisSwitch`, `scripts/policy_divergence_report.py`) but no MoE exists to jitter. |
| B-10 Claim audit: recurrence does not eliminate frame-stacking | PASS | Architecture claim audit; no game content. | First-party and stronger than the cited source: same references as B-6, controlled A/B on our harness with the honest protocol on both arms. |

### Section C — System 2, planner / mapping / goal conditioning

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| C-1 C-Planning: graph search strictly over *visited* states | PASS | The model-citizen item of the entire report. "Search only over states the system itself visited" is the "Telemetry in" clause restated as an algorithm; it has no channel through which content could enter. | Enforced **architecturally, not by convention**: `RoomIndex.adj` only gains an edge from a transition a worker took (`_room_step:4063` → `_room_transit:4185`); `record_edge` **raises `ValueError` on `kind == "warp"`** (:4148 guard). Missing: any learned distance metric — routing weight is a count prior with an articulation term. |
| C-2 SGRL: action-pruning mask driven by an LLM/VLM goal source | MIXED | The mask is game-agnostic machinery. The goal *source* is not: Tier 3 bans "LLM guidance of rewards or exploration inside a run," and a VLM planning a 1986 title has tens of thousands of pages of walkthrough in its priors. **Secondary hazard:** hard masks are themselves a stealthy injection channel — a mask encoding which actions are invalid in a room is per-game shaping wearing a mechanism's clothes. | Mask built with a negative: `src/training/hazard_mask.py` — model-derived (never authored), substrate frozen, non-optional escape hatch (`n_fully_vetoed` counted), `enabled=False` byte-identical. Legal goal source built: `select()` (`go_explore_solve.py:5615`) with deep/ortho/room-router/count/doors arms. |
| C-3 Go-Explore: archive, deterministic return, explore forward | PASS | Sanctioned at Tier 1 with a pre-authored verbatim claim sentence. | Built twice: `src/training/go_explore.py` (trainer archive, injected `cell_fn`, `W_LOCATION_COEF`, `EXPLORE_AFTER_FIRST_CLEAR`) and `scripts/go_explore_solve.py` (solver, 7,781 lines, 5+ arms, lineage guards, resume). Zero marginal value in re-proposing. |
| C-4 VLM semantic planner polling every 2–5 s, emitting NL goals | **BANNED** | An LLM inside the run is the Tier-3 clause without modification, and the illustrative goal ("Navigate to dungeon entrance") is literally "any instruction naming where to go in a specific game." The two-timescale slow-planner/fast-executor *pattern* survives; the planner being a language model does not. | — |
| C-5 Topological graph memory (nodes, edges, edge attributes) | MIXED | The pattern strongly passes and "Visual Hashing Fingerprint" is exactly right. Three things must be stripped: `Screen ID (0x00–0x7F)` (a RAM index that also asserts a 128-screen world), the edge-attribute vocabulary `{Requires_Ladder, Requires_Raft, Bombable_Wall, Burnable_Bush}` (a hand-authored enumeration of the game's lock-and-key rules — this is the walkthrough), and the worked map excerpt with named destinations. | **Already built in the stripped form and validated on Zelda.** `nt_fingerprint:812` (blake2b-64 over a masked 2 KB nametable, mask emitted by `scripts/room_fp_calibrate.py` from variance over our own idle/walk frames), `record_edge:1027` with `EdgeStat`, `RoomIndex:955`. `docs/receipts/room_graph/RG1_zelda_2026-08-25.md`: 633–1024 rooms/run, 547–925 fade edges, router lift **1.33×–1.62×**, zero warp-minted edges, false-merge bbox audit 0/46. |
| C-6 RND detachment: novelty decays; new capability should reopen cells | MIXED | The detachment critique is a real published game-agnostic result. The instantiation names the map size ("128 overworld screens"), the item ("Bombs"), and the affordance ("un-bombed walls") — all three walkthrough content. | Detachment half built: RND exists (`src/models/rnd.py`, `src/models/tile_rnd.py`, opt-in); the **barren arm** retires cells whose bursts come back empty; the **doors arm** (`door_weight`, `_door_scan:5826`). Missing: nothing acts on capability change. `EdgeStat.cap_hist` records the correlation but `RoomIndex.adj` is **not** rekeyed on `(dst, cap_sig)` — deferred to v2 in `ITEM_SEMANTICS_ENGINE_2026-08-25.md` §8. |
| C-7 Go-Explore cell key `<Screen_ID, Tile_X, Tile_Y, Inventory_Bitmask, Key_Count>` | **BANNED** | Every non-abstract component is RAM-map-sourced. The *shape* (position bucket × capability signature) is a PASS pattern we already use; the sourcing of each field is the violation. | — |
| C-8 "Prioritize cells with un-tested destructible tiles; navigate along documented graph edges to target coordinates" | **BANNED** | Fails three independent ways: presumes we know which tiles are destructible; "documented" here means authored, not self-recorded; and "navigate directly to target coordinates" is an instruction naming where to go. | — |
| C-9a Polling frequency vs. frame-drop latency | MIXED | The latency question is legitimate; it inherits C-4's banned premise that the poller is a VLM. Re-asked about any slow-loop mapper it passes. | Measured: RG-1d on a confirmed-idle machine — baseline 1653–1672 sps, ON-arm 1601/1691/1690/1663 sps, all clearing the pre-registered 90% bar. Methodological caution: the *first* RG-1d baseline was invalidated by concurrent-workflow contention (1463 sps). |
| C-9b Contrastive loss preventing goal-representation collapse | PASS | A representation-learning question; no game content. | **Net-new.** No contrastive objective exists anywhere (`grep contrastive` over `src/`+`scripts/` returns only unrelated prose). |
| C-10 Claim audit: RND alone does not solve long-horizon adventure games | PASS | Algorithmic claim audit; consonant with Tier-1 machinery. | Both sides of the comparison exist here, so the claim can be made first-party. Supporting: the Castlevania hall record (five runs, ~10.7 h, ~77 M steps, zero crossings). |

### Section D — training-data pipeline

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| D-1 VPT: train an IDM on labeled human footage, pseudo-label a large corpus, clone | **BANNED** | Head-on collision with "hand-driven input segments in **any training input**." Pseudo-labeling changes how the human actions are *recovered*, not whose actions they are. **Nuance:** the IDM technique is game-agnostic and would pass on our own rollouts — where it is pointless, because our actions are already logged as receipts. Its only use is on footage whose actions we lack, which means someone else's play. | — |
| D-2 Emulator instrumentation (own receipts / third-party movies / raw RAM dumps) | MIXED | (a) own-run frame-synced inputs = telemetry, PASS. (b) third-party `.fm2`/`.bk2` movie files = recorded human or TAS input, BANNED. (c) raw unlabeled dumps = Tier-1 observation with disclosure; the banned part is the *labeling* step, and the report's cited source is a how-to for exactly that. | (a) `scripts/emit_fm2.py` (power-on-anchored FM2 from our own action bitmasks — "the reconstruction is exact, not approximate"); `src/training/tape_replay.py`. (b) `scripts/convert_fm2.py` **ingests** `.fm2`/`.fm3`/`.bk2` — see §8. (c) `get_ram_range` hard-`PyValueError`s above `0x0800` (`nes_core/src/python.rs:715`), making Metroid's quarantined cartridge-RAM addresses structurally unreachable. |
| D-3 Three-phase VPT pipeline: 50 h labeled + ~500 h YouTube/Twitch, RWBC, privileged RAM critic | **BANNED** | Phase 2 is the Tier-3 clause instantiated at scale. Every one of those 500 hours was played by someone who knew the map, the item order, and where the secrets were; the resulting policy's competence is downstream of that knowledge in a way no evaluation protocol can disentangle. | — |
| D-4 Parse TAS movies for "ground-truth topological maps, hidden staircase coordinates, optimal boss engagement distances" | **BANNED** | The single most flagrant item in the report. It declines TAS for imitation on sound grounds, then proposes it for the one purpose that is maximally banned. Worse than the use it rejects: imitating TAS inputs is a fragile injection that fails at deployment; extracting the map is a durable one that shapes every downstream decision and never shows up in an auditable rollout. | — |
| D-5 "2D IDM exceeds 94% action-prediction accuracy" | **BANNED** | Inherits the banned root; the claim exists solely to argue the 500-hour human corpus will label cleanly. The isolated observation (fixed-camera 2D is easier for inverse dynamics than free-camera 3D) is unobjectionable and has no legal use here. | — |
| D-6 Reward-weighted BC with a privileged-critic advantage `A^RAM` | MIXED | Advantage-weighted BC is standard machinery and is Tier-1 sanctioned **on the agent's own demos**. Here the demonstrations are human, converting the same loss into a human-imitation objective; and `A^RAM` inherits the labeled-address problem, putting a wiki inside a reward channel. | Legal form shipped and already evaluated: `src/training/sil.py`, `src/training/demo_bank.py` ("the bank never receives policy rollouts"), `ppo.py`'s `demo_anchor_loss`, `scripts/bc_pretrain.py`, `bc_distill.py`, `distill_level.py`, `distill_recovery.py`. |
| D-7 Privileged DAgger during offline BC | **BANNED** | DAgger relabels the states the learner visits; with a human-derived expert, every relabel is a fresh injection on exactly the states the learner would otherwise have had to solve. The pattern survives with a different expert — DAgger against our own Go-Explore solver is Tier-1 legal and is the standing research-grounded pivot. | — |
| D-8 "Minimum volume of pseudo-labeled human video for low-frequency item deployment (placing bombs at specific wall tiles)" | **BANNED** | Banned premise plus a banned example; calling the tiles "specific" concedes the question presumes a known set of them. | — |
| D-9 Does BC/DAgger pre-training collapse when transitioning to model-free RL? | MIXED | Stripped of the human-demo premise, a genuine question — **already answered here, twice, both negatives.** | `DOSSIER_V3_2026-07-23.md`: clone accuracy 1.0 → sticky **0.00**. `CLAIMS.md` smodice entry: 40,785-transition terminal-grounded dataset, seven-arm ablation, argmax 0.308 → **0.669** past the 0.60 gate, then **0.0 clears** on 50 honest episodes, median death at x=646 of 3,266. |
| D-10 Claim audit: TAS streams are not imitation data | PASS | Correct and game-agnostic; independently reinforces "presenting deterministic replay as learning" as a Tier-3 ban. | Doctrine + enforced boundary: the two-ledger split, with EXHIBITION explicitly covering routed replay chains and single-trajectory BC clones. |

### Section E — reward modeling / asymmetric actor-critic

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| E-1 Asymmetric actor-critic: privileged critic in training, discarded at deployment | PASS | Well-established and game-agnostic. The asymmetry itself is not the violation; whether an instantiation passes depends entirely on where the privileged vector's semantics come from. | **Net-new as an architecture.** Every critic in the repo sees exactly the actor's observation. Nearest kin: `src/training/hazard_model.py` is a value-style model on interventional data the actor never sees, discarded at deployment — asymmetry in the information-source sense. |
| E-2 PPO overfits background textures; decouple representation from reward computation | PASS | A generalization finding from a multi-game benchmark. | Partial. The decoupling is real (reward in `nes_core/src/rewards.rs`, observation built separately). The texture-level generalization study is net-new; `scripts/interference_falsifier.py` and `src/training/plr.py` are the nearest analogues. |
| E-3 Critic receives the 2 KB RAM vector — justified by naming `$066F`, `$00EB`, `$065C–$0671` | MIXED | **The most consequential adjudication in the report, so the split is stated precisely.** *Passes:* feeding the critic the entire undifferentiated 2 KB with no address interpreted, singled out, weighted, normalized, or named. No semantics are asserted; the sentence is authorable by a party who has never seen any game; the critic is discarded, so the honest sticky number still measures a vision-only actor from power-on. *Does not pass:* the justification as written and the instrumentation it implies — the moment three addresses are named, the design has consulted a RAM map. **Operational rule: grep the critic path for hex literals. If any appear, it has crossed the line.** *Non-purity hazard:* a critic confident about outcomes the actor cannot predict inflates advantage variance rather than reducing it — structurally the same failure that produced the 77.4%-veto result. | Legal variant is net-new **and cleaner than what we run today**: the hazard substrate's input is `OBS_DIM = 712`, built by `src/emulation/tile_observations/smb.py`, whose own docstring says the addresses "are from the standard NESdev SMB disassembly and match SethBling's MarI/O Lua script." Tier-1 sanctioned *with disclosure*, but emphatically not *discovered*. |
| E-4 Multi-tiered reward hierarchy (combat / traversal / milestone) | **BANNED** | Per-game reward shaping in its purest form, banned by name — and barred a second, independent way: Tier 2 freezes the five legacy `LEVEL_*` ladders and states "no new hand ladders will ever be authored." Half-hearts, shield blocks, hidden staircases, the identity of the major items, the Triforce as progression spine, the existence of bosses — none is derivable by a party with no prior. "+10.0 for uncovering a hidden staircase" pays the agent for executing a walkthrough step, making any later "the agent found the secret" claim circular. | Survives: the three-tier *shape*; a generic revisit/anti-dither penalty; a coverage bonus over self-discovered cells (the direct analogue of our every-256px ladder). **See §8 — a version of this hierarchy is already live in this repo.** |
| E-5 Domain randomization policy ("the overworld is static, so overfitting to it is desirable") | **BANNED** | Three counts, one easy to miss: (a) "the layout is static, therefore overfit" is a decision made by reading a map; (b) it contradicts honest-eval doctrine outright — sticky 0.25 and start-jitter exist to punish memorized layouts, and deliberately maximizing memorization then reporting the number is the mechanism behind "presenting deterministic replay as learning"; (c) **randomizing enemy spawn positions cannot be done from outside the emulator** — it requires writing game memory, and "RAM edits" is banned with no disclosure tier available. | Survives and is already mandated: frame-delay/action-repeat randomization is Machado sticky-actions, run at 0.25. |
| E-6 Does privileged critic access induce overfitting that fails under rendering artifacts? | PASS | Generic, well-posed, and the correct falsifier to attach to E-3's legal variant. | Perturbation harness exists (`scripts/eval_regime_stats.py`, `docs/receipts/eval_rng_regimes_2026-08-15.md`); no privileged critic exists to test. |
| E-7 Weighting to prevent safe farming over risky progression | MIXED | Reward-hacking avoidance is real and generic; the question presupposes E-4's banned tiers. Re-asked over the generic distance ladder plus time/death penalties it is legal and live. | Institutional scar tissue: `GenericReward` (`nes_core/src/rewards.rs:159`) freezes its score-candidate set after a 300-step warmup — a farming guard inside the mechanism; `configs/zelda.yaml`'s own comment history documents a motion-reward exploit and an up+A sword-spam collapse; the PPO structural fix (`1c7ef1f`); die-respawn eval inflation; `OPTIONS_NEGATIVE_2026-08-23.md`. |
| E-8 "Labeled RAM critics dominate visual reward parsers" | MIXED | As a narrow latency/variance claim, correct. As deployed — arguing labeled RAM strictly dominates self-discovered observables — **the preference runs the other way here.** A HUD region the system locates itself is the legal path; a labeled address is the banned one. "Noisier" is a cost we pay on purpose. | The legal path is proven: the PPU scroll odometer is FORGE-CERTIFIED **5/5** on `scripts/odometer_cert.py`, with Rygar/Ninja Gaiden SIGNAL-SOUND, Contra cross-validated 162 vs 163, Kung Fu reclassified as a skill wall not an instrument fault. Blind rediscovery is a shipped pattern (fight-gate `0x0398`), honest scope 1-for-4. |

### Section F — NES hardware constraints (titled "hardware," mostly game-specific)

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| F-1 Cite the Aldo Núñez complete disassembly | **BANNED** | Tier 3's list opens with "Game-disassembly knowledge," and the report's own description ("exact game logic routines… RAM address allocations") is a description of the banned category. **Exemption boundary, stated so it is not over-read:** using a disassembly to diagnose an *emulator* bug is explicitly exempt. Nothing in Section F is fidelity work — every use is agent-facing (sprite cadence into the frame stack, scroll-lock into training labels, a register table titled "for Privileged Critic Instrumentation"). | — |
| F-2 Cite the DataCrystal Zelda RAM map | **BANNED** | The definitional case of "RAM semantics beyond the observables the system discovers for itself." This one citation is the upstream provenance of B-7, C-5, C-7, E-3, F-4, F-5, F-6, G-3, G-4b and G-7. | — |
| F-3 Sprite-multiplexing cadence sourced to the disassembly (`OAMIndexOffset += 4`) | MIXED | The conclusion is fine and independently reachable (B-3 reaches it from NESdev; the period is measurable in our own captures). The derivation is not: it is sourced to the game's disassembly and names an engine-internal symbol. `$4014` is a hardware register and naming it is fine; `OAMIndexOffset` is a symbol from someone's reverse-engineered source. | k=4 shipped; primitives shipped; **the measurement has never been run.** Template for banking it: `classify_transition`'s constants (`pan_odo=(128, 384)`, `warp_scene_min`, settle) were all probe-derived and moved into `tests/fixtures/roomgraph/` so RG-0 replays them as a pytest before any live compute. |
| F-4 Screen-scroll input locking; label transition frames NOOP; key on `$0012 = 0x11` | MIXED | A fully legal version exists and is cheap: *measure* that inputs have no effect for a window after a transition. Banned: the provenance ("code disassembly confirms"), the asserted 32-frame constant, and keying on a labeled game-state address. `$4016`/`$4017` are hardware ports and are fine; `$0012` is game RAM with an asserted semantic and is not. **Why this one matters more than it looks:** "all transition frames are labeled as NOOP actions" is a *training-data labeling decision* — disassembly knowledge editing the training input is the most direct route from an injected fact into learned behavior. | **Already built.** `scripts/clear_detect.py` signal 3 (`differential_input_lock_probe:269`) is literally this: run N frames holding a direction and N frames of NOOP from the same state, diff the resulting RAM; near-identical RAM means input had no effect. Already voting in the 4-signal confluence. |
| F-5 "NES Memory Register Map for Privileged Critic Instrumentation" (nine rows) | **BANNED — every row** | The violation in concentrated form: a hand-authored address-to-semantic map from a third-party wiki, whose title states outright that it exists to be wired into the agent's training loop. No partial salvage — the table's entire content is the labels, and the labels are the banned thing. | — |
| F-6 Sub-pixel jitter and CNN aliasing, measured at `$0070`/`$0084` | MIXED | The underlying vision question is genuine and bears on our tile-decoder and low-res pipeline work. Stated with two labeled addresses as the measurement source, it is the banned form. | Legal position signal exists (odometer; `odometer_cert.py` check 3 is HUD-split immunity). The aliasing study is net-new and binds only on the pixel path, since the tile MLP is the default. |
| F-7 Are there configurations where OAM cycling drops a sprite for >4 consecutive frames? | PASS | The correct falsifier for the k=4 choice, answerable purely by measurement on our own captures. No addresses, no routes, no disassembly. | **Net-new.** `peek_oam` exists; nothing calls it for this purpose. k=4 is currently a default inherited from Atari convention plus one cited third-party result, not from any measurement on this core. |
| F-8 "Disassembly verification invalidates claims that transitions allow input buffering" | MIXED | The finding is probably true; the *epistemic warrant offered* is the banned one, and banned unnecessarily — press buttons during a transition on our own emulator and observe. A two-line experiment producing a receipt is strictly better than a citation: third-party verifiable, non-contaminating, and it works on the next game too. | Standing methodology, with the most quotable precedent in `docs/receipts/games/zelda_onboarding_2026-08-10.md` §0 — where a probe landed on an address the disassembly-derived block also names, "that coincidence is recorded below as an observation *after* the fact — the probe is the evidence." |

### Section G — evaluation methodology

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| G-1 Non-gameable intermediate milestones as methodology | PASS | The principle contains no game content and is strongly aligned with our claims policy. | Policy already: per-level rates are the scoreboard; chain rates reported with honest compounding math (twelve levels at 0.95 ≈ 0.54); every quoted rate names its harness. |
| G-2 Legitimacy criteria distinguishing play from engine exploitation | MIXED, leaning banned | Having a legitimacy criterion is valuable and game-agnostic. The specific criteria are lifted from a speedrun wiki, which is a walkthrough under another name — and a glitch catalog handed to the *designers* is itself a hint sheet. **Policy question the report does not surface:** if our system discovers an engine exploit by itself, disqualifying it imports an outside community's ruleset into our evaluation. A self-discovered exploit is arguably a legitimate result to report as what it is, not silently void. | "Did the claimed run happen" is mechanized: action receipts recorded and self-replay verified; `scripts/emit_fm2.py`; `_xram_local` re-derives the room ordinal against a **frozen** index and marks a diverging replay UNVERIFIED. No glitch-legitimacy policy exists anywhere — that is an owner decision, not an implementation. |
| G-3 Eleven weighted progression checkpoints with RAM-audit verification | **BANNED as instantiated (the concept survives)** | Fails twice, independently. (a) Every verification criterion is a RAM-map address or bitmask, all eleven footnoted to DataCrystal. (b) Every *objective description* is a walkthrough table of contents — read the middle column top to bottom and it is the game's critical path in order, with the required items and the dungeons named. A *weighted, ordered* milestone list functions as a curriculum: it tells the designers what order to solve the game in, and any later "the agent progressed" claim is measured against a yardstick that already encodes the answer. **The agent never has to read it for the contamination to land.** | See §6 for the legal reconstruction. Components exist; the assembly does not; the one piece that was tried **FAILED** (IS-1a). |
| G-4 Automated post-hoc adversarial auditor architecture | PASS | An out-of-band automated auditor that can disqualify a run after the fact is exactly the discipline our claims policy demands; automating it is an improvement over doing it by hand. | Partial. Post-hoc adjudication is a habit with machinery: `scripts/phase3_adjudicate.py`, `experiment_preflight.py`, `eval_regime_stats.py`, `compare_runs.py`, `campaign_report.py`, `make provenance-check`, plus pre-registration ("locked once written; corrections are dated addenda, never edits"). Missing: one named binary that ingests a run directory and returns pass/disqualify. |
| G-4a Screen-wrap glitch audit (5 px from an edge + a 1-frame perpendicular turn) | **BANNED** | The rule *is* the exploit recipe, transcribed from a speedrun wiki into a detector. Encoding it documents the exploit in our own codebase, and the 5 px threshold additionally requires a position source — in practice `$0070`/`$0084`. | — |
| G-4b Block-clip audit: flag Δp > 2 px/frame through solid collision tiles | MIXED | "Flag physically impossible position discontinuities" is a good game-agnostic anomaly detector. Three changes required: the position source must be visually estimated, the "solid collision tiles" oracle presumes a collision map we may not have, and the 2 px/frame constant must be measured not asserted. | Pieces exist, none of them a disqualifier: `odometer_cert.py` check 4 (a savestate load produces no spurious mega-delta) and check 5 (restore exactness); the wrap-aware fold; `clear_detect.py`'s `progress_median: K`, which median-filters exactly the impulse class that produced the Double Dragon false clear (72 → 846 → 88 in 5 steps, verdict `state_artifact`). |
| G-4c Inference isolation audit: verify the deployed process is hook-free | PASS | **The best item in the report.** It converts the central honest-eval commitment — the evaluated policy had no privileged access — from a promise into a mechanically verified property, checkable by a third party and enforceable in code. Zero game content; transfers to every title. | **Net-new, and building it collides with an existing design choice.** The posture exists (cold power-on, zero state loads at test time, `src/training/cold_probe.py` process isolation). The assertion does not — and on the default learning path the observation *did* touch a memory hook (`tile_observations/smb.py`), so a literal vision-only audit would **fail every tile-mode checkpoint by construction, including the flagship 0.76 1-1 number.** See §6 for the usable narrower form. |
| G-4d Verify item-prerequisite bitmasks prior to room entries | **BANNED** | Two injections at once: "item prerequisite bitmasks" is the RAM map, and "which items are prerequisites for which rooms" is the lock-and-key dependency graph — the walkthrough's core content. | — |
| G-5 Weight partial progress without letting reversible changes accumulate credit | MIXED | The anti-farming design question is real and generic; it inherits G-3's banned premise and uses light content framing. Re-asked over discovered milestones it passes. | Primitives shipped: `GenericReward`'s frozen post-warmup candidate set; the barren counter; `level_key: []` plus clear-count == 0 assertions in every Zelda room-graph gate run, precisely so coverage can never be scored as a win; `src/training/plr.py` (inverse-recent-success, no stage advancement, no mid-rollout capture); `reverts_seen` in `discover_item_bits.py`. Missing: a weighted partial-credit scoreboard. |
| G-6 Input-log anomaly threshold separating micro-stepping from noise | PASS | Purely a question about input-log statistics on our own receipts. | **Net-new.** Logs exist in quantity (`traces.pkl`, `sol_*.actions.npy`); `scripts/macro_mine.py` already made one relevant methodological choice (non-overlapping occurrence counting; drop single-action repeats under 8 steps). Nobody has fit a threshold. |
| G-7 Score is farmable; tie progress to irreversible bitmask changes | MIXED | First half is a genuine game-agnostic evaluation lesson matching our own history. The prescription re-imports the RAM map and names game content. | First half settled doctrine with first-party receipts (die-respawn inflation; the Double Dragon combat-blip false clear; `GenericReward`'s score-hunt freeze). Legal prescription: discovery half built and **FAILED its first gate** (IS-1a); weighting half unbuilt. |

### The source report's five-item roadmap

| Recommendation | Verdict | Reasoning | Already-built-as |
|---|---|---|---|
| R-1 Transition exploration from RND to topological Go-Explore | MIXED | The headline move is Tier-1 sanctioned; the cell key is C-7 verbatim and banned; "128 overworld screens" and "un-bombed walls" are map knowledge. | **Already built, and "low marginal value" understates it.** Go-Explore is the core search; the topological layer shipped and passed RG-1 on Zelda; RND here is an *option*, not the primary exploration mechanism, so "transition from RND" does not describe this system at all. |
| R-2 Asymmetric actor-critic with a RAM-derived training critic | MIXED | Same split as E-3. "Deploy an un-hooked, vision-only actor at runtime" is exactly right. Three problems: the citation is DataCrystal (signalling a labeled, not raw, implementation); "precise credit assignment for multi-dungeon item dependencies" only means anything if you already possess the dependency graph; and any *reward* from labeled RAM is banned outright under E-4. | Critic net-new; deployment posture already built. Our own attempt to *discover* such dependencies produced 13 root-caused false positives and zero real ones (IS-1a). |
| R-3 Goal-conditioned action masking at the System-1/System-2 handoff | MIXED | Contrastive waypoint embeddings over *visited* states pass under C-1's discipline; masking is game-agnostic machinery. The banned component is the goal source (C-4's VLM). | Mask built with a real negative; legal goal source built. Contrastive embeddings net-new. Live surface worth naming: `state_sig` entries are hand-pasted into a profile's YAML from the discovery ledger, and `ITEM_SEMANTICS_ENGINE_2026-08-25.md` §7.6 flags the likely authoring mistake (an item-bit address pasted into `room_sig:`, corrupting `room_id()`) as having **no structural guard**, only a documentation tripwire. |
| R-4 Adopt frame-stacking k=4 | PASS | The cleanest item in the roadmap: NES-generic, standard preprocessing, no addresses, no routes, no human data. | **Already built at exactly k=4 in both stackers.** Adopting it is a no-op; the only actionable content is the refinement — measure the flicker period (F-3) and attach F-7 as the standing falsifier — and both halves of that are net-new. |
| R-5 Privileged RWBC pre-training on ~500 h of pseudo-labeled human video | **BANNED** | Both halves fail independently: the corpus is the Tier-3 clause at maximum scale, and the labeled-RAM critic filtering it is the F-5 table wired into a loss function. There is no version that survives editing, because the corpus *is* the item. **Second, independent reason to decline:** we already ran this in legal form and it failed on its own terms (D-9). R-5 is simultaneously the most banned item in the roadmap and the least supported by our own evidence — we do not need the policy to reject it. | — |

**Tally:** A 5/4/0 · B 7/2/1 · C 4/4/3 · D 1/3/6 · E 3/3/2 · F 1/4/3 ·
G 4/4/3 · roadmap 1/3/1 — **26 PASS / 27 MIXED / 19 BANNED across 72
items** (PASS/MIXED/BANNED).

---

## 5. REJECTED — banned content, named verbatim

**This section exists so that nobody downstream implements a banned item
because the surrounding prose made it sound adopted.** Everything below
is quoted from the source report and is prohibited by `CLAIMS.md`. None
of it is to be typed into a config, a script, a reward function, an
evaluation predicate, a test fixture, or a prompt. The governing test in
every case:

> *Could this decision have been made by a party who has never seen the
> game?* If not, it is banned.

### 5.1 Banned sources — do not cite, do not open, do not consult

- **"Complete Disassembly of NES The Legend of Zelda (Aldo Núñez)"** —
  `https://github.com/aldonunez/zelda1-disassembly`,
  `https://computerarcheology.com/NES/Zelda/`. Tier 3's first named ban
  is "Game-disassembly knowledge."
- **"The Legend of Zelda/RAM map" — Data Crystal** —
  `https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda/RAM_map`. The
  definitional case of "RAM semantics beyond the observables the system
  discovers for itself."
- **"NES RAM (Mapping/Finding Values)" — FCEUX** —
  `https://fceux.com/web/help/NESRAMMappingFindingValues.html`. A
  tutorial for reverse-engineering which address means what; bringing it
  into the design loop is bringing the RAM-map methodology in by a side
  door.
- **"Screen Scrolling" — ZeldaSpeedRuns** —
  `https://www.zeldaspeedruns.com/loz/tech/screen-scrolling`. A speedrun
  wiki is a walkthrough under another name.

The *fidelity/mapper exemption does not reach any of these uses.* Using
a disassembly to diagnose an emulator bug is exempt. Every use proposed
in the source report is agent-facing: sprite cadence into the policy's
frame stack, scroll-lock into training-data labels, and a register table
whose title is "for Privileged Critic Instrumentation."

### 5.2 Banned addresses and labels — every one, verbatim

Do not add any of these to any file in this repo, in any form, including
comments. The table below is the single deliberate exception: a
quarantine ledger has to name what it quarantines or no grep can enforce
it. Nothing outside this table and §8.1 may carry them.

| Address / mask | Label asserted by the report | Where it appears |
|---|---|---|
| `$0070` | "Link's Current X-Coordinate (Sub-pixel Space)" | B-7, B-8, F-5, F-6, G-4a, G-4b |
| `$0084` | "Link's Current Y-Coordinate (Sub-pixel Space)" | B-7, B-8, F-5, F-6, G-4a, G-4b |
| `$00EB` | "Active Screen ID (Topological Node Index)"; range asserted `0x00–0x7F` | C-5, C-7, E-3, F-5, G-3 |
| `$066F` | "Link's Health (Units of 1/16th Heart)" | E-3, F-5 |
| `$0670` | "Link's Maximum Heart Containers" | F-5, G-3 (CP-03) |
| `$065C` | "Inventory Bomb Count" | E-3, F-5 |
| `$066E` | "Inventory Key Count" | C-7, F-5 |
| `$0671` (bits 0–7) | "Triforce Fragment Bitmask Array" | C-7, E-3, F-5, G-3 (CP-02, 04–10) |
| `$0012` (`0x05`, `0x11`, `0x1A`) | "Execution State (Play / Scroll Pan / Ending Credits)" | F-4, F-5, G-3 (CP-11) |
| `$0657`, `$0656` | Sword-tier flags | G-3 (CP-01, CP-03) |
| `$065C–$0671` | "inventory bitmasks" (as a region) | E-3, R-2 |

The table header in the source report reads **"NES Memory Register Map
for Privileged Critic Instrumentation."** There is no partial salvage:
the table's entire content is the labels, and the labels are the banned
thing. Nine addresses, nine Tier-3 injections, one source.

**Repo-side note, because an enforcement grep scoped to the source report
would miss the larger problem.** Nine of the eleven quarantined rows are
*already compiled into this repo* as named constants in `impl
ZeldaReward` (`nes_core/src/rewards.rs:433`–`:467`) — every row except
`$065C` and `$0012` — under labels that carry the same semantics
(`RAM_LINK_X`, `RAM_LINK_Y`, `RAM_MAP_X`, `RAM_HEARTS`,
`RAM_PARTIAL_HEARTS`, `RAM_KEYS`, `RAM_TRIFORCE`, `RAM_SWORD` /
`RAM_INVENTORY_START`, `RAM_B_ITEM`). The same block additionally names
`RAM_GANON_DEFEATED = 0x0672` and `RAM_BOMBS = 0x0658`, which the source
report does not contain at all. These predate this review and are adopted
from nothing in it; they are recorded here so the ledger is complete and
so §8.1's disposal question is scoped correctly. Nothing in this document
authorizes any of them.

### 5.3 Banned map, route and walkthrough content

- `"Nodes (V): Screen ID (0x00 to 0x7F)"` — simultaneously a RAM index
  and an assertion that the overworld has 128 screens.
- `"Edge Attributes: {Unblocked, Requires_Ladder, Requires_Raft,
  Bombable_Wall, Burnable_Bush}"` — a hand-authored enumeration of the
  game's lock-and-key traversal rules. **This is the walkthrough.** The
  item-semantics engine exists precisely to *discover* gate classes;
  being handed the answer set defeats the mechanism and voids any result
  it produces.
- The worked graph excerpt under the Topological Graph Memory diagram: a
  literal map fragment pairing screen indices with named destinations and
  the item each edge requires. Not transcribed here, for the same reason
  as the checkpoint table — the node indices are §5.2 content and the
  destination names are route knowledge. Banned in full.
- `"the 128 overworld screens"`, `"when the agent acquires a new item
  (e.g., Bombs)"`, `"previously visited screens containing un-bombed
  walls"`.
- `"cells containing un-tested destructible environmental tiles"`;
  `"navigate directly along documented graph edges back to target
  coordinates"` — the "Never" clause's "any instruction naming where to
  go in a specific game."
- `"Direct natural-language goal vectors (e.g., 'Navigate to dungeon
  entrance')"`.
- `"depleting bombs before discovering a mandatory secret wall"`;
  `"bombable wall flags"`; `"placing bombs at specific wall tiles"`.
- **The entire CP-01 … CP-11 checkpoint table** — all three columns,
  including Completion Weight. Read top to bottom, its eleven objective
  descriptions are the game's critical path in order: each names the item
  or the dungeon required at that step, and the weights rank them. They
  are deliberately **not transcribed here.** By this document's own G-3
  reasoning a weighted, ordered milestone list *is* a curriculum and the
  agent never has to read it for the contamination to land — so copying
  it into the repo in order to prove it was rejected would defeat the
  rejection. The source PDF is the record. The ban covers the table, its
  ordering, its item and dungeon names, and its per-row verification
  criteria (which are §5.2 addresses and bitmasks).
- `"TAS data is parsed via BizHawk Lua scripts strictly to extract
  ground-truth topological maps, hidden staircase coordinates, and
  optimal boss engagement distance metrics."` The single most flagrant
  item in the report — a literal coordinate list of the game's secrets
  plus combat strategy, harvested from expert human play.
- `"Verifies item prerequisite bitmasks prior to room entries"` — the
  lock-and-key dependency graph.
- `"placing Link 5 pixels from a screen edge and executing a 1-frame
  perpendicular D-Pad turn"` — an exploit recipe from a speedrun wiki.

### 5.4 Banned training data — human gameplay video

- `"Phase 2: Unlabeled Human Gameplay Video (YouTube / Twitch Streams)
  --> Pseudo-Label via IDM -> Construct Multi-Player Demonstration
  Corpus"`
- `"An Inverse Dynamics Model is trained on 50 hours of human gameplay
  captured via emulator instrumentation, pairing visual frames with
  ground-truth controller inputs"`
- `"The IDM processes ~500 hours of public Zelda 1 gameplay videos,
  generating pseudo-labeled demonstration tuples"`
- `"down-weighting suboptimal human mistakes (such as taking avoidable
  enemy damage)"`
- Use of third-party `.fm2` / `.bk2` movie files as training input.

Tier 3 bans "hand-driven input segments in **any training input**" —
note the breadth: any training input, not merely final weights or
evaluation. Pseudo-labeling changes how the human actions are
*recovered*, not whose actions they are, and by `CLAIMS.md`'s own
reasoning about automated walkthrough recall, **the ban does not weaken
because the transcription was automated.** Every one of those 500 hours
was played by someone who knew the map, the item order, and where the
secrets were; the resulting policy's competence is downstream of that
knowledge in a way no evaluation protocol can disentangle.

### 5.5 Banned reward shaping

Every term of the report's `R(t) = w1·R_combat + w2·R_traversal +
w3·R_milestone` hierarchy: `"+ Damage Dealt to Enemies (+1.0 per HP)"`,
`"- Damage Taken by Link (-2.0 per half-heart)"`, `"+ Evasive Shield
Blocking (+0.2 per successful block)"`, `"+ Uncovering Hidden Staircase
/ Secret Tile (+10.0)"`, `"+ Major Item Acquisition (Bow, Raft, Stepping
Stone) (+50.0)"`, `"+ Triforce Piece Collected (+200.0)"`, `"+ Boss
Defeated (+100.0)"`.

Banned twice over: by the "Never" clause's "per-game reward shaping," and
independently by Tier 2, which freezes the five legacy `LEVEL_*` ladders
and states that **"no new hand ladders will ever be authored."**

### 5.6 Banned experimental design

- `"Zelda 1 features a static overworld layout. Overfitting to the
  overworld tile layout is desirable."` — a design decision made by
  reading a map, and a direct contradiction of the honest protocol.
  Sticky 0.25 and start-jitter exist to punish memorized layouts.
- `"Domain randomization is applied exclusively to enemy spawn
  positions"` — **not implementable from outside the emulator.** Placing
  enemies requires writing game memory, and "RAM edits" is banned
  outright with no disclosure tier available.
- Any LLM/VLM emitting goals, rewards or exploration targets inside a
  run (`"SYSTEM 2: VLM SEMANTIC PLANNER | - Polls every 2-5 seconds"`).

---

## 6. Purity-compliant translation — the MIXED items

For each MIXED item: the same architectural pattern, rebuilt on this
project's own discovery tools. The general move, stated once: **wherever
the report says "the disassembly confirms X," substitute "measure X on
our own frames and record the receipt."** F-8 is the archetype — a
two-line experiment on our own emulator replaces a citation and yields
an artifact that is third-party verifiable, non-contaminating, and
portable to the next game.

**A-2 — out-of-band verifier.** Keep the pattern (policy sees pixels,
verifier scores out-of-band); replace "memory-scanning evaluator" with
what we already run: `scripts/clear_detect.py`'s 4-signal confluence
(3-of-4 to declare), the certified odometer, or `eval_game.py`'s
predicate path outside the observation stream. No new work.

**A-4 — rewind-and-attribute credit assignment.** Keep the rule; key it
on observables we discovered. `scripts/hazard_collect.py` already does
this and its labels are as clean as this repo gets. Two lessons must
travel with any reuse: (1) the label-provenance bug — an earlier
revision held the forked action for the whole horizon, answering the
wrong counterfactual and **vetoing 77.4% of a working policy's own
chosen actions** — is fixed and now disclosed per-dataset in `meta_json`
as `continuation_mode`; (2) `hazard_mask.NEG_MASK = -1.0e9` and *not*
`-inf`, because `-inf` makes the entropy term `0 * -inf = NaN`, the
trainer's NaN guard silently skips the actor update, and both arms then
evaluate identically **because they were the same actor**.

**A-6 — deadlock taxonomy and the movement quantum.** Do not assert
16 px; measure it. Fit a histogram (or autocorrelation) of observed
per-frame displacement from the odometer's own `dx` stream and record
the derived quantum with the run. For the macro-deadlock class, drop the
bomb example entirely and state it structurally: *an exhausted
consumable component of the state signature with no reachable
replenishment cell in the archive.* `scripts/discover_item_bits.py`
already splits flags from counted resources and routes counters to
`find_progress` in resource mode. Before proposing any new coverage-
statistic wall classifier, read `src/training/wall_taxonomy.py`'s
header: `WallClass.GATED` was removed after 22 candidate statistics over
103 archives all straddled — a solved archive read concentration 98.30,
3.2× the "gated" bracket. It now abstains with the descriptive
non-verdict `UNRESOLVED_CONCENTRATED`.

**A-8 — discrete-state preservation in a world model.** Point
`src/training/latent_cells.py` (VQ-VAE codebook over RAM windows, no
address privileged) at the question instead of naming "bombable wall
flags." It is currently unwired and trains only against synthetic
tensors; wiring it is the prerequisite.

**B-4 — MoE.** Legal form: *k anonymous experts with a learned router,
k chosen by a sweep, no semantic labels authored in advance.* If
post-hoc telemetry clustering shows the experts specialized into
interpretable regimes, that is a **finding** and may be reported as one
— the direction of inference is the whole point.
`src/training/multihead_policy.py` supplies the trunk+heads chassis and
the `export_head` compatibility path. Do not copy
`composite_policy.py`'s `label_from_ram` router; that one is hand-keyed
on RAM semantics and its own header insists results be reported as
hierarchical.

**B-8 — lattice-snapping macros.** Infer the movement quantum from our
own position telemetry (as in A-6), then expose "snap to discovered
lattice" as a macro with the quantum as a telemetry-derived knob
recorded with the run. The knob-recording discipline already exists
(`LINEAGE_KEY_AXES:303`, `room_fp_config_sha`, `stamp_stats_provenance`).
Drop "prior to entering doorways" as a trigger; the legal trigger is
whatever the frontier selection already uses.

**C-2 / C-9a / R-3 — planner and mask.** Keep the two-timescale
slow-planner/fast-executor pattern and the action mask; delete the
language model. The legal goal source is Go-Explore cell selection over
the archive under C-1's visited-states-only discipline —
`go_explore_solve.py:select()` with its existing arms. Any mask must be
learned or telemetry-derived, never authored, with its derivation
recorded alongside the run; `hazard_mask.py` is the template, including
its non-optional escape hatch.

**C-5 — topological graph.** Already the stripped form: nodes =
`nt_fingerprint` (blake2b-64 over a masked nametable, mask calibrated
from our own idle/walk frames by `scripts/room_fp_calibrate.py` with a
per-game receipt required under `docs/receipts/room_fp/`); edges =
transitions actually taken, with `EdgeStat` counts; attributes =
`EdgeStat.cap_hist`, mapping `str(cap_sig)` → traversal count where
`cap_sig` is a `state_sig` bit-vector slot and **bits are never named**.
The report's own example sentence — "edge 41 was impassable across 900
attempts; became passable after state-signature bit 7 appeared" — is
precisely Stage 3a `correlate_boundary_edges` in
`scripts/discover_item_bits.py`. Open debt: exemplar persistence (A-7)
and the `cap=1024` ceiling that saturated both ON-arm runs.

**C-6 — capability-change revisit.** Legal statement: *novelty decays
globally; when the agent's capability signature changes, cells
previously exhausted may become newly productive, and a purely global
novelty signal cannot represent that.* Implementation requires the
deferred `(dst, cap_sig)` adjacency rekey plus a `barren`-reset path on
capability change. This is the highest-value clean item in Section C.

**D-2 / D-6 / D-9 — training data.** The only legal demonstration source
is our own: `sil.py` (own clears), `demo_bank.py` (own solver demos,
allowlist-gated), `ppo.py`'s `demo_anchor_loss`. The advantage-weighted
BC formula is fine over those trajectories with advantages from our own
observables — but note it has already been evaluated and the answer to
D-9 is on record twice (clone 1.0 → sticky 0.00; smodice argmax 0.669 →
0.0 clears on 50 honest episodes). Re-asking is re-running a closed
experiment. `scripts/convert_fm2.py` must never be pointed at a
demonstration corpus; `configs/demo_allowlist.txt` + `make
provenance-check` + `checkpoints/QUARANTINE_tier3/` are the enforcement
that already answers this.

**E-3 / E-7 / E-8 / R-2 — the privileged critic.** The pattern is usable
**if and only if the critic's privileged input contains no interpreted
address.** Two legal constructions:

1. *Raw whole-RAM.* Feed the critic the entire undifferentiated 2 KB
   vector — no address named, sliced, weighted, normalized specially, or
   fed to an auxiliary head — disclosed in one sentence, discarded at
   deployment. Enforce with a test: **grep the critic path for hex
   literals; if any appear, it has crossed the line.** Precedent for
   mechanized enforcement exists on the key path —
   `LINEAGE_KEY_AXES` stamps `sig_arity`/`sig_sha` (the exact
   addresses/matches/mods) into the archive lineage and **refuses
   cross-schema resume**, so an address cannot be silently added
   mid-lineage. There is no equivalent guard on a critic input path
   today; one would have to be written.
2. *Discovered-observable critic.* Build the privileged vector from
   observables this project found for itself — a health-like byte via
   `find_hp_lives` / `lives_from_death_drives`, room identity via
   `nt_fingerprint`, position via the certified odometer, capability
   bits via `discover_item_bits.py` — **never from the disassembly's
   semantic labels.** Each component then carries a discovery receipt
   rather than a citation.

Whichever is built, pre-register **E-6** as its falsifier (does
privileged critic access induce overfitting that fails under rendering
artifacts?) and carry the engineering caveat: a critic confident about
outcomes the actor cannot predict inflates advantage variance —
structurally the same failure that produced the 77.4%-veto result.

On **E-8** specifically, the report's argument should be reversed rather
than adopted. It treats the noisy-but-legal visual/self-discovered
option as dominated by labeled RAM. Here it is the only admissible
member of the pair, and "noisier" is a cost we pay on purpose — with
evidence that it works: the odometer is certified 5/5, and the fight-gate
mechanism rediscovered `0x0398` blind. The honest status line the
odometer entry carries is the right template for any such claim: *the
instrument is certified, the games remain unsolved, and no clear is
attributed to it.*

**F-3 / F-4 / F-6 / F-8 — hardware constraints.** Replace every
"disassembly confirms" with a measurement:

- *Flicker cadence.* Sample `peek_oam` over our own captures, measure
  the multiplex period and the max consecutive-frames-a-sprite-is-absent
  (that is F-7), then choose k from the measurement. Bank the constants
  as fixtures under `tests/fixtures/` so an offline falsifier replays
  them as a pytest before any live compute — exactly how
  `classify_transition`'s `pan_odo=(128, 384)` window was derived and
  frozen.
- *Input-lock window.* Already shipped —
  `clear_detect.differential_input_lock_probe:269`. Use it; do not key
  anything on `$0012`, and do **not** propagate transition-frame NOOP
  labels into any training input.
- *Sub-pixel aliasing.* Ask the question against the odometer's
  visually-estimated position signal, which is both legal and the thing
  we actually care about aliasing in.

**G-2 — legitimacy criteria.** Keep the concept; do not import the
speedrun wiki's rules. This needs an owner decision before any code: if
our own search discovers an engine exploit, is that a legitimate result
reported as what it is, or a disqualification? Every other surprising
own-discovery in this project has been reported, not voided.

**G-3 / G-5 / G-7 — the milestone framework.** The concept survives; the
rows must be discovered. The valuable properties are all game-agnostic:
milestones should be **monotone, hard to farm, irreversible, verified
out-of-band, verified post-hoc, and weighted for partial credit.** None
of them requires knowing what a Triforce is. Legal reconstruction:
define milestones as **irreversible state-signature transitions** —
components of the system's own discovered state signature that no
observed trajectory ever reverses — ranked by rarity across the archive
and weighted by that rarity. Monotone by construction; gameable-
resistant by construction (a farmable action is by definition reversible
or repeatable); discovered rather than authored; never needs a proper
noun. Components already owned: `discover_item_bits.py` Stages 1–2
(`reverts_seen > 0` anywhere ⇒ **permanently rejected**, with a frozen
per-lineage idle prefilter, change-rate filter, and K-of-N cross-rollout
confirmation), `clear_detect.py` and `eval_game.py` for out-of-band
verification, `discover_observables.py` Gates 1–2 and the odometer for
progress axes, archive statistics for rarity, and the sticky/jitter
protocol for denominators. **Keep the "Completion Weight" column as a
concept — partial credit yields a gradient instead of a binary — and
discover the rows.**

The honest qualification, because the parts list is more ready than the
assembly: **the discovery half has one real gate attempt behind it and
it failed.** IS-1a (`runs/item_semantics/is1a/IS1A_VERDICT_2026-08-25.md`,
commit `cc0bc9e`), run over 12 replayed RG-1 Zelda traces: 607 keys →
**51 "confirmed"** raw; after a fair death-truncation using only the
already-claimed lives byte, 224 keys → **13 "confirmed."** Both false-
positive classes were root-caused and neither is an item: (A) Zelda's
death→continue-menu animation rewrites RAM 20–130 steps after `lives`
hits 0, and (B) the residual 13 fire at *identical elapsed-step offsets*
(7, 53, 69) across 8 independently-mined trajectories — the signature of
a deterministic engine-init artifact. Two mitigations are named and
**filed, not built**: death/lives-based rollout truncation ahead of
`scan_rollout`, and a cross-rollout transition-step consistency check
treating identical first-flip steps as artifact evidence rather than a
promotion vote. Also worth recording: Stage 3a returned **0 leads on all
four RG-1 runs** for a trivial reason — those runs predate the
`cap_hist` graft, so every edge reads the empty default. That is safe
degradation, not signal.

**G-4b — displacement anomaly audit.** Fit the per-frame displacement
distribution from our own rollouts and flag samples beyond a high
quantile; drop the "solid collision tiles" oracle entirely (it presumes
a collision map) and the asserted 2 px/frame constant. Position source =
the certified odometer. Precedent for measuring rather than asserting a
threshold: `pan_odo`. Precedent for the impulse class this will see:
`progress_median: K`, which was added after the Double Dragon 72 → 846 →
88 combat blip shipped a false clear.

**G-4c — inference isolation.** Adopt the *intent*, not the literal
wording. A pure "reads strictly raw RGB, zero memory hooks" assertion
would **fail every tile-mode checkpoint in this repo by construction**,
including the flagship 0.76 1-1 number, because
`src/emulation/tile_observations/smb.py` reads CPU RAM to build the
policy input. The usable form is a **declared-and-verified
observation-source assertion**: the run declares its encoder class (tile
/ pixel / odometer-pseudo-RAM), and the auditor mechanically verifies
the deployed policy read nothing outside that declaration, with the tile
class carrying its Tier-1 disclosure. That is enforceable today; a pure
hook-free assertion is a much larger product decision.

---

## 7. Revised roadmap (replaces the source report's five items)

Ordered by value per unit risk given what already exists. Each item
states whether it is net-new work or existing machinery to point at
Zelda.

**N-1 — Measure the OAM flicker cadence; make k a receipt.**
*Net-new, cheapest clean win in the audit.* Sample `peek_oam`
(`nes_core/src/python.rs:766`) over our own captures; measure the
multiplex period and the maximum consecutive frames a sprite is absent
(F-7); bank the constants as fixtures under `tests/fixtures/` with an
offline replay test, following the `classify_transition` precedent. k=4
is currently inherited from Atari convention plus one third-party
citation, not from any measurement on this core. Deliverable: a measured
constant with a receipt, portable to the whole library. Replaces R-4,
whose literal content (adopt k=4) is already a no-op.

**N-2 — Capability-change revisit arm.**
*Net-new mechanism on shipped chassis; highest-value clean item in
Section C.* When the state signature gains a component, reopen archive
cells whose action distribution was previously exhausted. Requires the
`(dst, cap_sig)` adjacency rekey that both item-semantics designs
deferred to v2 (`ITEM_SEMANTICS_ENGINE_2026-08-25.md` §8) plus a
`barren`-reset path. This is the only genuinely new content inside R-1;
everything else R-1 proposes is already built and receipted.

**N-3 — Close IS-1a's two filed mitigations, then build the weighting
column.**
*Mitigations filed-not-built; weighting net-new.* Land death/lives-based
rollout truncation ahead of `scan_rollout` and the cross-rollout
transition-step consistency check, re-run the IS-1a gate, and only then
build discovered-milestone weighting: rank irreversible state-signature
transitions by archive-wide rarity and turn that into partial-credit
evaluation. This is the legal reconstruction of G-3 and the missing half
of G-5/G-7. Pre-register before running; the gate that already failed
once is the reason.

**N-4 — Declared-encoder isolation audit.**
*Net-new; the best idea in the source report, in its usable form.* A run
declares its observation-source class; an auditor mechanically verifies
the deployed policy read nothing outside that declaration. Do **not**
build it as a literal vision-only assertion — see §6, G-4c. Natural
enforcement point for any future privileged-critic work: a critic that
reads RAM during training cannot leak into a deployment that is provably
within its declared class.

**N-5 — Point the shipped discovery stack at Zelda for a
discovered-observable critic, before writing any critic.**
*Existing machinery, new target.* If asymmetric actor-critic (E-1/E-3/
R-2) is pursued at all, the privileged vector must come from
`find_hp_lives` / `lives_from_death_drives` (health-like byte),
`nt_fingerprint` (room identity), the certified odometer (position), and
`discover_item_bits.py` (capability bits) — never from the report's
labels. Pre-register E-6 as the falsifier and ship a hex-literal test on
the critic path before a line of critic code is written. Note the
inversion worth stating plainly: this construction is **cleaner than the
substrate we already run**, since the tile observation is
disassembly-provenanced and Tier-1-disclosed rather than discovered.

**N-6 — Self-calibrated displacement-outlier disqualifier.**
*Net-new, small.* Fit per-frame displacement from our own rollouts over
the odometer signal; flag beyond a high quantile as a run-disqualifying
audit. G-4b minus the collision oracle and the asserted constant.

**N-7 — Input-log anomaly threshold.**
*Net-new, small.* Fit a threshold over the existing action-receipt
corpus (`traces.pkl`, `sol_*.actions.npy`). `scripts/macro_mine.py`
already made the right methodological choice (non-overlapping counting;
drop single-action repeats under 8 steps).

**N-8 — Owner decision: exploit legitimacy policy.**
*Not an implementation.* If our own search discovers an engine exploit,
is it a legitimate result or a disqualification? `CLAIMS.md` is silent.
Decide deliberately rather than inherit a speedrun community's ruleset.

**Deferred, and named so they are not mistaken for oversights:**
anonymous-expert MoE (B-4 — chassis exists; standing evidence from
`RECURRENT_AB_VERDICT` and `OPTIONS_NEGATIVE` argues for finishing
open lanes first); contrastive waypoint embeddings (C-9b — genuinely
absent, but downstream of N-2); any transformer arm (no evidence
supports spending here before the GRU salvage items).

**Must NOT be recommended — would re-invent or regress shipped,
receipted machinery:** building a topological room graph for Zelda
(RG-1 PASS); adopting frame-stacking k=4 (default since inception);
building a Go-Explore archive or selection arm (two implementations,
Tier-1 sanctioned); building an inert-input/transition detector by
measurement (`clear_detect.py` signal 3 already is one); building a
rewind-and-attribute credit-assignment wrapper (`hazard_collect.py`,
with a fixed provenance bug and a Phase-3 negative); re-asking whether
BC/DAgger collapses into RL (answered twice); adding a coverage-
concentration "is this wall GATED?" statistic (`WallClass.GATED` was
removed after 22 candidates over 103 archives all straddled); proposing
action-commitment options as a fresh idea (0/100 vs 8/100); proposing a
recurrent policy as the fix for the sticky wall (0.06 vs 0.76 — and the
honest framing is "mechanism untested," not "refuted").

---

## 8. Two open purity items this review surfaced

Neither was asked for; both bear on any Zelda work and both are stated
as **uncertain, not violated**, because that is what the evidence
supports.

**8.1 — `ZeldaReward` is the E-4 hierarchy, live in this repo.**
`nes_core/src/rewards.rs:372` defines `pub struct ZeldaReward`,
instantiated at `:4444`, with weights named `triforce_piece`,
`dungeon_enter`, `new_item`, `first_sword`, `key_collected`,
`magic_key`, `map_compass`, `all_fragments`, `level9_enter`,
`ganon_reached`, `win_bonus`. `configs/zelda.yaml` sets them
(`dungeon_enter: 3000.0`, `exploration_bonus: 50.0`,
`triforce_piece: 10000.0`) with a comment block reasoning explicitly
about "128 screens × 20" — i.e. map size.

**The config supplies only the weights.** The addresses those weights are
scored against are hardcoded in the Rust: `impl ZeldaReward`
(`nes_core/src/rewards.rs:433`–`:467`) declares its own labeled
constants, nine of which are rows of the §5.2 quarantine table (see the
repo-side note there). The profile's `ram_mapping:` block
(`configs/zelda.yaml:21`) carries the same labels a second time — its
in-file comment reads "Win chain (disassembly + emulator-verified)" over
`triforce_pieces: 0x0671` and `ganon_defeated: 0x0672` — but it is a
parallel copy consumed by other tooling, **not** the reward's source.
That distinction decides the remedy, and it is the reason this item is
larger than it first appears.

`CLAIMS.md`'s Tier-2 freeze names only "the five hand-calibrated LEVEL_* reward
ladders in nes_core/src/rewards.rs" (`LEVEL_1_1`, `LEVEL_1_2`,
`LEVEL_1_3`, `LEVEL_1_4`, `LEVEL_2_1`). `ZeldaReward` is not named in
that freeze, not in the Quarantine section, and not in
`configs/demo_allowlist.txt`'s enforcement scope.

The profile has an internally split posture worth noting exactly: its
`solve:` block carries an explicit purity statement — *"The pre-existing
`ram_mapping:` above descends from a disassembly and is NOT the source
for anything here — where a probe happens to land on the same address
that is a coincidence recorded in the receipt, not evidence"* — while
the reward path reads those same addresses anyway, out of its own Rust
constants, entirely outside the disclaimer's reach. The trainer never
reads `solve:`, so the disclaimer constrains the discovery lane and
nothing else. The precedent for disposal exists and is exact:
`docs/receipts/games/metroid_purity_quarantine_2026-08-10.md`, which
quarantined this class of contamination in `configs/metroid.yaml` under
a `quarantined_external_knowledge:` block with a rediscovery rule, and
stores the values **as strings, not ints**, so a tool folding
`ram_mapping` (`scripts/observatory.py` does `int(a)` over `.values()`)
raises instead of silently consuming banned addresses.

That precedent is exact but **only reaches the YAML copy.** Quarantining
`configs/zelda.yaml`'s `ram_mapping:` block would leave `ZeldaReward`
fully operational, because its addresses are compiled constants, not
config. Any disposal decision here therefore has two halves — the config
block, which the Metroid recipe already handles, and the Rust constants,
which nothing in this repo currently guards. Naming that split is the
whole content of this item; choosing the disposal is an owner call, and
tracing whether any current Learned-ledger run instantiates
`ZeldaReward` is the prerequisite nobody has done.

**8.2 — `scripts/convert_fm2.py` ingests third-party movie files.**
It converts FCEUX `.fm2`/`.fm3` and BizHawk `.bk2` movies into this
project's `.state.bin` action-byte format — the exact D-2b channel the
audit bans, already present. Its inverse, `scripts/emit_fm2.py`, is the
legal direction (emit a power-on-anchored FM2 from our own recorded eval
actions, as a TAS-grade receipt). The guard already exists and has been
exercised for real: `configs/demo_allowlist.txt` (authoritative),
`make provenance-check`, and `checkpoints/QUARANTINE_tier3/` holding
`demos_4_2_full.npz`, `demos_4_2_pilot.npz`, `full_4_2_solution.npy`,
`pilot_4_2.pt`. The correct statement is therefore not "do not build a
movie ingester" — it exists — but **"the allowlist is the mechanism that
already answers Section D, and Section D would require defeating it."**
Worth a sentence in any Zelda plan so nobody wires the ingester to a
"500 hours of gameplay video" idea by analogy.

---

## 9. Honest closing

A large fraction of this report's specific methodology is incompatible
with this project's binding rules as literally described, and no amount
of careful reading changes that. Its privileged RAM critic is specified
by naming three addresses out of a third-party wiki. Its eleven
evaluation checkpoints are a walkthrough table of contents with bitmask
receipts, footnoted to the same wiki on every row. Its entire
training-data pipeline rests on 500 hours of human gameplay video, which
Tier 3 bans in any training input and which pseudo-labeling does not
launder. Its reward hierarchy is per-game shaping, banned by name and
banned again by the Tier-2 freeze. Its domain-randomization plan
requires RAM edits, which have no disclosure tier at all. Section D
cannot be repaired by editing, because its load-bearing asset *is* the
banned thing; and independently of policy, we already ran the legal
version of that experiment and it failed on its own terms — clone
accuracy 1.0 collapsing to sticky 0.00, and an offline fit of 0.669
producing 0.0 clears on 50 honest episodes.

That verdict coexists with a real compliment. Several of the report's
architectural *patterns* are sound, and the two best of them are already
enforced here more strictly than the report proposes: C-1's
visited-states-only search discipline is architecturally guaranteed by
`record_edge` raising on a warp rather than promised in prose, and the
topological room graph the report sketches is shipped, pre-registered,
run four times on Zelda, and adjudicated PASS with a measured 1.33×–1.62×
router lift. Frame-stacking at k=4 is a no-op recommendation here.
Go-Explore is the core search. The rewind-and-attribute wrapper exists
and has already produced a documented negative. G-4c's inference
isolation audit is the best idea in the document and is genuinely
missing — though building it in its literal form would fail every
tile-mode checkpoint we have, including the flagship 0.76, which is
itself a fact worth sitting with rather than routing around.

The most useful thing to carry forward is not any single verdict but the
habit the report lacks: it never once derives anything from the agent's
own telemetry. Every fact it needs, it looks up. That is what produced
19 banned items — and in several cases, including the flicker cadence
and the input-lock window, the legal path was strictly cheaper and would
have produced a measured constant with a receipt instead of a citation
that cannot be checked. The purity line is not only a constraint here.
On this evidence it is also the better engineering.
