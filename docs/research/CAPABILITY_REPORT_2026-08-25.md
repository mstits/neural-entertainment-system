# Capability report — 2026-08-25 addendum

Scope: this is a short addendum to
`docs/research/CAPABILITY_REPORT_2026-08-24.md`, not a rewrite of it —
that report's two-week window (2026-08-11 → 2026-08-24) is closed and
its numbers stand unchanged. This addendum covers exactly two things
that landed after that window closed: the room-graph engine (commit
`3601c45`) and the launch of v27 (commits `3bb93ef`, `5995272`). Same
ledger tags — LEARNED, EXHIBITION, FORGE — and the same rule: nothing
claimed past what has an actual receipt. Full entries for both are in
`CLAIMS.md`; this is the narrative form.

---

## 1. Room-graph engine [FORGE] — RG-0 green, RG-1 registered and not yet run

### What it is

A new instrument, not a new search behavior: the solver can now
identify *which room it is in* game-agnostically, on games where the
existing scene ordinal is either noisy (Metroid: spurious scene bumps
at clamp/seam with no room actually left) or blind (Zelda cave/dungeon
fades, Rygar's blank doors — none of which move the scene ordinal at
all). Identity comes from a settled, masked blake2b-64 hash of the
physical 2 KB nametable VRAM, interned to a discovery-order ordinal and
carried through the already-shipped `area`/`room_sig`/`room_advance`
machinery via two new pseudo-RAM bytes — so room-keyed cells, transit
detection, and frontier tracking ride shipped code rather than new key
schema. New logic on top: a pan/fade/warp transition classifier from
integrated Δodometer + Δscene during the settle window (a Zelda death
is a *warp* — flat odometer, scene jump — and warps are recorded for
telemetry but mint no adjacency edge and are never routable, which is
what closes the "death mistaken for a navigable door" failure mode
without needing any per-game death observable), edge exemplars for
sticky-replay validation, an aliasing audit, a room-pool selection arm,
and restore-lockstep invariants (global structures append-only, per-
worker state re-derived on `_assign` rather than accumulated across
restores — the documented desync trap this class of engine is prone
to).

Design provenance: synthesized and judged from three independently
authored designs (`docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md`) —
D0 (minimal-diff, zero Rust/key-schema changes) taken as the shipping
chassis, with D1's detector (the offline falsifier, the classifier
state machine, edge exemplars, the aliasing audit) and D2's roadmap
(capability-bitmask keying for lock-and-key progression) banked
verbatim as the v2 direction. FORGE-class per the ledger's four-part
test: the need was found in the system's own probe telemetry, not a
walkthrough; authorship and adversarial review were agentic; it ships
default-off and byte-identical; and its status is stated honestly
below rather than rounded up.

### What's actually proven right now

**Byte-identity (flags off):** verified against pre-branch HEAD on a
16,000-step SMB solve — sha256-identical RAM, archive, and traces
across all 8 workers. This is the load-bearing safety property: the
engine cannot have silently changed behavior for any game that isn't
opted in.

**RG-0, the offline falsifier — 9/9 PASS.** This is a pytest suite
(`tests/test_rg0_roomgraph.py`) replaying banked probe fixtures, no
live emulation required — the cheap-premise-first gate the BINDING
sequencing rule requires before any live compute is spent. Re-run and
reconfirmed today: `9 passed`. It covers all five design-mandated
assertions plus four supporting ones: a Zelda east-exit pan mints
exactly one new node; a Zelda death mints a warp with zero edges; idle
Zelda hashes to a single value post-mask; Metroid's two doors mint
exactly two pan edges; Metroid's scene noise mints zero extra nodes;
plus death-aftermath-is-fades-never-warp-edges, the settle-threshold
hybrid-room negative check, mask/probe-idle-stability reproduction, and
fixture provenance stamping.

### What is NOT proven — stated plainly, not rounded up

**RG-1, the live gate, has not been run.** It is registered
(`docs/proposals/ROOMGRAPH_ENGINE_2026-08-24.md` §6): 4 unattended
90-minute Zelda runs, 12 workers each, with pre-registered pass
conditions — ≥30 distinct settled rooms by 90 minutes and ≥1 fade edge
(validity), router-selected distinct-room count ≥1.25× the router-off
control on both seeds (routing lift), an SMB 5-minute determinism
regression check with `room_fp` absent (integrity), SPS ≥90% of the
room_fp-off Zelda baseline (perf), and 20 sampled edges replayed from
their exemplar restore + action ring (edge validity). None of those
five numbers exist yet. There is no live room-graph traversal of
either Zelda or Metroid, no room count, no edge count, and no routing-
lift figure to cite. RG-1 is blocked on nothing but scheduling — the
machine is currently saturated running v27 (below) — and needs a clean
unattended 4×90-minute window once that clears.

**Bottom line:** the engine is built, tested offline, and safe to ship
default-off; it has not yet demonstrated a single live room graph on a
real game. Cite RG-0 as "the offline falsifier passed"; do not cite it
as "the room-graph engine works on Zelda."

---

## 2. v27: fresh-recovery run [LEARNED, registered] — launched, in progress, no verdict

### The open question this answers

The prior window closed with recovery distillation adjudicated FAIL
across three families (base method, KL-anchored cloning, on-policy
recovery PPO) against the banked backward-1-1 control — every gradient
that touched the consolidated 48k-checkpoint artifact made it worse.
The one thing not yet separated: whether that's because the artifact
is a genuinely isolated optimum, or because post-hoc fine-tuning at a
small parameter budget is the wrong lever and the same recovery states,
present in the curriculum from iteration 0 of a fresh run, behave
differently. v27 is that fresh run: 4 seeds × 250 iterations, 60 envs,
vanilla PPO on the SMB-tiles-pos encoder, gated against the control's
0.767.

### The DR amendment before launch

DR review of the pre-registration (`docs/proposals/
V27_FRESH_RECOVERY_2026-08-24.md`) returned Decision (B): launching an
*unmodified* fresh run risked producing a FAIL that's actually
uninterpretable, conflating a real capacity deficit with primacy bias
or dormant-unit collapse — a confound the prereg's own clause required
resolving before spending the budget. The mandated fix, ReDo (Sokar et
al.) dormant-neuron recycling, is now implemented and shipped
(commit `3bb93ef`): layer-mean-normalized dormancy score (correcting
the DR's own "layer max" phrasing), tau=0.025 checked every gradient
iteration post-GA-warmup, `fc1`/`fc2` hidden units only, Kaiming reinit
with zeroed outgoing columns, and exact-slice Adam-moment clears.
Single arm, ReDo-on for all 4 seeds — per the DR's own outcome mapping,
cumulative-recycle telemetry alone is enough to tell whether the
mechanism was live or inert, so no separate ReDo-off control was run.
17 new tests, default off, byte-identical when off. ADDENDUM 2
(`5995272`, same-day follow-up) root-caused the V7 agreement-bound
pilot check against LayerNorm effects via a tau-sweep (0.05–0.35) and
PASSed under the corrected condition, clearing the last item that had
blocked launch.

### Status as of this writing — IN PROGRESS, not a verdict

Only seed0 has started
(`checkpoints/mario_1_1_v27_recovery_seed0/`); seeds 1–3 have not yet
begun. Seed0 is at iteration 26 of 250. From its live run log:
`best_fitness` climbed 3633 → 4196 and `success_rate` bounced 1–4%
over the last five logged iterations (noisy this early, as expected);
throughput ~2,300–2,420 env-steps/s (~154–161× realtime); ReDo is
armed and reporting `dormant fc1 0/64 fc2 0/32 recycled 0 cum 0 agree
1.0000` — no dormant units found yet, so the recycling mechanism has
not had occasion to fire this early in training. This machine is
currently dedicated to this run; no other heavy compute should contend
with it until it either completes or hits a checkpoint milestone worth
pausing at.

**No honest-protocol number exists for v27 yet, at any seed, and none
should be inferred from the training-telemetry `success_rate` field
above** — that is in-progress training telemetry, not a cold-eval
figure, and the two are never to be conflated per the claims policy's
honest-evaluation-protocol section. The eventual result (pass against
the 0.767 gate, fail, or void) gets its own CLAIMS.md entry when the
run — all 4 seeds — actually finishes.

---

## Unchanged from the prior window

Everything in `CAPABILITY_REPORT_2026-08-24.md` — the odometer, scene
detection, death-semantics fixes, the onboarding wave, the search
capabilities, the sticky-wall decomposition, the recovery-assay and
recovery-distillation verdicts, the honest scoreboard (1-1 0.767 /
1-2 38% / 1-3 21% / 1-4 51%), and the retraction register — stands as
written. This addendum adds two rows to the ledger, not a
reconciliation of the rest.
