# The Totality Basis — which games span "any game"

*2026-08-08. The question: what is the minimal set of games such that
beating them certifies the system can beat (nearly) any NES game in
show mode, and how do we measure true progress toward that?*

## Games are not the unit — mechanism classes are

"Beat N famous games" is the wrong target; games are bundles of
mechanism classes, and totality = covering the classes. The NES action
library decomposes into roughly ten:

| # | Mechanism class | Purest exemplar | Status |
|---|---|---|---|
| 1 | Linear momentum platforming (dense progress axis, frame precision) | SMB1 | **CERTIFIED** (beaten live, 32/32) |
| 2 | Coverage/maze (momentum-vs-coverage taxonomy) | SMB 4-4 / 8-4 | **CERTIFIED** (coverage recipes, live) |
| 3 | Committed-action combat platforming (knockback punish, holds) | Castlevania | 90% — hall = last mile |
| 4 | Vertical/orthogonal progress (scoring axis ⊥ progress) | CV block-3 hall; Kid Icarus | IN FLIGHT (--ortho arm) |
| 5 | Boss state machines + projectile pressure | Contra; Mega Man | open (adapter partly built) |
| 6 | Pure reactive pattern timing (zero navigation) | Punch-Out!! | open (ASM path exists) |
| 7 | Non-spatial planning (no gx-shaped state at all) | Tetris Type-B | open — cheap, high-signal |
| 8 | Room-graph / item-gated open world (no progress byte) | Metroid | open — the big research lift |
| 9 | Interaction-discovery top-down (triggers, inventory) | Zelda | open (win-predicate wired) |
| 10 | Menu/economy/text (RPGs) | Dragon Warrior | OUT OF SCOPE v1 (declared) |

**The basis set (8 games):** SMB1 ✅, Castlevania, Contra, Mega Man,
Punch-Out, Tetris-B, Metroid, Zelda. Neither {Mario, Metroid, Zelda}
(misses 5/6/7) nor {Contra, SMB3, Tetris} (misses 8/9 — the biggest
class) spans the space. SMB3 adds little over SMB1+coverage; it is
content, not a new axiom.

## The totality instrument — how "true progress" is measured

1. **Capability matrix** (above): a class is CERTIFIED only by a
   receipted show-mode clear of its exemplar. Basis progress = 2/8.
2. **The League**: every cycle, sample K games *stratified-randomly*
   from the 793-boot library, point show mode at them UNATTENDED with
   a fixed budget, and score tiers: T1 = verified level-1 clear,
   T2 = deep run (≥half the game), T3 = credits roll. The trend of
   unattended tier rates over random samples — not hand-picked games —
   is the honest totality number.
3. **"Point at any game" gate** (pre-registered): ≥80% of sampled
   games reach T1 within 1 h unattended; ≥⅓ reach T3 within a
   show-night budget; zero fabricated clears (detector-verified).

## Dependency-ordered path (each unlocks the next)

1. **CV hall** (class 4) — in flight; certifies 3+4 together.
2. **Confluence clear-detector v2** (combat-blip + room-transition
   modes still unfixed) — THE league prerequisite: unattended totality
   is unmeasurable without trustworthy generic clear detection. This,
   plus game-over/menu auto-recovery (launcher backlog), makes
   "unattended" real.
3. **Contra** (5) → **Mega Man** (5 hardened + light gating).
4. **Tetris-B** (7) — cheap test that the archive abstraction isn't
   secretly gx-shaped; excellent show content.
5. **Punch-Out** (6) — pure timing axiom; showstopper.
6. **Metroid** (8) — the research lift: solver must build its own
   room graph. **Zelda** (9) rides on it.
7. **League mode**: random-game show nights, tier scoreboard public.

Classes 1-2 took ~6 weeks including all infrastructure. 3-7 reuse that
infrastructure; 8 is the one genuine research unknown remaining.
