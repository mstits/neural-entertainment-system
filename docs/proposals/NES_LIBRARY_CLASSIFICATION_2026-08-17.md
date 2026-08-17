# The NES library against the totality basis — what the ten classes actually cover

Date: 2026-08-17. Companion to `TOTALITY_BASIS_2026-08-08.md`, which
defines ten mechanism classes and an eight-game basis set but classifies
only the exemplars. This document applies the basis to the *whole*
library on disk and reports what it covers and what it does not.

Corpus: 795 ROM files, **789 unique titles** after stripping region and
revision tags (`roms/*.nes`).

## Headline finding: the basis is an ACTION-GAME basis

Classes 1-9 describe navigation, combat, timing, and planning under a
progress signal. Roughly a quarter of the library is not that kind of
game at all. Measured by title-keyword sweep (counts are exact for the
keyword sets used; a handful of titles match two buckets):

| Bucket | Titles | Status against the basis |
|---|---:|---|
| Sports (ball/team/olympic) | 76 | **no class** |
| Racing / driving | 44 | **no class** |
| RPG / menu-economy | 28 | class 10, **declared out of scope v1** |
| Board / card / casino / quiz | 20 | **no class** |
| Edutainment / fitness | 17 | **no class** |
| Light-gun | 13 | **no class** (needs the Zapper) |
| **Out-of-scope subtotal** | **~198 (25%)** | |
| Remainder — action/adventure/platform/shooter/puzzle | **~591 (75%)** | classes 1-9 |

So the honest coverage statement is: *the totality basis spans the
action-game portion of the library, about three quarters of titles by
count.* The claim "point at any game" should either carry that scope
explicitly, or the basis needs new classes (sports and racing alone are
120 titles — more than any single in-scope class).

Sports and racing are not exotic, either: both have dense continuous
progress signals and would likely be tractable, but neither has a
"clear" in the level sense — success is a score comparison or a lap
count, which the current clear-detector vocabulary cannot express. That
is a detector question before it is a solver question.

## Classes 1-9 applied to the in-scope remainder

Exemplars are from the basis document; the additional titles are
assignments from this pass. Counts are estimates over the remainder
except where the basis already certifies a class.

### 1 — Linear momentum platforming (CERTIFIED, SMB1 32/32)
The single largest in-scope class, ~150-200 titles.
Super Mario Bros 1/2/3, Lost Levels, Adventure Island series, Kung Fu,
Ninja Gaiden I-III, Battletoads, Bucky O'Hare, Chip 'n Dale 1/2,
Darkwing Duck, Little Nemo, Power Blade, Journey to Silius, Shatterhand,
Vice: Project Doom, Rockin' Kats, Felix the Cat, Kirby's Adventure,
Bionic Commando, City Connection, BreakThru, Bonk's Adventure.

### 2 — Coverage / maze (CERTIFIED via SMB 4-4, 8-4 recipes)
Whole-game members are rarer than in-level members; ~30-50 titles.
Bomberman 1/2, Boulder Dash, Clu Clu Land, Lode Runner, Dig Dug II,
Bubble Bobble, Crystal Mines, Devil World, Castlequest.

### 3 — Committed-action combat platforming (blocks 0-2 clear; hall open)
~60-80 titles.
Castlevania I/III, Ghosts 'n Goblins, Rygar, Astyanax, Demon Sword,
Legendary Wings, Bad Dudes, Double Dragon I-III, River City Ransom,
TMNT II/III, Battletoads-Double Dragon, Conan, Cowboy Kid.

### 4 — Vertical / orthogonal progress (IN FLIGHT, --ortho arm)
~15-25 titles.
Kid Icarus, Ice Climber, Balloon Fight, DuckTales, Rainbow Islands,
Castelian, Air Fortress.

### 5 — Boss state machines + projectile pressure (adapter partly built)
~70-90 titles once shooters are included.
Contra, Super C, Mega Man 1-6, Gradius, Life Force, Abadox,
Blaster Master, Metal Storm, Burai Fighter, Captain Skyhawk, 1942,
1943, Twin Cobra, Zanac, Guardian Legend, Cybernoid, Alpha Mission.

### 6 — Pure reactive pattern timing (ASM path exists)
~10-20 titles.
Punch-Out!! / Mike Tyson's, Best of the Best, Karate Champ,
Yie Ar Kung-Fu, Pro Wrestling (timing-dominant).

### 7 — Non-spatial planning (open; cheap, high-signal)
~40-60 titles.
Tetris (both), Dr. Mario, Yoshi, Klax, Hatris, Palamedes, Adventures of
Lolo 1-3, Crackout, Arkanoid, Puzznic, Pipe Dream, Solomon's Key.

### 8 — Room-graph / item-gated open world (THE research lift)
~25-40 titles.
Metroid, Castlevania II, Blaster Master, Faxanadu, Battle of Olympus,
Goonies II, Zelda II, Legacy of the Wizard, Milon's Secret Castle,
Wizards & Warriors, Solstice, Deadly Towers, Clash at Demonhead,
Rad Gravity, Air Fortress.

### 9 — Interaction-discovery top-down (win predicate wired)
~15-25 titles.
The Legend of Zelda, Crystalis, StarTropics, Willow, Guardian Legend
(hybrid with 5), Battle of Olympus (hybrid with 8), Deja Vu,
Shadowgate, Uninvited (the last three are adventure/parser hybrids and
arguably want their own class).

### 10 — Menu / economy / text (OUT OF SCOPE v1)
28 titles measured.
Dragon Warrior I-IV, Final Fantasy, Ultima III-V, Bard's Tale,
AD&D Pool of Radiance / Hillsfar / Heroes of the Lance, Might & Magic,
Wizardry, Destiny of an Emperor, Nobunaga's Ambition, Genghis Khan,
Romance of the Three Kingdoms, Sweet Home, Faria, Deep Dungeon,
Swords and Serpents.

## Consequences for the League's stratified-random sample

The League samples "stratified-randomly from the 793-booting library."
Today the strata do not exist as data — the frozen six-game roster was
hand-picked. Two options, both honest:

1. **Scope the draw**: sample only from the ~591 in-scope titles and say
   so in the overlay ("mechanism classes 1-9; sports, racing, light-gun,
   board and RPG titles are out of scope v1").
2. **Extend the basis**: add classes for score-comparison sports,
   lap-count racing, and quiz/board — each needs a clear predicate
   before it needs a solver, which is a detector-workstream question.

Either way, the per-ROM class label needs to become data (a column in a
manifest, not prose) before "stratified-random" means anything
verifiable. That labeling pass is mechanical for the out-of-scope
buckets (keyword + mapper + a progress-byte probe) and judgment for the
1/3/5 boundaries, which blur in hybrids like Ninja Gaiden and Rygar.

## Honesty notes

- Out-of-scope counts are exact for the keyword sets listed in the
  generating commands; in-scope per-class counts are ESTIMATES from this
  pass, not a per-title labeling. Nothing here is a receipt.
- Hybrids are real and common: Rygar is 3+8, Guardian Legend is 5+9,
  Zelda II is 3+8+9, Battle of Olympus is 3+8. A single-label scheme
  will misfile them; the manifest should allow a primary and secondary
  class.
- No claim about tractability is made here. Class membership says which
  machinery *applies*, not that the game falls.
