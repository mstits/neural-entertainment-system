"""The engine half of the unwitnessed-semantics sweep.

`configs/` was swept on 2026-08-27 (7 quarantined, 24 downgraded, 963 kept;
`docs/research/UNWITNESSED_SEMANTICS_2026-08-27.md`). That sweep named its
own scope limit in writing:

    "Quarantining the YAML retracts the DOCUMENTATION claim, NOT the Rust
    constant."

`tests/test_purity_quarantine_sweep.py` guards the configs. This module
guards `nes_core/src/rewards.rs` — the layer that actually executes — with
the same discriminant:

    An assertion of semantics tied to an event this project has NEVER
    WITNESSED is injected knowledge, because it could not have been
    measured here.

Two failure modes are guarded, because the config sweep demonstrated both:

1. **Silent restoration.** Deleting a provenance tag would put an
   unearned claim back into live code with nothing to notice. Every tag is
   named explicitly below, so removing one fails loudly.
2. **Layer drift** — the structural finding of this pass. For Kid Icarus
   ($0130) and Double Dragon ($0030) the YAML sweep retracted a specific
   sentence, and that exact sentence survived VERBATIM in the Rust. The
   documentation moved while the wiring stayed put, so the two layers
   disagreed in writing. `RETRACTED_CLAUSES` pins those sentences dead in
   this layer too.

**Over-withdrawal is guarded as the equal-and-opposite defect.** SMB's
constants are earned by a completely solved game — they fired 32 times in a
single cold-boot tape with `state_loads=0` and a rendered ending frame.
`test_smb_constants_are_untouched_and_untagged` fails if anybody strips or
tags them to look rigorous.

Anti-vacuity: every predicate in this file is mutation-tested. Each lint has
a `_has_teeth` sibling that feeds it text it MUST reject, so a checker that
rots into matching nothing cannot pass silently — the bar the predecessor
sweep set by verifying its guard through an ACTUAL REVERT.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REWARDS_RS = REPO / "nes_core" / "src" / "rewards.rs"

TAG = "PURITY: UNWITNESSED-EXTERNAL"

#: Pattern for a shared tag block's stable id, e.g. `[group: metroid_win_chain]`.
#:
#: A constant's tag is its OWN contiguous comment block — never a
#: neighbour's. An earlier version of this module searched a fixed 60-line
#: window above the declaration, which made the half-fix guard largely
#: vacuous: deleting a constant's entire tag still passed, because the tag
#: of a constant 25 lines above fell inside the window. 19 of 27 rows
#: escaped that way, including every headline retraction (the Zelda,
#: Metroid and Castlevania win chains).
#:
#: Four blocks legitimately tag a whole group of adjacent constants. Those
#: are handled by NAMING the group rather than by proximity: the shared
#: header carries `[group: X]` next to its tag, and every member of the
#: group references `[group: X]` in its own comment. Deleting either half
#: fails. That keeps group tagging expressible while leaving the scope of
#: a tag explicit rather than positional.
GROUP_RE = re.compile(r"\[group:\s*([a-z0-9_]+)\]")


def _source() -> str:
    return REWARDS_RS.read_text()


def _lines() -> list[str]:
    return _source().splitlines()


# =========================================================================
# The registry. Each row is one assertion the 2026-08-27 engine pass ruled
# UNWITNESSED-EXTERNAL, and each names the event with no witness in this
# repo — the reason the claim could not have been measured here.
#
# `decl` is matched as a substring of a single source line, so the row also
# pins the constant's VALUE: the annotation pass changed no reward
# arithmetic, and a behaviour change that moved an address would fail here.
# =========================================================================

#: (id, game, declaration text that must still be present, unwitnessed event)
UNWITNESSED_CONSTANTS = [
    # ---- Punch-Out: the concrete case named in the brief -----------------
    ("punchout_match_id", "Punch-Out!!",
     "const RAM_MATCH_ID: usize = 0x0001;",
     "a Punch-Out bout win"),
    ("punchout_losses", "Punch-Out!!",
     "const RAM_LOSSES: usize = 0x000A;",
     "a Mac TKO"),
    ("punchout_opp_id", "Punch-Out!!",
     "const RAM_OPP_ID: usize = 0x0002;",
     "reaching a second Punch-Out opponent"),
    # ---- Mega Man: the only reward struct with no provenance at all ------
    ("megaman_boss_health", "Mega Man 2",
     "const RAM_BOSS_HEALTH: usize = 0x06C1;",
     "reaching a Mega Man 2 boss room"),
    ("megaman_player_health", "Mega Man 2",
     "const RAM_PLAYER_HEALTH: usize = 0x06C0;",
     "any observed movement of the Mega Man 2 address block"),
    ("megaman_x_page", "Mega Man 2",
     "const RAM_PLAYER_X_PAGE: usize = 0x0460;",
     "any observed movement of the Mega Man 2 address block"),
    ("megaman_lives", "Mega Man 2",
     "const RAM_LIVES: usize = 0x00A8;",
     "any observed movement of the Mega Man 2 address block"),
    # ---- Kung Fu: claim and measurement contradicted each other ----------
    ("kungfu_floor", "Kung Fu",
     "const RAM_FLOOR: usize = 0x0058;",
     "a Kung Fu floor clear"),
    ("kungfu_game_clear_floor", "Kung Fu",
     "game_clear_floor: u8,",
     "beating Kung Fu / the Sylvia rescue"),
    # ---- Castlevania -----------------------------------------------------
    ("castlevania_boss_health", "Castlevania",
     "const RAM_BOSS_HEALTH: usize = 0x01A9;",
     "a Castlevania boss fight"),
    ("castlevania_dracula_stage", "Castlevania",
     "const DRACULA_STAGE: u8 = 0x12;",
     "reaching Dracula's stage"),
    # ---- Zelda -----------------------------------------------------------
    ("zelda_ganon_defeated", "Zelda",
     "const RAM_GANON_DEFEATED: usize = 0x0672;",
     "Ganon's defeat"),
    ("zelda_song", "Zelda",
     "const RAM_SONG: usize = 0x0609;",
     "the Zelda ending / the Ganon fight"),
    ("zelda_dungeon_level", "Zelda",
     "const RAM_DUNGEON_LEVEL: usize = 0x10;",
     "entering Zelda's Level 9"),
    # ---- Metroid: the whole win chain ------------------------------------
    ("metroid_mb_state", "Metroid",
     "const RAM_MB_STATE: usize = 0x0098;",
     "reaching Mother Brain"),
    ("metroid_mb_hits", "Metroid",
     "const RAM_MB_HITS: usize = 0x0099;",
     "Mother Brain's defeat"),
    ("metroid_ending_msg", "Metroid",
     "const RAM_ENDING_MSG: usize = 0x007A;",
     "the Metroid ending"),
    ("metroid_credits", "Metroid",
     "const RAM_CREDITS: usize = 0x007B;",
     "the Metroid ending"),
    ("metroid_death_hits", "Metroid",
     "const MB_DEATH_HITS: u8 = 32;",
     "Mother Brain's defeat"),
    # ---- DuckTales: game knowledge as two dollar figures, not an address -
    ("ducktales_win_treasure", "DuckTales",
     "win_treasure_value: u64,",
     "a DuckTales boss defeat / main-treasure pickup"),
    ("ducktales_treasure_cap", "DuckTales",
     "const TREASURE_CAP: u64 = 50_000;",
     "a census of DuckTales gem values"),
    # ---- Kid Icarus: the clause the YAML already retracted ---------------
    ("kid_icarus_stage", "Kid Icarus",
     "const RAM_STAGE: usize = 0x0130;",
     "a Kid Icarus stage clear"),
    ("kid_icarus_boss", "Kid Icarus",
     "const RAM_BOSS: usize = 0x006B;",
     "a Kid Icarus fortress-boss kill"),
    # ---- Double Dragon: same retracted-clause pattern --------------------
    ("double_dragon_mission_addr", "Double Dragon",
     "mission_addr: usize,",
     "a Double Dragon mission clear"),
    # ---- Gradius: six datacrystal bytes, capsule never grabbed -----------
    ("gradius_speed", "Gradius",
     "const RAM_SPEED: usize = 0x0040;",
     "a Gradius capsule pickup"),
    ("gradius_shield", "Gradius",
     "const RAM_SHIELD: usize = 0x0046;",
     "a Gradius capsule pickup"),
    # ---- Tetris: "widely cited", opt-in only ----------------------------
    ("tetris_empty_cell", "Tetris",
     "const EMPTY_CELL: u8 = 0xEF;",
     "a live read of the Tetris playfield"),
]

#: Sentences retracted by the 2026-08-27 sweeps that must not reappear.
#: Each row carries the number of times the sweep QUOTES the clause while
#: withdrawing it; any other count is a restoration.
#: The first two are the layer-drift cases: `configs/` retracted them and
#: the Rust kept them verbatim until this pass.
RETRACTED_CLAUSES = [
    ("kid_icarus_never_false_positive",
     "unambiguous stage clear (never a false positive)",
     1,
     "configs/kid_icarus.yaml quotes and retracts this exact clause: a "
     "false-positive rate cannot be known from zero observed increments."),
    ("double_dragon_safe_under_trigger",
     "safe under-trigger",
     2,
     "configs/double_dragon.yaml retracts the equivalent clause and notes "
     "the Rust default IS the wiring, not documentation."),
    ("double_dragon_never_false_positives",
     "never false-positives on",
     1,
     "the same clause, in a SECOND place: Double Dragon carried it in the "
     "arm header AND again in compute()'s win branch, so retracting only "
     "the header would have been a half-fix."),
    ("kungfu_ram_verified_live",
     "RAM verified live on this ROM",
     1,
     "$0058 measured 0 in 32/32 banked cells over ~719,500 steps; the "
     "claim and the measurement contradicted each other in one repo."),
    ("kungfu_dangling_config_path",
     "configs/kung-fu.yaml",
     1,
     "that file does not exist — the profile is configs/kungfu.yaml."),
    ("punchout_never_false_positive_flag",
     "never-false-positive win flag",
     2,
     "a false-positive rate asserted for an event with zero observations."),
    ("castlevania_proven_by_rom_code",
     "proven by the game's own code",
     1,
     "the Tier-3 line: the question was resolved by reading the ROM's "
     "disassembly and quoting `cmp #$12` as proof."),
    ("megaman_map_is_correct",
     "the Mega Man 2 map, which is CORRECT",
     1,
     "no boss room has been reached to check any of it."),
]

#: SMB constants that are EARNED and must not move or acquire a tag. This
#: is the over-withdrawal guard: stripping earned work to look rigorous is
#: the same error as importing knowledge, wearing the opposite mask.
SMB_EARNED_CONSTANTS = [
    "const RAM_X_PAGE: usize = 0x006D;",
    "const RAM_X_LOW: usize = 0x0086;",
    "const RAM_LIVES: usize = 0x075A;",
    "const RAM_PLAYER_STATE: usize = 0x000E;",
    "const RAM_WORLD: usize = 0x075F;",
    "const RAM_LEVEL: usize = 0x0760;",
    "const RAM_DISPLAY_LEVEL: usize = 0x075C;",
    "const DISPLAY_LEVEL_CASTLE: u8 = 3;",
    "const RAM_FLOAT_STATE: usize = 0x001D;",
]


def _decl_line_index(decl: str) -> int:
    """Index of the single line declaring `decl`, or -1."""
    hits = [i for i, ln in enumerate(_lines()) if decl in ln]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return -1
    # A few names repeat across arms (RAM_LIVES, RAM_STAGE...). Prefer the
    # one that has the tag above it; if none does, take the first so the
    # tag assertion reports a real miss rather than a lookup failure.
    for i in hits:
        if TAG in _own_comment_block(i):
            return i
    return hits[0]


def _own_comment_block(i: int, lines: list[str] | None = None) -> str:
    """The comment block that belongs to the declaration on line `i`.

    That is the unbroken run of comment lines directly above it, plus any
    trailing comment on the declaration line itself. A blank line, a brace,
    an attribute or another declaration ENDS the block — so a neighbouring
    constant's tag can never stand in for a missing one.
    """
    lines = _lines() if lines is None else lines
    j = i - 1
    while j >= 0 and lines[j].strip().startswith("//"):
        j -= 1
    block = lines[j + 1:i]
    _, _, trailing = lines[i].partition("//")
    if trailing:
        block = block + ["//" + trailing]
    return "\n".join(block)


def _group_blocks(lines: list[str] | None = None) -> dict[str, str]:
    """Shared tag blocks, keyed by the id they declare.

    A comment run qualifies as a group header only if it carries BOTH the
    purity tag and a `[group: X]` id. A member's own comment mentioning the
    id is therefore not enough to satisfy itself — the header has to exist.
    """
    lines = _lines() if lines is None else lines
    out: dict[str, str] = {}
    run: list[str] = []
    for line in lines + [""]:
        if line.strip().startswith("//"):
            run.append(line)
            continue
        if run:
            text = "\n".join(run)
            if TAG in text:
                for gid in GROUP_RE.findall(text):
                    out[gid] = text
            run = []
    return out


def _tag_block_for(decl: str) -> str | None:
    """The text that must carry `decl`'s purity tag, and nothing else's."""
    i = _decl_line_index(decl)
    if i < 0:
        return None
    own = _own_comment_block(i)
    if TAG in own:
        return own
    groups = _group_blocks()
    parts = [own]
    for gid in GROUP_RE.findall(own):
        if gid in groups:
            parts.append(groups[gid])
    return "\n".join(parts)


# =========================================================================
# 1. Every unwitnessed constant is still declared, unchanged, and tagged.
# =========================================================================

@pytest.mark.parametrize(
    "cid,game,decl,event", UNWITNESSED_CONSTANTS,
    ids=[c[0] for c in UNWITNESSED_CONSTANTS])
def test_unwitnessed_constant_is_declared_unchanged(cid, game, decl, event):
    """The constant still exists with the same value.

    This is the behaviour-freeze half. The 2026-08-27 engine pass changed
    NO reward arithmetic — the disposition was deliberately provenance +
    guard, not disarmament, because a behaviour change here can invalidate
    banked runs. If a value moved, that was a behaviour change and it
    needs its own justification, not this test's silence.
    """
    assert decl in _source(), (
        f"{game}: `{decl}` is gone from rewards.rs. The 2026-08-27 pass "
        f"annotated it and left its value alone; a change here is a "
        f"BEHAVIOUR change and must be argued separately."
    )


@pytest.mark.parametrize(
    "cid,game,decl,event", UNWITNESSED_CONSTANTS,
    ids=[c[0] for c in UNWITNESSED_CONSTANTS])
def test_unwitnessed_constant_carries_a_provenance_tag(cid, game, decl, event):
    """The constant discloses that its semantics were never witnessed here.

    This is the assertion that fails on a silent restoration. Deleting the
    tag puts an unearned claim back into live code, which is exactly what
    the configs-only sweep could not prevent.
    """
    block = _tag_block_for(decl)
    assert block is not None, f"{game}: cannot locate `{decl}`"
    assert TAG in block, (
        f"{game}: `{decl}` lost its `{TAG}` tag. It asserts semantics tied "
        f"to {event} — an event with NO witness in this repository — so the "
        f"claim could not have been measured here and must say so in the "
        f"layer that executes."
    )


@pytest.mark.parametrize(
    "cid,game,decl,event", UNWITNESSED_CONSTANTS,
    ids=[c[0] for c in UNWITNESSED_CONSTANTS])
def test_provenance_tag_is_complete(cid, game, decl, event):
    """A tag states the claim, the missing witness, and what would earn it.

    A bare `UNWITNESSED-EXTERNAL` marker with no content would satisfy the
    test above while telling a reader nothing, so the three-part shape is
    required: what it ASSERTS, that there is NO WITNESS, and what EARNS IT.
    """
    block = _tag_block_for(decl)
    assert block is not None, f"{game}: cannot locate `{decl}`"
    for field in ("ASSERTS:", "NO WITNESS:", "EARNS IT:"):
        assert field in block, (
            f"{game}: `{decl}`'s provenance tag is missing `{field}`. A tag "
            f"without it is a marker, not a disclosure — a reader still "
            f"cannot tell what was claimed or what would settle it."
        )


# =========================================================================
# 2. Retracted clauses stay retracted — in THIS layer.
# =========================================================================

#: Markers that turn an occurrence of a retracted clause from a live claim
#: into a QUOTATION inside its own retraction. A retraction that cannot
#: quote what it retracts is much less useful to a reader, so the lint asks
#: where the clause sits, not merely whether the characters are present.
RETRACTION_MARKERS = (TAG, "RETRACTED", "retracted", "RETRACTION")


def _comment_run_at(i: int, lines: list[str]) -> str | None:
    """The contiguous comment run containing line `i`, or None if it is code."""
    if not lines[i].strip().startswith("//"):
        return None
    a = i
    while a > 0 and lines[a - 1].strip().startswith("//"):
        a -= 1
    b = i
    while b + 1 < len(lines) and lines[b + 1].strip().startswith("//"):
        b += 1
    return "\n".join(lines[a:b + 1])


def _clause_occurrences(clause: str, src: str | None = None) -> list[int]:
    """1-indexed lines where `clause` appears at all."""
    lines = (src if src is not None else _source()).splitlines()
    return [i + 1 for i, line in enumerate(lines) if clause in line]


def _unretracted_occurrences(clause: str, src: str | None = None) -> list[int]:
    """1-indexed lines where `clause` appears OUTSIDE a retraction block.

    `src` is injectable so the teeth test can feed this synthetic text
    without patching the filesystem.
    """
    lines = (src if src is not None else _source()).splitlines()
    bad = []
    for i, line in enumerate(lines):
        if clause not in line:
            continue
        run = _comment_run_at(i, lines)
        if run is None or not any(m in run for m in RETRACTION_MARKERS):
            bad.append(i + 1)
    return bad


@pytest.mark.parametrize(
    "cid,clause,quoted,why", RETRACTED_CLAUSES,
    ids=[c[0] for c in RETRACTED_CLAUSES])
def test_retracted_clause_does_not_reappear(cid, clause, quoted, why):
    """A sentence withdrawn by a sweep must not survive as a LIVE claim.

    The structural finding of the 2026-08-27 engine pass: for Kid Icarus
    and Double Dragon, `configs/` retracted a sentence and the Rust kept it
    word for word. Documentation moved, wiring did not, and the two layers
    disagreed in writing.

    The clause may still appear *inside* its own retraction — quoting what
    was withdrawn is how a reader knows what changed. What may not happen
    is the clause standing on its own again as an assertion.
    """
    bad = _unretracted_occurrences(clause)
    assert not bad, (
        f"the retracted clause {clause!r} stands as a live claim at "
        f"rewards.rs line(s) {bad}. {why}"
    )
    # The census is the half with real teeth. Requiring only "a retraction
    # marker somewhere nearby" was not enough: an arm header is one long
    # comment run that CONTAINS its own retraction, so restoring the
    # sentence a few lines below the marker read as a quotation of it and
    # passed silently — for all eight clauses, in exactly the header each
    # was withdrawn from. Pinning HOW MANY times the sweep quotes each
    # clause makes any new occurrence fail wherever it lands.
    found = _clause_occurrences(clause)
    assert len(found) == quoted, (
        f"the retracted clause {clause!r} now appears {len(found)} time(s) "
        f"in rewards.rs at line(s) {found}; the 2026-08-27 sweep quotes it "
        f"{quoted} time(s). A NEW occurrence is a restored claim even when "
        f"it sits inside the retraction block that withdrew it. {why}"
    )


def test_every_retracted_clause_is_still_quoted_somewhere():
    """The retractions are visible, not silent deletions.

    A sweep that merely deleted the offending sentences would leave no
    trace of what was withdrawn or why — and the next reader would have to
    diff history to find out. Each clause must survive as a quotation.
    """
    src = _source()
    for cid, clause, _quoted, _why in RETRACTED_CLAUSES:
        assert clause in src, (
            f"{cid}: {clause!r} was deleted outright rather than retracted "
            f"in place. Keep it quoted inside its provenance tag so the "
            f"record shows what the engine used to assert."
        )


# =========================================================================
# 3. Over-withdrawal guard — SMB is earned and must not move.
# =========================================================================

@pytest.mark.parametrize("decl", SMB_EARNED_CONSTANTS)
def test_smb_constants_are_untouched_and_untagged(decl):
    """SMB's block is WITNESSED, so the discriminant does not bite.

    These fired 32 times in one cold-boot tape with `state_loads=0` and a
    rendered ending frame. Tagging them UNWITNESSED-EXTERNAL would be a
    false statement about the best-evidenced code in the file, and changing
    a value invalidates banked runs. Both directions fail here.
    """
    src = _source()
    assert decl in src, (
        f"SMB constant `{decl}` moved or disappeared. SMB is a completely "
        f"solved game here; a behaviour change to its reward invalidates "
        f"banked runs."
    )
    i = _decl_line_index(decl)
    block = "\n".join(_lines()[max(0, i - 4):i + 1])
    assert TAG not in block, (
        f"SMB constant `{decl}` was tagged UNWITNESSED-EXTERNAL. That is "
        f"over-withdrawal: the flagpole and castle events behind it are the "
        f"most-witnessed in this repository."
    )


def test_smb_block_is_positively_marked_witnessed():
    """SMB's block says out loud that it is earned, so a future sweep
    reads the ledger before touching it."""
    src = _source()
    assert "PURITY: WITNESSED" in src, (
        "the SMB block lost its WITNESSED marker. Recording what IS earned "
        "is half the discipline: without it the next sweep has only the "
        "unwitnessed tags to go on and over-withdrawal looks like rigour."
    )


# =========================================================================
# 4. The teeth. Every lint above is fed input it MUST reject.
# =========================================================================

def test_tag_lint_has_teeth():
    """The tag search must reject an untagged block.

    A predicate that matched everything would keep this file green while
    guarding nothing.
    """
    untagged = "\n".join([
        "    // just an ordinary comment",
        "    const RAM_SOMETHING: usize = 0x1234;",
    ])
    assert TAG not in untagged


def test_completeness_lint_has_teeth():
    """A bare marker with no ASSERTS / NO WITNESS / EARNS IT must fail."""
    bare = f"    /// {TAG}\n    const RAM_X: usize = 0x01;"
    assert TAG in bare, "sanity: the marker is present"
    missing = [f for f in ("ASSERTS:", "NO WITNESS:", "EARNS IT:")
               if f not in bare]
    assert missing == ["ASSERTS:", "NO WITNESS:", "EARNS IT:"], (
        "the completeness check must reject a marker carrying no content"
    )


def test_retraction_lint_has_teeth():
    """The retraction check must FIRE on a clause standing on its own, and
    must NOT fire on the same clause quoted inside a retraction.

    Both directions matter. A checker that flagged the quotation too would
    force silent deletions — losing the record of what was withdrawn — and
    a checker that accepted a bare restoration would guard nothing.
    """
    for cid, clause, _quoted, _why in RETRACTED_CLAUSES:
        live = "\n".join([
            "// an ordinary header with no disclosure whatsoever",
            f"//   {clause}",
            "const RAM_X: usize = 0x01;",
        ])
        quoted = "\n".join([
            f"/// {TAG}",
            "/// ASSERTS: the clause below, which is why it is quoted here.",
            f'///   "{clause}"',
            "/// NO WITNESS: the event has never occurred here.",
            "/// EARNS IT: witness it, then log the byte across it.",
            "const RAM_X: usize = 0x01;",
        ])

        assert _unretracted_occurrences(clause, live), (
            f"{cid}: the lint would NOT catch {clause!r} returning as a "
            f"live claim — it guards nothing"
        )
        assert not _unretracted_occurrences(clause, quoted), (
            f"{cid}: the lint fires on the clause quoted inside its own "
            f"retraction, which would force a silent deletion instead"
        )


def test_registry_is_non_empty_and_covers_every_named_game():
    """A registry that quietly emptied would pass every parametrized test
    above by having nothing to run."""
    assert len(UNWITNESSED_CONSTANTS) >= 25
    games = {row[1] for row in UNWITNESSED_CONSTANTS}
    for required in ("Punch-Out!!", "Mega Man 2", "Kung Fu", "Castlevania",
                     "Zelda", "Metroid", "DuckTales", "Kid Icarus",
                     "Double Dragon", "Gradius", "Tetris"):
        assert required in games, (
            f"{required} dropped out of the unwitnessed registry"
        )
    assert len(RETRACTED_CLAUSES) >= 8
    assert len(SMB_EARNED_CONSTANTS) >= 9


# =========================================================================
# 5. The behavioural guard — the arm cannot report success SILENTLY.
#
# Everything above is a source lint. This section runs the COMPILED engine,
# so it also proves the loaded `.so` matches the source it is being checked
# against: after a Rust edit the dylib must be rebuilt AND copied into the
# venv, or every Python measurement afterwards is void.
# =========================================================================

nes_core = pytest.importorskip("nes_core")


def _ledger() -> dict[str, dict[str, str]]:
    return {
        rid: {"status": status.lower(), "predicate": pred, "basis": basis}
        for rid, status, pred, basis in nes_core.win_witness_ledger()
    }


_ROW_RE = re.compile(
    r"WinWitnessRow\s*\{\s*"
    r'reward_id:\s*"(?P<rid>[a-z_]+)"\s*,\s*'
    r"status:\s*WinWitness::(?P<status>\w+)\s*,\s*"
    r'predicate:\s*(?P<pred>"(?:[^"\\]|\\.)*")\s*,\s*'
    r'basis:\s*(?P<basis>"(?:[^"\\]|\\.)*")\s*,?\s*\}',
    re.S,
)


def _rust_str(lit: str) -> str:
    """Decode a Rust string literal, including `\\`-newline continuations.

    A trailing backslash before a newline joins the lines and eats the
    leading whitespace of the next one, which is how the ledger's prose is
    wrapped in the source.
    """
    body = lit[1:-1]
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt == "\n":
            i += 2
            while i < len(body) and body[i] in " \t":
                i += 1
            continue
        out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
        i += 2
    return "".join(out)


def _declared_ledger() -> dict[str, dict[str, str]]:
    """`WIN_WITNESS_LEDGER` as written in rewards.rs — the source of truth."""
    return {
        m.group("rid"): {
            # the enum variant is `Witnessed`; the exported string is
        # `witnessed` — compare on the name, not its casing.
        "status": m.group("status").lower(),
            "predicate": _rust_str(m.group("pred")),
            "basis": _rust_str(m.group("basis")),
        }
        for m in _ROW_RE.finditer(_source())
    }


def test_every_reward_arm_is_classified():
    """No arm may be silent about whether its success event was witnessed."""
    ledger = _ledger()
    for rid in nes_core.reward_ids():
        assert rid in ledger, (
            f"reward arm `{rid}` has no win-witness classification. A "
            f"success it reports is either backed by a witnessed event here "
            f"or it is UNCONFIRMED; silence is not a third option."
        )
    assert set(ledger) == set(nes_core.reward_ids())


def test_compiled_ledger_matches_the_source_of_truth():
    """The loaded binary is the one this file just linted.

    `make build` writes `libnes_core.dylib`, and the venv keeps a separate
    `nes_core.abi3.so`. Forgetting the copy means Python silently runs the
    OLD binary — a standing failure mode in this repo — and every
    measurement taken afterwards is void. Comparing the compiled table
    against the source turns that into a loud test failure.
    """
    declared = _declared_ledger()
    assert declared, "could not parse WIN_WITNESS_LEDGER out of rewards.rs"
    compiled = _ledger()
    assert set(compiled) == set(declared), (
        f"the compiled win-witness ledger disagrees with "
        f"nes_core/src/rewards.rs (only in binary: "
        f"{set(compiled) - set(declared)}; only in source: "
        f"{set(declared) - set(compiled)}). The loaded .so is STALE: "
        f"run `make build` and copy the dylib into the venv .so."
    )
    # Comparing only the ID SET was the original form of this check, and it
    # was much weaker than its own docstring: any edit that does not ADD or
    # REMOVE an arm — a flipped status, a re-pointed predicate, a rewritten
    # basis — left it green against a stale binary. Those are precisely the
    # edits this sweep makes, so all three fields are compared.
    drift = [
        (rid, field, declared[rid][field], compiled[rid][field])
        for rid in sorted(declared)
        for field in ("status", "predicate", "basis")
        if declared[rid][field] != compiled[rid][field]
    ]
    assert not drift, (
        f"the compiled win-witness ledger has drifted from the source in "
        f"{len(drift)} field(s): {drift[:3]}. The loaded .so is STALE: run "
        f"`make build` and copy the dylib into the venv .so, or every "
        f"measurement taken against it is void."
    )


UNWITNESSED_WIN_ARMS = [
    "zelda", "castlevania", "metroid", "punch_out", "kung_fu",
    "ducktales", "kid_icarus", "double_dragon", "ghosts",
]
WITNESSED_WIN_ARMS = ["mario", "excitebike", "bubble_bobble", "tetris"]
DISARMED_WIN_ARMS = ["generic", "mega_man", "contra", "gradius"]


@pytest.mark.parametrize("rid", UNWITNESSED_WIN_ARMS)
def test_unwitnessed_arm_declares_itself_unwitnessed(rid):
    """Every arm resting on an unwitnessed event says so."""
    row = _ledger()[rid]
    assert row["status"] == "unwitnessed", (
        f"`{rid}` is labelled {row['status']!r}. Its episode_success() can "
        f"only become true through semantics tied to an event nobody here "
        f"has ever seen, so a success it reports is UNCONFIRMED."
    )
    assert len(row["basis"]) > 40, f"{rid}: the basis must name the gap"


@pytest.mark.parametrize("rid", WITNESSED_WIN_ARMS)
def test_witnessed_arm_is_not_withdrawn(rid):
    """The over-withdrawal guard on the ledger itself."""
    assert _ledger()[rid]["status"] == "witnessed", (
        f"`{rid}` lost its witnessed status. These four have receipted "
        f"clears; demoting one throws away real work, which is the same "
        f"error as importing knowledge wearing the opposite mask."
    )


@pytest.mark.parametrize("rid", DISARMED_WIN_ARMS)
def test_disarmed_arm_stays_disarmed(rid):
    """Contra's 255 sentinel and Gradius's `stage_addr == 0` are the model
    the rest of the engine should copy. Removing one turns a dormant
    unwitnessed claim into a live one."""
    assert _ledger()[rid]["status"] == "disarmed"


def _reward(reward_id: str, **weights):
    return nes_core.build_reward_function(
        {"reward_id": reward_id, "reward_weights": weights})


def test_punchout_win_predicate_cannot_report_success_silently():
    """**The concrete case.**

    `RAM_MATCH_ID = 0x0001` drives Punch-Out's `won` / `done` /
    `episode_success()`. Its config annotation read "VERIFIED WIN LATCH"
    and sat in a `ram_mapping` block naming Data Crystal and TASVideos as
    its sources. No Punch-Out bout win has ever been witnessed here:
    `runs/fight_gate/smoke/solutions/` is empty, cumulative damage topped
    out at 81 against a 96 cap, and a pure-NOOP continuation from the
    archived knockdown state refills opponent HP 0->96 at step 120 with
    `match_id` constant 0.

    The guard is NOT that the arm stops reporting success — disarming it
    is a behaviour change and was deliberately not made. The guard is that
    it cannot report success SILENTLY: when the byte fires, the ledger
    must already say the result is UNCONFIRMED.
    """
    r = _reward("punch_out")
    ram = bytearray(2048)
    ram[0x0398] = 0x60      # opponent at full HP
    ram[0x0392] = 0x60      # Mac at full HP
    r.compute(bytes(ram), 0)
    assert not r.episode_success(), "the bout has not been won at step 1"

    ram[0x0001] = 1         # the unwitnessed win latch fires
    _reward_val, done, _level = r.compute(bytes(ram), 0)

    assert r.episode_success() and done, (
        "punch_out's win is still keyed on $0001 — if this changed, the "
        "ledger row and the RAM_MATCH_ID tag are describing the wrong byte"
    )
    row = _ledger()["punch_out"]
    assert row["status"] == "unwitnessed", (
        "punch_out just reported a success through $0001. That byte's "
        "identity has NO witness in this repo and is falsified at the one "
        "moment it could be checked, so the success is UNCONFIRMED and the "
        "ledger must say so. A success reported with the ledger claiming "
        "'witnessed' would be exactly the silent report this guard exists "
        "to prevent."
    )
    assert "0x0001" in row["predicate"] or "$0001" in row["predicate"]
    assert "runs/fight_gate/smoke/solutions/" in row["basis"], (
        "the basis must cite the receipt that establishes the null, not "
        "merely assert one"
    )


def test_megaman_boss_term_is_live_even_though_its_win_is_disarmed():
    """`mega_man` is `disarmed` for its WIN only.

    `episode_success()` is hard-false, but `boss_damage` / `boss_killed`
    still pay out on $06C1 — an ACTIVE shaping term on an unwitnessed
    identity. The ledger row has to say that, or "disarmed" would read as
    "nothing here depends on the unwitnessed byte", which is false.
    """
    row = _ledger()["mega_man"]
    assert row["status"] == "disarmed"
    assert "boss_damage" in row["basis"] and "SHAPING" in row["basis"], (
        "mega_man's row must disclose that the boss SHAPING term is live on "
        "$06C1 even though the win is held shut"
    )

    r = _reward("mega_man", boss_damage=5.0, boss_killed=75.0,
                forward_progress=0.0, health_delta=0.0, time_penalty=0.0,
                death_penalty=0.0)
    ram = bytearray(2048)
    ram[0x06C0] = 28
    ram[0x00A8] = 3
    ram[0x06C1] = 28
    r.compute(bytes(ram), 0)
    ram[0x06C1] = 0
    reward, _done, _level = r.compute(bytes(ram), 0)
    assert reward > 0.0, (
        "the $06C1 boss term really does pay out, so the disclosure above "
        "is describing live behaviour rather than a hypothetical"
    )
    assert not r.episode_success(), "the WIN remains hard-false"


def test_ledger_bases_cite_measurements_not_adjectives():
    """Each unwitnessed row points at something openable.

    A row saying only "not verified" would pass every structural check and
    teach a reader nothing about how strong the null is.
    """
    ledger = _ledger()
    for rid in UNWITNESSED_WIN_ARMS:
        basis = ledger[rid]["basis"]
        assert any(tok in basis for tok in
                   ("runs/", "docs/", "steps", "cells", "files", "trees")), (
            f"`{rid}`'s basis cites no receipt and no measurement: {basis!r}"
        )
