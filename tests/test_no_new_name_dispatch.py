"""Shrink-only inventory of display-name substring dispatch in production
Python.

BREACH PATH 1 (2026-08-26) was `if name.contains("zelda")` in the Rust
reward dispatch handing `configs/legend_of_zelda.yaml` — a 31-line,
address-free, lint-clean profile that declared no reward at all — the
quarantined disassembly-sourced win predicate, because its title contains
the word "Zelda". That door is now nailed shut on the Rust side: dispatch
takes an explicit `reward_id` and the display name is not a parameter of
the function any more, so the defect cannot be re-expressed there.

Python is a different story. Verification of that fix found two further
sites nobody had disclosed, doing the SAME thing with the SAME quarantined
bytes: `src/diagnostics/worker_debug.py` selected `ZELDA_LINK_X = 0x0070`
/ `ZELDA_LINK_Y = 0x0084` under `if "zelda" in self.game`, and
`scripts/diagnose.py` read the identical pair under
`"zelda" in profile["name"].lower()`. Both were quarantined addresses
arriving by inference from a title. `src/audio/ram_music.py` carried a
whole substring->address table of the same shape. All three are fixed.

The lesson is that a fixed instance is not a fixed class: these were found
by hand, twice, after two rounds of review said the breach was closed.
This test makes the remaining sites visible and countable so the next one
cannot arrive silently.

THE RULE IS SHRINK-ONLY. `ALLOWED` may lose entries; it may never gain
one. An entry that no longer matches its file is a hard failure, not a
tidy-up, so fixing a site forces the inventory down and a stale list
cannot quietly certify an empty tree. This is deliberately NOT a lint you
satisfy by adding a line.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The historical display-name tokens. FROZEN — documented never to grow.
#: New reward arms are selected by `reward_id`, so a new game does not need
#: a token here; adding one would be re-opening the door this file guards.
TOKENS = (r"mario|zelda|contra|mega ?man|castlevania|metroid|tetris|"
          r"bubble ?bobble|punch-? ?out|kung ?fu|gradius|excitebike|ghosts|"
          r"ducktales|kid ?icarus|double ?dragon|smb")

#: Matches the dangerous shape specifically: a quoted game token used as a
#: SUBSTRING test (`"zelda" in name`). An exact comparison against an
#: explicit slug a program owns (`g["name"] == "mario"` over a catalog
#: entry, `reward_id == "mario"`) is the fix this project moved to, so
#: flagging it would push people back toward inference.
PATTERN = re.compile(r"""["'](?:""" + TOKENS + r""")["']\s*in\s*""",
                     re.IGNORECASE)

SCANNED_DIRS = ("src", "scripts")

#: Every surviving site, with why it is still here. Each value is the
#: reason; the key is (repo-relative path, 1-based line number is NOT used
#: — line numbers churn — the matched source line, stripped).
ALLOWED: dict[tuple[str, str], str] = {
    # ---- SMB training behaviour. Disclosed debt from the reward_id
    # migration. `reward_id == "mario"` is the drop-in replacement, but
    # trainer.py's `"mario" in name or "smb" in name` is a strictly WIDER
    # set than the reward arm (it includes configs/smb_4_4_micro.yaml),
    # so swapping them changes training behaviour and needs its own
    # before/after on SMB. None of these selects a RAM address.
    ("src/training/trainer.py",
     '"mario" in str(game_profile.get("name", "")).lower()'):
        "selects the SMB tile-observation path",
    ("src/training/trainer.py",
     'self._is_smb_profile: bool = "mario" in profile_name or "smb" in profile_name'):
        "selects SMB curriculum/go-explore behaviour; wider set than the reward arm",
    ("src/training/trainer.py",
     'if "mario" in str(self.game_profile.get("name", "")).lower():'):
        "selects SMB checkpoint/curriculum handling",
    ("src/training/exploration_controller.py",
     '_is_mario = "mario" in str(t.game_profile.get("name", "")).lower()'):
        "selects max_x go-explore scoring for SMB",
    ("src/gui/main_window.py",
     'is_smb = "mario" in str(prof.get("name", "")).lower()'):
        "GUI-only: shows SMB-specific controls",

    # ---- Offline tooling keyed on FILE PATHS, not on RAM. These pick
    # which run directories or action-token sets to load; no address, no
    # reward, no clear verdict depends on them.
    ("scripts/clear_detect.py",
     'if "mario" in low or "smb" in low or "1-1" in low or "8-1" in low:'):
        "filters banked run dirs by root_state path convention",
    ("scripts/macro_mine.py", '"castlevania" in root_state'):
        "picks an action-token vocabulary from the run's path",
    ("scripts/macro_mine.py", 'or "castlevania" in prof_str'):
        "same site, second clause",
}

#: `scripts/migrate_reward_id.py` is the migration tool that deliberately
#: REPLAYS the deleted substring predicate to freeze what each config
#: resolved to on migration day. It is a historical record of the defect,
#: which is the one legitimate reason to contain it.
EXEMPT_FILES = {"scripts/migrate_reward_id.py"}


def _hits() -> dict[tuple[str, str], int]:
    found: dict[tuple[str, str], int] = {}
    for d in SCANNED_DIRS:
        for path in sorted((REPO / d).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if rel in EXEMPT_FILES:
                continue
            for line in path.read_text(errors="replace").splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or not PATTERN.search(stripped):
                    continue
                found[(rel, stripped)] = found.get((rel, stripped), 0) + 1
    return found


def test_no_new_display_name_dispatch_site_appeared() -> None:
    new = sorted(k for k in _hits() if k not in ALLOWED)
    assert not new, (
        "new display-name substring dispatch:\n  "
        + "\n  ".join(f"{p}: {src}" for p, src in new)
        + "\n\nA display name is not a declaration. Select behaviour from "
          "an explicit profile key (`reward_id`, or a new declared field) "
          "instead — see the module docstring for what this cost the "
          "purity line the last time."
    )


def test_the_inventory_only_shrinks() -> None:
    """A stale entry must fail loudly. Otherwise fixing a site leaves a
    line behind that keeps the count looking the same, and eventually the
    whole list means nothing."""
    hits = _hits()
    stale = sorted(k for k in ALLOWED if k not in hits)
    assert not stale, (
        "these ALLOWED entries no longer match anything — delete them "
        "(the inventory is shrink-only, and a stale entry hides how much "
        f"is really left):\n  " + "\n  ".join(f"{p}: {s}" for p, s in stale))


def test_the_scanner_can_actually_find_a_site() -> None:
    """Guard the guard. A regex that stopped matching would make both
    tests above pass on any tree at all, including one that had just
    re-added `if "zelda" in name: return ZeldaReward`."""
    assert PATTERN.search('if "zelda" in profile["name"].lower():')
    assert PATTERN.search('if "mega man" in n:')
    assert PATTERN.search('"metroid" in self.game')
    assert not PATTERN.search("reward_id == 'zelda'"), (
        "an explicit reward_id comparison is the FIX, not the defect — "
        "flagging it would push people back toward the name")
    assert _hits(), (
        "the scanner found nothing at all in src/ or scripts/; the "
        "inventory below would then be vacuous")


#: Files allowed to carry the quarantined 0x0070/0x0084 coordinate pair
#: with NO name-token conjunction (see the test below for why the
#: conjunction was dropped entirely). Each entry states why the file is
#: legitimate: this is deliberately a SEPARATE list from EXEMPT_FILES
#: above — that one exempts the SUBSTRING-DISPATCH scan (selecting
#: behaviour by matching a name), a different defect shape than these
#: two, which never dispatch on a name at all.
QUARANTINED_PAIR_EXEMPT: dict[str, str] = {
    "src/gui/play_window.py": (
        "generic GUI<->headless RAM trace dumped for EVERY loaded ROM "
        "regardless of game (mixes Zelda's 0x70/0x84 with SMB's "
        "0x86/0xCE/0x1D/0x770/0x772 in one fixed dump) — fidelity/parity "
        "debugging output, never read by a reward or win-predicate path"
    ),
    "scripts/tracing/nes_core_nmi_trace.py": (
        "single-purpose NMI-timing diagnostic hardcoded to one Zelda ROM "
        "path; a fidelity tool comparing PC/RAM traces against nes-py, "
        "never wired into training or a win predicate"
    ),
    "scripts/zelda_cave_entry_repro.py": (
        "cycle-accuracy regression guard for the cave-entry transition "
        "timing fix (advance_one_frame cycle-lock) — compares nes_core "
        "vs nes-py RAM at one frame to confirm emulator timing, not game "
        "outcome; never wired into training or a win predicate"
    ),
}


def test_the_quarantined_zelda_pair_is_gone_from_production_python() -> None:
    """The specific regression: 0x0070 and 0x0084 (q_link_x / q_link_y)
    were read under a "zelda"-in-name test in two diagnostics files.

    A THIRD and FOURTH file were found 2026-08-27 carrying the exact same
    pair while evading the original fix's guard, which required a literal
    quoted `"zelda"` token alongside the addresses: both files identify
    the ROM only through its `.nes` filename ("Legend of Zelda, The (USA)
    (Rev A).nes"), never through the bare word the old regex demanded.
    Deleting a game-name comment or writing the ROM path instead of a
    literal string was enough to fall outside the guard's sight, even
    though the addresses stayed exactly as quarantined.

    The fix is structural, not another keyword to type correctly: the
    ADDRESS PAIR alone is now disqualifying, with no name-token
    conjunction to opt out of. Co-occurrence of both quarantined
    coordinates in production Python is specific enough on its own — the
    conjunction never bought precision, only a way to evade silently.
    Legitimate uses are named explicitly in QUARANTINED_PAIR_EXEMPT
    above, with a reason, so an exemption is visible and auditable
    instead of just absent from a regex's field of view.
    """
    offenders = []
    for d in SCANNED_DIRS:
        for path in sorted((REPO / d).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if rel in EXEMPT_FILES or rel in QUARANTINED_PAIR_EXEMPT:
                continue
            # Code only: this file's own explanatory comments name the
            # addresses, and a scanner that counted those would be unable
            # to describe the defect it guards.
            code = "\n".join(
                l for l in path.read_text(errors="replace").splitlines()
                if not l.strip().startswith("#"))
            has_addr = re.search(r"0x0*70\b", code) and re.search(r"0x0*84\b", code)
            if has_addr:
                offenders.append(rel)
    assert not offenders, (
        f"{offenders} carry the quarantined 0x0070/0x0084 coordinate pair "
        f"in production Python with no exemption on record — remove the "
        f"pair or add a reasoned QUARANTINED_PAIR_EXEMPT entry naming why "
        f"it's legitimate")


def test_quarantined_pair_exemptions_are_not_stale() -> None:
    """Mirrors `test_the_inventory_only_shrinks`'s discipline for the
    other allowlist in this file: an exemption for a file that no longer
    exists, or no longer actually carries the pair, is dead weight that
    would hide the exemption list quietly drifting from reality."""
    for rel, reason in QUARANTINED_PAIR_EXEMPT.items():
        path = REPO / rel
        assert path.exists(), f"exempted file {rel!r} no longer exists"
        assert reason.strip(), f"exempted file {rel!r} has no stated reason"
        code = "\n".join(
            l for l in path.read_text(errors="replace").splitlines()
            if not l.strip().startswith("#"))
        has_addr = re.search(r"0x0*70\b", code) and re.search(r"0x0*84\b", code)
        assert has_addr, (
            f"{rel} is exempted for carrying 0x0070/0x0084 but no longer "
            f"does — delete the stale exemption")
