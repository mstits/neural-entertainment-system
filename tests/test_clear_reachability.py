"""The guard that would have caught the `() > ()` census.

WHAT WENT WRONG, so the tests can be aimed at it. `GenericGame.is_clear`
opens with `if self.level_key(ram) > tuple(start_key)`. For the 40 profiles
shipping `level_key: []` that is `() > ()` — False for every RAM state that
can exist. Millions of emulator steps banked `solutions: 0`, and ten
documents cited that constant as though a search had asked a question and
got "no". Nothing in the repo ever checked whether the question COULD be
answered.

Two guards already in the tree failed the same way and are the reason these
tests are written the way they are: `adjudicate_soak._check_budget`'s
"banked N solution(s) — missed clear" branch, and the Zelda receipt's
"fabrication tripwire CLEAN". Both report PASS identically whether the
clear detector is sound or entirely absent. So every test below is written
against one question:

    WHAT WOULD THIS TEST REPORT IF THE MECHANISM WERE ABSENT?

`test_the_guard_is_not_a_constant` and the mutation tests answer it
directly: each takes a profile the guard accepts, breaks exactly one thing,
and requires the verdict to MOVE. A guard that returned REACHABLE for
everything, or NONE for everything, fails them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from clear_reachability import (  # noqa: E402
    DEGENERATE,
    LIVE_CONFLUENCE_SIGNALS,
    MAX_RAM_BYTE,
    NONE,
    NONE_SILENT,
    REACHABLE,
    UNFIREABLE,
    CoordConstantMissing,
    Reachability,
    clear_reachability,
    enforce,
    launch_banner,
)


# --------------------------------------------------------------------------
# Fixtures: minimal profiles, one knob at a time.
# --------------------------------------------------------------------------

def _profile(**solve) -> dict:
    """A solve block with the required keys, overridable one at a time."""
    base = {
        "rom": "roms/does-not-need-to-exist.nes",
        "progress": {"lo": 0x003E, "hi": 0x003F},
        "y": 0x0002,
        "level_key": [],
        "lives": 0x0032,
    }
    base.update(solve)
    return {"solve": base}


# --------------------------------------------------------------------------
# 1. The identity the census found.
# --------------------------------------------------------------------------

def test_the_empty_level_key_identity_is_reported_not_hidden() -> None:
    """`() > ()` is False. A profile resting on it alone has no reachable
    clear predicate, and the verdict has to SAY so — the whole defect was
    that this state was indistinguishable from "searched, found nothing"."""
    # The identity itself, stated in the test. Parenthesised deliberately:
    # `() > () is False` is a CHAINED comparison in Python and means
    # `(() > ()) and (() is False)`, which is False for reasons that have
    # nothing to do with the claim.
    assert (() > ()) is False
    r = clear_reachability(_profile())
    # 2026-08-28 (§5.1): a bare constant now classifies as the SILENT
    # class — announced-instead-of-inferred is enforced, not aspirational.
    assert r.verdict == NONE_SILENT
    assert r.can_bank_a_solution is False
    assert "constant" in r.reason


def test_a_live_level_key_can_advance() -> None:
    r = clear_reachability(_profile(level_key=[0x0028]))
    assert r.verdict == REACHABLE
    assert r.via == "level_key"
    assert r.can_bank_a_solution is True


def test_a_missing_level_key_is_treated_as_empty_not_crashed() -> None:
    """configs/legend_of_zelda.yaml and configs/tetris_usa.yaml ship a
    partial solve block with no level_key at all. The lint must classify
    them, not raise — a guard that dies on the malformed input is a guard
    that gets excluded from the sweep."""
    prof = _profile()
    del prof["solve"]["level_key"]
    assert clear_reachability(prof).verdict == NONE_SILENT


def test_a_profile_with_no_solve_block_is_not_a_solver_profile() -> None:
    assert clear_reachability({"action_space": [[]]}).verdict == NONE


# --------------------------------------------------------------------------
# 2. The confluence vote arithmetic — the Gradius case.
# --------------------------------------------------------------------------

def test_confluence_on_an_odometer_profile_is_refused() -> None:
    """The live Gradius defect. `coord` needs the progress readout to fall
    by >= COORD_RESET_DROP_MIN, but nes_core's odo_fold_frame re-anchors
    across a scene cut instead of integrating and _xram clamps backwards
    motion to 0, so the odometer never falls. With `tally` the only other
    live signal and min_signals defaulting to 2, the vote tops out at 1."""
    r = clear_reachability(_profile(
        progress={"source": "odometer", "axis": "x"},
        clear={"mode": "confluence"}))
    assert r.verdict == UNFIREABLE
    assert r.via == "confluence"
    assert "odo_fold_frame" in r.reason


def test_confluence_on_a_fight_gate_profile_is_refused() -> None:
    """Same shape, different monotone integral: fight_gate accumulates
    damage and never falls either."""
    r = clear_reachability(_profile(
        progress={"source": "fight_gate", "foe_hp": 0x0398},
        clear={"mode": "confluence"}))
    assert r.verdict == UNFIREABLE


def test_confluence_on_a_single_byte_progress_is_refused_on_arithmetic() -> None:
    """A single unsigned RAM byte tops out at 255, below the drop `coord`
    demands. No emulator needed to know this and no game knowledge either."""
    r = clear_reachability(_profile(
        progress={"lo": 0x0070}, clear={"mode": "confluence"}))
    assert r.verdict == UNFIREABLE
    assert str(MAX_RAM_BYTE) in r.reason


def test_confluence_on_a_sixteen_bit_pair_is_allowed_through() -> None:
    """THE OVER-CORRECTION GUARD. Contra ships confluence on a {lo, hi}
    pair, so `coord`'s drop is arithmetically possible. Whether `tally`
    has any referent in that game is a fact about the game, obtainable
    only by measuring it — asserting it from memory is the authored-
    semantics class CLAIMS.md Tier 3 forbids. So this profile passes, and
    the reason string says plainly that passing is not evidence it works."""
    r = clear_reachability(_profile(clear={"mode": "confluence"}))
    assert r.verdict == REACHABLE
    assert "not evidence that it works" in r.reason


def test_arming_the_apu_vote_no_longer_rescues_a_dead_coord() -> None:
    """CONTRACT CHANGE, 2026-08-26 — Rule 5, the required class.

    This used to pass: tally(1) + apu(1) reached min_signals=2 with coord
    dead, and the ceiling arithmetic called that REACHABLE. The escape
    hatch is now closed, because the shape it permits is the one that was
    measured firing. runs/clear_control_2026-08-26/bb_offline_r99.json:
    the detector crossed at frame 320 on {audio: 1, tally: 1, lock: 1,
    coord: 0} — three corroborators summing to exactly THRESHOLD with zero
    transition evidence — 1736 frames before the true clear at 2056.

    Corroborators agree with each other about a scene change none of them
    observed. At least one eligible signal has to be TRANSITION EVIDENCE
    ("a scene committed" / "the world did not come back"), and with the
    six shelf signals unwired, `coord` is the only one there is.

    No shipped profile is affected: contra and contra_blank are the only
    two declaring confluence and neither arms apu_weight."""
    r = clear_reachability(_profile(
        progress={"source": "odometer", "axis": "x"},
        clear={"mode": "confluence", "apu_weight": 1.0}))
    assert r.verdict == UNFIREABLE
    assert "TRANSITION EVIDENCE" in r.reason


def test_min_signals_above_the_ceiling_is_refused() -> None:
    """`min_signals: 3` with no apu_weight asks for a third vote that is
    never constructed — the streaming detector derives only tally and
    coord from a RAM snapshot."""
    r = clear_reachability(_profile(
        clear={"mode": "confluence", "min_signals": 3}))
    assert r.verdict == UNFIREABLE
    assert LIVE_CONFLUENCE_SIGNALS == 2


def test_min_signals_at_the_ceiling_is_allowed() -> None:
    r = clear_reachability(_profile(
        clear={"mode": "confluence", "min_signals": 3, "apu_weight": 1.0}))
    assert r.verdict == REACHABLE


# --------------------------------------------------------------------------
# 3. The other hooks.
# --------------------------------------------------------------------------

def test_byte_change_without_an_addr_is_refused() -> None:
    """`self._clear_addr is None` makes the branch return False for every
    state — a second way to ship an identity."""
    r = clear_reachability(_profile(clear={"mode": "byte_change"}))
    assert r.verdict == UNFIREABLE


def test_byte_change_with_an_addr_is_reachable() -> None:
    r = clear_reachability(_profile(
        clear={"mode": "byte_change", "addr": 0x0050, "direction": "down"}))
    assert r.verdict == REACHABLE


def test_a_zero_threshold_score_jump_is_refused_as_degenerate() -> None:
    """The opposite failure and the one that actually fabricates:
    `(v - prev) >= 0` is true almost every step. A predicate that is a
    constant TRUE is as broken as one that is a constant FALSE, and this
    repo has shipped both."""
    r = clear_reachability(_profile(clear={"mode": "score_jump"}))
    assert r.verdict == DEGENERATE
    assert r.ok is False


def test_a_real_score_jump_threshold_is_reachable() -> None:
    r = clear_reachability(_profile(
        clear={"mode": "score_jump", "threshold": 5000}))
    assert r.verdict == REACHABLE


def test_an_unknown_clear_mode_is_refused() -> None:
    """A typo'd mode falls through every branch of is_clear and returns
    False silently. Today that ships as a profile that searches forever."""
    r = clear_reachability(_profile(clear={"mode": "confluance"}))
    assert r.verdict == UNFIREABLE


def test_a_finale_with_a_mismatched_level_key_arity_is_refused() -> None:
    """`is_finale` compares `tuple(start_key) == tuple(f['level_key'])`.
    The entrance key has len(solve.level_key) entries, so a finale literal
    of a different length is never equal to it."""
    r = clear_reachability(_profile(
        level_key=[], finale={"addr": 0x000E, "value": 2, "level_key": [7, 3]}))
    assert r.verdict == UNFIREABLE
    assert r.via == "finale"


def test_a_matching_finale_is_reachable() -> None:
    """Excitebike's shape: both keys empty, so `() == ()` is True and the
    hook reduces to the addr/value test."""
    r = clear_reachability(_profile(
        level_key=[], finale={"addr": 0x000E, "value": 2, "level_key": []}))
    assert r.verdict == REACHABLE
    assert r.via == "finale"


def test_a_finale_missing_its_addr_is_refused() -> None:
    r = clear_reachability(_profile(level_key=[], finale={"level_key": []}))
    assert r.verdict == UNFIREABLE


# --------------------------------------------------------------------------
# 4. The vacuity check itself: would this test notice the mechanism going away?
# --------------------------------------------------------------------------

def test_the_guard_is_not_a_constant() -> None:
    """The property `adjudicate_soak._check_budget` and the Zelda
    "fabrication tripwire" both lacked: three inputs, three DIFFERENT
    answers. A guard that always says one thing cannot fail, and a guard
    that cannot fail certifies nothing."""
    verdicts = {
        clear_reachability(_profile()).verdict,
        clear_reachability(_profile(level_key=[0x0028])).verdict,
        clear_reachability(_profile(
            progress={"source": "odometer", "axis": "x"},
            clear={"mode": "confluence"})).verdict,
    }
    assert verdicts == {NONE_SILENT, REACHABLE, UNFIREABLE}


@pytest.mark.parametrize("mutation,expected", [
    # Each row: break exactly one thing about an accepted profile and
    # require the verdict to move off REACHABLE.
    (dict(level_key=[]), NONE_SILENT),
    (dict(level_key=[], clear={"mode": "confluence"},
          progress={"source": "odometer", "axis": "x"}), UNFIREABLE),
    (dict(level_key=[], clear={"mode": "byte_change"}), UNFIREABLE),
    (dict(level_key=[], clear={"mode": "score_jump", "threshold": 0}),
     DEGENERATE),
])
def test_breaking_one_thing_moves_the_verdict(mutation, expected) -> None:
    accepted = clear_reachability(_profile(level_key=[0x0028]))
    assert accepted.verdict == REACHABLE
    assert clear_reachability(_profile(**mutation)).verdict == expected


def test_the_coord_constant_is_read_from_the_detector_not_copied() -> None:
    """If somebody retunes COORD_RESET_DROP_MIN, this guard must move with
    it. Verified by reading the real constant out of the real module —
    when nes_core is importable, the AST-parsed value and the imported one
    must agree, which is the only way a copy could hide."""
    from clear_reachability import _coord_drop_min
    parsed = _coord_drop_min()
    try:
        from clear_detect import COORD_RESET_DROP_MIN
    except Exception:                      # no compiled nes_core in this env
        pytest.skip("clear_detect not importable (needs the nes_core build)")
    assert parsed == int(COORD_RESET_DROP_MIN)


def test_a_renamed_coord_constant_raises_instead_of_defaulting(
        tmp_path, monkeypatch) -> None:
    """The guard must not invent a plausible 300 when it loses sight of the
    real one. Substituting a default is how a check keeps reporting PASS
    after the thing it checks has gone."""
    import clear_reachability as cr
    fake = tmp_path / "scripts"
    fake.mkdir()
    (fake / "clear_detect.py").write_text("COORD_RESET_DROP_SOMETHING = 300\n")
    monkeypatch.setattr(cr, "REPO", tmp_path)
    with pytest.raises(CoordConstantMissing):
        cr._coord_drop_min()


# --------------------------------------------------------------------------
# 5. Enforcement and the launch banner.
# --------------------------------------------------------------------------

def test_enforce_refuses_an_unfireable_profile() -> None:
    prof = _profile(progress={"source": "odometer", "axis": "x"},
                    clear={"mode": "confluence"})
    with pytest.raises(SystemExit) as exc:
        enforce(prof, "configs/example.yaml")
    assert "REFUSED" in str(exc.value)
    assert "configs/example.yaml" in str(exc.value)


def test_enforce_allows_an_announced_coverage_baseline_to_run() -> None:
    """NONE is not an error — once ANNOUNCED. The requirement was always
    "announce the blindness, not forbid it"; as of 2026-08-28 the
    announcement is a declared key, and a SILENT baseline refuses at
    launch exactly like UNFIREABLE, before any emulator second is spent.
    All 37 shipped baselines carry the acknowledgment."""
    prof = _profile()
    prof["solve"]["no_clear_predicate"] = "no witnessed clear; unminted"
    r = enforce(prof)
    assert r.verdict == NONE

    with pytest.raises(SystemExit):
        enforce(_profile())  # silent -> refused at launch


def test_the_launch_banner_names_the_constant_for_a_blind_profile() -> None:
    """The one line that would have prevented all ten distorted claims:
    printed at launch, before the hours are spent, in the log every
    receipt is written from."""
    banner = launch_banner(_profile(), "configs/rygar.yaml")
    assert banner is not None
    assert "NO REACHABLE CLEAR PREDICATE" in banner
    assert "configs/rygar.yaml" in banner
    assert "searched and found none" in banner


def test_there_is_no_banner_when_a_predicate_exists() -> None:
    assert launch_banner(_profile(level_key=[0x0028])) is None


# --------------------------------------------------------------------------
# 6. The shipped corpus.
# --------------------------------------------------------------------------

def _solve_profiles() -> list[Path]:
    out = []
    for p in sorted((REPO / "configs").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("solve"), dict):
            out.append(p)
    return out


@pytest.mark.parametrize("path", _solve_profiles(), ids=lambda p: p.name)
def test_no_shipped_profile_declares_a_hook_it_cannot_fire(path: Path) -> None:
    """The regression bar. A profile may honestly have NO clear predicate
    (36 do); it may not ADVERTISE one that is provably inert. Gradius
    failed this from 2026-08-24, when the League onboarding wave swapped
    its progress to the odometer, until 2026-08-26."""
    r = clear_reachability(yaml.safe_load(path.read_text()))
    assert r.ok, f"{path.name}: {r.verdict} — {r.reason}"


def test_the_corpus_is_not_uniform() -> None:
    """Cheap anti-vacuity check on the sweep above: if every profile
    resolved to the same verdict, the parametrized test would be
    asserting nothing about the guard's discrimination."""
    verdicts = {clear_reachability(yaml.safe_load(p.read_text())).verdict
                for p in _solve_profiles()}
    assert {NONE, REACHABLE} <= verdicts


def test_the_cli_exits_nonzero_on_a_refusal(tmp_path: Path) -> None:
    """The lint has to be usable from a Makefile, which means the exit
    code has to carry the verdict."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(_profile(
        progress={"source": "odometer", "axis": "x"},
        clear={"mode": "confluence"})))
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_profile(level_key=[0x0028])))

    script = str(REPO / "scripts" / "clear_reachability.py")
    rc_bad = subprocess.run([sys.executable, script, str(bad)],
                            capture_output=True, text=True)
    rc_good = subprocess.run([sys.executable, script, str(good)],
                             capture_output=True, text=True)
    assert rc_bad.returncode == 1, rc_bad.stdout
    assert rc_good.returncode == 0, rc_good.stdout
    assert "UNFIREABLE" in rc_bad.stdout


def test_reachability_is_hashable_and_frozen() -> None:
    """It gets stashed in receipts and compared; accidental mutation of a
    verdict after the fact is the kind of thing that produced this mess."""
    r = clear_reachability(_profile())
    assert isinstance(r, Reachability)
    with pytest.raises(Exception):
        r.verdict = REACHABLE          # type: ignore[misc]


# ---------------------------------------------------------------------------
# NONE_SILENT (direction review §5.1, 2026-08-28): a profile whose
# is_clear is the `() > ()` constant must ANNOUNCE it or the lint exits
# non-zero. "Announced instead of inferred" was the docstring's own
# promise; this makes it enforced. 37 profiles carried the silent
# constant for months while `solutions: 0` was read as a search result.
# ---------------------------------------------------------------------------

def test_silent_empty_coverage_baseline_is_not_ok():
    from clear_reachability import clear_reachability, NONE_SILENT
    r = clear_reachability({"solve": {"level_key": []}})
    assert r.verdict == NONE_SILENT
    assert not r.ok, "a silent constant must fail the lint"
    assert not r.can_bank_a_solution


def test_acknowledged_coverage_baseline_is_ok_but_still_cannot_bank():
    from clear_reachability import clear_reachability, NONE
    r = clear_reachability({"solve": {
        "level_key": [],
        "no_clear_predicate": "no witnessed clear; deliberately unminted",
    }})
    assert r.verdict == NONE and r.ok
    assert not r.can_bank_a_solution, (
        "acknowledgment changes the lint verdict, never the constant"
    )


def test_empty_or_blank_acknowledgment_stays_silent():
    from clear_reachability import clear_reachability, NONE_SILENT
    for bad in ("", "   ", None, True, 7):
        r = clear_reachability({"solve": {
            "level_key": [], "no_clear_predicate": bad,
        }})
        assert r.verdict == NONE_SILENT, (
            f"a {bad!r} acknowledgment must not satisfy the announcement "
            "requirement — an empty ack is the silent case wearing a key"
        )


def test_every_shipped_profile_is_announced_or_reachable():
    """The roster-level enforcement: no configs/*.yaml may carry the
    silent constant. This is the test that makes a NEW silent profile
    impossible to add — the exact property the census lacked."""
    import pathlib
    import yaml
    from clear_reachability import clear_reachability, NONE_SILENT
    root = pathlib.Path(__file__).resolve().parent.parent
    silent = []
    for p in sorted((root / "configs").glob("*.yaml")):
        prof = yaml.safe_load(open(p)) or {}
        if "solve" not in prof:
            continue
        if clear_reachability(prof).verdict == NONE_SILENT:
            silent.append(p.name)
    assert not silent, (
        f"{len(silent)} profile(s) carry the `() > ()` constant with no "
        f"acknowledgment: {silent[:6]} — declare solve.no_clear_predicate "
        "or mint a predicate"
    )
