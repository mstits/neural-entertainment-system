"""Every top-level `configs/*.yaml` that declares a `solve:` block must
either construct through `make_game()` or say, by name, why it cannot.

D3 (docs/research/CLEAR_GAP_CLOSURE_2026-08-26.md §1c rows 20 + 35):
`legend_of_zelda.yaml` and `tetris_usa.yaml` both shipped a `solve:`
stub (rom + progress only) with no `y` / `level_key` / `lives`, so
`make_game()` raised a bare `KeyError('y')` before a single search step
could run — and nothing upstream could tell "not onboarded yet" apart
from "behaviourally proven to have no y" (Tetris genuinely has no
jump/ballistic axis; it is a static-playfield puzzle game). This test
is the roster-wide falsifier `clear_reachability.py` §10 recommended:
construct every `solve:`-bearing profile, and require an explicit,
reasoned `constructible: false` for the ones that legitimately can't.

`make_game()` never touches a ROM or the emulator (GenericGame.__init__
only parses the profile dict), so this test needs no game assets and
runs in milliseconds — the same "cheapest possible check" discipline
`tests/test_profile_configs.py`'s trainer-boot-contract test already
uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.go_explore_solve import ProfileNotConstructible, make_game

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _solve_profiles() -> list[Path]:
    """Every top-level profile YAML carrying a `solve:` block."""
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and data.get("solve"):
            out.append(path)
    return out


@pytest.mark.parametrize("profile_path", _solve_profiles(), ids=lambda p: p.name)
def test_solve_profile_constructs_or_declares_why_not(profile_path: Path) -> None:
    """A profile with a `solve:` block must either build a working
    GenericGame, or its `solve:` block must carry BOTH
    `constructible: false` and a non-empty `reason:` explaining what
    was tried and why it failed — never a silent KeyError."""
    data = yaml.safe_load(profile_path.read_text())
    solve = data["solve"]
    if solve.get("constructible") is False:
        reason = solve.get("reason")
        assert reason and str(reason).strip(), (
            f"{profile_path.name}: constructible: false with no reason "
            f"recorded — a refusal with no evidence is as bad as a "
            f"silent KeyError."
        )
        with pytest.raises(ProfileNotConstructible):
            make_game(data)
        return
    try:
        make_game(data)
    except KeyError as exc:
        pytest.fail(
            f"{profile_path.name}: make_game() raised bare {exc!r} — "
            f"either fill in the missing solve.{exc.args[0]} key "
            f"(discovered behaviourally, never guessed) or mark this "
            f"profile `constructible: false` with a reason."
        )


def test_tetris_usa_is_explicitly_non_constructible() -> None:
    """Regression guard for D3: this profile must never again crash
    with a bare KeyError('y') — the wave-3 discovery record (0 progress
    candidates, no y ballistic signature in any of 3 directions) means
    the honest answer is a named refusal, not a stub."""
    data = yaml.safe_load((CONFIG_DIR / "tetris_usa.yaml").read_text())
    assert data["solve"].get("constructible") is False
    with pytest.raises(ProfileNotConstructible):
        make_game(data)


def test_legend_of_zelda_constructs() -> None:
    """Regression guard for D3: this profile's solve: block must carry
    a real, behaviourally-discovered y / level_key / lives triple (see
    runs/onboard_wave4/discover_legend_of_zelda_right.json) and build a
    working GenericGame instead of raising KeyError('y')."""
    data = yaml.safe_load((CONFIG_DIR / "legend_of_zelda.yaml").read_text())
    game = make_game(data)
    assert game.__class__.__name__ == "GenericGame"
    assert game._y is not None
    assert game._lives is not None
