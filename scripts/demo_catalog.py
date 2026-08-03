#!/usr/bin/env python3
"""Index of everything the demo console can show.

Scans the repo for real, on-disk assets:

  - every game profile the solver stack knows about (configs/*.yaml with a
    `solve:` section -> the generic Go-Explore adapter in
    scripts/go_explore_solve.py's GenericGame; SMB is the one built-in
    exception -- its profiles carry no `solve:` key by design, so
    go_explore_solve.make_game() falls back to the byte-exact SmbGame
    adapter when `solve` is absent)

  - every banked win or research frontier under runs/: the SMB 32-level
    full-clear chain, the Castlevania block chain + its open block-3
    frontier, Contra's base-wall coverage grind, each newer game's
    Go-Explore bootstrap archive, and any standalone level solves.

Nothing here is fabricated or hand-typed: every "banked" entry is read off
a real file under runs/ at scan time, so the catalog can never claim a win
that doesn't have a receipt on disk.

CLI:
    python scripts/demo_catalog.py           # pretty-printed
    python scripts/demo_catalog.py --json    # JSON (for a demo-console UI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"

sys.path.insert(0, str(REPO / "scripts"))
try:
    from train_game import DEFAULT_ROMS  # noqa: E402  (per-game canonical ROM)
except Exception:
    DEFAULT_ROMS = {}

# Games that have no `solve:` section yet but are real, configured games the
# demo console should still list (has_adapter=False until a solve adapter is
# written for them). "mario" is the one built-in-adapter exception -- see
# module docstring.
EXTRA_GAMES = ["mario", "zelda", "tetris", "punchout"]

# Human display names for banked-entry labels. Falls back to the profile's
# own `name:` field, then a title-cased slug, when a game isn't listed here.
PRETTY_NAMES = {
    "mario": "SMB",
    "contra": "Contra",
    "castlevania": "Castlevania",
    "bubble_bobble": "Bubble Bobble",
    "double_dragon": "Double Dragon",
    "ducktales": "DuckTales",
    "excitebike": "Excitebike",
    "ghosts_n_goblins": "Ghosts'n Goblins",
    "gradius": "Gradius",
    "kid_icarus": "Kid Icarus",
    "kirby": "Kirby's Adventure",
    "kungfu": "Kung Fu",
    "megaman": "Mega Man 2",
    "metroid": "Metroid",
}

# Legacy/campaign run-directory roots that don't sit at runs/<name>/. Every
# other solve-adapter game's bootstrap archive lives at runs/<name>/*.
CAMPAIGN_ROOTS = {
    "contra": ["breadth_contra"],
    "castlevania": ["cv_chain_a", "cv_chain_hw", "cv_chain_hw2", "cv_chain_poweron", "cv_smoke"],
}


# ---------------------------------------------------------------------------
# small path / io helpers
# ---------------------------------------------------------------------------

def _yaml(path: Path) -> dict:
    try:
        d = yaml.safe_load(path.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def rel(p: Any) -> Optional[str]:
    """Repo-relative path string. Absolute paths under REPO are shortened;
    already-relative paths (the project convention for every config/receipt
    field) pass through unchanged."""
    if p is None:
        return None
    s = str(p)
    if not s:
        return None
    path = Path(s)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO))
        except ValueError:
            return s
    return s


def _display_name(short: str, profile: dict) -> str:
    if short in PRETTY_NAMES:
        return PRETTY_NAMES[short]
    name = profile.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return short.replace("_", " ").title()


# ---------------------------------------------------------------------------
# game roster
# ---------------------------------------------------------------------------

def discover_solve_game_names() -> list[str]:
    """Short names of every configs/<name>.yaml with a genuine top-level
    `solve:` mapping (not just the substring "solve:" in a comment/description
    -- distinguishes contra.yaml, which has one, from contra_screen9.yaml or
    the composite_*.yaml routing configs, which merely mention it in prose)."""
    names = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        profile = _yaml(path)
        if isinstance(profile.get("solve"), dict):
            names.append(path.stem)
    return names


def resolve_rom(name: str, profile: dict) -> str:
    """rom: solve.rom > profile rom_path > per-game canonical default
    (scripts/train_game.py's DEFAULT_ROMS -- the same table the headless
    launcher uses to resolve a game to a concrete ROM)."""
    solve = profile.get("solve")
    if isinstance(solve, dict) and solve.get("rom"):
        return str(solve["rom"])
    if profile.get("rom_path"):
        return str(profile["rom_path"])
    return DEFAULT_ROMS.get(name, "")


# ---------------------------------------------------------------------------
# banked-entry scanners
# ---------------------------------------------------------------------------

def scan_smb(banked: list[dict]) -> None:
    """The verified 32-level power-on -> THANK YOU MARIO chain, its
    per-level demo-selectable clips, and the standalone pre-chain research
    solves that predate it (runs/ge_1_2_solve etc.)."""
    chain_dir = RUNS / "live_show" / "smb_4_4_micro"
    verify = _json(chain_dir / "chain_verify.json")
    if verify:
        levels = verify.get("levels", [])
        n = len(levels)
        ok = bool(verify.get("ok"))
        first_lvl = levels[0]["level"] if levels else None
        last_lvl = levels[-1]["level"] if levels else None
        total_actions = verify.get("total_actions") or sum(
            int(lv.get("actions", 0)) for lv in levels
        )
        banked.append({
            "run_dir": rel(chain_dir),
            "kind": "chain",
            "label": (
                f"SMB {first_lvl} -> {last_lvl} full clear "
                f"({n}/32 levels, cold power-on, zero deaths, "
                f"{'VERIFIED' if ok else 'verification FAILED'})"
            ),
            "level": f"{first_lvl}..{last_lvl}" if levels else None,
            "cleared": ok and n >= 32,
            "depth": n,
            "actions_path": None,
            "root_state": rel(chain_dir / "entrance_start.state"),
        })
        for lv in levels:
            label_name = lv.get("level")
            lvl_dir = chain_dir / f"lvl_{label_name}"
            sol_json = lvl_dir / "solutions" / "sol_000.json"
            sol_npy = lvl_dir / "solutions" / "sol_000.actions.npy"
            sol = _json(sol_json) or {}
            banked.append({
                "run_dir": rel(lvl_dir),
                "kind": "solution",
                "label": (
                    f"SMB {label_name} clear "
                    f"({lv.get('actions')} actions, from its entrance)"
                ),
                "level": label_name,
                "cleared": bool(lv.get("ok")),
                "depth": lv.get("actions"),
                "actions_path": rel(sol_npy) if sol_npy.is_file() else None,
                "root_state": rel(sol.get("root_state")),
            })

    # Standalone research-era solves that predate the verified chain above
    # (each is a single level, solved in isolation, not chained end to end).
    for tag, level_label in (
        ("ge_1_2_solve", "1-2"), ("ge_1_3_solve", "1-3"),
        ("ge_1_4_solve", "1-4"), ("ge_2_1_solve", "2-1"),
    ):
        d = RUNS / tag
        sol_dir = d / "solutions"
        if not sol_dir.is_dir():
            continue
        sols = []
        for sp in sorted(sol_dir.glob("sol_*.json")):
            sd = _json(sp)
            if sd:
                sols.append((sp, sd))
        if not sols:
            continue
        best_path, best = min(sols, key=lambda t: t[1].get("actions", 1 << 30))
        npy = best_path.with_name(best_path.stem + ".actions.npy")
        stats = _json(d / "archive.stats.json") or {}
        banked.append({
            "run_dir": rel(d),
            "kind": "solution",
            "label": (
                f"SMB {level_label} standalone Go-Explore solve "
                f"(research, pre-chain; {len(sols)} solutions banked, "
                f"best {best.get('actions')} actions)"
            ),
            "level": level_label,
            "cleared": True,
            "depth": best.get("actions"),
            "actions_path": rel(npy) if npy.is_file() else None,
            "root_state": rel(best.get("root_state")),
        })
        # a frontier line too -- these archives kept exploring past the
        # solved segment (e.g. ge_1_2_solve's 8180 cells / frontier 3048).
        if stats:
            banked.append({
                "run_dir": rel(d),
                "kind": "frontier",
                "label": (
                    f"SMB {level_label} solve-archive frontier "
                    f"({stats.get('cells', 0):,} cells explored, "
                    f"best score {stats.get('best_score')})"
                ),
                "level": level_label,
                "cleared": False,
                "depth": stats.get("best_score"),
                "actions_path": None,
                "root_state": None,
            })


def _chain_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out


def scan_castlevania(banked: list[dict]) -> None:
    """The deepest verified power-on chain (whichever cv_chain_*/chain.jsonl
    solved the most consecutive blocks) plus the open block-3 exploration
    frontier (the deepest archive among every lvl_*_3* / lvl_03* attempt --
    block 3 has never been solved, per the block-3 bat-wake investigation)."""
    best_chain: Optional[tuple[Path, list[dict]]] = None
    for root in (RUNS / r for r in CAMPAIGN_ROOTS["castlevania"]):
        jsonl = root / "chain.jsonl"
        if not jsonl.is_file():
            continue
        entries = _chain_jsonl(jsonl)
        if best_chain is None or len(entries) > len(best_chain[1]):
            best_chain = (root, entries)

    if best_chain:
        root, entries = best_chain
        solved = [e for e in entries if e.get("status") == "solved"]
        if solved:
            last_solved = solved[-1]
            next_label = last_solved.get("next_label")
            root_state = root / "poweron_root.state"
            banked.append({
                "run_dir": rel(root),
                "kind": "chain",
                "label": (
                    f"Castlevania power-on -> block {next_label} chain "
                    f"({len(solved)} blocks solved back-to-back, "
                    f"block {next_label} frontier still open)"
                ),
                "level": f"0..{last_solved.get('level')}",
                "cleared": False,
                "depth": len(solved),
                "actions_path": rel(last_solved.get("solution")),
                "root_state": rel(root_state) if root_state.is_file() else None,
            })

    # Open block-3 frontier: the biggest archive among every lvl_*_3*/lvl_03*
    # attempt across all cv_chain_* campaigns.
    best_dir = None
    best_stats: dict = {}
    for root in (RUNS / r for r in CAMPAIGN_ROOTS["castlevania"]):
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if not (("_3" in d.name) or d.name.startswith("lvl_03")):
                continue
            stats = _json(d / "archive.stats.json")
            if not stats:
                continue
            if stats.get("cells", 0) > best_stats.get("cells", -1):
                best_dir, best_stats = d, stats
    if best_dir is not None:
        banked.append({
            "run_dir": rel(best_dir),
            "kind": "frontier",
            "label": (
                f"Castlevania block 3 exploration frontier "
                f"({best_stats.get('cells', 0):,} cells, "
                f"best score {best_stats.get('best_score')}, unsolved)"
            ),
            "level": "3",
            "cleared": False,
            "depth": best_stats.get("best_score"),
            "actions_path": None,
            "root_state": None,
        })


def scan_contra(banked: list[dict]) -> None:
    """Contra has no level_key yet (coverage baseline, per configs/contra.yaml
    solve.level_key: []) so there is no clearable "level" -- just the
    base-wall coverage grind's deepest archive."""
    root = RUNS / CAMPAIGN_ROOTS["contra"][0]
    if not root.is_dir():
        return
    best_dir = None
    best_stats: dict = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        stats = _json(d / "archive.stats.json")
        if not stats:
            continue
        if stats.get("cells", 0) > best_stats.get("cells", -1):
            best_dir, best_stats = d, stats
    if best_dir is None:
        return
    root_state = root / "poweron_root.state"
    banked.append({
        "run_dir": rel(best_dir),
        "kind": "frontier",
        "label": (
            f"Contra base-wall grind, {best_stats.get('cells', 0):,} cells "
            f"(best score {best_stats.get('best_score')})"
        ),
        "level": None,
        "cleared": False,
        "depth": best_stats.get("best_score"),
        "actions_path": None,
        "root_state": rel(root_state) if root_state.is_file() else None,
    })


def _fmt_wd(wd: Any) -> str:
    """A world/door coordinate (e.g. [1] or [0, 2]) as "1" / "0-2" instead of
    a Python list repr."""
    if isinstance(wd, (list, tuple)):
        return "-".join(str(x) for x in wd)
    return str(wd)


def _root_state_of(d: Path) -> Optional[str]:
    roots = _json(d / "roots.json")
    if not isinstance(roots, dict):
        return None
    for entry in roots.values():
        if isinstance(entry, dict) and entry.get("path"):
            return rel(entry["path"])
    return None


def scan_generic_bootstrap(name: str, banked: list[dict]) -> None:
    """Every other solve-adapter game: its Go-Explore bootstrap archive(s)
    directly under runs/<name>/*, each read from its own archive.stats.json.
    Any solutions/sol_*.json whose start_wd != clear_wd is a genuine
    level/round clear and gets its own "solution" entry too."""
    root = RUNS / name
    if not root.is_dir():
        return
    pretty = PRETTY_NAMES.get(name, name.replace("_", " ").title())
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        stats = _json(d / "archive.stats.json")
        if not stats:
            continue

        sol_dir = d / "solutions"
        clears = []
        if sol_dir.is_dir():
            for sp in sorted(sol_dir.glob("sol_*.json")):
                sd = _json(sp)
                if sd and sd.get("start_wd") != sd.get("clear_wd"):
                    clears.append((sp, sd))
        if clears:
            sp, sd = min(clears, key=lambda t: t[1].get("actions", 1 << 30))
            npy = sp.with_name(sp.stem + ".actions.npy")
            banked.append({
                "run_dir": rel(d),
                "kind": "solution",
                "label": (
                    f"{pretty} {_fmt_wd(sd.get('start_wd'))} -> "
                    f"{_fmt_wd(sd.get('clear_wd'))} clear "
                    f"({sd.get('actions')} actions, {d.name})"
                ),
                "level": _fmt_wd(sd.get("clear_wd")),
                "cleared": True,
                "depth": sd.get("actions"),
                "actions_path": rel(npy) if npy.is_file() else None,
                "root_state": rel(sd.get("root_state")) or _root_state_of(d),
            })

        banked.append({
            "run_dir": rel(d),
            "kind": "frontier",
            "label": (
                f"{pretty} {d.name} bootstrap frontier "
                f"({stats.get('cells', 0):,} cells, "
                f"best score {stats.get('best_score')})"
            ),
            "level": None,
            "cleared": False,
            "depth": stats.get("best_score"),
            "actions_path": None,
            "root_state": _root_state_of(d),
        })


SPECIAL_SCANNERS = {
    "mario": scan_smb,
    "castlevania": scan_castlevania,
    "contra": scan_contra,
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_catalog() -> dict:
    """Build the demo console's game+banked-wins index.

    Returns {"games": [...], "generated": None} per the catalog contract --
    see this module's docstring and each game dict's fields:
      name, profile, rom, start_state, has_adapter, banked[]
    """
    solve_names = discover_solve_game_names()
    seen = set(solve_names)
    ordered_names = list(solve_names) + [g for g in EXTRA_GAMES if g not in seen]

    games = []
    for name in ordered_names:
        profile_path = CONFIGS / f"{name}.yaml"
        profile = _yaml(profile_path)
        has_solve = isinstance(profile.get("solve"), dict)
        has_adapter = has_solve or name == "mario"

        banked: list[dict] = []
        scanner = SPECIAL_SCANNERS.get(name)
        if scanner is not None:
            scanner(banked)
        elif has_solve:
            scan_generic_bootstrap(name, banked)

        games.append({
            "name": name,
            "profile": rel(profile_path) if profile_path.is_file() else f"configs/{name}.yaml",
            "rom": resolve_rom(name, profile),
            "start_state": rel(profile.get("start_state_path")),
            "has_adapter": has_adapter,
            "banked": banked,
        })

    return {"games": games, "generated": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pretty_print(catalog: dict) -> None:
    games = catalog["games"]
    print(f"{len(games)} games in the demo catalog\n")
    for g in games:
        adapter = "adapter" if g["has_adapter"] else "no adapter"
        print(f"* {g['name']:16s} [{adapter:11s}] profile={g['profile']}")
        print(f"    rom:         {g['rom']}")
        print(f"    start_state: {g['start_state']}")
        if not g["banked"]:
            print("    banked:      (none)")
            continue
        print(f"    banked ({len(g['banked'])}):")
        for b in g["banked"]:
            flag = "CLEARED" if b["cleared"] else "frontier"
            depth = b["depth"] if b["depth"] is not None else "?"
            print(f"      - [{b['kind']:8s} {flag:8s} depth={depth!s:<8}] {b['label']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a pretty listing")
    args = ap.parse_args()

    catalog = build_catalog()
    if args.json:
        print(json.dumps(catalog, indent=2))
    else:
        _pretty_print(catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
