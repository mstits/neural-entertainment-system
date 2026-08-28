"""Single-command training launcher per game.

Usage:
    python scripts/train_game.py --game mario
    python scripts/train_game.py --game zelda --iters 100 --seed 7
    python scripts/train_game.py --setup-check
    python scripts/train_game.py --setup-game --game contra

Resolves a logical `--game` name to the canonical config YAML in
`configs/`, resolves + MD5-validates the ROM (fuzzily, so a differently
named dump in `roms/` still trains), seeds every RNG from `--seed` for
reproducibility, instantiates the Trainer in headless mode (no GUI), and
runs the trainer's loop. Designed to be invoked from the Makefile
target `make train GAME=<name>`, but works standalone.

The headless mode reuses the same Trainer used by the GUI — same
`vanilla_ppo` loop, same curriculum, same auto-resume — minus the
Qt frame sink. Output goes to stdout + the per-game metrics file.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# NOTE: the heavy `Trainer` import (pulls in torch + nes_core) is deferred
# into `main()` so the light-weight ROM resolver + setup helpers below can
# be imported — and unit-tested — without loading the whole training stack.


# Logical-name → canonical config path. Some games have multiple
# profiles (mario_canonical, mario_tiles, mario_vanilla_ppo); the
# launcher picks the one designated as the per-game default for the
# unified-thesis work. Other profiles remain available for explicit
# selection via --profile.
DEFAULT_PROFILES: dict[str, str] = {
    "mario": "configs/mario_vanilla_ppo.yaml",
    "smb": "configs/mario_vanilla_ppo.yaml",
    "contra": "configs/contra.yaml",
    "megaman": "configs/megaman.yaml",
    "mega_man": "configs/megaman.yaml",
    "castlevania": "configs/castlevania.yaml",
    "zelda": "configs/zelda.yaml",
    "metroid": "configs/metroid.yaml",
    "punchout": "configs/punchout.yaml",
    "punch-out": "configs/punchout.yaml",
    "punch_out": "configs/punchout.yaml",
    "kungfu": "configs/kungfu.yaml",
    "kung_fu": "configs/kungfu.yaml",
    "gradius": "configs/gradius.yaml",
    "excitebike": "configs/excitebike.yaml",
    "ghosts": "configs/ghosts_n_goblins.yaml",
    "ghosts_n_goblins": "configs/ghosts_n_goblins.yaml",
    "ducktales": "configs/ducktales.yaml",
    "kid_icarus": "configs/kid_icarus.yaml",
    "kidicarus": "configs/kid_icarus.yaml",
    "double_dragon": "configs/double_dragon.yaml",
    "doubledragon": "configs/double_dragon.yaml",
    "tetris": "configs/tetris.yaml",
    "bubble_bobble": "configs/bubble_bobble.yaml",
    "bubblebobble": "configs/bubble_bobble.yaml",
}

# Per-game canonical ROM. Profile YAMLs don't declare rom_path (the
# GUI prompts the user at launch); the headless launcher hard-codes
# the canonical US-release ROM for each game. Override via --rom.
DEFAULT_ROMS: dict[str, str] = {
    "mario": "roms/Super Mario Bros. (World).nes",
    "smb": "roms/Super Mario Bros. (World).nes",
    "contra": "roms/Contra (USA).nes",
    # Mega Man 2 — the reward fn + profile RAM map are calibrated for
    # MM2 ($06C0 health, $00A8 lives); the MM1 ROM mismatches them.
    "megaman": "roms/Mega Man 2 (USA).nes",
    "mega_man": "roms/Mega Man 2 (USA).nes",
    "castlevania": "roms/Castlevania (USA).nes",
    "zelda": "roms/Legend of Zelda, The (USA) (Rev A).nes",
    "metroid": "roms/Metroid (USA).nes",
    # Punch-Out!! (USA) — the Mr. Dream retail dump (mapper 9 / MMC2). The
    # profile RAM map + start state are calibrated to THIS dump; the Mike
    # Tyson dump differs on later-opponent offsets.
    "punchout": "roms/Punch-Out!! (USA).nes",
    "punch-out": "roms/Punch-Out!! (USA).nes",
    "punch_out": "roms/Punch-Out!! (USA).nes",
    "kungfu": "roms/Kung Fu (Japan, USA).nes",
    "kung_fu": "roms/Kung Fu (Japan, USA).nes",
    "gradius": "roms/Gradius (USA).nes",
    "excitebike": "roms/Excitebike (Japan, USA).nes",
    "ghosts": "roms/Ghosts'n Goblins (USA).nes",
    "ghosts_n_goblins": "roms/Ghosts'n Goblins (USA).nes",
    "ducktales": "roms/DuckTales (USA).nes",
    "kid_icarus": "roms/Kid Icarus (USA, Europe).nes",
    "kidicarus": "roms/Kid Icarus (USA, Europe).nes",
    "double_dragon": "roms/Double Dragon (USA).nes",
    "doubledragon": "roms/Double Dragon (USA).nes",
    "tetris": "roms/Tetris (USA).nes",
    "bubble_bobble": "roms/Bubble Bobble (USA).nes",
    "bubblebobble": "roms/Bubble Bobble (USA).nes",
}


# --------------------------------------------------------------------------
# ROM resolution + integrity
# --------------------------------------------------------------------------

# Words that identify a game's *edition* rather than its identity — dropped
# when deriving fuzzy-match tokens so a region/version tag can't false-match.
_ROM_NOISE_WORDS = frozenset({
    "the", "and", "world", "usa", "europe", "japan", "rev",
})


def rom_md5(path: str | Path) -> str:
    """Whole-file MD5 of a ROM, as a lowercase hex string.

    Matches the No-Intro / `rom_hashes` convention used across
    `configs/*.yaml` (the hash is taken over the entire `.nes` file,
    iNES header included). Streamed so a multi-megabyte ROM never
    balloons memory.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_rom_hashes(profile: dict) -> list[str]:
    """Collect the ROM MD5s a profile declares, lowercased, empties dropped.

    Reads both `rom_hashes` (a list — the canonical config field) and
    `expected_md5` (a scalar — the trainer's legacy lock field). Empty
    placeholders (`""`, unfilled TODO stubs) are ignored so an
    unpopulated hash never triggers a spurious mismatch warning.
    """
    out: list[str] = []
    rh = profile.get("rom_hashes")
    if isinstance(rh, (list, tuple)):
        out.extend(str(x) for x in rh)
    elif isinstance(rh, str):
        out.append(rh)
    em = profile.get("expected_md5")
    if isinstance(em, str):
        out.append(em)
    return [x.strip().lower() for x in out if x and x.strip()]


def _nes_files(roms_dir: str | Path) -> list[Path]:
    """Sorted list of `*.nes` files directly under `roms_dir` (or `[]`)."""
    d = Path(roms_dir)
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() == ".nes"
    )


def _significant_tokens(name: str) -> set[str]:
    """Identity words (>=4 chars, minus region/filler noise) from a ROM name."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {w for w in words if len(w) >= 4 and w not in _ROM_NOISE_WORDS}


# Standalone installment markers — roman numerals II-X or arabic digits
# 2-9 as a whole word-token, never a substring. Franchise entries share
# every significant word of their title with the original ("double",
# "dragon", "zelda", "mega", "man") so token overlap alone cannot tell
# "Double Dragon" from "Double Dragon II", or "Mega Man 2" from "Mega
# Man 3" — only the marker differs, and each installment has its own
# RAM map. Deliberately UNFILTERED by the >=4-char rule above: the
# differentiator here is exactly the short token that rule discards.
_SEQUEL_MARKER_RE = re.compile(
    r"^(?:ii|iii|iv|v|vi|vii|viii|ix|x|[2-9])$"
)


def _installment_markers(name: str) -> set[str]:
    """Standalone sequel/installment tokens in a ROM name (see above)."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {w for w in words if _SEQUEL_MARKER_RE.match(w)}


def _missing_installment(stem: str, canonical_markers: set[str]) -> bool:
    """True if the canonical name names an installment the candidate lacks.

    This is the reverse direction of the sequel check. A marker counts as
    present if it is a standalone word (`Mega Man 2`) or glued to the end
    of one (`megaman2.nes`, a legitimate retag), so the rule rejects a
    dump that does not encode the installment at all without rejecting a
    renamed dump of the right one.
    """
    words = re.findall(r"[a-z0-9]+", stem.lower())
    for marker in canonical_markers:
        if not any(w == marker or (w.endswith(marker) and w != marker
                                   and w[:-len(marker)].isalpha())
                   for w in words):
            return True
    return False


def match_roms(
    game_key: str, canonical_name: str, available: list[Path],
) -> list[Path]:
    """Rank `available` `.nes` files by how well they match a game, best-first.

    Non-matches are excluded. Ordering:
      1. exact basename == the canonical filename (case-insensitive), then
      2. the game key or a significant word of the canonical name appears
         as a substring of the file's stem (case-insensitive) — UNLESS the
         candidate's installment markers (II, III, 2, 3, ...) DIFFER from
         the canonical name's in either direction, which makes it a
         different game in the same franchise, never a fuzzy match, no
         matter how much token overlap its title has with the canonical
         one.
    An exact basename match, when present, always ranks ahead of fuzzy hits.
    """
    canonical_base = Path(canonical_name).name.lower() if canonical_name else ""
    canonical_stem = Path(canonical_name).stem if canonical_name else ""
    tokens = _significant_tokens(canonical_stem)
    canonical_markers = _installment_markers(canonical_stem)
    key = game_key.lower().strip()
    exact: list[Path] = []
    fuzzy: list[Path] = []
    for p in available:
        stem = p.stem.lower()
        if canonical_base and p.name.lower() == canonical_base:
            exact.append(p)
            continue
        if (_installment_markers(stem) - canonical_markers
                or _missing_installment(stem, canonical_markers)):
            # A different installment; token overlap doesn't count. The
            # comparison is SYMMETRIC on purpose. A one-sided
            # `candidate - canonical` blocked only original <- sequel, and
            # left the reverse wide open: every canonical name that CARRIES
            # a marker matched the original's dump. Measured against the
            # real configs, `lost_levels` bound `Super Mario Bros.
            # (World).nes`, Castlevania III bound Castlevania, DuckTales 2
            # bound DuckTales, Ninja Gaiden II bound Ninja Gaiden and
            # Double Dragon II bound Double Dragon — five franchise
            # collisions, one of them on a game with a witnessed clear.
            continue
        if (key and key in stem) or any(t in stem for t in tokens):
            fuzzy.append(p)
    return exact + fuzzy


def resolve_rom(
    game_key: str,
    explicit_rom: Optional[str],
    profile: dict,
    roms_dir: str | Path = "roms",
) -> str:
    """Resolve a game to a concrete ROM path, fuzzily when needed.

    Priority: `--rom` > profile `rom_path` > per-game canonical default.
    When none of those exist verbatim, scan `roms_dir` for a fuzzy match on
    the game name so a user who dropped in `roms/mario.nes` (or a differently
    tagged regional dump) still trains without editing config. Raises a clean
    `SystemExit` — listing the candidates and the exact expected filename(s)
    — when the match is ambiguous or empty.
    """
    if explicit_rom:
        if not Path(explicit_rom).exists():
            raise SystemExit(f"--rom not found: {explicit_rom}")
        return explicit_rom

    prof_rom = profile.get("rom_path")
    if prof_rom and Path(prof_rom).exists():
        return str(prof_rom)

    canonical = DEFAULT_ROMS.get(game_key.lower().strip())
    if canonical and Path(canonical).exists():
        return canonical

    # Nothing verbatim — fuzzy-scan the ROM directory.
    expected = [n for n in (canonical, prof_rom) if n]
    if not expected:
        raise SystemExit(
            f"No ROM resolvable for game={game_key!r}. Pass --rom "
            f"<file.nes> or declare rom_path in the profile."
        )
    canonical_name = expected[0]
    available = _nes_files(roms_dir)
    matches = match_roms(game_key, canonical_name, available)
    expected_names = "\n    ".join(sorted(set(expected)))

    if len(matches) == 1:
        resolved = matches[0]
        logging.getLogger("train_game").warning(
            "[launcher] exact ROM '%s' not found; using fuzzy match '%s' "
            "from %s/", canonical_name, resolved.name, roms_dir,
        )
        return str(resolved)

    if not matches:
        present = ", ".join(p.name for p in available) or "(none)"
        raise SystemExit(
            f"No ROM found for game={game_key!r} in {roms_dir}/.\n"
            f"  Expected one of:\n    {expected_names}\n"
            f"  Present .nes files: {present}\n"
            f"  Drop the ROM into {roms_dir}/ or pass --rom <file.nes>."
        )
    cand = "\n    ".join(p.name for p in matches)
    raise SystemExit(
        f"Ambiguous ROM for game={game_key!r} in {roms_dir}/ — "
        f"{len(matches)} candidates matched:\n    {cand}\n"
        f"  Rename the intended one to the canonical name:\n    {expected_names}\n"
        f"  or select it explicitly with --rom <file.nes>."
    )


def validate_rom_md5(rom_path: str, profile: dict) -> str:
    """Compute the ROM's whole-file MD5 and check it against the profile's
    declared hashes. Warns clearly on mismatch but never raises — a dirty
    dump still boots, and the warning is the signal the user needs. Returns
    the computed MD5 so callers can record it in the run manifest.
    """
    md5 = rom_md5(rom_path)
    declared = profile_rom_hashes(profile)
    log = logging.getLogger("train_game")
    name = Path(rom_path).name
    if not declared:
        log.info(
            "[launcher] ROM %s md5=%s (profile declares no rom_hashes to "
            "validate against; add it to lock the dump)", name, md5,
        )
    elif md5 in declared:
        log.info("[launcher] ROM %s md5 OK: %s", name, md5)
    else:
        log.warning(
            "[launcher] ROM MD5 MISMATCH for %s: got %s, profile expects one "
            "of %s. RAM addresses may be shifted — reward functions can "
            "silently operate on the wrong layout.", name, md5, declared,
        )
    return md5


def resolve_start_state(profile: dict, rom_path: str) -> Optional[str]:
    """Resolve a start-state file for a run, or `None` if none exists.

    A profile that DECLARES `start_state_path` must point at a real
    file: a missing declared path is a hard error, not a fallback.
    (The old sidecar/None fallback silently trained a different level —
    or the title-screen demo — than the one the profile configured.)
    Only when the profile declares nothing do we fall back to the
    `<rom_stem>_start.state.bin` sidecar next to the ROM, or `None`
    when that is absent too.
    """
    declared = profile.get("start_state_path")
    if declared:
        if not Path(str(declared)).is_file():
            raise SystemExit(
                f"start_state_path={declared!r} is declared in the profile "
                f"but does not exist (or is not a regular file). Refusing "
                f"to fall back to a sidecar or cold boot — that silently "
                f"trains the wrong level. Fix the profile's "
                f"start_state_path, capture the state (scripts/"
                f"capture_start_state.py), or remove the key to use the "
                f"'<rom>_start.state.bin' sidecar."
            )
        return str(declared)
    sidecar = Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
    if sidecar.exists():
        return str(sidecar)
    return None


# --------------------------------------------------------------------------
# Onboarding: `make setup-check` / `make setup-game GAME=<x>`
# --------------------------------------------------------------------------

def _load_profile(game: str) -> Optional[dict]:
    """Best-effort load of a game's default profile YAML (or `None`)."""
    try:
        path = resolve_profile_path(game, None)
        with open(path) as f:
            return yaml.safe_load(f)
    except (SystemExit, OSError, yaml.YAMLError):
        return None


def run_setup_check() -> int:
    """Print an environment + per-game ROM readiness report.

    Verifies the Python venv, the `nes_core` import, and torch's MPS
    backend, then lists every per-game ROM with its exact expected
    filename and present/absent (+ MD5) status. Deliberately crash-proof:
    a missing ROM or unbuilt extension is reported, never raised — the
    whole point is to tell a fresh clone what's still needed.
    """
    print("== NES-Evolve setup check ==")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"[python]   {sys.version.split()[0]}  @ {sys.executable}")
    print(f"[venv]     {'active' if in_venv else 'NOT active — run: source .venv/bin/activate'}")

    try:
        import nes_core  # noqa: F401
        has_info = hasattr(nes_core, "rom_info")
        print(f"[nes_core] import OK (rom_info={'yes' if has_info else 'no'})")
    except Exception as exc:  # noqa: BLE001 — report, never crash
        print(f"[nes_core] IMPORT FAILED: {exc}  — run: make build")

    try:
        import torch
        mps = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
        print(f"[torch]    {torch.__version__}  MPS={'available' if mps else 'unavailable'}")
    except Exception as exc:  # noqa: BLE001 — report, never crash
        print(f"[torch]    IMPORT FAILED: {exc}")

    print("\n== ROMs ==  (game -> expected filename : status)")
    by_rom: dict[str, list[str]] = {}
    for game, rom in DEFAULT_ROMS.items():
        by_rom.setdefault(rom, []).append(game)
    for rom, games in sorted(by_rom.items()):
        aliases = "/".join(sorted(games))
        if not Path(rom).exists():
            print(f"  {aliases:16s} {rom} : ABSENT")
            continue
        try:
            md5 = rom_md5(rom)
            declared = profile_rom_hashes(_load_profile(games[0]) or {})
            if not declared:
                status = f"present  md5={md5} (no hash pinned)"
            elif md5 in declared:
                status = f"present  md5 OK ({md5})"
            else:
                status = f"present  md5 MISMATCH (got {md5})"
        except Exception as exc:  # noqa: BLE001 — report, never crash
            status = f"present  (hash failed: {exc})"
        print(f"  {aliases:16s} {rom} : {status}")

    print(
        "\nMissing ROMs are expected on a fresh clone — drop each dump into "
        "roms/ under the exact filename above, then re-run `make setup-check`."
    )
    return 0


def run_setup_game(game: str) -> int:
    """Validate/hash a game's ROM and capture a start-state if one is missing.

    Resolves the ROM (fuzzily), validates its MD5, then checks for a
    start-state; if absent, shells out to `scripts/capture_start_state.py`
    to generate the sidecar. Returns the capture's exit code (0 when a
    start-state already existed or capture succeeded).
    """
    profile = _load_profile(game)
    if profile is None:
        raise SystemExit(
            f"unknown game {game!r} or its profile is missing. "
            f"Known: {sorted(set(DEFAULT_PROFILES))}"
        )

    rom_path = resolve_rom(game, None, profile)
    print(f"[setup] {game}: ROM -> {rom_path}")
    validate_rom_md5(rom_path, profile)

    start_state = resolve_start_state(profile, rom_path)
    if start_state:
        print(f"[setup] {game}: start-state present -> {start_state}")
        return 0

    print(f"[setup] {game}: no start-state found — capturing one "
          f"(scripts/capture_start_state.py) ...")
    capture = ROOT / "scripts" / "capture_start_state.py"
    rc = subprocess.call([sys.executable, str(capture), "--game", game])
    if rc == 0:
        print(f"[setup] {game}: start-state captured.")
    else:
        print(f"[setup] {game}: automatic capture failed (rc={rc}). Some "
              f"games (e.g. Zelda's save-file registration) need manual menu "
              f"navigation — capture a state from the GUI instead.")
    return rc


def _seed_everything(seed: int) -> None:
    """Seed Python `random`, numpy, and torch RNGs from `seed` at launch.

    Runs before the Trainer is built so any module-level RNG use during
    construction is already deterministic. Numpy/torch are imported lazily
    so importing this module stays light for the setup/resolver helpers.
    """
    import random
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:  # noqa: BLE001 — seeding is best-effort
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
            _torch.mps.manual_seed(seed)
    except Exception:  # noqa: BLE001 — seeding is best-effort
        pass


def resolve_profile_path(game: str, explicit: Optional[str]) -> Path:
    """Resolve --game / --profile args to a concrete config file."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--profile not found: {explicit}")
        return p
    key = game.lower().strip()
    if key not in DEFAULT_PROFILES:
        raise SystemExit(
            f"unknown game {game!r}. Known: {sorted(set(DEFAULT_PROFILES))}"
        )
    p = Path(DEFAULT_PROFILES[key])
    if not p.exists():
        raise FileNotFoundError(f"profile file missing: {p}")
    return p


def _pid_start_time(pid: int) -> Optional[str]:
    """Best-effort start-time fingerprint for `pid` (via `ps -o lstart=`).

    Used to tell a still-running process apart from an unrelated one
    that has since recycled the same PID. Returns None if `ps` is
    unavailable or the PID can't be resolved — callers must treat that
    as "unknown", never as "confirmed different process".
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _run_lock_pid_is_live(pid: int, recorded_start: str) -> bool:
    """True iff `pid` is still the process that recorded `recorded_start`.

    A bare `os.kill(pid, 0)` only proves *some* process currently holds
    that PID (success), or that one exists but isn't ours
    (PermissionError) — not that it's the same process that wrote the
    run-lock. Under an unclean shutdown (OOM-kill, power loss) macOS can
    reissue the dead trainer's PID to an unrelated process within
    seconds of boot, which would make a stale `.run.lock` look live
    forever and permanently block every later unattended restart against
    that checkpoint dir. Comparing start-time fingerprints catches PID
    reuse; when either fingerprint can't be determined this stays
    conservative and reports live, matching the pre-fix behavior.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # some process holds this PID, not us — check the fingerprint
    if not recorded_start:
        return True
    current_start = _pid_start_time(pid)
    return current_start is None or current_start == recorded_start


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-command headless training launcher per game.",
    )
    parser.add_argument(
        "--game", type=str, default=None,
        help=f"Game logical name. One of: {sorted(set(DEFAULT_PROFILES))}.",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Explicit profile YAML path (overrides --game default).",
    )
    parser.add_argument(
        "--rom", type=str, default=None,
        help="Explicit ROM path (overrides --game canonical ROM).",
    )
    parser.add_argument(
        "--iters", type=int, default=10_000,
        help="Generations / vanilla_ppo iters to run.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Override num_instances from the profile. Scheduling note "
             "(M4-class, 12P+4E cores): 13-16 envs is the measured sour "
             "spot — workers spill onto E-cores and gate the pool "
             "barrier; prefer <=12 or >=24 (24 measured +19%% pool "
             "throughput vs 16).",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Auto-resume from latest checkpoint (default: on).",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a fresh run (ignores existing checkpoints).",
    )
    parser.add_argument(
        "--no-supervise", action="store_true",
        help="Disable the crash supervisor (default: crashes are logged "
             "and the run relaunches with auto-resume, up to 5 times).",
    )
    parser.add_argument(
        "--strict-config", action="store_true",
        help="Reject (rather than warn on) unregistered profile keys — "
             "catches typo'd/dead knobs that would be silently ignored.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility (default: 0). Seeds python, "
             "numpy and torch, and is recorded in the run manifest.",
    )
    parser.add_argument(
        "--setup-check", action="store_true",
        help="Print an environment + per-game ROM readiness report and exit.",
    )
    parser.add_argument(
        "--setup-game", action="store_true",
        help="Validate/hash --game's ROM and capture a start-state if one "
             "is missing, then exit (no training).",
    )
    args = parser.parse_args()

    if args.setup_check:
        return run_setup_check()
    if args.setup_game:
        if not args.game:
            parser.error("--setup-game requires --game")
        return run_setup_game(args.game)

    if not args.game and not args.profile:
        parser.error("either --game or --profile must be specified")

    # Seed every RNG from --seed BEFORE the trainer is built so the whole
    # run — construction included — is reproducible. Recorded in the manifest.
    _seed_everything(args.seed)

    profile_path = resolve_profile_path(args.game or "mario", args.profile)
    with open(profile_path) as f:
        profile = yaml.safe_load(f)

    # Fail fast on a malformed profile rather than crashing mid-run.
    from src.training.profile_utils import validate_profile
    _problems = validate_profile(profile)
    if _problems:
        raise SystemExit(
            f"invalid profile {profile_path}:\n  - " + "\n  - ".join(_problems)
        )

    # ROM resolution priority: --rom > profile rom_path > per-game default,
    # then a fuzzy scan of roms/ so a differently-named dump still resolves.
    rom_path = resolve_rom(args.game or "", args.rom, profile)
    # Validate the resolved dump against the profile's declared MD5s (warns
    # loudly on mismatch, never fatal); reuse the hash for the run manifest.
    rom_md5_hex = validate_rom_md5(rom_path, profile)

    # Start state resolution. Without one, the emulator cold-boots to
    # the title screen and trains on the attract-mode demo (inputs
    # ignored). Prefer the profile's value; fall back to the canonical
    # `<rom>_start.state.bin` sidecar if present. Fail loud if neither
    # resolves — silent title-screen training looks like "the agent
    # won't learn" and wastes entire runs.
    start_state = resolve_start_state(profile, rom_path)
    if start_state is None:
        raise SystemExit(
            f"No start state for game={args.game!r} (profile + sidecar both "
            f"missing). Training would cold-boot to the title-screen demo "
            f"where inputs are ignored. Provide start_state_path in the "
            f"profile or place a '<rom>_start.state.bin' next to the ROM "
            f"(run `make setup-game GAME={args.game or '<game>'}` to capture one)."
        )
    if start_state != profile.get("start_state_path"):
        logging.getLogger("train_game").warning(
            "[launcher] using start-state sidecar %s (profile declared "
            "none)", start_state,
        )

    num_instances = args.num_envs or int(
        profile.get("reinforce", {}).get("num_envs", 60)
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train_game")
    log.info(
        "[launcher] profile=%s game=%s num_envs=%d iters=%d resume=%s",
        profile_path.name, profile.get("name"), num_instances,
        args.iters, args.resume,
    )

    # Deferred so the resolver/setup helpers above stay importable without
    # loading torch + nes_core (see the module-header note).
    from src.training.trainer import Trainer

    trainer = Trainer(
        rom_path=rom_path,
        game_profile=profile,
        num_instances=num_instances,
        population_size=num_instances,  # vanilla_ppo doesn't use a population
        start_state_path=start_state,
        env_spec=profile.get("env_spec", "nes_core"),
        max_episode_steps=int(profile.get("max_episode_steps", 1000)),
        seed=args.seed,
    )

    # Schema check: a misspelled reinforce knob is silently ignored
    # otherwise (the parsed-but-inert bug class). Warn loudly at startup.
    try:
        from src.training.config_schema import check_profile
        check_profile(profile, strict=args.strict_config, logger=log)
    except Exception as exc:
        if args.strict_config:
            raise
        log.warning("[launcher] config schema check errored: %s", exc)

    log.info("[launcher] checkpoint_dir = %s", trainer.checkpoint_dir)

    # Run-lock: two trainers writing the same checkpoint dir corrupt each
    # other's checkpoints and metrics (interleaved saves, torn files,
    # numbering clashes). A stale lock from a dead process is reclaimed;
    # a live one aborts. Cheap insurance against a double `nohup` launch
    # or a supervisor relaunch racing a not-quite-dead predecessor.
    import os as _os
    _lock = trainer.checkpoint_dir / ".run.lock"
    try:
        if _lock.exists():
            try:
                _lock_lines = _lock.read_text().splitlines()
                _old_pid = int(_lock_lines[0].split()[0])
                _old_start = _lock_lines[1] if len(_lock_lines) > 1 else ""
            except (ValueError, IndexError):
                log.warning("[launcher] reclaiming stale run-lock %s", _lock)
            else:
                if _run_lock_pid_is_live(_old_pid, _old_start):
                    log.error(
                        "[launcher] checkpoint dir is locked by live PID %d "
                        "(%s). Refusing to run two trainers on one dir. Stop "
                        "the other run or pick a different dir.",
                        _old_pid, _lock,
                    )
                    return 1
                log.warning("[launcher] reclaiming stale run-lock %s", _lock)
        _lock.write_text(
            f"{_os.getpid()}\n{_pid_start_time(_os.getpid()) or ''}\n"
            f"{rom_path}\n"
        )
        import atexit as _atexit
        _atexit.register(lambda: _lock.exists() and _lock.unlink())
    except OSError as exc:
        log.warning("[launcher] could not establish run-lock (%s); "
                    "continuing without it", exc)

    # Reproducibility manifest: ROM + its MD5 + start state + seed + git
    # commit + pinned hyperparameters, written to the checkpoint dir before
    # the run starts. With the code at that commit and the run's
    # metrics.jsonl, the result becomes reproducible/citable. Best-effort —
    # never block training on a manifest write.
    try:
        from src.training.run_manifest import write_run_manifest
        mpath = write_run_manifest(
            trainer.checkpoint_dir,
            game=profile.get("name"),
            rom_path=rom_path,
            start_state_path=start_state,
            seed=args.seed,
            rom_md5=rom_md5_hex,
            profile=profile,
            num_envs=num_instances,
            frame_skip=getattr(
                trainer, "frame_skip", int(profile.get("frame_skip", 16))
            ),
        )
        log.info("[launcher] wrote run manifest: %s", mpath)
    except Exception as exc:  # noqa: BLE001 — provenance is non-fatal
        log.warning("[launcher] run manifest write failed: %s", exc)

    resume_from = None
    if args.resume:
        # For vanilla_ppo, auto-resume is built into _run_vanilla_ppo
        # itself and is gated by `fresh_start` below. For GA modes,
        # resolve the latest gen_*.pt here so Trainer.run() actually
        # resumes instead of falling through to a fresh `ga.initialize()`.
        # `--no-resume` forces a fresh run through both paths.
        from src.training.checkpointing import find_latest_checkpoint
        latest = find_latest_checkpoint(trainer.checkpoint_dir)
        resume_from = str(latest) if latest else None

    # Self-healing supervisor: a training process must NEVER die and stay
    # dead. Any trainer exception (numeric collapse, worker panic, MPS
    # hiccup) is logged with its traceback and the run is relaunched with
    # auto-resume, which rolls back to the latest on-disk checkpoint (at
    # most one save interval of progress). Backoff between attempts;
    # repeated immediate failures abort loudly rather than looping — a
    # checkpoint that crashes on load needs a human, not a hot loop.
    import time as _time
    import traceback as _tb
    def _patch_redo_manifest() -> None:
        # Best-effort provenance patch (V31_REDO_SURGICAL_2026-08-27.md
        # §12 item 6) — runs after EVERY attempt (success, VOID abort, or
        # crash) via `finally` below, never masking whatever exception is
        # in flight. `scripts/redo_arm_gate.py` remains the source of
        # truth at verdict time; this is the convenience copy.
        if not getattr(trainer, "redo_enabled", False):
            return
        try:
            import statistics as _stats

            from src.training.run_manifest import update_run_manifest_redo
            agrees = getattr(trainer, "_redo_agree_log", [])
            fracs = getattr(trainer, "_redo_fracs_on_fire", [])
            # F3' TURNOVER (V32 §6.2): fraction of recycle events after
            # the first whose index SET shares >= 1 index with the
            # immediately preceding event's index set. None with fewer
            # than 2 events — there is no preceding event to compare.
            _seq = getattr(trainer, "_redo_fc2_index_sequence", [])
            _repeat_rate = None
            if len(_seq) >= 2:
                _shared = sum(
                    1 for _prev, _cur in zip(_seq, _seq[1:])
                    if set(_prev) & set(_cur)
                )
                _repeat_rate = _shared / (len(_seq) - 1)
            update_run_manifest_redo(
                trainer.checkpoint_dir,
                redo_tau=trainer.redo_tau,
                cum_recycled=trainer._redo_cum_recycled,
                recycle_events=trainer._redo_recycle_events,
                first_recycle_iter=trainer._redo_first_recycle_iter,
                median_agree=(
                    float(_stats.median(agrees)) if agrees else None
                ),
                median_dose_frac=(
                    float(_stats.median(fracs)) if fracs else None
                ),
                distinct_fc2_indices=len(
                    getattr(trainer, "_redo_fc2_index_counts", {})
                ),
                redo_mode=getattr(trainer, "redo_mode", "threshold"),
                redo_bottom_k=getattr(trainer, "redo_bottom_k", 0),
                redo_repeat_rate=_repeat_rate,
            )
        except Exception as exc:  # noqa: BLE001 — provenance is non-fatal
            log.warning("[launcher] redo manifest patch failed: %s", exc)

    max_restarts = 0 if args.no_supervise else 5
    attempt = 0
    while True:
        _t0 = _time.time()
        try:
            try:
                trainer.run(
                    num_generations=args.iters, resume_from=resume_from,
                    fresh_start=(not args.resume) and attempt == 0,
                )
                return 0
            except KeyboardInterrupt:
                log.info("[supervisor] interrupted — clean stop")
                return 0
            # BaseException, NOT Exception: a Rust worker panic crosses
            # the PyO3 boundary as pyo3_runtime.PanicException, which
            # subclasses BaseException — `except Exception` would let it
            # kill the process, defeating the supervisor for the exact
            # failure class (worker panic) it exists to survive.
            # SystemExit still propagates (a deliberate exit is not a
            # crash to restart). A ReDo VOID abort (arming deadline,
            # in-run dose ceiling) is also a RuntimeError caught here —
            # with `--no-supervise` (max_restarts=0) it aborts on the
            # first attempt exactly as registered, never masked by a
            # resume-and-retry loop.
            except SystemExit:
                raise
            except BaseException:  # noqa: BLE001 — the supervisor's job
                attempt += 1
                log.error(
                    "[supervisor] trainer crashed (attempt %d/%d):\n%s",
                    attempt, max_restarts, _tb.format_exc(),
                )
                if attempt > max_restarts:
                    log.error(
                        "[supervisor] restart budget exhausted — aborting"
                    )
                    return 1
                if _time.time() - _t0 < 120:
                    # Crashed within two minutes of starting: likely a
                    # poisoned checkpoint or config, not a transient.
                    # Longer backoff, and give up sooner via the attempt
                    # budget.
                    _time.sleep(60)
                else:
                    _time.sleep(15)
                log.info(
                    "[supervisor] relaunching with auto-resume "
                    "(attempt %d)", attempt,
                )
        finally:
            # Runs after EVERY attempt — success, VOID abort, or crash —
            # so the manifest reflects whatever ReDo telemetry
            # accumulated even when the run never reaches `return 0`.
            _patch_redo_manifest()


if __name__ == "__main__":
    sys.exit(main())
