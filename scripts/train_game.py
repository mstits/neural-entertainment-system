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


def match_roms(
    game_key: str, canonical_name: str, available: list[Path],
) -> list[Path]:
    """Rank `available` `.nes` files by how well they match a game, best-first.

    Non-matches are excluded. Ordering:
      1. exact basename == the canonical filename (case-insensitive), then
      2. the game key or a significant word of the canonical name appears
         as a substring of the file's stem (case-insensitive).
    An exact basename match, when present, always ranks ahead of fuzzy hits.
    """
    canonical_base = Path(canonical_name).name.lower() if canonical_name else ""
    tokens = _significant_tokens(Path(canonical_name).stem) if canonical_name else set()
    key = game_key.lower().strip()
    exact: list[Path] = []
    fuzzy: list[Path] = []
    for p in available:
        stem = p.stem.lower()
        if canonical_base and p.name.lower() == canonical_base:
            exact.append(p)
        elif (key and key in stem) or any(t in stem for t in tokens):
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

    Prefers the profile's `start_state_path`; falls back to a
    `<rom_stem>_start.state.bin` sidecar next to the ROM. Only returns a
    path that actually exists on disk.
    """
    declared = profile.get("start_state_path")
    if declared and Path(str(declared)).exists():
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
            "[launcher] using start-state sidecar %s (profile declared none "
            "or a missing path)", start_state,
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

    log.info("[launcher] checkpoint_dir = %s", trainer.checkpoint_dir)

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
        # itself and is gated by `fresh_start` below. For GA modes, we'd
        # resolve a latest gen_*.pt here; passing None lets the trainer
        # handle it. `--no-resume` forces a fresh run through both paths.
        pass

    trainer.run(
        num_generations=args.iters, resume_from=resume_from,
        fresh_start=not args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
