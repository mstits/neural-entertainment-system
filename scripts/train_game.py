"""Single-command training launcher per game.

Usage:
    python scripts/train_game.py --game mario
    python scripts/train_game.py --game zelda --iters 100

Resolves a logical `--game` name to the canonical config YAML in
`configs/`, instantiates the Trainer in headless mode (no GUI), and
runs the trainer's loop. Designed to be invoked from the Makefile
target `make train GAME=<name>`, but works standalone.

The headless mode reuses the same Trainer used by the GUI — same
`vanilla_ppo` loop, same curriculum, same auto-resume — minus the
Qt frame sink. Output goes to stdout + the per-game metrics file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.trainer import Trainer  # noqa: E402
from src.training.profile_utils import derive_checkpoint_dir  # noqa: E402


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
}


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
        help="Override num_instances from the profile.",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Auto-resume from latest checkpoint (default: on).",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force a fresh run (ignores existing checkpoints).",
    )
    args = parser.parse_args()

    if not args.game and not args.profile:
        parser.error("either --game or --profile must be specified")

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

    # ROM resolution priority: --rom > profile rom_path > per-game default.
    rom_path = args.rom or profile.get("rom_path") or DEFAULT_ROMS.get(
        (args.game or "").lower().strip()
    )
    if rom_path is None:
        raise SystemExit(
            f"No ROM resolvable for game={args.game!r}. Pass --rom or "
            f"declare rom_path in {profile_path}."
        )
    if not Path(rom_path).exists():
        raise SystemExit(f"ROM file missing: {rom_path}")

    # Start state resolution. Without one, the emulator cold-boots to
    # the title screen and trains on the attract-mode demo (inputs
    # ignored). Prefer the profile's value; fall back to the canonical
    # `<rom>_start.state.bin` sidecar if present. Fail loud if neither
    # resolves — silent title-screen training looks like "the agent
    # won't learn" and wastes entire runs.
    start_state = profile.get("start_state_path")
    if not start_state:
        sidecar = Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
        if sidecar.exists():
            start_state = str(sidecar)
            log = logging.getLogger("train_game")
            log.warning(
                "[launcher] profile has no start_state_path; using sidecar %s",
                sidecar,
            )
    if not start_state or not Path(str(start_state)).exists():
        raise SystemExit(
            f"No start state for game={args.game!r} (profile + sidecar both "
            f"missing). Training would cold-boot to the title-screen demo "
            f"where inputs are ignored. Provide start_state_path in the "
            f"profile or place a '<rom>_start.state.bin' next to the ROM."
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

    trainer = Trainer(
        rom_path=rom_path,
        game_profile=profile,
        num_instances=num_instances,
        population_size=num_instances,  # vanilla_ppo doesn't use a population
        start_state_path=start_state,
        env_spec=profile.get("env_spec", "nes_core"),
        max_episode_steps=int(profile.get("max_episode_steps", 1000)),
    )

    log.info("[launcher] checkpoint_dir = %s", trainer.checkpoint_dir)

    resume_from = None
    if args.resume:
        # For vanilla_ppo, auto-resume is built into _run_vanilla_ppo
        # itself. For GA modes, we'd resolve a latest gen_*.pt here.
        # Either way, passing None lets the trainer handle it.
        pass

    trainer.run(num_generations=args.iters, resume_from=resume_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
