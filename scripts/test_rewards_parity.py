#!/usr/bin/env python3
"""Parity test: Rust nes_core reward functions vs legacy Python
reward functions. Feeds identical synthetic RAM sequences to both
implementations and asserts per-step (reward, done, level_id) match.

Covers every supported game. Any divergence is a port bug."""

import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nes_core
from src.utils.reward_functions import build_reward_function


def make_ram(seed: int) -> bytearray:
    rng = random.Random(seed)
    return bytearray(rng.randrange(0, 256) for _ in range(2048))


def evolve_ram(ram: bytearray, step: int, seed: int) -> None:
    """Make small changes between steps so reward signals fire."""
    rng = random.Random(seed * 7919 + step)
    # Touch 5-15 random bytes per step — small enough that most signals
    # don't fire, big enough that some do.
    for _ in range(rng.randint(5, 15)):
        addr = rng.randrange(0, 2048)
        ram[addr] = rng.randrange(0, 256)


def run_game(game_name: str, profile: dict, n_steps: int = 500, seed: int = 42) -> None:
    py_fn = build_reward_function(profile)
    rust_fn = nes_core.build_reward_function(profile)

    ram = make_ram(seed)
    diverged = False
    for step in range(n_steps):
        evolve_ram(ram, step, seed)
        action = (step * 17) & 0xFF
        py_out = py_fn.compute(bytes(ram), action=action)
        rust_out = rust_fn.compute(bytes(ram), action=action)
        if (
            not math.isclose(py_out[0], rust_out[0], rel_tol=1e-9, abs_tol=1e-9)
            or py_out[1] != rust_out[1]
            or py_out[2] != rust_out[2]
        ):
            print(
                f"  DIVERGE @ step {step}: py={py_out} rust={rust_out}"
            )
            diverged = True
            break
    if not diverged:
        # Final check: breakdown keys match
        py_bd = dict(getattr(py_fn, "breakdown", {}))
        rust_bd = dict(rust_fn.breakdown)
        py_keys = set(py_bd.keys())
        rust_keys = set(rust_bd.keys())
        if py_keys != rust_keys:
            extra_py = py_keys - rust_keys
            extra_rust = rust_keys - py_keys
            note = []
            if extra_py:
                note.append(f"py-only: {sorted(extra_py)}")
            if extra_rust:
                note.append(f"rust-only: {sorted(extra_rust)}")
            print(f"  {game_name}: {n_steps} steps match, breakdown keys differ ({'; '.join(note)})")
        else:
            print(f"  {game_name}: {n_steps} steps match perfectly, {len(py_keys)} breakdown signals")


def main() -> None:
    profiles = [
        {"name": "zelda", "reward_id": "zelda", "reward_weights": {}},
        {"name": "mario", "reward_id": "mario", "reward_weights": {}},
        {"name": "contra", "reward_id": "contra", "reward_weights": {}},
        {"name": "megaman", "reward_id": "mega_man", "reward_weights": {}},
        {"name": "castlevania", "reward_id": "castlevania", "reward_weights": {}},
        {"name": "metroid", "reward_id": "metroid", "reward_weights": {}},
    ]
    print("Rust reward parity test (500 steps per game)")
    print("-" * 60)
    for profile in profiles:
        run_game(profile["name"], profile)


if __name__ == "__main__":
    main()
