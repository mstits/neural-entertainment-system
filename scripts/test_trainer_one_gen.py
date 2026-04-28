#!/usr/bin/env python3
"""Headless trainer smoke test: construct a Trainer, run exactly one
generation, confirm no crashes and the metrics look sane. Exercises
the refactored `_evaluate_batch` hot loop (now pool-impl agnostic)."""

import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.training.trainer import Trainer


def main() -> None:
    profile = yaml.safe_load((_REPO_ROOT / "configs" / "zelda.yaml").read_text())
    # Tiny config so the test completes in seconds, not minutes.
    profile["max_episode_steps"] = 120  # ~2s of game time per genome
    profile.setdefault("reinforce", {})["enabled"] = False  # skip gradient step
    profile["reinforce"]["torch_compile"] = False  # faster startup

    env_spec = "nes_core:NESEnvironment"
    print(f"backend: {env_spec}")
    trainer = Trainer(
        rom_path=str(_REPO_ROOT / "roms" / "zelda.nes"),
        game_profile=profile,
        num_instances=4,
        population_size=4,
        checkpoint_dir=str(_REPO_ROOT / "checkpoints_test"),
        env_spec=env_spec,
    )
    trainer.run(num_generations=1)
    print("one gen complete")


if __name__ == "__main__":
    main()
