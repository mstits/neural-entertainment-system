"""Per-run reproducibility manifest.

Each training run writes a `run_manifest.json` into its checkpoint dir
capturing exactly what's needed to reproduce it: ROM + its MD5, start
state, seed, the git commit of the code, and the load-bearing
hyperparameters. Combined with the code at that commit and the
metrics.jsonl the run emits, a result becomes reproducible and citable
rather than "it worked on my machine that one time."

Kept deliberately small — provenance + config only. The achieved
results live in metrics.jsonl / eval.jsonl (the scoreboard joins them).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

# The hyperparameters that actually change learning behavior — worth
# pinning so a manifest fully specifies the run alongside the seed+commit.
_PINNED_HYPERPARAMS = (
    "lr", "gamma", "gae_lambda", "ppo_clip_eps", "value_coef",
    "entropy_coef", "grad_clip", "rollout_steps", "steps",
    "ppo_minibatch_size", "value_loss", "rnd_intrinsic_coef",
    "rnd_loss_coef", "tile_frame_stack",
)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
    except Exception:
        return None


def write_run_manifest(
    checkpoint_dir,
    *,
    game: Optional[str],
    rom_path: str,
    start_state_path: Optional[str],
    seed: Optional[int],
    profile: dict,
    num_envs: int,
    frame_skip: int,
    rom_md5: Optional[str] = None,
    created_at: Optional[float] = None,
) -> Path:
    """Write `<checkpoint_dir>/run_manifest.json` atomically; return its path.

    `seed` and `rom_md5` are the two provenance fields that make a run
    reproducible: the seed pins every RNG, and the MD5 pins the exact ROM
    dump (a re-dumped/patched ROM shifts RAM addresses and silently breaks
    reward functions). Both are recorded verbatim; a caller that leaves
    them as ``None`` records ``null`` (still a valid, if weaker, manifest).
    """
    rl = profile.get("reinforce", {}) or {}
    manifest = {
        "game": game,
        "rom_path": str(rom_path),
        "rom_md5": rom_md5,
        "start_state_path": str(start_state_path) if start_state_path else None,
        "seed": seed,
        "git_commit": _git_commit(),
        "created_at": created_at if created_at is not None else time.time(),
        "trainer_mode": rl.get("trainer_mode", "vanilla_ppo"),
        "encoder": rl.get("encoder", "nature_dqn"),
        "device": rl.get("device"),
        "num_envs": int(num_envs),
        "frame_skip": int(frame_skip),
        "hyperparams": {k: rl[k] for k in _PINNED_HYPERPARAMS if k in rl},
    }
    path = Path(checkpoint_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)
    return path
