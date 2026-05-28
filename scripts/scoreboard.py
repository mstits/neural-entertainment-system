"""Multi-game scoreboard: one row per game showing training progress.

Reads the per-game checkpoint dirs and their eval logs (if present),
prints a compact table showing for each game:
  - whether any training has happened (checkpoint exists)
  - the highest-numbered iter checkpoint
  - the latest eval result (mean_return, mean_max_byte, clear_rate)
  - curriculum stage if applicable

This is the developer-facing "where are we across all six games"
view. Phase 0 acceptance criterion: this command produces a
multi-game summary even if only mario has been trained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.profile_utils import profile_slug  # noqa: E402

# Single source of truth for "the 6 games" — keep in lockstep with
# the thesis doc and DEFAULT_PROFILES in train_game.py.
GAMES = [
    ("mario", "Super Mario Bros."),
    ("contra", "Contra"),
    ("megaman", "Mega Man"),
    ("castlevania", "Castlevania"),
    ("zelda", "The Legend of Zelda"),
    ("metroid", "Metroid"),
]


def summarize(game_key: str, display_name: str) -> dict:
    slug = profile_slug(display_name)
    ckpt_dir = Path("checkpoints") / slug
    if not ckpt_dir.exists():
        return {"game": game_key, "status": "untrained", "dir": str(ckpt_dir)}

    iters = sorted(
        ckpt_dir.glob("vanilla_ppo_iter_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    out = {
        "game": game_key,
        "display": display_name,
        "dir": str(ckpt_dir),
        "status": "trained" if iters else "untrained",
        "latest_iter": int(iters[-1].stem.rsplit("_", 1)[-1]) if iters else None,
        "n_checkpoints": len(iters),
    }

    # Curriculum stage indicator (per-game subdir convention).
    curr_dir = ckpt_dir / "smb_curriculum"
    if curr_dir.exists():
        stages = sorted(curr_dir.glob("stage_*.state"))
        out["curriculum_stage"] = len(stages)
    else:
        out["curriculum_stage"] = 0

    # Latest eval row.
    eval_log = ckpt_dir / "eval.jsonl"
    if eval_log.exists():
        try:
            with open(eval_log) as f:
                lines = [ln for ln in f if ln.strip()]
            if lines:
                latest = json.loads(lines[-1])
                out["last_eval"] = {
                    "mean_return": latest.get("mean_return"),
                    "mean_max_byte": latest.get("mean_max_byte"),
                    "clear_rate": latest.get("clear_rate"),
                    "timestamp": latest.get("timestamp"),
                }
        except Exception:
            pass

    return out


def main() -> int:
    rows = [summarize(key, name) for key, name in GAMES]

    # Pretty table.
    print(f"{'game':<14} {'status':<10} {'iter':>6} {'stages':>7} "
          f"{'mean_max_byte':>14} {'clear%':>7}")
    print("-" * 70)
    for r in rows:
        iter_str = str(r.get("latest_iter") or "-")
        stage_str = str(r.get("curriculum_stage", 0))
        mb = r.get("last_eval", {}).get("mean_max_byte")
        mb_str = f"{mb:.1f}" if mb is not None else "-"
        cr = r.get("last_eval", {}).get("clear_rate")
        cr_str = f"{cr*100:.1f}%" if cr is not None else "-"
        print(f"{r['game']:<14} {r['status']:<10} {iter_str:>6} "
              f"{stage_str:>7} {mb_str:>14} {cr_str:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
