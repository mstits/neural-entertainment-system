"""Multi-game scoreboard — mission control across all six games.

Joins three per-game artifacts under checkpoints/<game_slug>/:
  * run_manifest.json  — provenance (mode, device, seed, git commit)
  * metrics.jsonl      — live training progress (best fitness, clears,
                         entropy, throughput) from the current run
  * eval.jsonl         — latest eval (clear rate, furthest stage)

into one developer-facing "where are we across all games" view. Reads
only on-disk artifacts (no training), so it's safe to run any time.

Phase 0 acceptance: produces a multi-game summary even if only one game
has been trained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.profile_utils import profile_slug  # noqa: E402

# Single source of truth for "the 6 games" — keep in lockstep with the
# thesis doc and DEFAULT_PROFILES in train_game.py.
GAMES = [
    ("mario", "Super Mario Bros."),
    ("contra", "Contra"),
    ("megaman", "Mega Man"),
    ("castlevania", "Castlevania"),
    ("zelda", "The Legend of Zelda"),
    ("metroid", "Metroid"),
]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def summarize(game_key: str, display_name: str, base: Path = Path("checkpoints")) -> dict:
    slug = profile_slug(display_name)
    ckpt_dir = Path(base) / slug
    out: dict = {"game": game_key, "display": display_name, "slug": slug,
                 "dir": str(ckpt_dir), "status": "untrained"}
    if not ckpt_dir.exists():
        return out

    iters = sorted(
        ckpt_dir.glob("vanilla_ppo_iter_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    out["status"] = "trained" if iters else "untrained"
    out["latest_iter"] = int(iters[-1].stem.rsplit("_", 1)[-1]) if iters else None
    out["n_checkpoints"] = len(iters)

    # Curriculum stages captured (SMB).
    curr = ckpt_dir / "smb_curriculum"
    out["curriculum_stage"] = (
        len(sorted(curr.glob("stage_*.state"))) if curr.exists() else 0
    )

    # Live training metrics (current run).
    rows = _read_jsonl(ckpt_dir / "metrics.jsonl")
    if rows:
        out["status"] = "trained"
        out["n_iters"] = len(rows)
        out["best_fitness"] = max(
            (float(r.get("best_fitness", 0.0) or 0.0) for r in rows), default=0.0
        )
        out["total_clears"] = int(sum(
            int(r.get("vanilla_ppo_clears", 0) or 0) for r in rows
        ))
        last = rows[-1]
        out["last_gen"] = last.get("generation")
        out["last_entropy"] = last.get("ppo_entropy")
        out["realtime_x"] = last.get("vanilla_ppo_realtime_x")

    # Provenance.
    man = _read_json(ckpt_dir / "run_manifest.json")
    if man:
        out["mode"] = man.get("trainer_mode")
        out["device"] = man.get("device")
        out["seed"] = man.get("seed")
        out["commit"] = man.get("git_commit")

    # Latest eval.
    evals = _read_jsonl(ckpt_dir / "eval.jsonl")
    if evals:
        e = evals[-1]
        out["eval_clear_rate"] = e.get("clear_rate")
        out["eval_max_byte"] = e.get("mean_max_byte")

    return out


def _fmt(v, spec="", dash="-"):
    if v is None:
        return dash
    try:
        return format(v, spec)
    except (ValueError, TypeError):
        return str(v)


def main() -> int:
    rows = [summarize(k, n) for k, n in GAMES]

    print(f"{'game':<13}{'status':<9}{'iter':>7}{'best_fit':>10}"
          f"{'clears':>8}{'entropy':>9}{'RTx':>7}{'stage':>6}{'eval%':>7}")
    print("-" * 76)
    for r in rows:
        rtx = r.get("realtime_x")
        cr = r.get("eval_clear_rate")
        print(
            f"{r['game']:<13}"
            f"{r['status']:<9}"
            f"{_fmt(r.get('latest_iter')):>7}"
            f"{_fmt(r.get('best_fitness'), '.0f'):>10}"
            f"{_fmt(r.get('total_clears')):>8}"
            f"{_fmt(r.get('last_entropy'), '.2f'):>9}"
            f"{(_fmt(rtx, '.0f') + 'x') if rtx is not None else '-':>7}"
            f"{r.get('curriculum_stage', 0):>6}"
            f"{(_fmt(cr * 100, '.0f') + '%') if cr is not None else '-':>7}"
        )

    # Provenance footer for trained games.
    trained = [r for r in rows if r.get("commit") or r.get("mode")]
    if trained:
        print("\nprovenance (run_manifest.json):")
        for r in trained:
            print(f"  {r['game']:<13} mode={r.get('mode','?')} "
                  f"device={r.get('device','?')} seed={r.get('seed')} "
                  f"commit={r.get('commit','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
