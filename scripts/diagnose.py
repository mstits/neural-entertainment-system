"""
Training diagnostic: `scripts/diagnose.py [checkpoint]` produces a report
covering:

  * Best genome's action distribution over N replay steps
  * Map cells visited + Link's position range
  * Reward breakdown (which signals fire) during replay
  * Plateau detection against the metrics log
  * Whether the policy has collapsed to a single action

Run after any training session to quickly answer "is it actually learning"
without firing up the GUI.

Usage:
    python scripts/diagnose.py                           # auto latest
    python scripts/diagnose.py checkpoints/gen_00030.pt  # specific
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_latest(checkpoint_dir: Path) -> Path | None:
    files = sorted(checkpoint_dir.glob("gen_*.pt"))
    return files[-1] if files else None


def _replay_report(rom: str, profile_path: str, ckpt_path: str,
                   start_state: str | None, n_steps: int = 500) -> dict:
    import nes_core

    from src.emulation.frame_utils import (
        BUTTON_A, BUTTON_B, BUTTON_DOWN, BUTTON_LEFT, BUTTON_NOOP,
        BUTTON_RIGHT, BUTTON_SELECT, BUTTON_START, BUTTON_UP,
        FrameStacker,
    )

    NESEnvironment = nes_core.NESEnvironment
    from src.models.policy_network import PolicyNetwork
    from src.utils.reward_functions import build_reward_function

    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    action_space = profile["action_space"]
    num_actions = len(action_space)

    name_to_bit = {
        "NOOP": BUTTON_NOOP, "A": BUTTON_A, "B": BUTTON_B,
        "up": BUTTON_UP, "down": BUTTON_DOWN,
        "left": BUTTON_LEFT, "right": BUTTON_RIGHT,
        "start": BUTTON_START, "select": BUTTON_SELECT,
    }

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    genomes = sorted(data["population"],
                     key=lambda g: g.get("fitness", float("-inf")),
                     reverse=True)
    best = genomes[0]

    # Infer num_actions from the saved state_dict — if the checkpoint was
    # produced with a different action-space size (e.g., before we removed
    # Start from the Zelda profile), we need to match it to load weights.
    ckpt_num_actions = best["state_dict"]["fc2.bias"].shape[0]
    if ckpt_num_actions != num_actions:
        print(
            f"NOTE: checkpoint has {ckpt_num_actions} actions but active "
            f"profile '{profile['name']}' has {num_actions}. Running "
            f"replay with checkpoint's action space for diagnostics only."
        )
        # Pad action_space to match if shorter, truncate if longer.
        while len(action_space) < ckpt_num_actions:
            action_space.append([])
        action_space = action_space[:ckpt_num_actions]
        num_actions = ckpt_num_actions

    net = PolicyNetwork(num_actions=num_actions)
    net.load_state_dict(best["state_dict"])
    net.eval()

    env = NESEnvironment(
        rom_path=rom, frame_skip=4,
        start_state_path=start_state,
    )
    first = env.reset()
    stacker = FrameStacker()
    stacker.reset(first)

    reward_fn = build_reward_function(profile)
    reward_fn.reset()

    _solve = profile.get("solve") or {}
    _px = (_solve.get("progress") or {}).get("lo")
    _py = _solve.get("y")
    pos_addrs = (int(_px), int(_py)) if _px is not None and _py is not None else None

    actions = Counter()
    positions = []
    map_cells = set()
    total_reward = 0.0

    for step in range(n_steps):
        obs = stacker.push(env.get_frame())
        obs_t = torch.from_numpy(obs).float().div_(255.0)
        # Stochastic sampling so a degenerate softmax still reveals itself
        # via near-100% concentration on one action.
        a = net.act(obs_t, deterministic=False)
        actions[a] += 1
        bitmask = 0
        for b in action_space[a]:
            # Skip unknown button names instead of KeyError-ing the whole
            # diagnostic. Custom profiles sometimes use buttons not in the
            # standard mapping; warning is more useful than crashing here.
            bit = name_to_bit.get(b)
            if bit is None:
                continue
            bitmask |= bit
        frame, done = env.step(bitmask)
        ram = env.get_ram_range(0, 2048).tobytes()
        # Pass the action so reward functions that key off it (e.g. Zelda
        # wall-bump) score this replay accurately.
        r, rew_done, level_id = reward_fn.compute(ram, action=bitmask)
        total_reward += r
        # Position / map. The coordinate pair comes from the profile's own
        # solve block (the addresses it declares and receipts), never from
        # a display-name substring picking hand-authored constants: this
        # used to read `0x0070`/`0x0084` whenever the name contained
        # "zelda", and both of those are addresses under this project's
        # purity quarantine. No solve coordinates declared -> no position
        # column, which is a missing diagnostic, not a wrong one.
        if pos_addrs is not None:
            positions.append((env.get_ram(pos_addrs[0]),
                              env.get_ram(pos_addrs[1])))
        map_cells.add(level_id)
        if done or rew_done:
            break
    env.close()

    top_action = actions.most_common(1)[0] if actions else (None, 0)
    top_pct = 100.0 * top_action[1] / max(1, sum(actions.values()))

    return {
        "best_genome_id": best["genome_id"],
        "best_fitness": best.get("fitness", 0.0),
        "replay_steps": sum(actions.values()),
        "replay_reward": round(total_reward, 3),
        "distinct_actions": len(actions),
        "top_action_pct": round(top_pct, 1),
        "top_action_idx": top_action[0],
        "action_distribution": {
            str(idx): pct for idx, pct in zip(
                [i for i, _ in actions.most_common()],
                [100.0 * n / sum(actions.values()) for _, n in actions.most_common()],
            )
        },
        "map_cells_visited": sorted(map_cells),
        "distinct_map_cells": len(map_cells),
        "distinct_positions": len(set(positions)),
        "reward_breakdown": dict(getattr(reward_fn, "breakdown", {})),
    }


def _plateau_report(metrics_path: Path, window: int = 10) -> dict:
    """Is best-fitness climbing, flat, or regressing?"""
    if not metrics_path.exists():
        return {"status": "no-metrics", "path": str(metrics_path)}
    lines = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return {"status": "too-few-gens", "count": len(lines)}

    bests = [m.get("best_fitness", 0.0) for m in lines]
    recent = bests[-window:]
    prev = bests[-2 * window : -window] if len(bests) >= 2 * window else None
    trend = "climbing" if recent[-1] > recent[0] else "flat-or-regressing"
    delta = recent[-1] - recent[0]

    info = {
        "generations": len(lines),
        "best_first": bests[0],
        "best_last": bests[-1],
        "last_window_avg": round(sum(recent) / len(recent), 2),
        "last_window_delta": round(delta, 2),
        "trend": trend,
    }
    if prev:
        info["prev_window_avg"] = round(sum(prev) / len(prev), 2)
        info["generation_over_generation_delta"] = round(
            (sum(recent) / len(recent)) - (sum(prev) / len(prev)), 2
        )
        if abs(info["generation_over_generation_delta"]) < 2.0:
            info["status"] = "plateau"
        elif info["generation_over_generation_delta"] > 0:
            info["status"] = "learning"
        else:
            info["status"] = "regressing"
    return info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--profile", default="configs/zelda.yaml")
    p.add_argument("--rom", default="roms/zelda.nes")
    p.add_argument("--start-state", default="roms/zelda.ines1_start.state.bin")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--metrics", default="checkpoints/metrics.jsonl")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    ckpt = Path(args.checkpoint) if args.checkpoint else _load_latest(Path("checkpoints"))
    if ckpt is None or not ckpt.exists():
        print("No checkpoint found.", file=sys.stderr)
        return 1

    print(f"== Diagnostic for {ckpt} ==\n")

    print("-- Plateau / progress --")
    prog = _plateau_report(Path(args.metrics))
    print(json.dumps(prog, indent=2))
    print()

    print(f"-- Replay ({args.steps} steps from best genome) --")
    if Path(args.rom).exists():
        rpt = _replay_report(
            args.rom, args.profile, str(ckpt),
            args.start_state if Path(args.start_state).exists() else None,
            n_steps=args.steps,
        )
        # Pretty-print truncated action distribution
        rpt["action_distribution"] = {
            k: round(v, 1) for k, v in list(rpt["action_distribution"].items())[:5]
        }
        rpt["map_cells_visited"] = rpt["map_cells_visited"][:8] + (
            ["…"] if len(rpt["map_cells_visited"]) > 8 else []
        )
        print(json.dumps(rpt, indent=2))

        # Terse verdict
        if rpt["top_action_pct"] > 70:
            print("\nVERDICT: action collapse — policy picks one action >70%.")
        elif rpt["distinct_map_cells"] <= 1:
            print("\nVERDICT: stuck in one screen — no exploration.")
        else:
            print("\nVERDICT: multi-action, multi-cell behavior — training has signal.")
    else:
        print(f"rom {args.rom} not found; skipping replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
