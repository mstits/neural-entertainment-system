"""Non-cheating wavefront potential reward (Potential-Based Reward Shaping).

Implements the dense distance-to-goal shaping from the research brief
(docs/research/LEARNED_STICKY_DISTILLATION_BRIEF_2026-07-22.md) WITHOUT any
game-internals / disassembly: the distance map D(cell) is derived purely from
Go-Explore's own VERIFIED solution trajectories (search output), not from the
ROM's level layout.

Adaptation from the brief: the brief builds Dijkstra over the Go-Explore cell
archive's transition graph. Our archive stores cells + per-cell action traces
but not explicit edges, and reconstructing edges means replaying 8k traces. The
verified SOLUTION trajectories already give a monotone distance-to-goal
(remaining steps), which is exactly the geodesic potential PBRS needs along the
reachable path. Off-path recovery is owned by DARS (generate_dars_recovery.py),
so this module only needs the on/near-path potential. D(cell) = min over all
solution visits of (steps remaining to the clear). Off-map states project to the
nearest same-area cell by x-bucket (a sensible potential that still points
"forward"), or fall back to the level's max distance (far from goal).

PBRS invariance (Ng et al. 1999): F(s,a,s') = gamma*Phi(s') - Phi(s) with
Phi(s) = -alpha*D(cell(s)) leaves the optimal policy unchanged (no reward
loops), while densifying the gradient so PPO isn't starved over a long horizon.

Spatial cell = (area_byte, x_abs // XB, y // YB) from RAM — same coordinates the
solver/eval already read, no new game knowledge.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# RAM addresses (SMB) — the same ones go_explore_solve.py / eval_game.py use.
R_X_PAGE, R_X_LOW = 0x006D, 0x0086
R_YPOS = 0x00CE
R_AREA = 0x0760

XB = 16   # x-bucket width (pixels) for the spatial cell
YB = 32   # y-band width


def _x(ram) -> int:
    return (int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW])


def wave_cell(ram) -> tuple:
    """Spatial cell key from RAM: (area, x-bucket, y-band). Game-agnostic."""
    return (int(ram[R_AREA]), _x(ram) // XB, int(ram[R_YPOS]) // YB)


def build_distance_map(
    solution_action_files,
    root_state_path: str,
    profile: dict,
    rom_path: str,
) -> dict:
    """Replay each solution from the entrance; return {cell: min_steps_to_goal}.

    Steps-to-goal for a state = (len(actions) - index_at_that_state): how many
    action-steps of the verified solution remain until the clear. Taking the MIN
    across all solutions and all visits gives the shortest verified distance to
    the goal for each reached cell — a geodesic-style potential over the
    reachable path.

    CRITICAL — transition-frame exclusion (fix 2026-07-22): SMB reads x==0 & y==0
    for a handful of frames whenever a level LOADS a new area (pipe/door
    transitions reset the scroll before the new area paints). Those loading
    frames all collapse to the same cell key (area, 0, 0) even though they occur
    at very different points along the trajectory — e.g. in 1-2 the area-2 entry
    (~700 steps remaining) AND a later internal pipe (~262 remaining) both read
    (2,0,0). The min-over-visits reduction then poisons the ENTRY cell with the
    LATER cell's small distance, so Phi spikes on area entry and every real step
    forward DROPS Phi (a ~-56 shaping cliff that actively punishes progress and
    traps the policy at the area boundary). Skipping x==0 & y==0 frames removes
    the aliasing; the real gameplay cells on either side carry the gradient and
    the runtime `_distance` projection covers the transition instant.
    """
    from nes_core import Pool
    from src.training.profile_utils import action_space_to_bitmasks
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    fs = int(profile.get("frame_skip", 4))
    root = Path(root_state_path).read_bytes()

    pool = Pool(rom_path=rom_path, num_workers=1, frame_skip=fs)
    pool.set_headless(True)
    dmap: dict = {}
    skipped = 0
    for f in solution_action_files:
        actions = np.load(f).astype(np.int64)
        n = len(actions)
        pool.reset_all()
        pool.load_worker_state(0, root)
        r = pool.step_all(np.zeros(1, dtype=np.uint8))  # convention NOOP
        ram = r[0][2]
        for i, a in enumerate(actions):
            remaining = n - i
            if _x(ram) == 0 and int(ram[R_YPOS]) == 0:
                skipped += 1  # loading/transition frame — do not record (aliases)
            else:
                c = wave_cell(ram)
                if c not in dmap or remaining < dmap[c]:
                    dmap[c] = remaining
            r = pool.step_all(np.array([bitmasks[int(a)]], dtype=np.uint8))
            ram = r[0][2]
        # goal cell: 0 remaining (only if it is a real, non-transition frame)
        if not (_x(ram) == 0 and int(ram[R_YPOS]) == 0):
            dmap[wave_cell(ram)] = 0
    pool.shutdown()
    print(f"[wavefront] build: skipped {skipped} loading/transition (x==0&y==0) "
          f"frames to avoid cell aliasing", flush=True)
    return dmap


@dataclass
class WavefrontPotential:
    """Dense distance-to-goal potential with NON-FARMABLE PBRS shaping.

    POSITIVE-SHIFTED formulation (research recipe, 2026-07-22): Phi(s) =
    phi_target * (1 - D(s)/D_start), so Phi in [0, phi_target], Phi(entrance)=0,
    Phi(goal)=phi_target, and Phi(s) >= 0 for all valid states. Combined with
    Phi(terminal)=0 on DEATH (F = gamma*0 - Phi(s) = -Phi(s) <= 0), this makes
    dying strictly non-positive in shaping — the telescoping sum over any
    non-completing path is <= 0, so progress-then-die can NOT be farmed. (The
    prior negative-Phi form was farmable: the greedy policy collected +168 of
    un-canceled approach shaping then died at gx 183 — confirmed by the 3-mode
    diagnostic 2026-07-22.)

    Off-map states project to the nearest same-area x-bucket (a drifted state
    still gets a forward-pointing potential); unseen area -> D_start (Phi 0)."""

    dmap: dict
    phi_target: float = 100.0
    gamma: float = 0.99
    d_start: float = 0.0
    _by_area: Optional[dict] = None

    def __post_init__(self):
        self._by_area = {}
        for (area, xb, yb), d in self.dmap.items():
            self._by_area.setdefault(area, []).append((xb, yb, d))
        for area in self._by_area:
            self._by_area[area].sort()
        # D_start = the max distance in the map (the level entrance).
        self.d_start = float(max(self.dmap.values(), default=1) or 1)

    @classmethod
    def load(cls, path, phi_target: float = 100.0, gamma: float = 0.99,
             alpha: float = None):
        # `alpha` accepted for backward-compat callers; ignored (positive-shift
        # uses phi_target instead of a raw scale).
        with open(path, "rb") as f:
            dmap = pickle.load(f)
        return cls(dmap=dmap, phi_target=float(phi_target), gamma=float(gamma))

    def _distance(self, ram) -> float:
        c = wave_cell(ram)
        d = self.dmap.get(c)
        if d is not None:
            return float(d)
        # Off-map projection is Y-AWARE (fix 2026-07-22): only inherit distance
        # from mapped cells within +/-1 y-band. The old x-only projection made
        # every altitude at a given x reward-equivalent — which handed the 1-2
        # CEILING route (yb 0-1, never visited by any solution) the floor
        # route's full forward gradient. Ceiling running has zero death risk,
        # so under sticky training the policy rationally took it — straight
        # into the warp-room dead end (area 0, off-map) where it wandered to
        # timeout every episode. With the y-tolerance, states outside the
        # expert trajectories' y-envelope read D_start (Phi=0): no gradient off
        # the route, and climbing off it costs -Phi immediately (a cliff the
        # value function can actually bootstrap, unlike a dead end ~200 steps
        # ahead). Jump arcs are unaffected — the solutions' own arcs put their
        # y-bands in the map.
        area = int(ram[R_AREA])
        cand = self._by_area.get(area)
        if not cand:
            return self.d_start
        xb = _x(ram) // XB
        yb = int(ram[R_YPOS]) // YB
        near = [t for t in cand if abs(t[1] - yb) <= 1]
        if not near:
            return self.d_start
        # Geodesic extension, not adoption: inherit the cell's D PLUS the
        # travel cost to reach it (~3.6 D per x-bucket along the solutions).
        # Without this, an off-envelope state borrows a distant cell's D for
        # free — the ceiling at xb54 read the expert's high-arc cell at xb32
        # and kept a forward gradient 22 buckets past it.
        step_x = self.d_start / 204.0  # measured: 742 D over ~204 xb
        proj = min(d + step_x * abs(cxb - xb) for cxb, cyb, d in near)
        return float(min(proj, self.d_start))

    def potential(self, ram) -> float:
        """Phi(s) = phi_target * (1 - D(s)/D_start), in [0, phi_target]."""
        return self.phi_target * (1.0 - self._distance(ram) / self.d_start)

    def shaped_increment(self, prev_ram, ram) -> float:
        """Live-transition F(s,a,s') = gamma*Phi(s') - Phi(s)."""
        return self.gamma * self.potential(ram) - self.potential(prev_ram)


def _cli():
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description="Build a wavefront distance map "
                                 "from Go-Explore solution traces.")
    ap.add_argument("--solutions", nargs="+", required=True,
                    help=".npy solution action files (all from the same root).")
    ap.add_argument("--root-state", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--rom", default="roms/Super Mario Bros. (World).nes")
    ap.add_argument("--out", required=True, help="Output .pkl distance map.")
    args = ap.parse_args()
    profile = yaml.safe_load(Path(args.profile).read_text())
    dmap = build_distance_map(args.solutions, args.root_state, profile, args.rom)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dmap, f, protocol=pickle.HIGHEST_PROTOCOL)
    ds = list(dmap.values())
    print(f"[wavefront] {len(dmap)} cells; dist range [{min(ds)}, {max(ds)}] "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    _cli()
