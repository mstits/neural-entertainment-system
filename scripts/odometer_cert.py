#!/usr/bin/env python3
"""Odometer certification: five automated checks on a real game.

The PPU scroll odometer (v3 savestate envelope) is only trusted as a
progress observable after ALL five checks pass on a game with a known
ground truth (SMB: HUD split + horizontal scroll + hold-still start).

  1. hold-forward monotonicity  — odometer_x rises substantially and
     never regresses by more than a jitter budget while holding right
  2. hold-still exact flatness  — zero input => odometer stays exactly 0
  3. HUD-split immunity         — per-step |dx| stays bounded while the
     status bar (fixed scroll=0 region) is on screen every frame; a
     modal-filter failure shows up as ±screen-width spikes
  4. discontinuity behaviour    — savestate load mid-run produces NO
     spurious mega-delta (restore replaces anchor, never integrates)
  5. restore exactness          — odometer reads back the saved value
     to the digit after a load_worker_state round-trip

Fail any => the build is quarantined (exit 1, verdict json says which).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(profile: str, steps: int, out: str | None) -> int:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np, yaml, nes_core
    from src.training.profile_utils import action_space_to_bitmasks

    prof = yaml.safe_load((REPO / profile).read_text())
    rom = prof.get("rom_path") or (prof.get("solve") or {}).get("rom")
    start = prof["start_state_path"]
    space = prof["action_space"]
    bm = action_space_to_bitmasks(space)
    right = next((i for i, a in enumerate(space) if a == ["right"]), 1)

    def fresh_pool():
        pool = nes_core.Pool(rom_path=str(REPO / rom), num_workers=1,
                             frame_skip=int(prof.get("frame_skip", 4)))
        pool.set_headless(True)
        pool.set_skip_preprocess(True)
        pool.set_odometer_enabled(True)
        pool.load_worker_state(0, (REPO / start).read_bytes())
        return pool

    checks: dict[str, dict] = {}
    noop = np.zeros(1, dtype=np.uint8)

    # -- 1+3: hold forward; track per-step deltas ------------------------
    pool = fresh_pool()
    a = np.array([bm[right]], dtype=np.uint8)
    xs = []
    for _ in range(steps):
        pool.step_all(a)
        xs.append(pool.get_odometer_per_worker()[0][0])
    xs = np.array(xs)
    dxs = np.diff(xs)
    total = int(xs[-1] - xs[0])
    max_regress = int(-(dxs.min())) if len(dxs) and dxs.min() < 0 else 0
    checks["hold_forward_monotonic"] = {
        "passed": total > 200 and max_regress <= 16,
        "total_dx": total, "max_regress": max_regress,
    }
    # frame_skip frames per step -> generous bound; a modal failure on
    # the HUD region jumps by ~±256 in a single step.
    max_step = int(np.abs(dxs).max()) if len(dxs) else 0
    checks["hud_split_immunity"] = {
        "passed": max_step < 64,
        "max_abs_step_dx": max_step,
    }

    # -- 2: hold still ---------------------------------------------------
    pool = fresh_pool()
    for _ in range(min(steps, 300)):
        pool.step_all(noop)
    ox, oy = pool.get_odometer_per_worker()[0]
    checks["hold_still_flat"] = {
        "passed": ox == 0 and oy == 0, "x": int(ox), "y": int(oy),
    }

    # -- 4+5: restore mid-run --------------------------------------------
    pool = fresh_pool()
    for _ in range(120):
        pool.step_all(a)
    blob = pool.save_worker_state(0)
    saved = pool.get_odometer_per_worker()[0]
    for _ in range(120):
        pool.step_all(a)
    drifted = pool.get_odometer_per_worker()[0]
    pool.load_worker_state(0, blob)
    restored = pool.get_odometer_per_worker()[0]
    checks["restore_exact"] = {
        "passed": restored == saved and drifted != saved,
        "saved": list(saved), "drifted": list(drifted),
        "restored": list(restored),
    }
    pool.step_all(a)
    after = pool.get_odometer_per_worker()[0]
    jump = abs(after[0] - restored[0]) + abs(after[1] - restored[1])
    checks["no_restore_discontinuity"] = {
        "passed": jump < 64, "first_step_delta_after_restore": int(jump),
    }

    ok = all(c["passed"] for c in checks.values())
    verdict = {"profile": profile, "steps": steps,
               "passed": ok, "checks": checks}
    print(json.dumps(verdict, indent=2))
    print(f"\nodometer certification: {'PASS (5/5)' if ok else 'FAIL — QUARANTINE'}")
    for name, c in checks.items():
        print(f"  [{'ok' if c['passed'] else 'XX'}] {name}")
    if out:
        p = REPO / out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(verdict, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="configs/mario_1_1_backward.yaml")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default=None)
    raise SystemExit(run(**vars(ap.parse_args())))
