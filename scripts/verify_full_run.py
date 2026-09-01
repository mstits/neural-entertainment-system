"""Reproducible cold-boot replay of the banked flagship tape.

Same per-frame path as scripts/play_tape.py's --headless-verify (cold
boot: NESEnvironment(ROM) -> reset() -> step() only, no state loads),
plus the per-level world/level checks from the out-of-repo probe at
reports/macos-emulation-and-training/2026-09-01-ground-truth-execution/
receipts/per_level_probe.py, folded into this repo as a committed,
re-runnable script. Writes a JSON receipt that records the exact build
(git commit + installed .so identity) it ran against, so a stale
`.venv` build can never be passed off as a fresh verification.

Usage:
  python scripts/verify_full_run.py                 # writes
      docs/receipts/full_run/replay_<YYYY-MM-DD>.json
  python scripts/verify_full_run.py --out path.json # explicit output

Exit 1 if any level boundary mismatches or the final opermode != 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nes_core  # noqa: E402

ROM = REPO / "roms/Super Mario Bros. (World).nes"
TAPE = REPO / "docs/receipts/full_run/full_tape.npy"
RECEIPTS = REPO / "docs/receipts/full_run/receipts.json"
FRAME_SKIP = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    so_path = Path(nes_core.__file__).resolve()
    if so_path.name == "__init__.py":
        so_path = so_path.parent / "nes_core.abi3.so"
    return {
        "git_commit": commit,
        "so_path": str(so_path),
        "so_mtime": so_path.stat().st_mtime,
        "so_sha256": sha256_file(so_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                     help="output path (default: "
                          "docs/receipts/full_run/replay_<today>.json)")
    args = ap.parse_args()

    rc = json.loads(RECEIPTS.read_text())
    tape = np.load(TAPE)
    tape_sha256 = hashlib.sha256(tape.tobytes()).hexdigest()
    if tape_sha256 != rc["tape_sha256"]:
        print(f"*** tape hash mismatch: {tape_sha256} != "
              f"{rc['tape_sha256']}", file=sys.stderr)
        return 1

    marks = {lv["end_step"]: lv for lv in rc["levels"]}
    build = build_provenance()
    print(f"[verify] build: commit={build['git_commit'][:12]} "
          f"so={build['so_path']} sha256={build['so_sha256'][:16]}...",
          flush=True)

    env = nes_core.NESEnvironment(str(ROM))
    env.reset()
    levels = []
    t0 = time.perf_counter()
    for i, mask in enumerate(tape):
        for _ in range(FRAME_SKIP):
            env.step(int(mask))
        step = i + 1
        if step in marks:
            lv = marks[step]
            w = int(env.get_ram(0x75F))
            l = int(env.get_ram(0x75C))
            if "wd_after" in lv:
                observed = [w, l]
                expected = lv["wd_after"]
            else:
                observed = int(env.get_ram(0x770))
                expected = 2
            ok = observed == expected
            levels.append({
                "level": lv["level"],
                "end_step": step,
                "wd_after_expected": expected,
                "wd_after_observed": observed,
                "ok": ok,
            })
            print(f"[verify] {lv['level']:<20} end_step={step:<6} "
                  f"expected={expected} observed={observed} "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
        if i % 4000 == 0:
            print(f"[verify] step {i}/{len(tape)}", flush=True)

    wall_s = time.perf_counter() - t0
    opermode = int(env.get_ram(0x770))
    all_ok = (
        opermode == 2
        and len(levels) == len(marks)
        and all(lv["ok"] for lv in levels)
    )

    receipt = {
        "rom": str(ROM.relative_to(REPO)),
        "rom_sha256": sha256_file(ROM),
        "tape_sha256": tape_sha256,
        "build": build,
        "cmd": " ".join([sys.executable] + sys.argv),
        "wall_s": wall_s,
        "opermode": opermode,
        "state_loads": 0,
        "cold_boot": True,
        "levels": levels,
        "all_ok": all_ok,
    }

    if args.out:
        out = Path(args.out)
    else:
        out = REPO / "docs/receipts/full_run" / (
            f"replay_{time.strftime('%Y-%m-%d')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[verify] done in {wall_s:.0f}s: opermode={opermode} "
          f"all_ok={all_ok} -> wrote {out}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
