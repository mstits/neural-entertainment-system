"""Select the online-campaign restart set from a full minted ladder.

`scripts/mint_backward_states.py` verifies a banked solver tape end-to-end
(it aborts unless the replay reaches the banked clear on THIS machine
lineage) and mints one savestate per action step. This script reduces that
dense ladder to the sparse rung set the 1-2 online campaign restarts from
— x ~= 2500 / 2000 / 1500 / 1000 / 500 / 0 — and writes:

* `index.json` in the same schema `backward_curriculum.load_index` reads
  (the trainer's `reinforce.backward_curriculum.states_dir` points here);
* `manifest.json` with per-state provenance: target x, actual gx, tape
  step, source solution/root, and the sha256 of every state blob.

Segment rule: 1-2 re-bases its x odometer twice (overworld -> underground,
underground -> exit), so "gx 2500" only means the bottleneck approach
inside the UNDERGROUND segment. Non-zero targets resolve within the
earliest segment whose max gx covers the largest target; target 0 is the
ladder head (tape step 0, the true level entrance). START STATES ONLY —
no action labels leave the ladder (Dossier v3: naive BC on these tapes
was eliminated; the tape is a curriculum, not a teacher).

Usage:
  python scripts/select_restart_states.py \
      --ladder <minted ladder dir> \
      --out checkpoints/online_1_2/restart_states \
      --targets 2500,2000,1500,1000,500,0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.training.backward_curriculum import (  # noqa: E402
    DEFAULT_GX_RESET_MAX, DEFAULT_GX_TOLERANCE, StateEntry, load_index,
    write_index,
)

DEFAULT_TARGETS = (2500, 2000, 1500, 1000, 500, 0)
# Largest |gx - target| accepted before the selection is an error. One
# ladder stride at full run speed is ~12 px; 64 px (4 tiles) is generous
# without letting a target silently land on the wrong side of a gap.
DEFAULT_TOLERANCE_PX = 64


def split_segments(
    entries: Sequence[StateEntry],
    tolerance: int = DEFAULT_GX_TOLERANCE,
    reset_max: int = DEFAULT_GX_RESET_MAX,
) -> list[list[StateEntry]]:
    """Split a ladder at odometer re-bases — same rule as `gx_report`.

    A new segment opens on an area-byte change, or on a gx drop beyond
    `tolerance` that lands at or below `reset_max` (the coordinate
    wrapped at a pipe/vine/castle; 1-2's second re-base keeps area 2).
    """
    segs: list[list[StateEntry]] = []
    for prev, cur in zip([None] + list(entries), entries):
        rebase = prev is not None and (
            prev.area != cur.area
            or (prev.gx - cur.gx > tolerance and cur.gx <= reset_max)
        )
        if prev is None or rebase:
            segs.append([])
        segs[-1].append(cur)
    return segs


def pick_bottleneck_segment(
    segments: Sequence[Sequence[StateEntry]], max_target: int,
) -> list[StateEntry]:
    """The EARLIEST segment whose gx range covers the deepest target.

    Earliest, because a later re-based segment (1-2's exit run) can reach
    the same gx values in a different room — a rung minted there would
    "approach the bottleneck" from the wrong side of the pipe.
    """
    for seg in segments:
        if seg and max(e.gx for e in seg) >= max_target:
            return list(seg)
    spans = [(s[0].step, max(e.gx for e in s)) for s in segments if s]
    raise ValueError(
        f"no segment covers gx {max_target}: segment (start step, max gx) "
        f"pairs are {spans}"
    )


def select_entries(
    entries: Sequence[StateEntry],
    targets: Sequence[int] = DEFAULT_TARGETS,
    *,
    tolerance_px: int = DEFAULT_TOLERANCE_PX,
) -> list[tuple[int, StateEntry]]:
    """(target, entry) per target, sorted by tape step (ascending x).

    Target 0 is the ladder head — the true entrance — so tau 0 in the
    TauScheduler IS the honest start. All other targets pick the
    nearest-gx entry inside the bottleneck segment and must land within
    `tolerance_px` of their target.
    """
    entries = sorted(entries, key=lambda e: e.step)
    if not entries:
        raise ValueError("empty ladder: nothing to select from")
    targets = sorted(set(int(t) for t in targets))
    nonzero = [t for t in targets if t != 0]
    picked: list[tuple[int, StateEntry]] = []
    if 0 in targets:
        picked.append((0, entries[0]))
    if nonzero:
        seg = pick_bottleneck_segment(split_segments(entries), max(nonzero))
        for t in nonzero:
            best = min(seg, key=lambda e: (abs(e.gx - t), e.step))
            if abs(best.gx - t) > tolerance_px:
                raise ValueError(
                    f"target x={t}: nearest rung in the bottleneck segment "
                    f"is gx={best.gx} (step {best.step}), off by "
                    f"{abs(best.gx - t)} px > tolerance {tolerance_px}"
                )
            picked.append((t, best))
    picked.sort(key=lambda te: te[1].step)
    steps = [e.step for _, e in picked]
    if len(set(steps)) != len(steps):
        raise ValueError(
            f"two targets resolved to the same rung (steps {steps}); "
            f"widen the target spacing or re-mint a denser ladder"
        )
    return picked


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ladder", required=True,
                    help="Full minted ladder dir (mint_backward_states.py "
                         "output with index.json).")
    ap.add_argument("--out", required=True,
                    help="Restart-states dir the campaign config points at.")
    ap.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)),
                    help="Comma-separated gx targets (default "
                         "2500,2000,1500,1000,500,0).")
    ap.add_argument("--tolerance-px", type=int, default=DEFAULT_TOLERANCE_PX)
    args = ap.parse_args()

    targets = tuple(int(t) for t in args.targets.split(","))
    entries, meta = load_index(args.ladder)
    picked = select_entries(entries, targets, tolerance_px=args.tolerance_px)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = []
    for target, e in picked:
        src = Path(args.ladder) / e.file
        dst = out_dir / e.file
        shutil.copyfile(src, dst)
        states.append({
            "target_x": target,
            "gx": e.gx,
            "area": e.area,
            "step": e.step,
            "frame": e.frame,
            "file": e.file,
            "sha256": _sha256(dst),
        })

    sel_meta = dict(meta)
    sel_meta.pop("dir", None)
    sel_meta["selection"] = {
        "targets": list(targets),
        "tolerance_px": int(args.tolerance_px),
        "from_ladder": str(args.ladder),
        "selected_at": time.time(),
    }
    write_index(out_dir, [e for _, e in picked], sel_meta)

    # Resolve symlinks so provenance names the BANKED tape, not a staging
    # alias (the minter may have been pointed at a symlinked run layout).
    actions_path = meta.get("actions")
    if actions_path and Path(actions_path).exists():
        actions_path = str(Path(actions_path).resolve())
    manifest = {
        "purpose": "SMB 1-2 online-campaign restart states "
                   "(backward-curriculum rungs at the x~2674 bottleneck "
                   "approach)",
        "source_solution": {
            "actions": actions_path,
            "root_state": meta.get("root_state"),
            "profile": meta.get("profile"),
            "rom": meta.get("rom"),
            "reached_clear": meta.get("reached_clear"),
            "tail": meta.get("tail"),
            "hw": meta.get("hw"),
            "minted_at": meta.get("minted_at"),
        },
        "targets": list(targets),
        "tolerance_px": int(args.tolerance_px),
        "states": states,
        "created_at": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for s in states:
        print(f"[select] x={s['target_x']:>5} -> gx {s['gx']:>5} "
              f"(step {s['step']:>4}, area {s['area']}) {s['file']} "
              f"sha256 {s['sha256'][:16]}", flush=True)
    print(f"[select] {len(states)} restart states + index.json + "
          f"manifest.json -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
