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


def auto_targets(
    entries: Sequence[StateEntry],
    n_rungs: int = 6,
    *,
    round_to: int = 50,
) -> tuple[int, ...]:
    """Evenly spaced gx targets for a level whose range is not known.

    The 1-2 targets (2500/2000/1500/1000/500/0) were hand-chosen around a
    MEASURED barrier. A level with no measured barrier must not get
    invented coordinates, so this derives the ladder entirely from our
    own solver tape: take one segment's max gx (the tape's reach inside
    that segment), lay `n_rungs - 1` rungs at k/n_rungs of it for
    k = n_rungs-1 .. 1, floor each to a `round_to` multiple, and append 0
    (the true entrance, which is always the ladder head).

    WHICH segment is not free. `select_entries` resolves every non-zero
    target inside `pick_bottleneck_segment` — the EARLIEST segment whose
    range covers the deepest target — so if the scale came from a later,
    deeper segment the rungs would be measured against one room and
    minted in another. That is not hypothetical: on the banked 1-2
    ladder the segments reach gx 160 / 2674 / 3175, and scaling off the
    deepest (3175, the re-based exit run) puts the top target at 2600,
    which the earliest-covering rule then resolves back inside the 2674
    underground segment. Scale and rungs would disagree silently.

    So the segment is chosen to a FIXED POINT: start from the deepest,
    and while the earliest segment covering the resulting top target is
    an earlier one, re-scale off that earlier segment. The iteration only
    ever moves earlier, so it terminates in at most `len(segments)`
    rounds, and it ends on exactly the segment `select_entries` will use.
    On a single-segment tape (SMB 1-3 never leaves its area byte) the
    first round is already the fixed point and this is a no-op.

    Returned descending, matching DEFAULT_TARGETS' order.
    """
    if n_rungs < 2:
        raise ValueError(f"n_rungs must be >= 2, got {n_rungs}")
    if round_to < 1:
        raise ValueError(f"round_to must be >= 1, got {round_to}")
    if not entries:
        raise ValueError("empty ladder: nothing to select from")
    segs = [s for s in split_segments(sorted(entries, key=lambda e: e.step))
            if s]

    def _scale(seg):
        max_gx = max(e.gx for e in seg)
        out = [int(max_gx * k / n_rungs) // round_to * round_to
               for k in range(n_rungs - 1, 0, -1)]
        out.append(0)
        return max_gx, out

    seg = max(segs, key=lambda s: max(e.gx for e in s))
    for _ in range(len(segs)):
        max_gx, targets = _scale(seg)
        top = max(t for t in targets if t) if any(targets) else 0
        if top <= 0:
            break
        owner = pick_bottleneck_segment(segs, top)
        if owner[0].step >= seg[0].step:
            break
        seg = owner
    else:                                   # pragma: no cover - unreachable
        raise ValueError("auto targets did not settle on a segment")
    if len(set(targets)) != len(targets):
        raise ValueError(
            f"auto targets collapsed to {targets} (max gx {max_gx} is too "
            f"small for {n_rungs} rungs at {round_to}px granularity); pass "
            f"explicit --targets or lower --round-to")
    return tuple(targets)


def check_ladder_provenance(
    meta, *, level=None, root_state=None, ladder="<ladder>",
) -> None:
    """Raise unless a ladder's stamped origin is the one we asked for.

    The mint stamps `level` / `root_state` / `profile` into index.json
    meta and this script copies them forward, but NOTHING downstream
    reads them, so a ladder minted by another lane for another level
    from another root selects cleanly and ships. That has a live on-disk
    example: `checkpoints/backward_states/1-2`'s meta names root
    `.../smb_4_4_micro/entrance_after_1-1.state` and profile
    `configs/smb_4_4_micro.yaml` over a 1215-action tape, while the 1-2
    campaign starts at `stage_03.state` and shipped an 871-action
    stage_03-rooted selection.

    Both checks are opt-in (the default path is unchanged) but a ladder
    that CANNOT prove its origin fails whichever check was requested —
    a missing stamp is not a pass.
    """
    meta = meta if isinstance(meta, dict) else {}
    if level is not None:
        got = meta.get("level")
        if got is None or str(got) != str(level):
            raise SystemExit(
                f"[select] {ladder}: ladder level {got!r} != expected "
                f"{level!r} — this is another level's ladder")
    if root_state is not None:
        got = meta.get("root_state")
        if got is None or Path(str(got)).resolve() != Path(root_state).resolve():
            raise SystemExit(
                f"[select] {ladder}: ladder root_state {got!r} != expected "
                f"{root_state!r} — this ladder starts in a different room")


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
    ap.add_argument("--auto-targets", type=int, default=None, metavar="N",
                    help="Derive N evenly spaced targets from the ladder's "
                         "own deepest segment instead of --targets (for a "
                         "level with no measured barrier).")
    ap.add_argument("--round-to", type=int, default=50,
                    help="Granularity auto targets are floored to (px).")
    ap.add_argument("--expect-level", default=None, metavar="W-L",
                    help="Refuse a ladder whose index.json meta names a "
                         "different level. Pass this whenever --ladder "
                         "points outside your own lane.")
    ap.add_argument("--expect-root", default=None, metavar="PATH",
                    help="Refuse a ladder minted from a different root "
                         "blob than the campaign profile starts at.")
    args = ap.parse_args()

    entries, meta = load_index(args.ladder)
    check_ladder_provenance(meta, level=args.expect_level,
                            root_state=args.expect_root, ladder=args.ladder)
    if args.auto_targets is not None:
        targets = auto_targets(entries, args.auto_targets,
                               round_to=args.round_to)
        print(f"[select] auto targets ({args.auto_targets} rungs): "
              f"{list(targets)}", flush=True)
    else:
        targets = tuple(int(t) for t in args.targets.split(","))
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
    level = str(meta.get("level") or "1-2")
    manifest = {
        "purpose": (
            f"SMB {level} online-campaign restart states "
            f"(backward-curriculum rungs at gx {list(targets)})"),
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
