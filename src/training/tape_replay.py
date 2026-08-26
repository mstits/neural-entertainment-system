"""The repo's ONE banked-tape replay convention, in one place.

Every consumer of a solver tape — minting a backward-state ladder
(scripts/mint_backward_states.py), rebuilding demos
(scripts/replay_to_demos.py), extracting the next entrance
(go_explore_chain.extract_next_entrance), collecting a RAM trace for
objective discovery (scripts/discover_observables.find_progress_from_tapes)
— replays it the same way:

    Pool(rom, 1 worker, frame_skip) -> apply hw flags -> headless +
    skip_preprocess -> reset_all() -> load_worker_state(root blob) ->
    ONE no-op step (materializes RAM) -> one step per tape action, the
    action index resolved through the SOLVE profile's action_space
    bitmasks.

Three details in that sequence are load-bearing and each has cost a
debugging session in the past:

  * hw flags go on BEFORE reset_all(). `reset_alignment` changes boot
    cycle accounting, so setting it after the reset boots a different
    power-on lineage (measured 2026-08-07: RAM diverges at frame 11,
    $006F/$01FE). `apply_hw_flags` documents this; this module just
    never gets the order wrong.
  * the leading no-op is not cosmetic, and it must be an actual no-op.
    Without it the first RAM read is the pre-load buffer and every
    downstream index is off by one frame; with a reused controller
    buffer it replays the previous segment's last button into the new
    segment's settle frame (see `replay_segments`).
  * a tape only means anything against its OWN root blob and its OWN
    machine lineage. `load_root` runs the sidecar check
    (`<blob>.json` carrying `hw_flags`) so a mismatched lineage prints a
    named warning instead of masquerading as nondeterminism.

Row indexing: `play()` yields `(step, ram)` where `ram` is the memory
BEFORE action `step` runs, so `step` doubles as "actions applied so far".
The final yield (`step == len(actions)`) is the post-tape memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

RAM_SIZE = 0x800
NOOP = 0x00


def _ram(x) -> np.ndarray:
    """The pool hands RAM back as `bytes`; normalize to uint8[2048]."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(x), dtype=np.uint8)[:RAM_SIZE]
    return np.asarray(x, dtype=np.uint8).reshape(-1)[:RAM_SIZE]


@dataclass(frozen=True)
class TapeSegment:
    """One banked tape: its OWN root blob plus the action indices.

    `label` is provenance only (the level dir a chain segment came from);
    nothing in the replay path reads it.
    """

    root: bytes
    actions: np.ndarray
    label: str = ""

    def __len__(self) -> int:
        return int(np.asarray(self.actions).size)


# ---- provenance helpers (where a tape and its root live on disk) ------


def level_from_wd(wd) -> Optional[str]:
    """`[world, level]` solver coordinates -> the "W-L" name, or None.

    The solver banks zero-based coordinates (`start_wd [0, 2]` is 1-3),
    so this is the one place that translation lives.
    """
    try:
        w, d = int(wd[0]), int(wd[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return f"{w + 1}-{d + 1}"


def solution_paths(run_dir, level: str, index: int = 0) -> tuple[Path, Path]:
    """(actions.npy, sidecar.json) for `level` in a solver run dir.

    Two banked layouts exist and both must resolve:

    * MULTI-LEVEL — `<run>/lvl_<level>/solutions/sol_NNN.*`, what a chain
      solve writes when one run dir holds many levels;
    * SINGLE-LEVEL — `<run>/solutions/sol_NNN.*`, what a per-level solve
      writes when the run dir IS the level (`runs/ge_1_3_solve/`).

    The multi-level path always wins when its level dir exists, so every
    caller that already resolved keeps resolving to the same files. The
    flat fallback is taken ONLY when `lvl_<level>/` is absent and the
    flat pair is present, and it is provenance-gated: the flat sidecar's
    `start_wd` must name the level that was asked for. A run dir solved
    for a different level therefore raises here instead of handing back
    a tape that mints the wrong ladder under the right filename.
    """
    run_dir = Path(run_dir)
    lvl_dir = run_dir / f"lvl_{level}"
    base = lvl_dir / "solutions" / f"sol_{index:03d}"
    act, side = base.with_suffix(".actions.npy"), base.with_suffix(".json")
    if lvl_dir.is_dir() or act.exists() or side.exists():
        return act, side

    flat = run_dir / "solutions" / f"sol_{index:03d}"
    f_act, f_side = flat.with_suffix(".actions.npy"), flat.with_suffix(".json")
    if not (f_act.exists() and f_side.exists()):
        return act, side          # neither layout: report the canonical one
    try:
        rec = json.loads(f_side.read_text())
    except (OSError, ValueError):
        return act, side
    got = level_from_wd(rec.get("start_wd") or ())
    if got is not None and got != str(level):
        raise ValueError(
            f"{run_dir}: single-level layout holds {got} "
            f"(start_wd {rec.get('start_wd')}), not the requested "
            f"{level} — point --run at the {level} solve, or pass "
            f"--actions/--start-state explicitly")
    return f_act, f_side


def root_state_for(sidecar, run_dir) -> Path:
    """The tape's OWN root blob — traces do not transplant across roots.

    The solver banks it by absolute path in the solution sidecar; fall
    back to the run dir's entrance naming when the run has been moved.
    """
    sidecar, run_dir = Path(sidecar), Path(run_dir)
    rec = json.loads(sidecar.read_text())
    p = Path(rec["root_state"])
    if p.exists():
        return p
    alt = run_dir / p.name
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"{sidecar}: root_state {p} not found (also tried {alt})")


def load_root(path, hw_flags=()) -> bytes:
    """Read a root blob after checking its recorded machine lineage.

    Non-fatal by design (see `check_state_sidecar`): an unlabelled blob
    is still the common case, but a labelled one that disagrees with the
    flags this replay is about to use prints a named warning rather than
    producing silent divergence.
    """
    from scripts.go_explore_solve import check_state_sidecar

    path = Path(path)
    check_state_sidecar(path, list(hw_flags))
    return path.read_bytes()


def machine_from_profile(profile: dict, *, rom: Optional[str] = None,
                         hw_flags: Optional[str] = None,
                         profile_name: str = "<profile>",
                         who: str = "tape_replay") -> tuple:
    """(rom_abs, frame_skip, bitmasks, hw_flags) for a SOLVE profile.

    The profile a tape was SOLVED under is the only one that indexes it:
    its `action_space` maps the tape's integers to controller bitmasks
    and its `frame_skip` sets the replay cadence. `rom` / `hw_flags`
    override the profile (the CLI spelling wins), matching
    `resolve_hw_flags`.
    """
    from scripts.go_explore_solve import resolve_hw_flags

    from src.training.profile_utils import action_space_to_bitmasks

    rom_rel = (rom or profile.get("rom_path") or profile.get("rom")
               or (profile.get("solve") or {}).get("rom"))
    if not rom_rel:
        raise SystemExit(f"[{who}] {profile_name} has no rom_path/rom/"
                         f"solve.rom key — pass --rom explicitly")
    return (
        str(REPO / rom_rel),
        int(profile.get("frame_skip", 4)),
        action_space_to_bitmasks(profile["action_space"]),
        tuple(resolve_hw_flags(profile, hw_flags)),
    )


# ---- the replay itself -----------------------------------------------


class TapePlayer:
    """One headless worker pinned to a tape's machine lineage.

    Use as a context manager; `play()` is a generator so a caller can do
    work (save a state, score the RAM) at the exact frame it is handed,
    while the emulator sits still between yields.
    """

    def __init__(self, *, rom: str, bitmasks: Sequence[int], frame_skip: int,
                 hw_flags: Sequence[str] = ()) -> None:
        from nes_core import Pool

        from scripts.go_explore_solve import apply_hw_flags

        self.rom = str(rom)
        self.bitmasks = list(bitmasks)
        self.frame_skip = int(frame_skip)
        self.hw_flags = tuple(hw_flags)
        self.pool = Pool(rom_path=self.rom, num_workers=1,
                         frame_skip=self.frame_skip)
        apply_hw_flags(self.pool, self.hw_flags)  # BEFORE reset_all()
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self._buf = np.zeros(1, dtype=np.uint8)
        self._closed = False

    def __enter__(self) -> "TapePlayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.pool.shutdown()

    def _step(self, mask: int) -> np.ndarray:
        self._buf[0] = mask
        return _ram(self.pool.step_all(self._buf)[0][2])

    def play(self, root: bytes, actions) -> Iterator[tuple[int, np.ndarray]]:
        """Load `root`, no-op once, then step every action.

        Yields `(step, ram)` with `ram` the memory BEFORE action `step`
        runs; the last yield is `(len(actions), post-tape RAM)`.
        """
        self.pool.load_worker_state(0, bytes(root))
        yield 0, self._step(NOOP)          # the no-op materializes RAM
        for i, a in enumerate(np.asarray(actions).reshape(-1)):
            yield i + 1, self._step(self.bitmasks[int(a)])

    def save_state(self, step: Optional[int] = None) -> bytes:
        """Snapshot the worker exactly where `play()` last paused."""
        blob = self.pool.save_worker_state(0)
        if blob is None:
            where = "" if step is None else f" at step {step}"
            raise RuntimeError(f"save_worker_state failed{where}")
        return blob


def replay_tape(*, rom: str, root: bytes, actions, bitmasks: Sequence[int],
                frame_skip: int, hw_flags: Sequence[str] = ()) -> np.ndarray:
    """Replay one tape, returning its ((n_actions + 1), 2048) RAM trace."""
    actions = np.asarray(actions).reshape(-1)
    out = np.empty((actions.size + 1, RAM_SIZE), dtype=np.uint8)
    with TapePlayer(rom=rom, bitmasks=bitmasks, frame_skip=frame_skip,
                    hw_flags=hw_flags) as player:
        for step, ram in player.play(root, actions):
            out[step] = ram
    return out


def replay_segments(segments: Sequence[TapeSegment], *, rom: str,
                    bitmasks: Sequence[int], frame_skip: int,
                    hw_flags: Sequence[str] = ()) -> tuple:
    """Replay a CHAIN of tapes back to back on one worker.

    Returns `(memories (N, 2048) uint8, masks (N,) int32, seams)`.
    `masks[r]` is the controller bitmask that produced row r (0 on each
    segment's leading no-op row); `seams` holds each segment's first row.

    Concatenating segments approximates the single continuous
    playthrough learnfun was designed to consume, and our banked chains
    really are that: segment k+1's root is the entrance captured at the
    end of segment k, so a seam is a settle gap, not a teleport. It is
    also the difference between recovering the progress byte and missing
    it — Bubble Bobble's $0401 is invisible on a single round's tape and
    only surfaces on the 30-round chain.

    EVERY SEGMENT'S LEADING STEP IS A REAL NO-OP. That sounds like it
    goes without saying; it does not. The obvious way to write this loop
    is to hoist the controller buffer out and reuse it, which silently
    replays the PREVIOUS segment's last button press into the next
    segment's settle frame — the buffer still holds it. The
    proof-of-concept this module was lifted from did exactly that, and
    the artifact is not cosmetic: on the 30-round Bubble Bobble chain it
    perturbs 12541 of 16078 trace rows (the first divergence is the
    first seam, row 300, and it compounds from there) and it moves the
    round counter's lead-rank. `TapePlayer.play` sets NOOP explicitly, so
    a seam here is a settle frame with the controller released.
    """
    total = sum(len(s) + 1 for s in segments)
    out = np.empty((total, RAM_SIZE), dtype=np.uint8)
    masks = np.zeros(total, dtype=np.int32)
    seams: list[int] = []
    r = 0
    with TapePlayer(rom=rom, bitmasks=bitmasks, frame_skip=frame_skip,
                    hw_flags=hw_flags) as player:
        for seg in segments:
            seams.append(r)
            for step, ram in player.play(seg.root, seg.actions):
                out[r] = ram
                if step:
                    masks[r] = player.bitmasks[int(seg.actions[step - 1])]
                r += 1
    return out[:r], masks[:r], seams


# ---- chain resolution -------------------------------------------------


def _level_key(name: str) -> tuple:
    """Sort `lvl_1-2` before `lvl_1-10` and both before `lvl_2-1`."""
    stem = name[4:] if name.startswith("lvl_") else name
    parts = stem.replace("_", "-").split("-")
    try:
        return (0, tuple(int(p) for p in parts))
    except ValueError:
        return (1, (stem,))


def run_dir_chain(run_dir, *, solution: int = 0,
                  hw_flags: Sequence[str] = ()) -> list[TapeSegment]:
    """Every solved level in a solver run dir, in natural level order.

    A run dir that solved 1-1..8-4 becomes the 32-tape chain the SMB
    receipt was measured on; each segment carries its own root blob, so
    the seams are the banked entrances.
    """
    run_dir = Path(run_dir)
    out: list[TapeSegment] = []
    for d in sorted((p for p in run_dir.glob("lvl_*") if p.is_dir()),
                    key=lambda p: _level_key(p.name)):
        level = d.name[4:]
        act, sidecar = solution_paths(run_dir, level, solution)
        if not act.exists() or not sidecar.exists():
            state = ("no solutions/ dir" if not (d / "solutions").is_dir()
                     else "incomplete tape/sidecar pair")
            print(f"[tape_replay] {d.name}: {state} for "
                  f"sol_{solution:03d} — SKIPPING (an interrupted solve "
                  f"is otherwise indistinguishable from one never "
                  f"attempted); the returned chain will be shorter than "
                  f"the lvl_* dirs present", flush=True)
            continue
        root = root_state_for(sidecar, run_dir)
        out.append(TapeSegment(root=load_root(root, hw_flags),
                               actions=np.load(act).astype(np.int64),
                               label=d.name))
    if not out:
        raise FileNotFoundError(
            f"{run_dir}: no lvl_*/solutions/sol_{solution:03d}.actions.npy "
            f"tapes found")
    return out


def resolve_chain(spec, *, solution: int = 0, hw_flags: Sequence[str] = (),
                  shuffle_seed: int = 0) -> list[TapeSegment]:
    """Turn a chain spec into loaded `TapeSegment`s.

    Accepts, in order of what it is asked for most:
      * a solver RUN DIR  -> every solved level, in level order;
      * a chain JSON path -> `[{"root": ..., "actions": ...}, ...]`;
      * a list of those dicts, or of `TapeSegment`s already loaded.

    `shuffle_seed` builds the NEGATIVE CONTROL: the same tapes with their
    actions permuted, which produces a matched-length trajectory that is
    NOT a solve. It is the reference the concentration statistic is read
    against (and the control that proved total objective weight is not a
    validity gate — the broken trajectory scores 2.5x MORE weight).
    """
    if isinstance(spec, TapeSegment):
        specs: list = [spec]
    elif isinstance(spec, (str, Path)):
        p = Path(spec)
        if p.is_dir():
            segs = run_dir_chain(p, solution=solution, hw_flags=hw_flags)
            return _maybe_shuffle(segs, shuffle_seed)
        specs = json.loads(p.read_text())
    else:
        specs = list(spec)

    out: list[TapeSegment] = []
    for s in specs:
        if isinstance(s, TapeSegment):
            out.append(s)
            continue
        act = Path(s["actions"])
        root = Path(s["root"])
        out.append(TapeSegment(
            root=load_root(root if root.is_absolute() else REPO / root,
                           hw_flags),
            actions=np.load(act if act.is_absolute()
                            else REPO / act).astype(np.int64),
            label=str(s.get("_lvl") or act.parent.parent.name),
        ))
    if not out:
        raise ValueError(f"{spec!r}: empty chain")
    return _maybe_shuffle(out, shuffle_seed)


def _maybe_shuffle(segments: list[TapeSegment], seed: int) -> list[TapeSegment]:
    if not seed:
        return segments
    rng = np.random.default_rng(seed)
    return [TapeSegment(root=s.root, actions=rng.permutation(s.actions),
                        label=f"{s.label}~shuffled{seed}") for s in segments]
