"""Honest Go-Explore level solver for SMB (research-grounded pipeline, Phase 1).

Runs first-return-then-explore (Ecoffet et al. 2021) PURELY from a level's
ENTRANCE — no hand-crafted prefixes, no handoff seeds, no policy net. Search
(not gradient descent) does the hard exploration that model-free PPO cannot:
the deterministic Rust pool + microsecond save/restore lets the archive return
to any frontier cell for free, so a long correct path is discovered
incrementally instead of re-explored from scratch every episode.

This is the SOLVER half of the Go-Explore -> distillation pipeline (Blueprint
§2C). Solved trajectories are dumped to <out>/solutions/ as action traces +
provenance; the distillation step (BC + DAgger + 25% sticky) teaches a policy
to reproduce them robustly, and the honest metric is cold sticky+jitter from
the entrance. The search is a TEACHER; the deployed agent is the learned net.

Differences from the 2-1 campaign harvester (scripts/go_explore_2_1.py), which
crossed into hand-authored territory:
  * root ONLY at the level entrance (--root-state); NO prefix/handoff lineages.
  * lives-based death detection ($075A) — robust across enemy/pit/time deaths
    and multi-area levels, vs the fragile gx-drop heuristic.
  * (area-order, gx) progress so multi-area levels (1-2's entrance->underground)
    keep pushing the true frontier, not just the entrance area.
  * WARP GUARD: a level "clear" counts only when the world/level advance in the
    natural forward sequence (world unchanged, displayed level +1, OR a castle
    world-advance) — a warp-zone pipe (world jumps) is NOT a legit clear.

Usage:
  python scripts/go_explore_solve.py --out runs/ge_1_2 \
      --root-state checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_03.state \
      --profile configs/mario_1_2_solo.yaml --workers 8 --minutes 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.go_explore import GoExploreArchive, keep_exploring  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

ROM = str(REPO / "roms/Super Mario Bros. (World).nes")
DEATH_STATES = (6, 11)  # RAM $000E player-state: dying / death-pit
NOOP = 0

# RAM addresses (SMB).
R_X_PAGE, R_X_LOW = 0x006D, 0x0086
R_PSTATE = 0x000E
R_YPOS = 0x00CE
R_PHASE = 0x0009
R_AREA = 0x0760
R_WORLD = 0x075F
R_LEVEL = 0x075C
R_LIVES = 0x075A


# ---------------------------------------------------------------------
# Hardware-flag selection + lineage provenance.
#
# A save-state blob records the machine's STATE, never the machine's
# TIMING CONFIGURATION — `Nes::State` carries {ram, mapper, cpu, ppu,
# apu, input, cycles} and nothing else, and every `set_hw_*` flag is
# sticky config that survives `load_worker_state` untouched. So loading
# someone else's entrance blob into a stock Pool silently runs a
# DIFFERENT machine than the one that produced it, and the divergence
# looks like a game/AI mystery rather than a config mismatch.
#
# That is not hypothetical: the Castlevania `cv_chain_hw2` lineage was
# built by a scratch driver under four hw flags that no committed code
# path reproduced, and replaying its block-3 entrance on a stock Pool
# dies in the hall around frame ~15,900 where the lineage survives — the
# whole "bat-wake mystery". Flags are opt-in here (DEFAULT EMPTY, so
# every existing seeded solve is bit-identical) and the resolved set is
# recorded under the key `hw_provenance` in roots.json,
# archive.stats.json and every solution receipt, plus a flattened
# `<blob>.state.json` sidecar next to every state this code writes.
#
# `hw_provenance`, NOT `provenance`: `provenance` is RESERVED for the
# honest-origin string marker (`"search"`) that solution receipts carry
# and CLAIMS.md audits compare against literally. Do not reuse it here
# — a dict literal that binds one key twice keeps only the last value,
# which silently deleted that marker once already (2026-08-07).
# ---------------------------------------------------------------------

def available_hw_flags() -> tuple:
    """Flag names the INSTALLED nes_core exposes, derived from `Pool`'s
    `set_hw_*` methods so this list can never go stale against the
    wheel — a name that needs a rebuild (e.g. `frame_anchor` before the
    Pool-side port lands) is reported as unavailable rather than dying
    in an AttributeError halfway through a run."""
    return tuple(sorted(n[len("set_hw_"):] for n in dir(Pool)
                        if n.startswith("set_hw_")))


def resolve_hw_flags(profile: dict, cli: str | None = None) -> list:
    """Resolve the hw-flag set for this run: `--hw-flags` wins over the
    profile's `solve.hw_flags:` list, and the DEFAULT IS EMPTY.

    `--hw-flags none` (or an empty string) forces the empty set even
    when the profile pins one. Names are validated against the
    installed core and de-duplicated with order preserved."""
    if cli is not None:
        raw = cli.strip()
        names = ([] if raw.lower() in ("", "none")
                 else [p.strip() for p in raw.split(",") if p.strip()])
    else:
        names = list(profile.get("solve", {}).get("hw_flags") or [])
    out, seen = [], set()
    for n in names:
        n = str(n).strip()
        if n.startswith("set_hw_"):        # tolerate the method spelling
            n = n[len("set_hw_"):]
        if n in seen:
            continue
        if n not in available_hw_flags():
            raise SystemExit(
                f"[go_explore_solve] unknown hw flag {n!r}. The installed "
                f"nes_core exposes: {', '.join(available_hw_flags())}. "
                f"(A flag that exists in nes_core/src but not here needs a "
                f"wheel rebuild + the stale-.so copy step.)")
        seen.add(n)
        out.append(n)
    return out


def apply_hw_flags(pool, flags) -> None:
    """Apply resolved hw flags to a freshly-constructed Pool.

    ORDER IS LOAD-BEARING: call this after `Pool(...)` and BEFORE
    `reset_all()`. `reset_alignment` changes boot cycle accounting, so
    setting it after the reset boots a different power-on lineage
    (measured 2026-08-07: RAM diverges at frame 11, $006F/$01FE). The
    other flags are order-insensitive; they are applied together here
    so there is exactly one ordering rule to remember. This is the same
    recipe scripts/tracing/build_cv_tape.py documents for the CV tape."""
    for name in flags:
        getattr(pool, f"set_hw_{name}")(True)


def _core_build_id() -> dict:
    """Identify the nes_core binary actually loaded. The dist version is
    static across rebuilds, so the extension module's digest is what
    actually pins the machine a lineage was produced on."""
    if _core_build_id._cache is None:
        import importlib.metadata as md

        import nes_core
        info = {"dist_version": None, "module": None, "sha256_16": None}
        try:
            info["dist_version"] = md.version("nes_core")
        except Exception:
            pass
        try:
            so = Path(nes_core.__file__).with_name("nes_core.abi3.so")
            if not so.exists():
                so = Path(nes_core.nes_core.__file__)
            info["module"] = so.name
            info["sha256_16"] = hashlib.sha256(so.read_bytes()).hexdigest()[:16]
        except Exception:
            pass
        _core_build_id._cache = info
    return dict(_core_build_id._cache)


_core_build_id._cache = None


def hw_provenance(flags, frame_skip: int) -> dict:
    """The machine description that must travel with any state blob."""
    return {"hw_flags": list(flags), "frame_skip": int(frame_skip),
            "nes_core": _core_build_id()}


def sidecar_path(state_path) -> Path:
    """`foo.state` -> `foo.state.json`. Sidecar, NOT the blob: the
    savestate format stays frozen and config never enters it."""
    p = Path(state_path)
    return p.with_name(p.name + ".json")


def write_state_sidecar(state_path, provenance: dict, extra: dict | None = None):
    """Record the machine a state blob was produced on, next to it."""
    rec = dict(provenance)
    if extra:
        rec.update(extra)
    rec["blob"] = Path(state_path).name
    path = sidecar_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return path


def check_state_sidecar(state_path, flags) -> bool:
    """Warn LOUDLY when a root blob's recorded lineage disagrees with the
    flags this run is about to use. Returns True when they agree (or no
    sidecar exists to check against). Deliberately non-fatal: an
    unlabelled blob is the common case until backfill catches up, and a
    warning that names both sets is what turns an invisible config
    mismatch into a five-second diagnosis."""
    path = sidecar_path(state_path)
    if not path.exists():
        return True
    try:
        want = list(json.loads(path.read_text()).get("hw_flags", []))
    except (OSError, ValueError) as e:
        print(f"[go_explore_solve] unreadable state sidecar {path}: {e}",
              flush=True)
        return True
    if sorted(want) == sorted(flags):
        return True
    print(f"[go_explore_solve] *** HW-FLAG LINEAGE MISMATCH *** "
          f"{Path(state_path).name} was built with {want or '[]'} but this "
          f"run uses {list(flags) or '[]'}. The restored machine is not the "
          f"machine that produced this state; expect divergence that looks "
          f"like nondeterminism. Pass --hw-flags {','.join(want) or 'none'} "
          f"to match the lineage.", flush=True)
    return False


def stamp_stats_provenance(archive_path, provenance: dict) -> None:
    """Merge run provenance into the archive's stats sidecar.

    `GoExploreArchive.save()` owns archive.stats.json; this re-opens it
    right after and adds the machine description under `hw_provenance`,
    so a banked run can be re-created without guessing which flags it
    ran under. Uses its own `.prov.tmp` scratch name so it can never
    race the archive's own `.stats.json.tmp`.

    The key is `hw_provenance`, never `provenance`: `provenance` is the
    reserved honest-origin marker (`"search"`) on solution receipts, and
    one token must not mean a string in one artifact and a dict in its
    sibling."""
    p = Path(archive_path).with_suffix(".stats.json")
    try:
        stats = json.loads(p.read_text())
    except (OSError, ValueError):
        return
    if stats.get("hw_provenance") == provenance:
        return
    stats["hw_provenance"] = provenance
    tmp = p.with_name(p.name + ".prov.tmp")
    try:
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, p)
    except OSError as e:
        print(f"[go_explore_solve] could not stamp provenance into {p}: {e}",
              flush=True)


def _gx(ram) -> int:
    return (int(ram[R_X_PAGE]) << 8) | int(ram[R_X_LOW])


def _wd(ram) -> tuple:
    """Displayed (world, level), both 0-indexed."""
    return (int(ram[R_WORLD]), int(ram[R_LEVEL]))


GX_BUCKET = 16   # overridable via --gx-bucket (micro-search: 8)
Y_BAND = 32      # overridable via --y-band (micro-search: 16)


def cell_fn(ram) -> tuple:
    # (area, step-phase, y-band, gx bucket). Phase = (frame>>2)&7 is
    # step-granular at frame_skip 4 (all 8 classes reachable via step-count
    # variance). gx bucket is LAST so the archive's horizontal_neighbors
    # frontier bonus applies unmodified.
    # vx sign disambiguates travel direction: doubling back through
    # previously-visited coordinates is a DISTINCT cell, so the archive
    # explores backtracking maneuvers instead of pruning them as loops
    # (heuristic-inversion recipe, maze consultation 2026-07-24).
    # $0057 is a signed velocity byte; take the two's-complement in pure
    # Python so an out-of-int8-range value (e.g. 252 = -4) casts cleanly
    # (np.int8(252) now raises a NumPy out-of-bound DeprecationWarning).
    vx = int(ram[0x57])
    if vx >= 128:
        vx -= 256
    vsign = 0 if vx == 0 else (1 if vx > 0 else 2)
    return (int(ram[R_AREA]), (int(ram[R_PHASE]) >> 2) & 7, vsign,
            int(ram[R_YPOS]) // Y_BAND, _gx(ram) // GX_BUCKET)


def is_forward_clear(start_wd: tuple, ram) -> bool:
    """True iff (world, level) advanced in the NATURAL forward sequence.

    Legit 1-2 -> 1-3: world unchanged, level +1. Legit castle world advance
    (e.g. 1-4 -> 2-1): world +1 (level resets to 0). A WARP-ZONE pipe jumps
    the world by >1 or advances the world without having been on a castle —
    guarded out so warps never count as a clear (honesty requirement)."""
    sw, sl = start_wd
    w, l = _wd(ram)
    if w == sw and l == sl + 1:
        return True   # same-world level advance (the common case, incl. 1-2->1-3)
    if w == sw + 1 and l == 0 and sl >= 2:
        return True   # castle clear -> next world (only from an x-4, level idx>=2... )
    return False


def resolve_verify_bank(args) -> bool:
    """Whether a clear candidate must reproduce from its root, in a fresh
    pool, before it is written to solutions/ (--verify-bank /
    --no-verify-bank).

    DEFAULTS TRUE, including when the attribute is absent — in-process
    constructions that predate the flag (the live show's hand-built args
    namespace) get verification too. That direction is deliberate and the
    opposite of the hw-flag/ortho knobs' "absent means off": those change what
    the search DOES, this only decides whether a claimed win is checked before
    it becomes a receipt, and banking trust must not be something a stale call
    site opts out of by omission."""
    return bool(getattr(args, "verify_bank", True))


def resolve_inversion_pin_secs(args) -> float:
    """Seconds the deep frontier must sit pinned before the heuristic-
    inversion arm engages (--inversion-pin-secs).

    Absent attribute — in-process constructions (the live show) that
    predate the flag — resolves to 180.0, the constant the receipted
    32-level SMB clear ran under, so those callers are unchanged. A
    NEGATIVE value disables the arm outright: the sampler then never
    flips to inverted weights, whatever the frontier does."""
    v = getattr(args, "inversion_pin_secs", None)
    return 180.0 if v is None else float(v)


def inversion_armed(pin_secs: float, pin_gx: int, gx: int, floor: float,
                    elapsed: float) -> bool:
    """Whether a worker sitting at `gx` should sample from the INVERTED
    weights this step.

    Four independent conditions, all measured off our own telemetry:
    the arm is enabled (`pin_secs` non-negative — a negative value is the
    --inversion-pin-secs disable sentinel, and it short-circuits FIRST
    because `elapsed >= -1` would otherwise always hold); the run has a
    real frontier to be pinned at (`pin_gx > 400`); the worker is inside
    the saturation window [floor, pin_gx+60] on a non-garbage gx; and the
    frontier has sat unmoved for at least `pin_secs`.

    A free function, not an inline expression, so the disable sentinel
    and the window boundaries are testable without a ROM and a Pool —
    the wiring is the whole deliverable of the flag."""
    return (pin_secs >= 0.0 and pin_gx > 400 and gx >= 0
            and floor <= gx <= pin_gx + 60 and elapsed >= pin_secs)


def inverted_weights(action_space) -> list:
    """Exploration bias for the saturation window: reward the maneuvers the
    forward heuristic structurally prunes (leftward, downward)."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "left" in b:
            w += 3.0
        if "down" in b:
            w += 3.0
        if "A" in b:
            w += 1.0
        if "right" in b:
            w -= 0.5
        ws.append(max(w, 0.2))
    return ws


def action_weights(action_space) -> list:
    """Right-biased sampling weights over the profile's action space."""
    ws = []
    for buttons in action_space:
        b = set(buttons)
        w = 1.0
        if "right" in b:
            w += 2.0
        if "A" in b:
            w += 2.0  # jumps carry gaps/pipes
        if "B" in b:
            w += 1.0
        ws.append(w)
    return ws


class SmbGame:
    @staticmethod
    def score_bonus(ram) -> int:
        return 0

    """SMB-engine adapter — wraps the module-level helpers verbatim, so
    solver behavior on SMB is byte-identical to the pre-adapter code
    (regression: seeded 1-1 run reproduces the same solution sha).

    Works for any game on the SMB1 ENGINE, not just SMB1 itself — the
    RAM map ($006D/$0086 x, $0760 area, $075F/$075C world/level, $075A
    lives) is engine-level. The Lost Levels (Super Mario Bros. 2 Japan /
    FDS) is the same engine with harder layouts, so a profile pointing
    `rom:` at it (and carrying NO solve: section) routes here and gets
    the full byte-exact SMB solver for free. A profile's optional `rom:`
    overrides the default SMB1 World ROM."""
    progress_cap = 7000   # transition-frame garbage guard (8-1 lesson)

    def __init__(self, profile: dict = None) -> None:
        r = (profile or {}).get("rom")
        self.rom = str(REPO / r) if r else ROM

    def progress(self, ram) -> int:
        return _gx(ram)

    def level_key(self, ram) -> tuple:
        return _wd(ram)

    def lives(self, ram) -> int:
        return int(ram[R_LIVES])

    def area(self, ram) -> int:
        return int(ram[R_AREA])

    def y(self, ram) -> int:
        return int(ram[R_YPOS])

    def swim(self, ram) -> int:
        return int(ram[0x1D])

    def cell_fn(self, ram) -> tuple:
        return cell_fn(ram)

    def is_clear(self, start_key: tuple, ram, ctx: dict | None = None) -> bool:
        # ctx accepted for call-site uniformity with GenericGame; SMB uses the
        # warp-guarded level_key advance only (behavior byte-identical).
        return is_forward_clear(start_key, ram)

    def is_dead(self, ram, start_lives: int) -> bool:
        # $00B5 = vertical screen page: 1 on-screen, 0 above (vines — legit),
        # >= 2 fallen below the screen. A pit fall reads B5=2..4 for ~120
        # frames BEFORE lives decrements (measured live, Lost Levels 2-2), so
        # without this check every frame of the fall is archivable as a
        # live cell — at 2-2 the deepest-gx frontier was ~entirely such
        # corpses (observatory receipt runs/lost_levels/ll_2_2_sectcap_fix/
        # observatory_v2.json: top predicate = the death timer $0795), and
        # every burst selected from them was doomed on arrival. Verified
        # zero false positives across the full receipted 8-4 clear replay
        # (1,735 actions incl. hold-Down pipe entries + the water section):
        # legit play never reads B5 >= 2.
        return (int(ram[R_LIVES]) < start_lives
                or int(ram[R_PSTATE]) in DEATH_STATES
                or int(ram[0xB5]) >= 2)

    def is_finale(self, start_key: tuple, ram) -> bool:
        # GAME-COMPLETE (8-4): the ending never advances the world/level
        # bytes — victory is operating mode $0770 == 2 with inputs locked.
        return int(ram[0x770]) == 2 and tuple(start_key) == (7, 3)

    def room_id(self, ram) -> tuple:
        # rid = (area type, swim flag, Bowser-slot) — verified 2026-07-26:
        # $074E changes 0.67/1k steps, only at full-screen transitions.
        bw = 1 if 0x2D in bytes(ram[0x14:0x1C]) else 0
        return (int(ram[0x74E]), int(ram[0x1D]), bw)

    def clear_verify_margin(self) -> int:
        """No NOOP margin: SMB's clear hooks (a level_key advance, and the
        $0770==2 finale) are STATELESS — pure functions of the current frame
        — so a replay's verdict lands on exactly the frame the live one did.
        See GenericGame.clear_verify_margin for the windowed case and why it
        needs one. Every adapter must define this; Solver.replay_verify calls
        it unconditionally, and its broad except would otherwise turn a
        missing method into a blanket 'error' verdict that rejects every real
        clear the game ever finds."""
        return 0

    @staticmethod
    def label(key: tuple) -> str:
        return f"{key[0] + 1}-{key[1] + 1}"


class GenericGame:
    """Profile-driven adapter: every address comes from the game profile's
    `solve:` section, whose bytes must be observationally verified first
    (scripts/verify_ram_map.py; receipts under docs/receipts/ram_verify/).

    Required solve keys: rom, progress {lo[, hi]}, y, level_key (list of
    addrs; any lexicographic advance = clear), lives (decrement = death).
    Optional: area, progress_cap, player_state + death_states, finale
    {addr, value, level_key} for the game's ending; clear {mode: ...} = an
    OPTIONAL second WIN-CONDITION signal for games whose level_key never
    advances on a clear (see __init__ / is_clear; default OFF = level_key
    only). The discrete-transition gate (room_advance) is a Solver-level
    option (see derive_transition_macros)."""

    def __init__(self, profile: dict) -> None:
        s = profile["solve"]
        self.rom = str(REPO / s["rom"])
        p = s["progress"]
        # Decoded HUD-field progress (optional): score/money games (DuckTales)
        # have no monotone spatial frontier — the objective is a multi-tile
        # decimal HUD field. `progress: {tiles: [addrs MSB-first], blank: B,
        # scale: N}` decodes those tiles (tile value = literal digit, blank
        # tile skipped) into an integer // scale. The monotone objective then
        # IS the frontier: banking higher-score states drives the solver
        # toward treasure-rich paths through the level. `lo`/`hi` is the
        # plain 1-2 byte mode (spatial games).
        self._ptiles = [int(a) for a in p["tiles"]] if "tiles" in p else None
        self._pblank = int(p.get("blank", 0x24))
        self._pscale = int(p.get("scale", 1))
        self._plo = int(p["lo"]) if "lo" in p else None
        self._phi = int(p["hi"]) if "hi" in p else None
        self._y = int(s["y"])
        self._lk = [int(a) for a in s["level_key"]]
        self._lives = int(s["lives"])
        self._area = int(s["area"]) if "area" in s else None
        self.progress_cap = int(s.get("progress_cap", 30000))
        self._pstate = int(s["player_state"]) if "player_state" in s else None
        self._death_states = tuple(s.get("death_states", ()))
        self._finale = s.get("finale")
        # Boss-fight progress (optional): a fixed-camera boss room pins the
        # gx frontier and position cells saturate in minutes (block-3 lesson:
        # 4.2M steps, 5,235 cells, zero gradient). `boss: {hp, start}` keys
        # cells on the boss-HP byte and scores damage dealt, giving the
        # archive a fight dimension. The byte must be discovered by
        # differential analysis of our own rollouts and receipted like every
        # other solve address.
        b = s.get("boss") or {}
        self._boss_hp = int(b["hp"]) if "hp" in b else None
        self._boss_start = int(b.get("start", 255))
        # Type-conditioned boss health (object-array engines): slots are
        # SHARED between fight objects and respawning mobs (Contra base:
        # wall guns type 54/68 + sensor 55 alternate with soldiers 33/44
        # in $0311-$0314; per-slot HP at $04BF-$04C2, causal fire receipt
        # 2026-07-31). hp = sum of hp_addrs[i] where type_addrs[i] holds
        # a listed type; reads as `start` when no fight object is live
        # (pre-wall AND post-destruction — the transit machinery owns the
        # post state).
        bt = s.get("boss_typed") or {}
        self._bt_types = frozenset(int(t) for t in bt.get("types", ()))
        self._bt_addrs = list(zip((int(a) for a in bt.get("type_addrs", ())),
                                  (int(a) for a in bt.get("hp_addrs", ()))))
        self._bt_start = int(bt.get("start", 12))
        # Movement-mode signature (optional): `state_sig: [{addr, match}]`.
        # Each entry contributes one bit: ram[addr] in match. Separates
        # aliased player modes at the same coordinates — the CV block-3
        # lesson: stair-climb states share (gx, y-band) with jump apexes,
        # tie-break to fewer steps, and the airborne dead-end wins the
        # cell; the staircase stays structurally invisible to the archive.
        # Optional 'mod': the bit tests (ram[addr] % mod) in match —
        # for PHASE observables (e.g. Contra's core open-window: frame
        # counter $001A % 16 in 7..14, discovered by pixel-phase mining
        # 2026-07-31, MI 0.53, accuracy 0.997). mod=0/absent = raw value.
        self._state_sig = [(int(e["addr"]),
                            frozenset(int(v) for v in e.get("match", ())),
                            int(e.get("mod", 0)))
                           for e in s.get("state_sig", ())]
        # Room signature (optional): `room_sig: [addr,...]` — bytes stable
        # within a room and different across rooms (found by before/after
        # transition diff of our own climbs). Feeds room_id, so the sect/
        # psig transit machinery (built on SMB's $074E) counts CV room
        # progress even though gx resets in every room.
        self._room_sig = tuple(int(a) for a in s.get("room_sig", ()))
        # WIN-CONDITION hook (optional, default OFF). Many games have no clean
        # level_key that advances on a clear (contra/kirby/ducktales all ship
        # `level_key: []` = coverage baseline: deep frontier cells accrue but a
        # clear NEVER fires). `solve: clear: {mode: ...}` lets is_clear ALSO
        # fire on a second, configured signal. level_key advance stays the sole
        # default clear, so a profile without a `clear:` key is byte-identical.
        # Modes:
        #   score_jump  {threshold}  — a single-step rise in the decoded
        #     progress (solve.progress, which for DuckTales already IS the
        #     money odometer) >= threshold. DuckTales's boss main-treasure is
        #     one atomic $1,000,000 pickup (>= the $50k
        #     largest gem) so a jump this large can ONLY be the clear; being
        #     per-step, gem farming can never fake it. threshold is in the same
        #     units as solve.progress (DuckTales scale /100 -> $1M = 10000).
        #   byte_change {addr[, direction: up|down|any][, target]} — ram[addr]
        #     moved off its level-start value (captured in note_start) in the
        #     given direction (default up): a stage/level counter incrementing
        #     on a real clear, without making it the level_key (which also gates
        #     the warp/dead machinery). `target` requires a specific landing.
        #   confluence  {[window][stride][min_signals][persist_checks]
        #                [progress_median]
        #                [room_veto: {steps[, addrs][, min_progress]}]} — the
        #     multi-signal clear_detect confluence, streamed over a bounded
        #     rolling window (clear_detect.StreamingConfluenceDetector; see its
        #     docstring for which signals are RAM-derivable in the hot loop).
        #
        #     DETECTOR v2 (2026-08-08) adds three knobs on top of the
        #     2026-08-06 lives-drop veto. ALL DEFAULT-INERT, so an existing
        #     `clear: {mode: confluence}` profile (contra, gradius) is
        #     byte-identical until it opts in:
        #       progress_median: K — median-filter the progress series over K
        #         trailing samples before the coord test. THE combat-blip fix
        #         (Double Dragon: 72 -> 846 -> 88 in 5 steps read as a level
        #         load). K=5 erases any impulse under ~2 samples and leaves a
        #         real load's step change intact. Default 1 = off.
        #       persist_checks: N — the confluence must additionally hold N
        #         consecutive checks before the latch closes. Second, cheaper
        #         filter; NOT sufficient alone (see the detector docstring for
        #         the window arithmetic). Default 1 = shipped behavior.
        #       room_veto: {steps: M[, addrs: [..]][, min_progress: P]} — a
        #         change of the room observable within the last M observations
        #         VETOES a fire unless `progress` has advanced by >= P (default
        #         1) measured from just before that change. This is the Kirby
        #         fix: a room-based game loads a fresh room far more often than
        #         it finishes a stage, and "a fresh room loading in" is
        #         literally what the coord signal fingerprints, so the only
        #         generic discriminator is whether the game's own progress
        #         observable actually moved forward across the transition.
        #         `addrs` names the room bytes explicitly; omitted, it uses the
        #         adapter's already-verified room identity (level_key + area +
        #         room_sig) — no new addresses either way. steps: 0 = disarmed
        #         (default). Progress samples above progress_cap are dropped
        #         rather than believed: a room load is exactly when the page
        #         byte reads transitional garbage, and one such sample used to
        #         satisfy the advance test in the very call that armed the
        #         veto — permanently disarming it. Set progress_cap for the
        #         game if the default 30000 is not right for its scale.
        #         PRECISION OVER RECALL, deliberately: on a game whose TERMINAL
        #         transition is observationally identical to an internal one
        #         (same room-byte change, same progress reset) this suppresses
        #         the terminal one too, and the hook degenerates toward "never
        #         fires". That is the intended failure direction — a missed
        #         clear costs one re-search, a fabricated one poisons
        #         go_explore_chain's next-entrance extraction and the receipt
        #         corpus. Raise recall by pointing `progress` at an observable
        #         that really does advance across the terminal transition, not
        #         by disarming the veto.
        cl = s.get("clear") or {}
        self._clear_mode = cl.get("mode")
        self._clear_threshold = float(cl.get("threshold", 0))
        self._clear_addr = int(cl["addr"]) if "addr" in cl else None
        self._clear_dir = str(cl.get("direction", "up"))
        self._clear_target = int(cl["target"]) if "target" in cl else None
        self._clear_baseline = None          # byte_change: set by note_start
        self._conf_window = int(cl.get("window", 240))
        self._conf_stride = int(cl.get("stride", 20))
        self._conf_min_signals = cl.get("min_signals")
        self._conf_persist = cl.get("persist_checks")
        self._conf_median = cl.get("progress_median")
        rv = cl.get("room_veto") or {}
        self._rv_steps = int(rv.get("steps", 0))
        self._rv_addrs = tuple(int(a) for a in rv.get("addrs", ()))
        self._rv_min_progress = int(rv.get("min_progress", 1))

    def clear_verify_margin(self) -> int:
        """Extra NOOP observations a replay must feed the clear hook, past the
        end of the trace, before "it did not reproduce" means anything.

        Zero for every STATELESS hook (level_key, finale, byte_change,
        score_jump): those fire on the exact frame the RAM satisfies them, so
        the replay's verdict lands on the same frame the live one did.

        NON-ZERO only for `confluence`, which is windowed AND phase-dependent.
        The streaming detector evaluates only when its own observation counter
        satisfies `_n % stride == 0`, and that counter starts when the ctx is
        created. Live, _assign() builds a fresh ctx per BURST, so the fire
        lands at (F - burst_start) % stride == 0; a replay from the ROOT builds
        its ctx at index 0, so its checks sit at multiples of stride from
        there. The two phases agree only when burst_start % stride == 0, and
        because the banked trace is cut exactly at F there are no steps left
        for the phase to re-align in — the replay's last check falls BEFORE the
        level-load signature and returns 'no_clear'. Measured over the real
        detector on a genuine no-life-lost clear: 136 of 200 burst alignments
        (68%) were rejected with no margin, 0 of 200 with a margin of
        stride-1 or more. Without this, --verify-bank being ON by default
        would have made the four-game confluence gate report zero solutions,
        with 'this is a fabricated clear' printed over every real win.

        stride * persist_checks, so the replay gets a full check period per
        consecutive confluence the live latch required. At the defaults that
        is 20 actions (~54 ms), against candidates of 500-1,700."""
        if self._clear_mode != "confluence":
            return 0
        persist = max(1, int(self._conf_persist or 1))
        return max(0, self._conf_stride) * persist

    def note_start(self, ram) -> None:
        """Record the level-entrance baseline the byte_change WIN-CONDITION
        compares against (called once from Solver.seed at the entrance). No-op
        unless a byte_change clear hook is configured."""
        if self._clear_mode == "byte_change" and self._clear_addr is not None:
            self._clear_baseline = int(ram[self._clear_addr])

    def progress(self, ram) -> int:
        if self._ptiles is not None:
            v = 0
            for a in self._ptiles:          # MSB-first digit tiles
                t = int(ram[a])
                if t < 10:                   # blank tile (e.g. 0x24) skipped
                    v = v * 10 + t
            return v // self._pscale
        v = int(ram[self._plo])
        if self._phi is not None:
            v |= int(ram[self._phi]) << 8
        return v

    def level_key(self, ram) -> tuple:
        return tuple(int(ram[a]) for a in self._lk)

    def lives(self, ram) -> int:
        return int(ram[self._lives])

    def area(self, ram) -> int:
        return int(ram[self._area]) if self._area is not None else 0

    def y(self, ram) -> int:
        return int(ram[self._y])

    def swim(self, ram) -> int:
        return 0

    def _typed_hp(self, ram) -> int:
        live = [int(ram[h]) for t, h in self._bt_addrs
                if int(ram[t]) in self._bt_types]
        return sum(live) if live else self._bt_start

    def cell_fn(self, ram) -> tuple:
        # Same arity as SMB's cell (selection caches index key[-5]/key[-1];
        # boss HP and the mode bits ride in free middle slots so those
        # stay intact).
        if self._bt_addrs:
            hp = self._typed_hp(ram)
        elif self._boss_hp is not None:
            hp = int(ram[self._boss_hp])
        else:
            hp = 0
        sig = 0
        for i, (a, m, md) in enumerate(self._state_sig):
            v = int(ram[a]) % md if md else int(ram[a])
            if v in m:
                sig |= 1 << i
        return (self.area(ram), hp, sig,
                self.y(ram) // Y_BAND, self.progress(ram) // GX_BUCKET)

    def score_bonus(self, ram) -> int:
        # Damage dealt dominates within-room gx differences (a whip hit is
        # worth more frontier than any sidestep) without touching the
        # cross-room score ordering scale.
        if self._bt_addrs:
            return max(0, self._bt_start - self._typed_hp(ram)) * 2000
        if self._boss_hp is None:
            return 0
        return max(0, self._boss_start - int(ram[self._boss_hp])) * 2000

    def room_veto_key(self, ram) -> tuple:
        """The observable whose CHANGE means 'a different room/level just
        loaded'. `clear.room_veto.addrs` when the profile names bytes
        explicitly; otherwise this adapter's own already-verified room
        identity (level_key + area + room_sig). No new addresses either way."""
        if self._rv_addrs:
            return tuple(int(ram[a]) for a in self._rv_addrs)
        return self.room_id(ram)

    # Room-veto verdicts (see room_veto_step).
    RV_NONE, RV_HOLD, RV_DISCARD = "none", "hold", "discard"

    def room_veto_step(self, ram, ctx: dict) -> str:
        """Advance this worker's room/progress bookkeeping by ONE observation
        and say what the room veto wants done with a clear vote right now.

        Must be called on every observation, not only the ones that fire: it
        IS the history.

          RV_NONE     nothing to veto — no room change is pending, or progress
                      has since advanced past where it stood just before one.
          RV_HOLD     a room change is pending and progress has not advanced
                      yet: suppress a fire but KEEP the evidence; the advance
                      may still arrive inside the window.
          RV_DISCARD  the window expired with no advance, so that was an
                      ordinary room transition: suppress AND throw the
                      evidence away. Fires exactly once per room change.

        The three-state shape is load-bearing, and a plain boolean is not
        enough in either direction. Suppress-only leaks: the samples that
        produced the fire sit in the detector's rolling window, so the same
        false fire simply lands again the moment the window expires.
        Discard-immediately over-blocks: it destroys the evidence before the
        progress advance that would have EXONERATED the transition has had its
        `steps` observations to show up, which would make the escape clause
        unreachable and the veto a blanket mute.

        GARBAGE-READ GUARD (2026-08-08): every progress sample is filtered
        through progress_cap before it is used or stored. A room load is
        precisely when the page byte reads transitional garbage, and the escape
        clause runs on the arming frame too — so an unguarded read let ONE
        garbage sample satisfy `prog - prog_before >= min_progress` in the very
        call that armed the veto, permanently disarming it and re-banking the
        Kirby false positive. Replay verification cannot catch that: the
        emulator is deterministic, so the replay reproduces the same garbage
        read and the same fabricated verdict. Every other consumer of
        progress() in this file already guards it the same way (observe()'s
        `gx > game.progress_cap` skip, explore()'s cap clamp); this is the one
        that did not."""
        n = ctx["_rv_n"] = ctx.get("_rv_n", 0) + 1
        key = self.room_veto_key(ram)
        raw = self.progress(ram)
        # None == "not an observation of progress at all", not "zero": a
        # garbage sample must neither advance the escape test nor become the
        # baseline a later escape is measured from.
        prog = None if raw > self.progress_cap else raw
        prev_key = ctx.get("_rv_key")
        prev_prog = ctx.get("_rv_prog_prev")
        ctx["_rv_key"] = key
        if prog is not None:
            ctx["_rv_prog_prev"] = prog     # last GOOD sample, never garbage
        if prev_key is not None and key != prev_key:
            ctx["_rv_at"] = n
            # Progress as it read on the step BEFORE the room changed — the
            # transition frame itself already reads the new room's value.
            # Both may be garbage (None); the veto then arms with no baseline
            # and simply cannot be escaped, which is the safe direction.
            ctx["_rv_prog_before"] = prev_prog if prev_prog is not None else prog
        at = ctx.get("_rv_at")
        if at is None:
            return self.RV_NONE
        before = ctx.get("_rv_prog_before")
        if (prog is not None and before is not None
                and prog - int(before) >= self._rv_min_progress):
            ctx["_rv_at"] = None          # the transition earned its keep
            return self.RV_NONE
        if n - at <= self._rv_steps:
            return self.RV_HOLD
        ctx["_rv_at"] = None              # act once per room change
        return self.RV_DISCARD

    def is_clear(self, start_key: tuple, ram, ctx: dict | None = None) -> bool:
        # Forward = lexicographic advance of the level key (stage counters
        # increment; a game-over reset reads backward and lands in `dead`).
        if self.level_key(ram) > tuple(start_key):
            return True
        # Optional configured WIN-CONDITION (default OFF). ctx is the caller's
        # per-worker state dict; None (e.g. the seed observe) short-circuits the
        # stateful modes so only the level_key advance above can fire.
        m = self._clear_mode
        if not m or ctx is None:
            return False
        if m == "score_jump":
            v = self.progress(ram)
            prev = ctx.get("_clear_prev")
            ctx["_clear_prev"] = v
            return prev is not None and (v - prev) >= self._clear_threshold
        if m == "byte_change":
            if self._clear_addr is None or self._clear_baseline is None:
                return False
            cur = int(ram[self._clear_addr])
            if self._clear_target is not None:
                return cur == self._clear_target and cur != self._clear_baseline
            if self._clear_dir == "up":
                return cur > self._clear_baseline
            if self._clear_dir == "down":
                return cur < self._clear_baseline
            return cur != self._clear_baseline
        if m == "confluence":
            det = ctx.get("_clear_det")
            if det is None:
                import sys as _sys
                _sd = str(Path(__file__).resolve().parent)   # scripts/ on path
                if _sd not in _sys.path:
                    _sys.path.insert(0, _sd)
                from clear_detect import StreamingConfluenceDetector
                det = ctx["_clear_det"] = StreamingConfluenceDetector(
                    self.progress, window=self._conf_window,
                    stride=self._conf_stride,
                    min_signals=self._conf_min_signals,
                    persist_checks=self._conf_persist,
                    progress_median=self._conf_median)
            fired = det.push(ram)
            # ROOM-TRANSITION VETO (v2, 2026-08-08). Bookkeeping runs on
            # every observation, so it is evaluated before the early
            # returns below; disarmed profiles (room_veto.steps 0, the
            # default) skip it entirely and stay byte-identical.
            rv = (self.room_veto_step(ram, ctx) if self._rv_steps > 0
                  else self.RV_NONE)
            # Lives-drop veto (2026-08-06): confirmed empirically on
            # Gradius that a real death (explosion + entity-slot wipe +
            # respawn) trips this detector's "coord" signal identically
            # to a real level load. Solver.observe() now resolves
            # is_dead() BEFORE is_clear() (v2), so this is belt-and-
            # braces for that path — but it is also the only guard for
            # any caller that checks the clear hook on its own. Track
            # this worker's own lives across steps (not the archive-wide
            # start_lives baseline, which only bounds a whole lineage,
            # not a single frame) and refuse to fire the same step lives
            # just dropped.
            cur_lives = self.lives(ram)
            prev_lives = ctx.get("_clear_prev_lives")
            ctx["_clear_prev_lives"] = cur_lives
            # ---- every per-step bookkeeping side effect is now done; only
            # verdicts below this line, so no early return can leave a
            # tracker one step stale.
            if rv == self.RV_DISCARD:
                # Drop the evidence with the verdict — including evidence
                # that has not fired yet. The samples spanning an ordinary
                # room load ARE the coord signature, and they stay in the
                # detector's rolling window for another ~window/stride
                # checks; leaving them there only postpones the same false
                # fire past the veto's own horizon.
                det.reset()
                return False
            if not fired:
                return False
            if prev_lives is not None and cur_lives < prev_lives:
                return False
            if rv == self.RV_HOLD:
                # Suppress, but keep the window: a progress advance inside
                # the remaining veto steps still exonerates this fire.
                return False
            return True
        return False

    def is_dead(self, ram, start_lives: int) -> bool:
        if self.lives(ram) < start_lives:
            return True
        return (self._pstate is not None
                and int(ram[self._pstate]) in self._death_states)

    def is_finale(self, start_key: tuple, ram) -> bool:
        f = self._finale
        return (bool(f) and tuple(start_key) == tuple(f["level_key"])
                and int(ram[int(f["addr"])]) == int(f["value"]))

    def room_id(self, ram) -> tuple:
        return (self.level_key(ram) + (self.area(ram),)
                + tuple(int(ram[a]) for a in self._room_sig))

    @staticmethod
    def label(key: tuple) -> str:
        return "-".join(str(x) for x in key)


def make_game(profile: dict):
    """SMB-engine profiles carry no `solve:` section — they get the
    byte-exact SMB adapter (which reads an optional `rom:` override, so
    Lost Levels / any SMB1-engine game works). A profile with `solve:`
    opts into the generic path."""
    return GenericGame(profile) if "solve" in profile else SmbGame(profile)


def derive_transition_macros(action_space: list, room_advance: dict | None) -> list:
    """Discrete-transition gate (the Kirby lesson). Room-based games stall
    because forward progress cannot gradient the player into a DISCRETE
    'grounded + UP' door/room entry — the camera clamps at the scroll limit so
    world-X gives no pull toward the door. When `solve: room_advance:` is
    configured this AUTO-DERIVES the door-entry maneuver macros (up / up+jump /
    up+right holds) straight from the profile's own action space — no
    hand-authored hold_macros needed — for the solver to inject near the deep
    frontier. Returns [(action_idx, hold_steps)]; empty when the space has no
    UP move (the gate is then inert). This is the exact settle-then-HOLD
    mechanism the existing hold_macros use (Solver.explore), just auto-built and
    frontier-biased rather than fired uniformly.

    Config: `room_advance: {addr[, steps=20][, p=0.05][, near=24][, buttons]}`.
    `addr` = the room/area-id byte whose advance we want to provoke (also
    surfaced as telemetry). `buttons` overrides the auto-derivation with an
    explicit list of button combos."""
    if not room_advance:
        return []
    steps = int(room_advance.get("steps", 20))
    want = room_advance.get("buttons")
    combos = ([set(b) for b in want] if want
              else [set(c) for c in action_space if "up" in set(c)])
    out = []
    for combo in combos:
        idx = next((i for i, c in enumerate(action_space) if set(c) == combo),
                   None)
        if idx is not None:
            out.append((idx, steps))
    return out


def update_stall(stall: dict, n_cells: int, now: float) -> None:
    """Pure state transition for the flat-archive stall watchdog: bumps
    `flat_windows` when the archive hasn't grown since the last call,
    resets it otherwise. Caller decides the polling cadence and what to
    do once flat_windows crosses its own threshold (>=2 == 2+ minutes
    with zero new cells, matching live_solve_show.py's convention)."""
    stall["flat_windows"] = (stall["flat_windows"] + 1
                             if n_cells <= stall["last_cells"] else 0)
    stall["last_cells"], stall["last_t"] = n_cells, now


def column_extremes(keys, mode: str) -> dict:
    """Per-gx-column ceiling (mode 'up') or floor (mode 'down') of the
    ORTHOGONAL axis: {gx bucket -> extreme y-band} over the given cell
    keys. key[-1] is the gx bucket, key[-2] the y-band (both SmbGame and
    GenericGame cells end (..., y_band, gx_bucket)). 'up' is min because
    the NES y axis grows downward — a smaller band is higher up."""
    pick = min if mode == "up" else max
    ext: dict = {}
    for k in keys:
        col, yb = k[-1], k[-2]
        cur = ext.get(col)
        ext[col] = yb if cur is None else pick(cur, yb)
    return ext


def ortho_pool(cells, mode: str, band: int) -> list:
    """The orthogonal frontier: cells within `band` y-bands of THEIR OWN
    column's extreme. Per-column, not global — a global cutoff would let
    one tall column's ceiling define the whole level and silently drop
    every other column's top-of-stack cell (the aliasing trap)."""
    ext = column_extremes([c.key for c in cells], mode)
    return [c for c in cells if abs(c.key[-2] - ext[c.key[-1]]) <= band]


def ortho_armed(mode: str, pin_time: float, now: float,
                pin_secs: float) -> bool:
    """Whether the orthogonal arm is live: a mode is selected AND the
    primary (x) frontier has been pinned at least `pin_secs`. Reuses the
    solver's own self-measured `_pin_time` — the same saturation signal
    the heuristic-inversion window already runs on, no new machinery and
    no external map."""
    return mode not in (None, "", "off") and now - pin_time >= pin_secs


def count_wmax(door_weight: float, ortho_weight: float) -> float:
    """Exact Wmax for the count arm's O(1) rejection sampling. The prior
    is W = 1/sqrt(times_chosen+1) * (score_norm + 0.1) <= 1.1, times any
    armed multiplier (R4 doors, orthogonal frontier). Under-stating it
    would silently truncate the prior; 1.1 is the legacy value both
    multipliers reduce to when off."""
    return 1.1 * max(door_weight, 1.0) * max(ortho_weight, 1.0)


class Solver:
    def __init__(self, args) -> None:
        # Cell granularity globals are consumed by cell_fn at call time.
        # main() sets them for subprocess runs; IN-PROCESS constructions
        # (the live show) previously bypassed that, silently ignoring
        # args.gx_bucket/y_band — the escalation ladder's micro-search
        # arm would have been a no-op. Apply here so both paths agree
        # (subprocess path re-applies the same values: no change).
        global GX_BUCKET, Y_BAND
        GX_BUCKET = int(getattr(args, "gx_bucket", GX_BUCKET))
        Y_BAND = int(getattr(args, "y_band", Y_BAND))
        self.args = args
        self.out = Path(args.out)
        (self.out / "solutions").mkdir(parents=True, exist_ok=True)
        profile = yaml.safe_load(Path(args.profile).read_text())
        self.bitmasks = action_space_to_bitmasks(profile["action_space"])
        self.weights = np.array(action_weights(profile["action_space"]))
        self.weights /= self.weights.sum()
        self.inv_weights = np.array(inverted_weights(profile["action_space"]))
        self.inv_weights /= self.inv_weights.sum()
        self.game = make_game(profile)
        # Hardware-flag selection (opt-in; empty by default, so a run
        # with neither --hw-flags nor solve.hw_flags: is bit-identical
        # to every pre-existing seeded solve). Resolved BEFORE the pool
        # exists so a bad name fails before any work is done.
        self.hw_flags = resolve_hw_flags(profile, getattr(args, "hw_flags", None))
        self.frame_skip = int(profile.get("frame_skip", 4))
        self.pool = Pool(rom_path=self.game.rom, num_workers=args.workers,
                         frame_skip=self.frame_skip)
        # MUST precede reset_all() — see apply_hw_flags' docstring.
        apply_hw_flags(self.pool, self.hw_flags)
        self.pool.set_headless(True)
        self.pool.set_skip_preprocess(True)
        self.pool.reset_all()
        self.provenance = hw_provenance(self.hw_flags, self.frame_skip)
        if self.hw_flags:
            print(f"[go_explore_solve] hw flags: {self.hw_flags}", flush=True)
        self.rng = np.random.default_rng(args.seed)
        # Sustained-hold macros (generic mechanism, profile-selected): at a
        # small per-step probability a worker settles briefly then HOLDS one
        # input for N steps — producing the long consecutive holds that
        # stochastic sampling almost never emits (stair mounts, pipe
        # entries). The 8-4 pipe-entry macro was this mechanism hardcoded;
        # macro steps are recorded verbatim in traces like any other action.
        self.macros = []          # (action_idx, hold_steps, weight)
        self.macro_p = 0.0
        for m in profile.get("solve", {}).get("hold_macros", []):
            want = set(m["buttons"])
            idx = next((i for i, c in enumerate(profile["action_space"])
                        if set(c) == want), None)
            if idx is None:
                print(f"[solver] hold_macro {m['buttons']} not in "
                      f"action_space — skipped", flush=True)
                continue
            p = float(m.get("p", 0.02))
            self.macros.append((idx, int(m.get("steps", 20)), p))
            self.macro_p += p
        if self.macros:
            w = np.array([m[2] for m in self.macros])
            self._macro_weights = w / w.sum()
        # Discrete-transition gate (the Kirby lesson). When solve.room_advance
        # is configured, auto-derive door/room-entry maneuver macros from the
        # action space (derive_transition_macros) and inject them with elevated
        # probability ONLY when a worker sits at the deep frontier — where the
        # coordinate frontier has stalled and the next progress is a discrete
        # 'grounded + UP' door entry, not another step rightward. Reuses the
        # hold_macros settle-then-hold machinery; empty (inert) for every SMB /
        # non-room profile, so the default path is byte-identical.
        ra = (profile.get("solve", {}) or {}).get("room_advance")
        self.transition_macros = derive_transition_macros(
            profile["action_space"], ra)
        self.room_advance_addr = (int(ra["addr"]) if ra and "addr" in ra
                                  else None)
        self.transition_p = float(ra.get("p", 0.05)) if ra else 0.0
        self.transition_near = int(ra.get("near", 24)) if ra else 24
        self._transition_injections = 0   # telemetry: door macros injected
        self.max_room = 0                 # telemetry: deepest room_advance byte
        if ra and not self.transition_macros:
            print("[solver] room_advance configured but action_space has no "
                  "'up' maneuver — discrete-transition gate inert", flush=True)
        self.archive = GoExploreArchive(self.game.cell_fn, seed=args.seed)
        self.traces: dict = {}            # cell key -> (root_id, trace bytes)
        self.roots: dict = {}             # root_id -> {path, start_wd, lives}
        self.start_wd = (0, 0)
        self.start_lives = 0
        # Progress = (area order, gx). We track the DEEPEST area reached and
        # the max gx within it, so the deep-frontier bias follows Mario into
        # the underground instead of pinning on the entrance area.
        self.max_area = 0
        self.max_gx_in_area: dict = {}    # area -> max gx seen
        self._pin_time = time.time()      # last frontier advance (inversion gate)
        # Resolved once (the gate is read per worker per step).
        self._inv_pin_secs = resolve_inversion_pin_secs(args)
        self._loop_dest_min = None        # min gx observed right after a loop
        self.max_sect = 0                 # deepest section-transit count seen
        self.n_solutions = 0
        self.sol_counter = 0
        self.best_sol_len = 10 ** 9
        # REPLAY-VERIFIED BANKING (v2, 2026-08-08). Default ON — measured
        # cost is ~2.7 ms per action of one candidate (see replay_verify),
        # i.e. seconds against a 25-45 minute solve. Absent attribute (the
        # live show's in-process SimpleNamespace) also resolves ON: banking
        # trust is not something an older call site should silently opt out
        # of. --no-verify-bank is the explicit escape.
        self.verify_bank = resolve_verify_bank(args)
        self.verify_checks = 0
        self.verify_rejections = 0
        self.steps_done = 0
        self.t0 = time.time()
        self.stop = False
        # Maze-coverage recipes (research round 2), flag-gated so the
        # default path stays byte-identical to the receipted campaign:
        # sel_mode "count" swaps the uniform arm for Go-Explore's native
        # count-based prior W = 1/sqrt(times_chosen+1) * (score_norm +
        # 0.1) via O(1) rejection sampling (the archive's own weighted
        # selection was bypassed for being an O(N) scan — this restores
        # the prior without the scan). frontier_throttle N > 0 arms
        # DD-RRT-style boundary suppression: a cell whose bursts yield
        # nothing novel N times in a row is excluded from the deep-
        # frontier band (stop throwing bursts at the physical wall).
        self.sel_mode = str(getattr(args, "sel_mode", "legacy"))
        self.frontier_throttle = int(getattr(args, "frontier_throttle", 0))
        # R4 door discovery (research round 2): cut vertices of the archive
        # transition graph are the maze's doors — every route between the
        # regions they separate passes through them. Edges are recorded from
        # our own rollouts (cell-to-cell transitions, no game internals); an
        # async pass re-derives articulation points every door_interval and
        # the count arm up-weights door cells by door_weight. 0 = off, and
        # the default path stays byte-identical to the receipted campaign.
        self.door_weight = float(getattr(args, "door_weight", 0.0))
        self.door_interval = float(getattr(args, "door_interval", 45.0))
        # ORTHOGONAL-FRONTIER arm (CV hall diagnosis, 2026-08-08), flag-
        # gated, default off => byte-identical to the receipted SMB
        # lineage. The solver's only reward axis is x (score = sect*10000
        # + gx, and CV's sect never moves), so the deep arm samples only
        # the deepest gx buckets while every climbing cell sits at LOW gx:
        # a vertical frontier is unreachable by construction, and the
        # count arm's score term penalizes it a second time. When the x
        # frontier has been pinned for `ortho_pin_secs` (the same self-
        # measured _pin_time the inversion window uses), a third arm
        # samples the top-of-column cells of the orthogonal axis with a
        # PURE count prior — no score term, so the x-axis score cannot
        # suppress a climb. SELECTION-SIDE ONLY: `score` and the cell key
        # are untouched, so domination, solution receipts, cross-run
        # comparability and --resume-archive all keep working.
        self.ortho_mode = str(getattr(args, "ortho", "off"))
        self.ortho_pin_secs = float(getattr(args, "ortho_pin_secs", 120.0))
        self.ortho_bias = float(getattr(args, "ortho_bias", 0.30))
        self.ortho_band = int(getattr(args, "ortho_band", 1))
        self.ortho_weight = float(getattr(args, "ortho_weight", 4.0))
        self.ortho_macro_p = float(getattr(args, "ortho_macro_p", 0.0))
        self._ortho_pool: list = []       # cached top-of-column cells
        self._ortho_ids: set = set()      # their keys, for O(1) membership
        self._ortho_ext: dict = {}        # gx bucket -> column extreme
        self._ortho_best = None           # best y-band any worker reached
        self._ortho_time = time.time()    # last orthogonal improvement
        self._ortho_deep_yband = None     # extreme y-band AT the x frontier
        self._ortho_selections = 0
        self._ortho_cols_improved = 0
        # v6-report recipes (2026-07-30), flag-gated, default byte-identical:
        # time_bins: append floor(log2(steps+1)) to the key prefix (after
        # sect, so key[0]/key[-N] indexing is untouched). Time-Myopic
        # Go-Explore: slow structural trajectories (stairs, waits) stop
        # losing the fewer-steps domination tie-break to drive-by states,
        # and wait-gated triggers become explorable at archive scale.
        # kill_key: append a capped cumulative entity-kill count (observed
        # 1->0 transitions of profile-configured entity-slot flags) —
        # prerequisite/kill-count gates become a searchable dimension.
        self.time_bins = bool(getattr(args, "time_bins", False))
        self.kill_key = bool(getattr(args, "kill_key", False))
        self.kill_key_local = bool(
            (profile.get("solve", {}) or {}).get("kill_key_local", False))
        eslots = (profile.get("solve", {}) or {}).get("entity_slots")
        # Optional stride: object-array engines (Contra: state bytes at
        # $0368 + k*$20) space their per-entity flags; contiguous ranges
        # (CV: $0450..$0457) are stride 1, the default.
        self.entity_slots = (range(int(eslots["lo"]), int(eslots["hi"]),
                                   int(eslots.get("stride", 1)))
                             if eslots else None)
        self._key_ids: dict = {}          # cell key -> int id (interned)
        self._adj: dict = {}              # id -> set of neighbor ids
        self._doors: frozenset = frozenset()
        self._door_lock = threading.Lock()
        self._door_thread: threading.Thread | None = None
        self._last_door_t = time.time()
        self._flush_thread: threading.Thread | None = None
        self._recorded_new = False
        # Optional spectator hook: called every pool step with
        # (results, solver) — the FULL per-worker results list, so the
        # live show can render one worker or the whole swarm. None in
        # headless runs (zero overhead).
        self.step_hook = None

    def _step0(self, a: int):
        acts = np.zeros(self.args.workers, dtype=np.uint8)
        acts[0] = self.bitmasks[a]
        return self.pool.step_all(acts)[0][2]

    # ---- record path -------------------------------------------------

    def observe(self, wid: int, ram, trace: list, steps: int,
                root_id: str, loops: int = 0,
                route_sig: tuple = (), sect: int = 0,
                psig: tuple = (), ctx: dict | None = None,
                kills: int = 0) -> str:
        """Record one reached state. Returns 'dead' | 'clear' | 'live'."""
        game = self.game
        # DEATH IS RESOLVED FIRST (v2, 2026-08-08). Any life lost, or an
        # explicit dying state = death; lives-based detection is robust
        # across enemy/pit/time deaths and multi-area levels.
        #
        # This used to run LAST, after is_finale and is_clear, which made
        # every clear-detector false positive win the race unconditionally:
        # on Gradius a real death (lives 3->2, progress reset to 0) was
        # banked as a win at 205 actions, and go_explore_chain.py would then
        # have extracted the next level's "entrance" out of a corpse. A step
        # that reads as BOTH dead and clear now resolves dead — a missed real
        # clear costs one re-search, a fabricated one poisons the chain and
        # the receipt corpus.
        #
        # Verified against every banked receipt this reorder could touch
        # before shipping it, so it is unconditional rather than flag-gated:
        # replaying regress_pre 1-1, ge_1_2/1_3/1_4/2_1 sol_000, the 4-4 maze
        # solve (1,104 actions) and the 8-4 FINALE (1,735 actions) from their
        # own roots reproduces the clear at exactly the recorded action count
        # with is_dead() FALSE at that frame in all 7 cases. Nothing banked
        # depends on the old order.
        if game.is_dead(ram, self.start_lives):
            return "dead"
        # GAME-COMPLETE check (8-4 finale): the ending never advances the
        # world/level bytes (there is no next level) — the victory state is
        # operating mode $0770 == 2 with inputs locked (verified 2026-07-27,
        # THANK YOU MARIO screen). Without this the winning trajectory sits
        # in the archive invisible, as it did for 1.5 hours on the night the
        # game was first beaten.
        #
        # Then the warp-guarded clear. ctx carries the per-worker state the
        # optional WIN-CONDITION hook needs (score_jump prev value, streaming-
        # confluence window); None at seed = level_key-only. Both go through
        # _dump_solution, which replay-verifies the candidate before anything
        # is written; a candidate that fails to reproduce is NOT a clear, so
        # the lineage is abandoned as dead rather than left to re-fire on the
        # same latched evidence next step.
        if game.is_finale(self.start_wd, ram):
            return ("clear" if self._dump_solution(root_id, trace, ram, steps)
                    else "dead")
        if game.is_clear(self.start_wd, ram, ctx):
            return ("clear" if self._dump_solution(root_id, trace, ram, steps)
                    else "dead")
        # A level-key change that ISN'T a forward clear = a warp (or a
        # backward reload / game-over reset): do not record it (would
        # poison the archive with off-level cells). DEBOUNCED (2026-08-06):
        # transition animations can blip the key byte through stale values
        # — Bubble Bobble's umbrella-warp counter dips to the PREVIOUS
        # round mid-animation, and the un-debounced kill executed every
        # lineage ~6 steps into the warp (round 52: 1,666 kills, 0 cells,
        # 0 solutions across 3 runs). A momentary blip now survives
        # (nothing is recorded during it); only a PERSISTENT off-key state
        # (a real reset/warp, >= 3 consecutive observations) still dies.
        if game.level_key(ram) != self.start_wd:
            if ctx is not None:
                n = ctx["_key_mm"] = ctx.get("_key_mm", 0) + 1
                if n < 3:
                    return "live"   # blip: keep stepping, record nothing
            return "dead"
        elif ctx is not None:
            ctx["_key_mm"] = 0
        gx = game.progress(ram)
        if gx > game.progress_cap:
            # Transition-frame garbage read (page byte mid-load reads huge);
            # real SMB levels reach ~6,300 px (8-1) — the old 3900 cap silently
            # froze the 8-1 frontier at 3900 (states past it never archived).
            return "live"
        area = game.area(ram)
        if gx > self.max_gx_in_area.get(area, 0):
            self.max_gx_in_area[area] = gx
            self._pin_time = time.time()  # frontier moved: inversion stays off
        if area > self.max_area:
            self.max_area = area
        # Orthogonal frontier: a NEW best y-band forces a selection-cache
        # rebuild immediately. The cache otherwise waits for the 2%-growth
        # trigger (~2 min at CV's ~700 cells/min), so a freshly-climbed
        # cell would sit unselectable for exactly as long as the climb
        # takes to decay — the arm would never compound its own progress.
        if self.ortho_mode != "off":
            yb = game.y(ram) // Y_BAND
            best = self._ortho_best
            if best is None or (yb < best if self.ortho_mode == "up"
                                else yb > best):
                self._ortho_best = yb
                self._ortho_time = time.time()
                self._sel_cells = None
        # Domination score = gx within the cell; deeper-x wins, ties to fewer
        # steps (elites keep shortening). Peek-then-record: only pay
        # save_worker_state for a new/dominating cell.
        # MAZE FIX (2026-07-24): the trajectory's loop-count is the LEADING
        # key component. Castle mazes (4-4/7-4/8-4) loop wrong paths back —
        # observable as a discontinuous gx->0 collapse in our own rollouts —
        # so gx alone aliases first-pass and looped-back states and the
        # frontier saturates (4-4: pinned at gx 2064, 0 solutions, 1.5M
        # records). With the counter, the same coordinates on different
        # maze phases are DIFFERENT cells and search explores each pass.
        # Search-derived (a property of the agent's own path), no internals.
        #
        # SECOND MAZE FIX (same day, v4): the loop-count alone still aliased
        # right-route and wrong-route states at the same coordinates — the
        # game tracks the current pass's route in internal state, and the
        # wall-2102 diagnostic showed the deepest lineages were wrong-route
        # spirals from which EVERY action loops back. (A raw RAM-hash key
        # separated routes but exploded the archive to 585k one-visit cells
        # — timers/enemies churn every frame.) The ROUTE SIGNATURE is the
        # compact middle path: the trajectory's own y-band at each 512-px
        # gx-boundary crossing since the last loop event — the observable
        # footprint of the route the current pass has taken. Derived purely
        # from our own rollout; bounded cardinality (<=4 entries, 4 bands).
        # v5: key on the DISCOVERED route-tracker bytes. Differential analysis
        # of our own rollouts (32 passes, /tmp/ram_diff2.py) found $0742 and
        # $07F8 change rarely, at single consistent fork positions — the
        # empirical signature of the game's route-tracking state. Keying on
        # their VALUES separates right-route from wrong-route states with
        # bounded cardinality (vs the raw-hash explosion / sig blindness).
        # Archive-eligibility gate (8-4 lesson): lineages beyond loop-phase 6
        # are wrong-route spirals — they reach deep gx but are unwinnable, and
        # archiving them bloated the archive to 3.6M cells (sps 3300->1700)
        # while poisoning deep-cell selection. They may still EXPLORE (their
        # forks feed discovery) but do not enter the selection pool.
        if loops > 6:
            return "live"
        if sect > self.max_sect:
            self.max_sect = sect
            self._pin_time = time.time()
        # v7: CONTENT-aware cells. Hypothesis shift — the seam wrap (gx->0)
        # may fire on the CORRECT route too, loading the NEXT section's
        # layout; coordinate keys then alias post-seam progress with
        # wrong-route repeats and every gx-based metric is blind to success
        # by construction. The tile buffer ($0500-$06BF, the drawn layout) is
        # the observable content signature: same coords + same section hash
        # alike, same coords + ADVANCED section hash differently. Bounded
        # cardinality (distinct layouts only), unlike the v4 full-RAM hash.
        # SECTION-aware key (8-4 fix, verified empirically 2026-07-25):
        # correct pipe transits change the section pointer $0750 while the
        # area byte $0760 stays constant — confirmed by differential
        # comparison of our own pre/post-gate archive states (2 -> 229).
        # The transit count leads the key and dominates the score, so
        # section progress is the frontier even when gx jumps backward.
        # psig = the last-4 section-pointer values = PIPE-PATH IDENTITY.
        # sect alone aliases different pipe sequences at equal transit
        # counts; 8-4's final checks discriminate by the route taken, so
        # each path is its own frontier (fix 2026-07-26).
        tb = int(steps + 1).bit_length() if self.time_bins else 0
        kk = min(int(kills), 15) if self.kill_key else 0
        key = (sect, tb, kk, psig, loops, route_sig) + game.cell_fn(ram)
        # R4 edge recording: a cell-to-cell transition in OUR OWN rollout is
        # an edge of the maze's traversal graph. Interned ids keep the
        # adjacency compact at castle-archive scale (1M+ cells).
        if ctx is not None and self.door_weight > 0:
            pk = ctx.get("cur_key")
            if pk is not None and pk != key:
                ids = self._key_ids
                ia = ids.get(pk)
                if ia is None:
                    ia = ids[pk] = len(ids)
                ib = ids.get(key)
                if ib is None:
                    ib = ids[key] = len(ids)
                with self._door_lock:
                    self._adj.setdefault(ia, set()).add(ib)
                    self._adj.setdefault(ib, set()).add(ia)
            ctx["cur_key"] = key
        score = sect * 10000 + gx + game.score_bonus(ram)
        cur = self.archive.cells.get(key)
        dom = (cur is None or score > cur.best_score + 1e-9
               or (abs(score - cur.best_score) <= 1e-9 and steps < cur.best_steps))
        if dom:
            blob = self.pool.save_worker_state(wid)
            if blob is not None and self.archive.record(ram, blob, score, steps,
                                                        key=key):
                self.traces[key] = (root_id, bytes(trace), loops, route_sig,
                                    sect, psig, kills)
                self._recorded_new = True
                # R2 credit signal (fix 2026-08-09): _assign() debits the
                # source cell's `barren` counter for every burst and reads
                # `prev["yielded"]` to credit the productive ones — but
                # NOTHING ever wrote that key, so the credit branch was
                # dead and `barren` was really "times ever selected". With
                # --frontier-throttle N every archive cell went permanently
                # barren after N selections, which silently retired the
                # deep-frontier band AND the orthogonal arm (its candidate
                # loop skips barren cells and then has no pick to return).
                # Receipt: runs/bubble_bobble/r68_retry_ortho.log — armed
                # 1,800 s, ortho_pool 18, ortho_selections 0. The burst
                # taught the archive something (a new cell or a dominating
                # improvement), so its root is not a wall.
                if ctx is not None:
                    ctx["yielded"] = True
        else:
            self.archive.record(ram, None, score, steps, key=key)
        return "live"

    def replay_verify(self, root_id: str, trace) -> dict:
        """Re-simulate `trace` from its own root in a FRESH single-worker pool
        and report whether the clear reproduces.

        The whole banking chain — go_explore_chain.py's next-entrance
        extraction, CLAIMS.md receipts, the show ledger — trusts one bit
        produced by a detector that has fabricated wins on three of seven
        games (2026-08-06: a Gradius death, a Double Dragon combat blip, a
        Kirby room transition). A clear that cannot be reproduced from the
        root is not a clear, and the check is cheap enough to be
        unconditional: measured on this machine, a full replay of a banked
        solution costs ~2.7 ms per action (887 actions 2.4 s, 1,104 actions
        3.3 s, 1,735 actions 4.7 s; pool construction is not measurable
        against it), against solve runs of 25-45 minutes that dump a handful
        of candidates. Hence --verify-bank defaults ON, with --no-verify-bank
        as the escape hatch.

        FRESH pool, never the search pool: the search pool's workers hold live
        lineage state and its worker 0 is mid-burst. Verdict is judged against
        the SAME start_wd/start_lives baseline the live hook used, so this
        reproduces the live judgement rather than inventing a second one.
        A fresh ctx is used so the streaming detector must re-earn its own
        evidence rather than inheriting the latch that produced the fire.

        A windowed clear hook also gets a NOOP MARGIN past the end of the
        trace (GenericGame.clear_verify_margin, 0 for every stateless hook, so
        SMB/CV replays are unchanged). The live and replayed detectors count
        their evaluation stride from different origins — per-burst vs. root —
        and the trace ends on the live fire frame, so without a margin a
        genuine confluence clear fails to reproduce ~68% of the time purely on
        phase. Only the CLEAR check runs in the margin: is_dead stays bounded
        to the trace, because what happens to an idle player AFTER the win is
        not evidence against it.

        Returns {"ok": bool, "verdict": str, "at": int|None, "elapsed_s":
        float, "n_actions": int, "margin": int[, "error": str]} — never
        raises: an infrastructure failure is reported as a non-reproduction
        (fail closed), because banking an unverified clear is the outcome this
        exists to prevent. `at` > n_actions means the clear reproduced inside
        the margin (a phase difference, not a different trajectory)."""
        t0 = time.time()
        out = {"ok": False, "verdict": "error", "at": None,
               "n_actions": len(trace), "margin": 0, "elapsed_s": 0.0}
        pool = None
        try:
            root = self.roots.get(root_id)
            if root is None:
                raise KeyError(f"unknown root_id {root_id!r}")
            blob = Path(root["path"]).read_bytes()
            pool = Pool(rom_path=self.game.rom, num_workers=1,
                        frame_skip=self.frame_skip)
            # Same recipe (and the same load-bearing ordering) as the
            # search pool, so the replay runs the SAME machine.
            apply_hw_flags(pool, self.hw_flags)
            pool.set_headless(True)
            pool.set_skip_preprocess(True)
            pool.reset_all()
            pool.load_worker_state(0, blob)
            acts = np.zeros(1, dtype=np.uint8)
            pool.step_all(acts)          # rooting NOOP, exactly as seed() does
            ctx: dict = {}
            n = len(trace)
            margin = self.game.clear_verify_margin()
            out["margin"] = margin
            # NOOP tail, exactly as clear_detect.run_episode pads an episode
            # for the same reason. The trajectory is not extended; the hook is
            # only given the observations it needs to reach the verdict it
            # already reached live.
            for i, a in enumerate(list(trace) + [0] * margin):
                acts[0] = self.bitmasks[int(a)]
                ram = pool.step_all(acts)[0][2]
                if i < n and self.game.is_dead(ram, self.start_lives):
                    out.update(verdict="dead", at=i + 1)
                    break
                if (self.game.is_finale(self.start_wd, ram)
                        or self.game.is_clear(self.start_wd, ram, ctx)):
                    out.update(ok=True, verdict="clear", at=i + 1)
                    break
            else:
                out.update(verdict="no_clear", at=None)
        except Exception as exc:                      # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if pool is not None:
                try:
                    pool.shutdown()
                except Exception:                     # noqa: BLE001
                    pass
        out["elapsed_s"] = round(time.time() - t0, 3)
        return out

    def _dump_solution(self, root_id: str, trace: list, ram, steps) -> bool:
        """Bank a solution. Returns whether the event counts as a REAL clear.

        False means the candidate failed replay verification — it was not a
        clear at all, so observe() abandons the lineage instead of reporting
        one. True with nothing written is the ordinary "re-clear that isn't
        materially shorter" case: real, just not worth another receipt."""
        if len(trace) >= self.best_sol_len - 8:
            return True  # keep only materially-shorter re-clears
        if self.verify_bank:
            v = self.replay_verify(root_id, list(trace))
            self.verify_checks += 1
            if not v["ok"]:
                self.verify_rejections += 1
                sys.stderr.write(
                    f"[go_explore_solve] *** CLEAR REJECTED (replay "
                    f"verification) *** root={root_id} {len(trace)} actions "
                    f"did NOT reproduce: {json.dumps(v)} — NOTHING banked. "
                    f"This is a fabricated clear, not a near miss: treat the "
                    f"profile's clear hook as unsafe until it is explained.\n")
                sys.stderr.flush()
                print(f"[go_explore_solve] CLEAR REJECTED {json.dumps(v)}",
                      flush=True)
                return False
            print(f"[go_explore_solve] replay-verified clear "
                  f"{json.dumps(v)}", flush=True)
        self.best_sol_len = len(trace)
        n = self.sol_counter
        self.sol_counter += 1
        self.n_solutions += 1
        base = self.out / "solutions" / f"sol_{n:03d}"
        np.save(str(base) + ".actions.npy", np.array(trace, dtype=np.int64))
        (base.parent / (base.name + ".json")).write_text(json.dumps({
            "provenance": "search",
            # Full effective invocation (audit finding: receipts recorded
            # neither workers nor profile nor buckets, making parity
            # claims between runs unverifiable after the fact).
            "solver_args": {k: v for k, v in vars(self.args).items()
                            if isinstance(v, (str, int, float, bool))},
            "root_id": root_id,
            "root_state": self.roots[root_id]["path"],
            # The machine this trace was produced on. A solution replayed
            # under different hw flags is not the same trajectory.
            #
            # NAMED `hw_provenance`, NOT `provenance`: `"provenance":
            # "search"` above is the honest-origin marker every banked
            # receipt carries and that CLAIMS.md audits compare against
            # the literal string (docs/proposals/smb_oneshot_campaign.md
            # 543/596). Reusing the key here shadowed it silently — a
            # dict literal keeps the LAST binding, so the marker was
            # deleted with no error. The token `provenance` means
            # exactly one thing corpus-wide: origin. The machine
            # description travels under `hw_provenance` everywhere
            # (receipts, roots.json, archive.stats.json).
            "hw_provenance": self.provenance,
            "start_wd": list(self.start_wd),
            "clear_wd": list(self.game.level_key(ram)),
            "steps": steps, "actions": len(trace),
            # Whether this receipt's clear was re-simulated from the root in a
            # fresh pool before it was written. Recorded so an audit can tell a
            # verified receipt from a pre-v2 (or --no-verify-bank) one without
            # re-running anything.
            "replay_verified": bool(self.verify_bank),
        }, indent=2) + "\n")
        print(f"[go_explore_solve] *** SOLUTION {n} *** root={root_id} "
              f"{len(trace)} actions, {self.start_wd}->"
              f"{self.game.level_key(ram)}", flush=True)
        return True

    # ---- seeding: root ONLY (honest) ---------------------------------

    def seed(self) -> None:
        path = self.args.root_state
        # The root blob carries no timing config of its own; if a
        # sidecar records the lineage it was built under, say so before
        # a mismatch turns into hours of "the game behaves differently".
        check_state_sidecar(path, self.hw_flags)
        self.pool.load_worker_state(0, Path(path).read_bytes())
        r = self._step0(NOOP)  # convention: load root, one NOOP, then actions
        self.start_wd = self.game.level_key(r)
        self.start_lives = self.game.lives(r)
        self.max_area = self.game.area(r)
        # WIN-CONDITION byte_change: capture the entrance baseline the hook
        # compares against (no-op for SMB / non-configured profiles).
        if hasattr(self.game, "note_start"):
            self.game.note_start(r)
        self.roots["entrance"] = {"path": str(path),
                                  "start_wd": list(self.start_wd),
                                  "lives": self.start_lives,
                                  "hw_provenance": self.provenance}
        self.observe(0, r, [], 0, "entrance")
        prev = getattr(self.args, "resume_archive", None)
        if prev:
            prev = Path(prev)
            self.archive.load(prev / "archive.pkl")
            with open(prev / "traces.pkl", "rb") as f:
                self.traces.update(pickle.load(f))
            saved_roots = json.loads((prev / "roots.json").read_text())
            for rid, info in saved_roots.items():
                self.roots.setdefault(rid, info)
            # Rebuild the frontier trackers the loaded cells imply.
            for c in self.archive.cells.values():
                area, gx = c.key[-5], c.key[-1] * GX_BUCKET
                sect = c.key[0]
                if area > self.max_area:
                    self.max_area = area
                if gx > self.max_gx_in_area.get(area, 0):
                    self.max_gx_in_area[area] = gx
                if sect > self.max_sect:
                    self.max_sect = sect
            print(f"[seed] RESUMED archive from {prev}: "
                  f"{len(self.archive.cells)} cells, "
                  f"{len(self.traces)} traces, max_area={self.max_area}, "
                  f"max_sect={self.max_sect}", flush=True)
        print(f"[seed] rooted at {path} wd={self.start_wd} lives="
              f"{self.start_lives} area={self.max_area}; archive="
              f"{json.dumps(self.archive.stats())}", flush=True)

    # ---- frontier selection ------------------------------------------

    def _refresh_sel_cache(self) -> None:
        """Rebuild the selection caches. A full archive scan ran on EVERY
        worker reassignment (every episode end) — O(330k) three times over
        (state filter, deep filter, weighted fallback) — starving the Rust
        pool at 1.6/16 cores with sps decayed 2378->475. Rebuild only when
        the archive grows 2% or the deepest area advances."""
        self._sel_cells = [c for c in self.archive.cells.values()
                           if c.state is not None]
        deep_area = self._sel_cells
        if self.ortho_mode != "off":
            # Restrict the ORTHO POOL — and only it — to the DEEPEST area:
            # an earlier area's column ceilings are not the wall being
            # climbed, and mixing them would let a tall stale column
            # define this one's frontier. Keyed on the MODE (static), not
            # on armed-ness (time-varying) — a cache whose contents depend
            # on when it happened to be rebuilt is not reproducible.
            #
            # SCOPED TO THE ARM (fix): this filter used to overwrite
            # self._sel_cells, i.e. it restricted the ENTIRE selection
            # pool. record() bumps self.max_area BEFORE the `loops > 6`
            # archive-eligibility early-out, so a castle-maze spiral
            # (4-4/7-4/8-4 — the exact case the loop counter exists for)
            # that crosses an area boundary advances max_area with ZERO
            # state-bearing cells in that area. The pool then emptied,
            # select() returned None, and _assign reset EVERY worker to
            # the entrance root — permanently, because entrance bursts
            # archive into the earlier area that the same filter discards.
            # Reproduced against snapshot e0f59e9 (6 cells in area 3,
            # max_area 4: mode=off -> 6 cells; mode=up -> 0, select None).
            # The ortho arm is additive: an empty deep-area subset now
            # just makes the arm fall through, exactly as an evicted pool
            # already did, and the count/legacy arms keep the full pool.
            deep_area = [c for c in self._sel_cells
                         if c.key[-5] == self.max_area]
            ext = column_extremes([c.key for c in deep_area],
                                  self.ortho_mode)
            better = min if self.ortho_mode == "up" else max
            prev = self._ortho_ext
            # Cumulative (monotone) count of column ceilings this run has
            # pushed — the arm's own partial-progress signal, independent
            # of whether a solution ever lands.
            self._ortho_cols_improved += sum(
                1 for col, yb in ext.items()
                if col in prev and yb != prev[col]
                and better(yb, prev[col]) == yb)
            self._ortho_ext = ext
            self._ortho_pool = ortho_pool(deep_area, self.ortho_mode,
                                          self.ortho_band)
            self._ortho_ids = {c.key for c in self._ortho_pool}
        self._sel_n = len(self.archive.cells)
        self._sel_area = self.max_area
        self._sel_maxscore = max(
            (c.best_score for c in self._sel_cells), default=1.0) or 1.0
        deep = [c for c in self._sel_cells
                if c.key[0] == self.max_sect and c.key[-5] == self.max_area]
        self._sel_deep = deep
        if deep:
            minl = min(c.key[0] for c in deep)
            lowl = [c for c in deep if c.key[0] <= minl + 1]
            self._sel_topgx = max(c.key[-1] for c in deep)
            # Near-frontier bands precomputed (the per-call band filter over
            # `deep` was itself O(N) in one-area castle levels).
            _f24 = self._sel_topgx - 24
            self._sel_band24 = [c for c in deep if c.key[-1] >= _f24]
            self._sel_lowl_band24 = [c for c in lowl if c.key[-1] >= _f24]
        else:
            self._sel_topgx = 0
            self._sel_band24 = []
            self._sel_lowl_band24 = []
        if self.ortho_mode != "off":
            # The literal figure the pre-registered gate names: the
            # extreme y-band reached inside the deep band the primary arm
            # actually samples (gx buckets >= topgx-24), inside the
            # deepest area (the arm's own subset, not the whole pool).
            ybs = [c.key[-2] for c in deep_area
                   if c.key[-1] >= self._sel_topgx - 24]
            self._ortho_deep_yband = (
                (min(ybs) if self.ortho_mode == "up" else max(ybs))
                if ybs else None)

    def _ortho_armed(self) -> bool:
        return ortho_armed(self.ortho_mode, self._pin_time, time.time(),
                           self.ortho_pin_secs)

    def select(self):
        if (getattr(self, "_sel_cells", None) is None
                or len(self.archive.cells) > self._sel_n * 1.02
                or self._sel_area != self.max_area):
            self._refresh_sel_cache()
        cells = self._sel_cells
        if not cells:
            return None
        # Deep-frontier arm: bias toward cells in the DEEPEST area reached,
        # near its max gx — this follows Mario through area transitions.
        if self.rng.random() < self.args.deep_bias:
            pool_band = (self._sel_lowl_band24
                         if (self.rng.random() < 0.7 and self._sel_lowl_band24)
                         else self._sel_band24)
            if pool_band and self.frontier_throttle > 0:
                # DD-RRT boundary suppression: cells whose bursts came
                # back empty `throttle` times in a row are walls — stop
                # sampling them; if the whole band is walls, fall through
                # to the count arm (least-visited exploration).
                pool_band = [c for c in pool_band
                             if getattr(c, "barren", 0) < self.frontier_throttle]
            if pool_band:
                floor = self._sel_topgx - int(self.rng.integers(0, 24))
                band = [c for c in pool_band if c.key[-1] >= floor]
                if band:
                    cell = band[int(self.rng.integers(len(band)))]
                    cell.times_chosen += 1
                    cell.explored = True
                    return cell
        # Orthogonal-frontier arm (between the deep and count arms so
        # deep_bias semantics are unchanged): once the x frontier is
        # pinned, spend `ortho_bias` of the remaining budget on the
        # top-of-column cells with a PURE count prior W =
        # 1/sqrt(times_chosen+1) (Wmax 1.0) — deliberately no score term,
        # since the score is exactly what makes a climb look worthless.
        # Same O(1) rejection sampling and same barren skip as the other
        # arms; an empty pool, or one that is all wall, falls through.
        armed = self._ortho_armed()
        if armed and self._ortho_pool and self.rng.random() < self.ortho_bias:
            opool = self._ortho_pool
            pick = None
            for _ in range(64):
                cand = opool[int(self.rng.integers(len(opool)))]
                if (self.frontier_throttle > 0
                        and getattr(cand, "barren", 0) >= self.frontier_throttle):
                    continue
                pick = cand
                if self.rng.random() < 1.0 / (cand.times_chosen + 1) ** 0.5:
                    break
            if pick is not None:
                pick.times_chosen += 1
                pick.explored = True
                self._ortho_selections += 1
                return pick
        if self.sel_mode == "count":
            # Count-based prior via O(1) rejection sampling: accept cell
            # with prob W/Wmax, W = 1/sqrt(times_chosen+1) * (score_norm
            # + 0.1), Wmax = 1.1. Expected O(few) draws; bounded at 64.
            # R4: door cells (articulation points of the transition graph)
            # get door_weight x — Wmax scales so the sampling stays exact.
            ms = self._sel_maxscore
            doors = self._doors
            dw = self.door_weight if doors else 0.0
            # Orthogonal frontier gets the same treatment as doors: an
            # up-weight inside the count arm, with Wmax scaled so the
            # rejection sampling stays EXACT when both multipliers arm.
            ow = self.ortho_weight if armed else 1.0
            wmax = count_wmax(dw, ow)
            pick = None
            for _ in range(64):
                pick = cells[int(self.rng.integers(len(cells)))]
                # R2 extension (block-3 doomed-tip lesson): a cell whose
                # bursts die yielding nothing `throttle` times in a row is
                # skipped here too — deterministic death states (e.g. a
                # committed bat swoop) otherwise drain the count arm's
                # budget forever, since depth keeps their score high.
                if (self.frontier_throttle > 0
                        and getattr(pick, "barren", 0) >= self.frontier_throttle):
                    continue
                w = ((1.0 / (pick.times_chosen + 1) ** 0.5)
                     * (pick.best_score / ms + 0.1))
                if dw > 0 and self._key_ids.get(pick.key) in doors:
                    w *= dw
                if ow > 1.0 and pick.key in self._ortho_ids:
                    w *= ow
                if self.rng.random() < w / wmax:
                    break
            pick.times_chosen += 1
            pick.explored = True
            return pick
        # Legacy: uniform over the cached list (the archive's weighted
        # selection was an O(N) scan; the deep arm supplies direction).
        return cells[int(self.rng.integers(len(cells)))]

    # ---- R4: door discovery (articulation points, async) --------------

    @staticmethod
    def _articulation_points(adj: dict) -> set:
        """Iterative Hopcroft-Tarjan cut vertices of an undirected graph
        given as {node: iterable-of-neighbors}. Pure derived-from-rollouts
        structure — no game internals."""
        disc: dict = {}
        low: dict = {}
        ap: set = set()
        timer = 0
        for start in adj:
            if start in disc:
                continue
            disc[start] = low[start] = timer
            timer += 1
            root_children = 0
            stack = [(start, None, iter(adj.get(start, ())))]
            while stack:
                node, parent, it = stack[-1]
                pushed = False
                for nb in it:
                    if nb == parent:
                        continue
                    if nb in disc:
                        if disc[nb] < low[node]:
                            low[node] = disc[nb]
                    else:
                        disc[nb] = low[nb] = timer
                        timer += 1
                        if node == start:
                            root_children += 1
                        stack.append((nb, node, iter(adj.get(nb, ()))))
                        pushed = True
                        break
                if not pushed:
                    stack.pop()
                    if stack:
                        pnode = stack[-1][0]
                        if low[node] < low[pnode]:
                            low[pnode] = low[node]
                        if pnode != start and low[node] >= disc[pnode]:
                            ap.add(pnode)
            if root_children > 1:
                ap.add(start)
        return ap

    def _door_scan_qos(self, *a, **k):
        # Housekeeping: run the periodic articulation-point scan on the
        # E-cores (QOS_BACKGROUND) so it never evicts worker cache lines.
        from src.training.qos import demote_current_thread
        demote_current_thread()
        return self._door_scan(*a, **k)

    def _door_scan(self) -> None:
        """Snapshot the adjacency and publish the current door set. Runs in
        a daemon thread; the snapshot copy holds the edge lock briefly (the
        4-4 frozen-show lesson: never scan a live million-entry structure
        from the hot loop)."""
        try:
            with self._door_lock:
                snap = {k: tuple(v) for k, v in self._adj.items()}
            self._doors = frozenset(self._articulation_points(snap))
        except Exception as exc:
            print(f"[door_scan] failed (doors unchanged): {exc}", flush=True)

    def _maybe_scan_doors(self, now: float) -> None:
        if self.door_weight <= 0:
            return
        if now - self._last_door_t < self.door_interval:
            return
        if self._door_thread is not None and self._door_thread.is_alive():
            return
        self._last_door_t = now
        self._door_thread = threading.Thread(target=self._door_scan_qos,
                                             daemon=True)
        self._door_thread.start()

    def _assign(self, wid: int, prev: dict | None = None) -> dict:
        # R2 bookkeeping: credit or debit the cell the finished burst
        # was rooted at. A burst that recorded nothing novel increments
        # the source cell's barren counter; any novelty resets it.
        if prev is not None and prev.get("key") is not None:
            src = self.archive.cells.get(prev["key"])
            if src is not None:
                if prev.get("yielded"):
                    src.barren = 0
                else:
                    src.barren = getattr(src, "barren", 0) + 1
        cell = self.select()
        if cell is None:
            # Fall back to the entrance root.
            self.pool.load_worker_state(wid, Path(self.args.root_state).read_bytes())
            return {"key": None, "root": "entrance", "trace": [], "steps": 0,
                    "left": self.args.burst, "loops": 0, "prev_gx": -1,
                    "sig": (), "sect": 0, "p0750": None, "psig": (),
                    "kills": 0, "eslots": None, "ortho": False,
                    "prev": int(self.rng.choice(len(self.weights), p=self.weights))}
        self.pool.load_worker_state(wid, cell.state)
        rec = self.traces[cell.key]
        root_id, tb, loops, sig = rec[0], rec[1], rec[2], rec[3]
        sect = rec[4] if len(rec) > 4 else 0
        psig = rec[5] if len(rec) > 5 else ()
        # prev_gx -1: no loop detection on the restore step (the load frame
        # reads transitional garbage; the first real step re-arms it).
        return {"key": cell.key, "root": root_id, "trace": list(tb),
                "steps": cell.best_steps, "left": self.args.burst,
                "loops": loops, "prev_gx": -1, "sig": sig,
                "sect": sect, "p0750": None, "psig": psig, "cur_key": cell.key,
                "kills": rec[6] if len(rec) > 6 else 0, "eslots": None,
                # Burst rooted at an orthogonal-frontier cell: explore()
                # rolls the hold-macro at ortho_macro_p for these.
                "ortho": cell.key in self._ortho_ids,
                "prev": int(self.rng.choice(len(self.weights), p=self.weights))}

    def explore(self) -> None:
        args = self.args
        game = self.game
        cap = game.progress_cap
        ctx = [self._assign(i) for i in range(args.workers)]
        acts = np.zeros(args.workers, dtype=np.uint8)
        self.t0 = last_progress = last_flush = time.time()
        self._stall = {"last_cells": 0, "last_t": self.t0, "flat_windows": 0}
        deadline = self.t0 + args.minutes * 60 if args.minutes > 0 else None
        while not self.stop:
            # Pin clock read ONCE per step rather than once per worker:
            # the inversion gate is a call now, so it can no longer
            # short-circuit the time.time() away. Against a 180 s
            # threshold a one-step-stale reading is noise, and this is
            # strictly cheaper than the per-worker read it replaces.
            _pin_elapsed = time.time() - self._pin_time
            for i, c in enumerate(ctx):
                # Heuristic inversion inside the self-measured saturation
                # window [pin-300, pin+60]: the frontier pin is where OUR
                # search saturates (live telemetry, not any external map);
                # inside it, sample from inverted weights so leftward /
                # downward entries get explored instead of pruned.
                # SATURATION-TRIGGERED inversion (fix 2026-07-25): always-on
                # inversion sabotaged standard levels — at 5-3's frontier the
                # solver sampled left/down exactly where a full-speed rightward
                # jump was needed (chain stall). The maze maneuver hunt now
                # arms only after the frontier has been pinned for
                # --inversion-pin-secs (default 180 s = the receipted
                # constant; negative disables the arm outright).
                _pin = self.max_gx_in_area.get(self.max_area, 0)
                _floor = (self._loop_dest_min
                          if self._loop_dest_min is not None
                          else _pin - 300)
                _w = (self.inv_weights
                      if inversion_armed(self._inv_pin_secs, _pin,
                                         c.get("gx", -1), _floor,
                                         _pin_elapsed)
                      else self.weights)
                # Discrete-transition gate (Kirby lesson): at the deep frontier,
                # inject an auto-derived door-entry maneuver (settle then HOLD
                # up) so a room/area advance can fire where coordinate progress
                # has stalled. Checked before the ordinary macro roll so it wins
                # the (shared) macro slot at the frontier; inert everywhere else.
                if (self.transition_macros and c.get("macro_left", 0) <= 0
                        and c.get("at_frontier")
                        and self.rng.random() < self.transition_p):
                    ti = int(self.rng.integers(len(self.transition_macros)))
                    c["macro_a"], c["macro_hold"] = self.transition_macros[ti]
                    c["macro_left"] = c["macro_hold"] + 6
                    self._transition_injections += 1
                # Bursts rooted at an orthogonal-frontier cell roll the
                # hold macro at ortho_macro_p instead of the profile's own
                # rate: a stair mount IS a sustained up/up+right hold, and
                # the profile-declared macros fire at p=0.02, so the one
                # maneuver the climb needs is the rarest thing the sampler
                # emits exactly where it matters. Cheap checks first —
                # with the arm off (macro_p 0) this costs one comparison.
                _mp = (self.ortho_macro_p
                       if (self.ortho_macro_p > 0 and c.get("ortho")
                           and self._ortho_armed())
                       else self.macro_p)
                if (c.get("macro_left", 0) <= 0 and self.macros
                        and self.rng.random() < _mp):
                    mi = int(self.rng.choice(len(self.macros),
                                             p=self._macro_weights))
                    c["macro_a"], c["macro_hold"] = self.macros[mi][:2]
                    c["macro_left"] = c["macro_hold"] + 6   # settle, then hold
                if c.get("macro_left", 0) > 0:
                    c["macro_left"] -= 1
                    a = NOOP if c["macro_left"] >= c["macro_hold"] \
                        else c["macro_a"]
                else:
                    a = c["prev"] if self.rng.random() < args.sticky else \
                        int(self.rng.choice(len(_w), p=_w))
                c["prev"] = a
                c["pending"] = a
                acts[i] = self.bitmasks[a]
            results = self.pool.step_all(acts)
            self.steps_done += args.workers
            if self.step_hook is not None:
                try:
                    self.step_hook(results, self)
                except Exception:
                    pass
            for i, c in enumerate(ctx):
                ram = results[i][2]
                c["trace"].append(c["pending"])
                c["steps"] += 1
                c["left"] -= 1
                # Loop-back detection: a discontinuous backward gx jump on a
                # non-garbage frame advances the trajectory's maze phase
                # (capped so wrong-path spirals can't explode the archive).
                # Route signature: the y-band recorded at each 512-px gx
                # boundary the pass crosses (reset on loop-back) — the
                # observable footprint of THIS pass's route choice.
                _lgx = game.progress(ram)
                c["gx"] = _lgx if _lgx <= cap else c.get("gx", -1)
                # Discrete-transition gate: is this worker sitting AT the deep
                # frontier (deepest area reached, within `near` gx buckets of
                # its max)? If so the next step may inject a door-entry macro.
                # Only computed when the gate is armed (non-SMB room profiles).
                if self.transition_macros:
                    _fgx = self.max_gx_in_area.get(self.max_area, 0)
                    c["at_frontier"] = (
                        game.area(ram) == self.max_area and c["gx"] >= 0
                        and c["gx"] >= _fgx - self.transition_near * GX_BUCKET)
                    if self.room_advance_addr is not None:
                        _rv = int(ram[self.room_advance_addr])
                        if _rv > self.max_room:
                            self.max_room = _rv
                # ROOM IDENTITY (SMB: verified 2026-07-26, $074E changes
                # 0.67/1k steps, only at full-screen transitions). A rid
                # CHANGE is a true room transition.
                if self.kill_key and self.entity_slots is not None:
                    _es = tuple(int(ram[a]) for a in self.entity_slots)
                    pes = c.get("eslots")
                    if pes is not None:
                        # nonzero->zero: CV's slots are 0/1 flags (same
                        # result); object-array engines (Contra) hold
                        # state values, so ==1 missed most despawns.
                        c["kills"] = c.get("kills", 0) + sum(
                            1 for p, q in zip(pes, _es) if p != 0 and q == 0)
                    c["eslots"] = _es
                    # Local mode: reset the counter whenever the
                    # trajectory sets a new progress high-water mark.
                    # Run-and-gun games kill constantly — cumulative kk
                    # saturates its cap long before the wall (Contra v3:
                    # 92% of cells at kk=15, ALL wall cells capped =
                    # zero gradient exactly where the fight is). With
                    # the reset, kk counts kills SINCE progress froze —
                    # at a fixed-camera fight, that IS the fight.
                    if self.kill_key_local:
                        if _lgx <= cap and _lgx > c.get("kk_max_gx", -1):
                            c["kk_max_gx"] = _lgx
                            c["kills"] = 0
                _rid = game.room_id(ram)
                _transit = (c["p0750"] is not None and _rid != c["p0750"])
                if _transit and c["sect"] < self.args.sect_cap:
                    c["sect"] += 1
                    c["psig"] = _rid
                c["p0750"] = _rid
                if _lgx <= cap:
                    if c["prev_gx"] >= 0:
                        if _transit:
                            c["prev_gx"] = _lgx   # section change: re-arm, no loop
                        elif _lgx < c["prev_gx"] - 100:
                            if c["loops"] < 8:
                                c["loops"] += 1
                            c["sig"] = ()   # new pass, fresh route
                        elif _lgx // 512 != c["prev_gx"] // 512:
                            c["sig"] = (c["sig"] + (game.y(ram) // 64,))[-4:]
                    c["prev_gx"] = _lgx
                if (self.args.swim_gx_ceiling > 0 and game.swim(ram) == 1
                        and _lgx <= cap
                        and _lgx > self.args.swim_gx_ceiling):
                    ctx[i] = self._assign(i, prev=c)
                    continue
                status = self.observe(i, ram, c["trace"], c["steps"],
                                      c["root"], c["loops"], c["sig"],
                                      c["sect"], c["psig"], ctx=c,
                                      kills=c.get("kills", 0))
                # Finisher extension: the level-END transition (exit pipe /
                # flag slide) can run many steps with gx frozen, so a burst
                # from the deepest cell can end just short of the wd advance.
                # A burst ending in the deepest-area top band gets one +200
                # extension so it can actually complete the clear.
                if (status == "live" and c["left"] <= 0
                        and not c.get("extended")
                        and game.area(ram) == self.max_area
                        and _lgx // 16 >= self.max_gx_in_area.get(self.max_area, 0) // 16 - 3):
                    c["left"] += 200
                    c["extended"] = True
                if status != "live" or c["left"] <= 0 or c["steps"] >= args.max_steps:
                    ctx[i] = self._assign(i, prev=c)
            now = time.time()
            self._maybe_scan_doors(now)
            if now - last_progress >= 60:
                last_progress = now
                self.progress_line(now - self.t0)
            if now - last_flush >= args.flush_secs:
                last_flush = now
                self._maybe_flush_async()
            if deadline and now >= deadline:
                break
            if not keep_exploring(self.n_solutions, args.want_solutions):
                break

    # ---- reporting / persistence -------------------------------------

    def progress_line(self, elapsed: float) -> None:
        # Flat-archive stall watchdog (ported from live_solve_show.py,
        # 2026-08-06): this standalone driver runs most of the fleet's
        # actual compute but had no equivalent — a run in this exact
        # 60s-cadence path died silently and went unnoticed for 32+ min
        # during the review that flagged this gap. Two consecutive 60s
        # windows (this method's own cadence) with zero new cells warns
        # loudly; it cannot detect the process dying outright (nothing
        # runs after that), only an in-process hang/freeze while alive.
        n_cells = len(self.archive)
        update_stall(self._stall, n_cells, time.time())
        if self._stall["flat_windows"] >= 2:
            sys.stderr.write(
                f"[go_explore_solve] STALL WARNING: stuck at {n_cells} "
                f"cells for {self._stall['flat_windows']} min straight — "
                f"possible frozen/dead state\n")
        line = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed),
            "cells": n_cells,
            "max_area": self.max_area,
            "max_gx_in_max_area": self.max_gx_in_area.get(self.max_area, 0),
            "max_sect": self.max_sect,
            "solutions": self.n_solutions,
            "best_sol_actions": (self.best_sol_len if self.n_solutions else None),
            "steps": self.steps_done,
            "sps": round(self.steps_done / max(elapsed, 1e-9)),
            "stall_flat_windows": self._stall["flat_windows"],
        }
        if self.ortho_mode != "off":
            # ortho_deep_yband is the pre-registered gate figure: the
            # extreme y-band inside the band the PRIMARY arm samples, so
            # a climb that only happens off-frontier cannot flatter it.
            # pinned_secs is the arm's own arming clock, printed so a log
            # reader can tell "inert" from "armed and not working".
            line["ortho_best_yband"] = self._ortho_best
            line["ortho_deep_yband"] = self._ortho_deep_yband
            line["ortho_pool"] = len(self._ortho_pool)
            line["ortho_selections"] = self._ortho_selections
            line["ortho_cols_improved"] = self._ortho_cols_improved
            line["pinned_secs"] = round(time.time() - self._pin_time)
        if self.door_weight > 0:
            line["doors"] = len(self._doors)
            line["edges"] = sum(len(v) for v in self._adj.values()) // 2
        if self.transition_macros:
            line["door_macros_injected"] = self._transition_injections
            if self.room_advance_addr is not None:
                line["max_room"] = self.max_room
        # Fabricated-clear telemetry: silent while nothing has been rejected
        # (so ordinary runs' progress lines are unchanged), loud the moment a
        # candidate fails to reproduce — the four-game detector gate reads
        # exactly this number.
        if getattr(self, "verify_rejections", 0):
            line["verify_checks"] = self.verify_checks
            line["verify_rejections"] = self.verify_rejections
        with open(self.out / "progress.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[go_explore_solve] {json.dumps(line)}", flush=True)

    def flush(self) -> None:
        # A background _flush_async_qos daemon (see _maybe_flush_async) may
        # still be mid-write to the SAME archive.pkl.tmp path this writes
        # to — join it first so the two writers never race the same inode
        # (found by adversarial review, 2026-08-06: explore() can break
        # with a flush thread in flight, and this synchronous flush is the
        # last-chance persist on exit, so it's always correct to wait).
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._flush_thread.join()
        self.archive.save(self.out / "archive.pkl")
        stamp_stats_provenance(self.out / "archive.pkl", self.provenance)
        with open(self.out / "traces.pkl", "wb") as f:
            pickle.dump(self.traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        (self.out / "roots.json").write_text(json.dumps(self.roots, indent=2) + "\n")

    def _maybe_flush_async(self) -> None:
        """Non-blocking periodic flush: a multi-GB archive.pkl pickle dump
        run synchronously on the hot loop is the receipted cause of a
        1563->322 sps stall (src/training/qos.py's docstring). Take a
        cheap shallow-copy snapshot HERE on the solve thread (a dict()
        copy is a single bulk operation under the GIL — safe against the
        'dict changed size during iteration' race a live multi-GB dict
        would hit if handed directly to another thread), then do the slow
        pickling on a QOS_CLASS_BACKGROUND daemon thread so it lands on
        the E-cores and never evicts a P-core worker's cache lines. Skips
        (rather than queuing) if the previous flush hasn't finished yet —
        the search keeps running either way; the next interval tries
        again with fresher data."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        cells_snapshot = dict(self.archive.cells)
        traces_snapshot = dict(self.traces)
        roots_snapshot = dict(self.roots)
        counters = (self.archive.total_records, self.archive.total_new_cells,
                    self.archive.total_improvements)
        self._flush_thread = threading.Thread(
            target=self._flush_async_qos,
            args=(cells_snapshot, traces_snapshot, roots_snapshot, counters),
            daemon=True)
        self._flush_thread.start()

    def _flush_async_qos(self, cells_snapshot, traces_snapshot,
                         roots_snapshot, counters) -> None:
        from src.training.qos import demote_current_thread
        demote_current_thread()
        try:
            snap = GoExploreArchive(self.game.cell_fn, seed=self.args.seed)
            snap._cells = cells_snapshot
            snap.total_records, snap.total_new_cells, snap.total_improvements = counters
            snap.save(self.out / "archive.pkl")
            stamp_stats_provenance(self.out / "archive.pkl", self.provenance)
            with open(self.out / "traces.pkl", "wb") as f:
                pickle.dump(traces_snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            (self.out / "roots.json").write_text(
                json.dumps(roots_snapshot, indent=2) + "\n")
        except Exception as e:
            print(f"[go_explore_solve] background flush failed (search "
                  f"continues, will retry next interval): {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-state", required=True,
                    help="Level ENTRANCE save-state (honest root; no prefix).")
    ap.add_argument("--profile", required=True,
                    help="Source of action_space + frame_skip only.")
    ap.add_argument("--hw-flags", type=str, default=None, metavar="a,b,c",
                    help="Comma-separated nes_core hw timing flags to set on "
                         "the pool (e.g. mmio_read_timing,nmi_poll_timing), "
                         "overriding the profile's solve.hw_flags. "
                         "'none' forces the empty set. DEFAULT: whatever the "
                         "profile pins, else NONE — a state blob records no "
                         "timing config, so a lineage built under flags must "
                         "be re-solved under the same flags or the restored "
                         "machine is not the machine that produced it.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--burst", type=int, default=64)
    ap.add_argument("--sticky", type=float, default=0.5)
    ap.add_argument("--deep-bias", type=float, default=0.4)
    ap.add_argument("--minutes", type=float, default=0,
                    help="Wall-clock budget (0 = until solution quota).")
    ap.add_argument("--want-solutions", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--flush-secs", type=float, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-bank", dest="verify_bank", action="store_true",
                    default=True,
                    help="Re-simulate every clear candidate from its root in "
                         "a fresh pool and refuse to bank it unless the clear "
                         "reproduces (DEFAULT ON; measured ~2.7 ms/action, "
                         "i.e. seconds per candidate against a 25-45 min "
                         "solve).")
    ap.add_argument("--no-verify-bank", dest="verify_bank",
                    action="store_false",
                    help="Bank clear candidates unverified (pre-2026-08-08 "
                         "behavior). Receipts written this way record "
                         "replay_verified: false.")
    ap.add_argument("--swim-gx-ceiling", type=int, default=0,
                    help="If >0: in swim rooms ($001D=1), lineages crossing "
                         "this gx are terminated — forces attempt density "
                         "onto the section's floor pipes instead of the "
                         "scroll-buffer wrap (8-4 water separating expt).")
    ap.add_argument("--sect-cap", type=int, default=16,
                    help="Cap on the room-transition counter (`sect`, part "
                         "of the cell key) a lineage can accumulate; room_id() "
                         "stops updating `psig` past this count. Sized for "
                         "SMB1's 8-4 (~5 rooms, room_id churns 0.67/1k steps) "
                         "— default preserves the exact byte-identical "
                         "behavior of the verified 32-level SMB clear. Raise "
                         "this for ROMs where room_id() churns faster (e.g. "
                         "Lost Levels measured 1.15-8.8/1k steps), or the cap "
                         "saturates within the first burst and every "
                         "post-saturation state collapses onto one frozen "
                         "cell-key slot for the rest of the run.")
    ap.add_argument("--gx-bucket", type=int, default=16,
                    help="Cell gx granularity px (micro-search: 8).")
    ap.add_argument("--y-band", type=int, default=32,
                    help="Cell y granularity px (micro-search: 16).")
    ap.add_argument("--sel-mode", choices=("legacy", "count"),
                    default="legacy",
                    help="count = Go-Explore count-based selection prior "
                         "via O(1) rejection sampling (maze-coverage R1).")
    ap.add_argument("--frontier-throttle", type=int, default=0,
                    help="If >0: exclude cells whose bursts came back "
                         "empty this many times in a row from the deep-"
                         "frontier band (DD-RRT boundary suppression, R2).")
    ap.add_argument("--door-weight", type=float, default=0.0,
                    help="If >0: up-weight archive cells that are cut "
                         "vertices (doors) of the rollout transition graph "
                         "by this factor in the count arm (Hopcroft-Tarjan "
                         "sub-goals, R4). Needs --sel-mode count. 5 = "
                         "research-recipe default.")
    ap.add_argument("--door-interval", type=float, default=45.0,
                    help="Seconds between async door (articulation-point) "
                         "recomputations (R4).")
    ap.add_argument("--time-bins", action="store_true",
                    help="Append floor(log2(steps)) to the key prefix "
                         "(Time-Myopic Go-Explore, v6 recipe 4): slow "
                         "structural paths survive domination; wait-gates "
                         "become searchable. Old-format archives resume "
                         "with duplicated (still-valid) cells.")
    ap.add_argument("--kill-key", action="store_true",
                    help="Append capped cumulative entity kills (profile "
                         "solve.entity_slots {lo,hi} flag bytes, observed "
                         "1->0) to the key prefix — prerequisite/kill-count "
                         "gates become a searchable dimension (v6 mech B).")
    ap.add_argument("--resume-archive", type=str, default=None, metavar="DIR",
                    help="Resume from a prior run's flushed out-dir: load "
                         "archive.pkl + traces.pkl + roots.json and continue "
                         "exploring instead of starting from one root cell. "
                         "Iterating on a single wall keeps every hard-won "
                         "frontier cell (e.g. CV block-3's stair funnel).")
    ap.add_argument("--inversion-pin-secs", type=float, default=180.0,
                    metavar="SECS",
                    help="Seconds the deep frontier must sit pinned before "
                         "the heuristic-inversion arm engages (leftward/"
                         "downward sampling inside the self-measured "
                         "saturation window). DEFAULT 180 = the constant "
                         "the verified 32-level SMB clear ran under, so the "
                         "banked receipts stay reproducible. Lower it on "
                         "walls that saturate early; -1 (any negative) "
                         "disables the arm entirely — the sampler keeps the "
                         "forward weights no matter how long the frontier "
                         "stays pinned.")
    # --- orthogonal (vertical) progress arm ------------------------------
    # The inversion arm above is horizontal: it un-prunes left/down when
    # gx saturates. On vertically-structured walls (shafts, climbs) the
    # search needs the mirror image — treat one vertical direction as the
    # progress axis. Every knob below is inert while --ortho is off (the
    # default), so a run that omits them samples exactly as before and
    # every banked SMB receipt's solver_args replays unchanged.
    ap.add_argument("--ortho", choices=("off", "up", "down"), default="off",
                    help="Orthogonal (vertical) progress axis for the "
                         "saturation arm: 'up' or 'down' names the "
                         "direction to treat as forward once horizontal "
                         "progress is pinned. DEFAULT off = no vertical "
                         "arm, sampling is bit-identical to the receipted "
                         "campaign.")
    ap.add_argument("--ortho-pin-secs", type=float, default=120.0,
                    metavar="SECS",
                    help="Seconds the frontier must sit pinned before the "
                         "ortho arm engages (the vertical counterpart of "
                         "--inversion-pin-secs; shorter by default because "
                         "a shaft saturates faster than a run of ground).")
    ap.add_argument("--ortho-bias", type=float, default=0.30, metavar="P",
                    help="Probability that an armed selection restarts a "
                         "worker from the ortho pool (top-of-column cells) "
                         "instead of falling through to the ordinary arms. "
                         "Sampled with a PURE count prior — no score term, "
                         "since the score is what buries a climb. 0 = "
                         "never.")
    ap.add_argument("--ortho-band", type=int, default=1, metavar="N",
                    help="Half-width, in y-bands (--y-band px each), of the "
                         "window around EACH gx column's own ortho extreme "
                         "that enters the pool. 1 = the column's frontier "
                         "band plus its two neighbours.")
    ap.add_argument("--ortho-weight", type=float, default=4.0, metavar="W",
                    help="Multiplier applied to pooled cells inside the "
                         "count arm's rejection sampling while the ortho "
                         "arm is armed (Wmax scales with it, so the prior "
                         "stays exact). 1.0 = no bias. Needs --sel-mode "
                         "count to have any effect.")
    ap.add_argument("--ortho-macro-p", type=float, default=0.0, metavar="P",
                    help="Per-step probability of injecting a profile-"
                         "declared sustained-hold macro in bursts ROOTED at "
                         "an ortho-pool cell (the long consecutive holds a "
                         "climb needs and stochastic sampling almost never "
                         "emits; profile rates are ~0.02). DEFAULT 0 = off, "
                         "so the macro slot behaves exactly as today.")
    args = ap.parse_args()
    global GX_BUCKET, Y_BAND
    GX_BUCKET, Y_BAND = args.gx_bucket, args.y_band

    s = Solver(args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(s, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(s, "stop", True))
    s.seed()
    s.flush()
    s.explore()
    s.flush()
    s.progress_line(time.time() - s.t0)
    print(f"[go_explore_solve] done: {json.dumps(s.archive.stats())}, "
          f"{s.n_solutions} solutions -> {s.out}", flush=True)
    s.pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
