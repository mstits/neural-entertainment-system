"""C4 curriculum-glue golden — the pre-extraction safety net for Task 4.

This is Task 0 / test C4 of `docs/proposals/trainer_decomposition_plan.md`:
the behavior anchor for the *trainer-side glue* around the SMB save-state
curriculum, so the eventual `Curriculum` extraction (plan §4 Task 4) can be
proven behavior-preserving. It pins the plumbing the trainer keeps; it does
NOT re-test the DECISIONS, which already live in
`src/training/oneshot_curriculum.py` / `src/training/smb_substage_ladder.py`
and are covered by `test_oneshot_curriculum` / `test_smb_substage_ladder`.

What "glue" means here (each pinned below):

  1. Curriculum disk I/O — `_load_smb_curriculum_from_disk` (a real callable
     seam): the `stage_NN.state` / `stage_NN.meta.json` file-set contract,
     the `[None]`/`[0]` stage-0 seeding, the contiguous-stage break, the
     anchor-from-sidecar parse, the fresh-start ignore + inherit opt-in, and
     the non-SMB skip. Round-tripped against a fixture GENERATED FROM THE
     REAL METHOD (regen with CHAR_GOLDEN_REGEN=1).
  2. The scalar capture gate predicate (`wl_packed > cur_anchor` + lives>=1
     + first-detection-wins + not-consolidating), mirrored and anchored to
     the exact trainer source so a reworded gate trips a test.
  3. The `wl_packed` RAM packing + lives byte layout.
  4. The `should_advance` rolling-mean gate.
  5. The scalar warm-start `env_stage` distribution (60% current-stage,
     round-robin the rest over earlier stages).
  6. The die-respawn `prev_completion_total` re-arm (plan §4.5 / §2.2): the
     off-by-one that, if dropped, under-reports success ~5-6x at stage>=1.
  7. Integration guard (ROM-gated): a short cold tile+CPU run advances the
     curriculum ZERO times and writes ZERO stage files — no spurious
     capture/advance out of the 1-1 start state.

Determinism basis (per the plan §3): the disk-I/O and predicate tests are
pure / seedless and ROM-free. The one integration guard runs the tile path
on `device=cpu` with a fixed seed. No pixel/MPS path is exercised.

Explicitly NOT covered (documented honestly so nobody over-trusts this net):

  * Driving the full loop to an ACTUAL mid-rollout capture or a real
    stage advance. That needs Mario to physically cross a level boundary
    (area byte > anchor) within a rollout, which a tiny deterministic run
    from the 1-1 start state cannot reach in << 10s. The capture/advance
    WRITE path is therefore pinned by (a) mirroring its predicate, (b)
    anchoring its exact file-naming + sidecar-schema source text, and (c)
    the read side of the same contract via the disk-load round-trip — not
    by observing a live write. This is the sanctioned fallback in the C4
    spec ("pin the capture-gate predicate and the advance-detection logic
    at their seams and document the rest as gaps").
  * The legacy no-sidecar anchor-peek branch of `_load_smb_curriculum_from_disk`
    (loads the blob into pool worker 0 to read `ram[$0760]`): needs a live
    pool + a real save state, out of scope for a ROM-free unit test.
  * The ladder, consolidation, PLR, Go-Explore, and backward-curriculum
    warm-start partitions: their assignment math is `oneshot_curriculum`'s
    and is unit-tested there; only the SCALAR mode's trainer-side
    distribution is mirrored here.
  * The cold-eval probe, forgetting/consolidate triggers, and coefficient
    schedules (C0 covers the loop's numeric trajectory end-to-end).
"""

from __future__ import annotations

import json
import os
import queue as _queue
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_tiles_vanilla.yaml"
_FIXTURE = ROOT / "tests" / "fixtures" / "char_curriculum_glue_diskload.json"

_TRAINER_SRC = (ROOT / "src" / "training" / "trainer.py").read_text()


def _vppo_body() -> str:
    """The text of `_run_vanilla_ppo` (where all the inline glue lives), so a
    source anchor cannot match an identical string in an unrelated method."""
    start = _TRAINER_SRC.find("def _run_vanilla_ppo")
    end = _TRAINER_SRC.find("def _emit_metrics", start)
    assert start >= 0 and end > start, "could not bound _run_vanilla_ppo body"
    return _TRAINER_SRC[start:end]


# ---------------------------------------------------------------------------
# A bare Trainer shell exposing just what `_load_smb_curriculum_from_disk`
# touches — no pool / ROM / emulator (the meta-sidecar path never reads the
# pool).
# ---------------------------------------------------------------------------
def _bare_curriculum_trainer(
    tmp: Path, *, active: bool = True, inherit_on_fresh: bool = False
):
    from src.training.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.checkpoint_dir = Path(tmp)
    t._smb_curriculum_active = active
    t._inherit_curriculum_on_fresh = inherit_on_fresh
    t.pool = None  # only touched by the legacy no-sidecar branch (not exercised)
    return t


# Canonical on-disk curriculum: (stage index, anchor area-byte, state blob).
# Anchors mimic real SMB packed world<<4|area values (1-2=2, 1-3=5, 1-4=6).
_DISK_STAGES: tuple[tuple[int, int, bytes], ...] = (
    (1, 2, b"CURRICULUM-STAGE-01-STATE-BLOB"),
    (2, 5, b"CURRICULUM-STAGE-02-STATE-BLOB"),
    (3, 6, b"CURRICULUM-STAGE-03-STATE-BLOB"),
)


def _write_disk_curriculum(
    checkpoint_dir: Path, stages=_DISK_STAGES
) -> None:
    """Lay down `stage_NN.state` + `stage_NN.meta.json` exactly as the
    trainer's advance block writes them (per trainer.py ~7948-7964)."""
    d = checkpoint_dir / "smb_curriculum"
    d.mkdir(parents=True, exist_ok=True)
    for n, anchor, blob in stages:
        (d / f"stage_{n:02d}.state").write_bytes(blob)
        (d / f"stage_{n:02d}.meta.json").write_text(
            json.dumps({"anchor": int(anchor)})
        )


def _load_result_summary(states, anchors, stage) -> dict:
    return {
        "anchors": [int(a) for a in anchors],
        "stage": int(stage),
        "n_states": len(states),
        "states_hex": [s.hex() if s is not None else None for s in states],
    }


# ---------------------------------------------------------------------------
# Group 1 — curriculum disk I/O (the real callable seam) + golden fixture.
# ---------------------------------------------------------------------------
def test_disk_load_roundtrips_stages_anchors_and_states() -> None:
    """Pin: `_load_smb_curriculum_from_disk` recovers `(states, anchors,
    stage)` from a `stage_NN.state`/`.meta.json` set exactly — stage-0 is
    seeded `None`/anchor-0, each contiguous stage appends its blob + sidecar
    anchor and bumps `stage`. Compared to a fixture GENERATED FROM THE REAL
    METHOD so a change in the load glue (dropped None-seed, mis-parsed
    anchor, off-by-one stage) trips this test."""
    with tempfile.TemporaryDirectory(prefix="c4_diskload_") as tmp:
        _write_disk_curriculum(Path(tmp))
        t = _bare_curriculum_trainer(Path(tmp))
        states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=False
        )
        got = _load_result_summary(states, anchors, stage)

    if os.environ.get("CHAR_GOLDEN_REGEN") == "1":
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(got, indent=2) + "\n")
        pytest.skip(f"regenerated golden fixture -> {_FIXTURE} (review the diff)")

    assert _FIXTURE.exists(), (
        f"golden fixture missing: {_FIXTURE}. Generate it with "
        f"CHAR_GOLDEN_REGEN=1 .venv/bin/pytest {Path(__file__).name}"
    )
    want = json.loads(_FIXTURE.read_text())
    assert got == want, (
        "disk-load glue changed vs golden. If intended, regenerate "
        "(CHAR_GOLDEN_REGEN=1) and review the diff."
    )

    # Cross-check the fixture actually encodes the file-set contract we wrote,
    # so the golden can't silently rot into a tautology.
    assert want["anchors"] == [0, 2, 5, 6]
    assert want["stage"] == 3
    assert want["n_states"] == 4
    assert want["states_hex"][0] is None, "stage 0 must be the cold-boot None seed"
    assert want["states_hex"][1] == _DISK_STAGES[0][2].hex()


def test_disk_load_stops_at_first_missing_stage() -> None:
    """Pin: the loader consumes CONTIGUOUS stages only — a gap ends the scan.
    Writing stage_01 and stage_03 (no stage_02) must load only stage 1."""
    with tempfile.TemporaryDirectory(prefix="c4_gap_") as tmp:
        _write_disk_curriculum(
            Path(tmp),
            stages=(_DISK_STAGES[0], _DISK_STAGES[2]),  # 1 and 3, gap at 2
        )
        t = _bare_curriculum_trainer(Path(tmp))
        states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=False
        )
    assert stage == 1, "a gap at stage_02 must stop the scan at stage 1"
    assert anchors == [0, 2]
    assert len(states) == 2
    assert states[0] is None and states[1] == _DISK_STAGES[0][2]


def test_disk_load_empty_dir_is_cold_boot_stage_zero() -> None:
    """Pin: with no saved stages the loader returns the single cold-boot
    stage (`[None]` / `[0]` / stage 0)."""
    with tempfile.TemporaryDirectory(prefix="c4_empty_") as tmp:
        t = _bare_curriculum_trainer(Path(tmp))
        states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=False
        )
    assert (states, anchors, stage) == ([None], [0], 0)


def test_fresh_start_ignores_saved_stages_but_leaves_files() -> None:
    """Pin: `fresh_start=True` (without the inherit opt-in) returns stage 0
    and IGNORES saved stages — a fresh run trains from 1-1 rather than
    silently inheriting a prior run's mid-game curriculum — yet the
    `stage_NN.state` files stay on disk (a paused run must remain resumable)."""
    with tempfile.TemporaryDirectory(prefix="c4_fresh_") as tmp:
        _write_disk_curriculum(Path(tmp))
        t = _bare_curriculum_trainer(Path(tmp), inherit_on_fresh=False)
        states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=True
        )
        assert (states, anchors, stage) == ([None], [0], 0), (
            "a fresh start must not inherit the saved curriculum"
        )
        # Files untouched.
        d = Path(tmp) / "smb_curriculum"
        remaining = sorted(p.name for p in d.glob("stage_*.state"))
        assert remaining == ["stage_01.state", "stage_02.state", "stage_03.state"], (
            "fresh start must leave the saved stage files on disk"
        )


def test_fresh_start_inherits_when_opted_in() -> None:
    """Pin: `reinforce.inherit_curriculum_on_fresh: true`
    (`_inherit_curriculum_on_fresh=True`) restores the historical
    inherit-on-fresh behavior — a fresh run then loads the saved stages."""
    with tempfile.TemporaryDirectory(prefix="c4_inherit_") as tmp:
        _write_disk_curriculum(Path(tmp))
        t = _bare_curriculum_trainer(Path(tmp), inherit_on_fresh=True)
        _states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=True
        )
    assert stage == 3 and anchors == [0, 2, 5, 6], (
        "inherit_on_fresh must load saved stages even on a fresh start"
    )


def test_non_smb_game_skips_curriculum_entirely() -> None:
    """Pin: a non-Mario game (`_smb_curriculum_active=False`) NEVER scans the
    stage files — plain single-stage vanilla PPO — even if stage files exist
    on disk (the `range(0)` skip in the loader)."""
    with tempfile.TemporaryDirectory(prefix="c4_nonsmb_") as tmp:
        _write_disk_curriculum(Path(tmp))  # files present but must be ignored
        t = _bare_curriculum_trainer(Path(tmp), active=False)
        states, anchors, stage = t._load_smb_curriculum_from_disk(
            fresh_start=False
        )
    assert (states, anchors, stage) == ([None], [0], 0), (
        "a non-SMB game must skip the area-byte curriculum entirely"
    )


# ---------------------------------------------------------------------------
# Group 2 — the scalar capture gate predicate (mirror + source anchor).
# ---------------------------------------------------------------------------
def _scalar_capture_fires(
    *, active: bool, consolidate_on: bool, pending_set: bool,
    wl_packed: int, cur_anchor: int, lives: int,
) -> bool:
    """Mirror of the scalar SMB capture gate, trainer.py ~6744-6748:

        elif (self._smb_curriculum_active
                and not consolidate_on
                and smb_pending_capture is None
                and wl_packed > cur_anchor
                and int(ram[0x075A]) >= 1):
    """
    return (
        active
        and not consolidate_on
        and not pending_set
        and wl_packed > cur_anchor
        and lives >= 1
    )


def test_scalar_capture_gate_predicate_bites_on_every_condition() -> None:
    """Pin the capture gate's truth table: it fires iff the curriculum is
    active, not consolidating, no capture is already pending, the packed
    world-level byte is STRICTLY past the current anchor, AND lives>=1. Each
    condition individually vetoes the capture."""
    base = dict(
        active=True, consolidate_on=False, pending_set=False,
        wl_packed=2, cur_anchor=0, lives=1,
    )
    assert _scalar_capture_fires(**base) is True, "base case must capture"

    # Strictly-greater gate: equal byte does NOT capture (the off-by-one that
    # stalled the curriculum lived exactly here).
    assert _scalar_capture_fires(**{**base, "wl_packed": 0}) is False
    assert _scalar_capture_fires(**{**base, "cur_anchor": 2}) is False  # equal
    assert _scalar_capture_fires(**{**base, "wl_packed": 3, "cur_anchor": 2}) is True

    # lives >= 1 required (a last-life crossing is skipped).
    assert _scalar_capture_fires(**{**base, "lives": 0}) is False

    # First-detection-wins: an already-pending capture blocks a second.
    assert _scalar_capture_fires(**{**base, "pending_set": True}) is False

    # Consolidation freezes capture; inactive curriculum never captures.
    assert _scalar_capture_fires(**{**base, "consolidate_on": True}) is False
    assert _scalar_capture_fires(**{**base, "active": False}) is False


def test_scalar_capture_gate_source_anchor() -> None:
    """Anchor the mirror to the real gate so a reworded predicate trips."""
    body = _vppo_body()
    for needle in (
        "and not consolidate_on",
        "and smb_pending_capture is None",
        "and wl_packed > cur_anchor",
        "and int(ram[0x075A]) >= 1",
    ):
        assert needle in body, (
            f"scalar capture-gate clause {needle!r} changed — update the "
            f"mirror in this characterization test deliberately."
        )


# ---------------------------------------------------------------------------
# Group 3 — wl_packed RAM packing + lives byte layout.
# ---------------------------------------------------------------------------
def _pack_wl(world_byte: int, area_byte: int) -> int:
    """Mirror of trainer.py ~6632:
    `(int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)`."""
    return (int(world_byte) << 4) | (int(area_byte) & 0x0F)


def test_wl_packed_arithmetic() -> None:
    """Pin the world/area packing that both the capture gate and the advance
    metric read off RAM."""
    assert _pack_wl(0, 0) == 0            # 1-1 start
    assert _pack_wl(0, 2) == 2            # 1-2 underground main
    assert _pack_wl(0, 5) == 5           # 1-3
    assert _pack_wl(1, 0) == 16          # 2-1 (world byte shifts by 4)
    # The area byte is masked to its low nibble.
    assert _pack_wl(0, 0x12) == 2
    assert _pack_wl(1, 0x1F) == 0x1F     # (1<<4)|0xF


def test_wl_packed_and_lives_ram_layout_source_anchor() -> None:
    """Anchor the packed-byte formula and lives byte ($075A) to source."""
    body = _vppo_body()
    assert (
        "wl_packed = (int(ram[0x075F]) << 4) | (int(ram[0x0760]) & 0x0F)"
        in body
    ), "the world/area packing formula changed"
    assert "int(ram[0x075A]) >= 1" in body, "the lives byte ($075A) changed"


# ---------------------------------------------------------------------------
# Group 4 — the should_advance rolling-mean gate (mirror + source anchor).
# ---------------------------------------------------------------------------
def _should_advance(
    *, pending_set: bool, history_len: int, window: int,
    rolling_mean: float, pct: float, consolidating: bool,
    consolidate_on: bool, ladder_on: bool, stage: int, ladder_size: int,
) -> bool:
    """Mirror of the advance gate, trainer.py ~7877-7887:

        should_advance = (
            smb_pending_capture is not None
            and len(smb_pastfrac_history) >= SMB_ADVANCE_WINDOW
            and rolling_mean >= SMB_ADVANCE_PCT
            and not consolidating
            and not consolidate_on
            and (not ladder_on or smb_curriculum_stage < LADDER_SIZE - 1)
        )
    """
    return (
        pending_set
        and history_len >= window
        and rolling_mean >= pct
        and not consolidating
        and not consolidate_on
        and (not ladder_on or stage < ladder_size - 1)
    )


def test_should_advance_predicate_bites_on_every_condition() -> None:
    """Pin the advance gate: needs a pending capture, a FULL rolling window,
    a rolling mean at/above the threshold, and neither consolidation flag —
    and, in ladder mode, room below the final rung."""
    base = dict(
        pending_set=True, history_len=5, window=5, rolling_mean=0.5, pct=0.5,
        consolidating=False, consolidate_on=False, ladder_on=False,
        stage=0, ladder_size=48,
    )
    assert _should_advance(**base) is True, "base case must advance"

    assert _should_advance(**{**base, "pending_set": False}) is False
    assert _should_advance(**{**base, "history_len": 4}) is False   # window unfilled
    assert _should_advance(**{**base, "rolling_mean": 0.49}) is False  # below pct
    assert _should_advance(**{**base, "consolidating": True}) is False
    assert _should_advance(**{**base, "consolidate_on": True}) is False

    # Rolling-mean is a >= gate; exactly at threshold advances.
    assert _should_advance(**{**base, "rolling_mean": 0.5, "pct": 0.5}) is True

    # Ladder mode: the final rung can never advance further.
    ladder = {**base, "ladder_on": True, "ladder_size": 4}
    assert _should_advance(**{**ladder, "stage": 2}) is True   # 2 < 3
    assert _should_advance(**{**ladder, "stage": 3}) is False  # final rung


def test_advance_gate_and_constants_source_anchor() -> None:
    """Anchor the advance gate clauses AND the default threshold/window/
    current-stage-fraction constants to source."""
    body = _vppo_body()
    for needle in (
        "smb_pending_capture is not None",
        "and len(smb_pastfrac_history) >= SMB_ADVANCE_WINDOW",
        "and rolling_mean >= SMB_ADVANCE_PCT",
        "and not consolidating",
        "and (not ladder_on or smb_curriculum_stage < LADDER_SIZE - 1)",
    ):
        assert needle in body, f"advance-gate clause {needle!r} changed"
    # Default constants the gate is measured against.
    assert "SMB_ADVANCE_PCT = 0.50" in body, "advance threshold default changed"
    assert "SMB_ADVANCE_WINDOW = 5" in body, "advance window default changed"
    assert "SMB_CURRENT_STAGE_FRAC = 0.6" in body, "current-stage frac changed"


def test_advance_writes_stage_files_source_anchor() -> None:
    """Anchor the advance WRITE path's file-set contract — the exact
    `stage_NN.state` / `stage_NN.meta.json` names + the `{"anchor": N}`
    sidecar schema + the atomic tmp+replace — so the write side of the
    contract the disk-load round-trip verifies cannot drift."""
    body = _vppo_body()
    assert 'f"stage_{new_stage:02d}.state"' in body, "advance .state name changed"
    assert 'f"stage_{new_stage:02d}.meta.json"' in body, "advance sidecar name changed"
    assert 'json.dumps({"anchor": int(new_anchor)})' in body, "sidecar schema changed"
    assert "_tmp_stage.replace(stage_path)" in body, "atomic state write changed"


# ---------------------------------------------------------------------------
# Group 5 — the scalar warm-start env_stage distribution (mirror + anchor).
# ---------------------------------------------------------------------------
def _scalar_env_stage(num_envs: int, stage: int, frac: float) -> np.ndarray:
    """Mirror of the scalar mixed-stage warm-start, trainer.py ~9209-9216:

        n_current = max(1, int(round(num_envs * SMB_CURRENT_STAGE_FRAC)))
        env_stage[:n_current] = smb_curriculum_stage
        for j, i in enumerate(range(n_current, num_envs)):
            env_stage[i] = j % smb_curriculum_stage
    """
    env_stage = np.zeros(num_envs, dtype=int)
    n_current = max(1, int(round(num_envs * frac)))
    env_stage[:n_current] = stage
    for j, i in enumerate(range(n_current, num_envs)):
        env_stage[i] = j % stage
    return env_stage


def test_scalar_warm_start_env_stage_distribution() -> None:
    """Pin: ~60% of envs sit at the current (hardest) stage; the rest
    round-robin over earlier stages [0..stage-1] for even coverage."""
    # 8 envs, stage 3: round(8*0.6)=round(4.8)=5 current, 3 spread -> 0,1,2.
    dist8 = _scalar_env_stage(8, 3, 0.6)
    assert dist8.tolist() == [3, 3, 3, 3, 3, 0, 1, 2]
    counts8 = {int(k): int(v) for k, v in zip(*np.unique(dist8, return_counts=True))}
    assert counts8 == {0: 1, 1: 1, 2: 1, 3: 5}

    # 2 envs, stage 1: round(1.2)=1 current, 1 spread -> 0 (only earlier stage).
    dist2 = _scalar_env_stage(2, 1, 0.6)
    assert dist2.tolist() == [1, 0]

    # The current-stage count is always >= 1 even when the fraction rounds to 0.
    dist1 = _scalar_env_stage(1, 2, 0.6)  # round(0.6)=1 -> [2]
    assert dist1.tolist() == [2]

    # Every env lands on a valid stage index (0..stage).
    d = _scalar_env_stage(30, 4, 0.6)
    assert d.min() >= 0 and d.max() <= 4


def test_scalar_warm_start_distribution_source_anchor() -> None:
    body = _vppo_body()
    assert "int(round(num_envs * SMB_CURRENT_STAGE_FRAC))" in body
    assert "env_stage[:n_current] = smb_curriculum_stage" in body
    assert "env_stage[i] = j % smb_curriculum_stage" in body


# ---------------------------------------------------------------------------
# Group 6 — the die-respawn prev_completion_total re-arm (plan §4.5 / §2.2).
# ---------------------------------------------------------------------------
def _detect_clears(sequence, *, rearm_on_death: bool) -> int:
    """Faithful mirror of the per-step clear-detector + its die-respawn re-arm.

    Per-step clear detection (trainer.py ~6882-6901), run every active step:
        cur_comp = reward_fns[i].breakdown["completion"]  # cumulative
        if cur_comp > prev_completion_total[i] + 1e-6:
            n_clears_this_iter += 1
        prev_completion_total[i] = cur_comp        # <- also self-heals prev

    On a within-rollout death auto-reset with a warm-start state, the reward
    fn's cumulative `completion` is zeroed by `reward_fns[i].reset()`, so the
    trainer ALSO re-arms the detector on that same step (trainer.py ~6992):
    `prev_completion_total[i] = 0.0`.

    The re-arm BITES in the curriculum stage>=1 case the code comment calls
    out: the reload drops Mario deep, and the reward fn's first scored step
    already carries completion credit (cur jumps straight to a bonus with NO
    intervening zero step to self-heal `prev` via the assignment above). Left
    stale-high, that clear fails the strict `> prev + 1e-6` diff and is
    dropped — the "under-reports success ~5-6x at stage>=1" bug. This mirror
    models exactly that ordering; in the shallow stage-0 case an intervening
    zero step self-heals `prev` and the re-arm is belt-and-suspenders.

    `sequence` is a list of ("comp", cumulative_value) steps and ("death",)
    resets.
    """
    prev = 0.0
    clears = 0
    for step in sequence:
        if step[0] == "comp":
            cur = float(step[1])
            if cur > prev + 1e-6:
                clears += 1
            prev = cur  # trainer.py ~6901: every step retracks prev
        elif step[0] == "death":
            # reward_fns[i].reset() zeroed the cumulative completion.
            if rearm_on_death:
                prev = 0.0  # trainer.py ~6992: the die-respawn re-arm
            # else: prev stays stale-high — the bug the re-arm fixes.
    return clears


def test_clear_detector_uses_strict_positive_diff() -> None:
    """Pin the clear-detector's strict `cur > prev + 1e-6` rule: only a
    positive jump in cumulative completion counts; a flat or shrinking value
    does not. (Re-arm irrelevant here — no death in the sequence.)"""
    # Two real clears in one episode (0 -> 1 -> 2 cumulative); the repeated
    # values between them do not double-count.
    seq = [
        ("comp", 0.0), ("comp", 1.0), ("comp", 1.0),  # one clear (0->1)
        ("comp", 2.0), ("comp", 2.0),                  # one more (1->2)
    ]
    assert _detect_clears(seq, rearm_on_death=True) == 2
    # A sub-epsilon wobble is NOT a clear.
    assert _detect_clears(
        [("comp", 0.0), ("comp", 5e-7)], rearm_on_death=True
    ) == 0


def test_die_respawn_rearm_reenables_clear_detection() -> None:
    """Pin: the die-respawn re-arm lets a SECOND episode's clear be counted
    after an in-rollout respawn into a deep (stage>=1) warm-start whose first
    scored step already carries completion. Two clears at completion=1.0,
    separated by a death auto-reset, with NO intervening zero step:

      * with the re-arm -> BOTH clears counted (correct);
      * without it -> only the first (the ~5-6x under-report bug).
    """
    seq = [
        ("comp", 1.0),   # episode 1 clears (0 -> 1.0); prev retracks to 1.0
        ("death",),       # auto-reset: cumulative completion zeroed
        ("comp", 1.0),   # deep warm-start's first scored step already carries it
    ]
    assert _detect_clears(seq, rearm_on_death=True) == 2, (
        "the re-arm must let the post-respawn clear be detected"
    )
    assert _detect_clears(seq, rearm_on_death=False) == 1, (
        "without the re-arm the stale prev masks the second clear "
        "(the under-report bug this pin guards against)"
    )


def test_die_respawn_rearm_source_anchor() -> None:
    """Anchor the re-arm sites: the death auto-reset re-arm (>=2 inline
    occurrences), the iter-boundary sync, and the clear-diff itself."""
    body = _vppo_body()
    assert body.count("prev_completion_total[i] = 0.0") >= 2, (
        "the die-respawn / stage-0-reseed re-arm(s) changed"
    )
    assert "prev_completion_total[:] = 0.0" in body, (
        "the iter-boundary clear-detector sync changed"
    )
    assert "cur_comp > prev_completion_total[i] + 1e-6" in body, (
        "the clear-detection diff changed"
    )


# ---------------------------------------------------------------------------
# Group 7 — integration guard (ROM-gated): a short cold run must not
# spuriously capture or advance the curriculum.
# ---------------------------------------------------------------------------
def _seed_all(s: int = 1234) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@pytest.mark.skipif(
    not _SMB_ROM.exists() or not _PROFILE.exists(),
    reason="SMB ROM / mario_tiles_vanilla profile not present.",
)
def test_short_cold_run_makes_no_spurious_capture_or_advance() -> None:
    """Run a tiny real tile+CPU vanilla_ppo loop from the 1-1 start state and
    pin the glue's stability: Mario cannot cross a level boundary in this many
    steps, so the scalar curriculum must stay at stage 0 for EVERY emitted
    iter and write ZERO `stage_NN.state` files. This bites if the capture gate
    or advance logic regresses into firing from the cold-boot state (byte 0)."""
    _seed_all(1234)
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 32
    profile["reinforce"]["ppo_minibatch_size"] = 16
    profile["reinforce"]["device"] = "cpu"

    from src.training.trainer import Trainer

    metrics_q: _queue.Queue = _queue.Queue()
    with tempfile.TemporaryDirectory(prefix="c4_intgr_") as tmp:
        trainer = Trainer(
            rom_path=str(_SMB_ROM),
            game_profile=profile,
            num_instances=2,
            population_size=2,
            checkpoint_dir=tmp,
            start_state_path=profile.get("start_state_path"),
            env_spec="nes_core",
            max_episode_steps=200,
            metrics_queue=metrics_q,
            device_override="cpu",
            seed=1234,
        )
        # Confirm the profile really drives the scalar SMB curriculum glue.
        assert trainer._smb_curriculum_active is True
        trainer.run(num_generations=2, resume_from=None, fresh_start=True)

        curr_dir = Path(tmp) / "smb_curriculum"
        stage_files = (
            sorted(p.name for p in curr_dir.glob("stage_*.state"))
            if curr_dir.exists() else []
        )

    assert stage_files == [], (
        f"a short cold run captured/advanced the curriculum and wrote "
        f"{stage_files} — the capture gate fired spuriously from the 1-1 start."
    )

    rows = []
    while not metrics_q.empty():
        rows.append(metrics_q.get_nowait())
    ppo_rows = [m for m in rows if "ppo_loss" in m]
    assert ppo_rows, "no ppo rows emitted — the loop never finished an iteration"
    for r in ppo_rows:
        assert "vanilla_ppo_curriculum_stage" in r, (
            "the SMB-active loop must emit vanilla_ppo_curriculum_stage"
        )
        assert int(r["vanilla_ppo_curriculum_stage"]) == 0, (
            "the curriculum must not advance past stage 0 in a short cold run"
        )
