"""Exploration owner for the vanilla-PPO conductor.

`ExplorationController` owns the exploration LIFECYCLE that used to be inlined
in `Trainer._run_vanilla_ppo` (and, for the RND build, `_reinforce_update`):
the RND module build (both the vanilla and GA build sites, unified here), the
resume-pending RND-state application, the count-based frontier bonus, and the
generic Go-Explore archive build / per-step record / periodic save. The logic
lives here; the conductor keeps the thin per-step guards and calls these
methods in place of the inline blocks.

The controller holds a reference back to the owning `Trainer` and reads/writes
the trainer-owned exploration state through `self.trainer` so the rest of the
loop sees exactly the attributes it expects. In particular:

  * `self.trainer._rnd` stays the RND module's home — `PPOUpdater` reads it
    directly during the intrinsic fold + predictor cache (Task 2), so this
    controller sets it exactly as the two inline build sites did and never
    moves the attribute off the trainer.
  * `self.trainer._gx_counts` stays the count table's home — `CheckpointManager`
    persists it and the C3 golden reads it back; only the update+bonus LOGIC
    moves here.
  * `self.trainer._go_explore` stays the archive's home — the leave-alone CGSA
    and iter-boundary warm-start/return blocks reference it as a conductor
    local (`go_explore_archive is self.trainer._go_explore`), so this
    controller sets and reads it in place.

Every read, side effect, and log message is preserved verbatim, just relocated.
This is the third strangler step of
`docs/proposals/trainer_decomposition_plan.md` (Task 3).

NOT owned here (reported as entangled with later tasks — see the module's
extraction note in the Task-3 handoff): the Go-Explore unstick BURST
(arm/tick/harvest + burst record + burst state) mutates the Task-4 Curriculum
save-state ladder, and the iter-boundary / inline archive RETURNS are woven
into the Task-5 warm-start mechanics and the leave-alone CGSA selector. Those
stay in the conductor until Curriculum (Task 4) and RolloutCollector (Task 5)
extract the surrounding seams.
"""
from __future__ import annotations

import logging
import math

from src.models.rnd import RND

log = logging.getLogger(__name__)


class ExplorationController:
    """Exploration lifecycle owner for one training run.

    `trainer` is the owning `Trainer`; the controller reads and writes the
    trainer-owned exploration state (`_rnd`, `_pending_rnd_state`, `_gx_counts`,
    `_gx_count_beta`, `_go_explore`, `pool`, `device`, `checkpoint_dir`) through
    the `self.trainer` ref so the loop sees exactly the state it expects.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    # ------------------------------------------------------------------ RND
    def build_rnd(self, *, log_msg: str) -> None:
        """Lazily build the RND module on `self.trainer._rnd` (once, on first
        use, so it lands on the current device) and apply any resume-pending
        state. No-op when RND is disabled or already built.

        Callers must invoke this again whenever `rnd_intrinsic_coef` may have
        risen off a zero baseline (e.g. after a consolidate/consolidate_level
        schedule step) — the build is gated on the CURRENT coefficient, not
        just on startup, so a run that starts with RND off and ramps it up
        later still gets the module built the first time it's actually
        needed.

        `log_msg` is the caller's exact log format string (the vanilla and GA
        build sites differ only in that prefix — "[vanilla_ppo] RND enabled"
        vs "RND intrinsic motivation enabled" — so it is passed in to preserve
        each site's line verbatim). The four format args are identical at both
        sites. Predictor params train in the same Adam as the policy; the
        encoder dispatches on observation kind (CNN `RND` pixel, `TileRND` MLP
        tile).
        """
        t = self.trainer
        if t.rnd_intrinsic_coef > 0.0 and t._rnd is None:
            if t._is_tile_mode:
                from src.models.tile_rnd import TileRND
                t._rnd = TileRND(
                    feature_dim=t._tile_feature_dim
                ).to(t.device)
            else:
                t._rnd = RND(in_channels=4).to(t.device)
            log.info(
                log_msg,
                "tile_mlp" if t._is_tile_mode else "pixel_cnn",
                t._rnd.num_params,
                t.rnd_intrinsic_coef,
                t.rnd_loss_coef,
            )
            # A build triggered by a mid-run coefficient ramp lands after
            # `_ppo_optimizer` already exists (built once, over `net`
            # params plus whatever `_rnd` held at the time — None here).
            # Without this, the predictor trains via `.backward()` but
            # `optimizer.step()` never touches it: same silent drop
            # `_build_ppo_optimizer` documents for the rollback rebuild,
            # just reached from the other direction (RND arrives late
            # instead of the optimizer being rebuilt without it).
            if getattr(t, "_ppo_optimizer", None) is not None:
                t._ppo_optimizer.add_param_group(
                    {"params": list(t._rnd.predictor.parameters())}
                )
            self.apply_pending_rnd_state()

    def apply_pending_rnd_state(self) -> None:
        """Load RND state stashed from a resumed checkpoint into the
        freshly-built module. RND builds lazily on the first update — after
        resume runs — so the state can't be applied at load time. No-op when
        nothing is pending. Called right after each lazy build."""
        t = self.trainer
        if t._pending_rnd_state is None:
            return
        if t._rnd is None:
            log.warning(
                "[vanilla_ppo] resumed checkpoint staged an RND state but "
                "RND is disabled this run (rnd_intrinsic_coef=%s) — the "
                "restored predictor/normalization stats are being dropped "
                "and will NOT carry into this run's checkpoints",
                t.rnd_intrinsic_coef,
            )
            t._pending_rnd_state = None
            return
        try:
            t._rnd.load_state_dict(t._pending_rnd_state, strict=False)
            log.info(
                "[vanilla_ppo] restored RND state (predictor + normalization) "
                "from checkpoint — novelty memory preserved across resume"
            )
        except Exception as exc:
            log.warning(
                "[vanilla_ppo] RND state restore failed (fresh init): %s", exc
            )
        t._pending_rnd_state = None

    # ---------------------------------------------------------- count bonus
    def count_bonus(self, wl_packed: int, x_pos: int) -> float:
        """Count-based frontier bonus for one (world/level, 64px gx-bucket)
        visit: `beta / sqrt(n)` after incrementing the cumulative visit count.

        Reproduces the inline per-step map verbatim — the `(wl_packed << 9) |
        (x_pos >> 6)` key and the `beta / sqrt(n)` decay — updating the
        trainer-owned `_gx_counts` table in place. Returns 0.0 when the bonus
        is disabled (`_gx_count_beta == 0.0`), matching the old else-branch that
        wrote `bonus_buf[t, i] = 0.0`.
        """
        t = self.trainer
        if t._gx_count_beta > 0.0:
            _gxk = (wl_packed << 9) | (x_pos >> 6)
            _gxn = t._gx_counts.get(_gxk, 0) + 1
            t._gx_counts[_gxk] = _gxn
            return t._gx_count_beta / math.sqrt(_gxn)
        return 0.0

    # ------------------------------------------------------------ Go-Explore
    def build_go_explore(self, ge_cfg: dict, *, fresh_start: bool = False) -> tuple:
        """Build the generic Go-Explore archive (first-return-then-explore).

        Called only when go_explore is enabled (the conductor gates on the
        mutual-exclusion `go_explore_on`, which it still owns for the CGSA
        coupling). Constructs the cell fn, loads any persisted archive, sets
        `self.trainer._go_explore`, and returns `(archive, use_max_x)` — the
        conductor rebuilds its `_ge_score` closure (which captures the rollout
        trackers) from `use_max_x` exactly as before. Every log line is
        preserved verbatim.

        `fresh_start=True` (GUI "Resume" unticked / headless `--no-resume`)
        skips loading any on-disk archive — a from-scratch run must not
        inherit a prior run's visited cells and saved-state blobs, or the
        next iter boundary can teleport the brand-new policy into a state
        (e.g. deep into a later level) it never earned. The on-disk
        `archive.pkl` is never deleted or moved (a paused experiment must
        stay resumable); it is only skipped.
        """
        t = self.trainer
        from src.training.go_explore import (
            GoExploreArchive, ram_bytes_cell, ram_downsample_cell,
            smb_gx_phase_cell,
        )
        _cell_cfg = dict(ge_cfg.get("cell", {}) or {})
        _cell_type = str(_cell_cfg.get("type", "ram_downsample"))
        if _cell_type == "smb_gx_phase":
            cell_fn = smb_gx_phase_cell(
                gx_bucket=int(_cell_cfg.get("gx_bucket", 16)),
                y_bucket=int(_cell_cfg.get("y_bucket", 32)),
            )
        elif _cell_type == "ram_bytes":
            _addrs = [
                int(a) for a in _cell_cfg.get(
                    "addresses", [0x075F, 0x0760, 0x006D]
                )
            ]
            cell_fn = ram_bytes_cell(
                _addrs, bucket=int(_cell_cfg.get("bucket", 1))
            )
        else:
            cell_fn = ram_downsample_cell(
                stride=int(_cell_cfg.get("stride", 64)),
                bucket=int(_cell_cfg.get("bucket", 16)),
            )
        go_explore_archive = GoExploreArchive(
            cell_fn, seed=int(ge_cfg.get("seed", t.seed) or 0)
        )
        t._go_explore = go_explore_archive
        _ge_path = t.checkpoint_dir / "go_explore" / "archive.pkl"
        if fresh_start:
            if _ge_path.exists():
                log.info(
                    "[vanilla_ppo] fresh start requested — ignoring saved "
                    "go_explore archive at %s (file left untouched); "
                    "starting from an empty archive",
                    _ge_path,
                )
        elif _ge_path.exists():
            try:
                go_explore_archive.load(_ge_path)
                log.info(
                    "[vanilla_ppo] go_explore: resumed archive (%d cells)",
                    len(go_explore_archive),
                )
            except Exception as e:
                log.warning("[vanilla_ppo] go_explore reload failed: %s", e)
        _score_kind = str(ge_cfg.get("score", "auto"))
        _is_mario = "mario" in str(t.game_profile.get("name", "")).lower()
        _use_max_x = _score_kind == "max_x" or (
            _score_kind == "auto" and _is_mario
        )
        log.info(
            "[vanilla_ppo] go_explore ENABLED: cell=%s, score=%s "
            "(SMB curriculum OFF)",
            _cell_type, "max_x" if _use_max_x else "ep_return",
        )
        return go_explore_archive, _use_max_x

    def record_cell(self, i: int, ram, score: float, steps: int) -> None:
        """Discretize this state into a cell and remember the best-known way to
        reach it. Bound cost by only paying for `pool.save_worker_state` on a
        NEW cell or a strict improvement (peek the archive first).

        The conductor keeps the `go_explore_archive is not None and not done`
        guard and computes `score` (its `_ge_score(i)`) and `steps`
        (`ep_lengths[i]`) from the rollout trackers, so this method's domination
        rule + pool interaction are the extracted half.
        """
        arc = self.trainer._go_explore
        _gc = arc.cells.get(arc.cell_fn(ram))
        _dom = (
            _gc is None
            or score > _gc.best_score + 1e-9
            or (abs(score - _gc.best_score) <= 1e-9
                and steps < _gc.best_steps)
        )
        if _dom:
            try:
                _blob = self.trainer.pool.save_worker_state(i)
            except Exception:
                _blob = None
            if _blob is not None:
                arc.record(ram, _blob, score, steps)
        else:
            arc.record(ram, None, score, steps)

    def save_go_explore(self) -> None:
        """Persist the Go-Explore archive. The conductor keeps the
        `it % save_every == 0` cadence guard; this is the save I/O half."""
        try:
            self.trainer._go_explore.save(
                self.trainer.checkpoint_dir / "go_explore" / "archive.pkl"
            )
        except Exception as exc:
            log.warning(
                "[vanilla_ppo] go_explore archive save failed: %s", exc
            )
