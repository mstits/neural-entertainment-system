"""Checkpoint owner for the vanilla-PPO conductor.

The `CheckpointManager` owns the three checkpoint responsibilities that used
to be inlined in `_run_vanilla_ppo`: the resume scan/load, the reproducibility
run manifest, and the periodic iter-checkpoint save (with its numeric-collapse
poison guard and atomic write). The logic lives here; the trainer keeps only a
thin `_maybe_resume_vanilla_ppo` shim for pre-existing direct callers.

The manager holds a reference back to the owning `Trainer` so that the resume
scan can stage the same `_pending_*` / `_vppo_resumed_from_iter` / `_gx_counts`
attributes the inline code did, and the save can read the live `_rnd` module,
`_gx_counts` table, and `checkpoint_dir` — every attribute assignment and side
effect is preserved verbatim, just relocated. This is the first strangler step
of `docs/proposals/trainer_decomposition_plan.md` (Task 1).
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


class CheckpointManager:
    """Resume + save owner for one vanilla-PPO run.

    `trainer` is the owning `Trainer`; the manager reads and writes the
    trainer-owned checkpoint state (`_pending_*`, `_gx_counts`, `_rnd`,
    `checkpoint_dir`, `device`) through the `self.trainer` ref so the loop sees
    exactly the state it expects.
    """

    def __init__(self, trainer, *, checkpoint_dir, device=None, game_name=None):
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.game_name = game_name

    def resume(self, net, optimizer, *, fresh_start: bool = False) -> int:
        """Load the newest vanilla_ppo_iter_*.pt on top of the freshly
        built net + optimizer and return the absolute iteration offset to
        continue checkpoint numbering from (0 = nothing loaded).

        The GA-path resume at trainer.run() loads `gen_*.pt` and is a
        no-op for our `vanilla_ppo_iter_*.pt` format, so this is the only
        resume a vanilla_ppo run gets. `fresh_start` is the user's
        explicit from-scratch request — the GUI "Resume" checkbox
        unticked, or headless `--no-resume` — and skips the scan so the
        run starts from random weights even when a checkpoint is sitting
        in the dir. Default False preserves the historical auto-resume so
        headless scripts that rely on it are unaffected.

        Sets `trainer._vppo_resumed_from_iter` to the resumed iter (or None
        when nothing was loaded) so the caller can surface it to the GUI.
        """
        self.trainer._vppo_resumed_from_iter = None
        if fresh_start:
            # MOVE prior checkpoints aside, don't just ignore them. A
            # fresh run numbers its own checkpoints from 0, below any
            # abandoned longer run's; if left in place, a supervisor
            # crash-relaunch (which runs fresh_start=False) would resume
            # the OLD run by highest-number and silently discard the
            # fresh run — the "fresh run resumed at stage N" confusion,
            # now automatable by the self-healing supervisor. Relocating
            # them makes the fresh run's numbering unambiguous and
            # leaves the old checkpoints recoverable under prevrun_*/.
            existing = list(
                self.trainer.checkpoint_dir.glob("vanilla_ppo_iter_*.pt")
            )
            if existing:
                from datetime import datetime as _dt
                dest = self.trainer.checkpoint_dir / f"prevrun_{_dt.now():%Y%m%d_%H%M%S}"
                try:
                    dest.mkdir(parents=True, exist_ok=True)
                    for p in existing:
                        p.rename(dest / p.name)
                    log.info(
                        "[vanilla_ppo] fresh start — moved %d prior "
                        "checkpoint(s) to %s so a relaunch can't resume "
                        "them", len(existing), dest,
                    )
                except OSError as exc:
                    log.warning(
                        "[vanilla_ppo] fresh start — could not relocate "
                        "%d prior checkpoint(s) (%s); they are ignored "
                        "this launch but a crash-relaunch may resume "
                        "them", len(existing), exc,
                    )
            return 0

        iter_offset = 0
        # Tear-tolerant resume: walk candidates NEWEST->OLDEST and load
        # the first that opens. A single torn newest checkpoint (a death
        # mid-save, now made rare by the atomic write above but still
        # possible on older runs) must NOT trigger a silent from-scratch
        # restart while ~50 good checkpoints sit on disk — that converts
        # a long run into random-weights training with only a log line,
        # and under the supervisor it happens unattended. A corrupt file
        # is renamed aside so it stops winning the sort, and starting
        # fresh is an ERROR (not a shrug) reached only when NOTHING loads.
        candidates = sorted(
            self.trainer.checkpoint_dir.glob("vanilla_ppo_iter_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            reverse=True,
        )
        state = None
        latest = None
        for cand in candidates:
            try:
                state = torch.load(str(cand), map_location=self.trainer.device)
                latest = cand
                break
            except Exception as exc:
                corrupt = cand.with_suffix(".pt.corrupt")
                try:
                    cand.rename(corrupt)
                except OSError:
                    pass
                log.error(
                    "[vanilla_ppo] checkpoint %s failed to load (%s) — "
                    "quarantined as %s, trying next-older",
                    cand.name, exc, corrupt.name,
                )
        if state is not None and isinstance(state, dict) \
                and "net_state_dict" in state:
            net.load_state_dict(state["net_state_dict"], strict=False)
            try:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as exc:
                log.warning(
                    "[vanilla_ppo] optimizer state load failed "
                    "(continuing with fresh Adam state): %s", exc,
                )
            iter_offset = int(state.get("iter", 0) or 0)
            if "rnd_state_dict" in state:
                self.trainer._pending_rnd_state = state["rnd_state_dict"]
            if "prmdp_adv_net_state_dict" in state:
                self.trainer._pending_prmdp_adv = (
                    state["prmdp_adv_net_state_dict"],
                    state.get("prmdp_adv_optimizer_state_dict"),
                )
            if "anticollapse" in state:
                self.trainer._pending_anticollapse = state["anticollapse"]
            if "backward_curriculum" in state:
                self.trainer._pending_backward_curriculum = state["backward_curriculum"]
            if "curriculum_resume" in state:
                self.trainer._pending_curriculum_resume = state["curriculum_resume"]
            if "gx_counts" in state:
                self.trainer._gx_counts = {
                    int(k): int(v) for k, v in state["gx_counts"].items()
                }
            self.trainer._vppo_resumed_from_iter = iter_offset
            log.info(
                "[vanilla_ppo] RESUMED net + optimizer from %s "
                "(saved at iter %d; continuing checkpoint numbering)",
                latest, iter_offset,
            )
        elif candidates:
            log.error(
                "[vanilla_ppo] NO loadable checkpoint among %d candidate(s)"
                " — starting from random weights", len(candidates),
            )
        return iter_offset

    def write_manifest(
        self, *, game, rom_path, start_state_path, seed, profile,
        num_envs, frame_skip,
    ) -> None:
        """Write the reproducibility run manifest (ROM + start state + seed +
        git commit + pinned hyperparameters) to the checkpoint dir.

        Best-effort — never blocks training.
        """
        try:
            from src.training.run_manifest import write_run_manifest
            mpath = write_run_manifest(
                self.checkpoint_dir,
                game=game,
                rom_path=rom_path,
                start_state_path=start_state_path,
                seed=seed,
                profile=profile,
                num_envs=num_envs,
                frame_skip=frame_skip,
            )
            log.info("[vanilla_ppo] wrote run manifest: %s", mpath)
        except Exception as exc:
            log.warning("[vanilla_ppo] run manifest write failed: %s", exc)

    def save_iter(
        self, *, net, optimizer, adv_net, adv_opt, bwd_on, bwd_sched,
        anticollapse, it, global_it, curriculum_resume=None,
    ) -> bool:
        """Save the periodic vanilla_ppo iter checkpoint, poison-guarded.

        The loop-local collaborators (`net`, `optimizer`, the PR-MDP
        adversary, the backward scheduler, the anti-collapse baseline,
        `it`/`global_it`) arrive as arguments; the live `_rnd`, `_gx_counts`,
        and `checkpoint_dir` are read from the trainer exactly as the inline
        block did. `anticollapse` is the `(best_net_snapshot,
        best_snapshot_fitness, collapse_strikes)` rollback baseline.
        `curriculum_resume` is the SMB curriculum's rolling advance history +
        Go-Explore burst bookkeeping (builtin-typed dict, or None to omit).

        Returns `_params_finite` so the conductor's winner-retention gate is
        identical to the inline `if it > 0 and it % 10 == 0 and _params_finite:`
        block. The write is skipped (poison guard) on non-finite params.
        """
        best_net_snapshot, best_snapshot_fitness, collapse_strikes = anticollapse
        _adv_net = adv_net
        _adv_opt = adv_opt
        # Checkpoint every 10 iters. No GA to save — just the policy
        # network's state_dict (plus the ABSOLUTE iter counter for
        # resume). Filenames use global_it so a resumed run never
        # overwrites checkpoints saved before the resume point.
        _params_finite = True
        if it > 0 and it % 10 == 0:
            # Poison guard: a numerically-collapsed net must never
            # become the auto-resume point, or the crash supervisor
            # would relaunch straight back into NaN. Refusing the
            # save keeps the last GOOD checkpoint as the rollback.
            # Checks the NET, the RND module, AND the optimizer
            # moments: grad clipping covers net.parameters() only, so
            # the RND predictor (same Adam) can go non-finite while
            # the net stays finite — a save landing there would
            # persist finite-net + NaN-RND/NaN-moments as the resume
            # point, and the load path restores them unvalidated.
            def _all_finite(sd) -> bool:
                return all(
                    torch.isfinite(v).all()
                    for v in sd.values()
                    if torch.is_tensor(v) and v.is_floating_point()
                )
            _params_finite = _all_finite(net.state_dict())
            if _params_finite and self.trainer._rnd is not None:
                _params_finite = _all_finite(self.trainer._rnd.state_dict())
            if _params_finite:
                for _grp in optimizer.state.values():
                    if not all(
                        torch.isfinite(t).all()
                        for t in _grp.values()
                        if torch.is_tensor(t) and t.is_floating_point()
                    ):
                        _params_finite = False
                        break
            if not _params_finite:
                log.error(
                    "[vanilla_ppo] REFUSING checkpoint save at iter "
                    "%d: non-finite parameters (numeric collapse) — "
                    "auto-resume will roll back to the last good "
                    "checkpoint", global_it,
                )
        if it > 0 and it % 10 == 0 and _params_finite:
            ckpt_path = (
                self.trainer.checkpoint_dir
                / f"vanilla_ppo_iter_{global_it:05d}.pt"
            )
            try:
                _ckpt_payload = {
                    "iter": global_it,
                    "net_state_dict": {
                        k: v.detach().cpu()
                        for k, v in net.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                # Persist RND (predictor weights + obs/reward running
                # stats) so resume keeps the novelty memory instead of
                # re-rewarding every already-explored state.
                if self.trainer._rnd is not None:
                    _ckpt_payload["rnd_state_dict"] = {
                        k: v.detach().cpu()
                        for k, v in self.trainer._rnd.state_dict().items()
                    }
                # Persist the PR-MDP adversary (net + optimizer) so a
                # resume continues the arms race instead of pitting the
                # trained protagonist against a fresh random opponent.
                if _adv_net is not None:
                    _ckpt_payload["prmdp_adv_net_state_dict"] = {
                        k: v.detach().cpu()
                        for k, v in _adv_net.state_dict().items()
                    }
                    _ckpt_payload["prmdp_adv_optimizer_state_dict"] = (
                        _adv_opt.state_dict()
                    )
                # Persist the gx-visitation table so a bounce doesn't
                # re-inflate already-trodden buckets' bonuses.
                if self.trainer._gx_counts:
                    _ckpt_payload["gx_counts"] = self.trainer._gx_counts
                # Persist the backward cursor (tau + the trailing rung
                # window + entrance counters) so a resume continues the
                # ladder where it stopped. Without it, --resume put tau
                # back at the ladder top and the run re-walked ground it
                # had already earned — B4 v3 resumed at iter 80 with
                # tau=757/757, so the at-entrance winner gate (which
                # selects this mode's deliverable) never fired again.
                if bwd_on and bwd_sched is not None:
                    _ckpt_payload["backward_curriculum"] = (
                        bwd_sched.state_dict()
                    )
                # Persist the anti-collapse rollback baseline (best-ever
                # healthy snapshot + its fitness + strike count) so a
                # resumed run keeps its collapse-recovery memory. The
                # snapshot is already cpu-cloned at capture time.
                if best_net_snapshot is not None:
                    _ckpt_payload["anticollapse"] = {
                        "best_net_snapshot": best_net_snapshot,
                        "best_snapshot_fitness": float(best_snapshot_fitness),
                        "collapse_strikes": int(collapse_strikes),
                    }
                # Persist the curriculum's rolling advance history + the
                # Go-Explore burst state so a resume continues the advance
                # gate and stall clock where they stopped, instead of
                # re-earning the rolling window from an empty history.
                if curriculum_resume is not None:
                    _ckpt_payload["curriculum_resume"] = curriculum_resume
                # Atomic write: tmp + fsync + os.replace, so a death
                # mid-save (OOM/SIGKILL — and the full CPU state copy
                # above widens that window) can never leave a
                # TRUNCATED highest-numbered checkpoint that the
                # resume scan would then pick and fail to load. The
                # old bare torch.save-to-final-path could, converting
                # a long run into a silent from-scratch restart under
                # the supervisor. Mirrors save_checkpoint_atomic.
                _tmp_path = ckpt_path.with_suffix(".pt.tmp")
                torch.save(_ckpt_payload, str(_tmp_path))
                try:
                    _fd = os.open(str(_tmp_path), os.O_RDONLY)
                    try:
                        os.fsync(_fd)
                    finally:
                        os.close(_fd)
                except OSError:
                    pass
                os.replace(str(_tmp_path), str(ckpt_path))
                log.info("[vanilla_ppo] saved checkpoint: %s", ckpt_path)
            except Exception as exc:
                log.warning("[vanilla_ppo] checkpoint save failed: %s", exc)
        return _params_finite
