"""End-to-end resume round-trip for the vanilla_ppo training path.

Resumability is a promised, user-facing feature — the GUI "Resume"
checkbox and `train ... --resume auto` both rely on a crashed or
stopped run picking up exactly where it left off — yet it had ZERO
automated coverage. The failure mode this guards is not hypothetical:
a resumed run that silently restarts its iteration counter at 0
overwrites the checkpoints saved before the resume point, which is how
a good overnight run once destroyed its own best weights (see the
resume-safe-numbering comment in Trainer._run_vanilla_ppo).

Unlike the GA path (which resumes via `run(resume_from=<gen ckpt>)`),
the vanilla_ppo loop AUTO-resumes: on entry it scans `checkpoint_dir`
for the latest `vanilla_ppo_iter_*.pt` and loads net + optimizer +
absolute-iter offset on top of its freshly-built policy. So "resume
enabled" for this mode just means "point a fresh Trainer at the same
checkpoint dir" — no flag required. That is exactly what the GUI and
CLI do on restart, so it is what this test drives.

The loop only writes a checkpoint every 10 iters, so a genuine
artifact requires 11 iters (it=0..10, saved at it=10). The test then
constructs two further, independently-built Trainers on the same dir:

  * one runs 0 iters, so we can observe the resumed net + optimizer
    state BEFORE any training mutates it — proving the latest
    checkpoint was loaded byte-for-byte (criteria a + c);
  * one runs another 11 iters, so the resume-applied iter offset
    surfaces as a checkpoint numbered 20 (10 saved + 10 resumed)
    while the pre-resume checkpoint at 10 is left untouched — proving
    it continues from the saved iteration, not 0 (criterion b).

It is a wiring guard, not a learning test: ~22 tiny iterations can't
learn anything. Marked slow because it builds three real pools and
runs the real rollout -> GAE -> PPO -> checkpoint loop.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
_SMB_ROM = ROOT / "roms" / "Super Mario Bros. (World).nes"
_PROFILE = ROOT / "configs" / "mario_vanilla_ppo.yaml"

# Checkpoint cadence in Trainer._run_vanilla_ppo: `it > 0 and it % 10 == 0`.
_SAVE_EVERY = 10

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _SMB_ROM.exists() or not _PROFILE.exists(),
        reason="SMB ROM / vanilla_ppo profile not present.",
    ),
]


def _tiny_profile() -> dict:
    """The real vanilla_ppo profile shrunk so an iter runs in ~0.1s."""
    with open(_PROFILE) as f:
        profile = yaml.safe_load(f)
    profile["reinforce"]["rollout_steps"] = 16
    profile["reinforce"]["steps"] = 2          # K PPO epochs
    profile["reinforce"]["ppo_minibatch_size"] = 16
    return profile


def _make_trainer(profile: dict, checkpoint_dir: str):
    from src.training.trainer import Trainer

    return Trainer(
        rom_path=str(_SMB_ROM),
        game_profile=profile,
        num_instances=4,
        population_size=4,
        checkpoint_dir=checkpoint_dir,   # explicit -> honored verbatim, isolated
        start_state_path=profile.get("start_state_path"),
        env_spec="nes_core",
        max_episode_steps=200,
    )


def test_vanilla_ppo_resume_roundtrip() -> None:
    profile = _tiny_profile()

    with tempfile.TemporaryDirectory(prefix="resume_rt_") as tmp:
        ckpt_dir = Path(tmp)

        # ---- Phase A: fresh run writes a genuine checkpoint ----
        first = _make_trainer(profile, tmp)
        assert str(first.checkpoint_dir) == tmp  # no per-game slug on explicit dir
        first.run(num_generations=_SAVE_EVERY + 1, resume_from=None)

        saved = ckpt_dir / f"vanilla_ppo_iter_{_SAVE_EVERY:05d}.pt"
        assert saved.exists(), (
            "the training loop never wrote a checkpoint — expected "
            f"{saved.name} after {_SAVE_EVERY + 1} iters (saves every "
            f"{_SAVE_EVERY})"
        )
        state = torch.load(str(saved), map_location="cpu")
        assert state["iter"] == _SAVE_EVERY
        # Adam accumulated moments over the run — non-empty optimizer state
        # is the thing a resume must restore (a fresh Adam has state == {}).
        assert len(state["optimizer_state_dict"]["state"]) > 0
        saved_weights = {k: v.clone() for k, v in state["net_state_dict"].items()}

        # ---- Phase B: a FRESH Trainer on the same dir must LOAD it ----
        # run(0) executes the real auto-resume scan (which runs before the
        # iteration loop) and then stops, so we observe the restored state
        # before any PPO update can mutate it.
        loader = _make_trainer(profile, tmp)
        assert loader._ppo_net is None  # nothing built yet
        loader.run(num_generations=0, resume_from=None)

        # (a) latest checkpoint loaded into the freshly-built policy net.
        assert loader._ppo_net is not None, (
            "resume scan did not build/populate the policy net"
        )
        restored = loader._ppo_net.state_dict()
        assert set(restored) == set(saved_weights), "net architecture drift"
        for k, want in saved_weights.items():
            assert torch.equal(restored[k].cpu(), want.cpu()), (
                f"resumed weight {k!r} does not match the saved checkpoint"
            )
        # (c) optimizer state restored — non-empty only because it was
        # loaded (run(0) never steps the optimizer itself).
        assert loader._ppo_optimizer is not None
        assert len(loader._ppo_optimizer.state_dict()["state"]) > 0, (
            "optimizer Adam moments were not restored on resume"
        )

        # ---- Phase C: a resumed run CONTINUES the iteration counter ----
        cont = _make_trainer(profile, tmp)
        cont.run(num_generations=_SAVE_EVERY + 1, resume_from=None)

        # (b) resume applied the saved iter as an offset: the next
        # checkpoint is numbered (saved_iter + save_every) = 20, NOT a
        # second 10 that would clobber the pre-resume artifact.
        continued = ckpt_dir / f"vanilla_ppo_iter_{2 * _SAVE_EVERY:05d}.pt"
        assert continued.exists(), (
            "resumed run did not continue the absolute iteration counter — "
            f"expected {continued.name} (iter_offset not applied); a resume "
            "that restarts at 0 silently overwrites pre-resume checkpoints"
        )
        assert saved.exists(), (
            "the pre-resume checkpoint was overwritten — resume-safe "
            "numbering regressed"
        )
        cont_state = torch.load(str(continued), map_location="cpu")
        assert cont_state["iter"] == 2 * _SAVE_EVERY
