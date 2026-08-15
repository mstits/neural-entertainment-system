"""P1-4 — curriculum resume state survives the iter checkpoint.

`CheckpointManager.save_iter` omitted the SMB curriculum's rolling
advance history (`smb_stage_clear_history` / `smb_pastfrac_history`) and
the Go-Explore burst state (`ge_burst_*`), so every resume reset the
advance gate's rolling mean and the stall clock to fresh — a resumed run
re-earned window fills it had already banked and lost mid-burst
bookkeeping. The blob now rides the checkpoint under `curriculum_resume`
and is staged on resume as `_pending_curriculum_resume` for the loop's
consume site. Round-trips through the REAL save (`save_iter`) and the
REAL resume (`_maybe_resume_vanilla_ppo`), reusing the ROM-free
`_bare_tile_trainer` shell per the C1 pattern.

Backward compatibility is part of the contract: an older checkpoint
without the key must load clean and leave the pending blob unset, so the
loop falls back to the historical fresh-reset behavior.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.training.checkpoint_manager import CheckpointManager
from tests.test_vanilla_ppo_characterization import _bare_tile_trainer

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_SRC = (ROOT / "src" / "training" / "trainer.py").read_text()
_CKPT_SRC = (ROOT / "src" / "training" / "checkpoint_manager.py").read_text()


def _seed_all(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _cpu_state(net) -> dict:
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


_BLOB = {
    "smb_stage_clear_history": [3, 5, 4],
    "smb_pastfrac_history": [0.2, 0.4, 0.35, 0.5, 0.45],
    "ge_burst_active": True,
    "ge_burst_remaining": 17,
    "ge_burst_quota": 6,
    "ge_iters_since_advance": 42,
    "ge_bursts_done": 2,
}


def _save_iter(trainer, net, opt, *, it: int, **extra) -> bool:
    """Drive the REAL `save_iter` with the minimal collaborators."""
    mgr = CheckpointManager(
        trainer,
        checkpoint_dir=trainer.checkpoint_dir,
        device=torch.device("cpu"),
    )
    return mgr.save_iter(
        net=net,
        optimizer=opt,
        adv_net=None,
        adv_opt=None,
        bwd_on=False,
        bwd_sched=None,
        anticollapse=(None, float("-inf"), 0),
        it=it,
        global_it=it,
        **extra,
    )


def _fresh_shell_resume(tmp: Path):
    t = _bare_tile_trainer(tmp)
    net = t._make_network()
    net.to("cpu")
    opt = t._build_ppo_optimizer(net)
    iter_offset = t._maybe_resume_vanilla_ppo(net, opt, fresh_start=False)
    return t, net, iter_offset


def test_curriculum_resume_blob_roundtrips() -> None:
    """Populated histories + burst state saved by the real `save_iter` come
    back verbatim as `_pending_curriculum_resume` through the real resume."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="p14_cr_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = src._make_network()
        net.to("cpu")
        opt = src._build_ppo_optimizer(net)

        saved = _save_iter(src, net, opt, it=10, curriculum_resume=dict(_BLOB))
        assert saved is True
        assert (tmp / "vanilla_ppo_iter_00010.pt").exists()

        t, _net, iter_offset = _fresh_shell_resume(tmp)

        assert iter_offset == 10
        cr = getattr(t, "_pending_curriculum_resume", None)
        assert cr is not None, "curriculum blob was not staged on resume"
        assert cr == _BLOB


def test_old_checkpoint_without_blob_loads_clean() -> None:
    """A pre-P1-4 checkpoint (no `curriculum_resume` key) must resume without
    error and leave the pending blob unset — the loop then falls back to the
    historical fresh-reset defaults."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="p14_old_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = src._make_network()
        net.to("cpu")
        opt = src._build_ppo_optimizer(net)

        torch.save(
            {
                "iter": 20,
                "net_state_dict": _cpu_state(net),
                "optimizer_state_dict": opt.state_dict(),
            },
            str(tmp / "vanilla_ppo_iter_00020.pt"),
        )

        t, _net, iter_offset = _fresh_shell_resume(tmp)

        assert iter_offset == 20
        assert getattr(t, "_pending_curriculum_resume", None) is None


def test_save_iter_without_blob_omits_key() -> None:
    """Callers that do not pass `curriculum_resume` keep writing the exact
    pre-P1-4 payload shape (no stray key for older readers to trip on)."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="p14_omit_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = src._make_network()
        net.to("cpu")
        opt = src._build_ppo_optimizer(net)

        assert _save_iter(src, net, opt, it=10) is True
        payload = torch.load(
            str(tmp / "vanilla_ppo_iter_00010.pt"), map_location="cpu"
        )
        assert "curriculum_resume" not in payload


def test_consume_and_call_sites_present_in_source() -> None:
    """Anchor the loop-side wiring (inline in `_run_vanilla_ppo`, so it is
    source-anchored rather than executed — the C1 convention): the save call
    passes the live locals, the consume site restores every field, and the
    restored-vs-fresh verdict is logged."""
    assert 'curriculum_resume={' in _TRAINER_SRC
    assert '_cr = getattr(self, "_pending_curriculum_resume", None)' \
        in _TRAINER_SRC
    for field in _BLOB:
        assert f'_cr.get("{field}"' in _TRAINER_SRC, (
            f"consume site does not restore {field}"
        )
    assert "curriculum resume state RESTORED" in _TRAINER_SRC
    assert "curriculum resume state: fresh reset" in _TRAINER_SRC
    # Save + resume halves in the manager.
    assert '_ckpt_payload["curriculum_resume"] = curriculum_resume' in _CKPT_SRC
    assert 'self.trainer._pending_curriculum_resume = ' \
        'state["curriculum_resume"]' in _CKPT_SRC
