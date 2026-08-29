"""Disk-floor guards on the two training launchers + the consecutive-
save-failure halt in CheckpointManager (external audit 2026-08-29).

Before this: only the engine scheduler checked free disk; a
hand-launched trainer or campaign on a 91%-full volume ran for weeks
while ENOSPC checkpoint saves degraded into `log.warning` — compute
burned for output that would not survive a crash.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.train_game as tg  # noqa: E402
import scripts.run_online_campaign as camp  # noqa: E402


# ---------------------------------------------------------------------------
# train_game.py: launch-time floor
# ---------------------------------------------------------------------------


def test_low_disk_refuses(monkeypatch):
    monkeypatch.setattr(tg, "_disk_free_gb", lambda p: 5.0)
    assert tg._check_disk_floor(REPO) is False


def test_low_disk_with_override_proceeds(monkeypatch):
    monkeypatch.setattr(tg, "_disk_free_gb", lambda p: 5.0)
    assert tg._check_disk_floor(REPO, allow_low_disk=True) is True


def test_enough_disk_proceeds(monkeypatch):
    monkeypatch.setattr(tg, "_disk_free_gb", lambda p: tg.DISK_FLOOR_GB + 1)
    assert tg._check_disk_floor(REPO) is True


def test_guard_error_fails_open(monkeypatch):
    def boom(p):
        raise OSError("statvfs hiccup")
    monkeypatch.setattr(tg, "_disk_free_gb", boom)
    assert tg._check_disk_floor(REPO) is True, (
        "a stat error must never be the reason a run is refused")


def test_main_refuses_before_doing_anything_heavy(monkeypatch):
    """With low disk, main() must return 1 BEFORE profile resolution —
    proven by passing a bogus --game that would otherwise SystemExit
    at resolution time."""
    monkeypatch.setattr(tg, "_disk_free_gb", lambda p: 5.0)
    monkeypatch.setattr(sys, "argv",
                        ["train_game.py", "--game", "no-such-game-xyz"])
    assert tg.main() == 1


def test_main_floor_does_not_fire_spuriously(monkeypatch):
    """Same bogus game with plenty of disk: the run must get PAST the
    floor (and die at profile resolution instead)."""
    monkeypatch.setattr(tg, "_disk_free_gb",
                        lambda p: tg.DISK_FLOOR_GB + 100)
    monkeypatch.setattr(sys, "argv",
                        ["train_game.py", "--game", "no-such-game-xyz"])
    with pytest.raises(SystemExit):
        tg.main()


# ---------------------------------------------------------------------------
# run_online_campaign.py: launch-time + per-tick floor
# ---------------------------------------------------------------------------


def test_campaign_breach_reason_below_floor(monkeypatch):
    monkeypatch.setattr(camp, "disk_free_gb", lambda *a: 3.0)
    reason = camp.disk_floor_breach()
    assert reason and "disk floor" in reason


def test_campaign_no_breach_with_headroom(monkeypatch):
    monkeypatch.setattr(camp, "disk_free_gb",
                        lambda *a: camp.DISK_FLOOR_GB + 1)
    assert camp.disk_floor_breach() is None


def test_campaign_guard_error_fails_open(monkeypatch):
    def boom(*a):
        raise OSError("hiccup")
    monkeypatch.setattr(camp, "disk_free_gb", boom)
    assert camp.disk_floor_breach() is None


def test_campaign_main_refuses_to_start_below_floor(monkeypatch):
    monkeypatch.setattr(camp, "disk_free_gb", lambda *a: 3.0)
    monkeypatch.setattr(camp, "run_campaign",
                        lambda **kw: 0)
    monkeypatch.setattr(sys, "argv", ["run_online_campaign.py"])
    with pytest.raises(SystemExit, match="REFUSING"):
        camp.main()


def test_campaign_main_starts_with_headroom(monkeypatch):
    monkeypatch.setattr(camp, "disk_free_gb",
                        lambda *a: camp.DISK_FLOOR_GB + 1)
    sentinel: list = []
    monkeypatch.setattr(camp, "run_campaign",
                        lambda **kw: sentinel.append(kw) or 0)
    monkeypatch.setattr(sys, "argv", ["run_online_campaign.py"])
    assert camp.main() == 0
    assert sentinel, "run_campaign must be reached when disk is fine"


# ---------------------------------------------------------------------------
# CheckpointManager: consecutive-save-failure escalation
# ---------------------------------------------------------------------------


def _manager_and_net(tmp_path):
    import torch
    from src.training.checkpoint_manager import CheckpointManager
    trainer = SimpleNamespace(
        checkpoint_dir=tmp_path,
        device="cpu",
        _rnd=None,
        _gx_counts={},
        game_profile={"reinforce": {}},
    )
    mgr = CheckpointManager(trainer, checkpoint_dir=tmp_path, device="cpu")
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters())
    return mgr, net, opt


def _save(mgr, net, opt, it):
    return mgr.save_iter(
        net=net, optimizer=opt, adv_net=None, adv_opt=None,
        bwd_on=False, bwd_sched=None,
        anticollapse=(None, 0.0, 0), it=it, global_it=it,
    )


def test_three_consecutive_save_failures_halt_the_run(tmp_path, monkeypatch):
    import torch
    mgr, net, opt = _manager_and_net(tmp_path)
    notified: list = []
    broken = types.ModuleType("src.training.notifications")
    broken.notify_macos = lambda *a, **kw: notified.append(a)
    monkeypatch.setitem(sys.modules, "src.training.notifications", broken)

    def enospc(*a, **kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(torch, "save", enospc)

    _save(mgr, net, opt, 10)   # 1st failure: warn, continue
    _save(mgr, net, opt, 20)   # 2nd: warn, continue
    with pytest.raises(RuntimeError, match="consecutive"):
        _save(mgr, net, opt, 30)   # 3rd: halt loudly
    assert notified, "the halt must push a notification"


def test_a_successful_save_resets_the_failure_counter(tmp_path, monkeypatch):
    import torch
    mgr, net, opt = _manager_and_net(tmp_path)
    real_save = torch.save

    def enospc(*a, **kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(torch, "save", enospc)
    _save(mgr, net, opt, 10)
    _save(mgr, net, opt, 20)
    assert mgr._consecutive_save_failures == 2
    monkeypatch.setattr(torch, "save", real_save)
    _save(mgr, net, opt, 30)   # succeeds
    assert mgr._consecutive_save_failures == 0
    assert (tmp_path / "vanilla_ppo_iter_00030.pt").exists()
    # Two MORE failures after a success must not halt (not consecutive
    # with the first two).
    monkeypatch.setattr(torch, "save", enospc)
    _save(mgr, net, opt, 40)
    _save(mgr, net, opt, 50)
    assert mgr._consecutive_save_failures == 2


def test_notification_failure_does_not_mask_the_halt(tmp_path, monkeypatch):
    import torch
    mgr, net, opt = _manager_and_net(tmp_path)
    broken = types.ModuleType("src.training.notifications")

    def boom(*a, **kw):
        raise RuntimeError("osascript exploded")
    broken.notify_macos = boom
    monkeypatch.setitem(sys.modules, "src.training.notifications", broken)

    def enospc(*a, **kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(torch, "save", enospc)
    _save(mgr, net, opt, 10)
    _save(mgr, net, opt, 20)
    with pytest.raises(RuntimeError, match="consecutive"):
        _save(mgr, net, opt, 30)
