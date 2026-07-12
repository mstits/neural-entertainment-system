"""Guard: the vanilla_ppo auto-resume scan honors an explicit
from-scratch request.

ISSUE-4 (2026-07-12 validation): the GUI "Resume" checkbox was a no-op
for vanilla_ppo — `_run_vanilla_ppo` always scanned checkpoint_dir for
`vanilla_ppo_iter_*.pt` and resumed regardless, so an unticked box (and
headless `--no-resume`) silently continued a supposedly-fresh run.

The scan now lives in `_maybe_resume_vanilla_ppo(net, optimizer,
fresh_start=...)`:
  * fresh_start=True  -> skip the scan, start from random weights;
  * fresh_start=False / default -> load the newest checkpoint (the
    historical behavior headless scripts rely on).

These tests exercise the decision path directly (Trainer.__new__ + a
temp checkpoint dir) so they never boot an emulator or run training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.trainer import Trainer


def _make_trainer(tmp_path):
    # Same pattern as test_vanilla_ppo_mode: build a Trainer shell
    # without the (slow) env/pool boot and inject only the attributes
    # the resume helper touches.
    t = Trainer.__new__(Trainer)
    t.checkpoint_dir = tmp_path
    t.device = torch.device("cpu")
    return t


def _write_ckpt(path, iter_num, fill=None):
    net = nn.Linear(4, 2)
    if fill is not None:
        with torch.no_grad():
            for p in net.parameters():
                p.fill_(fill)
    opt = torch.optim.Adam(net.parameters())
    torch.save(
        {
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "iter": iter_num,
        },
        str(path),
    )


def test_fresh_start_skips_existing_checkpoint(tmp_path):
    t = _make_trainer(tmp_path)
    _write_ckpt(tmp_path / "vanilla_ppo_iter_00010.pt", 10)

    net = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters())
    offset = t._maybe_resume_vanilla_ppo(net, opt, fresh_start=True)

    assert offset == 0
    assert t._vppo_resumed_from_iter is None


def test_default_resumes_from_latest_checkpoint(tmp_path):
    t = _make_trainer(tmp_path)
    # Saved weights are all 0.5 so we can prove they were actually loaded.
    _write_ckpt(tmp_path / "vanilla_ppo_iter_00010.pt", 10, fill=0.5)

    net = nn.Linear(4, 2)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    opt = torch.optim.Adam(net.parameters())

    # Default (no signal) resumes — backward compatible with headless.
    offset = t._maybe_resume_vanilla_ppo(net, opt)

    assert offset == 10
    assert t._vppo_resumed_from_iter == 10
    for p in net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))


def test_fresh_false_explicit_resumes(tmp_path):
    t = _make_trainer(tmp_path)
    _write_ckpt(tmp_path / "vanilla_ppo_iter_00020.pt", 20)

    net = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters())
    offset = t._maybe_resume_vanilla_ppo(net, opt, fresh_start=False)

    assert offset == 20
    assert t._vppo_resumed_from_iter == 20


def test_picks_highest_iter_not_lexical(tmp_path):
    t = _make_trainer(tmp_path)
    # Lexical sort would put _00009 after _000100; numeric sort must win.
    _write_ckpt(tmp_path / "vanilla_ppo_iter_00009.pt", 9)
    _write_ckpt(tmp_path / "vanilla_ppo_iter_00100.pt", 100)

    net = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters())
    offset = t._maybe_resume_vanilla_ppo(net, opt)

    assert offset == 100
    assert t._vppo_resumed_from_iter == 100


def test_no_checkpoint_starts_fresh(tmp_path):
    t = _make_trainer(tmp_path)

    net = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters())
    offset = t._maybe_resume_vanilla_ppo(net, opt)

    assert offset == 0
    assert t._vppo_resumed_from_iter is None
