"""P1 — `save_iter`'s tmp checkpoint path must be per-writer, not deterministic.

Audit finding (checkpoint_manager.py:327): the tmp path used by the
tmp+fsync+os.replace atomic-write sequence was derived solely from the final
checkpoint's name (`ckpt_path.with_suffix(".pt.tmp")`), with no pid/uuid
component. Two processes sharing a `checkpoint_dir` (both resumed from the
same latest checkpoint) hit the same `it % 10 == 0` boundary at the same
`global_it` and both open the IDENTICAL tmp file for `torch.save`, which
performs many buffered writes rather than one atomic syscall — the two
writers' output can interleave into a torn blob, and whichever `os.replace`
runs last promotes that torn file to the canonical checkpoint name. This
defeats the single-writer atomicity guarantee the surrounding code comment
claims.

The fix salts the tmp filename with `os.getpid()` and a `uuid4` fragment so
concurrent writers targeting the same `global_it` always open distinct
inodes. These tests pin that contract directly against the REAL `save_iter`
(via `CheckpointManager`, reusing the ROM-free `_bare_tile_trainer` shell
from the C1 pattern) rather than against a mirror.
"""

from __future__ import annotations

import os
import random
import tempfile
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.training.checkpoint_manager import CheckpointManager
from tests.test_vanilla_ppo_characterization import _bare_tile_trainer


def _seed_all(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def test_save_iter_tmp_path_is_not_deterministic_across_writers() -> None:
    """Two `save_iter` calls at the SAME `global_it` (the racing-writer
    scenario: two processes resumed from the same checkpoint, both hitting
    the same `it % 10 == 0` boundary) must never target the identical tmp
    file. Before the fix, `ckpt_path.with_suffix(".pt.tmp")` was derived
    only from the final name, so both calls produced the exact same tmp
    path — a second writer would race `torch.save`'s buffered writes onto
    the shared inode of the first."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="p1_tmp_unique_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = src._make_network()
        net.to("cpu")
        opt = src._build_ppo_optimizer(net)

        seen_tmp_paths: list[str] = []
        real_replace = os.replace

        def _spy_replace(src_path, dst_path):
            seen_tmp_paths.append(str(src_path))
            return real_replace(src_path, dst_path)

        with mock.patch(
            "src.training.checkpoint_manager.os.replace",
            side_effect=_spy_replace,
        ):
            assert _save_iter(src, net, opt, it=10) is True
            assert _save_iter(src, net, opt, it=10) is True

        assert len(seen_tmp_paths) == 2
        assert seen_tmp_paths[0] != seen_tmp_paths[1], (
            "two save_iter calls at the same global_it produced the "
            "identical tmp path — a second writer would race torch.save "
            "onto the same inode as the first"
        )
        # Salted with this process's pid, per the fix.
        for p in seen_tmp_paths:
            assert f".{os.getpid()}." in p, (
                f"tmp path {p!r} is not salted with the writer's pid"
            )


def test_concurrent_save_iter_same_global_it_does_not_corrupt() -> None:
    """End-to-end regression for the race itself: two threads standing in
    for two processes that share a `checkpoint_dir` and both resumed to the
    same `global_it`, calling the REAL `save_iter` concurrently for the
    same iter number. Before the fix both threads' `torch.save` calls could
    land on the identical tmp path and interleave into a torn blob that the
    later `os.replace` promotes to the canonical name. With unique tmp
    paths, neither thread's write can be corrupted by the other, and
    whichever `os.replace` lands last always produces one WHOLE, loadable
    checkpoint (never a blend of two, never a torn file)."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="p1_concurrent_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)

        errors: list[BaseException] = []
        results: list[bool] = []
        lock = threading.Lock()

        def _writer(seed: int) -> None:
            try:
                torch.manual_seed(seed)
                net = src._make_network()
                net.to("cpu")
                opt = src._build_ppo_optimizer(net)
                ok = _save_iter(src, net, opt, it=10)
                with lock:
                    results.append(ok)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_writer, args=(seed,)) for seed in (1, 2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not any(t.is_alive() for t in threads), "writer thread hung"
        assert not errors, f"writer thread(s) raised: {errors}"
        assert results == [True, True]

        ckpt_path = tmp / "vanilla_ppo_iter_00010.pt"
        assert ckpt_path.exists()
        # A torn/interleaved payload would fail to unpickle here, or would
        # load with a missing/corrupt tensor.
        payload = torch.load(str(ckpt_path), map_location="cpu")
        assert "net_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert payload["iter"] == 10

        # No stray tmp files left behind by either writer.
        leftovers = list(tmp.glob("*.tmp*"))
        assert leftovers == [], f"tmp file(s) left behind: {leftovers}"
