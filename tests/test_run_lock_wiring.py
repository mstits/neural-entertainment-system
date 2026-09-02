"""Every solver-chain writer holds a run-lock on its shared output path
(the 2026-09-01 [process] backfill entry, MISTAKES.md).

`src/utils/run_lock.py` guards `scripts/train_game.py` and
`scripts/collect_substrate_pairs.py` directly (see tests/test_run_lock.py
for the lock primitive itself). This file checks the REMAINING writers
each wire the same lock in: one parametrized test per script, importing
that script's own lock-path helper and driving it through the real
`acquire`/`release` from src/utils/run_lock — a second acquire on the
SAME lock path a script would compute must be refused.

`scripts/discover_observables.py` also gets a receipt-path test: the
--selftest write is tmp-file + os.replace (a reader must never observe
a partial write), and the write lands exactly at the receipt path the
lock is guarding.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.run_lock import acquire, read_lock, release  # noqa: E402

from scripts.discover_observables import (  # noqa: E402
    selftest_lock_path,
    selftest_receipt_path,
)
from scripts.go_explore_chain import chain_lock_path  # noqa: E402
from scripts.go_explore_solve import Solver, solver_lock_path  # noqa: E402
from scripts.merge_recovery_ladder import merge_lock_path  # noqa: E402
import scripts.merge_recovery_ladder as merge_recovery_ladder  # noqa: E402
from scripts.run_online_campaign import campaign_lock_path  # noqa: E402
from scripts.run_v31_eval_ladder import (  # noqa: E402
    _receipt_path,
    ladder_lock_path,
    main as ladder_main,
)

# (label, lock-path helper, args to call it with) — one row per wired
# writer. `out` is a throwaway tmp_path target; only the helper's own
# path-shaping is under test, not the script's full write path (each
# script's docstring/comment at its acquire() call site names the
# incident this guards against).
WIRED = [
    ("go_explore_chain", chain_lock_path, lambda out: (out,)),
    ("go_explore_solve", solver_lock_path, lambda out: (out,)),
    ("run_v31_eval_ladder", ladder_lock_path, lambda out: (out,)),
    ("run_online_campaign", campaign_lock_path, lambda out: (out,)),
    ("merge_recovery_ladder", merge_lock_path, lambda out: (out,)),
    ("discover_observables_selftest", selftest_lock_path, lambda out: (out,)),
]


@pytest.mark.parametrize("label,path_fn,args_fn", WIRED, ids=[w[0] for w in WIRED])
def test_second_acquire_on_wired_lock_path_is_refused(tmp_path, label, path_fn, args_fn):
    out = tmp_path / "out"
    lock = path_fn(*args_fn(out))
    lock.parent.mkdir(parents=True, exist_ok=True)

    assert acquire(lock) is None, f"{label}: first acquire must succeed"
    holder = acquire(lock)
    assert holder is not None, (
        f"{label}: a second acquire on {lock} (the exact path "
        f"{path_fn.__name__} computes) must be refused while the first "
        "holder is live"
    )
    assert holder.pid == os.getpid()

    release(lock)
    assert not lock.exists(), f"{label}: release must remove the lockfile"
    assert acquire(lock) is None, f"{label}: acquire must succeed again after release"
    release(lock)


def test_merge_lock_path_survives_out_dir_rmtree(tmp_path):
    """merge_recovery_ladder's --force path rmtrees out_dir; the lock
    must live beside it, not inside it, or the rmtree would delete a
    lock the same process just took."""
    out_dir = tmp_path / "ladder_v27"
    out_dir.mkdir()
    lock = merge_lock_path(out_dir)
    assert lock.parent == out_dir.parent
    assert acquire(lock) is None
    import shutil
    shutil.rmtree(out_dir)
    assert lock.exists(), "the lock must survive an out_dir rmtree"
    release(lock)


def test_selftest_lock_and_receipt_share_one_runs_dir(tmp_path):
    """The --selftest lock and its receipt are fixed, unparameterized
    paths (no per-invocation naming) — both must resolve under whatever
    runs_dir override is given, so a test (or a caller) can't lock one
    directory while the real write lands in another."""
    receipt = selftest_receipt_path(tmp_path)
    lock = selftest_lock_path(tmp_path)
    assert receipt.parent == tmp_path
    assert lock.parent == tmp_path
    assert receipt.name == "discover_observables_selftest.json"
    assert lock.name == ".discover_observables_selftest.run.lock"


def test_selftest_write_is_tmp_file_then_atomic_replace(tmp_path, monkeypatch):
    """A crash between the write and the rename must never leave a
    truncated receipt at the real path — either the old good receipt
    is still there, or the new complete one is."""
    receipt = selftest_receipt_path(tmp_path)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"stale": true}\n')

    # Reproduce _selftest's own write tail without running the full
    # (emulator-heavy) self-test: same tmp-suffix + os.replace idiom.
    tmp = receipt.with_suffix(receipt.suffix + f".tmp{os.getpid()}")
    tmp.write_text('{"fresh": true}\n')
    assert receipt.read_text() == '{"stale": true}\n', (
        "writing the tmp file must not touch the real receipt"
    )
    os.replace(tmp, receipt)
    assert receipt.read_text() == '{"fresh": true}\n'
    assert not tmp.exists()


def test_selftest_source_actually_uses_tmp_file_and_os_replace():
    """Ties the write-pattern test above to the REAL `_selftest` body:
    a regression that reverts the write tail to a bare `write_text` on
    the receipt path would make this fail even though the OS-level
    idiom test above still passes on its own."""
    import inspect

    import scripts.discover_observables as discover_observables

    src = inspect.getsource(discover_observables._selftest)
    assert "os.replace(" in src, (
        "the receipt write must go through os.replace for atomicity"
    )
    assert ".tmp" in src, "the atomic write needs a tmp-file target"
    # The final write must not write straight to the receipt path
    # (out/_out) via write_text — only the tmp file may.
    assert "out.write_text(" not in src and "_out.write_text(" not in src


# ---------------------------------------------------------------------------
# Real end-to-end wiring checks: drive the actual acquire() call at each
# script's real call site (not just its lock-path helper), for the three
# writers cheap enough to exercise without a ROM. The lock is acquired
# strictly before any ROM/profile I/O in all three, so a fake --profile
# that fails fast is enough to prove the lock was actually taken by real
# production code, not simulated by the test.
# ---------------------------------------------------------------------------

def test_solver_init_holds_real_lock_and_refuses_second_construction(tmp_path):
    from types import SimpleNamespace

    out_dir = tmp_path / "solve_out"
    fake_args = SimpleNamespace(out=str(out_dir), root_state="fake.state",
                                profile=str(tmp_path / "missing_profile.yaml"))

    # First construction: passes the lock, then dies loading the (fake)
    # profile — expected, and irrelevant to what's under test here.
    with pytest.raises(Exception) as first:
        Solver(fake_args)
    assert not isinstance(first.value, SystemExit), (
        "first construction must fail on the missing profile, not the lock"
    )
    lock = solver_lock_path(out_dir)
    assert lock.exists(), "the real Solver.__init__ lock acquire must have run"
    assert read_lock(lock).pid == os.getpid()

    # Second construction, same --out: must be refused by the SAME PID's
    # own held lock before it ever touches the profile again.
    with pytest.raises(SystemExit) as second:
        Solver(fake_args)
    assert "is locked by live PID" in str(second.value)
    assert str(os.getpid()) in str(second.value)

    release(lock)


def test_ladder_main_holds_real_lock_and_refuses_second_run(tmp_path):
    out_dir = tmp_path / "ladder_out"
    # Pre-bank the one receipt the (seed, iter, eval-seed) grid below
    # would produce, so `jobs` is empty and main() reaches `return`
    # having acquired-and-held the lock without spawning eval_game.py.
    out_dir.mkdir(parents=True)
    _receipt_path(out_dir, "prefix", 0, 10, 0).write_text("{}")
    argv = ["--profile-template", "configs/does_not_exist_{seed}.yaml",
            "--checkpoint-dir-template", "checkpoints/does_not_exist_{seed}",
            "--receipt-prefix", "prefix", "--out-dir", str(out_dir),
            "--seeds", "0", "--iters", "10", "--eval-seeds", "0"]

    rc_first = ladder_main(argv)
    assert rc_first == 0, "zero queued jobs must not be reported as a failure"
    lock = ladder_lock_path(out_dir)
    assert lock.exists(), "the real main() lock acquire must have run"
    assert read_lock(lock).pid == os.getpid()

    rc_second = ladder_main(argv)
    assert rc_second == 75, (
        "a second run against the same --out-dir, same PID's lock still "
        "held, must be refused with EXIT_HELD"
    )

    release(lock)


def test_merge_main_holds_real_lock_and_refuses_second_run(tmp_path, monkeypatch):
    out_dir = tmp_path / "merge_out"
    argv = ["merge_recovery_ladder.py", "--out", str(out_dir),
            "--profile", "definitely_missing_profile.yaml"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(Exception) as first:
        merge_recovery_ladder.main()
    assert not isinstance(first.value, SystemExit), (
        "first run must fail on the missing profile (FileNotFoundError), "
        "not the lock"
    )
    lock = merge_lock_path(out_dir)
    assert lock.exists(), "the real main() lock acquire must have run"
    assert read_lock(lock).pid == os.getpid()

    with pytest.raises(SystemExit) as second:
        merge_recovery_ladder.main()
    assert "is locked by live PID" in str(second.value)

    release(lock)


# ---------------------------------------------------------------------------
# The shared "emulator-pool" resource lock (src/utils/run_lock.
# acquire_resource): unlike the per-`--out` locks above, this one is keyed
# by a NAME shared across scripts, not a path any one of them owns — it
# guards the physical emulator pool itself, which collect_substrate_pairs.
# py, interference_falsifier.py, eval_shared_substrate.py, and
# go_explore_solve.py can each step from an unrelated --out. Real
# end-to-end + revert-verify coverage for eval_shared_substrate.run()
# lives in tests/test_eval_shared_substrate.py (cheap: no ROM needed,
# full cfg redirection). The checks below are the remaining piece: every
# wired script names the SAME resource string (a typo in any one of them
# would silently defeat the cross-script guard, undetectably from any
# single script's own tests) and actually calls acquire_resource at its
# real call site.
# ---------------------------------------------------------------------------

import scripts.collect_substrate_pairs as collect_substrate_pairs  # noqa: E402
import scripts.eval_shared_substrate as eval_shared_substrate  # noqa: E402
import scripts.interference_falsifier as interference_falsifier  # noqa: E402
import scripts.go_explore_solve as go_explore_solve  # noqa: E402

POOL_RESOURCE_WRITERS = [
    ("collect_substrate_pairs", collect_substrate_pairs, "emulator-pool"),
    ("interference_falsifier", interference_falsifier, "emulator-pool"),
    ("eval_shared_substrate", eval_shared_substrate, "emulator-pool"),
]


@pytest.mark.parametrize("label,mod,expected", POOL_RESOURCE_WRITERS,
                        ids=[w[0] for w in POOL_RESOURCE_WRITERS])
def test_pool_resource_writer_names_the_shared_resource(label, mod, expected):
    """A typo'd resource name (e.g. "emulator_pool" or "emulator-poool")
    would compile, import, and pass every OTHER test in that script's own
    file — it only fails here, where all four names are compared."""
    assert mod.POOL_RESOURCE == expected


def test_go_explore_solve_names_the_shared_resource():
    # go_explore_solve.py uses a leading-underscore module constant (it
    # already had a public `POOL_RESOURCE`-shaped name collision risk
    # with none of its exports, so this one is private like its other
    # `_acquire_run_lock`-style internals).
    assert go_explore_solve._POOL_RESOURCE == "emulator-pool"


def test_collect_substrate_pairs_main_acquires_the_pool_resource_lock():
    import inspect
    src = inspect.getsource(collect_substrate_pairs.main)
    assert "acquire_resource(POOL_RESOURCE" in src
    assert "release_resource(POOL_RESOURCE" in src


def test_interference_falsifier_run_acquires_the_pool_resource_lock():
    import inspect
    src = inspect.getsource(interference_falsifier.run)
    assert "acquire_resource(POOL_RESOURCE" in src
    assert "release_resource(POOL_RESOURCE" in src


def test_go_explore_solve_init_acquires_the_pool_resource_lock():
    import inspect
    src = inspect.getsource(go_explore_solve.Solver.__init__)
    assert "_acquire_pool_resource_lock(_POOL_RESOURCE" in src
    assert "_release_pool_resource_lock(_POOL_RESOURCE" in src
    # Acquired strictly before Pool(...) exists, same discipline as the
    # --out lock it sits beside.
    assert src.index("_acquire_pool_resource_lock(") < src.index("Pool(")


def test_collect_substrate_pairs_docstring_references_the_lock_not_prose():
    """The module docstring used to say, in prose, 'the operator gate is
    upstream — do not run while anything else steps the pool'. That's
    now an enforced call, not an instruction to a human; the docstring
    should point at it instead of repeating the old prose."""
    doc = collect_substrate_pairs.__doc__
    assert "acquire_resource" in doc
    assert "operator gate is upstream" not in doc


def test_solver_init_holds_real_pool_resource_lock_and_refuses_second(
        tmp_path, monkeypatch):
    """Real end-to-end: pre-acquire the resource lock exactly as another
    writer would (a different --out, same shared resource name), then
    construct a Solver against a DIFFERENT --out with a REAL constructible
    profile (configs/1942.yaml — make_game() never touches a ROM, per
    tests/test_solve_profiles_construct.py, so this needs no game asset
    and stays cheap). The path lock above can't be what refuses this —
    --out differs — so a refusal here can only be the pool resource lock,
    caught strictly before Pool(...) would ever open the ROM."""
    from types import SimpleNamespace

    import src.utils.run_lock as run_lock

    monkeypatch.setattr(run_lock, "_RESOURCE_LOCK_DIR", tmp_path / "locks")
    held = run_lock.acquire_resource("emulator-pool", extra="other-writer")
    assert held is None, "precondition: this test must hold the lock first"

    out_dir = tmp_path / "solve_out"
    fake_args = SimpleNamespace(out=str(out_dir), root_state="fake.state",
                               profile="configs/1942.yaml")
    try:
        with pytest.raises(SystemExit) as excinfo:
            go_explore_solve.Solver(fake_args)
        assert "emulator-pool is locked by live PID" in str(excinfo.value)
        assert str(os.getpid()) in str(excinfo.value)
    finally:
        run_lock.release_resource("emulator-pool")
    release(solver_lock_path(out_dir))
