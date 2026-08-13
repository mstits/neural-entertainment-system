"""C1 characterization golden — the CheckpointManager anchor.

Task 0 / test C1 of `docs/proposals/trainer_decomposition_plan.md`: the
pre-extraction safety net for the checkpoint seam that Task 1 will move into a
`CheckpointManager`. It pins the resume + save CONTRACT of
`Trainer._maybe_resume_vanilla_ppo` (trainer.py ~4578) and the iter-checkpoint
save block (the `vanilla_ppo_iter_{global_it:05d}.pt` writer + the `_all_finite`
poison guard, ~9448-9563). A pure structural move must keep every assertion here
green.

This file EXTENDS — it does not duplicate — the two checkpoint round-trip tests
already in `test_vanilla_ppo_characterization.py`
(`test_checkpoint_payload_roundtrips_net_and_offset` pins net weights + optimizer
offset; `test_checkpoint_roundtrips_rnd_state_when_present` pins the RND blob). It
reuses that file's `_bare_tile_trainer` shell (constructor injection, no ROM /
pool / emulator) and covers the three round-trip surfaces those two tests do NOT:

  * `_gx_counts`         — the Go-Explore visitation table, incl. the resume
                            path's `int(k): int(v)` key/value coercion.
  * `anticollapse`       — the anti-collapse rollback blob (best-net snapshot +
                            snapshot fitness + collapse-strike count) round-tripped
                            into `_pending_anticollapse` and re-extractable by the
                            loop's consume site (trainer.py ~5056).
  * poison guard         — `_all_finite`'s actual truth table, and the end-to-end
                            REFUSAL contract: a NaN-bearing net is refused at save
                            time so the real resume path falls back to the last
                            good checkpoint (the resume loader itself does NOT
                            validate — the guard is the sole protection).

Determinism basis (per the plan's §3): tile mode on `device=cpu` with a fixed
seed. These are exact-identity round-trips and boolean-logic pins generated from
the REAL code in-test (the save-block payload is mirrored; the LOAD path exercised
is the real `_maybe_resume_vanilla_ppo`), so no external fixture is recorded — the
round-trip identity and the source-anchored truth table ARE the golden. That
matches the fixture-free philosophy of the two round-trip tests this file extends;
the value-level fixture lives with the C0 master golden.

Explicitly NOT covered (documented honestly so nobody over-trusts this net):
  * The atomic tmp+fsync+os.replace write mechanics (9550-9560) and the winner-
    retention `save_winner` branch (9565+) — those are separate surfaces the plan
    scopes to Task 1's own move-verification, not the resume/round-trip contract.
  * The PR-MDP adversary and backward-curriculum resume blobs — orthogonal
    subsystems keyed on their own attributes, out of this anchor's scope.
  * The optimizer-moment half of the poison guard (9469-9476) is anchored to the
    source text but not exercised end-to-end (it is an inline loop, not
    `_all_finite`); the net/RND half — which uses `_all_finite` — is what C1 pins.
  * The save block is inline in `_run_vanilla_ppo`, not a callable, so the SAVE
    side is mirrored (payload dict + poison-guard predicate) and anchored to the
    trainer source; only the LOAD side runs real code. If either mirror drifts
    from the source the anchoring assertions trip.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Reuse the sibling characterization file's ROM-free trainer shell rather than
# re-declaring it (anti-duplication, per the C1 brief). `_bare_tile_trainer`
# wires exactly the attributes the net build + resume path touch, incl.
# `_gx_counts = {}` and `_pending_anticollapse = None`, which these tests need.
from tests.test_vanilla_ppo_characterization import _bare_tile_trainer

ROOT = Path(__file__).resolve().parent.parent
_TRAINER_SRC = (ROOT / "src" / "training" / "trainer.py").read_text()
# Task 1 of the trainer-decomposition plan relocated the resume scan and the
# iter-save block into `CheckpointManager`; the source-anchored assertions
# below now read that module (the behavioral round-trip tests are unchanged and
# still exercise the real code through `_maybe_resume_vanilla_ppo`).
_CKPT_SRC = (ROOT / "src" / "training" / "checkpoint_manager.py").read_text()


def _seed_all(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _perturbed_net(trainer):
    """A real tile net nudged away from init so weight equality is meaningful."""
    net = trainer._make_network()
    net.to("cpu")
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return net


def _cpu_state(net) -> dict:
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def _write_ckpt(tmp: Path, iter_n: int, payload: dict) -> Path:
    """Mirror the save block's file naming and write the given payload."""
    path = tmp / f"vanilla_ppo_iter_{iter_n:05d}.pt"
    torch.save(payload, str(path))
    return path


def _fresh_shell_resume(tmp: Path):
    """A fresh trainer shell + net + optimizer, resumed through the REAL path.

    Returns (trainer, net, iter_offset)."""
    t = _bare_tile_trainer(tmp)
    net = t._make_network()
    net.to("cpu")
    opt = t._build_ppo_optimizer(net)
    iter_offset = t._maybe_resume_vanilla_ppo(net, opt, fresh_start=False)
    return t, net, iter_offset


# ---------------------------------------------------------------------------
# `_gx_counts` — the Go-Explore visitation table round-trip.
# ---------------------------------------------------------------------------
def test_gx_counts_roundtrips_in_real_int_format() -> None:
    """Pin the FAITHFUL round-trip: the real save writes `self._gx_counts` as a
    `dict[int, int]` (trainer.py 1172 / 6649-6650), and resume restores it equal
    so a bounce doesn't re-inflate already-trodden buckets' bonuses.

    (The resume loader runs `torch.load` at its default `weights_only=True`, so
    the on-disk table must be builtin-typed — numpy-scalar entries would make the
    whole checkpoint fail to load and get quarantined. The real int table is
    exactly what loads.)"""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="c1_gx_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = _perturbed_net(src)
        opt = src._build_ppo_optimizer(net)

        gx_saved = {12345: 7, 0: 1, 2 ** 20 + 3: 5}
        _write_ckpt(
            tmp,
            30,
            {
                "iter": 30,
                "net_state_dict": _cpu_state(net),
                "optimizer_state_dict": opt.state_dict(),
                "gx_counts": gx_saved,
            },
        )

        t, _resumed_net, iter_offset = _fresh_shell_resume(tmp)

        assert iter_offset == 30
        assert t._gx_counts == {12345: 7, 0: 1, 2 ** 20 + 3: 5}
        for k, v in t._gx_counts.items():
            assert type(k) is int and type(v) is int


def test_gx_counts_resume_coerces_keys_and_values_to_int() -> None:
    """Pin the resume path's defensive `{int(k): int(v) ...}` normalization
    (trainer.py 4688-4690). Probed with a string-keyed/valued table — the natural
    non-int form a table would take if it had been JSON-round-tripped, and
    weights-only-safe unlike numpy scalars — so removing the `int()` coercion
    trips the equality (string keys `!=` int keys)."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="c1_gxc_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = _perturbed_net(src)
        opt = src._build_ppo_optimizer(net)

        _write_ckpt(
            tmp,
            31,
            {
                "iter": 31,
                "net_state_dict": _cpu_state(net),
                "optimizer_state_dict": opt.state_dict(),
                "gx_counts": {"12345": "7", "0": "1"},
            },
        )

        t, _resumed_net, iter_offset = _fresh_shell_resume(tmp)

        assert iter_offset == 31
        assert t._gx_counts == {12345: 7, 0: 1}, (
            "resume must coerce string keys/values to int (the int() normalization)"
        )
        for k, v in t._gx_counts.items():
            assert type(k) is int, f"gx key {k!r} not coerced to builtin int"
            assert type(v) is int, f"gx value {v!r} not coerced to builtin int"


def test_gx_counts_save_gate_and_resume_coercion_present_in_source() -> None:
    """Anchor the two halves of the gx contract in the trainer source so the
    mirrored save + the exercised resume can't silently drift: the save side is
    gated on a truthy (non-empty) table, and the resume side re-keys to int."""
    assert 'if self.trainer._gx_counts:' in _CKPT_SRC, (
        "the gx_counts save gate (`if self.trainer._gx_counts:`) changed or "
        "vanished."
    )
    assert '_ckpt_payload["gx_counts"] = self.trainer._gx_counts' in _CKPT_SRC
    assert 'int(k): int(v) for k, v in state["gx_counts"].items()' in _CKPT_SRC, (
        "the resume-side gx_counts int-coercion changed — update this test "
        "deliberately."
    )


# ---------------------------------------------------------------------------
# `anticollapse` — the rollback-baseline blob round-trip.
# ---------------------------------------------------------------------------
def test_anticollapse_blob_roundtrips_into_pending_and_is_reextractable() -> None:
    """Pin: the anti-collapse blob (best-net snapshot + snapshot fitness +
    collapse-strike count) round-trips into `_pending_anticollapse`
    bit-identically, and the loop's consume site (trainer.py ~5056-5062) can
    re-extract the exact three fields — so a post-restart entropy melt can still
    roll back to the pre-restart healthy snapshot instead of measuring collapse
    against -inf."""
    _seed_all()
    with tempfile.TemporaryDirectory(prefix="c1_ac_") as _tmp:
        tmp = Path(_tmp)
        src = _bare_tile_trainer(tmp)
        net = _perturbed_net(src)
        opt = src._build_ppo_optimizer(net)

        # Blob shaped exactly as the save block emits (9537-9541): the snapshot
        # is a cpu-cloned state_dict, fitness a float, strikes an int.
        snap = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        blob = {
            "best_net_snapshot": snap,
            "best_snapshot_fitness": 123.456789,
            "collapse_strikes": 3,
        }
        _write_ckpt(
            tmp,
            40,
            {
                "iter": 40,
                "net_state_dict": _cpu_state(net),
                "optimizer_state_dict": opt.state_dict(),
                "anticollapse": blob,
            },
        )

        t, _resumed_net, iter_offset = _fresh_shell_resume(tmp)

        assert iter_offset == 40
        ac = t._pending_anticollapse
        assert ac is not None, "anti-collapse blob was not staged on resume"
        assert set(ac.keys()) == {
            "best_net_snapshot",
            "best_snapshot_fitness",
            "collapse_strikes",
        }
        # Snapshot tensors bit-identical.
        assert set(ac["best_net_snapshot"].keys()) == set(snap.keys())
        for k in snap:
            assert torch.equal(ac["best_net_snapshot"][k], snap[k]), (
                f"anti-collapse snapshot param {k} did not round-trip identically"
            )
        # Scalars exact.
        assert ac["best_snapshot_fitness"] == 123.456789
        assert ac["collapse_strikes"] == 3

        # The loop's consume site (mirror of trainer.py 5056-5062) must recover
        # the same three fields the resume staged.
        _ac = getattr(t, "_pending_anticollapse", None)
        assert _ac
        best_net_snapshot = _ac.get("best_net_snapshot") or None
        best_snapshot_fitness = float(_ac.get("best_snapshot_fitness", float("-inf")))
        collapse_strikes = int(_ac.get("collapse_strikes", 0))
        assert best_net_snapshot is not None
        for k in snap:
            assert torch.equal(best_net_snapshot[k], snap[k])
        assert best_snapshot_fitness == pytest.approx(123.456789, rel=0, abs=1e-6)
        assert collapse_strikes == 3


def test_anticollapse_save_and_consume_sites_present_in_source() -> None:
    """Anchor both ends of the anti-collapse contract so the mirrors above can't
    drift from the real save/resume/consume code."""
    # Save side (CheckpointManager.save_iter).
    assert 'if best_net_snapshot is not None:' in _CKPT_SRC
    assert '_ckpt_payload["anticollapse"] = {' in _CKPT_SRC
    assert '"best_net_snapshot": best_net_snapshot,' in _CKPT_SRC
    assert '"best_snapshot_fitness": float(best_snapshot_fitness),' in _CKPT_SRC
    assert '"collapse_strikes": int(collapse_strikes),' in _CKPT_SRC
    # Resume side (CheckpointManager.resume).
    assert 'self.trainer._pending_anticollapse = state["anticollapse"]' in _CKPT_SRC
    # Consume side (still in `_run_vanilla_ppo`).
    assert '_ac = getattr(self, "_pending_anticollapse", None)' in _TRAINER_SRC
    assert 'best_net_snapshot = _ac.get("best_net_snapshot") or None' in _TRAINER_SRC


# ---------------------------------------------------------------------------
# Poison guard — `_all_finite`'s truth table + the end-to-end refusal contract.
# ---------------------------------------------------------------------------
def _all_finite(sd) -> bool:
    """Mirror of the nested `_all_finite` in `_run_vanilla_ppo` (trainer.py
    ~9459-9464). Anchored to source by
    `test_poison_guard_source_matches_mirror`."""
    return all(
        torch.isfinite(v).all()
        for v in sd.values()
        if torch.is_tensor(v) and v.is_floating_point()
    )


def test_all_finite_truth_table() -> None:
    """Pin `_all_finite`'s ACTUAL behavior (the poison-guard predicate):

      * an all-finite state_dict is finite;
      * a NaN or an Inf in ANY floating-point tensor makes it non-finite;
      * NON-tensor values (e.g. an int step counter) are IGNORED;
      * NON-floating-point tensors (e.g. int64) are IGNORED — the guard only
        polices float params, so an int tensor can never flip the verdict;
      * an empty state_dict is finite (`all([]) is True`)."""
    finite = {"w": torch.zeros(3), "b": torch.ones(2)}
    assert _all_finite(finite) is True

    nan_sd = {"w": torch.zeros(3), "b": torch.tensor([0.0, float("nan")])}
    assert _all_finite(nan_sd) is False, "a NaN float param must read non-finite"

    inf_sd = {"w": torch.tensor([float("inf"), 0.0]), "b": torch.ones(2)}
    assert _all_finite(inf_sd) is False, "an Inf float param must read non-finite"

    # Non-tensor values are skipped by `torch.is_tensor(v)`.
    with_nontensor = {"w": torch.zeros(3), "step": 42, "name": "policy"}
    assert _all_finite(with_nontensor) is True

    # Non-float tensors are skipped by `v.is_floating_point()`; an int tensor
    # cannot hold NaN, and the guard does not inspect it regardless.
    with_int_tensor = {"w": torch.zeros(3), "n": torch.tensor([1, 2, 3])}
    assert _all_finite(with_int_tensor) is True

    assert _all_finite({}) is True, "an empty state_dict is vacuously finite"


def test_poison_guard_source_matches_mirror() -> None:
    """Anchor the real `_all_finite` definition + the refusal gate so the mirror
    above and the refusal contract below cannot silently drift from the code."""
    assert 'def _all_finite(sd) -> bool:' in _CKPT_SRC
    assert 'torch.isfinite(v).all()' in _CKPT_SRC
    assert 'if torch.is_tensor(v) and v.is_floating_point()' in _CKPT_SRC
    # Net + RND both routed through _all_finite.
    assert '_params_finite = _all_finite(net.state_dict())' in _CKPT_SRC
    assert '_params_finite = _all_finite(self.trainer._rnd.state_dict())' in _CKPT_SRC
    # The optimizer-moment half (a separate inline loop, not _all_finite).
    assert 'for _grp in optimizer.state.values():' in _CKPT_SRC
    # The save is gated on the finiteness verdict, and refusal is logged.
    assert 'if it > 0 and it % 10 == 0 and _params_finite:' in _CKPT_SRC
    assert 'REFUSING checkpoint save at iter ' in _CKPT_SRC


def test_poison_guard_refusal_makes_resume_fall_back_to_last_good() -> None:
    """End-to-end refusal contract through the REAL resume path.

    Part A (danger): the resume loader does NOT itself validate — if a NaN-net
    checkpoint is the newest file, `_maybe_resume_vanilla_ppo` loads it and
    hands back a poisoned net. This is WHY the save-side guard must refuse.

    Part B (protection): the poison guard refuses to write the NaN checkpoint
    (`_all_finite(nan_net) is False` -> the iter-N file is never created), so the
    real resume falls back to the last GOOD checkpoint — finite weights, the good
    iter offset."""
    _seed_all()

    good_iter, poison_iter = 10, 20

    def _good_and_poison(tmp: Path):
        shell = _bare_tile_trainer(tmp)
        good_net = _perturbed_net(shell)
        good_opt = shell._build_ppo_optimizer(good_net)
        good_payload = {
            "iter": good_iter,
            "net_state_dict": _cpu_state(good_net),
            "optimizer_state_dict": good_opt.state_dict(),
        }
        # A numerically-collapsed net (every param NaN).
        poison_net = _perturbed_net(shell)
        with torch.no_grad():
            for p in poison_net.parameters():
                p.mul_(float("nan"))
        return good_net, good_payload, poison_net

    # --- Part A: no guard -> newest (poison) checkpoint IS loaded, unvalidated.
    with tempfile.TemporaryDirectory(prefix="c1_poison_a_") as _tmp:
        tmp = Path(_tmp)
        good_net, good_payload, poison_net = _good_and_poison(tmp)
        _write_ckpt(tmp, good_iter, good_payload)
        # Guard bypassed: the poison checkpoint reaches disk as the newest file.
        _write_ckpt(
            tmp,
            poison_iter,
            {
                "iter": poison_iter,
                "net_state_dict": _cpu_state(poison_net),
                "optimizer_state_dict": _bare_tile_trainer(tmp)
                ._build_ppo_optimizer(poison_net)
                .state_dict(),
            },
        )
        _t, resumed_net, iter_offset = _fresh_shell_resume(tmp)
        assert iter_offset == poison_iter, "resume takes the newest checkpoint"
        assert not _all_finite(resumed_net.state_dict()), (
            "resume loads the poison net verbatim — the loader does not validate"
        )

    # --- Part B: guard active -> poison save refused -> resume uses last good.
    with tempfile.TemporaryDirectory(prefix="c1_poison_b_") as _tmp:
        tmp = Path(_tmp)
        good_net, good_payload, poison_net = _good_and_poison(tmp)
        _write_ckpt(tmp, good_iter, good_payload)

        # The real guard's predicate on the NaN net (net half of 9465-9467).
        params_finite = _all_finite(poison_net.state_dict())
        assert params_finite is False, "the NaN net must read as non-finite"
        if params_finite:  # mirror: save only when finite (trainer.py 9484)
            _write_ckpt(
                tmp,
                poison_iter,
                {
                    "iter": poison_iter,
                    "net_state_dict": _cpu_state(poison_net),
                    "optimizer_state_dict": good_payload["optimizer_state_dict"],
                },
            )
        # Refusal means the iter-20 file was never written.
        assert not (tmp / f"vanilla_ppo_iter_{poison_iter:05d}.pt").exists()

        t, resumed_net, iter_offset = _fresh_shell_resume(tmp)
        assert iter_offset == good_iter, "resume rolls back to the last good iter"
        assert t._vppo_resumed_from_iter == good_iter
        assert _all_finite(resumed_net.state_dict()), (
            "the resumed net must be the finite, good checkpoint"
        )
        for k in good_net.state_dict():
            assert torch.equal(
                resumed_net.state_dict()[k], good_net.state_dict()[k].cpu()
            ), f"resumed param {k} is not the good checkpoint's weight"
