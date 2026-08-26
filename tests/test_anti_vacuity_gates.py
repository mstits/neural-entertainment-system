"""Standing guard against the idiom behind four vacuous gates found and
fixed on 2026-08-26 (is_clear's `() > ()`, area()'s literal-0 default,
the D6 camera-static override, and a progress column certified sound by
`passed = not findings` on a column that could never hold anything).
The common thread, stated in the postmortem: nothing was checking what
these would report if the mechanism were absent.

scripts/anti_vacuity_scan.py finds every place in scripts/ and src/ that
computes a verdict as the bare absence of a findings-shaped collection
(`passed = not findings`, `"ok": len(errors) == 0`, ...). That shape is
not automatically wrong -- assess()'s `"passed": not instrument` below is
exactly this idiom and is legitimate, because `instrument` is populated
only by real, independently-measured checks. What makes a site SAFE is
a demonstrated ability to report both a pass and a fail; nothing else
about the syntax says so.

This file is the other half: an explicit, hand-reviewed registry of
every site the scanner currently finds, each paired with a positive and
a negative case built directly against the real function (no ROM, no
Pool -- every fixture is a synthetic input engineered to land on one
specific measured branch). Two tests enforce it:

  * a NEW site the scanner finds that is not in REGISTRY fails loudly,
    by name, with the fix ("add a negative-control case and register
    it") spelled out -- so a future idiom instance cannot land silently;
  * every REGISTERED site is re-proven, on every run, to still return
    both polarities -- so a refactor that quietly makes one of these
    gates unable to fail (the exact D6 shape: a branch that used to set
    the fail case gets deleted) is caught here even if every other test
    file still passes.

Deliberately narrow. It does not try to detect vacuity by itself --
only the AST shape from the postmortem, only in scripts/+src/, only the
five-word verdict-name list anti_vacuity_scan.py documents. A checker
that guessed at semantics would need constant suppressions; this one
either finds the exact shape or it doesn't.
"""
from __future__ import annotations

import numpy as np

from scripts.anti_vacuity_scan import scan_repo


# ===========================================================================
# Self-contained positive/negative fixtures, one pair per registered site.
# Each calls the real function under test directly; none of these import
# fixtures from another test file, so this file's coverage does not rot
# if some other test's helpers are renamed or restructured.
# ===========================================================================

# --- scripts/clear_detect.py: input_lock_preflight -------------------------
#
# Minimal duck-typed envs matching the exact surface _run_lock_branches
# and measure_input_lock_null call (save_state/load_state/step/
# get_ram_range/peek_oam/get_audio) -- see tests/test_input_lock_signal.py
# for the fuller version this is a deliberately small cousin of.

class _LockFakeEnv:
    RAM_SIZE = 2048
    OAM_SIZE = 256

    def __init__(self, rule):
        self._rule = rule
        self._state = {"t": 0}

    def save_state(self):
        return dict(self._state)

    def load_state(self, blob):
        self._state = dict(blob)

    def step(self, mask):
        self._state = self._rule(self._state, int(mask))

    def get_audio(self):
        return np.zeros(0, dtype=np.int16)

    def get_ram_range(self, start, length):
        return self._render()[0]

    def peek_oam(self):
        return bytes(self._render()[1])

    def _render(self):
        raise NotImplementedError


class _ResponsiveLockEnv(_LockFakeEnv):
    """RAM genuinely tracks which input each branch held: an unlocked,
    ordinary-play surface."""

    def _render(self):
        pos = self._state.get("pos", 0) & 0xFF
        ram = np.zeros(self.RAM_SIZE, dtype=np.uint8)
        ram[0] = pos
        oam = np.zeros(self.OAM_SIZE, dtype=np.uint8)
        oam[0] = pos
        return ram, oam


class _FrozenLockEnv(_LockFakeEnv):
    """RAM/OAM evolve identically no matter what input each branch held --
    input has zero causal effect, the mechanism every LOCKED reading
    (pause, cutscene, death animation) shares."""

    def _render(self):
        c = self._state.get("c", 0) & 0xFF
        ram = np.zeros(self.RAM_SIZE, dtype=np.uint8)
        ram[0] = c
        oam = np.zeros(self.OAM_SIZE, dtype=np.uint8)
        return ram, oam


def _input_lock_preflight_case(env_cls, rule, seed):
    from scripts.clear_detect import InputLockSignal, input_lock_preflight

    bitmasks = (0x00, 0x01, 0x02, 0x04, 0x08)
    rng = np.random.default_rng(seed)
    sig = InputLockSignal(bitmasks, probe_frames=10, branches=4,
                          quantile=0.01, lock_frac=0.6, rng=rng)
    env = env_cls(rule)
    return input_lock_preflight(sig, env, settle_steps=30)


def _input_lock_preflight_positive():
    return _input_lock_preflight_case(
        _ResponsiveLockEnv,
        lambda s, mask: {"pos": s.get("pos", 0) + mask + 1}, seed=1001)


def _input_lock_preflight_negative():
    return _input_lock_preflight_case(
        _FrozenLockEnv, lambda s, mask: {"c": s.get("c", 0) + 1}, seed=1002)


# --- scripts/night2_runner.py: gate_verdict --------------------------------

_NIGHT2_CFG = {"gate_honest_median_baseline": 2059.0,
              "gate_det_median_baseline": 2979.0}


def _night2_summ(median: float, n: int = 10, status: str = "ok") -> dict:
    return {"status": status, "n_episodes": n, "median_max_x": median}


def _gate_verdict_positive():
    from scripts.night2_runner import gate_verdict
    return gate_verdict(_night2_summ(2200.0), _night2_summ(3100.0),
                        cfg=_NIGHT2_CFG)


def _gate_verdict_negative():
    from scripts.night2_runner import gate_verdict
    # Exactly AT the honest baseline: pre-registered as strictly greater,
    # so equal must fail.
    return gate_verdict(_night2_summ(2059.0), _night2_summ(3100.0),
                        cfg=_NIGHT2_CFG)


# --- scripts/progress_signal_gate.py: assess -------------------------------

def _assess_trace():
    return list(range(788))


def _assess_positive():
    from scripts.progress_signal_gate import assess
    return assess(_assess_trace(), None, True, raw_direction=787)


def _assess_negative():
    from scripts.progress_signal_gate import assess
    # Same rebased trace, only the raw (un-rebased) direction flips --
    # the 1942 false-PASS this function exists to reject.
    return assess(_assess_trace(), None, True, raw_direction=-787)


# --- scripts/discover_observables.py: behavioural_lives_verdict -----------

_LIVES_RAM = 64
_LIVES_ADDR = 0x0A
_LIVES_N = 320


def _lives_col(level: int = 2, n: int = _LIVES_N):
    return np.full(n, level, dtype=np.uint8)


def _lives_toggle(level: int = 2, first: int = 3, period: int = 15,
                  width: int = 2, n: int = _LIVES_N):
    """Falls to 0 and recovers to `level` every `period` steps -- an
    attack/animation counter, not a stock (Bad Dudes $00CD's own shape)."""
    col = np.full(n, level, dtype=np.uint8)
    for t in range(first, n, period):
        col[t:min(t + width, n)] = 0
    return col


def _lives_arm(lives_col, n: int = _LIVES_N) -> dict:
    log = np.zeros((n, _LIVES_RAM), dtype=np.uint8)
    log[:, _LIVES_ADDR] = lives_col
    return {"log": log, "steps": n,
            "odo": np.zeros((n, 2), dtype=np.int64),
            "scene": np.zeros(n, dtype=np.int64), "reset_threshold": 350.0}


def _behavioural_lives_verdict_positive():
    from scripts.discover_observables import behavioural_lives_verdict
    # A byte that never touches zero and never oscillates in any arm:
    # none of the three measured failure modes can fire.
    probe = {"arms": {"hold": _lives_arm(_lives_col()),
                      "idle": _lives_arm(_lives_col()),
                      "mash": _lives_arm(_lives_col())},
             "odometer": True, "steps": _LIVES_N}
    return behavioural_lives_verdict({"addr": _LIVES_ADDR}, probe)


def _behavioural_lives_verdict_negative():
    from scripts.discover_observables import behavioural_lives_verdict
    # The mash arm oscillates well past OSC_MAX_CYCLES -- an animation
    # counter, rejected on the oscillation check alone.
    probe = {"arms": {"hold": _lives_arm(_lives_col()),
                      "idle": _lives_arm(_lives_col()),
                      "mash": _lives_arm(_lives_toggle())},
             "odometer": True, "steps": _LIVES_N}
    return behavioural_lives_verdict({"addr": _LIVES_ADDR}, probe)


# ===========================================================================
# The registry itself: (file, qualname) -> (verdict key, positive, negative)
# `file`/`func` must match anti_vacuity_scan.VacuityHit exactly.
# ===========================================================================

REGISTRY = {
    ("scripts/clear_detect.py", "input_lock_preflight"): (
        "ok", _input_lock_preflight_positive, _input_lock_preflight_negative),
    ("scripts/night2_runner.py", "gate_verdict"): (
        "passed", _gate_verdict_positive, _gate_verdict_negative),
    ("scripts/progress_signal_gate.py", "assess"): (
        "passed", _assess_positive, _assess_negative),
    ("scripts/discover_observables.py", "behavioural_lives_verdict"): (
        "passed", _behavioural_lives_verdict_positive,
        _behavioural_lives_verdict_negative),
}


def test_no_new_vacuous_shaped_gates_without_a_registered_proof():
    """Every `passed = not <x>`-shaped site in scripts/+src/ must be a key
    in REGISTRY. This is the drift catch: add a new one of these without
    registering a negative-control proof, and this fails by name, with
    the two-line fix spelled out, rather than banking a fifth vacuous
    gate the way the four in the postmortem got banked."""
    hits = scan_repo()
    unregistered = [h for h in hits if (h.file, h.func) not in REGISTRY]
    assert not unregistered, (
        "New vacuous-shaped gate(s) found (verdict computed as the bare "
        "absence of a findings-collection) with no registered proof they "
        "can report both a pass and a fail:\n" +
        "\n".join(
            f"  {h.file}:{h.line} func={h.func} "
            f'"{h.target}" = {h.shape.replace("<x>", h.source)}'
            for h in unregistered) +
        "\n\nFix: add a positive- and negative-control case for the "
        "function above to tests/test_anti_vacuity_gates.py and register "
        "it in REGISTRY, proving the gate can actually report a failure "
        "-- not just that nothing has objected yet.")


def test_registered_gates_demonstrate_both_polarities():
    """Every REGISTRY entry is re-proven on every run, not just recorded
    once: a refactor that quietly removes the one branch capable of
    failing a gate (the exact D6 shape) is caught here even if no other
    test file happens to exercise that branch anymore."""
    failures = []
    for (file, func), (key, positive, negative) in REGISTRY.items():
        pos_result = positive()
        neg_result = negative()
        if not pos_result.get(key):
            failures.append(
                f"{file}:{func} — positive case did not yield a truthy "
                f"{key!r} ({pos_result.get(key)!r}); the demonstrated-pass "
                f"case for this gate is broken")
        if neg_result.get(key):
            failures.append(
                f"{file}:{func} — negative case did NOT yield a falsy "
                f"{key!r} ({neg_result.get(key)!r}); this gate can no "
                f"longer be observed to fail, which is exactly the D6/"
                f"is_clear/area() shape this file exists to catch")
    assert not failures, "\n".join(failures)


def test_registry_has_no_duplicate_or_orphaned_entries():
    """Cheap self-check on the registry's own shape: every key must be a
    2-tuple of non-empty strings, and every value a 3-tuple whose two
    callables are actually callable. Catches a copy-paste registry entry
    before it ever reaches the two tests above."""
    for key, value in REGISTRY.items():
        assert isinstance(key, tuple) and len(key) == 2
        file, func = key
        assert file and func
        assert isinstance(value, tuple) and len(value) == 3
        verdict_key, positive, negative = value
        assert isinstance(verdict_key, str) and verdict_key
        assert callable(positive) and callable(negative)
        assert positive is not negative, (
            f"{key}: positive and negative cases must be different calls")
