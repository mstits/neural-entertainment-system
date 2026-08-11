"""B6 — the consolidation gate's probe distribution + accept arithmetic.

Every test here is anchored to a banked receipt, not to a preference:

  * `checkpoints/mario_1_1_backward_v4/run.log` — fifteen of sixteen gate
    probes on target 1-1 read 0.000 and one read 1.000, because
    `_clevel_probe_level` / `_gate_probe` forwarded none of the perturbation
    settings, leaving every probe episode the same deterministic replay and
    the accept resting on a single sample.
  * `checkpoints/mario_1_1_backward_v4/consolidate_1-1.json` — the run closed
    at `{"best_tgt_rate": 1.0, "target_rate": 0.0, "sustain": 0, "iter": 260}`.
    Nothing strictly improves on a best of 1.0, so nothing could be accepted
    for 144 iterations. That is the ratchet.
  * `checkpoints/mario_1_1_backward_v4/eval.jsonl` — rows carrying a
    sticky-0.25 + jitter-16 clear rate with no record of the protocol.

Four layers:
  1. the probe-settings resolver (pure) — defaults are v4-inert, the B6 keys
     resolve to an honest distribution;
  2. `cold_probe` command-line threading (mocked subprocess) — the flags
     actually reach `eval_game.py`, and are absent at the defaults;
  3. the trainer's two call sites — both thread the resolved settings (this
     is the test that fails against the pre-B6 source);
  4. the Wilson accept arithmetic and the ratchet break.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

import src.training.cold_probe as cold_probe
from src.training import oneshot_curriculum as oc

_ROOT = Path(__file__).resolve().parent.parent
_TRAINER_SRC = (_ROOT / "src" / "training" / "trainer.py").read_text()
_V6 = _ROOT / "configs" / "mario_1_1_backward_v6.yaml"
_V4 = _ROOT / "configs" / "mario_1_1_backward_v4.yaml"


def _v6_probe_cfg() -> dict:
    prof = yaml.safe_load(_V6.read_text())
    return prof["reinforce"]["consolidate_level"]["probe"]


# ==========================================================================
# 1. The probe-settings resolver (pure)
# ==========================================================================

def test_absent_keys_resolve_to_the_v4_probe_exactly():
    """The regression gate: with the B6 keys absent the probe is the
    deterministic single-replay probe v4 ran, and `probe_kwargs` adds nothing
    that would change eval_game's command line."""
    s = oc.resolve_probe_settings({"every": 12, "episodes": 1,
                                   "max_steps": 1500})
    assert (s.sticky_prob, s.start_jitter, s.eval_seed) == (0.0, 0, None)
    assert s.eval_workers == 1 and s.eval_rng == "shared-stream"
    assert s.stochastic is False
    # ...but the EFFECTIVE seed is not None: an omitted flag lands on
    # eval_game's own default, which is what the receipt must record.
    assert s.effective_eval_seed == oc.EVAL_GAME_DEFAULT_SEED == 0
    assert s.probe_kwargs() == {
        "episodes": 1, "max_steps": 1500, "sticky_prob": 0.0,
        "start_jitter": 0, "eval_seed": None, "eval_workers": 1,
        "eval_rng": "shared-stream",
    }


def test_empty_and_none_probe_blocks_resolve_to_the_shipped_defaults():
    for cfg in (None, {}):
        s = oc.resolve_probe_settings(cfg)
        assert (s.every, s.episodes, s.max_steps) == (25, 8, 1500)
        assert s.stochastic is False


def test_v6_keys_resolve_to_an_honest_distribution():
    s = oc.resolve_probe_settings(_v6_probe_cfg())
    assert s.episodes == 30
    assert s.sticky_prob == pytest.approx(0.25)
    assert s.start_jitter == 16
    assert s.eval_seed == 7
    assert s.eval_workers == 8
    assert s.stochastic is True


def test_parallel_stochastic_probe_forces_the_per_episode_rng():
    """eval_game refuses >1 worker on the shared stream for a stochastic run
    (episode i's draws depend on how many draws episodes 0..i-1 consumed), and
    a refused probe is a dead gate — so the resolver picks the only mode that
    works instead of letting the run discover it at the first probe."""
    s = oc.resolve_probe_settings(
        {"episodes": 30, "sticky_prob": 0.25, "eval_workers": 8}
    )
    assert s.eval_rng == "per-episode"
    # ...but a DETERMINISTIC parallel probe is reproducible either way, so the
    # default is left alone.
    d = oc.resolve_probe_settings({"episodes": 30, "eval_workers": 8})
    assert d.eval_rng == "shared-stream"


def test_selection_seed_must_not_be_a_reporting_seed():
    """The gate selects; the deliverable is reported on
    REPORTING_EVAL_SEEDS. Sharing the stream lets the gate pick the network
    that happens to win on the very seed it will be graded on."""
    honest = {"sticky_prob": 0.25, "start_jitter": 16}
    assert oc.REPORTING_EVAL_SEEDS == (0, 1)
    assert oc.resolve_probe_settings(
        {**honest, "eval_seed": 7}).seed_collision is False
    for bad in oc.REPORTING_EVAL_SEEDS:
        assert oc.resolve_probe_settings(
            {**honest, "eval_seed": bad}).seed_collision
    # The v6 profile is configured correctly.
    assert oc.resolve_probe_settings(_v6_probe_cfg()).seed_collision is False


def test_an_omitted_selection_seed_is_a_reporting_seed():
    """THE HOLE A "configured seed" CHECK LEAVES OPEN. Arming sticky + jitter
    and simply not writing `eval_seed` is the easiest way to configure this
    block, and it is not neutral: `cold_probe.probe` omits the flag, eval_game
    falls back to --eval-seed 0, and 0 is REPORTING_EVAL_SEEDS[0]. The gate
    would then select on the exact stream the deliverable is published on,
    with no key anywhere in the profile that looks wrong."""
    s = oc.resolve_probe_settings({"episodes": 30, "sticky_prob": 0.25,
                                   "start_jitter": 16})
    assert s.eval_seed is None                       # nothing configured...
    assert s.effective_eval_seed == 0                # ...but 0 is what runs
    assert s.stochastic is True
    assert s.seed_collision is True


def test_a_deterministic_probe_on_seed_zero_is_not_flagged_as_a_collision():
    """A probe that consumes no randomness draws from no stream, so it cannot
    share one. That keeps the v4-equivalent (deterministic) profile clean —
    its own, louder defect is reported separately — and keeps the guard from
    firing on every legacy consolidation profile in the tree."""
    s = oc.resolve_probe_settings({"episodes": 1, "eval_seed": 0})
    assert s.stochastic is False and s.seed_collision is False


# ==========================================================================
# 2. cold_probe threads the settings onto eval_game's command line
# ==========================================================================

@pytest.fixture
def rom_file(tmp_path) -> str:
    p = tmp_path / "fake.nes"
    p.write_bytes(b"\x00" * 16)
    return str(p)


@pytest.fixture(autouse=True)
def _reset_seq_memo():
    cold_probe._SEQUENTIAL_SUPPORTED = None
    yield
    cold_probe._SEQUENTIAL_SUPPORTED = None


def _probe_argv(rom_file: str, **kwargs) -> list:
    ok = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "game": "m", "status": "ok", "n_episodes": kwargs.get("episodes", 8),
            "mean_return": 0.0, "mean_length": 0.0, "max_byte_seen": 0,
            "clear_rate": 0.0, "seq_clear_rate": 0.5,
        }),
        stderr="",
    )
    net = {"fc1.weight": torch.zeros(2, 2)}
    with mock.patch.object(subprocess, "run", return_value=ok) as m:
        cold_probe.probe(net, {"name": "Super Mario Bros."}, sequential=True,
                         rom_path=rom_file, **kwargs)
    return list(m.call_args[0][0])


def test_default_probe_argv_carries_no_perturbation_flags(rom_file):
    argv = _probe_argv(rom_file, episodes=1, max_steps=1500)
    for flag in ("--sticky-prob", "--start-jitter", "--eval-seed",
                 "--eval-workers", "--eval-rng"):
        assert flag not in argv, f"{flag} leaked into the v4-equivalent probe"


def test_v6_probe_kwargs_reach_eval_game(rom_file):
    """THE REPAIR, end to end through the probe: the honest protocol is on the
    command line, so `eval_game.run_consumes_randomness` is True and the 30
    episodes are 30 draws rather than 30 copies of one replay."""
    kwargs = oc.resolve_probe_settings(_v6_probe_cfg()).probe_kwargs()
    argv = _probe_argv(rom_file, **kwargs)
    assert argv[argv.index("--episodes") + 1] == "30"
    assert argv[argv.index("--sticky-prob") + 1] == "0.25"
    assert argv[argv.index("--start-jitter") + 1] == "16"
    assert argv[argv.index("--eval-seed") + 1] == "7"
    assert argv[argv.index("--eval-workers") + 1] == "8"
    assert argv[argv.index("--eval-rng") + 1] == "per-episode"


# ==========================================================================
# 3. The trainer's two consolidation call sites
# ==========================================================================

def _fn_body(src: str, header: str) -> str:
    """The source of a nested `def` up to the next dedented statement."""
    i = src.index(header)
    indent = len(src[:i].rsplit("\n", 1)[-1])
    out = [src[i:src.index("\n", i)]]
    for line in src[src.index("\n", i) + 1:].split("\n"):
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("header", ["def _clevel_probe_level(_entry):",
                                    "def _gate_probe(_entry):"])
def test_both_consolidation_probes_thread_the_resolved_distribution(header):
    """FAILS AGAINST THE PRE-B6 SOURCE. v4's two probe call sites passed
    `episodes=clevel_eps` and nothing else — no sticky, no jitter, no seed —
    which is why its gate measured a replay. Both must now expand the one
    resolved settings object, so they cannot drift apart from each other or
    from the profile."""
    body = _fn_body(_TRAINER_SRC, header)
    assert "**clevel_probe_kwargs" in body, body
    # The episode count now rides inside the resolved kwargs; a stray explicit
    # `episodes=` would be a duplicate-keyword TypeError at the first probe.
    assert "episodes=" not in body, body
    # The probe must report how many episodes it actually scored, or the
    # Wilson bound has no n.
    assert "cold_n_episodes" in body, body


def test_gate_passes_the_accept_arithmetic_through_to_the_pure_sequencer():
    call = _TRAINER_SRC[_TRAINER_SRC.index("_gate = oc.gate_step("):]
    call = call[:call.index("\n                )")]
    for kw in ("target_n=", "use_wilson_bound=", "wilson_confidence=",
               "accept_rule=", "best_decay="):
        assert kw in call, call


# ==========================================================================
# 4. Wilson arithmetic + the ratchet break
# ==========================================================================

def test_wilson_lower_bound_known_values():
    # Textbook Wilson score interval at 95%.
    assert oc.wilson_lower_bound(15, 30) == pytest.approx(0.33154, abs=5e-5)
    assert oc.wilson_lower_bound(30, 30) == pytest.approx(0.88649, abs=5e-5)
    assert oc.wilson_lower_bound(0, 30) == 0.0
    # THE v4 CASE: one lucky episode is not evidence of a 100% policy.
    assert oc.wilson_lower_bound(1, 1) == pytest.approx(0.20655, abs=5e-5)
    # More evidence for the same rate => a tighter (higher) bound.
    assert oc.wilson_lower_bound(60, 60) > oc.wilson_lower_bound(30, 30)
    # Never outside [0, 1]; an unmeasured probe bounds nothing.
    assert oc.wilson_lower_bound(0, 0) == 0.0
    assert 0.0 <= oc.wilson_lower_bound(29, 30) <= 1.0
    # A tighter confidence gives a higher bound.
    assert oc.wilson_lower_bound(15, 30, confidence=0.80) > \
        oc.wilson_lower_bound(15, 30, confidence=0.99)


def test_wilson_lower_bound_is_below_the_point_estimate():
    for k, n in [(1, 1), (5, 30), (15, 30), (29, 30), (1, 100)]:
        assert oc.wilson_lower_bound(k, n) < k / n + 1e-12


def test_gate_defaults_are_byte_identical_to_the_pre_b6_gate():
    """Every pre-B6 assertion still holds, and the added fields do not change
    action/sustain/done."""
    d = oc.gate_step(regressed=None, target_rate=0.6, best_rate=0.5,
                     sustain=0, bar=0.75, need=3)
    assert d["action"] == "accept" and d["done"] is False
    assert d["target_lb"] is None
    assert d["gate_metric"] == pytest.approx(0.6)
    assert d["best_rate"] == pytest.approx(0.6)      # point estimate stored
    h = oc.gate_step(regressed=None, target_rate=0.5, best_rate=0.5,
                     sustain=0, bar=0.75, need=3)
    assert h["action"] == "hold" and h["best_rate"] == pytest.approx(0.5)
    r = oc.gate_step(regressed="1-2", target_rate=1.0, best_rate=0.5,
                     sustain=2, bar=0.75, need=3)
    assert r["action"] == "rollback" and r["sustain"] == 0
    assert r["best_rate"] == pytest.approx(0.5)      # rollback never re-bases


def test_v4_ratchet_reproduced_under_the_old_rule():
    """The failure, first, so the fix has something to be a fix OF: one
    deterministic probe reads 1.000, best becomes 1.0, and no later probe —
    however good — can ever accept again."""
    best = 0.0
    lucky = oc.gate_step(regressed=None, target_rate=1.0, best_rate=best,
                         sustain=0, bar=1.0, need=4, target_n=1)
    assert lucky["action"] == "accept"
    best = lucky["best_rate"]
    assert best == pytest.approx(1.0)
    for rate in (0.0, 0.5, 0.9, 1.0):
        d = oc.gate_step(regressed=None, target_rate=rate, best_rate=best,
                         sustain=0, bar=1.0, need=4, target_n=1)
        assert d["action"] == "hold", rate     # locked out, exactly as v4 was


def test_wilson_rule_refuses_to_let_one_lucky_probe_pin_the_bar():
    """THE RATCHET BREAK. The same 1/1 probe cannot accept under the B6 rule,
    because its lower bound (0.21) does not clear a best that a 30-episode
    measurement has already established."""
    kw = dict(regressed=None, sustain=0, bar=0.5, need=3,
              use_wilson_bound=True, wilson_confidence=0.95,
              accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    lucky = oc.gate_step(target_rate=1.0, target_n=1, best_rate=0.30, **kw)
    assert lucky["action"] == "hold"
    assert lucky["target_lb"] == pytest.approx(0.20655, abs=5e-5)
    # A real 30-episode improvement does accept, and stores the POINT
    # estimate of that 30-episode measurement as the new bar.
    real = oc.gate_step(target_rate=20 / 30, target_n=30, best_rate=0.30, **kw)
    assert real["action"] == "accept"
    assert real["target_lb"] == pytest.approx(0.48780, abs=5e-5)
    assert real["best_rate"] == pytest.approx(20 / 30)


def test_the_new_bar_stays_reachable_after_an_accept():
    """v4's post-accept bar was arithmetically unreachable. B6's is not: a
    genuinely better 30-episode policy still clears the incumbent's point
    estimate, while a merely-noisy one does not."""
    kw = dict(regressed=None, sustain=0, bar=0.5, need=3, target_n=30,
              use_wilson_bound=True, accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    best = oc.gate_step(target_rate=20 / 30, best_rate=0.0, **kw)["best_rate"]
    assert oc.gate_step(target_rate=22 / 30, best_rate=best,
                        **kw)["action"] == "hold"     # not yet distinguishable
    assert oc.gate_step(target_rate=29 / 30, best_rate=best,
                        **kw)["action"] == "accept"


def test_best_decay_re_estimates_a_stale_incumbent():
    """The second arm of the ratchet break, default-inert: on a non-accepting
    probe the incumbent erodes toward what the policy actually does now, so a
    high-water can never lock the gate out permanently."""
    assert oc.decay_best(1.0, 0.0, 0.0) == pytest.approx(1.0)   # default: inert
    assert oc.decay_best(1.0, 0.0, 0.25) == pytest.approx(0.75)
    assert oc.decay_best(1.0, 0.4, 1.0) == pytest.approx(0.4)
    assert oc.decay_best(1.0, None, 0.9) == pytest.approx(1.0)  # failed probe
    d = oc.gate_step(regressed=None, target_rate=0.0, best_rate=1.0,
                     sustain=0, bar=0.5, need=3, target_n=30, best_decay=0.5)
    assert d["action"] == "hold" and d["best_rate"] == pytest.approx(0.5)


def test_probe_failed_recognises_the_trainers_sentinel_not_just_none():
    """`decay_best(observed=None)` is not the failed-probe guarantee, because
    the trainer never produces None here: `_gate_probe` returns None and the
    caller converts it to -1.0 before the gate sees it. The guard has to
    recognise THAT value, and the zero-episode case."""
    assert oc.probe_failed(None, 30) is True
    assert oc.probe_failed(-1.0, 0) is True
    assert oc.probe_failed(-1.0, 30) is True     # sentinel with a stale n
    assert oc.probe_failed(0.0, 30) is False     # a real 0/30 IS a measurement
    assert oc.probe_failed(0.6, None) is False   # pre-B6 callers omit n


def test_a_failed_probe_never_moves_the_incumbent_under_decay():
    """THE PRODUCTION PATH, not the None convenience. Driven with exactly what
    `_run_vanilla_ppo` passes when an eval subprocess fails: target_rate -1.0
    and target_n 0. Left unguarded, the -1.0 is a measurement of "worse than
    anything possible" and the re-estimation walks the bar off the bottom:
    best 0.60 at decay 0.25 would become 0.20, and 0.5 would give -0.20."""
    for decay in (0.25, 0.5, 1.0):
        d = oc.gate_step(regressed=None, target_rate=-1.0, best_rate=0.60,
                         sustain=2, bar=0.5, need=3, target_n=0,
                         best_decay=decay)
        assert d["action"] == "hold"
        assert d["best_rate"] == pytest.approx(0.60), decay
        assert d["sustain"] == 0 and d["done"] is False
    # Same under the bounded rule, where the failed probe also cannot be
    # bounded and so must not be allowed to accept.
    d = oc.gate_step(regressed=None, target_rate=-1.0, best_rate=0.60,
                     sustain=0, bar=0.5, need=3, target_n=0, best_decay=0.5,
                     use_wilson_bound=True,
                     accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    assert d["action"] == "hold" and d["best_rate"] == pytest.approx(0.60)
    assert d["target_lb"] is None


def test_the_decay_can_only_lower_the_bar_never_raise_it():
    """`best` describes the ACCEPTED SNAPSHOT, and a hold did not change it —
    so a hold must not raise the bar. Reachable under the bounded rule: a
    probe's point estimate can exceed `best` while its lower bound does not."""
    d = oc.gate_step(regressed=None, target_rate=20 / 30, best_rate=0.60,
                     sustain=0, bar=0.5, need=3, target_n=30, best_decay=0.5,
                     use_wilson_bound=True,
                     accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    assert d["action"] == "hold"          # LB 0.488 does not clear 0.60
    assert d["best_rate"] == pytest.approx(0.60)   # ...and 0.667 does not lift


def test_the_decay_never_falls_below_the_accepted_snapshots_own_bound():
    """The floor. Decay without one lets a late collapse (v4's final iterate
    read greedy 0.00) walk the bar to ~0, after which a clearly worse network
    accepts and overwrites the deliverable. The floor is the accepting probe's
    own 95% lower bound: below that, a candidate is not credibly better than
    the snapshot already saved."""
    kw = dict(regressed=None, sustain=0, bar=0.5, need=3, target_n=30,
              use_wilson_bound=True, accept_rule=oc.ACCEPT_RULE_WILSON_LB,
              best_decay=0.5)
    acc = oc.gate_step(target_rate=20 / 30, best_rate=0.0, best_floor=None,
                       **kw)
    assert acc["action"] == "accept"
    assert acc["best_rate"] == pytest.approx(20 / 30)
    assert acc["best_floor"] == pytest.approx(0.48780, abs=5e-5)
    # Now collapse. However many probes read 0.000, the bar stops at the floor.
    best, floor = acc["best_rate"], acc["best_floor"]
    for _ in range(10):
        d = oc.gate_step(target_rate=0.0, best_rate=best, best_floor=floor,
                         **kw)
        assert d["action"] == "hold"
        best, floor = d["best_rate"], d["best_floor"]
    assert best == pytest.approx(0.48780, abs=5e-5)
    # ...so a 0.30 probe (LB 0.166) still cannot take the snapshot. Nor can a
    # re-measurement of the SAME rate, which is the point of the floor: 20/30
    # bounds at exactly 0.4878 and the accept is strict. One episode better
    # does clear it, so the gate stays open to real progress.
    assert oc.gate_step(target_rate=9 / 30, best_rate=best, best_floor=floor,
                        **kw)["action"] == "hold"
    assert oc.gate_step(target_rate=20 / 30, best_rate=best, best_floor=floor,
                        **kw)["action"] == "hold"
    assert oc.gate_step(target_rate=21 / 30, best_rate=best, best_floor=floor,
                        **kw)["action"] == "accept"


def test_the_floor_is_inert_without_a_bound():
    """Pre-B6 callers pass no floor and get none, so the clamp can never lift
    a negative incumbent (the trainer's `best_tgt_rate = -1.0` when the mode's
    BASELINE probe fails) up to 0.0 behind their back."""
    d = oc.gate_step(regressed=None, target_rate=0.0, best_rate=-1.0,
                     sustain=0, bar=0.5, need=3)
    assert d["action"] == "accept"      # 0.0 > -1.0, exactly as before
    d = oc.gate_step(regressed=None, target_rate=0.0, best_rate=0.5,
                     sustain=0, bar=0.5, need=3)
    assert d["action"] == "hold" and d["best_rate"] == pytest.approx(0.5)
    assert d["best_floor"] is None


def test_done_fires_on_the_sustained_wilson_bound_not_the_point_estimate():
    """The DONE condition is the deliverable's bar: LB >= accept_bar for
    `accept_probes` consecutive probes. A point estimate that clears 0.50 on a
    bound that does not must NOT terminate the run."""
    kw = dict(regressed=None, best_rate=1.0, bar=0.5, need=3, target_n=30,
              use_wilson_bound=True, accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    # 17/30 = 0.567 point, LB 0.390 -> below the bar, sustain resets.
    weak = oc.gate_step(target_rate=17 / 30, sustain=2, **kw)
    assert weak["sustain"] == 0 and weak["done"] is False
    # 24/30 = 0.80 point, LB 0.626 -> clears; three in a row terminates.
    s, done = 0, False
    for _ in range(3):
        d = oc.gate_step(target_rate=24 / 30, sustain=s, **kw)
        s, done = d["sustain"], d["done"]
    assert s == 3 and done is True


def test_the_bounded_rule_fails_CLOSED_not_back_onto_the_point_estimate():
    """THE FAIL-OPEN DEFECT. A profile that names `wilson_lb_gt_best_point`
    believes a lucky probe cannot pin its bar. If the bound is skipped for any
    reason and the comparison silently reverts to the point estimate, that
    profile gets v4's ratchet while its log says otherwise — so the two ways
    the bound can go missing must both be refused, loudly."""
    # (a) The rule named, `use_wilson_bound` left off. A 1/1 probe reading
    #     1.000 must NOT accept and must NOT set best to 1.0.
    d = oc.gate_step(regressed=None, target_rate=1.0, best_rate=0.30,
                     sustain=0, bar=0.5, need=3, target_n=1,
                     use_wilson_bound=False,
                     accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    assert d["action"] == "hold"
    assert d["target_lb"] == pytest.approx(0.20655, abs=5e-5)
    assert d["best_rate"] == pytest.approx(0.30)
    # (b) No episode count threaded at all — a wiring bug, not a run-time
    #     condition, so it stops the caller instead of quietly point-accepting.
    for kw in ({"accept_rule": oc.ACCEPT_RULE_WILSON_LB},
               {"use_wilson_bound": True}):
        with pytest.raises(ValueError, match="target_n"):
            oc.gate_step(regressed=None, target_rate=1.0, best_rate=0.30,
                         sustain=0, bar=0.5, need=3, target_n=None, **kw)
    # (c) The probe ran but scored nothing: hold, and leave the bar alone.
    d = oc.gate_step(regressed=None, target_rate=1.0, best_rate=0.30,
                     sustain=0, bar=0.5, need=3, target_n=0,
                     use_wilson_bound=True,
                     accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    assert d["action"] == "hold" and d["best_rate"] == pytest.approx(0.30)


def test_gate_step_itself_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="accept_rule"):
        oc.gate_step(regressed=None, target_rate=0.5, best_rate=0.1,
                     sustain=0, bar=0.5, need=3, accept_rule="wilson")


def test_unknown_accept_rule_is_rejected_by_the_registry():
    assert set(oc.ACCEPT_RULES) == {"point_gt_best", "wilson_lb_gt_best_point"}
    # An unrecognised rule must never silently fall back to point-accepting;
    # the trainer raises, and the schema validator warns.
    from src.training.config_schema import validate_profile
    prof = {"name": "x", "reinforce": {"consolidate_level": {
        "accept_rule": "wilson_lb", "target": "1-1",
    }}}
    assert any("accept_rule" in w for w in validate_profile(prof))


def test_trainer_raises_on_an_unknown_accept_rule():
    """Static guard on the raise site: a typo'd rule must stop the run, not
    quietly restore the v4 arithmetic under a B6 profile."""
    assert "accept_rule must be one of" in _TRAINER_SRC


def _trainer_block(start: str, end: str) -> str:
    i = _TRAINER_SRC.index(start)
    return _TRAINER_SRC[i:_TRAINER_SRC.index(end, i)]


def test_the_reporting_seed_guard_stops_the_run_rather_than_logging():
    """A warning does not stop a probe, and `--strict-config` cannot catch
    this one at all: every key involved is registered and individually valid,
    and the commonest way to hit it is to OMIT `eval_seed` entirely. So the
    guard has to be a raise, at mode start, before the block is spent."""
    honest = {"episodes": 30, "sticky_prob": 0.25, "start_jitter": 16}
    # Omitted seed -> eval_game's default 0 -> a reporting seed.
    with pytest.raises(ValueError, match="reporting seeds"):
        oc.require_selection_seed(oc.resolve_probe_settings(honest))
    # Explicitly set to a reporting seed -> same refusal.
    for bad in oc.REPORTING_EVAL_SEEDS:
        with pytest.raises(ValueError, match="reporting seeds"):
            oc.require_selection_seed(
                oc.resolve_probe_settings({**honest, "eval_seed": bad}))
    # The v6 profile, and the v4-equivalent deterministic probe, both pass.
    oc.require_selection_seed(oc.resolve_probe_settings(_v6_probe_cfg()))
    oc.require_selection_seed(oc.resolve_probe_settings({"episodes": 1}))
    # ...and the trainer calls it, at mode start, instead of branching on the
    # flag itself (which is how it became a warning the first time).
    assert "_oc_gate.require_selection_seed(clevel_probe_cfg)" in _TRAINER_SRC
    assert "if clevel_probe_cfg.seed_collision" not in _TRAINER_SRC


def test_the_gate_receipts_record_the_effective_seed_never_null():
    """The v4 eval rows recorded a perturbed number with `eval_seed: null`.
    Writing the CONFIGURED seed into the new gate receipts would reproduce
    that inside the very block added to stop it, because the configured seed
    is None precisely when the run silently used 0."""
    block = _trainer_block("clevel_protocol = {", "}\n")
    assert '"eval_seed": clevel_probe_cfg.effective_eval_seed' in block, block
    assert '"eval_seed_configured": clevel_probe_cfg.eval_seed' in block
    # ...and both receipts write that one object, so they cannot disagree.
    assert _TRAINER_SRC.count('"protocol": dict(clevel_protocol)') == 2


def test_the_trainer_threads_the_incumbent_floor_through_the_gate():
    call = _TRAINER_SRC[_TRAINER_SRC.index("_gate = oc.gate_step("):]
    call = call[:call.index("\n                )")]
    assert "best_floor=clevel_best_floor" in call, call
    assert 'clevel_best_floor = _gate["best_floor"]' in _TRAINER_SRC


# ==========================================================================
# The v6 profile itself
# ==========================================================================

def test_v6_is_v4_plus_exactly_the_pre_registered_keys():
    v4 = yaml.safe_load(_V4.read_text())["reinforce"]["consolidate_level"]
    v6 = yaml.safe_load(_V6.read_text())["reinforce"]["consolidate_level"]
    assert set(v6) - set(v4) == {
        "use_wilson_bound", "wilson_confidence", "accept_rule", "best_decay",
    }
    assert set(v6["probe"]) - set(v4["probe"]) == {
        "sticky_prob", "start_jitter", "eval_seed", "eval_workers",
    }
    # The gate block, and nothing else. Everything outside it is v4 verbatim.
    assert v6["schedule"] == v4["schedule"]
    assert v6["protect"] == v4["protect"] == []
    assert v6["target"] == v4["target"] == "1-1"
    assert v6["target_entry_state"] == v4["target_entry_state"]
    assert v6["cooldown"] == v4["cooldown"]
    assert v6["probe"]["every"] == v4["probe"]["every"] == 12
    assert v6["probe"]["max_steps"] == v4["probe"]["max_steps"] == 1500
    # The pre-registered values.
    assert v6["probe"]["episodes"] == 30
    assert v6["use_wilson_bound"] is True
    assert v6["wilson_confidence"] == 0.95
    assert v6["accept_rule"] == "wilson_lb_gt_best_point"
    assert v6["accept_bar"] == 0.50
    assert v6["accept_probes"] == 3
    # BOTH halves of the ratchet break, not just the bound. With the bound
    # alone, an accept at 20/30 sets the next bar at 26/30 and an accept at
    # 30/30 sets it at 1.0 against a maximum attainable LB of 0.886 — which is
    # v4's lockout with extra steps. See test_wilson_alone_still_locks_out.
    assert v6["best_decay"] == 0.25


def test_wilson_alone_still_locks_out_which_is_why_v6_arms_the_decay():
    """The arithmetic behind `best_decay: 0.25` in the profile, pinned so the
    two cannot drift apart. Both cells are measured, not asserted."""
    kw = dict(regressed=None, sustain=0, bar=0.5, need=3, target_n=30,
              use_wilson_bound=True, accept_rule=oc.ACCEPT_RULE_WILSON_LB)
    # (1) Accept at 20/30 with NO decay -> the next accept needs 26/30, which
    #     is above the flagship pair's honest ceiling (0.65 / 0.667).
    best = oc.gate_step(target_rate=20 / 30, best_rate=0.0, **kw)["best_rate"]
    assert best == pytest.approx(20 / 30)
    needed = min(k for k in range(31)
                 if oc.wilson_lower_bound(k, 30) > best)
    assert needed == 26
    # (2) A 30/30 probe pins the bar at 1.0, and no probe can ever bound above
    #     it — v4's lockout, reproduced under the "fixed" rule.
    pinned = oc.gate_step(target_rate=1.0, best_rate=0.0, **kw)["best_rate"]
    assert pinned == pytest.approx(1.0)
    assert oc.wilson_lower_bound(30, 30) < 1.0
    for k in range(31):
        assert oc.gate_step(target_rate=k / 30, best_rate=pinned,
                            **kw)["action"] == "hold"
    # (3) With the profile's decay armed, that 1.0 erodes back toward what the
    #     policy reproduces instead of standing for the rest of the run.
    b, f = pinned, oc.wilson_lower_bound(30, 30)
    for _ in range(6):
        d = oc.gate_step(target_rate=18 / 30, best_rate=b, best_floor=f,
                         best_decay=0.25, **kw)
        b, f = d["best_rate"], d["best_floor"]
    assert b < pinned
    assert b == pytest.approx(f)      # ...and settles on the snapshot's bound
    # (A 30/30 probe is still terminal at n=30 — nothing bounds above 0.886 —
    #  but that run has already met DONE: LB 0.886 >= accept_bar 0.50 for three
    #  probes exits the run with the snapshot in hand. The lockout that matters
    #  is the one that strands a MEDIOCRE incumbent, and that is what erodes.)


def test_v6_rest_of_the_profile_matches_v4():
    """The donor's lineage is load-bearing (a strict=False resume silently
    drops mismatched tensors), so everything outside the gate block must be
    identical to v4."""
    v4 = yaml.safe_load(_V4.read_text())
    v6 = yaml.safe_load(_V6.read_text())
    for key in ("rom_path", "frame_skip", "start_state_path", "ram_mapping",
                "reward_weights", "action_space", "ga_params",
                "max_episode_steps"):
        assert v6[key] == v4[key], key
    r4 = dict(v4["reinforce"])
    r6 = dict(v6["reinforce"])
    r4.pop("consolidate_level")
    r6.pop("consolidate_level")
    assert r6 == r4
    assert v6["name"] != v4["name"]      # ...but it banks in its own dir


def test_v6_passes_strict_config_validation():
    from src.training.config_schema import check_profile
    check_profile(yaml.safe_load(_V6.read_text()), strict=True)


# ==========================================================================
# The eval receipt schema
# ==========================================================================

def test_receipt_schema_source_emits_the_protocol_fields():
    src = (_ROOT / "scripts" / "eval_game.py").read_text()
    block = src[src.index('result = {'):src.index('eval_log = ckpt_dir')]
    for field in ('"sticky_prob"', '"start_jitter"', '"eval_seed"',
                  '"stochastic"'):
        assert field in block, f"{field} missing from the eval.jsonl row"


_ROM = _ROOT / "roms" / "Super Mario Bros. (World).nes"
_START = _ROOT / "runs" / "live_show" / "smb_4_4_micro" / "entrance_start.state"
_DONOR = (_ROOT / "checkpoints" / "mario_1_1_backward"
          / "vanilla_ppo_iter_00080.pt")


@pytest.mark.slow
@pytest.mark.skipif(
    not (_ROM.exists() and _START.exists() and _DONOR.exists()),
    reason="needs the SMB ROM, the 1-1 entrance start state and the B4 donor",
)
@pytest.mark.parametrize("perturbed", [True, False])
def test_eval_jsonl_row_records_its_own_protocol_never_null(perturbed):
    """The v4 rows (checkpoints/mario_1_1_backward_v4/eval.jsonl) carried a
    sticky-0.25 + jitter-16 number with no record that it was one. A rate whose
    protocol is not in its own row cannot be cited. Both arms are checked: the
    perturbed run must record what perturbed it, and the UNperturbed run must
    record honest zeros rather than nulls, so the two rows are comparable."""
    import scripts.eval_game as eval_game

    prof = yaml.safe_load(_V6.read_text())
    prof["start_state_path"] = str(_START)
    slug = eval_game.derive_checkpoint_dir("./checkpoints", prof["name"]).name
    extra = (["--sticky-prob", "0.25", "--start-jitter", "16"]
             if perturbed else [])
    with tempfile.TemporaryDirectory(prefix="b6_receipt_") as td:
        td = Path(td)
        (td / "checkpoints" / slug).mkdir(parents=True)
        pf = td / "probe_profile.yaml"
        pf.write_text(yaml.safe_dump(prof))
        r = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "eval_game.py"),
             "--game", "mario", "--profile", str(pf), "--rom", str(_ROM),
             "--checkpoint", str(_DONOR),
             "--episodes", "1", "--max-steps", "60",
             "--sequential", "--level-clear", "--start-state", str(_START),
             "--eval-seed", "7"] + extra,
            cwd=str(td), capture_output=True, text=True, timeout=900,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert json.loads(r.stdout)["status"] == "ok", r.stdout[-800:]
        rows = [json.loads(x) for x in
                (td / "checkpoints" / slug / "eval.jsonl").read_text().splitlines()]
    assert rows, "no eval.jsonl row written"
    row = rows[-1]
    for field in ("sticky_prob", "start_jitter", "eval_seed", "stochastic"):
        assert field in row and row[field] is not None, row
    assert row["eval_seed"] == 7
    assert row["sticky_prob"] == pytest.approx(0.25 if perturbed else 0.0)
    assert row["start_jitter"] == (16 if perturbed else 0)
    assert row["stochastic"] is perturbed
