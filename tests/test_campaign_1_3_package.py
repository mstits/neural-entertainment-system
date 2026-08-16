"""The 1-3 campaign package: profile schema, controller parameterization,
the audit-7 rung-progress kill, and the auto-target ladder adapter.

Token-bound tests only — no emulator, no ROM, no checkpoints. What is
under test:

* `configs/mario_1_3_online_v1.yaml` is a schema-clean clone of the
  proven 1-2 online profile, retargeted to the 1-3 entrance, with the
  net dims that are a CONTRACT with the BC anchor it will load;
* `configs/campaign_1_3.yaml` merges over the controller's CONFIG and
  PHASES, and the merge REFUSES the mistakes that would cost a window
  (unknown key, wrong type, unknown gate, out-of-order phase);
* the merge is pure — the 1-2 defaults the controller ships with are
  byte-identical afterwards (golden-pinned here as well as in
  tests/test_online_campaign.py);
* `rung_progress_kill` fires only on a real, armed, stalled reverse
  walk;
* `auto_targets` derives an evenly spaced ladder from a tape's own
  reach, and the targets it produces resolve through `select_entries`.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_online_campaign import (  # noqa: E402
    CONFIG, KNOWN_GATES, PHASES, build_phase_profile, merge_campaign_config,
    phase_pins_entrance, restart_provenance_ok, rung_progress_kill,
)
from scripts.select_restart_states import (  # noqa: E402
    auto_targets, check_ladder_provenance, pick_bottleneck_segment,
    select_entries, split_segments,
)
from src.training.backward_curriculum import StateEntry  # noqa: E402
from src.training.config_schema import check_profile  # noqa: E402

PROFILE_1_3 = ROOT / "configs" / "mario_1_3_online_v1.yaml"
PROFILE_1_2 = ROOT / "configs" / "mario_1_2_online_v2.yaml"
CAMPAIGN_1_3 = ROOT / "configs" / "campaign_1_3.yaml"
ENTRANCE_1_3 = (
    "checkpoints/super_mario_bros_one_shot_tiles/smb_curriculum/stage_05.state")


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# ---- the 1-3 base profile ---------------------------------------------


def test_1_3_profile_passes_the_strict_schema():
    """A misspelled knob is a silently-ignored mechanism; the campaign
    launches with --strict-config, so the profile must be clean here."""
    assert check_profile(_yaml(PROFILE_1_3), strict=False) == []


def test_1_3_profile_starts_at_the_solver_root_entrance():
    """stage_05 is the root all four banked 1-3 clears were searched from
    (runs/ge_1_3_solve/roots.json) AND the parallel lane's backward root.
    One entrance, one convention — a different start state would make the
    dmap, the rungs and the honest probe three different problems."""
    prof = _yaml(PROFILE_1_3)
    assert prof["start_state_path"] == ENTRANCE_1_3


def test_1_3_anchor_and_dmap_are_1_3_artifacts_not_1_2_ones():
    rl = _yaml(PROFILE_1_3)["reinforce"]
    assert rl["wavefront_reward"]["dmap"] == (
        "checkpoints/wavefront/mario_1_3_dmap.pkl")
    assert rl["kl_anchor_checkpoint"] == (
        "checkpoints/bc_1_3/anchor_h256/vanilla_ppo_iter_00000.pt")
    assert rl["backward_curriculum"]["states_dir"] == (
        "checkpoints/online_1_3/restart_states")
    # The parallel lane's dense ladder is READ-ONLY input to selection;
    # the trainer must never be pointed straight at it.
    assert "backward_states" not in rl["backward_curriculum"]["states_dir"]


def test_1_3_net_dims_are_the_bc_anchor_contract():
    """kl_anchor.load_actor_into raises on any shape mismatch, so these
    three lines and the BC training flags (--hidden-dim 256
    --trunk-dim 64) are one contract."""
    rl = _yaml(PROFILE_1_3)["reinforce"]
    assert rl["tile_hidden_dim"] == 256
    assert rl["tile_trunk_dim"] == 64
    assert rl["layernorm"] is True
    assert rl["encoder"] == "smb_tiles_pos"
    assert rl["recurrent"] is False


def test_1_3_gammas_are_equal_for_pbrs_invariance():
    rl = _yaml(PROFILE_1_3)["reinforce"]
    assert rl["gamma"] == 0.99
    assert rl["wavefront_reward"]["gamma"] == rl["gamma"]
    assert rl["wave_terminal_rule"] == "monotone"


def test_1_3_carries_the_proven_1_2_hyperparameters_verbatim():
    """The 1-2 -> 1-3 comparison must not be confounded: everything that
    is not a 1-3 INPUT is copied from the proven profile."""
    a = _yaml(PROFILE_1_2)["reinforce"]
    b = _yaml(PROFILE_1_3)["reinforce"]
    for key in ("rollout_steps", "num_envs", "ppo_minibatch_size", "steps",
                "lr", "ppo_clip_eps", "grad_clip", "value_loss", "value_coef",
                "gae_lambda", "entropy_coef", "entropy_coef_max",
                "rnd_intrinsic_coef", "async_pipeline", "symlog_rewards",
                "sticky_action_prob", "sticky_episode_boundary_reset",
                "kl_beta_start", "kl_beta_end", "kl_beta_decay_steps",
                "actor_freeze_steps"):
        assert b[key] == a[key], key
    assert b["sil"] == a["sil"]
    assert b["adversary"] == a["adversary"]
    assert "mode" not in b["adversary"], "the adversary must ship inert"
    pa, pb = _yaml(PROFILE_1_2), _yaml(PROFILE_1_3)
    assert pb["action_space"] == pa["action_space"]
    assert pb["ram_mapping"] == pa["ram_mapping"]
    assert pb["reward_weights"] == pa["reward_weights"]
    assert pb["rom_hashes"] == pa["rom_hashes"]
    assert pb["frame_skip"] == pa["frame_skip"]
    assert pb["max_episode_steps"] == pa["max_episode_steps"]


def test_1_3_rung_budget_is_the_proven_1_2_budget_verbatim():
    """PROVE-IT (defect: unearned retune). An earlier draft shipped
    base 250 / per_entry 220, justified as "the same ratio 1-2 ran at".
    1-2's actual ratio is 650 budget / 312 remaining tape steps = 2.08x,
    not the claimed 1.5x, and where 1-3's deepest rung lands on its
    540-step tape is unknowable until the ladder is minted. A budget
    tuned against an unmeasured quantity is an un-calibrated threshold,
    so the proven values ship and the retune waits for the mint."""
    bwd = _yaml(PROFILE_1_3)["reinforce"]["backward_curriculum"]
    proven = _yaml(PROFILE_1_2)["reinforce"]["backward_curriculum"]
    assert bwd["rung_step_budget"] == proven["rung_step_budget"]
    assert bwd["rung_step_budget"] == {"base": 300, "per_entry": 350}
    assert bwd["count_truncations"] is True
    assert bwd["min_attempts"] == 30, "3-episode gates are mirages"
    assert bwd["tau_init"] == -1
    assert bwd["advance_actions"] == 1
    assert 0.0 < bwd["entrance_weight"] < 1.0


# ---- the campaign document ---------------------------------------------


def test_campaign_1_3_merges_and_retargets_every_path():
    cfg, phases = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    assert cfg["base_profile"] == "configs/mario_1_3_online_v1.yaml"
    assert cfg["run_dir"] == "runs/online_1_3"
    assert cfg["campaign_log"] == "runs/online_1_3/campaign.jsonl"
    assert cfg["restart_states_dir"] == "checkpoints/online_1_3/restart_states"
    assert cfg["campaign_name"] == "smb_1_3_online_v1"
    assert cfg["probe_game"] == "mario_1_3_online_v1"
    assert len(phases) == 6


def test_campaign_1_3_keeps_the_honest_probe_protocol():
    """The protocol IS the purity line: 30 episodes, greedy, sticky 0.25,
    jitter 16, cold from the entrance. Only the seed may move."""
    cfg, _ = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    assert cfg["probe_episodes"] == CONFIG["probe_episodes"] == 30
    assert cfg["probe_sticky"] == CONFIG["probe_sticky"] == 0.25
    assert cfg["probe_jitter"] == CONFIG["probe_jitter"] == 16
    assert cfg["probe_seed"] != CONFIG["probe_seed"]


def test_campaign_1_3_arms_the_rung_kill_where_the_cursor_actually_moves():
    """PROVE-IT (defect: kill armed where it cannot fire). The threshold
    is CUMULATIVE campaign steps, so it must clear the whole pre-walk
    schedule (a phase-local 25M could not fire in the ~5M of phase steps
    1-2's reverse walk actually took) while still leaving the reverse
    walk's own budget to save."""
    cfg, phases = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    limit = cfg["kill_rung_halfway_steps"]
    assert limit == 65_000_000

    walk = [p for p in phases if p["gate"] == "restart_at_entrance"]
    assert len(walk) == 1, "exactly one phase owns the reverse walk"
    pre_walk = sum(p["budget_env_steps"] for p in phases
                   if p["idx"] < walk[0]["idx"])
    assert limit >= pre_walk, (
        "a cumulative threshold below the pre-walk budget can fire while "
        "the curriculum is still legitimately walking in phases 0-2")
    assert limit < pre_walk + walk[0]["budget_env_steps"], (
        "a threshold past the walk's own budget can never pre-empt the "
        "abort it exists to save")
    # The one precedent: 1-2 attempt 7 opened its reverse-walk phase at
    # 70.04M cumulative with tau already 0, so this threshold must not
    # have fired on the only campaign we can check.
    assert limit < 70_041_600


def test_rung_kill_is_armed_in_every_unpinned_phase():
    """PROVE-IT: the curriculum lives in the BASE profile, so tau
    advances from phase 0. Only entrance-pinned phases have no cursor to
    falsify. Arming solely on the reverse-walk gate would watch the one
    phase where the precedent says the walk is already finished."""
    _, phases = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    base_bwd = _yaml(PROFILE_1_3)["reinforce"]["backward_curriculum"]
    assert not base_bwd.get("pin_entrance"), (
        "the base profile must leave the curriculum live, else 'unpinned' "
        "no longer identifies the phases whose cursor moves")
    watched = [p["idx"] for p in phases if not phase_pins_entrance(p)]
    assert watched == [0, 1, 2, 3], watched
    pinned = [p["idx"] for p in phases if phase_pins_entrance(p)]
    assert pinned == [4, 5], pinned
    walk = [p["idx"] for p in phases if p["gate"] == "restart_at_entrance"]
    assert set(walk) < set(watched), (
        "the reverse-walk phase must be a STRICT subset of what is "
        "watched — that gap is the defect")


def test_campaign_1_3_inherits_every_unlisted_1_2_threshold():
    """Silence in the document means "keep the proven value" — the kill
    table is not re-tuned per level except where a reason is written."""
    cfg, _ = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    for key in ("kill_kl_threshold", "kill_kl_sustain_steps",
                "kill_vloss_spike_ratio", "kill_vloss_recovery_steps",
                "kill_vloss_baseline_rows", "kill_phase1_no_sil_clear_steps",
                "gate_critic_cv", "gate_critic_window_iters",
                "gate_min_phase0_steps", "gate_det_clears",
                "gate_sticky_clear", "entropy_start", "entropy_end",
                "entropy_decay_steps", "probe_every_env_steps"):
        assert cfg[key] == CONFIG[key], key


def test_campaign_1_3_phase_profiles_are_schema_clean_and_chain():
    """Every phase profile is what the trainer actually launches with."""
    cfg, phases = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    base = _yaml(ROOT / cfg["base_profile"])
    for phase in phases:
        prof = build_phase_profile(base, phase, cum_env_steps=0)
        assert check_profile(prof, strict=False) == [], phase["name"]
        # Same name every phase => every phase resumes one checkpoint dir.
        assert prof["name"] == base["name"]
    armed = [p["name"] for p in phases
             if (p.get("overrides", {}).get("reinforce", {})
                 .get("adversary", {}).get("mode"))]
    assert armed == ["hardening"]


def test_campaign_1_3_entropy_decays_across_the_schedule():
    cfg, phases = merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    base = _yaml(ROOT / cfg["base_profile"])
    first = build_phase_profile(base, phases[0], cum_env_steps=0)
    last = build_phase_profile(
        base, phases[-1], cum_env_steps=cfg["entropy_decay_steps"])
    assert first["reinforce"]["entropy_coef"] == pytest.approx(
        cfg["entropy_start"])
    assert last["reinforce"]["entropy_coef"] == pytest.approx(
        cfg["entropy_end"])


# ---- merge validation (the mistakes that cost a window) ----------------


def test_merge_rejects_an_unregistered_config_key():
    with pytest.raises(ValueError, match="unknown CONFIG key"):
        merge_campaign_config(CONFIG, PHASES,
                              {"config": {"kill_kl_treshold": 0.9}})


def test_merge_rejects_a_wrongly_typed_value():
    with pytest.raises(ValueError, match="expected"):
        merge_campaign_config(CONFIG, PHASES,
                              {"config": {"probe_episodes": "thirty"}})


def test_merge_rejects_an_unknown_top_level_key():
    with pytest.raises(ValueError, match="unknown campaign-config key"):
        merge_campaign_config(CONFIG, PHASES, {"phase": []})


def test_merge_rejects_an_unknown_gate():
    doc = {"phases": [{"idx": 0, "name": "x", "gate": "vibes",
                       "budget_env_steps": 1}]}
    with pytest.raises(ValueError, match="unknown gate"):
        merge_campaign_config(CONFIG, PHASES, doc)


def test_merge_rejects_out_of_order_phase_indices():
    doc = {"phases": [{"idx": 1, "name": "x", "gate": "budget",
                       "budget_env_steps": 1}]}
    with pytest.raises(ValueError, match="sequential"):
        merge_campaign_config(CONFIG, PHASES, doc)


def test_merge_rejects_a_phase_missing_its_budget():
    doc = {"phases": [{"idx": 0, "name": "x", "gate": "budget"}]}
    with pytest.raises(ValueError, match="missing"):
        merge_campaign_config(CONFIG, PHASES, doc)


def test_every_registered_gate_is_one_the_1_2_schedule_uses_or_knows():
    assert {p["gate"] for p in PHASES} <= KNOWN_GATES


# ---- 1-2 goldens: the parameterization changes nothing by default ------


def test_merge_does_not_mutate_the_controller_defaults():
    before_cfg = copy.deepcopy(CONFIG)
    before_phases = copy.deepcopy(PHASES)
    merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))
    assert CONFIG == before_cfg
    assert PHASES == before_phases


def test_1_2_defaults_are_byte_preserved():
    """Golden pins on the shipped 1-2 campaign — the retargeting work
    must not have moved a single 1-2 number."""
    assert CONFIG["base_profile"] == "configs/mario_1_2_online_v2.yaml"
    assert CONFIG["run_dir"] == "runs/online_1_2"
    assert CONFIG["campaign_log"] == "runs/online_1_2/campaign.jsonl"
    assert CONFIG["restart_states_dir"] == "checkpoints/online_1_2/restart_states"
    assert CONFIG["campaign_name"] == "smb_1_2_online_v2"
    assert CONFIG["probe_game"] == "mario_1_2_online_v2"
    assert CONFIG["probe_seed"] == 20260814
    assert CONFIG["bottleneck_x"] == 2674
    assert CONFIG["bottleneck_survival_x"] == 2600
    assert CONFIG["kill_probe_median_floor"] == 150.0
    # The audit-7 kill ships DISARMED for 1-2: its ledger predates the
    # criterion, and arming a new kill on a banked campaign would change
    # a pre-registration after the fact.
    assert CONFIG["kill_rung_halfway_steps"] == 0
    assert [(p["idx"], p["name"], p["gate"], p["budget_env_steps"])
            for p in PHASES] == [
        (0, "critic_warmup", "critic_warmup", 15_000_000),
        (1, "local_clear", "det_local_clears", 40_000_000),
        (2, "sticky_local", "budget", 2_000_000),
        (3, "reverse_walk", "restart_at_entrance", 60_000_000),
        (4, "hardening", "budget", 20_000_000),
        (5, "consolidation", "budget", 10_000_000),
    ]


def test_empty_campaign_doc_is_the_identity():
    cfg, phases = merge_campaign_config(CONFIG, PHASES, {})
    assert cfg == CONFIG
    assert phases == PHASES


# ---- audit-7 rung-progress kill ----------------------------------------

ARMED = {"kill_rung_halfway_steps": 25_000_000}
SIX = {"tau": 5, "n_entries": 6}          # deepest rung, no advances yet


def test_rung_kill_fires_on_a_stalled_walk_past_the_threshold():
    reason = rung_progress_kill(
        state=SIX, curriculum_env_steps=25_000_000, cfg=ARMED)
    assert reason is not None and "rung-progress" in reason
    assert "tau=5/5" in reason


def test_rung_kill_silent_before_the_threshold():
    assert rung_progress_kill(
        state=SIX, curriculum_env_steps=24_999_999, cfg=ARMED) is None


def test_rung_kill_silent_once_tau_passes_the_midpoint():
    # 6 rungs -> midpoint index 2; tau 3 is still short, tau 2 is past.
    assert rung_progress_kill(
        state={"tau": 3, "n_entries": 6}, curriculum_env_steps=30_000_000,
        cfg=ARMED) is not None
    assert rung_progress_kill(
        state={"tau": 2, "n_entries": 6}, curriculum_env_steps=30_000_000,
        cfg=ARMED) is None
    assert rung_progress_kill(
        state={"tau": 0, "n_entries": 6}, curriculum_env_steps=30_000_000,
        cfg=ARMED) is None


def test_rung_kill_disarmed_by_default_never_fires():
    assert rung_progress_kill(
        state=SIX, curriculum_env_steps=10**12, cfg=CONFIG) is None


def test_rung_kill_never_fires_on_missing_or_degenerate_state():
    """An unreadable checkpoint is a probe problem; a kill that fires on
    missing data is a coin flip."""
    for state in (None, {}, {"tau": 5}, {"n_entries": 6},
                  {"tau": 0, "n_entries": 1}, "not-a-dict"):
        assert rung_progress_kill(
            state=state, curriculum_env_steps=10**9, cfg=ARMED) is None


# ---- auto-target ladder adapter ----------------------------------------


def _single_segment_ladder(n: int = 540, max_gx: int = 2515):
    """1-3-shaped tape: one area byte, monotone gx, 540 steps."""
    return [StateEntry(step=i, frame=i * 4, gx=int(max_gx * i / (n - 1)),
                       area=3, file=f"s_{i:06d}.state") for i in range(n)]


def _three_segment_ladder():
    """1-2-shaped tape: intro / underground / re-based exit."""
    entries = []
    for i in range(10):
        entries.append(StateEntry(step=i, frame=i * 4, gx=i * 16, area=1,
                                  file=f"s_{i:06d}.state"))
    for i in range(100):
        step = 10 + i
        entries.append(StateEntry(step=step, frame=step * 4, gx=i * 27,
                                  area=2, file=f"s_{step:06d}.state"))
    for i in range(41):
        step = 110 + i
        entries.append(StateEntry(step=step, frame=step * 4, gx=i * 20,
                                  area=2, file=f"s_{step:06d}.state"))
    return entries


def test_auto_targets_are_six_descending_rungs_ending_at_the_entrance():
    t = auto_targets(_single_segment_ladder(), 6)
    assert len(t) == 6
    assert t[-1] == 0
    assert list(t) == sorted(t, reverse=True)
    assert all(x % 50 == 0 for x in t)


def test_auto_targets_span_the_tape_without_touching_the_clear():
    """The deepest rung must be a real approach, not the clear frame."""
    max_gx = 2515
    t = auto_targets(_single_segment_ladder(max_gx=max_gx), 6)
    assert t[0] < max_gx
    assert t[0] >= int(max_gx * 0.75)


def test_auto_targets_resolve_through_the_real_selector():
    """The adapter is only useful if select_entries accepts its output."""
    ladder = _single_segment_ladder()
    t = auto_targets(ladder, 6)
    picked = select_entries(ladder, t, tolerance_px=64)
    assert len(picked) == 6
    assert picked[0][1].step == 0, "target 0 is the ladder head"
    gxs = [e.gx for _, e in picked]
    assert gxs == sorted(gxs), "rungs are ordered by tape step == by gx"


def _deep_exit_ladder():
    """1-2-SHAPED AS IT REALLY IS: the re-based exit run out-reaches the
    underground segment (gx 160 / 2674 / 3175 on the banked ladder). The
    old fixture capped the exit run at 800px, so it never exercised the
    case where the deepest segment is NOT the one select_entries uses."""
    entries = []
    for i in range(116):                      # intro, area 1, max gx 160
        entries.append(StateEntry(step=i, frame=i * 4, gx=i * 160 // 115,
                                  area=1, file=f"s_{i:06d}.state"))
    for i in range(856):                      # underground, max gx 2674
        step = 116 + i
        entries.append(StateEntry(step=step, frame=step * 4,
                                  gx=i * 2674 // 855, area=2,
                                  file=f"s_{step:06d}.state"))
    for i in range(122):                      # re-based exit, max gx 3175
        step = 972 + i
        entries.append(StateEntry(step=step, frame=step * 4,
                                  gx=i * 3175 // 121, area=2,
                                  file=f"s_{step:06d}.state"))
    return entries


def test_auto_targets_use_the_deepest_segment_on_a_rebased_tape():
    """On the 1-2-shaped ladder the underground segment (max gx 2673)
    owns the ladder, not the 800px exit run or the 144px intro."""
    t = auto_targets(_three_segment_ladder(), 6)
    assert t[0] > 2000
    picked = select_entries(_three_segment_ladder(), t, tolerance_px=64)
    assert len(picked) == 6


def test_auto_targets_scale_and_resolve_in_the_SAME_segment():
    """PROVE-IT (defect: scale and rungs came from different segments).

    On the real 1-2 ladder shape, scaling off the deepest segment (3175,
    the post-rebase exit run) put the top target at 2600, which
    select_entries then resolved inside the EARLIER 2674 underground
    segment — the scale from one room, the rungs from another. The rule
    must settle on one segment and it must be the one select_entries
    uses."""
    ladder = _deep_exit_ladder()
    segs = [s for s in split_segments(sorted(ladder, key=lambda e: e.step))
            if s]
    assert [max(e.gx for e in s) for s in segs] == [160, 2674, 3175], (
        "fixture must reproduce the banked 1-2 segment reaches")

    t = auto_targets(ladder, 6)
    owner = pick_bottleneck_segment(segs, max(x for x in t if x))
    assert max(e.gx for e in owner) == 2674, (
        "the underground segment must own the ladder, scale included")
    assert t[0] == 2200, t                    # 2674 * 5/6 floored to 50
    assert t[0] < 2674, "the deepest rung is an approach, not the reach"

    # Every resolved rung lands inside the segment that set the scale.
    span = {e.step for e in owner} | {ladder[0].step}
    assert all(e.step in span for _, e in select_entries(ladder, t))


def test_auto_targets_scale_off_the_deepest_when_nothing_earlier_covers():
    """The fixed point is only reached when an EARLIER segment covers the
    top target. A tape whose deepest segment is also the earliest that
    covers it must keep scaling off the deepest — no silent demotion."""
    ladder = _single_segment_ladder()
    assert auto_targets(ladder, 6)[0] == 2050   # 2515 * 5/6 floored to 50


def test_auto_targets_refuse_a_ladder_too_short_to_split():
    with pytest.raises(ValueError, match="collapsed"):
        auto_targets(_single_segment_ladder(n=20, max_gx=100), 6)


def test_auto_targets_are_self_consistent_on_the_banked_1_2_ladder():
    """The rule run on a REAL ladder, checked for the property it can
    actually be held to.

    This is deliberately NOT "the auto targets match the hand-chosen
    2500/2000/1500/1000/500/0". That comparison was never external
    validation: this ladder's meta names root entrance_after_1-1.state
    over a 1215-action tape, whereas the hand-chosen numbers were picked
    on the stage_03-rooted 871-action tape the campaign shipped from,
    AND they were picked around a MEASURED barrier at 2674 rather than
    by even spacing — 2500 is 0.94x the segment reach, even spacing puts
    the top rung at 0.83x. Two rules can disagree there and both be
    right. What the rule must never do is disagree with ITSELF, so what
    is asserted is the invariant: one segment sets the scale and that
    same segment resolves every rung. (Read-only; skipped when absent.)"""
    ladder = ROOT / "checkpoints" / "backward_states" / "1-2"
    if not (ladder / "index.json").exists():
        pytest.skip("1-2 dense ladder not present")
    from src.training.backward_curriculum import load_index
    entries, _ = load_index(ladder)
    segs = [s for s in split_segments(sorted(entries, key=lambda e: e.step))
            if s]
    t = auto_targets(entries, 6)
    owner = pick_bottleneck_segment(segs, max(x for x in t if x))
    scale_reach = max(e.gx for e in owner)
    assert t[0] == int(scale_reach * 5 / 6) // 50 * 50, (t, scale_reach)
    span = {e.step for e in owner} | {sorted(entries,
                                            key=lambda e: e.step)[0].step}
    assert all(e.step in span for _, e in select_entries(entries, t))


# ---- restart-set provenance (Branch A: consuming another lane's ladder) --


def _campaign_1_3():
    return merge_campaign_config(CONFIG, PHASES, _yaml(CAMPAIGN_1_3))[0]


def _good_meta():
    return {"level": "1-3", "root_state": ENTRANCE_1_3,
            "profile": "configs/mario_1_3_online_v1.yaml", "n_actions": 540}


def test_provenance_accepts_this_lanes_own_ladder():
    ok, why = restart_provenance_ok(
        _good_meta(), _yaml(PROFILE_1_3), _campaign_1_3())
    assert ok, why


def test_provenance_accepts_the_shipped_1_2_restart_set():
    """The gate must not break the campaign it was proven on."""
    d = ROOT / "checkpoints" / "online_1_2" / "restart_states"
    if not (d / "index.json").exists():
        pytest.skip("1-2 restart set not present")
    from src.training.backward_curriculum import load_index
    _, meta = load_index(d)
    ok, why = restart_provenance_ok(meta, _yaml(PROFILE_1_2), CONFIG)
    assert ok, why


def test_provenance_rejects_the_foreign_rooted_ladder_on_disk():
    """PROVE-IT (defect: Branch A had no provenance gate). The live
    example: checkpoints/backward_states/1-2 is schema-perfect and every
    other check passes on it, but its root is a 4-4 show-lane entrance
    and its profile is smb_4_4_micro."""
    ladder = ROOT / "checkpoints" / "backward_states" / "1-2"
    if not (ladder / "index.json").exists():
        pytest.skip("1-2 dense ladder not present")
    from src.training.backward_curriculum import load_index
    _, meta = load_index(ladder)
    assert meta.get("level") == "1-2" and "smb_4_4_micro" in str(
        meta.get("root_state")), "fixture premise: right level, wrong root"
    ok, why = restart_provenance_ok(meta, _yaml(PROFILE_1_2), CONFIG)
    assert not ok and "root_state" in why, why


def test_provenance_rejects_a_foreign_level():
    meta = {**_good_meta(), "level": "1-2"}
    ok, why = restart_provenance_ok(
        meta, _yaml(PROFILE_1_3), _campaign_1_3())
    assert not ok and "level" in why


def test_provenance_refuses_an_unstamped_ladder():
    """A ladder that cannot prove its origin is not a pass."""
    for meta in (None, {}, {"level": "1-3"}, {"root_state": ENTRANCE_1_3},
                 "not-a-dict"):
        ok, _ = restart_provenance_ok(
            meta, _yaml(PROFILE_1_3), _campaign_1_3())
        assert not ok, meta


def test_campaign_1_3_declares_the_level_the_gate_checks():
    cfg = _campaign_1_3()
    assert cfg["campaign_level"] == "1-3"
    assert CONFIG["campaign_level"] == "1-2", "1-2 default byte-preserved"


def test_select_cli_provenance_gate_refuses_a_foreign_ladder():
    """The same gate at the point of consumption, so Branch A fails at
    selection rather than 65M env-steps into a campaign."""
    good = _good_meta()
    check_ladder_provenance(good, level="1-3", root_state=ENTRANCE_1_3)
    with pytest.raises(SystemExit, match="level"):
        check_ladder_provenance(good, level="1-1")
    with pytest.raises(SystemExit, match="root_state"):
        check_ladder_provenance(good, root_state="/some/other.state")
    with pytest.raises(SystemExit):
        check_ladder_provenance({}, level="1-3")
    # Opt-in: an unchecked call is still the untouched default path.
    check_ladder_provenance({"level": "9-9"})


def test_auto_targets_reject_bad_arguments():
    with pytest.raises(ValueError):
        auto_targets(_single_segment_ladder(), 1)
    with pytest.raises(ValueError):
        auto_targets([], 6)
