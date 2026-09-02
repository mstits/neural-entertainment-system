"""Profile config validation — catch the silent-typo / dead-knob class.

A misspelled reinforce key (``entropy_flooor: 0.5``) is silently
ignored today: ``rl_cfg.get("entropy_floor", default)`` returns the
default and the intended setting never takes effect — a whole class of
"why isn't my knob doing anything" bugs (the review counted ~193
unregistered/dead knobs; sticky_action_prob and eval --start-state were
each parsed-but-inert for weeks).

This validator holds the registry of reinforce keys the trainer
actually consumes and flags any key in a profile's ``reinforce`` block
that is not registered. It is intentionally a WARN-by-default (a
genuinely new knob should not hard-block a run) with a ``strict`` mode
that raises — used by the test suite and available via CLI so CI-less
local runs can opt into hard rejection.

Keeping this registry in sync is enforced by test_config_schema.py,
which re-derives the consumed set from the trainer source and asserts
it is a subset of the registry (a newly-consumed key must be
registered, or the test fails).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

# Reinforce keys the trainer consumes. Kept in sync with trainer.py by
# test_config_schema.py (which parses rl_cfg.get(...) sites). Sub-block
# dict keys (consolidate_level.*, cold_eval.*, etc.) are validated only
# at the top level here; their internals are owned by their own readers.
KNOWN_REINFORCE_KEYS: frozenset[str] = frozenset({
    # Phase 3 of the hazard substrate: freeze the model and veto actions
    # above a predicted death probability. Genuinely consumed by
    # Trainer._make_network, which is the bar this set enforces.
    "hazard_mask",
    # Action-commitment options (OPTIONS_PREREG_2026-08-22): consumed by
    # Trainer._make_network, the rollout overlay, and eval_game.
    "commitment_options",
    "actor_freeze_steps", "advance", "adversary", "asm_bulk_cycles",
    "async_pipeline", "autocast_fp16",
    "backward_curriculum",
    "batched_render", "bc_demo_path", "bc_epochs", "bc_replay_enabled",
    "bc_replay_epochs", "bc_replay_every_gens", "bc_replay_max_buffer",
    "bc_replay_train_window",
    # Vanilla-PPO checkpoint rotation (external audit 2026-08-28: the
    # path never rotated and checkpoints/ hit 98 GB). Default 0 = keep
    # ALL — behavior-preserving, and mandatory for any campaign whose
    # registered scoring reads its full grid (v32 cross-fit, the
    # corrected peak ladders). Consumed by CheckpointManager.save_iter.
    "checkpoint_keep_last",
    "cold_eval", "consolidate",
    "consolidate_level", "demo_anchor_coef", "demo_anchor_coef_final",
    "demo_anchor_decay_iters", "demo_anchor_decay_start",
    "demo_anchor_enabled", "demo_anchor_margin", "demo_anchor_minibatch",
    "demo_anchor_paths", "device", "drq_aug", "drq_pad", "enabled",
    "encoder", "entropy_coef", "entropy_coef_max", "entropy_floor",
    # Per-episode metrics sidecar (commit a9b3355, "observability:
    # per-episode metrics sidecar"). Consumed by Trainer at
    # trainer.py:5194/7393 to gate the sidecar writer.
    "episode_metrics",
    "cgsa", "episodes_per_genome", "freeze_pre_ppo_elite", "gae_lambda", "gamma",
    "go_explore", "go_explore_fallback", "grad_clip",
    "gx_count_bonus_coef", "inherit_curriculum_on_fresh",
    "kl_anchor_checkpoint", "kl_anchor_loss_coef", "kl_beta_decay_steps",
    "kl_beta_end", "kl_beta_start", "layernorm",
    "lr", "max_steps_per_traj", "num_envs", "num_instances",
    "pace_multiplier", "panic_isolation", "plr_enabled", "ppo_clip_eps",
    "ppo_minibatch_size", "preprocess_f16", "preserve_elite_diversity",
    "prmdp", "recurrent", "recurrent_env_minibatch",
    # ReDo — Recycling Dormant neurons (V27_FRESH_RECOVERY_2026-08-24.md
    # AMENDMENT 1, B4). Default OFF; the v27 seed configs set
    # redo_enabled: true explicitly.
    # `redo_mode` / `redo_bottom_k` are V32_REDO_BOTTOM_K_2026-08-28.md
    # §2: the rank-based selection rule. Default mode is "threshold",
    # i.e. the pre-v32 behaviour, so no existing config changes meaning.
    "redo_bottom_k", "redo_check_every_iters", "redo_enabled", "redo_mode",
    "redo_reset_optimizer_moments", "redo_sample_batch", "redo_tau",
    "rnd_intrinsic_coef",
    "rnd_loss_coef", "rnd_predictor_update_fraction", "rollout_steps",
    "sam_rho", "sil", "smb_curriculum", "steps", "sticky_action_prob",
    "sticky_episode_boundary_reset", "substage_ladder",
    "symlog_rewards", "tile_frame_stack", "tile_hidden_dim", "tile_trunk_dim",
    "top_k", "torch_compile",
    "trainer_mode", "value_coef", "value_loss", "vmap_forward",
    "warm_start", "warmup_gens_ga_only", "wave_terminal_rule",
    "wavefront_reward",
})

# Sub-keys of `reinforce.backward_curriculum`. This block is the one
# sub-block that is validated one level deeper than its parent, because
# it is the block pre-registered runs are configured through and a
# silently-ignored knob there costs a 2.5-hour attended window (B5 run
# 2's advance window starved on scoring semantics). Kept in sync with
# the trainer by test_config_schema.py, which re-derives the consumed
# set from the `_bwd_cfg.get(...)` sites. Sub-SUB-blocks
# (`entropy_guard`, `rung_step_budget`) are validated by their own
# readers, both of which RAISE on an unknown or missing field.
KNOWN_BACKWARD_CURRICULUM_KEYS: frozenset[str] = frozenset({
    "advance_actions", "advance_threshold", "count_truncations", "enabled",
    "entrance_weight", "entropy_guard", "min_attempts", "pin_entrance",
    "rung_step_budget", "states_dir", "tau_init", "truncation_is_failure",
    "window_frames",
})

# Sub-keys of `reinforce.consolidate_level` and of its `probe` block.
# Validated a level (and two levels) deeper for the same reason the
# backward-curriculum block is: it is how a pre-registered consolidation
# run is configured, and a silently-ignored knob there costs the whole
# attended window. The B6 receipt makes the cost concrete — v4's gate ran
# a deterministic single-replay probe because the honest-protocol keys
# had nowhere to be written, and a typo'd `sticky_probb` would reproduce
# exactly that failure while looking configured.
KNOWN_CONSOLIDATE_LEVEL_KEYS: frozenset[str] = frozenset({
    "accept_bar", "accept_probes", "accept_rule", "best_decay", "cooldown",
    "enabled", "probe", "protect", "schedule", "seed_globs", "target",
    "target_entry_state", "tol", "use_wilson_bound", "wilson_confidence",
    "winner",
})

KNOWN_CONSOLIDATE_PROBE_KEYS: frozenset[str] = frozenset({
    "episodes", "eval_rng", "eval_seed", "eval_workers", "every",
    "max_steps", "start_jitter", "sticky_prob",
})

# Sub-keys of `reinforce.sil` (self-imitation buffer) and
# `reinforce.adversary` (kernel-matched binary adversary). Both are
# campaign blocks a pre-registered online run is configured through —
# same one-level-deeper rationale as backward_curriculum: a typo'd
# `bc_coef` or `budget_penalty` silently reverting to the default costs
# the attended window.
KNOWN_SIL_KEYS: frozenset[str] = frozenset({
    "bc_coef", "buffer_size", "enabled",
})

KNOWN_ADVERSARY_KEYS: frozenset[str] = frozenset({
    "budget_penalty", "clip", "entropy_coef", "epochs", "lr", "mode",
})

# Mirrors kernel_adversary's accepted modes. Literal (not imported) so the
# validator stays importable with no training deps, same as _ACCEPT_RULES.
_ADVERSARY_MODES: frozenset[str] = frozenset({"kernel_sticky"})

# Top-level profile keys read outside `reinforce` — by the launcher,
# the GA path (`ga_params`), the Dreamer path (`dreamer`), the
# composite-eval manifest family (`levels`, `hysteresis_k`), and
# `tensorboard` (trainer.py's `game_profile.get("tensorboard", True)`,
# gating MetricsSink's TensorBoard writer).
KNOWN_TOP_KEYS: frozenset[str] = frozenset({
    "name", "reward_id", "game", "rom_path", "rom", "rom_hashes", "expected_md5",
    "rom_md5", "description", "frame_skip", "max_episode_steps",
    "start_state_path", "start_state", "ram_mapping", "reinforce", "solve",
    "reward_weights", "action_space", "curriculum", "env_spec", "seed",
    "ga_params", "dreamer", "levels", "hysteresis_k", "stop_after_worlds",
    "plr_levels", "tensorboard",
})

# Sub-keys of `solve` (go_explore_solve.py's GenericGame). Seeded
# 2026-09-01 from the 45 solve-shaped configs/*.yaml profiles plus two
# consumed-but-unadopted keys (hw_flags, stasis) - census in
# reports/.../2026-09-01-outstanding/config-and-fidelity.md.
KNOWN_SOLVE_KEYS: frozenset[str] = frozenset({
    "rom", "progress", "y", "lives", "level_key", "no_clear_predicate",
    "clear", "area", "state_sig", "player_state", "death_states",
    "progress_cap", "hold_macros", "room_advance", "entity_slots",
    "kill_key_local", "boss_typed", "room_sig", "room_fp", "boss",
    "finale", "transit_source", "area_key", "min_blank_frames",
    "constructible", "reason", "hw_flags", "stasis",
})


# Mirrors oneshot_curriculum.ACCEPT_RULES. Duplicated as a literal rather
# than imported so this validator stays importable with no training deps
# (the launcher validates before any of the heavy modules load).
_ACCEPT_RULES: frozenset[str] = frozenset({
    "point_gt_best", "wilson_lb_gt_best_point",
})


class ConfigSchemaError(ValueError):
    """Raised in strict mode when a profile has unregistered keys."""


def validate_profile(profile: dict[str, Any]) -> list[str]:
    """Return a list of human-readable warnings for unregistered keys.

    Empty list == clean. Does not mutate the profile.
    """
    warnings: list[str] = []
    for k in profile:
        if k not in KNOWN_TOP_KEYS:
            warnings.append(
                f"unknown top-level profile key {k!r} "
                f"(typo? it will be silently ignored)"
            )
    rl = profile.get("reinforce")
    if isinstance(rl, dict):
        for k in rl:
            if k not in KNOWN_REINFORCE_KEYS:
                warnings.append(
                    f"unknown reinforce key {k!r} — NOT consumed by the "
                    f"trainer (typo? renamed? it will be silently ignored)"
                )
        bwd = rl.get("backward_curriculum")
        if isinstance(bwd, dict):
            for k in bwd:
                if k not in KNOWN_BACKWARD_CURRICULUM_KEYS:
                    warnings.append(
                        f"unknown reinforce.backward_curriculum key {k!r} — "
                        f"NOT consumed by the trainer (typo? renamed? it "
                        f"will be silently ignored)"
                    )
            # `count_truncations` is the registered name and
            # `truncation_is_failure` the original alias; the trainer
            # reads the former first, so carrying both means one of them
            # is inert — the exact bug class this validator exists for.
            if "count_truncations" in bwd and "truncation_is_failure" in bwd:
                warnings.append(
                    "reinforce.backward_curriculum sets both "
                    "'count_truncations' and its legacy alias "
                    "'truncation_is_failure'; only 'count_truncations' is "
                    "read — drop the alias"
                )
        clev = rl.get("consolidate_level")
        if isinstance(clev, dict):
            for k in clev:
                if k not in KNOWN_CONSOLIDATE_LEVEL_KEYS:
                    warnings.append(
                        f"unknown reinforce.consolidate_level key {k!r} — "
                        f"NOT consumed by the trainer (typo? renamed? it "
                        f"will be silently ignored)"
                    )
            probe = clev.get("probe")
            if isinstance(probe, dict):
                for k in probe:
                    if k not in KNOWN_CONSOLIDATE_PROBE_KEYS:
                        warnings.append(
                            f"unknown reinforce.consolidate_level.probe key "
                            f"{k!r} — NOT consumed by the trainer (typo? "
                            f"renamed? it will be silently ignored)"
                        )
            rule = clev.get("accept_rule")
            if rule is not None and str(rule) not in _ACCEPT_RULES:
                warnings.append(
                    f"reinforce.consolidate_level.accept_rule {rule!r} is not "
                    f"one of {sorted(_ACCEPT_RULES)} — the trainer raises on "
                    f"an unknown rule rather than silently point-accepting"
                )
        sil = rl.get("sil")
        if isinstance(sil, dict):
            for k in sil:
                if k not in KNOWN_SIL_KEYS:
                    warnings.append(
                        f"unknown reinforce.sil key {k!r} — NOT consumed by "
                        f"the trainer (typo? renamed? it will be silently "
                        f"ignored)"
                    )
        adv = rl.get("adversary")
        if isinstance(adv, dict):
            for k in adv:
                if k not in KNOWN_ADVERSARY_KEYS:
                    warnings.append(
                        f"unknown reinforce.adversary key {k!r} — NOT "
                        f"consumed by the trainer (typo? renamed? it will "
                        f"be silently ignored)"
                    )
            mode = adv.get("mode")
            if mode is not None and str(mode) not in _ADVERSARY_MODES:
                warnings.append(
                    f"reinforce.adversary.mode {mode!r} is not one of "
                    f"{sorted(_ADVERSARY_MODES)} — the trainer raises on an "
                    f"unknown mode rather than silently training without an "
                    f"adversary"
                )
    sv = profile.get("solve")
    if isinstance(sv, dict):
        for k in sv:
            if k not in KNOWN_SOLVE_KEYS:
                warnings.append(
                    f"unknown solve key {k!r} - NOT recognized by any "
                    f"validator (typo? renamed? it will be silently ignored)"
                )
    return warnings


def check_profile(profile: dict[str, Any], *, strict: bool = False,
                  logger=None) -> list[str]:
    """Validate and surface warnings (log.warning), or raise in strict.

    Returns the warning list. The trainer launcher calls this at startup
    so a misspelled knob is loud instead of silent.
    """
    warnings = validate_profile(profile)
    if warnings:
        msg = "profile schema: " + "; ".join(warnings)
        if strict:
            raise ConfigSchemaError(msg)
        if logger is not None:
            logger.warning("[config] %s", msg)
        else:
            print(f"[config] WARNING: {msg}")
    return warnings


class ReachabilityDerivationError(RuntimeError):
    """The inert-key derivation could not find the structures it parses.

    Raised rather than returning an empty result. An analysis that silently
    reports "nothing is inert" because a refactor moved the code it reads is
    a vacuous check, and this project has already shipped six of those.
    """


# Config-block names the trainer reads `reinforce` keys out of.
_CFG_NAMES = frozenset({"rl_cfg", "_rl_cfg", "reinforce_cfg", "reinforce"})


def _trainer_class(tree: ast.AST) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            return node
    raise ReachabilityDerivationError("no class Trainer in the trainer source")


def _self_calls(node: ast.AST) -> set[str]:
    """Names of `self.foo(...)` calls anywhere under `node`."""
    out: set[str] = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"):
            out.add(n.func.attr)
    return out


def _cfg_keys_in(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in _CFG_NAMES
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.add(n.args[0].value)
    return out


def vanilla_reachable_methods(trainer_src: Path) -> set[str]:
    """Trainer methods reachable from `run()` when `trainer_mode: vanilla_ppo`.

    `Trainer.run()` dispatches with an early return:

        if self.vanilla_ppo_mode:
            self._run_vanilla_ppo(...)
            return
        for gen in range(num_generations):
            self._run_one_generation(gen)   # <- never reached

    so the whole GA loop below that branch is dead in the mode EVERY banked
    learning-track run uses. This walks `run()`'s body up to and including
    the early-return branch, then takes the transitive closure over
    `self.foo(...)` calls.
    """
    tree = ast.parse(Path(trainer_src).read_text())
    cls = _trainer_class(tree)
    methods = {n.name: n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    run = methods.get("run")
    if run is None:
        raise ReachabilityDerivationError("Trainer has no run() method")

    entry: set[str] = set()
    found_dispatch = False
    for stmt in run.body:
        if (isinstance(stmt, ast.If)
                and any(isinstance(n, ast.Attribute)
                        and n.attr == "vanilla_ppo_mode"
                        for n in ast.walk(stmt.test))
                and any(isinstance(n, ast.Return) for n in ast.walk(stmt))):
            entry |= _self_calls(stmt)
            found_dispatch = True
            break
        entry |= _self_calls(stmt)
    if not found_dispatch:
        raise ReachabilityDerivationError(
            "could not find the `if self.vanilla_ppo_mode: ...; return` "
            "dispatch in Trainer.run() — the reachability derivation is "
            "reading a structure that no longer exists")

    graph = {name: _self_calls(m) for name, m in methods.items()}
    seen: set[str] = set()
    stack = list(entry)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(graph.get(name, ()))
    return seen


def inert_reinforce_keys_under_vanilla_ppo(
    trainer_src: Path, sibling_srcs: tuple[Path, ...] = (),
) -> dict[str, list[str]]:
    """`reinforce` keys the trainer parses but CANNOT act on under vanilla_ppo.

    Returns `{key: [methods that consume it]}` for every key whose every
    consumption site lies outside the vanilla-reachable call graph. These are
    knobs a profile can set, that `--strict-config` accepts, that appear in
    the run manifest, and that do nothing.

    Why this is derived from source rather than listed: the registry check
    the suite already runs (`consumed_reinforce_keys_from_source`) is
    ONE-WAY. It enforces "every key the trainer consumes is registered",
    which is why all 34 schema tests pass green while a dozen keys in the
    flagship recipe are inert. Nothing enforced the other direction, and a
    hand-maintained list of dead knobs would rot the first time someone
    moved a method.

    `sibling_srcs` are other modules that reach into the trainer by
    attribute (`ppo_updater.py` does `t.ppo_clip_eps`). Any attribute they
    touch is treated as live — the failure that matters here is a FALSE
    "this knob is inert", so the analysis errs toward silence.
    """
    trainer_src = Path(trainer_src)
    tree = ast.parse(trainer_src.read_text())
    cls = _trainer_class(tree)
    methods = {n.name: n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    reachable = vanilla_reachable_methods(trainer_src)

    # key -> the self.<attr> names it is assigned to
    key2attrs: dict[str, set[str]] = {}
    for meth in methods.values():
        for n in ast.walk(meth):
            if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                continue
            keys = _cfg_keys_in(n.value) if n.value is not None else set()
            if len(keys) != 1:
                continue  # ambiguous: two keys folded into one attribute
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    key2attrs.setdefault(next(iter(keys)), set()).add(t.attr)

    # Attributes any sibling module reads off a trainer handle are live.
    sibling_attrs: set[str] = set()
    for p in sibling_srcs:
        try:
            stree = ast.parse(Path(p).read_text())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(stree):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                sibling_attrs.add(n.attr)

    # attr -> methods that READ it (the __init__ assignment is not a use)
    def _readers(attr: str) -> set[str]:
        out: set[str] = set()
        for name, meth in methods.items():
            if name == "__init__":
                continue
            for n in ast.walk(meth):
                if (isinstance(n, ast.Attribute) and n.attr == attr
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                        and isinstance(n.ctx, ast.Load)):
                    out.add(name)
                    break
        return out

    inert: dict[str, list[str]] = {}
    for key, attrs in key2attrs.items():
        if attrs & sibling_attrs:
            continue
        sites: set[str] = set()
        for a in attrs:
            sites |= _readers(a)
        if not sites or (sites & reachable):
            continue
        inert[key] = sorted(sites)
    return inert


def consumed_reinforce_keys_from_source(trainer_src: Path) -> set[str]:
    """Re-derive the keys the trainer reads from `reinforce` config, by
    parsing `<cfg>.get("key"...)` sites. Used by the schema test to keep
    KNOWN_REINFORCE_KEYS honest — a newly-consumed key that is not
    registered fails the test."""
    txt = trainer_src.read_text()
    pat = re.compile(
        r'(?:rl_cfg|_rl_cfg|reinforce_cfg|reinforce)\.get\('
        r'\s*["\']([a-z0-9_]+)["\']'
    )
    return set(pat.findall(txt))


def consumed_backward_curriculum_keys_from_source(
    trainer_src: Path,
) -> set[str]:
    """Same, one level deeper: the `reinforce.backward_curriculum` keys the
    trainer reads (`_bwd_cfg.get("key"...)`). Used by the schema test so a
    new curriculum knob cannot ship unregistered."""
    txt = trainer_src.read_text()
    pat = re.compile(r'_bwd_cfg\.get\(\s*["\']([a-z0-9_]+)["\']')
    return set(pat.findall(txt))


def consumed_consolidate_level_keys_from_source(trainer_src: Path) -> set[str]:
    """The `reinforce.consolidate_level` keys the trainer reads
    (`_clevel_cfg.get("key"...)`). Same drift guard, applied to the block a
    pre-registered consolidation run is configured through."""
    txt = trainer_src.read_text()
    pat = re.compile(r'_clevel_cfg\.get\(\s*["\']([a-z0-9_]+)["\']')
    return set(pat.findall(txt))


def consumed_consolidate_probe_keys_from_source(probe_src: Path) -> set[str]:
    """The `reinforce.consolidate_level.probe` keys the resolver reads.

    Unlike the other two scanners this parses `oneshot_curriculum.py`, not
    the trainer: the probe block is resolved in ONE place
    (`resolve_probe_settings`) precisely so the gate's baseline probe and its
    per-cycle probe cannot drift apart the way v4's did.
    """
    txt = probe_src.read_text()
    pat = re.compile(r'cfg\.get\(\s*["\']([a-z0-9_]+)["\']')
    return set(pat.findall(txt))
