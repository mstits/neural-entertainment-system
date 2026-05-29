"""Helpers for per-game profile handling.

Centralizes the rules for translating a profile's `name` field into
a filesystem-safe slug used for per-game checkpoint directories,
metric files, eval outputs, and any other on-disk artifact.

The slug is deterministic: `"Super Mario Bros."` → `"super_mario_bros"`,
`"The Legend of Zelda"` → `"the_legend_of_zelda"`, `"Mega Man"` →
`"mega_man"`. Stable across runs so the trainer's auto-resume logic
can find the latest checkpoint after a code change without manual
intervention.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


_SLUG_TRIM = re.compile(r"[^a-z0-9]+")

# Canonical controller-button name → NES bitmask. Mirrors the nes-py
# layout pinned by `src.emulation.frame_utils` and guarded by
# `tests/test_button_bit_alignment.py`. Duplicated here (rather than
# imported at module load) so profile_utils stays import-light for the
# launcher/eval scripts; the values are asserted against frame_utils in
# the unit tests so they can never silently drift.
BUTTON_NAME_TO_BIT: dict[str, int] = {
    "NOOP": 0x00,
    "A": 0x01,
    "B": 0x02,
    "select": 0x04,
    "start": 0x08,
    "up": 0x10,
    "down": 0x20,
    "left": 0x40,
    "right": 0x80,
}


def profile_slug(name: Optional[str]) -> str:
    """Return a filesystem-safe slug for a profile's display name.

    Lowercases the input, replaces any run of non-alphanumerics with
    a single underscore, strips leading/trailing underscores. Empty
    or None input returns `"unknown"` so the caller always gets a
    valid path component.
    """
    if not name:
        return "unknown"
    slug = _SLUG_TRIM.sub("_", name.lower()).strip("_")
    return slug or "unknown"


def derive_checkpoint_dir(
    base: str | Path,
    profile_name: Optional[str],
) -> Path:
    """Resolve the checkpoint directory for a profile.

    When `base` is the legacy default (`./checkpoints` or any string
    equal to it), append the profile's slug as a subdirectory. When
    `base` is anything else (an explicit override), use it verbatim.

    This preserves backwards compatibility for callers that pass an
    explicit non-default `checkpoint_dir` (e.g., test fixtures), and
    automatically gives every game its own subtree when callers
    don't override.
    """
    base_p = Path(base)
    is_default = base_p == Path("./checkpoints") or base_p == Path("checkpoints")
    if not is_default:
        return base_p
    return base_p / profile_slug(profile_name)


def action_space_to_bitmasks(action_space: list) -> tuple[int, ...]:
    """Convert a profile's `action_space` to a tuple of NES bitmasks.

    `action_space` is a list of button-name lists, e.g.
    `[[], ["right"], ["right", "A"]]`. Each inner list is OR-folded
    into a single controller bitmask. The empty list is NOOP (0x00).

    Centralizes the conversion so the trainer (hot path) and the
    headless eval/launch scripts agree on exactly one mapping. The
    trainer's `_build_bitmask_table` delegates here.

    Raises ValueError on a bare-string entry (a common YAML mistake:
    `- right+A` instead of `- [right, A]`, which would iterate
    char-by-char and explode) or an unknown button name.
    """
    table: list[int] = []
    for idx, buttons in enumerate(action_space):
        if isinstance(buttons, str):
            raise ValueError(
                f"action_space[{idx}] is a string ({buttons!r}); each "
                f"entry must be a list of button names (or [] for NOOP). "
                f'Example: ["right", "A"].'
            )
        bitmask = 0
        for b in buttons:
            if b not in BUTTON_NAME_TO_BIT:
                raise ValueError(
                    f"action_space[{idx}] references unknown button {b!r}; "
                    f"valid names: {sorted(BUTTON_NAME_TO_BIT.keys())}"
                )
            bitmask |= BUTTON_NAME_TO_BIT[b]
        table.append(bitmask)
    return tuple(table)


def validate_profile(profile: dict) -> list[str]:
    """Validate a game profile's load-bearing fields; return a list of
    human-readable problems (empty list == valid).

    Lightweight schema check (no external dep) for the keys the trainer
    and launcher actually depend on, so a malformed YAML fails at load
    with a clear message instead of crashing mid-run or — worse —
    training silently on bad config. Unknown/extra keys are allowed
    (profiles carry game-specific extensions by design).
    """
    problems: list[str] = []
    if not isinstance(profile, dict):
        return ["profile root is not a mapping"]

    name = profile.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append("name: must be a non-empty string")

    action_space = profile.get("action_space")
    if action_space is not None:
        try:
            action_space_to_bitmasks(action_space)
        except (ValueError, TypeError) as e:
            problems.append(f"action_space: {e}")

    rw = profile.get("reward_weights")
    if rw is not None and not isinstance(rw, dict):
        problems.append("reward_weights: must be a mapping")

    rl = profile.get("reinforce")
    if rl is not None:
        if not isinstance(rl, dict):
            problems.append("reinforce: must be a mapping")
        else:
            enc = rl.get("encoder")
            if enc is not None and not isinstance(enc, str):
                problems.append("reinforce.encoder: must be a string")
            # Hyperparameters, when present, must be numeric — a stray
            # quote ("3e-4" vs 3e-4) otherwise blows up at optimizer build.
            for k in (
                "lr", "gamma", "gae_lambda", "ppo_clip_eps", "value_coef",
                "entropy_coef", "grad_clip", "rnd_intrinsic_coef",
                "rnd_loss_coef", "rollout_steps", "steps", "ppo_minibatch_size",
                "tile_frame_stack",
            ):
                # bool is an int subclass — reject it explicitly so a
                # YAML `lr: true` is caught here, not at optimizer build.
                if k in rl and (
                    isinstance(rl[k], bool) or not isinstance(rl[k], (int, float))
                ):
                    problems.append(
                        f"reinforce.{k}: must be a number, got "
                        f"{type(rl[k]).__name__} ({rl[k]!r})"
                    )
    return problems


def resolve_encoder(profile: dict) -> tuple[Any, int, int]:
    """Resolve a profile's observation encoder from `reinforce.encoder`.

    Returns `(extractor, feature_dim, stacked_dim)` where:
      * `extractor` is a `TileObservation` instance (has `.extract`),
      * `feature_dim` is the per-frame feature length,
      * `stacked_dim` is `feature_dim * tile_frame_stack` — the actual
        input width the policy network sees.

    The stack size comes from `reinforce.tile_frame_stack` (default 4).
    Reads the same keys the trainer uses so eval/launch and training
    build identically-shaped observations. Raises ValueError when the
    profile names no encoder or an unknown one (pixel encoders are not
    routed here yet — they have their own CNN path in the trainer).
    """
    from src.emulation.tile_observations import get_extractor

    rl = profile.get("reinforce", {}) or {}
    name = rl.get("encoder")
    if not name:
        raise ValueError(
            "profile has no reinforce.encoder; cannot resolve an "
            "observation encoder for headless eval/launch."
        )
    extractor = get_extractor(name)
    feature_dim = int(extractor.feature_dim)
    stack_size = int(rl.get("tile_frame_stack", 4))
    return extractor, feature_dim, feature_dim * stack_size
