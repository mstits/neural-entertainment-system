"""Monotone (peak-augmented) wavefront shaping — pure helpers.

The legacy wavefront PBRS pays F = gamma*Phi(s') - Phi(s) live and charges
-peak(Phi) on any non-clearing terminal. Under
``reinforce.wave_terminal_rule: "monotone"`` the shaping stream switches to
the peak-augmented potential Phi~(s_t) = max over the episode so far of
Phi: the per-step term becomes

    F_t = gamma * max(peak, Phi(s_{t+1})) - peak

which is zero (at gamma=1) unless this step sets a NEW episode peak, and a
TRUE death charges -peak — the Grzes zero-terminal rule on the augmented
potential, so any advance-retreat-advance-death episode telescopes to
exactly zero net shaping. Stall/timeout cuts (the WAVE_LOST_K off-envelope
cut) are TRUNCATIONS instead: no terminal charge, and the GAE sweep
bootstraps V(s) rather than 0 (Pardo partial-episode bootstrapping —
`batched_gae`'s ``trunc_buf``).

Helpers are pure so the telescoping property is unit-testable without the
trainer loop; the trainer's inline wavefront block calls them per step.
"""
from __future__ import annotations

_VALID_RULES = ("monotone",)


def resolve_wave_terminal_rule(value) -> bool:
    """Map the ``wave_terminal_rule`` config value to "monotone on?".

    Absent/empty = legacy behavior (False). Anything other than
    "monotone" raises — a typo'd rule silently reverting to the legacy
    charge would burn an attended campaign window.
    """
    if value is None:
        return False
    rule = str(value).strip()
    if not rule:
        return False
    if rule not in _VALID_RULES:
        raise ValueError(
            f"reinforce.wave_terminal_rule {rule!r} is not one of "
            f"{list(_VALID_RULES)} (or absent for the legacy -peak rule)"
        )
    return True


def monotone_wave_step(
    peak: float, phi: float, gamma: float
) -> tuple[float, float]:
    """One live step of the monotone stream.

    Returns ``(F, new_peak)`` where ``F = gamma*max(peak, phi) - peak``.
    ``peak`` starts at 0.0 each episode, so the first step pays the spawn
    potential up front and the death charge refunds it exactly.
    """
    new_peak = phi if phi > peak else peak
    return gamma * new_peak - peak, new_peak


def wave_terminal_charge(peak: float, *, is_clear: bool, truncated: bool) -> float:
    """Terminal shaping term under the monotone rule.

    Clears keep their banked peak increments (the completion bonus is the
    reward); truncations charge nothing (the episode did not end — the
    value bootstrap carries it); true deaths refund the whole peak.
    """
    if is_clear or truncated or peak <= 0.0:
        return 0.0
    return -float(peak)
