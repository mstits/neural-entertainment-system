"""Measure a clear signal's NULL from a profile's own play, and refuse to guess.

WHY THIS EXISTS. Three of the six clear signals refuse a global default in
their own docstrings, in writing:

  * SceneCutSignal -- "NO DEFENSIBLE GLOBAL DEFAULT for scene_min/blank_min
    ... A profile must measure its own null d_scene/d_blank distribution ...
    and set both minimums above the observed null, or declare the signal
    `enabled: false` with a reason -- never guess a number."
  * entity_wipe_windows -- "min_bytes and tol MUST come from a measured
    per-profile null ... A profile whose null has no separable upper tail
    should declare this signal `enabled: false` rather than pick a number."
  * LockReleaseNoveltyTrack -- lock_max and m are "REQUIRED, no default".

Every document in this campaign that cites a null says "scripts/
clear_calibrate.py does not exist". It does now, for the first of those
three. That matters more than its size: COORD_RESET_DROP_MIN = 300 is an
SMB-shaped constant that was applied to 45 games and made `coord`
arithmetically dead on most of them, and the only thing that stops the
replacement signals repeating it is a measurement per profile.

WHAT IT MEASURES. The largest per-check window delta this profile's own
ordinary play produces, for the two quantities SceneCutSignal gates on:

    d_scene  -- change in the odometer scene ordinal across the window
    d_blank  -- dropped (blackout) folds across the window

over the SAME window/stride arithmetic SceneCutSignal itself uses, so the
number measured is the number the signal will be compared against, not a
proxy for it. The recommended gate is then the smallest integer strictly
above the observed null -- and when the null is not separable (it already
reaches the magnitudes a real transition produces), the recommendation is
`enabled: false` with the measured reason attached, not a bigger number.

TWO SOURCES OF "ORDINARY PLAY", and the difference is the whole reliability
of the answer:

  --drive       NOOP + forward-hold from the profile's own start state.
                Cheap, available for every profile, and WEAK: a few hundred
                frames of standing still is not evidence about what an hour
                of real play does.
  --runs        the PRE-CLEAR portion of banked solution tapes. Strong: it
                is this game's real play, at the real frame cadence, for
                thousands of frames, and it is scored only up to the
                adapter's own clear predicate so the transition being
                calibrated FOR cannot leak into the null it is measured
                against. Use this whenever a tape exists.

WHAT WOULD THIS REPORT IF THE MECHANISM WERE ABSENT? A profile whose scene
ordinal and blank counter never move at all reports a null of (0, 0) and a
gate of (1, 1) -- and that is indistinguishable, HERE, from a game whose
odometer is not wired. So the receipt also carries `n_checks` and the
POST-clear deltas when a tape supplies a clear frame: a null of 0 next to a
post-clear delta of 0 means the surface is silent for this game and the
signal must be declined, while a null of 0 next to a post-clear delta of 4
means the surface separates. Reporting the gate without that second column
would be a number travelling without its meaning, which is the failure this
whole campaign is named after.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nes_core  # noqa: E402
from clear_detect import SCENE_CUT_STRIDE, SCENE_CUT_WINDOW  # noqa: E402
from go_explore_solve import make_game  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402


class SceneCutNull:
    """Rolling max of (d_scene, d_blank) over SceneCutSignal's own window.

    Deliberately a separate, tiny reimplementation of the window arithmetic
    rather than a call into SceneCutSignal: the signal needs a gate to be
    constructed, and asking a gated instrument what its null is would mean
    the answer depended on the guess being calibrated away."""

    def __init__(self, window: int = SCENE_CUT_WINDOW,
                 stride: int = SCENE_CUT_STRIDE):
        self.window = max(2, int(window))
        self.stride = max(1, int(stride))
        self._buf: deque = deque(maxlen=self.window)
        self._n = 0
        self.max_scene = 0
        self.max_blank = 0
        self.n_checks = 0

    def push(self, scene: int, blank: int) -> None:
        self._n += 1
        self._buf.append((int(scene), int(blank)))
        if self._n % self.stride or len(self._buf) < 2:
            return
        self.n_checks += 1
        self.max_scene = max(self.max_scene, self._buf[-1][0] - self._buf[0][0])
        self.max_blank = max(self.max_blank, self._buf[-1][1] - self._buf[0][1])

    def as_dict(self) -> dict:
        return {"max_d_scene": self.max_scene, "max_d_blank": self.max_blank,
                "n_checks": self.n_checks, "window": self.window,
                "stride": self.stride}


def _env_for(profile: dict):
    game = make_game(profile)
    env = nes_core.NESEnvironment(game.rom, frame_skip=1)
    env.reset()
    env.set_odometer_enabled(True)
    return game, env


def measure_drive(profile: dict, noop_frames: int = 400,
                  hold_frames: int = 80) -> dict:
    """NOOP then forward-hold from the profile's own start state.

    The forward-hold is deliberately SHORT: continued long enough it runs
    the player into a hazard, and a death is a REAL re-anchor event, not
    seam noise -- it would poison the very null it is meant to measure.
    (tests/test_scene_cut_signal.py makes the same split for the same
    reason, and names the measured SMB case.)"""
    game, env = _env_for(profile)
    space = [list(a) for a in profile["action_space"]]
    masks = action_space_to_bitmasks(space)
    fwd = next((i for i, a in enumerate(space) if "right" in a), 0)
    start = profile.get("start_state_path")
    try:
        if start and (REPO / start).exists():
            env.load_state((REPO / start).read_bytes())
        null = SceneCutNull()
        for _ in range(noop_frames):
            env.step(0)
            null.push(env.get_odometer_scene(), env.get_odometer_blank())
        for _ in range(hold_frames):
            env.step(int(masks[fwd]))
            null.push(env.get_odometer_scene(), env.get_odometer_blank())
    finally:
        env.close()
    out = null.as_dict()
    out.update(source="drive", noop_frames=noop_frames,
               hold_frames=hold_frames,
               start_state=start, post_clear=None)
    return out


def measure_tape(profile: dict, base: str) -> dict:
    """The PRE-CLEAR portion of one banked solution tape.

    Scored only up to the adapter's own clear predicate, so the transition
    this gate exists to catch cannot leak into the null it is measured
    against. The post-clear deltas are reported alongside as the
    separability column -- a null of 0 means nothing on its own."""
    game, env = _env_for(profile)
    fs = int(profile.get("frame_skip", 4))
    space = [list(a) for a in profile["action_space"]]
    masks = action_space_to_bitmasks(space)
    meta = json.loads(Path(str(base) + ".json").read_text())
    actions = np.load(str(base) + ".actions.npy").tolist()
    null, post = SceneCutNull(), SceneCutNull()
    truth, f = None, -1
    try:
        env.load_state((REPO / meta["root_state"]).read_bytes())
        for _ in range(fs):
            env.step(0)
        ram0 = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
        if hasattr(game, "note_start"):
            game.note_start(ram0)
        start_wd = tuple(game.level_key(ram0))
        ctx: dict = {}
        for a in list(actions) + [0] * 90:
            m = int(masks[int(a)])
            for _ in range(fs):
                env.step(m)
                f += 1
                ram = np.array(env.get_ram_range(0, 2048), dtype=np.uint8)
                if truth is None and (game.is_clear(start_wd, ram, ctx)
                                      or game.is_finale(start_wd, ram)):
                    truth = f
                (null if truth is None else post).push(
                    env.get_odometer_scene(), env.get_odometer_blank())
    finally:
        env.close()
    out = null.as_dict()
    out.update(source="tape", tape=str(base), true_clear_frame=truth,
               n_frames=f + 1, post_clear=post.as_dict())
    return out


def recommend(nulls: list[dict]) -> dict:
    """The gate, or a refusal. One integer above the worst observed null.

    A null that already reaches what the post-clear window produces is NOT
    separable, and the honest output there is `enabled: false` carrying the
    measurement -- never a larger number chosen until the null fits under
    it."""
    n_scene = max(n["max_d_scene"] for n in nulls)
    n_blank = max(n["max_d_blank"] for n in nulls)
    posts = [n["post_clear"] for n in nulls if n.get("post_clear")]
    p_scene = max((p["max_d_scene"] for p in posts), default=None)
    p_blank = max((p["max_d_blank"] for p in posts), default=None)
    scene_min, blank_min = n_scene + 1, n_blank + 1
    separable = None
    if posts:
        separable = (p_scene >= scene_min) or (p_blank >= blank_min)
    out = {"null_max_d_scene": n_scene, "null_max_d_blank": n_blank,
           "post_clear_max_d_scene": p_scene, "post_clear_max_d_blank": p_blank,
           "separable": separable,
           "n_checks": sum(n["n_checks"] for n in nulls)}
    if separable is False:
        out["recommendation"] = {"enabled": False}
        out["reason"] = (
            f"NOT SEPARABLE: the measured null reaches d_scene={n_scene} / "
            f"d_blank={n_blank} and the post-clear window only reaches "
            f"d_scene={p_scene} / d_blank={p_blank}, so no gate separates "
            "ordinary play from the transition. Declining the signal is the "
            "honest output; a larger number chosen until the null fits under "
            "it would be COORD_RESET_DROP_MIN again.")
        return out
    out["recommendation"] = {"scene_min": scene_min, "blank_min": blank_min}
    out["reason"] = (
        f"measured null over {out['n_checks']} checks of this profile's own "
        f"play: max d_scene={n_scene}, max d_blank={n_blank}. The gate is "
        f"the smallest integer strictly above it."
        + ("" if separable is None else
           f" Separability: the post-clear window reaches d_scene={p_scene} / "
           f"d_blank={p_blank}, which clears the gate."))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True, help="configs/*.yaml")
    ap.add_argument("--signal", default="scene_cut", choices=("scene_cut",),
                    help="only scene_cut is implemented; entity_wipe and "
                         "lock_release_novelty still have no calibrator and "
                         "must not be armed without one")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="solution basenames (without .json/.actions.npy). "
                         "Strongly preferred over --drive.")
    ap.add_argument("--drive", action="store_true",
                    help="fall back to a NOOP + forward-hold drive from the "
                         "profile's own start state")
    ap.add_argument("--out", default=None, help="write the receipt here")
    args = ap.parse_args(argv)

    path = REPO / args.profile if not Path(args.profile).is_absolute() \
        else Path(args.profile)
    prof = yaml.safe_load(path.read_text())
    nulls = []
    if args.runs:
        for base in args.runs:
            b = base if Path(base).is_absolute() else str(REPO / base)
            nulls.append(measure_tape(prof, b))
    if args.drive or not nulls:
        nulls.append(measure_drive(prof))
    rec = recommend(nulls)
    receipt = {"profile": args.profile, "signal": args.signal,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "measurements": nulls, **rec}
    print(json.dumps(receipt, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[clear_calibrate] receipt written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
