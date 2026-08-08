"""Segment composition probe: from a banked state at gx X, how often does a
policy reach gx Y under the honest noise model?

This is the S3 measurement the whole World 1-2 verdict rests on
(`docs/research/RESULTS_1_2_HONEST_PROTOCOL_2026-07-24.md`: "S3 probe:
gx1500 -> gx2200 under sticky", 0/30 on every seed). The original lived only
in /tmp and is gone; this is that instrument, restored as a first-class
script.

Why it is separate from `eval_game.py`: eval scores a LEVEL predicate
(flagpole / area transition). A composition probe scores a SEGMENT predicate
-- "reached global x >= Y, alive, still in the area it started in" -- which is
the only way to ask whether verified local robustness composes into traversal.
Everything else (policy construction, obs assembly, sticky/jitter semantics,
the step loop's shape) is deliberately identical to `eval_game.py`, and the
tile/pixel policy wrappers and `select_action` are imported from it so
observation assembly and the action-selection vocabulary can never drift.

Start state selection:
  --start-state PATH   probe from an exact save-state file, or
  --from-gx N          pick the NEAREST state in --bank whose measured gx is
                       closest to N. Bank filenames only hint (`..._a2_gx1490
                       .state`); the chosen state's gx/area are re-measured
                       from RAM after loading and both the requested and the
                       measured values are reported.

Defaults are the honest protocol (sticky 0.25, start-jitter 16, 30 episodes)
because that is the only regime this instrument is for. Report greedy AND
sampled -- run it twice with --action-select.

Reproducing the published 1-2 negative (the B3 control):

    python scripts/segment_probe.py --game mario \\
        --profile configs/mario_1_2_cgsa.yaml \\
        --checkpoint checkpoints/mario_1_2_cgsa_s1/vanilla_ppo_iter_05990.pt \\
        --bank checkpoints/ge_1_2_rungs_gx --from-gx 1500 --from-area 2 \\
        --to-gx 2200 --episodes 30 --sticky-prob 0.25 --action-select greedy

Output is a single JSON object on stdout (progress goes to stderr) and a row
appended to `checkpoints/<game_slug>/segment_probe.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import FrameStacker, TileFeatureStacker  # noqa: E402
from src.models.policy_network import PolicyNetwork  # noqa: E402
from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, derive_checkpoint_dir, resolve_encoder,
)
from src.utils.reward_functions import build_reward_function  # noqa: E402
from eval_game import (  # noqa: E402
    _ACTION_SELECT_MODES, _PixelPolicy, _TilePolicy, resolve_eval_checkpoint,
    resolve_pixel_encoder, select_action,
)
from train_game import DEFAULT_PROFILES, DEFAULT_ROMS, resolve_profile_path  # noqa: E402

# SMB RAM the eval harness already reads for telemetry (same addresses as
# eval_game's gx/stage proxies and robustify_level's segment predicate). These
# are MEASUREMENT taps, never an observation or a reward input.
R_GX_HI = 0x006D
R_GX_LO = 0x0086
R_PSTATE = 0x000E       # 6 / 11 == dying-or-dead animation
R_AREA = 0x0760         # internal area index (NOT the printed level number)
_DYING = (6, 11)

# Bank filename -> area. Ordered most-specific first: the act-indexed bank
# names states `r_a438_ar2_gx1812` (a438 = action count, ar2 = area), so a
# bare `_a(\d+)` probe must be the LAST resort or it reads the action count
# as an area.
_AREA_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"_area(\d+)"),
    re.compile(r"_ar(\d+)"),
    re.compile(r"_a(\d+)"),
)
_GX_PATTERN = re.compile(r"gx(\d+)")


def gx_of(ram) -> int:
    """Global x (page << 8 | in-page x) — the harness-wide gx convention."""
    return (int(ram[R_GX_HI]) << 8) | int(ram[R_GX_LO])


def alive(ram) -> bool:
    """False once the death/shrink animation has started (robustify's
    segment predicate uses exactly this test)."""
    return int(ram[R_PSTATE]) not in _DYING


def bank_gx(path) -> Optional[int]:
    """`gx` hint parsed from a banked state's filename, or None."""
    m = _GX_PATTERN.search(Path(path).stem)
    return int(m.group(1)) if m else None


def bank_area(path) -> Optional[int]:
    """Area hint parsed from a banked state's filename, or None."""
    stem = Path(path).stem
    for pat in _AREA_PATTERNS:
        m = pat.search(stem)
        if m:
            return int(m.group(1))
    return None


def scan_bank(dirs) -> list[dict]:
    """Every `*.state` under `dirs` that carries a gx hint, as
    `{"path", "gx", "area"}` sorted by (gx, path). Non-recursive per dir —
    a bank is a flat rung directory."""
    out: list[dict] = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.state")):
            gx = bank_gx(f)
            if gx is None:
                continue
            out.append({"path": str(f), "gx": gx, "area": bank_area(f)})
    out.sort(key=lambda e: (e["gx"], e["path"]))
    return out


def select_bank_state(entries, from_gx: int,
                      area: Optional[int] = None) -> Optional[dict]:
    """Nearest banked state to `from_gx`, optionally filtered to `area`.

    Ties (a rung either side, equidistant) resolve to the SHALLOWER one so a
    repeat probe never silently gets an easier start, then by path so the
    choice is deterministic across filesystems.
    """
    pool = [e for e in entries if area is None or e["area"] == area]
    if not pool:
        return None
    return min(pool, key=lambda e: (abs(e["gx"] - from_gx), e["gx"], e["path"]))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k/n. Reported because this instrument's
    whole job is to distinguish "0 of 30" from "0" — 0/30 is
    [0.000, 0.114], which is exactly the gate-mirage the campaign was
    burned by (3-episode gates hiding true success < 1/400)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, (centre - margin) / denom),
            min(1.0, (centre + margin) / denom))


def summarize_episodes(records: list[dict], *, to_gx: int) -> dict:
    """Aggregate per-episode records into the probe's verdict block.

    `records`: dicts with "outcome" in {reached, died, timeout, left_area},
    "max_gx", "steps", "end_gx". Pure — unit-testable without an emulator.
    """
    n = len(records)
    max_gxs = [int(r["max_gx"]) for r in records]
    reached = [r for r in records if r["outcome"] == "reached"]
    k = len(reached)
    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    lo, hi = wilson_interval(k, n)
    steps_to_reach = [int(r["steps"]) for r in reached]
    return {
        "to_gx": int(to_gx),
        "episodes": n,
        "reached": k,
        "reach_rate": (k / n) if n else 0.0,
        "reach_ci95": [lo, hi],
        "outcomes": outcomes,
        "max_gx_per_episode": max_gxs,
        "mean_max_gx": float(np.mean(max_gxs)) if max_gxs else 0.0,
        "median_max_gx": float(np.median(max_gxs)) if max_gxs else 0.0,
        "min_max_gx": int(min(max_gxs)) if max_gxs else 0,
        "best_max_gx": int(max(max_gxs)) if max_gxs else 0,
        "mean_steps_to_reach": (
            float(np.mean(steps_to_reach)) if steps_to_reach else None
        ),
    }


def build_probe_policy(profile: dict, state: dict, bitmasks):
    """(policy, is_recurrent, pool_preprocess_f16) for a loaded checkpoint.

    Mirrors `eval_game.eval_one_game`'s construction exactly — same encoder
    dispatch, same `strict=False` load with a LOUD failure on missing core
    weights, same `_TilePolicy` / `_PixelPolicy` wrappers — so a probe scores
    the same policy the eval harness would. Raises ValueError on an
    unsupported profile or a partial load.
    """
    pixel_encoder: Optional[str] = None
    try:
        extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    except ValueError as tile_err:
        pixel_encoder = resolve_pixel_encoder(profile)
        if pixel_encoder is None:
            raise ValueError(f"unsupported profile: {tile_err}") from tile_err

    if pixel_encoder is None:
        net, is_recurrent = build_tile_policy_from_checkpoint(
            state, num_actions=len(bitmasks), feature_dim=stacked_dim,
        )
        load_res = net.load_state_dict(state["net_state_dict"], strict=False)
        if load_res.missing_keys:
            raise ValueError(
                "checkpoint does not provide all network weights for "
                f"{type(net).__name__}(num_actions={len(bitmasks)}, "
                f"feature_dim={stacked_dim}); missing="
                f"{list(load_res.missing_keys)}"
            )
        net.eval()
        stacker = TileFeatureStacker(
            stack_size=stacked_dim // feature_dim, feature_dim=feature_dim,
        )
        return _TilePolicy(net, stacker, extractor, is_recurrent), is_recurrent, False

    rl = profile.get("reinforce", {}) or {}
    preprocess_f16 = bool(rl.get("preprocess_f16", False))
    frame_stack = int(rl.get("frame_stack", 4))
    net = PolicyNetwork(
        num_actions=len(bitmasks), frame_stack=frame_stack, frame_size=84,
        encoder=pixel_encoder, use_layernorm=bool(rl.get("layernorm", True)),
    )
    load_res = net.load_state_dict(state["net_state_dict"], strict=False)
    if load_res.missing_keys:
        raise ValueError(
            "checkpoint does not provide all network weights for "
            f"{type(net).__name__}(num_actions={len(bitmasks)}, "
            f"encoder={pixel_encoder!r}, frame_stack={frame_stack}); missing="
            f"{list(load_res.missing_keys)}"
        )
    net.eval()
    stacker = FrameStacker(
        stack_size=frame_stack,
        dtype=np.float16 if preprocess_f16 else np.uint8,
    )
    return _PixelPolicy(net, stacker, preprocess_f16), False, preprocess_f16


def run_segment(pool, pol, bitmasks, blob, reward_fn, device, *, to_gx: int,
                episodes: int, sticky_prob: float, start_jitter: int,
                action_select: str, eval_seed: int, max_steps: int,
                temperature: float = 1.0,
                progress=None) -> tuple[list[dict], dict]:
    """Run `episodes` segment attempts from `blob`; return (records, start).

    One episode = load the banked state, optional Machado no-op jitter, then
    step until one of:
      reached    gx >= to_gx, alive, still in the start area  (the SUCCESS)
      died       reward fn terminal or the pool's auto-reset done flag
      left_area  the area byte changed (gx restarts in the new area, so the
                 segment measurement is over — reported, never counted)
      timeout    max_steps exhausted

    The action stream is eval_game's, bit-for-bit: one `RandomState(eval_seed)`
    consumed first by the jitter draw and then by the sticky draws, plus one
    `torch.Generator(eval_seed)` for `--action-select sampled` (eval_game's
    `select_action` does the drawing), so a run here and an `eval_game` run
    with the same seed see the same perturbation sequence.
    """
    rng = np.random.RandomState(eval_seed)
    action_gen = torch.Generator(device="cpu")
    action_gen.manual_seed(int(eval_seed))
    records: list[dict] = []
    start_info: dict = {}
    for ep in range(episodes):
        pool.reset_all()
        pool.load_worker_state(0, blob)
        reward_fn.reset()
        # load_worker_state restores RAM but produces no frame until the next
        # step; the noop flushes the post-restore frame for the stacker seed
        # (the load/noop/reset convention the trainer and eval both use).
        init = pool.step_all(np.zeros(1, dtype=np.uint8))
        # Report the rung's OWN gx (pre-jitter): the no-op frames drift Mario a
        # few px, and a start position that moves with --start-jitter would
        # make two probes of the same rung look like different segments.
        if not start_info:
            start_info = {"start_gx": gx_of(init[0][2]),
                          "start_area": int(init[0][2][R_AREA])}
        if start_jitter > 0:
            for _ in range(int(rng.randint(0, start_jitter + 1))):
                init = pool.step_all(np.zeros(1, dtype=np.uint8))
        ram0 = init[0][2]
        start_gx = gx_of(ram0)
        start_area = int(ram0[R_AREA])
        obs = pol.reset(init[0])
        hidden = pol.initial_hidden(device)
        ep_max_gx = start_gx
        end_gx = start_gx
        outcome = "timeout"
        prev_action_idx = 0
        step = -1                       # so `step + 1` is 0 steps, not 1, when
        for step in range(max_steps):   # max_steps is 0 and the loop never runs
            with torch.no_grad():
                logits, hidden = pol.logits(obs, hidden)
                action_idx = select_action(
                    logits, action_select, temperature, action_gen,
                )
            if sticky_prob > 0.0 and step > 0 and rng.random() < sticky_prob:
                action_idx = prev_action_idx
            prev_action_idx = action_idx
            bitmask = bitmasks[action_idx]
            r = pool.step_all(np.array([bitmask], dtype=np.uint8))
            ram = r[0][2]
            _, rew_done, _ = reward_fn.compute(ram, action=int(bitmask))
            obs = pol.push(r[0])
            area = int(ram[R_AREA])
            if area != start_area:
                outcome = "left_area"
                break
            end_gx = gx_of(ram)
            if end_gx > ep_max_gx:
                ep_max_gx = end_gx
            if end_gx >= to_gx and alive(ram):
                outcome = "reached"
                break
            if rew_done or bool(r[0][3]):
                outcome = "died"
                break
        records.append({
            "outcome": outcome, "max_gx": ep_max_gx,
            "steps": step + 1, "end_gx": end_gx,
        })
        if progress is not None:
            progress(ep, records[-1])
    return records, start_info


def probe(game: str, profile_path: Path, rom_path: str, *, to_gx: int,
          from_gx: Optional[int] = None, from_area: Optional[int] = None,
          banks: Optional[list[str]] = None, start_state: Optional[str] = None,
          episodes: int = 30, sticky_prob: float = 0.25,
          start_jitter: int = 16, action_select: str = "greedy",
          temperature: float = 1.0, eval_seed: int = 0, max_steps: int = 1500,
          max_gx_delta: int = 200, latest: bool = False,
          iter_: Optional[int] = None, checkpoint: Optional[str] = None,
          progress=None) -> dict:
    """Resolve checkpoint + start state, run the probe, return the result dict."""
    if action_select not in _ACTION_SELECT_MODES:
        raise ValueError(
            f"action_select must be one of {sorted(_ACTION_SELECT_MODES)}, "
            f"got {action_select!r}"
        )
    if not (float(temperature) > 0.0):
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    with open(profile_path) as f:
        profile = yaml.safe_load(f)
    ckpt_dir = derive_checkpoint_dir("./checkpoints", profile.get("name"))
    ckpt = resolve_eval_checkpoint(
        ckpt_dir, game, latest=latest, iter_=iter_, checkpoint=checkpoint,
    )
    if ckpt is None:
        return {"game": game, "status": "no_checkpoint", "ckpt_dir": str(ckpt_dir)}

    # --- start state: explicit path, or nearest rung in the bank ------------
    bank_entry: Optional[dict] = None
    if start_state is not None:
        ss_path = Path(start_state)
    else:
        if from_gx is None:
            return {"game": game, "status": "no_start_state",
                    "detail": "pass --start-state PATH or --from-gx N (+ --bank DIR)"}
        entries = scan_bank(banks or [])
        if not entries:
            return {"game": game, "status": "empty_bank",
                    "banks": list(banks or []),
                    "detail": "no *.state files with a gx hint in --bank"}
        bank_entry = select_bank_state(entries, from_gx, area=from_area)
        if bank_entry is None:
            return {"game": game, "status": "empty_bank",
                    "banks": list(banks or []), "from_area": from_area,
                    "detail": f"no banked state in area {from_area}"}
        if abs(bank_entry["gx"] - from_gx) > max_gx_delta:
            # Silently probing from gx 0 because the bank has nothing near the
            # requested rung would fabricate a much harder (or easier) segment.
            return {"game": game, "status": "no_bank_state_near_gx",
                    "from_gx": from_gx, "nearest": bank_entry,
                    "max_gx_delta": max_gx_delta,
                    "detail": "nearest banked state is farther than --max-gx-delta"}
        ss_path = Path(bank_entry["path"])
    if not ss_path.exists():
        return {"game": game, "status": "no_such_start_state",
                "start_state": str(ss_path)}
    blob = ss_path.read_bytes()

    try:
        bitmasks = action_space_to_bitmasks(profile["action_space"])
    except (KeyError, ValueError) as e:
        return {"game": game, "status": "unsupported_profile", "detail": str(e)}
    state = torch.load(str(ckpt), map_location="cpu")
    device = torch.device("cpu")
    try:
        pol, is_recurrent, preprocess_f16 = build_probe_policy(profile, state, bitmasks)
    except ValueError as e:
        return {"game": game, "status": "checkpoint_mismatch",
                "checkpoint": str(ckpt), "detail": str(e)}

    profile_start = profile.get("start_state_path") or (
        Path(rom_path).with_name(Path(rom_path).stem + "_start.state.bin")
    )
    pool = Pool(
        rom_path=rom_path, num_workers=1, frame_skip=4,
        start_state_path=(
            str(profile_start) if Path(str(profile_start)).exists() else None
        ),
    )
    if preprocess_f16:
        pool.set_preprocess_f16(True)
    reward_fn = build_reward_function(profile)
    try:
        records, start_info = run_segment(
            pool, pol, bitmasks, blob, reward_fn, device, to_gx=to_gx,
            episodes=episodes, sticky_prob=sticky_prob,
            start_jitter=start_jitter, action_select=action_select,
            temperature=temperature, eval_seed=eval_seed,
            max_steps=max_steps, progress=progress,
        )
    finally:
        pool.shutdown()

    result = {
        "game": game,
        "status": "ok",
        "profile": str(profile_path),
        "checkpoint": str(ckpt),
        "recurrent": bool(is_recurrent),
        "start_state": str(ss_path),
        "from_gx_requested": from_gx,
        "from_area_requested": from_area,
        # Measured from RAM after the load — the filename is only a hint, and
        # the post-load noop (plus jitter) advances gx a few px.
        "from_gx_measured": start_info.get("start_gx"),
        "from_area_measured": start_info.get("start_area"),
        "sticky_prob": float(sticky_prob),
        "start_jitter": int(start_jitter),
        "action_select": str(action_select),
        "temperature": float(temperature),
        "eval_seed": int(eval_seed),
        "max_steps": int(max_steps),
        "timestamp": time.time(),
    }
    if bank_entry is not None:
        result["bank_entry"] = bank_entry
    result.update(summarize_episodes(records, to_gx=to_gx))
    if from_area is not None and start_info.get("start_area") != from_area:
        # The bank's filename hint disagreed with RAM: the probe ran, but it
        # ran somewhere else than asked. Loud in the record, never silent.
        result["status"] = "bank_area_mismatch"
    log = ckpt_dir / "segment_probe.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Segment composition probe: gx X -> gx Y under sticky actions.")
    ap.add_argument("--game", type=str, default="mario",
                    help=f"Game logical name: {sorted(set(DEFAULT_PROFILES))}")
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--rom", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                    help="Probe an exact checkpoint file by path.")
    ap.add_argument("--iter", type=int, default=None, dest="iter_", metavar="N",
                    help="Probe an exact vanilla_ppo_iter_NNNNN.pt.")
    ap.add_argument("--latest", action="store_true",
                    help="Probe the freshest trained checkpoint instead of the "
                         "retained winner.")
    ap.add_argument("--from-gx", type=int, default=None, dest="from_gx",
                    help="Start from the banked state whose gx is nearest N.")
    ap.add_argument("--from-area", type=int, default=None, dest="from_area",
                    help="Restrict the bank to this area index (banks mix the "
                         "1-2 overworld stub and the underground main area at "
                         "overlapping gx).")
    ap.add_argument("--bank", type=str, action="append", default=None,
                    metavar="DIR",
                    help="Directory of banked *.state rungs (repeatable). "
                         "Default: the profile's checkpoint dir.")
    ap.add_argument("--start-state", type=str, default=None, dest="start_state",
                    metavar="PATH",
                    help="Probe from an exact save-state file (bypasses --from-gx).")
    ap.add_argument("--to-gx", type=int, required=True, dest="to_gx",
                    help="Success = reaching this global x alive, in the start area.")
    ap.add_argument("--episodes", type=int, default=30,
                    help="Protocol minimum is 30; 3-episode gates are mirages.")
    ap.add_argument("--sticky-prob", type=float, default=0.25, dest="sticky_prob",
                    help="Machado sticky-actions probability (protocol: 0.25).")
    ap.add_argument("--start-jitter", type=int, default=16, dest="start_jitter",
                    help="Machado no-op starts: up to this many no-op frames.")
    ap.add_argument("--action-select", type=str, default="greedy",
                    choices=list(_ACTION_SELECT_MODES), dest="action_select",
                    help="argmax, or a seeded draw from the policy's own "
                         "softmax. Same vocabulary as eval_game — report BOTH.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for --action-select sampled.")
    ap.add_argument("--eval-seed", type=int, default=0, dest="eval_seed")
    ap.add_argument("--max-steps", type=int, default=1500, dest="max_steps")
    ap.add_argument("--max-gx-delta", type=int, default=200, dest="max_gx_delta",
                    help="Refuse to probe if the nearest banked state is this "
                         "much farther than --from-gx.")
    ap.add_argument("--json-out", type=str, default=None, dest="json_out",
                    metavar="PATH", help="Also write the result JSON here.")
    args = ap.parse_args()

    profile_path = resolve_profile_path(args.game, args.profile)
    rom_path = args.rom or DEFAULT_ROMS.get(args.game.lower().strip())
    if rom_path is None or not Path(rom_path).exists():
        print(json.dumps({"game": args.game, "status": "no_rom",
                          "rom_path": rom_path}))
        return 1
    if not (args.temperature > 0.0):
        print(json.dumps({"game": args.game, "status": "bad_temperature",
                          "detail": f"--temperature must be > 0, got "
                                    f"{args.temperature}"}))
        return 1
    banks = args.bank
    if banks is None:
        with open(profile_path) as f:
            banks = [str(derive_checkpoint_dir(
                "./checkpoints", (yaml.safe_load(f) or {}).get("name")))]

    def _progress(ep: int, rec: dict) -> None:
        print(f"[segment] ep {ep + 1}/{args.episodes} {rec['outcome']} "
              f"max_gx={rec['max_gx']} steps={rec['steps']}",
              file=sys.stderr, flush=True)

    result = probe(
        args.game, profile_path, rom_path, to_gx=args.to_gx,
        from_gx=args.from_gx, from_area=args.from_area, banks=banks,
        start_state=args.start_state, episodes=args.episodes,
        sticky_prob=args.sticky_prob, start_jitter=args.start_jitter,
        action_select=args.action_select, temperature=args.temperature,
        eval_seed=args.eval_seed,
        max_steps=args.max_steps, max_gx_delta=args.max_gx_delta,
        latest=args.latest, iter_=args.iter_, checkpoint=args.checkpoint,
        progress=_progress,
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
