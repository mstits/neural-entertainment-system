"""Composite (level-keyed hierarchical) evaluator for the SMB one-shot cold run.

Plays ONE cold emulator session from the 1-1 start and switches which policy
checkpoint produces actions based on the CURRENT displayed level read from RAM —
the level-keyed hierarchical policy that is the one-shot push's banker. Per-level
specialists are composed into the single cold run the campaign's definition of
done demands, and the result is reported HONESTLY as a hierarchical policy
(``"hierarchical": true`` in the JSON), never as a monolithic net.

Usage:
    python scripts/eval_composite.py --manifest configs/composite_scout.yaml
    python scripts/eval_composite.py --manifest m.yaml --episodes 4 --max-steps 2400

Manifest format (YAML):

    name: composite_scout        # optional; names the eval.jsonl subdir
    game: mario                  # optional (default: mario) — DEFAULT_ROMS lookup
    rom: <path>                  # optional; else DEFAULT_ROMS[game]
    start_state: <path>          # optional; else the scoring level's profile start
    frame_skip: 4                # optional; else the scoring profile's / 4
    hysteresis_k: 2              # optional; consecutive-frame switch threshold
    levels:
      "1-1": {ckpt: <path>, profile: <path>}
      "1-2": {ckpt: <path>, profile: <path>}
      "1-3": {ckpt: <path>, profile: <path>}
      "1-4": {ckpt: <path>, profile: <path>}
      default: {ckpt: <path>, profile: <path>}   # catch-all for unmapped levels

Each level maps to a (checkpoint, profile) pair; the profile drives that level's
observation encoder (tile OR pixel) and action space, so MIXED encoders compose
in one session. The router reads the displayed (world, level) each step (the same
RAM semantics `smb_sequential` uses), resolves the level -> net (exact, else
``default``), and switches only after K=2 consecutive agreeing frames so a
transient byte flicker at a transition can't thrash nets. On a committed switch
the incoming net's frame/feature stack is rebuilt and reset-seeded from the
current frame (a scene-cut reset; see `composite_policy`).

Scoring is a full `SequentialTracker` over the session (seq_clear, furthest_seq,
furthest_any, warp_taken) plus per-level segment stats (entered at / cleared /
died / timeout). A single JSON line is printed to stdout and a row is appended to
``<out-dir>/eval.jsonl`` (composite-specific dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.emulation.rust_pool_adapter import RustPool  # noqa: E402
from src.training.composite_policy import (  # noqa: E402
    CompositeController, build_level_net, run_episode, summarize,
)
from src.training.profile_utils import profile_slug  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402

# Reuse the launcher's logical-name -> ROM table so eval agrees with train.
sys.path.insert(0, str(ROOT / "scripts"))
from train_game import DEFAULT_ROMS  # noqa: E402


def _err(**kw) -> dict:
    return kw


def _scoring_key(levels: dict) -> str:
    """Pick the level whose profile drives the cold start / reward fn / frame skip.

    Prefer ``"1-1"`` (the DoD cold-start level), then ``"default"``, then the
    lowest-ranked mapped level.
    """
    if "1-1" in levels:
        return "1-1"
    if "default" in levels:
        return "default"
    def rank(k: str):
        try:
            w, l = k.split("-", 1)
            return (int(w), int(l))
        except (ValueError, AttributeError):
            return (99, 99)
    return sorted(levels.keys(), key=rank)[0]


def eval_composite(
    manifest_path: Path,
    episodes: int,
    max_steps: int,
    seed: int,
    rom: Optional[str] = None,
    start_state: Optional[str] = None,
    out_dir: Optional[str] = None,
    device_str: str = "cpu",
    hysteresis_k: Optional[int] = None,
    capture_handoffs: Optional[str] = None,
    stop_after_worlds: int = 1,
    sticky_prob: float = 0.0,
    start_jitter: int = 0,
    record_actions: Optional[str] = None,
) -> dict:
    if not manifest_path.exists():
        return _err(status="no_manifest", manifest=str(manifest_path))
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    levels = manifest.get("levels")
    if not isinstance(levels, dict) or not levels:
        return _err(status="bad_manifest",
                    detail="manifest needs a non-empty `levels` mapping",
                    manifest=str(manifest_path))

    game = str(manifest.get("game", "mario")).lower().strip()
    device = torch.device(device_str)
    torch.manual_seed(seed)

    # Load every specialist up front, deduped by (ckpt, profile) so a
    # champion-everywhere manifest loads the net once and shares it.
    scoring_key = _scoring_key(levels)
    scoring_profile: Optional[dict] = None
    cache: dict[tuple, object] = {}
    nets: dict[str, object] = {}
    gx_routes: dict[str, list] = {}
    entry_opts: dict[str, dict] = {}
    for key, entry in levels.items():
        if not isinstance(entry, dict) or "ckpt" not in entry or "profile" not in entry:
            return _err(status="bad_manifest",
                        detail=f"level {key!r} needs both `ckpt` and `profile`",
                        manifest=str(manifest_path))
        ckpt = str(entry["ckpt"])
        prof_path = str(entry["profile"])
        if not Path(ckpt).exists():
            return _err(status="missing_checkpoint", level=key, ckpt=ckpt)
        if not Path(prof_path).exists():
            return _err(status="missing_profile", level=key, profile=prof_path)
        profile = yaml.safe_load(Path(prof_path).read_text())
        if key == scoring_key:
            scoring_profile = profile
        cache_key = (str(Path(ckpt).resolve()), str(Path(prof_path).resolve()))
        if cache_key not in cache:
            try:
                cache[cache_key] = build_level_net(profile, ckpt, device, label=key)
            except (ValueError, KeyError, RuntimeError) as e:
                return _err(status="load_failed", level=key, ckpt=ckpt,
                            profile=prof_path, detail=str(e))
        nets[key] = cache[cache_key]
        if isinstance(entry.get("entry"), dict):
            entry_opts[key] = {
                "noop_pad": int(entry["entry"].get("noop_pad", 0)),
                "continuous_stack": bool(
                    entry["entry"].get("continuous_stack", False)
                ),
            }
        # Optional intra-level gx-keyed handoffs: a sorted stage list of
        # {at_gx, ckpt, profile}. Each stage net loads through the same
        # cache; the controller switches to it (one-way, disclosed in the
        # output as gx_switches) once the level's x crosses at_gx.
        for sw in entry.get("gx_switches") or []:
            try:
                at_gx = int(sw["at_gx"])
                s_ckpt = str(sw["ckpt"])
                s_prof = str(sw["profile"])
            except (KeyError, TypeError, ValueError):
                return _err(status="bad_manifest", level=key,
                            detail="each gx_switches entry needs "
                                   "at_gx, ckpt and profile")
            if not Path(s_ckpt).exists():
                return _err(status="missing_checkpoint",
                            level=f"{key}@gx{at_gx}", ckpt=s_ckpt)
            if not Path(s_prof).exists():
                return _err(status="missing_profile",
                            level=f"{key}@gx{at_gx}", profile=s_prof)
            s_profile = yaml.safe_load(Path(s_prof).read_text())
            s_cache_key = (str(Path(s_ckpt).resolve()),
                           str(Path(s_prof).resolve()))
            if s_cache_key not in cache:
                try:
                    cache[s_cache_key] = build_level_net(
                        s_profile, s_ckpt, device, label=f"{key}@gx{at_gx}",
                    )
                except (ValueError, KeyError, RuntimeError) as e:
                    return _err(status="load_failed", level=f"{key}@gx{at_gx}",
                                ckpt=s_ckpt, profile=s_prof, detail=str(e))
            s_key = f"{key}@gx{at_gx}"
            nets[s_key] = cache[s_cache_key]
            gx_routes.setdefault(key, []).append(
                (at_gx, s_key, int(sw.get("noop_pad", 0)),
                 bool(sw.get("continuous_stack", False)))
            )
    for key in gx_routes:
        gx_routes[key].sort()

    # ROM + cold-start state + frame_skip resolution.
    rom_path = rom or manifest.get("rom") or DEFAULT_ROMS.get(game)
    if not rom_path or not Path(rom_path).exists():
        return _err(status="no_rom", rom_path=rom_path, game=game)
    start = (
        start_state or manifest.get("start_state")
        or (scoring_profile or {}).get("start_state_path")
    )
    if start is not None and not Path(str(start)).exists():
        return _err(status="no_start_state", start_state=str(start))
    frame_skip = int(
        manifest.get("frame_skip")
        or (scoring_profile or {}).get("frame_skip", 4)
    )
    k = int(hysteresis_k if hysteresis_k is not None
            else manifest.get("hysteresis_k", 2))

    # The reward fn is the game's Rust object (built from the scoring level's
    # profile); used ONLY for the terminal (death) signal + a loose return, so
    # the composite's stop condition matches eval_game --sequential exactly.
    reward_fn = build_reward_function(scoring_profile)

    pool = RustPool(
        rom_path=rom_path, num_workers=1, frame_skip=frame_skip,
        start_state_path=str(start) if start else None, seed=seed,
    )
    pool.start()
    try:
        records = []
        for _ep in range(episodes):
            controller = CompositeController(
                nets, levels.keys(), device, k=k, gx_routes=gx_routes,
                entry_opts=entry_opts,
            )
            # Per-episode isolation: one off-map level (KeyError from
            # _select) or any episode-level fault must not discard the
            # whole batch and lose every completed episode's log row.
            # Record a sentinel and keep going.
            try:
                rec = run_episode(pool, controller, reward_fn, max_steps,
                                  capture_dir=capture_handoffs,
                                  stop_after_worlds=stop_after_worlds,
                                  sticky_prob=sticky_prob,
                                  start_jitter=start_jitter,
                                  seed=seed * 10_000 + _ep,
                                  record_dir=record_actions,
                                  episode_idx=_ep)
            except Exception as exc:  # noqa: BLE001 — isolate the batch
                import traceback as _tb
                print(f"[eval_composite] episode {_ep} faulted "
                      f"({exc}); recording sentinel and continuing:\n"
                      f"{_tb.format_exc()}", file=sys.stderr)
                rec = {
                    "seq_clear": False, "warp_taken": False,
                    "furthest_seq": None, "furthest_any": None,
                    "worlds_cleared": 0, "furthest_nowarp": None,
                    "cleared": False, "return": 0.0, "length": 0,
                    "max_byte": 0, "switches": 0, "gx_switches": [],
                    "level_max_gx": {}, "segments": [],
                    "end_reason": "episode_fault", "fault": str(exc),
                }
            records.append(rec)
    finally:
        pool.shutdown()

    result = summarize(
        records, manifest_path=str(manifest_path), game=game, n_episodes=episodes,
    )
    result["timestamp"] = time.time()
    # Self-describing protocol fields: a logged row proves how it was run.
    result["sticky_prob"] = sticky_prob
    result["start_jitter"] = start_jitter
    result["seed"] = seed
    try:
        import subprocess as _sp
        result["git_commit"] = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        result["git_commit"] = None
    try:
        import hashlib as _hl
        result["rom_md5"] = _hl.md5(Path(rom_path).read_bytes()).hexdigest()
    except Exception:
        result["rom_md5"] = None
    try:
        import hashlib as _hl
        _ckpt_md5_cache: dict[str, str] = {}
        def _ckpt_md5(path: str) -> str:
            rp = str(Path(path).resolve())
            if rp not in _ckpt_md5_cache:
                _ckpt_md5_cache[rp] = _hl.md5(Path(path).read_bytes()).hexdigest()
            return _ckpt_md5_cache[rp]
        # Provenance for what was ACTUALLY loaded: checkpoints/ is gitignored
        # and can drift in place (same path, different bytes) without the
        # manifest or git_commit changing, so a banked row needs its own
        # content hash to be trustworthy.
        result["level_checkpoints"] = {
            key: {"ckpt": str(entry["ckpt"]), "profile": str(entry["profile"]),
                  "ckpt_md5": _ckpt_md5(entry["ckpt"])}
            for key, entry in levels.items()
        }
    except Exception:
        result["level_checkpoints"] = None
    result["hysteresis_k"] = k
    result["levels"] = sorted(levels.keys())
    if gx_routes:
        # Routed seams are disclosed, never hidden: the result names every
        # gx-keyed handoff the manifest declares, alongside the per-episode
        # gx_switches events summarize() carries through.
        result["gx_routed"] = True
        result["level_entry_opts"] = entry_opts
        result["gx_routes"] = {
            key: [[gx, name, pad, cont] for gx, name, pad, cont in stages]
            for key, stages in gx_routes.items()
        }

    # Append to a composite-specific eval log.
    slug = profile_slug(manifest.get("name")) if manifest.get("name") else "composite"
    log_dir = Path(out_dir) if out_dir else (ROOT / "checkpoints" / f"composite_{slug}")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "eval.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    result["eval_log"] = str(log_dir / "eval.jsonl")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=str, required=True,
                   help="Composite manifest YAML (level -> {ckpt, profile}).")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=2400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rom", type=str, default=None,
                   help="ROM path override (else manifest.rom / DEFAULT_ROMS).")
    p.add_argument("--start-state", type=str, default=None,
                   help="Cold-start state override (else the scoring profile's).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory for eval.jsonl (else checkpoints/composite_*).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hysteresis", type=int, default=None, dest="hysteresis_k",
                   metavar="K",
                   help="Consecutive agreeing frames before a net switch (default 2).")
    p.add_argument("--capture-handoffs", type=str, default=None, metavar="DIR",
                   help="Save the emulator state at each first committed level "
                        "switch as DIR/handoff_<label>.state (the exact entry "
                        "frame the incoming specialist takes over from).")
    p.add_argument("--stop-after-worlds", type=int, default=1, metavar="N",
                   help="End the episode after N real castle clears (default "
                        "1: the World-1 DoD). 0 = never stop on clears — play "
                        "until death/timeout and report how far the chain got "
                        "(worlds_cleared / furthest_nowarp).")
    p.add_argument("--sticky-prob", type=float, default=0.0, metavar="P",
                   help="Honest-eval protocol: repeat the previous action "
                        "with probability P (Machado et al. sticky actions; "
                        "0.25 is the standard). Default 0 = deterministic.")
    p.add_argument("--record-actions", default=None, dest="record_actions",
                   help="Directory for per-episode action receipts: the "
                        "post-sticky stepped mask per step (.npy) + a "
                        "protocol sidecar (.json).")
    p.add_argument("--start-jitter", type=int, default=0, metavar="N",
                   help="Hold up to N random noop steps before control "
                        "begins, desynchronizing hazard phases from the "
                        "trained trajectory. Default 0.")
    args = p.parse_args()

    result = eval_composite(
        Path(args.manifest), args.episodes, args.max_steps, args.seed,
        rom=args.rom, start_state=args.start_state, out_dir=args.out_dir,
        device_str=args.device, hysteresis_k=args.hysteresis_k,
        record_actions=args.record_actions,
        capture_handoffs=args.capture_handoffs,
        stop_after_worlds=args.stop_after_worlds,
        sticky_prob=args.sticky_prob,
        start_jitter=args.start_jitter,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
