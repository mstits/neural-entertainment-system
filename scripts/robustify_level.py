"""Self-imitation robustifier: distill a policy's own stochastic level
clears into its argmax path.

Single-state mode (default): run the checkpoint STOCHASTICALLY from a
fixed entry state, keep the (obs, action) trajectories of episodes that
clear the level (LevelClearTracker predicate — the same one the
consolidation gate probes), behavior-clone those demos into the same
net (CE on actions, fine-tuned from the source checkpoint), then greedy
re-eval from the same entry and save the best clone.

Ladder mode (--ladder deepest,...,shallowest): backward induction
within one level. Weld the deepest rung first, then step the start
state backwards; each round's demos run from that rung THROUGH the rest
of the level to the clear, so the BC dataset always covers the full
suffix — no forgetting between rounds. While collecting, the emulator
state is harvested at every first x-bucket crossing (go-explore style);
if a rung's gap to the welded basin is too wide for stochastic
bridging, the deepest harvested intermediate is inserted as a new rung
and the stalled rung is retried after it.

Usage:
  python robustify_level.py --profile configs/smb_1_4_go_explore.yaml \
      --checkpoint .../vanilla_ppo_iter_02120.pt --out .../robust_1_4.pt \
      [--start-state PATH | --ladder P1,P2,...] [--clears 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nes_core import Pool  # noqa: E402
from src.emulation.frame_utils import TileFeatureStacker  # noqa: E402
from src.models.tile_policy import build_tile_policy_from_checkpoint  # noqa: E402
from src.training.profile_utils import (  # noqa: E402
    action_space_to_bitmasks, resolve_encoder,
)
from src.training.smb_sequential import LevelClearTracker  # noqa: E402
from src.utils.reward_functions import build_reward_function  # noqa: E402
from eval_game import _TilePolicy  # noqa: E402
from train_game import DEFAULT_ROMS  # noqa: E402

HARVEST_PX = 150  # x-bucket width for intermediate-state harvesting


def make_policy(profile: dict, sd: dict, bitmasks) -> tuple:
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    net, is_recurrent = build_tile_policy_from_checkpoint(
        sd, num_actions=len(bitmasks), feature_dim=stacked_dim,
    )
    res = net.load_state_dict(sd["net_state_dict"], strict=False)
    assert not res.missing_keys, f"missing keys: {res.missing_keys}"
    net.eval()
    stacker = TileFeatureStacker(
        stack_size=stacked_dim // feature_dim, feature_dim=feature_dim,
    )
    return net, _TilePolicy(net, stacker, extractor, is_recurrent), is_recurrent


def _gx(ram) -> int:
    return (int(ram[0x006D]) << 8) | int(ram[0x0086])


def _level_key(ram) -> tuple:
    return (int(ram[0x075F]), int(ram[0x075C]))


def run_episode(pool, pol, bitmasks, blob, max_steps, mode, device, reward_fn,
                seed=0, collect=False, harvest=None, explore_eps=0.0,
                greedy_after=0, greedy_after_gx=0, sticky_prob=0.0,
                success_gx=0):
    """One episode. Returns (outcome, traj) with outcome in
    'clear' | 'died' | 'timeout'. `harvest`: optional dict[bucket->blob]
    filled at every first x-bucket crossing while still inside the
    start level (post-step snapshot, go-explore style). `explore_eps`:
    probability of a uniform-random action during sampling — keeps
    collection diverse even after BC rounds collapse the policy's
    entropy onto the demo actions (without it, a sharpened clone
    replays one failing trajectory N times and never finds a clear)."""
    g = torch.Generator().manual_seed(seed)
    pool.reset_all()
    if blob is not None:
        pool.load_worker_state(0, blob)
    reward_fn.reset()
    init = pool.step_all(np.zeros(1, dtype=np.uint8))
    obs = pol.reset(init[0])
    hidden = pol.initial_hidden(device)
    tracker = LevelClearTracker()
    tracker.update(init[0][2])
    start_key = _level_key(init[0][2])
    traj = []
    _prev_a = 0
    joined = False  # position-triggered argmax handover latch
    last_ram = init[0][2]
    for _step in range(max_steps):
        with torch.no_grad():
            logits, hidden = pol.logits(obs, hidden)
        # Hybrid rollout: a weld is a TRAJECTORY-weld — a stochastic run
        # reaching the welded region arrives with different enemy/firebar
        # phases than the welded seed state, so a full stochastic clear
        # can be near-impossible even 150px from a welded basin. With
        # greedy_after > 0, sample only the unwelded gap, then hand the
        # episode to argmax and let the welded policy finish closed-loop.
        # greedy_after_gx does the same POSITION-triggered: hand over the
        # first time Mario physically crosses into the welded basin —
        # correct when the gap is a long gauntlet and a step-count
        # trigger would fire argmax mid-gauntlet in unwelded territory.
        # The demo (stochastic prefix + greedy suffix) is BC-consistent.
        if greedy_after_gx and not joined:
            joined = _gx(last_ram) >= greedy_after_gx
        if mode == "sample" and not joined and not (
                greedy_after and _step >= greedy_after):
            # Match the TRAINING noise model: a net trained with sticky
            # actions has learned to expect action persistence — sampling
            # it without sticky systematically shortens every hold and
            # can collapse its clear rate ~two orders of magnitude.
            if (sticky_prob > 0.0 and _step > 0
                    and torch.rand(1, generator=g).item() < sticky_prob):
                a = _prev_a
            elif explore_eps > 0 and torch.rand(1, generator=g).item() < explore_eps:
                a = int(torch.randint(len(bitmasks), (1,), generator=g).item())
            else:
                probs = torch.softmax(logits[0], dim=-1)
                a = int(torch.multinomial(probs, 1, generator=g).item())
        else:
            a = int(torch.argmax(logits[0]).item())
        _prev_a = a
        if collect:
            traj.append((np.array(obs, dtype=np.int8, copy=True), a))
        r = pool.step_all(np.array([bitmasks[a]], dtype=np.uint8))
        ram = r[0][2]
        last_ram = ram
        _, rew_done, _ = reward_fn.compute(ram, action=int(bitmasks[a]))
        tracker.update(ram)
        if (harvest is not None and _level_key(ram) == start_key
                and not tracker.seq_clear):
            b = _gx(ram) // HARVEST_PX
            if b not in harvest:
                harvest[b] = pool.save_worker_state(0)
        obs = pol.push(r[0])
        # Segment predicate: with success_gx set, the episode SUCCEEDS on
        # reaching that global x alive — the front-half weld target for
        # gx-keyed intra-level routing (the back half is a separately
        # welded net; the composite hands over at the boundary).
        if success_gx and _gx(ram) >= success_gx and int(ram[0x000E]) not in (6, 11):
            return "clear", (traj if collect else None)
        if tracker.seq_clear:
            return "clear", (traj if collect else None)
        if rew_done or bool(r[0][3]):
            return "died", None
    return "timeout", None


def collect_demos(pool, pol, bitmasks, blob, max_steps, device, reward_fn,
                  want, cap, harvest=None, tag="", explore_eps=0.0,
                  greedy_after=0, greedy_after_gx=0, sticky_prob=0.0,
                  success_gx=0):
    demos, seen = [], 0
    while len(demos) < want and seen < cap:
        outcome, traj = run_episode(
            pool, pol, bitmasks, blob, max_steps, "sample", device,
            reward_fn, seed=seen, collect=True, harvest=harvest,
            explore_eps=explore_eps, greedy_after=greedy_after,
            greedy_after_gx=greedy_after_gx, sticky_prob=sticky_prob,
            success_gx=success_gx,
        )
        seen += 1
        if outcome == "clear":
            demos.append(traj)
        if seen % 40 == 0 or len(demos) == want:
            print(f"[collect{tag}] {seen} eps -> {len(demos)} clears",
                  flush=True)
    return demos, seen


def collect_demos_parallel(net, extractor, feature_dim, stack_size, pool,
                           bitmasks, blob, max_steps, reward_fn_factory, want,
                           cap, *, tag="", explore_eps=0.0, greedy_after_gx=0,
                           sticky_prob=0.0, success_gx=0, workers=10,
                           base_seed=0, harvest=None):
    """N-worker lockstep version of collect_demos — collects `want` clears (or
    `cap` total episodes) across `workers` parallel envs, batching the policy
    forward. Same outcome semantics as run_episode('sample'); each worker draws
    from its own RNG so trajectories stay diverse. Restart convention matches
    run_episode: reload blob, one NOOP step, re-seed the stacker."""
    W = int(workers)
    stackers = [TileFeatureStacker(stack_size=stack_size, feature_dim=feature_dim)
                for _ in range(W)]
    rfns = [reward_fn_factory() for _ in range(W)]
    gens = [torch.Generator().manual_seed(base_seed * 100000 + i) for i in range(W)]
    pool.reset_all()
    if blob is not None:
        for i in range(W):
            pool.load_worker_state(i, blob)
    for r in rfns:
        r.reset()
    init = pool.step_all(np.zeros(W, dtype=np.uint8))
    obs = [stackers[i].reset(extractor.extract(init[i][2])) for i in range(W)]
    trackers = [LevelClearTracker() for _ in range(W)]
    for i in range(W):
        trackers[i].update(init[i][2])
    trajs = [[] for _ in range(W)]
    prev_a = [0] * W
    joined = [False] * W
    steps = [0] * W
    fresh = [False] * W
    last_ram = [init[i][2] for i in range(W)]
    start_key = [_level_key(init[i][2]) for i in range(W)]
    demos: list = []
    seen = 0
    acts = np.zeros(W, dtype=np.uint8)

    def _restart(i):
        if blob is not None:
            pool.load_worker_state(i, blob)
        rfns[i].reset()
        trackers[i] = LevelClearTracker()
        trajs[i] = []
        steps[i] = 0
        prev_a[i] = 0
        joined[i] = False
        fresh[i] = True   # next step is a NOOP re-seed (load,noop,reset convention)

    while len(demos) < want and seen < cap:
        X = torch.from_numpy(np.stack(obs)).float()
        with torch.no_grad():
            logits, _ = net.forward_ac(X)
        for i in range(W):
            if fresh[i]:
                acts[i] = bitmasks[0]  # NOOP to materialize the reloaded state
                continue
            if greedy_after_gx and not joined[i]:
                joined[i] = _gx(last_ram[i]) >= greedy_after_gx
            if not joined[i]:
                if (sticky_prob > 0.0 and steps[i] > 0
                        and torch.rand(1, generator=gens[i]).item() < sticky_prob):
                    a = prev_a[i]
                elif explore_eps > 0 and torch.rand(1, generator=gens[i]).item() < explore_eps:
                    a = int(torch.randint(len(bitmasks), (1,), generator=gens[i]).item())
                else:
                    probs = torch.softmax(logits[i], dim=-1)
                    a = int(torch.multinomial(probs, 1, generator=gens[i]).item())
            else:
                a = int(torch.argmax(logits[i]).item())
            prev_a[i] = a
            trajs[i].append((np.array(obs[i], dtype=np.int8, copy=True), a))
            acts[i] = bitmasks[a]
        r = pool.step_all(acts)
        for i in range(W):
            ram = r[i][2]
            last_ram[i] = ram
            if fresh[i]:
                # Re-seed after the reload NOOP; this step is not part of any
                # episode (matches run_episode's init NOOP, not counted).
                obs[i] = stackers[i].reset(extractor.extract(ram))
                trackers[i].update(ram)
                start_key[i] = _level_key(ram)
                fresh[i] = False
                continue
            steps[i] += 1
            _, rew_done, _ = rfns[i].compute(ram, action=int(acts[i]))
            trackers[i].update(ram)
            if (harvest is not None and _level_key(ram) == start_key[i]
                    and not trackers[i].seq_clear):
                b = _gx(ram) // HARVEST_PX
                if b not in harvest:
                    harvest[b] = pool.save_worker_state(i)
            obs[i] = stackers[i].push(extractor.extract(ram))
            cleared = bool(
                (success_gx and _gx(ram) >= success_gx
                 and int(ram[0x000E]) not in (6, 11))
                or trackers[i].seq_clear
            )
            ended = cleared or rew_done or bool(r[i][3]) or steps[i] >= max_steps
            if ended:
                if cleared:
                    demos.append(trajs[i])
                seen += 1
                if seen % 40 == 0 or len(demos) == want:
                    print(f"[collect{tag}||{W}w] {seen} eps -> {len(demos)} "
                          f"clears", flush=True)
                if len(demos) >= want or seen >= cap:
                    break
                _restart(i)
    return demos, seen


def greedy_rate(pool, pol, bitmasks, blob, max_steps, device, reward_fn,
                episodes, success_gx=0, sticky_prob=0.0) -> float:
    """Acceptance clear rate from `blob`. With sticky_prob>0 this evaluates
    under the SAME sticky-action noise as the honest gate (mode='sample',
    NO exploration) rather than pure argmax — so a rung is only welded when
    the policy is genuinely sticky-robust there, not merely greedy-competent.
    (The greedy-acceptance vs sticky-gate mismatch was welding fragile rungs
    that then stalled stochastic collection at the next rung.)"""
    mode = "sample" if sticky_prob > 0.0 else "greedy"
    c = 0
    for ep in range(episodes):
        outcome, _ = run_episode(
            pool, pol, bitmasks, blob, max_steps, mode, device,
            reward_fn, seed=1000 + ep, success_gx=success_gx,
            explore_eps=0.0, sticky_prob=sticky_prob,
        )
        c += outcome == "clear"
    return c / max(1, episodes)


def _bc_fit(net, pol, pool, bitmasks, blob, demos, args, device, reward_fn,
            tag="", epochs=None):
    """Fit `net` to the demos by CE; greedy-verify periodically; return
    (best_rate, best_sd)."""
    X = torch.tensor(
        np.concatenate([np.stack([o for o, _ in t]) for t in demos]),
        dtype=torch.float32,
    )
    if X.ndim == 3:
        X = X.squeeze(1)
    Y = torch.tensor(
        np.concatenate([np.array([a for _, a in t]) for t in demos]),
        dtype=torch.long,
    )
    print(f"[bc{tag}] dataset: {X.shape[0]} steps from {len(demos)} clears",
          flush=True)
    n_epochs = epochs if epochs is not None else args.bc_epochs
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_rate, best_sd = -1.0, None
    for epoch in range(n_epochs):
        logits, _ = net.forward_ac(X)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net.forward_ac(X)[0].argmax(-1) == Y).float().mean()
            rate = greedy_rate(pool, pol, bitmasks, blob, args.max_steps,
                               device, reward_fn, args.eval_episodes,
                               success_gx=getattr(args, "success_gx", 0),
                               sticky_prob=getattr(args, "accept_sticky", 0.0))
            print(f"[bc{tag}] epoch {epoch+1} loss {loss.item():.4f} "
                  f"acc {acc.item():.3f} greedy {rate:.2f}", flush=True)
            if rate > best_rate:
                best_rate = rate
                best_sd = {k: v.clone() for k, v in net.state_dict().items()}
            if rate == 1.0:
                break
            net.train()
    return best_rate, best_sd


def bc_round(net, pol, pool, bitmasks, blob, demos, args, device, reward_fn,
             tag=""):
    """BC the demos into `net`. Multi-demo CE first; if no clone
    greedy-clears, fall back to cloning each demo ALONE (restarting from
    the pre-round weights each attempt). Distinct successful trajectories
    give CONFLICTING action labels at similar observations — a 98%+
    accurate majority-vote clone can stitch them into an infeasible
    hybrid path — while an exact single-trajectory replay is feasible by
    construction in a deterministic env (modulo observation aliasing)."""
    pre = {k: v.detach().clone() for k, v in net.state_dict().items()}
    rate, sd = _bc_fit(net, pol, pool, bitmasks, blob, demos, args, device,
                       reward_fn, tag=tag)
    if rate > 0:
        return rate, sd
    for i, d in enumerate(demos):
        net.load_state_dict(pre)
        rate, sd = _bc_fit(net, pol, pool, bitmasks, blob, [d], args, device,
                           reward_fn, tag=f"{tag}:solo{i}",
                           epochs=args.bc_epochs + 200)
        if rate > 0:
            return rate, sd
    net.load_state_dict(pre)
    return 0.0, None


def blob_gx(pool, blob) -> int:
    pool.reset_all()
    pool.load_worker_state(0, blob)
    r = pool.step_all(np.zeros(1, dtype=np.uint8))
    return _gx(r[0][2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--ladder", default=None,
                    help="Comma-separated rung state paths, deepest first; "
                         "backward induction with auto-harvested "
                         "intermediates. Overrides --start-state.")
    ap.add_argument("--clears", type=int, default=10)
    ap.add_argument("--episode-cap", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--bc-epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10,
                    help="Parallel envs for demo collection (the bottleneck). "
                         "BC/eval stay single-env.")
    ap.add_argument("--explore-eps", type=float, default=0.15,
                    help="Uniform-random action probability during "
                         "collection (exploration that survives BC "
                         "entropy collapse).")
    ap.add_argument("--greedy-after", type=int, default=0,
                    help="During collection, switch from sampling to pure "
                         "argmax after this many steps — bridge the "
                         "unwelded gap stochastically, then let the "
                         "welded greedy policy finish (0 = off).")
    ap.add_argument("--success-gx", type=int, default=0,
                    help="Segment weld: success = reaching this global x "
                         "alive (instead of clearing the level). For "
                         "gx-keyed intra-level routing front halves.")
    ap.add_argument("--accept-sticky", type=float, default=0.0,
                    help="Evaluate rung ACCEPTANCE under this sticky "
                         "prob (match the honest gate) instead of argmax "
                         "— welds only sticky-robust rungs.")
    ap.add_argument("--sticky-prob", type=float, default=0.0,
                    help="Match the training noise model during collection: "
                         "repeat the previous action with this probability "
                         "(use the profile's sticky_action_prob for nets "
                         "trained with sticky actions).")
    ap.add_argument("--greedy-after-gx", type=int, default=0,
                    help="Position-triggered argmax handover: switch to "
                         "pure argmax the first time Mario's global x "
                         "crosses this value (the welded basin edge). "
                         "Correct for long-gauntlet gaps where a step "
                         "trigger would fire mid-gauntlet (0 = off).")
    args = ap.parse_args()

    profile = yaml.safe_load(Path(args.profile).read_text())
    bitmasks = action_space_to_bitmasks(profile["action_space"])
    sd = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cpu")
    net, pol, is_recurrent = make_policy(profile, sd, bitmasks)
    assert not is_recurrent, "robustifier currently supports stateless nets"
    reward_fn = build_reward_function(profile)
    extractor, feature_dim, stacked_dim = resolve_encoder(profile)
    stack_size = stacked_dim // feature_dim

    rom = DEFAULT_ROMS.get("mario", "roms/Super Mario Bros. (World).nes")
    # 1-worker pool for single-env BC-eval / greedy_rate / blob_gx (unchanged).
    pool = Pool(rom_path=rom, num_workers=1, frame_skip=4,
                start_state_path=None)
    # W-worker pool for parallel demo collection (the bottleneck) — this is
    # what keeps all cores busy instead of one.
    cpool = Pool(rom_path=rom, num_workers=int(args.workers), frame_skip=4,
                 start_state_path=None)
    cpool.set_headless(True)
    cpool.set_skip_preprocess(True)

    if args.ladder:
        rungs = [("rung:" + Path(p).stem, Path(p).read_bytes())
                 for p in args.ladder.split(",")]
    else:
        ss = args.start_state or profile.get("start_state_path")
        rungs = [("entry", Path(ss).read_bytes() if ss else None)]

    def save(rate: float, welded: list) -> None:
        out = dict(sd)
        out["net_state_dict"] = {
            k: v.clone() for k, v in net.state_dict().items()
        }
        out["robustified"] = {
            "source": args.checkpoint,
            "greedy_rate": rate,
            "ladder": bool(args.ladder),
            "welded_rungs": list(welded),
        }
        torch.save(out, args.out)

    harvested: dict[int, bytes] = {}
    used_buckets: set[int] = set()
    welded: list = []
    welded_min_gx: Optional[int] = None  # shallowest welded rung so far
    final_rate = 0.0
    stall = 0
    _rung_i = 0
    while rungs:
        name, blob = rungs.pop(0)
        _rung_i += 1
        rung_gx = blob_gx(pool, blob) if blob is not None else 0
        print(f"=== {name} (gx={rung_gx}) ===", flush=True)
        # Dynamic argmax handover at the welded-basin edge: explore only the
        # NEW gap (this rung -> the shallowest welded rung), then let the
        # already-welded greedy finish. A static --greedy-after-gx (or none)
        # forces stochastic play across the whole remaining level, and
        # uniform explore-eps noise over ~500 steps destroys even freshly
        # welded segments (0/400 collections from INSIDE the basin).
        # Dynamic basin edge once welds accrue; the static CLI value only
        # bootstraps the FIRST rungs of a warm-started run (welded_min_gx
        # is None in a fresh process even when the checkpoint is welded).
        _handover = (welded_min_gx if welded_min_gx is not None
                     else args.greedy_after_gx)
        demos, seen = collect_demos_parallel(
            net, extractor, feature_dim, stack_size, cpool, bitmasks, blob,
            args.max_steps, lambda: build_reward_function(profile),
            args.clears, args.episode_cap, harvest=harvested, tag=f":{name}",
            explore_eps=args.explore_eps, greedy_after_gx=_handover,
            sticky_prob=args.sticky_prob, success_gx=args.success_gx,
            workers=int(args.workers), base_seed=_rung_i,
        )
        if demos:
            rate, best_sd = bc_round(net, pol, pool, bitmasks, blob, demos,
                                     args, device, reward_fn, tag=f":{name}")
            if best_sd is None or rate <= 0.0:
                print(f"FAIL: BC found no greedy-clearing clone at {name}")
                pool.shutdown()
                cpool.shutdown()
                return 2
            stall = 0
            net.load_state_dict(best_sd)
            net.eval()
            final_rate = rate
            welded.append({"rung": name, "gx": rung_gx, "greedy": rate})
            if welded_min_gx is None or rung_gx < welded_min_gx:
                welded_min_gx = rung_gx
            # Persist after every weld so a crash/restart keeps progress.
            save(rate, welded)
            continue
        # No stochastic clears: the gap between this rung and the welded
        # basin is too wide. Bridge with the deepest harvested
        # intermediate STRICTLY BETWEEN the rung and the basin edge —
        # a bucket at/inside the basin extends nothing — then retry.
        edge = welded_min_gx if welded_min_gx is not None else 10 ** 9
        cands = {b: v for b, v in harvested.items()
                 if b not in used_buckets
                 and rung_gx < b * HARVEST_PX < edge}
        if not cands or stall >= 12:
            print(f"FAIL: no clears at {name} and no unused intermediate "
                  f"in ({rung_gx}, {edge}) to bridge with (stall={stall})")
            pool.shutdown()
            cpool.shutdown()
            return 1
        b = max(cands)
        used_buckets.add(b)
        stall += 1
        print(f"[bridge] inserting harvested bucket {b} "
              f"(gx~{b * HARVEST_PX}) before retrying {name}", flush=True)
        rungs.insert(0, (name, blob))
        rungs.insert(0, (f"harvest_b{b}", cands[b]))

    save(final_rate, welded)
    print(f"[done] greedy clear rate {final_rate:.2f} -> {args.out}")
    pool.shutdown()
    cpool.shutdown()
    return 0 if final_rate > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
