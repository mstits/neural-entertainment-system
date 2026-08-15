"""Differential verification of castlevania.yaml's asserted `entity_slots`.

BLOCKER: `solve.entity_slots: {lo: 0x0450, hi: 0x0458}` is the only byte
range in the CV solve block with no receipt — its sole provenance is a
prose comment. Every other solve byte is receipted
(docs/receipts/ram_verify/castlevania.json, docs/receipts/observatory/
cv_lvl02_acceptance.json). This script measures the range from our own
rollouts and either receipts it or refutes it.

PURITY: no map, no disassembly, no walkthrough. The protocol sweeps the
WHOLE page 0x0400-0x04FF and every action in the profile's own action
space; which addresses and which button "matter" is an output, never an
input. The claimed bounds lo/hi are treated as a HYPOTHESIS TO FALSIFY,
not as a region to confirm.

Three independent verdicts. The class under test is a SLOT-FLAG ARRAY,
so the criteria are the ones that class implies -- not verify_ram_map's
stable-state criteria, which grade a byte on NOT changing and are
inapplicable to a byte whose whole purpose is to toggle. Churn and
occupancy are therefore reported descriptively and deliberately excluded
from the verdicts (see `descriptive_only` in the receipt).

  V1 BOUNDS (structural, derives lo/hi rather than confirming them).
     Compute every maximal contiguous run of binary-{0,1} bytes in the
     whole page. The claim passes only if the LONGEST such run is
     exactly [lo, hi). A claimed range that is merely one run among many
     equal-length runs is unsupported; a longer run elsewhere refutes
     the bounds outright.
  V2 SPAWN/DESPAWN (dynamic). Every address in the range shows at least
     one 0->1 and at least one 1->0 edge, and the 8 columns are not
     mutual duplicates (>= DISTINCT_MIN distinct column signatures --
     8 identical columns would be one flag mirrored, not 8 slots).
  V3 KILL DIFFERENTIAL. Pooled over the range, some held action's 1->0
     edge rate exceeds the NOOP arm's by >= KILL_LIFT x. If nothing the
     agent does elevates the despawn rate, the 1->0 edges are ambient
     and `--kill-key` / `kill_mask` have no differential basis.

Read-only w.r.t. the repo; writes one JSON receipt to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
sys.path.insert(0, str(REPO))

from nes_core import Pool  # noqa: E402
from src.training.profile_utils import action_space_to_bitmasks  # noqa: E402

RAM_SIZE = 0x800
PAGE_LO, PAGE_HI = 0x0400, 0x0500

CARD_MAX = 4
OCC_MIN = 0.05
CHURN_MAX = 5.0
KILL_LIFT = 2.0
DISTINCT_MIN = 4


def apply_hw_flags(pool, flags):
    for name in flags:
        getattr(pool, f"set_hw_{name}")(True)


class Rig:
    def __init__(self, rom, state, frame_skip, hw_flags):
        self.pool = Pool(rom_path=rom, num_workers=1, frame_skip=frame_skip)
        apply_hw_flags(self.pool, hw_flags)
        self.pool.set_headless(True)
        self.pool.reset_all()
        self.state = state
        self.reload()

    def reload(self):
        self.pool.load_worker_state(0, self.state)

    def run(self, masks):
        log = np.empty((len(masks), RAM_SIZE), dtype=np.uint8)
        for i, m in enumerate(masks):
            r = self.pool.step_all(np.array([int(m)], dtype=np.uint8))
            log[i] = np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]
        return log


def edges(col):
    """(0 -> nonzero, nonzero -> 0) transition counts for one byte."""
    nz = col != 0
    on = int(np.count_nonzero(~nz[:-1] & nz[1:]))
    off = int(np.count_nonzero(nz[:-1] & ~nz[1:]))
    return on, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(REPO / "configs/castlevania.yaml"))
    ap.add_argument("--root-state",
                    default=str(REPO / "runs/cv_chain_hw2/entrances/"
                                       "entrance_after_2.state"))
    ap.add_argument("--site", default="block3_hall_entrance")
    ap.add_argument("--hold-steps", type=int, default=1500)
    ap.add_argument("--random-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prof = yaml.safe_load(Path(a.profile).read_text())
    solve = prof["solve"]
    rom = str(REPO / solve["rom"])
    fs = int(prof.get("frame_skip", 4))
    sidecar = Path(a.root_state + ".json")
    hw = (list(json.loads(sidecar.read_text()).get("hw_flags", []))
          if sidecar.exists() else [])
    masks = action_space_to_bitmasks(prof["action_space"])
    state = Path(a.root_state).read_bytes()

    t0 = time.time()
    rig = Rig(rom, state, fs, hw)

    arms = {}
    for ai, m in enumerate(masks):
        rig.reload()
        arms[f"hold:{'+'.join(prof['action_space'][ai]) or 'NOOP'}"] = \
            rig.run([m] * a.hold_steps)

    rng = np.random.default_rng(a.seed)
    rig.reload()
    sched, cur = [], 0
    for _ in range(a.random_steps):
        if rng.random() > 0.5:
            cur = int(rng.integers(len(masks)))
        sched.append(masks[cur])
    arms["sticky_random"] = rig.run(sched)
    elapsed = time.time() - t0

    allrows = np.concatenate(list(arms.values()), axis=0)
    rnd = arms["sticky_random"]
    noop_key = next(k for k in arms if k.endswith("NOOP"))

    lives_a = int(solve["lives"])
    plo, phi = int(solve["progress"]["lo"]), int(solve["progress"]["hi"])

    per = {}
    for addr in range(PAGE_LO, PAGE_HI):
        col_all = allrows[:, addr]
        vals = Counter(col_all.tolist())
        card = len(vals)
        zfrac = float(np.count_nonzero(col_all == 0) / col_all.size)
        churn = 0.0
        rc = rnd[:, addr]
        churn = float(np.count_nonzero(rc[1:] != rc[:-1]) / len(rc) * 1000.0)
        on_t = off_t = 0
        per_arm = {}
        for k, log in arms.items():
            on, off = edges(log[:, addr])
            on_t += on
            off_t += off
            per_arm[k] = {"on": on, "off": off,
                          "off_per_1k": off / len(log) * 1000.0}
        s1 = card <= CARD_MAX
        s2 = OCC_MIN <= zfrac <= 1.0 - OCC_MIN
        s3 = on_t > 0 and off_t > 0
        s4 = churn < CHURN_MAX
        per[addr] = {
            "addr": f"0x{addr:04x}", "cardinality": card,
            "top_values": dict(sorted(vals.items(), key=lambda kv: -kv[1])[:4]),
            "zero_fraction": round(zfrac, 4),
            "churn_1k_random": round(churn, 3),
            "onsets": on_t, "clears": off_t,
            "S1_low_cardinality": s1, "S2_occupancy_toggles": s2,
            "S3_spawn_and_despawn": s3, "S4_not_scratchpad": s4,
            "qualifies": bool(s1 and s2 and s3 and s4),
            "per_arm": per_arm,
        }

    lo, hi = int(solve["entity_slots"]["lo"]), int(solve["entity_slots"]["hi"])
    claimed = list(range(lo, hi))

    # --- V1 BOUNDS: derive the binary-flag runs, then compare to the claim.
    binary = [x for x in range(PAGE_LO, PAGE_HI)
              if per[x]["cardinality"] == 2 and set(per[x]["top_values"]) == {0, 1}]
    runs, cur = [], [binary[0]] if binary else []
    for x in binary[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    if cur:
        runs.append(cur)
    runs.sort(key=len, reverse=True)
    longest = runs[0] if runs else []
    v1 = bool(longest and longest[0] == lo and longest[-1] + 1 == hi
              and (len(runs) == 1 or len(runs[1]) < len(longest)))

    # --- V2 SPAWN/DESPAWN + column non-duplication.
    both_edges = {f"0x{x:04x}": bool(per[x]["onsets"] > 0 and per[x]["clears"] > 0)
                  for x in claimed}
    sigs = {hash(allrows[:, x].tobytes()) for x in claimed}
    v2 = bool(all(both_edges.values()) and len(sigs) >= DISTINCT_MIN)

    # --- V3 KILL DIFFERENTIAL, per slot and pooled over the range.
    kill = {}
    for addr in claimed:
        base = per[addr]["per_arm"][noop_key]["off_per_1k"]
        best = max(((k, v["off_per_1k"]) for k, v in per[addr]["per_arm"].items()
                    if k != "sticky_random"), key=lambda kv: kv[1])
        kill[f"0x{addr:04x}"] = {
            "noop_clear_per_1k": round(base, 3),
            "best_arm": best[0], "best_clear_per_1k": round(best[1], 3),
            "lift_vs_noop": (round(best[1] / base, 3) if base > 0
                             else ("inf" if best[1] > 0 else 0.0)),
        }
    pooled = {}
    for k in arms:
        if k == "sticky_random":
            continue
        off = sum(per[x]["per_arm"][k]["off"] for x in claimed)
        pooled[k] = off / len(arms[k]) * 1000.0
    pool_base = pooled[noop_key]
    pool_best = max(pooled.items(), key=lambda kv: kv[1])
    pool_lift = (pool_best[1] / pool_base if pool_base > 0
                 else (float("inf") if pool_best[1] > 0 else 0.0))
    v3 = bool(pool_lift >= KILL_LIFT)

    deaths = int(np.count_nonzero(
        np.diff(allrows[:, lives_a].astype(np.int16)) < 0))
    prog = allrows[:, plo].astype(np.int32) + 256 * allrows[:, phi].astype(np.int32)

    receipt = {
        "what": "differential verification of configs/castlevania.yaml "
                "solve.entity_slots (asserted lo=0x0450 hi=0x0458)",
        "why": "the one solve byte-range with no receipt; the gate-opener "
               "arm's (a0)/(a1) and Phase 0 rest on it",
        "purity": "whole-page 0x0400-0x04FF sweep, every action in the "
                  "profile's own action space; claimed bounds treated as a "
                  "hypothesis to falsify. No map/disassembly/walkthrough.",
        "site": a.site,
        "root_state": a.root_state,
        "hw_flags": hw, "frame_skip": fs, "seed": a.seed,
        "steps": {"per_hold_arm": a.hold_steps, "n_hold_arms": len(masks),
                  "sticky_random": a.random_steps,
                  "total": int(allrows.shape[0])},
        "wall_secs": round(elapsed, 1),
        "criteria": {"CARD_MAX": CARD_MAX, "KILL_LIFT": KILL_LIFT,
                     "DISTINCT_MIN": DISTINCT_MIN},
        "descriptive_only": (
            "churn_1k_random and zero_fraction are reported per address but "
            "are NOT verdict inputs. verify_ram_map's CHURN_MAX=5/1k grades a "
            "byte on being STABLE, which is the wrong test for a flag array "
            "whose function is to toggle; applying it here would refuse the "
            "class by construction. The per-address S1..S4/qualifies fields "
            "are that inapplicable grading, retained for transparency."),
        "site_sanity": {
            "deaths_observed": deaths,
            "progress_min": int(prog.min()), "progress_max": int(prog.max()),
            "level_key_values": sorted({int(v) for v in
                                        allrows[:, int(solve['level_key'][0])]}),
        },
        "verdict": {
            "V1_bounds_derived_match_claim": v1,
            "V2_spawn_despawn_and_distinct": v2,
            "V3_kill_differential": v3,
            "all_pass": bool(v1 and v2 and v3),
            "claimed_addresses": [f"0x{x:04x}" for x in claimed],
            "binary_runs_in_page": [
                {"lo": f"0x{r[0]:04x}", "hi": f"0x{r[-1] + 1:04x}", "len": len(r)}
                for r in runs],
            "both_edges_per_address": both_edges,
            "distinct_column_signatures": len(sigs),
            "pooled_clear_per_1k_by_arm": {
                k: round(v, 3)
                for k, v in sorted(pooled.items(), key=lambda kv: -kv[1])},
            "pooled_best_arm": pool_best[0],
            "pooled_noop_clear_per_1k": round(pool_base, 3),
            "pooled_lift_vs_noop": (round(pool_lift, 3)
                                    if pool_lift != float("inf") else "inf"),
        },
        "kill_differential": kill,
        "per_address": {v["addr"]: v for v in
                        (per[x] for x in range(PAGE_LO, PAGE_HI))},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(receipt, indent=2, sort_keys=False))

    print(f"steps {allrows.shape[0]}  wall {elapsed:.1f}s  deaths {deaths}")
    print(f"progress range {prog.min()}..{prog.max()}  "
          f"level_key {receipt['site_sanity']['level_key_values']}")
    print(f"V1 bounds derived==claim : {v1}   runs="
          f"{[(f'0x{r[0]:04x}-0x{r[-1] + 1:04x}', len(r)) for r in runs[:3]]}")
    print(f"V2 spawn/despawn+distinct: {v2}   distinct_cols={len(sigs)}/8")
    print(f"V3 kill differential     : {v3}   best={pool_best[0]} "
          f"{pool_best[1]:.2f} vs NOOP {pool_base:.2f} per 1k "
          f"(lift {pool_lift:.2f}x)")
    for x in claimed:
        p = per[x]
        print(f"  {p['addr']} card={p['cardinality']:3d} "
              f"z={p['zero_fraction']:.3f} churn={p['churn_1k_random']:7.2f} "
              f"on={p['onsets']:5d} off={p['clears']:5d} "
              f"lift={kill[p['addr']]['lift_vs_noop']} "
              f"top={list(p['top_values'])[:3]}")
    print(f"receipt -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
