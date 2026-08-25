"""IS-1a runner (Stages 1-3a): replays already-banked Zelda RG-1 probe
traces (traces.pkl action sequences from the real 90-minute RG-1 runs)
against roms/zelda_start_ctrl.state.bin to reconstruct real per-step RAM
logs, feeds them to discover_item_bits.py Stages 1-2, and runs Stage 3a
over the real banked room_index.json files. Not committed (orchestrator
commits) and not part of the shipped engine -- a one-shot receipt
generator for docs/receipts/item_bits / runs/item_semantics/is1a/.

No RAM map, no walkthrough: every RAM address scanned is our own
console RAM ($0000-$07FF) read the way get_ram_range/Pool.step_all
already read it. The action sequences replayed are exactly the ones
the real Go-Explore archive already banked (mined, never authored).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from nes_core import Pool

from scripts.discover_item_bits import (
    RAM_SIZE, confirm_across_rollouts, correlate_boundary_edges,
    idle_mask_from_rom,
)

ROM = str(REPO / "roms/Legend of Zelda, The (USA) (Rev A).nes")
ROOT_STATE = REPO / "roms/zelda_start_ctrl.state.bin"
RUNS = [
    REPO / "runs/room_graph/rg1_zelda_seed0_bias025",
    REPO / "runs/room_graph/rg1_zelda_seed0_bias000",
    REPO / "runs/room_graph/rg1_zelda_seed1_bias025",
    REPO / "runs/room_graph/rg1_zelda_seed1_bias000",
]
# Already-verified Zelda observables (zelda_onboarding_2026-08-10.md):
# progress.lo (link_x), y (link_y), lives (fine HP). Excluded per §6
# item 4 (dedup) -- these are ALREADY claimed by an existing observable,
# not candidate item bits.
CLAIMED_ADDRS = frozenset({0x0070, 0x0084, 0x0670})
N_TRACES_PER_RUN = 3          # spread across 4 banked runs => 12 rollouts
MAX_TRACE_LEN = 2600          # RG-1's own measured max (2573) -- no cap needed
                              # in practice, kept as an explicit bound.


def pick_traces(traces: dict, k: int) -> list:
    """Pick k traces spanning the run's own length distribution
    (shortest / median / longest thirds) -- real banked data, no
    cherry-picking for outcome."""
    items = sorted(traces.items(), key=lambda kv: len(kv[1][1]))
    if not items:
        return []
    n = len(items)
    idxs = sorted({0, n // 2, n - 1})
    while len(idxs) < k and len(idxs) < n:
        cand = int(len(idxs) / k * n)
        if cand not in idxs:
            idxs.append(cand)
        else:
            break
    idxs = sorted(set(idxs))[:k]
    return [items[i] for i in idxs]


LIVES_ADDR = 0x0670  # configs/zelda.yaml solve.lives -- ALREADY a verified,
                     # claimed observable (zelda_onboarding_2026-08-10.md),
                     # not new game knowledge. Used here only to truncate a
                     # replayed rollout at the death->CONTINUE-menu boundary
                     # before scanning, mirroring discover_observables.py's
                     # own settle/reset truncation discipline for this exact
                     # game (its docstring names Zelda's death-into-menu
                     # animation as its hardest reset-detection case).


def replay_trace(pool, root_state_bytes: bytes, action_bytes: bytes) -> np.ndarray:
    """Deterministic replay of a real banked action sequence from the
    real root state. Returns RAM($0..RAM_SIZE) per step, shape
    (len(action_bytes), RAM_SIZE)."""
    pool.load_worker_state(0, root_state_bytes)
    n = len(action_bytes)
    log = np.empty((n, RAM_SIZE), dtype=np.uint8)
    for t in range(n):
        r = pool.step_all(np.array([action_bytes[t]], dtype=np.uint8))
        log[t] = np.frombuffer(bytes(r[0][2]), dtype=np.uint8)[:RAM_SIZE]
    return log


def truncate_at_death(log: np.ndarray) -> np.ndarray:
    """Drop every step from the first lives==0 frame onward -- the
    death animation + CONTINUE/SAVE/RETRY menu tail that
    zelda_onboarding_2026-08-10.md documents as this game's hardest
    reset-detection case (mass-RAM-churn detection fails on it, so it
    is truncated explicitly here by the ALREADY-CLAIMED lives byte
    rather than left to look like a monotone capability flag)."""
    dead = np.where(log[:, LIVES_ADDR] == 0)[0]
    return log if len(dead) == 0 else log[: int(dead[0])]


def main() -> int:
    root_bytes = ROOT_STATE.read_bytes()
    receipt: dict = {"root_state": str(ROOT_STATE.relative_to(REPO)),
                     "rom": str(Path(ROM).relative_to(REPO)),
                     "claimed_addrs": sorted(CLAIMED_ADDRS)}

    # --- idle-prefilter: one short single-worker idle probe -----------
    print("[is1a] computing idle mask ...", flush=True)
    idle_mask = idle_mask_from_rom(ROM, str(ROOT_STATE), frame_skip=4,
                                   forward="right", seed=1)
    receipt["idle_mask_size"] = len(idle_mask)
    receipt["idle_mask_addrs"] = sorted(idle_mask)
    print(f"[is1a] idle mask: {len(idle_mask)} addrs -> {sorted(idle_mask)}",
         flush=True)

    # --- Stage 1-2: replay real banked traces, scan for flag bits -----
    pool = Pool(rom_path=ROM, num_workers=1, frame_skip=4)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()

    rollout_logs = []
    rollout_ids = []
    trace_meta = []
    for run_dir in RUNS:
        with open(run_dir / "traces.pkl", "rb") as f:
            traces = pickle.load(f)
        picks = pick_traces(traces, N_TRACES_PER_RUN)
        for key, rec in picks:
            action_bytes = bytes(rec[1])[:MAX_TRACE_LEN]
            if len(action_bytes) < 2:
                continue
            rid = f"{run_dir.name}:{key}"
            print(f"[is1a] replaying {rid} ({len(action_bytes)} actions) ...",
                 flush=True)
            log = replay_trace(pool, root_bytes, action_bytes)
            rollout_logs.append(log)
            rollout_ids.append(rid)
            trace_meta.append({"rollout_id": rid, "n_actions": len(action_bytes),
                               "run": run_dir.name, "cell_key": list(key)})
    receipt["n_rollouts"] = len(rollout_logs)
    receipt["rollouts"] = trace_meta

    # Persist raw logs for follow-up false-positive-class diagnostics
    # (not part of the ledger; a receipt artifact only).
    logs_path = Path(__file__).parent / "is1a_rollout_logs.npz"
    np.savez_compressed(logs_path,
                        **{f"log_{i}": log for i, log in enumerate(rollout_logs)},
                        rollout_ids=np.array(rollout_ids, dtype=object))
    print(f"[is1a] saved raw rollout logs -> {logs_path}", flush=True)

    def _summarize(ledger):
        by_status = {"candidate": [], "confirmed": [], "rejected": []}
        for key, cand in ledger.items():
            by_status[cand.status].append({
                "addr": cand.addr, "bit": cand.bit,
                "monotone_rollouts": cand.monotone_rollouts,
                "total_rollouts": cand.total_rollouts,
                "reverts_seen": cand.reverts_seen,
                "change_rate": round(cand.change_rate, 4),
                "first_seen": cand.first_seen,
            })
        return by_status

    # --- RAW: scan the rollouts exactly as replayed (may include a
    # death->CONTINUE-menu tail if the mined lineage died in-window).
    ledger_raw = confirm_across_rollouts(
        rollout_logs, confirm_k=3, claimed_addrs=CLAIMED_ADDRS,
        idle_mask=idle_mask, rollout_ids=rollout_ids)
    raw_status = _summarize(ledger_raw)
    receipt["stage12_raw_untruncated"] = {
        "note": "Stage1-2 over the replayed rollouts AS-IS, no death "
               "truncation. Kept for the record: this is the reading "
               "any caller gets if they feed Stage1-2 raw RAM without "
               "excluding post-death frames.",
        "n_keys_seen": len(ledger_raw),
        "n_confirmed": len(raw_status["confirmed"]),
        "n_candidate": len(raw_status["candidate"]),
        "n_rejected": len(raw_status["rejected"]),
        "confirmed": raw_status["confirmed"],
    }
    print(f"[is1a] RAW ledger: {len(ledger_raw)} keys, "
         f"{len(raw_status['confirmed'])} confirmed, "
         f"{len(raw_status['candidate'])} candidate, "
         f"{len(raw_status['rejected'])} rejected", flush=True)

    # --- DEATH-TRUNCATED: the methodologically fair run -- truncate
    # each rollout at the already-claimed `lives` ($0670) byte hitting
    # 0, mirroring discover_observables.py's own settle/reset
    # discipline for this exact game. No new address, no new game
    # knowledge -- lives is already wired into configs/zelda.yaml
    # solve.lives and already excluded via CLAIMED_ADDRS.
    truncated_logs = [truncate_at_death(log) for log in rollout_logs]
    n_truncated = sum(1 for a, b in zip(rollout_logs, truncated_logs)
                      if len(b) < len(a))
    truncated_logs = [log for log in truncated_logs if len(log) >= 2]
    truncated_ids = [rid for rid, log in
                     zip(rollout_ids, [truncate_at_death(l) for l in rollout_logs])
                     if len(log) >= 2]
    ledger = confirm_across_rollouts(
        truncated_logs, confirm_k=3, claimed_addrs=CLAIMED_ADDRS,
        idle_mask=idle_mask, rollout_ids=truncated_ids)
    by_status = _summarize(ledger)
    receipt["stage12"] = {
        "note": "Death-truncated (the fair run): each rollout cut at "
               "the first lives==0 frame before scanning.",
        "n_rollouts_truncated_for_death": n_truncated,
        "n_keys_seen": len(ledger),
        "n_confirmed": len(by_status["confirmed"]),
        "n_candidate": len(by_status["candidate"]),
        "n_rejected": len(by_status["rejected"]),
        "confirmed": by_status["confirmed"],
        "candidate": by_status["candidate"],
    }
    print(f"[is1a] DEATH-TRUNCATED ledger ({n_truncated}/{len(rollout_logs)} "
         f"rollouts cut for death): {len(ledger)} keys, "
         f"{len(by_status['confirmed'])} confirmed, "
         f"{len(by_status['candidate'])} candidate, "
         f"{len(by_status['rejected'])} rejected", flush=True)

    # Doctrine cross-check (reporting only, never used as scanner input):
    # do rupees ($066D) / keys ($066E) -- the disassembly-sourced
    # ram_mapping addresses NEVER consulted as evidence -- ever move
    # across any replayed rollout? Confirms the "already-banked
    # rollouts never show a pickup" premise the gate depends on,
    # exactly as zelda_onboarding_2026-08-10.md §4 already found for
    # its own probes.
    rupees_moved = any(bool(np.any(log[:, 0x066D] != log[0, 0x066D]))
                       for log in rollout_logs)
    keys_moved = any(bool(np.any(log[:, 0x066E] != log[0, 0x066E]))
                     for log in rollout_logs)
    receipt["doctrine_crosscheck_reporting_only"] = {
        "note": "0x066D/0x066E are disassembly-sourced ram_mapping "
               "addresses (configs/zelda.yaml); NEVER consulted as "
               "scanner input, checked here only to confirm the "
               "true-negative premise the gate depends on.",
        "rupees_addr_0x066D_ever_changed": rupees_moved,
        "keys_addr_0x066E_ever_changed": keys_moved,
    }

    # --- Stage 3a: correlational read over the real banked room_index.json
    stage3a = {}
    for run_dir in RUNS:
        with open(run_dir / "room_index.json") as f:
            ri = json.load(f)
        n_edges = sum(len(dsts) for dsts in ri.get("adj", {}).values())
        has_cap_hist = any(
            "cap_hist" in e
            for dsts in ri.get("adj", {}).values() for e in dsts.values())
        leads_by_bit = {}
        for bit in range(8):
            leads = correlate_boundary_edges(ri, bit)
            leads_by_bit[bit] = len(leads)
        stage3a[run_dir.name] = {
            "n_edges": n_edges,
            "cap_hist_key_present_pregraft_run": has_cap_hist,
            "leads_per_bit": leads_by_bit,
            "total_leads": sum(leads_by_bit.values()),
        }
        print(f"[is1a] Stage3a {run_dir.name}: {n_edges} edges, "
             f"cap_hist present={has_cap_hist}, "
             f"total leads={sum(leads_by_bit.values())}", flush=True)
    receipt["stage3a"] = stage3a

    out = Path(__file__).parent / "is1a_stage12_3a_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"[is1a] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
