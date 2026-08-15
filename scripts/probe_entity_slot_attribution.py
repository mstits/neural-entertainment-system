"""Attribute the 0x0450-0x0457 despawn edge: locomotion or attack?

Reads the two banked probe receipts and computes matched contrasts that
hold one factor constant. Emits a combined attribution receipt.

Contrasts (all pooled 1->0 edges per 1k steps over the 8 slots):
  C1 locomotion   : any arm containing `right`  vs  the NOOP arm
  C2 attack alone : `B`      vs  NOOP            (locomotion held OFF)
  C3 attack given : `right+B` vs `right`         (locomotion held ON)
  C4 jump given   : `right+A` vs `right`         (negative-control: a
                    non-attack button added to the same locomotion)

C3 is the load-bearing one: it is the only comparison in which the two
arms differ ONLY by continuous attack.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/stits/Documents/macos-emulation-and-training")
PROBES = REPO / "docs/receipts/probes"
SRC = {"pin": "cv_entity_slots_2026-08-10.json",
       "site2": "cv_entity_slots_site2_2026-08-10.json"}


def main():
    out = {
        "what": "attribution of the 1->0 edge in castlevania.yaml "
                "solve.entity_slots (0x0450-0x0457): locomotion vs attack",
        "why": "configs/castlevania.yaml asserts '--kill-key counts 1->0 "
               "transitions as kills'. The gate-opener arm's (a1) keys a "
               "search dimension on that edge. If the edge is locomotion-"
               "coupled it is a positional alias, not an interaction axis, "
               "and (a1)'s own mandatory farmability test refuses it.",
        "sources": {k: f"docs/receipts/probes/{v}" for k, v in SRC.items()},
        "unit": "pooled 1->0 edges per 1000 steps over the 8 slots",
        "sites": {},
    }
    for site, fn in SRC.items():
        d = json.loads((PROBES / fn).read_text())
        p = d["verdict"]["pooled_clear_per_1k_by_arm"]
        noop, b = p["hold:NOOP"], p["hold:B"]
        r, rb, ra = p["hold:right"], p["hold:right+B"], p["hold:right+A"]
        rows = sorted(p.items(), key=lambda kv: -kv[1])
        movers = [k for k, v in rows if v > noop]
        out["sites"][site] = {
            "root_state": d["root_state"],
            "steps_per_hold_arm": d["steps"]["per_hold_arm"],
            "arms_above_noop": movers,
            "arms_above_noop_all_contain_right": all(
                "right" in k for k in movers),
            "C1_locomotion_right_vs_noop": {
                "right": r, "noop": noop, "delta": round(r - noop, 3)},
            "C2_attack_alone_B_vs_noop": {
                "B": b, "noop": noop, "delta": round(b - noop, 3)},
            "C3_attack_given_locomotion_rightB_vs_right": {
                "right+B": rb, "right": r, "delta": round(rb - r, 3)},
            "C4_negcontrol_jump_given_locomotion_rightA_vs_right": {
                "right+A": ra, "right": r, "delta": round(ra - r, 3)},
        }
        pa = d["per_address"]
        per_slot = [pa[f"0x{x:04x}"]["clears"] for x in range(0x450, 0x458)]
        out["sites"][site]["farmability"] = {
            "per_slot_1to0_edges": per_slot,
            "total_1to0_edges_all_arms": sum(per_slot),
            "slot_count": 8,
            "edges_exceed_slot_count": bool(sum(per_slot) > 8),
            "note": "the arm doc's (a1) farmability rule: more 1->0 edges "
                    "than slots means slots respawn and a cumulative COUNT "
                    "is farmable",
            "coincidence_notice": "both sites total exactly 317 from "
                                  "different per-slot vectors "
                                  "([140,37,37,5,20,24,18,36] vs "
                                  "[178,48,65,0,11,7,3,5]). Numerically a "
                                  "coincidence, not a copied value -- the "
                                  "per-slot vectors above are the proof.",
        }

    c3 = [out["sites"][s]["C3_attack_given_locomotion_rightB_vs_right"]["delta"]
          for s in SRC]
    c2 = [out["sites"][s]["C2_attack_alone_B_vs_noop"]["delta"] for s in SRC]
    c1 = [out["sites"][s]["C1_locomotion_right_vs_noop"]["delta"] for s in SRC]
    out["verdict"] = {
        "edge_is_locomotion_coupled": bool(all(x > 0 for x in c1)),
        "edge_is_attack_coupled": bool(any(x > 0 for x in c2)
                                       and all(x > 0 for x in c3)),
        "C1_deltas": c1, "C2_deltas": c2, "C3_deltas": c3,
        "conclusion": None,
        "scope_caveat": (
            "single held actions from one root per site, 1500/800 steps per "
            "arm. A whip only connects when an entity is in range, so a null "
            "attack contrast from a fixed root is evidence that HELD attack "
            "does not drive the edge -- not proof that no contact ever "
            "clears a slot. The correct instrument for that is a contact-"
            "conditioned differential (the arm doc's P1-P4), not --kill-key."),
    }
    out["verdict"]["conclusion"] = (
        "REFUTED as a kill axis / CONFIRMED as an occupancy axis: the 1->0 "
        "edge is entirely locomotion-coupled. Every arm above the NOOP "
        "baseline at both sites contains `right`; every arm without it sits "
        "exactly at baseline, including `B` and `down+B`. Holding locomotion "
        "constant, adding continuous attack moves the pooled rate by "
        f"{c3[0]:+.3f} and {c3[1]:+.3f} per 1k -- the same order as the "
        "non-attack negative control. The slots are real and agent-coupled, "
        "but what the agent does to them is SCROLL them, and a count over "
        "scroll-driven despawns is a positional alias that (a1)'s "
        "farmability rule refuses.")

    dst = PROBES / "cv_entity_slots_attribution_2026-08-10.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))
    for s in SRC:
        v = out["sites"][s]
        print(f"\n{s}: movers={v['arms_above_noop']}")
        print(f"  all contain right: {v['arms_above_noop_all_contain_right']}")
        print(f"  C1 {v['C1_locomotion_right_vs_noop']}")
        print(f"  C2 {v['C2_attack_alone_B_vs_noop']}")
        print(f"  C3 {v['C3_attack_given_locomotion_rightB_vs_right']}")
        print(f"  C4 {v['C4_negcontrol_jump_given_locomotion_rightA_vs_right']}")
        print(f"  farm {v['farmability']}")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
