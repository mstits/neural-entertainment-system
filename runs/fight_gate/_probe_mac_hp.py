import sys, json
sys.path.insert(0, "/Users/stits/Documents/macos-emulation-and-training/scripts")
import discover_observables as do

rom = "/Users/stits/Documents/macos-emulation-and-training/roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A).nes"
state = open("/Users/stits/Documents/macos-emulation-and-training/roms/Mike Tyson's Punch-Out!! (Japan, USA) (Rev A)_start.state.bin", "rb").read()

disc = do.Discoverer(rom, state, 4, "right", 1)  # same seed=1 as the discover_all run
atk = disc.attack_mash()
apr = disc.approach_retreat()
disc.close()

# Unrestricted ranking (top=2048) so nothing gets truncated out of view.
full = do.fight_health_from_drives(atk, apr, self_hp_addr=0x00BC, top=2048)

def find_rank(cands, addr):
    for i, c in enumerate(cands):
        if c["addr"] == addr:
            return i, c
    return None, None

out = {}
for label, addr in [("mac_hp_0x392", 0x392), ("mac_hp_mirror_0x391", 0x391),
                     ("opp_hp_0x398", 0x398)]:
    ri, c = find_rank(full["foe_hp_candidates"], addr)
    si, sc = find_rank(full["self_hp_candidates"], addr)
    out[label] = {"foe_rank": ri, "foe_entry": c, "self_rank": si, "self_entry": sc}

# Also raw decrement-consensus directly, independent of classification bucket.
atk_logs = [__import__("numpy").asarray(d["log"]) for d in atk]
apr_logs = [__import__("numpy").asarray(d["log"]) for d in apr]
for label, addr in [("mac_hp_0x392", 0x392), ("mac_hp_mirror_0x391", 0x391)]:
    fh1 = do._decrement_consensus(atk_logs, addr, min_agree=do.DEATH_MIN_AGREE)
    fh2 = do._decrement_consensus(apr_logs, addr, min_agree=do.DEATH_MIN_AGREE)
    out[label]["raw_fh1_attack"] = fh1
    out[label]["raw_fh2_defense"] = fh2

print(json.dumps(out, indent=2, default=int))
with open("/Users/stits/Documents/macos-emulation-and-training/runs/fight_gate/mac_hp_probe.json", "w") as f:
    json.dump(out, f, indent=2, default=int)
