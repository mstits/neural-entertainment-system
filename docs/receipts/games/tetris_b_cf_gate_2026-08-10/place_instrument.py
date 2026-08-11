"""B-TYPE placement instrument: drive the quota to 0 and bank the action trace.

Emits actions as INDICES INTO configs/tetris_b.yaml's action_space, produced on
exactly the machine counterfactual_probe replays on (Pool, frame_skip 2, no hw
flags, root blob + one rooting NOOP), so the banked trace is directly feedable
to the gate.

Instrument only -- nothing it discovers is written into the profile.
"""
import json
import sys
import time
import numpy as np
import nes_core as nc

ROM = "roms/Tetris (USA).nes"
ROOT = "roms/Tetris (USA)_btype_start.state.bin"
B = {'A': nc.BUTTON_A, 'B': nc.BUTTON_B, 'left': nc.BUTTON_LEFT,
     'right': nc.BUTTON_RIGHT, 'down': nc.BUTTON_DOWN}
SPACE = [[], ['left'], ['right'], ['down'], ['A'], ['B'],
         ['down', 'left'], ['down', 'right']]
MASK = np.array([sum(B[b] for b in a) for a in SPACE], dtype=np.uint8)
NOOP, LEFT, RIGHT, DOWN, ROT = 0, 1, 2, 3, 4
BOARD = (0x0400, 0x04C8)
EMPTY = 0xEF
QUOTA = 0x0050
SETTLE = 22
DROP_CAP = 44


def board(ram):
    return [[0 if ram[BOARD[0] + r * 10 + c] == EMPTY else 1
             for c in range(10)] for r in range(20)]


def bcd(v):
    return (v >> 4) * 10 + (v & 0xF)


def score(bd, lines):
    h = [0] * 10
    for c in range(10):
        for r in range(20):
            if bd[r][c]:
                h[c] = 20 - r
                break
    holes = 0
    for c in range(10):
        seen = False
        for r in range(20):
            if bd[r][c]:
                seen = True
            elif seen:
                holes += 1
    bump = sum(abs(h[c] - h[c + 1]) for c in range(9))
    return (-0.510066 * sum(h) + 3.0 * lines
            - 0.35663 * holes - 0.184483 * bump)


class Cand:
    """Per-worker adaptive placement: rotate, shift, soft-drop to lock, settle."""

    def __init__(self, rot, dcol, base_occ):
        self.plan = [ROT, NOOP] * rot
        self.plan += [(RIGHT if dcol > 0 else LEFT), NOOP] * abs(dcol)
        self.i = 0
        self.base_occ = base_occ
        self.phase = 0          # 0 = plan, 1 = drop, 2 = settle, 3 = done
        self.settle = 0
        self.drop = 0
        self.actions = []

    def next(self):
        if self.phase == 0:
            if self.i < len(self.plan):
                a = self.plan[self.i]
                self.i += 1
            else:
                self.phase, a = 1, DOWN
        elif self.phase == 1:
            a = DOWN
        elif self.phase == 2:
            a = NOOP
        else:
            return None
        self.actions.append(a)
        return a

    def observe(self, ram):
        occ = sum(1 for v in ram[BOARD[0]:BOARD[1]] if v != EMPTY)
        if self.phase == 1:
            self.drop += 1
            if occ != self.base_occ or self.drop >= DROP_CAP:
                self.phase = 2
        elif self.phase == 2:
            self.settle += 1
            if self.settle >= SETTLE:
                self.phase = 3


def run(max_pieces=90, budget_s=280.0):
    pool = nc.Pool(rom_path=ROM, num_workers=4, frame_skip=2)
    pool.set_headless(True)
    pool.set_skip_preprocess(True)
    pool.reset_all()
    blob = open(ROOT, 'rb').read()
    for w in range(4):
        pool.load_worker_state(w, blob)
    acts = np.zeros(4, dtype=np.uint8)
    ram = pool.step_all(acts)[0][2]          # the rooting NOOP
    committed = pool.save_worker_state(0)
    trace = []
    timeline = []                            # (step, quota, occ, row, state)
    t0 = time.time()
    base_q = bcd(ram[QUOTA])

    for piece in range(max_pieces):
        if time.time() - t0 > budget_s:
            print("[instrument] budget exhausted", flush=True)
            break
        base_occ = sum(1 for v in ram[BOARD[0]:BOARD[1]] if v != EMPTY)
        q_before = bcd(ram[QUOTA])
        cands = [(r, d) for r in range(4) for d in range(-5, 5)]
        results = []
        for batch0 in range(0, len(cands), 4):
            chunk = cands[batch0:batch0 + 4]
            objs = [Cand(r, d, base_occ) for r, d in chunk]
            for w in range(len(chunk)):
                pool.load_worker_state(w, committed)
            live = list(range(len(chunk)))
            for _ in range(200):
                if not live:
                    break
                for w in range(4):
                    a = objs[w].next() if w < len(chunk) else None
                    acts[w] = MASK[a if a is not None else NOOP]
                out = pool.step_all(acts)
                for w in list(live):
                    objs[w].observe(out[w][2])
                    if objs[w].phase == 3:
                        live.remove(w)
            for w, (r, d) in enumerate(chunk):
                rm = out[w][2]
                q = bcd(rm[QUOTA])
                dead = rm[0x0048] == 10
                results.append((score(board(rm), q_before - q) - (1e6 if dead else 0),
                                r, d, objs[w].actions))
        results.sort(key=lambda x: -x[0])
        best = results[0]
        # commit it on worker 0, recording the per-step timeline
        pool.load_worker_state(0, committed)
        for a in best[3]:
            acts[0] = MASK[a]
            ram = pool.step_all(acts)[0][2]
            trace.append(a)
            timeline.append((len(trace), ram[QUOTA],
                             sum(1 for v in ram[BOARD[0]:BOARD[1]] if v != EMPTY),
                             ram[0x0041], ram[0x0048], ram[0x00D8]))
        committed = pool.save_worker_state(0)
        q = bcd(ram[QUOTA])
        if q != q_before:
            print(f"[piece {piece:3d}] CLEAR {q_before - q}  quota {q_before}->{q}  "
                  f"step {len(trace)}  t={time.time()-t0:.1f}s", flush=True)
        if ram[QUOTA] == 0:
            print(f"[instrument] WIN at piece {piece}, step {len(trace)}, "
                  f"{time.time()-t0:.1f}s", flush=True)
            break
        if ram[0x0048] == 10:
            print(f"[instrument] TOP OUT at piece {piece}", flush=True)
            break

    pool.shutdown()
    return trace, timeline, ram


if __name__ == "__main__":
    mp = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    bs = float(sys.argv[2]) if len(sys.argv) > 2 else 280.0
    tr, tl, ram = run(mp, bs)
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/tetb_gate_probe"
    np.save(out + "_trace.npy", np.array(tr, dtype=np.int64))
    with open(out + "_timeline.json", "w") as f:
        json.dump(tl, f)
    print(json.dumps({"steps": len(tr), "quota": ram[QUOTA],
                      "state": ram[0x0048], "d8": ram[0x00D8]}))
