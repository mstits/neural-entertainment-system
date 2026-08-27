"""Detect NMI firings in our emulator by watching PC jump to the NMI
vector address (E45B for Zelda Rev A). Compare with nes-py's trace."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import nes_core


ROM = REPO / "roms" / "Legend of Zelda, The (USA) (Rev A).nes"
TAPE = REPO / "roms" / "zelda_start_419.state.bin"
NMI_VECTOR_TARGET = 0xE45B  # Zelda's NMI handler entry


def main():
    env = nes_core.NESEnvironment(rom_path=str(ROM), frame_skip=1)
    env.reset()
    tape = list(TAPE.read_bytes())

    # Fast-forward using env.step for frames we don't care about
    for i, b in enumerate(tape[:1005]):
        env.step(int(b) & 0xFF)

    # For frames 1005..1015, walk instruction-by-instruction
    prev_ppu_frame = env.ppu_state()[1]
    frame_index = 1005
    nmi_seen_this_frame = False
    frame_end_snapshots = []

    # Apply frame 1005 button, then step by instruction through frame 1020
    env.set_buttons(int(tape[1005]) & 0xFF)

    while frame_index < 1020:
        # Before step, snapshot PPU frame
        st_before = env.ppu_state()
        pc_before = env.cpu_state()[0]
        pc_after, opcode_done, _ = env.step_one_instruction()
        st_after = env.ppu_state()

        # NMI detection: PC jumped to NMI vector target AND wasn't there before
        if pc_after == NMI_VECTOR_TARGET and pc_before != NMI_VECTOR_TARGET:
            # Only report once per frame
            if not nmi_seen_this_frame:
                print(f"f{frame_index}: NMI_ENTRY cpu_cyc={st_after[4]} "
                      f"ppu_sl={st_before[2]} ppu_slc={st_before[3]}")
                nmi_seen_this_frame = True

        # Detect frame boundary. $11/$84/$70/$657 are read only for this
        # fidelity trace (cross-checking our CPU/PPU/NMI timing against
        # nes-py on this specific ROM) — never for reward or win logic.
        # $84/$70 are quarantined as TRAINING semantics in
        # configs/zelda.yaml (see tests/test_no_new_name_dispatch.py's
        # QUARANTINED_PAIR_EXEMPT for this file's standing exemption).
        if st_after[1] > prev_ppu_frame:
            ram = env.get_ram_range(0, 0x800)
            print(f"f{frame_index}: FRAME_END cpu={st_after[4]} "
                  f"$06={ram[0x06]:02X} $11={ram[0x11]:02X} "
                  f"$84={ram[0x84]:02X} $70={ram[0x70]:02X} "
                  f"$657={ram[0x657]:02X}")
            prev_ppu_frame = st_after[1]
            frame_index += 1
            nmi_seen_this_frame = False
            if frame_index < len(tape):
                env.set_buttons(int(tape[frame_index]) & 0xFF)


if __name__ == "__main__":
    main()
