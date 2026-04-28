-- Mesen 2 Lua: SMB walk-right scenario, per-frame RAM dump.
--
-- Scenario (same as /tmp/repro_mario_floor.py):
--   frames   0..49  : blank (title settles)
--   frames  50..55  : Start pressed (enter game)
--   frames  56..105 : blank (level loads)
--   frames 106..305 : Right held (walk Mario)
--
-- Per-frame, dump the full $0000-$07FF RAM (2KB) to a binary file at
-- offset (frame * 2048). The Python diff harness will then find the
-- first frame where nes_core and Mesen RAM diverge, and within that
-- frame the first byte that differs.
--
-- Run:
--   /Applications/Mesen.app/Contents/MacOS/Mesen \
--     --testRunner --noaudio --novideo --noinput \
--     scripts/tracing/mesen_smb_walk.lua \
--     "roms/Super Mario Bros. (World).nes" \
--     --timeout=60

local DUMP_PATH = "/tmp/mesen_smb_walk_ram.bin"
local STATUS_PATH = "/tmp/mesen_smb_walk.status.txt"
local TOTAL_FRAMES = 306

-- Per-frame input bitmask in our standard layout
-- (bit 7 = Right, 3 = Start, 0 = A, etc.).
local function input_for_frame(f)
    if f >= 50 and f < 56 then return 0x08 end       -- Start
    if f >= 106 and f < 306 then return 0x80 end     -- Right
    return 0
end

local function mask_to_mesen(m)
    return {
        a      = (m & 0x01) ~= 0,
        b      = (m & 0x02) ~= 0,
        select = (m & 0x04) ~= 0,
        start  = (m & 0x08) ~= 0,
        up     = (m & 0x10) ~= 0,
        down   = (m & 0x20) ~= 0,
        left   = (m & 0x40) ~= 0,
        right  = (m & 0x80) ~= 0,
    }
end

local dump_file = io.open(DUMP_PATH, "wb")
local status_file = io.open(STATUS_PATH, "w")
status_file:write("script-loaded\n"); status_file:flush()

local frame = 0

function on_input_polled()
    local m = input_for_frame(frame)
    emu.setInput(0, mask_to_mesen(m))
end

function on_end_frame()
    -- Snapshot $0000-$07FF (2KB internal RAM) at the END of this frame
    if dump_file == nil then return end
    local buf = {}
    for addr = 0, 0x7FF do
        buf[#buf + 1] = string.char(emu.read(addr, emu.memType.nesMemory))
    end
    dump_file:write(table.concat(buf))
    frame = frame + 1
    if frame >= TOTAL_FRAMES then
        dump_file:close(); dump_file = nil
        status_file:write("done frames=" .. frame .. "\n")
        status_file:close()
        emu.stop(0)
    end
end

emu.addEventCallback(on_input_polled, emu.eventType.inputPolled)
emu.addEventCallback(on_end_frame, emu.eventType.endFrame)
status_file:write("callbacks-registered\n"); status_file:flush()
