-- Mesen 2 Lua: CV lockstep tape replay, per-frame RAM dump.
-- Reads /tmp/cv_tape.bin (one input bitmask byte per frame, our layout),
-- dumps $0000-$07FF per frame to /tmp/mesen_cv_ram.bin.
-- Run:
--   /Applications/Mesen.app/Contents/MacOS/Mesen \
--     --testRunner --noaudio --novideo --noinput \
--     scripts/tracing/mesen_cv_tape_dump.lua \
--     "roms/Castlevania (USA).nes" --timeout=180

local TAPE_PATH = "/tmp/cv_tape.bin"
local DUMP_PATH = "/tmp/mesen_cv_ram.bin"
local STATUS_PATH = "/tmp/mesen_cv_tape.status.txt"

local tf = io.open(TAPE_PATH, "rb")
local tape = tf:read("*a")
tf:close()
local TOTAL_FRAMES = #tape

local dump = io.open(DUMP_PATH, "wb")
local frame = 0

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

local function on_input(_)
    if frame < TOTAL_FRAMES then
        local m = string.byte(tape, frame + 1)
        emu.setInput(mask_to_mesen(m), 0)
    end
end

local function on_frame_end()
    local bytes = {}
    for a = 0, 0x7FF do
        bytes[#bytes + 1] = string.char(emu.read(a, emu.memType.nesMemory))
    end
    dump:write(table.concat(bytes))
    frame = frame + 1
    if frame >= TOTAL_FRAMES then
        dump:close()
        local st = io.open(STATUS_PATH, "w")
        st:write("done " .. frame .. " frames\n")
        st:close()
        emu.stop(0)
    end
end

emu.addEventCallback(on_input, emu.eventType.inputPolled)
emu.addEventCallback(on_frame_end, emu.eventType.endFrame)
