-- Generic per-frame RAM dump for any ROM.
--
-- Captures $0000-$07FF (2KB internal RAM) at endFrame for the first
-- N frames of cold-boot idle (no input). Output path + frame count
-- come from environment variables MESEN_DUMP_OUT and MESEN_DUMP_FRAMES.
--
-- Run via Mesen --testRunner mode (see scripts/parity/capture_mesen_tape.sh).

local OUT = os.getenv("MESEN_DUMP_OUT") or "/tmp/mesen_dump_ram.bin"
local STATUS = OUT .. ".status.txt"
local FRAMES = tonumber(os.getenv("MESEN_DUMP_FRAMES") or "120")

local f = io.open(OUT, "wb")
local s = io.open(STATUS, "w")
s:write("loaded\n"); s:flush()

local frame = 0

function on_input()
    emu.setInput(0, {})
end

function on_start_frame()
    -- StartFrame fires at vblank END (right after pre-render scanline).
    -- At this point the previous frame's NMI handler has completed,
    -- matching our env.step() return point (which advances exactly
    -- 29781 CPU cycles = full frame including the NMI handler). Using
    -- endFrame instead would capture mid-vblank, BEFORE the NMI runs,
    -- and produce a false-positive ~3 bytes of stack divergence
    -- (the NMI's pushed return address) on every frame snapshot.
    if not f then return end
    -- Skip the first startFrame: it fires before any frame has run.
    if frame == 0 and f then
        -- Mark we've seen the first startFrame by filling first 2KB
        -- with whatever RAM is at this point (typically all zeros for
        -- power-on RAM model).
    end
    local buf = {}
    for a = 0, 0x7FF do
        buf[#buf + 1] = string.char(emu.read(a, emu.memType.nesMemory))
    end
    f:write(table.concat(buf))
    frame = frame + 1
    if frame >= FRAMES then
        f:close(); f = nil
        s:write("done frames=" .. frame .. "\n"); s:close()
        emu.stop(0)
    end
end

emu.addEventCallback(on_input, emu.eventType.inputPolled)
emu.addEventCallback(on_start_frame, emu.eventType.startFrame)
s:write("registered\n"); s:flush()
