-- Mesen 2 Lua: CV tape replay + CPU instruction trace in a frame window.
-- Tape: /tmp/cv_tape.bin (one bitmask byte per frame, our layout).
-- Trace window frames set below; output /tmp/mesen_cv_trace.txt.
-- Run headless:
--   /Applications/Mesen.app/Contents/MacOS/Mesen --testRunner --noaudio \
--     --novideo --noinput scripts/tracing/mesen_cv_trace.lua "roms/CV.nes"

local TAPE_PATH = "/tmp/cv_tape.bin"
local LOG_PATH = "/tmp/mesen_cv_trace.txt"
local TRACE_START = 3790
local TRACE_END = 3800
local MAX_INSTR = 400000

local tf = io.open(TAPE_PATH, "rb")
local tape = tf:read("*a")
tf:close()
local TOTAL_FRAMES = #tape

local log_file = io.open(LOG_PATH, "w")
local frame = 0
local instr_count = 0

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
        emu.setInput(mask_to_mesen(string.byte(tape, frame + 1)), 0)
    end
end

local function on_frame_end()
    frame = frame + 1
    if frame > TRACE_END + 2 or frame >= TOTAL_FRAMES then
        if log_file then log_file:close(); log_file = nil end
        emu.stop(0)
    end
end

local function on_exec(address, value)
    if log_file == nil or instr_count >= MAX_INSTR then return end
    if frame < TRACE_START or frame > TRACE_END then return end
    -- this Mesen build returns a FLAT state table with dotted keys
    local s = emu.getState()
    local pc = s["cpu.pc"]
    local op = emu.read(pc, emu.memType.nesDebug)
    log_file:write(string.format(
        "%04X %02X A:%02X X:%02X Y:%02X P:%02X SP:%02X PPU:%3d,%3d CYC:%d F:%d\n",
        pc, op, s["cpu.a"], s["cpu.x"], s["cpu.y"], s["cpu.ps"], s["cpu.sp"],
        s["ppu.scanline"], s["ppu.cycle"], s["cpu.cycleCount"], frame))
    instr_count = instr_count + 1
end

emu.addEventCallback(on_input, emu.eventType.inputPolled)
emu.addEventCallback(on_frame_end, emu.eventType.endFrame)
emu.addMemoryCallback(on_exec, emu.callbackType.exec, 0, 0xFFFF)
