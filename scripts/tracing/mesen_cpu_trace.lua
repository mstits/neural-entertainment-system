-- Mesen 2 Lua trace script for Zelda cave-bug comparison.
--
-- Auto-plays the Zelda tape (zelda_start_419.state.bin) from our
-- training repo through Mesen, logging every CPU instruction to
-- /tmp/mesen_zelda_trace.txt in a format matching
-- scripts/tracing/nes_core_cpu_trace.py.
--
-- How to run:
--   1. Open Mesen 2, load Zelda ROM (File -> Open Game)
--   2. Immediately Power Cycle (Cmd+R / Ctrl+R) so frame 0 is clean
--   3. Tools -> Scripts -> Open Script Window (or Ctrl+Shift+S)
--   4. File -> Open -> select this file, click "Run"
--   5. The script auto-feeds buttons frame-by-frame for ~3354 frames
--      (~60 seconds) and logs the CPU trace
--   6. When the emulator pauses on its own, trace is saved

local TAPE_PATH = "/Users/stits/Documents/macos-emulation-and-training/roms/zelda_start_419.state.bin"
local LOG_PATH = "/tmp/mesen_zelda_trace.txt"
local MAX_FRAMES = 1020       -- enough to cover the cave-entry at f1011
local MAX_INSTR = 500000      -- safety cap; ~5000 frames worth
local TRACE_START_FRAME = 1000 -- skip early boot logging to keep file small
local TRACE_END_FRAME = 1020

-- Our tape uses this bitmask layout (nes-py / LaiNES convention):
--   bit 7 = Right, 6 = Left, 5 = Down, 4 = Up
--   bit 3 = Start, 2 = Select, 1 = B, 0 = A
-- Mesen's Lua setInput() expects a table of booleans by name.
local function mask_to_mesen_input(m)
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

-- Sanity banner so we can see the script started
emu.log("[MESEN_TRACE] Script loaded")
emu.displayMessage("Trace", "Script loaded, trying IO")

-- Read the tape
local tape = {}
do
    local ok, f = pcall(io.open, TAPE_PATH, "rb")
    if not ok then
        emu.log("[MESEN_TRACE] io.open failed with error: " .. tostring(f))
        emu.displayMessage("Trace", "io.open raised: " .. tostring(f))
        return
    end
    if not f then
        emu.log("[MESEN_TRACE] io.open returned nil for tape: " .. TAPE_PATH)
        emu.displayMessage("Trace", "Cannot open tape (IO disabled?)")
        return
    end
    local data = f:read("*a")
    f:close()
    for i = 1, #data do
        tape[i] = string.byte(data, i)
    end
    emu.log("[MESEN_TRACE] Loaded " .. #tape .. " tape frames")
    emu.displayMessage("Trace", "Loaded " .. #tape .. " tape frames")
end

local log_file = io.open(LOG_PATH, "w")
if not log_file then
    emu.log("[MESEN_TRACE] Cannot open log: " .. LOG_PATH)
    emu.displayMessage("Trace", "Cannot open log")
    return
end
emu.log("[MESEN_TRACE] Log open, starting autoplay")

local frame_count = 0
local instr_count = 0

function on_input_polled()
    frame_count = frame_count + 1
    if frame_count <= #tape and frame_count <= MAX_FRAMES then
        local mask = tape[frame_count]
        emu.setInput(0, mask_to_mesen_input(mask))
    else
        -- End of tape: stop logging and pause emulator
        if log_file then
            log_file:close()
            log_file = nil
            emu.displayMessage("Trace",
                "Done: " .. instr_count .. " instr, " .. frame_count .. " frames")
            emu.stop()
        end
    end
end

function on_exec(address, value, addressType)
    if log_file == nil or instr_count >= MAX_INSTR then
        return
    end
    -- Only log within the target frame window.
    if frame_count < TRACE_START_FRAME or frame_count > TRACE_END_FRAME then
        return
    end
    local state = emu.getState()
    local cpu = state.cpu
    local ppu = state.ppu
    local op = emu.read(cpu.pc, emu.memType.nesDebug)
    log_file:write(string.format(
        "%04X %02X A:%02X X:%02X Y:%02X P:%02X SP:%02X PPU:%3d,%3d CYC:%d\n",
        cpu.pc, op,
        cpu.a, cpu.x, cpu.y, cpu.ps, cpu.sp,
        ppu.scanline, ppu.cycle, cpu.cycleCount
    ))
    instr_count = instr_count + 1
end

emu.addEventCallback(on_input_polled, emu.eventType.inputPolled)
emu.addMemoryCallback(on_exec, emu.callbackType.exec, 0, 0xFFFF)
emu.displayMessage("Trace",
    "Autoplay + trace started -> " .. LOG_PATH)
