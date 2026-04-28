-- Mesen 2 Lua trace script: SMB cold-boot CPU instruction trace.
--
-- Captures the first MAX_FRAMES of cold-boot CPU instructions from Super
-- Mario Bros (USA), no input pressed. This is the "ground truth" oracle
-- for diffing nes_core's CPU/PPU/cycle behavior on the same scenario.
--
-- How to run (headless via Mesen --testRunner):
--   /Applications/Mesen.app/Contents/MacOS/Mesen \
--     --testRunner --noaudio --novideo --noinput \
--     scripts/tracing/mesen_smb_coldboot.lua \
--     "roms/Super Mario Bros. (World).nes" \
--     --timeout=120
--
-- Output format (per line, matching nes_core_smb_coldboot.py):
--   PC OP A:AA X:XX Y:YY P:PP SP:SP PPU:sl,cyc CYC:total
--
-- emu.getState() returns a flat map with dotted keys (`cpu.pc`,
-- `ppu.scanline`, etc.), not the nested table the older docs imply.

local LOG_PATH = "/tmp/mesen_smb_coldboot.txt"
local STATUS_PATH = "/tmp/mesen_smb_coldboot.status.txt"
local MAX_FRAMES = 200

local log_file = io.open(LOG_PATH, "w")
local status_file = io.open(STATUS_PATH, "w")
status_file:write("script-loaded\n"); status_file:flush()

local frame_count = 0
local instr_count = 0
local done = false

function on_input_polled()
    emu.setInput(0, {a=false,b=false,select=false,start=false,
                     up=false,down=false,left=false,right=false})
end

function on_frame_start()
    frame_count = frame_count + 1
    if frame_count > MAX_FRAMES and not done then
        done = true
        if log_file then log_file:close(); log_file = nil end
        status_file:write("done frames=" .. (frame_count - 1) ..
                          " instr=" .. instr_count .. "\n")
        status_file:close()
        emu.stop(0)
    end
end

function on_exec(address, value, addressType)
    if log_file == nil then return end
    local s = emu.getState()
    local pc = s["cpu.pc"]
    local op = emu.read(pc, emu.memType.nesDebug)
    log_file:write(string.format(
        "%04X %02X A:%02X X:%02X Y:%02X P:%02X SP:%02X PPU:%3d,%3d CYC:%d\n",
        pc, op,
        s["cpu.a"], s["cpu.x"], s["cpu.y"], s["cpu.ps"], s["cpu.sp"],
        s["ppu.scanline"], s["ppu.cycle"], s["cpu.cycleCount"]
    ))
    instr_count = instr_count + 1
end

emu.addEventCallback(on_input_polled, emu.eventType.inputPolled)
emu.addEventCallback(on_frame_start, emu.eventType.startFrame)
emu.addMemoryCallback(on_exec, emu.callbackType.exec, 0, 0xFFFF)
status_file:write("callbacks-registered\n"); status_file:flush()
