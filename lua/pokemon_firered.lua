-- mGBA Lua Script for Pokemon FireRed (Simplified)
-- Uses file-based communication with Python

local IPC_DIR = "ipc/"
local STATE_FILE = IPC_DIR .. "mgba_state.txt"
local ACTION_FILE = IPC_DIR .. "mgba_action.txt"


-- FireRed RAM addresses
local PLAYER_HP_CUR = 0x02023D70
local PLAYER_HP_MAX = 0x02023D72
local OPP_HP_CUR    = 0x02023F18
local OPP_HP_MAX    = 0x02023F1A
local BATTLE_FLAG   = 0x02023BC4
local MAP_GROUP     = 0x02036DFC
local MAP_NUM       = 0x02036DFE
local BADGES        = 0x020244E8

-- Read 16-bit value (little endian)
local function u16(addr)
    local low = emu:read8(addr)
    local high = emu:read8(addr + 1)
    return low + high * 256
end

-- Communication files
-- local STATE_FILE = "mgba_state.txt"
-- local ACTION_FILE = "mgba_action.txt"

-- Write state to file for Python to read
local function write_state()
    local in_battle = emu:read8(BATTLE_FLAG)
    local player_hp = u16(PLAYER_HP_CUR)
    local player_hp_max = u16(PLAYER_HP_MAX)
    local opp_hp = u16(OPP_HP_CUR)
    local opp_hp_max = u16(OPP_HP_MAX)
    local map_group = emu:read8(MAP_GROUP)
    local map_num = emu:read8(MAP_NUM)
    local badges = emu:read8(BADGES)
    
    -- Format as JSON
    local json = string.format(
        '{"in_battle":%d,"player_hp":%d,"player_hp_max":%d,"opp_hp":%d,"opp_hp_max":%d,"map_group":%d,"map_num":%d,"badges":%d}',
        in_battle,
        player_hp,
        player_hp_max,
        opp_hp,
        opp_hp_max,
        map_group,
        map_num,
        badges
    )
    
    local file = io.open(STATE_FILE, "w")
    if file then
        file:write(json)
        file:close()
    end
end

-- Read action from file (written by Python)
local function read_action()
    local file = io.open(ACTION_FILE, "r")
    if not file then
        return nil
    end
    
    local action = file:read("*l")
    file:close()
    
    -- Delete the action file
    os.remove(ACTION_FILE)
    
    return action
end

-- Execute action - SIMPLIFIED (no clearing)
local function execute_action(action)
    if not action or action == "NONE" then
        return
    end
    
    -- Just press the button (mGBA will auto-release after frame)
    if action == "A" then
        emu:addKey("a")
    elseif action == "B" then
        emu:addKey("b")
    elseif action == "UP" then
        emu:addKey("up")
    elseif action == "DOWN" then
        emu:addKey("down")
    elseif action == "LEFT" then
        emu:addKey("left")
    elseif action == "RIGHT" then
        emu:addKey("right")
    elseif action == "START" then
        emu:addKey("start")
    elseif action == "SELECT" then
        emu:addKey("select")
    elseif action == "L" then
        emu:addKey("l")
    elseif action == "R" then
        emu:addKey("r")
    end
end

-- Frame counter for debugging
local frame_count = 0

-- Frame callback
callbacks:add("frame", function()
    frame_count = frame_count + 1
    
    -- Write state every frame
    write_state()
    
    -- Read and execute action
    local action = read_action()
    if action then
        execute_action(action)
    end
    
    -- Log every 60 frames (1 second)
    if frame_count % 60 == 0 then
        console:log("Frame " .. frame_count .. " - Running normally")
    end
end)

console:log("===========================================")
console:log("mGBA Pokemon FireRed Script Loaded!")
console:log("===========================================")
console:log("State file: " .. STATE_FILE)
console:log("Action file: " .. ACTION_FILE)
console:log("Ready for Python!")
console:log("===========================================")