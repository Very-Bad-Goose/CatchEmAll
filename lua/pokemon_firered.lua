local socket = require("socket")
local client = socket.tcp()
client:connect("127.0.0.1", 5005)
client:settimeout(0)

-- FireRed RAM addresses
PLAYER_HP_CUR = 0x02023D70
PLAYER_HP_MAX = 0x02023D72
OPP_HP_CUR    = 0x02023F18
OPP_HP_MAX    = 0x02023F1A
BATTLE_FLAG  = 0x02023BC4
MAP_GROUP    = 0x02036DFC
MAP_NUM      = 0x02036DFE

local function u16(addr)
    return memory.readbyte(addr) + memory.readbyte(addr+1)*256
end

while true do
    local state = {
        in_battle = memory.readbyte(BATTLE_FLAG),
        player_hp = u16(PLAYER_HP_CUR),
        player_hp_max = u16(PLAYER_HP_MAX),
        opp_hp = u16(OPP_HP_CUR),
        opp_hp_max = u16(OPP_HP_MAX),
        map_group = memory.readbyte(MAP_GROUP),
        map_num = memory.readbyte(MAP_NUM)
    }

    client:send(tostring(state) .. "\n")

    local action = client:receive()
    if action == "A" then joypad.set({A=true}) end
    if action == "B" then joypad.set({B=true}) end
    if action == "UP" then joypad.set({Up=true}) end
    if action == "DOWN" then joypad.set({Down=true}) end
    if action == "LEFT" then joypad.set({Left=true}) end
    if action == "RIGHT" then joypad.set({Right=true}) end
    if action == "START" then joypad.set({Start=true}) end

    emu.frameadvance()
end
