import time

print("Waiting for mGBA...")

import sys

# mGBA exposes comm through stdin/stdout
# Run Python FROM mGBA using --script OR pipe
# So we poll stdin

while True:
    line = sys.stdin.readline().strip()
    if not line:
        time.sleep(0.01)
        continue

    parts = line.split(",")
    state = {
        "in_battle": int(parts[0]),
        "player_hp": int(parts[1]),
        "player_hp_max": int(parts[2]),
        "opp_hp": int(parts[3]),
        "opp_hp_max": int(parts[4]),
        "map_group": int(parts[5]),
        "map_num": int(parts[6]),
    }

    print(state)

    # Dumb policy
    if state["in_battle"]:
        print("A", flush=True)
    else:
        print("RIGHT", flush=True)
