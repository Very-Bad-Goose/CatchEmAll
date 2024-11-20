  GNU nano 7.2                 Utils/reward_system.py

def calculate_reward(info):
    # Example: Reward for defeating opponents
    if info.get("opponent_defeated", False):
        return 10
    return -0.1  # Small penalty for idle time
