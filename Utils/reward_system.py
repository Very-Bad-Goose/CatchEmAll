def calculate_reward(info):
    """
    Calculate reward based on game events and player progress.
    
    Args:
        info (dict): A dictionary containing game information.
    
    Returns:
        float: The calculated reward for the current step.
    """
    reward = 0  # Initialize reward

    # Reward for defeating opponents
    if info.get("opponent_defeated", False):
        reward += 10  # Encourage AI to defeat opponents

    # Reward for collecting items
    if info.get("item_collected", False):
        reward += 5  # Reward for collecting items like potions or Poké Balls

    # Reward for advancing to new areas
    if info.get("new_area_reached", False):
        reward += 15  # Reward for exploring and progressing through the game

    # Reward for winning battles
    if info.get("battle_won", False):
        reward += 20  # Larger reward for successfully completing a battle

    # Reward for capturing Pokémon
    if info.get("pokemon_captured", False):
        reward += 25  # High reward for capturing a Pokémon

    # Penalty for idle time or unnecessary steps
    if info.get("idle_time", 0) > 5:  # Example: idle_time > 5 seconds
        reward -= 1  # Penalize idling

    # Penalty for taking damage
    if info.get("player_damage", 0) > 0:
        reward -= info["player_damage"] * 0.1  # Penalize based on the amount of damage taken

    # Reward for maintaining high health
    if info.get("player_health", 100) > 80:  # Example: health above 80%
        reward += 2  # Small positive reinforcement for staying healthy

    # Ensure a small penalty for every step to encourage efficiency
    reward -= 0.1  # Small penalty for each step taken

    return reward
