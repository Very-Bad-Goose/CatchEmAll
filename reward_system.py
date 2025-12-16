def calculate_reward(self, info):
    reward = 0

    # Winning a battle is a major reward
    if info.get("battle_won"):
        reward += 50

    # Deal damage (shaped reward)
    opp_hp, opp_max = self.get_opponent_health()
    if opp_max > 0:
        damage_fraction = 1 - (opp_hp / opp_max)
        reward += damage_fraction * 2  # small shaping reward

    # Avoid punishment if HP is full (start of battle)
    player_hp, player_max = self.get_player_health()
    if player_hp < player_max:
        reward -= (player_max - player_hp) / player_max * 1.5

    # Capturing Pokémon
    if info.get("pokemon_captured"):
        reward += 30

    # Exploration reward
    if info.get("new_area_reached"):
        reward += 5

    # Idle penalty
    idle_time = info.get("idle_time", 0)
    reward -= idle_time * 0.01

    return reward
