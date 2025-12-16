import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import Box, Discrete
from collections import deque

class PokemonFireRedEnv(Env):
    def __init__(self):
        super().__init__()
        # Use FireRed instead of Emerald
        self.env = retro.make(game="PokemonFireRed-GBA")
        
        # Observation space (stacked frames)
        self.observation_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
        
        # Action space
        self.action_space = Discrete(self.env.action_space.n)
        
        # Frame stack
        self.frame_stack = deque(maxlen=4)
        
        # Tracking variables
        self.last_area = None
        self.idle_counter = 0
        self.last_opponent_hp = None

    def preprocess_frame(self, frame):
        """Preprocesses the game frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

    def stack_frames(self, frame):
        """Stacks the latest frame with previous frames."""
        if len(self.frame_stack) == 0:
            self.frame_stack.extend([frame] * 4)
        else:
            self.frame_stack.append(frame)
        return np.array(self.frame_stack)

    def read_u16(self, ram, addr):
        """Read 16-bit value from RAM (little endian)."""
        return ram[addr] | (ram[addr + 1] << 8)
    
    def read_u32(self, ram, addr):
        """Read 32-bit value from RAM (little endian)."""
        return (ram[addr] | (ram[addr + 1] << 8) | 
                (ram[addr + 2] << 16) | (ram[addr + 3] << 24))
    
    # FireRed-specific memory addresses
    def in_battle(self):
        """Check if currently in battle."""
        ram = self.env.get_ram()
        # FireRed battle flag address
        return ram[0x02022B4C] != 0
    
    def get_player_health(self):
        """Get player's current Pokemon HP."""
        ram = self.env.get_ram()
        # FireRed player HP addresses
        cur_hp = self.read_u16(ram, 0x02024284)
        max_hp = self.read_u16(ram, 0x02024286)
        return cur_hp, max_hp
    
    def get_opponent_health(self):
        """Get opponent's current Pokemon HP."""
        ram = self.env.get_ram()
        # FireRed opponent HP addresses
        cur_hp = self.read_u16(ram, 0x02024744)
        max_hp = self.read_u16(ram, 0x02024746)
        return cur_hp, max_hp

    def get_badges(self):
        """Get number of badges earned."""
        ram = self.env.get_ram()
        # FireRed badges byte
        badges_byte = ram[0x020244E8]
        return bin(badges_byte).count('1')
    
    def get_party_size(self):
        """Get number of Pokemon in party."""
        ram = self.env.get_ram()
        return ram[0x02024284]
    
    def get_money(self):
        """Get player money."""
        ram = self.env.get_ram()
        return self.read_u32(ram, 0x0202494C)

    def check_battle_won(self):
        """Check if battle was just won."""
        if not self.in_battle():
            # If we just left battle and opponent HP was 0
            if self.last_opponent_hp is not None and self.last_opponent_hp == 0:
                return True
        return False

    def check_new_area(self):
        """Check if player entered a new area."""
        ram = self.env.get_ram()
        # FireRed map bank and map number
        current_area = (ram[0x02036DFC], ram[0x02036DFD])
        
        if self.last_area is None:
            self.last_area = current_area
            return False
        
        changed = current_area != self.last_area
        self.last_area = current_area
        return changed

    def calculate_idle_time(self, new_area, opponent_defeated):
        """Track idle time (no progress)."""
        if new_area or opponent_defeated:
            self.idle_counter = 0
        else:
            self.idle_counter += 1
        return self.idle_counter

    def calculate_reward(self, info):
        """Calculate reward based on game state."""
        reward = 0

        # Major achievements
        if info.get("battle_won"):
            reward += 100
        
        if info.get("badge_earned"):
            reward += 500
        
        if info.get("new_area_reached"):
            reward += 10

        # Battle rewards (damage dealing)
        if self.in_battle():
            opp_hp, opp_max = self.get_opponent_health()
            if opp_max > 0:
                damage_fraction = 1 - (opp_hp / opp_max)
                reward += damage_fraction * 5
        
        # Health penalty (losing health)
        player_hp, player_max = self.get_player_health()
        if player_max > 0 and player_hp < player_max:
            health_loss = (player_max - player_hp) / player_max
            reward -= health_loss * 3

        # Idle penalty (to encourage exploration)
        idle_time = info.get("idle_time", 0)
        if idle_time > 100:
            reward -= 0.1

        return reward

    def step(self, action):
        """Execute action and return observation."""
        obs, _, done, info = self.env.step(action)
        
        # Preprocess frame
        preprocessed_frame = self.preprocess_frame(obs)
        stacked_obs = self.stack_frames(preprocessed_frame)
        
        # Track opponent HP for battle win detection
        if self.in_battle():
            self.last_opponent_hp, _ = self.get_opponent_health()
        
        # Gather game state info
        info["in_battle"] = self.in_battle()
        info["battle_won"] = self.check_battle_won()
        info["new_area_reached"] = self.check_new_area()
        info["badges"] = self.get_badges()
        info["party_size"] = self.get_party_size()
        
        # Check for badge earned
        if not hasattr(self, "last_badges"):
            self.last_badges = info["badges"]
        info["badge_earned"] = info["badges"] > self.last_badges
        self.last_badges = info["badges"]
        
        # Calculate idle time
        info["idle_time"] = self.calculate_idle_time(
            info["new_area_reached"], 
            info.get("battle_won", False)
        )
        
        # Custom reward
        reward = self.calculate_reward(info)
        
        return stacked_obs, reward, done, info

    def reset(self):
        """Reset the environment."""
        obs = self.env.reset()
        preprocessed_frame = self.preprocess_frame(obs)
        
        # Clear frame stack
        self.frame_stack.clear()
        stacked_obs = self.stack_frames(preprocessed_frame)
        
        # Reset tracking variables
        self.last_area = None
        self.idle_counter = 0
        self.last_opponent_hp = None
        self.last_badges = 0
        
        return stacked_obs

    def render(self, mode="human"):
        """Render the environment."""
        self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()