import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import Box, Discrete
from collections import deque

class PokemonEnv(Env):
    def __init__(self):
        super().__init__()
        self.env = retro.make(game="PokemonEmerald-GBA")  # Ensure the ROM is available
        # Observation space (stacked frames)
        self.observation_space = Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)
        # Action space
        self.action_space = Discrete(self.env.action_space.n)
        # Frame stack
        self.frame_stack = deque(maxlen=4)

    def preprocess_frame(self, frame):
        """
        Preprocesses the game frame by converting it to grayscale, resizing, and normalizing.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

    def stack_frames(self, frame):
        """
        Stacks the latest frame with previous frames to create a multi-frame observation.
        """
        if len(self.frame_stack) == 0:
            # Initialize stack with the same frame
            self.frame_stack.extend([frame] * 4)
        else:
            # Add the latest frame
            self.frame_stack.append(frame)
        # Return stacked frames as numpy array
        return np.array(self.frame_stack)

    def step(self, action):
        """
        Takes an action in the environment and calculates the custom reward based on game events.
        """
        obs, _, done, info = self.env.step(action)

        # Preprocess and stack frames
        preprocessed_frame = self.preprocess_frame(obs)
        stacked_obs = self.stack_frames(preprocessed_frame)

        # Add custom keys to `info` (replace these with actual game logic)
        info["opponent_defeated"] = self.check_opponent_defeat(info)
        info["item_collected"] = self.check_item_collection(info)
        info["new_area_reached"] = self.check_new_area(info)
        info["battle_won"] = self.check_battle_win(info)
        info["pokemon_captured"] = self.check_pokemon_capture(info)
        info["idle_time"] = self.calculate_idle_time(info)
        info["player_damage"] = self.calculate_player_damage(info)
        info["player_health"] = self.get_player_health(info)

        # Calculate custom reward
        reward = self.calculate_reward(info)

        return stacked_obs, reward, done, info

    def reset(self):
        """
        Resets the environment and initializes the frame stack.
        """
        obs = self.env.reset()
        preprocessed_frame = self.preprocess_frame(obs)
        stacked_obs = self.stack_frames(preprocessed_frame)
        return stacked_obs

    def render(self, mode="human"):
        """
        Renders the environment.
        """
        self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()

    # Reward system and helper methods as before...
