import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import Discrete, Box

class PokemonEnv(Env):
    def __init__(self):
        super().__init__()
        self.env = retro.make(game="PokemonRed-GameBoy")
        self.observation_space = Box(
            low=0, high=255, shape=(224, 240, 3), dtype=np.uint8
        )  # Resized frame
        self.action_space = Discrete(self.env.action_space.n)

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (240, 224))
        return frame

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess_frame(obs)
        reward = self.calculate_reward(reward, info)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.preprocess_frame(obs)

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()

    def calculate_reward(self, base_reward, info):
        # Define custom reward based on info
        return base_reward
