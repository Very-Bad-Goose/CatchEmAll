from stable_baselines3 import DQN
from environments.pokemon_env import PokemonEnv

def train_model():
    env = PokemonEnv()
    model = DQN("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("models/pokemon_dqn")
    env.close()

if __name__ == "__main__":
    train_model()
