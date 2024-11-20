from stable_baselines3 import DQN
from environments.pokemon_env import PokemonEnv

def evaluate_model():
    env = PokemonEnv()
    model = DQN.load("models/pokemon_dqn")
    obs = env.reset()

    for _ in range(1000):  # Test for 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    # Train or evaluate
    from models.train_model import train_model
    train_model()
    evaluate_model()
