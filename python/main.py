# Train: python main.py --train
# Evaluate: python main.py --evaluate




from stable_baselines3 import DQN
from Environments.pokemon_env import PokemonEnv
from Models.train_model import train_model

def evaluate_model():
    """Evaluate a trained model."""
    env = PokemonEnv()
    model = DQN.load("models/pokemon_dqn")  # Load the trained model
    obs = env.reset()

    for _ in range(1000):  # Test for 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

def test_environment():
    """Test the environment to ensure preprocessing and stacking work correctly."""
    env = PokemonEnv()
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")  # Should be (4, 84, 84)

    for _ in range(10):  # Test for 10 steps
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step observation shape: {obs.shape}, Reward: {reward}")
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Pokémon AI")
    parser.add_argument("--train", action="store_true", help="Train the AI model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--test", action="store_true", help="Test the environment")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.test:
        test_environment()
    else:
        print("Please specify an action: --train, --evaluate, or --test")
