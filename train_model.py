import os
import argparse
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from pokemon_env import PokemonFireRedEnv

def train_model(algorithm="DQN", total_timesteps=1000000):
    """Train the Pokemon AI model."""
    
    # Create directories
    os.makedirs("./models/checkpoints/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    # Create the environment
    env = PokemonFireRedEnv()
    env = Monitor(env, "./logs/")
    
    print(f"Training with {algorithm}...")
    
    if algorithm == "DQN":
        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.01,
            target_update_interval=1000,
            train_freq=4,
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )
    elif algorithm == "PPO":
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/",
        name_prefix=f"pokemon_{algorithm.lower()}"
    )
    
    # Start training
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    print("Training completed!")
    
    # Save final model
    model_path = f"./models/pokemon_{algorithm.lower()}_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

def evaluate_model(model_path, episodes=10):
    """Evaluate a trained model."""
    env = PokemonFireRedEnv()
    
    # Try to determine algorithm from filename
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)
    
    total_rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            env.render()
            
            # Print interesting info
            if info.get("battle_won"):
                print(f"Battle won at step {steps}!")
            if info.get("badge_earned"):
                print(f"Badge earned! Total: {info['badges']}")
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f}, Steps: {steps}")
    
    print(f"\nAverage reward: {sum(total_rewards)/len(total_rewards):.2f}")
    env.close()

def test_environment():
    """Test the environment setup."""
    print("Testing environment...")
    env = PokemonFireRedEnv()
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    print("\nTaking 20 random steps...")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: Reward={reward:.2f}, In Battle={info.get('in_battle', False)}, "
              f"Badges={info.get('badges', 0)}")
        
        env.render()
        
        if done:
            print("Episode ended, resetting...")
            obs = env.reset()
    
    env.close()
    print("Environment test completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon FireRed AI")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--test", action="store_true", help="Test environment")
    parser.add_argument("--algorithm", type=str, default="DQN", choices=["DQN", "PPO"],
                       help="Algorithm to use (DQN or PPO)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default="./models/pokemon_dqn_final",
                       help="Path to model for evaluation")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    if args.train:
        train_model(algorithm=args.algorithm, total_timesteps=args.timesteps)
    elif args.evaluate:
        evaluate_model(model_path=args.model_path, episodes=args.episodes)
    elif args.test:
        test_environment()
    else:
        print("Please specify: --train, --evaluate, or --test")