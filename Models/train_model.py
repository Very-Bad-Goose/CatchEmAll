import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from environments.pokemon_env import PokemonEnv
from utils.logger import logger

def train_model():
    # Create the environment
    env = PokemonEnv()

    model = DQN(
        policy="CnnPolicy",           # The policy architecture; "CnnPolicy" processes image-based inputs using a Convolutional Neural Network (CNN).
        env=env,                      # The environment instance for the AI to interact with.
        learning_rate=1e-4,           # The step size for updating the neural network; smaller values slow down learning but improve stability.
        buffer_size=50000,            # Maximum size of the replay buffer; stores past experiences for training the model.
        learning_starts=1000,         # Number of steps before the training process begins, allowing the replay buffer to fill up with initial data.
        batch_size=32,                # Number of experiences sampled from the replay buffer for each training step.
        gamma=0.99,                   # Discount factor for future rewards; balances the importance of immediate vs. long-term rewards.
        exploration_fraction=0.1,     # Fraction of the total training steps during which the exploration rate (epsilon) decreases linearly.
        exploration_final_eps=0.02,   # Minimum exploration rate (epsilon); determines how much random exploration continues after the initial phase.
        target_update_interval=500,   # Number of steps between updates to the target network, which stabilizes training.
        train_freq=4,                 # Number of steps between each training update; can be adjusted to balance training frequency.
        verbose=1                     # Controls the level of logging; 1 enables training progress output, 0 suppresses it.
        )



    # Callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path="./models/checkpoints/",
        name_prefix="pokemon_dqn"
    )

    # Start training
    logger.info("Starting training...")
    model.learn(
        total_timesteps=500000,  # Train for 500k steps
        callback=checkpoint_callback
    )
    logger.info("Training completed.")

    # Save the final model
    model_path = "models/pokemon_dqn_final"
    model.save(model_path)
    logger.info(f"Model saved at {model_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Ensure checkpoint directory exists
    os.makedirs("./models/checkpoints/", exist_ok=True)

    # Train the model
    train_model()
