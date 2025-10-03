import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# Training configuration
training_period = 250  # Record video every 250 episodes
num_training_episodes = 10_001  # Total training episodes
env_name = "CartPole-v1"  #  CarRacing-v3

# Set up logging for episode statistics
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Create environment with periodic video recording
env = gym.make(env_name, render_mode="rgb_array")

# Record videos periodically (every 250 episodes)
env = RecordVideo(
    env,
    video_folder=f"{env_name}-training",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0,
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)

print(f"Starting training for {num_training_episodes} episodes")
print(f"Videos will be recorded every {training_period} episodes")
print(f"Videos saved to: {env_name}-training/")

for episode_num in range(num_training_episodes):
    obs, info = env.reset()
    episode_over = False

    while not episode_over:
        # Replace with your actual training agent
        action = env.action_space.sample()  # Random policy for demonstration
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    # Log episode statistics (available in info after episode ends)
    if "episode" in info:
        episode_data = info["episode"]
        logging.info(
            f"Episode {episode_num}: "
            f"reward={episode_data['r']:.1f}, "
            f"length={episode_data['l']}, "
            f"time={episode_data['t']:.2f}s"
        )

        # Additional analysis for milestone episodes
        if episode_num % 1000 == 0:
            # Look at recent performance (last 100 episodes)
            recent_rewards = list(env.return_queue)[-100:]
            if recent_rewards:
                avg_recent = sum(recent_rewards) / len(recent_rewards)
                print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")

env.close()
