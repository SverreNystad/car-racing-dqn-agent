import random
import torch
from tqdm import trange
from loguru import logger
from src.agents.random_agent import RandomAgent
from src.agents.DQN_agent import DQNAgent
from src.environment import create_env

SEED = 42

if __name__ == "__main__":
    num_training_episodes = 30_001  # Total training episodes
    environment = "CartPole-v1"  # "LunarLander-v3"  # "CarRacing-v3"  # CartPole-v1 # LunarLander-v3
    env = create_env(environment, False)
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    warmup_steps = 5_000
    total_steps = 0

    agent = DQNAgent(
        state_shape=observation_shape,
        action_size=action_size,
        learning_rate=0.001,
        epsilon_decay_steps=num_training_episodes,
        discount_factor=0.99,
        batch_size=128,
        tau=0.005,
    )

    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    for episode_num in trange(num_training_episodes):
        obs, info = env.reset()
        episode_over = False

        while not episode_over:
            action = agent.choose_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_over = bool(terminated or truncated)
            agent.store(obs, action, float(reward), episode_over, next_obs)

            obs = next_obs
            total_steps += 1
            episode_over = terminated or truncated

            if total_steps >= warmup_steps:
                agent.update()
        # Log episode statistics (available in info after episode ends)
        if "episode" in info:
            episode_data = info["episode"]
            logger.info(
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
                    logger.info(
                        f"  -> Average reward over last 100 episodes: {avg_recent:.1f}"
                    )

    env.close()
