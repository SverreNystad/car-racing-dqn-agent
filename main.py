import random
import torch
from tqdm import trange
from loguru import logger
import os
from dotenv import load_dotenv
import wandb

from src.agents.DQN_agent import DQNAgent
from src.environment import create_env

load_dotenv()
SEED = 42

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found in environment variables.")

wandb.login(key=wandb_api_key)


if __name__ == "__main__":
    num_training_episodes = 30_001  # Total training episodes
    environment = "CarRacing-v3"  # "LunarLander-v3"  # "CarRacing-v3"  # CartPole-v1 # LunarLander-v3
    wandb.init(
        project=f"DQN-{environment}",
        config={
            "epsilon_decay_steps": num_training_episodes,
            "environment": environment,
        },
        mode="online",
    )
    env = create_env(environment, False, training_record_frequency=10)
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    warmup_steps = 1_000
    total_steps = 0
    run_name = f"DQN-{environment}-seed{SEED}"

    agent = DQNAgent(
        state_shape=observation_shape,
        action_size=action_size,
        learning_rate=0.001,
        epsilon_decay_steps=num_training_episodes,
        discount_factor=0.99,
        batch_size=128,
        tau=0.005,
    )

    wandb.config.update(
        {
            "state_shape": observation_shape,
            "action_size": action_size,
            "learning_rate": agent.learning_rate,
            "discount_factor": agent.discount_factor,
            "batch_size": agent.batch_size,
            "tau": agent.tau,
            "epsilon_start": agent.epsilon_start,
            "epsilon_end": agent.epsilon_end,
            "epsilon_decay_steps": agent.epsilon_decay_steps,
            "device": str(agent.device),
        },
        allow_val_change=True,
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
                metrics = agent.update()
                wandb.log(metrics, step=total_steps)
        # Log episode statistics (available in info after episode ends)
        if "episode" in info:
            episode_data = info["episode"]
            ep_logs = {
                "episode/reward": float(episode_data.get("r", 0.0)),
                "episode/length": int(episode_data.get("l", 0)),
                "episode/time_s": float(episode_data.get("t", 0.0)),
                "episode/idx": episode_num,
                "step/total_steps": total_steps,
            }
            wandb.log(ep_logs, step=total_steps)

            logger.info(
                f"Episode {episode_num}: "
                f"reward={episode_data['r']:.1f}, "
                f"length={episode_data['l']}, "
                f"time={episode_data['t']:.2f}s"
            )

            # Additional analysis for milestone episodes
            if episode_num % 100 == 0:
                # Look at recent performance (last 100 episodes)
                recent_rewards = list(env.return_queue)[-100:]
                if recent_rewards:
                    avg_recent = sum(recent_rewards) / len(recent_rewards)
                    logger.info(
                        f"  -> Average reward over last 100 episodes: {avg_recent:.1f}"
                    )
                    wandb.log(
                        {"episode/avg_reward_last_100": avg_recent}, step=total_steps
                    )

    env.close()
