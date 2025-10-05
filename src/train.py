import os
from loguru import logger
import wandb
from tqdm import trange
from gymnasium import Env
import glob

from src.agent import Agent
from src.configuration import TrainingConfiguration


def train(config: TrainingConfiguration, env: Env, agent: Agent):
    total_steps = 0
    best_reward = float("-inf")
    last_logged_video: str | None = None

    for episode_num in trange(config.num_training_episodes):
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

            if total_steps >= config.warmup_steps:
                metrics = agent.update()
                wandb.log(metrics, step=total_steps)

        # Log episode statistics (available in info after episode ends)
        if "episode" in info:
            episode_data = info["episode"]
            current_reward = float(episode_data.get("r", 0.0))
            ep_logs = {
                "episode/reward": current_reward,
                "episode/length": int(episode_data.get("l", 0)),
                "episode/time_s": float(episode_data.get("t", 0.0)),
                "episode/idx": episode_num,
                "step/total_steps": total_steps,
            }
            wandb.log(ep_logs, step=total_steps)

            logger.info(
                f"Episode {episode_num}: "
                f"reward={current_reward:.1f}, "
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
            # Upload video
            video_dir = getattr(env, "video_folder_name", None)
            if video_dir:
                newest = _latest_video(video_dir)
                if newest and newest != last_logged_video:
                    wandb.log(
                        {
                            "videos/training": wandb.Video(
                                newest,
                                caption=f"Episode {episode_num}",
                                format="mp4",
                            )
                        },
                        step=total_steps,
                    )
                    last_logged_video = newest

            # Save the model if it's the best so far
            if config.shall_checkpoint_model and best_reward < current_reward:
                best_reward = max(best_reward, current_reward)

                agent.save_policy(
                    f"best_model_episode_{episode_num}_reward_{best_reward:.2f}"
                )

    env.close()


def _latest_video(video_dir: str) -> str | None:
    """Return the path of the newest .mp4 file in video_dir, or None."""
    if not os.path.isdir(video_dir):
        return None
    mp4s = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not mp4s:
        return None
    return max(mp4s, key=os.path.getmtime)
