import random
import gymnasium
import numpy as np
import torch
import os
from dotenv import load_dotenv
import wandb
from gymnasium.utils.play import play
import argparse

from src.agents.DQN_agent import DQNAgent
from src.configuration import load_config
from src.environment import create_env
from src.train import train

load_dotenv()
SEED = 42

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY not found in environment variables.")

wandb.login(key=wandb_api_key)


def train_dqn_agent(config_path: str = "dqn-agent.yaml"):
    config = load_config(config_path)
    wandb.init(
        project=config.wandb.project_name,
        config=config.model_dump(),
        mode=config.wandb.mode,
    )
    env = create_env(**config.env.model_dump())
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(
        state_shape=observation_shape,
        action_size=action_size,
        learning_rate=config.training.learning_rate,
        epsilon_decay_steps=config.training.epsilon_decay_steps,
        discount_factor=config.training.discount_factor,
        batch_size=config.training.batch_size,
        tau=config.training.tau,
    )

    wandb.config.update(
        {
            "state_shape": observation_shape,
            "action_size": action_size,
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

    train(config.training, env, agent)


def run_a_agent(agent_name: str, config_path: str = "dqn-agent.yaml"):
    config = load_config(config_path)
    wandb.init(
        project=config.wandb.project_name,
        config=config.model_dump(),
        mode=config.wandb.mode,
    )
    env = create_env(**config.env.model_dump())
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(
        state_shape=observation_shape,
        action_size=action_size,
    )
    agent.load_policy(agent_name)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.choose_action(obs)
        obs, reward, terminated, _, _ = env.step(action)
        obs = obs / 255
        done = terminated
        total_reward += float(reward)
        env.render()
    print(f"Total reward: {total_reward}")
    env.close()


def run_game_using_keyboard():
    env = gymnasium.make("CarRacing-v3", render_mode="rgb_array")
    """
    0: do nothing
    1: steer right
    2: steer left
    3: gas
    4: brake
    """
    env.reset()

    total_reward = 0.0

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        nonlocal total_reward
        total_reward += rew
        if terminated or truncated:
            print(f"Total reward: {total_reward:.2f}")
            total_reward = 0.0
        return [
            rew,
        ]

    noop_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    keys_to_action = {
        "d": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # steer right
        "a": np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # steer left
        "w": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # gas
        "s": np.array([0.0, 0.0, 0.8], dtype=np.float32),
        # Combined keys can be tuples mapping to combined actions
        ("a", "w"): np.array([-1.0, 1.0, 0.0], dtype=np.float32),
        ("d", "w"): np.array([1.0, 1.0, 0.0], dtype=np.float32),
    }

    play(
        env,
        keys_to_action=keys_to_action,
        noop=noop_action,
        callback=callback,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DQN for CarRacing-v3: train, run a saved policy, or play with keyboard."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train a DQN agent.")
    p_train.add_argument(
        "--config",
        default="dqn-agent.yaml",
        help="Path to config YAML (default: dqn-agent.yaml)",
    )

    # run (evaluate) saved agent
    p_run = subparsers.add_parser("run", help="Run an already trained agent.")
    p_run.add_argument(
        "--agent-name",
        required=True,
        help="Identifier/path for the saved policy (e.g. 'user/project/model:v60').",
    )
    p_run.add_argument(
        "--config",
        default="dqn-agent.yaml",
        help="Path to config YAML (default: dqn-agent.yaml)",
    )

    # play (keyboard)
    subparsers.add_parser("play", help="Play CarRacing-v3 with the keyboard.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train_dqn_agent(config_path=args.config)

    elif args.command == "run":
        run_a_agent(agent_name=args.agent_name, config_path=args.config)

    elif args.command == "play":
        run_game_using_keyboard()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
