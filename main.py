import random
import torch
import os
from dotenv import load_dotenv
import wandb

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


def train_dqn_agent():
    config = load_config("dqn-agent.yaml")
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


if __name__ == "__main__":
    train_dqn_agent()
