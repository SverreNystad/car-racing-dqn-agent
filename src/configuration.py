import os
from typing import Literal
from pydantic import BaseModel
import yaml

CONFIG_PATH = "config"


class Configuration(BaseModel):
    env: "EnvConfiguration"
    training: "TrainingConfiguration"
    wandb: "WANDBConfiguration"


class TrainingConfiguration(BaseModel):
    num_training_episodes: int = 30_001
    warmup_steps: int = 1_000
    seed: int = 42
    shall_checkpoint_model: bool = True

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 40_000
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 64
    tau: float = 0.005

    replay_buffer_size: int = 100_000
    """size of the storage, i.e. maximum number of elements stored in the buffer."""


class EnvConfiguration(BaseModel):
    env_id: str
    continuous: bool = False
    video_folder_name: str | None = None
    action_repeat: int = 4
    training_record_frequency: int = 250
    frame_size: tuple[int, int] = (84, 84)


class WANDBConfiguration(BaseModel):
    project_name: str
    entity: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = (
        "online"  # "offline" or "disabled" for no logging
    )


def load_config(filename: str) -> "Configuration":
    """
    Load a configuration file from the config directory.
    """
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Configuration(**raw_config)
