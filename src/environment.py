from loguru import logger
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
)


def create_env(
    env_id: str,
    continuous: bool = False,
    video_folder_name: str | None = None,
    action_repeat: int = 4,
    training_record_frequency: int = 250,
    frame_size: tuple[int, int] = (84, 84),
) -> gym.Env:
    # Some environments (for example CartPole) do not accept a `continuous`
    try:
        env = gym.make(env_id, render_mode="rgb_array", continuous=continuous)
    except TypeError as e:
        logger.warning(f"Creating env {env_id} without 'continuous' argument: {e}")
        env = gym.make(env_id, render_mode="rgb_array")

    # Only apply ResizeObservation for image-like observation spaces.
    # Some environments (e.g. CartPole) provide a low-dimensional observation (vector) rather than pixel observations.
    obs_shape = env.observation_space.shape
    if obs_shape and len(obs_shape) in {2, 3}:
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, frame_size)

    env = ActionFrameRepeater(env, action_repeat=action_repeat)
    if not video_folder_name:
        video_folder_name = f"{env_id}-training"
    env = RecordVideo(
        env,
        video_folder=video_folder_name,
        name_prefix="training",
        episode_trigger=lambda x: x % training_record_frequency == 0,
    )
    env = RecordEpisodeStatistics(env)
    return env


class ActionFrameRepeater(gym.Wrapper):
    """
    A wrapper for skipping frames in the environment to speed up training.

    Args:
        env (gymnasium.Env) : The environment to apply the wrapper to.
        skip (int) : The number of frames to skip.
    """

    def __init__(self, env: gym.Env, action_repeat: int = 4):
        super().__init__(env)
        if action_repeat < 1:
            raise ValueError("Action repeat must be at least 1")
        self._steps = action_repeat

    def step(self, action: int) -> tuple:
        """
        Executes the action for the specified number of frames, accumulating rewards.
        """
        total_reward = 0.0
        for _ in range(self._steps):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated:
                break
        return state, total_reward, terminated, truncated, info
