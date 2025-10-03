from src.environment import create_env
import gymnasium as gym
import pytest


@pytest.mark.parametrize(
    "env_id,continuous",
    [
        ("CartPole-v1", False),
        ("CarRacing-v3", True),
    ],
)
def test_create_env(env_id, continuous):
    env = create_env(env_id, continuous=continuous)
    assert isinstance(env, gym.Env)
    env.close()
