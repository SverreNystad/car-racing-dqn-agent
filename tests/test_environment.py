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


def test_invalid_env_configuration():
    with pytest.raises(ValueError):
        create_env("CarRacing-v3", frame_size=(-84, 84))


def test_invalid_action_repeat():
    with pytest.raises(ValueError):
        create_env(
            "CarRacing-v3",
            action_repeat=0,
        )


def test_negative_action_repeat():
    with pytest.raises(ValueError):
        create_env(
            "CarRacing-v3",
            action_repeat=-1,
        )
