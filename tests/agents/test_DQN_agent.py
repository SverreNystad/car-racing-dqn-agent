import pytest
from loguru import logger

from src.agents.DQN_agent import DQNAgent
from src.environment import create_env


@pytest.mark.parametrize("env_id", ["CartPole-v1", "CarRacing-v3"])
def test_choose_action_dqn_agent(env_id):
    env = create_env(env_id)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    logger.info(f"State shape: {state_shape}, Action size: {action_size}")
    agent = DQNAgent(state_shape=state_shape, action_size=action_size)

    obs, info = env.reset()
    action = agent.choose_action(obs)
    assert env.action_space.contains(action), "Action is not valid"
    env.step(action)
    env.close()
