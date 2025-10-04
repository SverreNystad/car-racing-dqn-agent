import copy
import pytest
from loguru import logger
import torch

from src.train import train
from src.configuration import TrainingConfiguration
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


@pytest.mark.parametrize("env_id", ["CartPole-v1"])
def test_training_dqn_agent_changes_policy(env_id: str):
    config = TrainingConfiguration(num_training_episodes=2, warmup_steps=0)
    env = create_env(env_id)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, action_size=action_size, batch_size=4)

    # Deep copy of network weights before training
    initial_network = copy.deepcopy(agent.policy_network.state_dict())

    train(config, env, agent)
    # Check if the network weights have changed after training
    for key in initial_network:
        assert not torch.equal(
            initial_network[key], agent.policy_network.state_dict()[key]
        ), f"Parameters for {key} did not change after training."
