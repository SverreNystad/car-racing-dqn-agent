import numpy as np

from src.agent import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: int) -> None:
        self._action_space = action_space

    def choose_action(self, observation) -> int:
        return np.random.randint(0, self._action_space)

    def update(self) -> dict:
        return {}

    def store(
        self,
        observation,
        action,
        reward,
        terminated,
        next_observation,
    ) -> None:
        pass

    def save_policy(self, policy_name: str) -> None:
        pass

    def load_policy(self, policy_name: str) -> None:
        pass
