from typing import Protocol, Union

from numpy import ndarray
from torch import Tensor


class Agent(Protocol):
    def choose_action(self, observation: Union[ndarray, Tensor]) -> int:
        """
        Get the action to take based on the current observation.
        """
        ...

    def store(
        self,
        observation: Union[ndarray, Tensor],
        action: int,
        reward: float,
        terminated: bool,
        next_observation: Union[ndarray, Tensor],
    ) -> None:
        """
        Store the experience in the agent's replay buffer.
        """
        ...

    def update(
        self,
    ) -> dict:
        """
        Update the agent's knowledge based on the experiences stored in the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample from the replay buffer for each update
        Returns:
            dict: A dictionary containing relevant metrics from the update process (e.g., loss values).
        """
        ...

    def save_policy(self, policy_name: str) -> None:
        """
        Save the current policy.

        Args:
            policy_name (str): The name to use when saving the policy.
        """
        ...

    def load_policy(self, policy_name: str) -> None:
        """
        Load a saved policy into the agent.

        Args:
            policy_name (str): The name of the policy to load.
        """
        ...
