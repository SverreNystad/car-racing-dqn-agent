import numpy as np

from tensordict import TensorDict
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from src.agent import Agent
from src.network import DQN


class DQNAgent(Agent):
    def __init__(
        self,
        state_shape: tuple,
        action_size: int,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        batch_size: int = 64,
        tau: float = 0.005,
    ):
        """
        Initializes the DQN agent with the given parameters.

        Args:
            state_shape (tuple): The shape of the state/observation space.
            action_size (int): The number of possible actions.
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon after each episode.
            epsilon_start (float): Starting value of epsilon.
            epsilon_end (float): Minimum value of epsilon.
            learning_rate (float): Learning rate for the optimizer.
            discount_factor (float): Discount factor "gamma" for future rewards.
            batch_size (int): Number of experiences to sample from the replay buffer for each update.
            tau (float): Soft update parameter for target network updates.
        """
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.tau = tau

        # if GPU is to be used
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = DQN(
            input_shape=state_shape,
            action_space_size=action_size,
        ).to(self.device)
        # Create the target network and initialize it with the same weights as the main network
        self.target_model = DQN(
            input_shape=state_shape,
            action_space_size=action_size,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                300000,
            ),
            batch_size=[batch_size],
        )

    def choose_action(self, observation) -> int:
        # Epsilon-greedy action selection
        if self.epsilon > np.random.rand():
            return np.random.randint(self.action_size)

        state = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            q_values = self.model(state)
            action = int(torch.argmax(q_values, dim=1).item())
            return action

    def store(
        self,
        observation,
        action: int,
        reward: float,
        terminated: bool,
        next_observation,
    ) -> None:
        self.replay_buffer.add(
            TensorDict(
                {
                    "observation": torch.tensor(observation),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_observation": torch.tensor(next_observation),
                    "terminated": torch.tensor(terminated),
                },
                batch_size=[],
            )
        )

    def update(
        self,
        batch_size: int,
    ) -> None:
        pass

    def _sample_experiences(self, amount: int) -> tuple:
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough experiences in the replay buffer to sample.")

        sampled_batch = self.replay_buffer.sample(amount)
        observations = sampled_batch["observation"].to(self.device)
        actions = sampled_batch["action"].squeeze().to(self.device)
        rewards = sampled_batch["reward"].squeeze().to(self.device)
        next_observations = sampled_batch["next_observation"].to(self.device)
        terminated = sampled_batch["terminated"].squeeze().to(self.device)

        return (
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
        )
