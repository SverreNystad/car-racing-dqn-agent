import numpy as np

from tensordict import TensorDict
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch.nn.utils import clip_grad_norm_

from src.agent import Agent
from src.network import DQN


class DQNAgent(Agent):
    def __init__(
        self,
        state_shape: tuple,
        action_size: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 40_000,
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
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.tau = tau

        # if GPU is to be used
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.policy_network = DQN(
            input_shape=state_shape,
            action_space_size=action_size,
        ).to(self.device)
        # Create the target network and initialize it with the same weights as the main network
        self.target_model = DQN(
            input_shape=state_shape,
            action_space_size=action_size,
        ).to(self.device)
        self.target_model.load_state_dict(self.policy_network.state_dict())

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                300000,
            ),
            batch_size=[batch_size],
        )

        self.optimizer = torch.optim.AdamW(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )
        self.criterion = torch.nn.SmoothL1Loss()

        self.global_step = 0

    def choose_action(self, observation) -> int:
        # Epsilon-greedy action selection
        if self.epsilon > np.random.rand():
            return np.random.randint(self.action_size)

        self.global_step += 1
        self.decay_epsilon()

        state = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_network(state)
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
                    "observation": torch.tensor(observation, dtype=torch.float32),
                    "action": torch.tensor(action, dtype=torch.long),
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "next_observation": torch.tensor(
                        next_observation, dtype=torch.float32
                    ),
                    "terminated": torch.tensor(terminated, dtype=torch.bool),
                },
                batch_size=[],
            )
        )

    def update(
        self,
    ) -> None:
        if len(self.replay_buffer) < self.batch_size:
            # Not enough experiences to sample a full batch
            return

        obs, actions, rewards, next_obs, terminated = self._sample_experiences(
            self.batch_size
        )

        # Ensure correct dtypes/shapes
        actions = actions.long().unsqueeze(1)  # [B, 1]
        rewards = rewards.float()  # [B]
        not_done = (~terminated.bool()).float()  # [B]

        # Q(s,a) for taken actions
        q_values = self.policy_network(obs)  # [B, A]
        q_sa = q_values.gather(1, actions).squeeze(1)  # [B]

        with torch.no_grad():
            # Double DQN: action from online net, value from target net
            next_actions = self.policy_network(next_obs).argmax(
                1, keepdim=True
            )  # [B, 1]
            next_q_target = (
                self.target_model(next_obs).gather(1, next_actions).squeeze(1)
            )  # [B]

            target = rewards + self.discount_factor * not_done * next_q_target  # [B]

        # Loss and optimization
        loss = self.criterion(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network: θ' ← τθ + (1-τ)θ'
        with torch.no_grad():
            for p_tgt, p in zip(
                self.target_model.parameters(), self.policy_network.parameters()
            ):
                p_tgt.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
        self.decay_epsilon()
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def _sample_experiences(self, amount: int) -> tuple:
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough experiences in the replay buffer to sample.")

        sampled_batch = self.replay_buffer.sample(amount)
        observations = sampled_batch["observation"].to(self.device)
        actions = sampled_batch["action"].to(self.device)
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

    def decay_epsilon(self) -> None:
        self.global_step += 1
        # progress in [0,1]
        p = min(1.0, self.global_step / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + p * (self.epsilon_end - self.epsilon_start)
