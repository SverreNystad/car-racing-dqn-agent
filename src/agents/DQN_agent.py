from loguru import logger
import numpy as np

from tensordict import TensorDict
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch.nn.utils import clip_grad_norm_
import wandb

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

        # Track gradients/params (lightweight default)
        wandb.watch(self.policy_network, log="gradients", log_freq=1000)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=50_000,
            ),
            # batch_size=[batch_size],
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
        self._decay_epsilon()

        state = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_network(state)
            action = int(torch.argmax(q_values, dim=1).item())
            return action

    def store(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_observation: np.ndarray,
    ) -> None:
        # Inputs are either frames (images) or 1D vectors (e.g., CartPole)
        # Can save frames as uint8 to save memory, but otherwise float32 to not lose precision
        observation_dtype = torch.uint8 if observation.ndim == 3 else torch.float32

        td = TensorDict(
            {
                "observation": torch.as_tensor(
                    observation, dtype=observation_dtype, device="cpu"
                ).contiguous(),
                "action": torch.as_tensor(action, dtype=torch.long, device="cpu").view(
                    ()
                ),
                "reward": torch.as_tensor(
                    reward, dtype=torch.float32, device="cpu"
                ).view(()),
                "terminated": torch.as_tensor(
                    terminated, dtype=torch.bool, device="cpu"
                ).view(()),
                "next_observation": torch.as_tensor(
                    next_observation, dtype=observation_dtype, device="cpu"
                ).contiguous(),
            },
            batch_size=[],
        )

        self.replay_buffer.add(td)

    def update(
        self,
    ) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            # Not enough experiences to sample a full batch
            return {}

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
        self._decay_epsilon()
        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def _sample_experiences(self, amount: int) -> tuple:
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough experiences in the replay buffer to sample.")

        batch = self.replay_buffer.sample(amount)
        batch = batch.to(self.device, non_blocking=True)

        obs = batch["observation"]
        nxt = batch["next_observation"]

        # For image observations stored as uint8, convert to float and scale to [0,1]
        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255)
            nxt = nxt.float().div_(255)

        actions = batch["action"]
        rewards = batch["reward"]
        terminated = batch["terminated"]

        return obs, actions, rewards, nxt, terminated

    def _decay_epsilon(self) -> None:
        self.global_step += 1
        # progress in [0,1]
        p = min(1.0, self.global_step / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + p * (self.epsilon_end - self.epsilon_start)

    def save_policy(self, policy_name: str) -> None:
        """
        Saves the policy network's parameters to WandB artifacts.
        """

        # Save your model.
        FILE_NAME = f"models/{policy_name}.pth"
        torch.save(self.policy_network.state_dict(), FILE_NAME)
        # Save as artifact for version control.
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(FILE_NAME)
        wandb.log_artifact(artifact)
        logger.info(f"Saved model as {policy_name}.pth and logged to WandB.")

    def load_policy(self, policy_path: str) -> None:
        """
        Loads the policy network's parameters from a file.
        """
        logger.info(f"Loading model from {policy_path}")
        self.policy_network.load_state_dict(
            torch.load(policy_path, map_location=self.device)
        )
        self.target_model.load_state_dict(self.policy_network.state_dict())
        logger.info(f"Loaded model from {policy_path}.")
        self.epsilon = 0


def _download_model_file(policy_name: str) -> str:
    artifact = wandb.use_artifact(policy_name, type="model")
    artifact_dir = artifact.download()
    return _find_latest_model_file(artifact_dir, policy_name)


def _find_latest_model_file(artifact_dir: str, policy_name: str) -> str:
    # The artifact may not contain a file named exactly after the artifact
    # reference. Search for .pth files under the downloaded directory and
    # pick the most appropriate one. Prefer an exact match if present,
    # otherwise choose the largest .pth (most likely the full model).
    import os
    import glob

    # Candidate: exact file (artifact_dir + policy_name + .pth)
    exact_path = os.path.join(artifact_dir, f"{policy_name}.pth")
    pth_path = None
    if os.path.isfile(exact_path):
        pth_path = exact_path
    else:
        # Search recursively for .pth files
        candidates = glob.glob(
            os.path.join(artifact_dir, "**", "*.pth"), recursive=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .pth files found in artifact directory '{artifact_dir}'. "
                f"Contents: {os.listdir(artifact_dir)}"
            )
        # Choose the largest candidate file (heuristic for best model file)
        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        pth_path = candidates[0]
    return pth_path
