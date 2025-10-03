"""Defines the neural network architecture used by the DQN agents."""

from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, action_space_size: int):
        super().__init__()

        # When not using image input, use a MLP
        if len(input_shape) == 1:
            self.net = nn.Sequential(
                nn.Linear(input_shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, action_space_size),
            )
            return

        channel_n, height, width = input_shape
        if height != 84 or width != 84:
            error_text = f"DQN model requires input of a (84, 84)-shape. \
                           Input of a ({height, width})-shape was passed."
            raise ValueError(error_text)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_size),
        )

    def forward(self, input):
        return self.net(input)
