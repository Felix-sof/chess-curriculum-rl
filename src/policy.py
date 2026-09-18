"""CNN feature extractor for the (12, 8, 8) board tensor, plugged into SB3 policies."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ChessCNN(BaseFeaturesExtractor):
    """Three-layer convolutional trunk producing a fixed-size embedding.

    Applied identically to policy and value heads via SB3's
    ``features_extractor_class`` mechanism.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
