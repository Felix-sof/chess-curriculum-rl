"""Policy+value network for MCTS self-play training (mcts.py, selfplay.py, train_selfplay.py).

A standalone PyTorch module (not tied to SB3, unlike policy.ChessCNN/the
PPO pipeline): given a (12, 8, 8) board tensor, outputs move logits over the
full action space (masked to legal moves by the caller) and a scalar value
estimate in [-1, 1] for the side to move.
"""

from __future__ import annotations

import torch
from torch import nn

from env import ACTION_SPACE_SIZE


class AZNetwork(nn.Module):
    def __init__(self, channels: int = 128, num_res_blocks: int = 6, features_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([_ResBlock(channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, ACTION_SPACE_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(board)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
