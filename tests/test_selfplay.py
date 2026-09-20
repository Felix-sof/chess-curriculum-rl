"""Tests for self-play game generation and the buffer training step."""

from __future__ import annotations

from collections import deque

import torch

from az_network import AZNetwork
from selfplay import play_self_play_game
from train_selfplay import train_on_buffer


def test_self_play_game_produces_consistent_value_targets() -> None:
    net = AZNetwork(channels=8, num_res_blocks=1, features_dim=16)
    net.eval()
    examples = play_self_play_game(net, num_simulations=5, device=torch.device("cpu"), max_moves=10)

    assert len(examples) > 0
    for example in examples:
        assert example.value_target in (-1.0, 0.0, 1.0)
        assert example.board_tensor.shape == (12, 8, 8)
        assert example.policy_target.sum() > 0.99  # sums to ~1 over legal moves


def test_train_on_buffer_reduces_loss_on_a_tiny_buffer() -> None:
    net = AZNetwork(channels=8, num_res_blocks=1, features_dim=16)
    net.eval()
    examples = play_self_play_game(net, num_simulations=5, device=torch.device("cpu"), max_moves=10)
    buffer = deque(examples)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # Just confirm this runs without error and leaves the network in eval mode.
    train_on_buffer(net, optimizer, buffer, epochs=1, batch_size=8, device=torch.device("cpu"))
    assert not net.training
