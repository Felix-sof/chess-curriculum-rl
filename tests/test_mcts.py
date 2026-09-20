"""Tests for the MCTS implementation. Uses an untrained (random-weight)
network throughout: search correctness for forced mates should hold
regardless of network quality, since terminal positions are scored exactly.
"""

from __future__ import annotations

import chess
import pytest
import torch

from az_network import AZNetwork
from mcts import policy_target_vector, run_mcts, visit_count_policy


@pytest.fixture(scope="module")
def tiny_network() -> AZNetwork:
    net = AZNetwork(channels=16, num_res_blocks=1, features_dim=32)
    net.eval()
    return net


def test_mcts_finds_mate_in_one(tiny_network: AZNetwork) -> None:
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")

    root = run_mcts(
        tiny_network, board, num_simulations=100, device=torch.device("cpu"), add_root_noise=False
    )
    moves, probs = visit_count_policy(root, temperature=0)
    chosen = moves[int(probs.argmax())]

    board.push(chosen)
    assert board.is_checkmate()


def test_visit_count_policy_is_a_probability_distribution(tiny_network: AZNetwork) -> None:
    board = chess.Board()
    root = run_mcts(tiny_network, board, num_simulations=20, device=torch.device("cpu"))
    moves, probs = visit_count_policy(root, temperature=1.0)
    assert len(moves) == len(root.children)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0).all()


def test_policy_target_vector_matches_legal_moves(tiny_network: AZNetwork) -> None:
    board = chess.Board()
    # Enough simulations that every legal move gets at least one visit (a
    # move with 0 visits legitimately gets 0 probability, which isn't a bug).
    root = run_mcts(tiny_network, board, num_simulations=200, device=torch.device("cpu"))
    moves, probs = visit_count_policy(root, temperature=1.0)
    target = policy_target_vector(moves, probs, board.turn)
    assert target.sum() == pytest.approx(1.0, abs=1e-5)
    assert (target > 0).sum() == len(moves)
