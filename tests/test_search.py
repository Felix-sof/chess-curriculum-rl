"""Tests for the alpha-beta search module.

Search correctness (finding forced mates/tactics) should hold regardless of
how good the underlying model is, since terminal positions are scored
exactly rather than through the network -- so these tests use a tiny,
untrained model rather than requiring a real checkpoint.
"""

from __future__ import annotations

import chess
import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env import ChessEnv
from policy import ChessCNN
from search import best_move


def _mask_fn(env: ChessEnv):
    return env.action_masks()


@pytest.fixture(scope="module")
def tiny_model() -> MaskablePPO:
    """An untrained (random-weight) MaskablePPO model, just to exercise the search API."""
    env = ActionMasker(
        ChessEnv(stockfish_path="unused", skill_level=0),
        _mask_fn,
    )
    policy_kwargs = {
        "features_extractor_class": ChessCNN,
        "features_extractor_kwargs": {"features_dim": 16},
    }
    return MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs, n_steps=32, device="cpu")


def test_search_finds_mate_in_one(tiny_model: MaskablePPO) -> None:
    # Fool's mate: after 1. f3 e5 2. g4, Black has Qh4# available.
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")

    move = best_move(tiny_model, board, depth=2)
    board.push(move)

    assert board.is_checkmate()


def test_search_returns_a_legal_move_at_various_depths(tiny_model: MaskablePPO) -> None:
    # With an untrained (random-weight) model, non-terminal leaf scores are
    # meaningless, so this only checks the search mechanics stay correct
    # (legal output, no crash) at each depth -- not move quality.
    board = chess.Board()
    for depth in (1, 2, 3):
        move = best_move(tiny_model, board, depth=depth)
        assert move in board.legal_moves


def test_search_prefers_forced_mate_over_other_moves(tiny_model: MaskablePPO) -> None:
    # Black's king is boxed in by its own pawns; White has both quiet moves
    # and the back-rank mate Qd8# available. A deep-enough search must find
    # it via exact terminal scoring, regardless of the (untrained) network's
    # leaf judgement.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1")
    move = best_move(tiny_model, board, depth=2)
    board.push(move)
    assert board.is_checkmate()
