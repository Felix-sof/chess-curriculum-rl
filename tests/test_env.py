"""Tests for the chess environment: action encoding, board tensor, and masking.

Tests that require a real Stockfish binary (reset/step against the engine)
are skipped automatically when Stockfish isn't found on PATH.
"""

from __future__ import annotations

import shutil

import chess
import numpy as np
import pytest

from env import (
    ACTION_SPACE_SIZE,
    ChessEnv,
    action_to_move,
    board_to_tensor,
    compute_action_mask,
    material_balance,
    move_to_action,
)

STOCKFISH_AVAILABLE = shutil.which("stockfish") is not None
requires_stockfish = pytest.mark.skipif(not STOCKFISH_AVAILABLE, reason="stockfish binary not found on PATH")


def test_move_action_round_trip_white_to_move() -> None:
    board = chess.Board()
    for move in board.legal_moves:
        action = move_to_action(move, board.turn)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert action_to_move(action, board.turn) == move


def test_move_action_round_trip_black_to_move() -> None:
    board = chess.Board()
    board.push_san("e4")
    assert board.turn == chess.BLACK
    for move in board.legal_moves:
        action = move_to_action(move, board.turn)
        assert action_to_move(action, board.turn) == move


def test_move_action_round_trip_promotion() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    promotion_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert promotion_moves, "fixture position should offer promotion moves"
    for move in promotion_moves:
        action = move_to_action(move, board.turn)
        assert action_to_move(action, board.turn) == move


def test_board_to_tensor_shape_and_range() -> None:
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    # Starting position: 16 pieces per side.
    assert tensor[:6].sum() == 16
    assert tensor[6:].sum() == 16


def test_board_to_tensor_is_canonical_for_side_to_move() -> None:
    # Give White and Black different pawn counts, then set Black to move:
    # channel 0 (mover) should reflect Black's count, channel 6 (opponent) White's.
    board = chess.Board()
    board.remove_piece_at(chess.E2)  # White: 7 pawns, Black: 8 pawns
    board.turn = chess.BLACK
    tensor = board_to_tensor(board)
    assert tensor[0].sum() == 8  # Black (side to move) pawns
    assert tensor[6].sum() == 7  # White (opponent) pawns


def test_compute_action_mask_matches_legal_moves() -> None:
    board = chess.Board()
    mask = compute_action_mask(board)
    assert mask.sum() == len(list(board.legal_moves))
    for move in board.legal_moves:
        assert mask[move_to_action(move, board.turn)]


def test_material_balance_starting_position_is_zero() -> None:
    board = chess.Board()
    assert material_balance(board, chess.WHITE) == 0
    assert material_balance(board, chess.BLACK) == 0


def test_material_balance_reflects_captures() -> None:
    board = chess.Board()
    board.remove_piece_at(chess.D7)  # remove a black pawn
    assert material_balance(board, chess.WHITE) == 1
    assert material_balance(board, chess.BLACK) == -1


def test_chess_env_action_masks_without_engine() -> None:
    # action_masks() only touches env.board, not the Stockfish process.
    env = ChessEnv(stockfish_path="stockfish", skill_level=0)
    mask = env.action_masks()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.sum() == len(list(env.board.legal_moves))


def test_chess_env_set_skill_level_clamps_range() -> None:
    env = ChessEnv(stockfish_path="stockfish", skill_level=0)
    env.set_skill_level(-5)
    assert env.skill_level == 0
    env.set_skill_level(99)
    assert env.skill_level == 21


def test_chess_env_ends_on_claimable_repetition() -> None:
    # A shuffling policy (e.g. Nf3/Ng1 back and forth) can reach a
    # threefold-repeatable position, which python-chess only treats as
    # game-over when explicitly asked to consider a claimable draw. Without
    # that, an under-trained agent could shuffle for a very long time before
    # the much-later automatic fivefold/75-move cutoff.
    env = ChessEnv(stockfish_path="definitely-not-a-real-engine", skill_level=0)
    board = env.board
    for _ in range(2):
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")
    assert not board.is_game_over()  # not over without considering a claim
    assert board.is_game_over(claim_draw=True)  # but a draw is claimable

    env.agent_color = chess.WHITE
    env._prev_material = material_balance(board, chess.WHITE)
    action = move_to_action(chess.Move.from_uci("g1f3"), board.turn)
    _obs, reward, terminated, _truncated, info = env.step(action)

    assert terminated
    assert reward == 0.0
    assert info["result"] == "1/2-1/2"


def test_chess_env_bootstrap_stage_needs_no_engine() -> None:
    # skill_level 0 is a uniform-random mover; a full episode should run
    # without ever spawning a Stockfish process (a bogus path is fine here).
    env = ChessEnv(stockfish_path="definitely-not-a-real-engine", skill_level=0)
    try:
        obs, _info = env.reset()
        assert obs.shape == (12, 8, 8)

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 300:
            mask = env.action_masks()
            legal_actions = np.flatnonzero(mask)
            action = int(np.random.choice(legal_actions))
            obs, reward, terminated, truncated, _info = env.step(action)
            steps += 1

        assert terminated
        assert reward in (-1.0, 0.0, 1.0)
    finally:
        env.close()


@requires_stockfish
def test_chess_env_skill_level_one_maps_to_stockfish_zero() -> None:
    env = ChessEnv(stockfish_path="stockfish", skill_level=1)
    try:
        engine = env._ensure_engine()
        assert env._configured_skill == 0
        assert engine is not None
    finally:
        env.close()


@requires_stockfish
def test_chess_env_reset_and_step_reach_terminal_state() -> None:
    env = ChessEnv(stockfish_path="stockfish", skill_level=1, engine_think_time=0.01)
    try:
        obs, info = env.reset()
        assert obs.shape == (12, 8, 8)
        assert "agent_color" in info

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 300:
            mask = env.action_masks()
            legal_actions = np.flatnonzero(mask)
            action = int(np.random.choice(legal_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated
        assert reward in (-1.0, 0.0, 1.0)
        assert "result" in info
    finally:
        env.close()
