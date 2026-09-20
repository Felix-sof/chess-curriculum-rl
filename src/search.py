"""Depth-limited alpha-beta search using a trained model as move-orderer and leaf evaluator.

No retraining needed: this adds real lookahead on top of any trained
MaskablePPO checkpoint, purely at inference time. It directly targets a
concrete weakness observed during training -- a policy network with no
search often can't convert a winning position into checkmate, even when it
correctly judges the position as winning. Alpha-beta search finds forced
mates and tactics exactly (via terminal-position detection), regardless of
how good the network's own judgement is beyond the search horizon; the
network's policy output is only used to order moves for better pruning, and
its value output only scores leaf positions.

Usage (as a library):
    from search import best_move
    move = best_move(model, board, depth=3)
"""

from __future__ import annotations

import chess
import torch
from sb3_contrib import MaskablePPO

from env import board_to_tensor, compute_action_mask, move_to_action


def _evaluate(model: MaskablePPO, board: chess.Board) -> float:
    """Value estimate for `board`, from the perspective of the side to move."""
    obs = board_to_tensor(board)
    obs_tensor, _ = model.policy.obs_to_tensor(obs[None, ...])
    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)
    return float(value.item())


def _ordered_legal_moves(model: MaskablePPO, board: chess.Board) -> list[chess.Move]:
    """Legal moves ordered best-first by the policy's prior probability, for better pruning."""
    legal_moves = list(board.legal_moves)
    if len(legal_moves) <= 1:
        return legal_moves

    obs = board_to_tensor(board)
    mask = compute_action_mask(board)
    obs_tensor, _ = model.policy.obs_to_tensor(obs[None, ...])
    mask_tensor = torch.as_tensor(mask[None, ...])
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
        probs = distribution.distribution.probs[0]

    scored = [(probs[move_to_action(move, board.turn)].item(), move) for move in legal_moves]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in scored]


def _negamax(model: MaskablePPO, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """Negamax alpha-beta search; returns a score from the side-to-move's perspective."""
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0
    if depth == 0:
        return _evaluate(model, board)

    best_score = -float("inf")
    for move in _ordered_legal_moves(model, board):
        board.push(move)
        score = -_negamax(model, board, depth - 1, -beta, -alpha)
        board.pop()
        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break  # beta cutoff: the opponent won't let play reach this branch
    return best_score


def best_move(model: MaskablePPO, board: chess.Board, depth: int = 3) -> chess.Move:
    """Pick the side-to-move's best move, searching `depth` ply ahead.

    `depth=1` is equivalent to the network's own greedy choice (no lookahead
    beyond scoring each immediate reply); depth 3-4 is a reasonable balance
    of strength and speed for a single CPU-bound search.
    """
    legal_moves = _ordered_legal_moves(model, board)
    chosen = legal_moves[0]
    best_score = -float("inf")
    alpha, beta = -float("inf"), float("inf")
    for move in legal_moves:
        board.push(move)
        score = -_negamax(model, board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best_score:
            best_score = score
            chosen = move
        alpha = max(alpha, best_score)
    return chosen
