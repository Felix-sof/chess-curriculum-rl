"""Monte Carlo Tree Search guided by a policy+value network (AlphaZero-style).

Unlike src/search.py (plain alpha-beta, used at inference time only on top
of an already-trained PPO model), this MCTS is the core of a different
training approach entirely: the network's raw policy output is never played
directly. Instead, at every move, many simulations build a search tree --
selecting via PUCT, expanding leaves with the network's policy priors and
value estimate, and backing values up the tree -- and the *visit-count
distribution* over root moves becomes both the move actually played and the
training target for the policy (see selfplay.py, train_selfplay.py).
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch
from torch import nn

from env import ACTION_SPACE_SIZE, action_to_move, board_to_tensor, compute_action_mask, move_to_action


class MCTSNode:
    """One position in the search tree. ``value_sum``/``visit_count`` are
    always from this node's own side-to-move's perspective."""

    __slots__ = ("children", "expanded", "prior", "value_sum", "visit_count")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, MCTSNode] = {}
        self.expanded = False

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _evaluate_and_expand(
    node: MCTSNode, board: chess.Board, network: nn.Module, device: torch.device
) -> float:
    """Run the network on `board`, set `node`'s children from legal-move
    priors, and return the value estimate (side-to-move's perspective)."""
    obs = board_to_tensor(board)
    mask = compute_action_mask(board)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = network(obs_tensor)
    policy_logits = policy_logits[0].cpu().numpy()
    value = float(value.item())

    legal_actions = np.flatnonzero(mask)
    logits = policy_logits[legal_actions]
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    for action, prob in zip(legal_actions, probs, strict=True):
        move = action_to_move(int(action), board.turn)
        node.children[move] = MCTSNode(prior=float(prob))
    node.expanded = True
    return value


def _select_child(node: MCTSNode, c_puct: float) -> tuple[chess.Move, MCTSNode]:
    """PUCT selection among `node`'s children."""
    total_visits = sum(child.visit_count for child in node.children.values())
    explore = c_puct * math.sqrt(total_visits + 1)

    best_score = -float("inf")
    best_move: chess.Move | None = None
    best_child: MCTSNode | None = None
    for move, child in node.children.items():
        # child.value() is from the CHILD's side-to-move perspective, i.e.
        # the opponent relative to `node` -- negate to score from node's view.
        score = -child.value() + explore * child.prior / (1 + child.visit_count)
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    assert best_move is not None and best_child is not None
    return best_move, best_child


def run_mcts(
    network: nn.Module,
    board: chess.Board,
    num_simulations: int,
    device: torch.device,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_root_noise: bool = True,
) -> MCTSNode:
    """Build a search tree of `num_simulations` simulations rooted at `board`
    and return the root (its children's visit counts are the search result).
    """
    root = MCTSNode(prior=1.0)
    _evaluate_and_expand(root, board, network, device)

    if add_root_noise and root.children:
        moves = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
        for move, eta in zip(moves, noise, strict=True):
            child = root.children[move]
            child.prior = (1 - dirichlet_eps) * child.prior + dirichlet_eps * float(eta)

    for _ in range(num_simulations):
        node = root
        search_board = board.copy()
        path = [node]

        while node.expanded and node.children:
            move, node = _select_child(node, c_puct)
            search_board.push(move)
            path.append(node)

        if search_board.is_game_over(claim_draw=True):
            outcome = search_board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                value = 0.0
            else:
                value = 1.0 if outcome.winner == search_board.turn else -1.0
        else:
            value = _evaluate_and_expand(node, search_board, network, device)

        for path_node in reversed(path):
            path_node.visit_count += 1
            path_node.value_sum += value
            value = -value  # perspective flips one ply back up the tree

    return root


def visit_count_policy(root: MCTSNode, temperature: float = 1.0) -> tuple[list[chess.Move], np.ndarray]:
    """The search result as a probability distribution over root moves,
    proportional to visit counts. `temperature=0` is argmax (greedy)."""
    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    if temperature == 0:
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
    else:
        scaled = visits ** (1.0 / temperature)
        probs = scaled / scaled.sum()
    return moves, probs


def policy_target_vector(moves: list[chess.Move], probs: np.ndarray, turn: chess.Color) -> np.ndarray:
    """Expand a (moves, probs) pair into a dense vector over the full action
    space, for use as a policy training target."""
    target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for move, prob in zip(moves, probs, strict=True):
        target[move_to_action(move, turn)] = prob
    return target
