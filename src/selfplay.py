"""Generate self-play training games using MCTS (see mcts.py).

Each move is chosen by running `num_simulations` MCTS simulations from the
current position; the resulting visit-count distribution is both the move
actually played (sampled, with a temperature schedule -- exploratory early
in the game, closer to greedy later) and the policy training target for
that position. Once the game ends, its actual outcome is filled in as the
value target for every recorded position, from that position's own
side-to-move's perspective (matching how the network's value output and
MCTS's backup are both defined).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch

from az_network import AZNetwork
from env import board_to_tensor
from mcts import policy_target_vector, run_mcts, visit_count_policy


@dataclass
class SelfPlayExample:
    board_tensor: np.ndarray  # (12, 8, 8) float32, canonical to the mover
    policy_target: np.ndarray  # (ACTION_SPACE_SIZE,) float32, sums to 1
    turn: chess.Color  # side to move when this position was recorded
    value_target: float = field(default=0.0)  # filled in once the game ends


def play_self_play_game(
    network: AZNetwork,
    num_simulations: int,
    device: torch.device,
    c_puct: float = 1.5,
    temperature_moves: int = 20,
    max_moves: int = 300,
) -> list[SelfPlayExample]:
    """Play one game of the network against itself via MCTS, returning a
    training example for every position reached."""
    board = chess.Board()
    examples: list[SelfPlayExample] = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        temperature = 1.0 if ply < temperature_moves else 0.1
        root = run_mcts(network, board, num_simulations, device, c_puct=c_puct, add_root_noise=True)
        moves, probs = visit_count_policy(root, temperature=temperature)

        examples.append(
            SelfPlayExample(
                board_tensor=board_to_tensor(board),
                policy_target=policy_target_vector(moves, probs, board.turn),
                turn=board.turn,
            )
        )

        move = moves[int(np.random.choice(len(moves), p=probs))]
        board.push(move)
        ply += 1

    outcome = board.outcome(claim_draw=True)
    for example in examples:
        if outcome is None or outcome.winner is None:
            example.value_target = 0.0
        else:
            example.value_target = 1.0 if outcome.winner == example.turn else -1.0

    return examples


def generate_games(
    network: AZNetwork,
    num_games: int,
    num_simulations: int,
    device: torch.device,
    c_puct: float = 1.5,
    temperature_moves: int = 20,
) -> list[SelfPlayExample]:
    all_examples: list[SelfPlayExample] = []
    for game in range(1, num_games + 1):
        examples = play_self_play_game(network, num_simulations, device, c_puct, temperature_moves)
        all_examples.extend(examples)
        wins = sum(1 for e in examples if e.value_target > 0)
        print(f"  self-play game {game}/{num_games}: {len(examples)} plies, {wins} winning-side positions")
    return all_examples


def save_examples(examples: list[SelfPlayExample], out_path: Path) -> None:
    boards = np.stack([e.board_tensor for e in examples]).astype(np.float32)
    policies = np.stack([e.policy_target for e in examples]).astype(np.float32)
    values = np.array([e.value_target for e in examples], dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, boards=boards, policies=policies, values=values)
    print(f"Saved {len(examples)} positions to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate self-play games via MCTS.")
    parser.add_argument(
        "--network", type=str, default=None, help="Path to an AZNetwork state dict; random init if omitted."
    )
    parser.add_argument("--out", type=str, default="logs/selfplay/games.npz")
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=128)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = AZNetwork(
        channels=args.channels, num_res_blocks=args.res_blocks, features_dim=args.features_dim
    )
    if args.network:
        network.load_state_dict(torch.load(args.network, map_location=device))
    network.to(device)
    network.eval()

    examples = generate_games(network, args.num_games, args.num_simulations, device)
    save_examples(examples, Path(args.out))


if __name__ == "__main__":
    main()
