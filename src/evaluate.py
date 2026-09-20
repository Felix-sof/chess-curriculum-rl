"""Evaluate a trained model's win rate against a fixed Stockfish skill level.

Usage:
    python src/evaluate.py --model logs/checkpoints/ppo_chess_final.zip --stockfish-level 5
"""

from __future__ import annotations

import argparse
from typing import Any

from sb3_contrib import MaskablePPO

from env import ChessEnv, move_to_action, stockfish_level_to_skill_level
from search import best_move


def evaluate(
    model_path: str,
    stockfish_path: str,
    stockfish_level: int,
    num_games: int,
    engine_think_time: float,
    search_depth: int = 1,
) -> dict[str, Any]:
    """Play ``num_games`` against Stockfish at a fixed level and report the outcome tally.

    ``stockfish_level`` is the literal Stockfish Skill Level (0-20); it's
    converted to ``ChessEnv``'s ``skill_level`` scale (which reserves the low
    end for the random-move bootstrap and depth-ramp stages). ``search_depth``
    > 1 picks moves via alpha-beta search (src/search.py) instead of the
    network's single greedy forward pass.
    """
    model = MaskablePPO.load(model_path)
    env = ChessEnv(
        stockfish_path=stockfish_path,
        skill_level=stockfish_level_to_skill_level(stockfish_level),
        engine_think_time=engine_think_time,
    )

    wins = losses = draws = 0
    try:
        for game in range(1, num_games + 1):
            obs, _info = env.reset()
            terminated = truncated = False
            reward = 0.0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                if search_depth > 1:
                    move = best_move(model, env.board, depth=search_depth)
                    action = move_to_action(move, env.board.turn)
                else:
                    action_masks = env.action_masks()
                    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))

            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
            print(f"Game {game}/{num_games}: result={info.get('result', '?')} reward={reward:+.1f}")
    finally:
        env.close()

    total = wins + losses + draws
    win_rate = wins / total if total else 0.0
    summary = {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}
    print(summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained chess model against Stockfish.")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to a saved MaskablePPO .zip checkpoint."
    )
    parser.add_argument("--stockfish-path", type=str, default="stockfish")
    parser.add_argument("--stockfish-level", type=int, default=5)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--think-time", type=float, default=0.1)
    parser.add_argument(
        "--search-depth",
        type=int,
        default=1,
        help="Ply of alpha-beta lookahead per move (1 = the network's own greedy choice, no search).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(
        args.model,
        args.stockfish_path,
        args.stockfish_level,
        args.num_games,
        args.think_time,
        args.search_depth,
    )


if __name__ == "__main__":
    main()
