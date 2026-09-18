"""Evaluate a trained model's win rate against a fixed Stockfish skill level.

Usage:
    python src/evaluate.py --model logs/checkpoints/ppo_chess_final.zip --stockfish-level 5
"""

from __future__ import annotations

import argparse
from typing import Any

from sb3_contrib import MaskablePPO

from env import ChessEnv


def evaluate(
    model_path: str,
    stockfish_path: str,
    stockfish_level: int,
    num_games: int,
    engine_think_time: float,
) -> dict[str, Any]:
    """Play ``num_games`` against Stockfish at a fixed level and report the outcome tally.

    ``stockfish_level`` is the literal Stockfish Skill Level (0-20); it's
    converted to ``ChessEnv``'s ``skill_level`` (which reserves 0 for the
    random-move bootstrap opponent, shifting Stockfish levels to 1-21).
    """
    model = MaskablePPO.load(model_path)
    env = ChessEnv(
        stockfish_path=stockfish_path,
        skill_level=stockfish_level + 1,
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args.model, args.stockfish_path, args.stockfish_level, args.num_games, args.think_time)


if __name__ == "__main__":
    main()
