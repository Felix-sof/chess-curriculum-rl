"""Live, hands-off viewing mode: watch a trained model play continuously.

No user input is required once started -- games play out automatically,
one after another, with the board printed after every move.

Usage:
    python src/watch.py --model logs/checkpoints/ppo_chess_final.zip \
        --opponent stockfish --stockfish-level 5 --num-games 3
    python src/watch.py --model logs/checkpoints/ppo_chess_final.zip --opponent self
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import chess
from sb3_contrib import MaskablePPO

from env import (
    ChessEnv,
    action_to_move,
    board_to_tensor,
    compute_action_mask,
    move_to_action,
    stockfish_level_to_skill_level,
)
from search import best_move


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 stdout so Unicode chess glyphs print reliably.

    Windows consoles often default to a legacy codepage (e.g. cp1254) that
    can't encode chess piece characters, which would otherwise crash on the
    very first board print.
    """
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def play_stockfish_game(
    model: MaskablePPO, env: ChessEnv, delay: float, search_depth: int
) -> tuple[str, float]:
    """Play one game of the agent against the environment's Stockfish opponent."""
    obs, _info = env.reset()
    env.render()
    terminated = truncated = False
    reward = 0.0
    info: dict = {}
    while not (terminated or truncated):
        if search_depth > 1:
            action = move_to_action(best_move(model, env.board, depth=search_depth), env.board.turn)
        else:
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        time.sleep(delay)
        env.render()
    return info.get("result", "?"), reward


def play_self_game(model: MaskablePPO, delay: float, search_depth: int) -> tuple[str, float]:
    """Play one game of the model against itself, one board, both colors.

    Because observations and actions are encoded canonically from the
    side-to-move's perspective (see env.board_to_tensor), the same model
    can drive both colors without any special-casing.
    """
    board = chess.Board()
    print(board.unicode(borders=True))
    # claim_draw=True: an under-trained model can shuffle between two "safe"
    # moves forever otherwise, since repetition isn't a claim by default.
    while not board.is_game_over(claim_draw=True):
        if search_depth > 1:
            move = best_move(model, board, depth=search_depth)
        else:
            obs = board_to_tensor(board)
            action_masks = compute_action_mask(board)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            move = action_to_move(int(action), board.turn)
        board.push(move)
        time.sleep(delay)
        print(board.unicode(borders=True))

    result = board.result(claim_draw=True)
    if result == "1-0":
        reward = 1.0
    elif result == "0-1":
        reward = -1.0
    else:
        reward = 0.0
    return result, reward


def summarize(reward: float) -> str:
    if reward > 0:
        return "WIN"
    if reward < 0:
        return "LOSS"
    return "DRAW"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch the trained agent play continuously, no input required."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to a saved MaskablePPO .zip checkpoint."
    )
    parser.add_argument("--opponent", choices=["stockfish", "self"], default="stockfish")
    parser.add_argument("--stockfish-path", type=str, default="stockfish")
    parser.add_argument("--stockfish-level", type=int, default=5)
    parser.add_argument("--num-games", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to pause between moves.")
    parser.add_argument(
        "--search-depth",
        type=int,
        default=1,
        help="Ply of alpha-beta lookahead per move (1 = the network's own greedy choice, no search).",
    )
    return parser


def main() -> None:
    _ensure_utf8_stdout()
    args = build_arg_parser().parse_args()
    model = MaskablePPO.load(args.model)

    env: ChessEnv | None = None
    if args.opponent == "stockfish":
        # --stockfish-level is the literal Stockfish Skill Level (0-20).
        env = ChessEnv(
            stockfish_path=args.stockfish_path,
            skill_level=stockfish_level_to_skill_level(args.stockfish_level),
            render_mode="human",
        )

    score = {"WIN": 0, "LOSS": 0, "DRAW": 0}
    try:
        for game in range(1, args.num_games + 1):
            print(f"\n=== Game {game}/{args.num_games} (opponent={args.opponent}) ===")
            if args.opponent == "stockfish":
                assert env is not None
                result, reward = play_stockfish_game(model, env, args.delay, args.search_depth)
            else:
                result, reward = play_self_game(model, args.delay, args.search_depth)

            outcome = summarize(reward)
            score[outcome] += 1
            print(f"Game {game} result: {result} ({outcome})")
            print(f"Score so far -> Wins: {score['WIN']}  Losses: {score['LOSS']}  Draws: {score['DRAW']}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
