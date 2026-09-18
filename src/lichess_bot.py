"""Optional Lichess Bot API integration for the trained model.

Connects to Lichess as a BOT account (via the ``berserk`` client), accepts
challenges, and plays games using the trained policy. Kept separate from
the training pipeline so it never blocks or is required by it.

IMPORTANT -- before running this:
  1. You need a dedicated Lichess BOT account (not a regular account, and
     an account cannot be converted to BOT once it has played a rated
     game). Create one at https://lichess.org and upgrade it following
     https://lichess.org/api#tag/Bot/operation/botAccountUpgrade
  2. Generate a personal API token for that account with the
     ``bot:play`` scope: https://lichess.org/account/oauth/token
  3. Set it as an environment variable, e.g. (PowerShell):
         $env:LICHESS_BOT_TOKEN = "your-token-here"
     Never commit this token to source control.

This module does not create the account or token for you -- that step is
manual and intentionally left to you.

Usage:
    python src/lichess_bot.py --model logs/checkpoints/ppo_chess_final.zip
"""

from __future__ import annotations

import argparse
import os

import berserk
import chess
from sb3_contrib import MaskablePPO

from env import action_to_move, board_to_tensor, compute_action_mask


def play_game(client: berserk.Client, model: MaskablePPO, game_id: str) -> None:
    """Stream one Lichess game and respond with the model's moves on our turns."""
    board = chess.Board()
    my_color: chess.Color | None = None

    for event in client.bots.stream_game_state(game_id):
        if event["type"] == "gameFull":
            my_color = chess.WHITE if event["white"].get("id") == client.account.get()["id"] else chess.BLACK
            moves = event["state"]["moves"].split()
        elif event["type"] == "gameState":
            moves = event["moves"].split()
        else:
            continue

        board = chess.Board()
        for uci_move in moves:
            board.push_uci(uci_move)

        if board.is_game_over(claim_draw=True) or my_color is None or board.turn != my_color:
            continue

        obs = board_to_tensor(board)
        action_masks = compute_action_mask(board)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        move = action_to_move(int(action), board.turn)
        client.bots.make_move(game_id, move.uci())


def run(model_path: str, token: str) -> None:
    model = MaskablePPO.load(model_path)
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    print("Connected to Lichess as a BOT account. Waiting for challenges (Ctrl+C to stop)...")
    for event in client.bots.stream_incoming_events():
        if event["type"] == "challenge":
            challenge = event["challenge"]
            if challenge["variant"]["key"] == "standard":
                client.bots.accept_challenge(challenge["id"])
            else:
                client.bots.decline_challenge(challenge["id"])
        elif event["type"] == "gameStart":
            play_game(client, model, event["game"]["id"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained model as a Lichess BOT.")
    parser.add_argument("--model", type=str, required=True, help="Path to a MaskablePPO .zip checkpoint.")
    parser.add_argument(
        "--token-env",
        type=str,
        default="LICHESS_BOT_TOKEN",
        help="Name of the environment variable holding the Lichess BOT API token.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"No Lichess API token found in ${args.token_env}. See this file's module "
            "docstring for how to create a BOT account and token."
        )
    run(args.model, token)


if __name__ == "__main__":
    main()
