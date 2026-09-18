"""A minimal UCI (Universal Chess Interface) wrapper around the trained model.

Lets the trained PPO policy be plugged into any UCI-compatible GUI (Arena,
ChessBase, cutechess, ...) as a chess engine. Speaks UCI over stdin/stdout;
supports the core handshake and play loop: ``uci``, ``isready``,
``ucinewgame``, ``position``, ``go``, ``stop``, ``quit``.

Usage:
    python src/uci_engine.py --model logs/checkpoints/ppo_chess_final.zip

Most GUIs launch the engine executable with no arguments, so point
``--model`` at a fixed path (the default below) or pass it as an extra
command-line argument in the GUI's "add engine" dialog if it supports one.
"""

from __future__ import annotations

import argparse
import sys

import chess
from sb3_contrib import MaskablePPO

from env import action_to_move, board_to_tensor, compute_action_mask

ENGINE_NAME = "CurriculumRLChess"
ENGINE_AUTHOR = "chess-curriculum-rl"
DEFAULT_MODEL_PATH = "logs/checkpoints/ppo_chess_final.zip"


class UCIEngine:
    """Tracks board state across a UCI session and picks moves with the trained policy."""

    def __init__(self, model_path: str) -> None:
        self.model = MaskablePPO.load(model_path)
        self.board = chess.Board()

    def set_position(self, tokens: list[str]) -> None:
        """Handle a ``position [startpos|fen <fen>] [moves ...]`` command's tokens."""
        idx = 0
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            idx += 1
            fen_tokens = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_tokens.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_tokens))
        else:
            return

        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            for uci_move in tokens[idx:]:
                self.board.push_uci(uci_move)

    def best_move(self) -> str:
        """Pick the model's best legal move for the current position, in UCI notation."""
        if self.board.is_game_over() or not any(self.board.legal_moves):
            return "0000"
        obs = board_to_tensor(self.board)
        action_masks = compute_action_mask(self.board)
        action, _states = self.model.predict(obs, action_masks=action_masks, deterministic=True)
        move = action_to_move(int(action), self.board.turn)
        return move.uci()


def _send(line: str) -> None:
    print(line, flush=True)


def run(model_path: str) -> None:
    engine: UCIEngine | None = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        command, args = tokens[0], tokens[1:]

        if command == "uci":
            _send(f"id name {ENGINE_NAME}")
            _send(f"id author {ENGINE_AUTHOR}")
            _send("uciok")
        elif command == "isready":
            if engine is None:
                engine = UCIEngine(model_path)
            _send("readyok")
        elif command == "ucinewgame":
            engine = UCIEngine(model_path)
        elif command == "position":
            if engine is None:
                engine = UCIEngine(model_path)
            engine.set_position(args)
        elif command == "go":
            if engine is None:
                engine = UCIEngine(model_path)
            _send(f"bestmove {engine.best_move()}")
        elif command == "stop":
            if engine is not None:
                _send(f"bestmove {engine.best_move()}")
        elif command == "quit":
            break
        # Unknown commands (e.g. "setoption", "debug") are silently ignored.


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained model as a UCI chess engine.")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to a MaskablePPO .zip checkpoint."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
