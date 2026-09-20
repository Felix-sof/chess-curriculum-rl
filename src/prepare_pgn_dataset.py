"""Extract (position, human move) training pairs from a Lichess PGN database dump.

Feeds the supervised-pretraining step (src/pretrain.py): before any
reinforcement learning happens, the policy can be warm-started on real human
games so it starts from "plays reasonably" rather than random weights.
Downloads are not automated here -- get a monthly dump (zstd-compressed)
from https://database.lichess.org/ and pass its path in.

Usage:
    python src/prepare_pgn_dataset.py --pgn-zst data/lichess_2013-01.pgn.zst \
        --out data/pretrain_positions.npz --max-positions 1000000
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

from env import board_to_tensor, move_to_action


def iter_positions(pgn_zst_path: Path, min_elo: int):
    """Yield (board_tensor_uint8, action_label) pairs from every rated game
    in the dump whose players both meet ``min_elo``."""
    decompressor = zstandard.ZstdDecompressor()
    with pgn_zst_path.open("rb") as compressed, decompressor.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                return

            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", "0"))
                black_elo = int(headers.get("BlackElo", "0"))
            except ValueError:
                continue
            if white_elo < min_elo or black_elo < min_elo:
                continue

            board = game.board()
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    break  # malformed/variant game; stop trusting this one
                tensor = (board_to_tensor(board) > 0.5).astype(np.uint8)
                action = move_to_action(move, board.turn)
                yield tensor, action
                board.push(move)


def build_dataset(pgn_zst_path: Path, out_path: Path, max_positions: int, min_elo: int) -> None:
    boards = np.empty((max_positions, 12, 8, 8), dtype=np.uint8)
    actions = np.empty((max_positions,), dtype=np.int32)

    count = 0
    for tensor, action in iter_positions(pgn_zst_path, min_elo):
        boards[count] = tensor
        actions[count] = action
        count += 1
        if count % 50_000 == 0:
            print(f"  extracted {count:,} positions...")
        if count >= max_positions:
            break

    boards = boards[:count]
    actions = actions[:count]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, boards=boards, actions=actions)
    print(f"Saved {count:,} positions to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a supervised-pretraining dataset from a Lichess PGN dump."
    )
    parser.add_argument(
        "--pgn-zst", type=str, required=True, help="Path to a lichess_db_standard_rated_*.pgn.zst file."
    )
    parser.add_argument("--out", type=str, default="data/pretrain_positions.npz")
    parser.add_argument("--max-positions", type=int, default=1_000_000)
    parser.add_argument(
        "--min-elo", type=int, default=1600, help="Skip games where either player is rated below this."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_dataset(Path(args.pgn_zst), Path(args.out), args.max_positions, args.min_elo)


if __name__ == "__main__":
    main()
