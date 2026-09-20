"""AlphaZero-style training: generate self-play games via MCTS, train the
network on them, then repeat with the improved network as the new search
guide. Unlike train.py (PPO against a Stockfish-backed curriculum), this
never touches Stockfish at all -- the opponent is always the network's own
current self, and search is baked into both play and data generation.

Usage:
    python src/train_selfplay.py --log-dir logs/selfplay_run1 --iterations 20 \
        --games-per-iter 30 --num-simulations 40
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from az_network import AZNetwork
from selfplay import SelfPlayExample, generate_games


def train_on_buffer(
    network: AZNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: deque[SelfPlayExample],
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    boards = np.stack([e.board_tensor for e in buffer])
    policies = np.stack([e.policy_target for e in buffer])
    values = np.array([e.value_target for e in buffer], dtype=np.float32)

    dataset = TensorDataset(torch.from_numpy(boards), torch.from_numpy(policies), torch.from_numpy(values))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network.train()
    for epoch in range(1, epochs + 1):
        total_policy_loss = total_value_loss = 0.0
        for board_batch, policy_batch, value_batch in loader:
            board_batch = board_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = network(board_batch)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_batch * log_probs).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value_pred, value_batch)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item() * board_batch.size(0)
            total_value_loss += value_loss.item() * board_batch.size(0)

        n = len(dataset)
        print(
            f"  epoch {epoch}/{epochs}: policy_loss={total_policy_loss / n:.4f} "
            f"value_loss={total_value_loss / n:.4f}"
        )
    network.eval()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play training loop.")
    parser.add_argument("--log-dir", type=str, default="logs/selfplay")
    parser.add_argument(
        "--init-from", type=str, default=None, help="Optional AZNetwork state dict to start from."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iter", type=int, default=30)
    parser.add_argument("--num-simulations", type=int, default=40)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=15)
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Positions kept across iterations.")
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs over the buffer per iteration."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--features-dim", type=int, default=128)
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="Save a checkpoint every N iterations."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_dir = Path(args.log_dir)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    network = AZNetwork(
        channels=args.channels, num_res_blocks=args.res_blocks, features_dim=args.features_dim
    )
    if args.init_from:
        network.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Initialized from {args.init_from}")
    network.to(device)
    network.eval()

    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    buffer: deque[SelfPlayExample] = deque(maxlen=args.buffer_size)

    for iteration in range(1, args.iterations + 1):
        print(f"\n=== Iteration {iteration}/{args.iterations} ===")
        t0 = time.time()

        examples = generate_games(
            network,
            args.games_per_iter,
            args.num_simulations,
            device,
            c_puct=args.c_puct,
            temperature_moves=args.temperature_moves,
        )
        buffer.extend(examples)
        t1 = time.time()
        print(f"  self-play: {len(examples)} new positions ({t1 - t0:.1f}s), buffer size {len(buffer)}")

        train_on_buffer(network, optimizer, buffer, args.epochs, args.batch_size, device)
        t2 = time.time()
        print(f"  training: {t2 - t1:.1f}s")

        if iteration % args.checkpoint_every == 0:
            checkpoint_path = checkpoints_dir / f"az_iter{iteration}.pt"
            torch.save(network.state_dict(), checkpoint_path)
            print(f"  saved {checkpoint_path}")

    final_path = checkpoints_dir / "az_final.pt"
    torch.save(network.state_dict(), final_path)
    print(f"\nTraining complete. Final network saved to {final_path}")


if __name__ == "__main__":
    main()
