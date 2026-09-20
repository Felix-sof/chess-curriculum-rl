"""Supervised pretraining: warm-start the CNN feature extractor on human moves.

Before any reinforcement learning happens, the policy's convolutional trunk
(the same `ChessCNN` used by `train.py`) is trained as a plain classifier --
given a position, predict the human's actual move -- on real games from
lichess.org's public database (see prepare_pgn_dataset.py). This gives PPO a
"plays somewhat sensibly" starting point instead of random weights, the same
idea used to bootstrap early AlphaGo. Only the feature extractor is saved;
`train.py --pretrained-features <path>` loads it into a fresh MaskablePPO
model before RL fine-tuning begins. The policy/value heads still start fresh
and are learned entirely through RL, since the human data isn't rich enough
to say what the value of a position is or match SB3's own head architecture.

Usage:
    python src/pretrain.py --dataset data/pretrain_positions.npz --epochs 5 \
        --out logs/pretrained_features.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.utils.data import DataLoader, Dataset

from env import ACTION_SPACE_SIZE
from policy import ChessCNN


class PositionDataset(Dataset):
    """Wraps the (board, human-move) arrays saved by prepare_pgn_dataset.py."""

    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.boards = data["boards"]  # uint8 (N, 12, 8, 8), 0/1
        self.actions = data["actions"]  # int32 (N,)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        board = torch.from_numpy(self.boards[idx]).float()
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return board, action


class PretrainNet(nn.Module):
    """ChessCNN + a plain linear move classifier, for supervised pretraining only."""

    def __init__(self, features_dim: int) -> None:
        super().__init__()
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float32)
        self.features_extractor = ChessCNN(observation_space, features_dim=features_dim)
        self.move_head = nn.Linear(features_dim, ACTION_SPACE_SIZE)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.move_head(self.features_extractor(board))


def train(
    dataset_path: str,
    out_path: str,
    features_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PositionDataset(dataset_path)
    val_size = max(1, len(dataset) // 20)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Device: {device}. Train positions: {train_size:,}. Val positions: {val_size:,}.")

    model = PretrainNet(features_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for boards, actions in train_loader:
            boards, actions = boards.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(boards)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * boards.size(0)
        train_loss = total_loss / train_size

        model.eval()
        correct = 0
        with torch.no_grad():
            for boards, actions in val_loader:
                boards, actions = boards.to(device), actions.to(device)
                logits = model(boards)
                correct += (logits.argmax(dim=1) == actions).sum().item()
        val_acc = correct / val_size

        print(f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_move_accuracy={val_acc:.4f}")

    torch.save(model.features_extractor.state_dict(), out_path)
    print(f"Saved pretrained feature extractor to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the CNN feature extractor on human moves.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to a .npz file from prepare_pgn_dataset.py."
    )
    parser.add_argument("--out", type=str, default="logs/pretrained_features.pt")
    parser.add_argument(
        "--features-dim", type=int, default=256, help="Must match train.py's policy.features_dim."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args.dataset, args.out, args.features_dim, args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
