"""Evaluate every checkpoint from a training run to see the learning curve at a glance.

Useful after a long run: instead of only trusting the final checkpoint, this
plays a handful of games against a fixed Stockfish level for each saved
checkpoint (oldest to newest) and prints a table, so you can quickly see
whether win rate is actually trending up, has plateaued, or the run is worth
abandoning in favor of different hyperparameters -- before deciding whether
to `--resume` it or start over.

Usage:
    python src/sweep_checkpoints.py --checkpoints-dir logs/run2/checkpoints --num-games 10 \
        --out-csv logs/run2/checkpoint_sweep.csv --plot assets/checkpoint_win_rate.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from evaluate import evaluate

_STEP_RE = re.compile(r"(\d+)_steps\.zip$")

# Reference palette (see dataviz skill): categorical slot 1 (blue).
_COLOR_WIN_RATE = "#2a78d6"
_COLOR_GRID = "#e1e0d9"
_COLOR_AXIS = "#c3c2b7"
_COLOR_INK = "#0b0b0b"
_COLOR_MUTED = "#52514e"


def _sorted_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """All ``*_steps.zip`` checkpoints in the directory, sorted by step count."""
    found = []
    for path in checkpoints_dir.glob("*_steps.zip"):
        match = _STEP_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def sweep(
    checkpoints_dir: Path,
    stockfish_path: str,
    stockfish_level: int,
    num_games: int,
    think_time: float,
    out_csv: Path | None,
) -> list[dict]:
    checkpoints = _sorted_checkpoints(checkpoints_dir)
    if not checkpoints:
        print(f"No '*_steps.zip' checkpoints found in {checkpoints_dir}")
        return []

    rows = []
    for steps, path in checkpoints:
        print(f"\n=== {path.name} ({steps} steps) ===")
        summary = evaluate(str(path), stockfish_path, stockfish_level, num_games, think_time)
        rows.append({"steps": steps, "checkpoint": path.name, **summary})

    print("\nsteps      win_rate  wins  draws  losses")
    for row in rows:
        print(
            f"{row['steps']:<10} {row['win_rate']:<9.2f} {row['wins']:<5} "
            f"{row['draws']:<6} {row['losses']:<7}"
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["steps", "checkpoint", "wins", "losses", "draws", "win_rate"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved results to {out_csv}")

    return rows


def plot(rows: list[dict], stockfish_level: int, out_path: Path) -> None:
    """Render a win-rate-vs-training-steps chart from `sweep()`'s results."""
    import matplotlib.pyplot as plt

    steps = [row["steps"] for row in rows]
    win_rates = [row["win_rate"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(steps, win_rates, color=_COLOR_WIN_RATE, linewidth=2, marker="o", markersize=4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Checkpoint win rate vs. Stockfish level {stockfish_level}",
        color=_COLOR_INK,
        fontsize=12,
        loc="left",
    )
    ax.set_xlabel("Training steps", color=_COLOR_MUTED)
    ax.set_ylabel("Win rate", color=_COLOR_MUTED)
    ax.grid(True, color=_COLOR_GRID, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_COLOR_AXIS)
    ax.tick_params(colors=_COLOR_MUTED)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved chart to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint in a directory against Stockfish."
    )
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--stockfish-path", type=str, default="stockfish")
    parser.add_argument("--stockfish-level", type=int, default=0)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--think-time", type=float, default=0.1)
    parser.add_argument("--out-csv", type=str, default=None, help="Optional path to save results as CSV.")
    parser.add_argument(
        "--plot", type=str, default=None, help="Optional path to save a win-rate-vs-steps chart (PNG)."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = sweep(
        Path(args.checkpoints_dir),
        args.stockfish_path,
        args.stockfish_level,
        args.num_games,
        args.think_time,
        Path(args.out_csv) if args.out_csv else None,
    )
    if args.plot:
        plot(rows, args.stockfish_level, Path(args.plot))


if __name__ == "__main__":
    main()
