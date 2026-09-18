"""Render the training-progress chart (win rate + curriculum level) from a curriculum log.

Usage:
    python src/plot_progress.py --log logs/curriculum_log.csv --out assets/training_progress.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Reference palette (see dataviz skill): categorical slot 1 (blue) and slot 2 (orange).
COLOR_WIN_RATE = "#2a78d6"
COLOR_LEVEL = "#eb6834"
COLOR_THRESHOLD = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"


def load_log(log_path: Path) -> tuple[list[int], list[float], list[int]]:
    episodes, win_rates, levels = [], [], []
    with log_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            win_rates.append(float(row["win_rate"]))
            levels.append(int(row["level"]))
    return episodes, win_rates, levels


def render(log_path: Path, out_path: Path, promotion_threshold: float = 0.7) -> None:
    episodes, win_rates, levels = load_log(log_path)

    fig, (ax_win, ax_level) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 2], facecolor="#fcfcfb"
    )

    ax_win.set_facecolor("#fcfcfb")
    ax_win.plot(episodes, win_rates, color=COLOR_WIN_RATE, linewidth=2)
    ax_win.axhline(promotion_threshold, color=COLOR_THRESHOLD, linewidth=1, linestyle="--")
    ax_win.set_ylim(0, 1.05)
    ax_win.set_title("Rolling win rate vs. Stockfish", color=COLOR_INK, fontsize=12, loc="left")
    ax_win.set_ylabel("Win rate", color=COLOR_MUTED)
    ax_win.grid(True, color=COLOR_GRID, linewidth=1)
    ax_win.spines[["top", "right"]].set_visible(False)
    ax_win.spines[["left", "bottom"]].set_color(COLOR_AXIS)
    ax_win.tick_params(colors=COLOR_MUTED)

    ax_level.set_facecolor("#fcfcfb")
    ax_level.step(episodes, levels, color=COLOR_LEVEL, linewidth=2, where="post")
    ax_level.set_title("Curriculum: Stockfish skill level", color=COLOR_INK, fontsize=12, loc="left")
    ax_level.set_xlabel("Episode", color=COLOR_MUTED)
    ax_level.set_ylabel("Skill level", color=COLOR_MUTED)
    ax_level.set_ylim(-0.5, 20.5)
    ax_level.grid(True, color=COLOR_GRID, linewidth=1)
    ax_level.spines[["top", "right"]].set_visible(False)
    ax_level.spines[["left", "bottom"]].set_color(COLOR_AXIS)
    ax_level.tick_params(colors=COLOR_MUTED)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved chart to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot win rate and curriculum level from a training log.")
    parser.add_argument("--log", type=str, default="logs/curriculum_log.csv")
    parser.add_argument("--out", type=str, default="assets/training_progress.png")
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    render(Path(args.log), Path(args.out), args.threshold)


if __name__ == "__main__":
    main()
