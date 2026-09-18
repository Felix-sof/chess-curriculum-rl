"""Curriculum tracking: promote the Stockfish skill level based on rolling win rate."""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections import deque
from pathlib import Path
from typing import Literal

GameResult = Literal["win", "loss", "draw"]

_RESULT_VALUES: dict[GameResult, float] = {"win": 1.0, "draw": 0.5, "loss": 0.0}


class CurriculumManager:
    """Tracks the rolling win rate over the last ``window_size`` games and
    promotes the Stockfish ``skill_level`` (0-20) whenever it meets
    ``promotion_threshold``, logging every result to a CSV file.
    """

    def __init__(
        self,
        start_level: int = 0,
        max_level: int = 20,
        window_size: int = 50,
        promotion_threshold: float = 0.7,
        log_path: str | Path | None = None,
    ) -> None:
        self.level = start_level
        self.max_level = max_level
        self.window_size = window_size
        self.promotion_threshold = promotion_threshold
        self.log_path = Path(log_path) if log_path is not None else None
        self.state_path = self.log_path.with_name("curriculum_state.json") if self.log_path else None

        self._history: deque[float] = deque(maxlen=window_size)
        self._episode_count = 0

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                with self.log_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        ["episode", "timestamp", "level", "result", "win_rate", "promoted"]
                    )

    def win_rate(self) -> float:
        """Win rate over the games recorded in the current rolling window."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def is_max_level(self) -> bool:
        return self.level >= self.max_level

    def record_result(self, result: GameResult) -> bool:
        """Record a finished game's result and promote the level if warranted.

        Returns True if this call caused a promotion.
        """
        self._episode_count += 1
        self._history.append(_RESULT_VALUES[result])

        promoted = False
        if (
            not self.is_max_level()
            and len(self._history) >= self.window_size
            and self.win_rate() >= self.promotion_threshold
        ):
            self.level += 1
            self._history.clear()
            promoted = True

        self._log(result, promoted)
        self._save_state()
        return promoted

    def _save_state(self) -> None:
        """Persist the current level so a later ``--resume`` can pick up the
        curriculum where this run left off, instead of restarting at level 0."""
        if self.state_path is None:
            return
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump({"level": self.level}, f)

    @staticmethod
    def load_level(state_path: str | Path) -> int | None:
        """Read a previously saved level from ``state_path``, or None if absent."""
        path = Path(state_path)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)["level"]

    def _log(self, result: GameResult, promoted: bool) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    self._episode_count,
                    _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    self.level,
                    result,
                    round(self.win_rate(), 4),
                    promoted,
                ]
            )
