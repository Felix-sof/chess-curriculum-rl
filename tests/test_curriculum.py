"""Unit tests for CurriculumManager's level-promotion logic."""

from __future__ import annotations

from pathlib import Path

from curriculum import CurriculumManager


def test_starts_at_configured_level() -> None:
    curriculum = CurriculumManager(start_level=3)
    assert curriculum.level == 3
    assert curriculum.win_rate() == 0.0


def test_no_promotion_before_window_fills() -> None:
    curriculum = CurriculumManager(start_level=0, window_size=5, promotion_threshold=0.7)
    for _ in range(4):
        promoted = curriculum.record_result("win")
        assert not promoted
    assert curriculum.level == 0


def test_promotes_when_threshold_met() -> None:
    curriculum = CurriculumManager(start_level=0, window_size=5, promotion_threshold=0.6)
    results = ["win", "win", "win", "loss", "loss"]  # 3/5 = 0.6
    promoted_flags = [curriculum.record_result(r) for r in results]
    assert promoted_flags == [False, False, False, False, True]
    assert curriculum.level == 1


def test_stays_when_below_threshold() -> None:
    curriculum = CurriculumManager(start_level=0, window_size=4, promotion_threshold=0.75)
    results = ["win", "win", "loss", "loss"]  # 0.5 win rate
    promoted_flags = [curriculum.record_result(r) for r in results]
    assert not any(promoted_flags)
    assert curriculum.level == 0


def test_history_resets_after_promotion() -> None:
    curriculum = CurriculumManager(start_level=0, window_size=3, promotion_threshold=1.0)
    curriculum.record_result("win")
    curriculum.record_result("win")
    curriculum.record_result("win")  # promotes here, history clears
    assert curriculum.level == 1
    assert curriculum.win_rate() == 0.0
    curriculum.record_result("loss")
    assert curriculum.win_rate() == 0.0  # single loss in a fresh window


def test_does_not_exceed_max_level() -> None:
    curriculum = CurriculumManager(start_level=20, max_level=20, window_size=2, promotion_threshold=0.5)
    curriculum.record_result("win")
    promoted = curriculum.record_result("win")
    assert not promoted
    assert curriculum.level == 20
    assert curriculum.is_max_level()


def test_draw_counts_as_half_win() -> None:
    curriculum = CurriculumManager(start_level=0, window_size=2, promotion_threshold=0.5)
    curriculum.record_result("draw")
    promoted = curriculum.record_result("draw")
    assert promoted
    assert curriculum.level == 1


def test_logs_to_csv(tmp_path: Path) -> None:
    log_path = tmp_path / "curriculum_log.csv"
    curriculum = CurriculumManager(start_level=0, window_size=2, promotion_threshold=0.5, log_path=log_path)
    curriculum.record_result("win")
    curriculum.record_result("loss")

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("episode,timestamp,level,result,win_rate,promoted")
    assert len(lines) == 3  # header + 2 results


def test_saves_and_loads_level_for_resume(tmp_path: Path) -> None:
    log_path = tmp_path / "curriculum_log.csv"
    curriculum = CurriculumManager(start_level=0, window_size=2, promotion_threshold=0.5, log_path=log_path)
    curriculum.record_result("win")
    curriculum.record_result("win")  # promotes to level 1

    state_path = tmp_path / "curriculum_state.json"
    assert state_path.exists()
    assert CurriculumManager.load_level(state_path) == 1


def test_load_level_missing_file_returns_none(tmp_path: Path) -> None:
    assert CurriculumManager.load_level(tmp_path / "does_not_exist.json") is None
