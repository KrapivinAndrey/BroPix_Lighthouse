"""
Тесты для модуля processing.speed_limit_checker.
"""

import time

import pytest

from src.processing.speed_limit_checker import SpeedLimitChecker, SpeedLimitState


def test_speed_limit_checker_init() -> None:
    """SpeedLimitChecker должен корректно инициализироваться."""
    checker = SpeedLimitChecker(speed_limit_kmh=10.0, red_hold_seconds=2.0)
    assert checker.speed_limit_kmh == 10.0
    assert checker.red_hold_seconds == 2.0


def test_speed_limit_checker_speed_exceeded() -> None:
    """Флаг превышения должен устанавливаться при превышении скорости."""
    checker = SpeedLimitChecker(speed_limit_kmh=8.0, red_hold_seconds=2.0)
    now = time.time()

    state = checker.update_state(
        prev_is_over_limit=False,
        prev_last_over_limit_time=None,
        now=now,
        current_speed_kmh=10.0,
    )

    assert state.is_over_limit is True
    assert state.last_over_limit_time == now


def test_speed_limit_checker_speed_below_limit() -> None:
    """Флаг превышения не должен устанавливаться при скорости ниже лимита."""
    checker = SpeedLimitChecker(speed_limit_kmh=8.0, red_hold_seconds=2.0)
    now = time.time()

    state = checker.update_state(
        prev_is_over_limit=False,
        prev_last_over_limit_time=None,
        now=now,
        current_speed_kmh=5.0,
    )

    assert state.is_over_limit is False
    assert state.last_over_limit_time is None


def test_speed_limit_checker_red_hold() -> None:
    """Логика red_hold должна удерживать флаг после снижения скорости."""
    checker = SpeedLimitChecker(speed_limit_kmh=8.0, red_hold_seconds=2.0)
    t0 = time.time()

    # Превышение скорости
    state1 = checker.update_state(
        prev_is_over_limit=False,
        prev_last_over_limit_time=None,
        now=t0,
        current_speed_kmh=10.0,
    )
    assert state1.is_over_limit is True

    # Скорость упала ниже лимита, но прошло меньше red_hold_seconds
    t1 = t0 + 1.0
    state2 = checker.update_state(
        prev_is_over_limit=state1.is_over_limit,
        prev_last_over_limit_time=state1.last_over_limit_time,
        now=t1,
        current_speed_kmh=5.0,
    )
    assert state2.is_over_limit is True

    # Прошло больше red_hold_seconds, флаг должен сброситься
    t2 = t0 + 3.0
    state3 = checker.update_state(
        prev_is_over_limit=state2.is_over_limit,
        prev_last_over_limit_time=state2.last_over_limit_time,
        now=t2,
        current_speed_kmh=5.0,
    )
    assert state3.is_over_limit is False


def test_speed_limit_checker_none_speed() -> None:
    """Обработка случая, когда скорость неизвестна (None)."""
    checker = SpeedLimitChecker(speed_limit_kmh=8.0, red_hold_seconds=2.0)
    now = time.time()

    # Если скорость None и не было предыдущего превышения, флаг False
    state1 = checker.update_state(
        prev_is_over_limit=False,
        prev_last_over_limit_time=None,
        now=now,
        current_speed_kmh=None,
    )
    assert state1.is_over_limit is False

    # Если было превышение и прошло меньше red_hold_seconds, флаг остаётся True
    t0 = time.time()
    state2 = checker.update_state(
        prev_is_over_limit=True,
        prev_last_over_limit_time=t0,
        now=t0 + 1.0,
        current_speed_kmh=None,
    )
    assert state2.is_over_limit is True
