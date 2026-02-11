from __future__ import annotations

import time
from typing import Dict, Any

from src.camera import display_video_stream  # noqa: F401  # импорт для соответствия стилю проекта


def _update_track_state(
    prev: Dict[str, Any],
    now: float,
    current_speed: float | None,
    speed_limit_kmh: float,
    red_hold_seconds: float,
) -> Dict[str, Any]:
    """
    Вспомогательная функция, воспроизводящая логику задержки красного сигнала
    для одного трека из display_video_stream.
    """
    is_over_limit_prev = bool(prev.get("is_over_limit", False))
    last_over_limit_time_prev = prev.get("last_over_limit_time")

    if current_speed is not None and current_speed > speed_limit_kmh:
        is_over_limit = True
        last_over_limit_time = now
    else:
        if (
            is_over_limit_prev
            and isinstance(last_over_limit_time_prev, (int, float))
            and now - float(last_over_limit_time_prev) <= red_hold_seconds
        ):
            is_over_limit = True
            last_over_limit_time = last_over_limit_time_prev
        else:
            is_over_limit = False
            last_over_limit_time = last_over_limit_time_prev

    return {
        "is_over_limit": is_over_limit,
        "last_over_limit_time": last_over_limit_time,
    }


def test_red_hold_keeps_over_limit_for_some_time() -> None:
    """Флаг превышения должен удерживаться в течение red_hold_seconds после падения скорости ниже порога."""

    speed_limit_kmh = 8.0
    red_hold_seconds = 2.0

    t0 = time.time()

    # Начальное состояние: впервые превысили скорость
    state = _update_track_state(
        prev={},
        now=t0,
        current_speed=10.0,
        speed_limit_kmh=speed_limit_kmh,
        red_hold_seconds=red_hold_seconds,
    )
    assert state["is_over_limit"] is True
    assert state["last_over_limit_time"] == t0

    # Через 1 секунду скорость упала ниже порога, но флаг ещё должен быть True
    t1 = t0 + 1.0
    state = _update_track_state(
        prev=state,
        now=t1,
        current_speed=5.0,
        speed_limit_kmh=speed_limit_kmh,
        red_hold_seconds=red_hold_seconds,
    )
    assert state["is_over_limit"] is True

    # Через 3 секунды от первого превышения (больше red_hold_seconds) флаг должен сброситься
    t2 = t0 + 3.0
    state = _update_track_state(
        prev=state,
        now=t2,
        current_speed=5.0,
        speed_limit_kmh=speed_limit_kmh,
        red_hold_seconds=red_hold_seconds,
    )
    assert state["is_over_limit"] is False

