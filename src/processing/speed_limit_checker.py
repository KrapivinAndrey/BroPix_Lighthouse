from __future__ import annotations

"""
Логика определения превышения скорости с учётом задержки красного сигнала.

Поведение синхронизировано с _update_track_state из tests/test_red_hold_logic.py.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpeedLimitState:
    """Состояние флага превышения скорости для одного трека."""

    is_over_limit: bool
    last_over_limit_time: Optional[float]


class SpeedLimitChecker:
    """
    Инкапсулирует логику «держать красный ещё N секунд после того,
    как скорость опустилась ниже порога».
    """

    def __init__(self, speed_limit_kmh: float, red_hold_seconds: float) -> None:
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.red_hold_seconds = float(red_hold_seconds)

    def update_state(
        self,
        prev_is_over_limit: bool,
        prev_last_over_limit_time: Optional[float],
        *,
        now: float,
        current_speed_kmh: Optional[float],
    ) -> SpeedLimitState:
        """
        Обновить состояние превышения скорости для одного трека.

        Аргументы аналогичны _update_track_state в тестах, но разделены по полям.
        """
        last_over_limit_time: Optional[float]

        if current_speed_kmh is not None and current_speed_kmh > self.speed_limit_kmh:
            is_over_limit = True
            last_over_limit_time = now
        else:
            if (
                prev_is_over_limit
                and isinstance(prev_last_over_limit_time, (int, float))
                and now - float(prev_last_over_limit_time) <= self.red_hold_seconds
            ):
                is_over_limit = True
                last_over_limit_time = float(prev_last_over_limit_time)
            else:
                is_over_limit = False
                last_over_limit_time = (
                    float(prev_last_over_limit_time)
                    if isinstance(prev_last_over_limit_time, (int, float))
                    else None
                )

        return SpeedLimitState(
            is_over_limit=is_over_limit,
            last_over_limit_time=last_over_limit_time,
        )

