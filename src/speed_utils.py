"""
Утилиты для расчёта псевдо-скорости объектов по смещению в пикселях.

Скорость оценивается на основе изменения положения центра объекта между
двумя моментами времени и фиксированного коэффициента перевода пикселей
в условные метры.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

DEFAULT_PX_TO_M_SCALE: float = 0.05
"""Условные метры на один пиксель (5 см/px по умолчанию)."""


def compute_speed_kmh(
    last_pos: Tuple[float, float],
    last_time: float,
    current_pos: Tuple[float, float],
    current_time: float,
    px_to_m_scale: float = DEFAULT_PX_TO_M_SCALE,
    min_distance_px: float = 1.0,
) -> Optional[float]:
    """
    Рассчитать псевдо-скорость объекта в км/ч по смещению в пикселях.

    Args:
        last_pos: Предыдущая позиция центра (x, y) в пикселях.
        last_time: Время предыдущей позиции (секунды, time.time()).
        current_pos: Текущая позиция центра (x, y) в пикселях.
        current_time: Время текущей позиции (секунды).
        px_to_m_scale: Коэффициент перевода пикселей в метры.
        min_distance_px: Минимальное смещение в пикселях, ниже которого
            движение считается шумом и скорость не рассчитывается.

    Returns:
        Оценка скорости в км/ч или None, если скорость рассчитать нельзя
        (слишком маленькое dt, нет заметного смещения и т.п.).
    """
    dt = current_time - last_time
    if dt <= 0:
        return None

    dx = current_pos[0] - last_pos[0]
    dy = current_pos[1] - last_pos[1]
    dist_px = math.hypot(dx, dy)

    if dist_px < min_distance_px:
        return None

    # Переводим расстояние в условные метры и затем в км/ч.
    dist_m = dist_px * px_to_m_scale
    v_m_s = dist_m / dt
    v_kmh = v_m_s * 3.6

    # Защита от некорректных значений (NaN, inf).
    if not math.isfinite(v_kmh):
        return None

    return v_kmh

