import math
import time

import pytest  # type: ignore[import]

from src.speed_utils import DEFAULT_PX_TO_M_SCALE, compute_speed_kmh


def test_compute_speed_kmh_basic() -> None:
    """Базовый расчёт скорости при заметном смещении и нормальном dt."""
    last_pos = (0.0, 0.0)
    current_pos = (10.0, 0.0)
    last_time = 0.0
    current_time = 1.0  # 1 секунда

    v_kmh = compute_speed_kmh(
        last_pos,
        last_time,
        current_pos,
        current_time,
        px_to_m_scale=DEFAULT_PX_TO_M_SCALE,
    )

    assert v_kmh is not None
    # 10 px * 0.05 м/px = 0.5 м за 1 с → 0.5 * 3.6 = 1.8 км/ч
    assert math.isclose(v_kmh, 1.8, rel_tol=1e-3)


def test_compute_speed_kmh_zero_dt_returns_none() -> None:
    """При dt <= 0 скорость не рассчитывается."""
    v_kmh = compute_speed_kmh(
        (0.0, 0.0),
        last_time=1.0,
        current_pos=(10.0, 0.0),
        current_time=1.0,
    )
    assert v_kmh is None


def test_compute_speed_kmh_too_small_distance_returns_none() -> None:
    """Слишком маленькое смещение считается шумом и даёт None."""
    v_kmh = compute_speed_kmh(
        (0.0, 0.0),
        last_time=0.0,
        current_pos=(0.5, 0.5),
        current_time=1.0,
        min_distance_px=2.0,
    )
    assert v_kmh is None


def test_compute_speed_kmh_custom_scale() -> None:
    """Поддерживается переопределение коэффициента px_to_m_scale."""
    v_kmh = compute_speed_kmh(
        (0.0, 0.0),
        last_time=0.0,
        current_pos=(20.0, 0.0),
        current_time=2.0,
        px_to_m_scale=0.1,
    )
    # 20 px * 0.1 м/px = 2 м / 2 c = 1 м/с → 3.6 км/ч
    assert v_kmh is not None
    assert math.isclose(v_kmh, 3.6, rel_tol=1e-3)

