"""
Тесты для модуля processing.track_manager.
"""

import time

import pytest

from src.processing.track_manager import TrackData, TrackManager


def test_track_manager_init() -> None:
    """TrackManager должен корректно инициализироваться."""
    manager = TrackManager(ttl_seconds=2.0)
    assert manager._ttl_seconds == 2.0
    assert len(manager.tracks) == 0


def test_track_manager_upsert() -> None:
    """Метод upsert должен создавать или обновлять трек."""
    manager = TrackManager()
    now = time.time()

    track = manager.upsert(
        1,
        last_pos=(10.0, 20.0),
        last_time=now,
        speed_kmh=5.0,
        is_over_limit=False,
        last_over_limit_time=None,
    )

    assert track.last_pos == (10.0, 20.0)
    assert track.last_time == now
    assert track.speed_kmh == 5.0
    assert track.is_over_limit is False
    assert 1 in manager.tracks


def test_track_manager_get() -> None:
    """Метод get должен возвращать трек по ID."""
    manager = TrackManager()
    now = time.time()

    manager.upsert(
        1,
        last_pos=(10.0, 20.0),
        last_time=now,
        speed_kmh=5.0,
        is_over_limit=False,
        last_over_limit_time=None,
    )

    track = manager.get(1)
    assert track is not None
    assert track.last_pos == (10.0, 20.0)

    assert manager.get(999) is None


def test_track_manager_cleanup_expired() -> None:
    """Метод cleanup_expired должен удалять устаревшие треки."""
    manager = TrackManager(ttl_seconds=1.0)
    t0 = time.time()

    manager.upsert(1, last_pos=(0, 0), last_time=t0, speed_kmh=None, is_over_limit=False, last_over_limit_time=None)
    manager.upsert(2, last_pos=(0, 0), last_time=t0 + 0.5, speed_kmh=None, is_over_limit=False, last_over_limit_time=None)

    assert 1 in manager.tracks
    assert 2 in manager.tracks

    # Очистка через 1.5 секунды (трек 1 устарел, трек 2 ещё актуален)
    manager.cleanup_expired(t0 + 1.5)

    assert 1 not in manager.tracks
    assert 2 in manager.tracks
