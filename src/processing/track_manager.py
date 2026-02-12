from __future__ import annotations

"""
Управление треками объектов (track_id) для расчёта скорости и флага превышения.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TrackData:
    """Состояние одного трека объекта."""

    last_pos: Tuple[float, float]
    last_time: float
    speed_kmh: Optional[float]
    is_over_limit: bool
    last_over_limit_time: Optional[float]


class TrackManager:
    """Простое хранилище треков с поддержкой TTL."""

    def __init__(self, ttl_seconds: float = 2.0) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._tracks: Dict[int, TrackData] = {}

    @property
    def tracks(self) -> Dict[int, TrackData]:
        """Прямой доступ к словарю треков (для чтения/итерации)."""
        return self._tracks

    def get(self, track_id: int) -> Optional[TrackData]:
        return self._tracks.get(track_id)

    def upsert(
        self,
        track_id: int,
        *,
        last_pos: Tuple[float, float],
        last_time: float,
        speed_kmh: Optional[float],
        is_over_limit: bool,
        last_over_limit_time: Optional[float],
    ) -> TrackData:
        data = TrackData(
            last_pos=last_pos,
            last_time=last_time,
            speed_kmh=speed_kmh,
            is_over_limit=is_over_limit,
            last_over_limit_time=last_over_limit_time,
        )
        self._tracks[track_id] = data
        return data

    def cleanup_expired(self, now: float) -> None:
        """Удалить треки, которые устарели по времени last_time."""
        expired: list[int] = []
        for track_id, data in self._tracks.items():
            if now - data.last_time > self._ttl_seconds:
                expired.append(track_id)

        for track_id in expired:
            self._tracks.pop(track_id, None)

