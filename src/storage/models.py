from __future__ import annotations

"""
Модели данных для хранения событий превышения скорости.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpeedExceedanceEvent:
    """
    Событие превышения скорости.

    Содержит информацию о моменте, когда объект превысил установленный лимит скорости.
    """

    id: int
    timestamp: float  # Unix timestamp
    track_id: Optional[int]
    speed_kmh: float
    speed_limit_kmh: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_score: float

    @classmethod
    def from_db_row(cls, row: tuple) -> SpeedExceedanceEvent:
        """
        Создать объект из строки базы данных.

        Args:
            row: Кортеж из SQLite запроса в порядке:
                 (id, timestamp, track_id, speed_kmh, speed_limit_kmh,
                  bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_score)
        """
        (
            event_id,
            timestamp,
            track_id,
            speed_kmh,
            speed_limit_kmh,
            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,
            detection_score,
        ) = row
        return cls(
            id=event_id,
            timestamp=timestamp,
            track_id=track_id,
            speed_kmh=speed_kmh,
            speed_limit_kmh=speed_limit_kmh,
            bbox=(bbox_x1, bbox_y1, bbox_x2, bbox_y2),
            detection_score=detection_score,
        )

    def to_dict(self) -> dict:
        """Преобразовать в словарь для JSON сериализации."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "speed_kmh": self.speed_kmh,
            "speed_limit_kmh": self.speed_limit_kmh,
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
            "detection_score": self.detection_score,
        }
