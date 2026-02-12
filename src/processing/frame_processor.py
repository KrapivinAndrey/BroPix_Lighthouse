from __future__ import annotations

"""
Единый обработчик детекций для расчёта скорости и флага превышения.

FrameProcessor не занимается захватом кадров и вызовом окон OpenCV —
он только принимает список детекций и обновляет состояние треков.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.processing.speed_limit_checker import SpeedLimitChecker
from src.processing.track_manager import TrackData, TrackManager
from src.speed_utils import compute_speed_kmh

try:
    from src.lighthouse.controller import LighthouseController
except ImportError:
    # Для обратной совместимости, если модуль lighthouse ещё не создан
    LighthouseController = None  # type: ignore

try:
    from src.storage.models import SpeedExceedanceEvent
    from src.storage.service import DataStorageService
except ImportError:
    # Для обратной совместимости, если модуль storage ещё не создан
    SpeedExceedanceEvent = None  # type: ignore
    DataStorageService = None  # type: ignore


Detection = Tuple[int, int, int, int, float, Optional[int]]


@dataclass
class ProcessedDetection:
    """Результат обработки одной детекции."""

    bbox: Tuple[int, int, int, int]
    score: float
    track_id: Optional[int]
    speed_kmh: Optional[float]
    is_over_limit: bool


@dataclass
class ProcessedFrame:
    """
    Результат обработки кадра.

    frame тут хранится для удобства, чтобы потребители могли при желании
    модифицировать тот же объект (нарисовать рамки и текст).
    """

    frame: np.ndarray
    detections: List[ProcessedDetection]
    tracks: Dict[int, TrackData]
    any_speed_exceeded: bool


class FrameProcessor:
    """
    Обработчик детекций: расчёт скорости, логика red-hold и TTL треков.

    Детектор (YOLO) вызывается вне этого класса — сюда приходят уже
    готовые детекции в формате:
        (x1, y1, x2, y2, score, track_id | None)
    """

    def __init__(
        self,
        *,
        speed_limit_kmh: float,
        red_hold_seconds: float,
        px_to_m_scale: float,
        track_ttl_seconds: float = 2.0,
        min_distance_px: float = 1.0,
        speed_func: Callable[..., Optional[float]] = compute_speed_kmh,
        lighthouse_controller: Optional[LighthouseController] = None,
        data_storage: Optional[DataStorageService] = None,
    ) -> None:
        self._speed_limit_kmh = float(speed_limit_kmh)
        self._red_hold_seconds = float(red_hold_seconds)
        self._px_to_m_scale = float(px_to_m_scale)
        self._min_distance_px = float(min_distance_px)
        self._track_manager = TrackManager(ttl_seconds=track_ttl_seconds)
        self._checker = SpeedLimitChecker(
            speed_limit_kmh=self._speed_limit_kmh,
            red_hold_seconds=self._red_hold_seconds,
        )
        self._speed_func = speed_func
        # lighthouse_controller может быть None, если модуль lighthouse не импортирован
        # или если контроллер не передан явно
        self._lighthouse_controller: Optional[LighthouseController] = lighthouse_controller
        # data_storage для записи событий превышения скорости
        self._data_storage: Optional[DataStorageService] = data_storage

    @property
    def track_manager(self) -> TrackManager:
        return self._track_manager

    def process(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        *,
        now: float,
    ) -> ProcessedFrame:
        """
        Обработать список детекций и обновить состояние треков.

        Возвращает ProcessedFrame с информацией по каждой детекции и
        общим флагом any_speed_exceeded.
        """
        self._track_manager.cleanup_expired(now)

        processed: List[ProcessedDetection] = []
        any_speed_exceeded = False

        for det in detections:
            if len(det) == 5:
                x1, y1, x2, y2, score = det
                track_id: Optional[int] = None
            elif len(det) == 6:
                x1, y1, x2, y2, score, track_id = det
            else:
                # Неизвестный формат — пропускаем.
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center = (cx, cy)

            speed_kmh: Optional[float] = None
            is_over_limit = False

            if track_id is not None and track_id >= 0:
                prev = self._track_manager.get(track_id)
                prev_speed = prev.speed_kmh if prev is not None else None
                prev_is_over_limit = prev.is_over_limit if prev is not None else False
                prev_last_over_limit_time = (
                    prev.last_over_limit_time if prev is not None else None
                )

                if prev is not None:
                    speed_kmh = self._speed_func(
                        prev.last_pos,
                        prev.last_time,
                        center,
                        now,
                        px_to_m_scale=self._px_to_m_scale,
                        min_distance_px=self._min_distance_px,
                    )

                current_speed = speed_kmh if speed_kmh is not None else prev_speed

                state = self._checker.update_state(
                    prev_is_over_limit=prev_is_over_limit,
                    prev_last_over_limit_time=prev_last_over_limit_time,
                    now=now,
                    current_speed_kmh=current_speed,
                )

                self._track_manager.upsert(
                    track_id,
                    last_pos=center,
                    last_time=now,
                    speed_kmh=current_speed,
                    is_over_limit=state.is_over_limit,
                    last_over_limit_time=state.last_over_limit_time,
                )

                is_over_limit = state.is_over_limit
                if is_over_limit:
                    any_speed_exceeded = True

                    # Записываем событие превышения скорости при переходе состояния
                    # (только если трек перешел из False в True)
                    if (
                        self._data_storage is not None
                        and SpeedExceedanceEvent is not None
                        and not prev_is_over_limit
                        and is_over_limit
                    ):
                        try:
                            # Создаем событие для записи в БД
                            event = SpeedExceedanceEvent(
                                id=0,  # Будет присвоен БД
                                timestamp=now,
                                track_id=track_id,
                                speed_kmh=current_speed if current_speed is not None else 0.0,
                                speed_limit_kmh=self._speed_limit_kmh,
                                bbox=(x1, y1, x2, y2),
                                detection_score=score,
                            )
                            self._data_storage.record_speed_exceedance(event)
                        except Exception as e:
                            # Логируем ошибку, но не прерываем обработку кадра
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.warning(f"Не удалось записать событие превышения скорости: {e}")

            processed.append(
                ProcessedDetection(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    track_id=track_id,
                    speed_kmh=speed_kmh,
                    is_over_limit=is_over_limit,
                )
            )

        result = ProcessedFrame(
            frame=frame,
            detections=processed,
            tracks=self._track_manager.tracks.copy(),
            any_speed_exceeded=any_speed_exceeded,
        )

        # Обновляем состояние маяка, если контроллер передан
        if self._lighthouse_controller is not None:
            self._lighthouse_controller.set_state(any_speed_exceeded)

        return result

