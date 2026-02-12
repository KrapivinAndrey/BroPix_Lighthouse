"""
Тесты для модуля processing.frame_processor.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from src.processing.frame_processor import FrameProcessor, ProcessedDetection
from src.processing.speed_limit_checker import SpeedLimitState
from src.processing.track_manager import TrackData, TrackManager


def test_frame_processor_init() -> None:
    """FrameProcessor должен корректно инициализироваться."""
    processor = FrameProcessor(
        speed_limit_kmh=10.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.05,
    )
    assert processor._speed_limit_kmh == 10.0
    assert processor._red_hold_seconds == 2.0
    assert processor._px_to_m_scale == 0.05


def test_frame_processor_process_empty_detections() -> None:
    """Обработка пустого списка детекций должна возвращать пустой результат."""
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.05,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process(frame=frame, detections=[], now=time.time())

    assert len(result.detections) == 0
    assert result.any_speed_exceeded is False
    assert len(result.tracks) == 0


def test_frame_processor_process_detection_without_track_id() -> None:
    """Детекция без track_id должна обрабатываться без расчёта скорости."""
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.05,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections: List[Tuple[int, int, int, int, float, Optional[int]]] = [
        (10, 20, 30, 40, 0.9, None)
    ]

    result = processor.process(frame=frame, detections=detections, now=time.time())

    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.bbox == (10, 20, 30, 40)
    assert det.score == 0.9
    assert det.track_id is None
    assert det.speed_kmh is None
    assert det.is_over_limit is False


def test_frame_processor_process_detection_with_track_id() -> None:
    """Детекция с track_id должна создавать трек."""
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.05,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections: List[Tuple[int, int, int, int, float, Optional[int]]] = [
        (10, 20, 30, 40, 0.9, 1)
    ]

    now = time.time()
    result = processor.process(frame=frame, detections=detections, now=now)

    assert len(result.detections) == 1
    assert 1 in result.tracks
    track = result.tracks[1]
    assert track.last_pos == (20.0, 30.0)  # центр bbox
    assert track.last_time == now


def test_frame_processor_speed_calculation() -> None:
    """Расчёт скорости должен работать при наличии предыдущей позиции трека."""
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.1,  # 10 см на пиксель
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    # Первая детекция
    result1 = processor.process(
        frame=frame,
        detections=[(0, 0, 10, 10, 0.9, 1)],
        now=t0,
    )
    assert result1.detections[0].speed_kmh is None  # Нет предыдущей позиции

    t1 = t0 + 1.0  # Через 1 секунду
    # Вторая детекция со смещением на 10 пикселей
    result2 = processor.process(
        frame=frame,
        detections=[(10, 0, 20, 10, 0.9, 1)],
        now=t1,
    )
    speed = result2.detections[0].speed_kmh
    assert speed is not None
    # 10 px * 0.1 м/px = 1 м за 1 с = 1 м/с = 3.6 км/ч
    assert abs(speed - 3.6) < 0.1


def test_frame_processor_speed_limit_exceeded() -> None:
    """Флаг превышения скорости должен устанавливаться при превышении лимита."""
    processor = FrameProcessor(
        speed_limit_kmh=5.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.1,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    processor.process(frame=frame, detections=[(0, 0, 10, 10, 0.9, 1)], now=t0)

    t1 = t0 + 1.0
    # Смещение на 20 пикселей за 1 секунду = 2 м/с = 7.2 км/ч > 5.0 км/ч
    result = processor.process(
        frame=frame,
        detections=[(20, 0, 30, 10, 0.9, 1)],
        now=t1,
    )

    assert result.any_speed_exceeded is True
    assert result.detections[0].is_over_limit is True


def test_frame_processor_red_hold_logic() -> None:
    """Логика red_hold должна удерживать флаг превышения после снижения скорости."""
    processor = FrameProcessor(
        speed_limit_kmh=5.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.1,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    processor.process(frame=frame, detections=[(0, 0, 10, 10, 0.9, 1)], now=t0)

    t1 = t0 + 1.0
    # Превышение скорости
    result1 = processor.process(
        frame=frame,
        detections=[(20, 0, 30, 10, 0.9, 1)],
        now=t1,
    )
    assert result1.any_speed_exceeded is True

    t2 = t1 + 0.5  # Через 0.5 секунды после превышения
    # Скорость упала ниже лимита, но флаг должен остаться
    result2 = processor.process(
        frame=frame,
        detections=[(21, 0, 31, 10, 0.9, 1)],  # Минимальное смещение
        now=t2,
    )
    assert result2.any_speed_exceeded is True

    t3 = t1 + 2.5  # Через 2.5 секунды после превышения (> red_hold_seconds)
    # Флаг должен сброситься
    result3 = processor.process(
        frame=frame,
        detections=[(22, 0, 32, 10, 0.9, 1)],
        now=t3,
    )
    assert result3.any_speed_exceeded is False


def test_frame_processor_track_cleanup() -> None:
    """Устаревшие треки должны удаляться по TTL."""
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.05,
        track_ttl_seconds=1.0,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t0 = time.time()
    processor.process(frame=frame, detections=[(0, 0, 10, 10, 0.9, 1)], now=t0)

    assert 1 in processor.track_manager.tracks

    t1 = t0 + 1.5  # Через 1.5 секунды (> TTL)
    processor.process(frame=frame, detections=[], now=t1)

    # Трек должен быть удалён
    assert 1 not in processor.track_manager.tracks
