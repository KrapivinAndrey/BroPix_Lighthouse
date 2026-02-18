"""
Интеграционные тесты для записи событий через FrameProcessor.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from src.processing.frame_processor import FrameProcessor
from src.storage.database import init_database
from src.storage.service import DataStorageService


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Создать временную базу данных для тестов."""
    db_path = tmp_path / "test_lighthouse.db"
    init_database(db_path)
    return db_path


def test_frame_processor_records_events(temp_db: Path) -> None:
    """FrameProcessor должен записывать события при переходе трека в состояние превышения."""
    storage = DataStorageService(db_path=temp_db)
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.1,
        data_storage=storage,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()

    # Первая детекция - трек создается, но превышения еще нет
    result1 = processor.process(
        frame=frame,
        detections=[(0, 0, 10, 10, 0.9, 1)],
        now=t0,
    )
    assert result1.any_speed_exceeded is False

    # Вторая детекция с большой скоростью - должно быть превышение и запись события
    t1 = t0 + 1.0
    # Смещение на 20 пикселей за 1 секунду = 2 м/с = 7.2 км/ч > 8.0 км/ч при масштабе 0.1
    # Но нужно больше смещения для превышения 8.0 км/ч
    # При масштабе 0.1 м/px: для 8 км/ч = 2.22 м/с нужно 22.2 пикселя за секунду
    result2 = processor.process(
        frame=frame,
        detections=[(25, 0, 35, 10, 0.9, 1)],  # Смещение на 25 пикселей
        now=t1,
    )
    assert result2.any_speed_exceeded is True

    # Проверяем, что событие записано в БД
    events = storage.get_events()
    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].speed_kmh > 8.0


def test_frame_processor_no_duplicate_events(temp_db: Path) -> None:
    """FrameProcessor не должен дублировать события для одного трека."""
    storage = DataStorageService(db_path=temp_db)
    processor = FrameProcessor(
        speed_limit_kmh=8.0,
        red_hold_seconds=2.0,
        px_to_m_scale=0.1,
        data_storage=storage,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()

    # Первая детекция
    processor.process(frame=frame, detections=[(0, 0, 10, 10, 0.9, 1)], now=t0)

    # Вторая детекция с превышением - событие должно записаться
    t1 = t0 + 1.0
    processor.process(frame=frame, detections=[(25, 0, 35, 10, 0.9, 1)], now=t1)

    # Третья детекция - трек все еще в состоянии превышения, но новое событие не должно записаться
    t2 = t1 + 0.5
    processor.process(frame=frame, detections=[(30, 0, 40, 10, 0.9, 1)], now=t2)

    # Должно быть только одно событие (при переходе из False в True)
    events = storage.get_events()
    assert len(events) == 1
