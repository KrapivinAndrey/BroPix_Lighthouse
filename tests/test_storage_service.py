"""
Тесты для модуля storage.service.
"""

import time
from pathlib import Path

import pytest

from src.storage.database import init_database
from src.storage.models import SpeedExceedanceEvent
from src.storage.service import DataStorageService


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Создать временную базу данных для тестов."""
    db_path = tmp_path / "test_lighthouse.db"
    init_database(db_path)
    return db_path


def test_data_storage_service_init(temp_db: Path) -> None:
    """DataStorageService должен корректно инициализироваться."""
    service = DataStorageService(db_path=temp_db)
    assert service._db_path == temp_db


def test_record_speed_exceedance(temp_db: Path) -> None:
    """Запись события превышения скорости должна работать."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    event = SpeedExceedanceEvent(
        id=0,
        timestamp=now,
        track_id=1,
        speed_kmh=10.0,
        speed_limit_kmh=8.0,
        bbox=(10, 20, 30, 40),
        detection_score=0.9,
    )

    event_id = service.record_speed_exceedance(event)
    assert event_id > 0


def test_get_events_empty(temp_db: Path) -> None:
    """Получение событий из пустой БД должно вернуть пустой список."""
    service = DataStorageService(db_path=temp_db)
    events = service.get_events()
    assert len(events) == 0


def test_get_events_with_data(temp_db: Path) -> None:
    """Получение событий должно возвращать записанные события."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    # Записываем несколько событий
    for i in range(3):
        event = SpeedExceedanceEvent(
            id=0,
            timestamp=now + i,
            track_id=i,
            speed_kmh=10.0 + i,
            speed_limit_kmh=8.0,
            bbox=(10, 20, 30, 40),
            detection_score=0.9,
        )
        service.record_speed_exceedance(event)

    events = service.get_events(limit=10)
    assert len(events) == 3
    # События должны быть отсортированы по времени (новые первыми)
    assert events[0].timestamp > events[1].timestamp


def test_get_events_pagination(temp_db: Path) -> None:
    """Пагинация должна работать корректно."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    # Записываем 5 событий
    for i in range(5):
        event = SpeedExceedanceEvent(
            id=0,
            timestamp=now + i,
            track_id=i,
            speed_kmh=10.0,
            speed_limit_kmh=8.0,
            bbox=(10, 20, 30, 40),
            detection_score=0.9,
        )
        service.record_speed_exceedance(event)

    # Первая страница (2 записи)
    events_page1 = service.get_events(limit=2, offset=0)
    assert len(events_page1) == 2

    # Вторая страница (2 записи)
    events_page2 = service.get_events(limit=2, offset=2)
    assert len(events_page2) == 2

    # Третья страница (1 запись)
    events_page3 = service.get_events(limit=2, offset=4)
    assert len(events_page3) == 1


def test_get_events_time_filter(temp_db: Path) -> None:
    """Фильтрация по времени должна работать."""
    service = DataStorageService(db_path=temp_db)
    t0 = time.time()

    # Записываем события в разное время
    for i in range(5):
        event = SpeedExceedanceEvent(
            id=0,
            timestamp=t0 + i * 3600,  # Каждое событие через час
            track_id=i,
            speed_kmh=10.0,
            speed_limit_kmh=8.0,
            bbox=(10, 20, 30, 40),
            detection_score=0.9,
        )
        service.record_speed_exceedance(event)

    # Фильтр: события между t0+1h и t0+3h (включительно)
    events = service.get_events(start_time=t0 + 3600, end_time=t0 + 3 * 3600)
    assert len(events) == 3  # Должно быть 3 события (на t0+1h, t0+2h и t0+3h включительно)


def test_get_statistics(temp_db: Path) -> None:
    """Получение статистики должно возвращать корректные данные."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    # Записываем несколько событий с разными скоростями
    # Используем прошлые временные метки, чтобы они точно попали в статистику
    speeds = [10.0, 12.0, 15.0, 9.0]
    for i, speed in enumerate(speeds):
        event = SpeedExceedanceEvent(
            id=0,
            timestamp=now - (len(speeds) - i),  # События в прошлом
            track_id=i % 2,  # Два уникальных объекта
            speed_kmh=speed,
            speed_limit_kmh=8.0,
            bbox=(10, 20, 30, 40),
            detection_score=0.9,
        )
        service.record_speed_exceedance(event)

    # Используем явный end_time для гарантии, что все события попадут в выборку
    stats = service.get_statistics(end_time=now + 1)

    assert stats["total_events"] == 4
    assert stats["total_objects"] == 2
    # Средняя скорость: (10.0 + 12.0 + 15.0 + 9.0) / 4 = 11.5
    assert stats["avg_speed_kmh"] == pytest.approx(11.5, rel=0.1)
    assert stats["max_speed_kmh"] == 15.0
    assert isinstance(stats["events_by_hour"], dict)


def test_get_total_events_count(temp_db: Path) -> None:
    """Подсчет общего количества событий должен работать."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    for i in range(3):
        event = SpeedExceedanceEvent(
            id=0,
            timestamp=now + i,
            track_id=i,
            speed_kmh=10.0,
            speed_limit_kmh=8.0,
            bbox=(10, 20, 30, 40),
            detection_score=0.9,
        )
        service.record_speed_exceedance(event)

    total = service.get_total_events_count()
    assert total == 3

    # С фильтром по времени
    total_filtered = service.get_total_events_count(start_time=now + 1)
    assert total_filtered == 2


def test_export_to_json(temp_db: Path) -> None:
    """Экспорт в JSON должен возвращать валидный JSON."""
    import json

    service = DataStorageService(db_path=temp_db)
    now = time.time()

    event = SpeedExceedanceEvent(
        id=0,
        timestamp=now,
        track_id=1,
        speed_kmh=10.0,
        speed_limit_kmh=8.0,
        bbox=(10, 20, 30, 40),
        detection_score=0.9,
    )
    service.record_speed_exceedance(event)

    json_data = service.export_to_json()
    events_list = json.loads(json_data)

    assert isinstance(events_list, list)
    assert len(events_list) == 1
    assert events_list[0]["speed_kmh"] == 10.0
    assert events_list[0]["track_id"] == 1


def test_cleanup_old_events(temp_db: Path) -> None:
    """Очистка старых событий должна удалять записи старше указанного периода."""
    service = DataStorageService(db_path=temp_db)
    now = time.time()

    # Записываем старое событие (35 дней назад)
    old_event = SpeedExceedanceEvent(
        id=0,
        timestamp=now - 35 * 24 * 3600,
        track_id=1,
        speed_kmh=10.0,
        speed_limit_kmh=8.0,
        bbox=(10, 20, 30, 40),
        detection_score=0.9,
    )
    service.record_speed_exceedance(old_event)

    # Записываем новое событие (1 день назад)
    new_event = SpeedExceedanceEvent(
        id=0,
        timestamp=now - 1 * 24 * 3600,
        track_id=2,
        speed_kmh=10.0,
        speed_limit_kmh=8.0,
        bbox=(10, 20, 30, 40),
        detection_score=0.9,
    )
    service.record_speed_exceedance(new_event)

    # Очищаем события старше 30 дней
    deleted = service.cleanup_old_events(days_to_keep=30)
    assert deleted == 1

    # Проверяем, что осталось только новое событие
    events = service.get_events()
    assert len(events) == 1
    assert events[0].track_id == 2
