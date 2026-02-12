from __future__ import annotations

"""
Сервис для работы с хранилищем данных (SQLite).

Предоставляет методы для записи событий превышения скорости,
получения статистики и экспорта данных.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

from src.storage.database import init_database, migrate_database
from src.storage.models import SpeedExceedanceEvent

logger = logging.getLogger(__name__)


class DataStorageService:
    """
    Сервис для работы с хранилищем событий превышения скорости.

    Потокобезопасный доступ к SQLite базе данных.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Инициализация сервиса хранения данных.

        Args:
            db_path: Путь к файлу SQLite базы данных.
                    Если None, используется data/lighthouse.db в корне проекта.
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "lighthouse.db"

        self._db_path = Path(db_path)
        self._lock = threading.Lock()

        # Инициализируем базу данных, если её нет
        if not self._db_path.exists():
            init_database(self._db_path)
        else:
            # Проверяем и применяем миграции, если нужно
            migrate_database(self._db_path)

    def record_speed_exceedance(self, event: SpeedExceedanceEvent) -> int:
        """
        Записать событие превышения скорости в базу данных.

        Args:
            event: Событие превышения скорости (без поля id, оно будет создано автоматически)

        Returns:
            ID созданной записи в базе данных
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO speed_exceedance_events
                    (timestamp, track_id, speed_kmh, speed_limit_kmh,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp,
                        event.track_id,
                        event.speed_kmh,
                        event.speed_limit_kmh,
                        event.bbox[0],
                        event.bbox[1],
                        event.bbox[2],
                        event.bbox[3],
                        event.detection_score,
                    ),
                )
                conn.commit()
                event_id = cursor.lastrowid
                if event_id is None:
                    raise RuntimeError("Не удалось получить ID созданной записи")
                logger.debug(f"Записано событие превышения скорости: ID={event_id}")
                return event_id
            finally:
                conn.close()

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[SpeedExceedanceEvent]:
        """
        Получить список событий превышения скорости.

        Args:
            limit: Максимальное количество записей
            offset: Смещение для пагинации
            start_time: Начало периода (Unix timestamp), None = без ограничения
            end_time: Конец периода (Unix timestamp), None = без ограничения

        Returns:
            Список событий, отсортированных по времени (новые первыми)
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Формируем SQL запрос с фильтрами
                query = """
                    SELECT id, timestamp, track_id, speed_kmh, speed_limit_kmh,
                           bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_score
                    FROM speed_exceedance_events
                    WHERE 1=1
                """
                params: List[Any] = []

                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [SpeedExceedanceEvent.from_db_row(row) for row in rows]
            finally:
                conn.close()

    def get_total_events_count(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> int:
        """
        Получить общее количество событий (для пагинации).

        Args:
            start_time: Начало периода
            end_time: Конец периода

        Returns:
            Общее количество событий, соответствующих фильтрам
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                query = "SELECT COUNT(*) FROM speed_exceedance_events WHERE 1=1"
                params: List[Any] = []

                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                cursor.execute(query, params)
                result = cursor.fetchone()
                return int(result[0]) if result else 0
            finally:
                conn.close()

    def get_statistics(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Получить статистику по событиям превышения скорости.

        Args:
            start_time: Начало периода (Unix timestamp)
            end_time: Конец периода (Unix timestamp), None = текущее время

        Returns:
            Словарь со статистикой:
            - total_events: общее количество событий
            - total_objects: количество уникальных объектов (по track_id)
            - avg_speed_kmh: средняя скорость превышений
            - max_speed_kmh: максимальная скорость
            - events_by_hour: распределение событий по часам (0-23)
        """
        if end_time is None:
            end_time = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Базовый WHERE для фильтров
                where_clause = "WHERE 1=1"
                params: List[Any] = []

                if start_time is not None:
                    where_clause += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    where_clause += " AND timestamp <= ?"
                    params.append(end_time)

                # Общее количество событий
                # Используем те же параметры для фильтрации
                count_params = params.copy()
                cursor.execute(f"SELECT COUNT(*) FROM speed_exceedance_events {where_clause}", count_params)
                total_events = int(cursor.fetchone()[0])

                # Количество уникальных объектов
                cursor.execute(
                    f"""
                    SELECT COUNT(DISTINCT track_id)
                    FROM speed_exceedance_events
                    {where_clause} AND track_id IS NOT NULL
                    """,
                    params,
                )
                total_objects = int(cursor.fetchone()[0])

                # Средняя и максимальная скорость
                cursor.execute(
                    f"""
                    SELECT AVG(speed_kmh), MAX(speed_kmh)
                    FROM speed_exceedance_events
                    {where_clause}
                    """,
                    params,
                )
                result = cursor.fetchone()
                avg_speed_kmh = float(result[0]) if result[0] is not None else 0.0
                max_speed_kmh = float(result[1]) if result[1] is not None else 0.0

                # Распределение по часам
                cursor.execute(
                    f"""
                    SELECT CAST((timestamp % 86400) / 3600 AS INTEGER) as hour, COUNT(*)
                    FROM speed_exceedance_events
                    {where_clause}
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    params,
                )
                events_by_hour_raw = cursor.fetchall()
                events_by_hour: Dict[int, int] = {hour: count for hour, count in events_by_hour_raw}

                return {
                    "total_events": total_events,
                    "total_objects": total_objects,
                    "avg_speed_kmh": round(avg_speed_kmh, 2),
                    "max_speed_kmh": round(max_speed_kmh, 2),
                    "events_by_hour": events_by_hour,
                }
            finally:
                conn.close()

    def export_to_json(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> str:
        """
        Экспортировать события в JSON строку.

        Args:
            start_time: Начало периода
            end_time: Конец периода

        Returns:
            JSON строка с массивом событий
        """
        events = self.get_events(limit=10000, offset=0, start_time=start_time, end_time=end_time)
        events_dict = [event.to_dict() for event in events]
        return json.dumps(events_dict, ensure_ascii=False, indent=2)

    def cleanup_old_events(self, days_to_keep: int = 30) -> int:
        """
        Удалить события старше указанного количества дней.

        Args:
            days_to_keep: Количество дней для хранения событий

        Returns:
            Количество удаленных записей
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM speed_exceedance_events WHERE timestamp < ?",
                    (cutoff_time,),
                )
                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Удалено {deleted_count} старых событий (старше {days_to_keep} дней)")
                return deleted_count
            finally:
                conn.close()
